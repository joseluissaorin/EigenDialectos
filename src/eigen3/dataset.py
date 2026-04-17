"""Dialect MLM dataset with variety labels and balanced batch sampling.

Two dataset classes:
  DialectMLMDataset   — for transformer training (MLM + classification + SupCon)
  SubwordDCLDataset   — legacy skip-gram dataset (kept for backward compat)

Plus a ``BalancedVarietySampler`` (v2) that yields batches containing an
equal number of samples from every variety, which is what SupCon + XBM
need to see meaningful within-batch positives.
"""

from __future__ import annotations

import logging
import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from eigen3.constants import (
    AFFINITY_BASE,
    ALL_REGIONALISMS,
    ALL_VARIETIES,
    VARIETY_AFFINITIES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transformer dataset
# ---------------------------------------------------------------------------

class DialectMLMDataset(Dataset):
    """Context-window MLM dataset with variety labels.

    Each item: tokenized text with variety prefix, MLM labels, variety id.
    Dynamic masking: 15% of tokens masked at __getitem__ time.
    """

    def __init__(
        self,
        corpus_by_variety: dict[str, list[str]],
        tokenizer,
        variety_token_ids: dict[str, int],
        max_length: int = 256,
        mlm_prob: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.variety_token_ids = variety_token_ids
        self.max_length = max_length
        self.mlm_prob = mlm_prob
        self._rng = random.Random(seed)

        self.variety_names: list[str] = sorted(corpus_by_variety.keys())
        self.variety_to_idx: dict[str, int] = {v: i for i, v in enumerate(self.variety_names)}

        # Flatten corpus into (text, variety_id) pairs
        self._samples: list[tuple[str, int]] = []
        for variety, docs in corpus_by_variety.items():
            v_idx = self.variety_to_idx[variety]
            for doc in docs:
                doc = doc.strip()
                if doc:
                    self._samples.append((doc, v_idx))

        # Shuffle deterministically
        self._rng.shuffle(self._samples)

        # Get special token IDs for masking
        self._mask_token_id = tokenizer.mask_token_id
        self._cls_token_id = tokenizer.cls_token_id
        self._sep_token_id = tokenizer.sep_token_id
        self._pad_token_id = tokenizer.pad_token_id
        self._special_ids = set(variety_token_ids.values()) | {
            self._cls_token_id, self._sep_token_id, self._pad_token_id,
        }

        logger.info("DialectMLMDataset: %d samples, %d varieties",
                     len(self._samples), len(self.variety_names))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text, variety_id = self._samples[idx]

        # Prepend variety token
        variety_name = self.variety_names[variety_id]
        var_token = f"[VAR_{variety_name}]"
        text_with_var = f"{var_token} {text}"

        # Tokenize
        encoding = self.tokenizer(
            text_with_var,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)    # (seq_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Dynamic MLM masking
        mlm_input_ids, mlm_labels = self._apply_mlm_masking(input_ids.clone())

        return {
            "input_ids": mlm_input_ids,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
            "variety_id": torch.tensor(variety_id, dtype=torch.long),
        }

    def _apply_mlm_masking(
        self, input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply dynamic MLM masking (15% of non-special tokens).

        80% -> [MASK], 10% -> random, 10% -> keep original.
        """
        labels = torch.full_like(input_ids, -100)  # -100 = ignore
        seq_len = input_ids.size(0)

        for i in range(seq_len):
            token_id = input_ids[i].item()
            if token_id in self._special_ids:
                continue
            if random.random() > self.mlm_prob:
                continue

            labels[i] = token_id
            r = random.random()
            if r < 0.8:
                input_ids[i] = self._mask_token_id
            elif r < 0.9:
                input_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
            # else: keep original (10%)

        return input_ids, labels


class DialectBatchCollator:
    """Simple stacking collator for v2 training.

    With ``BalancedVarietySampler`` providing class-balanced batches,
    the contrastive objective (SupCon) uses the variety labels directly
    and no explicit pair mining is needed.
    """

    def __init__(self, n_varieties: int = 8) -> None:
        self.n_varieties = n_varieties

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids":      torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "mlm_labels":     torch.stack([b["mlm_labels"] for b in batch]),
            "variety_ids":    torch.stack([b["variety_id"] for b in batch]),
        }


# Backward-compatible alias — external imports keep working.
DialectContrastiveCollator = DialectBatchCollator


class BalancedVarietySampler(Sampler[list[int]]):
    """Yield batches with exactly ``samples_per_variety`` items per variety.

    Batch size = samples_per_variety * n_varieties. With the defaults
    (k=4, n=8) this gives 32-example batches — the same batch size used
    in v1, so grad-accum math carries over. Epoch length is bounded by
    the smallest per-variety pool: ``floor(min_pool / k)`` batches per
    epoch. Because the corpus is run through ``balance_corpus`` upstream
    every pool should be close to equal in size, so very little data is
    left on the table.
    """

    def __init__(
        self,
        variety_ids: np.ndarray,
        samples_per_variety: int = 4,
        n_varieties: int = 8,
        seed: int = 42,
    ) -> None:
        super().__init__(data_source=None)
        self.k = samples_per_variety
        self.n_var = n_varieties
        self.batch_size = self.k * self.n_var
        self._rng = np.random.default_rng(seed)

        variety_ids = np.asarray(variety_ids)
        self._pools = [np.where(variety_ids == v)[0] for v in range(n_varieties)]
        pool_sizes = [len(p) for p in self._pools]
        missing = [v for v, s in enumerate(pool_sizes) if s == 0]
        if missing:
            raise ValueError(
                f"BalancedVarietySampler: varieties without samples: {missing}"
            )
        min_size = min(pool_sizes)
        self._batches_per_epoch = min_size // self.k
        if self._batches_per_epoch == 0:
            raise ValueError(
                f"BalancedVarietySampler: smallest pool has only {min_size} "
                f"samples but k={self.k}"
            )

        logger.info(
            "BalancedVarietySampler: k=%d, n_var=%d, batch=%d, "
            "batches_per_epoch=%d (min pool=%d)",
            self.k, self.n_var, self.batch_size,
            self._batches_per_epoch, min_size,
        )

    def __iter__(self) -> Iterator[list[int]]:
        shuffled = [self._rng.permutation(p) for p in self._pools]
        for b in range(self._batches_per_epoch):
            batch: list[int] = []
            for v in range(self.n_var):
                start = b * self.k
                batch.extend(int(x) for x in shuffled[v][start:start + self.k])
            self._rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self._batches_per_epoch


# ---------------------------------------------------------------------------
# Legacy skip-gram dataset (kept for backward compatibility)
# ---------------------------------------------------------------------------

def _get_affinity(v1: str, v2: str) -> float:
    """Get pairwise variety affinity (symmetric)."""
    return VARIETY_AFFINITIES.get(
        (v1, v2),
        VARIETY_AFFINITIES.get((v2, v1), AFFINITY_BASE),
    )


class SubwordDCLDataset(Dataset):
    """Pre-materialized skip-gram pairs with affinity-weighted negative sampling.

    All samples stored as a contiguous numpy array for MPS throughput.
    Each row: (center, ctx_same, ctx_other, variety_a, variety_b, is_regionalism)
    """

    def __init__(
        self,
        corpus_by_variety: dict[str, list[str]],
        tokenizer,
        window_size: int = 5,
        neg_samples: int = 5,
        regionalism_set: frozenset[str] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.regionalism_set = regionalism_set or ALL_REGIONALISMS
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        self.variety_names: list[str] = sorted(corpus_by_variety.keys())
        self.variety_to_idx: dict[str, int] = {v: i for i, v in enumerate(self.variety_names)}
        self.vocab_size: int = tokenizer.vocab_size

        # Tokenize and build skip-gram pairs
        variety_pools: dict[int, np.ndarray] = {}
        skipgram_pairs: list[tuple[int, int, int, int]] = []

        for variety_name, docs in corpus_by_variety.items():
            v_idx = self.variety_to_idx[variety_name]
            pool_tokens: list[int] = []

            for doc in docs:
                doc = doc.strip()
                if not doc:
                    continue

                words = doc.lower().split()
                token_ids = tokenizer.encode(doc)
                if len(token_ids) < 2:
                    continue

                is_reg = int(any(w in self.regionalism_set for w in words))
                pool_tokens.extend(token_ids)

                n = len(token_ids)
                for i in range(n):
                    left = max(0, i - window_size)
                    right = min(n, i + window_size + 1)
                    for j in range(left, right):
                        if j == i:
                            continue
                        skipgram_pairs.append((token_ids[i], token_ids[j], v_idx, is_reg))

            variety_pools[v_idx] = np.array(pool_tokens, dtype=np.int64) if pool_tokens else np.array([], dtype=np.int64)

        n_base = len(skipgram_pairs)
        all_v_indices = sorted(variety_pools.keys())

        neg_probs = self._build_neg_sampling_probs(all_v_indices)

        n_total = n_base * neg_samples
        data = np.empty((n_total, 6), dtype=np.int64)

        for i, (center, ctx_same, v_a, is_reg) in enumerate(skipgram_pairs):
            probs = neg_probs[v_a]
            other_indices = [v for v in all_v_indices if v != v_a]
            base_idx = i * neg_samples
            for k in range(neg_samples):
                if other_indices and len(probs) > 0:
                    v_b = self._np_rng.choice(other_indices, p=probs)
                else:
                    v_b = v_a
                pool_b = variety_pools.get(v_b)
                if pool_b is not None and len(pool_b) > 0:
                    ctx_other = int(pool_b[self._np_rng.randint(len(pool_b))])
                else:
                    ctx_other = self._np_rng.randint(self.vocab_size)
                data[base_idx + k] = (center, ctx_same, ctx_other, v_a, v_b, is_reg)

        self._data = data
        self._n_total = n_total

        logger.info(
            "SubwordDCLDataset: vocab=%d, varieties=%d, pairs=%d, total=%d",
            self.vocab_size, len(self.variety_names), n_base, n_total,
        )

    def _build_neg_sampling_probs(self, all_v_indices: list[int]) -> dict[int, np.ndarray]:
        """Affinity-weighted negative sampling: high affinity -> less negative sampling."""
        neg_probs: dict[int, np.ndarray] = {}
        for v_a in all_v_indices:
            others = [v for v in all_v_indices if v != v_a]
            if not others:
                neg_probs[v_a] = np.array([])
                continue

            v_a_name = self.variety_names[v_a]
            weights = []
            for v_b in others:
                v_b_name = self.variety_names[v_b]
                aff = _get_affinity(v_a_name, v_b_name)
                weights.append(1.0 - aff)

            weights = np.array(weights, dtype=np.float64)
            total = weights.sum()
            if total > 0:
                weights /= total
            else:
                weights = np.ones(len(others)) / len(others)
            neg_probs[v_a] = weights

        return neg_probs

    def __len__(self) -> int:
        return self._n_total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        row = self._data[idx]
        return tuple(torch.tensor(x, dtype=torch.long) for x in row)
