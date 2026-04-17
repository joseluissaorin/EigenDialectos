"""Subword-level skip-gram dataset with cross-variety negative sampling.

Like :class:`DCLDataset` but operates on BPE subword tokens from a shared
:class:`SharedSubwordTokenizer`.  Regionalism status is inherited from the
parent word: if the word is a regionalism, all of its constituent subword
tokens are marked as regionalisms.

All training samples are **pre-materialized** into a contiguous numpy array
at construction time so that ``__getitem__`` is a single array row lookup —
critical for MPS GPU throughput where Python-per-sample overhead dominates.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from eigendialectos.embeddings.dcl.regionalisms import ALL_REGIONALISMS

logger = logging.getLogger(__name__)


class SubwordDCLDataset(Dataset):
    """Yields tuples for subword-level Dialect-Contrastive Loss training.

    Each sample is:
        ``(subword_idx, ctx_idx_same, ctx_idx_other, variety_a, variety_b, is_regionalism)``

    All samples are pre-materialized at construction for fast GPU training.

    Parameters
    ----------
    corpus_by_variety:
        Mapping from variety name to list of text documents.
    tokenizer:
        A trained :class:`SharedSubwordTokenizer` instance.
    window_size:
        Skip-gram context window radius (subword tokens on each side).
    neg_samples:
        Number of times each positive pair is repeated with different
        cross-variety negatives.
    regionalism_set:
        Words considered to be regionalisms.  All subword tokens of a
        regionalism word are marked as such.
    subsampling_threshold:
        Threshold for frequent-subword subsampling (Mikolov formula).
        Set to 0 to disable.
    seed:
        Random seed.
    """

    def __init__(
        self,
        corpus_by_variety: dict[str, list[str]],
        tokenizer,  # SharedSubwordTokenizer
        window_size: int = 5,
        neg_samples: int = 5,
        regionalism_set: set[str] | None = None,
        subsampling_threshold: float = 1e-4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.regionalism_set = (
            regionalism_set if regionalism_set is not None else ALL_REGIONALISMS
        )
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)
        self._tokenizer = tokenizer

        # ------------------------------------------------------------------
        # 1. Variety bookkeeping
        # ------------------------------------------------------------------
        self.variety_names: list[str] = sorted(corpus_by_variety.keys())
        self.variety_to_idx: dict[str, int] = {
            v: i for i, v in enumerate(self.variety_names)
        }
        self.vocab_size: int = tokenizer.vocab_size

        # ------------------------------------------------------------------
        # 2. Tokenize corpus and track regionalism status per subword position
        # ------------------------------------------------------------------
        variety_token_lists: dict[int, list[list[int]]] = {}
        variety_reg_masks: dict[int, list[list[bool]]] = {}

        global_counts: Counter[int] = Counter()
        total_tokens = 0

        for variety_name, docs in corpus_by_variety.items():
            v_idx = self.variety_to_idx[variety_name]
            token_docs: list[list[int]] = []
            reg_docs: list[list[bool]] = []

            for doc in docs:
                doc_text = doc.strip()
                if not doc_text:
                    continue

                subword_ids, reg_mask = self._tokenize_with_regionalism(doc_text)

                if len(subword_ids) < 2:
                    continue

                token_docs.append(subword_ids)
                reg_docs.append(reg_mask)

                for sid in subword_ids:
                    global_counts[sid] += 1
                    total_tokens += 1

            variety_token_lists[v_idx] = token_docs
            variety_reg_masks[v_idx] = reg_docs

        # ------------------------------------------------------------------
        # 3. Compute subsampling keep-probabilities
        # ------------------------------------------------------------------
        keep_prob: dict[int, float] = {}
        if subsampling_threshold > 0 and total_tokens > 0:
            for token_id, count in global_counts.items():
                freq = count / total_tokens
                keep_prob[token_id] = min(
                    1.0,
                    math.sqrt(subsampling_threshold / freq)
                    + subsampling_threshold / freq,
                )

        # ------------------------------------------------------------------
        # 4. Generate skip-gram pairs (as Python list, then vectorize)
        # ------------------------------------------------------------------
        skipgram_pairs: list[tuple[int, int, int, int]] = []
        # (center_subword, context_subword, variety_idx, is_regionalism_int)

        for v_idx, doc_lists in variety_token_lists.items():
            reg_docs = variety_reg_masks[v_idx]
            for doc_tokens, doc_regs in zip(doc_lists, reg_docs):
                n = len(doc_tokens)
                for i in range(n):
                    center = doc_tokens[i]

                    if center in keep_prob:
                        if self._rng.random() > keep_prob[center]:
                            continue

                    is_reg = int(doc_regs[i])
                    left = max(0, i - self.window_size)
                    right = min(n, i + self.window_size + 1)
                    for j in range(left, right):
                        if j == i:
                            continue
                        skipgram_pairs.append(
                            (center, doc_tokens[j], v_idx, is_reg)
                        )

        n_base = len(skipgram_pairs)

        # ------------------------------------------------------------------
        # 5. Build per-variety token pools as numpy arrays
        # ------------------------------------------------------------------
        variety_pools_np: dict[int, np.ndarray] = {}
        for v_idx, doc_lists in variety_token_lists.items():
            pool: list[int] = []
            for doc_tokens in doc_lists:
                pool.extend(doc_tokens)
            variety_pools_np[v_idx] = np.array(pool, dtype=np.int64)

        all_variety_indices = sorted(variety_pools_np.keys())

        # ------------------------------------------------------------------
        # 5b. Build affinity-weighted negative sampling probabilities.
        #     Linguistically distant varieties are sampled MORE as negatives,
        #     so CAN-CAR pairs are rarer negatives (weaker repulsion).
        # ------------------------------------------------------------------
        neg_probs = self._build_neg_sampling_probs(all_variety_indices)

        # ------------------------------------------------------------------
        # 6. Pre-materialize ALL samples into a contiguous numpy array
        #    This eliminates per-sample Python overhead in __getitem__.
        # ------------------------------------------------------------------
        logger.info(
            "Pre-materializing %d × %d = %d samples ...",
            n_base, self.neg_samples, n_base * self.neg_samples,
        )

        n_total = n_base * self.neg_samples
        # Columns: center, ctx_same, ctx_other, variety_a, variety_b, is_reg
        data = np.empty((n_total, 6), dtype=np.int64)

        for i, (center, ctx_same, v_a, is_reg) in enumerate(skipgram_pairs):
            probs = neg_probs[v_a]  # Pre-computed probabilities
            other_indices = [v for v in all_variety_indices if v != v_a]
            base_idx = i * self.neg_samples
            for k in range(self.neg_samples):
                v_b = self._np_rng.choice(other_indices, p=probs) if other_indices else v_a
                pool_b = variety_pools_np.get(v_b)
                if pool_b is not None and len(pool_b) > 0:
                    ctx_other = int(pool_b[self._np_rng.randint(len(pool_b))])
                else:
                    ctx_other = self._np_rng.randint(self.vocab_size)
                data[base_idx + k] = (center, ctx_same, ctx_other, v_a, v_b, is_reg)

        self._data = data
        self._total_len = n_total

        # Free temporary structures
        del skipgram_pairs, variety_token_lists, variety_reg_masks
        del variety_pools_np

        logger.info(
            "SubwordDCLDataset: vocab=%d, varieties=%d, "
            "skip-gram pairs=%d, total samples=%d (%.1f MB)",
            self.vocab_size,
            len(self.variety_names),
            n_base,
            self._total_len,
            self._data.nbytes / 1e6,
        )

    def _build_neg_sampling_probs(
        self,
        all_variety_indices: list[int],
    ) -> dict[int, np.ndarray]:
        """Build per-variety negative sampling probabilities.

        Linguistically distant varieties are sampled MORE as negatives.
        This encodes variety affinity without modifying the loss function.
        CAN-CAR pairs appear less as negatives → weaker effective repulsion.
        """
        # Variety affinity: 0 = unrelated, 1 = identical
        # Based on standard Spanish dialectology groupings.
        _AFFINITY = {
            # Strongest links (should cluster together)
            ("ES_CAN", "ES_CAR"): 0.92,   # Atlantic Spanish: historical migration
            ("ES_AND", "ES_AND_BO"): 0.90, # Andalusian sub-varieties
            # Southern Cone
            ("ES_CHI", "ES_RIO"): 0.70,
            # Caribbean basin / Latin American connections
            ("ES_CAR", "ES_MEX"): 0.35,
            ("ES_CAR", "ES_CHI"): 0.30,
            ("ES_CAR", "ES_RIO"): 0.30,
            ("ES_MEX", "ES_CHI"): 0.35,
            ("ES_MEX", "ES_RIO"): 0.35,
            # Iberian / Canarian connections
            ("ES_AND", "ES_CAN"): 0.50,   # Andalusian → Canarian historical
            ("ES_PEN", "ES_AND"): 0.25,
            ("ES_PEN", "ES_CAN"): 0.20,
        }
        base_affinity = 0.10

        idx_to_name = {i: n for n, i in self.variety_to_idx.items()}
        neg_probs: dict[int, np.ndarray] = {}

        for v_a in all_variety_indices:
            others = [v for v in all_variety_indices if v != v_a]
            if not others:
                neg_probs[v_a] = np.array([])
                continue

            name_a = idx_to_name[v_a]
            weights = []
            for v_b in others:
                name_b = idx_to_name[v_b]
                pair = (name_a, name_b)
                rev_pair = (name_b, name_a)
                aff = _AFFINITY.get(pair, _AFFINITY.get(rev_pair, base_affinity))
                # Negative sampling weight = 1 - affinity (distant → MORE negatives)
                weights.append(1.0 - aff)

            weights = np.array(weights, dtype=np.float64)
            weights /= weights.sum()
            neg_probs[v_a] = weights

            logger.debug(
                "  Neg sampling %s: %s",
                name_a,
                ", ".join(f"{idx_to_name[v]}={w:.2f}" for v, w in zip(others, weights)),
            )

        return neg_probs

    def _tokenize_with_regionalism(
        self, text: str
    ) -> tuple[list[int], list[bool]]:
        """Tokenize text and propagate regionalism status from words to subwords."""
        words = text.split()
        all_ids: list[int] = []
        all_regs: list[bool] = []

        for word in words:
            is_reg = word.lower() in self.regionalism_set
            piece_ids = self._tokenizer.tokenize_word(word)
            all_ids.extend(piece_ids)
            all_regs.extend([is_reg] * len(piece_ids))

        return all_ids, all_regs

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, index: int) -> tuple[int, int, int, int, int, int]:
        """Return a single training sample as plain Python ints.

        The default collate_fn batches these into tensors efficiently.
        """
        row = self._data[index]
        return int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])

    def as_tensor_dataset(self) -> torch.utils.data.TensorDataset:
        """Convert to a TensorDataset for maximum DataLoader throughput.

        Returns a TensorDataset whose iteration yields the same 6-tuple
        but via pure tensor slicing (no Python __getitem__ per sample).
        """
        t = torch.from_numpy(self._data)
        return torch.utils.data.TensorDataset(
            t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4], t[:, 5],
        )

    def get_vocab_size(self) -> int:
        """Return the BPE vocabulary size."""
        return self.vocab_size
