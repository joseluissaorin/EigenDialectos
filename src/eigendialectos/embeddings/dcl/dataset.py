"""Skip-gram dataset with cross-variety negative sampling for DCL training.

Builds a shared vocabulary from all varieties, generates (word, context)
skip-gram pairs per variety, and provides cross-variety negative contexts.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import torch
from torch.utils.data import Dataset

from eigendialectos.embeddings.dcl.regionalisms import ALL_REGIONALISMS


class DCLDataset(Dataset):
    """Yields tuples for Dialect-Contrastive Loss training.

    Each sample is:
        ``(word_idx, ctx_idx_same, ctx_idx_other, variety_a, variety_b, is_regionalism)``

    Parameters
    ----------
    corpus_by_variety:
        Mapping from variety name (e.g. ``"ES_RIO"``) to a list of
        already-tokenised documents (each document is a single string
        of whitespace-separated tokens).
    window_size:
        Skip-gram context window radius (words on each side).
    neg_samples:
        Number of cross-variety negative context pairs generated per
        positive pair.  Currently each positive pair generates one
        cross-variety negative pair (the ``neg_samples`` parameter
        controls how many times a positive pair is repeated with
        different random cross-variety contexts).
    regionalism_set:
        Custom set of regionalism words.  If ``None``, the curated
        :data:`ALL_REGIONALISMS` is used.
    min_count:
        Minimum token frequency (across all varieties combined) for
        a word to be included in the shared vocabulary.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        corpus_by_variety: dict[str, list[str]],
        window_size: int = 5,
        neg_samples: int = 5,
        regionalism_set: set[str] | None = None,
        min_count: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.regionalism_set = regionalism_set if regionalism_set is not None else ALL_REGIONALISMS
        self._rng = random.Random(seed)

        # ------------------------------------------------------------------
        # 1. Build shared vocabulary across all varieties
        # ------------------------------------------------------------------
        self.variety_names: list[str] = sorted(corpus_by_variety.keys())
        self.variety_to_idx: dict[str, int] = {
            v: i for i, v in enumerate(self.variety_names)
        }

        # Count token frequencies across all varieties
        global_counts: Counter[str] = Counter()
        for docs in corpus_by_variety.values():
            for doc in docs:
                for token in doc.split():
                    global_counts[token] += 1

        # Filter by min_count and sort for determinism
        filtered_tokens = sorted(
            tok for tok, cnt in global_counts.items() if cnt >= min_count
        )
        self.vocab: list[str] = filtered_tokens
        self.word2idx: dict[str, int] = {w: i for i, w in enumerate(self.vocab)}
        self.vocab_size: int = len(self.vocab)

        # Pre-compute which vocab indices are regionalisms
        self._regionalism_indices: set[int] = {
            self.word2idx[w]
            for w in self.regionalism_set
            if w in self.word2idx
        }

        # ------------------------------------------------------------------
        # 2. Build per-variety tokenised corpora (as index lists)
        # ------------------------------------------------------------------
        self._variety_token_lists: dict[int, list[list[int]]] = {}
        for variety_name, docs in corpus_by_variety.items():
            v_idx = self.variety_to_idx[variety_name]
            indexed_docs: list[list[int]] = []
            for doc in docs:
                indexed = [
                    self.word2idx[tok]
                    for tok in doc.split()
                    if tok in self.word2idx
                ]
                if len(indexed) >= 2:
                    indexed_docs.append(indexed)
            self._variety_token_lists[v_idx] = indexed_docs

        # ------------------------------------------------------------------
        # 3. Generate skip-gram pairs per variety
        # ------------------------------------------------------------------
        # Each entry: (word_idx, ctx_idx, variety_idx)
        self._skipgram_pairs: list[tuple[int, int, int]] = []
        for v_idx, doc_lists in self._variety_token_lists.items():
            for doc_tokens in doc_lists:
                n = len(doc_tokens)
                for i, center in enumerate(doc_tokens):
                    left = max(0, i - self.window_size)
                    right = min(n, i + self.window_size + 1)
                    for j in range(left, right):
                        if j == i:
                            continue
                        self._skipgram_pairs.append(
                            (center, doc_tokens[j], v_idx)
                        )

        # ------------------------------------------------------------------
        # 4. Build per-variety token pools for cross-variety sampling
        # ------------------------------------------------------------------
        # Flattened list of all token indices per variety (for random draws)
        self._variety_token_pools: dict[int, list[int]] = {}
        for v_idx, doc_lists in self._variety_token_lists.items():
            pool: list[int] = []
            for doc_tokens in doc_lists:
                pool.extend(doc_tokens)
            self._variety_token_pools[v_idx] = pool

        self._all_variety_indices = list(self._variety_token_pools.keys())

        # ------------------------------------------------------------------
        # 5. Expand pairs by neg_samples (each positive pair gets
        #    neg_samples cross-variety negative draws at __getitem__ time)
        # ------------------------------------------------------------------
        self._total_len = len(self._skipgram_pairs) * self.neg_samples

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, index: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return a single training sample.

        Returns
        -------
        tuple of six tensors (all scalar / 0-d):
            ``(word_idx, ctx_idx_same, ctx_idx_other, variety_a, variety_b, is_regionalism)``
        """
        # Map expanded index back to the base skip-gram pair
        pair_idx = index // self.neg_samples
        word_idx, ctx_idx_same, variety_a = self._skipgram_pairs[pair_idx]

        # Pick a DIFFERENT variety for the cross-variety negative
        other_varieties = [
            v for v in self._all_variety_indices if v != variety_a
        ]
        if len(other_varieties) == 0:
            # Degenerate case: only one variety present
            variety_b = variety_a
        else:
            variety_b = self._rng.choice(other_varieties)

        # Sample a random context token from variety B
        pool_b = self._variety_token_pools.get(variety_b, [])
        if len(pool_b) > 0:
            ctx_idx_other = self._rng.choice(pool_b)
        else:
            # Fallback: use a random vocabulary index
            ctx_idx_other = self._rng.randint(0, self.vocab_size - 1)

        is_reg = word_idx in self._regionalism_indices

        return (
            torch.tensor(word_idx, dtype=torch.long),
            torch.tensor(ctx_idx_same, dtype=torch.long),
            torch.tensor(ctx_idx_other, dtype=torch.long),
            torch.tensor(variety_a, dtype=torch.long),
            torch.tensor(variety_b, dtype=torch.long),
            torch.tensor(is_reg, dtype=torch.bool),
        )

    def get_vocab(self) -> list[str]:
        """Return the shared vocabulary list."""
        return list(self.vocab)

    def get_word2idx(self) -> dict[str, int]:
        """Return the word-to-index mapping."""
        return dict(self.word2idx)
