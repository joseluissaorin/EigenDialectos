"""Compose word-level embeddings from trained subword (BPE) embeddings.

After subword-level DCL training produces per-variety subword embedding
tables, this module composes word-level vectors by mean-pooling the
constituent subword embeddings.  The result is a vocabulary where
*every word in every variety* has an embedding — no intersection
bottleneck.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SubwordToWordComposer:
    """Compose word embeddings from subword embeddings via mean pooling.

    Parameters
    ----------
    tokenizer:
        A trained :class:`SharedSubwordTokenizer`.
    subword_embeddings:
        Dict mapping variety name to ``(bpe_vocab_size, embedding_dim)``
        numpy array of trained subword embeddings.
    """

    def __init__(
        self,
        tokenizer,  # SharedSubwordTokenizer
        subword_embeddings: dict[str, np.ndarray],
    ) -> None:
        self._tokenizer = tokenizer
        self._subword_embs = subword_embeddings

    def compose_word(self, word: str, variety: str) -> np.ndarray:
        """Compose a single word's embedding for a given variety.

        Returns the mean of its constituent subword embeddings.
        """
        piece_ids = self._tokenizer.tokenize_word(word)
        if not piece_ids:
            # Fallback: return zero vector
            dim = next(iter(self._subword_embs.values())).shape[1]
            return np.zeros(dim, dtype=np.float32)
        emb_table = self._subword_embs[variety]
        # Clamp indices to valid range
        piece_ids = [min(pid, emb_table.shape[0] - 1) for pid in piece_ids]
        piece_vecs = emb_table[piece_ids]  # (n_pieces, dim)
        return piece_vecs.mean(axis=0).astype(np.float32)

    def compose_vocabulary(
        self,
        word_list: list[str],
        variety: str,
    ) -> np.ndarray:
        """Compose embeddings for an entire vocabulary.

        Parameters
        ----------
        word_list:
            List of words to compose.
        variety:
            Variety code (e.g. ``"ES_RIO"``).

        Returns
        -------
        np.ndarray
            ``(len(word_list), embedding_dim)`` float32 matrix.
        """
        vectors = np.array(
            [self.compose_word(w, variety) for w in word_list],
            dtype=np.float32,
        )
        return vectors


def build_union_vocabulary(
    corpus_by_variety: dict[str, list[str]],
    min_count: int = 3,
) -> list[str]:
    """Build a union vocabulary from all varieties.

    A word is included if it appears at least ``min_count`` times in
    the combined corpus of *any* single variety.  This avoids the
    intersection bottleneck that discards dialect-specific words.

    Parameters
    ----------
    corpus_by_variety:
        Mapping from variety name to list of text documents.
    min_count:
        Minimum occurrences in any single variety for inclusion.

    Returns
    -------
    list[str]
        Sorted list of vocabulary words.
    """
    from collections import Counter

    per_variety_counts: dict[str, Counter] = {}
    for variety, docs in corpus_by_variety.items():
        counter: Counter = Counter()
        for doc in docs:
            for word in doc.strip().lower().split():
                word = word.strip(".,;:!?¿¡\"'()[]{}…-—–")
                if word:
                    counter[word] += 1
        per_variety_counts[variety] = counter

    # Union: include word if it reaches min_count in ANY variety
    vocab_set: set[str] = set()
    for counter in per_variety_counts.values():
        for word, count in counter.items():
            if count >= min_count:
                vocab_set.add(word)

    vocab = sorted(vocab_set)
    logger.info(
        "Union vocabulary: %d words (min_count=%d, %d varieties)",
        len(vocab), min_count, len(corpus_by_variety),
    )
    return vocab
