"""N-gram language model perplexity evaluation for dialect fidelity."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from eigendialectos.constants import DialectCode
from eigendialectos.types import CorpusSlice


# ======================================================================
# Simple n-gram language model (from scratch)
# ======================================================================

class NgramLM:
    """Add-k smoothed n-gram language model.

    Parameters
    ----------
    n : int
        N-gram order (e.g., 3 for trigrams).
    k : float
        Additive smoothing constant (default 0.01).
    """

    def __init__(self, n: int = 3, k: float = 0.01) -> None:
        self.n = n
        self.k = k
        self._counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._context_totals: dict[tuple[str, ...], int] = defaultdict(int)
        self._vocab: set[str] = set()

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def train(self, texts: list[str]) -> None:
        """Train the LM on a list of texts."""
        for text in texts:
            tokens = self._tokenize(text)
            # Pad the beginning with (n-1) <s> tokens
            padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1 : i])
                word = padded[i]
                self._counts[context][word] += 1
                self._context_totals[context] += 1
                self._vocab.add(word)
        # Ensure at least a minimal vocab so we never divide by zero
        self._vocab.add("<unk>")

    # ---------------------------------------------------------
    # Probability
    # ---------------------------------------------------------

    def log_prob(self, token: str, context: tuple[str, ...]) -> float:
        """Log probability of *token* given *context* (add-k smoothed)."""
        count = self._counts[context].get(token, 0)
        total = self._context_totals.get(context, 0)
        V = len(self._vocab)
        prob = (count + self.k) / (total + self.k * V)
        return math.log(prob)

    def sequence_log_prob(self, text: str) -> float:
        """Total log probability of *text* under this LM."""
        tokens = self._tokenize(text)
        padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        total_log_p = 0.0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1 : i])
            word = padded[i]
            total_log_p += self.log_prob(word, context)
        return total_log_p

    # ---------------------------------------------------------
    # Perplexity
    # ---------------------------------------------------------

    def perplexity(self, text: str) -> float:
        """Perplexity of *text* under this LM."""
        tokens = self._tokenize(text)
        N = len(tokens) + 1  # +1 for </s>
        if N == 0:
            return float("inf")
        log_p = self.sequence_log_prob(text)
        return math.exp(-log_p / N)

    # ---------------------------------------------------------
    # Tokenisation (simple whitespace-based)
    # ---------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()


# ======================================================================
# Perplexity evaluator
# ======================================================================

class PerplexityEvaluator:
    """Evaluate dialectal fidelity using per-dialect n-gram language models.

    Parameters
    ----------
    lm_per_dialect : dict[DialectCode, NgramLM] or None
        Pre-trained LMs.  If ``None`` you must call :meth:`build_ngram_lms`
        before evaluating.
    """

    def __init__(
        self,
        lm_per_dialect: dict[DialectCode, Any] | None = None,
    ) -> None:
        self.lm_per_dialect: dict[DialectCode, NgramLM] = lm_per_dialect or {}

    # ---------------------------------------------------------
    # Build LMs
    # ---------------------------------------------------------

    def build_ngram_lms(
        self,
        train_data: dict[DialectCode, CorpusSlice],
        n: int = 3,
    ) -> None:
        """Train one n-gram LM per dialect.

        Parameters
        ----------
        train_data : dict[DialectCode, CorpusSlice]
            Training corpus slices keyed by dialect.
        n : int
            N-gram order.
        """
        self.lm_per_dialect = {}
        for dialect, corpus_slice in train_data.items():
            lm = NgramLM(n=n)
            texts = [s.text for s in corpus_slice.samples]
            lm.train(texts)
            self.lm_per_dialect[dialect] = lm

    # ---------------------------------------------------------
    # Perplexity queries
    # ---------------------------------------------------------

    def compute_perplexity(self, text: str, dialect: DialectCode) -> float:
        """Perplexity of *text* under the language model for *dialect*.

        Parameters
        ----------
        text : str
        dialect : DialectCode

        Returns
        -------
        float
            Perplexity value (lower means the text is more likely under that model).

        Raises
        ------
        KeyError
            If no LM has been built for *dialect*.
        """
        lm = self.lm_per_dialect.get(dialect)
        if lm is None:
            raise KeyError(f"No language model available for dialect {dialect.value}")
        return lm.perplexity(text)

    def cross_dialect_perplexity(self, text: str) -> dict[DialectCode, float]:
        """Compute perplexity of *text* under every available dialect LM.

        Parameters
        ----------
        text : str

        Returns
        -------
        dict[DialectCode, float]
        """
        return {
            dialect: lm.perplexity(text)
            for dialect, lm in self.lm_per_dialect.items()
        }

    # ---------------------------------------------------------
    # Dialect fidelity
    # ---------------------------------------------------------

    def evaluate_dialect_fidelity(
        self,
        texts: dict[DialectCode, list[str]],
    ) -> dict[str, Any]:
        """For each text, check if the lowest-perplexity dialect matches the expected one.

        Parameters
        ----------
        texts : dict[DialectCode, list[str]]
            Texts keyed by their expected dialect.

        Returns
        -------
        dict
            ``total_correct``, ``total``, ``accuracy``, per-dialect breakdown.
        """
        total = 0
        correct = 0
        per_dialect: dict[str, dict[str, Any]] = {}

        for expected_dialect, text_list in texts.items():
            d_total = 0
            d_correct = 0
            for text in text_list:
                pp_map = self.cross_dialect_perplexity(text)
                if not pp_map:
                    continue
                best = min(pp_map, key=lambda d: pp_map[d])
                d_total += 1
                if best == expected_dialect:
                    d_correct += 1
            per_dialect[expected_dialect.value] = {
                "correct": d_correct,
                "total": d_total,
                "accuracy": d_correct / d_total if d_total > 0 else 0.0,
            }
            total += d_total
            correct += d_correct

        return {
            "total_correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0.0,
            "per_dialect": per_dialect,
        }
