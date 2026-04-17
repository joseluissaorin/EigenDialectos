"""Spectral fingerprinting: embed text -> project onto eigenmodes -> P(dialect|text).

Pipeline:
    1. Embed input text as bag-of-words centroid in shared embedding space.
    2. For each dialect's eigendecomposition, project the centroid onto the
       eigenvector matrix P to obtain a mode activation vector.
    3. Compare the activation profile to precomputed reference profiles for each
       dialect via cosine similarity.
    4. Apply softmax with temperature to produce a calibrated probability
       distribution over dialects.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np

from eigen3.types import EigenDecomp, ScoreResult
from eigen3.constants import ALL_VARIETIES, REFERENCE_VARIETY, REGIONALISMS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-záéíóúüñ]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer with basic punctuation stripping."""
    return _WORD_RE.findall(text.lower())


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two real-valued vectors.

    Returns 0.0 when either vector has zero norm.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling.

    Parameters
    ----------
    logits : np.ndarray
        Raw scores (1-D).
    temperature : float
        Temperature > 0.  Lower values sharpen the distribution;
        higher values flatten it toward uniform.

    Returns
    -------
    np.ndarray
        Probability vector that sums to 1.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - scaled.max()  # numerical stability
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


# ---------------------------------------------------------------------------
# DialectScorer
# ---------------------------------------------------------------------------

class DialectScorer:
    """Score and classify text by dialect using spectral fingerprinting.

    The scorer projects text embeddings onto eigenmode bases derived from
    per-dialect transformation matrices (W), then compares the resulting
    activation profiles to precomputed reference profiles.

    Parameters
    ----------
    embeddings : dict[str, np.ndarray]
        Per-variety embedding matrices keyed by dialect code.
        Each matrix has shape ``(vocab_size, dim)``.
    vocab : list[str]
        Shared vocabulary aligned with the embedding matrices' rows.
    decomps : dict[str, EigenDecomp]
        Per-variety eigendecompositions of W matrices, keyed by dialect code.
    reference : str
        The reference variety whose embedding space is used for text
        embedding.  Defaults to ``REFERENCE_VARIETY`` (ES_PEN).
    """

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        vocab: list[str],
        decomps: dict[str, EigenDecomp],
        reference: str = REFERENCE_VARIETY,
    ) -> None:
        self.embeddings = embeddings
        self.vocab = vocab
        self.decomps = decomps
        self.reference = reference

        # Build fast word -> index lookup
        self._word2idx: dict[str, int] = {w: i for i, w in enumerate(vocab)}

        # Embedding dimension (inferred from reference embeddings)
        self._dim: int = embeddings[reference].shape[1]

        # Reference embedding matrix used for text embedding
        self._ref_emb: np.ndarray = embeddings[reference]

        # Compute IDF weights for vocabulary (downweight common function words)
        self._idf_weights = self._compute_idf_weights(embeddings)

        # Pre-compute dialect reference profiles
        self._dialect_profiles: dict[str, np.ndarray] = {}
        self._compute_dialect_profiles()

        # Sorted variety list for scoring (consistent ordering)
        self._varieties_sorted: list[str] = sorted(self._dialect_profiles.keys())
        self._var_idx: dict[str, int] = {
            v: i for i, v in enumerate(self._varieties_sorted)
        }

        # Per-word dialect affinities based on embedding divergence
        self._word_affinities: np.ndarray = self._compute_word_affinities()

        logger.info(
            "DialectScorer initialized: %d varieties, vocab=%d, dim=%d",
            len(decomps), len(vocab), self._dim,
        )

    def _compute_idf_weights(self, embeddings: dict[str, np.ndarray]) -> np.ndarray:
        """Compute IDF-like weights based on embedding variance across varieties.

        Words that have similar embeddings across all varieties (function words
        like 'de', 'el', 'en') get low weight. Words that differ across varieties
        (regionalisms, dialectal markers) get high weight. This is a spectral
        IDF that uses embedding distances rather than document frequencies.
        """
        vocab_size = len(self.vocab)
        n_varieties = len(embeddings)

        if n_varieties < 2:
            return np.ones(vocab_size, dtype=np.float64)

        # For each word, compute variance of its embeddings across varieties
        varieties = sorted(embeddings.keys())
        all_embs = np.stack([embeddings[v] for v in varieties])  # (n_var, vocab, dim)
        # Variance across varieties per word: mean of per-dim variances
        var_per_word = all_embs.var(axis=0).mean(axis=1)  # (vocab,)

        # Transform to IDF-like weight: higher variance = more dialectally important
        # Use log-scaled variance + baseline to avoid zero weights
        weights = np.log1p(var_per_word * 1000) + 1.0
        # Normalize to mean=1
        weights = weights / weights.mean()

        return weights.astype(np.float64)

    def _compute_word_affinities(self) -> np.ndarray:
        """Per-word dialect affinity via embedding divergence from cross-variety mean.

        For each word, measures how much each variety's embedding vector
        differs from the average across all varieties.  Divergences are
        z-score-normalized per variety so that varieties with generally
        higher divergence (e.g. due to larger training corpora) do not
        dominate.  The result is softmax-normalized per word.

        Returns
        -------
        np.ndarray
            Shape ``(vocab_size, n_varieties)`` where columns follow the
            order of ``self._varieties_sorted``.
        """
        varieties = self._varieties_sorted
        n_var = len(varieties)
        vocab_size = len(self.vocab)

        if n_var < 2:
            return np.full((vocab_size, max(n_var, 1)), 1.0 / max(n_var, 1))

        # Stack all variety embeddings: (n_var, vocab_size, dim)
        all_embs = np.stack([self.embeddings[v] for v in varieties])

        # Cross-variety mean per word: (vocab_size, dim)
        mean_emb = all_embs.mean(axis=0)

        # L2 divergence of each variety from the mean: (n_var, vocab_size)
        divergences = np.linalg.norm(all_embs - mean_emb[np.newaxis], axis=2)

        # Z-score per variety: removes systematic bias where some varieties
        # have uniformly higher divergence (better-trained embeddings)
        var_mean = divergences.mean(axis=1, keepdims=True)   # (n_var, 1)
        var_std = divergences.std(axis=1, keepdims=True) + 1e-12
        z_div = (divergences - var_mean) / var_std            # (n_var, vocab_size)

        # Softmax on z-scores per word → probability-like affinities
        max_z = z_div.max(axis=0, keepdims=True)
        exp_z = np.exp(z_div - max_z)  # numerically stable
        affinities = exp_z / exp_z.sum(axis=0, keepdims=True)

        return affinities.T  # (vocab_size, n_var)

    # ------------------------------------------------------------------
    # Text embedding
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text as IDF-weighted bag-of-words centroid.

        Words that vary more across dialects (high spectral IDF) get higher
        weight, while common function words are downweighted. This focuses
        the embedding on dialectally significant vocabulary.

        Words not in the vocabulary are silently ignored. If no words match,
        returns the zero vector.
        """
        tokens = _tokenize(text)
        vectors: list[np.ndarray] = []
        weights: list[float] = []

        for token in tokens:
            idx = self._word2idx.get(token)
            if idx is not None:
                vectors.append(self._ref_emb[idx])
                weights.append(float(self._idf_weights[idx]))

        if not vectors:
            logger.warning("No vocabulary words found in text; returning zero vector.")
            return np.zeros(self._dim, dtype=np.float64)

        # IDF-weighted average
        w = np.array(weights, dtype=np.float64)
        w_sum = w.sum()
        if w_sum < 1e-12:
            centroid = np.mean(vectors, axis=0).astype(np.float64)
        else:
            centroid = np.average(vectors, axis=0, weights=w).astype(np.float64)
        return centroid

    # ------------------------------------------------------------------
    # Mode activation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_mode_activations(
        centroid: np.ndarray,
        decomp: EigenDecomp,
    ) -> np.ndarray:
        """Project a centroid onto the eigenvector columns of P.

        For each eigenvector column ``P[:, k]``, the activation is the
        projection coefficient: ``a_k = P_inv[k, :] @ centroid``.

        Since eigenvalues/vectors may be complex, we take the magnitude of
        each activation to produce a real-valued activation profile.

        Parameters
        ----------
        centroid : np.ndarray
            Text centroid of shape ``(dim,)``.
        decomp : EigenDecomp
            Eigendecomposition of the dialect's W matrix.

        Returns
        -------
        np.ndarray
            Real-valued activation vector of shape ``(n_modes,)``.
        """
        # P_inv @ centroid gives the coordinate in the eigenbasis
        activations = decomp.P_inv @ centroid.astype(np.complex128)
        return np.abs(activations).astype(np.float64)

    # ------------------------------------------------------------------
    # Dialect profile precomputation
    # ------------------------------------------------------------------

    def _compute_dialect_profiles(self) -> None:
        """Precompute reference activation profiles for each dialect.

        For each dialect, the profile is obtained by:
            1. Computing the centroid of that dialect's full embedding matrix.
            2. Projecting it onto the corresponding eigendecomposition.

        The resulting profile captures the "average spectral fingerprint" of
        each dialect and serves as the reference point for cosine comparison.
        """
        for variety, decomp in self.decomps.items():
            if variety not in self.embeddings:
                logger.warning(
                    "No embeddings for variety %s; skipping profile.", variety,
                )
                continue

            # Dialect centroid: mean of all word embeddings for this variety
            emb_matrix = self.embeddings[variety]  # (V, dim)
            dialect_centroid = np.mean(emb_matrix, axis=0).astype(np.float64)

            # Project onto eigenmodes
            profile = self._compute_mode_activations(dialect_centroid, decomp)
            self._dialect_profiles[variety] = profile

        logger.info(
            "Computed dialect profiles for %d varieties.", len(self._dialect_profiles),
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, text: str, temperature: float = 1.0) -> ScoreResult:
        """Score text against all known dialects.

        Uses a Naive-Bayes-style aggregation of per-word dialect
        affinities (derived from embedding divergence across varieties)
        combined with explicit regionalism bonuses.

        Parameters
        ----------
        text : str
            Raw input text to classify.
        temperature : float
            Softmax temperature (default 1.0).  Lower values produce
            sharper distributions; higher values produce flatter ones.

        Returns
        -------
        ScoreResult
            Contains ``probabilities`` (dict mapping dialect code to float),
            ``mode_activations`` (np.ndarray of activations under the top
            dialect), and ``top_dialect`` (str).
        """
        tokens = _tokenize(text)
        n_var = len(self._varieties_sorted)

        # Accumulate log-likelihoods from per-word affinities
        log_scores = np.zeros(n_var, dtype=np.float64)
        n_words = 0

        for token in tokens:
            idx = self._word2idx.get(token)
            if idx is None:
                continue
            idf = self._idf_weights[idx]
            affinity = self._word_affinities[idx]  # (n_var,)
            log_scores += idf * np.log(affinity + 1e-10)
            n_words += 1

        # Regionalism bonus (additive in log-space ≈ multiplicative likelihood)
        # Includes morphological matching: strip -s and -es for regular plurals
        _REGIONALISM_BONUS = 3.0
        has_regionalism = False
        for variety, regs in REGIONALISMS.items():
            vi = self._var_idx.get(variety)
            if vi is None:
                continue
            n_matches = 0
            seen: set[str] = set()
            for t in tokens:
                if t in seen:
                    continue
                seen.add(t)
                if t in regs:
                    n_matches += 1
                elif t.endswith("es") and len(t) > 3 and t[:-2] in regs:
                    n_matches += 1
                elif t.endswith("s") and len(t) > 2 and t[:-1] in regs:
                    n_matches += 1
            if n_matches:
                log_scores[vi] += n_matches * _REGIONALISM_BONUS
                has_regionalism = True

        if n_words == 0 and not has_regionalism:
            # No vocabulary words and no regionalism matches — uniform
            uniform = 1.0 / max(n_var, 1)
            probs_dict = {v: uniform for v in self._varieties_sorted}
            return ScoreResult(
                probabilities=probs_dict,
                mode_activations=np.zeros(self._dim, dtype=np.float64),
                top_dialect=(
                    self._varieties_sorted[0] if self._varieties_sorted else ""
                ),
            )

        # Convert log-scores to probability distribution
        probs_array = _softmax(log_scores, temperature=temperature)
        probabilities = {
            v: float(p) for v, p in zip(self._varieties_sorted, probs_array)
        }

        top_idx = int(np.argmax(probs_array))
        top_dialect = self._varieties_sorted[top_idx]

        # Mode activations via spectral projection (for visualization / explorer)
        centroid = self.embed_text(text)
        if top_dialect in self.decomps:
            mode_activations = self._compute_mode_activations(
                centroid, self.decomps[top_dialect],
            )
        else:
            mode_activations = np.zeros(self._dim, dtype=np.float64)

        return ScoreResult(
            probabilities=probabilities,
            mode_activations=mode_activations,
            top_dialect=top_dialect,
        )

    def batch_score(
        self,
        texts: list[str],
        temperature: float = 1.0,
    ) -> list[ScoreResult]:
        """Score multiple texts.

        Parameters
        ----------
        texts : list[str]
            Input texts to classify.
        temperature : float
            Softmax temperature applied to every text.

        Returns
        -------
        list[ScoreResult]
            One ``ScoreResult`` per input text, in the same order.
        """
        return [self.score(t, temperature=temperature) for t in texts]

    def classify(self, text: str) -> str:
        """Classify a text and return the top dialect code.

        Convenience wrapper around :meth:`score` that discards the full
        probability distribution and returns only the winning dialect.

        Parameters
        ----------
        text : str
            Raw input text to classify.

        Returns
        -------
        str
            Dialect code of the highest-probability variety.
        """
        return self.score(text).top_dialect

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def score_detailed(
        self,
        text: str,
        temperature: float = 1.0,
    ) -> dict:
        """Return detailed scoring internals for debugging.

        Returns a dictionary with:
            - ``centroid``: text embedding centroid
            - ``tokens_found``: number of vocabulary words matched
            - ``tokens_total``: number of tokens in text
            - ``activations``: per-variety activation vectors
            - ``similarities``: per-variety cosine similarities
            - ``result``: the ``ScoreResult``
        """
        tokens = _tokenize(text)
        tokens_found = sum(1 for t in tokens if t in self._word2idx)
        centroid = self.embed_text(text)

        activations: dict[str, np.ndarray] = {}
        similarities: dict[str, float] = {}

        for variety, profile in self._dialect_profiles.items():
            if variety not in self.decomps:
                continue
            decomp = self.decomps[variety]
            act = self._compute_mode_activations(centroid, decomp)
            activations[variety] = act
            similarities[variety] = _cosine_similarity(act, profile)

        result = self.score(text, temperature=temperature)

        return {
            "centroid": centroid,
            "tokens_found": tokens_found,
            "tokens_total": len(tokens),
            "activations": activations,
            "similarities": similarities,
            "result": result,
        }

    def top_k_dialects(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k dialects by probability.

        Parameters
        ----------
        text : str
            Raw input text.
        k : int
            Number of top dialects to return.

        Returns
        -------
        list[tuple[str, float]]
            List of ``(dialect_code, probability)`` pairs sorted descending.
        """
        result = self.score(text)
        sorted_probs = sorted(
            result.probabilities.items(), key=lambda x: x[1], reverse=True,
        )
        return sorted_probs[:k]

    def confusion_matrix(
        self,
        texts: list[str],
        labels: list[str],
    ) -> np.ndarray:
        """Build a confusion matrix from labeled examples.

        Parameters
        ----------
        texts : list[str]
            Input texts.
        labels : list[str]
            Ground-truth dialect codes aligned with ``texts``.

        Returns
        -------
        np.ndarray
            Confusion matrix of shape ``(n_varieties, n_varieties)`` where
            rows are true labels and columns are predicted labels.
            Variety ordering follows ``ALL_VARIETIES``.
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) must have "
                f"the same length.",
            )

        varieties = [v for v in ALL_VARIETIES if v in self.decomps]
        variety_idx = {v: i for i, v in enumerate(varieties)}
        n = len(varieties)
        cm = np.zeros((n, n), dtype=np.int64)

        for text, true_label in zip(texts, labels):
            if true_label not in variety_idx:
                continue
            pred = self.classify(text)
            if pred not in variety_idx:
                continue
            cm[variety_idx[true_label], variety_idx[pred]] += 1

        return cm

    def accuracy(self, texts: list[str], labels: list[str]) -> float:
        """Compute classification accuracy.

        Parameters
        ----------
        texts : list[str]
            Input texts.
        labels : list[str]
            Ground-truth dialect codes.

        Returns
        -------
        float
            Fraction of texts correctly classified.
        """
        if not texts:
            return 0.0
        correct = sum(
            1 for text, label in zip(texts, labels) if self.classify(text) == label
        )
        return correct / len(texts)

    def get_dialect_profile(self, variety: str) -> np.ndarray | None:
        """Return the precomputed reference profile for a dialect.

        Returns
        -------
        np.ndarray or None
            Activation profile, or None if the variety is not available.
        """
        return self._dialect_profiles.get(variety)

    def available_varieties(self) -> list[str]:
        """Return the list of varieties this scorer can classify against."""
        return list(self._dialect_profiles.keys())

    def __repr__(self) -> str:
        return (
            f"DialectScorer(varieties={len(self._dialect_profiles)}, "
            f"vocab={len(self.vocab)}, dim={self._dim}, "
            f"reference={self.reference!r})"
        )
