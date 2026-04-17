"""Core transform engine for the SDC compiler.

For each linguistic level:
1. Embed units using per-level embedding lookup
2. Use dual-path search: direct cross-variety similarity + spectral-biased search
3. Score candidates by semantic fidelity, novelty, context, and eigenvalue analysis
4. Return best replacement with full traceability metadata

The spectral transform W(α) = P Λ^α P^{-1} can be numerically unstable when
the eigenvector matrix P is ill-conditioned (cond > 1e10).  In those cases we
fall back to a direct embedding-similarity approach, using the eigenvalue
analysis only for traceability metadata.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from eigendialectos.spectral.stack import SpectralStack
from eigendialectos.types import LevelEmbedding

logger = logging.getLogger(__name__)

# Condition-number threshold: above this we consider the transform unreliable
_COND_THRESHOLD = 1e10
# Minimum cosine similarity for a kNN candidate to be accepted
_MIN_COSINE_SIM = 0.25
# Maximum ratio of ||transformed|| / ||source|| before we declare numerical blow-up
_MAX_NORM_RATIO = 50.0

# Function words / stop words that should NEVER be transformed.
# Embeddings for these are noisy across varieties — transforming them
# produces garbage because their meaning is purely syntactic.
_STOP_WORDS = frozenset({
    # Determiners
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "lo", "al", "del",
    # Prepositions
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde",
    "en", "entre", "hacia", "hasta", "para", "por", "según",
    "sin", "so", "sobre", "tras",
    # Conjunctions
    "y", "e", "o", "u", "ni", "que", "si", "pero", "mas", "sino",
    "aunque", "porque", "como", "cuando", "donde",
    # Pronouns (subject/object)
    "yo", "tú", "él", "ella", "nosotros", "nosotras", "vosotros",
    "vosotras", "ellos", "ellas", "usted", "ustedes",
    "me", "te", "se", "nos", "os", "le", "les",
    "mi", "ti", "mí",
    # Possessives
    "su", "sus", "mi", "mis", "tu", "tus", "nuestro", "nuestra",
    "nuestros", "nuestras", "vuestro", "vuestra",
    # Demonstratives
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas",
    "esto", "eso", "aquello",
    # Common adverbs
    "no", "sí", "ya", "muy", "más", "menos", "bien", "mal",
    "también", "tampoco", "ahora", "aquí", "allí", "así",
    # Auxiliaries / modals
    "ser", "estar", "haber", "tener", "ir",
    "es", "está", "hay", "son", "están", "era", "fue",
    "ha", "he", "has", "han", "hemos",
    # Common verbs too generic to transform
    "ser", "estar", "tener", "hacer", "poder", "decir", "dar",
    "ver", "saber", "querer", "llegar", "pasar",
})


class SpectralTransformer:
    """Transform engine: spectral-biased kNN lookup in target space.

    Uses a dual-path approach:
    - **Spectral path**: Apply W(α) and search near the transformed vector.
      Only used when the eigenvector matrix is well-conditioned.
    - **Direct path**: Find target words whose embeddings are closest to the
      source embedding.  Always available as a fallback.

    The final candidate list merges both paths before scoring.

    Parameters
    ----------
    spectral_stack : SpectralStack
        Pre-fitted spectral stack with per-level eigendecompositions.
    source_embeddings : dict mapping level int to source LevelEmbedding
    target_embeddings : dict mapping level int to target LevelEmbedding
    """

    def __init__(
        self,
        spectral_stack: SpectralStack,
        source_embeddings: dict[int, LevelEmbedding],
        target_embeddings: dict[int, LevelEmbedding],
    ) -> None:
        self.stack = spectral_stack
        self.source_emb = source_embeddings
        self.target_emb = target_embeddings

        # Pre-compute normalized target vectors for cosine similarity
        self._target_normed: dict[int, npt.NDArray[np.float64]] = {}
        for level, emb in target_embeddings.items():
            norms = np.linalg.norm(emb.vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self._target_normed[level] = emb.vectors / norms

        # Check transform reliability per level
        self._transform_reliable: dict[int, bool] = {}
        for level in spectral_stack.level_eigen:
            eigen = spectral_stack.level_eigen[level]
            P = eigen.eigenvectors
            cond = np.linalg.cond(P.real if np.iscomplexobj(P) else P)
            reliable = cond < _COND_THRESHOLD
            self._transform_reliable[level] = reliable
            if not reliable:
                logger.warning(
                    "Level %d: eigenvector matrix cond=%.2e > threshold %.2e; "
                    "using direct similarity (spectral transform unreliable).",
                    level, cond, _COND_THRESHOLD,
                )

    def transform_level(
        self,
        level: int,
        units: list[str],
        alpha: float,
        context: Optional[list[str]] = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Transform units at one level, returning (replacement, metadata) pairs.

        Parameters
        ----------
        level : int
            Linguistic level (1-5).
        units : list[str]
            Units to transform at this level.
        alpha : float
            Dialectal intensity (0=identity, 1=full).
        context : optional list of surrounding units

        Returns
        -------
        list of (replacement_string, metadata_dict) tuples.
        """
        if level not in self.source_emb or level not in self.target_emb:
            return [(u, {"changed": False, "reason": "no_embeddings"}) for u in units]

        src = self.source_emb[level]
        tgt = self.target_emb[level]
        use_spectral = self._transform_reliable.get(level, False)
        results: list[tuple[str, dict[str, Any]]] = []

        for unit in units:
            unit_lower = unit.lower()

            # Skip function words — their embeddings are too noisy for
            # cross-variety transformation
            if unit_lower in _STOP_WORDS:
                results.append((unit, {"changed": False, "reason": "stop_word"}))
                continue

            # Look up source embedding
            if unit_lower not in src.vocabulary:
                results.append((unit, {"changed": False, "reason": "oov_source"}))
                continue

            src_idx = src.vocabulary[unit_lower]
            src_vec = src.vectors[src_idx]  # (dim,)

            # --- Direct path: kNN on source vector in target space ---
            direct_candidates = self.knn_search(src_vec, tgt, k=15)

            # --- Spectral path: transform then kNN (only if reliable) ---
            spectral_candidates: list[tuple[str, float]] = []
            transformed = src_vec  # default: no transform
            if use_spectral and level in self.stack.level_eigen:
                try:
                    t_vec = self.stack.transform(level, src_vec, alpha=alpha)
                    # Validate: check for numerical blow-up
                    src_norm = np.linalg.norm(src_vec)
                    t_norm = np.linalg.norm(t_vec)
                    if src_norm > 1e-10 and t_norm / src_norm < _MAX_NORM_RATIO:
                        transformed = t_vec
                        spectral_candidates = self.knn_search(t_vec, tgt, k=15)
                    else:
                        logger.debug(
                            "Norm blow-up for '%s': ||t||/||s|| = %.1f; using direct path.",
                            unit_lower, t_norm / max(src_norm, 1e-10),
                        )
                except Exception:
                    logger.debug("Spectral transform failed for '%s'; using direct.", unit_lower)

            # Merge candidate lists (deduplicate, keep best similarity)
            merged = self._merge_candidates(direct_candidates, spectral_candidates)

            # Filter: minimum cosine similarity, exclude stop words from results
            merged = [
                (w, s) for w, s in merged
                if s >= _MIN_COSINE_SIM and w.lower() not in _STOP_WORDS
            ]

            # Score candidates
            scored = self.score_candidates(
                merged, unit_lower, context or [], level, alpha
            )

            if not scored:
                results.append((unit, {"changed": False, "reason": "no_candidates"}))
                continue

            best_word, best_score = scored[0]

            # If best candidate is the same word, keep original
            if best_word.lower() == unit_lower:
                results.append((unit, {"changed": False, "reason": "no_change"}))
                continue

            # Determine the responsible eigenvector (for traceability)
            eigen_info = self._identify_eigenvector(src_vec, transformed, level)

            # Preserve original casing pattern
            replacement = self._match_case(unit, best_word)

            metadata: dict[str, Any] = {
                "changed": True,
                "original": unit,
                "replacement": replacement,
                "score": best_score,
                "confidence": best_score,
                "level": level,
                "alpha": alpha,
                "candidates_considered": len(scored),
                "used_spectral": use_spectral and len(spectral_candidates) > 0,
                **eigen_info,
            }

            results.append((replacement, metadata))

        return results

    @staticmethod
    def _merge_candidates(
        list_a: list[tuple[str, float]],
        list_b: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Merge two candidate lists, keeping the higher similarity for duplicates."""
        merged: dict[str, float] = {}
        for word, sim in list_a:
            w = word.lower()
            merged[w] = max(merged.get(w, -1.0), sim)
        for word, sim in list_b:
            w = word.lower()
            merged[w] = max(merged.get(w, -1.0), sim)
        result = sorted(merged.items(), key=lambda x: -x[1])
        return result

    def knn_search(
        self,
        query_vector: npt.NDArray[np.float64],
        target_emb: LevelEmbedding,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Find k nearest neighbors in target embedding space via cosine similarity.

        Parameters
        ----------
        query_vector : ndarray, shape (dim,)
        target_emb : LevelEmbedding
        k : int

        Returns
        -------
        list of (word, cosine_similarity) sorted descending.
        """
        level = target_emb.level
        query_norm = np.linalg.norm(query_vector)
        if query_norm < 1e-10:
            return []

        query_normed = query_vector / query_norm

        if level in self._target_normed:
            normed = self._target_normed[level]
        else:
            norms = np.linalg.norm(target_emb.vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            normed = target_emb.vectors / norms

        # Cosine similarity: dot product of normalized vectors
        similarities = normed @ query_normed  # (vocab_size,)

        # Top k
        actual_k = min(k, len(similarities) - 1)
        if actual_k < 1:
            return []
        top_k_idx = np.argpartition(-similarities, actual_k)[:k]
        top_k_idx = top_k_idx[np.argsort(-similarities[top_k_idx])]

        results = [
            (target_emb.labels[i], float(similarities[i]))
            for i in top_k_idx
        ]

        return results

    def score_candidates(
        self,
        candidates: list[tuple[str, float]],
        original: str,
        context: list[str],
        level: int,
        alpha: float,
    ) -> list[tuple[str, float]]:
        """Score candidates by multiple criteria.

        Scoring:
        1. Cosine similarity (semantic fidelity) — weight 0.5
        2. Morphological similarity (shared prefix/suffix) — weight 0.2
        3. Novelty: different from original — weight 0.1
        4. Alpha-gated — weight 0.2

        Returns only candidates with combined score >= 0.35.
        """
        if not candidates:
            return []

        context_set = set(w.lower() for w in context)
        scored: list[tuple[str, float]] = []

        for word, cos_sim in candidates:
            w_lower = word.lower()
            is_same = w_lower == original.lower()

            # Morphological similarity: shared prefix or suffix
            morph_sim = self._morphological_similarity(original.lower(), w_lower)

            novelty = 0.0 if is_same else 0.15
            ctx_bonus = 0.05 if w_lower in context_set else 0.0

            alpha_penalty = 0.0
            if alpha < 0.3 and not is_same:
                alpha_penalty = -0.4 * (0.3 - alpha)

            combined = (
                0.5 * cos_sim
                + 0.2 * morph_sim
                + 0.1 * novelty
                + 0.2 * alpha
                + ctx_bonus
                + alpha_penalty
            )

            scored.append((word, combined))

        scored.sort(key=lambda x: -x[1])

        # Filter out low-quality matches
        scored = [(w, s) for w, s in scored if s >= 0.35]

        return scored

    @staticmethod
    def _morphological_similarity(a: str, b: str) -> float:
        """Compute morphological similarity as shared-prefix ratio + suffix bonus.

        Words that share a long common prefix (same stem) or ending (same
        conjugation/derivation) are more likely to be valid dialectal variants.
        """
        if not a or not b:
            return 0.0

        # Shared prefix length
        prefix_len = 0
        for ca, cb in zip(a, b):
            if ca == cb:
                prefix_len += 1
            else:
                break

        prefix_ratio = prefix_len / max(len(a), len(b))

        # Shared suffix (last 3 chars)
        suffix_match = 0.0
        for i in range(1, min(4, min(len(a), len(b)) + 1)):
            if a[-i] == b[-i]:
                suffix_match += 1.0 / 3.0
            else:
                break

        # Length similarity (penalize very different lengths)
        len_ratio = min(len(a), len(b)) / max(len(a), len(b))

        return 0.5 * prefix_ratio + 0.3 * suffix_match + 0.2 * len_ratio

    def _identify_eigenvector(
        self,
        src_vec: npt.NDArray[np.float64],
        transformed: npt.NDArray[np.float64],
        level: int,
    ) -> dict[str, Any]:
        """Identify which eigenvector contributed most to the transformation.

        Projects the change vector onto each eigenvector and finds the one
        with maximum projection.
        """
        if level not in self.stack.level_eigen:
            return {"eigenvector_idx": -1, "eigenvalue": 0.0, "eigenvector_contribution": 0.0}

        eigen = self.stack.level_eigen[level]
        change = transformed - src_vec

        change_norm = np.linalg.norm(change)
        if change_norm < 1e-10:
            return {"eigenvector_idx": -1, "eigenvalue": 0.0, "eigenvector_contribution": 0.0}

        # Project change onto each eigenvector
        projections = np.abs(np.real(eigen.eigenvectors.T @ change.astype(np.complex128)))
        best_idx = int(np.argmax(projections))

        return {
            "eigenvector_idx": best_idx,
            "eigenvalue": complex(eigen.eigenvalues[best_idx]),
            "eigenvector_contribution": float(projections[best_idx] / change_norm),
        }

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        """Preserve the casing pattern of the original in the replacement."""
        if original.isupper():
            return replacement.upper()
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement
