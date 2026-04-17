"""Fisher Information Matrix analysis for dialect identification.

FIM eigenvalues identify which dimensions of the embedding space carry
the most signal for distinguishing between dialect varieties. This is
the information-theoretic complement to the algebraic eigenvalues from
transformation matrices.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from eigendialectos.types import FisherResult

logger = logging.getLogger(__name__)


class FisherInformationAnalysis:
    """Fisher Information analysis for dialect discrimination.

    Treats dialect identification as a classification problem and
    computes the FIM to identify which embedding dimensions carry
    the most discriminative signal.
    """

    def compute_fim(
        self,
        embeddings: dict[str, npt.NDArray[np.float64]],
        vocabulary: list[str] | None = None,
    ) -> FisherResult:
        """Compute the Fisher Information Matrix for dialect classification.

        Uses Linear Discriminant Analysis (LDA) formulation:
        FIM ≈ S_w^{-1} S_b

        where S_b is the between-class scatter and S_w is the within-class
        scatter. The eigenvalues of this matrix indicate which directions
        in embedding space best separate the dialect classes.

        Parameters
        ----------
        embeddings : dict
            Mapping from dialect name to embedding matrix, shape (n_words, dim).
            All matrices must have the same n_words and dim.
        vocabulary : list[str] or None
            Word labels corresponding to rows.

        Returns
        -------
        FisherResult
        """
        dialect_names = sorted(embeddings.keys())
        n_classes = len(dialect_names)

        if n_classes < 2:
            raise ValueError("Need at least 2 dialect varieties for FIM analysis")

        # Stack all embeddings: each class contributes its full embedding matrix
        # Treat each word position as a sample, each dialect as a class
        # E_i has shape (n_words, dim)
        first_key = dialect_names[0]
        n_words, dim = embeddings[first_key].shape

        # Global mean across all classes and words
        all_data = np.stack([embeddings[d] for d in dialect_names], axis=0)  # (n_classes, n_words, dim)
        global_mean = all_data.mean(axis=(0, 1))  # (dim,)

        # Between-class scatter: S_b = Σ_c n_c (μ_c - μ)(μ_c - μ)^T
        S_b = np.zeros((dim, dim), dtype=np.float64)
        class_means: dict[str, npt.NDArray] = {}
        for d in dialect_names:
            mu_c = embeddings[d].mean(axis=0)  # (dim,)
            class_means[d] = mu_c
            diff = mu_c - global_mean
            S_b += n_words * np.outer(diff, diff)

        # Within-class scatter: S_w = Σ_c Σ_i (x_i^c - μ_c)(x_i^c - μ_c)^T
        S_w = np.zeros((dim, dim), dtype=np.float64)
        for d in dialect_names:
            centered = embeddings[d] - class_means[d]  # (n_words, dim)
            S_w += centered.T @ centered  # (dim, dim)

        # Regularize S_w for invertibility
        S_w_reg = S_w + 1e-6 * np.eye(dim)

        # FIM ≈ S_w^{-1} @ S_b
        try:
            S_w_inv = np.linalg.inv(S_w_reg)
        except np.linalg.LinAlgError:
            S_w_inv = np.linalg.pinv(S_w_reg)

        fim = S_w_inv @ S_b

        # Eigendecompose FIM
        fim_eigenvalues, fim_eigenvectors = np.linalg.eig(fim)

        # Sort by magnitude (descending)
        order = np.argsort(-np.abs(fim_eigenvalues))
        fim_eigenvalues = np.real(fim_eigenvalues[order])
        fim_eigenvectors = np.real(fim_eigenvectors[:, order])

        # Find most diagnostic words: project each word's variance onto top FIM eigenvectors
        most_diagnostic = self._find_diagnostic_words(
            embeddings, fim_eigenvectors, vocabulary, top_k=20
        )

        logger.info(
            "FIM: top 5 eigenvalues = %s, condition number = %.2e",
            fim_eigenvalues[:5],
            float(fim_eigenvalues[0] / max(fim_eigenvalues[-1], 1e-15)),
        )

        return FisherResult(
            fim=fim.astype(np.float64),
            fim_eigenvalues=fim_eigenvalues.astype(np.float64),
            fim_eigenvectors=fim_eigenvectors.astype(np.float64),
            most_diagnostic=most_diagnostic,
        )

    def _find_diagnostic_words(
        self,
        embeddings: dict[str, npt.NDArray[np.float64]],
        fim_eigenvectors: npt.NDArray[np.float64],
        vocabulary: list[str] | None,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """Find words with highest variance along top FIM directions.

        For each word, compute how much its cross-dialect variance
        aligns with the top FIM eigenvectors (most discriminative directions).

        Parameters
        ----------
        embeddings : dict of dialect -> (n_words, dim) arrays
        fim_eigenvectors : (dim, dim) matrix, columns sorted by eigenvalue
        vocabulary : optional word labels
        top_k : number of top words to return

        Returns
        -------
        list of (word, informativeness_score) sorted descending.
        """
        dialect_names = sorted(embeddings.keys())
        n_words = next(iter(embeddings.values())).shape[0]

        if vocabulary is None:
            vocabulary = [f"word_{i}" for i in range(n_words)]

        # Use top-3 FIM directions
        n_directions = min(3, fim_eigenvectors.shape[1])
        top_dirs = fim_eigenvectors[:, :n_directions]  # (dim, n_directions)

        scores = np.zeros(n_words, dtype=np.float64)

        for w_idx in range(n_words):
            # Get this word's embedding across all dialects
            word_vectors = np.array([embeddings[d][w_idx] for d in dialect_names])  # (n_classes, dim)
            # Cross-dialect variance
            word_var = np.var(word_vectors, axis=0)  # (dim,)

            # Project variance onto top FIM directions
            projected_var = word_var @ top_dirs  # (n_directions,)
            scores[w_idx] = float(np.sum(np.abs(projected_var)))

        # Top-k
        top_indices = np.argsort(-scores)[:top_k]
        result = [(vocabulary[i], float(scores[i])) for i in top_indices]

        return result
