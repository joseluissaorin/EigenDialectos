"""Multi-level spectral stack transform.

Applies separate transformation matrices W_i per linguistic level,
enabling per-level α control (the "mixing board" metaphor).

C_{A→B}(t) = ∏_ℓ W_{A→B}^(ℓ)(α_ℓ) ∘ π_ℓ(t)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from eigendialectos.constants import LinguisticLevel
from eigendialectos.spectral.eigendecomposition import eigendecompose
from eigendialectos.spectral.transformation import compute_transformation_matrix
from eigendialectos.types import (
    EigenDecomposition,
    EmbeddingMatrix,
    LevelEmbedding,
    SpectralStackResult,
    TransformationMatrix,
)

logger = logging.getLogger(__name__)


class SpectralStack:
    """Multi-level spectral transform: separate W_i and eigenspectrum per level.

    Each linguistic level (morpheme, word, phrase, sentence, discourse) gets
    its own transformation matrix and eigendecomposition, allowing independent
    α control per level.
    """

    def __init__(self, levels: Optional[list[int]] = None) -> None:
        self.levels = levels or [lv.value for lv in LinguisticLevel]
        self.level_W: dict[int, npt.NDArray[np.float64]] = {}
        self.level_eigen: dict[int, EigenDecomposition] = {}
        self._fitted = False

    def fit(
        self,
        source_embeddings: dict[int, LevelEmbedding],
        target_embeddings: dict[int, LevelEmbedding],
        method: str = "lstsq",
        regularization: float = 0.01,
    ) -> SpectralStack:
        """Compute W_i for each level using shared vocabulary at that level.

        For each level:
        1. Find shared units between source and target embeddings
        2. Build aligned embedding matrices
        3. Compute W via existing compute_transformation_matrix
        4. Eigendecompose W

        Parameters
        ----------
        source_embeddings : dict mapping level int to LevelEmbedding
        target_embeddings : dict mapping level int to LevelEmbedding
        method : str
            Method for computing W ('lstsq', 'procrustes', 'nuclear')
        regularization : float
            Regularization strength for lstsq/nuclear methods

        Returns
        -------
        self
        """
        from eigendialectos.constants import DialectCode

        for level in self.levels:
            if level not in source_embeddings or level not in target_embeddings:
                logger.warning("Level %d missing from embeddings, skipping", level)
                continue

            src = source_embeddings[level]
            tgt = target_embeddings[level]

            # Find shared vocabulary at this level
            shared_vocab = sorted(
                set(src.vocabulary.keys()) & set(tgt.vocabulary.keys())
            )
            if len(shared_vocab) < 5:
                logger.warning(
                    "Level %d has only %d shared units, skipping",
                    level,
                    len(shared_vocab),
                )
                continue

            # Build aligned matrices: (dim, n_shared)
            src_indices = [src.vocabulary[w] for w in shared_vocab]
            tgt_indices = [tgt.vocabulary[w] for w in shared_vocab]
            E_src = src.vectors[src_indices].T  # (dim, n_shared)
            E_tgt = tgt.vectors[tgt_indices].T

            src_emb = EmbeddingMatrix(
                data=E_src,
                vocab=shared_vocab,
                dialect_code=DialectCode.ES_PEN,  # source = neutral
            )
            tgt_emb = EmbeddingMatrix(
                data=E_tgt,
                vocab=shared_vocab,
                dialect_code=DialectCode.ES_PEN,  # placeholder
            )

            W_tm = compute_transformation_matrix(
                source=src_emb,
                target=tgt_emb,
                method=method,
                regularization=regularization,
            )
            self.level_W[level] = W_tm.data

            # Eigendecompose
            self.level_eigen[level] = eigendecompose(W_tm)

            logger.info(
                "Level %d: W shape %s, %d shared units, top eigenvalue |λ₁|=%.4f",
                level,
                W_tm.data.shape,
                len(shared_vocab),
                np.max(np.abs(self.level_eigen[level].eigenvalues)),
            )

        self._fitted = True
        return self

    def fit_from_matrices(
        self,
        W_matrices: dict[int, npt.NDArray[np.float64]],
    ) -> SpectralStack:
        """Fit directly from pre-computed W matrices per level.

        Useful when transformation matrices are already computed.
        """
        from eigendialectos.constants import DialectCode

        for level, W in W_matrices.items():
            self.level_W[level] = W
            tm = TransformationMatrix(
                data=W,
                source_dialect=DialectCode.ES_PEN,
                target_dialect=DialectCode.ES_PEN,
                regularization=0.0,
            )
            self.level_eigen[level] = eigendecompose(tm)

        self._fitted = True
        return self

    def transform(
        self,
        level: int,
        vectors: npt.NDArray[np.float64],
        alpha: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Apply W^(ℓ)(α) = P Λ^α P^{-1} transform at given level.

        Parameters
        ----------
        level : int
            Linguistic level (1-5)
        vectors : ndarray, shape (n, dim) or (dim,)
            Input vectors to transform
        alpha : float
            Dialectal intensity (0 = identity, 1 = full transform)

        Returns
        -------
        ndarray
            Transformed vectors, same shape as input
        """
        if level not in self.level_eigen:
            raise KeyError(f"Level {level} not fitted. Available: {list(self.level_eigen.keys())}")

        eigen = self.level_eigen[level]
        P = eigen.eigenvectors
        eigenvalues = eigen.eigenvalues
        P_inv = eigen.eigenvectors_inv

        # Compute Λ^α (handle complex eigenvalues via complex power)
        Lambda_alpha = np.diag(eigenvalues.astype(np.complex128) ** alpha)

        # W(α) = P Λ^α P^{-1}
        W_alpha = (P @ Lambda_alpha @ P_inv).real

        # Apply: if vectors is (n, dim), result is (n, dim)
        was_1d = vectors.ndim == 1
        if was_1d:
            vectors = vectors.reshape(1, -1)

        result = (W_alpha @ vectors.T).T

        if was_1d:
            result = result.ravel()

        return result.astype(np.float64)

    def transform_all(
        self,
        level_vectors: dict[int, npt.NDArray[np.float64]],
        alphas: dict[int, float],
    ) -> SpectralStackResult:
        """Apply full spectral stack transform across all levels.

        Parameters
        ----------
        level_vectors : dict mapping level -> (n, dim) vectors
        alphas : dict mapping level -> α intensity

        Returns
        -------
        SpectralStackResult
        """
        level_transforms: dict[int, npt.NDArray[np.float64]] = {}

        for level in self.levels:
            if level not in level_vectors:
                continue
            alpha = alphas.get(level, 1.0)
            level_transforms[level] = self.transform(level, level_vectors[level], alpha)

        return SpectralStackResult(
            level_results=dict(self.level_eigen),
            level_transforms=level_transforms,
            alphas=alphas,
        )

    def get_eigenspectrum(self, level: int) -> tuple[npt.NDArray, npt.NDArray]:
        """Get eigenvalues and eigenvectors for a specific level.

        Returns
        -------
        eigenvalues, eigenvectors
        """
        if level not in self.level_eigen:
            raise KeyError(f"Level {level} not fitted")
        eigen = self.level_eigen[level]
        return eigen.eigenvalues, eigen.eigenvectors

    @property
    def fitted_levels(self) -> list[int]:
        """Return list of levels that have been fitted."""
        return sorted(self.level_eigen.keys())
