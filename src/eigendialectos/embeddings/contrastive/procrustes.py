"""Orthogonal Procrustes alignment for cross-dialect embedding spaces.

This is the mathematically critical module of EigenDialectos.  It
computes the optimal orthogonal transformation W* that minimises:

    ||W X - Y||_F   subject to   W^T W = I

Closed-form solution via SVD of Y^T X:

    U S V^T = SVD(Y^T X)
    W* = V U^T

This guarantees W* is in O(d) (the orthogonal group), which preserves
inner products, distances and angles -- essential for the downstream
spectral analysis of dialect transformation matrices.

References
----------
- Schonemann, P.H. (1966). "A generalized solution of the orthogonal
  Procrustes problem." *Psychometrika*, 31, 1-10.
- Conneau, A. et al. (2018). "Word Translation Without Parallel Data."
  *ICLR*.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from eigendialectos.types import EmbeddingMatrix

logger = logging.getLogger(__name__)


class ProcrustesAligner:
    """Orthogonal Procrustes alignment between two embedding spaces.

    Given source embedding matrix X and target Y (with rows being word
    vectors for shared anchor words), finds the orthogonal matrix W*
    that best maps X into Y's space.
    """

    def __init__(self, normalize: bool = True) -> None:
        """
        Parameters
        ----------
        normalize:
            If ``True``, unit-normalise and mean-centre the anchor
            vectors before computing the Procrustes solution.  This
            follows the best practice from Conneau et al. (2018).
        """
        self._normalize = normalize
        self._W: np.ndarray | None = None

    def align(
        self,
        source: EmbeddingMatrix,
        target: EmbeddingMatrix,
        anchors: list[str] | None = None,
    ) -> np.ndarray:
        """Compute the Procrustes alignment matrix.

        Parameters
        ----------
        source:
            Embedding matrix for the source dialect.
        target:
            Embedding matrix for the target (reference) dialect.
        anchors:
            Shared vocabulary words to use as anchor pairs.  If
            ``None``, the intersection of both vocabularies is used.

        Returns
        -------
        np.ndarray
            Orthogonal matrix W of shape ``(d, d)`` such that
            ``(W @ source_vec)`` approximates the corresponding
            ``target_vec`` for each anchor word.
        """
        # --- Determine anchor pairs ---
        src_word2idx = {w: i for i, w in enumerate(source.vocab)}
        tgt_word2idx = {w: i for i, w in enumerate(target.vocab)}

        if anchors is None:
            anchors = sorted(set(source.vocab) & set(target.vocab))
        else:
            anchors = [w for w in anchors if w in src_word2idx and w in tgt_word2idx]

        if len(anchors) == 0:
            raise ValueError(
                "No anchor words found in both source and target vocabularies. "
                "Provide explicit anchors or ensure vocabularies overlap."
            )

        logger.info(
            "Procrustes alignment: %d anchor words, dim=%d, "
            "source=%s -> target=%s",
            len(anchors), source.dim,
            source.dialect_code.value, target.dialect_code.value,
        )

        # --- Extract anchor sub-matrices ---
        src_idx = [src_word2idx[w] for w in anchors]
        tgt_idx = [tgt_word2idx[w] for w in anchors]

        X = source.data[src_idx].astype(np.float64)  # (n, d)
        Y = target.data[tgt_idx].astype(np.float64)  # (n, d)

        # --- Optional normalisation ---
        if self._normalize:
            X = X - X.mean(axis=0, keepdims=True)
            Y = Y - Y.mean(axis=0, keepdims=True)
            x_norms = np.linalg.norm(X, axis=1, keepdims=True)
            y_norms = np.linalg.norm(Y, axis=1, keepdims=True)
            X = X / np.maximum(x_norms, 1e-12)
            Y = Y / np.maximum(y_norms, 1e-12)

        # --- Orthogonal Procrustes via SVD ---
        #
        # Minimise  ||W X^T - Y^T||_F   (treating rows as vectors)
        # Equivalently, for row-wise convention:
        #   W* = argmin_W || X W^T - Y ||_F  s.t. W in O(d)
        #
        # Standard formulation (column vectors in d-space):
        #   M = Y^T X                     (d x d)
        #   U S V^T = SVD(M)
        #   W = U V^T
        #
        # We want W such that X @ W.T approx Y, i.e.
        #   min_W ||X W^T - Y||_F, W^T W = I
        # Let M = X^T Y (d x d), SVD: U S V^T = SVD(M)
        # Then W^T = U V^T, so W = V U^T.
        #
        # Alternatively: min_W ||X W - Y||_F => M = X^T Y, W = U V^T
        # We use the convention W maps source -> target:
        #   aligned_source = source.data @ W
        M = X.T @ Y  # (d, d)
        U, _S, Vt = np.linalg.svd(M)

        # W = U V^T gives the optimal orthogonal solution
        W = U @ Vt

        # Ensure W is a proper rotation (det = +1) rather than
        # a reflection, which can happen when the data is rank-deficient.
        # Conneau et al. (2018) standard: flip the sign of the last
        # column of U if det < 0.
        #
        # For high-dimensional matrices np.linalg.det can overflow,
        # so we use np.linalg.slogdet which returns (sign, log|det|).
        sign, _ = np.linalg.slogdet(W)
        if sign < 0:
            U[:, -1] *= -1
            W = U @ Vt

        self._W = W
        self._verify_orthogonality(W)

        return W

    def transform(
        self,
        source: EmbeddingMatrix,
        W: np.ndarray | None = None,
    ) -> EmbeddingMatrix:
        """Apply the alignment matrix to all source vectors.

        Parameters
        ----------
        source:
            Source embedding matrix.
        W:
            Explicit alignment matrix.  If ``None``, uses the last
            matrix computed by :meth:`align`.

        Returns
        -------
        EmbeddingMatrix
            Aligned embedding matrix in the target space.
        """
        if W is None:
            W = self._W
        if W is None:
            raise RuntimeError("No alignment matrix available.  Call align() first.")

        aligned = source.data.astype(np.float64) @ W
        return EmbeddingMatrix(
            data=aligned,
            vocab=list(source.vocab),
            dialect_code=source.dialect_code,
        )

    @staticmethod
    def _verify_orthogonality(W: np.ndarray, tol: float = 1e-6) -> None:
        """Assert that W^T W = I within tolerance."""
        d = W.shape[0]
        product = W.T @ W
        deviation = np.linalg.norm(product - np.eye(d), ord="fro")
        if deviation > tol:
            logger.warning(
                "Procrustes matrix deviates from orthogonality: "
                "||W^T W - I||_F = %.2e (tol=%.2e)",
                deviation, tol,
            )
        else:
            logger.debug(
                "Orthogonality verified: ||W^T W - I||_F = %.2e", deviation
            )

    @property
    def alignment_matrix(self) -> np.ndarray | None:
        """The last computed alignment matrix, or None."""
        return self._W
