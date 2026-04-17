"""Compute linear transformation matrices between dialect embedding spaces."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import orthogonal_procrustes

from eigendialectos.constants import DialectCode
from eigendialectos.spectral.utils import (
    check_condition_number,
    regularize_matrix,
    safe_inverse,
)
from eigendialectos.types import EmbeddingMatrix, TransformationMatrix


def compute_transformation_matrix(
    source: EmbeddingMatrix,
    target: EmbeddingMatrix,
    method: str = "lstsq",
    regularization: float = 0.01,
    weights: npt.NDArray[np.float64] | None = None,
) -> TransformationMatrix:
    r"""Compute the transformation matrix *W* mapping *source* to *target*.

    Given embedding matrices ``E_source`` (d x V) and ``E_target`` (d x V),
    find ``W`` (d x d) such that ``W @ E_source ≈ E_target``.

    Parameters
    ----------
    source : EmbeddingMatrix
        Source dialect embeddings, shape ``(d, V)``.
    target : EmbeddingMatrix
        Target dialect embeddings, shape ``(d, V)``.
    method : str
        * ``'lstsq'`` -- Ridge regression:
          ``W = E_target @ E_source^T @ (E_source @ E_source^T + λI)^{-1}``
        * ``'procrustes'`` -- Orthogonal Procrustes: constrain *W* to O(d).
        * ``'nuclear'`` -- Nuclear-norm regularised least squares via SVD
          soft-thresholding.
    regularization : float
        Regularisation strength ``λ`` (used by ``'lstsq'`` and
        ``'nuclear'``).
    weights : ndarray of shape (V,), optional
        Per-word weights for weighted least squares.  Higher weight =
        word contributes more to the W fit.  Used to downweight noisy
        words (e.g. English contaminants with unstable embeddings).
        Only supported with ``method='lstsq'``.

    Returns
    -------
    TransformationMatrix

    Raises
    ------
    ValueError
        If shapes are incompatible or *method* is unknown.
    """
    E_s: npt.NDArray[np.float64] = np.asarray(source.data, dtype=np.float64)
    E_t: npt.NDArray[np.float64] = np.asarray(target.data, dtype=np.float64)

    if E_s.shape != E_t.shape:
        raise ValueError(
            f"Source shape {E_s.shape} != target shape {E_t.shape}. "
            "Both embedding matrices must share the same (d, V) shape."
        )

    d, V = E_s.shape

    if method == "lstsq":
        W = _lstsq(E_s, E_t, regularization, weights=weights)
    elif method == "procrustes":
        W = _procrustes(E_s, E_t)
    elif method == "nuclear":
        W = _nuclear(E_s, E_t, regularization)
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose from 'lstsq', 'procrustes', 'nuclear'."
        )

    return TransformationMatrix(
        data=W,
        source_dialect=source.dialect_code,
        target_dialect=target.dialect_code,
        regularization=regularization,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _lstsq(
    E_s: npt.NDArray[np.float64],
    E_t: npt.NDArray[np.float64],
    lam: float,
    weights: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    r"""Ridge-regression solution, optionally weighted.

    Without weights:

    .. math::
        W = E_t \, E_s^\top \bigl(E_s \, E_s^\top + \lambda I\bigr)^{-1}

    With per-word weights ``w_i``:

    .. math::
        W = E_t \, D \, E_s^\top \bigl(E_s \, D \, E_s^\top + \lambda I\bigr)^{-1}

    where ``D = \text{diag}(w_1, \ldots, w_V)``.  This is equivalent to
    weighted least squares — words with higher weights contribute more
    to the fit.  Used to downweight noisy words (e.g. English contaminants
    with unstable cross-variety embeddings).

    Parameters
    ----------
    weights : ndarray of shape (V,), optional
        Per-word weights.  If None, uniform weights (standard ridge).
    """
    d = E_s.shape[0]

    if weights is not None:
        # Weighted: scale columns by sqrt(w) for each word
        sqrt_w = np.sqrt(weights)[np.newaxis, :]  # (1, V)
        E_s_w = E_s * sqrt_w  # (d, V) — each column scaled
        E_t_w = E_t * sqrt_w
        gram = E_s_w @ E_s_w.T
        cross = E_t_w @ E_s_w.T
    else:
        gram = E_s @ E_s.T
        cross = E_t @ E_s.T

    # Regularise: gram + λI
    gram_reg = regularize_matrix(gram, lambda_reg=lam, method="ridge")
    check_condition_number(gram_reg)
    # W = cross @ inv(gram_reg)
    gram_inv = safe_inverse(gram_reg)
    W = cross @ gram_inv
    return W.astype(np.float64)


def _procrustes(
    E_s: npt.NDArray[np.float64],
    E_t: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Orthogonal Procrustes solution.

    Finds the orthogonal matrix *R* minimising ``||E_t - R @ E_s||_F``.

    scipy.linalg.orthogonal_procrustes solves ``min ||B - A @ R||``
    for column-major layout, so we transpose: ``A = E_s^T``, ``B = E_t^T``
    which yields ``R`` such that ``E_t^T ≈ E_s^T @ R``  =>  ``E_t ≈ R^T @ E_s``.
    """
    # orthogonal_procrustes(A, B) finds R s.t. ||A @ R - B||_F is minimised
    R, _ = orthogonal_procrustes(E_s.T, E_t.T)
    # W = R^T so that W @ E_s ≈ E_t  (verify: E_s^T @ R ≈ E_t^T => R^T @ E_s ≈ E_t)
    # Actually orthogonal_procrustes returns R minimising ||A R - B||
    # So E_s^T @ R ≈ E_t^T  =>  R^T @ E_s ≈ E_t  =>  W = R^T
    W = R.T
    return W.astype(np.float64)


def _nuclear(
    E_s: npt.NDArray[np.float64],
    E_t: npt.NDArray[np.float64],
    lam: float,
) -> npt.NDArray[np.float64]:
    r"""Nuclear-norm regularised least-squares via SVD soft-thresholding.

    1. Compute the un-regularised least-squares solution ``W_ls``.
    2. SVD: ``W_ls = U Sigma V^T``.
    3. Soft-threshold singular values: ``sigma_i <- max(sigma_i - lambda, 0)``.
    4. Reconstruct ``W = U diag(sigma_thresh) V^T``.
    """
    # Least-squares solution first (without nuclear penalty)
    W_ls = _lstsq(E_s, E_t, lam=0.0)
    # SVD
    U, sigma, Vt = np.linalg.svd(W_ls, full_matrices=False)
    # Soft-threshold
    sigma_thresh = np.maximum(sigma - lam, 0.0)
    # Reconstruct
    W = (U * sigma_thresh[np.newaxis, :]) @ Vt
    return W.astype(np.float64)


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def compute_all_transforms(
    embeddings: dict[DialectCode, EmbeddingMatrix],
    reference: DialectCode,
    method: str = "lstsq",
    regularization: float = 0.01,
) -> dict[DialectCode, TransformationMatrix]:
    """Compute transformations from *reference* dialect to every other.

    Parameters
    ----------
    embeddings : dict
        Mapping from ``DialectCode`` to ``EmbeddingMatrix``.
    reference : DialectCode
        The source (reference) dialect.
    method : str
        Method forwarded to :func:`compute_transformation_matrix`.
    regularization : float
        Regularisation strength forwarded to
        :func:`compute_transformation_matrix`.

    Returns
    -------
    dict[DialectCode, TransformationMatrix]
        Keys are the *target* dialect codes.  The reference dialect itself
        is included (identity-like transform).
    """
    if reference not in embeddings:
        raise ValueError(f"Reference dialect {reference} not in embeddings")

    source = embeddings[reference]
    transforms: dict[DialectCode, TransformationMatrix] = {}
    for code, target in embeddings.items():
        transforms[code] = compute_transformation_matrix(
            source=source,
            target=target,
            method=method,
            regularization=regularization,
        )
    return transforms
