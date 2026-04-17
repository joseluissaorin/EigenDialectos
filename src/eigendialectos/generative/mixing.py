"""Dialect mixing: combine multiple dialect transforms into blends.

Supports both linear (weighted sum) and log-Euclidean (geometric)
interpolation strategies, as well as mixing at the eigenvalue level.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm, logm

from eigendialectos.constants import DialectCode
from eigendialectos.types import EigenDecomposition, TransformationMatrix

# ---------------------------------------------------------------------------
# Tolerance for weight validation
# ---------------------------------------------------------------------------
_WEIGHT_TOL: float = 1e-6


def _validate_weights(weights: list[float]) -> None:
    """Ensure mixing weights sum to 1 (within tolerance).

    Parameters
    ----------
    weights : list of float
        Mixing coefficients beta_k.

    Raises
    ------
    ValueError
        If the weights do not sum to 1.0 within *_WEIGHT_TOL*.
    """
    total = sum(weights)
    if abs(total - 1.0) > _WEIGHT_TOL:
        raise ValueError(
            f"Mixing weights must sum to 1.0, got {total:.8f} "
            f"(difference {abs(total - 1.0):.2e})"
        )


def mix_dialects(
    transforms: list[tuple[TransformationMatrix, float]],
) -> TransformationMatrix:
    """Linear dialect mixing: W_mix = sum(beta_k * W_k).

    Parameters
    ----------
    transforms : list of (TransformationMatrix, float)
        Pairs of (transform, weight).  Weights must sum to 1.

    Returns
    -------
    TransformationMatrix
        The blended transformation matrix.

    Raises
    ------
    ValueError
        If weights do not sum to 1 or list is empty.
    """
    if not transforms:
        raise ValueError("Cannot mix an empty list of transforms.")

    weights = [w for _, w in transforms]
    _validate_weights(weights)

    W_mix = np.zeros_like(transforms[0][0].data, dtype=np.float64)
    for tm, beta in transforms:
        W_mix += beta * np.asarray(tm.data, dtype=np.float64)

    # Derive dialect metadata from the first transform
    first_tm = transforms[0][0]
    return TransformationMatrix(
        data=W_mix,
        source_dialect=first_tm.source_dialect,
        target_dialect=DialectCode.ES_PEN,  # mixed dialect
        regularization=0.0,
    )


def log_euclidean_mix(
    transforms: list[tuple[TransformationMatrix, float]],
) -> TransformationMatrix:
    """Log-Euclidean dialect mixing: W_mix = expm(sum(beta_k * logm(W_k))).

    This provides geometrically correct interpolation on the manifold of
    invertible matrices, respecting the Lie group structure.

    Parameters
    ----------
    transforms : list of (TransformationMatrix, float)
        Pairs of (transform, weight).  Weights must sum to 1.

    Returns
    -------
    TransformationMatrix
        The blended transformation matrix.

    Raises
    ------
    ValueError
        If weights do not sum to 1 or list is empty.
    """
    if not transforms:
        raise ValueError("Cannot mix an empty list of transforms.")

    weights = [w for _, w in transforms]
    _validate_weights(weights)

    n = transforms[0][0].data.shape[0]
    log_sum = np.zeros((n, n), dtype=np.complex128)

    for tm, beta in transforms:
        W = np.asarray(tm.data, dtype=np.float64)
        log_W = logm(W)
        log_sum += beta * log_W

    W_mix = expm(log_sum)

    # Discard negligible imaginary components
    if np.allclose(W_mix.imag, 0.0, atol=1e-10):
        W_mix = W_mix.real

    first_tm = transforms[0][0]
    return TransformationMatrix(
        data=np.asarray(W_mix, dtype=np.float64),
        source_dialect=first_tm.source_dialect,
        target_dialect=DialectCode.ES_PEN,  # mixed dialect
        regularization=0.0,
    )


def mix_eigendecompositions(
    eigens: list[tuple[EigenDecomposition, float]],
) -> EigenDecomposition:
    """Mix dialect transforms at the eigenvalue level.

    Uses a shared eigenvector basis (from the first decomposition) and
    interpolates eigenvalues as a weighted geometric mean:

        lambda_mix = prod(|lambda_k|^{beta_k}) * exp(i * sum(beta_k * theta_k))

    Parameters
    ----------
    eigens : list of (EigenDecomposition, float)
        Pairs of (eigendecomposition, weight).  Weights must sum to 1.

    Returns
    -------
    EigenDecomposition
        Blended eigendecomposition using the eigenvector basis of the first
        entry.

    Raises
    ------
    ValueError
        If weights do not sum to 1, list is empty, or eigenvalue array
        sizes are inconsistent.
    """
    if not eigens:
        raise ValueError("Cannot mix an empty list of eigendecompositions.")

    weights = [w for _, w in eigens]
    _validate_weights(weights)

    # Validate consistent sizes
    n = len(eigens[0][0].eigenvalues)
    for eigen, _ in eigens:
        if len(eigen.eigenvalues) != n:
            raise ValueError(
                f"All eigendecompositions must have the same number of "
                f"eigenvalues. Expected {n}, got {len(eigen.eigenvalues)}."
            )

    # Weighted geometric interpolation of eigenvalues
    # log-domain: log(lambda_mix) = sum(beta_k * log(lambda_k))
    log_mixed = np.zeros(n, dtype=np.complex128)
    for eigen, beta in eigens:
        eigenvalues = np.asarray(eigen.eigenvalues, dtype=np.complex128)
        magnitudes = np.abs(eigenvalues)
        angles = np.angle(eigenvalues)

        # Use log of magnitude (safe against zeros)
        safe_log_mag = np.where(magnitudes > 0, np.log(magnitudes), 0.0)
        log_mixed += beta * (safe_log_mag + 1j * angles)

    # Reconstruct eigenvalues from log-domain
    mixed_eigenvalues = np.exp(log_mixed)

    # Use eigenvector basis from the first decomposition
    first_eigen = eigens[0][0]
    return EigenDecomposition(
        eigenvalues=mixed_eigenvalues,
        eigenvectors=first_eigen.eigenvectors.copy(),
        eigenvectors_inv=first_eigen.eigenvectors_inv.copy(),
        dialect_code=first_eigen.dialect_code,
    )
