"""DIAL: Dialectal Interpolation via Algebraic Linearization.

The core innovation of EigenDialectos. Given the eigendecomposition of a
dialect transformation matrix W = P Lambda P^{-1}, the DIAL transform
computes a continuous family of matrices parameterised by alpha:

    W(alpha) = P Lambda^alpha P^{-1}

where alpha controls dialectal intensity:
    alpha = 0 --> identity (neutral Spanish)
    alpha = 1 --> original W (full dialect)
    alpha > 1 --> hyperdialect (exaggerated features)
    alpha < 0 --> inverse dialect
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from eigendialectos.constants import ALPHA_RANGE, DialectCode
from eigendialectos.types import EigenDecomposition, TransformationMatrix


def _eigenvalues_to_alpha(
    eigenvalues: npt.NDArray[np.complex128],
    alpha: float,
) -> npt.NDArray[np.complex128]:
    """Raise complex eigenvalues to a real power alpha.

    For a complex eigenvalue lambda = |lambda| e^{i theta}:
        lambda^alpha = |lambda|^alpha * e^{i alpha theta}

    Parameters
    ----------
    eigenvalues : ndarray of complex128
        Array of (possibly complex) eigenvalues.
    alpha : float
        Dialectal intensity parameter.

    Returns
    -------
    ndarray of complex128
        Eigenvalues raised to the power alpha.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
    magnitudes = np.abs(eigenvalues)
    angles = np.angle(eigenvalues)

    # |lambda|^alpha -- handle zero eigenvalues gracefully
    mag_alpha = np.where(magnitudes > 0, np.power(magnitudes, alpha), 0.0)

    # e^{i alpha theta}
    phase_alpha = np.exp(1j * alpha * angles)

    return (mag_alpha * phase_alpha).astype(np.complex128)


def apply_dial(
    eigen: EigenDecomposition,
    alpha: float,
) -> TransformationMatrix:
    """Apply the DIAL transform at intensity *alpha*.

    Computes W(alpha) = P diag(Lambda^alpha) P^{-1}.

    Parameters
    ----------
    eigen : EigenDecomposition
        Pre-computed eigendecomposition of a dialect transformation matrix.
    alpha : float
        Dialectal intensity. 0 = identity, 1 = original, >1 = hyperdialect,
        <0 = inverse dialect.

    Returns
    -------
    TransformationMatrix
        The interpolated transformation matrix.
    """
    lambda_alpha = _eigenvalues_to_alpha(eigen.eigenvalues, alpha)
    diag_alpha = np.diag(lambda_alpha)

    # W(alpha) = P Lambda^alpha P^{-1}
    W_alpha = eigen.eigenvectors @ diag_alpha @ eigen.eigenvectors_inv

    # Discard imaginary parts: for real-valued transforms with conjugate
    # eigenvalue pairs the imaginary residual is purely numerical noise.
    # For non-conjugate artificial inputs we keep only the real part since
    # TransformationMatrix stores float64 data.
    W_real = np.real(W_alpha)

    return TransformationMatrix(
        data=np.asarray(W_real, dtype=np.float64),
        source_dialect=DialectCode.ES_PEN,
        target_dialect=eigen.dialect_code,
        regularization=0.0,
    )


def dial_transform_embedding(
    embedding: npt.NDArray[np.floating],
    eigen: EigenDecomposition,
    alpha: float,
) -> npt.NDArray[np.float64]:
    """Apply the DIAL transform to an embedding vector or matrix.

    Parameters
    ----------
    embedding : ndarray
        A single embedding vector of shape ``(d,)`` or a batch of embeddings
        of shape ``(n, d)``.
    eigen : EigenDecomposition
        Eigendecomposition of the dialect transform.
    alpha : float
        Dialectal intensity parameter.

    Returns
    -------
    ndarray of float64
        Transformed embedding(s), same shape as input.
    """
    W = apply_dial(eigen, alpha)
    embedding = np.asarray(embedding, dtype=np.float64)

    if embedding.ndim == 1:
        result = W.data @ embedding
    elif embedding.ndim == 2:
        # Each row is a vector: result = (W @ E^T)^T = E @ W^T
        result = embedding @ W.data.T
    else:
        raise ValueError(
            f"Expected 1-D or 2-D embedding array, got shape {embedding.shape}"
        )

    return result.astype(np.float64)


def compute_dial_series(
    eigen: EigenDecomposition,
    alpha_range: tuple[float, float, float] = ALPHA_RANGE,
) -> list[TransformationMatrix]:
    """Compute DIAL transforms for a range of alpha values.

    Parameters
    ----------
    eigen : EigenDecomposition
        Eigendecomposition of the dialect transform.
    alpha_range : tuple of (start, stop, step)
        Range specification for alpha values. Default is the project-wide
        ``ALPHA_RANGE`` constant.

    Returns
    -------
    list of TransformationMatrix
        One transform per alpha value in the range ``[start, stop)`` with
        the given step.
    """
    start, stop, step = alpha_range
    alphas = np.arange(start, stop, step)
    return [apply_dial(eigen, float(a)) for a in alphas]
