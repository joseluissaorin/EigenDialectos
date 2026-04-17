"""Eigendecomposition and SVD of dialect transformation matrices."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

from eigendialectos.spectral.utils import safe_inverse
from eigendialectos.types import EigenDecomposition, TransformationMatrix


def eigendecompose(W: TransformationMatrix) -> EigenDecomposition:
    r"""Compute the eigendecomposition ``W = P \Lambda P^{-1}``.

    Parameters
    ----------
    W : TransformationMatrix
        Square transformation matrix.

    Returns
    -------
    EigenDecomposition
        Contains eigenvalues ``\lambda``, eigenvector matrix *P*, and
        ``P^{-1}``.

    Raises
    ------
    ValueError
        If the matrix is not square.
    """
    data = np.asarray(W.data, dtype=np.complex128)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f"Transformation matrix must be square, got shape {data.shape}")

    eigenvalues, P = np.linalg.eig(data)

    # Compute P^{-1} safely
    P_inv = safe_inverse(P.real).astype(np.complex128)
    # If P has imaginary parts, use complex inverse
    if np.any(np.abs(P.imag) > 1e-12):
        try:
            P_inv = np.linalg.inv(P)
        except np.linalg.LinAlgError:
            P_inv = np.linalg.pinv(P)
            warnings.warn(
                "Eigenvector matrix is singular; using pseudo-inverse.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Validate reconstruction: P @ diag(eigenvalues) @ P_inv ≈ W
    reconstructed = P @ np.diag(eigenvalues) @ P_inv
    reconstruction_error = np.linalg.norm(reconstructed - data, "fro")
    data_norm = np.linalg.norm(data, "fro")
    relative_error = reconstruction_error / max(data_norm, 1e-15)

    if relative_error > 1e-6:
        warnings.warn(
            f"Eigendecomposition reconstruction relative error = {relative_error:.2e} "
            f"(absolute = {reconstruction_error:.2e}). "
            "Results may be inaccurate for defective matrices.",
            RuntimeWarning,
            stacklevel=2,
        )

    return EigenDecomposition(
        eigenvalues=eigenvalues,
        eigenvectors=P,
        eigenvectors_inv=P_inv,
        dialect_code=W.target_dialect,
    )


def svd_decompose(
    W: TransformationMatrix,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Compute the SVD ``W = U \Sigma V^T``.

    Parameters
    ----------
    W : TransformationMatrix
        Transformation matrix (need not be square).

    Returns
    -------
    U : ndarray, shape (m, k)
        Left singular vectors.
    Sigma : ndarray, shape (k,)
        Singular values in descending order.
    Vt : ndarray, shape (k, n)
        Right singular vectors (transposed).

    Where ``k = min(m, n)``.
    """
    data = np.asarray(W.data, dtype=np.float64)
    U, sigma, Vt = np.linalg.svd(data, full_matrices=False)
    return U, sigma, Vt


def decompose(W: TransformationMatrix, method: str = "both") -> dict:
    """Decompose *W* using eigendecomposition, SVD, or both.

    Parameters
    ----------
    W : TransformationMatrix
        Transformation matrix.
    method : str
        ``'eigen'``, ``'svd'``, or ``'both'``.

    Returns
    -------
    dict
        Keys depend on *method*:

        * ``'eigen'``: ``{'eigendecomposition': EigenDecomposition}``
        * ``'svd'``: ``{'U': ndarray, 'Sigma': ndarray, 'Vt': ndarray}``
        * ``'both'``: all of the above.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    result: dict = {}

    if method in ("eigen", "both"):
        result["eigendecomposition"] = eigendecompose(W)

    if method in ("svd", "both"):
        U, Sigma, Vt = svd_decompose(W)
        result["U"] = U
        result["Sigma"] = Sigma
        result["Vt"] = Vt

    if method not in ("eigen", "svd", "both"):
        raise ValueError(f"Unknown method {method!r}. Choose 'eigen', 'svd', or 'both'.")

    return result
