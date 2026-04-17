"""Numerical stability helpers for spectral computations."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt


def check_condition_number(
    M: npt.NDArray[np.floating], threshold: float = 1e10
) -> float:
    """Return the condition number of *M*, warning if it exceeds *threshold*.

    Uses the 2-norm condition number (ratio of largest to smallest singular
    value).

    Parameters
    ----------
    M : ndarray
        Square or rectangular matrix.
    threshold : float
        If the condition number exceeds this value a ``RuntimeWarning`` is
        emitted.

    Returns
    -------
    float
        The condition number ``cond(M)``.
    """
    cond = float(np.linalg.cond(M))
    if cond > threshold:
        warnings.warn(
            f"Matrix condition number {cond:.2e} exceeds threshold "
            f"{threshold:.2e}. Results may be numerically unstable.",
            RuntimeWarning,
            stacklevel=2,
        )
    return cond


def regularize_matrix(
    M: npt.NDArray[np.floating],
    lambda_reg: float = 0.01,
    method: str = "ridge",
) -> npt.NDArray[np.float64]:
    """Apply Tikhonov (ridge) regularisation to a square matrix.

    Parameters
    ----------
    M : ndarray, shape (n, n)
        Square matrix to regularise.
    lambda_reg : float
        Regularisation strength.
    method : str
        Currently only ``'ridge'`` (Tikhonov) is supported: returns
        ``M + lambda_reg * I``.

    Returns
    -------
    ndarray
        Regularised matrix.

    Raises
    ------
    ValueError
        If *M* is not square or *method* is unknown.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {M.shape}")
    if method != "ridge":
        raise ValueError(f"Unknown regularisation method: {method!r}")
    n = M.shape[0]
    return (M + lambda_reg * np.eye(n)).astype(np.float64)


def handle_complex_eigenvalues(
    eigenvalues: npt.NDArray[np.complexfloating],
    method: str = "magnitude",
) -> npt.NDArray[np.float64]:
    """Convert complex eigenvalues to real values.

    Parameters
    ----------
    eigenvalues : ndarray
        Possibly-complex eigenvalue array.
    method : str
        Conversion strategy:

        * ``'magnitude'`` -- absolute value ``|lambda|``
        * ``'real_part'`` -- real component ``Re(lambda)``
        * ``'both'`` -- returns a 2-column array ``[|lambda|, Re(lambda)]``

    Returns
    -------
    ndarray
        Real-valued array (1-D for ``'magnitude'`` and ``'real_part'``;
        2-D of shape ``(n, 2)`` for ``'both'``).

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    eigenvalues = np.asarray(eigenvalues)
    if method == "magnitude":
        return np.abs(eigenvalues).astype(np.float64)
    if method == "real_part":
        return np.real(eigenvalues).astype(np.float64)
    if method == "both":
        magnitudes = np.abs(eigenvalues).astype(np.float64)
        real_parts = np.real(eigenvalues).astype(np.float64)
        return np.column_stack([magnitudes, real_parts])
    raise ValueError(f"Unknown method: {method!r}. Use 'magnitude', 'real_part', or 'both'.")


def stable_log(
    x: float | npt.NDArray[np.floating],
    eps: float = 1e-10,
) -> float | npt.NDArray[np.float64]:
    """Numerically stable natural logarithm: ``log(max(x, eps))``.

    Parameters
    ----------
    x : float or ndarray
        Input value(s).
    eps : float
        Floor value to prevent ``log(0)``.

    Returns
    -------
    float or ndarray
        ``ln(max(x, eps))``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    result = np.log(np.maximum(x_arr, eps))
    if result.ndim == 0:
        return float(result)
    return result


def is_orthogonal(M: npt.NDArray[np.floating], tol: float = 1e-6) -> bool:
    """Check whether *M* is orthogonal: ``M @ M^T approx I``.

    Parameters
    ----------
    M : ndarray, shape (n, n)
        Square matrix to test.
    tol : float
        Frobenius-norm tolerance for deviation from identity.

    Returns
    -------
    bool
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    n = M.shape[0]
    product = M @ M.T
    return bool(np.linalg.norm(product - np.eye(n), "fro") < tol)


def is_positive_definite(M: npt.NDArray[np.floating]) -> bool:
    """Check whether the symmetric part of *M* is positive-definite.

    Uses Cholesky decomposition; returns ``False`` on failure.

    Parameters
    ----------
    M : ndarray, shape (n, n)
        Square matrix to test.

    Returns
    -------
    bool
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    sym = 0.5 * (M + M.T)
    try:
        np.linalg.cholesky(sym)
        return True
    except np.linalg.LinAlgError:
        return False


def safe_inverse(
    M: npt.NDArray[np.floating],
    rcond: float = 1e-10,
) -> npt.NDArray[np.float64]:
    """Compute the inverse of *M*, falling back to the pseudo-inverse.

    If the matrix is singular (or nearly so, according to *rcond*), the
    Moore--Penrose pseudo-inverse is returned instead of raising an error.

    Parameters
    ----------
    M : ndarray, shape (n, n)
        Square matrix.
    rcond : float
        Cutoff for small singular values when computing the pseudo-inverse.

    Returns
    -------
    ndarray
        Inverse or pseudo-inverse of *M*.
    """
    try:
        cond = np.linalg.cond(M)
        if cond > 1.0 / rcond:
            warnings.warn(
                f"Matrix is near-singular (cond={cond:.2e}). "
                "Using pseudo-inverse.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.linalg.pinv(M, rcond=rcond).astype(np.float64)
        return np.linalg.inv(M).astype(np.float64)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Matrix is singular. Using pseudo-inverse.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.linalg.pinv(M, rcond=rcond).astype(np.float64)
