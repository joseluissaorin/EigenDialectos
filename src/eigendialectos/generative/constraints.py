"""Safety constraints and feasibility checks for DIAL transforms.

Ensures that generated transformation matrices remain numerically
stable and linguistically meaningful by checking condition numbers,
eigenvalue bounds, and other properties.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from eigendialectos.generative.dial import _eigenvalues_to_alpha
from eigendialectos.types import EigenDecomposition, TransformationMatrix


def validate_transform(
    W: TransformationMatrix,
    max_cond: float = 1000.0,
    max_eigenval: float = 100.0,
    min_eigenval: float = 0.001,
) -> tuple[bool, list[str]]:
    """Validate that a transformation matrix is numerically well-behaved.

    Parameters
    ----------
    W : TransformationMatrix
        The transform to validate.
    max_cond : float
        Maximum allowed condition number.
    max_eigenval : float
        Maximum allowed eigenvalue magnitude.
    min_eigenval : float
        Minimum allowed (non-zero) eigenvalue magnitude.

    Returns
    -------
    tuple of (bool, list[str])
        ``(is_valid, violations)`` where *violations* is a list of
        human-readable messages describing each failure.
    """
    violations: list[str] = []
    data = np.asarray(W.data, dtype=np.float64)

    # Check shape
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        violations.append(
            f"Transform must be a square matrix, got shape {data.shape}"
        )
        return (False, violations)

    # Condition number
    cond = float(np.linalg.cond(data))
    if cond > max_cond:
        violations.append(
            f"Condition number {cond:.2e} exceeds limit {max_cond:.2e}"
        )

    # Eigenvalue bounds
    eigenvalues = np.linalg.eigvals(data)
    magnitudes = np.abs(eigenvalues)

    max_mag = float(np.max(magnitudes))
    if max_mag > max_eigenval:
        violations.append(
            f"Maximum eigenvalue magnitude {max_mag:.4f} exceeds "
            f"limit {max_eigenval:.4f}"
        )

    # Check non-zero eigenvalues against minimum
    nonzero_mask = magnitudes > 1e-14
    if np.any(nonzero_mask):
        min_nonzero = float(np.min(magnitudes[nonzero_mask]))
        if min_nonzero < min_eigenval:
            violations.append(
                f"Minimum non-zero eigenvalue magnitude {min_nonzero:.6f} "
                f"is below limit {min_eigenval:.6f}"
            )

    # Check for NaN / Inf
    if np.any(np.isnan(data)):
        violations.append("Transform contains NaN values")
    if np.any(np.isinf(data)):
        violations.append("Transform contains Inf values")

    is_valid = len(violations) == 0
    return (is_valid, violations)


def clip_eigenvalues(
    eigen: EigenDecomposition,
    max_val: float = 100.0,
    min_val: float = 0.001,
) -> EigenDecomposition:
    """Clip eigenvalue magnitudes to a safe range.

    Preserves the phase (angle) of each eigenvalue while bounding the
    magnitude to ``[min_val, max_val]``.  Zero eigenvalues remain zero.

    Parameters
    ----------
    eigen : EigenDecomposition
        Original eigendecomposition.
    max_val : float
        Upper bound on eigenvalue magnitude.
    min_val : float
        Lower bound on non-zero eigenvalue magnitude.

    Returns
    -------
    EigenDecomposition
        New decomposition with clipped eigenvalues.
    """
    eigenvalues = np.asarray(eigen.eigenvalues, dtype=np.complex128)
    magnitudes = np.abs(eigenvalues)
    angles = np.angle(eigenvalues)

    # Clip magnitudes, but preserve zeros
    is_nonzero = magnitudes > 1e-14
    clipped_mag = np.where(
        is_nonzero,
        np.clip(magnitudes, min_val, max_val),
        0.0,
    )

    # Reconstruct eigenvalues with clipped magnitudes and original phases
    clipped_eigenvalues = clipped_mag * np.exp(1j * angles)

    return EigenDecomposition(
        eigenvalues=clipped_eigenvalues,
        eigenvectors=eigen.eigenvectors.copy(),
        eigenvectors_inv=eigen.eigenvectors_inv.copy(),
        dialect_code=eigen.dialect_code,
    )


def check_feasibility(
    alpha: float,
    eigen: EigenDecomposition,
    max_cond: float = 1000.0,
    max_eigenval: float = 100.0,
    min_eigenval: float = 0.001,
) -> tuple[bool, str]:
    """Check whether an alpha value produces a feasible transform.

    Performs a lightweight check by examining the eigenvalues raised to
    the power alpha, without reconstructing the full matrix.

    Parameters
    ----------
    alpha : float
        Proposed dialectal intensity.
    eigen : EigenDecomposition
        Eigendecomposition of the dialect transform.
    max_cond, max_eigenval, min_eigenval : float
        Bounds for feasibility (same semantics as :func:`validate_transform`).

    Returns
    -------
    tuple of (bool, str)
        ``(is_feasible, reason)`` where *reason* is empty on success or
        a description of the problem on failure.
    """
    lambda_alpha = _eigenvalues_to_alpha(eigen.eigenvalues, alpha)
    mags = np.abs(lambda_alpha)

    # Check for NaN or Inf
    if np.any(np.isnan(mags)) or np.any(np.isinf(mags)):
        return (False, f"alpha={alpha} produces NaN/Inf eigenvalues")

    # Maximum magnitude
    max_mag = float(np.max(mags))
    if max_mag > max_eigenval:
        return (
            False,
            f"alpha={alpha} produces eigenvalue magnitude {max_mag:.4f} "
            f"exceeding limit {max_eigenval:.4f}",
        )

    # Minimum non-zero magnitude
    nonzero_mask = mags > 1e-14
    if np.any(nonzero_mask):
        min_nonzero = float(np.min(mags[nonzero_mask]))
        if min_nonzero < min_eigenval:
            return (
                False,
                f"alpha={alpha} produces eigenvalue magnitude {min_nonzero:.6f} "
                f"below limit {min_eigenval:.6f}",
            )

    # Condition number proxy: ratio of max to min non-zero magnitude
    if np.any(nonzero_mask):
        min_nonzero = float(np.min(mags[nonzero_mask]))
        if min_nonzero > 0:
            cond_proxy = max_mag / min_nonzero
            if cond_proxy > max_cond:
                return (
                    False,
                    f"alpha={alpha} produces condition number proxy "
                    f"{cond_proxy:.2e} exceeding limit {max_cond:.2e}",
                )

    return (True, "")
