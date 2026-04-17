"""Per-mode alpha control — the core v3 innovation.

Instead of a single scalar exponent on *all* eigenvalues, we assign each
eigenmode its own intensity alpha_i.  The parametric family of operators is:

    W(a) = P @ diag(lambda_1^{a_1}, ..., lambda_n^{a_n}) @ P^{-1}

where ``a = (a_1, ..., a_n)`` is an *AlphaVector*.

Key properties:
    * a = 1  ⟹  W(a) = W   (original transformation)
    * a = 0  ⟹  W(a) = I   (identity; no dialectal shift)
    * Intermediate a continuously interpolates dialect intensity.
    * Individual modes can be isolated, suppressed or composed.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from eigen3.types import AlphaVector, EigenDecomp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_W_alpha(decomp: EigenDecomp, alpha: AlphaVector) -> np.ndarray:
    """Build the parametric operator W(alpha) from a decomposition and alpha vector.

    Formula
    -------
        W(a) = P @ diag(lambda_1^{a_1}, ..., lambda_n^{a_n}) @ P_inv

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition of the original W matrix.
    alpha : AlphaVector
        Per-mode intensity vector.  Length must equal ``decomp.n_modes``.

    Returns
    -------
    np.ndarray
        (n, n) real-valued matrix obtained by taking the real part of the
        reconstruction (imaginary residuals from conjugate-pair arithmetic
        are typically < 1e-12).

    Raises
    ------
    ValueError
        If the alpha vector length does not match the number of modes.
    """
    n = decomp.n_modes
    if len(alpha) != n:
        raise ValueError(
            f"AlphaVector length ({len(alpha)}) != number of modes ({n})"
        )

    # np.power handles complex eigenvalues correctly:
    # (r * e^{i*theta})^a  =  r^a * e^{i*a*theta}
    powered = np.power(decomp.eigenvalues, alpha.values)  # (n,) complex
    Lambda_alpha = np.diag(powered)                       # (n, n) complex

    W_alpha = decomp.P @ Lambda_alpha @ decomp.P_inv
    return W_alpha.real


def _validate_mode_idx(decomp: EigenDecomp, mode_idx: int) -> None:
    """Raise *ValueError* if *mode_idx* is out of range."""
    if not 0 <= mode_idx < decomp.n_modes:
        raise ValueError(
            f"mode_idx {mode_idx} out of range [0, {decomp.n_modes})"
        )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def isolate_mode(
    decomp: EigenDecomp,
    mode_idx: int,
    strength: float = 1.0,
) -> np.ndarray:
    """Produce W(alpha) with *only* one active mode.

    All alpha entries are set to 0 (identity contribution) except
    ``alpha[mode_idx] = strength``.

    This is the simplest way to inspect what a single eigenmode "does":
    apply the returned matrix to a word vector and compare the result to
    the original vector.

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition.
    mode_idx : int
        Index of the eigenmode to isolate.
    strength : float
        Exponent applied to the selected eigenvalue.  1.0 reproduces the
        original eigenvalue contribution; values > 1 amplify it.

    Returns
    -------
    np.ndarray
        (n, n) matrix with a single active mode.
    """
    _validate_mode_idx(decomp, mode_idx)
    alpha = AlphaVector.zeros(decomp.n_modes)
    alpha.values[mode_idx] = strength
    return compute_W_alpha(decomp, alpha)


def suppress_mode(decomp: EigenDecomp, mode_idx: int) -> np.ndarray:
    """Produce W(alpha) with one mode suppressed (set to identity).

    All alpha entries are 1 (full contribution) except
    ``alpha[mode_idx] = 0`` (suppressed).

    Comparing the output to the original W reveals what information
    the suppressed mode carried.

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition.
    mode_idx : int
        Index of the eigenmode to suppress.

    Returns
    -------
    np.ndarray
        (n, n) matrix with the target mode zeroed out.
    """
    _validate_mode_idx(decomp, mode_idx)
    alpha = AlphaVector.ones(decomp.n_modes)
    alpha.values[mode_idx] = 0.0
    return compute_W_alpha(decomp, alpha)


def compose_modes(
    decomp: EigenDecomp,
    mode_indices: Sequence[int],
    strengths: Sequence[float],
) -> np.ndarray:
    """Produce W(alpha) activating a specific set of modes.

    Only the listed modes receive non-zero alpha; all others remain at 0
    (identity).

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition.
    mode_indices : sequence of int
        Mode indices to activate.
    strengths : sequence of float
        Per-mode strengths (same order as *mode_indices*).

    Returns
    -------
    np.ndarray
        (n, n) matrix with the selected modes active.

    Raises
    ------
    ValueError
        If *mode_indices* and *strengths* have different lengths, or if
        any mode index is out of range.
    """
    if len(mode_indices) != len(strengths):
        raise ValueError(
            f"mode_indices ({len(mode_indices)}) and strengths "
            f"({len(strengths)}) must have the same length"
        )

    alpha = AlphaVector.zeros(decomp.n_modes)
    for idx, s in zip(mode_indices, strengths):
        _validate_mode_idx(decomp, idx)
        alpha.values[idx] = s

    return compute_W_alpha(decomp, alpha)


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def interpolate_alpha(
    alpha_a: AlphaVector,
    alpha_b: AlphaVector,
    t: float,
) -> AlphaVector:
    """Linearly interpolate between two alpha vectors.

    Parameters
    ----------
    alpha_a, alpha_b : AlphaVector
        Endpoints of the interpolation.
    t : float
        Interpolation parameter in [0, 1].  ``t=0`` returns *alpha_a*,
        ``t=1`` returns *alpha_b*.

    Returns
    -------
    AlphaVector
        Interpolated alpha vector.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0, 1], got {t}")
    if len(alpha_a) != len(alpha_b):
        raise ValueError("Alpha vectors must have the same length")
    values = (1.0 - t) * alpha_a.values + t * alpha_b.values
    return AlphaVector(values=values)


def alpha_gradient(
    decomp: EigenDecomp,
    alpha: AlphaVector,
    target_emb: np.ndarray,
    source_emb: np.ndarray,
) -> np.ndarray:
    """Numerical gradient of ||W(alpha) @ source - target||^2 w.r.t. alpha.

    Uses centered finite differences for each alpha component.

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition.
    alpha : AlphaVector
        Current alpha vector.
    target_emb : np.ndarray
        (vocab_size, dim) target embedding matrix.
    source_emb : np.ndarray
        (vocab_size, dim) source embedding matrix.

    Returns
    -------
    np.ndarray
        (n_modes,) gradient vector.
    """
    eps = 1e-5
    grad = np.zeros(decomp.n_modes, dtype=np.float64)

    for i in range(decomp.n_modes):
        # Forward
        alpha_fwd = AlphaVector(values=alpha.values.copy())
        alpha_fwd.values[i] += eps
        W_fwd = compute_W_alpha(decomp, alpha_fwd)
        loss_fwd = np.sum((source_emb @ W_fwd.T - target_emb) ** 2)

        # Backward
        alpha_bwd = AlphaVector(values=alpha.values.copy())
        alpha_bwd.values[i] -= eps
        W_bwd = compute_W_alpha(decomp, alpha_bwd)
        loss_bwd = np.sum((source_emb @ W_bwd.T - target_emb) ** 2)

        grad[i] = (loss_fwd - loss_bwd) / (2.0 * eps)

    return grad


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def mode_contribution(decomp: EigenDecomp, mode_idx: int) -> float:
    """Relative energy contribution of a single eigenmode.

    Defined as  |lambda_i|^2 / sum_j |lambda_j|^2.

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition.
    mode_idx : int
        Index of the eigenmode.

    Returns
    -------
    float
        Fraction of total energy carried by the mode.
    """
    _validate_mode_idx(decomp, mode_idx)
    mags_sq = np.abs(decomp.eigenvalues) ** 2
    total = mags_sq.sum()
    if total == 0:
        return 0.0
    return float(mags_sq[mode_idx] / total)


def energy_spectrum(decomp: EigenDecomp) -> np.ndarray:
    """Return the full energy spectrum (|lambda_i|^2 / total) for all modes.

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition.

    Returns
    -------
    np.ndarray
        (n_modes,) array of energy fractions summing to 1.
    """
    mags_sq = np.abs(decomp.eigenvalues) ** 2
    total = mags_sq.sum()
    if total == 0:
        return np.zeros_like(mags_sq, dtype=np.float64)
    return mags_sq / total


def reconstruction_error(decomp: EigenDecomp, alpha: AlphaVector) -> float:
    """Frobenius-norm error between W(alpha) and W_original.

    Useful for measuring how far a particular alpha setting deviates from
    the learned transformation.

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition.
    alpha : AlphaVector
        Per-mode alpha vector.

    Returns
    -------
    float
        ||W(alpha) - W_original||_F
    """
    W_alpha = compute_W_alpha(decomp, alpha)
    return float(np.linalg.norm(W_alpha - decomp.W_original, "fro"))
