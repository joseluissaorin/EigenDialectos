"""Numerical stability: condition number, safe inverse, regularization."""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv, pinv


def safe_inverse(M: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Regularized matrix inverse: inv(M + reg * I)."""
    n = M.shape[0]
    return inv(M + reg * np.eye(n))


def check_condition(W: np.ndarray) -> float:
    """Condition number of W."""
    return float(np.linalg.cond(W))


def regularize_W(
    W: np.ndarray,
    method: str = "tikhonov",
    strength: float = 1e-4,
) -> np.ndarray:
    """Regularize a transformation matrix.

    Parameters
    ----------
    method : str
        "tikhonov" → (1-s)*W + s*I
        "pseudo" → use pseudo-inverse reconstruction
    strength : float
        Regularization strength (0 = no regularization, 1 = identity).
    """
    n = W.shape[0]
    if method == "tikhonov":
        return (1 - strength) * W + strength * np.eye(n)
    elif method == "pseudo":
        # SVD-based regularization: clip small singular values
        U, S, Vt = np.linalg.svd(W)
        S_reg = np.where(S > strength, S, strength)
        return (U * S_reg) @ Vt
    else:
        raise ValueError(f"Unknown method: {method}")


def pseudo_inverse_fallback(W: np.ndarray, cond_threshold: float = 1e6) -> np.ndarray:
    """Use pseudo-inverse if condition number exceeds threshold."""
    cond = check_condition(W)
    if cond > cond_threshold:
        return pinv(W)
    return inv(W)
