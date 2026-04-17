"""Eigendecomposition W = P @ diag(eigenvalues) @ P_inv, eigenspectrum, and entropy."""

from __future__ import annotations

import logging

import numpy as np
from scipy import linalg

from eigen3.types import EigenDecomp, EigenSpectrum

logger = logging.getLogger(__name__)


def eigendecompose(W: np.ndarray, variety: str = "") -> EigenDecomp:
    """Eigendecompose W = P @ diag(eigenvalues) @ P_inv.

    Eigenvalues sorted by magnitude (descending).
    """
    W = W.astype(np.float64)
    eigenvalues, P = linalg.eig(W)

    # Sort by magnitude (descending)
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    P = P[:, order]

    # Compute P_inv
    P_inv = linalg.inv(P)

    return EigenDecomp(
        P=P,
        eigenvalues=eigenvalues,
        P_inv=P_inv,
        W_original=W.copy(),
        variety=variety,
    )


def eigenspectrum(eigenvalues: np.ndarray) -> EigenSpectrum:
    """Compute eigenspectrum: sorted magnitudes, Shannon entropy, effective rank."""
    magnitudes = np.sort(np.abs(eigenvalues))[::-1]

    # Shannon entropy on normalized magnitudes
    total = magnitudes.sum()
    if total > 0:
        p = magnitudes / total
        p = p[p > 0]  # remove zeros for log
        entropy = float(-np.sum(p * np.log(p)))
    else:
        entropy = 0.0

    # Effective rank = exp(entropy)
    effective_rank = int(round(np.exp(entropy)))

    return EigenSpectrum(
        magnitudes=magnitudes,
        entropy=entropy,
        effective_rank=effective_rank,
    )


def reconstruct_W(decomp: EigenDecomp, k: int | None = None) -> np.ndarray:
    """Reconstruct W from eigendecomposition, optionally using top-k eigenvalues.

    If k is None, uses all eigenvalues (full reconstruction).
    """
    eigenvalues = decomp.eigenvalues.copy()
    if k is not None:
        eigenvalues[k:] = 0.0
    Lambda = np.diag(eigenvalues)
    return (decomp.P @ Lambda @ decomp.P_inv).real
