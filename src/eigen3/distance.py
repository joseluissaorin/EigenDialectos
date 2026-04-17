"""Spectral distance metrics: Frobenius, EMD on eigenvalues, subspace angles."""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd


def frobenius_distance(W1: np.ndarray, W2: np.ndarray) -> float:
    """Frobenius norm of W1 - W2."""
    return float(np.linalg.norm(W1 - W2, "fro"))


def spectral_distance(eig1: np.ndarray, eig2: np.ndarray) -> float:
    """Earth Mover's Distance on sorted eigenvalue magnitudes.

    Both arrays should be sorted descending by magnitude.
    """
    m1 = np.sort(np.abs(eig1))[::-1]
    m2 = np.sort(np.abs(eig2))[::-1]
    n = min(len(m1), len(m2))
    return float(np.sum(np.abs(m1[:n] - m2[:n])))


def subspace_distance(P1: np.ndarray, P2: np.ndarray, k: int = 10) -> float:
    """Subspace angle between top-k eigenvector spans.

    Returns the largest principal angle in [0, pi/2].
    """
    U1 = P1[:, :k].real  # (n, k)
    U2 = P2[:, :k].real

    # Orthogonalize
    Q1, _ = np.linalg.qr(U1)
    Q2, _ = np.linalg.qr(U2)

    # Principal angles via SVD of Q1^T @ Q2
    _, sigmas, _ = svd(Q1.T @ Q2)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(np.max(angles))


def distance_matrix(
    W_dict: dict[str, np.ndarray],
    metric: str = "frobenius",
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise distance matrix for all varieties.

    Parameters
    ----------
    W_dict : dict
        Variety name -> W matrix (or eigenvalues for spectral metric).
    metric : str
        "frobenius" or "spectral"

    Returns
    -------
    D : np.ndarray
        (n, n) symmetric distance matrix.
    labels : list[str]
        Variety names in matrix order.
    """
    labels = sorted(W_dict.keys())
    n = len(labels)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if metric == "frobenius":
                d = frobenius_distance(W_dict[labels[i]], W_dict[labels[j]])
            elif metric == "spectral":
                d = spectral_distance(W_dict[labels[i]], W_dict[labels[j]])
            else:
                raise ValueError(f"Unknown metric: {metric}")
            D[i, j] = d
            D[j, i] = d

    return D, labels
