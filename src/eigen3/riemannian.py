"""SPD manifold geometry for transformation matrices.

Geodesic distances, metric tensors, and Ricci curvature on the
symmetric-positive-definite (SPD) manifold.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
from scipy.linalg import logm, sqrtm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_spd(W: np.ndarray) -> np.ndarray:
    """Convert an arbitrary square matrix to SPD via its Gram matrix W^T W.

    A small ridge is added to guarantee strict positive-definiteness.
    """
    G = W.T @ W
    # Symmetrise (guard against float round-off)
    G = 0.5 * (G + G.T)
    # Ridge for strict positive-definiteness
    G += 1e-10 * np.eye(G.shape[0])
    return G


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """Real part of principal matrix square root."""
    return np.real(sqrtm(M))


def _matrix_log(M: np.ndarray) -> np.ndarray:
    """Real part of principal matrix logarithm."""
    return np.real(logm(M))


def _matrix_inv_sqrt(M: np.ndarray) -> np.ndarray:
    """M^{-1/2} via eigendecomposition for numerical stability."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12)
    return (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T


# ---------------------------------------------------------------------------
# Geodesic distance on SPD manifold
# ---------------------------------------------------------------------------

def geodesic_distance(W_a: np.ndarray, W_b: np.ndarray) -> float:
    """Riemannian geodesic distance between two points on the SPD manifold.

    Given transformation matrices W_a, W_b, first projects them onto SPD
    via G = W^T W, then computes::

        d(A, B) = || log(A^{-1/2} B A^{-1/2}) ||_F

    Parameters
    ----------
    W_a, W_b : np.ndarray
        Arbitrary square matrices (same shape).

    Returns
    -------
    float
        Geodesic distance on the SPD manifold.
    """
    A = _to_spd(W_a)
    B = _to_spd(W_b)

    A_inv_sqrt = _matrix_inv_sqrt(A)
    inner = A_inv_sqrt @ B @ A_inv_sqrt

    # Symmetrise to avoid spurious imaginary parts in logm
    inner = 0.5 * (inner + inner.T)
    log_inner = _matrix_log(inner)

    return float(np.linalg.norm(log_inner, "fro"))


# ---------------------------------------------------------------------------
# Metric tensor (numerical approximation)
# ---------------------------------------------------------------------------

def metric_tensor(W: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Numerical metric tensor at point W on the SPD manifold.

    Perturbs W in each pair of matrix directions (i,j) and (k,l) and
    computes the local inner product via finite differences of the
    geodesic distance.

    The returned tensor is indexed by flattened matrix entries, giving
    shape (n*n, n*n).

    Parameters
    ----------
    W : np.ndarray
        Square matrix defining the manifold point (projected to SPD internally).
    epsilon : float
        Perturbation magnitude.

    Returns
    -------
    np.ndarray
        (n*n, n*n) symmetric metric tensor.
    """
    n = W.shape[0]
    dim = n * n
    G = np.zeros((dim, dim))

    for a in range(dim):
        for b in range(a, dim):
            i_a, j_a = divmod(a, n)
            i_b, j_b = divmod(b, n)

            # Perturbation matrices
            E_a = np.zeros_like(W)
            E_a[i_a, j_a] = epsilon
            E_b = np.zeros_like(W)
            E_b[i_b, j_b] = epsilon

            # Four-point stencil: g_{ab} ≈ (d(W+Ea+Eb, W) + d(W, W)
            #   - d(W+Ea, W) - d(W+Eb, W)) / eps^2
            d_pp = geodesic_distance(W + E_a + E_b, W)
            d_00 = 0.0  # d(W, W) = 0
            d_p0 = geodesic_distance(W + E_a, W)
            d_0p = geodesic_distance(W + E_b, W)

            g_ab = (d_pp ** 2 - d_p0 ** 2 - d_0p ** 2 + d_00 ** 2) / (2.0 * epsilon ** 2)
            G[a, b] = g_ab
            G[b, a] = g_ab

    return G


# ---------------------------------------------------------------------------
# Ollivier-Ricci curvature proxy
# ---------------------------------------------------------------------------

def ricci_curvature(W_dict: dict[str, np.ndarray]) -> dict[str, float]:
    """Ollivier-Ricci curvature proxy for each variety.

    Uses pairwise geodesic distances to estimate curvature as the
    normalised deficit between the average geodesic distance to
    neighbours and the expected flat-space distance.

    For each variety *v*, the curvature proxy is::

        kappa(v) = 1 - mean_w[ d_geodesic(v,w) ] / d_mean

    where d_mean is the global mean geodesic distance. Positive values
    indicate local clustering (positive curvature); negative values
    indicate local spreading (negative curvature).

    Parameters
    ----------
    W_dict : dict[str, np.ndarray]
        Variety name -> transformation matrix.

    Returns
    -------
    dict[str, float]
        Variety name -> curvature proxy.
    """
    labels = sorted(W_dict.keys())
    n = len(labels)

    # Pairwise geodesic distance matrix
    D = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        d = geodesic_distance(W_dict[labels[i]], W_dict[labels[j]])
        D[i, j] = d
        D[j, i] = d

    # Global mean distance (excluding diagonal)
    if n < 2:
        return {labels[0]: 0.0} if n == 1 else {}
    d_mean = D.sum() / (n * (n - 1))
    if d_mean < 1e-15:
        return {lab: 0.0 for lab in labels}

    curvatures: dict[str, float] = {}
    for i, lab in enumerate(labels):
        # Average distance from this variety to all others
        neighbours = np.delete(D[i], i)
        avg_d = neighbours.mean()
        curvatures[lab] = 1.0 - avg_d / d_mean

    return curvatures
