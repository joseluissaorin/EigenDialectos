"""Matrix Lie algebra operations for transformation matrices.

Provides generators, commutators, bracket matrices, interpolation,
and round-trip consistency checks via logm / expm.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
from scipy.linalg import expm, logm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lie algebra generator
# ---------------------------------------------------------------------------

def generator(W: np.ndarray) -> np.ndarray:
    """Compute the Lie algebra element A = logm(W).

    Parameters
    ----------
    W : np.ndarray
        Square matrix (transformation matrix).

    Returns
    -------
    np.ndarray
        Lie algebra element (may be complex if W has negative eigenvalues).
    """
    return logm(W.astype(np.float64))


# ---------------------------------------------------------------------------
# Commutator
# ---------------------------------------------------------------------------

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix commutator [A, B] = AB - BA.

    Parameters
    ----------
    A, B : np.ndarray
        Square matrices of the same shape.

    Returns
    -------
    np.ndarray
        The commutator matrix.
    """
    return A @ B - B @ A


# ---------------------------------------------------------------------------
# Bracket matrix (pairwise commutator norms)
# ---------------------------------------------------------------------------

def bracket_matrix(
    generators: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Pairwise commutator Frobenius norms.

    For each pair (i, j) of generators, computes::

        B[i, j] = || [gen_i, gen_j] ||_F

    Parameters
    ----------
    generators : dict[str, np.ndarray]
        Variety name -> Lie algebra element (from :func:`generator`).

    Returns
    -------
    B : np.ndarray
        (n, n) symmetric matrix of commutator norms.
    labels : list[str]
        Variety names in matrix order.
    """
    labels = sorted(generators.keys())
    n = len(labels)
    B = np.zeros((n, n))

    for i, j in combinations(range(n), 2):
        C = commutator(generators[labels[i]], generators[labels[j]])
        norm = float(np.linalg.norm(C, "fro"))
        B[i, j] = norm
        B[j, i] = norm

    return B, labels


# ---------------------------------------------------------------------------
# Lie-group geodesic interpolation
# ---------------------------------------------------------------------------

def lie_interpolate(
    W_a: np.ndarray,
    W_b: np.ndarray,
    t: float,
) -> np.ndarray:
    """Geodesic interpolation on the matrix Lie group.

    Computes::

        W(t) = expm((1-t) * logm(W_a) + t * logm(W_b))

    Parameters
    ----------
    W_a, W_b : np.ndarray
        Square matrices of the same shape.
    t : float
        Interpolation parameter in [0, 1].
        t=0 returns W_a, t=1 returns W_b.

    Returns
    -------
    np.ndarray
        Interpolated matrix on the Lie group.
    """
    W_a = W_a.astype(np.float64)
    W_b = W_b.astype(np.float64)

    log_a = logm(W_a)
    log_b = logm(W_b)

    interpolated_log = (1.0 - t) * log_a + t * log_b
    return np.real(expm(interpolated_log))


# ---------------------------------------------------------------------------
# Round-trip consistency check
# ---------------------------------------------------------------------------

def roundtrip_check(W: np.ndarray) -> float:
    """Measure round-trip error || expm(logm(W)) - W ||_F.

    A small value confirms that the matrix logarithm and exponential
    are numerically consistent for this matrix.

    Parameters
    ----------
    W : np.ndarray
        Square matrix.

    Returns
    -------
    float
        Frobenius norm of the round-trip residual.
    """
    W = W.astype(np.float64)
    W_reconstructed = expm(logm(W))
    return float(np.linalg.norm(np.real(W_reconstructed) - W, "fro"))


# ---------------------------------------------------------------------------
# Convenience: generate all Lie algebra elements
# ---------------------------------------------------------------------------

def generators_from_matrices(
    W_dict: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute Lie algebra generators for all transformation matrices.

    Parameters
    ----------
    W_dict : dict[str, np.ndarray]
        Variety name -> transformation matrix.

    Returns
    -------
    dict[str, np.ndarray]
        Variety name -> Lie algebra element.
    """
    gens: dict[str, np.ndarray] = {}
    for name, W in W_dict.items():
        gens[name] = generator(W)
    return gens


# ---------------------------------------------------------------------------
# Structure constants (advanced)
# ---------------------------------------------------------------------------

def structure_constants(
    generators: dict[str, np.ndarray],
) -> np.ndarray:
    """Approximate structure constants of the Lie algebra spanned by the generators.

    For a set of n generators {A_1, ..., A_n}, the structure constants
    f^k_{ij} are defined by  [A_i, A_j] = sum_k f^k_{ij} A_k.

    We approximate f^k_{ij} by least-squares projection of each
    commutator onto the generator basis.

    Parameters
    ----------
    generators : dict[str, np.ndarray]
        Variety name -> generator matrix.

    Returns
    -------
    np.ndarray
        (n, n, n) array of structure constants f[i, j, k].
    """
    labels = sorted(generators.keys())
    n = len(labels)
    dim = generators[labels[0]].size

    # Stack generators as flattened vectors
    G = np.zeros((n, dim), dtype=np.complex128)
    for idx, lab in enumerate(labels):
        G[idx] = generators[lab].ravel()

    # Pseudo-inverse for projection
    G_pinv = np.linalg.pinv(G)  # (dim, n)

    f = np.zeros((n, n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            C = commutator(generators[labels[i]], generators[labels[j]])
            # Project onto basis: coefficients = G_pinv^T @ C.ravel()
            coeffs = G_pinv.T @ C.ravel()
            f[i, j, :] = coeffs

    return np.real_if_close(f)
