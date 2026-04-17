"""Distance metrics between dialect transformation matrices and spectra."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import wasserstein_distance

from eigendialectos.constants import DialectCode
from eigendialectos.spectral.utils import handle_complex_eigenvalues
from eigendialectos.types import (
    DialectalSpectrum,
    EigenDecomposition,
    TransformationMatrix,
)


def frobenius_distance(
    W_a: TransformationMatrix,
    W_b: TransformationMatrix,
) -> float:
    r"""Frobenius distance between two transformation matrices.

    .. math::
        d_F(W_a, W_b) = \|W_a - W_b\|_F

    Parameters
    ----------
    W_a, W_b : TransformationMatrix

    Returns
    -------
    float
        Non-negative distance.
    """
    A = np.asarray(W_a.data, dtype=np.float64)
    B = np.asarray(W_b.data, dtype=np.float64)
    return float(np.linalg.norm(A - B, "fro"))


def spectral_distance(
    spec_a: DialectalSpectrum,
    spec_b: DialectalSpectrum,
) -> float:
    """Earth Mover's Distance between two eigenvalue distributions.

    Eigenvalue magnitudes are normalised to sum to 1 and the
    Wasserstein-1 distance is computed using ``scipy.stats.wasserstein_distance``.

    Parameters
    ----------
    spec_a, spec_b : DialectalSpectrum

    Returns
    -------
    float
        Non-negative EMD.
    """
    ev_a = np.asarray(spec_a.eigenvalues_sorted, dtype=np.float64)
    ev_b = np.asarray(spec_b.eigenvalues_sorted, dtype=np.float64)

    # Pad to same length
    max_len = max(len(ev_a), len(ev_b))
    a_pad = np.zeros(max_len, dtype=np.float64)
    b_pad = np.zeros(max_len, dtype=np.float64)
    a_pad[: len(ev_a)] = ev_a
    b_pad[: len(ev_b)] = ev_b

    # Normalise
    sum_a = np.sum(a_pad)
    sum_b = np.sum(b_pad)
    if sum_a < 1e-15 or sum_b < 1e-15:
        return 0.0

    norm_a = a_pad / sum_a
    norm_b = b_pad / sum_b

    positions = np.arange(max_len, dtype=np.float64)
    return float(wasserstein_distance(positions, positions, u_weights=norm_a, v_weights=norm_b))


def subspace_distance(
    P_a: npt.NDArray,
    P_b: npt.NDArray,
    k: int = 10,
) -> float:
    r"""Subspace distance using top-*k* eigenvectors.

    .. math::
        d_S = \|P_a^{(k)} {P_a^{(k)}}^\dagger - P_b^{(k)} {P_b^{(k)}}^\dagger\|_F

    where ``P^{(k)}`` denotes the first *k* columns of the eigenvector
    matrix.  We use the Hermitian conjugate for complex eigenvectors.

    Parameters
    ----------
    P_a, P_b : ndarray
        Eigenvector matrices (columns are eigenvectors).
    k : int
        Number of leading eigenvectors to use.

    Returns
    -------
    float
        Non-negative distance.
    """
    P_a = np.asarray(P_a, dtype=np.complex128)
    P_b = np.asarray(P_b, dtype=np.complex128)

    k_a = min(k, P_a.shape[1])
    k_b = min(k, P_b.shape[1])

    Pa_k = P_a[:, :k_a]
    Pb_k = P_b[:, :k_b]

    # Orthonormalise via QR for numerical stability
    Qa, _ = np.linalg.qr(Pa_k)
    Qb, _ = np.linalg.qr(Pb_k)

    # Projection matrices
    proj_a = Qa @ Qa.conj().T
    proj_b = Qb @ Qb.conj().T

    return float(np.linalg.norm(proj_a - proj_b, "fro").real)


def entropy_distance(H_a: float, H_b: float) -> float:
    """Absolute difference between two entropy values.

    Parameters
    ----------
    H_a, H_b : float

    Returns
    -------
    float
        ``|H_a - H_b|``.
    """
    return abs(H_a - H_b)


def combined_distance(
    W_a: TransformationMatrix,
    W_b: TransformationMatrix,
    spec_a: DialectalSpectrum,
    spec_b: DialectalSpectrum,
    H_a: float,
    H_b: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted combination of multiple distance metrics.

    Parameters
    ----------
    W_a, W_b : TransformationMatrix
    spec_a, spec_b : DialectalSpectrum
    H_a, H_b : float
        Entropies.
    weights : dict, optional
        Keys: ``'frobenius'``, ``'spectral'``, ``'entropy'``.
        Default: equal weights normalised to sum to 1.

    Returns
    -------
    float
        Combined distance.
    """
    if weights is None:
        weights = {"frobenius": 1.0, "spectral": 1.0, "entropy": 1.0}

    total_weight = sum(weights.values())
    if total_weight < 1e-15:
        return 0.0

    d_frob = frobenius_distance(W_a, W_b)
    d_spec = spectral_distance(spec_a, spec_b)
    d_ent = entropy_distance(H_a, H_b)

    combined = (
        weights.get("frobenius", 0.0) * d_frob
        + weights.get("spectral", 0.0) * d_spec
        + weights.get("entropy", 0.0) * d_ent
    ) / total_weight

    return float(combined)


def compute_distance_matrix(
    transforms: dict[DialectCode, TransformationMatrix],
    spectra: dict[DialectCode, DialectalSpectrum],
    entropies: dict[DialectCode, float],
    method: str = "combined",
    weights: dict[str, float] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute the full pairwise distance matrix across dialects.

    Parameters
    ----------
    transforms : dict
        Mapping from ``DialectCode`` to ``TransformationMatrix``.
    spectra : dict
        Mapping from ``DialectCode`` to ``DialectalSpectrum``.
    entropies : dict
        Mapping from ``DialectCode`` to entropy values.
    method : str
        Distance metric: ``'frobenius'``, ``'spectral'``, ``'entropy'``,
        or ``'combined'``.
    weights : dict, optional
        Weights for the combined metric (ignored by other methods).

    Returns
    -------
    ndarray, shape (n, n)
        Symmetric distance matrix where ``D[i, j]`` is the distance
        between the *i*-th and *j*-th dialects.  The ordering follows
        the sorted keys of *transforms*.

    Raises
    ------
    ValueError
        If *method* is unknown or if the dict keys are inconsistent.
    """
    codes = sorted(transforms.keys(), key=lambda c: c.value)
    n = len(codes)
    D = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = codes[i], codes[j]

            if method == "frobenius":
                d = frobenius_distance(transforms[ci], transforms[cj])
            elif method == "spectral":
                d = spectral_distance(spectra[ci], spectra[cj])
            elif method == "entropy":
                d = entropy_distance(entropies[ci], entropies[cj])
            elif method == "combined":
                d = combined_distance(
                    transforms[ci],
                    transforms[cj],
                    spectra[ci],
                    spectra[cj],
                    entropies[ci],
                    entropies[cj],
                    weights=weights,
                )
            else:
                raise ValueError(
                    f"Unknown method {method!r}. "
                    "Choose 'frobenius', 'spectral', 'entropy', or 'combined'."
                )

            D[i, j] = d
            D[j, i] = d

    return D
