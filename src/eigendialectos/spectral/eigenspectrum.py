"""Eigenspectrum analysis -- ordering, energy, entropy, comparison."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import wasserstein_distance

from eigendialectos.spectral.entropy import compute_dialectal_entropy
from eigendialectos.spectral.utils import handle_complex_eigenvalues
from eigendialectos.types import (
    DialectalSpectrum,
    EigenDecomposition,
    TransformationMatrix,
)


def compute_eigenspectrum(eigen: EigenDecomposition) -> DialectalSpectrum:
    """Compute the dialectal spectrum from an eigendecomposition.

    The eigenvalues are converted to magnitudes, sorted in descending order,
    and the spectral entropy is computed.

    Parameters
    ----------
    eigen : EigenDecomposition
        Result of :func:`~eigendialectos.spectral.eigendecomposition.eigendecompose`.

    Returns
    -------
    DialectalSpectrum
    """
    magnitudes = handle_complex_eigenvalues(eigen.eigenvalues, method="magnitude")
    # Sort descending
    sorted_indices = np.argsort(magnitudes)[::-1]
    eigenvalues_sorted = magnitudes[sorted_indices]

    entropy = compute_dialectal_entropy(eigenvalues_sorted)

    return DialectalSpectrum(
        eigenvalues_sorted=eigenvalues_sorted,
        entropy=entropy,
        dialect_code=eigen.dialect_code,
    )


def compare_spectra(
    spec_a: DialectalSpectrum,
    spec_b: DialectalSpectrum,
) -> dict:
    """Compare two dialectal spectra and return similarity metrics.

    Parameters
    ----------
    spec_a, spec_b : DialectalSpectrum
        Spectra to compare.

    Returns
    -------
    dict
        Keys:

        * ``'emd'`` -- Earth Mover's Distance between normalised spectra.
        * ``'entropy_diff'`` -- ``|H_a - H_b|``.
        * ``'correlation'`` -- Pearson correlation of eigenvalue vectors
          (truncated to common length).
        * ``'energy_overlap'`` -- dot product of normalised eigenvalue
          vectors (cosine similarity).
        * ``'dialect_a'`` -- dialect code of *spec_a*.
        * ``'dialect_b'`` -- dialect code of *spec_b*.
    """
    ev_a = np.asarray(spec_a.eigenvalues_sorted, dtype=np.float64)
    ev_b = np.asarray(spec_b.eigenvalues_sorted, dtype=np.float64)

    # Pad shorter to same length with zeros
    max_len = max(len(ev_a), len(ev_b))
    ev_a_pad = np.zeros(max_len, dtype=np.float64)
    ev_b_pad = np.zeros(max_len, dtype=np.float64)
    ev_a_pad[: len(ev_a)] = ev_a
    ev_b_pad[: len(ev_b)] = ev_b

    # Normalise for distribution-based metrics
    sum_a = np.sum(ev_a_pad)
    sum_b = np.sum(ev_b_pad)
    norm_a = ev_a_pad / max(sum_a, 1e-15)
    norm_b = ev_b_pad / max(sum_b, 1e-15)

    # Earth Mover's Distance on normalised spectra
    emd = float(wasserstein_distance(
        np.arange(max_len), np.arange(max_len),
        u_weights=norm_a, v_weights=norm_b,
    ))

    # Entropy difference
    entropy_diff = abs(spec_a.entropy - spec_b.entropy)

    # Pearson correlation
    if np.std(ev_a_pad) < 1e-15 or np.std(ev_b_pad) < 1e-15:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(ev_a_pad, ev_b_pad)[0, 1])

    # Energy overlap (cosine similarity of eigenvalue vectors)
    norm_a_l2 = np.linalg.norm(ev_a_pad)
    norm_b_l2 = np.linalg.norm(ev_b_pad)
    if norm_a_l2 < 1e-15 or norm_b_l2 < 1e-15:
        energy_overlap = 0.0
    else:
        energy_overlap = float(np.dot(ev_a_pad, ev_b_pad) / (norm_a_l2 * norm_b_l2))

    return {
        "emd": emd,
        "entropy_diff": entropy_diff,
        "correlation": correlation,
        "energy_overlap": energy_overlap,
        "dialect_a": spec_a.dialect_code,
        "dialect_b": spec_b.dialect_code,
    }


def rank_k_approximation(
    eigen: EigenDecomposition,
    k: int,
) -> TransformationMatrix:
    r"""Reconstruct a rank-*k* approximation of the transformation matrix.

    .. math::
        W_k = P_k \, \operatorname{diag}(\lambda_1, \dots, \lambda_k) \, P_k^{-1}

    where the top-*k* eigenvalues are selected by magnitude.

    Parameters
    ----------
    eigen : EigenDecomposition
        Full eigendecomposition.
    k : int
        Number of eigenvalues to retain.

    Returns
    -------
    TransformationMatrix
        Approximated transformation (real part only).

    Raises
    ------
    ValueError
        If ``k`` is out of range.
    """
    n = len(eigen.eigenvalues)
    if k < 1 or k > n:
        raise ValueError(f"k={k} is out of range [1, {n}]")

    magnitudes = np.abs(eigen.eigenvalues)
    top_k_indices = np.argsort(magnitudes)[::-1][:k]

    P = eigen.eigenvectors
    P_inv = eigen.eigenvectors_inv

    # Build diagonal with only top-k eigenvalues, rest zeroed
    Lambda_k = np.zeros(n, dtype=np.complex128)
    Lambda_k[top_k_indices] = eigen.eigenvalues[top_k_indices]

    W_k = P @ np.diag(Lambda_k) @ P_inv

    # Take real part (imaginary should be negligible for real input)
    W_k_real = np.real(W_k).astype(np.float64)

    return TransformationMatrix(
        data=W_k_real,
        source_dialect=eigen.dialect_code,
        target_dialect=eigen.dialect_code,
        regularization=0.0,
    )
