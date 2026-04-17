"""Dialectal entropy computation and comparison."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from eigendialectos.constants import DialectCode
from eigendialectos.spectral.utils import handle_complex_eigenvalues, stable_log


def compute_dialectal_entropy(
    spectrum_or_eigenvalues: object,
    epsilon: float = 1e-10,
    base: str = "natural",
) -> float:
    r"""Compute the spectral entropy of a dialect's eigenvalue distribution.

    .. math::
        H_i = -\sum_j p_j \ln p_j, \quad p_j = \frac{|\lambda_j|}{\sum_k |\lambda_k|}

    Parameters
    ----------
    spectrum_or_eigenvalues : DialectalSpectrum or ndarray
        Either a ``DialectalSpectrum`` (uses ``.eigenvalues_sorted``) or a
        raw eigenvalue array.
    epsilon : float
        Small constant to avoid ``log(0)``.
    base : str
        Logarithm base: ``'natural'`` (ln), ``'2'`` (log2), or ``'10'``
        (log10).

    Returns
    -------
    float
        Non-negative entropy value.

    Raises
    ------
    ValueError
        If *base* is not recognised.
    """
    # Import here to avoid circular import at module level
    from eigendialectos.types import DialectalSpectrum

    if isinstance(spectrum_or_eigenvalues, DialectalSpectrum):
        eigenvalues = np.asarray(
            spectrum_or_eigenvalues.eigenvalues_sorted, dtype=np.float64
        )
    else:
        eigenvalues = np.asarray(spectrum_or_eigenvalues)

    # Convert complex to real magnitudes if needed
    if np.iscomplexobj(eigenvalues):
        eigenvalues = handle_complex_eigenvalues(eigenvalues, method="magnitude")

    magnitudes = np.abs(eigenvalues).astype(np.float64)

    total = np.sum(magnitudes)
    if total < epsilon:
        return 0.0

    # Normalise to probability distribution
    probs = magnitudes / total

    # Compute entropy: H = -sum(p * log(p))
    # Only include terms where p > 0 to avoid 0*log(0)
    mask = probs > 0
    p = probs[mask]

    if base == "natural":
        log_p = np.log(np.maximum(p, epsilon))
    elif base == "2":
        log_p = np.log2(np.maximum(p, epsilon))
    elif base == "10":
        log_p = np.log10(np.maximum(p, epsilon))
    else:
        raise ValueError(f"Unknown base {base!r}. Use 'natural', '2', or '10'.")

    entropy = -float(np.sum(p * log_p))
    return entropy


def compare_entropies(
    entropies: dict[DialectCode, float],
) -> dict:
    """Compare entropy values across dialects.

    Parameters
    ----------
    entropies : dict
        Mapping from ``DialectCode`` to entropy values.

    Returns
    -------
    dict
        * ``'rankings'`` -- list of ``(DialectCode, entropy)`` sorted
          descending by entropy.
        * ``'mean'`` -- mean entropy.
        * ``'std'`` -- standard deviation of entropies.
        * ``'min'`` -- ``(DialectCode, entropy)`` with lowest entropy.
        * ``'max'`` -- ``(DialectCode, entropy)`` with highest entropy.
        * ``'range'`` -- difference between max and min entropy.
        * ``'interpretation'`` -- textual summary.
    """
    if not entropies:
        return {
            "rankings": [],
            "mean": 0.0,
            "std": 0.0,
            "min": None,
            "max": None,
            "range": 0.0,
            "interpretation": "No entropy data provided.",
        }

    values = np.array(list(entropies.values()), dtype=np.float64)
    codes = list(entropies.keys())

    rankings = sorted(zip(codes, values, strict=False), key=lambda x: -x[1])
    mean_h = float(np.mean(values))
    std_h = float(np.std(values))

    min_idx = int(np.argmin(values))
    max_idx = int(np.argmax(values))
    min_pair = (codes[min_idx], float(values[min_idx]))
    max_pair = (codes[max_idx], float(values[max_idx]))

    range_h = float(values[max_idx] - values[min_idx])

    # Interpretation
    if range_h < 0.1:
        interp = (
            "All dialects have very similar spectral entropy, suggesting "
            "comparable structural complexity."
        )
    elif range_h < 0.5:
        interp = (
            f"Moderate variation in spectral entropy. "
            f"{max_pair[0].value} shows highest complexity "
            f"(H={max_pair[1]:.4f}), while {min_pair[0].value} shows "
            f"lowest (H={min_pair[1]:.4f})."
        )
    else:
        interp = (
            f"Substantial variation in spectral entropy. "
            f"{max_pair[0].value} has significantly higher complexity "
            f"(H={max_pair[1]:.4f}) compared to {min_pair[0].value} "
            f"(H={min_pair[1]:.4f}), suggesting different degrees of "
            f"dialectal divergence from the reference variety."
        )

    return {
        "rankings": [(code, float(val)) for code, val in rankings],
        "mean": mean_h,
        "std": std_h,
        "min": min_pair,
        "max": max_pair,
        "range": range_h,
        "interpretation": interp,
    }
