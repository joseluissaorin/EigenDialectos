"""Factor analysis utilities for tensor decomposition results."""

from __future__ import annotations

import numpy as np

from eigendialectos.constants import DialectCode


def analyze_factors(
    decomposition: dict,
    dialect_codes: list[DialectCode],
    vocab: list[str] | None = None,
) -> dict:
    """Map decomposition factor vectors back to linguistic interpretations.

    Works with both Tucker and CP decomposition results.

    Parameters
    ----------
    decomposition : dict
        Result from ``tucker_decompose`` or ``cp_decompose``.
    dialect_codes : list[DialectCode]
        Ordered dialect codes corresponding to the third tensor mode.
    vocab : list[str] | None
        Optional vocabulary labels for the first two modes.

    Returns
    -------
    dict
        Keys:
        - ``dialect_loadings``: dict mapping each DialectCode to its
          factor loading vector (from the third factor matrix C).
        - ``factor_variance``: per-factor explained variance proportion.
        - ``top_vocab_per_factor``: if vocab given, top-loaded vocabulary
          items per factor (from factor matrix A).
        - ``n_factors``: number of components / factors.
    """
    factors = decomposition["factor_matrices"]

    # Third-mode factor matrix gives dialect loadings
    C = factors[2]  # shape (m, r) or (m, n_components)
    n_factors = C.shape[1]

    # Dialect loadings: map each dialect to its row in C
    dialect_loadings: dict[str, np.ndarray] = {}
    for i, code in enumerate(dialect_codes):
        if i < C.shape[0]:
            dialect_loadings[code.value] = C[i, :].copy()

    # Per-factor variance contribution
    # For Tucker: use core tensor norms per slice
    # For CP: use weights if available
    if "weights" in decomposition:
        weights = decomposition["weights"]
        total_w = float(np.sum(weights**2))
        factor_variance = (
            (weights**2 / total_w).tolist() if total_w > 0 else [1.0 / n_factors] * n_factors
        )
    elif "core_tensor" in decomposition:
        core = decomposition["core_tensor"]
        # Variance per factor from core tensor
        factor_var = []
        for r in range(min(n_factors, core.shape[2])):
            slice_norm = float(np.sum(core[:, :, r] ** 2))
            factor_var.append(slice_norm)
        total = sum(factor_var) if factor_var else 1.0
        factor_variance = [v / total for v in factor_var] if total > 0 else []
    else:
        factor_variance = [1.0 / n_factors] * n_factors

    result: dict = {
        "dialect_loadings": dialect_loadings,
        "factor_variance": factor_variance,
        "n_factors": n_factors,
    }

    # Top vocabulary items per factor (first-mode factor matrix A)
    if vocab is not None:
        A = factors[0]  # shape (d, r)
        top_vocab: dict[int, list[str]] = {}
        for r in range(n_factors):
            if r < A.shape[1]:
                col = np.abs(A[:, r])
                n_top = min(10, len(vocab), A.shape[0])
                top_indices = np.argsort(col)[-n_top:][::-1]
                top_vocab[r] = [
                    vocab[idx] for idx in top_indices if idx < len(vocab)
                ]
        result["top_vocab_per_factor"] = top_vocab

    return result


def find_shared_factors(
    decomposition: dict, threshold: float = 0.5
) -> list[int]:
    """Find factors where multiple varieties have high loadings.

    A factor is "shared" when at least two dialects have absolute
    loading above *threshold* times the maximum loading on that factor.

    Parameters
    ----------
    decomposition : dict
        Result from tensor decomposition.
    threshold : float
        Relative threshold (0-1) for considering a loading as "high".

    Returns
    -------
    list[int]
        Indices of shared factors.
    """
    factors = decomposition["factor_matrices"]
    C = factors[2]  # dialect mode, shape (m, r)

    shared: list[int] = []
    for r in range(C.shape[1]):
        col = np.abs(C[:, r])
        max_val = float(np.max(col))
        if max_val == 0:
            continue
        high_count = int(np.sum(col >= threshold * max_val))
        if high_count >= 2:
            shared.append(r)

    return shared


def find_variety_specific_factors(
    decomposition: dict, threshold: float = 0.7
) -> dict[str, list[int]]:
    """Find factors dominated by a single dialect variety.

    A factor is "variety-specific" when one dialect's absolute loading
    accounts for more than *threshold* of the total absolute loading
    on that factor.

    Parameters
    ----------
    decomposition : dict
        Result from tensor decomposition.
    threshold : float
        Dominance threshold (0-1).

    Returns
    -------
    dict[str, list[int]]
        Mapping from dialect code value to list of factor indices that
        the dialect dominates.
    """
    factors = decomposition["factor_matrices"]
    C = factors[2]  # shape (m, r)

    specific: dict[str, list[int]] = {}

    for r in range(C.shape[1]):
        col = np.abs(C[:, r])
        total = float(np.sum(col))
        if total == 0:
            continue
        proportions = col / total
        dominant_idx = int(np.argmax(proportions))
        if proportions[dominant_idx] >= threshold:
            # We don't have dialect_codes here, so use index as key
            key = str(dominant_idx)
            specific.setdefault(key, []).append(r)

    return specific
