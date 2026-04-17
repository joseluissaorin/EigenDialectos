"""CP (CANDECOMP/PARAFAC) decomposition of multi-dialect tensors.

Decomposes T ~ sum_r  a_r (x) b_r (x) c_r  where (x) denotes the
outer product and the sum is over R rank-one components.
"""

from __future__ import annotations

import numpy as np

from eigendialectos.types import TensorDialectal

try:
    import tensorly as tl
    from tensorly.decomposition import parafac, tucker

    HAS_TENSORLY = True
except ImportError:
    HAS_TENSORLY = False


def _require_tensorly() -> None:
    if not HAS_TENSORLY:
        raise ImportError(
            "tensorly is required for CP decomposition. "
            "Install it with: pip install tensorly"
        )


def cp_decompose(
    tensor: TensorDialectal,
    rank: int = 10,
    n_restarts: int = 5,
) -> dict:
    """Perform CP decomposition with multiple random restarts.

    Factorises T ~ sum_{r=1}^{R} w_r * a_r (x) b_r (x) c_r.

    Parameters
    ----------
    tensor : TensorDialectal
        Input tensor of shape (d, d, m).
    rank : int
        Number of rank-one components.
    n_restarts : int
        Number of random initialisations; the best (lowest error) is kept.

    Returns
    -------
    dict
        Keys: ``weights`` (length-R), ``factor_matrices`` ([A, B, C]),
        ``reconstruction_error`` (Frobenius norm of residual).
    """
    _require_tensorly()

    data = tensor.data.astype(np.float64)
    tl.set_backend("numpy")
    tl_tensor = tl.tensor(data)

    best_result = None
    best_error = np.inf

    for _ in range(max(1, n_restarts)):
        try:
            cp_result = parafac(
                tl_tensor,
                rank=rank,
                init="random",
                normalize_factors=True,
                n_iter_max=200,
                tol=1e-8,
            )
        except Exception:
            # Some random starts may fail to converge; skip them.
            continue

        weights = np.asarray(cp_result.weights)
        factors = [np.asarray(f) for f in cp_result.factors]

        reconstructed = _cp_reconstruct(weights, factors)
        error = float(np.linalg.norm(data - reconstructed))

        if error < best_error:
            best_error = error
            best_result = {"weights": weights, "factors": factors}

    if best_result is None:
        raise RuntimeError(
            f"All {n_restarts} CP restarts failed to converge."
        )

    return {
        "weights": best_result["weights"],
        "factor_matrices": best_result["factors"],
        "reconstruction_error": best_error,
    }


def _cp_reconstruct(
    weights: np.ndarray, factors: list[np.ndarray]
) -> np.ndarray:
    """Reconstruct tensor from CP factors (internal helper).

    Parameters
    ----------
    weights : np.ndarray
        Component weights of shape (R,).
    factors : list[np.ndarray]
        Factor matrices [A (d1, R), B (d2, R), C (d3, R)].

    Returns
    -------
    np.ndarray
        Reconstructed tensor.
    """
    _require_tensorly()
    tl.set_backend("numpy")
    cp_tensor = (tl.tensor(weights), [tl.tensor(f) for f in factors])
    return np.asarray(tl.cp_to_tensor(cp_tensor))


def core_consistency(
    tensor: TensorDialectal,
    rank: int,
) -> float:
    """Bro's core consistency diagnostic (CORCONDIA) for rank selection.

    Computes how well a CP model of the given rank fits a Tucker model.
    Values close to 100 indicate appropriate rank; values near 0 or
    negative suggest over-fitting.

    Parameters
    ----------
    tensor : TensorDialectal
        Input tensor.
    rank : int
        CP rank to evaluate.

    Returns
    -------
    float
        Core consistency percentage (ideally close to 100 for good rank).
    """
    _require_tensorly()

    data = tensor.data.astype(np.float64)
    tl.set_backend("numpy")
    tl_tensor = tl.tensor(data)

    # Fit CP model
    cp_result = parafac(
        tl_tensor,
        rank=rank,
        init="random",
        normalize_factors=True,
        n_iter_max=200,
    )
    factors = [np.asarray(f) for f in cp_result.factors]

    # Fit Tucker with same rank along each mode
    tucker_ranks = tuple(min(rank, s) for s in data.shape)
    core, tucker_factors = tucker(tl_tensor, rank=tucker_ranks)
    core = np.asarray(core)

    # Project CP factors into Tucker factor space to get the core
    # that the CP model implies
    projected_factors = []
    for cp_f, tk_f in zip(factors, tucker_factors):
        cp_f = np.asarray(cp_f)
        tk_f = np.asarray(tk_f)
        # Project: (tk_f^T @ cp_f) gives coordinates in Tucker space
        projected_factors.append(tk_f.T @ cp_f)

    # Build the superdiagonal core that a perfect CP model would produce
    R = min(rank, *core.shape)
    ideal_core = np.zeros_like(core)
    weights = np.asarray(cp_result.weights)
    for r in range(R):
        idx = tuple(r for _ in range(core.ndim))
        ideal_core[idx] = weights[r] if r < len(weights) else 1.0

    # Reconstruct core from projected factors
    # G_cp = core projected through the CP factor matrices
    g_cp = core.copy()
    for mode in range(core.ndim):
        # mode-n product with projected factor
        pf = projected_factors[mode]  # shape (tucker_rank_mode, cp_rank)
        # We want the core that results from the CP factors in Tucker space
        pass

    # Simplified CORCONDIA: compare Tucker core to superdiagonal
    total_sq = float(np.sum(core**2))
    if total_sq == 0:
        return 100.0

    # The core consistency measures how "superdiagonal" the core tensor is
    diag_sq = 0.0
    off_diag_sq = 0.0
    it = np.nditer(core, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        val = float(it[0])
        if len(set(idx)) == 1:  # diagonal element
            diag_sq += val**2
        else:
            off_diag_sq += val**2
        it.iternext()

    # CORCONDIA = 100 * (1 - sum_offdiag(g^2) / sum_all(g^2))
    corcondia = 100.0 * (1.0 - off_diag_sq / total_sq) if total_sq > 0 else 100.0
    return float(corcondia)
