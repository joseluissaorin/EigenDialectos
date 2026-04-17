"""Tucker decomposition of multi-dialect tensors.

Decomposes T ~ G x_1 A x_2 B x_3 C where G is the core tensor and
A, B, C are factor matrices along each mode.
"""

from __future__ import annotations

import numpy as np

from eigendialectos.types import TensorDialectal

try:
    import tensorly as tl
    from tensorly.decomposition import tucker

    HAS_TENSORLY = True
except ImportError:
    HAS_TENSORLY = False


def _require_tensorly() -> None:
    if not HAS_TENSORLY:
        raise ImportError(
            "tensorly is required for Tucker decomposition. "
            "Install it with: pip install tensorly"
        )


def tucker_decompose(
    tensor: TensorDialectal,
    ranks: tuple[int, int, int] = (10, 10, 4),
) -> dict:
    """Perform Tucker decomposition on the multi-dialect tensor.

    Factorises T ~ G x_1 A x_2 B x_3 C.

    Parameters
    ----------
    tensor : TensorDialectal
        Input tensor of shape (d, d, m).
    ranks : tuple[int, int, int]
        Desired ranks for each mode.  Clamped to the actual tensor
        dimensions when a rank exceeds the corresponding mode size.

    Returns
    -------
    dict
        Keys: ``core_tensor`` (G), ``factor_matrices`` ([A, B, C]),
        ``reconstruction_error`` (Frobenius norm of residual).
    """
    _require_tensorly()

    data = tensor.data.astype(np.float64)
    tensor_shape = data.shape

    # Clamp ranks to actual dimensions
    clamped_ranks = tuple(
        min(r, s) for r, s in zip(ranks, tensor_shape)
    )

    tl.set_backend("numpy")
    tl_tensor = tl.tensor(data)

    core, factors = tucker(tl_tensor, rank=clamped_ranks)

    # Compute reconstruction error
    reconstructed = tucker_reconstruct(core, factors)
    error = float(np.linalg.norm(data - reconstructed))

    return {
        "core_tensor": np.asarray(core),
        "factor_matrices": [np.asarray(f) for f in factors],
        "reconstruction_error": error,
    }


def tucker_reconstruct(
    core: np.ndarray, factors: list[np.ndarray]
) -> np.ndarray:
    """Reconstruct the full tensor from Tucker factors.

    Parameters
    ----------
    core : np.ndarray
        Core tensor G of shape (r1, r2, r3).
    factors : list[np.ndarray]
        Factor matrices [A, B, C] with shapes (d1, r1), (d2, r2), (d3, r3).

    Returns
    -------
    np.ndarray
        Reconstructed tensor of shape (d1, d2, d3).
    """
    _require_tensorly()

    tl.set_backend("numpy")
    tl_core = tl.tensor(np.asarray(core))
    tl_factors = [tl.tensor(np.asarray(f)) for f in factors]

    reconstructed = tl.tucker_to_tensor((tl_core, tl_factors))
    return np.asarray(reconstructed)


def explained_variance(
    tensor: TensorDialectal, core: np.ndarray, factors: list[np.ndarray]
) -> float:
    """Compute proportion of variance explained by the Tucker decomposition.

    Parameters
    ----------
    tensor : TensorDialectal
        Original tensor.
    core : np.ndarray
        Core tensor from decomposition.
    factors : list[np.ndarray]
        Factor matrices from decomposition.

    Returns
    -------
    float
        Value in [0, 1] representing fraction of variance explained.
    """
    _require_tensorly()

    data = tensor.data.astype(np.float64)
    reconstructed = tucker_reconstruct(core, factors)
    residual_norm_sq = float(np.sum((data - reconstructed) ** 2))
    total_norm_sq = float(np.sum(data**2))

    if total_norm_sq == 0.0:
        return 1.0

    return 1.0 - residual_norm_sq / total_norm_sq
