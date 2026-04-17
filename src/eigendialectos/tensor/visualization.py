"""Visualization utilities for tensor decomposition results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eigendialectos.constants import DialectCode
from eigendialectos.types import TensorDialectal

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import tensorly as tl

    HAS_TENSORLY = True
except ImportError:
    HAS_TENSORLY = False


def _require_matplotlib() -> None:
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib and seaborn are required for visualization. "
            "Install with: pip install matplotlib seaborn"
        )


def plot_core_tensor(
    core: np.ndarray, save_path: Path | None = None
) -> None:
    """Heatmap visualization of Tucker core tensor slices.

    Shows one heatmap per slice along the third mode (dialect factor axis).

    Parameters
    ----------
    core : np.ndarray
        Core tensor G from Tucker decomposition, shape (r1, r2, r3).
    save_path : Path | None
        If given, save the figure to this path instead of showing it.
    """
    _require_matplotlib()

    n_slices = core.shape[2]
    n_cols = min(4, n_slices)
    n_rows = (n_slices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False
    )

    vmax = float(np.max(np.abs(core)))
    vmin = -vmax

    for k in range(n_slices):
        row, col = divmod(k, n_cols)
        ax = axes[row][col]
        sns.heatmap(
            core[:, :, k],
            ax=ax,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            center=0,
            square=True,
            cbar=True,
            cbar_kws={"shrink": 0.7},
        )
        ax.set_title(f"Core slice k={k}")
        ax.set_xlabel("Factor (mode 2)")
        ax.set_ylabel("Factor (mode 1)")

    # Hide unused axes
    for k in range(n_slices, n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Tucker Core Tensor Slices", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_factor_loadings(
    factors: list[np.ndarray],
    dialect_codes: list[DialectCode],
    save_path: Path | None = None,
) -> None:
    """Bar charts of factor loadings per dialect (third-mode factor matrix).

    Parameters
    ----------
    factors : list[np.ndarray]
        Factor matrices [A, B, C] from decomposition.
    dialect_codes : list[DialectCode]
        Ordered dialect codes corresponding to the third mode.
    save_path : Path | None
        If given, save the figure instead of displaying it.
    """
    _require_matplotlib()

    C = factors[2]  # shape (m, n_factors)
    n_factors = C.shape[1]
    n_dialects = C.shape[0]
    labels = [code.value for code in dialect_codes[:n_dialects]]

    n_cols = min(4, n_factors)
    n_rows = (n_factors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )

    colors = sns.color_palette("husl", n_dialects)

    for r in range(n_factors):
        row, col = divmod(r, n_cols)
        ax = axes[row][col]
        vals = C[:, r]
        bars = ax.bar(range(n_dialects), vals, color=colors)
        ax.set_xticks(range(n_dialects))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Factor {r}")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("Loading")

    for k in range(n_factors, n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Dialect Factor Loadings", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_reconstruction_error(
    tensor: TensorDialectal,
    max_rank: int = 20,
    save_path: Path | None = None,
) -> None:
    """Scree plot of CP reconstruction error vs. rank.

    Parameters
    ----------
    tensor : TensorDialectal
        The multi-dialect tensor.
    max_rank : int
        Maximum CP rank to evaluate.
    save_path : Path | None
        If given, save figure instead of showing.
    """
    _require_matplotlib()
    if not HAS_TENSORLY:
        raise ImportError("tensorly is required for reconstruction error plot.")

    from eigendialectos.tensor.cp import cp_decompose

    data = tensor.data.astype(np.float64)
    total_norm = float(np.linalg.norm(data))

    ranks = list(range(1, max_rank + 1))
    errors: list[float] = []
    relative_errors: list[float] = []

    for r in ranks:
        try:
            result = cp_decompose(tensor, rank=r, n_restarts=3)
            err = result["reconstruction_error"]
        except RuntimeError:
            err = float("nan")

        errors.append(err)
        rel_err = err / total_norm if total_norm > 0 else float("nan")
        relative_errors.append(rel_err)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ranks, errors, "o-", color="steelblue", markersize=4)
    ax1.set_xlabel("CP Rank")
    ax1.set_ylabel("Reconstruction Error (Frobenius)")
    ax1.set_title("Absolute Reconstruction Error")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ranks, relative_errors, "s-", color="darkorange", markersize=4)
    ax2.set_xlabel("CP Rank")
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Relative Reconstruction Error")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Reconstruction Error vs. CP Rank", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
