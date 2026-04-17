"""Tensor decomposition visualizations (factor loadings, CP, reconstruction error)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from eigendialectos.constants import DialectCode  # noqa: E402
from eigendialectos.visualization._colors import dialect_label  # noqa: E402


def _save_or_return(fig: plt.Figure, save_path: Optional[Path]) -> plt.Figure:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 1. Factor loadings heatmap
# ---------------------------------------------------------------------------

def plot_factor_loadings_heatmap(
    factors: np.ndarray,
    dialect_codes: list[DialectCode],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap of factor loadings (dialects x factors).

    Parameters
    ----------
    factors:
        2D array of shape (n_dialects, n_factors).
    dialect_codes:
        Ordered list of dialect codes corresponding to rows.
    """
    labels = [dialect_label(dc) for dc in dialect_codes]
    n_factors = factors.shape[1] if factors.ndim == 2 else 1

    fig, ax = plt.subplots(
        figsize=(max(7, n_factors * 0.8), max(4, len(labels) * 0.5))
    )
    sns.heatmap(
        factors,
        ax=ax,
        xticklabels=[f"Factor {i}" for i in range(n_factors)],
        yticklabels=labels,
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        annot=True,
        fmt=".2f",
    )
    ax.set_title("Factor loadings per dialect")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 2. CP decomposition component visualization
# ---------------------------------------------------------------------------

def plot_cp_components(
    weights: np.ndarray,
    factors: list[np.ndarray],
    top_k: int = 5,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Visualize top-k CP decomposition components.

    Parameters
    ----------
    weights:
        1D array of component weights (length R).
    factors:
        List of factor matrices, one per tensor mode.
    top_k:
        Number of top components to display.
    """
    k = min(top_k, len(weights))
    n_modes = len(factors)

    # Sort by weight magnitude
    order = np.argsort(-np.abs(weights))[:k]
    sorted_weights = weights[order]

    fig, axes = plt.subplots(1, n_modes + 1, figsize=(4 * (n_modes + 1), max(4, k * 0.5)))
    if n_modes + 1 == 1:
        axes = [axes]

    # Component weights bar chart
    ax_w = axes[0]
    colours = plt.cm.viridis(np.linspace(0.2, 0.8, k))
    ax_w.barh(np.arange(k), sorted_weights, color=colours)
    ax_w.set_yticks(np.arange(k))
    ax_w.set_yticklabels([f"Comp {order[i]}" for i in range(k)], fontsize=9)
    ax_w.set_xlabel("Weight")
    ax_w.set_title("Component weights")
    ax_w.invert_yaxis()

    # Factor matrices for each mode
    for m in range(n_modes):
        ax = axes[m + 1]
        factor_sub = factors[m][:, order]  # select top-k columns
        sns.heatmap(
            factor_sub,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            xticklabels=[f"C{order[i]}" for i in range(k)],
            yticklabels=False,
            linewidths=0,
        )
        ax.set_title(f"Mode {m} factors")

    fig.suptitle(f"Top-{k} CP components", fontsize=13)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Reconstruction scree plot
# ---------------------------------------------------------------------------

def plot_reconstruction_scree(
    errors: list[float],
    ranks: list[int],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Scree plot of reconstruction error vs. tensor rank."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(ranks, errors, marker="o", color="#0072B2", linewidth=2, markersize=6)
    ax.fill_between(ranks, errors, alpha=0.1, color="#0072B2")

    ax.set_xlabel("Tensor rank")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Reconstruction error vs. tensor rank")
    ax.grid(True, alpha=0.25)

    # Mark the elbow if there are enough points
    if len(errors) >= 3:
        diffs = np.diff(errors)
        second_diffs = np.diff(diffs)
        if len(second_diffs) > 0:
            elbow_idx = int(np.argmax(second_diffs)) + 1  # index in ranks
            ax.axvline(
                x=ranks[elbow_idx],
                color="#D55E00",
                linestyle="--",
                linewidth=1.2,
                label=f"Elbow at rank {ranks[elbow_idx]}",
            )
            ax.legend(fontsize=9)

    fig.tight_layout()
    return _save_or_return(fig, save_path)
