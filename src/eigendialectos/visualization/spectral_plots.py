"""Spectral analysis visualizations for dialectal eigenvalue spectra."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from eigendialectos.constants import DialectCode  # noqa: E402
from eigendialectos.types import DialectalSpectrum  # noqa: E402
from eigendialectos.visualization._colors import (  # noqa: E402
    DIALECT_COLORS,
    dialect_label,
)


def _save_or_return(fig: plt.Figure, save_path: Optional[Path]) -> plt.Figure:
    """Optionally save figure to disk, then return it."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 1. Grouped bar chart of top-k eigenvalue magnitudes per variety
# ---------------------------------------------------------------------------

def plot_eigenvalue_bars(
    spectra: dict[DialectCode, DialectalSpectrum],
    top_k: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Grouped bar chart of the top-k eigenvalue magnitudes per dialect variety."""
    dialects = sorted(spectra.keys(), key=lambda d: d.value)
    n_dialects = len(dialects)

    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.6), 6))

    bar_width = 0.8 / max(n_dialects, 1)
    indices = np.arange(top_k)

    for i, dc in enumerate(dialects):
        vals = np.abs(spectra[dc].eigenvalues_sorted[:top_k])
        # Pad if fewer eigenvalues than top_k
        if len(vals) < top_k:
            vals = np.concatenate([vals, np.zeros(top_k - len(vals))])
        offset = (i - n_dialects / 2 + 0.5) * bar_width
        ax.bar(
            indices + offset,
            vals,
            width=bar_width,
            label=dialect_label(dc),
            color=DIALECT_COLORS.get(dc, "#333333"),
            alpha=0.85,
        )

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("|eigenvalue|")
    ax.set_title(f"Top-{top_k} eigenvalue magnitudes by dialect")
    ax.set_xticks(indices)
    ax.set_xticklabels([str(j) for j in range(top_k)], fontsize=7)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Eigenvalue decay curves (overlaid lines)
# ---------------------------------------------------------------------------

def plot_eigenvalue_decay(
    spectra: dict[DialectCode, DialectalSpectrum],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlaid line plots of eigenvalue decay for all dialect varieties."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for dc in sorted(spectra.keys(), key=lambda d: d.value):
        vals = np.abs(spectra[dc].eigenvalues_sorted)
        ax.plot(
            np.arange(len(vals)),
            vals,
            label=dialect_label(dc),
            color=DIALECT_COLORS.get(dc, "#333333"),
            linewidth=1.5,
        )

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("|eigenvalue|")
    ax.set_title("Eigenvalue decay curves")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Cumulative energy curves
# ---------------------------------------------------------------------------

def plot_cumulative_energy(
    spectra: dict[DialectCode, DialectalSpectrum],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Cumulative energy curves for each dialect spectrum."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for dc in sorted(spectra.keys(), key=lambda d: d.value):
        cum = spectra[dc].cumulative_energy
        ax.plot(
            np.arange(len(cum)),
            cum,
            label=dialect_label(dc),
            color=DIALECT_COLORS.get(dc, "#333333"),
            linewidth=1.5,
        )

    ax.axhline(y=0.95, color="grey", linestyle="--", linewidth=0.8, label="95% energy")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title("Cumulative spectral energy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Entropy comparison bar chart
# ---------------------------------------------------------------------------

def plot_entropy_comparison(
    entropies: dict[DialectCode, float],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart of dialectal entropy per variety, sorted descending."""
    sorted_items = sorted(entropies.items(), key=lambda x: x[1], reverse=True)
    labels = [dialect_label(dc) for dc, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [DIALECT_COLORS.get(dc, "#333333") for dc, _ in sorted_items]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Spectral entropy")
    ax.set_title("Dialectal entropy comparison")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Eigenspectrum heatmap
# ---------------------------------------------------------------------------

def plot_eigenspectrum_heatmap(
    spectra: dict[DialectCode, DialectalSpectrum],
    top_k: int = 30,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap with dialects as rows and eigenvalue index as columns."""
    dialects = sorted(spectra.keys(), key=lambda d: d.value)
    labels = [dialect_label(dc) for dc in dialects]

    matrix = np.zeros((len(dialects), top_k))
    for i, dc in enumerate(dialects):
        vals = np.abs(spectra[dc].eigenvalues_sorted[:top_k])
        matrix[i, : len(vals)] = vals

    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.35), max(4, len(dialects) * 0.6)))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=[str(j) for j in range(top_k)],
        yticklabels=labels,
        cmap="viridis",
        linewidths=0.3,
    )
    ax.set_xlabel("Eigenvalue index")
    ax.set_title("Eigenspectrum heatmap")
    fig.tight_layout()
    return _save_or_return(fig, save_path)
