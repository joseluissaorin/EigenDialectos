"""Dialect relationship visualizations: distance matrices, dendrograms, MDS."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from sklearn.manifold import MDS  # noqa: E402

from eigendialectos.constants import DialectCode  # noqa: E402
from eigendialectos.visualization._colors import (  # noqa: E402
    DIALECT_COLORS,
    dialect_label,
)


def _save_or_return(fig: plt.Figure, save_path: Optional[Path]) -> plt.Figure:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 1. Distance matrix heatmap
# ---------------------------------------------------------------------------

def plot_dialect_distance_matrix(
    distances: np.ndarray,
    dialect_codes: list[DialectCode],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap of pairwise dialect distances."""
    labels = [dialect_label(dc) for dc in dialect_codes]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(7, n * 0.8), max(6, n * 0.7)))
    sns.heatmap(
        distances,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        square=True,
    )
    ax.set_title("Pairwise dialect distances")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Dendrogram
# ---------------------------------------------------------------------------

def plot_dialect_dendrogram(
    distances: np.ndarray,
    dialect_codes: list[DialectCode],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Hierarchical clustering dendrogram from a distance matrix."""
    labels = [dialect_label(dc) for dc in dialect_codes]

    # Convert square distance matrix to condensed form
    condensed = squareform(distances, checks=False)
    linked = linkage(condensed, method="ward")

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.0), 6))
    dendrogram(
        linked,
        labels=labels,
        ax=ax,
        leaf_rotation=35,
        leaf_font_size=10,
        color_threshold=0,
        above_threshold_color="#333333",
    )
    ax.set_title("Hierarchical clustering of dialects")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 3. MDS projection
# ---------------------------------------------------------------------------

def plot_dialect_mds(
    distances: np.ndarray,
    dialect_codes: list[DialectCode],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """2D MDS projection of dialects from a pairwise distance matrix."""
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(distances)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, dc in enumerate(dialect_codes):
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            c=DIALECT_COLORS.get(dc, "#333333"),
            s=120,
            zorder=3,
            edgecolors="white",
            linewidth=1.2,
        )
        ax.annotate(
            dialect_label(dc),
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    ax.set_title("MDS projection of dialect distances")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _save_or_return(fig, save_path)
