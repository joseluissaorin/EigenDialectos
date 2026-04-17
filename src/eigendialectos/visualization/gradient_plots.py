"""Gradient / alpha-intensity visualizations for dialectal transformations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

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
# 1. Alpha vs. metric curves
# ---------------------------------------------------------------------------

def plot_alpha_gradient(
    alpha_values: list[float],
    metrics: dict[str, list[float]],
    dialect: DialectCode,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot metric curves as a function of alpha (dialectal intensity).

    Parameters
    ----------
    alpha_values:
        Horizontal axis values (e.g. 0.0, 0.1, ..., 1.5).
    metrics:
        Mapping from metric name to its values at each alpha.
    dialect:
        The dialect under analysis.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.tab10
    for i, (name, vals) in enumerate(sorted(metrics.items())):
        ax.plot(
            alpha_values,
            vals,
            label=name,
            color=cmap(i % 10),
            linewidth=1.8,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("Alpha (dialectal intensity)")
    ax.set_ylabel("Metric value")
    ax.set_title(f"Alpha gradient -- {dialect_label(dialect)}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Feature activation heatmap across alpha values
# ---------------------------------------------------------------------------

def plot_feature_activation_heatmap(
    alpha_values: list[float],
    features: dict[str, list[float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap of feature activations (rows) across alpha values (columns)."""
    feature_names = sorted(features.keys())
    matrix = np.array([features[f] for f in feature_names])

    fig, ax = plt.subplots(figsize=(max(8, len(alpha_values) * 0.5), max(5, len(feature_names) * 0.4)))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=[f"{a:.2f}" for a in alpha_values],
        yticklabels=feature_names,
        cmap="RdYlBu_r",
        linewidths=0.3,
    )
    ax.set_xlabel("Alpha")
    ax.set_title("Feature activation across dialectal intensity")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Threshold annotations on a score curve
# ---------------------------------------------------------------------------

def plot_threshold_annotations(
    alpha_values: list[float],
    scores: list[float],
    recognition_threshold: float,
    naturalness_threshold: float,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Line plot with vertical threshold annotations.

    Parameters
    ----------
    alpha_values:
        X-axis values.
    scores:
        Y-axis score values (e.g. classifier confidence).
    recognition_threshold:
        Alpha value at which the dialect becomes recognisable.
    naturalness_threshold:
        Alpha value above which output ceases to sound natural.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(alpha_values, scores, color="#0072B2", linewidth=2, marker="o", markersize=4)

    ax.axvline(
        x=recognition_threshold,
        color="#009E73",
        linestyle="--",
        linewidth=1.5,
        label=f"Recognition threshold (alpha={recognition_threshold:.2f})",
    )
    ax.axvline(
        x=naturalness_threshold,
        color="#D55E00",
        linestyle="--",
        linewidth=1.5,
        label=f"Naturalness threshold (alpha={naturalness_threshold:.2f})",
    )

    # Shade the "sweet spot" region
    ax.axvspan(
        recognition_threshold,
        naturalness_threshold,
        alpha=0.1,
        color="#56B4E9",
        label="Sweet spot",
    )

    ax.set_xlabel("Alpha (dialectal intensity)")
    ax.set_ylabel("Score")
    ax.set_title("Dialectal intensity thresholds")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _save_or_return(fig, save_path)
