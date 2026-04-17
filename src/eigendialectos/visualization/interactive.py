"""Interactive Plotly-based visualizations for dialectal analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from eigendialectos.constants import DialectCode
from eigendialectos.types import DialectalSpectrum, EmbeddingMatrix
from eigendialectos.visualization._colors import DIALECT_COLORS, dialect_label


# ---------------------------------------------------------------------------
# 1. Spectral dashboard
# ---------------------------------------------------------------------------

def create_spectral_dashboard(
    spectra: dict[DialectCode, DialectalSpectrum],
    distances: np.ndarray,
    entropies: dict[DialectCode, float],
) -> go.Figure:
    """Interactive plotly figure with dropdown for dialect selection.

    Panels:
      - Eigenvalue decay (all dialects, selected highlighted)
      - Cumulative energy (all dialects, selected highlighted)
      - Entropy bar chart
    """
    dialects = sorted(spectra.keys(), key=lambda d: d.value)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Eigenvalue decay",
            "Cumulative energy",
            "Spectral entropy",
            "Distance matrix",
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "heatmap"}],
        ],
    )

    # --- Eigenvalue decay traces (one per dialect) ---
    for dc in dialects:
        vals = np.abs(spectra[dc].eigenvalues_sorted)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(vals))),
                y=vals.tolist(),
                mode="lines",
                name=dialect_label(dc),
                line=dict(color=DIALECT_COLORS.get(dc, "#333333")),
                legendgroup=dc.value,
                showlegend=True,
            ),
            row=1, col=1,
        )

    # --- Cumulative energy traces ---
    for dc in dialects:
        cum = spectra[dc].cumulative_energy
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cum))),
                y=cum.tolist(),
                mode="lines",
                name=dialect_label(dc),
                line=dict(color=DIALECT_COLORS.get(dc, "#333333")),
                legendgroup=dc.value,
                showlegend=False,
            ),
            row=1, col=2,
        )

    # --- Entropy bar chart ---
    sorted_items = sorted(entropies.items(), key=lambda x: x[1], reverse=True)
    fig.add_trace(
        go.Bar(
            x=[dialect_label(dc) for dc, _ in sorted_items],
            y=[v for _, v in sorted_items],
            marker_color=[DIALECT_COLORS.get(dc, "#333333") for dc, _ in sorted_items],
            showlegend=False,
        ),
        row=2, col=1,
    )

    # --- Distance heatmap ---
    labels = [dialect_label(dc) for dc in dialects]
    fig.add_trace(
        go.Heatmap(
            z=distances.tolist(),
            x=labels,
            y=labels,
            colorscale="YlOrRd",
            showlegend=False,
        ),
        row=2, col=2,
    )

    # --- Dropdown to highlight a single dialect ---
    buttons = []
    n_dialects = len(dialects)
    for idx, dc in enumerate(dialects):
        # Visibility: all eigenvalue traces + all cum energy traces + entropy bar + heatmap
        vis = []
        # eigenvalue traces
        for j in range(n_dialects):
            vis.append(True)
        # cum energy traces
        for j in range(n_dialects):
            vis.append(True)
        # entropy + heatmap always visible
        vis.append(True)
        vis.append(True)

        # Opacity trick: widen the selected dialect trace
        line_widths_ev = [1 if j != idx else 3.5 for j in range(n_dialects)]
        line_widths_ce = [1 if j != idx else 3.5 for j in range(n_dialects)]

        update_traces: list[dict] = []
        for j in range(n_dialects):
            update_traces.append({"line.width": line_widths_ev[j]})
        for j in range(n_dialects):
            update_traces.append({"line.width": line_widths_ce[j]})

        buttons.append(
            dict(
                label=dialect_label(dc),
                method="update",
                args=[{"visible": vis}],
            )
        )

    buttons.insert(
        0,
        dict(
            label="All dialects",
            method="update",
            args=[{"visible": [True] * (2 * n_dialects + 2)}],
        ),
    )

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.5,
                xanchor="center",
                y=1.12,
                yanchor="top",
            )
        ],
        height=750,
        title_text="Spectral Dashboard",
        template="plotly_white",
    )

    fig.update_xaxes(title_text="Index", row=1, col=1)
    fig.update_yaxes(title_text="|eigenvalue|", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Components", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative energy", row=1, col=2)

    return fig


# ---------------------------------------------------------------------------
# 2. Embedding explorer (3D scatter)
# ---------------------------------------------------------------------------

def create_embedding_explorer(
    embeddings: dict[DialectCode, EmbeddingMatrix],
    vocab: dict[DialectCode, list[str]],
    dialect_codes: list[DialectCode],
) -> go.Figure:
    """3D PCA scatter of word embeddings with hover annotations showing words."""
    # Collect data
    all_data: list[np.ndarray] = []
    all_labels: list[DialectCode] = []
    all_words: list[str] = []

    rng = np.random.default_rng(42)
    n_sample = 300

    for dc in dialect_codes:
        if dc not in embeddings:
            continue
        mat = embeddings[dc].data
        words = vocab.get(dc, embeddings[dc].vocab)
        n = min(n_sample, mat.shape[0])
        idx = rng.choice(mat.shape[0], size=n, replace=False)
        all_data.append(mat[idx])
        all_labels.extend([dc] * n)
        all_words.extend([words[i] for i in idx])

    combined = np.vstack(all_data)
    pca = PCA(n_components=3, random_state=42)
    proj = pca.fit_transform(combined)

    fig = go.Figure()
    for dc in dialect_codes:
        mask = np.array([l == dc for l in all_labels])
        if not np.any(mask):
            continue
        words_dc = [w for w, m in zip(all_words, mask) if m]
        fig.add_trace(
            go.Scatter3d(
                x=proj[mask, 0].tolist(),
                y=proj[mask, 1].tolist(),
                z=proj[mask, 2].tolist(),
                mode="markers",
                name=dialect_label(dc),
                marker=dict(
                    size=3,
                    color=DIALECT_COLORS.get(dc, "#333333"),
                    opacity=0.7,
                ),
                text=words_dc,
                hovertemplate="%{text}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        title="3D Embedding Explorer (PCA)",
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})",
        ),
        height=700,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Gradient slider
# ---------------------------------------------------------------------------

def create_gradient_slider(
    alpha_data: dict[str, object],
) -> go.Figure:
    """Interactive alpha slider showing transformation progression.

    Parameters
    ----------
    alpha_data:
        Dictionary with keys:
        - "alpha_values": list[float]
        - "features": dict[str, list[float]]  (feature_name -> values at each alpha)
        - "scores": list[float] (optional overall score at each alpha)
    """
    alpha_values: list[float] = alpha_data["alpha_values"]  # type: ignore[assignment]
    features: dict[str, list[float]] = alpha_data.get("features", {})  # type: ignore[assignment]
    scores: list[float] = alpha_data.get("scores", [])  # type: ignore[assignment]

    fig = go.Figure()

    # Add a trace per feature
    for name, vals in sorted(features.items()):
        fig.add_trace(
            go.Scatter(
                x=alpha_values,
                y=vals,
                mode="lines+markers",
                name=name,
            )
        )

    # Add overall score if provided
    if scores:
        fig.add_trace(
            go.Scatter(
                x=alpha_values,
                y=scores,
                mode="lines+markers",
                name="Overall score",
                line=dict(width=3, dash="dash"),
            )
        )

    # Build slider steps
    steps = []
    for i, alpha in enumerate(alpha_values):
        step = dict(
            method="update",
            label=f"{alpha:.2f}",
            args=[
                {},
                {
                    "title": f"Dialectal transformation at alpha = {alpha:.2f}",
                    "shapes": [
                        dict(
                            type="line",
                            x0=alpha, x1=alpha,
                            y0=0, y1=1,
                            yref="paper",
                            line=dict(color="red", width=2, dash="dot"),
                        )
                    ],
                },
            ],
        )
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Alpha: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(
        sliders=sliders,
        title="Dialectal transformation progression",
        xaxis_title="Alpha",
        yaxis_title="Value",
        height=550,
        template="plotly_white",
    )

    return fig
