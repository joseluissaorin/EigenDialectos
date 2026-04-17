"""Embedding space visualizations (t-SNE, UMAP, PCA, alignment quality)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from eigendialectos.constants import DialectCode  # noqa: E402
from eigendialectos.types import EmbeddingMatrix  # noqa: E402
from eigendialectos.visualization._colors import (  # noqa: E402
    DIALECT_COLORS,
    DIALECT_MARKERS,
    dialect_label,
)


def _save_or_return(fig: plt.Figure, save_path: Optional[Path]) -> plt.Figure:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def _subsample_embeddings(
    embeddings: dict[DialectCode, EmbeddingMatrix],
    n_words: int,
) -> tuple[np.ndarray, list[DialectCode]]:
    """Subsample and concatenate embedding matrices, returning data + dialect labels."""
    all_data: list[np.ndarray] = []
    all_labels: list[DialectCode] = []

    rng = np.random.default_rng(42)
    for dc in sorted(embeddings.keys(), key=lambda d: d.value):
        mat = embeddings[dc].data
        n = min(n_words, mat.shape[0])
        idx = rng.choice(mat.shape[0], size=n, replace=False)
        all_data.append(mat[idx])
        all_labels.extend([dc] * n)

    return np.vstack(all_data), all_labels


# ---------------------------------------------------------------------------
# 1. t-SNE
# ---------------------------------------------------------------------------

def plot_embeddings_tsne(
    embeddings: dict[DialectCode, EmbeddingMatrix],
    n_words: int = 500,
    perplexity: int = 30,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """t-SNE projection of word embeddings coloured by dialect."""
    data, labels = _subsample_embeddings(embeddings, n_words)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, len(data) - 1)),
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    proj = tsne.fit_transform(data)

    fig, ax = plt.subplots(figsize=(10, 8))
    for dc in sorted(set(labels), key=lambda d: d.value):
        mask = np.array([l == dc for l in labels])
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            c=DIALECT_COLORS.get(dc, "#333333"),
            marker=DIALECT_MARKERS.get(dc, "o"),
            label=dialect_label(dc),
            alpha=0.6,
            s=15,
            edgecolors="none",
        )

    ax.set_title("t-SNE projection of dialect embeddings")
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 2. UMAP
# ---------------------------------------------------------------------------

def plot_embeddings_umap(
    embeddings: dict[DialectCode, EmbeddingMatrix],
    n_words: int = 500,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """UMAP projection of word embeddings coloured by dialect."""
    import umap  # optional heavy dependency

    data, labels = _subsample_embeddings(embeddings, n_words)

    reducer = umap.UMAP(n_components=2, random_state=42)
    proj = reducer.fit_transform(data)

    fig, ax = plt.subplots(figsize=(10, 8))
    for dc in sorted(set(labels), key=lambda d: d.value):
        mask = np.array([l == dc for l in labels])
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            c=DIALECT_COLORS.get(dc, "#333333"),
            marker=DIALECT_MARKERS.get(dc, "o"),
            label=dialect_label(dc),
            alpha=0.6,
            s=15,
            edgecolors="none",
        )

    ax.set_title("UMAP projection of dialect embeddings")
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Alignment quality (before / after Procrustes)
# ---------------------------------------------------------------------------

def plot_alignment_quality(
    before: dict[DialectCode, EmbeddingMatrix],
    after: dict[DialectCode, EmbeddingMatrix],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Side-by-side PCA scatter: embedding spaces before and after Procrustes alignment."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, data_dict, title in [
        (axes[0], before, "Before alignment"),
        (axes[1], after, "After alignment"),
    ]:
        combined, labels = _subsample_embeddings(data_dict, n_words=300)
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(combined)

        for dc in sorted(set(labels), key=lambda d: d.value):
            mask = np.array([l == dc for l in labels])
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=DIALECT_COLORS.get(dc, "#333333"),
                marker=DIALECT_MARKERS.get(dc, "o"),
                label=dialect_label(dc),
                alpha=0.6,
                s=15,
                edgecolors="none",
            )

        ax.set_title(title)
        ax.legend(fontsize=7, markerscale=2)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Embedding alignment quality", fontsize=14)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 4. PCA scree plot
# ---------------------------------------------------------------------------

def plot_pca_variance(
    embeddings: EmbeddingMatrix,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Scree plot of PCA variance explained for a single embedding matrix."""
    n_components = min(50, embeddings.data.shape[1], embeddings.data.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(embeddings.data)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    indices = np.arange(1, len(explained) + 1)

    ax1.bar(indices, explained, color="#0072B2", alpha=0.7, label="Individual")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Variance explained (fraction)")

    ax2 = ax1.twinx()
    ax2.plot(indices, cumulative, color="#D55E00", linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative variance explained")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.95, color="grey", linestyle="--", linewidth=0.8)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    ax1.set_title(f"PCA scree plot ({dialect_label(embeddings.dialect_code)})")
    fig.tight_layout()
    return _save_or_return(fig, save_path)
