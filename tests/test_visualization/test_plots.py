"""Tests for the visualization module.

Every test uses small synthetic data and the Agg backend so no display is required.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import pytest  # noqa: E402

from eigendialectos.constants import DialectCode  # noqa: E402
from eigendialectos.types import DialectalSpectrum, EmbeddingMatrix  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIALECTS_SUBSET = [DialectCode.ES_PEN, DialectCode.ES_MEX, DialectCode.ES_RIO]

DIM = 30
VOCAB_SIZE = 60
N_EIGENVALUES = 40


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def spectra(rng: np.random.Generator) -> dict[DialectCode, DialectalSpectrum]:
    """Synthetic spectral data for three dialects."""
    result: dict[DialectCode, DialectalSpectrum] = {}
    for dc in DIALECTS_SUBSET:
        raw = np.sort(rng.exponential(scale=2.0, size=N_EIGENVALUES))[::-1]
        total = raw.sum()
        entropy = -float(np.sum((raw / total) * np.log(raw / total + 1e-12)))
        result[dc] = DialectalSpectrum(
            eigenvalues_sorted=raw,
            entropy=entropy,
            dialect_code=dc,
        )
    return result


@pytest.fixture()
def entropies(spectra: dict[DialectCode, DialectalSpectrum]) -> dict[DialectCode, float]:
    return {dc: sp.entropy for dc, sp in spectra.items()}


@pytest.fixture()
def embeddings(rng: np.random.Generator) -> dict[DialectCode, EmbeddingMatrix]:
    result: dict[DialectCode, EmbeddingMatrix] = {}
    for dc in DIALECTS_SUBSET:
        data = rng.normal(size=(VOCAB_SIZE, DIM)).astype(np.float64)
        vocab = [f"word_{dc.value}_{i}" for i in range(VOCAB_SIZE)]
        result[dc] = EmbeddingMatrix(data=data, vocab=vocab, dialect_code=dc)
    return result


@pytest.fixture()
def single_embedding(rng: np.random.Generator) -> EmbeddingMatrix:
    data = rng.normal(size=(VOCAB_SIZE, DIM)).astype(np.float64)
    vocab = [f"w{i}" for i in range(VOCAB_SIZE)]
    return EmbeddingMatrix(data=data, vocab=vocab, dialect_code=DialectCode.ES_PEN)


@pytest.fixture()
def distance_matrix(rng: np.random.Generator) -> np.ndarray:
    n = len(DIALECTS_SUBSET)
    upper = rng.uniform(0.1, 1.0, size=(n, n))
    sym = (upper + upper.T) / 2
    np.fill_diagonal(sym, 0)
    return sym


@pytest.fixture()
def alpha_values() -> list[float]:
    return [round(x * 0.1, 2) for x in range(16)]  # 0.0 .. 1.5


@pytest.fixture()
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Spectral plot tests
# ---------------------------------------------------------------------------

class TestSpectralPlots:
    def test_eigenvalue_bars_returns_figure(self, spectra: dict) -> None:
        from eigendialectos.visualization.spectral_plots import plot_eigenvalue_bars

        fig = plot_eigenvalue_bars(spectra, top_k=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_eigenvalue_bars_saves(self, spectra: dict, tmp_dir: Path) -> None:
        from eigendialectos.visualization.spectral_plots import plot_eigenvalue_bars

        path = tmp_dir / "bars.png"
        fig = plot_eigenvalue_bars(spectra, top_k=10, save_path=path)
        assert path.exists()
        assert path.stat().st_size > 0
        plt.close(fig)

    def test_eigenvalue_decay_returns_figure(self, spectra: dict) -> None:
        from eigendialectos.visualization.spectral_plots import plot_eigenvalue_decay

        fig = plot_eigenvalue_decay(spectra)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_eigenvalue_decay_saves(self, spectra: dict, tmp_dir: Path) -> None:
        from eigendialectos.visualization.spectral_plots import plot_eigenvalue_decay

        path = tmp_dir / "decay.png"
        fig = plot_eigenvalue_decay(spectra, save_path=path)
        assert path.exists()
        plt.close(fig)

    def test_cumulative_energy_returns_figure(self, spectra: dict) -> None:
        from eigendialectos.visualization.spectral_plots import plot_cumulative_energy

        fig = plot_cumulative_energy(spectra)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cumulative_energy_saves(self, spectra: dict, tmp_dir: Path) -> None:
        from eigendialectos.visualization.spectral_plots import plot_cumulative_energy

        path = tmp_dir / "cumul.png"
        fig = plot_cumulative_energy(spectra, save_path=path)
        assert path.exists()
        plt.close(fig)

    def test_entropy_comparison_returns_figure(self, entropies: dict) -> None:
        from eigendialectos.visualization.spectral_plots import plot_entropy_comparison

        fig = plot_entropy_comparison(entropies)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_entropy_comparison_saves(self, entropies: dict, tmp_dir: Path) -> None:
        from eigendialectos.visualization.spectral_plots import plot_entropy_comparison

        path = tmp_dir / "entropy.png"
        fig = plot_entropy_comparison(entropies, save_path=path)
        assert path.exists()
        plt.close(fig)

    def test_eigenspectrum_heatmap_returns_figure(self, spectra: dict) -> None:
        from eigendialectos.visualization.spectral_plots import plot_eigenspectrum_heatmap

        fig = plot_eigenspectrum_heatmap(spectra, top_k=15)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_eigenspectrum_heatmap_saves(self, spectra: dict, tmp_dir: Path) -> None:
        from eigendialectos.visualization.spectral_plots import plot_eigenspectrum_heatmap

        path = tmp_dir / "heatmap.png"
        fig = plot_eigenspectrum_heatmap(spectra, top_k=15, save_path=path)
        assert path.exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Embedding plot tests
# ---------------------------------------------------------------------------

class TestEmbeddingPlots:
    def test_tsne_returns_figure(self, embeddings: dict) -> None:
        from eigendialectos.visualization.embedding_plots import plot_embeddings_tsne

        fig = plot_embeddings_tsne(embeddings, n_words=20, perplexity=5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_tsne_saves(self, embeddings: dict, tmp_dir: Path) -> None:
        from eigendialectos.visualization.embedding_plots import plot_embeddings_tsne

        path = tmp_dir / "tsne.png"
        fig = plot_embeddings_tsne(embeddings, n_words=20, perplexity=5, save_path=path)
        assert path.exists()
        plt.close(fig)

    def test_alignment_quality_returns_figure(self, embeddings: dict) -> None:
        from eigendialectos.visualization.embedding_plots import plot_alignment_quality

        fig = plot_alignment_quality(before=embeddings, after=embeddings)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_alignment_quality_saves(self, embeddings: dict, tmp_dir: Path) -> None:
        from eigendialectos.visualization.embedding_plots import plot_alignment_quality

        path = tmp_dir / "alignment.png"
        fig = plot_alignment_quality(before=embeddings, after=embeddings, save_path=path)
        assert path.exists()
        plt.close(fig)

    def test_pca_variance_returns_figure(self, single_embedding: EmbeddingMatrix) -> None:
        from eigendialectos.visualization.embedding_plots import plot_pca_variance

        fig = plot_pca_variance(single_embedding)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pca_variance_saves(self, single_embedding: EmbeddingMatrix, tmp_dir: Path) -> None:
        from eigendialectos.visualization.embedding_plots import plot_pca_variance

        path = tmp_dir / "pca.png"
        fig = plot_pca_variance(single_embedding, save_path=path)
        assert path.exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Dialect map tests
# ---------------------------------------------------------------------------

class TestDialectMaps:
    def test_distance_matrix_returns_figure(self, distance_matrix: np.ndarray) -> None:
        from eigendialectos.visualization.dialect_maps import plot_dialect_distance_matrix

        fig = plot_dialect_distance_matrix(distance_matrix, DIALECTS_SUBSET)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_distance_matrix_saves(self, distance_matrix: np.ndarray, tmp_dir: Path) -> None:
        from eigendialectos.visualization.dialect_maps import plot_dialect_distance_matrix

        path = tmp_dir / "dist.png"
        fig = plot_dialect_distance_matrix(distance_matrix, DIALECTS_SUBSET, save_path=path)
        assert path.exists()
        plt.close(fig)

    def test_dendrogram_returns_figure(self, distance_matrix: np.ndarray) -> None:
        from eigendialectos.visualization.dialect_maps import plot_dialect_dendrogram

        fig = plot_dialect_dendrogram(distance_matrix, DIALECTS_SUBSET)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dendrogram_saves(self, distance_matrix: np.ndarray, tmp_dir: Path) -> None:
        from eigendialectos.visualization.dialect_maps import plot_dialect_dendrogram

        path = tmp_dir / "dendro.png"
        fig = plot_dialect_dendrogram(distance_matrix, DIALECTS_SUBSET, save_path=path)
        assert path.exists()
        plt.close(fig)

    def test_mds_returns_figure(self, distance_matrix: np.ndarray) -> None:
        from eigendialectos.visualization.dialect_maps import plot_dialect_mds

        fig = plot_dialect_mds(distance_matrix, DIALECTS_SUBSET)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_mds_saves(self, distance_matrix: np.ndarray, tmp_dir: Path) -> None:
        from eigendialectos.visualization.dialect_maps import plot_dialect_mds

        path = tmp_dir / "mds.png"
        fig = plot_dialect_mds(distance_matrix, DIALECTS_SUBSET, save_path=path)
        assert path.exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Gradient plot tests
# ---------------------------------------------------------------------------

class TestGradientPlots:
    def test_alpha_gradient_returns_figure(self, alpha_values: list[float]) -> None:
        from eigendialectos.visualization.gradient_plots import plot_alpha_gradient

        metrics = {
            "confidence": [float(np.sin(a * 2)) for a in alpha_values],
            "perplexity": [float(np.exp(-a)) for a in alpha_values],
        }
        fig = plot_alpha_gradient(alpha_values, metrics, DialectCode.ES_MEX)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_alpha_gradient_saves(self, alpha_values: list[float], tmp_dir: Path) -> None:
        from eigendialectos.visualization.gradient_plots import plot_alpha_gradient

        metrics = {"confidence": [float(np.sin(a * 2)) for a in alpha_values]}
        path = tmp_dir / "alpha.png"
        fig = plot_alpha_gradient(alpha_values, metrics, DialectCode.ES_MEX, save_path=path)
        assert path.exists()
        plt.close(fig)

    def test_feature_heatmap_returns_figure(self, alpha_values: list[float]) -> None:
        from eigendialectos.visualization.gradient_plots import plot_feature_activation_heatmap

        features = {
            "seseo": [float(a) for a in alpha_values],
            "voseo": [float(1 - a) for a in alpha_values],
        }
        fig = plot_feature_activation_heatmap(alpha_values, features)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_feature_heatmap_saves(self, alpha_values: list[float], tmp_dir: Path) -> None:
        from eigendialectos.visualization.gradient_plots import plot_feature_activation_heatmap

        features = {"seseo": [float(a) for a in alpha_values]}
        path = tmp_dir / "feat_heatmap.png"
        fig = plot_feature_activation_heatmap(alpha_values, features, save_path=path)
        assert path.exists()
        plt.close(fig)

    def test_threshold_annotations_returns_figure(self, alpha_values: list[float]) -> None:
        from eigendialectos.visualization.gradient_plots import plot_threshold_annotations

        scores = [float(np.tanh(a)) for a in alpha_values]
        fig = plot_threshold_annotations(alpha_values, scores, 0.3, 1.1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_threshold_annotations_saves(self, alpha_values: list[float], tmp_dir: Path) -> None:
        from eigendialectos.visualization.gradient_plots import plot_threshold_annotations

        scores = [float(np.tanh(a)) for a in alpha_values]
        path = tmp_dir / "thresholds.png"
        fig = plot_threshold_annotations(alpha_values, scores, 0.3, 1.1, save_path=path)
        assert path.exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tensor plot tests
# ---------------------------------------------------------------------------

class TestTensorPlots:
    def test_factor_loadings_returns_figure(self, rng: np.random.Generator) -> None:
        from eigendialectos.visualization.tensor_plots import plot_factor_loadings_heatmap

        factors = rng.normal(size=(3, 5))
        fig = plot_factor_loadings_heatmap(factors, DIALECTS_SUBSET)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_factor_loadings_saves(self, rng: np.random.Generator, tmp_dir: Path) -> None:
        from eigendialectos.visualization.tensor_plots import plot_factor_loadings_heatmap

        factors = rng.normal(size=(3, 5))
        path = tmp_dir / "factors.png"
        fig = plot_factor_loadings_heatmap(factors, DIALECTS_SUBSET, save_path=path)
        assert path.exists()
        plt.close(fig)

    def test_cp_components_returns_figure(self, rng: np.random.Generator) -> None:
        from eigendialectos.visualization.tensor_plots import plot_cp_components

        weights = rng.uniform(0.5, 3.0, size=8)
        factor_list = [rng.normal(size=(10, 8)) for _ in range(3)]
        fig = plot_cp_components(weights, factor_list, top_k=4)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cp_components_saves(self, rng: np.random.Generator, tmp_dir: Path) -> None:
        from eigendialectos.visualization.tensor_plots import plot_cp_components

        weights = rng.uniform(0.5, 3.0, size=8)
        factor_list = [rng.normal(size=(10, 8)) for _ in range(3)]
        path = tmp_dir / "cp.png"
        fig = plot_cp_components(weights, factor_list, top_k=4, save_path=path)
        assert path.exists()
        plt.close(fig)

    def test_reconstruction_scree_returns_figure(self) -> None:
        from eigendialectos.visualization.tensor_plots import plot_reconstruction_scree

        errors = [1.0, 0.5, 0.3, 0.2, 0.18, 0.17, 0.165]
        ranks = [1, 2, 3, 4, 5, 6, 7]
        fig = plot_reconstruction_scree(errors, ranks)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_reconstruction_scree_saves(self, tmp_dir: Path) -> None:
        from eigendialectos.visualization.tensor_plots import plot_reconstruction_scree

        errors = [1.0, 0.5, 0.3, 0.2, 0.18, 0.17, 0.165]
        ranks = [1, 2, 3, 4, 5, 6, 7]
        path = tmp_dir / "scree.png"
        fig = plot_reconstruction_scree(errors, ranks, save_path=path)
        assert path.exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Interactive plot tests
# ---------------------------------------------------------------------------

class TestInteractivePlots:
    def test_spectral_dashboard_returns_go_figure(
        self,
        spectra: dict,
        distance_matrix: np.ndarray,
        entropies: dict,
    ) -> None:
        from eigendialectos.visualization.interactive import create_spectral_dashboard

        fig = create_spectral_dashboard(spectra, distance_matrix, entropies)
        assert isinstance(fig, go.Figure)

    def test_embedding_explorer_returns_go_figure(self, embeddings: dict) -> None:
        from eigendialectos.visualization.interactive import create_embedding_explorer

        vocab = {dc: emb.vocab for dc, emb in embeddings.items()}
        fig = create_embedding_explorer(embeddings, vocab, DIALECTS_SUBSET)
        assert isinstance(fig, go.Figure)

    def test_gradient_slider_returns_go_figure(self, alpha_values: list[float]) -> None:
        from eigendialectos.visualization.interactive import create_gradient_slider

        alpha_data = {
            "alpha_values": alpha_values,
            "features": {
                "seseo": [float(a) for a in alpha_values],
                "voseo": [float(1 - a) for a in alpha_values],
            },
            "scores": [float(np.tanh(a)) for a in alpha_values],
        }
        fig = create_gradient_slider(alpha_data)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Color palette consistency test
# ---------------------------------------------------------------------------

class TestColorPalette:
    def test_all_dialects_have_colors(self) -> None:
        from eigendialectos.visualization._colors import DIALECT_COLORS

        for dc in DialectCode:
            assert dc in DIALECT_COLORS, f"Missing colour for {dc}"

    def test_all_dialects_have_markers(self) -> None:
        from eigendialectos.visualization._colors import DIALECT_MARKERS

        for dc in DialectCode:
            assert dc in DIALECT_MARKERS, f"Missing marker for {dc}"

    def test_dialect_label_returns_string(self) -> None:
        from eigendialectos.visualization._colors import dialect_label

        for dc in DialectCode:
            label = dialect_label(dc)
            assert isinstance(label, str)
            assert len(label) > 0
