"""Tests for spectral stack and multi-granularity decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.spectral.stack import SpectralStack
from eigendialectos.spectral.multigranularity import MultiGranularityDecomposition


class TestSpectralStack:
    def test_fit_from_matrices(self):
        rng = np.random.default_rng(42)
        W = np.eye(10) + 0.1 * rng.standard_normal((10, 10))
        stack = SpectralStack(levels=[2])
        stack.fit_from_matrices({2: W})
        assert 2 in stack.level_eigen
        assert stack._fitted

    def test_transform_identity_at_alpha_zero(self):
        rng = np.random.default_rng(42)
        W = np.eye(10) + 0.1 * rng.standard_normal((10, 10))
        stack = SpectralStack(levels=[2])
        stack.fit_from_matrices({2: W})

        v = rng.standard_normal(10)
        result = stack.transform(2, v, alpha=0.0)
        np.testing.assert_allclose(result, v, atol=1e-10)

    def test_transform_full_at_alpha_one(self):
        rng = np.random.default_rng(42)
        W = np.eye(10) + 0.1 * rng.standard_normal((10, 10))
        stack = SpectralStack(levels=[2])
        stack.fit_from_matrices({2: W})

        v = rng.standard_normal(10)
        result = stack.transform(2, v, alpha=1.0)
        expected = W @ v
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_transform_batch(self):
        rng = np.random.default_rng(42)
        W = np.eye(10) + 0.1 * rng.standard_normal((10, 10))
        stack = SpectralStack(levels=[2])
        stack.fit_from_matrices({2: W})

        V = rng.standard_normal((5, 10))
        result = stack.transform(2, V, alpha=1.0)
        assert result.shape == (5, 10)

    def test_fitted_levels(self):
        rng = np.random.default_rng(42)
        stack = SpectralStack(levels=[1, 2, 3])
        stack.fit_from_matrices({
            1: np.eye(5) + 0.1 * rng.standard_normal((5, 5)),
            3: np.eye(5) + 0.1 * rng.standard_normal((5, 5)),
        })
        assert stack.fitted_levels == [1, 3]

    def test_missing_level_raises(self):
        stack = SpectralStack(levels=[2])
        stack.fit_from_matrices({2: np.eye(5)})
        with pytest.raises(KeyError):
            stack.transform(3, np.zeros(5))


class TestMultiGranularityDecomposition:
    def test_decomposition_reconstruction(self):
        rng = np.random.default_rng(42)
        W_matrices = {
            d.value: np.eye(10) + 0.05 * rng.standard_normal((10, 10))
            for d in DialectCode
        }
        mg = MultiGranularityDecomposition()
        results = mg.decompose(W_matrices)

        # Reconstruction errors should be near zero
        for d, error in results["reconstruction_errors"].items():
            assert error < 1e-10, f"Reconstruction error for {d}: {error}"

    def test_macro_eigenvalues_exist(self):
        rng = np.random.default_rng(42)
        W_matrices = {
            d.value: np.eye(10) + 0.05 * rng.standard_normal((10, 10))
            for d in DialectCode
        }
        mg = MultiGranularityDecomposition()
        results = mg.decompose(W_matrices)

        assert "macro" in results
        assert results["macro"]["eigenvalues"].shape == (10,)

    def test_hierarchical_spectrum(self):
        rng = np.random.default_rng(42)
        W_matrices = {
            d.value: np.eye(10) + 0.05 * rng.standard_normal((10, 10))
            for d in DialectCode
        }
        mg = MultiGranularityDecomposition()
        mg.decompose(W_matrices)

        spectrum = mg.get_hierarchical_spectrum("ES_PEN")
        assert "macro" in spectrum
        assert "zonal" in spectrum
        assert "dialect" in spectrum

    def test_explained_variance_ratios(self):
        rng = np.random.default_rng(42)
        W_matrices = {
            d.value: np.eye(10) + 0.05 * rng.standard_normal((10, 10))
            for d in DialectCode
        }
        mg = MultiGranularityDecomposition()
        mg.decompose(W_matrices)

        ratios = mg.explained_variance_ratio()
        for d, r in ratios.items():
            assert abs(r["macro"] + r["zonal"] + r["dialect"] - 1.0) < 0.01
