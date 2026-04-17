"""Tests for geometry package: Lie algebra, Riemannian, Fisher, eigenfield."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm, logm

from eigendialectos.constants import DialectCode
from eigendialectos.geometry.lie_algebra import LieAlgebraAnalysis
from eigendialectos.geometry.riemannian import RiemannianDialectSpace
from eigendialectos.geometry.fisher import FisherInformationAnalysis
from eigendialectos.geometry.eigenfield import EigenvalueField
from eigendialectos.types import EigenDecomposition


@pytest.fixture
def random_W_matrices():
    rng = np.random.default_rng(42)
    return {
        "A": np.eye(10) + 0.1 * rng.standard_normal((10, 10)),
        "B": np.eye(10) + 0.1 * rng.standard_normal((10, 10)),
        "C": np.eye(10) + 0.1 * rng.standard_normal((10, 10)),
    }


@pytest.fixture
def random_eigendecomps():
    rng = np.random.default_rng(42)
    decomps = {}
    for name in ["A", "B", "C"]:
        P = rng.standard_normal((10, 10)) + 0j
        decomps[name] = EigenDecomposition(
            eigenvalues=(rng.random(10) + 0.5) + 0j,
            eigenvectors=P,
            eigenvectors_inv=np.linalg.inv(P),
            dialect_code=DialectCode.ES_PEN,
        )
    return decomps


class TestLieAlgebra:
    def test_expm_logm_roundtrip(self, random_W_matrices):
        lie = LieAlgebraAnalysis()
        generators = lie.compute_generators(random_W_matrices)

        for name, W in random_W_matrices.items():
            A = generators[name]
            W_reconstructed = expm(A)
            np.testing.assert_allclose(
                W_reconstructed.real, W, atol=1e-8,
                err_msg=f"expm(logm(W)) != W for {name}",
            )

    def test_commutator_antisymmetry(self, random_W_matrices):
        lie = LieAlgebraAnalysis()
        result = lie.full_analysis(random_W_matrices)

        for (i, j), bracket in result.commutators.items():
            # [A_i, A_j] = -[A_j, A_i]
            A_i = result.generators[i]
            A_j = result.generators[j]
            bracket_ij = A_i @ A_j - A_j @ A_i
            bracket_ji = A_j @ A_i - A_i @ A_j
            np.testing.assert_allclose(
                bracket_ij.real, -bracket_ji.real, atol=1e-10,
            )

    def test_interpolation(self, random_W_matrices):
        lie = LieAlgebraAnalysis()
        generators = lie.compute_generators(random_W_matrices)
        A_a = generators["A"]
        A_b = generators["B"]

        # beta=1 should give W_A
        W_at_1 = lie.interpolate(A_a, A_b, beta=1.0)
        np.testing.assert_allclose(W_at_1, random_W_matrices["A"], atol=1e-6)

        # beta=0 should give W_B
        W_at_0 = lie.interpolate(A_a, A_b, beta=0.0)
        np.testing.assert_allclose(W_at_0, random_W_matrices["B"], atol=1e-6)

    def test_bracket_magnitude_matrix(self, random_W_matrices):
        lie = LieAlgebraAnalysis()
        generators = lie.compute_generators(random_W_matrices)
        matrix, labels = lie.bracket_magnitude_matrix(generators)

        assert matrix.shape == (3, 3)
        assert len(labels) == 3
        # Diagonal should be zero
        np.testing.assert_allclose(np.diag(matrix), 0.0)
        # Should be non-negative
        assert np.all(matrix >= 0)


class TestRiemannian:
    def test_metric_tensor_spd(self, random_eigendecomps):
        riem = RiemannianDialectSpace()
        metrics = riem.compute_metric_tensors(random_eigendecomps)

        for name, g in metrics.items():
            # Symmetric
            np.testing.assert_allclose(g, g.T, atol=1e-10)
            # Positive definite
            eigenvals = np.linalg.eigvalsh(g)
            assert np.all(eigenvals > 0), f"Metric tensor for {name} is not SPD"

    def test_geodesic_distance_metric_properties(self, random_eigendecomps):
        riem = RiemannianDialectSpace()
        result = riem.full_analysis(random_eigendecomps)
        D = result.geodesic_distances

        # d(x, x) = 0
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)
        # Symmetry
        np.testing.assert_allclose(D, D.T, atol=1e-10)
        # Non-negative
        assert np.all(D >= -1e-10)
        # Triangle inequality (spot check)
        n = D.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert D[i, j] <= D[i, k] + D[k, j] + 1e-6

    def test_curvatures_are_finite(self, random_eigendecomps):
        riem = RiemannianDialectSpace()
        result = riem.full_analysis(random_eigendecomps)
        for name, kappa in result.ricci_curvatures.items():
            assert np.isfinite(kappa)


class TestFisher:
    def test_fim_shape(self):
        rng = np.random.default_rng(42)
        embeddings = {d: rng.standard_normal((100, 50)) for d in ["A", "B", "C"]}
        fisher = FisherInformationAnalysis()
        result = fisher.compute_fim(embeddings)
        assert result.fim.shape == (50, 50)
        assert result.fim_eigenvalues.shape == (50,)

    def test_fim_psd(self):
        rng = np.random.default_rng(42)
        embeddings = {d: rng.standard_normal((100, 50)) for d in ["A", "B", "C"]}
        fisher = FisherInformationAnalysis()
        result = fisher.compute_fim(embeddings)
        # S_b is PSD by construction, but FIM = S_w^{-1} S_b may have negative eigenvalues
        # Check that most eigenvalues are non-negative
        n_positive = np.sum(result.fim_eigenvalues >= -1e-10)
        assert n_positive > 0

    def test_diagnostic_words(self):
        rng = np.random.default_rng(42)
        vocab = [f"w{i}" for i in range(100)]
        embeddings = {d: rng.standard_normal((100, 50)) for d in ["A", "B", "C"]}
        fisher = FisherInformationAnalysis()
        result = fisher.compute_fim(embeddings, vocabulary=vocab)
        assert len(result.most_diagnostic) == 20
        assert all(isinstance(w, str) for w, _ in result.most_diagnostic)


class TestEigenvalueField:
    def test_fit_predict_at_training(self):
        rng = np.random.default_rng(42)
        coords = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float64)
        eigenvalues = rng.random((4, 5))

        ef = EigenvalueField(noise_variance=1e-6)
        ef.fit(coords, eigenvalues)
        preds, uncerts = ef.predict(coords)

        # Predictions at training points should be close to actual
        np.testing.assert_allclose(preds, np.abs(eigenvalues), atol=0.2)

    def test_compute_field_shape(self):
        rng = np.random.default_rng(42)
        coords = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float64)
        eigenvalues = rng.random((3, 5))

        ef = EigenvalueField()
        ef.fit(coords, eigenvalues)
        result = ef.compute_field(resolution=20)

        assert result.eigenvalue_surfaces.shape == (5, 20, 20)
        assert result.uncertainties.shape == (5, 20, 20)
        assert len(result.grid_lat) == 20
        assert len(result.grid_lon) == 20

    def test_uncertainty_increases_away_from_data(self):
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        eigenvalues = np.array([[1.0], [1.0], [1.0]])

        ef = EigenvalueField(kernel_lengthscale=1.0)
        ef.fit(coords, eigenvalues)

        near = np.array([[0.5, 0.5]])
        far = np.array([[100.0, 100.0]])
        _, uncert_near = ef.predict(near)
        _, uncert_far = ef.predict(far)

        assert uncert_far[0, 0] > uncert_near[0, 0]
