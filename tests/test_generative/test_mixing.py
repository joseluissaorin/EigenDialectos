"""Tests for dialect mixing: linear, log-Euclidean, and eigenvalue-level."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.generative.mixing import (
    log_euclidean_mix,
    mix_dialects,
    mix_eigendecompositions,
)
from eigendialectos.types import EigenDecomposition, TransformationMatrix


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_transform(data: np.ndarray, dialect: DialectCode) -> TransformationMatrix:
    return TransformationMatrix(
        data=data,
        source_dialect=DialectCode.ES_PEN,
        target_dialect=dialect,
        regularization=0.0,
    )


def _make_eigen(
    eigenvalues: np.ndarray,
    P: np.ndarray,
    dialect: DialectCode,
) -> EigenDecomposition:
    return EigenDecomposition(
        eigenvalues=eigenvalues.astype(np.complex128),
        eigenvectors=P.astype(np.complex128),
        eigenvectors_inv=np.linalg.inv(P).astype(np.complex128),
        dialect_code=dialect,
    )


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def W_a():
    """A simple 3x3 transform for dialect A."""
    data = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
    return _make_transform(data, DialectCode.ES_RIO)


@pytest.fixture
def W_b():
    """A simple 3x3 transform for dialect B."""
    data = np.array([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
    return _make_transform(data, DialectCode.ES_MEX)


@pytest.fixture
def eigen_a():
    """Eigendecomposition for dialect A (diagonal)."""
    return _make_eigen(
        np.array([3.0, 2.0, 1.0]),
        np.eye(3),
        DialectCode.ES_RIO,
    )


@pytest.fixture
def eigen_b():
    """Eigendecomposition for dialect B (diagonal)."""
    return _make_eigen(
        np.array([1.0, 4.0, 2.0]),
        np.eye(3),
        DialectCode.ES_MEX,
    )


# ------------------------------------------------------------------
# Tests: weight validation
# ------------------------------------------------------------------

class TestWeightValidation:
    """Weights must sum to 1."""

    def test_weights_sum_to_one_ok(self, W_a, W_b):
        """Valid weights should not raise."""
        result = mix_dialects([(W_a, 0.5), (W_b, 0.5)])
        assert result.data.shape == (3, 3)

    def test_weights_not_summing_to_one(self, W_a, W_b):
        with pytest.raises(ValueError, match="sum to 1"):
            mix_dialects([(W_a, 0.3), (W_b, 0.3)])

    def test_weights_exceed_one(self, W_a, W_b):
        with pytest.raises(ValueError, match="sum to 1"):
            mix_dialects([(W_a, 0.8), (W_b, 0.8)])

    def test_empty_list(self):
        with pytest.raises(ValueError, match="empty"):
            mix_dialects([])

    def test_single_weight_one(self, W_a):
        """A single transform with weight 1 should return itself."""
        result = mix_dialects([(W_a, 1.0)])
        np.testing.assert_allclose(result.data, W_a.data, atol=1e-12)


# ------------------------------------------------------------------
# Tests: pure dialect (beta=1 for one, 0 for rest)
# ------------------------------------------------------------------

class TestPureDialect:
    """beta=1 for one dialect should produce that dialect's transform."""

    def test_beta_one_zero_gives_first(self, W_a, W_b):
        result = mix_dialects([(W_a, 1.0), (W_b, 0.0)])
        np.testing.assert_allclose(result.data, W_a.data, atol=1e-12)

    def test_beta_zero_one_gives_second(self, W_a, W_b):
        result = mix_dialects([(W_a, 0.0), (W_b, 1.0)])
        np.testing.assert_allclose(result.data, W_b.data, atol=1e-12)

    def test_equal_weights_is_average(self, W_a, W_b):
        result = mix_dialects([(W_a, 0.5), (W_b, 0.5)])
        expected = 0.5 * W_a.data + 0.5 * W_b.data
        np.testing.assert_allclose(result.data, expected, atol=1e-12)


# ------------------------------------------------------------------
# Tests: log-Euclidean mixing
# ------------------------------------------------------------------

class TestLogEuclideanMix:
    """Log-Euclidean interpolation tests."""

    def test_log_euclidean_weight_validation(self, W_a, W_b):
        with pytest.raises(ValueError, match="sum to 1"):
            log_euclidean_mix([(W_a, 0.3), (W_b, 0.3)])

    def test_log_euclidean_single_weight_one(self, W_a):
        """beta=1 for a single transform should recover it."""
        result = log_euclidean_mix([(W_a, 1.0)])
        np.testing.assert_allclose(result.data, W_a.data, atol=1e-10)

    def test_log_euclidean_pure_gives_original(self, W_a, W_b):
        """beta=1,0 should give the first transform."""
        result = log_euclidean_mix([(W_a, 1.0), (W_b, 0.0)])
        np.testing.assert_allclose(result.data, W_a.data, atol=1e-10)

    def test_log_euclidean_preserves_shape(self, W_a, W_b):
        result = log_euclidean_mix([(W_a, 0.5), (W_b, 0.5)])
        assert result.data.shape == (3, 3)

    def test_log_euclidean_is_deterministic(self, W_a, W_b):
        r1 = log_euclidean_mix([(W_a, 0.5), (W_b, 0.5)])
        r2 = log_euclidean_mix([(W_a, 0.5), (W_b, 0.5)])
        np.testing.assert_array_equal(r1.data, r2.data)

    def test_log_euclidean_differs_from_linear(self, W_a, W_b):
        """Log-Euclidean and linear mix should generally differ."""
        linear = mix_dialects([(W_a, 0.5), (W_b, 0.5)])
        log_euc = log_euclidean_mix([(W_a, 0.5), (W_b, 0.5)])
        # They should not be identical (unless degenerate)
        diff = np.linalg.norm(linear.data - log_euc.data)
        assert diff > 1e-10


# ------------------------------------------------------------------
# Tests: eigendecomposition mixing
# ------------------------------------------------------------------

class TestMixEigendecompositions:
    """Mixing at the eigenvalue level."""

    def test_weight_validation(self, eigen_a, eigen_b):
        with pytest.raises(ValueError, match="sum to 1"):
            mix_eigendecompositions([(eigen_a, 0.3), (eigen_b, 0.3)])

    def test_pure_weight_recovers_eigenvalues(self, eigen_a, eigen_b):
        """beta=1 for one should recover its eigenvalues."""
        result = mix_eigendecompositions([(eigen_a, 1.0), (eigen_b, 0.0)])
        np.testing.assert_allclose(
            np.abs(result.eigenvalues), np.abs(eigen_a.eigenvalues), atol=1e-10
        )

    def test_mixed_eigenvalues_are_intermediate(self, eigen_a, eigen_b):
        """Mixed eigenvalue magnitudes should be between the two originals."""
        result = mix_eigendecompositions([(eigen_a, 0.5), (eigen_b, 0.5)])
        mag_a = np.abs(eigen_a.eigenvalues)
        mag_b = np.abs(eigen_b.eigenvalues)
        mag_mix = np.abs(result.eigenvalues)

        # Geometric mean: for positive real eigenvalues,
        # mixed magnitude should be sqrt(a * b)
        expected = np.sqrt(mag_a * mag_b)
        np.testing.assert_allclose(mag_mix, expected, atol=1e-10)

    def test_inconsistent_sizes(self, eigen_a):
        small = _make_eigen(np.array([1.0, 2.0]), np.eye(2), DialectCode.ES_CHI)
        with pytest.raises(ValueError, match="same number"):
            mix_eigendecompositions([(eigen_a, 0.5), (small, 0.5)])

    def test_empty_list(self):
        with pytest.raises(ValueError, match="empty"):
            mix_eigendecompositions([])
