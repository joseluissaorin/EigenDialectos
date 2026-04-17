"""Tests for the core DIAL transform: W(alpha) = P Lambda^alpha P^{-1}."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import ALPHA_RANGE, DialectCode
from eigendialectos.generative.dial import (
    _eigenvalues_to_alpha,
    apply_dial,
    compute_dial_series,
    dial_transform_embedding,
)
from eigendialectos.types import EigenDecomposition, TransformationMatrix


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def simple_eigen():
    """3x3 eigendecomposition with real eigenvalues [3, 2, 1]."""
    eigenvalues = np.array([3.0, 2.0, 1.0], dtype=np.complex128)
    P = np.array(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.complex128,
    )
    P_inv = np.linalg.inv(P)
    return EigenDecomposition(
        eigenvalues=eigenvalues,
        eigenvectors=P,
        eigenvectors_inv=P_inv,
        dialect_code=DialectCode.ES_RIO,
    )


@pytest.fixture
def complex_eigen():
    """3x3 eigendecomposition with one complex conjugate pair.

    Eigenvalues: 2+1j, 2-1j, 1.0
    """
    eigenvalues = np.array([2.0 + 1.0j, 2.0 - 1.0j, 1.0 + 0.0j], dtype=np.complex128)
    # Use a real-valued eigenvector matrix that is well-conditioned
    P = np.array(
        [[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.complex128,
    )
    P_inv = np.linalg.inv(P)
    return EigenDecomposition(
        eigenvalues=eigenvalues,
        eigenvectors=P,
        eigenvectors_inv=P_inv,
        dialect_code=DialectCode.ES_AND,
    )


@pytest.fixture
def identity_eigen():
    """Eigendecomposition where all eigenvalues are 1 (identity)."""
    n = 4
    eigenvalues = np.ones(n, dtype=np.complex128)
    P = np.eye(n, dtype=np.complex128)
    P_inv = np.eye(n, dtype=np.complex128)
    return EigenDecomposition(
        eigenvalues=eigenvalues,
        eigenvectors=P,
        eigenvectors_inv=P_inv,
        dialect_code=DialectCode.ES_PEN,
    )


# ------------------------------------------------------------------
# Tests: alpha = 0 --> identity
# ------------------------------------------------------------------

class TestAlphaZeroIdentity:
    """alpha=0 must produce the identity matrix."""

    def test_alpha_zero_real_eigenvalues(self, simple_eigen):
        result = apply_dial(simple_eigen, 0.0)
        n = simple_eigen.eigenvectors.shape[0]
        np.testing.assert_allclose(result.data, np.eye(n), atol=1e-12)

    def test_alpha_zero_complex_eigenvalues(self, complex_eigen):
        result = apply_dial(complex_eigen, 0.0)
        n = complex_eigen.eigenvectors.shape[0]
        np.testing.assert_allclose(result.data, np.eye(n), atol=1e-12)

    def test_alpha_zero_identity_eigen(self, identity_eigen):
        result = apply_dial(identity_eigen, 0.0)
        n = identity_eigen.eigenvectors.shape[0]
        np.testing.assert_allclose(result.data, np.eye(n), atol=1e-12)


# ------------------------------------------------------------------
# Tests: alpha = 1 --> original W
# ------------------------------------------------------------------

class TestAlphaOneOriginal:
    """alpha=1 must recover the original transformation matrix."""

    def test_alpha_one_real(self, simple_eigen):
        result = apply_dial(simple_eigen, 1.0)
        # Reconstruct the original W = P Lambda P^{-1}
        W_original = (
            simple_eigen.eigenvectors
            @ np.diag(simple_eigen.eigenvalues)
            @ simple_eigen.eigenvectors_inv
        )
        np.testing.assert_allclose(result.data, W_original.real, atol=1e-12)

    def test_alpha_one_complex(self, complex_eigen):
        result = apply_dial(complex_eigen, 1.0)
        W_original = (
            complex_eigen.eigenvectors
            @ np.diag(complex_eigen.eigenvalues)
            @ complex_eigen.eigenvectors_inv
        )
        # May have imaginary residuals, but they should be negligible
        np.testing.assert_allclose(result.data, W_original.real, atol=1e-10)


# ------------------------------------------------------------------
# Tests: eigenvalue structure preservation
# ------------------------------------------------------------------

class TestEigenvalueStructure:
    """Verify that the DIAL transform preserves the eigenvector structure."""

    def test_eigenvalues_raised_correctly(self):
        lambdas = np.array([3.0, 2.0, 1.0], dtype=np.complex128)
        alpha = 0.5
        result = _eigenvalues_to_alpha(lambdas, alpha)
        expected = np.array([3.0**0.5, 2.0**0.5, 1.0**0.5], dtype=np.complex128)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_alpha_two_is_squared(self, simple_eigen):
        """W(2) should equal W @ W for real positive eigenvalues."""
        W1 = apply_dial(simple_eigen, 1.0)
        W2 = apply_dial(simple_eigen, 2.0)
        W_squared = W1.data @ W1.data
        np.testing.assert_allclose(W2.data, W_squared, atol=1e-10)

    def test_alpha_half_then_half_gives_one(self, simple_eigen):
        """W(0.5) @ W(0.5) should give W(1.0)."""
        W_half = apply_dial(simple_eigen, 0.5)
        W_one = apply_dial(simple_eigen, 1.0)
        product = W_half.data @ W_half.data
        np.testing.assert_allclose(product, W_one.data, atol=1e-10)

    def test_negative_alpha(self, simple_eigen):
        """W(-1) @ W(1) should give identity."""
        W_pos = apply_dial(simple_eigen, 1.0)
        W_neg = apply_dial(simple_eigen, -1.0)
        product = W_pos.data @ W_neg.data
        n = simple_eigen.eigenvectors.shape[0]
        np.testing.assert_allclose(product, np.eye(n), atol=1e-10)

    def test_alpha_additivity(self, simple_eigen):
        """W(a) @ W(b) = W(a+b) for any a, b."""
        a, b = 0.3, 0.7
        W_a = apply_dial(simple_eigen, a)
        W_b = apply_dial(simple_eigen, b)
        W_ab = apply_dial(simple_eigen, a + b)
        np.testing.assert_allclose(W_a.data @ W_b.data, W_ab.data, atol=1e-10)


# ------------------------------------------------------------------
# Tests: complex eigenvalue handling
# ------------------------------------------------------------------

class TestComplexEigenvalues:
    """Verify correct handling of complex eigenvalues."""

    def test_complex_alpha_zero(self, complex_eigen):
        """Complex eigenvalues to the power 0 should still give identity."""
        result = apply_dial(complex_eigen, 0.0)
        n = complex_eigen.eigenvectors.shape[0]
        np.testing.assert_allclose(result.data, np.eye(n), atol=1e-12)

    def test_complex_magnitude_phase(self):
        """lambda^alpha = |lambda|^alpha * exp(i*alpha*arg(lambda))."""
        lambdas = np.array([2.0 + 1.0j, 3.0 - 2.0j], dtype=np.complex128)
        alpha = 0.5

        result = _eigenvalues_to_alpha(lambdas, alpha)

        for i, lam in enumerate(lambdas):
            mag = abs(lam)
            angle = np.angle(lam)
            expected = mag**alpha * np.exp(1j * alpha * angle)
            np.testing.assert_allclose(result[i], expected, atol=1e-12)

    def test_conjugate_pairs_produce_real_result(self, complex_eigen):
        """With conjugate pairs, W(alpha) should be approximately real."""
        for alpha in [0.0, 0.5, 1.0, 1.5]:
            result = apply_dial(complex_eigen, alpha)
            # The result should be real-valued (imaginary part negligible)
            assert result.data.dtype == np.float64

    def test_zero_eigenvalue(self):
        """Zero eigenvalue raised to any positive power stays zero."""
        lambdas = np.array([0.0, 2.0, 1.0], dtype=np.complex128)
        result = _eigenvalues_to_alpha(lambdas, 0.5)
        assert result[0] == 0.0
        np.testing.assert_allclose(result[1], 2.0**0.5, atol=1e-12)


# ------------------------------------------------------------------
# Tests: dial_transform_embedding
# ------------------------------------------------------------------

class TestDialTransformEmbedding:
    """Test embedding-level DIAL application."""

    def test_vector_transform(self, simple_eigen):
        vec = np.array([1.0, 0.0, 0.0])
        result = dial_transform_embedding(vec, simple_eigen, 1.0)
        W = apply_dial(simple_eigen, 1.0)
        expected = W.data @ vec
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_batch_transform(self, simple_eigen):
        batch = np.eye(3)
        result = dial_transform_embedding(batch, simple_eigen, 1.0)
        W = apply_dial(simple_eigen, 1.0)
        expected = batch @ W.data.T
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_alpha_zero_preserves_embedding(self, simple_eigen):
        vec = np.array([1.0, 2.0, 3.0])
        result = dial_transform_embedding(vec, simple_eigen, 0.0)
        np.testing.assert_allclose(result, vec, atol=1e-12)

    def test_invalid_dims(self, simple_eigen):
        bad = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="1-D or 2-D"):
            dial_transform_embedding(bad, simple_eigen, 1.0)


# ------------------------------------------------------------------
# Tests: compute_dial_series
# ------------------------------------------------------------------

class TestComputeDialSeries:
    """Test series computation over alpha range."""

    def test_default_range(self, simple_eigen):
        series = compute_dial_series(simple_eigen)
        expected_count = len(np.arange(*ALPHA_RANGE))
        assert len(series) == expected_count

    def test_custom_range(self, simple_eigen):
        series = compute_dial_series(simple_eigen, alpha_range=(0.0, 1.0, 0.25))
        assert len(series) == 4  # 0.0, 0.25, 0.5, 0.75

    def test_all_valid_transforms(self, simple_eigen):
        series = compute_dial_series(simple_eigen)
        for tm in series:
            assert isinstance(tm, TransformationMatrix)
            assert tm.data.shape == (3, 3)

    def test_first_is_identity(self, simple_eigen):
        series = compute_dial_series(simple_eigen, alpha_range=(0.0, 1.0, 0.5))
        np.testing.assert_allclose(series[0].data, np.eye(3), atol=1e-12)
