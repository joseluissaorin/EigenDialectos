"""Tests for spectral.eigendecomposition module."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.spectral.eigendecomposition import (
    decompose,
    eigendecompose,
    svd_decompose,
)
from eigendialectos.types import TransformationMatrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def diagonal_transform():
    """Diagonal matrix whose eigenvalues are the diagonal entries."""
    diag_vals = np.array([5.0, 3.0, 1.0, 0.5])
    W = TransformationMatrix(
        data=np.diag(diag_vals),
        source_dialect=DialectCode.ES_PEN,
        target_dialect=DialectCode.ES_AND,
        regularization=0.0,
    )
    return W, diag_vals


@pytest.fixture
def random_square_transform(rng):
    """Random 10x10 transformation matrix."""
    data = rng.standard_normal((10, 10)).astype(np.float64)
    return TransformationMatrix(
        data=data,
        source_dialect=DialectCode.ES_PEN,
        target_dialect=DialectCode.ES_MEX,
        regularization=0.01,
    )


@pytest.fixture
def symmetric_transform(rng):
    """Symmetric (real eigenvalues) transformation matrix."""
    A = rng.standard_normal((8, 8)).astype(np.float64)
    S = (A + A.T) / 2.0
    return TransformationMatrix(
        data=S,
        source_dialect=DialectCode.ES_PEN,
        target_dialect=DialectCode.ES_CHI,
        regularization=0.0,
    )


# ---------------------------------------------------------------------------
# Tests: eigendecompose
# ---------------------------------------------------------------------------

class TestEigendecompose:
    """Tests for eigendecomposition."""

    def test_diagonal_eigenvalues(self, diagonal_transform):
        """Eigenvalues of a diagonal matrix should be its diagonal entries."""
        W, expected = diagonal_transform
        result = eigendecompose(W)
        actual = np.sort(np.real(result.eigenvalues))
        expected_sorted = np.sort(expected)
        np.testing.assert_allclose(actual, expected_sorted, atol=1e-10)

    def test_diagonal_rank(self, diagonal_transform):
        """All eigenvalues are non-zero, so rank = n."""
        W, _ = diagonal_transform
        result = eigendecompose(W)
        assert result.rank == 4

    def test_reconstruction(self, random_square_transform):
        r"""Test that P @ diag(eigenvalues) @ P^{-1} ≈ W."""
        W = random_square_transform
        result = eigendecompose(W)
        reconstructed = (
            result.eigenvectors
            @ np.diag(result.eigenvalues)
            @ result.eigenvectors_inv
        )
        np.testing.assert_allclose(
            np.real(reconstructed), W.data, atol=1e-8
        )

    def test_known_eigenstructure(self, known_transform):
        """Test eigendecomposition on the known_transform fixture."""
        result = eigendecompose(known_transform)
        eigenvalues_real = np.sort(np.real(result.eigenvalues))
        np.testing.assert_allclose(
            eigenvalues_real, [1.0, 2.0, 3.0], atol=1e-10
        )

    def test_symmetric_real_eigenvalues(self, symmetric_transform):
        """Symmetric matrices should have purely real eigenvalues."""
        result = eigendecompose(symmetric_transform)
        np.testing.assert_allclose(
            np.imag(result.eigenvalues), 0.0, atol=1e-10
        )

    def test_dialect_code_propagated(self, random_square_transform):
        """The target dialect code should be stored in the result."""
        result = eigendecompose(random_square_transform)
        assert result.dialect_code == DialectCode.ES_MEX

    def test_non_square_raises(self):
        """Non-square matrix should raise ValueError."""
        W = TransformationMatrix(
            data=np.ones((3, 5)),
            source_dialect=DialectCode.ES_PEN,
            target_dialect=DialectCode.ES_RIO,
            regularization=0.0,
        )
        with pytest.raises(ValueError, match="square"):
            eigendecompose(W)


# ---------------------------------------------------------------------------
# Tests: svd_decompose
# ---------------------------------------------------------------------------

class TestSVD:
    """Tests for SVD decomposition."""

    def test_svd_reconstruction(self, random_square_transform):
        r"""Test that U @ diag(Sigma) @ V^T ≈ W."""
        W = random_square_transform
        U, Sigma, Vt = svd_decompose(W)
        reconstructed = U @ np.diag(Sigma) @ Vt
        np.testing.assert_allclose(reconstructed, W.data, atol=1e-10)

    def test_svd_singular_values_non_negative(self, random_square_transform):
        """Singular values should all be >= 0."""
        _, Sigma, _ = svd_decompose(random_square_transform)
        assert np.all(Sigma >= 0)

    def test_svd_singular_values_sorted(self, random_square_transform):
        """Singular values should be in descending order."""
        _, Sigma, _ = svd_decompose(random_square_transform)
        assert np.all(np.diff(Sigma) <= 1e-15)

    def test_svd_orthogonal_factors(self, random_square_transform):
        """U and V should have orthonormal columns."""
        U, _, Vt = svd_decompose(random_square_transform)
        np.testing.assert_allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-10)
        np.testing.assert_allclose(Vt @ Vt.T, np.eye(Vt.shape[0]), atol=1e-10)

    def test_diagonal_svd(self, diagonal_transform):
        """SVD of diagonal matrix should give sorted diagonal as singular values."""
        W, diag_vals = diagonal_transform
        _, Sigma, _ = svd_decompose(W)
        np.testing.assert_allclose(Sigma, np.sort(np.abs(diag_vals))[::-1], atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: decompose (combined)
# ---------------------------------------------------------------------------

class TestDecompose:
    """Tests for the unified decompose function."""

    def test_both_returns_all_keys(self, random_square_transform):
        """method='both' should return eigen + SVD results."""
        result = decompose(random_square_transform, method="both")
        assert "eigendecomposition" in result
        assert "U" in result
        assert "Sigma" in result
        assert "Vt" in result

    def test_eigen_only(self, random_square_transform):
        """method='eigen' should return only eigendecomposition."""
        result = decompose(random_square_transform, method="eigen")
        assert "eigendecomposition" in result
        assert "U" not in result

    def test_svd_only(self, random_square_transform):
        """method='svd' should return only SVD."""
        result = decompose(random_square_transform, method="svd")
        assert "U" in result
        assert "eigendecomposition" not in result

    def test_invalid_method_raises(self, random_square_transform):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            decompose(random_square_transform, method="invalid")
