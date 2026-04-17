"""Tests for Tucker decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.tensor.construction import build_dialect_tensor
from eigendialectos.types import TensorDialectal, TransformationMatrix

try:
    import tensorly

    HAS_TENSORLY = True
except ImportError:
    HAS_TENSORLY = False

pytestmark = pytest.mark.skipif(
    not HAS_TENSORLY, reason="tensorly not installed"
)


@pytest.fixture
def small_tensor(rng):
    """Small tensor (5x5x3) for decomposition tests."""
    d = 5
    codes = [DialectCode.ES_PEN, DialectCode.ES_RIO, DialectCode.ES_MEX]
    transforms = {}
    for code in codes:
        data = rng.standard_normal((d, d))
        transforms[code] = TransformationMatrix(
            data=data,
            source_dialect=DialectCode.ES_PEN,
            target_dialect=code,
            regularization=0.0,
        )
    return build_dialect_tensor(transforms)


class TestTuckerDecompose:
    """Tests for tucker_decompose."""

    def test_returns_expected_keys(self, small_tensor):
        from eigendialectos.tensor.tucker import tucker_decompose

        result = tucker_decompose(small_tensor, ranks=(3, 3, 2))
        assert "core_tensor" in result
        assert "factor_matrices" in result
        assert "reconstruction_error" in result

    def test_factor_shapes(self, small_tensor):
        from eigendialectos.tensor.tucker import tucker_decompose

        result = tucker_decompose(small_tensor, ranks=(3, 3, 2))
        A, B, C = result["factor_matrices"]
        assert A.shape == (5, 3)
        assert B.shape == (5, 3)
        assert C.shape == (3, 2)

    def test_core_shape(self, small_tensor):
        from eigendialectos.tensor.tucker import tucker_decompose

        result = tucker_decompose(small_tensor, ranks=(3, 3, 2))
        core = result["core_tensor"]
        assert core.shape == (3, 3, 2)

    def test_reconstruction_within_tolerance(self, small_tensor):
        """Full-rank Tucker should reconstruct exactly."""
        from eigendialectos.tensor.tucker import tucker_decompose

        # Full ranks = exact reconstruction
        d, _, m = small_tensor.shape
        result = tucker_decompose(small_tensor, ranks=(d, d, m))
        assert result["reconstruction_error"] < 1e-8

    def test_low_rank_has_positive_error(self, small_tensor):
        from eigendialectos.tensor.tucker import tucker_decompose

        result = tucker_decompose(small_tensor, ranks=(2, 2, 1))
        assert result["reconstruction_error"] > 0

    def test_rank_clamping(self, small_tensor):
        """Ranks exceeding dimensions are clamped."""
        from eigendialectos.tensor.tucker import tucker_decompose

        result = tucker_decompose(small_tensor, ranks=(100, 100, 100))
        A, B, C = result["factor_matrices"]
        assert A.shape[1] <= 5
        assert C.shape[1] <= 3


class TestTuckerReconstruct:
    """Tests for tucker_reconstruct."""

    def test_round_trip(self, small_tensor):
        from eigendialectos.tensor.tucker import tucker_decompose, tucker_reconstruct

        result = tucker_decompose(small_tensor, ranks=(5, 5, 3))
        recon = tucker_reconstruct(result["core_tensor"], result["factor_matrices"])
        np.testing.assert_allclose(
            recon, small_tensor.data, atol=1e-8
        )


class TestExplainedVariance:
    """Tests for explained_variance."""

    def test_full_rank_is_one(self, small_tensor):
        from eigendialectos.tensor.tucker import explained_variance, tucker_decompose

        d, _, m = small_tensor.shape
        result = tucker_decompose(small_tensor, ranks=(d, d, m))
        ev = explained_variance(
            small_tensor, result["core_tensor"], result["factor_matrices"]
        )
        assert ev > 1.0 - 1e-8

    def test_low_rank_between_zero_and_one(self, small_tensor):
        from eigendialectos.tensor.tucker import explained_variance, tucker_decompose

        result = tucker_decompose(small_tensor, ranks=(2, 2, 1))
        ev = explained_variance(
            small_tensor, result["core_tensor"], result["factor_matrices"]
        )
        assert 0.0 <= ev <= 1.0
