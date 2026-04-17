"""Tests for CP (CANDECOMP/PARAFAC) decomposition."""

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
def rank1_tensor():
    """A known rank-1 tensor: T = a (x) b (x) c."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, 1.5, 2.5])
    c = np.array([1.0, -1.0])
    # Outer product: T[i,j,k] = a[i] * b[j] * c[k]
    data = np.einsum("i,j,k->ijk", a, b, c)

    codes = [DialectCode.ES_PEN, DialectCode.ES_RIO]
    return TensorDialectal(data=data, dialect_codes=codes)


@pytest.fixture
def small_tensor(rng):
    """Small random tensor (5x5x3) for general CP tests."""
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


class TestCPDecompose:
    """Tests for cp_decompose."""

    def test_returns_expected_keys(self, small_tensor):
        from eigendialectos.tensor.cp import cp_decompose

        result = cp_decompose(small_tensor, rank=3, n_restarts=2)
        assert "weights" in result
        assert "factor_matrices" in result
        assert "reconstruction_error" in result

    def test_factor_shapes(self, small_tensor):
        from eigendialectos.tensor.cp import cp_decompose

        rank = 3
        result = cp_decompose(small_tensor, rank=rank, n_restarts=2)
        A, B, C = result["factor_matrices"]
        assert A.shape == (5, rank)
        assert B.shape == (5, rank)
        assert C.shape == (3, rank)

    def test_weights_shape(self, small_tensor):
        from eigendialectos.tensor.cp import cp_decompose

        rank = 4
        result = cp_decompose(small_tensor, rank=rank, n_restarts=2)
        assert result["weights"].shape == (rank,)

    def test_rank1_near_exact(self, rank1_tensor):
        """CP decomposition of a rank-1 tensor with rank=1 should be near-exact."""
        from eigendialectos.tensor.cp import cp_decompose

        result = cp_decompose(rank1_tensor, rank=1, n_restarts=3)
        total_norm = float(np.linalg.norm(rank1_tensor.data))
        relative_error = result["reconstruction_error"] / total_norm
        assert relative_error < 0.01, (
            f"Rank-1 tensor CP error too large: {relative_error:.4f}"
        )

    def test_multiple_restarts_improve(self, small_tensor):
        """More restarts should give at least as good a result."""
        from eigendialectos.tensor.cp import cp_decompose

        result_1 = cp_decompose(small_tensor, rank=3, n_restarts=1)
        result_5 = cp_decompose(small_tensor, rank=3, n_restarts=5)
        # With more restarts, error should be <= (or very close)
        assert result_5["reconstruction_error"] <= result_1["reconstruction_error"] * 1.1


class TestCoreConsistency:
    """Tests for core_consistency."""

    def test_rank1_high_consistency(self, rank1_tensor):
        """Rank-1 tensor at rank=1 should have high core consistency."""
        from eigendialectos.tensor.cp import core_consistency

        cc = core_consistency(rank1_tensor, rank=1)
        assert cc > 50.0, f"Expected high consistency, got {cc:.1f}"

    def test_returns_float(self, small_tensor):
        from eigendialectos.tensor.cp import core_consistency

        cc = core_consistency(small_tensor, rank=2)
        assert isinstance(cc, float)
