"""Tests for the unified algebraic dialect model."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.algebra.model import DialectAlgebra
from eigendialectos.algebra.regionalism import decompose_regionalism
from eigendialectos.constants import DialectCode, FeatureCategory
from eigendialectos.types import EigenDecomposition, TransformationMatrix


@pytest.fixture
def algebra_3x3():
    """DialectAlgebra with 3x3 well-conditioned matrices for 3 dialects."""
    rng = np.random.default_rng(123)
    dim = 3
    codes = [DialectCode.ES_PEN, DialectCode.ES_RIO, DialectCode.ES_MEX]

    transforms = {}
    eigendecomps = {}

    for code in codes:
        if code == DialectCode.ES_PEN:
            # Identity for peninsular (reference)
            W = np.eye(dim)
        else:
            # Close-to-identity, positive-definite matrix
            A = rng.standard_normal((dim, dim)) * 0.1
            W = np.eye(dim) + A + A.T  # symmetric positive definite shift
            W = W / np.linalg.norm(W) * np.sqrt(dim)  # normalize

        eigenvalues, P = np.linalg.eig(W)
        P_inv = np.linalg.inv(P)

        transforms[code] = TransformationMatrix(
            data=W,
            source_dialect=DialectCode.ES_PEN,
            target_dialect=code,
            regularization=0.0,
        )
        eigendecomps[code] = EigenDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=P,
            eigenvectors_inv=P_inv,
            dialect_code=code,
        )

    return DialectAlgebra(transforms=transforms, eigendecomps=eigendecomps)


@pytest.fixture
def algebra_identity():
    """DialectAlgebra where all transforms are identity (exact group)."""
    dim = 5
    codes = [DialectCode.ES_PEN, DialectCode.ES_RIO]

    transforms = {}
    eigendecomps = {}

    for code in codes:
        W = np.eye(dim)
        eigenvalues = np.ones(dim)
        P = np.eye(dim)

        transforms[code] = TransformationMatrix(
            data=W,
            source_dialect=DialectCode.ES_PEN,
            target_dialect=code,
            regularization=0.0,
        )
        eigendecomps[code] = EigenDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=P,
            eigenvectors_inv=P,
            dialect_code=code,
        )

    return DialectAlgebra(transforms=transforms, eigendecomps=eigendecomps)


class TestCompose:
    """Tests for DialectAlgebra.compose."""

    def test_compose_result_shape(self, algebra_3x3):
        result = algebra_3x3.compose(DialectCode.ES_RIO, DialectCode.ES_MEX)
        assert result.data.shape == (3, 3)

    def test_compose_is_matrix_multiply(self, algebra_3x3):
        """compose(d1, d2) must equal W_d1 @ W_d2."""
        d1, d2 = DialectCode.ES_RIO, DialectCode.ES_MEX
        result = algebra_3x3.compose(d1, d2)
        expected = (
            algebra_3x3.transforms[d1].data @ algebra_3x3.transforms[d2].data
        )
        np.testing.assert_allclose(result.data, expected, atol=1e-12)

    def test_compose_with_identity(self, algebra_3x3):
        """Composing with identity (ES_PEN) should return the other matrix."""
        d = DialectCode.ES_RIO
        result = algebra_3x3.compose(d, DialectCode.ES_PEN)
        np.testing.assert_allclose(
            result.data, algebra_3x3.transforms[d].data, atol=1e-12
        )


class TestInvert:
    """Tests for DialectAlgebra.invert."""

    def test_compose_then_invert_gives_identity(self, algebra_3x3):
        """W^{-1} @ W should be close to identity."""
        for code in algebra_3x3.dialects:
            W = algebra_3x3.transforms[code].data
            W_inv = algebra_3x3.invert(code).data
            product = W_inv @ W
            np.testing.assert_allclose(
                product, np.eye(W.shape[0]), atol=1e-8
            )

    def test_invert_identity_is_identity(self, algebra_identity):
        inv = algebra_identity.invert(DialectCode.ES_PEN)
        np.testing.assert_allclose(inv.data, np.eye(5), atol=1e-12)


class TestInterpolate:
    """Tests for DialectAlgebra.interpolate."""

    def test_alpha_zero_gives_identity(self, algebra_3x3):
        """interpolate(d, 0) must return identity."""
        for code in algebra_3x3.dialects:
            result = algebra_3x3.interpolate(code, alpha=0.0)
            np.testing.assert_allclose(
                result.data, np.eye(algebra_3x3.dim), atol=1e-12
            )

    def test_alpha_one_gives_original(self, algebra_3x3):
        """interpolate(d, 1) must return W_d."""
        for code in algebra_3x3.dialects:
            result = algebra_3x3.interpolate(code, alpha=1.0)
            np.testing.assert_allclose(
                result.data,
                algebra_3x3.transforms[code].data,
                atol=1e-6,
            )

    def test_alpha_half_between(self, algebra_3x3):
        """interpolate(d, 0.5) should produce a matrix between I and W."""
        code = DialectCode.ES_RIO
        result = algebra_3x3.interpolate(code, alpha=0.5)
        W = algebra_3x3.transforms[code].data
        I = np.eye(algebra_3x3.dim)

        # Distance from I should be less than distance of W from I
        dist_from_I = np.linalg.norm(result.data - I, "fro")
        dist_W_from_I = np.linalg.norm(W - I, "fro")
        assert dist_from_I < dist_W_from_I + 1e-8


class TestProjectOntoSubspace:
    """Tests for DialectAlgebra.project_onto_subspace."""

    def test_projection_is_idempotent(self, algebra_3x3):
        """Projecting twice gives the same result."""
        code = DialectCode.ES_RIO
        # Define a 2-dimensional subspace
        V = np.eye(3)[:, :2]  # first 2 standard basis vectors

        proj1 = algebra_3x3.project_onto_subspace(code, V)
        # Build a temporary algebra with the projected result to project again
        transforms2 = {code: proj1}
        eigendecomps2: dict = {}
        alg2 = DialectAlgebra(transforms2, eigendecomps2)
        proj2 = alg2.project_onto_subspace(code, V)

        np.testing.assert_allclose(proj1.data, proj2.data, atol=1e-10)

    def test_full_subspace_preserves_matrix(self, algebra_3x3):
        """Projecting onto the full space returns the original."""
        code = DialectCode.ES_RIO
        V = np.eye(3)  # full space
        proj = algebra_3x3.project_onto_subspace(code, V)
        np.testing.assert_allclose(
            proj.data, algebra_3x3.transforms[code].data, atol=1e-10
        )


class TestIsApproximateGroup:
    """Tests for DialectAlgebra.is_approximate_group."""

    def test_identity_algebra_is_group(self, algebra_identity):
        """All-identity matrices should form an exact group."""
        result = algebra_identity.is_approximate_group(tol=1e-8)
        assert result["closure"] is True
        assert result["associativity"] is True
        assert result["identity"]["exists"] is True
        assert result["inverse"]["ok"] is True

    def test_returns_all_keys(self, algebra_3x3):
        result = algebra_3x3.is_approximate_group()
        assert "closure" in result
        assert "associativity" in result
        assert "identity" in result
        assert "inverse" in result
        assert "details" in result


class TestRegionalismDecomposition:
    """Tests for decompose_regionalism."""

    def test_components_sum_to_deviation(self):
        """Sum of additive components should approximate W - I."""
        rng = np.random.default_rng(77)
        d = 6
        W_data = np.eye(d) + rng.standard_normal((d, d)) * 0.05
        W = TransformationMatrix(
            data=W_data,
            source_dialect=DialectCode.ES_PEN,
            target_dialect=DialectCode.ES_RIO,
            regularization=0.0,
        )

        # Create orthogonal subspaces that span the full space
        # Use QR to get orthogonal basis, then split
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        subspaces = {
            FeatureCategory.LEXICAL: Q[:, 0:2],
            FeatureCategory.MORPHOSYNTACTIC: Q[:, 2:4],
            FeatureCategory.PHONOLOGICAL: Q[:, 4:6],
        }

        result = decompose_regionalism(W, subspaces)

        # Sum of deltas should approximate W - I when subspaces span the space
        total_delta = sum(tm.data for tm in result.values())
        deviation = W_data - np.eye(d)
        np.testing.assert_allclose(total_delta, deviation, atol=1e-10)

    def test_partial_subspace_gives_partial_reconstruction(self):
        """With partial subspaces, we should still get valid components."""
        rng = np.random.default_rng(88)
        d = 6
        W_data = np.eye(d) + rng.standard_normal((d, d)) * 0.1
        W = TransformationMatrix(
            data=W_data,
            source_dialect=DialectCode.ES_PEN,
            target_dialect=DialectCode.ES_AND,
            regularization=0.0,
        )

        subspaces = {
            FeatureCategory.LEXICAL: np.eye(d)[:, :2],
        }

        result = decompose_regionalism(W, subspaces)
        assert FeatureCategory.LEXICAL in result
        # Component norm should be <= deviation norm
        delta_norm = np.linalg.norm(result[FeatureCategory.LEXICAL].data)
        dev_norm = np.linalg.norm(W_data - np.eye(d))
        assert delta_norm <= dev_norm + 1e-10
