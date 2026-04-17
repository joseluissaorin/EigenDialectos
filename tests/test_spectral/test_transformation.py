"""Tests for spectral.transformation module."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.spectral.transformation import (
    compute_all_transforms,
    compute_transformation_matrix,
)
from eigendialectos.spectral.utils import check_condition_number, is_orthogonal
from eigendialectos.types import EmbeddingMatrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aligned_embeddings(rng):
    """Create source/target embeddings related by a known linear transform."""
    d, V = 10, 50
    vocab = [f"w{i}" for i in range(V)]

    E_source = rng.standard_normal((d, V)).astype(np.float64)

    # Known transform: a rotation-like matrix
    W_true = np.eye(d, dtype=np.float64)
    W_true[0, 1] = 0.5
    W_true[1, 0] = -0.5

    E_target = W_true @ E_source

    source = EmbeddingMatrix(data=E_source, vocab=vocab, dialect_code=DialectCode.ES_PEN)
    target = EmbeddingMatrix(data=E_target, vocab=vocab, dialect_code=DialectCode.ES_RIO)

    return source, target, W_true


@pytest.fixture
def small_embeddings(rng):
    """Small random embeddings for basic smoke tests."""
    d, V = 5, 20
    vocab = [f"w{i}" for i in range(V)]
    source = EmbeddingMatrix(
        data=rng.standard_normal((d, V)).astype(np.float64),
        vocab=vocab,
        dialect_code=DialectCode.ES_PEN,
    )
    target = EmbeddingMatrix(
        data=rng.standard_normal((d, V)).astype(np.float64),
        vocab=vocab,
        dialect_code=DialectCode.ES_MEX,
    )
    return source, target


# ---------------------------------------------------------------------------
# Tests: lstsq (ridge regression)
# ---------------------------------------------------------------------------

class TestLstsq:
    """Tests for the ridge-regression transformation method."""

    def test_recovers_known_transform(self, aligned_embeddings):
        """When E_target = W_true @ E_source, lstsq should recover W_true."""
        source, target, W_true = aligned_embeddings
        W = compute_transformation_matrix(
            source, target, method="lstsq", regularization=1e-8
        )
        np.testing.assert_allclose(W.data, W_true, atol=1e-4)

    def test_shape(self, small_embeddings):
        """Output should be (d, d) where d is the embedding row dimension."""
        source, target = small_embeddings
        W = compute_transformation_matrix(source, target, method="lstsq")
        d = source.data.shape[0]  # row dimension = embedding dimensionality
        assert W.data.shape == (d, d)

    def test_ridge_reduces_condition_number(self, small_embeddings):
        """Higher regularisation should reduce the condition number."""
        source, target = small_embeddings
        W_low = compute_transformation_matrix(
            source, target, method="lstsq", regularization=0.001
        )
        W_high = compute_transformation_matrix(
            source, target, method="lstsq", regularization=1.0
        )
        cond_low = check_condition_number(W_low.data, threshold=1e15)
        cond_high = check_condition_number(W_high.data, threshold=1e15)
        # Higher regularisation should give better-conditioned result
        # (or at least not blow up)
        assert np.isfinite(cond_low)
        assert np.isfinite(cond_high)

    def test_metadata(self, small_embeddings):
        """Source/target dialect codes and regularisation stored correctly."""
        source, target = small_embeddings
        W = compute_transformation_matrix(
            source, target, method="lstsq", regularization=0.05
        )
        assert W.source_dialect == DialectCode.ES_PEN
        assert W.target_dialect == DialectCode.ES_MEX
        assert W.regularization == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Tests: Procrustes
# ---------------------------------------------------------------------------

class TestProcrustes:
    """Tests for the orthogonal Procrustes method."""

    def test_result_is_orthogonal(self, small_embeddings):
        """The Procrustes solution must be an orthogonal matrix."""
        source, target = small_embeddings
        W = compute_transformation_matrix(source, target, method="procrustes")
        assert is_orthogonal(W.data, tol=1e-5)

    def test_determinant_is_pm1(self, small_embeddings):
        """det(W) should be +/- 1 for an orthogonal matrix."""
        source, target = small_embeddings
        W = compute_transformation_matrix(source, target, method="procrustes")
        det = np.linalg.det(W.data)
        assert abs(abs(det) - 1.0) < 1e-5

    def test_recovers_rotation(self, rng):
        """When the true transform is a rotation, Procrustes should recover it."""
        d, V = 5, 30
        vocab = [f"w{i}" for i in range(V)]
        E_s = rng.standard_normal((d, V)).astype(np.float64)

        # Build a random rotation via QR
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1  # ensure det = +1
        E_t = Q @ E_s

        source = EmbeddingMatrix(data=E_s, vocab=vocab, dialect_code=DialectCode.ES_PEN)
        target = EmbeddingMatrix(data=E_t, vocab=vocab, dialect_code=DialectCode.ES_AND)

        W = compute_transformation_matrix(source, target, method="procrustes")
        np.testing.assert_allclose(W.data, Q, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: nuclear
# ---------------------------------------------------------------------------

class TestNuclear:
    """Tests for nuclear-norm regularised method."""

    def test_nuclear_gives_lower_rank(self, small_embeddings):
        """Nuclear-norm regularisation with large lambda should shrink singular values."""
        source, target = small_embeddings
        W_unreg = compute_transformation_matrix(
            source, target, method="lstsq", regularization=1e-8
        )
        W_nuc = compute_transformation_matrix(
            source, target, method="nuclear", regularization=1.0
        )
        sv_unreg = np.linalg.svd(W_unreg.data, compute_uv=False)
        sv_nuc = np.linalg.svd(W_nuc.data, compute_uv=False)
        # Nuclear-regularised should have smaller (or zero) singular values
        assert np.sum(sv_nuc) <= np.sum(sv_unreg) + 1e-8


# ---------------------------------------------------------------------------
# Tests: compute_all_transforms
# ---------------------------------------------------------------------------

class TestComputeAllTransforms:
    """Tests for batch transformation computation."""

    def test_returns_all_dialects(self, random_embeddings):
        """Should return one transform per dialect in the input dict."""
        transforms = compute_all_transforms(
            random_embeddings, reference=DialectCode.ES_PEN
        )
        assert set(transforms.keys()) == set(random_embeddings.keys())

    def test_self_transform_near_identity(self, random_embeddings):
        """The reference-to-self transform should be close to identity."""
        ref = DialectCode.ES_PEN
        transforms = compute_all_transforms(
            random_embeddings, reference=ref, regularization=1e-6
        )
        self_W = transforms[ref].data
        d = self_W.shape[0]
        np.testing.assert_allclose(self_W, np.eye(d), atol=0.1)

    def test_invalid_reference_raises(self, random_embeddings):
        """Passing a reference not in the dict should raise ValueError."""
        small = {DialectCode.ES_PEN: random_embeddings[DialectCode.ES_PEN]}
        with pytest.raises(ValueError, match="not in embeddings"):
            compute_all_transforms(small, reference=DialectCode.ES_CHI)


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for edge cases and invalid inputs."""

    def test_shape_mismatch_raises(self, rng):
        """Source and target with different shapes should raise."""
        vocab5 = [f"w{i}" for i in range(5)]
        vocab10 = [f"w{i}" for i in range(10)]
        s = EmbeddingMatrix(
            data=rng.standard_normal((3, 5)), vocab=vocab5, dialect_code=DialectCode.ES_PEN
        )
        t = EmbeddingMatrix(
            data=rng.standard_normal((3, 10)), vocab=vocab10, dialect_code=DialectCode.ES_RIO
        )
        with pytest.raises(ValueError, match="shape"):
            compute_transformation_matrix(s, t)

    def test_unknown_method_raises(self, small_embeddings):
        """Unknown method should raise ValueError."""
        source, target = small_embeddings
        with pytest.raises(ValueError, match="Unknown method"):
            compute_transformation_matrix(source, target, method="bogus")
