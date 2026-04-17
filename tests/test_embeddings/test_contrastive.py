"""Tests for contrastive alignment algorithms (Procrustes, VecMap, MUSE).

These are the most critical tests in the project since the alignment
matrices feed directly into the spectral analysis pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.embeddings.contrastive.muse import MUSEAligner, _csls_score
from eigendialectos.embeddings.contrastive.procrustes import ProcrustesAligner
from eigendialectos.embeddings.contrastive.vecmap import VecMapAligner
from eigendialectos.types import EmbeddingMatrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_embedding(
    n_words: int,
    dim: int,
    dialect: DialectCode,
    seed: int = 42,
    prefix: str = "word",
) -> EmbeddingMatrix:
    """Create a random embedding matrix with named vocabulary."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_words, dim))
    vocab = [f"{prefix}_{i}" for i in range(n_words)]
    return EmbeddingMatrix(data=data.astype(np.float64), vocab=vocab, dialect_code=dialect)


def _make_aligned_pair(
    n_words: int = 50,
    dim: int = 30,
    seed: int = 42,
) -> tuple[EmbeddingMatrix, EmbeddingMatrix, np.ndarray]:
    """Create source/target pair where target = source @ Q for a known Q.

    Returns (source, target, Q) where Q is a random orthogonal matrix.
    """
    rng = np.random.default_rng(seed)

    # Random orthogonal matrix via QR decomposition
    A = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(A)
    # Ensure proper rotation (det = +1)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1

    vocab = [f"w_{i}" for i in range(n_words)]
    source_data = rng.standard_normal((n_words, dim)).astype(np.float64)
    target_data = (source_data @ Q).astype(np.float64)

    source = EmbeddingMatrix(data=source_data, vocab=vocab, dialect_code=DialectCode.ES_MEX)
    target = EmbeddingMatrix(data=target_data, vocab=vocab, dialect_code=DialectCode.ES_PEN)

    return source, target, Q


# ---------------------------------------------------------------------------
# Procrustes tests
# ---------------------------------------------------------------------------


class TestProcrustesAligner:
    """Tests for the orthogonal Procrustes alignment."""

    def test_alignment_matrix_is_orthogonal(self):
        """W^T W = I (the defining property)."""
        source, target, _ = _make_aligned_pair(n_words=40, dim=20)
        aligner = ProcrustesAligner(normalize=False)
        W = aligner.align(source, target)

        product = W.T @ W
        np.testing.assert_allclose(
            product, np.eye(W.shape[0]), atol=1e-10,
            err_msg="Procrustes matrix is not orthogonal: W^T W != I",
        )

    def test_alignment_matrix_has_det_plus_one(self):
        """det(W) = +1 (proper rotation, not reflection)."""
        source, target, _ = _make_aligned_pair(n_words=40, dim=20)
        aligner = ProcrustesAligner(normalize=False)
        W = aligner.align(source, target)

        det = np.linalg.det(W)
        assert abs(det - 1.0) < 1e-8, f"Expected det(W)=+1, got {det}"

    def test_recovers_known_rotation(self):
        """When target = source @ Q, Procrustes should recover W close to Q."""
        source, target, Q = _make_aligned_pair(n_words=80, dim=20, seed=123)
        aligner = ProcrustesAligner(normalize=False)
        W = aligner.align(source, target)

        # W should be close to Q (up to numerical precision)
        # Check: source @ W should be close to target
        aligned = source.data @ W
        error = np.linalg.norm(aligned - target.data, ord="fro")
        baseline = np.linalg.norm(target.data, ord="fro")
        relative_error = error / baseline
        assert relative_error < 1e-8, (
            f"Procrustes failed to recover known rotation: "
            f"relative error = {relative_error:.2e}"
        )

    def test_minimises_frobenius_distance(self):
        """The Procrustes solution should give smaller ||XW - Y||_F than a random orthogonal matrix."""
        source, target, _ = _make_aligned_pair(n_words=50, dim=20)
        aligner = ProcrustesAligner(normalize=False)
        W = aligner.align(source, target)

        aligned = source.data @ W
        procrustes_error = np.linalg.norm(aligned - target.data, ord="fro")

        # Random orthogonal matrix
        rng = np.random.default_rng(999)
        A = rng.standard_normal((20, 20))
        Q_rand, _ = np.linalg.qr(A)
        random_error = np.linalg.norm(source.data @ Q_rand - target.data, ord="fro")

        assert procrustes_error <= random_error + 1e-10, (
            f"Procrustes error ({procrustes_error:.4f}) >= "
            f"random error ({random_error:.4f})"
        )

    def test_with_normalisation(self):
        """Normalised Procrustes should still produce orthogonal W."""
        source, target, _ = _make_aligned_pair(n_words=40, dim=20)
        aligner = ProcrustesAligner(normalize=True)
        W = aligner.align(source, target)

        product = W.T @ W
        np.testing.assert_allclose(product, np.eye(W.shape[0]), atol=1e-10)

    def test_partial_anchors(self):
        """Alignment using a subset of vocabulary as anchors."""
        source, target, Q = _make_aligned_pair(n_words=60, dim=15)
        # Use only first 30 words as anchors
        anchors = source.vocab[:30]
        aligner = ProcrustesAligner(normalize=False)
        W = aligner.align(source, target, anchors=anchors)

        # Still orthogonal
        np.testing.assert_allclose(W.T @ W, np.eye(15), atol=1e-10)

        # Still a good alignment (not perfect since only partial anchors,
        # but for a known rotation it should be exact)
        aligned = source.data @ W
        error = np.linalg.norm(aligned - target.data, ord="fro")
        baseline = np.linalg.norm(target.data, ord="fro")
        assert error / baseline < 1e-8

    def test_transform(self):
        """transform() applies W to all source vectors."""
        source, target, _ = _make_aligned_pair(n_words=30, dim=10)
        aligner = ProcrustesAligner(normalize=False)
        W = aligner.align(source, target)
        aligned_em = aligner.transform(source)

        np.testing.assert_allclose(aligned_em.data, source.data @ W, atol=1e-10)
        assert aligned_em.vocab == source.vocab
        assert aligned_em.dialect_code == source.dialect_code

    def test_no_anchors_raises(self):
        """Should raise if no anchors overlap."""
        source = _random_embedding(10, 5, DialectCode.ES_MEX, prefix="src")
        target = _random_embedding(10, 5, DialectCode.ES_PEN, prefix="tgt")
        aligner = ProcrustesAligner()
        with pytest.raises(ValueError, match="No anchor words"):
            aligner.align(source, target)

    def test_alignment_matrix_property(self):
        source, target, _ = _make_aligned_pair(n_words=20, dim=10)
        aligner = ProcrustesAligner()
        assert aligner.alignment_matrix is None
        W = aligner.align(source, target)
        assert aligner.alignment_matrix is not None
        np.testing.assert_array_equal(aligner.alignment_matrix, W)

    def test_symmetry_property(self):
        """If W aligns X -> Y, then W^T should align Y -> X."""
        source, target, _ = _make_aligned_pair(n_words=40, dim=15)
        aligner = ProcrustesAligner(normalize=False)
        W_fwd = aligner.align(source, target)

        # Reverse alignment
        W_rev = aligner.align(target, source)

        # W_fwd @ W_rev should be close to identity
        product = W_fwd @ W_rev
        np.testing.assert_allclose(product, np.eye(15), atol=1e-6)

    def test_high_dimensional(self):
        """Procrustes works in higher dimensions too."""
        source, target, _ = _make_aligned_pair(n_words=100, dim=300, seed=7)
        aligner = ProcrustesAligner(normalize=False)
        W = aligner.align(source, target)

        # Check orthogonality via ||W^T W - I||_F (det overflows for d=300)
        np.testing.assert_allclose(W.T @ W, np.eye(300), atol=1e-8)
        aligned = source.data @ W
        error = np.linalg.norm(aligned - target.data, ord="fro")
        baseline = np.linalg.norm(target.data, ord="fro")
        assert error / baseline < 1e-8


# ---------------------------------------------------------------------------
# VecMap tests
# ---------------------------------------------------------------------------


class TestVecMapAligner:
    """Tests for the VecMap iterative self-learning aligner."""

    def test_produces_orthogonal_matrix(self):
        source, target, _ = _make_aligned_pair(n_words=50, dim=20)
        aligner = VecMapAligner(max_iter=5, normalize=False)
        W = aligner.align(source, target)

        np.testing.assert_allclose(W.T @ W, np.eye(20), atol=1e-10)

    def test_converges_on_known_rotation(self):
        source, target, Q = _make_aligned_pair(n_words=50, dim=20)
        aligner = VecMapAligner(max_iter=20, normalize=False)
        W = aligner.align(source, target)

        aligned = source.data @ W
        error = np.linalg.norm(aligned - target.data, ord="fro")
        baseline = np.linalg.norm(target.data, ord="fro")
        assert error / baseline < 1e-6

    def test_convergence_history(self):
        source, target, _ = _make_aligned_pair(n_words=40, dim=15)
        aligner = VecMapAligner(max_iter=10)
        aligner.align(source, target)

        history = aligner.convergence_history
        assert len(history) > 0
        assert "objective" in history[0]
        assert "n_anchors" in history[0]

    def test_transform(self):
        source, target, _ = _make_aligned_pair(n_words=30, dim=10)
        aligner = VecMapAligner(max_iter=3, normalize=False)
        W = aligner.align(source, target)
        aligned_em = aligner.transform(source)

        np.testing.assert_allclose(aligned_em.data, source.data @ W, atol=1e-10)

    def test_no_anchors_raises(self):
        source = _random_embedding(10, 5, DialectCode.ES_MEX, prefix="src")
        target = _random_embedding(10, 5, DialectCode.ES_PEN, prefix="tgt")
        aligner = VecMapAligner()
        with pytest.raises(ValueError, match="No seed anchors"):
            aligner.align(source, target)


# ---------------------------------------------------------------------------
# MUSE tests
# ---------------------------------------------------------------------------


class TestMUSEAligner:
    """Tests for the MUSE Procrustes + CSLS aligner."""

    def test_produces_orthogonal_matrix(self):
        source, target, _ = _make_aligned_pair(n_words=50, dim=20)
        aligner = MUSEAligner(max_iter=5, normalize=False)
        W = aligner.align(source, target)

        np.testing.assert_allclose(W.T @ W, np.eye(20), atol=1e-10)

    def test_converges_on_known_rotation(self):
        source, target, Q = _make_aligned_pair(n_words=50, dim=20)
        aligner = MUSEAligner(max_iter=20, k_csls=5, normalize=False)
        W = aligner.align(source, target)

        aligned = source.data @ W
        error = np.linalg.norm(aligned - target.data, ord="fro")
        baseline = np.linalg.norm(target.data, ord="fro")
        assert error / baseline < 1e-6

    def test_convergence_history(self):
        source, target, _ = _make_aligned_pair(n_words=40, dim=15)
        aligner = MUSEAligner(max_iter=5)
        aligner.align(source, target)

        history = aligner.convergence_history
        assert len(history) > 0

    def test_no_anchors_raises(self):
        source = _random_embedding(10, 5, DialectCode.ES_MEX, prefix="src")
        target = _random_embedding(10, 5, DialectCode.ES_PEN, prefix="tgt")
        aligner = MUSEAligner()
        with pytest.raises(ValueError, match="No seed anchors"):
            aligner.align(source, target)


# ---------------------------------------------------------------------------
# CSLS score tests
# ---------------------------------------------------------------------------


class TestCSLS:
    """Test the CSLS scoring function directly."""

    def test_shape(self):
        rng = np.random.default_rng(42)
        src = rng.standard_normal((10, 5))
        tgt = rng.standard_normal((15, 5))
        src = src / np.linalg.norm(src, axis=1, keepdims=True)
        tgt = tgt / np.linalg.norm(tgt, axis=1, keepdims=True)

        scores = _csls_score(src, tgt, k=3)
        assert scores.shape == (10, 15)

    def test_identical_vectors_high_score(self):
        """CSLS score should be highest for identical vectors."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 10))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        scores = _csls_score(X, X, k=2)
        # Diagonal should have the highest scores per row
        for i in range(5):
            assert scores[i, i] == pytest.approx(np.max(scores[i, :]), abs=1e-10)
