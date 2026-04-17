"""Tests for the CrossVarietyAligner orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.embeddings.alignment import CrossVarietyAligner
from eigendialectos.types import EmbeddingMatrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(
    dialect: DialectCode,
    n_words: int = 30,
    dim: int = 15,
    seed: int = 42,
    rotation: np.ndarray | None = None,
) -> EmbeddingMatrix:
    """Create a test embedding, optionally rotated from a base."""
    rng = np.random.default_rng(seed)
    vocab = [f"w_{i}" for i in range(n_words)]
    data = rng.standard_normal((n_words, dim)).astype(np.float64)
    if rotation is not None:
        data = data @ rotation
    return EmbeddingMatrix(data=data, vocab=vocab, dialect_code=dialect)


def _random_orthogonal(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrossVarietyAligner:
    """Tests for the high-level alignment orchestrator."""

    def test_aligns_all_to_reference(self):
        dim = 15
        base_seed = 42
        Q_mex = _random_orthogonal(dim, seed=10)
        Q_rio = _random_orthogonal(dim, seed=20)

        embeddings = {
            DialectCode.ES_PEN: _make_embedding(DialectCode.ES_PEN, dim=dim, seed=base_seed),
            DialectCode.ES_MEX: _make_embedding(
                DialectCode.ES_MEX, dim=dim, seed=base_seed, rotation=Q_mex,
            ),
            DialectCode.ES_RIO: _make_embedding(
                DialectCode.ES_RIO, dim=dim, seed=base_seed, rotation=Q_rio,
            ),
        }

        aligner = CrossVarietyAligner(method="procrustes", reference=DialectCode.ES_PEN)
        result = aligner.align_all(embeddings)

        # All dialects present
        assert set(result.keys()) == {DialectCode.ES_PEN, DialectCode.ES_MEX, DialectCode.ES_RIO}

        # Reference unchanged
        np.testing.assert_array_equal(
            result[DialectCode.ES_PEN].data,
            embeddings[DialectCode.ES_PEN].data,
        )

        # Aligned MEX should be close to PEN (since they share the same base)
        pen_data = result[DialectCode.ES_PEN].data
        mex_aligned = result[DialectCode.ES_MEX].data
        error = np.linalg.norm(mex_aligned - pen_data, ord="fro")
        baseline = np.linalg.norm(pen_data, ord="fro")
        assert error / baseline < 1e-6

    def test_alignment_matrices_stored(self):
        dim = 10
        embeddings = {
            DialectCode.ES_PEN: _make_embedding(DialectCode.ES_PEN, dim=dim),
            DialectCode.ES_MEX: _make_embedding(DialectCode.ES_MEX, dim=dim, seed=99),
        }

        aligner = CrossVarietyAligner(method="procrustes")
        aligner.align_all(embeddings)

        matrices = aligner.alignment_matrices
        assert DialectCode.ES_MEX in matrices
        assert DialectCode.ES_PEN not in matrices  # reference is identity
        W = matrices[DialectCode.ES_MEX]
        assert W.shape == (dim, dim)
        # Should be orthogonal
        np.testing.assert_allclose(W.T @ W, np.eye(dim), atol=1e-10)

    def test_reference_not_found_raises(self):
        embeddings = {
            DialectCode.ES_MEX: _make_embedding(DialectCode.ES_MEX, dim=10),
        }
        aligner = CrossVarietyAligner(reference=DialectCode.ES_PEN)
        with pytest.raises(ValueError, match="Reference dialect"):
            aligner.align_all(embeddings)

    def test_single_dialect_returns_unchanged(self):
        embeddings = {
            DialectCode.ES_PEN: _make_embedding(DialectCode.ES_PEN, dim=10),
        }
        aligner = CrossVarietyAligner()
        result = aligner.align_all(embeddings)
        assert len(result) == 1
        np.testing.assert_array_equal(
            result[DialectCode.ES_PEN].data,
            embeddings[DialectCode.ES_PEN].data,
        )

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown alignment method"):
            CrossVarietyAligner(method="nonexistent")

    def test_vecmap_method(self):
        dim = 10
        Q = _random_orthogonal(dim, seed=5)
        embeddings = {
            DialectCode.ES_PEN: _make_embedding(DialectCode.ES_PEN, dim=dim, seed=42),
            DialectCode.ES_CHI: _make_embedding(
                DialectCode.ES_CHI, dim=dim, seed=42, rotation=Q
            ),
        }
        aligner = CrossVarietyAligner(method="vecmap", max_iter=5)
        result = aligner.align_all(embeddings)
        assert DialectCode.ES_CHI in result

    def test_muse_method(self):
        dim = 10
        Q = _random_orthogonal(dim, seed=7)
        embeddings = {
            DialectCode.ES_PEN: _make_embedding(DialectCode.ES_PEN, dim=dim, seed=42),
            DialectCode.ES_CAR: _make_embedding(
                DialectCode.ES_CAR, dim=dim, seed=42, rotation=Q,
            ),
        }
        aligner = CrossVarietyAligner(method="muse", max_iter=5, k_csls=3)
        result = aligner.align_all(embeddings)
        assert DialectCode.ES_CAR in result

    def test_properties(self):
        aligner = CrossVarietyAligner(
            method="procrustes", reference=DialectCode.ES_RIO,
        )
        assert aligner.method == "procrustes"
        assert aligner.reference == DialectCode.ES_RIO

    def test_override_reference_and_method(self):
        dim = 10
        Q = _random_orthogonal(dim, seed=3)
        embeddings = {
            DialectCode.ES_PEN: _make_embedding(DialectCode.ES_PEN, dim=dim, seed=42),
            DialectCode.ES_MEX: _make_embedding(
                DialectCode.ES_MEX, dim=dim, seed=42, rotation=Q,
            ),
        }
        # Create with procrustes/PEN defaults, then override at call time
        aligner = CrossVarietyAligner(method="procrustes", reference=DialectCode.ES_PEN)
        result = aligner.align_all(
            embeddings,
            reference=DialectCode.ES_MEX,
            method="procrustes",
        )
        # Now MEX should be the reference (unchanged)
        np.testing.assert_array_equal(
            result[DialectCode.ES_MEX].data,
            embeddings[DialectCode.ES_MEX].data,
        )


# ---------------------------------------------------------------------------
# Registry tests (included here since they're lightweight)
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the model registry."""

    def test_register_and_get(self):
        from eigendialectos.embeddings.registry import (
            clear_registry,
            get_model,
            list_available,
            register_model,
        )
        from eigendialectos.embeddings.word.glove_model import GloVeModel

        clear_registry()
        register_model("test_glove", GloVeModel)
        assert "test_glove" in list_available()

        model = get_model("test_glove")
        assert isinstance(model, GloVeModel)
        clear_registry()

    def test_duplicate_raises(self):
        from eigendialectos.embeddings.registry import (
            clear_registry,
            register_model,
        )
        from eigendialectos.embeddings.word.glove_model import GloVeModel

        clear_registry()
        register_model("dup_test", GloVeModel)
        with pytest.raises(ValueError, match="already registered"):
            register_model("dup_test", GloVeModel)
        clear_registry()

    def test_unknown_model_raises(self):
        from eigendialectos.embeddings.registry import clear_registry, get_model

        clear_registry()
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent")
        clear_registry()

    def test_non_embedding_model_raises(self):
        from eigendialectos.embeddings.registry import clear_registry, register_model

        clear_registry()
        with pytest.raises(TypeError, match="subclass of EmbeddingModel"):
            register_model("bad", str)  # type: ignore[arg-type]
        clear_registry()

    def test_case_insensitive(self):
        from eigendialectos.embeddings.registry import (
            clear_registry,
            get_model,
            register_model,
        )
        from eigendialectos.embeddings.word.glove_model import GloVeModel

        clear_registry()
        register_model("MyModel", GloVeModel)
        model = get_model("mymodel")
        assert isinstance(model, GloVeModel)
        clear_registry()
