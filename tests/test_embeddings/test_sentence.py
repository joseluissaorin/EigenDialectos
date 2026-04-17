"""Tests for sentence-level embedding models (BETO, MarIA, SpanBERTa).

These models require transformers + torch, which are heavy dependencies.
Tests are skipped if not installed.  When available, we test only with
tiny inputs to keep CI fast.
"""

from __future__ import annotations

import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.embeddings.base import EmbeddingModel


# ---------------------------------------------------------------------------
# BETO
# ---------------------------------------------------------------------------

class TestBETOModel:
    """Tests for the BETO sentence embedding model."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_transformers(self):
        pytest.importorskip("transformers")
        pytest.importorskip("torch")

    def test_is_embedding_model(self):
        from eigendialectos.embeddings.sentence.beto_model import BETOModel

        assert issubclass(BETOModel, EmbeddingModel)

    def test_level(self):
        from eigendialectos.embeddings.sentence.beto_model import BETOModel

        model = BETOModel(dialect_code=DialectCode.ES_PEN, device="cpu")
        assert model.level() == "sentence"

    def test_default_dim(self):
        from eigendialectos.embeddings.sentence.beto_model import BETOModel

        model = BETOModel(device="cpu")
        # Before loading, should return default
        assert model.embedding_dim() == 768

    @pytest.mark.slow
    def test_encode_shape(self):
        """Requires network access to download the model."""
        from eigendialectos.embeddings.sentence.beto_model import BETOModel

        model = BETOModel(device="cpu")
        result = model.encode(["Hola mundo", "El gato negro"])
        assert result.shape[0] == 2
        assert result.shape[1] == 768

    @pytest.mark.slow
    def test_encode_words(self):
        from eigendialectos.embeddings.sentence.beto_model import BETOModel

        model = BETOModel(device="cpu")
        em = model.encode_words(["gato", "perro"])
        assert em.data.shape[0] == 2
        assert len(em.vocab) == 2


# ---------------------------------------------------------------------------
# MarIA
# ---------------------------------------------------------------------------

class TestMarIAModel:
    """Tests for the MarIA sentence embedding model."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_transformers(self):
        pytest.importorskip("transformers")
        pytest.importorskip("torch")

    def test_is_embedding_model(self):
        from eigendialectos.embeddings.sentence.maria_model import MarIAModel

        assert issubclass(MarIAModel, EmbeddingModel)

    def test_level(self):
        from eigendialectos.embeddings.sentence.maria_model import MarIAModel

        model = MarIAModel(device="cpu")
        assert model.level() == "sentence"


# ---------------------------------------------------------------------------
# SpanBERTa
# ---------------------------------------------------------------------------

class TestSpanBERTaModel:
    """Tests for the SpanBERTa sentence embedding model."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_transformers(self):
        pytest.importorskip("transformers")
        pytest.importorskip("torch")

    def test_is_embedding_model(self):
        from eigendialectos.embeddings.sentence.spanbert_model import SpanBERTaModel

        assert issubclass(SpanBERTaModel, EmbeddingModel)

    def test_level(self):
        from eigendialectos.embeddings.sentence.spanbert_model import SpanBERTaModel

        model = SpanBERTaModel(device="cpu")
        assert model.level() == "sentence"


# ---------------------------------------------------------------------------
# Import guard tests
# ---------------------------------------------------------------------------

class TestTransformerImportGuards:
    """Verify that import errors give clear install instructions."""

    def test_beto_import_error_message(self, monkeypatch):
        """If transformers not available, error message is helpful."""
        import eigendialectos.embeddings.sentence.beto_model as mod

        original = mod._HAS_TRANSFORMERS
        monkeypatch.setattr(mod, "_HAS_TRANSFORMERS", False)
        try:
            with pytest.raises(ImportError, match="pip install"):
                mod._require_transformers()
        finally:
            monkeypatch.setattr(mod, "_HAS_TRANSFORMERS", original)

    def test_maria_import_error_message(self, monkeypatch):
        import eigendialectos.embeddings.sentence.maria_model as mod

        original = mod._HAS_TRANSFORMERS
        monkeypatch.setattr(mod, "_HAS_TRANSFORMERS", False)
        try:
            with pytest.raises(ImportError, match="pip install"):
                mod._require_transformers()
        finally:
            monkeypatch.setattr(mod, "_HAS_TRANSFORMERS", original)

    def test_spanberta_import_error_message(self, monkeypatch):
        import eigendialectos.embeddings.sentence.spanbert_model as mod

        original = mod._HAS_TRANSFORMERS
        monkeypatch.setattr(mod, "_HAS_TRANSFORMERS", False)
        try:
            with pytest.raises(ImportError, match="pip install"):
                mod._require_transformers()
        finally:
            monkeypatch.setattr(mod, "_HAS_TRANSFORMERS", original)
