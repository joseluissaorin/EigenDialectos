"""Tests for subword embedding models (FastText, BPE)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.embeddings.base import EmbeddingModel
from eigendialectos.types import CorpusSlice, DialectSample

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_TEXTS = [
    "el gato negro duerme en la silla",
    "la casa grande tiene un jardin bonito",
    "los ninos juegan en el parque",
    "el perro corre por la calle",
    "la mujer camina hacia la tienda",
    "el hombre lee un libro interesante",
    "las flores rojas estan en la mesa",
    "el cielo azul se ve muy claro",
    "la ciudad tiene muchos edificios altos",
    "los pajaros cantan por la manana",
    "el rio fluye hacia el mar",
    "la montana es muy alta y fria",
    "el gato negro come su comida",
    "la casa tiene una puerta roja",
    "los ninos van a la escuela",
]


def _make_corpus(dialect: DialectCode = DialectCode.ES_PEN) -> CorpusSlice:
    samples = [
        DialectSample(
            text=t,
            dialect_code=dialect,
            source_id=f"test_{i}",
            confidence=1.0,
        )
        for i, t in enumerate(_SAMPLE_TEXTS)
    ]
    return CorpusSlice(samples=samples, dialect_code=dialect)


# ---------------------------------------------------------------------------
# BPE Model tests (no heavy dependencies)
# ---------------------------------------------------------------------------


class TestBPEModel:
    """Tests for the BPE subword model."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_tokenizers(self):
        pytest.importorskip("tokenizers")

    def test_is_embedding_model(self):
        from eigendialectos.embeddings.subword.bpe_model import BPEModel

        assert issubclass(BPEModel, EmbeddingModel)

    def test_train_and_encode(self):
        from eigendialectos.embeddings.subword.bpe_model import BPEModel

        model = BPEModel(
            dialect_code=DialectCode.ES_PEN,
            vocab_size=200,
            vector_size=32,
        )
        corpus = _make_corpus()
        model.train(corpus, config={"min_frequency": 1})

        assert model.is_trained
        assert model.level() == "subword"
        assert model.vocab_size() > 0
        assert model.embedding_dim() == 32

    def test_encode_returns_correct_shape(self):
        from eigendialectos.embeddings.subword.bpe_model import BPEModel

        model = BPEModel(vocab_size=200, vector_size=32)
        model.train(_make_corpus(), config={"min_frequency": 1})

        texts = ["gato negro", "casa grande"]
        result = model.encode(texts)
        assert result.shape == (2, 32)
        assert result.dtype == np.float64

    def test_encode_words_returns_embedding_matrix(self):
        from eigendialectos.embeddings.subword.bpe_model import BPEModel

        model = BPEModel(vocab_size=200, vector_size=32)
        model.train(_make_corpus(), config={"min_frequency": 1})

        em = model.encode_words(["gato", "casa", "perro"])
        assert em.data.shape[0] == 3
        assert em.data.shape[1] == 32
        assert len(em.vocab) == 3

    def test_save_and_load(self, tmp_path):
        from eigendialectos.embeddings.subword.bpe_model import BPEModel

        model = BPEModel(
            dialect_code=DialectCode.ES_MEX,
            vocab_size=200,
            vector_size=32,
        )
        model.train(_make_corpus(DialectCode.ES_MEX), config={"min_frequency": 1})
        original_encoding = model.encode(["gato negro"])

        save_dir = tmp_path / "bpe_model"
        model.save(save_dir)

        model2 = BPEModel(vocab_size=200, vector_size=32)
        model2.load(save_dir)

        assert model2.is_trained
        loaded_encoding = model2.encode(["gato negro"])
        np.testing.assert_allclose(original_encoding, loaded_encoding, atol=1e-6)

    def test_untrained_model_raises(self):
        from eigendialectos.embeddings.subword.bpe_model import BPEModel

        model = BPEModel(vocab_size=200, vector_size=32)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.encode(["hola"])


# ---------------------------------------------------------------------------
# FastText Model tests
# ---------------------------------------------------------------------------


class TestFastTextModel:
    """Tests for the FastText model (requires gensim)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_gensim(self):
        pytest.importorskip("gensim")

    def test_is_embedding_model(self):
        from eigendialectos.embeddings.subword.fasttext_model import FastTextModel

        assert issubclass(FastTextModel, EmbeddingModel)

    def test_train_and_encode(self):
        from eigendialectos.embeddings.subword.fasttext_model import FastTextModel

        model = FastTextModel(
            dialect_code=DialectCode.ES_PEN,
            vector_size=32,
            min_count=1,
            epochs=2,
        )
        model.train(_make_corpus())

        assert model.is_trained
        assert model.level() == "subword"
        assert model.vocab_size() > 0
        assert model.embedding_dim() == 32

    def test_encode_shape(self):
        from eigendialectos.embeddings.subword.fasttext_model import FastTextModel

        model = FastTextModel(vector_size=32, min_count=1, epochs=2)
        model.train(_make_corpus())

        result = model.encode(["gato", "casa"])
        assert result.shape == (2, 32)
        assert result.dtype == np.float64

    def test_encode_words(self):
        from eigendialectos.embeddings.subword.fasttext_model import FastTextModel

        model = FastTextModel(vector_size=32, min_count=1, epochs=2)
        model.train(_make_corpus())

        em = model.encode_words(["gato", "perro"])
        assert em.data.shape == (2, 32)

    def test_subword_oov(self):
        """FastText should handle OOV words via subword information."""
        from eigendialectos.embeddings.subword.fasttext_model import FastTextModel

        model = FastTextModel(vector_size=32, min_count=1, epochs=2)
        model.train(_make_corpus())

        # OOV word should still get a vector via character n-grams
        result = model.encode(["desconocidisimo"])
        assert result.shape == (1, 32)
        assert not np.allclose(result, 0)

    def test_save_and_load(self, tmp_path):
        from eigendialectos.embeddings.subword.fasttext_model import FastTextModel

        model = FastTextModel(vector_size=32, min_count=1, epochs=2)
        model.train(_make_corpus())
        original = model.encode(["gato"])

        model.save(tmp_path / "ft_model.bin")

        model2 = FastTextModel(vector_size=32)
        model2.load(tmp_path / "ft_model.bin")
        loaded = model2.encode(["gato"])
        np.testing.assert_allclose(original, loaded, atol=1e-6)

    def test_untrained_raises(self):
        from eigendialectos.embeddings.subword.fasttext_model import FastTextModel

        model = FastTextModel(vector_size=32)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.encode(["hola"])
