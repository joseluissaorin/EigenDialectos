"""Tests for word-level embedding models (Word2Vec, GloVe)."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.embeddings.base import EmbeddingModel
from eigendialectos.types import CorpusSlice, DialectSample

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_TEXTS = [
    "el gato negro duerme en la silla vieja",
    "la casa grande tiene un jardin bonito y verde",
    "los ninos juegan en el parque con sus amigos",
    "el perro corre por la calle principal",
    "la mujer camina hacia la tienda del barrio",
    "el hombre lee un libro interesante de ciencia",
    "las flores rojas estan en la mesa del comedor",
    "el cielo azul se ve muy claro hoy",
    "la ciudad tiene muchos edificios altos y modernos",
    "los pajaros cantan por la manana temprano",
    "el rio fluye hacia el mar profundo",
    "la montana es muy alta y fria en invierno",
    "el gato negro come su comida con hambre",
    "la casa tiene una puerta roja de madera",
    "los ninos van a la escuela cada dia",
    "el profesor ensena matematicas a los estudiantes",
    "la nina pequena juega con su muneca favorita",
    "el doctor trabaja en el hospital central",
    "los arboles verdes crecen en el bosque denso",
    "la comida esta muy rica y caliente ahora",
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
# GloVe Model tests (pure Python, no external deps)
# ---------------------------------------------------------------------------


class TestGloVeModel:
    """Tests for the GloVe SVD-approximation model."""

    def test_is_embedding_model(self):
        from eigendialectos.embeddings.word.glove_model import GloVeModel

        assert issubclass(GloVeModel, EmbeddingModel)

    def test_train_and_encode(self):
        from eigendialectos.embeddings.word.glove_model import GloVeModel

        model = GloVeModel(
            dialect_code=DialectCode.ES_PEN,
            vector_size=32,
            min_count=1,
            window=3,
        )
        model.train(_make_corpus())

        assert model.is_trained
        assert model.level() == "word"
        assert model.vocab_size() > 0
        assert model.embedding_dim() == 32

    def test_encode_returns_correct_shape(self):
        from eigendialectos.embeddings.word.glove_model import GloVeModel

        model = GloVeModel(vector_size=32, min_count=1)
        model.train(_make_corpus())

        result = model.encode(["el gato", "la casa"])
        assert result.shape == (2, 32)
        assert result.dtype == np.float64

    def test_encode_words(self):
        from eigendialectos.embeddings.word.glove_model import GloVeModel

        model = GloVeModel(vector_size=32, min_count=1)
        model.train(_make_corpus())

        em = model.encode_words(["gato", "casa", "perro"])
        assert em.data.shape[0] == 3
        assert em.data.shape[1] == 32
        assert em.vocab == ["gato", "casa", "perro"]

    def test_oov_gets_zero_vector(self):
        from eigendialectos.embeddings.word.glove_model import GloVeModel

        model = GloVeModel(vector_size=32, min_count=1)
        model.train(_make_corpus())

        # Completely OOV word in a sentence
        result = model.encode(["xyzzyx_nunca_visto"])
        assert result.shape == (1, 32)
        assert np.allclose(result, 0)

    def test_encode_words_filters_oov(self):
        from eigendialectos.embeddings.word.glove_model import GloVeModel

        model = GloVeModel(vector_size=32, min_count=1)
        model.train(_make_corpus())

        em = model.encode_words(["gato", "xyzzyx_unknown"])
        # Only "gato" should be present
        assert "gato" in em.vocab
        assert "xyzzyx_unknown" not in em.vocab

    def test_save_and_load(self, tmp_path):
        from eigendialectos.embeddings.word.glove_model import GloVeModel

        model = GloVeModel(
            dialect_code=DialectCode.ES_AND,
            vector_size=32,
            min_count=1,
        )
        model.train(_make_corpus(DialectCode.ES_AND))
        original = model.encode(["el gato"])

        save_dir = tmp_path / "glove_model"
        model.save(save_dir)

        model2 = GloVeModel()
        model2.load(save_dir)
        assert model2.is_trained
        loaded = model2.encode(["el gato"])
        np.testing.assert_allclose(original, loaded, atol=1e-6)

    def test_untrained_raises(self):
        from eigendialectos.embeddings.word.glove_model import GloVeModel

        model = GloVeModel(vector_size=32)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.encode(["hola"])

    def test_empty_corpus(self):
        from eigendialectos.embeddings.word.glove_model import GloVeModel

        model = GloVeModel(vector_size=32, min_count=100)
        # All words below min_count -> empty vocab
        corpus = CorpusSlice(
            samples=[
                DialectSample(text="a", dialect_code=DialectCode.ES_PEN,
                              source_id="x", confidence=1.0)
            ],
            dialect_code=DialectCode.ES_PEN,
        )
        model.train(corpus)
        assert model.vocab_size() == 0


# ---------------------------------------------------------------------------
# Word2Vec Model tests
# ---------------------------------------------------------------------------


class TestWord2VecModel:
    """Tests for the Word2Vec model (requires gensim)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_gensim(self):
        pytest.importorskip("gensim")

    def test_is_embedding_model(self):
        from eigendialectos.embeddings.word.word2vec_model import Word2VecModel

        assert issubclass(Word2VecModel, EmbeddingModel)

    def test_train_cbow(self):
        from eigendialectos.embeddings.word.word2vec_model import Word2VecModel

        model = Word2VecModel(vector_size=32, min_count=1, epochs=2, sg=0)
        model.train(_make_corpus())
        assert model.is_trained
        assert model.algorithm == "cbow"
        assert model.level() == "word"

    def test_train_skipgram(self):
        from eigendialectos.embeddings.word.word2vec_model import Word2VecModel

        model = Word2VecModel(vector_size=32, min_count=1, epochs=2, sg=1)
        model.train(_make_corpus())
        assert model.algorithm == "skipgram"

    def test_encode_shape(self):
        from eigendialectos.embeddings.word.word2vec_model import Word2VecModel

        model = Word2VecModel(vector_size=32, min_count=1, epochs=2)
        model.train(_make_corpus())

        result = model.encode(["el gato", "la casa"])
        assert result.shape == (2, 32)

    def test_encode_words(self):
        from eigendialectos.embeddings.word.word2vec_model import Word2VecModel

        model = Word2VecModel(vector_size=32, min_count=1, epochs=2)
        model.train(_make_corpus())

        em = model.encode_words(["gato", "casa"])
        assert em.data.shape == (2, 32)

    def test_save_and_load(self, tmp_path):
        from eigendialectos.embeddings.word.word2vec_model import Word2VecModel

        model = Word2VecModel(vector_size=32, min_count=1, epochs=2)
        model.train(_make_corpus())
        original = model.encode(["el gato"])

        model.save(tmp_path / "w2v_model.bin")

        model2 = Word2VecModel(vector_size=32)
        model2.load(tmp_path / "w2v_model.bin")
        loaded = model2.encode(["el gato"])
        np.testing.assert_allclose(original, loaded, atol=1e-6)

    def test_untrained_raises(self):
        from eigendialectos.embeddings.word.word2vec_model import Word2VecModel

        model = Word2VecModel(vector_size=32)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.encode(["hola"])
