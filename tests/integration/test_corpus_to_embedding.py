"""Integration test: corpus → embedding pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode


def test_synthetic_corpus_to_embeddings(tiny_corpus):
    """Test that synthetic corpus data can feed into embedding pipeline."""
    from eigendialectos.corpus.synthetic.fixtures import get_fixtures

    fixtures = get_fixtures()
    assert len(fixtures) == len(DialectCode)

    for code, samples in fixtures.items():
        # get_fixtures returns dict[DialectCode, list[DialectSample]]
        sample_list = samples if isinstance(samples, list) else samples.samples
        assert len(sample_list) > 0
        for sample in sample_list:
            assert len(sample.text) > 0
            assert sample.dialect_code == code


def test_corpus_preprocessing_pipeline():
    """Test preprocessing on raw text."""
    from eigendialectos.corpus.preprocessing.noise import clean_text

    raw = "  Vamos a coger la guagua 🚌 https://t.co/abc @usuario jajajajajaja  "
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "@" not in cleaned
    assert len(cleaned) < len(raw)
    assert len(cleaned.strip()) == len(cleaned)


def test_embedding_matrix_from_random_data(random_embeddings):
    """Test that embedding matrices have correct structure."""
    for code, emb in random_embeddings.items():
        assert emb.data.shape[0] == 50  # dim
        assert emb.data.shape[1] == 100  # vocab
        assert len(emb.vocab) == 100
        assert emb.dialect_code == code
