"""Shared test fixtures for EigenDialectos."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode, FeatureCategory
from eigendialectos.types import (
    CorpusSlice,
    DialectSample,
    DialectalSpectrum,
    EigenDecomposition,
    EmbeddingMatrix,
    TransformationMatrix,
)


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def all_dialect_codes():
    """All 8 dialect codes."""
    return list(DialectCode)


@pytest.fixture
def tiny_corpus():
    """Minimal corpus with 5 samples per dialect for testing."""
    samples = {}
    texts = {
        DialectCode.ES_PEN: [
            "Vamos a coger el autobús para ir al centro.",
            "¿Habéis visto la película que echan esta noche?",
            "Me he comprado un ordenador nuevo, es genial.",
            "Tío, no me mola nada tener que madrugar.",
            "Quedamos a las ocho en la plaza, ¿vale?",
        ],
        DialectCode.ES_AND: [
            "Vamoh a coger er autobú pa ir ar centro.",
            "¿Habéi vihto la película que echan ehta noche?",
            "Me he comprao un ordenadó nuevo, eh genial.",
            "Quillo, no me mola bah tené que madrugá.",
            "Quedamoh a lah ocho en la plasa, ¿vale?",
        ],
        DialectCode.ES_CAN: [
            "Vamos a coger la guagua para ir al centro.",
            "¿Han visto la película que echan esta noche?",
            "Me he comprado un ordenador nuevo, es brutal.",
            "Chacho, no me gusta nada tener que madrugar.",
            "Quedamos a las ocho en la plaza, ¿verdad?",
        ],
        DialectCode.ES_RIO: [
            "Vamos a tomar el colectivo para ir al centro.",
            "¿Vieron la película que dan esta noche?",
            "Me compré una computadora nueva, es genial.",
            "Che, no me copa nada tener que madrugar.",
            "Nos vemos a las ocho en la plaza, ¿dale?",
        ],
        DialectCode.ES_MEX: [
            "Vamos a tomar el camión para ir al centro.",
            "¿Ya vieron la película que pasan esta noche?",
            "Me compré una computadora nueva, está padrísima.",
            "Güey, no me late nada tener que madrugar.",
            "Nos vemos a las ocho en la plaza, ¿va?",
        ],
        DialectCode.ES_CAR: [
            "Vamos a coger la guagua pa' ir al centro.",
            "¿Ustedes vieron la película que dan esta noche?",
            "Me compré una computadora nueva, está brutal.",
            "Mijo, no me gusta na' tener que madrugar.",
            "Nos vemos a las ocho en la plaza, ¿oíste?",
        ],
        DialectCode.ES_CHI: [
            "Vamos a tomar la micro para ir al centro.",
            "¿Cacharon la película que dan esta noche?",
            "Me compré un computador nuevo, está la raja.",
            "Hueón, no me tinca nada tener que madrugar.",
            "Nos juntamos a las ocho en la plaza, ¿cachai?",
        ],
        DialectCode.ES_AND_BO: [
            "Vamos a tomar el bus para ir al centro.",
            "¿Han visto la película que pasan esta noche?",
            "Me he comprado una computadora nueva, es bonita.",
            "Oye, no me gusta nada tener que madrugar.",
            "Nos vemos a las ocho en la plaza, ¿ya?",
        ],
    }

    for code, code_texts in texts.items():
        samples[code] = CorpusSlice(
            samples=[
                DialectSample(
                    text=t,
                    dialect_code=code,
                    source_id="test_fixture",
                    confidence=1.0,
                    metadata={"index": i},
                )
                for i, t in enumerate(code_texts)
            ],
            dialect_code=code,
        )
    return samples


@pytest.fixture
def random_embeddings(rng):
    """Random embedding matrices for testing (dim=50, vocab=100)."""
    dim = 50
    vocab_size = 100
    vocab = [f"word_{i}" for i in range(vocab_size)]

    embeddings = {}
    for code in DialectCode:
        data = rng.standard_normal((dim, vocab_size)).astype(np.float32)
        embeddings[code] = EmbeddingMatrix(
            data=data, vocab=vocab, dialect_code=code
        )
    return embeddings


@pytest.fixture
def known_transform():
    """A hand-crafted transformation with known eigenstructure.

    3x3 matrix with eigenvalues [3, 2, 1] and known eigenvectors.
    """
    eigenvalues = np.array([3.0, 2.0, 1.0])
    P = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    P_inv = np.linalg.inv(P)
    W = P @ np.diag(eigenvalues) @ P_inv

    return TransformationMatrix(
        data=W,
        source_dialect=DialectCode.ES_PEN,
        target_dialect=DialectCode.ES_RIO,
        regularization=0.0,
    )


@pytest.fixture
def identity_transform():
    """Identity transformation (no dialectal change)."""
    dim = 50
    return TransformationMatrix(
        data=np.eye(dim),
        source_dialect=DialectCode.ES_PEN,
        target_dialect=DialectCode.ES_PEN,
        regularization=0.0,
    )


@pytest.fixture
def known_eigendecomposition():
    """Known eigendecomposition for testing."""
    eigenvalues = np.array([3.0, 2.0, 1.0])
    P = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    P_inv = np.linalg.inv(P)

    return EigenDecomposition(
        eigenvalues=eigenvalues,
        eigenvectors=P,
        eigenvectors_inv=P_inv,
        dialect_code=DialectCode.ES_RIO,
    )


@pytest.fixture
def sample_spectrum():
    """Sample dialectal spectrum for testing."""
    eigenvalues = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
    total = np.sum(np.abs(eigenvalues))
    probs = np.abs(eigenvalues) / total
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    return DialectalSpectrum(
        eigenvalues_sorted=eigenvalues,
        entropy=entropy,
        dialect_code=DialectCode.ES_AND,
    )
