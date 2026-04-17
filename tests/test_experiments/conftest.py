"""Shared fixtures for experiment tests.

Uses small synthetic data (dim=10, vocab=50, 3 dialects) for speed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.types import (
    EmbeddingMatrix,
    TransformationMatrix,
    EigenDecomposition,
)


# The 3 dialects used in lightweight tests
TINY_DIALECTS = [DialectCode.ES_PEN, DialectCode.ES_RIO, DialectCode.ES_AND]
TINY_DIM = 10
TINY_VOCAB = 50


@pytest.fixture
def tiny_config(tmp_path: Path) -> dict:
    """Minimal experiment config for fast tests."""
    return {
        "seed": 42,
        "dim": TINY_DIM,
        "vocab_size": TINY_VOCAB,
        "n_dialects": len(TINY_DIALECTS),
        "n_sentences": 5,
        "data_dir": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "output"),
        "regularization": 0.01,
        "max_holdout_pairs": 3,
    }


@pytest.fixture
def tiny_rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def tiny_embeddings(tiny_rng: np.random.Generator) -> dict[DialectCode, EmbeddingMatrix]:
    vocab = [f"w{i}" for i in range(TINY_VOCAB)]
    base = tiny_rng.standard_normal((TINY_DIM, TINY_VOCAB)).astype(np.float64)
    embs: dict[DialectCode, EmbeddingMatrix] = {}
    for code in TINY_DIALECTS:
        noise = (
            np.zeros_like(base) if code == DialectCode.ES_PEN
            else tiny_rng.standard_normal(base.shape) * 0.1
        )
        embs[code] = EmbeddingMatrix(data=base + noise, vocab=vocab, dialect_code=code)
    return embs


@pytest.fixture
def tiny_transforms(tiny_embeddings: dict[DialectCode, EmbeddingMatrix]) -> dict[DialectCode, TransformationMatrix]:
    from eigendialectos.spectral.transformation import compute_transformation_matrix
    ref = tiny_embeddings[DialectCode.ES_PEN]
    return {
        code: compute_transformation_matrix(ref, emb, method="lstsq", regularization=0.01)
        for code, emb in tiny_embeddings.items()
    }


@pytest.fixture
def tiny_eigendecomps(tiny_transforms: dict[DialectCode, TransformationMatrix]) -> dict[DialectCode, EigenDecomposition]:
    from eigendialectos.spectral.eigendecomposition import eigendecompose
    return {code: eigendecompose(W) for code, W in tiny_transforms.items()}
