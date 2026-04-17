"""Session-scoped fixtures for eigen3 tests.

Loads real v2 output data (embeddings, W matrices, eigendecompositions)
so that tests verify against actual trained artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUTS = _PROJECT_ROOT / "outputs"
_EMB_DIR = _OUTPUTS / "embeddings"
_DCL_DIR = _EMB_DIR / "dcl_subword"
_V2_DIR = _OUTPUTS / "v2_real"
_DATA_DIR = _PROJECT_ROOT / "data" / "processed"

ALL_VARIETIES = [
    "ES_PEN", "ES_AND", "ES_CAN", "ES_RIO",
    "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO",
]


# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    return _DATA_DIR


@pytest.fixture(scope="session")
def sample_corpus() -> dict[str, list[str]]:
    """Small sample corpus for unit tests (first 50 docs per variety)."""
    corpus: dict[str, list[str]] = {}
    for variety in ALL_VARIETIES:
        path = _DATA_DIR / f"{variety}.jsonl"
        docs: list[str] = []
        if path.exists():
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if i >= 50:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    text = doc.get("text", "").strip()
                    if text:
                        docs.append(text)
                except json.JSONDecodeError:
                    continue
        corpus[variety] = docs
    return corpus


@pytest.fixture(scope="session")
def full_corpus() -> dict[str, list[str]]:
    """Full corpus for integration tests."""
    corpus: dict[str, list[str]] = {}
    for variety in ALL_VARIETIES:
        path = _DATA_DIR / f"{variety}.jsonl"
        docs: list[str] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    text = doc.get("text", "").strip()
                    if text:
                        docs.append(text)
                except json.JSONDecodeError:
                    continue
        corpus[variety] = docs
    return corpus


# ---------------------------------------------------------------------------
# Vocabulary fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def vocab() -> list[str]:
    """Word-level vocabulary (43k+ words)."""
    path = _EMB_DIR / "vocab.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Embedding fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def word_embeddings_dict() -> dict[str, np.ndarray]:
    """Word-level embeddings per variety: (dim, vocab_size) -> transposed to (vocab_size, dim)."""
    result = {}
    for variety in ALL_VARIETIES:
        path = _EMB_DIR / f"{variety}.npy"
        if path.exists():
            emb = np.load(str(path))
            # Stored as (dim, vocab_size), transpose to (vocab_size, dim)
            if emb.shape[0] < emb.shape[1]:
                emb = emb.T
            result[variety] = emb.astype(np.float64)
    return result


@pytest.fixture(scope="session")
def bpe_embeddings_dict() -> dict[str, np.ndarray]:
    """BPE subword embeddings per variety: (8000, 100)."""
    result = {}
    for variety in ALL_VARIETIES:
        path = _DCL_DIR / f"{variety}_subword.npy"
        if path.exists():
            result[variety] = np.load(str(path)).astype(np.float64)
    return result


@pytest.fixture(scope="session")
def embedding_dim() -> int:
    return 100


@pytest.fixture(scope="session")
def embedding_meta() -> dict:
    path = _DCL_DIR / "meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# W matrix fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def W_dict() -> dict[str, np.ndarray]:
    """Transformation matrices W per variety (relative to PEN)."""
    path = _V2_DIR / "W_matrices.npz"
    data = np.load(str(path))
    return {k: data[k].astype(np.float64) for k in data.files}


@pytest.fixture(scope="session")
def W_pen(W_dict) -> np.ndarray:
    return W_dict["ES_PEN"]


# ---------------------------------------------------------------------------
# Eigendecomposition fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def decomps_dict(W_dict) -> dict[str, "EigenDecomp"]:
    """Eigendecompositions for all 8 W matrices."""
    from eigen3.decomposition import eigendecompose
    return {v: eigendecompose(W, variety=v) for v, W in W_dict.items()}


@pytest.fixture(scope="session")
def spectra_dict(decomps_dict) -> dict[str, "EigenSpectrum"]:
    """Eigenspectra for all 8 varieties."""
    from eigen3.decomposition import eigenspectrum
    return {v: eigenspectrum(d.eigenvalues) for v, d in decomps_dict.items()}


@pytest.fixture(scope="session")
def eigenvalues_all() -> dict[str, list[float]]:
    """Pre-computed eigenvalues from v2 output."""
    path = _V2_DIR / "eigenvalues_all.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tokenizer fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tokenizer_model_path() -> Path:
    return _EMB_DIR / "tokenizer" / "shared_bpe.model"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
