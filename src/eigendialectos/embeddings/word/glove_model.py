"""GloVe embedding model with a pure-Python SVD-based fallback.

Since the canonical GloVe implementation requires C compilation, this
module provides a self-contained Python approximation that builds a
weighted co-occurrence matrix and factorises it via truncated SVD,
producing embeddings with properties closely analogous to GloVe.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from eigendialectos.constants import EMBEDDING_DIMS, DialectCode
from eigendialectos.embeddings.base import EmbeddingModel
from eigendialectos.types import CorpusSlice, EmbeddingMatrix

logger = logging.getLogger(__name__)


class GloVeModel(EmbeddingModel):
    """GloVe-style word embedding model (pure-Python SVD approximation).

    The training procedure:
    1. Build a vocabulary from the corpus.
    2. Construct a weighted co-occurrence matrix ``X`` with harmonic
       distance weighting within a context window.
    3. Apply the GloVe weighting function ``f(X_ij)`` and take ``log(1 + X)``.
    4. Factorise via truncated SVD to obtain word vectors.

    Parameters
    ----------
    dialect_code:
        Target dialect variety.
    vector_size:
        Embedding dimensionality.
    window:
        Co-occurrence context window.
    min_count:
        Minimum token count to include in vocabulary.
    x_max:
        GloVe clipping parameter for the weighting function.
    alpha:
        GloVe weighting exponent.
    """

    def __init__(
        self,
        dialect_code: DialectCode | None = None,
        vector_size: int | None = None,
        window: int = 10,
        min_count: int = 2,
        x_max: float = 100.0,
        alpha: float = 0.75,
        **kwargs: Any,
    ) -> None:
        super().__init__(dialect_code=dialect_code, **kwargs)
        self._vector_size = vector_size or EMBEDDING_DIMS["word"]
        self._window = window
        self._min_count = min_count
        self._x_max = x_max
        self._alpha = alpha
        self._embeddings: np.ndarray | None = None
        self._token2id: dict[str, int] = {}
        self._id2token: dict[int, str] = {}

    # ------------------------------------------------------------------
    # EmbeddingModel interface
    # ------------------------------------------------------------------

    def train(self, corpus: CorpusSlice, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._dialect_code = corpus.dialect_code

        vector_size = cfg.get("vector_size", self._vector_size)
        window = cfg.get("window", self._window)
        min_count = cfg.get("min_count", self._min_count)
        x_max = cfg.get("x_max", self._x_max)
        alpha = cfg.get("alpha", self._alpha)

        sentences = [sample.text.split() for sample in corpus.samples]

        # --- Step 1: build vocabulary ---
        counter: Counter[str] = Counter()
        for sent in sentences:
            counter.update(sent)

        vocab_tokens = [tok for tok, cnt in counter.most_common() if cnt >= min_count]
        self._token2id = {tok: i for i, tok in enumerate(vocab_tokens)}
        self._id2token = {i: tok for tok, i in self._token2id.items()}
        n = len(vocab_tokens)

        logger.info(
            "Training GloVe-SVD (vocab=%d, dim=%d, window=%d) for %s",
            n, vector_size, window, corpus.dialect_code.value,
        )

        if n == 0:
            self._embeddings = np.zeros((0, vector_size), dtype=np.float64)
            self._is_trained = True
            return

        # --- Step 2: build co-occurrence matrix ---
        cooc = np.zeros((n, n), dtype=np.float64)
        for sent in sentences:
            ids = [self._token2id[w] for w in sent if w in self._token2id]
            for i, wid in enumerate(ids):
                start = max(0, i - window)
                end = min(len(ids), i + window + 1)
                for j in range(start, end):
                    if j != i:
                        dist = abs(i - j)
                        cooc[wid, ids[j]] += 1.0 / dist

        # Symmetrise
        cooc = cooc + cooc.T

        # --- Step 3: GloVe weighting and log transform ---
        weights = np.where(
            cooc < x_max,
            (cooc / x_max) ** alpha,
            1.0,
        )
        log_cooc = np.log1p(cooc)  # log(1 + X) to handle zeros
        weighted = weights * log_cooc

        # --- Step 4: truncated SVD ---
        rank = min(vector_size, n - 1)
        if rank <= 0:
            self._embeddings = np.random.randn(n, vector_size).astype(np.float64) * 0.01
        else:
            try:
                from scipy.sparse.linalg import svds

                safe_rank = min(rank, min(weighted.shape) - 1)
                if safe_rank > 0:
                    U, S, Vt = svds(weighted, k=safe_rank)
                    # GloVe-style: use W + W_tilde, approximated as U*sqrt(S) + V*sqrt(S)
                    sqrt_S = np.sqrt(S)[np.newaxis, :]
                    self._embeddings = (U * sqrt_S + Vt.T * sqrt_S) / 2.0
                else:
                    self._embeddings = np.random.randn(n, vector_size).astype(np.float64) * 0.01
            except ImportError:
                U, S, Vt = np.linalg.svd(weighted, full_matrices=False)
                sqrt_S = np.sqrt(S[:rank])[np.newaxis, :]
                self._embeddings = (U[:, :rank] * sqrt_S + Vt[:rank, :].T * sqrt_S) / 2.0

            # Pad to target dimensionality if SVD rank < vector_size
            if self._embeddings.shape[1] < vector_size:
                pad = np.zeros(
                    (n, vector_size - self._embeddings.shape[1]),
                    dtype=np.float64,
                )
                self._embeddings = np.hstack([self._embeddings, pad])

        self._is_trained = True
        logger.info("GloVe-SVD model trained: %d words, dim %d", n, self._embeddings.shape[1])

    def encode(self, texts: list[str]) -> np.ndarray:
        self._assert_trained()
        assert self._embeddings is not None
        dim = self._embeddings.shape[1]
        vectors = []
        for text in texts:
            words = text.split()
            word_vecs = []
            for w in words:
                if w in self._token2id:
                    word_vecs.append(self._embeddings[self._token2id[w]])
            if word_vecs:
                vectors.append(np.mean(word_vecs, axis=0))
            else:
                vectors.append(np.zeros(dim, dtype=np.float64))
        return np.vstack(vectors).astype(np.float64)

    def encode_words(self, words: list[str]) -> EmbeddingMatrix:
        self._assert_trained()
        assert self._embeddings is not None
        dim = self._embeddings.shape[1]
        present = [w for w in words if w in self._token2id]
        if not present:
            return EmbeddingMatrix(
                data=np.zeros((0, dim), dtype=np.float64),
                vocab=[],
                dialect_code=self._dialect_code or DialectCode.ES_PEN,
            )
        vecs = np.vstack(
            [self._embeddings[self._token2id[w]] for w in present]
        ).astype(np.float64)
        return EmbeddingMatrix(
            data=vecs,
            vocab=present,
            dialect_code=self._dialect_code or DialectCode.ES_PEN,
        )

    def save(self, path: Path) -> None:
        self._assert_trained()
        assert self._embeddings is not None
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        np.save(str(p / "embeddings.npy"), self._embeddings)
        vocab_data = {
            "token2id": self._token2id,
            "id2token": {str(k): v for k, v in self._id2token.items()},
        }
        (p / "vocab.json").write_text(
            json.dumps(vocab_data, ensure_ascii=False), encoding="utf-8"
        )
        self._save_meta(p)
        logger.info("GloVe model saved to %s", p)

    def load(self, path: Path) -> None:
        p = Path(path)
        self._embeddings = np.load(str(p / "embeddings.npy"))
        vocab_data = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
        self._token2id = vocab_data["token2id"]
        self._id2token = {int(k): v for k, v in vocab_data["id2token"].items()}
        meta = self._load_meta(p)
        if meta.get("dialect_code"):
            self._dialect_code = DialectCode(meta["dialect_code"])
        self._is_trained = True
        logger.info("GloVe model loaded from %s", p)

    def vocab_size(self) -> int:
        self._assert_trained()
        return len(self._token2id)

    def embedding_dim(self) -> int:
        if self._embeddings is not None:
            return int(self._embeddings.shape[1])
        return self._vector_size

    def level(self) -> str:
        return "word"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_trained(self) -> None:
        if not self._is_trained or self._embeddings is None:
            raise RuntimeError("Model has not been trained yet.  Call train() first.")
