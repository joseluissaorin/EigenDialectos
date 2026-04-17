"""BPE embedding model using HuggingFace tokenizers.

Trains a byte-pair-encoding tokenizer per dialect variety and learns
token embeddings via a lightweight embedding layer trained with a
simple skip-gram-like objective on the BPE token sequences.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from eigendialectos.constants import EMBEDDING_DIMS, DialectCode
from eigendialectos.embeddings.base import EmbeddingModel
from eigendialectos.types import CorpusSlice, EmbeddingMatrix

logger = logging.getLogger(__name__)

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    _HAS_TOKENIZERS = True
except ImportError:
    _HAS_TOKENIZERS = False


def _require_tokenizers() -> None:
    if not _HAS_TOKENIZERS:
        raise ImportError(
            "The 'tokenizers' library is required for BPEModel. "
            "Install with:  pip install tokenizers>=0.15  "
            "or:  pip install eigendialectos[subword]"
        )


class BPEModel(EmbeddingModel):
    """BPE subword embedding model.

    Workflow:
    1. Train a BPE tokenizer on the dialect corpus.
    2. Encode all corpus sentences into BPE token-id sequences.
    3. Learn dense embeddings for each BPE token using a lightweight
       co-occurrence / skip-gram approximation (SVD of a token
       co-occurrence matrix).

    Parameters
    ----------
    dialect_code:
        Target dialect variety.
    vocab_size:
        BPE vocabulary size.
    vector_size:
        Embedding dimensionality.
    window:
        Co-occurrence context window.
    min_frequency:
        Minimum pair frequency for BPE merges.
    """

    def __init__(
        self,
        dialect_code: DialectCode | None = None,
        vocab_size: int = 8000,
        vector_size: int | None = None,
        window: int = 5,
        min_frequency: int = 2,
        **kwargs: Any,
    ) -> None:
        _require_tokenizers()
        super().__init__(dialect_code=dialect_code, **kwargs)
        self._vocab_size_target = vocab_size
        self._vector_size = vector_size or EMBEDDING_DIMS["subword"]
        self._window = window
        self._min_frequency = min_frequency
        self._tokenizer: Tokenizer | None = None  # type: ignore[assignment]
        self._embeddings: np.ndarray | None = None
        self._id2token: dict[int, str] = {}
        self._token2id: dict[str, int] = {}

    # ------------------------------------------------------------------
    # EmbeddingModel interface
    # ------------------------------------------------------------------

    def train(self, corpus: CorpusSlice, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._dialect_code = corpus.dialect_code

        vocab_size = cfg.get("vocab_size", self._vocab_size_target)
        vector_size = cfg.get("vector_size", self._vector_size)
        window = cfg.get("window", self._window)
        min_frequency = cfg.get("min_frequency", self._min_frequency)

        texts = [s.text for s in corpus.samples]

        # --- Step 1: train BPE tokenizer ---
        logger.info(
            "Training BPE tokenizer (vocab=%d) on %d texts for %s",
            vocab_size, len(texts), corpus.dialect_code.value,
        )
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["[UNK]", "[PAD]"],
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        self._tokenizer = tokenizer

        actual_vocab = tokenizer.get_vocab()
        self._token2id = dict(actual_vocab)
        self._id2token = {v: k for k, v in actual_vocab.items()}

        # --- Step 2: encode corpus into BPE token sequences ---
        token_sequences: list[list[int]] = []
        for text in texts:
            encoding = tokenizer.encode(text)
            token_sequences.append(encoding.ids)

        # --- Step 3: build co-occurrence matrix and compute SVD ---
        n_tokens = len(actual_vocab)
        cooccurrence = np.zeros((n_tokens, n_tokens), dtype=np.float64)

        for seq in token_sequences:
            for i, token_id in enumerate(seq):
                start = max(0, i - window)
                end = min(len(seq), i + window + 1)
                for j in range(start, end):
                    if j != i:
                        distance = abs(i - j)
                        weight = 1.0 / distance
                        cooccurrence[token_id, seq[j]] += weight

        # Symmetrise and apply PPMI
        cooccurrence = cooccurrence + cooccurrence.T
        row_sums = cooccurrence.sum(axis=1, keepdims=True)
        col_sums = cooccurrence.sum(axis=0, keepdims=True)
        total = cooccurrence.sum()

        if total > 0:
            # Pointwise mutual information
            with np.errstate(divide="ignore", invalid="ignore"):
                pmi = np.log2(
                    (cooccurrence * total) / (row_sums * col_sums + 1e-16) + 1e-16
                )
            pmi = np.maximum(pmi, 0)  # positive PMI
        else:
            pmi = cooccurrence

        # Truncated SVD
        rank = min(vector_size, n_tokens - 1, pmi.shape[0] - 1)
        if rank <= 0:
            self._embeddings = np.random.randn(n_tokens, vector_size).astype(np.float64) * 0.01
        else:
            try:
                from scipy.sparse.linalg import svds

                # svds requires rank < min(matrix_shape)
                safe_rank = min(rank, min(pmi.shape) - 1)
                if safe_rank > 0:
                    U, S, _ = svds(pmi, k=safe_rank)
                    # Weight by sqrt(S) as in GloVe-style embeddings
                    self._embeddings = U * np.sqrt(S)[np.newaxis, :]
                    # Pad to target dimensionality if needed
                    if self._embeddings.shape[1] < vector_size:
                        pad = np.zeros(
                            (n_tokens, vector_size - self._embeddings.shape[1]),
                            dtype=np.float64,
                        )
                        self._embeddings = np.hstack([self._embeddings, pad])
                else:
                    self._embeddings = np.random.randn(n_tokens, vector_size).astype(np.float64) * 0.01
            except ImportError:
                # Fallback: full SVD via numpy
                U, S, _ = np.linalg.svd(pmi, full_matrices=False)
                self._embeddings = U[:, :rank] * np.sqrt(S[:rank])[np.newaxis, :]
                if self._embeddings.shape[1] < vector_size:
                    pad = np.zeros(
                        (n_tokens, vector_size - self._embeddings.shape[1]),
                        dtype=np.float64,
                    )
                    self._embeddings = np.hstack([self._embeddings, pad])

        self._is_trained = True
        logger.info(
            "BPE model trained: %d tokens, dim %d",
            n_tokens, self._embeddings.shape[1],
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        self._assert_trained()
        assert self._tokenizer is not None
        assert self._embeddings is not None

        vectors = []
        for text in texts:
            encoding = self._tokenizer.encode(text)
            ids = encoding.ids
            if len(ids) == 0:
                vectors.append(np.zeros(self._embeddings.shape[1], dtype=np.float64))
            else:
                token_vecs = self._embeddings[ids]
                vectors.append(token_vecs.mean(axis=0))
        return np.vstack(vectors).astype(np.float64)

    def encode_words(self, words: list[str]) -> EmbeddingMatrix:
        vecs = self.encode(words)
        return EmbeddingMatrix(
            data=vecs,
            vocab=list(words),
            dialect_code=self._dialect_code or DialectCode.ES_PEN,
        )

    def save(self, path: Path) -> None:
        self._assert_trained()
        assert self._tokenizer is not None
        assert self._embeddings is not None

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        self._tokenizer.save(str(p / "tokenizer.json"))
        np.save(str(p / "embeddings.npy"), self._embeddings)

        vocab_data = {
            "token2id": self._token2id,
            "id2token": {str(k): v for k, v in self._id2token.items()},
        }
        (p / "vocab.json").write_text(json.dumps(vocab_data, ensure_ascii=False), encoding="utf-8")
        self._save_meta(p)
        logger.info("BPE model saved to %s", p)

    def load(self, path: Path) -> None:
        _require_tokenizers()
        p = Path(path)

        self._tokenizer = Tokenizer.from_file(str(p / "tokenizer.json"))
        self._embeddings = np.load(str(p / "embeddings.npy"))

        vocab_data = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
        self._token2id = vocab_data["token2id"]
        self._id2token = {int(k): v for k, v in vocab_data["id2token"].items()}

        meta = self._load_meta(p)
        if meta.get("dialect_code"):
            self._dialect_code = DialectCode(meta["dialect_code"])

        self._is_trained = True
        logger.info("BPE model loaded from %s", p)

    def vocab_size(self) -> int:
        self._assert_trained()
        return len(self._token2id)

    def embedding_dim(self) -> int:
        if self._embeddings is not None:
            return int(self._embeddings.shape[1])
        return self._vector_size

    def level(self) -> str:
        return "subword"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_trained(self) -> None:
        if not self._is_trained or self._tokenizer is None:
            raise RuntimeError("Model has not been trained yet.  Call train() first.")
