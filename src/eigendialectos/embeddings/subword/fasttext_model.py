"""FastText embedding model backed by gensim.

Trains character n-gram (subword) embeddings per dialect variety.
Gracefully degrades when gensim is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from eigendialectos.constants import EMBEDDING_DIMS, DialectCode
from eigendialectos.embeddings.base import EmbeddingModel
from eigendialectos.types import CorpusSlice, EmbeddingMatrix

logger = logging.getLogger(__name__)

try:
    from gensim.models import FastText as _GensimFastText

    _HAS_GENSIM = True
except ImportError:
    _HAS_GENSIM = False


def _require_gensim() -> None:
    if not _HAS_GENSIM:
        raise ImportError(
            "gensim is required for FastTextModel. "
            "Install it with:  pip install gensim>=4.3  "
            "or:  pip install eigendialectos[subword]"
        )


class FastTextModel(EmbeddingModel):
    """Subword embedding model using gensim's FastText implementation.

    Parameters
    ----------
    dialect_code:
        Target dialect variety.
    vector_size:
        Embedding dimensionality (default from ``EMBEDDING_DIMS``).
    min_count:
        Minimum token frequency to include in vocabulary.
    window:
        Context window size.
    epochs:
        Training epochs (gensim calls this *epochs*).
    min_n / max_n:
        Character n-gram range.
    sg:
        Training algorithm. 0 = CBOW, 1 = Skip-gram.
    **kwargs:
        Additional arguments forwarded to gensim ``FastText``.
    """

    def __init__(
        self,
        dialect_code: DialectCode | None = None,
        vector_size: int | None = None,
        min_count: int = 2,
        window: int = 5,
        epochs: int = 10,
        min_n: int = 3,
        max_n: int = 6,
        sg: int = 0,
        **kwargs: Any,
    ) -> None:
        _require_gensim()
        super().__init__(dialect_code=dialect_code, **kwargs)
        self._vector_size = vector_size or EMBEDDING_DIMS["subword"]
        self._min_count = min_count
        self._window = window
        self._epochs = epochs
        self._min_n = min_n
        self._max_n = max_n
        self._sg = sg
        self._model: _GensimFastText | None = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # EmbeddingModel interface
    # ------------------------------------------------------------------

    def train(self, corpus: CorpusSlice, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._dialect_code = corpus.dialect_code

        # Tokenise simply on whitespace; downstream preprocessing is
        # expected to have normalised the text already.
        sentences = [sample.text.split() for sample in corpus.samples]

        vector_size = cfg.get("vector_size", self._vector_size)
        min_count = cfg.get("min_count", self._min_count)
        window = cfg.get("window", self._window)
        epochs = cfg.get("epochs", self._epochs)
        min_n = cfg.get("min_n", self._min_n)
        max_n = cfg.get("max_n", self._max_n)
        sg = cfg.get("sg", self._sg)

        logger.info(
            "Training FastText (dim=%d, epochs=%d) on %d sentences for %s",
            vector_size,
            epochs,
            len(sentences),
            corpus.dialect_code.value,
        )

        self._model = _GensimFastText(
            sentences=sentences,
            vector_size=vector_size,
            min_count=min_count,
            window=window,
            epochs=epochs,
            min_n=min_n,
            max_n=max_n,
            sg=sg,
            workers=cfg.get("workers", 1),
            seed=cfg.get("seed", 42),
        )
        self._is_trained = True

    def encode(self, texts: list[str]) -> np.ndarray:
        self._assert_trained()
        assert self._model is not None
        vecs = [self._model.wv.get_vector(t) for t in texts]
        return np.vstack(vecs).astype(np.float64)

    def encode_words(self, words: list[str]) -> EmbeddingMatrix:
        vecs = self.encode(words)
        return EmbeddingMatrix(
            data=vecs,
            vocab=list(words),
            dialect_code=self._dialect_code or DialectCode.ES_PEN,
        )

    def save(self, path: Path) -> None:
        self._assert_trained()
        assert self._model is not None
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(p))
        self._save_meta(p)
        logger.info("FastText model saved to %s", p)

    def load(self, path: Path) -> None:
        _require_gensim()
        p = Path(path)
        self._model = _GensimFastText.load(str(p))
        meta = self._load_meta(p)
        if meta.get("dialect_code"):
            self._dialect_code = DialectCode(meta["dialect_code"])
        self._is_trained = True
        logger.info("FastText model loaded from %s", p)

    def vocab_size(self) -> int:
        self._assert_trained()
        assert self._model is not None
        return len(self._model.wv)

    def embedding_dim(self) -> int:
        if self._model is not None:
            return int(self._model.wv.vector_size)
        return self._vector_size

    def level(self) -> str:
        return "subword"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_trained(self) -> None:
        if not self._is_trained or self._model is None:
            raise RuntimeError(
                "Model has not been trained yet.  Call train() first."
            )
