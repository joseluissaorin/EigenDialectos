"""Word2Vec embedding model backed by gensim.

Supports both CBOW (sg=0) and Skip-gram (sg=1) training modes.
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
    from gensim.models import Word2Vec as _GensimWord2Vec

    _HAS_GENSIM = True
except ImportError:
    _HAS_GENSIM = False


def _require_gensim() -> None:
    if not _HAS_GENSIM:
        raise ImportError(
            "gensim is required for Word2VecModel. "
            "Install it with:  pip install gensim>=4.3  "
            "or:  pip install eigendialectos[word]"
        )


class Word2VecModel(EmbeddingModel):
    """Word-level embedding model using gensim Word2Vec.

    Parameters
    ----------
    dialect_code:
        Target dialect variety.
    vector_size:
        Embedding dimensionality (default from ``EMBEDDING_DIMS``).
    min_count:
        Minimum token frequency.
    window:
        Context window size.
    epochs:
        Training epochs.
    sg:
        Training algorithm.  0 = CBOW, 1 = Skip-gram.
    ns_exponent:
        Negative-sampling exponent.
    negative:
        Number of negative samples.
    """

    def __init__(
        self,
        dialect_code: DialectCode | None = None,
        vector_size: int | None = None,
        min_count: int = 2,
        window: int = 5,
        epochs: int = 10,
        sg: int = 0,
        ns_exponent: float = 0.75,
        negative: int = 5,
        **kwargs: Any,
    ) -> None:
        _require_gensim()
        super().__init__(dialect_code=dialect_code, **kwargs)
        self._vector_size = vector_size or EMBEDDING_DIMS["word"]
        self._min_count = min_count
        self._window = window
        self._epochs = epochs
        self._sg = sg
        self._ns_exponent = ns_exponent
        self._negative = negative
        self._model: _GensimWord2Vec | None = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # EmbeddingModel interface
    # ------------------------------------------------------------------

    def train(self, corpus: CorpusSlice, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._dialect_code = corpus.dialect_code

        sentences = [sample.text.split() for sample in corpus.samples]

        vector_size = cfg.get("vector_size", self._vector_size)
        min_count = cfg.get("min_count", self._min_count)
        window = cfg.get("window", self._window)
        epochs = cfg.get("epochs", self._epochs)
        sg = cfg.get("sg", self._sg)
        negative = cfg.get("negative", self._negative)
        ns_exponent = cfg.get("ns_exponent", self._ns_exponent)

        algo_name = "Skip-gram" if sg else "CBOW"
        logger.info(
            "Training Word2Vec %s (dim=%d, epochs=%d) on %d sentences for %s",
            algo_name, vector_size, epochs, len(sentences),
            corpus.dialect_code.value,
        )

        self._model = _GensimWord2Vec(
            sentences=sentences,
            vector_size=vector_size,
            min_count=min_count,
            window=window,
            epochs=epochs,
            sg=sg,
            negative=negative,
            ns_exponent=ns_exponent,
            workers=cfg.get("workers", 1),
            seed=cfg.get("seed", 42),
        )
        self._is_trained = True

    def encode(self, texts: list[str]) -> np.ndarray:
        self._assert_trained()
        assert self._model is not None
        vectors = []
        for text in texts:
            words = text.split()
            word_vecs = []
            for w in words:
                if w in self._model.wv:
                    word_vecs.append(self._model.wv[w])
            if word_vecs:
                vectors.append(np.mean(word_vecs, axis=0))
            else:
                vectors.append(np.zeros(self._model.wv.vector_size, dtype=np.float64))
        return np.vstack(vectors).astype(np.float64)

    def encode_words(self, words: list[str]) -> EmbeddingMatrix:
        self._assert_trained()
        assert self._model is not None
        present = [w for w in words if w in self._model.wv]
        if not present:
            dim = self._model.wv.vector_size
            return EmbeddingMatrix(
                data=np.zeros((0, dim), dtype=np.float64),
                vocab=[],
                dialect_code=self._dialect_code or DialectCode.ES_PEN,
            )
        vecs = np.vstack([self._model.wv[w] for w in present]).astype(np.float64)
        return EmbeddingMatrix(
            data=vecs,
            vocab=present,
            dialect_code=self._dialect_code or DialectCode.ES_PEN,
        )

    def save(self, path: Path) -> None:
        self._assert_trained()
        assert self._model is not None
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(p))
        self._save_meta(p)
        logger.info("Word2Vec model saved to %s", p)

    def load(self, path: Path) -> None:
        _require_gensim()
        p = Path(path)
        self._model = _GensimWord2Vec.load(str(p))
        meta = self._load_meta(p)
        if meta.get("dialect_code"):
            self._dialect_code = DialectCode(meta["dialect_code"])
        self._is_trained = True
        logger.info("Word2Vec model loaded from %s", p)

    def vocab_size(self) -> int:
        self._assert_trained()
        assert self._model is not None
        return len(self._model.wv)

    def embedding_dim(self) -> int:
        if self._model is not None:
            return int(self._model.wv.vector_size)
        return self._vector_size

    def level(self) -> str:
        return "word"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def algorithm(self) -> str:
        """Return ``'skipgram'`` or ``'cbow'`` depending on configuration."""
        return "skipgram" if self._sg else "cbow"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_trained(self) -> None:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model has not been trained yet.  Call train() first.")
