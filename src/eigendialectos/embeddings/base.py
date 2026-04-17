"""Abstract base class for all embedding models in EigenDialectos."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from eigendialectos.constants import DialectCode
from eigendialectos.types import CorpusSlice, EmbeddingMatrix


class EmbeddingModel(ABC):
    """Abstract embedding model interface.

    All embedding backends (subword, word, sentence) implement this
    contract so that downstream alignment and spectral analysis
    remain agnostic to the concrete representation.
    """

    def __init__(self, dialect_code: DialectCode | None = None, **kwargs: Any) -> None:
        self._dialect_code = dialect_code
        self._is_trained = False
        self._config: dict[str, Any] = kwargs

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, corpus: CorpusSlice, config: dict[str, Any] | None = None) -> None:
        """Train the model on a dialect corpus slice.

        Parameters
        ----------
        corpus:
            A :class:`CorpusSlice` containing samples for one variety.
        config:
            Optional training hyperparameters.
        """

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into a 2-D embedding array.

        Parameters
        ----------
        texts:
            Raw text strings.  Interpretation depends on the model level:
            subword/word models treat each string as a single token or
            short phrase; sentence models treat each string as a full
            sentence.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), embedding_dim)``.
        """

    @abstractmethod
    def encode_words(self, words: list[str]) -> EmbeddingMatrix:
        """Encode a vocabulary list and return an ``EmbeddingMatrix``.

        This is the primary entry point used by the alignment layer.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to *path* (directory or file)."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a previously saved model from *path*."""

    @abstractmethod
    def vocab_size(self) -> int:
        """Return the number of tokens / words the model knows."""

    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of produced embeddings."""

    @abstractmethod
    def level(self) -> str:
        """Return the granularity level: ``'subword'``, ``'word'``, or ``'sentence'``."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def dialect_code(self) -> DialectCode | None:
        return self._dialect_code

    @dialect_code.setter
    def dialect_code(self, value: DialectCode) -> None:
        self._dialect_code = value

    def _save_meta(self, path: Path) -> None:
        """Write model metadata alongside saved artifacts."""
        meta = {
            "dialect_code": self._dialect_code.value if self._dialect_code else None,
            "level": self.level(),
            "embedding_dim": self.embedding_dim() if self._is_trained else None,
            "config": {k: v for k, v in self._config.items() if _is_json_serialisable(v)},
        }
        meta_path = Path(path)
        if meta_path.is_dir():
            meta_path = meta_path / "meta.json"
        else:
            meta_path = meta_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _load_meta(self, path: Path) -> dict[str, Any]:
        meta_path = Path(path)
        if meta_path.is_dir():
            meta_path = meta_path / "meta.json"
        else:
            meta_path = meta_path.with_suffix(".meta.json")
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {}

    def __repr__(self) -> str:
        cls = type(self).__name__
        dialect = self._dialect_code.value if self._dialect_code else "?"
        return f"<{cls} dialect={dialect} trained={self._is_trained}>"


def _is_json_serialisable(obj: Any) -> bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False
