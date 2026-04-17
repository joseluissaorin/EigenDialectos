"""Embedding training, encoding, and cross-variety alignment layer.

Public API
----------
- :class:`EmbeddingModel` -- abstract base for all backends
- :class:`FastTextModel`, :class:`Word2VecModel`, :class:`BETOModel` -- concrete backends
- :func:`register_model`, :func:`get_model`, :func:`list_available` -- registry
- :class:`CrossVarietyAligner` -- high-level alignment orchestrator
- :class:`ProcrustesAligner`, :class:`VecMapAligner`, :class:`MUSEAligner` -- alignment algorithms
"""

from eigendialectos.embeddings.alignment import CrossVarietyAligner
from eigendialectos.embeddings.base import EmbeddingModel
from eigendialectos.embeddings.contrastive.muse import MUSEAligner
from eigendialectos.embeddings.contrastive.procrustes import ProcrustesAligner
from eigendialectos.embeddings.contrastive.vecmap import VecMapAligner
from eigendialectos.embeddings.registry import (
    clear_registry,
    get_model,
    list_available,
    register_model,
)
from eigendialectos.embeddings.sentence.beto_model import BETOModel
from eigendialectos.embeddings.subword.fasttext_model import FastTextModel
from eigendialectos.embeddings.word.word2vec_model import Word2VecModel

__all__ = [
    "BETOModel",
    "CrossVarietyAligner",
    "EmbeddingModel",
    "FastTextModel",
    "MUSEAligner",
    "ProcrustesAligner",
    "VecMapAligner",
    "Word2VecModel",
    "clear_registry",
    "get_model",
    "list_available",
    "register_model",
]
