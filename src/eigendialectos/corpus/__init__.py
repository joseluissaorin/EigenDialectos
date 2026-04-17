"""Corpus loading, generation, and preprocessing for EigenDialectos."""

from eigendialectos.corpus.base import CorpusSource
from eigendialectos.corpus.dataset import DialectDataset
from eigendialectos.corpus.registry import (
    clear_registry,
    get_source,
    list_available,
    register_source,
)

__all__ = [
    "CorpusSource",
    "DialectDataset",
    "clear_registry",
    "get_source",
    "list_available",
    "register_source",
]
