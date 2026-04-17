"""Sentence-level embedding models (BETO, MarIA, SpanBERTa)."""

from eigendialectos.embeddings.sentence.beto_model import BETOModel
from eigendialectos.embeddings.sentence.maria_model import MarIAModel
from eigendialectos.embeddings.sentence.spanbert_model import SpanBERTaModel

__all__ = ["BETOModel", "MarIAModel", "SpanBERTaModel"]
