"""EigenDialectos: Spectral analysis framework for Spanish dialect variation."""

from __future__ import annotations

__version__ = "0.1.0"

# Configuration
from eigendialectos.config import (
    get_contrastive_config,
    get_embedding_config,
    load_config,
)

# Core types
from eigendialectos.types import (
    CorpusSlice,
    DialectalSpectrum,
    DialectSample,
    EigenDecomposition,
    EmbeddingMatrix,
    ExperimentResult,
    TensorDialectal,
    TransformationMatrix,
)

__all__ = [
    "__version__",
    # config
    "load_config",
    "get_embedding_config",
    "get_contrastive_config",
    # types
    "CorpusSlice",
    "DialectalSpectrum",
    "DialectSample",
    "EigenDecomposition",
    "EmbeddingMatrix",
    "ExperimentResult",
    "TensorDialectal",
    "TransformationMatrix",
]
