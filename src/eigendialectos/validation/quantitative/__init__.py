"""Quantitative validation sub-package."""

from eigendialectos.validation.quantitative.classification import (
    DialectClassifier,
    extract_eigenvalue_features,
)
from eigendialectos.validation.quantitative.holdout import HoldoutEvaluator
from eigendialectos.validation.quantitative.perplexity import (
    NgramLM,
    PerplexityEvaluator,
)

__all__ = [
    "DialectClassifier",
    "HoldoutEvaluator",
    "NgramLM",
    "PerplexityEvaluator",
    "extract_eigenvalue_features",
]
