"""Synthetic dialect data generation."""

from eigendialectos.corpus.synthetic.fixtures import (
    DIALECT_FEATURES,
    get_dialect_features,
    get_fixtures,
)
from eigendialectos.corpus.synthetic.generator import SyntheticGenerator
from eigendialectos.corpus.synthetic.templates import (
    DIALECT_TEMPLATES,
    DialectTemplate,
    TransformationRule,
    get_template,
    list_templates,
)

__all__ = [
    "DIALECT_FEATURES",
    "DIALECT_TEMPLATES",
    "DialectTemplate",
    "SyntheticGenerator",
    "TransformationRule",
    "get_dialect_features",
    "get_fixtures",
    "get_template",
    "list_templates",
]
