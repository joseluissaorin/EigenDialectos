"""Algebraic model module for dialect transformation algebra."""

from eigendialectos.algebra.lexical import LexicalOperator
from eigendialectos.algebra.model import DialectAlgebra
from eigendialectos.algebra.morphosyntactic import MorphosyntacticOperator
from eigendialectos.algebra.phonological import PhonologicalOperator
from eigendialectos.algebra.pragmatic import PragmaticOperator
from eigendialectos.algebra.regionalism import (
    decompose_regionalism,
    multiplicative_decomposition,
)

__all__ = [
    "DialectAlgebra",
    "LexicalOperator",
    "MorphosyntacticOperator",
    "PragmaticOperator",
    "PhonologicalOperator",
    "decompose_regionalism",
    "multiplicative_decomposition",
]
