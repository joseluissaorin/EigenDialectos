"""eigen3 — Ground-up rewrite of EigenDialectos with per-mode eigenvalue control.

W(α⃗) = P · diag(λ₁^α₁, ..., λₙ^αₙ) · P⁻¹
"""

from eigen3.constants import DialectCode, ALL_VARIETIES, REFERENCE_VARIETY
from eigen3.types import (
    AlphaVector,
    AnalysisResult,
    ChangeEntry,
    ComposeResult,
    DialectEmbeddings,
    EigenDecomp,
    EigenSpectrum,
    NullModelResult,
    PersistenceDiagram,
    ScoreResult,
    TransformationMatrix,
    TransformResult,
)
from eigen3.core import EigenDialectos

__all__ = [
    # Facade
    "EigenDialectos",
    # Constants
    "DialectCode",
    "ALL_VARIETIES",
    "REFERENCE_VARIETY",
    # Types
    "AlphaVector",
    "AnalysisResult",
    "ChangeEntry",
    "ComposeResult",
    "DialectEmbeddings",
    "EigenDecomp",
    "EigenSpectrum",
    "NullModelResult",
    "PersistenceDiagram",
    "ScoreResult",
    "TransformationMatrix",
    "TransformResult",
]
