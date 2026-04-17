"""Geometry package: Lie algebra, Riemannian, Fisher information, eigenvalue fields."""

from eigendialectos.geometry.lie_algebra import LieAlgebraAnalysis
from eigendialectos.geometry.riemannian import RiemannianDialectSpace
from eigendialectos.geometry.fisher import FisherInformationAnalysis
from eigendialectos.geometry.eigenfield import EigenvalueField

__all__ = [
    "LieAlgebraAnalysis",
    "RiemannianDialectSpace",
    "FisherInformationAnalysis",
    "EigenvalueField",
]
