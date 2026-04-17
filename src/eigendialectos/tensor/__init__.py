"""Tensor decomposition module for multi-dialect analysis."""

from eigendialectos.tensor.analysis import (
    analyze_factors,
    find_shared_factors,
    find_variety_specific_factors,
)
from eigendialectos.tensor.construction import build_dialect_tensor, extract_slice
from eigendialectos.tensor.cp import cp_decompose
from eigendialectos.tensor.tucker import tucker_decompose, tucker_reconstruct

__all__ = [
    "analyze_factors",
    "build_dialect_tensor",
    "cp_decompose",
    "extract_slice",
    "find_shared_factors",
    "find_variety_specific_factors",
    "tucker_decompose",
    "tucker_reconstruct",
]
