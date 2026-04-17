"""Generative module: DIAL transforms, mixing, and dialect generation."""

from __future__ import annotations

from eigendialectos.generative.constraints import (
    check_feasibility,
    clip_eigenvalues,
    validate_transform,
)
from eigendialectos.generative.dial import (
    apply_dial,
    compute_dial_series,
    dial_transform_embedding,
)
from eigendialectos.generative.generator import DialectGenerator
from eigendialectos.generative.intensity import IntensityController
from eigendialectos.generative.lora_integration import LoRADialectAdapter
from eigendialectos.generative.mixing import (
    log_euclidean_mix,
    mix_dialects,
    mix_eigendecompositions,
)

__all__ = [
    "apply_dial",
    "check_feasibility",
    "clip_eigenvalues",
    "compute_dial_series",
    "dial_transform_embedding",
    "DialectGenerator",
    "IntensityController",
    "log_euclidean_mix",
    "LoRADialectAdapter",
    "mix_dialects",
    "mix_eigendecompositions",
    "validate_transform",
]
