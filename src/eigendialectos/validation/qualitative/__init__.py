"""Qualitative validation sub-package."""

from eigendialectos.validation.qualitative.hyperdia import HyperdialectalEvaluator
from eigendialectos.validation.qualitative.survey import SurveyGenerator
from eigendialectos.validation.qualitative.turing_test import DialectalTuringTest

__all__ = [
    "DialectalTuringTest",
    "HyperdialectalEvaluator",
    "SurveyGenerator",
]
