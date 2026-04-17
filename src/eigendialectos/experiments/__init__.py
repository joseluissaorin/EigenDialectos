"""EigenDialectos experiment suite.

Each experiment is a self-contained subclass of :class:`Experiment` with its
own setup, run, evaluate, visualize, and report lifecycle.  The
:class:`ExperimentRunner` orchestrates discovery, configuration, execution,
and output persistence.
"""

from eigendialectos.experiments.base import Experiment
from eigendialectos.experiments.runner import ExperimentRunner

__all__ = [
    "Experiment",
    "ExperimentRunner",
]
