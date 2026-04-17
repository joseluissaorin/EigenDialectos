"""Experiment runner: discovery, dependency checking, execution, and output."""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path
from typing import Type

from eigendialectos.experiments.base import Experiment
from eigendialectos.types import ExperimentResult

logger = logging.getLogger(__name__)

# Registry of experiment_id -> (module_path, class_name)
_EXPERIMENT_REGISTRY: dict[str, tuple[str, str]] = {
    "exp1_spectral_map": (
        "eigendialectos.experiments.exp1_spectral_map",
        "SpectralMapExperiment",
    ),
    "exp2_full_generation": (
        "eigendialectos.experiments.exp2_full_generation",
        "FullGenerationExperiment",
    ),
    "exp3_dialectal_gradient": (
        "eigendialectos.experiments.exp3_dialectal_gradient",
        "DialectalGradientExperiment",
    ),
    "exp4_impossible_dialects": (
        "eigendialectos.experiments.exp4_impossible_dialects",
        "ImpossibleDialectsExperiment",
    ),
    "exp5_archaeology": (
        "eigendialectos.experiments.exp5_archaeology",
        "DialectalArchaeologyExperiment",
    ),
    "exp6_evolution": (
        "eigendialectos.experiments.exp6_evolution",
        "EvolutionExperiment",
    ),
    "exp7_zeroshot": (
        "eigendialectos.experiments.exp7_zeroshot",
        "ZeroshotTransferExperiment",
    ),
    # V2 experiments
    "exp_a_dialectal_genome": (
        "eigendialectos.experiments.exp_a_dialectal_genome",
        "DialectalGenomeExperiment",
    ),
    "exp_b_phase_transitions": (
        "eigendialectos.experiments.exp_b_phase_transitions",
        "PhaseTransitionsExperiment",
    ),
    "exp_c_eigenvalue_archaeology": (
        "eigendialectos.experiments.exp_c_eigenvalue_archaeology",
        "EigenvalueArchaeologyExperiment",
    ),
    "exp_d_synthetic_dialect": (
        "eigendialectos.experiments.exp_d_synthetic_dialect",
        "SyntheticDialectExperiment",
    ),
    "exp_e_code_switching": (
        "eigendialectos.experiments.exp_e_code_switching",
        "CodeSwitchingExperiment",
    ),
    "exp_f_eigenvalue_microscope": (
        "eigendialectos.experiments.exp_f_eigenvalue_microscope",
        "EigenvalueMicroscopeExperiment",
    ),
    "exp_g_cross_linguistic": (
        "eigendialectos.experiments.exp_g_cross_linguistic",
        "CrossLinguisticExperiment",
    ),
}


class ExperimentRunner:
    """Orchestrate experiment lifecycle: instantiate, configure, run, save.

    Parameters
    ----------
    config : dict
        Global configuration.  Each experiment may consume a nested dict
        keyed by its ``experiment_id`` (e.g. ``config["exp1_spectral_map"]``).
        Unrecognised keys are silently ignored.
    data_dir : Path
        Root directory for input data / corpora.
    output_dir : Path
        Root directory for experiment outputs.
    """

    def __init__(
        self,
        config: dict,
        data_dir: Path,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._instances: dict[str, Experiment] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_experiments(self) -> list[str]:
        """Return all registered experiment IDs (sorted)."""
        return sorted(_EXPERIMENT_REGISTRY.keys())

    def run_experiment(self, experiment_id: str) -> ExperimentResult:
        """Run a single experiment end-to-end.

        Steps:
        1. Instantiate the experiment class (lazy import).
        2. Check dependencies.
        3. ``setup`` with merged config.
        4. ``run``.
        5. ``evaluate``.
        6. ``visualize``.
        7. ``save``.

        Parameters
        ----------
        experiment_id : str
            Must be a key in the internal registry.

        Returns
        -------
        ExperimentResult

        Raises
        ------
        KeyError
            If the experiment_id is unknown.
        ImportError
            If a dependency is not available.
        """
        exp = self._get_or_create(experiment_id)
        self._check_dependencies(exp)

        # Build per-experiment config
        exp_config = self._build_config(experiment_id)

        logger.info("Setting up experiment %r ...", experiment_id)
        exp.setup(exp_config)

        logger.info("Running experiment %r ...", experiment_id)
        t0 = time.perf_counter()
        result = exp.run()
        elapsed = time.perf_counter() - t0
        logger.info(
            "Experiment %r finished in %.2f s", experiment_id, elapsed
        )

        # Evaluate
        eval_metrics = exp.evaluate(result)
        result.metrics["_evaluation"] = eval_metrics

        # Visualize
        try:
            fig_paths = exp.visualize(result)
            result.artifact_paths.extend(str(p) for p in fig_paths)
        except Exception:
            logger.warning(
                "Visualisation failed for %r; skipping.",
                experiment_id,
                exc_info=True,
            )

        # Save
        exp.save(result, self.output_dir)

        return result

    def run_all(self) -> dict[str, ExperimentResult]:
        """Run every registered experiment sequentially.

        Returns
        -------
        dict[str, ExperimentResult]
            Mapping from experiment_id to its result.  Experiments that
            raise are logged and skipped (the key is absent from the dict).
        """
        results: dict[str, ExperimentResult] = {}
        for exp_id in self.list_experiments():
            try:
                results[exp_id] = self.run_experiment(exp_id)
            except Exception:
                logger.error(
                    "Experiment %r failed; skipping.", exp_id, exc_info=True
                )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, experiment_id: str) -> Experiment:
        """Lazy-import and cache the experiment instance."""
        if experiment_id in self._instances:
            return self._instances[experiment_id]

        if experiment_id not in _EXPERIMENT_REGISTRY:
            raise KeyError(
                f"Unknown experiment {experiment_id!r}. "
                f"Available: {self.list_experiments()}"
            )

        module_path, class_name = _EXPERIMENT_REGISTRY[experiment_id]
        module = importlib.import_module(module_path)
        cls: Type[Experiment] = getattr(module, class_name)
        instance = cls()
        self._instances[experiment_id] = instance
        return instance

    @staticmethod
    def _check_dependencies(exp: Experiment) -> None:
        """Verify that every module in ``exp.dependencies`` can be imported."""
        for dep in exp.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError as exc:
                raise ImportError(
                    f"Experiment {exp.experiment_id!r} requires module "
                    f"{dep!r} which could not be imported: {exc}"
                ) from exc

    def _build_config(self, experiment_id: str) -> dict:
        """Merge global config with experiment-specific overrides."""
        base: dict = {
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
        }
        # Global defaults
        for key in ("seed", "dim", "n_dialects"):
            if key in self.config:
                base[key] = self.config[key]
        # Per-experiment overrides
        if experiment_id in self.config:
            base.update(self.config[experiment_id])
        return base
