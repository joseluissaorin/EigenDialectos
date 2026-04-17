"""Abstract base class for all EigenDialectos experiments."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eigendialectos.types import ExperimentResult

logger = logging.getLogger(__name__)


class Experiment(ABC):
    """Base class that every experiment must subclass.

    Provides a uniform lifecycle: ``setup -> run -> evaluate -> visualize -> report``
    plus serialisation helpers (``save``).

    Subclasses **must** define the four class-level attributes and implement
    the four abstract methods.

    Attributes
    ----------
    experiment_id : str
        Short machine-friendly identifier (e.g. ``"exp1_spectral_map"``).
    name : str
        Human-readable name.
    description : str
        One-paragraph summary of what the experiment does.
    dependencies : list[str]
        Module paths this experiment requires (used by the runner to
        verify availability before execution).
    """

    experiment_id: str = ""
    name: str = ""
    description: str = ""
    dependencies: list[str] = []

    def __init__(self) -> None:
        self._config: dict = {}
        self._is_setup: bool = False

    # ------------------------------------------------------------------
    # Lifecycle (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self, config: dict) -> None:
        """Prepare data, load models, pre-compute shared structures.

        Parameters
        ----------
        config : dict
            Experiment-specific configuration. Typical keys include
            ``"dim"``, ``"n_dialects"``, ``"seed"``, ``"data_dir"``, etc.
        """
        ...

    @abstractmethod
    def run(self) -> ExperimentResult:
        """Execute the core computation and return results.

        Returns
        -------
        ExperimentResult
        """
        ...

    @abstractmethod
    def evaluate(self, result: ExperimentResult) -> dict:
        """Compute evaluation metrics on an experiment result.

        Parameters
        ----------
        result : ExperimentResult

        Returns
        -------
        dict
            Metric name -> value mapping.
        """
        ...

    @abstractmethod
    def visualize(self, result: ExperimentResult) -> list[Path]:
        """Generate visualisation artefacts (figures, HTML, etc.).

        Parameters
        ----------
        result : ExperimentResult

        Returns
        -------
        list[Path]
            Paths to the generated files.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def report(self, result: ExperimentResult) -> str:
        """Return a human-readable textual report for *result*.

        The default implementation produces a Markdown-formatted summary
        including all metrics.  Subclasses may override for richer output.
        """
        lines: list[str] = [
            f"# {self.name}",
            "",
            f"**Experiment ID:** `{self.experiment_id}`",
            f"**Timestamp:** {result.timestamp}",
            "",
            "## Description",
            "",
            self.description,
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(result.config, indent=2, default=str),
            "```",
            "",
            "## Metrics",
            "",
        ]
        for key, value in result.metrics.items():
            lines.append(f"- **{key}:** {_format_metric(value)}")
        lines.append("")

        if result.artifact_paths:
            lines.append("## Artefacts")
            lines.append("")
            for p in result.artifact_paths:
                lines.append(f"- `{p}`")
            lines.append("")

        return "\n".join(lines)

    def save(self, result: ExperimentResult, output_dir: Path) -> None:
        """Persist *result* to *output_dir*.

        Creates:
        - ``<output_dir>/<experiment_id>/result.json``  --  serialised metrics
        - ``<output_dir>/<experiment_id>/report.md``     --  textual report
        """
        exp_dir = output_dir / self.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Serialise result
        result_path = exp_dir / "result.json"
        serialisable = {
            "experiment_id": result.experiment_id,
            "timestamp": result.timestamp,
            "config": _make_json_safe(result.config),
            "metrics": _make_json_safe(result.metrics),
            "artifact_paths": result.artifact_paths,
        }
        result_path.write_text(json.dumps(serialisable, indent=2, default=str))
        logger.info("Saved result to %s", result_path)

        # Write report
        report_path = exp_dir / "report.md"
        report_path.write_text(self.report(result))
        logger.info("Saved report to %s", report_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_result(
        self,
        metrics: dict,
        artifact_paths: list[str] | None = None,
    ) -> ExperimentResult:
        """Convenience factory that fills in boilerplate fields."""
        return ExperimentResult(
            experiment_id=self.experiment_id,
            metrics=metrics,
            artifact_paths=artifact_paths or [],
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=dict(self._config),
        )

    def _check_setup(self) -> None:
        if not self._is_setup:
            raise RuntimeError(
                f"Experiment {self.experiment_id!r} has not been set up. "
                "Call setup(config) first."
            )


# ------------------------------------------------------------------
# Module-private utilities
# ------------------------------------------------------------------

def _format_metric(value: object) -> str:
    """Pretty-print a single metric value."""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, np.ndarray):
        if value.size <= 10:
            return np.array2string(value, precision=4, separator=", ")
        return f"ndarray(shape={value.shape}, mean={value.mean():.4f})"
    return str(value)


def _make_json_safe(obj: object) -> object:
    """Recursively convert numpy types so ``json.dumps`` succeeds."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.complexfloating,)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
