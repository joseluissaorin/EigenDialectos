"""Tests for Experiment 4: Impossible Dialects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.experiments.base import Experiment
from eigendialectos.experiments.exp4_impossible_dialects import ImpossibleDialectsExperiment
from eigendialectos.types import ExperimentResult

_CFG = {"seed": 42, "dim": 10, "vocab_size": 50}


class TestImpossibleDialectsInstantiation:
    def test_can_instantiate(self):
        exp = ImpossibleDialectsExperiment()
        assert isinstance(exp, Experiment)
        assert exp.experiment_id == "exp4_impossible_dialects"

    def test_has_required_attrs(self):
        exp = ImpossibleDialectsExperiment()
        assert len(exp.name) > 0
        assert len(exp.description) > 0
        assert isinstance(exp.dependencies, list)
        assert len(exp.dependencies) > 0


class TestImpossibleDialectsSetup:
    def test_setup_with_synthetic_data(self):
        exp = ImpossibleDialectsExperiment()
        exp.setup(_CFG)
        assert exp._is_setup is True
        assert len(exp._transforms) == 8

    def test_run_fails_without_setup(self):
        exp = ImpossibleDialectsExperiment()
        with pytest.raises(RuntimeError, match="not been set up"):
            exp.run()


class TestImpossibleDialectsRun:
    @pytest.fixture(autouse=True)
    def _setup_exp(self):
        self.exp = ImpossibleDialectsExperiment()
        self.exp.setup(_CFG)
        self.result = self.exp.run()

    def test_produces_experiment_result(self):
        assert isinstance(self.result, ExperimentResult)
        assert self.result.experiment_id == "exp4_impossible_dialects"

    def test_metrics_keys(self):
        assert "impossible_combinations" in self.result.metrics
        assert "conflict_matrix" in self.result.metrics
        assert "dialect_order" in self.result.metrics

    def test_impossible_combinations_structure(self):
        combos = self.result.metrics["impossible_combinations"]
        assert len(combos) >= 3  # at least 3 impossible combos defined
        for c in combos:
            assert "name" in c
            assert "conflict_score" in c
            assert "coherence" in c
            assert "score" in c["coherence"]
            assert 0.0 <= c["conflict_score"] <= 1.0

    def test_conflict_matrix_symmetric(self):
        cm = np.array(self.result.metrics["conflict_matrix"])
        np.testing.assert_allclose(cm, cm.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(cm), 0.0, atol=1e-10)


class TestImpossibleDialectsEvaluate:
    def test_evaluate_returns_expected_keys(self):
        exp = ImpossibleDialectsExperiment()
        exp.setup(_CFG)
        result = exp.run()
        evaluation = exp.evaluate(result)

        assert isinstance(evaluation, dict)
        assert "mean_conflict_score" in evaluation
        assert "mean_coherence" in evaluation
        assert "n_incoherent" in evaluation


class TestImpossibleDialectsVisualize:
    def test_visualize_creates_figures(self, tmp_path):
        cfg = {**_CFG, "output_dir": str(tmp_path)}
        exp = ImpossibleDialectsExperiment()
        exp.setup(cfg)
        result = exp.run()

        try:
            paths = exp.visualize(result)
            assert isinstance(paths, list)
            for p in paths:
                assert isinstance(p, Path)
                assert p.exists()
        except ImportError:
            pytest.skip("matplotlib not available")


class TestImpossibleDialectsReport:
    def test_report_contains_expected_content(self):
        exp = ImpossibleDialectsExperiment()
        exp.setup(_CFG)
        result = exp.run()
        report = exp.report(result)
        assert isinstance(report, str)
        assert "Impossible" in report or "impossible" in report
        assert "voseo" in report.lower() or "seseo" in report.lower()
