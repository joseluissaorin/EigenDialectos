"""Tests for Experiment 2: Full Dialect Generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.experiments.base import Experiment
from eigendialectos.experiments.exp2_full_generation import FullGenerationExperiment
from eigendialectos.types import ExperimentResult

_CFG = {"seed": 42, "dim": 10, "vocab_size": 50, "n_sentences": 5}


class TestFullGenerationInstantiation:
    def test_can_instantiate(self):
        exp = FullGenerationExperiment()
        assert isinstance(exp, Experiment)
        assert exp.experiment_id == "exp2_full_generation"

    def test_has_required_attrs(self):
        exp = FullGenerationExperiment()
        assert len(exp.name) > 0
        assert len(exp.description) > 0
        assert isinstance(exp.dependencies, list)
        assert len(exp.dependencies) > 0


class TestFullGenerationSetup:
    def test_setup_with_synthetic_data(self):
        exp = FullGenerationExperiment()
        exp.setup(_CFG)
        assert exp._is_setup is True
        assert len(exp._eigendecomps) == 8

    def test_run_fails_without_setup(self):
        exp = FullGenerationExperiment()
        with pytest.raises(RuntimeError, match="not been set up"):
            exp.run()


class TestFullGenerationRun:
    @pytest.fixture(autouse=True)
    def _setup_exp(self):
        self.exp = FullGenerationExperiment()
        self.exp.setup(_CFG)
        self.result = self.exp.run()

    def test_produces_experiment_result(self):
        assert isinstance(self.result, ExperimentResult)
        assert self.result.experiment_id == "exp2_full_generation"

    def test_metrics_keys(self):
        assert "per_dialect" in self.result.metrics
        assert "mean_bleu" in self.result.metrics
        assert "mean_chrf" in self.result.metrics
        assert "mean_perplexity_proxy" in self.result.metrics

    def test_per_dialect_metrics_structure(self):
        per_dialect = self.result.metrics["per_dialect"]
        assert len(per_dialect) == 8
        for code, m in per_dialect.items():
            assert "bleu" in m
            assert "chrf" in m
            assert "perplexity_proxy" in m
            assert 0.0 <= m["bleu"] <= 1.0
            assert 0.0 <= m["chrf"] <= 1.0

    def test_mean_bleu_is_average(self):
        per_dialect = self.result.metrics["per_dialect"]
        expected = np.mean([m["bleu"] for m in per_dialect.values()])
        assert abs(self.result.metrics["mean_bleu"] - expected) < 1e-10


class TestFullGenerationEvaluate:
    def test_evaluate_returns_expected_keys(self):
        exp = FullGenerationExperiment()
        exp.setup(_CFG)
        result = exp.run()
        evaluation = exp.evaluate(result)

        assert isinstance(evaluation, dict)
        assert "mean_bleu" in evaluation
        assert "std_bleu" in evaluation
        assert "best_dialect_bleu" in evaluation
        assert "worst_dialect_bleu" in evaluation


class TestFullGenerationVisualize:
    def test_visualize_creates_figures(self, tmp_path):
        cfg = {**_CFG, "output_dir": str(tmp_path)}
        exp = FullGenerationExperiment()
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


class TestFullGenerationReport:
    def test_report_contains_expected_content(self):
        exp = FullGenerationExperiment()
        exp.setup(_CFG)
        result = exp.run()
        report = exp.report(result)
        assert isinstance(report, str)
        assert "Generation" in report
        assert "BLEU" in report
