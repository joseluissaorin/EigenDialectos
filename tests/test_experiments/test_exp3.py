"""Tests for Experiment 3: Dialectal Gradient (alpha sweep)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.experiments.base import Experiment
from eigendialectos.experiments.exp3_dialectal_gradient import DialectalGradientExperiment
from eigendialectos.types import ExperimentResult

# Reduced alpha range for fast tests
_CFG = {
    "seed": 42, "dim": 10, "vocab_size": 50, "n_sentences": 5,
    "alpha_start": 0.0, "alpha_stop": 1.0, "alpha_step": 0.5,
}


class TestDialectalGradientInstantiation:
    def test_can_instantiate(self):
        exp = DialectalGradientExperiment()
        assert isinstance(exp, Experiment)
        assert exp.experiment_id == "exp3_dialectal_gradient"

    def test_has_required_attrs(self):
        exp = DialectalGradientExperiment()
        assert len(exp.name) > 0
        assert len(exp.description) > 0
        assert isinstance(exp.dependencies, list)
        assert len(exp.dependencies) > 0


class TestDialectalGradientSetup:
    def test_setup_with_synthetic_data(self):
        exp = DialectalGradientExperiment()
        exp.setup(_CFG)
        assert exp._is_setup is True
        assert len(exp._eigendecomps) == 8

    def test_run_fails_without_setup(self):
        exp = DialectalGradientExperiment()
        with pytest.raises(RuntimeError, match="not been set up"):
            exp.run()


class TestDialectalGradientRun:
    @pytest.fixture(autouse=True)
    def _setup_exp(self):
        self.exp = DialectalGradientExperiment()
        self.exp.setup(_CFG)
        self.result = self.exp.run()

    def test_produces_experiment_result(self):
        assert isinstance(self.result, ExperimentResult)
        assert self.result.experiment_id == "exp3_dialectal_gradient"

    def test_metrics_keys(self):
        assert "curves" in self.result.metrics
        assert "alphas" in self.result.metrics

    def test_curves_structure(self):
        curves = self.result.metrics["curves"]
        assert len(curves) == 8
        for code, data in curves.items():
            assert "alphas" in data
            assert "confidences" in data
            assert "norms" in data
            assert "recognition_threshold" in data
            assert "naturalness_threshold" in data
            assert len(data["alphas"]) == len(data["confidences"])

    def test_confidence_at_alpha_zero_is_zero(self):
        curves = self.result.metrics["curves"]
        for code, data in curves.items():
            if 0.0 in data["alphas"]:
                idx = data["alphas"].index(0.0)
                assert data["confidences"][idx] == 0.0, (
                    f"Confidence at alpha=0 should be 0 for {code}"
                )

    def test_confidences_in_valid_range(self):
        curves = self.result.metrics["curves"]
        for code, data in curves.items():
            for c in data["confidences"]:
                assert 0.0 <= c <= 1.0, f"Confidence {c} out of [0,1] for {code}"


class TestDialectalGradientEvaluate:
    def test_evaluate_returns_expected_keys(self):
        exp = DialectalGradientExperiment()
        exp.setup(_CFG)
        result = exp.run()
        evaluation = exp.evaluate(result)

        assert isinstance(evaluation, dict)
        assert "recognition_thresholds" in evaluation
        assert "naturalness_thresholds" in evaluation
        assert "n_dialects_with_recognition" in evaluation


class TestDialectalGradientVisualize:
    def test_visualize_creates_figures(self, tmp_path):
        cfg = {**_CFG, "output_dir": str(tmp_path)}
        exp = DialectalGradientExperiment()
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


class TestDialectalGradientReport:
    def test_report_contains_expected_content(self):
        exp = DialectalGradientExperiment()
        exp.setup(_CFG)
        result = exp.run()
        report = exp.report(result)
        assert isinstance(report, str)
        assert "Gradient" in report or "gradient" in report
