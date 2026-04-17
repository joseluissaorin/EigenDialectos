"""Tests for Experiment 7: Zero-shot Dialect Transfer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.experiments.base import Experiment
from eigendialectos.experiments.exp7_zeroshot import ZeroshotTransferExperiment
from eigendialectos.types import ExperimentResult

_CFG = {"seed": 42, "dim": 10, "vocab_size": 50, "max_holdout_pairs": 3}


class TestZeroshotInstantiation:
    def test_can_instantiate(self):
        exp = ZeroshotTransferExperiment()
        assert isinstance(exp, Experiment)
        assert exp.experiment_id == "exp7_zeroshot"

    def test_has_required_attrs(self):
        exp = ZeroshotTransferExperiment()
        assert len(exp.name) > 0
        assert len(exp.description) > 0
        assert isinstance(exp.dependencies, list)
        assert len(exp.dependencies) > 0


class TestZeroshotSetup:
    def test_setup_with_synthetic_data(self):
        exp = ZeroshotTransferExperiment()
        exp.setup(_CFG)
        assert exp._is_setup is True
        assert len(exp._transforms) == 8
        assert exp._full_tensor is not None

    def test_run_fails_without_setup(self):
        exp = ZeroshotTransferExperiment()
        with pytest.raises(RuntimeError, match="not been set up"):
            exp.run()


class TestZeroshotRun:
    @pytest.fixture(autouse=True)
    def _setup_exp(self):
        self.exp = ZeroshotTransferExperiment()
        self.exp.setup(_CFG)
        self.result = self.exp.run()

    def test_produces_experiment_result(self):
        assert isinstance(self.result, ExperimentResult)
        assert self.result.experiment_id == "exp7_zeroshot"

    def test_metrics_keys(self):
        assert "holdout_results" in self.result.metrics
        assert "mean_frobenius_error" in self.result.metrics
        assert "mean_relative_error" in self.result.metrics
        assert "n_holdout_pairs" in self.result.metrics

    def test_holdout_results_structure(self):
        holdout = self.result.metrics["holdout_results"]
        assert len(holdout) <= _CFG["max_holdout_pairs"]
        for r in holdout:
            assert "held_a" in r
            assert "held_b" in r
            assert "frobenius_error_a" in r
            assert "frobenius_error_b" in r
            assert "relative_error_a" in r
            assert "relative_error_b" in r
            # Errors must be non-negative
            assert r["frobenius_error_a"] >= 0.0
            assert r["frobenius_error_b"] >= 0.0

    def test_reconstruction_error_bounded(self):
        error = self.result.metrics["mean_frobenius_error"]
        assert error >= 0.0
        # Should be finite
        assert np.isfinite(error)


class TestZeroshotEvaluate:
    def test_evaluate_returns_expected_keys(self):
        exp = ZeroshotTransferExperiment()
        exp.setup(_CFG)
        result = exp.run()
        evaluation = exp.evaluate(result)

        assert isinstance(evaluation, dict)
        assert "avg_relative_error_per_dialect" in evaluation
        assert "easiest_to_reconstruct" in evaluation
        assert "hardest_to_reconstruct" in evaluation
        assert "good_generalisation" in evaluation
        assert isinstance(evaluation["good_generalisation"], bool)


class TestZeroshotVisualize:
    def test_visualize_creates_figures(self, tmp_path):
        cfg = {**_CFG, "output_dir": str(tmp_path)}
        exp = ZeroshotTransferExperiment()
        exp.setup(cfg)
        result = exp.run()
        # Need to evaluate first so _evaluation is in metrics
        eval_data = exp.evaluate(result)
        result.metrics["_evaluation"] = eval_data

        try:
            paths = exp.visualize(result)
            assert isinstance(paths, list)
            for p in paths:
                assert isinstance(p, Path)
                assert p.exists()
        except ImportError:
            pytest.skip("matplotlib not available")


class TestZeroshotReport:
    def test_report_contains_expected_content(self):
        exp = ZeroshotTransferExperiment()
        exp.setup(_CFG)
        result = exp.run()
        report = exp.report(result)
        assert isinstance(report, str)
        assert "Zero-shot" in report or "Generalisation" in report or "error" in report.lower()
