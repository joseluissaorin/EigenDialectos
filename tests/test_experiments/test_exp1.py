"""Tests for Experiment 1: Spectral Map of Spanish Dialect Varieties."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.experiments.base import Experiment
from eigendialectos.experiments.exp1_spectral_map import SpectralMapExperiment
from eigendialectos.types import ExperimentResult

# Small config reused across tests
_CFG = {"seed": 42, "dim": 10, "vocab_size": 50}


class TestSpectralMapInstantiation:
    def test_can_instantiate(self):
        exp = SpectralMapExperiment()
        assert isinstance(exp, Experiment)
        assert exp.experiment_id == "exp1_spectral_map"

    def test_has_required_attrs(self):
        exp = SpectralMapExperiment()
        assert len(exp.name) > 0
        assert len(exp.description) > 0
        assert isinstance(exp.dependencies, list)
        assert len(exp.dependencies) > 0


class TestSpectralMapSetup:
    def test_setup_with_synthetic_data(self):
        exp = SpectralMapExperiment()
        exp.setup(_CFG)
        assert exp._is_setup is True
        assert len(exp._embeddings) == 8

    def test_run_fails_without_setup(self):
        exp = SpectralMapExperiment()
        with pytest.raises(RuntimeError, match="not been set up"):
            exp.run()


class TestSpectralMapRun:
    @pytest.fixture(autouse=True)
    def _setup_exp(self):
        self.exp = SpectralMapExperiment()
        self.exp.setup(_CFG)
        self.result = self.exp.run()

    def test_produces_experiment_result(self):
        assert isinstance(self.result, ExperimentResult)
        assert self.result.experiment_id == "exp1_spectral_map"

    def test_metrics_keys(self):
        assert "distance_matrix" in self.result.metrics
        assert "entropies" in self.result.metrics
        assert "dialect_order" in self.result.metrics
        assert "mean_distance" in self.result.metrics

    def test_distance_matrix_shape_and_symmetry(self):
        dist = np.array(self.result.metrics["distance_matrix"])
        n = len(self.result.metrics["dialect_order"])
        assert dist.shape == (n, n)
        np.testing.assert_allclose(dist, dist.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-10)

    def test_entropies_non_negative(self):
        for code, entropy in self.result.metrics["entropies"].items():
            assert entropy >= 0.0, f"Negative entropy for {code}"

    def test_all_dialects_present(self):
        order = self.result.metrics["dialect_order"]
        assert len(order) == 8


class TestSpectralMapEvaluate:
    def test_evaluate_returns_expected_keys(self):
        exp = SpectralMapExperiment()
        exp.setup(_CFG)
        result = exp.run()
        evaluation = exp.evaluate(result)

        assert isinstance(evaluation, dict)
        assert "mean_close_distance" in evaluation
        assert "mean_far_distance" in evaluation
        assert "ordering_correct" in evaluation
        assert isinstance(evaluation["ordering_correct"], bool)


class TestSpectralMapVisualize:
    def test_visualize_creates_figures(self, tmp_path):
        cfg = {**_CFG, "output_dir": str(tmp_path)}
        exp = SpectralMapExperiment()
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


class TestSpectralMapReportAndSave:
    def test_report_produces_string(self):
        exp = SpectralMapExperiment()
        exp.setup(_CFG)
        result = exp.run()
        report = exp.report(result)
        assert isinstance(report, str)
        assert "Spectral Map" in report
        assert "exp1_spectral_map" in report

    def test_save_creates_files(self, tmp_path):
        exp = SpectralMapExperiment()
        exp.setup(_CFG)
        result = exp.run()
        exp.save(result, tmp_path)
        assert (tmp_path / "exp1_spectral_map" / "result.json").exists()
        assert (tmp_path / "exp1_spectral_map" / "report.md").exists()
