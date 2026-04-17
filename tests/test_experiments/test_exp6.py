"""Tests for Experiment 6: Dialectal Evolution (Phylogenetic Analysis)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.experiments.base import Experiment
from eigendialectos.experiments.exp6_evolution import EvolutionExperiment
from eigendialectos.types import ExperimentResult

_CFG = {"seed": 42, "dim": 10, "vocab_size": 50, "k_top": 3}


class TestEvolutionInstantiation:
    def test_can_instantiate(self):
        exp = EvolutionExperiment()
        assert isinstance(exp, Experiment)
        assert exp.experiment_id == "exp6_evolution"

    def test_has_required_attrs(self):
        exp = EvolutionExperiment()
        assert len(exp.name) > 0
        assert len(exp.description) > 0
        assert isinstance(exp.dependencies, list)
        assert len(exp.dependencies) > 0


class TestEvolutionSetup:
    def test_setup_with_synthetic_data(self):
        exp = EvolutionExperiment()
        exp.setup(_CFG)
        assert exp._is_setup is True
        assert len(exp._eigendecomps) == 8

    def test_run_fails_without_setup(self):
        exp = EvolutionExperiment()
        with pytest.raises(RuntimeError, match="not been set up"):
            exp.run()


class TestEvolutionRun:
    @pytest.fixture(autouse=True)
    def _setup_exp(self):
        self.exp = EvolutionExperiment()
        self.exp.setup(_CFG)
        self.result = self.exp.run()

    def test_produces_experiment_result(self):
        assert isinstance(self.result, ExperimentResult)
        assert self.result.experiment_id == "exp6_evolution"

    def test_metrics_keys(self):
        assert "similarity_matrix" in self.result.metrics
        assert "distance_matrix" in self.result.metrics
        assert "dialect_order" in self.result.metrics
        assert "shared_axes_summary" in self.result.metrics
        assert "unique_axes" in self.result.metrics

    def test_similarity_matrix_valid(self):
        sim = np.array(self.result.metrics["similarity_matrix"])
        n = len(self.result.metrics["dialect_order"])
        assert sim.shape == (n, n)
        # Symmetric
        np.testing.assert_allclose(sim, sim.T, atol=1e-10)
        # Diagonal should be 1 (self-similarity)
        for i in range(n):
            assert sim[i, i] >= 0.99, f"Self-similarity should be ~1.0, got {sim[i,i]}"
        # All values in [0, 1]
        assert np.all(sim >= -0.01) and np.all(sim <= 1.01)

    def test_distance_matrix_consistent_with_similarity(self):
        sim = np.array(self.result.metrics["similarity_matrix"])
        dist = np.array(self.result.metrics["distance_matrix"])
        np.testing.assert_allclose(sim + dist, 1.0, atol=1e-10)


class TestEvolutionEvaluate:
    def test_evaluate_returns_expected_keys(self):
        exp = EvolutionExperiment()
        exp.setup(_CFG)
        result = exp.run()
        evaluation = exp.evaluate(result)

        assert isinstance(evaluation, dict)
        assert "cluster_similarities" in evaluation
        assert "mean_intra_cluster_similarity" in evaluation
        assert "mean_inter_cluster_similarity" in evaluation
        assert "intra_gt_inter" in evaluation


class TestEvolutionVisualize:
    def test_visualize_creates_figures(self, tmp_path):
        cfg = {**_CFG, "output_dir": str(tmp_path)}
        exp = EvolutionExperiment()
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


class TestEvolutionReport:
    def test_report_contains_expected_content(self):
        exp = EvolutionExperiment()
        exp.setup(_CFG)
        result = exp.run()
        report = exp.report(result)
        assert isinstance(report, str)
        assert "Evolution" in report or "Phylogenetic" in report
        assert "Shared" in report or "shared" in report
