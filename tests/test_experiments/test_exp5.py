"""Tests for Experiment 5: Dialectal Archaeology."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.experiments.base import Experiment
from eigendialectos.experiments.exp5_archaeology import DialectalArchaeologyExperiment
from eigendialectos.types import ExperimentResult

_CFG = {"seed": 42, "dim": 10, "vocab_size": 50}


class TestArchaeologyInstantiation:
    def test_can_instantiate(self):
        exp = DialectalArchaeologyExperiment()
        assert isinstance(exp, Experiment)
        assert exp.experiment_id == "exp5_archaeology"

    def test_has_required_attrs(self):
        exp = DialectalArchaeologyExperiment()
        assert len(exp.name) > 0
        assert len(exp.description) > 0
        assert isinstance(exp.dependencies, list)
        assert len(exp.dependencies) > 0


class TestArchaeologySetup:
    def test_setup_with_synthetic_data(self):
        exp = DialectalArchaeologyExperiment()
        exp.setup(_CFG)
        assert exp._is_setup is True
        assert exp._text_embeddings is not None

    def test_run_fails_without_setup(self):
        exp = DialectalArchaeologyExperiment()
        with pytest.raises(RuntimeError, match="not been set up"):
            exp.run()


class TestArchaeologyRun:
    @pytest.fixture(autouse=True)
    def _setup_exp(self):
        self.exp = DialectalArchaeologyExperiment()
        self.exp.setup(_CFG)
        self.result = self.exp.run()

    def test_produces_experiment_result(self):
        assert isinstance(self.result, ExperimentResult)
        assert self.result.experiment_id == "exp5_archaeology"

    def test_metrics_keys(self):
        assert "samples" in self.result.metrics
        assert "historical_texts" in self.result.metrics
        assert "feature_alignment" in self.result.metrics
        assert "n_samples" in self.result.metrics

    def test_samples_structure(self):
        samples = self.result.metrics["samples"]
        assert len(samples) >= 3  # at least 3 Golden Age texts built-in
        for s in samples:
            assert "title" in s
            assert "period" in s
            assert "per_dialect" in s
            assert "closest_dialect_inverse" in s
            # per_dialect should have entries for all 8 dialects
            assert len(s["per_dialect"]) == 8

    def test_inverse_displacement_positive(self):
        samples = self.result.metrics["samples"]
        for s in samples:
            for code, data in s["per_dialect"].items():
                assert data["inverse_displacement"] >= 0.0

    def test_historical_texts_are_strings(self):
        texts = self.result.metrics["historical_texts"]
        assert len(texts) >= 3
        for t in texts:
            assert isinstance(t, str)
            assert len(t) > 10


class TestArchaeologyEvaluate:
    def test_evaluate_returns_expected_keys(self):
        exp = DialectalArchaeologyExperiment()
        exp.setup(_CFG)
        result = exp.run()
        evaluation = exp.evaluate(result)

        assert isinstance(evaluation, dict)
        assert "closest_dialect_distribution" in evaluation
        assert "most_common_closest" in evaluation
        assert "avg_inverse_displacements" in evaluation


class TestArchaeologyVisualize:
    def test_visualize_creates_figures(self, tmp_path):
        cfg = {**_CFG, "output_dir": str(tmp_path)}
        exp = DialectalArchaeologyExperiment()
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


class TestArchaeologyReport:
    def test_report_contains_expected_content(self):
        exp = DialectalArchaeologyExperiment()
        exp.setup(_CFG)
        result = exp.run()
        report = exp.report(result)
        assert isinstance(report, str)
        assert "Archaeology" in report or "Diachronic" in report
        assert "Golden Age" in report or "Cervantes" in report or "proto" in report.lower()
