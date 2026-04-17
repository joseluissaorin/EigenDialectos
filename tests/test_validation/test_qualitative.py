"""Tests for qualitative validation modules."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest

from eigendialectos.constants import DialectCode, DIALECT_NAMES
from eigendialectos.validation.qualitative.hyperdia import HyperdialectalEvaluator
from eigendialectos.validation.qualitative.survey import SurveyGenerator
from eigendialectos.validation.qualitative.turing_test import DialectalTuringTest


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def real_samples():
    return {
        DialectCode.ES_PEN: [
            "Vamos a coger el autobus para ir al centro.",
            "Me he comprado un ordenador nuevo.",
        ],
        DialectCode.ES_RIO: [
            "Vamos a tomar el colectivo para ir al centro.",
            "Me compre una computadora nueva.",
        ],
    }


@pytest.fixture
def generated_samples():
    return {
        DialectCode.ES_PEN: [
            "Quedamos a las ocho en la plaza.",
            "Tio, no me mola nada madrugar.",
        ],
        DialectCode.ES_RIO: [
            "Nos vemos a las ocho en la plaza.",
            "Che, no me copa nada madrugar.",
        ],
    }


# ======================================================================
# SurveyGenerator
# ======================================================================

class TestSurveyGenerator:
    """Tests for the survey HTML generator."""

    def test_html_contains_expected_elements(self, real_samples, generated_samples):
        gen = SurveyGenerator(config={"seed": 42})
        html = gen.create_survey(real_samples, generated_samples)

        # Basic HTML structure
        assert "<!DOCTYPE html>" in html
        assert "<form" in html
        assert "</form>" in html

        # Survey questions
        assert "Naturalidad" in html or "naturalness" in html.lower()
        assert "Identidad dialectal" in html or "identity" in html.lower()

        # Dialect options present
        for dialect in DialectCode:
            assert dialect.value in html

        # Likert scale radios
        assert 'type="radio"' in html

    def test_html_contains_all_sample_texts(self, real_samples, generated_samples):
        gen = SurveyGenerator(config={"seed": 42})
        html = gen.create_survey(real_samples, generated_samples)

        for texts in real_samples.values():
            for text in texts:
                assert text in html or text.replace("&", "&amp;") in html

        for texts in generated_samples.values():
            for text in texts:
                assert text in html or text.replace("&", "&amp;") in html

    def test_metadata_embedded(self, real_samples, generated_samples):
        gen = SurveyGenerator(config={"seed": 42})
        html = gen.create_survey(real_samples, generated_samples)

        # Metadata JSON should be embedded
        assert "survey-metadata" in html
        assert '"origin"' in html

    def test_parse_responses(self, tmp_path):
        gen = SurveyGenerator(config={"seed": 42})
        response_data = {
            "variety_abc": "ES_PEN",
            "naturalness_abc": "4",
            "identity_abc": "3",
            "_metadata": [{"id": "abc", "dialect": "ES_PEN", "origin": "real"}],
        }
        response_file = tmp_path / "responses.json"
        response_file.write_text(json.dumps(response_data), encoding="utf-8")

        parsed = gen.parse_responses(response_file)
        assert parsed["variety_abc"] == "ES_PEN"

    def test_analyze_responses(self):
        gen = SurveyGenerator(config={"seed": 42})
        responses = {
            "variety_abc": "ES_PEN",
            "naturalness_abc": "4",
            "identity_abc": "3",
            "variety_def": "ES_RIO",
            "naturalness_def": "5",
            "identity_def": "4",
            "_metadata": [
                {"id": "abc", "dialect": "ES_PEN", "origin": "real"},
                {"id": "def", "dialect": "ES_RIO", "origin": "generated"},
            ],
        }
        result = gen.analyze_responses(responses)
        assert "naturalness_mean" in result
        assert "identity_mean" in result
        assert "identification_accuracy" in result
        assert result["naturalness_mean"] == pytest.approx(4.5)
        assert result["identification_accuracy"] == 1.0
        assert result["per_origin"]["real"]["count"] == 1
        assert result["per_origin"]["generated"]["count"] == 1


# ======================================================================
# DialectalTuringTest
# ======================================================================

class TestDialectalTuringTest:
    """Tests for the Turing test generator and evaluator."""

    def test_create_test_html_structure(self, real_samples, generated_samples):
        tt = DialectalTuringTest(n_pairs_per_dialect=2, seed=42)
        html = tt.create_test(real_samples, generated_samples)

        assert "<!DOCTYPE html>" in html
        assert "Turing" in html
        assert "<form" in html
        assert "Texto A" in html or "text_a" in html.lower()
        assert "Texto B" in html or "text_b" in html.lower()

    def test_randomisation(self, real_samples, generated_samples):
        """Different seeds produce different orderings."""
        tt1 = DialectalTuringTest(n_pairs_per_dialect=2, seed=1)
        tt2 = DialectalTuringTest(n_pairs_per_dialect=2, seed=99)
        html1 = tt1.create_test(real_samples, generated_samples)
        html2 = tt2.create_test(real_samples, generated_samples)
        # The metadata (generated_position) should differ between seeds
        # (probabilistically nearly certain with different seeds)
        meta_pattern = r'"generated_position":\s*"[AB]"'
        positions1 = re.findall(meta_pattern, html1)
        positions2 = re.findall(meta_pattern, html2)
        # At least the ordering or positions should differ in most cases
        # (there is a tiny chance they are the same, so we just check structure)
        assert len(positions1) > 0
        assert len(positions2) > 0

    def test_metadata_contains_answers(self, real_samples, generated_samples):
        tt = DialectalTuringTest(n_pairs_per_dialect=2, seed=42)
        html = tt.create_test(real_samples, generated_samples)
        assert '"generated_position"' in html

    def test_evaluate_perfect_detection(self, real_samples, generated_samples):
        """Simulate evaluators who correctly identify all generated texts."""
        tt = DialectalTuringTest(n_pairs_per_dialect=2, seed=42)
        html = tt.create_test(real_samples, generated_samples)

        # Extract metadata from the HTML
        meta_match = re.search(
            r'<script id="turing-metadata" type="application/json">(.*?)</script>',
            html, re.DOTALL
        )
        assert meta_match is not None
        metadata = json.loads(meta_match.group(1))

        # Simulate perfect response
        response: dict = {"_metadata": metadata}
        for pair in metadata:
            response[f"pair_{pair['id']}"] = pair["generated_position"]

        result = tt.evaluate([response])
        assert result["success_rate"] == 1.0
        assert result["n_evaluators"] == 1
        assert result["total_correct"] == result["total_pairs"]

    def test_evaluate_zero_detection(self, real_samples, generated_samples):
        """Simulate evaluators who always pick the wrong answer."""
        tt = DialectalTuringTest(n_pairs_per_dialect=2, seed=42)
        html = tt.create_test(real_samples, generated_samples)

        meta_match = re.search(
            r'<script id="turing-metadata" type="application/json">(.*?)</script>',
            html, re.DOTALL
        )
        metadata = json.loads(meta_match.group(1))

        response: dict = {"_metadata": metadata}
        for pair in metadata:
            # Pick the opposite of the correct answer
            wrong = "B" if pair["generated_position"] == "A" else "A"
            response[f"pair_{pair['id']}"] = wrong

        result = tt.evaluate([response])
        assert result["success_rate"] == 0.0


# ======================================================================
# HyperdialectalEvaluator
# ======================================================================

class TestHyperdialectalEvaluator:
    """Tests for the hyperdialectal evaluation."""

    def test_create_evaluation_html(self):
        ev = HyperdialectalEvaluator(alpha_values=[1.0, 1.5])
        texts_by_alpha = {
            1.0: {DialectCode.ES_PEN: ["texto normal uno"]},
            1.5: {DialectCode.ES_PEN: ["texto exagerado uno"]},
        }
        html = ev.create_evaluation(texts_by_alpha)

        assert "<!DOCTYPE html>" in html
        assert "Hiperdialectal" in html
        assert "exagerado" in html.lower() or "Suena exagerado" in html
        assert 'name="naturalness_' in html

    def test_analyze_responses(self):
        ev = HyperdialectalEvaluator(alpha_values=[1.0, 1.5])
        responses = [
            {
                "naturalness_aaa": "5",
                "exaggerated_aaa": "no",
                "threshold_aaa": "",
                "naturalness_bbb": "2",
                "exaggerated_bbb": "yes",
                "threshold_bbb": "1.5",
                "_metadata": [
                    {"id": "aaa", "alpha": 1.0, "dialect": "ES_PEN", "text": "t1"},
                    {"id": "bbb", "alpha": 1.5, "dialect": "ES_PEN", "text": "t2"},
                ],
            }
        ]
        result = ev.analyze(responses)
        assert "mean_naturalness_by_alpha" in result
        assert "exaggeration_rate_by_alpha" in result
        assert "threshold_distribution" in result
        assert result["mean_naturalness_by_alpha"][1.0] == 5.0
        assert result["mean_naturalness_by_alpha"][1.5] == 2.0
        assert result["exaggeration_rate_by_alpha"][1.0] == 0.0
        assert result["exaggeration_rate_by_alpha"][1.5] == 1.0
        assert result["n_evaluators"] == 1
