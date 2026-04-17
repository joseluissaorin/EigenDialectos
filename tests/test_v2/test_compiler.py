"""Tests for SDC compiler."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.compiler.parser import CompilerParser
from eigendialectos.compiler.reconstructor import TextReconstructor, _infer_gender
from eigendialectos.compiler.residual import ResidualCorrectionNetwork
from eigendialectos.types import ParsedText


class TestCompilerParser:
    def test_parse(self):
        parser = CompilerParser()
        result = parser.parse("El gato come pescado.")
        assert isinstance(result, ParsedText)
        assert len(result.words) >= 4


class TestTextReconstructor:
    def test_gender_inference(self):
        assert _infer_gender("gato") == "m"
        assert _infer_gender("gata") == "f"
        assert _infer_gender("mano") == "f"  # exception
        assert _infer_gender("problema") == "m"  # exception

    def test_agreement_correction(self):
        recon = TextReconstructor()
        # If noun changes gender, article should adjust
        tokens = ["el", "gata"]  # "el" should become "la"
        result = recon._apply_agreement(tokens)
        assert result[0].lower() == "la"

    def test_reconstruct_with_no_changes(self):
        parser = CompilerParser()
        recon = TextReconstructor()
        parsed = parser.parse("El gato come.")
        # No replacements
        result = recon.reconstruct(parsed, {})
        assert result == "El gato come."

    def test_reconstruct_preserves_punctuation(self):
        parser = CompilerParser()
        recon = TextReconstructor()
        parsed = parser.parse("¿Cómo estás?")
        result = recon.reconstruct(parsed, {})
        assert "¿" in result
        assert "?" in result


class TestResidualNetwork:
    def test_forward_pass(self):
        import torch
        model = ResidualCorrectionNetwork(vocab_size=500, d_model=64, num_layers=1)
        x = torch.randint(0, 500, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 500)

    def test_parameter_count(self):
        model = ResidualCorrectionNetwork(vocab_size=500, d_model=64, num_layers=1)
        n_params = model.num_parameters
        assert n_params > 0
        assert n_params < 10_000_000  # Should be small, not LLM-sized
