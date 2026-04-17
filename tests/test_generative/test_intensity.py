"""Tests for the IntensityController: sweep and threshold discovery."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.generative.intensity import IntensityController
from eigendialectos.types import EigenDecomposition


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def eigen():
    """Diagonal 4x4 eigendecomposition with eigenvalues > 1."""
    eigenvalues = np.array([3.0, 2.0, 1.5, 1.2], dtype=np.complex128)
    P = np.eye(4, dtype=np.complex128)
    P_inv = np.eye(4, dtype=np.complex128)
    return EigenDecomposition(
        eigenvalues=eigenvalues,
        eigenvectors=P,
        eigenvectors_inv=P_inv,
        dialect_code=DialectCode.ES_AND,
    )


@pytest.fixture
def embedding():
    """Unit embedding vector."""
    return np.array([1.0, 1.0, 1.0, 1.0])


@pytest.fixture
def controller():
    return IntensityController(tolerance=0.01, max_iterations=100)


# ------------------------------------------------------------------
# Tests: sweep
# ------------------------------------------------------------------

class TestSweep:
    """Test intensity sweep generation."""

    def test_sweep_count(self, controller, embedding, eigen):
        results = controller.sweep_intensities(
            embedding, eigen, start=0.0, stop=1.0, step=0.2
        )
        expected = len(np.arange(0.0, 1.0, 0.2))
        assert len(results) == expected

    def test_sweep_alphas_increase(self, controller, embedding, eigen):
        results = controller.sweep_intensities(
            embedding, eigen, start=0.0, stop=1.5, step=0.1
        )
        alphas = [a for a, _ in results]
        assert alphas == sorted(alphas)

    def test_sweep_first_is_identity(self, controller, embedding, eigen):
        """alpha=0 should return the original embedding."""
        results = controller.sweep_intensities(
            embedding, eigen, start=0.0, stop=1.0, step=0.5
        )
        alpha0, emb0 = results[0]
        assert alpha0 == pytest.approx(0.0)
        np.testing.assert_allclose(emb0, embedding, atol=1e-12)

    def test_sweep_norms_monotonically_increase(self, controller, embedding, eigen):
        """For eigenvalues > 1, norms should increase with alpha."""
        results = controller.sweep_intensities(
            embedding, eigen, start=0.0, stop=1.5, step=0.1
        )
        norms = [float(np.linalg.norm(emb)) for _, emb in results]
        for i in range(1, len(norms)):
            assert norms[i] >= norms[i - 1] - 1e-10


# ------------------------------------------------------------------
# Tests: generate_at_intensity
# ------------------------------------------------------------------

class TestGenerateAtIntensity:
    def test_returns_correct_shape(self, controller, embedding, eigen):
        result = controller.generate_at_intensity(embedding, eigen, 1.0)
        assert result.shape == embedding.shape

    def test_alpha_zero(self, controller, embedding, eigen):
        result = controller.generate_at_intensity(embedding, eigen, 0.0)
        np.testing.assert_allclose(result, embedding, atol=1e-12)


# ------------------------------------------------------------------
# Tests: recognition threshold
# ------------------------------------------------------------------

class TestRecognitionThreshold:
    """Binary search for the alpha where a classifier detects the dialect."""

    def test_always_recognised(self, controller, embedding, eigen):
        """If the classifier always says yes, threshold should be near 0."""

        class AlwaysTrue:
            def predict(self, emb):
                return True

        threshold = controller.find_recognition_threshold(
            embedding, eigen, AlwaysTrue(), low=0.0, high=2.0
        )
        assert threshold < 0.02  # near zero

    def test_never_recognised(self, controller, embedding, eigen):
        """If the classifier never detects, threshold should be high."""

        class AlwaysFalse:
            def predict(self, emb):
                return False

        threshold = controller.find_recognition_threshold(
            embedding, eigen, AlwaysFalse(), low=0.0, high=2.0
        )
        assert threshold == pytest.approx(2.0, abs=0.02)

    def test_threshold_at_known_point(self, controller, embedding, eigen):
        """Classifier that activates above norm threshold."""
        base_norm = float(np.linalg.norm(embedding))

        class NormClassifier:
            def __init__(self, threshold_norm):
                self.threshold_norm = threshold_norm

            def predict(self, emb):
                return float(np.linalg.norm(emb)) > self.threshold_norm

        # The transform is diagonal with eigenvalues > 1, so norm grows with alpha
        # Find the alpha where norm exceeds 1.5 * base_norm
        clf = NormClassifier(1.5 * base_norm)
        threshold = controller.find_recognition_threshold(
            embedding, eigen, clf, low=0.0, high=2.0
        )
        # Verify: at the found threshold, the classifier should say True
        transformed = controller.generate_at_intensity(embedding, eigen, threshold)
        assert float(np.linalg.norm(transformed)) >= 1.5 * base_norm - 0.1


# ------------------------------------------------------------------
# Tests: naturalness threshold
# ------------------------------------------------------------------

class TestNaturalnessThreshold:
    """Find the alpha where quality drops below a floor."""

    def test_always_good_quality(self, controller, embedding, eigen):
        """If quality never drops, threshold should be near high."""

        def always_good(emb):
            return 1.0

        threshold = controller.find_naturalness_threshold(
            embedding, eigen, always_good, quality_floor=0.5, low=0.0, high=2.0
        )
        assert threshold > 1.9

    def test_always_bad_quality(self, controller, embedding, eigen):
        """If quality is always bad, threshold should be at low."""

        def always_bad(emb):
            return 0.0

        threshold = controller.find_naturalness_threshold(
            embedding, eigen, always_bad, quality_floor=0.5, low=0.0, high=2.0
        )
        assert threshold == pytest.approx(0.0, abs=0.02)

    def test_degrading_quality(self, controller, embedding, eigen):
        """Quality inversely proportional to norm increase."""
        base_norm = float(np.linalg.norm(embedding))

        def quality_fn(emb):
            current_norm = float(np.linalg.norm(emb))
            # Quality degrades as norm increases
            return max(0.0, 1.0 - (current_norm - base_norm) / (3.0 * base_norm))

        threshold = controller.find_naturalness_threshold(
            embedding, eigen, quality_fn, quality_floor=0.5, low=0.0, high=2.0
        )
        # Verify quality at threshold is still acceptable
        transformed = controller.generate_at_intensity(embedding, eigen, threshold)
        assert quality_fn(transformed) >= 0.45  # allowing small tolerance
