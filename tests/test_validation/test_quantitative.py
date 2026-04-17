"""Tests for quantitative validation modules."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.types import CorpusSlice, DialectSample, DialectalSpectrum
from eigendialectos.validation.metrics import (
    compute_bleu,
    compute_chrf,
    compute_classification_accuracy,
    compute_confusion_matrix,
    compute_dialectal_perplexity_ratio,
    compute_eigenspectrum_divergence,
    compute_frobenius_error,
    compute_krippendorff_alpha,
)
from eigendialectos.validation.quantitative.classification import (
    DialectClassifier,
    extract_eigenvalue_features,
)
from eigendialectos.validation.quantitative.holdout import HoldoutEvaluator
from eigendialectos.validation.quantitative.perplexity import (
    NgramLM,
    PerplexityEvaluator,
)


# ======================================================================
# BLEU
# ======================================================================

class TestBLEU:
    """Tests for the BLEU metric."""

    def test_identical_strings(self):
        text = "el gato se sentó en la alfombra"
        assert compute_bleu(text, text) == pytest.approx(1.0)

    def test_completely_different(self):
        ref = "el gato se sentó en la alfombra"
        hyp = "xxx yyy zzz www qqq rrr ppp"
        assert compute_bleu(ref, hyp) == 0.0

    def test_partial_overlap(self):
        ref = "el gato se sentó en la alfombra roja del salón grande"
        hyp = "el gato se echó en la alfombra roja del cuarto grande"
        score = compute_bleu(ref, hyp)
        assert 0.0 < score < 1.0

    def test_empty_hypothesis(self):
        assert compute_bleu("algo de texto", "") == 0.0

    def test_empty_reference(self):
        assert compute_bleu("", "algo de texto") == 0.0

    def test_brevity_penalty(self):
        """Shorter hypothesis should receive a brevity penalty."""
        ref = "el gato se sentó en la alfombra roja"
        hyp_short = "el gato"
        hyp_full = "el gato se sentó en la alfombra roja"
        assert compute_bleu(ref, hyp_short) < compute_bleu(ref, hyp_full)

    def test_known_value(self):
        """Check a known BLEU value from a simple case."""
        ref = "a b c d"
        hyp = "a b c d"
        assert compute_bleu(ref, hyp) == pytest.approx(1.0)


# ======================================================================
# chrF
# ======================================================================

class TestChrF:
    """Tests for the chrF metric."""

    def test_identical(self):
        text = "hola mundo"
        assert compute_chrf(text, text) == pytest.approx(1.0)

    def test_completely_different(self):
        score = compute_chrf("aaa", "zzz")
        assert score == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        score = compute_chrf("hola mundo", "hola tierra")
        assert 0.0 < score < 1.0

    def test_empty(self):
        assert compute_chrf("hola", "") == 0.0
        assert compute_chrf("", "hola") == 0.0


# ======================================================================
# Perplexity ratio
# ======================================================================

class TestPerplexityRatio:
    """Tests for dialectal perplexity ratio."""

    def test_same_distribution_gives_one(self):
        probs = {"hola": 0.5, "mundo": 0.5}
        ratio = compute_dialectal_perplexity_ratio("hola mundo", probs, probs)
        assert ratio == pytest.approx(1.0)

    def test_better_target_gives_less_than_one(self):
        target_probs = {"hola": 0.8, "mundo": 0.2}
        baseline_probs = {"hola": 0.1, "mundo": 0.1}
        ratio = compute_dialectal_perplexity_ratio("hola mundo", target_probs, baseline_probs)
        assert ratio < 1.0


# ======================================================================
# Classification accuracy & confusion matrix
# ======================================================================

class TestClassificationMetrics:
    """Tests for classification metrics."""

    def test_perfect_accuracy(self):
        labels = [DialectCode.ES_PEN, DialectCode.ES_RIO, DialectCode.ES_MEX]
        assert compute_classification_accuracy(labels, labels) == 1.0

    def test_zero_accuracy(self):
        pred = [DialectCode.ES_PEN, DialectCode.ES_PEN]
        true = [DialectCode.ES_RIO, DialectCode.ES_RIO]
        assert compute_classification_accuracy(pred, true) == 0.0

    def test_confusion_matrix_shape(self):
        labels = [DialectCode.ES_PEN, DialectCode.ES_RIO, DialectCode.ES_MEX]
        pred = [DialectCode.ES_PEN, DialectCode.ES_PEN, DialectCode.ES_MEX]
        true = [DialectCode.ES_PEN, DialectCode.ES_RIO, DialectCode.ES_MEX]
        cm = compute_confusion_matrix(pred, true, labels)
        assert cm.shape == (3, 3)

    def test_confusion_matrix_diagonal(self):
        """Perfect predictions => all counts on the diagonal."""
        labels = [DialectCode.ES_PEN, DialectCode.ES_RIO]
        pred = [DialectCode.ES_PEN, DialectCode.ES_RIO, DialectCode.ES_PEN]
        true = [DialectCode.ES_PEN, DialectCode.ES_RIO, DialectCode.ES_PEN]
        cm = compute_confusion_matrix(pred, true, labels)
        assert cm[0, 0] == 2
        assert cm[1, 1] == 1
        assert cm[0, 1] == 0
        assert cm[1, 0] == 0

    def test_confusion_matrix_off_diagonal(self):
        """A single mis-prediction should appear off-diagonal."""
        labels = [DialectCode.ES_PEN, DialectCode.ES_RIO]
        pred = [DialectCode.ES_RIO]  # predicted RIO
        true = [DialectCode.ES_PEN]  # truth is PEN
        cm = compute_confusion_matrix(pred, true, labels)
        # cm[true_idx, pred_idx] -> cm[0, 1] == 1
        assert cm[0, 1] == 1


# ======================================================================
# Frobenius & spectral metrics
# ======================================================================

class TestMatrixMetrics:
    """Tests for Frobenius error and eigenspectrum divergence."""

    def test_frobenius_zero_for_identical(self):
        W = np.eye(5)
        assert compute_frobenius_error(W, W) == pytest.approx(0.0)

    def test_frobenius_positive_for_different(self):
        W1 = np.eye(3)
        W2 = np.ones((3, 3))
        assert compute_frobenius_error(W1, W2) > 0.0

    def test_eigenspectrum_divergence_zero_for_same(self):
        spec = np.array([3.0, 2.0, 1.0])
        assert compute_eigenspectrum_divergence(spec, spec) == pytest.approx(0.0, abs=1e-8)

    def test_eigenspectrum_divergence_positive(self):
        a = np.array([5.0, 3.0, 1.0])
        b = np.array([1.0, 1.0, 1.0])
        assert compute_eigenspectrum_divergence(a, b) > 0.0


# ======================================================================
# Krippendorff's alpha
# ======================================================================

class TestKrippendorffAlpha:
    """Tests for inter-annotator agreement."""

    def test_perfect_agreement(self):
        ratings = np.array([[1, 2, 3, 1, 2],
                            [1, 2, 3, 1, 2]], dtype=float)
        alpha = compute_krippendorff_alpha(ratings)
        assert alpha == pytest.approx(1.0, abs=0.01)

    def test_no_agreement_low_alpha(self):
        """Random/opposing ratings should produce alpha near or below 0."""
        ratings = np.array([[1, 2, 1, 2],
                            [2, 1, 2, 1]], dtype=float)
        alpha = compute_krippendorff_alpha(ratings)
        assert alpha < 0.5


# ======================================================================
# Perplexity evaluator (n-gram LM)
# ======================================================================

class TestPerplexityEvaluator:
    """Tests for n-gram LM training and perplexity evaluation."""

    @pytest.fixture
    def simple_train_data(self):
        """Two 'dialects' with clearly distinct vocabularies."""
        dialect_a = DialectCode.ES_PEN
        dialect_b = DialectCode.ES_RIO

        samples_a = [
            DialectSample(text="el ordenador funciona bien", dialect_code=dialect_a,
                          source_id="t", confidence=1.0),
            DialectSample(text="el ordenador es nuevo", dialect_code=dialect_a,
                          source_id="t", confidence=1.0),
            DialectSample(text="el ordenador tiene buena pantalla", dialect_code=dialect_a,
                          source_id="t", confidence=1.0),
        ]
        samples_b = [
            DialectSample(text="la computadora funciona bien", dialect_code=dialect_b,
                          source_id="t", confidence=1.0),
            DialectSample(text="la computadora es nueva", dialect_code=dialect_b,
                          source_id="t", confidence=1.0),
            DialectSample(text="la computadora tiene buena pantalla", dialect_code=dialect_b,
                          source_id="t", confidence=1.0),
        ]

        return {
            dialect_a: CorpusSlice(samples=samples_a, dialect_code=dialect_a),
            dialect_b: CorpusSlice(samples=samples_b, dialect_code=dialect_b),
        }

    def test_build_and_query(self, simple_train_data):
        ev = PerplexityEvaluator()
        ev.build_ngram_lms(simple_train_data, n=2)
        assert len(ev.lm_per_dialect) == 2

        pp = ev.compute_perplexity("el ordenador funciona bien", DialectCode.ES_PEN)
        assert pp > 0

    def test_cross_dialect_perplexity(self, simple_train_data):
        ev = PerplexityEvaluator()
        ev.build_ngram_lms(simple_train_data, n=2)

        cross = ev.cross_dialect_perplexity("el ordenador funciona bien")
        assert DialectCode.ES_PEN in cross
        assert DialectCode.ES_RIO in cross

    def test_dialect_fidelity(self, simple_train_data):
        ev = PerplexityEvaluator()
        ev.build_ngram_lms(simple_train_data, n=2)

        texts = {
            DialectCode.ES_PEN: ["el ordenador funciona bien"],
            DialectCode.ES_RIO: ["la computadora funciona bien"],
        }
        result = ev.evaluate_dialect_fidelity(texts)
        assert result["total"] == 2
        # With distinct enough vocabularies, fidelity should be high
        assert result["accuracy"] >= 0.5

    def test_ngram_lm_perplexity_finite(self):
        lm = NgramLM(n=2)
        lm.train(["hola mundo", "hola amigo"])
        pp = lm.perplexity("hola mundo")
        assert np.isfinite(pp) and pp > 0


# ======================================================================
# Classifier
# ======================================================================

class TestDialectClassifier:
    """Tests for dialect classification with separable synthetic data."""

    @pytest.fixture
    def separable_data(self):
        """Create linearly separable 2-class data."""
        rng = np.random.default_rng(42)
        n_per_class = 50
        dim = 10

        # Class 0: mean at +2
        X0 = rng.standard_normal((n_per_class, dim)) + 2.0
        # Class 1: mean at -2
        X1 = rng.standard_normal((n_per_class, dim)) - 2.0

        X = np.vstack([X0, X1])
        labels = (
            [DialectCode.ES_PEN] * n_per_class
            + [DialectCode.ES_RIO] * n_per_class
        )
        return X, labels

    def test_train_and_predict(self, separable_data):
        X, labels = separable_data
        clf = DialectClassifier(method="logistic_regression")
        clf.train(X, labels)
        preds = clf.predict(X)
        assert len(preds) == len(labels)
        acc = compute_classification_accuracy(preds, labels)
        # Highly separable data should give near-perfect accuracy
        assert acc > 0.9

    def test_predict_proba(self, separable_data):
        X, labels = separable_data
        clf = DialectClassifier(method="logistic_regression")
        clf.train(X, labels)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(labels), 2)
        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_evaluate_returns_expected_keys(self, separable_data):
        X, labels = separable_data
        clf = DialectClassifier(method="logistic_regression")
        clf.train(X, labels)
        result = clf.evaluate(X, labels)
        assert "accuracy" in result
        assert "per_class_f1" in result
        assert "confusion_matrix" in result

    def test_confusion_matrix_correct_shape(self, separable_data):
        X, labels = separable_data
        clf = DialectClassifier(method="logistic_regression")
        clf.train(X, labels)
        result = clf.evaluate(X, labels)
        cm = np.array(result["confusion_matrix"])
        assert cm.shape == (2, 2)


# ======================================================================
# Feature extraction from spectra
# ======================================================================

class TestExtractEigenvalueFeatures:
    """Tests for eigenvalue feature extraction."""

    def test_output_shape(self):
        spectra = {
            DialectCode.ES_PEN: DialectalSpectrum(
                eigenvalues_sorted=np.array([5.0, 3.0, 1.0]),
                entropy=1.5,
                dialect_code=DialectCode.ES_PEN,
            ),
            DialectCode.ES_RIO: DialectalSpectrum(
                eigenvalues_sorted=np.array([4.0, 2.0, 0.5]),
                entropy=1.3,
                dialect_code=DialectCode.ES_RIO,
            ),
        }
        X, labels = extract_eigenvalue_features(spectra)
        assert X.shape[0] == 2
        assert len(labels) == 2
        # Features = 3 eigenvalues + 4 extras (entropy, p25, p50, p75)
        assert X.shape[1] == 7

    def test_empty_spectra(self):
        X, labels = extract_eigenvalue_features({})
        assert X.shape[0] == 0
        assert labels == []


# ======================================================================
# Holdout evaluator
# ======================================================================

class TestHoldoutEvaluator:
    """Tests for the holdout evaluator."""

    @pytest.fixture
    def test_corpus(self):
        return {
            DialectCode.ES_PEN: CorpusSlice(
                samples=[
                    DialectSample(text="hola mundo", dialect_code=DialectCode.ES_PEN,
                                  source_id="t", confidence=1.0),
                    DialectSample(text="buenos dias", dialect_code=DialectCode.ES_PEN,
                                  source_id="t", confidence=1.0),
                ],
                dialect_code=DialectCode.ES_PEN,
            ),
        }

    def test_evaluate_generation(self, test_corpus):
        ev = HoldoutEvaluator(test_corpus)
        generated = {DialectCode.ES_PEN: ["hola mundo", "buenos dias"]}
        result = ev.evaluate_generation(generated)
        assert "macro_avg" in result
        assert result[DialectCode.ES_PEN.value]["bleu"] == pytest.approx(1.0)

    def test_report_nonempty(self, test_corpus):
        ev = HoldoutEvaluator(test_corpus)
        generated = {DialectCode.ES_PEN: ["hola mundo", "buenos dias"]}
        ev.evaluate_generation(generated)
        report = ev.report()
        assert "Generation Metrics" in report
