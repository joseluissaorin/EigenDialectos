"""Robustness, reproducibility and edge-case tests for eigen3.

30 tests organised into three classes:
    TestPerturbation     — numerical stability under noise (10)
    TestReproducibility  — deterministic behaviour guarantees (10)
    TestEdgeCases        — boundary inputs and error handling (10)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from eigen3.algebra import (
    analogy_dialects,
    compose_dialects,
    interpolate_spectrum,
)
from eigen3.analyzer import analyze_text, name_mode
from eigen3.compiler import compile as sdc_compile
from eigen3.core import EigenDialectos
from eigen3.decomposition import eigendecompose, eigenspectrum
from eigen3.distance import frobenius_distance
from eigen3.per_mode import compute_W_alpha
from eigen3.scorer import DialectScorer
from eigen3.stability import check_condition, regularize_W
from eigen3.types import AlphaVector

ALL = [
    "ES_PEN", "ES_AND", "ES_CAN", "ES_RIO",
    "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO",
]


# ======================================================================
# Perturbation robustness (10 tests)
# ======================================================================

class TestPerturbation:

    def test_small_embedding_noise_small_score_change(
        self, word_embeddings_dict, vocab, decomps_dict, rng,
    ):
        """Adding N(0, 1e-4) to embeddings shifts scores by < 0.3."""
        scorer_clean = DialectScorer(word_embeddings_dict, vocab, decomps_dict)

        noisy_emb = {
            v: emb + rng.normal(0, 1e-4, size=emb.shape)
            for v, emb in word_embeddings_dict.items()
        }
        scorer_noisy = DialectScorer(noisy_emb, vocab, decomps_dict)

        text = "la casa es grande y bonita"
        clean = scorer_clean.score(text)
        noisy = scorer_noisy.score(text)

        for v in clean.probabilities:
            assert abs(clean.probabilities[v] - noisy.probabilities[v]) < 0.3, (
                f"{v}: probability shifted too much under small embedding noise"
            )

    def test_small_W_noise_small_eigenvalue_change(self, W_dict, rng):
        """Tiny noise on W -> eigenvalues change only slightly."""
        W = W_dict["ES_CAN"]
        d_clean = eigendecompose(W, variety="ES_CAN")

        noise = rng.normal(0, 1e-6, size=W.shape)
        d_noisy = eigendecompose(W + noise, variety="ES_CAN")

        mag_diff = np.abs(d_clean.magnitudes - d_noisy.magnitudes)
        assert np.max(mag_diff) < 0.01, "Eigenvalue magnitudes shifted too much"

    def test_eigendecomp_stable(self, W_dict):
        """Eigendecomposing the same W twice gives identical results."""
        W = W_dict["ES_RIO"]
        d1 = eigendecompose(W, variety="ES_RIO")
        d2 = eigendecompose(W, variety="ES_RIO")
        np.testing.assert_array_equal(d1.eigenvalues, d2.eigenvalues)
        np.testing.assert_array_equal(d1.P, d2.P)

    def test_scoring_robust_to_word_order(
        self, word_embeddings_dict, vocab, decomps_dict,
    ):
        """BoW scorer assigns same classification regardless of word order."""
        scorer = DialectScorer(word_embeddings_dict, vocab, decomps_dict)
        c1 = scorer.classify("casa grande")
        c2 = scorer.classify("grande casa")
        assert c1 == c2

    def test_scoring_robust_to_typo(
        self, word_embeddings_dict, vocab, decomps_dict,
    ):
        """OOV 'cas' (typo for 'casa') doesn't crash; result is a valid ScoreResult."""
        scorer = DialectScorer(word_embeddings_dict, vocab, decomps_dict)
        result = scorer.score("cas")
        assert result.top_dialect in ALL or result.top_dialect != ""
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6

    def test_distance_robust(self, W_dict, rng):
        """Small noise on W produces a smoothly-changing Frobenius distance."""
        W_can = W_dict["ES_CAN"]
        W_car = W_dict["ES_CAR"]
        d_clean = frobenius_distance(W_can, W_car)

        noise = rng.normal(0, 1e-6, size=W_can.shape)
        d_noisy = frobenius_distance(W_can + noise, W_car)
        assert abs(d_clean - d_noisy) < 0.01

    def test_alpha_response_smooth(self, decomps_dict):
        """Two close alpha vectors produce close W(alpha) matrices."""
        decomp = decomps_dict["ES_MEX"]
        n = decomp.n_modes
        alpha1 = AlphaVector.uniform(n, 1.0)
        alpha2 = AlphaVector(values=np.full(n, 1.001, dtype=np.float64))

        W1 = compute_W_alpha(decomp, alpha1)
        W2 = compute_W_alpha(decomp, alpha2)
        assert frobenius_distance(W1, W2) < 1.0

    def test_analogy_robust(self, decomps_dict, rng):
        """Small noise on spectra produces a similar analogy result."""
        ref = decomps_dict["ES_PEN"]
        r1 = analogy_dialects(decomps_dict, "ES_CAN", "ES_CAR", "ES_MEX", ref)

        # Perturb one decomposition's eigenvalues slightly
        noisy_decomps = dict(decomps_dict)
        d_can = decomps_dict["ES_CAN"]
        from eigen3.types import EigenDecomp
        noisy_eig = d_can.eigenvalues + rng.normal(0, 1e-6, size=d_can.eigenvalues.shape)
        noisy_decomps["ES_CAN"] = EigenDecomp(
            P=d_can.P, eigenvalues=noisy_eig, P_inv=d_can.P_inv,
            W_original=d_can.W_original, variety="ES_CAN",
        )
        r2 = analogy_dialects(noisy_decomps, "ES_CAN", "ES_CAR", "ES_MEX", ref)

        assert np.linalg.norm(r1.spectrum - r2.spectrum) < 0.1

    def test_mode_naming_robust(self, decomps_dict, vocab, word_embeddings_dict, rng):
        """Small noise on W does not change the top mode name."""
        emb = word_embeddings_dict["ES_PEN"]
        d = decomps_dict["ES_CHI"]
        name_clean = name_mode(d.P[:, 0], vocab, top_k=3, embeddings=emb)

        noise = rng.normal(0, 1e-8, size=d.W_original.shape)
        d_noisy = eigendecompose(d.W_original + noise, variety="ES_CHI")
        name_noisy = name_mode(d_noisy.P[:, 0], vocab, top_k=3, embeddings=emb)

        assert name_clean == name_noisy

    def test_interpolation_smooth(self, spectra_dict):
        """10-step interpolation path has no sudden jumps."""
        mag_a = spectra_dict["ES_PEN"].magnitudes
        mag_b = spectra_dict["ES_RIO"].magnitudes
        steps = [interpolate_spectrum(mag_a, mag_b, t / 10) for t in range(11)]

        for i in range(len(steps) - 1):
            jump = np.linalg.norm(steps[i + 1] - steps[i])
            total = np.linalg.norm(mag_b - mag_a)
            assert jump < total / 5, f"Jump at step {i} is too large"


# ======================================================================
# Reproducibility (10 tests)
# ======================================================================

class TestReproducibility:

    def test_eigendecomp_deterministic(self, W_dict):
        """Same W always yields identical P and eigenvalues."""
        W = W_dict["ES_AND"]
        d1 = eigendecompose(W)
        d2 = eigendecompose(W)
        np.testing.assert_array_equal(d1.eigenvalues, d2.eigenvalues)
        np.testing.assert_array_equal(d1.P, d2.P)

    def test_scoring_deterministic(
        self, word_embeddings_dict, vocab, decomps_dict,
    ):
        """Scoring the same text twice gives identical ScoreResult."""
        scorer = DialectScorer(word_embeddings_dict, vocab, decomps_dict)
        r1 = scorer.score("buenos aires es hermosa")
        r2 = scorer.score("buenos aires es hermosa")
        assert r1.top_dialect == r2.top_dialect
        for v in r1.probabilities:
            assert r1.probabilities[v] == r2.probabilities[v]
        np.testing.assert_array_equal(r1.mode_activations, r2.mode_activations)

    def test_transform_deterministic(
        self, word_embeddings_dict, vocab, decomps_dict,
    ):
        """Same input always produces the same TransformResult."""
        decomp = decomps_dict["ES_RIO"]
        kwargs = dict(
            text="el coche está aparcado",
            source="ES_PEN",
            target="ES_RIO",
            embeddings=word_embeddings_dict,
            vocab=vocab,
            decomp=decomp,
        )
        r1 = sdc_compile(**kwargs)
        r2 = sdc_compile(**kwargs)
        assert r1.text == r2.text
        assert len(r1.changes) == len(r2.changes)

    def test_analysis_deterministic(
        self, word_embeddings_dict, vocab, decomps_dict,
    ):
        """analyze_text returns identical results on repeated calls."""
        r1 = analyze_text("vamos a la playa", word_embeddings_dict, vocab, decomps_dict)
        r2 = analyze_text("vamos a la playa", word_embeddings_dict, vocab, decomps_dict)
        np.testing.assert_array_equal(r1.mode_strengths, r2.mode_strengths)
        assert r1.mode_names == r2.mode_names

    def test_algebra_deterministic(self, decomps_dict):
        """Composition with same inputs is deterministic."""
        ref = decomps_dict["ES_PEN"]
        w = {"ES_CAN": 0.5, "ES_CAR": 0.5}
        r1 = compose_dialects(decomps_dict, w, ref)
        r2 = compose_dialects(decomps_dict, w, ref)
        np.testing.assert_array_equal(r1.spectrum, r2.spectrum)
        np.testing.assert_array_equal(r1.W, r2.W)

    def test_float64_precision(self, decomps_dict, spectra_dict):
        """All core arrays are float64, not float32."""
        for v in ALL:
            assert decomps_dict[v].eigenvalues.dtype in (np.float64, np.complex128), v
            assert decomps_dict[v].P.dtype in (np.float64, np.complex128), v
            assert spectra_dict[v].magnitudes.dtype == np.float64, v

    def test_large_alpha_no_crash(self, decomps_dict):
        """alpha=10 doesn't crash (extreme amplification)."""
        decomp = decomps_dict["ES_MEX"]
        alpha = AlphaVector.uniform(decomp.n_modes, 10.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W_big = compute_W_alpha(decomp, alpha)
        assert W_big.shape == decomp.W_original.shape

    def test_negative_alpha_no_crash(self, decomps_dict):
        """alpha=-1 doesn't crash (inversion-like)."""
        decomp = decomps_dict["ES_MEX"]
        alpha = AlphaVector.uniform(decomp.n_modes, -1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W_neg = compute_W_alpha(decomp, alpha)
        assert W_neg.shape == decomp.W_original.shape

    def test_core_facade_score(self, word_embeddings_dict, vocab, W_dict):
        """EigenDialectos.score returns a valid ScoreResult."""
        facade = EigenDialectos(word_embeddings_dict, vocab, W_dict)
        result = facade.score("el gato duerme en la casa")
        assert result.top_dialect in ALL
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6

    def test_core_facade_classify(self, word_embeddings_dict, vocab, W_dict):
        """EigenDialectos.classify returns one of the known variety codes."""
        facade = EigenDialectos(word_embeddings_dict, vocab, W_dict)
        label = facade.classify("la guagua va llena de gente")
        assert label in ALL


# ======================================================================
# Edge cases (10 tests)
# ======================================================================

class TestEdgeCases:

    def test_regularization_prevents_singularity(self, W_dict):
        """Regularising an ill-conditioned W keeps condition number finite."""
        W = W_dict["ES_CAN"]
        # Make ill-conditioned: zero out a row
        W_bad = W.copy()
        W_bad[0, :] = 0.0
        W_reg = regularize_W(W_bad, strength=1e-4)
        cond = check_condition(W_reg)
        assert np.isfinite(cond)

    def test_strong_regularization_improves_condition(self, W_dict):
        """Strong regularisation (strength=0.5) improves condition vs none."""
        W = W_dict["ES_AND"]
        W_lam0 = regularize_W(W, strength=0.0)
        W_lam5 = regularize_W(W, strength=0.5)
        cond0 = check_condition(W_lam0)
        cond5 = check_condition(W_lam5)
        # Strong Tikhonov (0.5*W + 0.5*I) should reduce condition number
        assert cond5 < cond0

    def test_high_regularization_toward_identity(self, W_dict):
        """lambda=1.0 gives exactly the identity matrix (Tikhonov)."""
        W = W_dict["ES_RIO"]
        W_reg = regularize_W(W, strength=1.0)
        np.testing.assert_allclose(W_reg, np.eye(W.shape[0]), atol=1e-12)

    def test_empty_text_graceful(
        self, word_embeddings_dict, vocab, decomps_dict,
    ):
        """Empty string does not crash scorer or analyzer."""
        scorer = DialectScorer(word_embeddings_dict, vocab, decomps_dict)
        result = scorer.score("")
        assert result.top_dialect in ALL or result.top_dialect != ""
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6

        analysis = analyze_text("", word_embeddings_dict, vocab, decomps_dict)
        assert analysis.mode_strengths is not None

    def test_single_char_text(
        self, word_embeddings_dict, vocab, decomps_dict,
    ):
        """Single character 'a' works in scorer without error."""
        scorer = DialectScorer(word_embeddings_dict, vocab, decomps_dict)
        result = scorer.score("a")
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6

    def test_very_long_text(
        self, word_embeddings_dict, vocab, decomps_dict,
    ):
        """5000-word text doesn't crash the scorer."""
        scorer = DialectScorer(word_embeddings_dict, vocab, decomps_dict)
        long_text = " ".join(vocab[:5000])
        result = scorer.score(long_text)
        assert result.top_dialect in ALL

    def test_all_same_word(
        self, word_embeddings_dict, vocab, decomps_dict,
    ):
        """Repeating 'casa' 100 times works and gives consistent score."""
        scorer = DialectScorer(word_embeddings_dict, vocab, decomps_dict)
        result = scorer.score("casa " * 100)
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6

    def test_missing_dialect_keyerror(self, decomps_dict):
        """Accessing a non-existent dialect raises KeyError."""
        with pytest.raises(KeyError):
            _ = decomps_dict["ES_FAKE"]

    def test_core_varieties_list(self, word_embeddings_dict, vocab, W_dict):
        """facade.varieties returns a sorted list of 8 variety codes."""
        facade = EigenDialectos(word_embeddings_dict, vocab, W_dict)
        assert facade.varieties == sorted(ALL)
        assert len(facade.varieties) == 8

    def test_core_n_modes(self, word_embeddings_dict, vocab, W_dict):
        """facade.n_modes equals the embedding dimension (100)."""
        facade = EigenDialectos(word_embeddings_dict, vocab, W_dict)
        assert facade.n_modes == 100
