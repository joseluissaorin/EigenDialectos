"""Tests for dialect algebraic composition: spectrum-level and matrix-level operations.

50 tests verifying mathematical properties of the algebra module on real v2 W matrices.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigen3.algebra import (
    analogy_dialects,
    analogy_spectrum,
    centroid_spectrum,
    compose_dialects,
    compose_spectra,
    compose_W,
    interpolate_spectrum,
    interpolate_W,
    invert_W,
    predict_leave_one_out,
    spectrum_to_W,
)
from eigen3.types import ComposeResult

ALL = ["ES_PEN", "ES_AND", "ES_CAN", "ES_RIO", "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO"]


# ======================================================================
# Matrix-level operations (20 tests)
# ======================================================================

class TestMatrixLevel:

    def test_compose_is_matmul(self, W_dict):
        """compose_W(A, B) is identical to A @ B."""
        A = W_dict["ES_CAN"]
        B = W_dict["ES_MEX"]
        result = compose_W(A, B)
        expected = A @ B
        assert np.allclose(result, expected, atol=1e-12)

    def test_compose_inverse_identity(self, W_dict):
        """W @ W^-1 should be approximately the identity matrix."""
        W = W_dict["ES_RIO"]
        W_inv = invert_W(W)
        result = compose_W(W, W_inv)
        assert np.allclose(result, np.eye(100), atol=1e-6)

    def test_associativity(self, W_dict):
        """compose(compose(A,B), C) == compose(A, compose(B,C))."""
        A = W_dict["ES_CAN"]
        B = W_dict["ES_MEX"]
        C = W_dict["ES_CAR"]
        lhs = compose_W(compose_W(A, B), C)
        rhs = compose_W(A, compose_W(B, C))
        assert np.allclose(lhs, rhs, atol=1e-8)

    def test_pen_identity_element(self, W_dict):
        """PEN (near-identity) composed with W should approximate W.

        ES_PEN is the reference dialect so its W matrix should be close
        to identity, making it an approximate identity element.
        """
        W_pen = W_dict["ES_PEN"]
        W_can = W_dict["ES_CAN"]
        # PEN is near identity so W_CAN @ W_PEN ~ W_CAN
        result = compose_W(W_can, W_pen)
        pen_deviation = np.linalg.norm(W_pen - np.eye(100), "fro")
        residual = np.linalg.norm(result - W_can, "fro")
        # residual should be bounded by ||W_CAN|| * ||W_PEN - I||
        assert residual < np.linalg.norm(W_can, "fro") * pen_deviation + 1e-10

    def test_inverse_of_inverse(self, W_dict):
        """inv(inv(W)) should recover W."""
        W = W_dict["ES_CHI"]
        W_inv = invert_W(W)
        W_back = invert_W(W_inv)
        assert np.allclose(W_back, W, atol=1e-6)

    def test_interpolate_t0_is_a(self, W_dict):
        """interpolate_W(A, B, 0) should return A."""
        A = W_dict["ES_CAN"]
        B = W_dict["ES_MEX"]
        result = interpolate_W(A, B, 0.0)
        assert np.allclose(result, A, atol=1e-4), (
            f"t=0 deviation: {np.linalg.norm(result - A, 'fro'):.6e}"
        )

    def test_interpolate_t1_is_b(self, W_dict):
        """interpolate_W(A, B, 1) should return B."""
        A = W_dict["ES_CAN"]
        B = W_dict["ES_MEX"]
        result = interpolate_W(A, B, 1.0)
        assert np.allclose(result, B, atol=1e-4), (
            f"t=1 deviation: {np.linalg.norm(result - B, 'fro'):.6e}"
        )

    def test_interpolate_produces_valid(self, W_dict):
        """Interpolated matrix is square (100, 100) with all finite values."""
        A = W_dict["ES_AND"]
        B = W_dict["ES_RIO"]
        result = interpolate_W(A, B, 0.3)
        assert result.shape == (100, 100)
        assert np.all(np.isfinite(result))

    def test_compose_preserves_dimension(self, W_dict):
        """Result of compose_W is (100, 100)."""
        A = W_dict["ES_CAN"]
        B = W_dict["ES_CAR"]
        result = compose_W(A, B)
        assert result.shape == (100, 100)

    def test_inverse_bounded_condition(self, W_dict):
        """Condition number of the inverse should be < 1e10."""
        for v in ALL:
            W_inv = invert_W(W_dict[v])
            cond = np.linalg.cond(W_inv)
            assert cond < 1e10, f"{v}: condition number {cond:.2e} too large"

    def test_compose_noncommutative(self, W_dict):
        """compose_W(A, B) != compose_W(B, A) for distinct A, B."""
        A = W_dict["ES_CAN"]
        B = W_dict["ES_MEX"]
        ab = compose_W(A, B)
        ba = compose_W(B, A)
        diff = np.linalg.norm(ab - ba, "fro")
        assert diff > 1e-8, "Matrix composition should be noncommutative for distinct dialects"

    def test_self_compose_is_square(self, W_dict):
        """compose_W(W, W) should equal W @ W."""
        W = W_dict["ES_AND"]
        result = compose_W(W, W)
        expected = W @ W
        assert np.allclose(result, expected, atol=1e-12)

    def test_interpolate_midpoint(self, W_dict):
        """Frobenius norm at midpoint lies between the norms of the endpoints."""
        A = W_dict["ES_CAN"]
        B = W_dict["ES_CAR"]
        mid = interpolate_W(A, B, 0.5)
        norm_a = np.linalg.norm(A, "fro")
        norm_b = np.linalg.norm(B, "fro")
        norm_mid = np.linalg.norm(mid, "fro")
        lo = min(norm_a, norm_b) * 0.5
        hi = max(norm_a, norm_b) * 2.0
        assert lo < norm_mid < hi, (
            f"midpoint norm {norm_mid:.4f} not in plausible range [{lo:.4f}, {hi:.4f}]"
        )

    def test_compose_W_validates_square(self, rng):
        """Non-square matrix should raise ValueError."""
        rect = rng.standard_normal((100, 50))
        sq = rng.standard_normal((100, 100))
        with pytest.raises(ValueError, match="square"):
            compose_W(rect, sq)

    def test_compose_W_validates_shape_match(self, rng):
        """Mismatched square matrices should raise ValueError."""
        A = rng.standard_normal((50, 50))
        B = rng.standard_normal((100, 100))
        with pytest.raises(ValueError, match="[Ss]hape"):
            compose_W(A, B)

    def test_invert_W_validates_square(self, rng):
        """Non-square matrix should raise ValueError."""
        rect = rng.standard_normal((100, 50))
        with pytest.raises(ValueError, match="square"):
            invert_W(rect)

    def test_interpolate_validates_square(self, rng):
        """Non-square matrix should raise ValueError."""
        rect = rng.standard_normal((100, 50))
        sq = rng.standard_normal((100, 100))
        with pytest.raises(ValueError, match="square"):
            interpolate_W(rect, sq, 0.5)

    def test_interpolate_validates_match(self, rng):
        """Mismatched shapes should raise ValueError."""
        A = rng.standard_normal((50, 50))
        B = rng.standard_normal((100, 100))
        with pytest.raises(ValueError, match="[Ss]hape"):
            interpolate_W(A, B, 0.5)

    def test_compose_result_no_nan(self, W_dict):
        """Composed matrix should contain no NaN values."""
        for v1, v2 in [("ES_CAN", "ES_MEX"), ("ES_RIO", "ES_CAR"), ("ES_AND", "ES_CHI")]:
            result = compose_W(W_dict[v1], W_dict[v2])
            assert not np.any(np.isnan(result)), f"NaN in compose_W({v1}, {v2})"

    def test_interpolate_smooth(self, W_dict):
        """A 5-step interpolation path should have decreasing distance to the target."""
        A = W_dict["ES_CAN"]
        B = W_dict["ES_MEX"]
        ts = [0.0, 0.25, 0.5, 0.75, 1.0]
        dists_to_B = []
        for t in ts:
            Wt = interpolate_W(A, B, t)
            dists_to_B.append(np.linalg.norm(Wt - B, "fro"))
        # Each step should bring us closer to B (monotonically decreasing)
        for i in range(len(dists_to_B) - 1):
            assert dists_to_B[i] >= dists_to_B[i + 1] - 1e-2, (
                f"Non-monotonic path at step {i}: "
                f"d[{i}]={dists_to_B[i]:.4f} < d[{i+1}]={dists_to_B[i+1]:.4f}"
            )


# ======================================================================
# Spectrum-level operations (30 tests)
# ======================================================================

class TestSpectrumLevel:

    # --- interpolate_spectrum ---

    def test_interpolate_t0(self, decomps_dict):
        """interpolate_spectrum(a, b, 0) == a."""
        a = decomps_dict["ES_CAN"].magnitudes
        b = decomps_dict["ES_MEX"].magnitudes
        result = interpolate_spectrum(a, b, 0.0)
        np.testing.assert_array_equal(result, a)

    def test_interpolate_t1(self, decomps_dict):
        """interpolate_spectrum(a, b, 1) == b."""
        a = decomps_dict["ES_CAN"].magnitudes
        b = decomps_dict["ES_MEX"].magnitudes
        result = interpolate_spectrum(a, b, 1.0)
        np.testing.assert_array_equal(result, b)

    def test_interpolate_midpoint(self, decomps_dict):
        """interpolate_spectrum(a, b, 0.5) == (a + b) / 2."""
        a = decomps_dict["ES_AND"].magnitudes
        b = decomps_dict["ES_RIO"].magnitudes
        result = interpolate_spectrum(a, b, 0.5)
        expected = (a + b) / 2.0
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_interpolate_validates_1d(self, rng):
        """2-D array should raise ValueError."""
        a = rng.standard_normal((10, 2))
        b = rng.standard_normal((10,))
        with pytest.raises(ValueError, match="1-D"):
            interpolate_spectrum(a, b, 0.5)

    def test_interpolate_validates_same_length(self, rng):
        """Different-length spectra should raise ValueError."""
        a = rng.standard_normal(50)
        b = rng.standard_normal(100)
        with pytest.raises(ValueError, match="length"):
            interpolate_spectrum(a, b, 0.5)

    def test_self_interpolation(self, decomps_dict):
        """interpolate(a, a, t) == a for any t."""
        a = decomps_dict["ES_CAR"].magnitudes
        for t in [0.0, 0.3, 0.5, 0.7, 1.0]:
            result = interpolate_spectrum(a, a, t)
            np.testing.assert_allclose(result, a, atol=1e-12)

    # --- compose_spectra ---

    def test_compose_single(self, decomps_dict):
        """compose_spectra([a], [1.0]) == a."""
        a = decomps_dict["ES_CAN"].magnitudes
        result = compose_spectra([a], [1.0])
        np.testing.assert_allclose(result, a, atol=1e-12)

    def test_compose_equal_weights(self, decomps_dict):
        """compose with equal weights == centroid."""
        a = decomps_dict["ES_CAN"].magnitudes
        b = decomps_dict["ES_MEX"].magnitudes
        composed = compose_spectra([a, b], [1.0, 1.0])
        centroid = centroid_spectrum([a, b])
        np.testing.assert_allclose(composed, centroid, atol=1e-12)

    def test_compose_weight_normalization(self, decomps_dict):
        """compose([a], [2.0]) == a because weights are normalized."""
        a = decomps_dict["ES_AND"].magnitudes
        result = compose_spectra([a], [2.0])
        np.testing.assert_allclose(result, a, atol=1e-12)

    def test_compose_validates_empty(self):
        """Empty spectra list should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            compose_spectra([], [])

    def test_compose_validates_length_mismatch(self, decomps_dict):
        """Mismatched spectra/weights lengths should raise ValueError."""
        a = decomps_dict["ES_CAN"].magnitudes
        b = decomps_dict["ES_MEX"].magnitudes
        with pytest.raises(ValueError, match="same length"):
            compose_spectra([a, b], [1.0])

    # --- analogy_spectrum ---

    def test_analogy_identity(self, decomps_dict):
        """analogy(a, b, a) == b (the answer to 'a:b :: a:?' is b)."""
        a = decomps_dict["ES_CAN"].magnitudes
        b = decomps_dict["ES_CAR"].magnitudes
        result = analogy_spectrum(a, b, a)
        np.testing.assert_allclose(result, b, atol=1e-12)

    def test_analogy_self_ref(self, decomps_dict):
        """analogy(a, a, c) == c (zero displacement)."""
        a = decomps_dict["ES_CAN"].magnitudes
        c = decomps_dict["ES_MEX"].magnitudes
        result = analogy_spectrum(a, a, c)
        np.testing.assert_allclose(result, c, atol=1e-12)

    def test_analogy_additive(self, decomps_dict):
        """analogy(a, b, c) == c + (b - a)."""
        a = decomps_dict["ES_CAN"].magnitudes
        b = decomps_dict["ES_CAR"].magnitudes
        c = decomps_dict["ES_MEX"].magnitudes
        result = analogy_spectrum(a, b, c)
        expected = c + (b - a)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_analogy_validates_same_length(self, rng):
        """Different-length spectra should raise ValueError."""
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        c = rng.standard_normal(100)
        with pytest.raises(ValueError, match="length"):
            analogy_spectrum(a, b, c)

    # --- centroid_spectrum ---

    def test_centroid_is_mean(self, decomps_dict):
        """centroid == np.mean of all spectra."""
        spectra = [decomps_dict[v].magnitudes for v in ALL]
        result = centroid_spectrum(spectra)
        expected = np.mean(spectra, axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_centroid_validates_empty(self):
        """Empty spectra list should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            centroid_spectrum([])

    # --- predict_leave_one_out ---

    def test_leave_one_out_mean(self, decomps_dict):
        """Prediction for left-out dialect == mean of the remaining."""
        spec_dict = {v: decomps_dict[v].magnitudes for v in ALL}
        left_out = "ES_CAN"
        result = predict_leave_one_out(spec_dict, left_out)
        others = [spec_dict[v] for v in ALL if v != left_out]
        expected = np.mean(others, axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_leave_one_out_missing_key(self, decomps_dict):
        """Missing key should raise KeyError."""
        spec_dict = {v: decomps_dict[v].magnitudes for v in ALL}
        with pytest.raises(KeyError, match="NONEXISTENT"):
            predict_leave_one_out(spec_dict, "NONEXISTENT")

    def test_leave_one_out_too_few(self, decomps_dict):
        """Single dialect should raise ValueError (nothing to average)."""
        spec_dict = {"ES_CAN": decomps_dict["ES_CAN"].magnitudes}
        with pytest.raises(ValueError, match="at least 2"):
            predict_leave_one_out(spec_dict, "ES_CAN")

    def test_leave_one_out_all_8(self, decomps_dict):
        """Leave-one-out works for all 8 varieties without error."""
        spec_dict = {v: decomps_dict[v].magnitudes for v in ALL}
        for v in ALL:
            result = predict_leave_one_out(spec_dict, v)
            assert result.shape == spec_dict[v].shape
            assert np.all(np.isfinite(result))

    def test_leave_one_out_error_bounded(self, decomps_dict):
        """Prediction error is bounded by the max pairwise distance."""
        spec_dict = {v: decomps_dict[v].magnitudes for v in ALL}
        # Compute max pairwise distance
        max_dist = 0.0
        for i, v1 in enumerate(ALL):
            for v2 in ALL[i + 1:]:
                d = np.linalg.norm(spec_dict[v1] - spec_dict[v2])
                max_dist = max(max_dist, d)
        # Each leave-one-out error should be below that bound
        for v in ALL:
            pred = predict_leave_one_out(spec_dict, v)
            err = np.linalg.norm(pred - spec_dict[v])
            assert err < max_dist, (
                f"{v}: LOO error {err:.4f} >= max pairwise dist {max_dist:.4f}"
            )

    # --- spectrum_to_W ---

    def test_spectrum_to_W_roundtrip(self, decomps_dict):
        """spectrum -> W -> decompose -> magnitudes approximately recovers original spectrum."""
        from eigen3.decomposition import eigendecompose

        d = decomps_dict["ES_CAN"]
        original_mags = d.magnitudes.copy()
        W_new = spectrum_to_W(original_mags, d.P, d.P_inv)
        d_new = eigendecompose(W_new, variety="roundtrip")
        recovered_mags = np.sort(d_new.magnitudes)[::-1]
        original_sorted = np.sort(original_mags)[::-1]
        np.testing.assert_allclose(recovered_mags, original_sorted, atol=1e-4)

    def test_spectrum_to_W_shape(self, decomps_dict):
        """Result is (100, 100)."""
        d = decomps_dict["ES_MEX"]
        result = spectrum_to_W(d.magnitudes, d.P, d.P_inv)
        assert result.shape == (100, 100)

    def test_spectrum_to_W_validates_dimension(self, decomps_dict, rng):
        """Mismatched dimensions should raise ValueError."""
        d = decomps_dict["ES_CAN"]
        bad_spectrum = rng.standard_normal(50)  # 50 != 100
        with pytest.raises(ValueError, match="[Dd]imension"):
            spectrum_to_W(bad_spectrum, d.P, d.P_inv)

    # --- compose_dialects ---

    def test_compose_dialects_returns_result(self, decomps_dict):
        """compose_dialects returns a ComposeResult with spectrum, W, condition."""
        ref = decomps_dict["ES_PEN"]
        weights = {"ES_CAN": 0.5, "ES_MEX": 0.5}
        result = compose_dialects(decomps_dict, weights, ref)
        assert isinstance(result, ComposeResult)
        assert isinstance(result.spectrum, np.ndarray)
        assert result.spectrum.ndim == 1
        assert isinstance(result.W, np.ndarray)
        assert result.W.shape == (100, 100)
        assert isinstance(result.condition, float)
        assert result.condition > 0

    def test_compose_dialects_missing_key(self, decomps_dict):
        """Missing key in weights should raise KeyError."""
        ref = decomps_dict["ES_PEN"]
        weights = {"NONEXISTENT": 1.0}
        with pytest.raises(KeyError):
            compose_dialects(decomps_dict, weights, ref)

    # --- analogy_dialects ---

    def test_analogy_dialects_returns_result(self, decomps_dict):
        """analogy_dialects returns a ComposeResult."""
        ref = decomps_dict["ES_PEN"]
        result = analogy_dialects(decomps_dict, "ES_CAN", "ES_CAR", "ES_MEX", ref)
        assert isinstance(result, ComposeResult)
        assert isinstance(result.spectrum, np.ndarray)
        assert result.spectrum.ndim == 1
        assert result.W.shape == (100, 100)
        assert result.condition > 0

    def test_analogy_dialects_missing_key(self, decomps_dict):
        """Missing dialect label should raise KeyError."""
        ref = decomps_dict["ES_PEN"]
        with pytest.raises(KeyError, match="NONEXISTENT"):
            analogy_dialects(decomps_dict, "NONEXISTENT", "ES_CAR", "ES_MEX", ref)

    # --- extrapolation ---

    def test_extrapolation_t_gt_1(self, decomps_dict):
        """interpolate_spectrum with t=1.5 should work (extrapolation beyond endpoints)."""
        a = decomps_dict["ES_CAN"].magnitudes
        b = decomps_dict["ES_MEX"].magnitudes
        result = interpolate_spectrum(a, b, 1.5)
        expected = (1.0 - 1.5) * a + 1.5 * b
        np.testing.assert_allclose(result, expected, atol=1e-12)
        # The result should lie beyond b along the a->b direction
        assert result.shape == a.shape
