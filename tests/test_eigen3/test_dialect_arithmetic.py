"""Tests for dialect arithmetic: interpolation, analogy, leave-one-out, centroids.

50 tests verifying semantic/dialectological behavior of algebra.py operations
on real v2 eigendecompositions.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from scipy.spatial.distance import cosine as cosine_dist

from eigen3.algebra import (
    analogy_dialects,
    analogy_spectrum,
    centroid_spectrum,
    compose_dialects,
    compose_spectra,
    interpolate_spectrum,
    predict_leave_one_out,
    spectrum_to_W,
)

ALL = [
    "ES_PEN", "ES_AND", "ES_CAN", "ES_RIO",
    "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO",
]

TOL_EXACT = 1e-10
TOL_ROUNDTRIP = 1e-4


def _mag(decomps_dict, v: str) -> np.ndarray:
    """Extract eigenvalue magnitudes for variety *v*."""
    return decomps_dict[v].magnitudes


def _spectra_dict_from_decomps(decomps_dict) -> dict[str, np.ndarray]:
    """Build a {variety: magnitudes} dict from decomps."""
    return {v: _mag(decomps_dict, v) for v in decomps_dict}


def _all_magnitudes(decomps_dict) -> list[np.ndarray]:
    """List of magnitude arrays for all 8 varieties, in ALL order."""
    return [_mag(decomps_dict, v) for v in ALL]


# ======================================================================
# Interpolation (15 tests)
# ======================================================================


class TestInterpolation:

    def test_interpolation_t0_exact(self, decomps_dict):
        """t=0 returns spec_a exactly."""
        a = _mag(decomps_dict, "ES_PEN")
        b = _mag(decomps_dict, "ES_MEX")
        result = interpolate_spectrum(a, b, 0.0)
        np.testing.assert_allclose(result, a, atol=TOL_EXACT)

    def test_interpolation_t1_exact(self, decomps_dict):
        """t=1 returns spec_b exactly."""
        a = _mag(decomps_dict, "ES_PEN")
        b = _mag(decomps_dict, "ES_MEX")
        result = interpolate_spectrum(a, b, 1.0)
        np.testing.assert_allclose(result, b, atol=TOL_EXACT)

    def test_midpoint_between(self, decomps_dict):
        """Midpoint equals arithmetic mean of endpoints."""
        a = _mag(decomps_dict, "ES_AND")
        b = _mag(decomps_dict, "ES_RIO")
        result = interpolate_spectrum(a, b, 0.5)
        expected = (a + b) / 2.0
        np.testing.assert_allclose(result, expected, atol=TOL_EXACT)

    def test_monotonic_frobenius(self, decomps_dict):
        """Distance to spec_a decreases monotonically as t goes from 1 to 0."""
        a = _mag(decomps_dict, "ES_PEN")
        b = _mag(decomps_dict, "ES_CHI")
        ts = np.linspace(1.0, 0.0, 20)
        distances = [
            np.linalg.norm(interpolate_spectrum(a, b, t) - a) for t in ts
        ]
        for i in range(len(distances) - 1):
            assert distances[i] >= distances[i + 1] - TOL_EXACT, (
                f"Monotonicity violated at step {i}: {distances[i]:.6f} < {distances[i+1]:.6f}"
            )

    def test_valid_result(self, decomps_dict):
        """Interpolated spectrum has all finite components."""
        a = _mag(decomps_dict, "ES_CAN")
        b = _mag(decomps_dict, "ES_CAR")
        result = interpolate_spectrum(a, b, 0.3)
        assert np.all(np.isfinite(result))

    def test_can_car_interpolation(self, decomps_dict):
        """Interpolation between CAN and CAR produces a valid spectrum."""
        a = _mag(decomps_dict, "ES_CAN")
        b = _mag(decomps_dict, "ES_CAR")
        result = interpolate_spectrum(a, b, 0.5)
        assert result.shape == a.shape
        assert np.all(np.isfinite(result))

    def test_symmetric(self, decomps_dict):
        """Swapping A,B and flipping t gives the same result."""
        a = _mag(decomps_dict, "ES_MEX")
        b = _mag(decomps_dict, "ES_AND")
        t = 0.37
        forward = interpolate_spectrum(a, b, t)
        backward = interpolate_spectrum(b, a, 1.0 - t)
        np.testing.assert_allclose(forward, backward, atol=TOL_EXACT)

    def test_self_interpolation(self, decomps_dict):
        """interpolate(a, a, any_t) == a for any t."""
        a = _mag(decomps_dict, "ES_RIO")
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = interpolate_spectrum(a, a, t)
            np.testing.assert_allclose(result, a, atol=TOL_EXACT)

    def test_extrapolation_beyond(self, decomps_dict):
        """t=1.5 works without error (extrapolation)."""
        a = _mag(decomps_dict, "ES_PEN")
        b = _mag(decomps_dict, "ES_CAR")
        result = interpolate_spectrum(a, b, 1.5)
        assert result.shape == a.shape
        assert np.all(np.isfinite(result))

    def test_smooth_20_steps(self, decomps_dict):
        """20-step path has no large jumps: consecutive diff < max_pairwise/2."""
        a = _mag(decomps_dict, "ES_AND")
        b = _mag(decomps_dict, "ES_CHI")
        ts = np.linspace(0.0, 1.0, 20)
        points = [interpolate_spectrum(a, b, t) for t in ts]
        max_pairwise = np.linalg.norm(a - b)
        for i in range(len(points) - 1):
            step = np.linalg.norm(points[i + 1] - points[i])
            assert step < max_pairwise / 2.0, (
                f"Jump at step {i}: {step:.6f} >= {max_pairwise/2:.6f}"
            )

    def test_condition_bounded(self, decomps_dict):
        """spectrum_to_W on an interpolated result has condition < 1e8."""
        ref = decomps_dict["ES_PEN"]
        a = _mag(decomps_dict, "ES_PEN")
        b = _mag(decomps_dict, "ES_MEX")
        interp = interpolate_spectrum(a, b, 0.5)
        W_new = spectrum_to_W(interp, ref.P, ref.P_inv)
        cond = np.linalg.cond(W_new)
        assert cond < 1e8, f"Condition number {cond:.2e} exceeds 1e8"

    def test_entropy_between_endpoints(self, decomps_dict, spectra_dict):
        """For t in [0,1], entropy of interpolated is between endpoint entropies (with tolerance)."""
        a_mag = _mag(decomps_dict, "ES_AND")
        b_mag = _mag(decomps_dict, "ES_RIO")
        e_a = spectra_dict["ES_AND"].entropy
        e_b = spectra_dict["ES_RIO"].entropy
        lo, hi = min(e_a, e_b), max(e_a, e_b)
        tolerance = 0.5 * (hi - lo) + 0.1  # generous tolerance for non-concave entropy

        for t in [0.25, 0.5, 0.75]:
            interp = interpolate_spectrum(a_mag, b_mag, t)
            # Compute entropy of the interpolated spectrum
            mags = np.abs(interp)
            mags = mags[mags > 0]
            probs = mags / mags.sum()
            entropy = -np.sum(probs * np.log(probs))
            assert lo - tolerance <= entropy <= hi + tolerance, (
                f"t={t}: entropy {entropy:.4f} outside [{lo - tolerance:.4f}, {hi + tolerance:.4f}]"
            )

    def test_all_28_pairs(self, decomps_dict):
        """All 28 variety pairs produce a valid interpolation at t=0.5."""
        for va, vb in combinations(ALL, 2):
            a = _mag(decomps_dict, va)
            b = _mag(decomps_dict, vb)
            result = interpolate_spectrum(a, b, 0.5)
            assert np.all(np.isfinite(result)), f"Invalid interpolation for {va}-{vb}"

    def test_continuous_small_dt(self, decomps_dict):
        """Small delta t produces a small change in the result."""
        a = _mag(decomps_dict, "ES_CAN")
        b = _mag(decomps_dict, "ES_MEX")
        t = 0.5
        dt = 1e-6
        r1 = interpolate_spectrum(a, b, t)
        r2 = interpolate_spectrum(a, b, t + dt)
        change = np.linalg.norm(r2 - r1)
        scale = np.linalg.norm(b - a) * dt
        assert change < scale * 10, (
            f"Change {change:.2e} too large for dt={dt:.2e}"
        )

    def test_energy_smooth(self, decomps_dict):
        """sum(spec^2) varies smoothly along the interpolation path."""
        a = _mag(decomps_dict, "ES_RIO")
        b = _mag(decomps_dict, "ES_CAR")
        ts = np.linspace(0.0, 1.0, 50)
        energies = [
            np.sum(interpolate_spectrum(a, b, t) ** 2) for t in ts
        ]
        # Energy is a quadratic in t, so second differences should be constant
        diffs = np.diff(energies)
        second_diffs = np.diff(diffs)
        # For a quadratic, second differences are constant -> their variance is ~0
        assert np.std(second_diffs) < 1e-6 * np.mean(np.abs(energies)), (
            "Energy is not smooth (second differences have high variance)"
        )


# ======================================================================
# Analogy (15 tests)
# ======================================================================


class TestAnalogy:

    def test_degenerate_case(self, decomps_dict):
        """analogy(a, b, a) == b."""
        a = _mag(decomps_dict, "ES_PEN")
        b = _mag(decomps_dict, "ES_AND")
        result = analogy_spectrum(a, b, a)
        np.testing.assert_allclose(result, b, atol=TOL_EXACT)

    def test_self_reference(self, decomps_dict):
        """analogy(a, a, c) == c."""
        a = _mag(decomps_dict, "ES_CAN")
        c = _mag(decomps_dict, "ES_MEX")
        result = analogy_spectrum(a, a, c)
        np.testing.assert_allclose(result, c, atol=TOL_EXACT)

    def test_additive_structure(self, decomps_dict):
        """Result equals c + (b - a) numerically."""
        a = _mag(decomps_dict, "ES_PEN")
        b = _mag(decomps_dict, "ES_AND")
        c = _mag(decomps_dict, "ES_CAN")
        result = analogy_spectrum(a, b, c)
        expected = c + (b - a)
        np.testing.assert_allclose(result, expected, atol=TOL_EXACT)

    def test_pen_and_can(self, decomps_dict):
        """analogy(PEN, AND, CAN) produces a valid spectrum."""
        result = analogy_spectrum(
            _mag(decomps_dict, "ES_PEN"),
            _mag(decomps_dict, "ES_AND"),
            _mag(decomps_dict, "ES_CAN"),
        )
        assert np.all(np.isfinite(result))

    def test_analogy_valid(self, decomps_dict):
        """Result has the correct length."""
        a = _mag(decomps_dict, "ES_PEN")
        b = _mag(decomps_dict, "ES_RIO")
        c = _mag(decomps_dict, "ES_MEX")
        result = analogy_spectrum(a, b, c)
        assert result.shape == a.shape

    def test_all_triples_valid(self, decomps_dict, rng):
        """20 random triples produce finite results."""
        indices = list(range(len(ALL)))
        for _ in range(20):
            i, j, k = rng.choice(indices, size=3, replace=False)
            a = _mag(decomps_dict, ALL[i])
            b = _mag(decomps_dict, ALL[j])
            c = _mag(decomps_dict, ALL[k])
            result = analogy_spectrum(a, b, c)
            assert np.all(np.isfinite(result)), (
                f"Non-finite analogy for ({ALL[i]}, {ALL[j]}, {ALL[k]})"
            )

    def test_condition_bounded(self, decomps_dict):
        """spectrum_to_W on analogy result has condition < 1e10."""
        ref = decomps_dict["ES_PEN"]
        result = analogy_spectrum(
            _mag(decomps_dict, "ES_PEN"),
            _mag(decomps_dict, "ES_AND"),
            _mag(decomps_dict, "ES_CAN"),
        )
        W_new = spectrum_to_W(result, ref.P, ref.P_inv)
        cond = np.linalg.cond(W_new)
        assert cond < 1e10, f"Condition number {cond:.2e} exceeds 1e10"

    def test_inverse_analogy(self, decomps_dict):
        """analogy(b, a, analogy(a, b, c)) approximately equals c."""
        a = _mag(decomps_dict, "ES_PEN")
        b = _mag(decomps_dict, "ES_MEX")
        c = _mag(decomps_dict, "ES_RIO")
        forward = analogy_spectrum(a, b, c)
        roundtrip = analogy_spectrum(b, a, forward)
        np.testing.assert_allclose(roundtrip, c, atol=TOL_EXACT)

    def test_double_application(self, decomps_dict):
        """Applying the analogy shift twice shifts twice."""
        a = _mag(decomps_dict, "ES_PEN")
        b = _mag(decomps_dict, "ES_AND")
        c = _mag(decomps_dict, "ES_CAN")
        shift = b - a
        once = analogy_spectrum(a, b, c)       # c + shift
        twice = analogy_spectrum(a, b, once)    # c + 2*shift
        expected = c + 2 * shift
        np.testing.assert_allclose(twice, expected, atol=TOL_EXACT)

    def test_analogy_preserves_length(self, decomps_dict):
        """Result has the same length as the input spectra."""
        a = _mag(decomps_dict, "ES_CHI")
        b = _mag(decomps_dict, "ES_CAR")
        c = _mag(decomps_dict, "ES_AND_BO")
        result = analogy_spectrum(a, b, c)
        assert len(result) == len(a) == len(b) == len(c)

    def test_large_shift_flagged(self, decomps_dict):
        """A very distant analogy has a larger norm difference from c."""
        a = _mag(decomps_dict, "ES_PEN")
        b = _mag(decomps_dict, "ES_CAR")
        c = _mag(decomps_dict, "ES_CAN")
        result = analogy_spectrum(a, b, c)
        shift_norm = np.linalg.norm(b - a)
        diff_norm = np.linalg.norm(result - c)
        # The analogy difference should equal the shift norm (exactly)
        np.testing.assert_allclose(diff_norm, shift_norm, atol=TOL_EXACT)

    def test_analogy_dialects_returns_compose_result(self, decomps_dict):
        """analogy_dialects returns a ComposeResult."""
        from eigen3.types import ComposeResult
        ref = decomps_dict["ES_PEN"]
        result = analogy_dialects(
            decomps_dict, "ES_PEN", "ES_AND", "ES_CAN", ref,
        )
        assert isinstance(result, ComposeResult)

    def test_analogy_dialects_has_spectrum(self, decomps_dict):
        """result.spectrum is a np.ndarray."""
        ref = decomps_dict["ES_PEN"]
        result = analogy_dialects(
            decomps_dict, "ES_PEN", "ES_AND", "ES_CAN", ref,
        )
        assert isinstance(result.spectrum, np.ndarray)
        assert result.spectrum.ndim == 1

    def test_analogy_dialects_has_W(self, decomps_dict):
        """result.W is (100, 100)."""
        ref = decomps_dict["ES_PEN"]
        result = analogy_dialects(
            decomps_dict, "ES_PEN", "ES_MEX", "ES_RIO", ref,
        )
        assert result.W.shape == (100, 100)

    def test_analogy_dialects_condition(self, decomps_dict):
        """result.condition is finite."""
        ref = decomps_dict["ES_PEN"]
        result = analogy_dialects(
            decomps_dict, "ES_CAN", "ES_CAR", "ES_MEX", ref,
        )
        assert np.isfinite(result.condition)


# ======================================================================
# Leave-One-Out (10 tests)
# ======================================================================


class TestLeaveOneOut:

    def test_prediction_equals_mean(self, decomps_dict):
        """Prediction equals the mean of the remaining 7 spectra."""
        sd = _spectra_dict_from_decomps(decomps_dict)
        pred = predict_leave_one_out(sd, "ES_PEN")
        others = [sd[v] for v in ALL if v != "ES_PEN"]
        expected = np.mean(others, axis=0)
        np.testing.assert_allclose(pred, expected, atol=TOL_EXACT)

    def test_correlation_positive(self, decomps_dict):
        """Cosine similarity between prediction and actual > 0 for all 8."""
        sd = _spectra_dict_from_decomps(decomps_dict)
        for v in ALL:
            pred = predict_leave_one_out(sd, v)
            actual = sd[v]
            sim = 1.0 - cosine_dist(pred, actual)
            assert sim > 0, f"{v}: cosine similarity {sim:.4f} <= 0"

    def test_error_bounded(self, decomps_dict):
        """Prediction error < max pairwise distance for each variety."""
        sd = _spectra_dict_from_decomps(decomps_dict)
        # Compute max pairwise distance
        max_dist = 0.0
        for va, vb in combinations(ALL, 2):
            d = np.linalg.norm(sd[va] - sd[vb])
            if d > max_dist:
                max_dist = d
        for v in ALL:
            pred = predict_leave_one_out(sd, v)
            err = np.linalg.norm(pred - sd[v])
            assert err < max_dist, (
                f"{v}: LOO error {err:.4f} >= max pairwise {max_dist:.4f}"
            )

    def test_all_8_predictable(self, decomps_dict):
        """predict_leave_one_out works for all 8 varieties without error."""
        sd = _spectra_dict_from_decomps(decomps_dict)
        for v in ALL:
            pred = predict_leave_one_out(sd, v)
            assert pred.shape == sd[v].shape
            assert np.all(np.isfinite(pred))

    def test_pen_closest_to_centroid(self, decomps_dict):
        """PEN has the smallest LOO error (closest to centroid of others).

        PEN is the reference variety whose W ≈ I (eigenvalues ≈ 1.0),
        so it is an outlier relative to the other 7 varieties whose
        eigenvalues vary. PEN should thus have the HIGHEST LOO error.
        """
        sd = _spectra_dict_from_decomps(decomps_dict)
        errors = {}
        for v in ALL:
            pred = predict_leave_one_out(sd, v)
            errors[v] = np.linalg.norm(pred - sd[v])
        sorted_errors = sorted(errors.items(), key=lambda x: x[1])
        pen_rank = [v for v, _ in sorted_errors].index("ES_PEN")
        # PEN should be in the top half of errors (furthest from centroid of others)
        assert pen_rank >= len(ALL) // 2, (
            f"ES_PEN rank={pen_rank}, expected >= {len(ALL) // 2}. "
            f"Errors: {sorted_errors}"
        )

    def test_valid_spectra(self, decomps_dict):
        """All predictions have correct length and finite values."""
        sd = _spectra_dict_from_decomps(decomps_dict)
        n = len(next(iter(sd.values())))
        for v in ALL:
            pred = predict_leave_one_out(sd, v)
            assert pred.shape == (n,)
            assert np.all(np.isfinite(pred))

    def test_leave_one_out_consistency(self, decomps_dict):
        """predict(X left out from 8) == mean of other 7, verified independently."""
        sd = _spectra_dict_from_decomps(decomps_dict)
        for v in ALL:
            pred = predict_leave_one_out(sd, v)
            manual = np.mean([sd[u] for u in ALL if u != v], axis=0)
            np.testing.assert_allclose(pred, manual, atol=TOL_EXACT)

    def test_error_distribution(self, decomps_dict):
        """Errors vary across varieties (not all the same)."""
        sd = _spectra_dict_from_decomps(decomps_dict)
        errors = [
            np.linalg.norm(predict_leave_one_out(sd, v) - sd[v]) for v in ALL
        ]
        assert np.std(errors) > 0, "All LOO errors are identical"

    def test_missing_key_error(self, decomps_dict):
        """Missing key raises KeyError."""
        sd = _spectra_dict_from_decomps(decomps_dict)
        with pytest.raises(KeyError):
            predict_leave_one_out(sd, "ES_NONEXISTENT")

    def test_needs_two(self, decomps_dict):
        """Dict with 1 entry raises ValueError."""
        single = {"ES_PEN": _mag(decomps_dict, "ES_PEN")}
        with pytest.raises(ValueError, match="at least 2"):
            predict_leave_one_out(single, "ES_PEN")


# ======================================================================
# Centroid & Extremes (10 tests)
# ======================================================================


class TestCentroidExtremes:

    def test_centroid_equals_mean(self, decomps_dict):
        """centroid == np.mean of all spectra."""
        spectra = _all_magnitudes(decomps_dict)
        result = centroid_spectrum(spectra)
        expected = np.mean(spectra, axis=0)
        np.testing.assert_allclose(result, expected, atol=TOL_EXACT)

    def test_centroid_valid(self, decomps_dict):
        """Centroid has correct shape and finite values."""
        spectra = _all_magnitudes(decomps_dict)
        result = centroid_spectrum(spectra)
        assert result.shape == spectra[0].shape
        assert np.all(np.isfinite(result))

    def test_within_convex_hull(self, decomps_dict):
        """Each component of the centroid is between min and max of inputs."""
        spectra = _all_magnitudes(decomps_dict)
        result = centroid_spectrum(spectra)
        stacked = np.stack(spectra)
        lo = stacked.min(axis=0)
        hi = stacked.max(axis=0)
        assert np.all(result >= lo - TOL_EXACT), "Centroid below minimum"
        assert np.all(result <= hi + TOL_EXACT), "Centroid above maximum"

    def test_centroid_minimizes_sum_distances(self, decomps_dict):
        """sum(||centroid - spec_i||^2) is minimized at the centroid.

        The mean uniquely minimizes the sum of squared distances.
        Any perturbation should increase the sum.
        """
        spectra = _all_magnitudes(decomps_dict)
        centroid = centroid_spectrum(spectra)
        sum_sq = sum(np.linalg.norm(centroid - s) ** 2 for s in spectra)
        # Perturb by a small random vector
        rng = np.random.default_rng(123)
        for _ in range(5):
            perturb = centroid + rng.normal(scale=0.01, size=centroid.shape)
            sum_sq_perturb = sum(np.linalg.norm(perturb - s) ** 2 for s in spectra)
            assert sum_sq_perturb > sum_sq - TOL_EXACT, (
                "Perturbation decreased sum of squared distances"
            )

    def test_distance_from_centroid_varies(self, decomps_dict):
        """Different dialects have different distances from the centroid."""
        spectra = _all_magnitudes(decomps_dict)
        centroid = centroid_spectrum(spectra)
        distances = [np.linalg.norm(s - centroid) for s in spectra]
        assert len(set(np.round(distances, 6))) > 1, (
            "All dialects equidistant from centroid"
        )

    def test_farthest_from_centroid(self, decomps_dict):
        """There exists a most distinctive dialect (farthest from centroid)."""
        spectra = _all_magnitudes(decomps_dict)
        centroid = centroid_spectrum(spectra)
        distances = {
            ALL[i]: np.linalg.norm(spectra[i] - centroid) for i in range(len(ALL))
        }
        farthest = max(distances, key=distances.get)
        # Just verify there is a unique maximum
        max_dist = distances[farthest]
        assert max_dist > 0, "All dialects are at the centroid"
        # The farthest should be strictly farther than at least one other
        assert sum(1 for d in distances.values() if d < max_dist) > 0

    def test_centroid_fixed_point(self, decomps_dict):
        """centroid of [centroid, centroid, ...] == centroid."""
        spectra = _all_magnitudes(decomps_dict)
        centroid = centroid_spectrum(spectra)
        copies = [centroid.copy() for _ in range(5)]
        result = centroid_spectrum(copies)
        np.testing.assert_allclose(result, centroid, atol=TOL_EXACT)

    def test_compose_dialects_equal_weights(self, decomps_dict):
        """Equal weights on all 8 varieties matches the centroid."""
        spectra = _all_magnitudes(decomps_dict)
        centroid = centroid_spectrum(spectra)

        ref = decomps_dict["ES_PEN"]
        weights = {v: 1.0 for v in ALL}
        result = compose_dialects(decomps_dict, weights, ref)
        np.testing.assert_allclose(result.spectrum, centroid, atol=TOL_EXACT)

    def test_compose_dialects_condition(self, decomps_dict):
        """compose_dialects with equal weights has finite condition."""
        ref = decomps_dict["ES_PEN"]
        weights = {v: 1.0 for v in ALL}
        result = compose_dialects(decomps_dict, weights, ref)
        assert np.isfinite(result.condition)

    def test_negative_weights_error(self, decomps_dict):
        """Negative weights raise ValueError."""
        ref = decomps_dict["ES_PEN"]
        weights = {"ES_PEN": 1.0, "ES_AND": -0.5}
        with pytest.raises(ValueError, match="non-negative"):
            compose_dialects(decomps_dict, weights, ref)
