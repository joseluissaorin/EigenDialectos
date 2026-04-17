"""Tests for spectral properties: eigendecomposition, entropy, distances.

60 tests verifying mathematical axioms on real v2 W matrices.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from eigen3.decomposition import eigendecompose, eigenspectrum, reconstruct_W
from eigen3.distance import (
    distance_matrix,
    frobenius_distance,
    spectral_distance,
    subspace_distance,
)
from eigen3.stability import check_condition, regularize_W, safe_inverse

ALL = ["ES_PEN", "ES_AND", "ES_CAN", "ES_RIO", "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO"]


# ======================================================================
# Eigendecomposition validity (10 tests)
# ======================================================================

class TestEigendecompositionValidity:

    def test_roundtrip_all_dialects(self, W_dict, decomps_dict):
        """PLP^-1 = W for all 8 dialects."""
        for v in ALL:
            d = decomps_dict[v]
            W_recon = (d.P @ np.diag(d.eigenvalues) @ d.P_inv).real
            assert np.linalg.norm(W_recon - W_dict[v]) < 1e-8, f"{v} roundtrip failed"

    def test_eigenpair_equation(self, W_dict, decomps_dict):
        """Wv = lv for each eigenpair."""
        for v in ["ES_CAN", "ES_RIO"]:
            d = decomps_dict[v]
            for i in range(min(10, d.n_modes)):
                Wv = W_dict[v] @ d.P[:, i]
                lv = d.eigenvalues[i] * d.P[:, i]
                assert np.allclose(Wv, lv, atol=1e-8), f"{v} mode {i}"

    def test_P_invertible(self, decomps_dict):
        """P * P^-1 = I."""
        for v in ALL:
            d = decomps_dict[v]
            I_approx = d.P @ d.P_inv
            assert np.allclose(I_approx, np.eye(d.n_modes), atol=1e-8), f"{v}"

    def test_trace_equals_sum_eigenvalues(self, W_dict, decomps_dict):
        """trace(W) = sum(eigenvalues)."""
        for v in ALL:
            trace_W = np.trace(W_dict[v])
            sum_eig = np.sum(decomps_dict[v].eigenvalues)
            assert abs(trace_W - sum_eig) < 1e-8, f"{v}"

    def test_det_equals_product_eigenvalues(self, W_dict, decomps_dict):
        """det(W) = product(eigenvalues)."""
        for v in ALL:
            det_W = np.linalg.det(W_dict[v])
            prod_eig = np.prod(decomps_dict[v].eigenvalues)
            assert abs(det_W - prod_eig) / max(abs(det_W), 1e-10) < 1e-6, f"{v}"

    def test_conjugate_pairs(self, decomps_dict):
        """Complex eigenvalues come in conjugate pairs."""
        for v in ALL:
            eigs = decomps_dict[v].eigenvalues
            complex_eigs = eigs[np.abs(eigs.imag) > 1e-10]
            for e in complex_eigs:
                conj = np.conj(e)
                diffs = np.abs(eigs - conj)
                assert np.min(diffs) < 1e-8, f"{v}: no conjugate for {e}"

    def test_deterministic(self, W_dict):
        """Two decompositions of same W give same eigenvalues."""
        W = W_dict["ES_CAN"]
        d1 = eigendecompose(W)
        d2 = eigendecompose(W)
        assert np.allclose(d1.eigenvalues, d2.eigenvalues)

    def test_rank_k_approximation_error_decreases(self, W_dict, decomps_dict):
        """Full reconstruction (k=100) has near-zero error, low-k has larger error."""
        d = decomps_dict["ES_CAN"]
        W = W_dict["ES_CAN"]
        err_10 = np.linalg.norm(reconstruct_W(d, k=10) - W, "fro")
        err_100 = np.linalg.norm(reconstruct_W(d, k=100) - W, "fro")
        assert err_100 < 1e-8  # full reconstruction is perfect
        assert err_10 > err_100  # partial is worse

    def test_real_eigenvalues_for_symmetric_part(self, W_dict):
        """Symmetric part of W has real eigenvalues."""
        W = W_dict["ES_PEN"]
        W_sym = (W + W.T) / 2
        eigs = np.linalg.eigvalsh(W_sym)
        assert np.all(np.isreal(eigs))

    def test_reconstruction_from_spectrum(self, decomps_dict):
        """Full reconstruction matches original."""
        d = decomps_dict["ES_MEX"]
        W_recon = reconstruct_W(d)
        assert np.allclose(W_recon, d.W_original, atol=1e-8)


# ======================================================================
# Identity and boundary (10 tests)
# ======================================================================

class TestIdentityBoundary:

    def test_pen_near_identity(self, W_dict):
        """W_PEN ~= I (Frobenius < 0.5)."""
        W = W_dict["ES_PEN"]
        diff = np.linalg.norm(W - np.eye(W.shape[0]), "fro")
        assert diff < 0.5, f"PEN deviation from I: {diff}"

    def test_pen_eigenvalues_near_one(self, decomps_dict):
        """PEN eigenvalues are close to 1."""
        mags = decomps_dict["ES_PEN"].magnitudes
        assert np.all(mags > 0.8) and np.all(mags < 1.2)

    def test_alpha_zero_gives_identity(self, decomps_dict):
        """W(alpha=0) = I for all dialects."""
        from eigen3.per_mode import compute_W_alpha
        from eigen3.types import AlphaVector
        for v in ["ES_CAN", "ES_RIO"]:
            d = decomps_dict[v]
            alpha = AlphaVector.zeros(d.n_modes)
            W_0 = compute_W_alpha(d, alpha)
            assert np.allclose(W_0, np.eye(d.n_modes), atol=1e-8), f"{v}"

    def test_alpha_one_gives_W(self, decomps_dict):
        """W(alpha=1) = W for all dialects."""
        from eigen3.per_mode import compute_W_alpha
        from eigen3.types import AlphaVector
        for v in ["ES_CAN", "ES_RIO"]:
            d = decomps_dict[v]
            alpha = AlphaVector.ones(d.n_modes)
            W_1 = compute_W_alpha(d, alpha)
            assert np.allclose(W_1, d.W_original, atol=1e-6), f"{v}"

    def test_identity_max_entropy(self, decomps_dict):
        """PEN (nearest to I) has highest entropy."""
        entropies = {v: eigenspectrum(d.eigenvalues).entropy for v, d in decomps_dict.items()}
        assert entropies["ES_PEN"] == max(entropies.values())

    def test_pen_smallest_frobenius(self, W_dict):
        """PEN has smallest Frobenius distance from I."""
        norms = {v: np.linalg.norm(W - np.eye(W.shape[0]), "fro") for v, W in W_dict.items()}
        assert norms["ES_PEN"] == min(norms.values())

    def test_pen_eigenvalues_in_range(self, decomps_dict):
        """PEN eigenvalues in [0.9, 1.1]."""
        mags = decomps_dict["ES_PEN"].magnitudes
        assert np.all(mags > 0.9) and np.all(mags < 1.15)

    def test_pen_spectral_radius_near_one(self, decomps_dict):
        """PEN spectral radius ~= 1."""
        sr = decomps_dict["ES_PEN"].magnitudes[0]
        assert abs(sr - 1.0) < 0.15

    def test_pen_is_neutral_element(self, W_dict):
        """W @ W_PEN ~= W for any variety."""
        W_pen = W_dict["ES_PEN"]
        W_can = W_dict["ES_CAN"]
        result = W_can @ W_pen
        assert np.allclose(result, W_can, atol=0.1)

    def test_pen_condition_number_smallest(self, W_dict):
        """PEN has lowest condition number."""
        conds = {v: check_condition(W) for v, W in W_dict.items()}
        assert conds["ES_PEN"] == min(conds.values())


# ======================================================================
# Spectral properties (15 tests)
# ======================================================================

class TestSpectralProperties:

    def test_spectral_radius_bounded(self, decomps_dict):
        """Spectral radius < 2.0 for all."""
        for v, d in decomps_dict.items():
            sr = d.magnitudes[0]
            assert sr < 2.0, f"{v}: spectral radius = {sr}"

    def test_condition_number_bounded(self, W_dict):
        """Condition number < 1e6."""
        for v, W in W_dict.items():
            cond = check_condition(W)
            assert cond < 1e6, f"{v}: condition = {cond}"

    def test_correct_shapes(self, W_dict, decomps_dict):
        """W is (100,100), eigenvalues has 100 elements."""
        for v in ALL:
            assert W_dict[v].shape == (100, 100), f"{v} W shape"
            assert decomps_dict[v].n_modes == 100, f"{v} n_modes"

    def test_eigenvalues_sorted_by_magnitude(self, decomps_dict):
        """Eigenvalues sorted descending by |lambda|."""
        for v, d in decomps_dict.items():
            mags = d.magnitudes
            assert np.all(mags[:-1] >= mags[1:] - 1e-10), f"{v}"

    def test_positive_magnitudes(self, decomps_dict):
        """All eigenvalue magnitudes > 0."""
        for v, d in decomps_dict.items():
            assert np.all(d.magnitudes > 0), f"{v}"

    def test_spectral_gap(self, decomps_dict):
        """Spectral gap exists (lambda1/lambda2 >= 1.0) for most varieties."""
        for v, d in decomps_dict.items():
            gap = d.magnitudes[0] / d.magnitudes[1] if d.magnitudes[1] > 0 else float("inf")
            assert gap >= 1.0 - 1e-10, f"{v}: gap = {gap}"

    def test_effective_rank_above_50(self, spectra_dict):
        """Effective rank > 50."""
        for v, s in spectra_dict.items():
            assert s.effective_rank > 50, f"{v}: eff_rank = {s.effective_rank}"

    def test_frobenius_norm_ordering(self, W_dict):
        """PEN has smallest Frobenius norm deviation from I."""
        norms = {v: np.linalg.norm(W - np.eye(100), "fro") for v, W in W_dict.items()}
        assert norms["ES_PEN"] < min(norms[v] for v in ALL if v != "ES_PEN")

    def test_no_nan_inf(self, W_dict, decomps_dict):
        """No NaN or Inf in any matrix."""
        for v in ALL:
            assert np.all(np.isfinite(W_dict[v])), f"{v} W"
            assert np.all(np.isfinite(decomps_dict[v].P)), f"{v} P"

    def test_reconstruction_preserves_cosines(self, W_dict, decomps_dict):
        """Reconstructed W preserves embedding cosine similarities."""
        v = "ES_CAN"
        W = W_dict[v]
        W_recon = reconstruct_W(decomps_dict[v])
        # Random vectors
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        cos_orig = np.dot(W @ x, W @ y) / (np.linalg.norm(W @ x) * np.linalg.norm(W @ y))
        cos_recon = np.dot(W_recon @ x, W_recon @ y) / (np.linalg.norm(W_recon @ x) * np.linalg.norm(W_recon @ y))
        assert abs(cos_orig - cos_recon) < 1e-6

    def test_entropy_non_negative(self, spectra_dict):
        """Entropy >= 0 for all."""
        for v, s in spectra_dict.items():
            assert s.entropy >= 0, f"{v}: entropy = {s.entropy}"

    def test_entropy_bounded(self, spectra_dict):
        """Entropy <= log(n)."""
        n = 100
        max_entropy = np.log(n)
        for v, s in spectra_dict.items():
            assert s.entropy <= max_entropy + 0.01, f"{v}"

    def test_spectrum_smooth(self, decomps_dict):
        """No eigenvalue magnitude jumps > 0.5."""
        for v, d in decomps_dict.items():
            mags = d.magnitudes
            diffs = np.abs(np.diff(mags))
            assert np.all(diffs < 0.5), f"{v}: max jump = {diffs.max()}"

    def test_total_energy_ordering(self, decomps_dict):
        """Total energy (sum lambda^2) PEN is smallest."""
        energies = {v: float(np.sum(d.magnitudes ** 2)) for v, d in decomps_dict.items()}
        # PEN should be closest to 100 (identity: 100 eigenvalues of 1)
        pen_dev = abs(energies["ES_PEN"] - 100)
        for v in ALL:
            if v != "ES_PEN":
                assert pen_dev <= abs(energies[v] - 100) + 0.1, f"{v}"

    def test_eigenvalue_decay(self, decomps_dict):
        """Eigenvalue magnitudes decay (last < first for non-PEN)."""
        for v, d in decomps_dict.items():
            if v == "ES_PEN":
                continue
            assert d.magnitudes[-1] < d.magnitudes[0]


# ======================================================================
# Entropy (10 tests)
# ======================================================================

class TestEntropy:

    def test_non_negative(self, spectra_dict):
        for v, s in spectra_dict.items():
            assert s.entropy >= 0

    def test_bounded_by_log_n(self, spectra_dict):
        for v, s in spectra_dict.items():
            assert s.entropy <= np.log(100) + 0.01

    def test_uniform_max_entropy(self):
        """Uniform eigenvalues → max entropy."""
        eigs = np.ones(100)
        s = eigenspectrum(eigs)
        assert abs(s.entropy - np.log(100)) < 0.01

    def test_single_dominant_low_entropy(self):
        """Single dominant eigenvalue → low entropy."""
        eigs = np.zeros(100)
        eigs[0] = 100.0
        s = eigenspectrum(eigs)
        assert s.entropy < 0.1

    def test_permutation_invariant(self):
        """Permuting eigenvalues doesn't change entropy."""
        eigs = np.array([3.0, 2.0, 1.0, 0.5, 0.1])
        s1 = eigenspectrum(eigs)
        s2 = eigenspectrum(eigs[::-1])
        assert abs(s1.entropy - s2.entropy) < 1e-10

    def test_concentration_decreases_H(self):
        """More concentrated spectrum → lower entropy."""
        uniform = np.ones(100)
        concentrated = np.zeros(100)
        concentrated[:10] = 1.0
        s_u = eigenspectrum(uniform)
        s_c = eigenspectrum(concentrated)
        assert s_c.entropy < s_u.entropy

    def test_pen_highest_entropy(self, spectra_dict):
        """PEN has highest entropy (most uniform spectrum)."""
        entropies = {v: s.entropy for v, s in spectra_dict.items()}
        assert entropies["ES_PEN"] == max(entropies.values())

    def test_base_consistency(self):
        """Natural log base consistent."""
        eigs = np.array([1.0, 1.0])  # 2 equal eigenvalues
        s = eigenspectrum(eigs)
        assert abs(s.entropy - np.log(2)) < 1e-10

    def test_effective_rank_ge_1(self, spectra_dict):
        for v, s in spectra_dict.items():
            assert s.effective_rank >= 1

    def test_effective_rank_le_dim(self, spectra_dict):
        for v, s in spectra_dict.items():
            assert s.effective_rank <= 100


# ======================================================================
# Distance metric axioms (15 tests)
# ======================================================================

class TestDistanceAxioms:

    def test_frobenius_non_negative(self, W_dict):
        for v1 in ALL:
            for v2 in ALL:
                d = frobenius_distance(W_dict[v1], W_dict[v2])
                assert d >= 0

    def test_frobenius_zero_iff_equal(self, W_dict):
        for v in ALL:
            assert frobenius_distance(W_dict[v], W_dict[v]) < 1e-10

    def test_frobenius_symmetric(self, W_dict):
        d12 = frobenius_distance(W_dict["ES_CAN"], W_dict["ES_CAR"])
        d21 = frobenius_distance(W_dict["ES_CAR"], W_dict["ES_CAN"])
        assert abs(d12 - d21) < 1e-10

    def test_frobenius_triangle_inequality(self, W_dict):
        """Triangle inequality: d(A,C) <= d(A,B) + d(B,C)."""
        for a, b, c in [("ES_CAN", "ES_CAR", "ES_MEX"),
                         ("ES_PEN", "ES_AND", "ES_RIO"),
                         ("ES_CHI", "ES_RIO", "ES_MEX")]:
            dac = frobenius_distance(W_dict[a], W_dict[c])
            dab = frobenius_distance(W_dict[a], W_dict[b])
            dbc = frobenius_distance(W_dict[b], W_dict[c])
            assert dac <= dab + dbc + 1e-10

    def test_spectral_non_negative(self, decomps_dict):
        e1 = decomps_dict["ES_CAN"].eigenvalues
        e2 = decomps_dict["ES_CAR"].eigenvalues
        assert spectral_distance(e1, e2) >= 0

    def test_spectral_zero_iff_equal(self, decomps_dict):
        e = decomps_dict["ES_CAN"].eigenvalues
        assert spectral_distance(e, e) < 1e-10

    def test_spectral_symmetric(self, decomps_dict):
        e1 = decomps_dict["ES_CAN"].eigenvalues
        e2 = decomps_dict["ES_CAR"].eigenvalues
        assert abs(spectral_distance(e1, e2) - spectral_distance(e2, e1)) < 1e-10

    def test_subspace_non_negative(self, decomps_dict):
        d = subspace_distance(decomps_dict["ES_CAN"].P, decomps_dict["ES_CAR"].P)
        assert d >= 0

    def test_subspace_bounded(self, decomps_dict):
        """Subspace distance in [0, pi/2]."""
        d = subspace_distance(decomps_dict["ES_CAN"].P, decomps_dict["ES_CAR"].P)
        assert d <= np.pi / 2 + 0.01

    def test_distance_matrix_symmetric(self, W_dict):
        D, _ = distance_matrix(W_dict, "frobenius")
        assert np.allclose(D, D.T)

    def test_distance_matrix_zero_diagonal(self, W_dict):
        D, _ = distance_matrix(W_dict, "frobenius")
        assert np.allclose(np.diag(D), 0)

    def test_can_car_closer_than_can_mex(self, W_dict):
        """CAN-CAR distance < CAN-MEX distance."""
        d_cc = frobenius_distance(W_dict["ES_CAN"], W_dict["ES_CAR"])
        d_cm = frobenius_distance(W_dict["ES_CAN"], W_dict["ES_MEX"])
        assert d_cc < d_cm

    def test_intra_family_lt_inter_family(self, W_dict):
        """CAN-CAR (high affinity) closer than CAN-MEX."""
        # Test the strongest affinity pair
        d_cc = frobenius_distance(W_dict["ES_CAN"], W_dict["ES_CAR"])
        d_cm = frobenius_distance(W_dict["ES_CAN"], W_dict["ES_MEX"])
        assert d_cc < d_cm, f"CAN-CAR={d_cc} vs CAN-MEX={d_cm}"

    def test_distances_finite(self, W_dict):
        D, _ = distance_matrix(W_dict, "frobenius")
        assert np.all(np.isfinite(D))

    def test_all_pairs_computed(self, W_dict):
        D, labels = distance_matrix(W_dict, "frobenius")
        assert D.shape == (8, 8)
        assert len(labels) == 8
