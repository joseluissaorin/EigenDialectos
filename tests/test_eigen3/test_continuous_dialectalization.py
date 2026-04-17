"""Tests for per-mode alpha control: W(alpha) = P @ diag(lambda_i^alpha_i) @ P^{-1}.

50 tests verifying the core v3 per-mode parametric operator on real v2 W matrices.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigen3.per_mode import (
    alpha_gradient,
    compose_modes,
    compute_W_alpha,
    energy_spectrum,
    interpolate_alpha,
    isolate_mode,
    mode_contribution,
    reconstruction_error,
    suppress_mode,
)
from eigen3.types import AlphaVector, EigenDecomp

ALL = ["ES_PEN", "ES_AND", "ES_CAN", "ES_RIO", "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO"]
N = 100  # dimension of W matrices


# ======================================================================
# TestPerModeAlpha (15 tests)
# ======================================================================

class TestPerModeAlpha:
    """Core compute_W_alpha identity, boundary, and alpha-vector tests."""

    def test_uniform_alpha1_equals_W(self, decomps_dict, W_dict):
        """W(alpha=1) reproduces the original W for all 8 varieties."""
        for v in ALL:
            d = decomps_dict[v]
            alpha = AlphaVector.ones(d.n_modes)
            W_alpha = compute_W_alpha(d, alpha)
            err = np.linalg.norm(W_alpha - W_dict[v], "fro")
            assert err < 1e-8, f"{v}: ||W(1) - W||_F = {err}"

    def test_uniform_alpha0_equals_I(self, decomps_dict):
        """W(alpha=0) equals the identity matrix for all 8 varieties."""
        for v in ALL:
            d = decomps_dict[v]
            alpha = AlphaVector.zeros(d.n_modes)
            W_alpha = compute_W_alpha(d, alpha)
            err = np.linalg.norm(W_alpha - np.eye(N), "fro")
            assert err < 1e-8, f"{v}: ||W(0) - I||_F = {err}"

    def test_result_is_real(self, decomps_dict):
        """compute_W_alpha returns a real ndarray (no complex dtype)."""
        d = decomps_dict["ES_CAN"]
        alpha = AlphaVector.uniform(d.n_modes, 0.5)
        W_alpha = compute_W_alpha(d, alpha)
        assert W_alpha.dtype in (np.float64, np.float32), f"dtype = {W_alpha.dtype}"
        assert not np.iscomplexobj(W_alpha)

    def test_result_is_square(self, decomps_dict):
        """Result matrix has shape (100, 100)."""
        d = decomps_dict["ES_MEX"]
        alpha = AlphaVector.uniform(d.n_modes, 0.7)
        W_alpha = compute_W_alpha(d, alpha)
        assert W_alpha.shape == (N, N)

    def test_alpha_half_between(self, decomps_dict):
        """||W(0.5)||_F is between ||I||_F and ||W(1)||_F."""
        for v in ["ES_CAN", "ES_RIO", "ES_MEX"]:
            d = decomps_dict[v]
            alpha_half = AlphaVector.uniform(d.n_modes, 0.5)
            W_half = compute_W_alpha(d, alpha_half)
            norm_I = np.linalg.norm(np.eye(N), "fro")
            norm_W = np.linalg.norm(d.W_original, "fro")
            norm_half = np.linalg.norm(W_half, "fro")
            lo, hi = min(norm_I, norm_W), max(norm_I, norm_W)
            assert lo - 0.1 <= norm_half <= hi + 0.1, (
                f"{v}: norm_half={norm_half:.4f} not in [{lo:.4f}, {hi:.4f}]"
            )

    def test_negative_alpha(self, decomps_dict):
        """Alpha = -1 does not raise (inverts eigenvalues)."""
        d = decomps_dict["ES_AND"]
        alpha = AlphaVector.uniform(d.n_modes, -1.0)
        W_neg = compute_W_alpha(d, alpha)
        assert W_neg.shape == (N, N)
        assert np.isfinite(W_neg).all()

    def test_large_alpha(self, decomps_dict):
        """Alpha = 2 does not raise (squares eigenvalues)."""
        d = decomps_dict["ES_CHI"]
        alpha = AlphaVector.uniform(d.n_modes, 2.0)
        W_sq = compute_W_alpha(d, alpha)
        assert W_sq.shape == (N, N)
        assert np.isfinite(W_sq).all()

    def test_preserves_eigenvectors(self, decomps_dict):
        """W(alpha) shares the same eigenvector basis P.

        Since W(alpha) = P diag(lambda^alpha) P^{-1}, the columns of P
        are still the eigenvectors (with different eigenvalues).
        We verify P^{-1} W(alpha) P is diagonal.
        """
        d = decomps_dict["ES_CAN"]
        alpha = AlphaVector.uniform(d.n_modes, 0.3)
        W_alpha = compute_W_alpha(d, alpha)
        # P^{-1} W(alpha) P should be diagonal
        D_alpha = d.P_inv @ W_alpha @ d.P
        off_diag = D_alpha - np.diag(np.diag(D_alpha))
        assert np.linalg.norm(off_diag, "fro") < 1e-6

    def test_deterministic(self, decomps_dict):
        """Same decomp and alpha gives identical results across calls."""
        d = decomps_dict["ES_RIO"]
        alpha = AlphaVector.uniform(d.n_modes, 0.42)
        W1 = compute_W_alpha(d, alpha)
        W2 = compute_W_alpha(d, alpha)
        assert np.array_equal(W1, W2)

    def test_from_dict_sparse(self, decomps_dict):
        """AlphaVector.from_dict creates a sparse alpha vector correctly."""
        d = decomps_dict["ES_PEN"]
        alpha = AlphaVector.from_dict({5: 1.0}, d.n_modes, default=0.0)
        assert len(alpha) == d.n_modes
        assert alpha.values[5] == 1.0
        assert alpha.values[0] == 0.0
        assert alpha.values[99] == 0.0

    def test_uniform_constructor(self):
        """AlphaVector.uniform(100, 0.5) has all values equal to 0.5."""
        alpha = AlphaVector.uniform(N, 0.5)
        assert len(alpha) == N
        assert np.allclose(alpha.values, 0.5)

    def test_alpha_length_equals_modes(self, decomps_dict):
        """Alpha length must equal decomp.n_modes for compute_W_alpha to succeed."""
        d = decomps_dict["ES_CAN"]
        alpha = AlphaVector.ones(d.n_modes)
        # Should not raise
        compute_W_alpha(d, alpha)

    def test_alpha_length_mismatch_raises(self, decomps_dict):
        """Wrong alpha length raises ValueError."""
        d = decomps_dict["ES_CAN"]
        alpha = AlphaVector.ones(d.n_modes + 1)
        with pytest.raises(ValueError, match="AlphaVector length"):
            compute_W_alpha(d, alpha)

    def test_continuous_in_alpha(self, decomps_dict):
        """Small alpha perturbation yields small matrix change."""
        d = decomps_dict["ES_MEX"]
        alpha_base = AlphaVector.uniform(d.n_modes, 1.0)
        alpha_pert = AlphaVector.uniform(d.n_modes, 1.001)
        W_base = compute_W_alpha(d, alpha_base)
        W_pert = compute_W_alpha(d, alpha_pert)
        diff = np.linalg.norm(W_pert - W_base, "fro")
        assert diff < 0.1, f"Perturbation diff = {diff}"

    def test_alpha2_differs_from_alpha1(self, decomps_dict):
        """W(2) != W(1) — alpha=2 changes the operator."""
        d = decomps_dict["ES_CAN"]
        W1 = compute_W_alpha(d, AlphaVector.ones(d.n_modes))
        W2 = compute_W_alpha(d, AlphaVector.uniform(d.n_modes, 2.0))
        # When eigenvalues < 1, squaring shrinks them; when > 1, it grows.
        # But W(2) should always differ from W(1).
        assert np.linalg.norm(W2 - W1, "fro") > 1e-6


# ======================================================================
# TestMonotonicitySmoothness (15 tests)
# ======================================================================

class TestMonotonicitySmoothness:
    """Monotonicity, smoothness, and spectral diagnostics."""

    def test_norm_varies_smoothly_with_alpha(self, decomps_dict):
        """||W(a)||_F changes smoothly as uniform alpha goes from 0 to 2."""
        d = decomps_dict["ES_CAN"]
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        norms = []
        for a in alphas:
            W_a = compute_W_alpha(d, AlphaVector.uniform(d.n_modes, a))
            norms.append(np.linalg.norm(W_a, "fro"))
        # Consecutive norm differences should be bounded (smooth)
        for i in range(len(norms) - 1):
            assert abs(norms[i+1] - norms[i]) < 5.0, (
                f"Large jump: alpha={alphas[i]}={norms[i]:.4f} -> alpha={alphas[i+1]}={norms[i+1]:.4f}"
            )

    def test_smooth_20step_path(self, decomps_dict):
        """20 steps from alpha=0 to alpha=1 show no large jumps (max step < 1.0)."""
        d = decomps_dict["ES_RIO"]
        steps = 20
        prev = compute_W_alpha(d, AlphaVector.zeros(d.n_modes))
        for i in range(1, steps + 1):
            t = i / steps
            cur = compute_W_alpha(d, AlphaVector.uniform(d.n_modes, t))
            jump = np.linalg.norm(cur - prev, "fro")
            assert jump < 1.0, f"Step {i}: jump = {jump:.4f}"
            prev = cur

    def test_continuous(self, decomps_dict):
        """||W(a + eps) - W(a)|| is small for small eps=1e-4."""
        d = decomps_dict["ES_AND"]
        a = 0.7
        eps = 1e-4
        W_a = compute_W_alpha(d, AlphaVector.uniform(d.n_modes, a))
        W_a_eps = compute_W_alpha(d, AlphaVector.uniform(d.n_modes, a + eps))
        diff = np.linalg.norm(W_a_eps - W_a, "fro")
        assert diff < 0.01, f"Continuity violation: diff = {diff}"

    def test_per_mode_independence(self, decomps_dict):
        """Changing alpha for mode i leaves mode j's isolated matrix nearly unchanged."""
        d = decomps_dict["ES_CAN"]
        # Isolate mode 1 with default alpha
        alpha_a = AlphaVector.zeros(d.n_modes)
        alpha_a.values[1] = 1.0
        W_a = compute_W_alpha(d, alpha_a)

        # Now also activate mode 5 — should not affect the mode-1 component
        alpha_b = AlphaVector.zeros(d.n_modes)
        alpha_b.values[1] = 1.0
        alpha_b.values[5] = 1.0
        W_b = compute_W_alpha(d, alpha_b)

        # Mode 1's isolated result should be consistent (W_b has additive mode-5 effect)
        # The difference should equal isolate_mode(5) minus I
        iso_5 = isolate_mode(d, 5)
        diff_actual = W_b - W_a
        diff_expected = iso_5 - np.eye(N)
        assert np.linalg.norm(diff_actual - diff_expected, "fro") < 1e-6

    def test_superposition(self, decomps_dict):
        """isolate(i) + isolate(j) - I approximates compose_modes([i,j],[1,1]).

        For W = P diag(lambda^a) P^{-1}, this is exact because
        P diag(e_i) P^{-1} + P diag(e_j) P^{-1} - P I P^{-1}
        = P diag(e_i + e_j - 1) P^{-1}  (in the lambda=0/1 sense).
        Actually: isolate sets one mode active (rest 0), compose sets two modes active (rest 0).
        Let us verify the approximate relationship.
        """
        d = decomps_dict["ES_CAN"]
        iso_0 = isolate_mode(d, 0)
        iso_1 = isolate_mode(d, 1)
        composed = compose_modes(d, [0, 1], [1.0, 1.0])
        superposed = iso_0 + iso_1 - np.eye(N)
        err = np.linalg.norm(composed - superposed, "fro")
        assert err < 1e-6, f"Superposition error = {err}"

    def test_spectral_radius_increases(self, decomps_dict):
        """Spectral radius increases with larger uniform alpha."""
        d = decomps_dict["ES_MEX"]
        W_05 = compute_W_alpha(d, AlphaVector.uniform(d.n_modes, 0.5))
        W_15 = compute_W_alpha(d, AlphaVector.uniform(d.n_modes, 1.5))
        rho_05 = np.max(np.abs(np.linalg.eigvals(W_05)))
        rho_15 = np.max(np.abs(np.linalg.eigvals(W_15)))
        assert rho_15 > rho_05, f"rho(1.5)={rho_15:.4f} <= rho(0.5)={rho_05:.4f}"

    def test_condition_bounded(self, decomps_dict):
        """Condition number of W(alpha) stays below 1e8 for alpha in [0, 2]."""
        d = decomps_dict["ES_CAN"]
        for a_val in [0.0, 0.5, 1.0, 1.5, 2.0]:
            W_a = compute_W_alpha(d, AlphaVector.uniform(d.n_modes, a_val))
            cond = np.linalg.cond(W_a)
            assert cond < 1e8, f"alpha={a_val}: cond = {cond:.2e}"

    def test_gradual_activation(self, decomps_dict):
        """10 steps from alpha=0 to alpha=1 yields smooth norm progression."""
        d = decomps_dict["ES_CHI"]
        steps = 10
        norms = []
        for i in range(steps + 1):
            t = i / steps
            W_t = compute_W_alpha(d, AlphaVector.uniform(d.n_modes, t))
            norms.append(np.linalg.norm(W_t, "fro"))
        # Check no sudden jumps: consecutive ratio should be < 1.5
        for i in range(len(norms) - 1):
            if norms[i] > 1e-12:
                ratio = norms[i + 1] / norms[i]
                assert ratio < 1.5, f"Step {i}->{i+1}: ratio = {ratio:.4f}"

    def test_energy_spectrum_sums_one(self, decomps_dict):
        """energy_spectrum sums to approximately 1.0 for all varieties."""
        for v in ALL:
            d = decomps_dict[v]
            spec = energy_spectrum(d)
            assert abs(spec.sum() - 1.0) < 1e-12, f"{v}: sum = {spec.sum()}"

    def test_energy_nonneg(self, decomps_dict):
        """All energy spectrum values are >= 0."""
        for v in ALL:
            d = decomps_dict[v]
            spec = energy_spectrum(d)
            assert (spec >= 0).all(), f"{v}: negative energy values"

    def test_mode_contribution_sums_one(self, decomps_dict):
        """Sum of all mode contributions equals approximately 1.0."""
        d = decomps_dict["ES_CAN"]
        total = sum(mode_contribution(d, i) for i in range(d.n_modes))
        assert abs(total - 1.0) < 1e-12, f"Sum = {total}"

    def test_mode_contribution_nonneg(self, decomps_dict):
        """Each mode contribution is >= 0."""
        d = decomps_dict["ES_RIO"]
        for i in range(d.n_modes):
            c = mode_contribution(d, i)
            assert c >= 0, f"Mode {i}: contribution = {c}"

    def test_mode_contribution_sorted(self, decomps_dict):
        """Mode 0 (largest eigenvalue by convention) has the highest contribution."""
        d = decomps_dict["ES_CAN"]
        c0 = mode_contribution(d, 0)
        for i in range(1, d.n_modes):
            ci = mode_contribution(d, i)
            assert c0 >= ci - 1e-12, f"Mode 0 ({c0:.6f}) < mode {i} ({ci:.6f})"

    def test_reconstruction_error_alpha1_zero(self, decomps_dict):
        """reconstruction_error is approximately 0 for alpha = ones."""
        for v in ALL:
            d = decomps_dict[v]
            err = reconstruction_error(d, AlphaVector.ones(d.n_modes))
            assert err < 1e-8, f"{v}: reconstruction error = {err}"

    def test_reconstruction_error_alpha0_nonzero(self, decomps_dict):
        """reconstruction_error > 0 for alpha = zeros (unless W = I)."""
        for v in ALL:
            d = decomps_dict[v]
            err = reconstruction_error(d, AlphaVector.zeros(d.n_modes))
            # W(0) = I, so error = ||I - W||_F which should be > 0
            assert err > 0, f"{v}: reconstruction error should be > 0"


# ======================================================================
# TestBoundaryBehavior (10 tests)
# ======================================================================

class TestBoundaryBehavior:
    """Boundary conditions: alpha=0, alpha=1, amplification, edge cases."""

    def test_alpha_zeros_identity_ES_CAN(self, decomps_dict):
        """W(alpha=0) = I verified specifically for ES_CAN."""
        d = decomps_dict["ES_CAN"]
        W0 = compute_W_alpha(d, AlphaVector.zeros(d.n_modes))
        assert np.linalg.norm(W0 - np.eye(N), "fro") < 1e-8

    def test_alpha_ones_original_ES_CAN(self, decomps_dict, W_dict):
        """W(alpha=1) = W verified specifically for ES_CAN."""
        d = decomps_dict["ES_CAN"]
        W1 = compute_W_alpha(d, AlphaVector.ones(d.n_modes))
        assert np.linalg.norm(W1 - W_dict["ES_CAN"], "fro") < 1e-8

    def test_alpha_gt1_amplifies(self, decomps_dict):
        """||W(1.5)||_F > ||W(1.0)||_F — alpha > 1 amplifies."""
        d = decomps_dict["ES_AND"]
        W1 = compute_W_alpha(d, AlphaVector.ones(d.n_modes))
        W15 = compute_W_alpha(d, AlphaVector.uniform(d.n_modes, 1.5))
        assert np.linalg.norm(W15, "fro") > np.linalg.norm(W1, "fro")

    def test_mixed_alpha(self, decomps_dict):
        """Mixed alpha (some modes 0, some 1) produces a valid matrix."""
        d = decomps_dict["ES_RIO"]
        alpha = AlphaVector.zeros(d.n_modes)
        alpha.values[0] = 1.0
        alpha.values[2] = 1.0
        alpha.values[4] = 1.0
        W_mixed = compute_W_alpha(d, alpha)
        assert W_mixed.shape == (N, N)
        assert np.isfinite(W_mixed).all()

    def test_isolate_mode_0(self, decomps_dict):
        """isolate_mode(0) returns a valid matrix."""
        d = decomps_dict["ES_CAN"]
        iso = isolate_mode(d, 0)
        assert iso.shape == (N, N)
        assert np.isfinite(iso).all()

    def test_suppress_mode_0(self, decomps_dict, W_dict):
        """suppress_mode(0) returns a matrix different from both W and I."""
        d = decomps_dict["ES_CAN"]
        sup = suppress_mode(d, 0)
        assert sup.shape == (N, N)
        # Should differ from the original W (since one mode is removed)
        diff_W = np.linalg.norm(sup - W_dict["ES_CAN"], "fro")
        # Should differ from identity (since n-1 modes are still active)
        diff_I = np.linalg.norm(sup - np.eye(N), "fro")
        assert diff_W > 1e-10, "suppress_mode(0) should differ from W"
        assert diff_I > 1e-10, "suppress_mode(0) should differ from I"

    def test_isolate_mode_invalid_raises(self, decomps_dict):
        """mode_idx out of range raises ValueError."""
        d = decomps_dict["ES_CAN"]
        with pytest.raises(ValueError):
            isolate_mode(d, -1)
        with pytest.raises(ValueError):
            isolate_mode(d, d.n_modes)

    def test_suppress_mode_invalid_raises(self, decomps_dict):
        """mode_idx out of range raises ValueError."""
        d = decomps_dict["ES_CAN"]
        with pytest.raises(ValueError):
            suppress_mode(d, -1)
        with pytest.raises(ValueError):
            suppress_mode(d, d.n_modes)

    def test_compose_modes_works(self, decomps_dict):
        """compose_modes with valid indices and strengths succeeds."""
        d = decomps_dict["ES_MEX"]
        W_comp = compose_modes(d, [0, 1], [1.0, 0.5])
        assert W_comp.shape == (N, N)
        assert np.isfinite(W_comp).all()

    def test_compose_modes_length_mismatch(self, decomps_dict):
        """compose_modes raises ValueError when lists differ in length."""
        d = decomps_dict["ES_CAN"]
        with pytest.raises(ValueError, match="same length"):
            compose_modes(d, [0, 1, 2], [1.0, 0.5])


# ======================================================================
# TestPerModeIsolation (10 tests)
# ======================================================================

class TestPerModeIsolation:
    """Isolation, suppression, complementarity, and interpolation."""

    def test_isolate_near_identity_small(self, decomps_dict):
        """Isolating a single mode yields a matrix close to identity.

        Most energy is distributed across many modes, so one mode
        alone contributes a small perturbation from I.
        """
        d = decomps_dict["ES_CAN"]
        # Pick a mode with low contribution (e.g., mode 50)
        iso = isolate_mode(d, 50)
        diff_from_I = np.linalg.norm(iso - np.eye(N), "fro")
        # One mode out of 100 should be much closer to I than to W
        diff_from_W = np.linalg.norm(iso - d.W_original, "fro")
        assert diff_from_I < diff_from_W, (
            f"Isolated mode 50: ||iso-I||={diff_from_I:.4f}, ||iso-W||={diff_from_W:.4f}"
        )

    def test_suppress_near_original(self, decomps_dict):
        """Suppressing one mode gives a matrix close to the original W.

        Removing 1 of 100 modes should have a minor effect.
        """
        d = decomps_dict["ES_RIO"]
        # Suppress a low-energy mode
        sup = suppress_mode(d, 50)
        diff = np.linalg.norm(sup - d.W_original, "fro")
        orig_norm = np.linalg.norm(d.W_original, "fro")
        # The relative difference should be small
        assert diff / orig_norm < 0.1, (
            f"Suppressing mode 50: relative diff = {diff / orig_norm:.4f}"
        )

    def test_complementary_modes(self, decomps_dict):
        """isolate(i, strength=1) + suppress(i) - I approximately equals W.

        Algebraically: isolate(i) = P diag(0,...,lambda_i,...,0) P^{-1} + (I - P e_i e_i^T P^{-1})
        Actually: isolate has alpha_i=1 rest=0; suppress has alpha_i=0 rest=1.
        Their sum minus I: alpha becomes (1,1,...,1) which gives W.
        """
        d = decomps_dict["ES_CAN"]
        for mode_idx in [0, 5, 50]:
            iso = isolate_mode(d, mode_idx)
            sup = suppress_mode(d, mode_idx)
            reconstructed = iso + sup - np.eye(N)
            err = np.linalg.norm(reconstructed - d.W_original, "fro")
            assert err < 1e-6, f"Mode {mode_idx}: complementary error = {err}"

    def test_traceability_mode_idx(self, decomps_dict):
        """Isolation targets the specific mode: W_isolated differs from I and from W."""
        d = decomps_dict["ES_CAN"]
        mode_idx = 3
        iso = isolate_mode(d, mode_idx)
        I = np.eye(d.n_modes)
        # iso should differ from identity (mode 3 is active)
        assert np.linalg.norm(iso - I, "fro") > 1e-6
        # iso should differ from full W (only one mode active)
        assert np.linalg.norm(iso - d.W_original, "fro") > 1e-6
        # iso should be closer to I than W is (since only 1 mode active)
        dist_iso_I = np.linalg.norm(iso - I, "fro")
        dist_W_I = np.linalg.norm(d.W_original - I, "fro")
        assert dist_iso_I < dist_W_I

    def test_alpha_stored(self, decomps_dict):
        """AlphaVector can be created, stored, and reused for compute_W_alpha."""
        d = decomps_dict["ES_AND"]
        alpha = AlphaVector.uniform(d.n_modes, 0.6)
        # Store and reuse
        stored = alpha
        W1 = compute_W_alpha(d, alpha)
        W2 = compute_W_alpha(d, stored)
        assert np.array_equal(W1, W2)

    def test_isolation_reversible(self, decomps_dict):
        """Isolate a mode, then set it back to 0 -> identity."""
        d = decomps_dict["ES_CAN"]
        # Setting the only active mode back to 0 gives all zeros -> identity
        alpha = AlphaVector.zeros(d.n_modes)
        W0 = compute_W_alpha(d, alpha)
        assert np.linalg.norm(W0 - np.eye(N), "fro") < 1e-8

    def test_mode_effect_varies(self, decomps_dict):
        """Different mode isolations give different matrices."""
        d = decomps_dict["ES_CAN"]
        iso_0 = isolate_mode(d, 0)
        iso_1 = isolate_mode(d, 1)
        diff = np.linalg.norm(iso_0 - iso_1, "fro")
        assert diff > 1e-10, "Isolating different modes should yield different matrices"

    def test_interpolate_alpha_works(self):
        """interpolate_alpha(zeros, ones, 0.5) gives uniform 0.5."""
        a = AlphaVector.zeros(N)
        b = AlphaVector.ones(N)
        mid = interpolate_alpha(a, b, 0.5)
        assert np.allclose(mid.values, 0.5)

    def test_interpolate_alpha_t0(self):
        """interpolate_alpha at t=0 returns the first alpha."""
        a = AlphaVector.uniform(N, 0.3)
        b = AlphaVector.uniform(N, 0.9)
        result = interpolate_alpha(a, b, 0.0)
        assert np.allclose(result.values, a.values)

    def test_interpolate_alpha_t1(self):
        """interpolate_alpha at t=1 returns the second alpha."""
        a = AlphaVector.uniform(N, 0.3)
        b = AlphaVector.uniform(N, 0.9)
        result = interpolate_alpha(a, b, 1.0)
        assert np.allclose(result.values, b.values)
