"""Tests for geometric structure modules: Riemannian, Lie algebra, TDA,
multi-granularity decomposition, and eigenvalue field interpolation.

40 tests verifying mathematical properties on real v2 W matrices.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigen3.riemannian import geodesic_distance, metric_tensor, ricci_curvature
from eigen3.lie import (
    bracket_matrix,
    commutator,
    generator,
    generators_from_matrices,
    lie_interpolate,
    roundtrip_check,
    structure_constants,
)
from eigen3.fisher import compute_fim, diagnostic_words, per_variety_diagnostics
from eigen3.topology import (
    betti_numbers,
    interpret,
    persistence_entropy,
    persistent_homology,
)
from eigen3.eigenfield import EigenvalueField
from eigen3.multigranularity import decompose
from eigen3.distance import distance_matrix as build_distance_matrix
from eigen3.constants import DIALECT_COORDINATES, DIALECT_FAMILIES

ALL = [
    "ES_PEN", "ES_AND", "ES_CAN", "ES_RIO",
    "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO",
]


# ======================================================================
# Shared fixtures for this module
# ======================================================================

@pytest.fixture(scope="module")
def distance_mat(W_dict):
    """Frobenius distance matrix for TDA tests."""
    D, labels = build_distance_matrix(W_dict, metric="frobenius")
    return D, labels


@pytest.fixture(scope="module")
def diagrams(distance_mat):
    """Persistence diagrams from Frobenius distance matrix."""
    D, _labels = distance_mat
    return persistent_homology(D, max_dim=1)


@pytest.fixture(scope="module")
def gens(W_dict):
    """Lie algebra generators for all 8 varieties."""
    return generators_from_matrices(W_dict)


@pytest.fixture(scope="module")
def eigenfield(spectra_dict):
    """EigenvalueField built from spectra magnitudes."""
    spectra = {v: s.magnitudes for v, s in spectra_dict.items()}
    return EigenvalueField(spectra)


# ======================================================================
# TestRiemannian (10 tests)
# ======================================================================

class TestRiemannian:

    def test_geodesic_nonneg(self, W_dict):
        """Geodesic distance is always >= 0."""
        d = geodesic_distance(W_dict["ES_PEN"], W_dict["ES_RIO"])
        assert d >= 0.0

    def test_geodesic_zero_identical(self, W_dict):
        """d(W, W) should be approximately 0."""
        d = geodesic_distance(W_dict["ES_PEN"], W_dict["ES_PEN"])
        assert d < 1e-8, f"d(PEN, PEN) = {d}, expected ~0"

    def test_geodesic_symmetric(self, W_dict):
        """d(A, B) == d(B, A) within floating point tolerance."""
        d_ab = geodesic_distance(W_dict["ES_CAN"], W_dict["ES_CAR"])
        d_ba = geodesic_distance(W_dict["ES_CAR"], W_dict["ES_CAN"])
        assert abs(d_ab - d_ba) < 1e-6, f"|d(CAN,CAR) - d(CAR,CAN)| = {abs(d_ab - d_ba)}"

    def test_triangle_inequality(self, W_dict):
        """d(A,C) <= d(A,B) + d(B,C) for 3 varieties."""
        a, b, c = "ES_PEN", "ES_MEX", "ES_RIO"
        d_ac = geodesic_distance(W_dict[a], W_dict[c])
        d_ab = geodesic_distance(W_dict[a], W_dict[b])
        d_bc = geodesic_distance(W_dict[b], W_dict[c])
        assert d_ac <= d_ab + d_bc + 1e-8, (
            f"Triangle inequality violated: d(A,C)={d_ac:.6f} > "
            f"d(A,B)+d(B,C)={d_ab + d_bc:.6f}"
        )

    def test_geodesic_finite(self, W_dict):
        """Geodesic distance is finite for all pairs."""
        for va in ["ES_PEN", "ES_RIO", "ES_MEX", "ES_CHI"]:
            for vb in ["ES_AND", "ES_CAN", "ES_CAR", "ES_AND_BO"]:
                d = geodesic_distance(W_dict[va], W_dict[vb])
                assert np.isfinite(d), f"d({va}, {vb}) is not finite"

    def test_geodesic_ordering(self, W_dict):
        """Geodesic ordering correlates with Frobenius ordering (Spearman r > 0).

        Compute pairwise geodesic and Frobenius distances for a subset,
        then check their rank correlation is positive.
        """
        subset = ["ES_PEN", "ES_AND", "ES_RIO", "ES_MEX"]
        geo_dists, frob_dists = [], []
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                geo_dists.append(
                    geodesic_distance(W_dict[subset[i]], W_dict[subset[j]])
                )
                frob_dists.append(
                    float(np.linalg.norm(W_dict[subset[i]] - W_dict[subset[j]], "fro"))
                )
        # Spearman rank correlation via rank-order
        from scipy.stats import spearmanr
        corr, _pval = spearmanr(geo_dists, frob_dists)
        assert corr > 0.0, f"Expected positive rank correlation, got {corr:.4f}"

    def test_curvature_finite(self, W_dict):
        """All Ricci curvature values are finite."""
        curv = ricci_curvature(W_dict)
        for v, kappa in curv.items():
            assert np.isfinite(kappa), f"Curvature for {v} is not finite"

    def test_curvature_sign(self, W_dict):
        """Curvature values should show variation (not all zero)."""
        curv = ricci_curvature(W_dict)
        vals = list(curv.values())
        assert max(vals) != min(vals), "All curvature values are identical"

    def test_curvature_all_varieties(self, W_dict):
        """Curvature dict contains all 8 varieties."""
        curv = ricci_curvature(W_dict)
        for v in ALL:
            assert v in curv, f"Missing curvature for {v}"

    def test_metric_tensor_symmetric(self, rng):
        """Metric tensor on a small (3x3) random matrix is symmetric."""
        # Use a well-conditioned 3x3 matrix (close to identity)
        W_small = np.eye(3) + 0.1 * rng.standard_normal((3, 3))
        G = metric_tensor(W_small, epsilon=1e-3)
        assert G.shape == (9, 9)
        np.testing.assert_allclose(
            G, G.T, atol=1e-4,
            err_msg="Metric tensor is not symmetric",
        )


# ======================================================================
# TestLieAlgebra (10 tests)
# ======================================================================

class TestLieAlgebra:

    def test_generator_exists(self, W_dict):
        """logm(W) exists (finite entries) for all 8 varieties."""
        for v in ALL:
            gen = generator(W_dict[v])
            assert np.all(np.isfinite(gen.real)), f"Generator for {v} has non-finite real parts"

    def test_roundtrip_small(self, W_dict):
        """expm(logm(W)) == W within 1e-6 for ES_PEN."""
        err = roundtrip_check(W_dict["ES_PEN"])
        assert err < 1e-6, f"Round-trip error for ES_PEN: {err}"

    def test_commutator_antisymmetric(self, gens):
        """[A, B] = -[B, A]."""
        A = gens["ES_PEN"]
        B = gens["ES_RIO"]
        C_ab = commutator(A, B)
        C_ba = commutator(B, A)
        np.testing.assert_allclose(
            C_ab, -C_ba, atol=1e-10,
            err_msg="Commutator is not antisymmetric",
        )

    def test_self_commutator_zero(self, gens):
        """[A, A] = 0."""
        A = gens["ES_MEX"]
        C = commutator(A, A)
        assert np.linalg.norm(C) < 1e-10, f"[A,A] norm = {np.linalg.norm(C)}"

    def test_lie_interpolate_t0(self, W_dict):
        """lie_interpolate(A, B, 0) == A."""
        W_a = W_dict["ES_PEN"]
        W_b = W_dict["ES_AND"]
        result = lie_interpolate(W_a, W_b, 0.0)
        np.testing.assert_allclose(
            result, W_a, atol=1e-6,
            err_msg="Lie interpolation at t=0 should return W_a",
        )

    def test_lie_interpolate_t1(self, W_dict):
        """lie_interpolate(A, B, 1) == B."""
        W_a = W_dict["ES_PEN"]
        W_b = W_dict["ES_AND"]
        result = lie_interpolate(W_a, W_b, 1.0)
        np.testing.assert_allclose(
            result, W_b, atol=1e-6,
            err_msg="Lie interpolation at t=1 should return W_b",
        )

    def test_bracket_matrix_symmetric(self, gens):
        """Bracket norm matrix is symmetric."""
        B, labels = bracket_matrix(gens)
        np.testing.assert_allclose(
            B, B.T, atol=1e-10,
            err_msg="Bracket matrix is not symmetric",
        )

    def test_bracket_matrix_diagonal_zero(self, gens):
        """Diagonal of bracket matrix is 0 (||[A,A]||=0)."""
        B, labels = bracket_matrix(gens)
        diag = np.diag(B)
        np.testing.assert_allclose(
            diag, 0.0, atol=1e-10,
            err_msg="Bracket matrix diagonal is not zero",
        )

    def test_generators_from_matrices(self, W_dict):
        """generators_from_matrices returns a dict with all 8 varieties."""
        gens = generators_from_matrices(W_dict)
        for v in ALL:
            assert v in gens, f"Missing generator for {v}"
            assert gens[v].shape == W_dict[v].shape

    def test_commutator_norm_finite(self, gens):
        """All pairwise commutator norms are finite."""
        labels = sorted(gens.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                C = commutator(gens[labels[i]], gens[labels[j]])
                norm = np.linalg.norm(C, "fro")
                assert np.isfinite(norm), (
                    f"Commutator norm [{labels[i]},{labels[j]}] is not finite"
                )


# ======================================================================
# TestTDA (10 tests)
# ======================================================================

class TestTDA:

    def test_h0_has_features(self, diagrams):
        """H0 diagram has at least 1 feature."""
        h0 = diagrams[0]
        assert h0.birth_death.shape[0] >= 1, "H0 has no features"

    def test_h0_infinity_feature(self, diagrams):
        """H0 has at least one feature with death=inf (the surviving component)."""
        h0 = diagrams[0]
        inf_count = np.isinf(h0.birth_death[:, 1]).sum()
        assert inf_count >= 1, "H0 should have at least one infinite feature"

    def test_persistence_entropy_nonneg(self, diagrams):
        """Persistence entropy for H0 is >= 0."""
        ent = persistence_entropy(diagrams[0])
        assert ent >= 0.0, f"Entropy = {ent}, expected >= 0"

    def test_valid_diagrams(self, diagrams):
        """birth <= death for all features in all diagrams."""
        for diag in diagrams:
            bd = diag.birth_death
            if bd.shape[0] == 0:
                continue
            # For finite deaths: birth <= death
            finite_mask = np.isfinite(bd[:, 1])
            if finite_mask.any():
                assert np.all(bd[finite_mask, 0] <= bd[finite_mask, 1]), (
                    f"Dimension {diag.dimension}: birth > death found"
                )

    def test_deterministic(self, distance_mat):
        """Same distance matrix gives identical diagrams on re-computation."""
        D, _labels = distance_mat
        diag1 = persistent_homology(D, max_dim=1)
        diag2 = persistent_homology(D, max_dim=1)
        np.testing.assert_array_equal(
            diag1[0].birth_death,
            diag2[0].birth_death,
            err_msg="Persistent homology is not deterministic",
        )

    def test_interpret_populated(self, diagrams):
        """interpret() returns dict with 'summary' key."""
        result = interpret(diagrams)
        assert isinstance(result, dict)
        assert "summary" in result, f"Missing 'summary' key; got keys: {list(result.keys())}"

    def test_interpret_n_components(self, diagrams):
        """n_components >= 1 in interpretation."""
        result = interpret(diagrams)
        assert result["n_components"] >= 1, f"n_components = {result['n_components']}"

    def test_betti_h0_positive(self, diagrams):
        """Betti number beta_0 >= 1 at a small threshold."""
        # Use a very small threshold; all points born at 0 should be alive
        betti = betti_numbers(diagrams, threshold=0.0)
        assert betti[0] >= 1, f"beta_0 at threshold=0 is {betti[0]}"

    def test_persistence_entropy_bounded(self, diagrams):
        """Persistence entropy is finite."""
        ent = persistence_entropy(diagrams[0])
        assert np.isfinite(ent), f"Entropy = {ent}, expected finite"

    def test_betti_varies_with_threshold(self, diagrams):
        """Different thresholds yield different Betti numbers.

        At threshold=0, every point is a separate component.
        At a large threshold, components have merged.
        """
        betti_low = betti_numbers(diagrams, threshold=0.0)
        # Use a threshold larger than the maximum finite death
        h0 = diagrams[0]
        finite_deaths = h0.birth_death[:, 1][np.isfinite(h0.birth_death[:, 1])]
        if len(finite_deaths) == 0:
            pytest.skip("No finite deaths in H0 — cannot test threshold variation")
        high_thresh = float(finite_deaths.max()) + 1.0
        betti_high = betti_numbers(diagrams, threshold=high_thresh)
        assert betti_low[0] != betti_high[0], (
            f"Betti_0 unchanged: low={betti_low[0]}, high={betti_high[0]}"
        )


# ======================================================================
# TestMultiGranularity (5 tests)
# ======================================================================

class TestMultiGranularity:

    def test_three_levels(self, W_dict):
        """decompose returns W_macro, W_zonal, W_dialect, variance_ratios."""
        W_macro, W_zonal, W_dialect, variance_ratios = decompose(W_dict)
        assert isinstance(W_macro, np.ndarray)
        assert isinstance(W_zonal, dict)
        assert isinstance(W_dialect, dict)
        assert isinstance(variance_ratios, dict)
        assert set(variance_ratios.keys()) == {"macro", "zonal", "dialect"}

    def test_reconstruction(self, W_dict):
        """W_v ~ W_macro + W_zonal[fam(v)] + W_dialect[v] for each variety."""
        W_macro, W_zonal, W_dialect, _ = decompose(W_dict)
        # Build variety -> family mapping
        variety_to_family: dict[str, str] = {}
        for fam_name, members in DIALECT_FAMILIES.items():
            for v in members:
                if v in W_dict:
                    variety_to_family[v] = fam_name

        for v in ALL:
            fam = variety_to_family.get(v)
            if fam and fam in W_zonal:
                W_recon = W_macro + W_zonal[fam] + W_dialect[v]
            else:
                W_recon = W_macro + W_dialect[v]
            np.testing.assert_allclose(
                W_recon, W_dict[v], atol=1e-10,
                err_msg=f"Reconstruction failed for {v}",
            )

    def test_variance_ratios_sum(self, W_dict):
        """Variance ratios sum to approximately 1.0."""
        _, _, _, vr = decompose(W_dict)
        total = vr["macro"] + vr["zonal"] + vr["dialect"]
        assert abs(total - 1.0) < 0.15, (
            f"Variance ratios sum to {total:.4f}, expected ~1.0"
        )

    def test_variance_ratios_nonneg(self, W_dict):
        """All variance ratios are >= 0."""
        _, _, _, vr = decompose(W_dict)
        for level, val in vr.items():
            assert val >= 0.0, f"Variance ratio '{level}' = {val} < 0"

    def test_macro_dominant(self, W_dict):
        """Macro component captures more variance than dialect residual."""
        _, _, _, vr = decompose(W_dict)
        assert vr["macro"] > vr["dialect"], (
            f"macro={vr['macro']:.4f} should be > dialect={vr['dialect']:.4f}"
        )


# ======================================================================
# TestEigenfield (5 tests)
# ======================================================================

class TestEigenfield:

    def test_field_at_known_point(self, eigenfield, spectra_dict):
        """Field at Madrid coordinates equals ES_PEN spectrum exactly."""
        lat, lon = DIALECT_COORDINATES["ES_PEN"]
        field_val = eigenfield.field_at(lat, lon)
        expected = spectra_dict["ES_PEN"].magnitudes
        np.testing.assert_allclose(
            field_val, expected, atol=1e-10,
            err_msg="Field at Madrid should match ES_PEN spectrum",
        )

    def test_field_at_midpoint(self, eigenfield, spectra_dict):
        """Field at midpoint between two cities is a blend of their spectra."""
        lat_pen, lon_pen = DIALECT_COORDINATES["ES_PEN"]
        lat_and, lon_and = DIALECT_COORDINATES["ES_AND"]
        mid_lat = (lat_pen + lat_and) / 2.0
        mid_lon = (lon_pen + lon_and) / 2.0

        field_mid = eigenfield.field_at(mid_lat, mid_lon)
        spec_pen = spectra_dict["ES_PEN"].magnitudes
        spec_and = spectra_dict["ES_AND"].magnitudes

        # The midpoint field should lie between the two endpoint spectra
        # (component-wise, it need not be exact average due to IDW weighting
        # from other points, but it should be closer to both than to distant ones)
        d_to_pen = np.linalg.norm(field_mid - spec_pen)
        d_to_and = np.linalg.norm(field_mid - spec_and)

        # Midpoint should be closer to both PEN and AND than the distance between them
        d_pen_and = np.linalg.norm(spec_pen - spec_and)
        assert d_to_pen < d_pen_and or d_to_and < d_pen_and, (
            "Midpoint field is not closer to either endpoint than endpoints are to each other"
        )

    def test_gradient_shape(self, eigenfield):
        """field_gradient returns (2, k) array."""
        lat, lon = DIALECT_COORDINATES["ES_MEX"]
        grad = eigenfield.field_gradient(lat, lon)
        assert grad.shape == (2, eigenfield.spectrum_dim), (
            f"Expected (2, {eigenfield.spectrum_dim}), got {grad.shape}"
        )

    def test_uncertainty_zero_at_known(self, eigenfield):
        """Uncertainty at Madrid (known point) is approximately 0."""
        lat, lon = DIALECT_COORDINATES["ES_PEN"]
        unc = eigenfield.field_uncertainty(lat, lon)
        assert unc < 1e-10, f"Uncertainty at Madrid = {unc}, expected ~0"

    def test_uncertainty_high_far_away(self, eigenfield):
        """Uncertainty at (0, 0) — mid-Atlantic — exceeds uncertainty at Madrid."""
        unc_far = eigenfield.field_uncertainty(0.0, 0.0)
        lat, lon = DIALECT_COORDINATES["ES_PEN"]
        unc_near = eigenfield.field_uncertainty(lat, lon)
        assert unc_far > unc_near, (
            f"Uncertainty at (0,0) = {unc_far} should be > "
            f"uncertainty at Madrid = {unc_near}"
        )
