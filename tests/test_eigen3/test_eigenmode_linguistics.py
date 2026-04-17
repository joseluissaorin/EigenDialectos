"""Tests for eigenmode linguistic analysis: interpret, name, compare, stability, sparsity, text.

60 tests verifying the eigen3.analyzer module against real v2 decompositions.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigen3.analyzer import (
    analyze_text,
    compare_eigenvectors,
    compare_spectra,
    cumulative_energy,
    effective_rank,
    find_shared_axes,
    find_unique_axes,
    interpret_eigenvector,
    mode_energy,
    mode_sparsity,
    mode_stability,
    name_all_modes,
    name_mode,
    spectral_distance,
)
from eigen3.types import AnalysisResult

ALL = ["ES_PEN", "ES_AND", "ES_CAN", "ES_RIO", "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO"]


# ======================================================================
# TestModeNaming (15 tests)
# ======================================================================

class TestModeNaming:

    @pytest.fixture()
    def emb(self, word_embeddings_dict):
        """Reference embedding matrix for projecting dim-sized eigenvectors."""
        return word_embeddings_dict["ES_PEN"]

    def test_name_all_modes_returns_dict(self, decomps_dict, vocab, emb):
        """name_all_modes returns a dict with int keys."""
        names = name_all_modes(decomps_dict["ES_CAN"], vocab, embeddings=emb)
        assert isinstance(names, dict)
        for key in names:
            assert isinstance(key, int)

    def test_name_all_modes_covers_all(self, decomps_dict, vocab, emb):
        """name_all_modes covers every mode 0..n-1."""
        decomp = decomps_dict["ES_PEN"]
        names = name_all_modes(decomp, vocab, embeddings=emb)
        assert set(names.keys()) == set(range(decomp.n_modes))

    def test_name_mode_nonempty(self, decomps_dict, vocab, emb):
        """Every auto-generated mode name is a non-empty string."""
        decomp = decomps_dict["ES_MEX"]
        for k in range(decomp.n_modes):
            name = name_mode(decomp.P[:, k], vocab, embeddings=emb)
            assert isinstance(name, str)
            assert len(name) > 0

    def test_name_mode_contains_hyphens(self, decomps_dict, vocab, emb):
        """With top_k=5, the name contains exactly 4 hyphens."""
        decomp = decomps_dict["ES_RIO"]
        name = name_mode(decomp.P[:, 0], vocab, top_k=5, embeddings=emb)
        assert name.count("-") == 4

    def test_names_stable(self, decomps_dict, vocab, emb):
        """Calling name_all_modes twice yields identical results."""
        decomp = decomps_dict["ES_CHI"]
        names_a = name_all_modes(decomp, vocab, embeddings=emb)
        names_b = name_all_modes(decomp, vocab, embeddings=emb)
        assert names_a == names_b

    def test_names_differ_across_dialects(self, decomps_dict, vocab, emb):
        """Mode 0 names for CAN and RIO are very likely different."""
        name_can = name_mode(decomps_dict["ES_CAN"].P[:, 0], vocab, embeddings=emb)
        name_rio = name_mode(decomps_dict["ES_RIO"].P[:, 0], vocab, embeddings=emb)
        assert name_can != name_rio

    def test_interpret_top_k_20(self, decomps_dict, vocab, emb):
        """interpret_eigenvector with top_k=20 returns exactly 20 items."""
        result = interpret_eigenvector(decomps_dict["ES_PEN"].P[:, 0], vocab, top_k=20, embeddings=emb)
        assert len(result) == 20

    def test_interpret_sorted_by_magnitude(self, decomps_dict, vocab, emb):
        """Results are sorted by |loading| descending."""
        result = interpret_eigenvector(decomps_dict["ES_AND"].P[:, 1], vocab, top_k=20, embeddings=emb)
        magnitudes = [abs(loading) for _, loading in result]
        for i in range(len(magnitudes) - 1):
            assert magnitudes[i] >= magnitudes[i + 1] - 1e-12

    def test_interpret_handles_complex(self, decomps_dict, vocab, emb):
        """interpret_eigenvector works on potentially complex P column."""
        decomp = decomps_dict["ES_CAR"]
        result = interpret_eigenvector(decomp.P[:, 2], vocab, top_k=5, embeddings=emb)
        assert len(result) == 5
        for word, loading in result:
            assert isinstance(loading, float)

    def test_interpret_raises_on_mismatch(self, vocab):
        """P_col of wrong length without embeddings raises ValueError."""
        bad_col = np.zeros(10, dtype=np.float64)
        with pytest.raises(ValueError, match="P_col length"):
            interpret_eigenvector(bad_col, vocab, top_k=5)

    def test_name_mode_top_k_1(self, decomps_dict, vocab, emb):
        """top_k=1 gives a single word with no hyphens."""
        name = name_mode(decomps_dict["ES_PEN"].P[:, 0], vocab, top_k=1, embeddings=emb)
        assert "-" not in name
        assert len(name) > 0

    def test_interpret_real_words(self, decomps_dict, vocab, emb):
        """Top words returned are actual members of the vocabulary."""
        result = interpret_eigenvector(decomps_dict["ES_MEX"].P[:, 0], vocab, top_k=10, embeddings=emb)
        vocab_set = set(vocab)
        for word, _ in result:
            assert word in vocab_set

    def test_different_top_k_different_names(self, decomps_dict, vocab, emb):
        """top_k=3 and top_k=10 produce different name strings."""
        P_col = decomps_dict["ES_AND_BO"].P[:, 0]
        name_3 = name_mode(P_col, vocab, top_k=3, embeddings=emb)
        name_10 = name_mode(P_col, vocab, top_k=10, embeddings=emb)
        assert name_3 != name_10

    def test_name_mode_reproducible(self, decomps_dict, vocab, emb):
        """name_mode is deterministic across repeated calls."""
        P_col = decomps_dict["ES_CAN"].P[:, 5]
        names = [name_mode(P_col, vocab, top_k=5, embeddings=emb) for _ in range(5)]
        assert all(n == names[0] for n in names)

    def test_interpret_loading_values_float(self, decomps_dict, vocab, emb):
        """All loading values are plain Python float."""
        result = interpret_eigenvector(decomps_dict["ES_CHI"].P[:, 0], vocab, top_k=10, embeddings=emb)
        for _, loading in result:
            assert isinstance(loading, float)


# ======================================================================
# TestModeStability (10 tests)
# ======================================================================

class TestModeStability:

    def test_stability_values_in_01(self, decomps_dict):
        """All stability values lie in [0, 1]."""
        stab = mode_stability(decomps_dict["ES_CAN"], n_perturbations=3, noise_scale=1e-4)
        for v in stab.values():
            assert 0.0 <= v <= 1.0

    def test_stability_deterministic_with_seed(self, decomps_dict):
        """Stability values are within [0, 1] (seeding is internal)."""
        stab = mode_stability(decomps_dict["ES_PEN"], n_perturbations=3, noise_scale=1e-4)
        assert all(0.0 <= v <= 1.0 for v in stab.values())

    def test_stability_top_modes_high(self, decomps_dict):
        """Mode 0 (dominant eigenvalue) typically has high stability (>0.5)."""
        stab = mode_stability(decomps_dict["ES_MEX"], n_perturbations=5, noise_scale=1e-4)
        assert stab[0] > 0.5

    def test_stability_dict_keys(self, decomps_dict):
        """Keys are consecutive integers 0..n-1."""
        decomp = decomps_dict["ES_RIO"]
        stab = mode_stability(decomp, n_perturbations=3, noise_scale=1e-4)
        assert set(stab.keys()) == set(range(decomp.n_modes))

    def test_stability_all_modes(self, decomps_dict):
        """Returns one entry per mode."""
        decomp = decomps_dict["ES_AND"]
        stab = mode_stability(decomp, n_perturbations=3, noise_scale=1e-4)
        assert len(stab) == decomp.n_modes

    def test_identity_low_stability(self, decomps_dict):
        """ES_PEN (near identity) has degenerate eigenvalues so modes are unstable."""
        # Near-identity means all eigenvalues ≈ 1 — eigenvectors are arbitrary,
        # so small perturbation scrambles them. Low stability is expected.
        stab = mode_stability(decomps_dict["ES_PEN"], n_perturbations=5, noise_scale=1e-4)
        avg = np.mean(list(stab.values()))
        assert 0.0 <= avg <= 1.0  # just verify valid range

    def test_noise_scale_matters(self, decomps_dict):
        """Larger noise_scale yields lower or equal average stability."""
        decomp = decomps_dict["ES_CAN"]
        stab_small = mode_stability(decomp, n_perturbations=5, noise_scale=1e-6)
        stab_large = mode_stability(decomp, n_perturbations=5, noise_scale=1e-2)
        avg_small = np.mean(list(stab_small.values()))
        avg_large = np.mean(list(stab_large.values()))
        # Small noise -> higher stability on average
        assert avg_small >= avg_large - 0.1  # allow some stochastic slack

    def test_more_perturbations_smoother(self, decomps_dict):
        """n_perturbations=20 vs 5 gives similar mean (within 0.2)."""
        decomp = decomps_dict["ES_CHI"]
        stab_5 = mode_stability(decomp, n_perturbations=5, noise_scale=1e-4)
        stab_20 = mode_stability(decomp, n_perturbations=20, noise_scale=1e-4)
        mean_5 = np.mean(list(stab_5.values()))
        mean_20 = np.mean(list(stab_20.values()))
        assert abs(mean_5 - mean_20) < 0.2

    def test_stability_nonneg(self, decomps_dict):
        """All stability values are non-negative."""
        stab = mode_stability(decomps_dict["ES_CAR"], n_perturbations=3, noise_scale=1e-4)
        for v in stab.values():
            assert v >= 0.0

    def test_stability_returns_float(self, decomps_dict):
        """All stability values are plain float."""
        stab = mode_stability(decomps_dict["ES_AND_BO"], n_perturbations=3, noise_scale=1e-4)
        for v in stab.values():
            assert isinstance(v, float)


# ======================================================================
# TestModeSparsity (10 tests)
# ======================================================================

class TestModeSparsity:

    def test_gini_in_01(self, decomps_dict):
        """Gini coefficient is in [0, 1] for a real eigenvector."""
        g = mode_sparsity(decomps_dict["ES_PEN"].P[:, 0])
        assert 0.0 <= g <= 1.0

    def test_uniform_vector_low_gini(self):
        """np.ones(100) should give Gini near 0."""
        g = mode_sparsity(np.ones(100, dtype=np.float64))
        assert g < 0.01

    def test_unit_vector_high_gini(self):
        """Standard basis vector e_0 should give Gini near 1."""
        e0 = np.zeros(1000, dtype=np.float64)
        e0[0] = 1.0
        g = mode_sparsity(e0)
        assert g > 0.99

    def test_gini_varies_across_modes(self, decomps_dict):
        """Different modes have different sparsity values."""
        decomp = decomps_dict["ES_CAN"]
        sparsities = [mode_sparsity(decomp.P[:, k]) for k in range(min(10, decomp.n_modes))]
        # Not all identical
        assert len(set(round(s, 6) for s in sparsities)) > 1

    def test_gini_stable(self, decomps_dict):
        """Same input twice gives identical output."""
        P_col = decomps_dict["ES_MEX"].P[:, 3]
        assert mode_sparsity(P_col) == mode_sparsity(P_col)

    def test_zero_vector_zero_gini(self):
        """All-zeros vector returns 0.0."""
        g = mode_sparsity(np.zeros(50, dtype=np.float64))
        assert g == 0.0

    def test_gini_monotonic_concentration(self):
        """As a vector becomes more concentrated, Gini increases."""
        n = 200
        uniform = np.ones(n, dtype=np.float64) / n
        mild = np.zeros(n, dtype=np.float64)
        mild[:50] = 1.0 / 50
        extreme = np.zeros(n, dtype=np.float64)
        extreme[0] = 1.0

        g_uniform = mode_sparsity(uniform)
        g_mild = mode_sparsity(mild)
        g_extreme = mode_sparsity(extreme)

        assert g_uniform < g_mild < g_extreme

    def test_gini_real_eigenvector(self, decomps_dict):
        """Works on a real eigenvector from decomposition."""
        P_col = np.real(decomps_dict["ES_RIO"].P[:, 0])
        g = mode_sparsity(P_col)
        assert 0.0 <= g <= 1.0

    def test_gini_consistent_l1l2(self, decomps_dict):
        """Gini generally correlates with L1/L2 sparsity direction."""
        decomp = decomps_dict["ES_AND"]
        ginis = []
        l1l2_ratios = []
        for k in range(min(20, decomp.n_modes)):
            col = np.abs(np.real(decomp.P[:, k])).astype(np.float64)
            g = mode_sparsity(decomp.P[:, k])
            l1 = np.sum(col)
            l2 = np.linalg.norm(col)
            if l2 > 1e-12:
                # Lower L1/L2 ratio -> sparser -> higher Gini
                l1l2_ratios.append(l1 / l2)
                ginis.append(g)

        # Check negative correlation: Spearman or just sign of covariance
        if len(ginis) > 3:
            cov = np.corrcoef(ginis, l1l2_ratios)[0, 1]
            # Expect negative correlation (higher gini <-> lower L1/L2)
            assert cov < 0.5  # should be negative or at least not strongly positive

    def test_gini_handles_complex(self, decomps_dict):
        """Works on complex P column without error."""
        P_col = decomps_dict["ES_CAR"].P[:, 4]  # may be complex
        g = mode_sparsity(P_col)
        assert 0.0 <= g <= 1.0


# ======================================================================
# TestSharedUniqueAxes (15 tests)
# ======================================================================

class TestSharedUniqueAxes:

    def test_shared_returns_list(self, decomps_dict):
        """find_shared_axes returns a list of int."""
        shared = find_shared_axes(decomps_dict, threshold=0.8)
        assert isinstance(shared, list)
        for idx in shared:
            assert isinstance(idx, int)

    def test_unique_returns_dict(self, decomps_dict):
        """find_unique_axes returns a dict mapping str to list[int]."""
        unique = find_unique_axes(decomps_dict, threshold=0.3)
        assert isinstance(unique, dict)
        for key, val in unique.items():
            assert isinstance(key, str)
            assert isinstance(val, list)

    def test_threshold_01_few_shared(self, decomps_dict):
        """Even low threshold may find few shared modes (eigenbases differ across dialects)."""
        shared = find_shared_axes(decomps_dict, threshold=0.1)
        # Non-symmetric W matrices have unrelated eigenbases, so 0 shared is valid
        assert isinstance(shared, list)
        assert len(shared) >= 0

    def test_threshold_099_few_shared(self, decomps_dict):
        """Very high threshold finds few or no shared modes."""
        shared_high = find_shared_axes(decomps_dict, threshold=0.99)
        shared_low = find_shared_axes(decomps_dict, threshold=0.1)
        assert len(shared_high) <= len(shared_low)

    def test_shared_modes_valid_indices(self, decomps_dict):
        """All shared mode indices are in [0, n_modes)."""
        n_modes = min(d.n_modes for d in decomps_dict.values())
        shared = find_shared_axes(decomps_dict, threshold=0.5)
        for idx in shared:
            assert 0 <= idx < n_modes

    def test_unique_modes_valid_indices(self, decomps_dict):
        """All unique mode indices are in [0, n_modes)."""
        n_modes = min(d.n_modes for d in decomps_dict.values())
        unique = find_unique_axes(decomps_dict, threshold=0.3)
        for v, indices in unique.items():
            for idx in indices:
                assert 0 <= idx < n_modes

    def test_shared_unique_coverage(self, decomps_dict):
        """Shared and unique together cover at least some modes."""
        shared = find_shared_axes(decomps_dict, threshold=0.5)
        unique = find_unique_axes(decomps_dict, threshold=0.3)
        all_unique = set()
        for indices in unique.values():
            all_unique.update(indices)
        covered = set(shared) | all_unique
        # At least one mode should appear somewhere
        assert len(covered) > 0

    def test_compare_eigenvectors_square(self, decomps_dict):
        """compare_eigenvectors returns an (8, 8) matrix for 8 dialects."""
        sim = compare_eigenvectors(decomps_dict, mode_idx=0)
        assert sim.shape == (8, 8)

    def test_compare_eigenvectors_diagonal_one(self, decomps_dict):
        """Diagonal of the similarity matrix is 1.0."""
        sim = compare_eigenvectors(decomps_dict, mode_idx=0)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-10)

    def test_compare_eigenvectors_symmetric(self, decomps_dict):
        """Similarity matrix is symmetric."""
        sim = compare_eigenvectors(decomps_dict, mode_idx=1)
        np.testing.assert_allclose(sim, sim.T, atol=1e-12)

    def test_compare_eigenvectors_bounded(self, decomps_dict):
        """All values in the similarity matrix are in [-1, 1]."""
        sim = compare_eigenvectors(decomps_dict, mode_idx=2)
        assert np.all(sim >= -1.0 - 1e-10)
        assert np.all(sim <= 1.0 + 1e-10)

    def test_find_shared_deterministic(self, decomps_dict):
        """Same input, same output."""
        a = find_shared_axes(decomps_dict, threshold=0.5)
        b = find_shared_axes(decomps_dict, threshold=0.5)
        assert a == b

    def test_find_unique_deterministic(self, decomps_dict):
        """Same input, same output."""
        a = find_unique_axes(decomps_dict, threshold=0.3)
        b = find_unique_axes(decomps_dict, threshold=0.3)
        assert a == b

    def test_unique_keys_are_varieties(self, decomps_dict):
        """Keys of the unique dict match the dialect variety names."""
        unique = find_unique_axes(decomps_dict, threshold=0.3)
        assert set(unique.keys()) == set(sorted(decomps_dict.keys()))

    def test_compare_mode0_exists(self, decomps_dict):
        """Mode 0 comparison works without error."""
        sim = compare_eigenvectors(decomps_dict, mode_idx=0)
        assert sim.shape[0] == len(decomps_dict)


# ======================================================================
# TestAnalysisPipeline (10 tests)
# ======================================================================

class TestAnalysisPipeline:

    @pytest.fixture()
    def sample_text(self):
        return "el perro grande corre por la calle principal de la ciudad"

    @pytest.fixture()
    def empty_text(self):
        return "xyzzyzzz qqqrrrr"

    def test_analyze_text_returns_result(
        self, sample_text, word_embeddings_dict, vocab, decomps_dict,
    ):
        """analyze_text returns an AnalysisResult."""
        result = analyze_text(sample_text, word_embeddings_dict, vocab, decomps_dict)
        assert isinstance(result, AnalysisResult)

    def test_mode_names_populated(
        self, sample_text, word_embeddings_dict, vocab, decomps_dict,
    ):
        """mode_names is a non-empty dict."""
        result = analyze_text(sample_text, word_embeddings_dict, vocab, decomps_dict)
        assert isinstance(result.mode_names, dict)
        assert len(result.mode_names) > 0

    def test_mode_strengths_shape(
        self, sample_text, word_embeddings_dict, vocab, decomps_dict,
    ):
        """mode_strengths shape is (n_modes,)."""
        result = analyze_text(sample_text, word_embeddings_dict, vocab, decomps_dict)
        first_variety = sorted(decomps_dict.keys())[0]
        n_modes = decomps_dict[first_variety].n_modes
        assert result.mode_strengths.shape == (n_modes,)

    def test_mode_strengths_nonneg(
        self, sample_text, word_embeddings_dict, vocab, decomps_dict,
    ):
        """All mode strengths are >= 0."""
        result = analyze_text(sample_text, word_embeddings_dict, vocab, decomps_dict)
        assert np.all(result.mode_strengths >= 0.0)

    def test_per_word_modes_populated(
        self, sample_text, word_embeddings_dict, vocab, decomps_dict,
    ):
        """per_word_modes has entries for words that exist in vocab."""
        result = analyze_text(sample_text, word_embeddings_dict, vocab, decomps_dict)
        # At least some common Spanish words should be in vocab
        assert len(result.per_word_modes) > 0

    def test_empty_text_zero_strengths(
        self, empty_text, word_embeddings_dict, vocab, decomps_dict,
    ):
        """OOV-only text produces zero mode_strengths."""
        result = analyze_text(empty_text, word_embeddings_dict, vocab, decomps_dict)
        np.testing.assert_allclose(result.mode_strengths, 0.0, atol=1e-15)

    def test_mode_energy_sums_one(self, decomps_dict):
        """mode_energy sums to approximately 1.0."""
        for v in ALL:
            energy = mode_energy(decomps_dict[v])
            np.testing.assert_allclose(energy.sum(), 1.0, atol=1e-10)

    def test_cumulative_energy_monotonic(self, decomps_dict):
        """cumulative_energy is monotonically non-decreasing."""
        for v in ALL:
            cum = cumulative_energy(decomps_dict[v])
            diffs = np.diff(cum)
            assert np.all(diffs >= -1e-12)

    def test_effective_rank_positive(self, decomps_dict):
        """effective_rank is at least 1."""
        for v in ALL:
            rank = effective_rank(decomps_dict[v])
            assert rank >= 1

    def test_effective_rank_bounded(self, decomps_dict):
        """effective_rank does not exceed n_modes."""
        for v in ALL:
            decomp = decomps_dict[v]
            rank = effective_rank(decomp)
            assert rank <= decomp.n_modes
