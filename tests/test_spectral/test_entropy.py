"""Tests for spectral.entropy module."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.spectral.entropy import compare_entropies, compute_dialectal_entropy
from eigendialectos.types import DialectalSpectrum


# ---------------------------------------------------------------------------
# Tests: compute_dialectal_entropy
# ---------------------------------------------------------------------------

class TestComputeDialectalEntropy:
    """Tests for the entropy computation."""

    def test_uniform_distribution(self):
        """Entropy of uniform distribution should be log(n)."""
        n = 10
        eigenvalues = np.ones(n, dtype=np.float64)
        H = compute_dialectal_entropy(eigenvalues)
        expected = np.log(n)
        assert H == pytest.approx(expected, rel=1e-6)

    def test_uniform_base2(self):
        """Entropy in base-2 of uniform distribution should be log2(n)."""
        n = 8
        eigenvalues = np.ones(n, dtype=np.float64)
        H = compute_dialectal_entropy(eigenvalues, base="2")
        expected = np.log2(n)
        assert H == pytest.approx(expected, rel=1e-6)

    def test_uniform_base10(self):
        """Entropy in base-10 of uniform distribution should be log10(n)."""
        n = 100
        eigenvalues = np.ones(n, dtype=np.float64)
        H = compute_dialectal_entropy(eigenvalues, base="10")
        expected = np.log10(n)
        assert H == pytest.approx(expected, rel=1e-6)

    def test_delta_distribution(self):
        """Entropy of a delta distribution (one non-zero) should be 0."""
        eigenvalues = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        H = compute_dialectal_entropy(eigenvalues)
        assert H == pytest.approx(0.0, abs=1e-8)

    def test_two_equal_values(self):
        """Entropy of [a, a, 0, 0, ...] should be log(2)."""
        eigenvalues = np.array([5.0, 5.0, 0.0, 0.0])
        H = compute_dialectal_entropy(eigenvalues)
        expected = np.log(2)
        assert H == pytest.approx(expected, rel=1e-6)

    def test_entropy_non_negative(self, rng):
        """Entropy should always be >= 0."""
        for _ in range(20):
            eigenvalues = np.abs(rng.standard_normal(50))
            H = compute_dialectal_entropy(eigenvalues)
            assert H >= -1e-10

    def test_entropy_at_most_log_n(self, rng):
        """Entropy should be <= log(n) for natural log."""
        n = 30
        eigenvalues = np.abs(rng.standard_normal(n))
        H = compute_dialectal_entropy(eigenvalues)
        assert H <= np.log(n) + 1e-8

    def test_accepts_spectrum_object(self, sample_spectrum):
        """Should accept a DialectalSpectrum as input."""
        H = compute_dialectal_entropy(sample_spectrum)
        assert H > 0

    def test_complex_eigenvalues(self):
        """Complex eigenvalues should be converted to magnitudes."""
        eigenvalues = np.array([1 + 1j, 1 - 1j, 2 + 0j])
        H = compute_dialectal_entropy(eigenvalues)
        assert H > 0

    def test_all_zeros(self):
        """All-zero eigenvalues should give entropy 0."""
        eigenvalues = np.zeros(5)
        H = compute_dialectal_entropy(eigenvalues)
        assert H == pytest.approx(0.0)

    def test_unknown_base_raises(self):
        """Unknown log base should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown base"):
            compute_dialectal_entropy(np.ones(5), base="e")

    def test_scaling_invariance(self):
        """Entropy should be invariant to positive scaling of eigenvalues.

        Since p_j = |lambda_j| / sum(|lambda_k|), multiplying all
        eigenvalues by a constant c > 0 does not change p_j.
        """
        eigenvalues = np.array([3.0, 2.0, 1.0, 0.5])
        H1 = compute_dialectal_entropy(eigenvalues)
        H2 = compute_dialectal_entropy(eigenvalues * 100.0)
        assert H1 == pytest.approx(H2, rel=1e-10)


# ---------------------------------------------------------------------------
# Tests: compare_entropies
# ---------------------------------------------------------------------------

class TestCompareEntropies:
    """Tests for entropy comparison across dialects."""

    def test_basic_comparison(self):
        """Check that comparison returns expected keys."""
        entropies = {
            DialectCode.ES_PEN: 1.5,
            DialectCode.ES_AND: 1.8,
            DialectCode.ES_MEX: 1.2,
        }
        result = compare_entropies(entropies)
        assert "rankings" in result
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert "range" in result
        assert "interpretation" in result

    def test_rankings_sorted_descending(self):
        """Rankings should be sorted by entropy, descending."""
        entropies = {
            DialectCode.ES_PEN: 1.0,
            DialectCode.ES_AND: 2.0,
            DialectCode.ES_MEX: 1.5,
        }
        result = compare_entropies(entropies)
        values = [v for _, v in result["rankings"]]
        assert values == sorted(values, reverse=True)

    def test_max_and_min(self):
        """Max and min should be correct."""
        entropies = {
            DialectCode.ES_PEN: 1.0,
            DialectCode.ES_AND: 3.0,
            DialectCode.ES_CHI: 2.0,
        }
        result = compare_entropies(entropies)
        assert result["max"] == (DialectCode.ES_AND, 3.0)
        assert result["min"] == (DialectCode.ES_PEN, 1.0)
        assert result["range"] == pytest.approx(2.0)

    def test_mean_and_std(self):
        """Mean and std should be numerically correct."""
        entropies = {
            DialectCode.ES_PEN: 1.0,
            DialectCode.ES_AND: 2.0,
            DialectCode.ES_MEX: 3.0,
        }
        result = compare_entropies(entropies)
        assert result["mean"] == pytest.approx(2.0)
        assert result["std"] == pytest.approx(np.std([1.0, 2.0, 3.0]))

    def test_empty_input(self):
        """Empty dict should return safe defaults."""
        result = compare_entropies({})
        assert result["rankings"] == []
        assert result["mean"] == 0.0

    def test_identical_entropies_interpretation(self):
        """When all entropies are equal, range is 0 and interpretation mentions similarity."""
        entropies = {
            DialectCode.ES_PEN: 1.5,
            DialectCode.ES_AND: 1.5,
            DialectCode.ES_MEX: 1.5,
        }
        result = compare_entropies(entropies)
        assert result["range"] == pytest.approx(0.0)
        assert "similar" in result["interpretation"].lower()
