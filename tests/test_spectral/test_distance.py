"""Tests for spectral.distance module."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.spectral.distance import (
    combined_distance,
    compute_distance_matrix,
    entropy_distance,
    frobenius_distance,
    spectral_distance,
    subspace_distance,
)
from eigendialectos.types import DialectalSpectrum, TransformationMatrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def W_pair(rng):
    """Two random 10x10 transformation matrices."""
    W_a = TransformationMatrix(
        data=rng.standard_normal((10, 10)),
        source_dialect=DialectCode.ES_PEN,
        target_dialect=DialectCode.ES_AND,
        regularization=0.01,
    )
    W_b = TransformationMatrix(
        data=rng.standard_normal((10, 10)),
        source_dialect=DialectCode.ES_PEN,
        target_dialect=DialectCode.ES_MEX,
        regularization=0.01,
    )
    return W_a, W_b


@pytest.fixture
def spectrum_pair():
    """Two dialectal spectra."""
    ev_a = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
    ev_b = np.array([4.0, 3.5, 2.5, 0.8, 0.2])

    def make_entropy(ev):
        mag = np.abs(ev)
        total = np.sum(mag)
        p = mag / total
        return -float(np.sum(p * np.log(p + 1e-10)))

    spec_a = DialectalSpectrum(
        eigenvalues_sorted=ev_a,
        entropy=make_entropy(ev_a),
        dialect_code=DialectCode.ES_AND,
    )
    spec_b = DialectalSpectrum(
        eigenvalues_sorted=ev_b,
        entropy=make_entropy(ev_b),
        dialect_code=DialectCode.ES_MEX,
    )
    return spec_a, spec_b


@pytest.fixture
def three_transforms(rng):
    """Three transforms for triangle inequality tests."""
    codes = [DialectCode.ES_AND, DialectCode.ES_MEX, DialectCode.ES_CHI]
    transforms = {}
    for code in codes:
        transforms[code] = TransformationMatrix(
            data=rng.standard_normal((8, 8)),
            source_dialect=DialectCode.ES_PEN,
            target_dialect=code,
            regularization=0.01,
        )
    return transforms


# ---------------------------------------------------------------------------
# Tests: frobenius_distance
# ---------------------------------------------------------------------------

class TestFrobeniusDistance:
    """Tests for the Frobenius distance."""

    def test_distance_to_self_is_zero(self, W_pair):
        """d(W, W) = 0."""
        W_a, _ = W_pair
        assert frobenius_distance(W_a, W_a) == pytest.approx(0.0)

    def test_symmetry(self, W_pair):
        """d(W_a, W_b) = d(W_b, W_a)."""
        W_a, W_b = W_pair
        assert frobenius_distance(W_a, W_b) == pytest.approx(
            frobenius_distance(W_b, W_a)
        )

    def test_non_negative(self, W_pair):
        """Distance should be >= 0."""
        W_a, W_b = W_pair
        assert frobenius_distance(W_a, W_b) >= 0

    def test_triangle_inequality(self, three_transforms):
        """d(A, C) <= d(A, B) + d(B, C)."""
        codes = list(three_transforms.keys())
        A, B, C = (three_transforms[c] for c in codes)
        d_AC = frobenius_distance(A, C)
        d_AB = frobenius_distance(A, B)
        d_BC = frobenius_distance(B, C)
        assert d_AC <= d_AB + d_BC + 1e-10

    def test_known_value(self):
        """Test on a known pair."""
        W_a = TransformationMatrix(
            data=np.eye(3),
            source_dialect=DialectCode.ES_PEN,
            target_dialect=DialectCode.ES_AND,
            regularization=0.0,
        )
        W_b = TransformationMatrix(
            data=np.zeros((3, 3)),
            source_dialect=DialectCode.ES_PEN,
            target_dialect=DialectCode.ES_MEX,
            regularization=0.0,
        )
        # ||I - 0||_F = sqrt(3)
        assert frobenius_distance(W_a, W_b) == pytest.approx(np.sqrt(3.0))


# ---------------------------------------------------------------------------
# Tests: spectral_distance
# ---------------------------------------------------------------------------

class TestSpectralDistance:
    """Tests for the spectral (EMD) distance."""

    def test_distance_to_self_is_zero(self, spectrum_pair):
        """d(spec, spec) = 0."""
        spec_a, _ = spectrum_pair
        assert spectral_distance(spec_a, spec_a) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self, spectrum_pair):
        """d(a, b) = d(b, a)."""
        spec_a, spec_b = spectrum_pair
        assert spectral_distance(spec_a, spec_b) == pytest.approx(
            spectral_distance(spec_b, spec_a)
        )

    def test_non_negative(self, spectrum_pair):
        """EMD should be >= 0."""
        spec_a, spec_b = spectrum_pair
        assert spectral_distance(spec_a, spec_b) >= 0


# ---------------------------------------------------------------------------
# Tests: subspace_distance
# ---------------------------------------------------------------------------

class TestSubspaceDistance:
    """Tests for the subspace distance."""

    def test_distance_to_self_is_zero(self, rng):
        """Subspace distance of a matrix with itself should be 0."""
        P = rng.standard_normal((10, 10))
        assert subspace_distance(P, P, k=5) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self, rng):
        """d(P_a, P_b) = d(P_b, P_a)."""
        Pa = rng.standard_normal((10, 10))
        Pb = rng.standard_normal((10, 10))
        assert subspace_distance(Pa, Pb, k=5) == pytest.approx(
            subspace_distance(Pb, Pa, k=5)
        )

    def test_orthogonal_subspaces_max_distance(self):
        """Two orthogonal 1-D subspaces in R^2 should have max distance."""
        Pa = np.array([[1.0], [0.0]])
        Pb = np.array([[0.0], [1.0]])
        d = subspace_distance(Pa, Pb, k=1)
        # ||e1 e1^T - e2 e2^T||_F = sqrt(2)
        assert d == pytest.approx(np.sqrt(2.0), rel=1e-5)


# ---------------------------------------------------------------------------
# Tests: entropy_distance
# ---------------------------------------------------------------------------

class TestEntropyDistance:
    """Tests for entropy distance."""

    def test_distance_to_self_is_zero(self):
        """d(H, H) = 0."""
        assert entropy_distance(1.5, 1.5) == pytest.approx(0.0)

    def test_symmetry(self):
        """d(a, b) = d(b, a)."""
        assert entropy_distance(1.0, 2.0) == pytest.approx(
            entropy_distance(2.0, 1.0)
        )

    def test_known_value(self):
        """|1.5 - 2.3| = 0.8."""
        assert entropy_distance(1.5, 2.3) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Tests: combined_distance
# ---------------------------------------------------------------------------

class TestCombinedDistance:
    """Tests for the weighted combined distance."""

    def test_self_distance_zero(self, W_pair, spectrum_pair):
        """Combined distance of object with itself should be 0."""
        W_a, _ = W_pair
        spec_a, _ = spectrum_pair
        d = combined_distance(W_a, W_a, spec_a, spec_a, spec_a.entropy, spec_a.entropy)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_custom_weights(self, W_pair, spectrum_pair):
        """Distance with only Frobenius weight should equal Frobenius distance."""
        W_a, W_b = W_pair
        spec_a, spec_b = spectrum_pair
        d = combined_distance(
            W_a, W_b, spec_a, spec_b,
            spec_a.entropy, spec_b.entropy,
            weights={"frobenius": 1.0, "spectral": 0.0, "entropy": 0.0},
        )
        expected = frobenius_distance(W_a, W_b)
        assert d == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Tests: compute_distance_matrix
# ---------------------------------------------------------------------------

class TestComputeDistanceMatrix:
    """Tests for the full pairwise distance matrix."""

    @pytest.fixture
    def full_data(self, rng):
        """Transforms, spectra, and entropies for 3 dialects."""
        codes = [DialectCode.ES_AND, DialectCode.ES_MEX, DialectCode.ES_CHI]
        transforms = {}
        spectra = {}
        entropies = {}

        for code in codes:
            data = rng.standard_normal((6, 6))
            transforms[code] = TransformationMatrix(
                data=data,
                source_dialect=DialectCode.ES_PEN,
                target_dialect=code,
                regularization=0.01,
            )
            ev = np.sort(np.abs(rng.standard_normal(6)))[::-1]
            mag = np.abs(ev)
            total = np.sum(mag)
            p = mag / total
            H = -float(np.sum(p * np.log(p + 1e-10)))
            spectra[code] = DialectalSpectrum(
                eigenvalues_sorted=ev, entropy=H, dialect_code=code
            )
            entropies[code] = H

        return transforms, spectra, entropies

    def test_shape(self, full_data):
        """Matrix should be (n, n)."""
        transforms, spectra, entropies = full_data
        D = compute_distance_matrix(transforms, spectra, entropies, method="frobenius")
        assert D.shape == (3, 3)

    def test_diagonal_is_zero(self, full_data):
        """Diagonal entries should be 0."""
        transforms, spectra, entropies = full_data
        D = compute_distance_matrix(transforms, spectra, entropies, method="frobenius")
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-15)

    def test_symmetry(self, full_data):
        """Distance matrix should be symmetric."""
        transforms, spectra, entropies = full_data
        D = compute_distance_matrix(transforms, spectra, entropies, method="frobenius")
        np.testing.assert_allclose(D, D.T)

    def test_non_negative(self, full_data):
        """All entries should be >= 0."""
        transforms, spectra, entropies = full_data
        D = compute_distance_matrix(transforms, spectra, entropies, method="combined")
        assert np.all(D >= -1e-15)

    def test_triangle_inequality_frobenius(self, full_data):
        """Frobenius distance matrix should satisfy triangle inequality."""
        transforms, spectra, entropies = full_data
        D = compute_distance_matrix(transforms, spectra, entropies, method="frobenius")
        n = D.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert D[i, k] <= D[i, j] + D[j, k] + 1e-10

    def test_unknown_method_raises(self, full_data):
        """Unknown method should raise ValueError."""
        transforms, spectra, entropies = full_data
        with pytest.raises(ValueError, match="Unknown method"):
            compute_distance_matrix(transforms, spectra, entropies, method="bogus")
