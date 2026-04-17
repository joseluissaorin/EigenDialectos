"""Tests for tensor construction and slicing."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.tensor.construction import build_dialect_tensor, extract_slice
from eigendialectos.types import TransformationMatrix


@pytest.fixture
def sample_transforms(rng):
    """Generate random transformation matrices for 4 dialects."""
    d = 10
    codes = [DialectCode.ES_PEN, DialectCode.ES_RIO, DialectCode.ES_MEX, DialectCode.ES_AND]
    transforms = {}
    for code in codes:
        data = rng.standard_normal((d, d))
        transforms[code] = TransformationMatrix(
            data=data,
            source_dialect=DialectCode.ES_PEN,
            target_dialect=code,
            regularization=0.0,
        )
    return transforms


class TestBuildDialectTensor:
    """Tests for build_dialect_tensor."""

    def test_correct_shape(self, sample_transforms):
        """Tensor shape must be (d, d, m)."""
        tensor = build_dialect_tensor(sample_transforms)
        d = 10
        m = len(sample_transforms)
        assert tensor.shape == (d, d, m)

    def test_consistent_dialect_ordering(self, sample_transforms):
        """Dialect codes must be sorted by enum value."""
        tensor = build_dialect_tensor(sample_transforms)
        expected = sorted(sample_transforms.keys(), key=lambda c: c.value)
        assert tensor.dialect_codes == expected

    def test_data_matches_input(self, sample_transforms):
        """Each slice must match the original matrix."""
        tensor = build_dialect_tensor(sample_transforms)
        for i, code in enumerate(tensor.dialect_codes):
            np.testing.assert_array_equal(
                tensor.data[:, :, i], sample_transforms[code].data
            )

    def test_empty_raises(self):
        """Empty transforms dict must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            build_dialect_tensor({})

    def test_shape_mismatch_raises(self, rng):
        """Inconsistent matrix shapes must raise ValueError."""
        transforms = {
            DialectCode.ES_PEN: TransformationMatrix(
                data=rng.standard_normal((5, 5)),
                source_dialect=DialectCode.ES_PEN,
                target_dialect=DialectCode.ES_PEN,
                regularization=0.0,
            ),
            DialectCode.ES_RIO: TransformationMatrix(
                data=rng.standard_normal((6, 6)),
                source_dialect=DialectCode.ES_PEN,
                target_dialect=DialectCode.ES_RIO,
                regularization=0.0,
            ),
        }
        with pytest.raises(ValueError, match="Shape mismatch"):
            build_dialect_tensor(transforms)

    def test_non_square_raises(self, rng):
        """Non-square matrices must raise ValueError."""
        transforms = {
            DialectCode.ES_PEN: TransformationMatrix(
                data=rng.standard_normal((5, 3)),
                source_dialect=DialectCode.ES_PEN,
                target_dialect=DialectCode.ES_PEN,
                regularization=0.0,
            ),
        }
        with pytest.raises(ValueError, match="square"):
            build_dialect_tensor(transforms)

    def test_single_dialect(self, rng):
        """Single dialect produces (d, d, 1) tensor."""
        d = 8
        transforms = {
            DialectCode.ES_CHI: TransformationMatrix(
                data=rng.standard_normal((d, d)),
                source_dialect=DialectCode.ES_PEN,
                target_dialect=DialectCode.ES_CHI,
                regularization=0.0,
            ),
        }
        tensor = build_dialect_tensor(transforms)
        assert tensor.shape == (d, d, 1)
        assert tensor.dialect_codes == [DialectCode.ES_CHI]


class TestExtractSlice:
    """Tests for extract_slice."""

    def test_round_trip(self, sample_transforms):
        """Build tensor, extract each slice, compare to original."""
        tensor = build_dialect_tensor(sample_transforms)
        for code in sample_transforms:
            extracted = extract_slice(tensor, code)
            np.testing.assert_array_almost_equal(
                extracted.data, sample_transforms[code].data
            )

    def test_missing_dialect_raises(self, sample_transforms):
        """Extracting absent dialect must raise KeyError."""
        tensor = build_dialect_tensor(sample_transforms)
        with pytest.raises(KeyError, match="ES_CAN"):
            extract_slice(tensor, DialectCode.ES_CAN)

    def test_extracted_is_copy(self, sample_transforms):
        """Extracted matrix must be an independent copy."""
        tensor = build_dialect_tensor(sample_transforms)
        code = tensor.dialect_codes[0]
        extracted = extract_slice(tensor, code)
        extracted.data[0, 0] = 999.0
        assert tensor.data[0, 0, 0] != 999.0
