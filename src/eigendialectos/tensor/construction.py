"""Tensor construction and slicing for multi-dialect representations."""

from __future__ import annotations

import numpy as np

from eigendialectos.constants import DialectCode
from eigendialectos.types import TensorDialectal, TransformationMatrix


def build_dialect_tensor(
    transforms: dict[DialectCode, TransformationMatrix],
) -> TensorDialectal:
    """Stack per-dialect transformation matrices into a 3-D tensor.

    Constructs T in R^{d x d x m} where m = number of dialect varieties.
    Matrices are ordered consistently by the enum value of each DialectCode.

    Parameters
    ----------
    transforms : dict[DialectCode, TransformationMatrix]
        Mapping from dialect code to its transformation matrix W_i.

    Returns
    -------
    TensorDialectal
        Tensor with shape (d, d, m) and corresponding dialect code ordering.

    Raises
    ------
    ValueError
        If transforms is empty or matrices have inconsistent shapes.
    """
    if not transforms:
        raise ValueError("transforms dict must be non-empty")

    # Sort by enum value for deterministic ordering
    sorted_codes = sorted(transforms.keys(), key=lambda c: c.value)

    # Validate that all matrices share the same shape
    shapes = {code: transforms[code].data.shape for code in sorted_codes}
    reference_shape = shapes[sorted_codes[0]]
    if len(reference_shape) != 2 or reference_shape[0] != reference_shape[1]:
        raise ValueError(
            f"Transformation matrices must be square; got shape {reference_shape}"
        )

    for code in sorted_codes[1:]:
        if shapes[code] != reference_shape:
            raise ValueError(
                f"Shape mismatch: {sorted_codes[0].value} has {reference_shape} "
                f"but {code.value} has {shapes[code]}"
            )

    # Stack along third axis: (d, d, m)
    stacked = np.stack([transforms[code].data for code in sorted_codes], axis=2)

    return TensorDialectal(data=stacked, dialect_codes=sorted_codes)


def extract_slice(
    tensor: TensorDialectal, dialect: DialectCode
) -> TransformationMatrix:
    """Extract a single variety's transformation matrix from the tensor.

    Parameters
    ----------
    tensor : TensorDialectal
        The multi-dialect tensor T in R^{d x d x m}.
    dialect : DialectCode
        The dialect whose matrix to extract.

    Returns
    -------
    TransformationMatrix
        The d x d matrix for the requested dialect.

    Raises
    ------
    KeyError
        If the dialect is not present in the tensor.
    """
    if dialect not in tensor.dialect_codes:
        raise KeyError(
            f"Dialect {dialect.value} not found in tensor. "
            f"Available: {[c.value for c in tensor.dialect_codes]}"
        )

    idx = tensor.dialect_codes.index(dialect)
    matrix_data = tensor.data[:, :, idx].copy()

    return TransformationMatrix(
        data=matrix_data,
        source_dialect=dialect,
        target_dialect=dialect,
        regularization=0.0,
    )
