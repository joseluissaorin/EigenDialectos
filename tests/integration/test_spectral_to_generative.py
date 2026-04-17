"""Integration test: spectral → generative pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.types import TransformationMatrix, EigenDecomposition


def test_dial_transform_alpha_zero(known_eigendecomposition):
    """DIAL at alpha=0 should give identity."""
    from eigendialectos.generative.dial import apply_dial

    W_identity = apply_dial(known_eigendecomposition, alpha=0.0)
    expected = np.eye(known_eigendecomposition.eigenvectors.shape[0])
    np.testing.assert_allclose(np.real(W_identity.data), expected, atol=1e-10)


def test_dial_transform_alpha_one(known_transform, known_eigendecomposition):
    """DIAL at alpha=1 should recover original transform."""
    from eigendialectos.generative.dial import apply_dial

    W_recovered = apply_dial(known_eigendecomposition, alpha=1.0)
    np.testing.assert_allclose(
        np.real(W_recovered.data), known_transform.data, atol=1e-10
    )


def test_dial_series_monotonic(known_eigendecomposition):
    """DIAL series should change monotonically with alpha."""
    from eigendialectos.generative.dial import compute_dial_series

    series = compute_dial_series(known_eigendecomposition, alpha_range=(0.0, 1.25, 0.25))
    assert len(series) == 5  # 0.0, 0.25, 0.5, 0.75, 1.0

    # Distance from identity should increase with alpha
    identity = np.eye(series[0].data.shape[0])
    distances = [np.linalg.norm(np.real(W.data) - identity) for W in series]
    for i in range(len(distances) - 1):
        assert distances[i] <= distances[i + 1] + 1e-10


def test_mixing_pure_weights(known_eigendecomposition):
    """Mixing with weight 1.0 for one dialect gives that dialect's transform."""
    from eigendialectos.generative.dial import apply_dial
    from eigendialectos.generative.mixing import mix_dialects

    W1 = apply_dial(known_eigendecomposition, alpha=1.0)
    # Create a trivial second transform
    W2_data = np.eye(W1.data.shape[0]) * 0.5
    W2 = TransformationMatrix(
        data=W2_data,
        source_dialect=DialectCode.ES_PEN,
        target_dialect=DialectCode.ES_AND,
        regularization=0.0,
    )

    W_mix = mix_dialects([(W1, 1.0), (W2, 0.0)])
    np.testing.assert_allclose(np.real(W_mix.data), np.real(W1.data), atol=1e-10)
