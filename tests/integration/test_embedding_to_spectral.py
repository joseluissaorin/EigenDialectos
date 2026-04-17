"""Integration test: embedding → spectral analysis pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.types import EmbeddingMatrix, TransformationMatrix


def test_embedding_to_transform(random_embeddings):
    """Test computing transformation matrices from embeddings."""
    from eigendialectos.spectral.transformation import compute_transformation_matrix

    ref = random_embeddings[DialectCode.ES_PEN]
    target = random_embeddings[DialectCode.ES_RIO]

    W = compute_transformation_matrix(ref, target)
    assert isinstance(W, TransformationMatrix)
    assert W.data.shape[0] == W.data.shape[1]  # Square matrix
    assert W.source_dialect == DialectCode.ES_PEN
    assert W.target_dialect == DialectCode.ES_RIO


def test_transform_to_eigendecomposition(known_transform):
    """Test eigendecomposition of a known transform."""
    from eigendialectos.spectral.eigendecomposition import eigendecompose

    eigen = eigendecompose(known_transform)
    assert eigen.eigenvalues is not None
    assert eigen.eigenvectors is not None

    # Reconstruct and check
    W_reconstructed = (
        eigen.eigenvectors
        @ np.diag(eigen.eigenvalues)
        @ eigen.eigenvectors_inv
    )
    np.testing.assert_allclose(
        np.real(W_reconstructed), known_transform.data, atol=1e-10
    )


def test_eigendecomposition_to_spectrum(known_transform):
    """Test full pipeline: transform → eigen → spectrum."""
    from eigendialectos.spectral.eigendecomposition import eigendecompose
    from eigendialectos.spectral.eigenspectrum import compute_eigenspectrum
    from eigendialectos.spectral.entropy import compute_dialectal_entropy

    eigen = eigendecompose(known_transform)
    spectrum = compute_eigenspectrum(eigen)

    assert spectrum.entropy >= 0
    assert len(spectrum.eigenvalues_sorted) == len(eigen.eigenvalues)
    # Eigenvalues should be sorted descending
    for i in range(len(spectrum.eigenvalues_sorted) - 1):
        assert spectrum.eigenvalues_sorted[i] >= spectrum.eigenvalues_sorted[i + 1]


def test_full_spectral_pipeline(random_embeddings):
    """Full pipeline from embeddings to distance matrix."""
    from eigendialectos.spectral.transformation import compute_all_transforms
    from eigendialectos.spectral.eigendecomposition import eigendecompose
    from eigendialectos.spectral.eigenspectrum import compute_eigenspectrum

    # Use 3 dialects for speed
    subset = {
        code: random_embeddings[code]
        for code in [DialectCode.ES_PEN, DialectCode.ES_RIO, DialectCode.ES_AND]
    }

    transforms = compute_all_transforms(subset, reference=DialectCode.ES_PEN)
    assert len(transforms) >= 2  # At least the non-reference dialects

    for code, W in transforms.items():
        eigen = eigendecompose(W)
        spectrum = compute_eigenspectrum(eigen)
        assert spectrum.entropy >= 0
