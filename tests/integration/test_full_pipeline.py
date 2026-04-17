"""Integration test: full end-to-end pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.types import (
    EmbeddingMatrix,
    TransformationMatrix,
    EigenDecomposition,
    DialectalSpectrum,
    TensorDialectal,
)


def test_full_pipeline_with_random_data(rng):
    """Run the complete pipeline from embeddings to experiment-like output."""
    from eigendialectos.spectral.transformation import compute_transformation_matrix
    from eigendialectos.spectral.eigendecomposition import eigendecompose
    from eigendialectos.spectral.eigenspectrum import compute_eigenspectrum
    from eigendialectos.spectral.entropy import compute_dialectal_entropy
    from eigendialectos.generative.dial import apply_dial

    dim = 20
    vocab_size = 50
    vocab = [f"w{i}" for i in range(vocab_size)]
    dialects = [DialectCode.ES_PEN, DialectCode.ES_RIO, DialectCode.ES_AND]

    # Step 1: Create embeddings
    embeddings = {}
    for code in dialects:
        data = rng.standard_normal((dim, vocab_size)).astype(np.float64)
        embeddings[code] = EmbeddingMatrix(data=data, vocab=vocab, dialect_code=code)

    # Step 2: Compute transforms
    ref = embeddings[DialectCode.ES_PEN]
    transforms = {}
    for code in dialects:
        if code == DialectCode.ES_PEN:
            continue
        W = compute_transformation_matrix(ref, embeddings[code])
        transforms[code] = W
        assert W.data.shape == (dim, dim)

    # Step 3: Eigendecompose
    eigendecomps = {}
    spectra = {}
    entropies = {}
    for code, W in transforms.items():
        eigen = eigendecompose(W)
        eigendecomps[code] = eigen

        spectrum = compute_eigenspectrum(eigen)
        spectra[code] = spectrum

        entropy = compute_dialectal_entropy(spectrum.eigenvalues_sorted)
        entropies[code] = entropy
        assert entropy >= 0

    # Step 4: DIAL transform
    for code, eigen in eigendecomps.items():
        # Alpha = 0 → identity
        W0 = apply_dial(eigen, alpha=0.0)
        np.testing.assert_allclose(
            np.real(W0.data), np.eye(dim), atol=1e-8
        )

        # Alpha = 1 → original
        W1 = apply_dial(eigen, alpha=1.0)
        np.testing.assert_allclose(
            np.real(W1.data), transforms[code].data, atol=1e-8
        )

        # Alpha = 0.5 → something between identity and original
        W05 = apply_dial(eigen, alpha=0.5)
        dist_to_identity = np.linalg.norm(np.real(W05.data) - np.eye(dim))
        dist_to_full = np.linalg.norm(np.real(W05.data) - transforms[code].data)
        # Should be closer to identity than full transform
        dist_identity_to_full = np.linalg.norm(transforms[code].data - np.eye(dim))
        assert dist_to_identity < dist_identity_to_full + 1e-8

    # Step 5: Build tensor
    from eigendialectos.tensor.construction import build_dialect_tensor

    tensor = build_dialect_tensor(transforms)
    assert isinstance(tensor, TensorDialectal)
    assert tensor.data.shape[2] == len(transforms)


def test_pipeline_produces_consistent_distances(rng):
    """Distance metrics should be self-consistent."""
    from eigendialectos.spectral.transformation import compute_transformation_matrix
    from eigendialectos.spectral.distance import frobenius_distance

    dim = 15
    vocab_size = 30
    vocab = [f"w{i}" for i in range(vocab_size)]

    ref_data = rng.standard_normal((dim, vocab_size))
    ref = EmbeddingMatrix(data=ref_data, vocab=vocab, dialect_code=DialectCode.ES_PEN)

    targets = {}
    for code in [DialectCode.ES_RIO, DialectCode.ES_AND, DialectCode.ES_MEX]:
        data = ref_data + rng.standard_normal((dim, vocab_size)) * 0.1
        targets[code] = EmbeddingMatrix(data=data, vocab=vocab, dialect_code=code)

    transforms = {}
    for code, emb in targets.items():
        transforms[code] = compute_transformation_matrix(ref, emb)

    # Self-distance should be 0
    for code, W in transforms.items():
        assert frobenius_distance(W, W) < 1e-10

    # Symmetry
    codes = list(transforms.keys())
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            d_ij = frobenius_distance(transforms[codes[i]], transforms[codes[j]])
            d_ji = frobenius_distance(transforms[codes[j]], transforms[codes[i]])
            assert abs(d_ij - d_ji) < 1e-10
