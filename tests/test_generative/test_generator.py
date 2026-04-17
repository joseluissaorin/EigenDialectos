"""End-to-end tests for the DialectGenerator pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.generative.generator import DialectGenerator
from eigendialectos.types import (
    EigenDecomposition,
    EmbeddingMatrix,
    TransformationMatrix,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

DIM = 10
VOCAB_SIZE = 20


@pytest.fixture
def vocab():
    return [f"word_{i}" for i in range(VOCAB_SIZE)]


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def neutral_embeddings(vocab, rng):
    """Neutral embedding matrix (n_words x dim)."""
    data = rng.standard_normal((VOCAB_SIZE, DIM)).astype(np.float64)
    # Normalise rows for cleaner cosine distances
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms
    return EmbeddingMatrix(
        data=data,
        vocab=vocab,
        dialect_code=DialectCode.ES_PEN,
    )


@pytest.fixture
def target_dialect():
    return DialectCode.ES_RIO


@pytest.fixture
def eigen(target_dialect, rng):
    """Eigendecomposition for the target dialect."""
    # Create a well-conditioned eigenvector matrix
    Q, _ = np.linalg.qr(rng.standard_normal((DIM, DIM)))
    Q = Q.astype(np.complex128)
    Q_inv = np.linalg.inv(Q)

    # Real positive eigenvalues close to 1 for stability
    eigenvalues = (1.0 + 0.3 * rng.standard_normal(DIM)).astype(np.complex128)
    # Ensure all magnitudes are positive
    eigenvalues = np.abs(eigenvalues) + 0.5

    return EigenDecomposition(
        eigenvalues=eigenvalues,
        eigenvectors=Q,
        eigenvectors_inv=Q_inv,
        dialect_code=target_dialect,
    )


@pytest.fixture
def transform(eigen):
    """Transformation matrix derived from eigendecomposition."""
    W = (
        eigen.eigenvectors
        @ np.diag(eigen.eigenvalues)
        @ eigen.eigenvectors_inv
    ).real.astype(np.float64)
    return TransformationMatrix(
        data=W,
        source_dialect=DialectCode.ES_PEN,
        target_dialect=eigen.dialect_code,
        regularization=0.0,
    )


@pytest.fixture
def generator(transform, eigen, vocab, neutral_embeddings, target_dialect):
    return DialectGenerator(
        transforms={target_dialect: transform},
        eigendecomps={target_dialect: eigen},
        vocab=vocab,
        embeddings=neutral_embeddings,
    )


# ------------------------------------------------------------------
# Tests: basic generation
# ------------------------------------------------------------------

class TestGenerate:
    """Basic end-to-end generation."""

    def test_generate_returns_string(self, generator, target_dialect):
        result = generator.generate("word_0 word_1 word_2", target_dialect)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_preserves_word_count(self, generator, target_dialect):
        text = "word_0 word_1 word_2 word_3"
        result = generator.generate(text, target_dialect)
        assert len(result.split()) == len(text.split())

    def test_generate_alpha_zero_identity(self, generator, target_dialect):
        """alpha=0 should map each word to itself (or nearest)."""
        text = "word_0 word_1 word_2"
        result = generator.generate(text, target_dialect, alpha=0.0)
        # At alpha=0, transform is identity, so nearest neighbour should
        # be the word itself
        assert result == text

    def test_generate_unknown_word_passthrough(self, generator, target_dialect):
        """Words not in vocab should pass through unchanged."""
        text = "word_0 UNKNOWN_WORD word_1"
        result = generator.generate(text, target_dialect, alpha=1.0)
        tokens = result.split()
        assert tokens[1] == "UNKNOWN_WORD"

    def test_generate_unknown_method(self, generator, target_dialect):
        with pytest.raises(ValueError, match="Unknown generation method"):
            generator.generate("word_0", target_dialect, method="neural")

    def test_generate_missing_dialect(self, generator):
        with pytest.raises(KeyError):
            generator.generate("word_0", DialectCode.ES_CHI)


# ------------------------------------------------------------------
# Tests: mixed generation
# ------------------------------------------------------------------

class TestGenerateMixed:
    """Dialect mixing in the generator."""

    def test_mixed_returns_string(self, generator, target_dialect, eigen, vocab, neutral_embeddings):
        # Add a second dialect
        rng = np.random.default_rng(99)
        Q2, _ = np.linalg.qr(rng.standard_normal((DIM, DIM)))
        Q2 = Q2.astype(np.complex128)
        eigenvalues2 = (np.abs(rng.standard_normal(DIM)) + 0.5).astype(np.complex128)

        eigen2 = EigenDecomposition(
            eigenvalues=eigenvalues2,
            eigenvectors=Q2,
            eigenvectors_inv=np.linalg.inv(Q2),
            dialect_code=DialectCode.ES_MEX,
        )
        W2 = (Q2 @ np.diag(eigenvalues2) @ np.linalg.inv(Q2)).real.astype(np.float64)
        tm2 = TransformationMatrix(
            data=W2,
            source_dialect=DialectCode.ES_PEN,
            target_dialect=DialectCode.ES_MEX,
            regularization=0.0,
        )

        gen = DialectGenerator(
            transforms={target_dialect: generator.transforms[target_dialect], DialectCode.ES_MEX: tm2},
            eigendecomps={target_dialect: eigen, DialectCode.ES_MEX: eigen2},
            vocab=vocab,
            embeddings=neutral_embeddings,
        )

        result = gen.generate_mixed(
            "word_0 word_1",
            {target_dialect: 0.5, DialectCode.ES_MEX: 0.5},
            alpha=1.0,
        )
        assert isinstance(result, str)
        assert len(result.split()) == 2

    def test_mixed_missing_dialect(self, generator, target_dialect):
        with pytest.raises(KeyError):
            generator.generate_mixed(
                "word_0",
                {target_dialect: 0.5, DialectCode.ES_CHI: 0.5},
            )


# ------------------------------------------------------------------
# Tests: gradient generation
# ------------------------------------------------------------------

class TestGenerateGradient:
    """Gradient across alpha from 0 to 1."""

    def test_gradient_count(self, generator, target_dialect):
        gradient = generator.generate_gradient("word_0 word_1", target_dialect, n_steps=8)
        assert len(gradient) == 8

    def test_gradient_starts_at_zero(self, generator, target_dialect):
        gradient = generator.generate_gradient("word_0", target_dialect, n_steps=5)
        alpha0, _ = gradient[0]
        assert alpha0 == pytest.approx(0.0)

    def test_gradient_ends_at_one(self, generator, target_dialect):
        gradient = generator.generate_gradient("word_0", target_dialect, n_steps=5)
        alpha_last, _ = gradient[-1]
        assert alpha_last == pytest.approx(1.0)

    def test_gradient_alphas_are_sorted(self, generator, target_dialect):
        gradient = generator.generate_gradient("word_0 word_1", target_dialect)
        alphas = [a for a, _ in gradient]
        assert alphas == sorted(alphas)

    def test_gradient_all_strings(self, generator, target_dialect):
        gradient = generator.generate_gradient("word_0 word_1", target_dialect, n_steps=4)
        for _, text in gradient:
            assert isinstance(text, str)
            assert len(text.split()) == 2


# ------------------------------------------------------------------
# Tests: embedding matrix orientation
# ------------------------------------------------------------------

class TestEmbeddingOrientation:
    """Test that the generator handles both row-major and column-major."""

    def test_column_major_embeddings(self, vocab, rng, eigen, transform, target_dialect):
        """Embedding matrix with shape (dim, vocab_size) should be transposed."""
        data = rng.standard_normal((DIM, VOCAB_SIZE)).astype(np.float64)
        norms = np.linalg.norm(data, axis=0, keepdims=True)
        data = data / norms

        emb = EmbeddingMatrix(data=data, vocab=vocab, dialect_code=DialectCode.ES_PEN)
        gen = DialectGenerator(
            transforms={target_dialect: transform},
            eigendecomps={target_dialect: eigen},
            vocab=vocab,
            embeddings=emb,
        )
        result = gen.generate("word_0 word_1", target_dialect, alpha=0.0)
        assert isinstance(result, str)
