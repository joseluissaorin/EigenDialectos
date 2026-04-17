"""End-to-end dialect generation pipeline.

Combines DIAL transforms, dialect mixing, and nearest-neighbour lookup
to convert neutral Spanish text into target dialect text at a specified
intensity.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

from eigendialectos.constants import DialectCode
from eigendialectos.generative.dial import apply_dial, dial_transform_embedding
from eigendialectos.generative.mixing import mix_dialects
from eigendialectos.types import (
    EigenDecomposition,
    EmbeddingMatrix,
    TransformationMatrix,
)


def _nearest_neighbour(
    vector: npt.NDArray[np.float64],
    reference_matrix: npt.NDArray[np.float64],
) -> int:
    """Find the index of the nearest neighbour in *reference_matrix*.

    Parameters
    ----------
    vector : ndarray, shape (d,)
        Query vector.
    reference_matrix : ndarray, shape (n, d)
        Reference embedding matrix (one row per word).

    Returns
    -------
    int
        Index of the closest row in *reference_matrix*.
    """
    # cdist expects 2-D inputs
    dists = cdist(vector.reshape(1, -1), reference_matrix, metric="cosine")
    return int(np.argmin(dists))


class DialectGenerator:
    """Generates dialectal text from neutral Spanish input.

    Parameters
    ----------
    transforms : dict mapping DialectCode to TransformationMatrix
        Pre-computed dialect transformation matrices.
    eigendecomps : dict mapping DialectCode to EigenDecomposition
        Pre-computed eigendecompositions.
    vocab : list of str
        Shared vocabulary (words in the embedding space).
    embeddings : EmbeddingMatrix
        Neutral (source) embedding matrix. Rows correspond to *vocab*.
    """

    def __init__(
        self,
        transforms: dict[DialectCode, TransformationMatrix],
        eigendecomps: dict[DialectCode, EigenDecomposition],
        vocab: list[str],
        embeddings: EmbeddingMatrix,
    ) -> None:
        self.transforms = transforms
        self.eigendecomps = eigendecomps
        self.vocab = vocab
        self.embeddings = embeddings

        # Build word-to-index map
        self._word2idx: dict[str, int] = {w: i for i, w in enumerate(vocab)}

        # Ensure embedding data is row-major (n_words x dim)
        emb_data = np.asarray(embeddings.data, dtype=np.float64)
        if emb_data.shape[0] == len(vocab):
            self._emb_matrix = emb_data
        elif emb_data.shape[1] == len(vocab):
            self._emb_matrix = emb_data.T
        else:
            raise ValueError(
                f"Embedding shape {emb_data.shape} is incompatible with "
                f"vocab size {len(vocab)}"
            )

    def _encode_word(self, word: str) -> npt.NDArray[np.float64] | None:
        """Look up the embedding vector for a word.

        Returns ``None`` if the word is not in the vocabulary.
        """
        idx = self._word2idx.get(word)
        if idx is None:
            return None
        return self._emb_matrix[idx].copy()

    def _decode_vector(
        self,
        vector: npt.NDArray[np.float64],
        reference_matrix: npt.NDArray[np.float64] | None = None,
    ) -> str:
        """Map an embedding vector to its nearest vocabulary word."""
        ref = reference_matrix if reference_matrix is not None else self._emb_matrix
        idx = _nearest_neighbour(vector, ref)
        return self.vocab[idx]

    def _get_target_embedding_matrix(
        self,
        target_dialect: DialectCode,
        alpha: float,
    ) -> npt.NDArray[np.float64]:
        """Build the target dialect embedding space.

        Applies the DIAL transform at *alpha* to the full neutral embedding
        matrix.
        """
        eigen = self.eigendecomps[target_dialect]
        return dial_transform_embedding(self._emb_matrix, eigen, alpha)

    def generate(
        self,
        text: str,
        target_dialect: DialectCode,
        alpha: float = 1.0,
        method: str = "algebraic",
    ) -> str:
        """Generate dialectal text from neutral Spanish input.

        Parameters
        ----------
        text : str
            Input text in neutral Spanish.
        target_dialect : DialectCode
            Target dialect variety.
        alpha : float
            Dialectal intensity (0 = neutral, 1 = full, >1 = hyper).
        method : str
            Generation method.  Currently only ``'algebraic'`` is supported:
            encode words, apply DIAL, decode via nearest-neighbour.

        Returns
        -------
        str
            Text transformed toward the target dialect at the given intensity.
        """
        if method != "algebraic":
            raise ValueError(
                f"Unknown generation method: {method!r}. "
                "Currently only 'algebraic' is supported."
            )

        if target_dialect not in self.eigendecomps:
            raise KeyError(
                f"No eigendecomposition available for {target_dialect.value}"
            )

        eigen = self.eigendecomps[target_dialect]
        target_matrix = self._get_target_embedding_matrix(target_dialect, alpha)

        tokens = text.split()
        result_tokens: list[str] = []

        for token in tokens:
            # Preserve punctuation and casing info
            clean = token.strip(".,;:!?\"'()[]{}").lower()
            prefix = token[: len(token) - len(token.lstrip(".,;:!?\"'()[]{}")) ]
            suffix = token[len(token.rstrip(".,;:!?\"'()[]{}")):]

            vec = self._encode_word(clean)
            if vec is None:
                # Unknown word -- pass through unchanged
                result_tokens.append(token)
                continue

            # Apply DIAL transform
            transformed = dial_transform_embedding(vec, eigen, alpha)

            # Find nearest neighbour in target space
            decoded = self._decode_vector(transformed, target_matrix)

            # Restore casing heuristic
            if token[0].isupper():
                decoded = decoded.capitalize()

            result_tokens.append(prefix + decoded + suffix)

        return " ".join(result_tokens)

    def generate_mixed(
        self,
        text: str,
        dialect_weights: dict[DialectCode, float],
        alpha: float = 1.0,
    ) -> str:
        """Generate text from a mixture of dialects.

        Parameters
        ----------
        text : str
            Input text in neutral Spanish.
        dialect_weights : dict mapping DialectCode to float
            Mixing weights (must sum to 1).
        alpha : float
            Dialectal intensity.

        Returns
        -------
        str
            Text with blended dialectal features.
        """
        # Build mixed transform
        transform_weight_pairs: list[tuple[TransformationMatrix, float]] = []
        for dialect, weight in dialect_weights.items():
            if dialect not in self.eigendecomps:
                raise KeyError(
                    f"No eigendecomposition available for {dialect.value}"
                )
            W = apply_dial(self.eigendecomps[dialect], alpha)
            transform_weight_pairs.append((W, weight))

        mixed_W = mix_dialects(transform_weight_pairs)

        # Apply mixed transform to each word
        tokens = text.split()
        result_tokens: list[str] = []

        # Build target space using mixed transform
        mixed_target = (self._emb_matrix @ mixed_W.data.T).astype(np.float64)

        for token in tokens:
            clean = token.strip(".,;:!?\"'()[]{}").lower()
            prefix = token[: len(token) - len(token.lstrip(".,;:!?\"'()[]{}")) ]
            suffix = token[len(token.rstrip(".,;:!?\"'()[]{}")):]

            vec = self._encode_word(clean)
            if vec is None:
                result_tokens.append(token)
                continue

            transformed = (mixed_W.data @ vec).astype(np.float64)
            decoded = self._decode_vector(transformed, mixed_target)

            if token[0].isupper():
                decoded = decoded.capitalize()

            result_tokens.append(prefix + decoded + suffix)

        return " ".join(result_tokens)

    def generate_gradient(
        self,
        text: str,
        target_dialect: DialectCode,
        n_steps: int = 16,
    ) -> list[tuple[float, str]]:
        """Generate a gradient of dialectal intensity from 0 to 1.

        Parameters
        ----------
        text : str
            Input text in neutral Spanish.
        target_dialect : DialectCode
            Target dialect variety.
        n_steps : int
            Number of evenly-spaced alpha values in [0, 1].

        Returns
        -------
        list of (alpha, text)
            Pairs of intensity and generated text.
        """
        alphas = np.linspace(0.0, 1.0, n_steps)
        results: list[tuple[float, str]] = []

        for a in alphas:
            generated = self.generate(text, target_dialect, alpha=float(a))
            results.append((float(a), generated))

        return results
