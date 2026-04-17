"""Lexical operator for identifying and extracting lexical-change subspaces.

Given known lexical substitution pairs (e.g. autobus -> guagua),
identifies the subspace where lexical changes dominate and extracts
the corresponding component from a transformation matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd

from eigendialectos.types import EmbeddingMatrix, TransformationMatrix


class LexicalOperator:
    """Operator for lexical variation subspace identification and extraction."""

    @staticmethod
    def identify_subspace(
        reference_emb: EmbeddingMatrix,
        target_emb: EmbeddingMatrix,
        word_pairs: list[tuple[str, str]],
    ) -> np.ndarray:
        """Identify the subspace where lexical changes dominate.

        Given known lexical substitution pairs, compute the difference
        vectors in embedding space and extract their principal subspace
        via SVD.

        Parameters
        ----------
        reference_emb : EmbeddingMatrix
            Embeddings for the reference dialect (rows = dims, cols = vocab).
        target_emb : EmbeddingMatrix
            Embeddings for the target dialect (same layout).
        word_pairs : list[tuple[str, str]]
            Pairs of (reference_word, target_word) representing known
            lexical substitutions.

        Returns
        -------
        np.ndarray
            Matrix of shape (d, k) whose columns span the lexical
            change subspace, where k = min(len(word_pairs), d).
        """
        ref_vocab_idx = {w: i for i, w in enumerate(reference_emb.vocab)}
        tgt_vocab_idx = {w: i for i, w in enumerate(target_emb.vocab)}

        diffs: list[np.ndarray] = []
        for ref_word, tgt_word in word_pairs:
            if ref_word not in ref_vocab_idx or tgt_word not in tgt_vocab_idx:
                continue
            ref_vec = reference_emb.data[:, ref_vocab_idx[ref_word]]
            tgt_vec = target_emb.data[:, tgt_vocab_idx[tgt_word]]
            diffs.append(tgt_vec - ref_vec)

        if not diffs:
            # Return zero subspace (single column of zeros)
            return np.zeros((reference_emb.dim, 1))

        # Stack difference vectors: (d, n_pairs)
        diff_matrix = np.column_stack(diffs)

        # SVD to extract principal directions
        U, S, _ = svd(diff_matrix, full_matrices=False)

        # Keep directions with non-negligible singular values
        threshold = 1e-10 * S[0] if S[0] > 0 else 1e-10
        k = max(1, int(np.sum(S > threshold)))

        return U[:, :k]

    @staticmethod
    def extract_component(
        W: TransformationMatrix, subspace: np.ndarray
    ) -> TransformationMatrix:
        """Extract the lexical component of a transformation.

        Projects the deviation (W - I) onto the given subspace.

        Parameters
        ----------
        W : TransformationMatrix
            Full dialect transformation.
        subspace : np.ndarray
            Subspace basis matrix (d, k).

        Returns
        -------
        TransformationMatrix
            The lexical component I + P @ (W - I) @ P.
        """
        data = W.data.astype(np.float64)
        d = data.shape[0]
        V = subspace.astype(np.float64)
        V_pinv = np.linalg.pinv(V)
        P = V @ V_pinv

        component = np.eye(d) + P @ (data - np.eye(d)) @ P

        return TransformationMatrix(
            data=component,
            source_dialect=W.source_dialect,
            target_dialect=W.target_dialect,
            regularization=0.0,
        )

    @staticmethod
    def lexical_distance(
        W_a: TransformationMatrix,
        W_b: TransformationMatrix,
        subspace: np.ndarray,
    ) -> float:
        """Compute the lexical distance between two dialect transformations.

        Measures the Frobenius norm of the difference between the lexical
        components of W_a and W_b.

        Parameters
        ----------
        W_a, W_b : TransformationMatrix
            Two dialect transformations.
        subspace : np.ndarray
            Lexical subspace basis (d, k).

        Returns
        -------
        float
            Distance >= 0.
        """
        comp_a = LexicalOperator.extract_component(W_a, subspace)
        comp_b = LexicalOperator.extract_component(W_b, subspace)
        return float(np.linalg.norm(comp_a.data - comp_b.data, "fro"))
