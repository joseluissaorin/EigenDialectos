"""Phonological operator for dialect transformation analysis.

Specialised for phonological-orthographic patterns such as seseo,
aspiration of /s/, yeismo, and other sound-level dialectal variation
as reflected in embedding space.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd

from eigendialectos.types import EmbeddingMatrix, TransformationMatrix


class PhonologicalOperator:
    """Operator for phonological variation subspace identification."""

    @staticmethod
    def identify_subspace(
        reference_emb: EmbeddingMatrix,
        target_emb: EmbeddingMatrix,
        word_pairs: list[tuple[str, str]],
    ) -> np.ndarray:
        """Identify the subspace where phonological changes dominate.

        Uses known phonological contrast pairs -- minimal pairs
        reflecting seseo (caza/casa), aspiration (estas/ehtah),
        yeismo (calle/cashe), etc. -- to extract the principal
        phonological variation directions.

        Parameters
        ----------
        reference_emb : EmbeddingMatrix
            Embeddings for the reference dialect.
        target_emb : EmbeddingMatrix
            Embeddings for the target dialect.
        word_pairs : list[tuple[str, str]]
            Phonological contrast pairs (reference_form, target_form).

        Returns
        -------
        np.ndarray
            Subspace basis matrix of shape (d, k).
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
            return np.zeros((reference_emb.dim, 1))

        diff_matrix = np.column_stack(diffs)
        U, S, _ = svd(diff_matrix, full_matrices=False)

        threshold = 1e-10 * S[0] if S[0] > 0 else 1e-10
        k = max(1, int(np.sum(S > threshold)))

        return U[:, :k]

    @staticmethod
    def extract_component(
        W: TransformationMatrix, subspace: np.ndarray
    ) -> TransformationMatrix:
        """Extract the phonological component of a transformation.

        Projects (W - I) onto the phonological subspace.

        Parameters
        ----------
        W : TransformationMatrix
            Full dialect transformation.
        subspace : np.ndarray
            Phonological subspace basis (d, k).

        Returns
        -------
        TransformationMatrix
            Component I + P @ (W - I) @ P.
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
    def phonological_distance(
        W_a: TransformationMatrix,
        W_b: TransformationMatrix,
        subspace: np.ndarray,
    ) -> float:
        """Compute the phonological distance between two transformations.

        Parameters
        ----------
        W_a, W_b : TransformationMatrix
            Two dialect transformations.
        subspace : np.ndarray
            Phonological subspace basis (d, k).

        Returns
        -------
        float
            Frobenius distance between phonological components.
        """
        comp_a = PhonologicalOperator.extract_component(W_a, subspace)
        comp_b = PhonologicalOperator.extract_component(W_b, subspace)
        return float(np.linalg.norm(comp_a.data - comp_b.data, "fro"))
