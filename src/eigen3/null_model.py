"""Random permutation null model for eigenvalue significance testing."""

from __future__ import annotations

import logging

import numpy as np
from scipy import linalg

from eigen3.types import NullModelResult

logger = logging.getLogger(__name__)


class NullModel:
    """Permutation test: shuffle variety labels and recompute eigenvalues."""

    def __init__(self, n_permutations: int = 100, seed: int = 42):
        self.n_permutations = n_permutations
        self.seed = seed

    def run(
        self,
        embeddings: dict[str, np.ndarray],
        reference: str,
        lambda_reg: float = 1e-4,
    ) -> dict[str, NullModelResult]:
        """Run null model for all varieties.

        Shuffles variety labels, recomputes W and eigenvalues.
        Compares observed eigenvalues to null distribution.
        """
        from eigen3.transformation import compute_W

        rng = np.random.default_rng(self.seed)
        varieties = sorted(embeddings.keys())
        ref_emb = embeddings[reference]

        results: dict[str, NullModelResult] = {}

        for variety in varieties:
            if variety == reference:
                continue

            # Observed eigenvalues
            obs_W = compute_W(ref_emb, embeddings[variety], lambda_reg)
            obs_eig = np.sort(np.abs(linalg.eig(obs_W.W)[0]))[::-1]

            # Null distribution
            all_embs = list(embeddings.values())
            n_modes = obs_eig.shape[0]
            null_eigs = np.zeros((self.n_permutations, n_modes))

            for p in range(self.n_permutations):
                # Shuffle which embedding is "target"
                perm_idx = rng.permutation(len(all_embs))
                target_emb = all_embs[perm_idx[0]]
                null_W = compute_W(ref_emb, target_emb, lambda_reg)
                null_eigs[p] = np.sort(np.abs(linalg.eig(null_W.W)[0]))[::-1]

            # p-values: fraction of null eigenvalues >= observed
            p_values = np.mean(null_eigs >= obs_eig[None, :], axis=0)
            significant = [i for i in range(n_modes) if p_values[i] < 0.05]

            results[variety] = NullModelResult(
                p_values=p_values,
                significant_modes=significant,
                n_permutations=self.n_permutations,
            )
            logger.info(
                "%s: %d/%d modes significant (p<0.05)",
                variety, len(significant), n_modes,
            )

        return results
