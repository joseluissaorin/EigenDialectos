"""VecMap-style iterative self-learning alignment.

Implements the approach from Artetxe et al. (2018):
1. Start from an initial seed dictionary (or Procrustes on anchors).
2. Iteratively:
   a. Compute the Procrustes alignment W from the current dictionary.
   b. Use W to translate source embeddings and build a new dictionary
      from mutual nearest neighbours.
   c. Repeat until convergence.

Reference
---------
Artetxe, M., Labaka, G., & Agirre, E. (2018). "A robust self-learning
method for fully unsupervised cross-lingual mappings of word embeddings."
*ACL*.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from eigendialectos.embeddings.contrastive.procrustes import ProcrustesAligner
from eigendialectos.types import EmbeddingMatrix

logger = logging.getLogger(__name__)


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def _build_dictionary_nn(
    src_aligned: np.ndarray,
    tgt: np.ndarray,
    src_vocab: list[str],
    tgt_vocab: list[str],
    k: int = 1,
) -> list[str]:
    """Build a seed dictionary via nearest-neighbour matching.

    Returns the list of anchor words that are mutual nearest
    neighbours in both directions.
    """
    # Normalise for cosine similarity
    src_n = _l2_normalize(src_aligned)
    tgt_n = _l2_normalize(tgt)

    # Similarity matrix: (n_src, n_tgt)
    sim = src_n @ tgt_n.T

    # Forward: for each source word, find nearest target
    fwd = np.argmax(sim, axis=1)
    # Backward: for each target word, find nearest source
    bwd = np.argmax(sim, axis=0)

    # Mutual nearest neighbours
    anchors: list[str] = []
    src_word2idx = {w: i for i, w in enumerate(src_vocab)}
    tgt_word2idx = {w: i for i, w in enumerate(tgt_vocab)}

    for si, ti in enumerate(fwd):
        if bwd[ti] == si:
            sw = src_vocab[si]
            tw = tgt_vocab[ti]
            # Only keep if same word (cross-dialect: same lemma)
            if sw == tw:
                anchors.append(sw)

    return anchors


class VecMapAligner:
    """VecMap-style iterative self-learning cross-lingual aligner.

    Parameters
    ----------
    max_iter:
        Maximum number of self-learning iterations.
    tol:
        Convergence tolerance on the change in objective.
    normalize:
        Whether to normalise embeddings before Procrustes.
    stochastic_interval:
        Apply stochastic dictionary induction every N iterations
        (adds some noise to escape local minima).
    """

    def __init__(
        self,
        max_iter: int = 50,
        tol: float = 1e-6,
        normalize: bool = True,
        stochastic_interval: int = 0,
    ) -> None:
        self._max_iter = max_iter
        self._tol = tol
        self._normalize = normalize
        self._stochastic_interval = stochastic_interval
        self._W: np.ndarray | None = None
        self._history: list[dict[str, Any]] = []

    def align(
        self,
        source: EmbeddingMatrix,
        target: EmbeddingMatrix,
        anchors: list[str] | None = None,
    ) -> np.ndarray:
        """Compute the alignment via iterative self-learning.

        Parameters
        ----------
        source:
            Source dialect embedding matrix.
        target:
            Target (reference) dialect embedding matrix.
        anchors:
            Initial seed dictionary.  If ``None``, the vocabulary
            intersection is used.

        Returns
        -------
        np.ndarray
            Orthogonal alignment matrix W of shape ``(d, d)``.
        """
        proc = ProcrustesAligner(normalize=self._normalize)

        # Initial alignment using seed anchors
        current_anchors = anchors
        if current_anchors is None:
            current_anchors = sorted(set(source.vocab) & set(target.vocab))

        if len(current_anchors) == 0:
            raise ValueError(
                "No seed anchors available. "
                "VecMap requires at least some vocabulary overlap or "
                "explicit seed pairs."
            )

        prev_objective = float("inf")
        self._history = []

        for iteration in range(self._max_iter):
            # Step 1: Procrustes on current dictionary
            W = proc.align(source, target, anchors=current_anchors)

            # Step 2: Translate source
            src_aligned = source.data.astype(np.float64) @ W

            # Step 3: Compute objective (mean squared error on anchors)
            src_w2i = {w: i for i, w in enumerate(source.vocab)}
            tgt_w2i = {w: i for i, w in enumerate(target.vocab)}
            anchor_src = np.array([src_aligned[src_w2i[w]] for w in current_anchors])
            anchor_tgt = np.array(
                [target.data[tgt_w2i[w]] for w in current_anchors]
            ).astype(np.float64)
            objective = float(np.mean(np.sum((anchor_src - anchor_tgt) ** 2, axis=1)))

            self._history.append({
                "iteration": iteration,
                "n_anchors": len(current_anchors),
                "objective": objective,
            })

            logger.info(
                "VecMap iter %d: %d anchors, objective=%.6f",
                iteration, len(current_anchors), objective,
            )

            # Check convergence
            if abs(prev_objective - objective) < self._tol:
                logger.info("VecMap converged at iteration %d", iteration)
                break
            prev_objective = objective

            # Step 4: Build new dictionary from mutual nearest neighbours
            new_anchors = _build_dictionary_nn(
                src_aligned, target.data.astype(np.float64),
                source.vocab, target.vocab,
            )

            # Stochastic perturbation to avoid local minima
            if (
                self._stochastic_interval > 0
                and iteration % self._stochastic_interval == 0
                and iteration > 0
            ):
                rng = np.random.default_rng(seed=iteration)
                keep = max(1, int(len(new_anchors) * 0.9))
                indices = rng.choice(len(new_anchors), size=keep, replace=False)
                new_anchors = [new_anchors[i] for i in sorted(indices)]

            if len(new_anchors) == 0:
                logger.warning(
                    "VecMap: no mutual NN found at iteration %d, "
                    "keeping previous dictionary.",
                    iteration,
                )
            else:
                current_anchors = new_anchors

        self._W = W
        return W

    def transform(
        self,
        source: EmbeddingMatrix,
        W: np.ndarray | None = None,
    ) -> EmbeddingMatrix:
        """Apply alignment matrix to all source embeddings."""
        if W is None:
            W = self._W
        if W is None:
            raise RuntimeError("No alignment matrix available.  Call align() first.")
        aligned = source.data.astype(np.float64) @ W
        return EmbeddingMatrix(
            data=aligned,
            vocab=list(source.vocab),
            dialect_code=source.dialect_code,
        )

    @property
    def alignment_matrix(self) -> np.ndarray | None:
        return self._W

    @property
    def convergence_history(self) -> list[dict[str, Any]]:
        return list(self._history)
