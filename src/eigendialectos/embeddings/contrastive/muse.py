"""MUSE-style alignment with Procrustes refinement and CSLS criterion.

Implements the approach from Conneau et al. (2018):
1. (Optional) adversarial seed dictionary generation -- omitted here
   since we assume some vocabulary overlap between Spanish dialects.
2. Procrustes refinement with CSLS (Cross-domain Similarity Local
   Scaling) for dictionary induction.
3. Iterative refinement until convergence.

CSLS mitigates the hubness problem where some target vectors are
nearest neighbours of many source vectors, by penalising such "hubs".

Reference
---------
Conneau, A., Lample, G., Ranzato, M.A., Denoyer, L., & Jegou, H.
(2018). "Word Translation Without Parallel Data." *ICLR*.
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


def _csls_score(
    src: np.ndarray,
    tgt: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Compute the CSLS similarity matrix.

    CSLS(x, y) = 2 cos(x, y) - r_T(x) - r_S(y)

    where r_T(x) = mean of top-k cos(x, y_j) over target neighbours,
    and r_S(y) = mean of top-k cos(x_i, y) over source neighbours.

    Parameters
    ----------
    src:
        L2-normalised source matrix (n_src, d).
    tgt:
        L2-normalised target matrix (n_tgt, d).
    k:
        Number of neighbours for the mean similarity.

    Returns
    -------
    np.ndarray
        CSLS score matrix of shape (n_src, n_tgt).
    """
    # Cosine similarities (already normalised)
    sim = src @ tgt.T  # (n_src, n_tgt)

    # r_T(x): for each source, mean similarity to k nearest targets
    k_tgt = min(k, sim.shape[1])
    if k_tgt > 0:
        topk_tgt = np.partition(sim, -k_tgt, axis=1)[:, -k_tgt:]
        r_T = topk_tgt.mean(axis=1, keepdims=True)  # (n_src, 1)
    else:
        r_T = np.zeros((sim.shape[0], 1))

    # r_S(y): for each target, mean similarity to k nearest sources
    k_src = min(k, sim.shape[0])
    if k_src > 0:
        topk_src = np.partition(sim, -k_src, axis=0)[-k_src:, :]
        r_S = topk_src.mean(axis=0, keepdims=True)  # (1, n_tgt)
    else:
        r_S = np.zeros((1, sim.shape[1]))

    csls = 2 * sim - r_T - r_S
    return csls


def _build_dictionary_csls(
    src_aligned: np.ndarray,
    tgt: np.ndarray,
    src_vocab: list[str],
    tgt_vocab: list[str],
    k_csls: int = 10,
    max_pairs: int = 0,
) -> list[str]:
    """Build a seed dictionary using CSLS-based nearest neighbours.

    Returns shared words that are mutual CSLS nearest neighbours.
    """
    src_n = _l2_normalize(src_aligned)
    tgt_n = _l2_normalize(tgt)

    csls = _csls_score(src_n, tgt_n, k=k_csls)

    # Forward: best target for each source
    fwd = np.argmax(csls, axis=1)
    # Backward: best source for each target
    bwd = np.argmax(csls, axis=0)

    anchors: list[str] = []
    for si, ti in enumerate(fwd):
        if bwd[ti] == si:
            sw = src_vocab[si]
            tw = tgt_vocab[ti]
            if sw == tw:
                anchors.append(sw)

    if max_pairs > 0 and len(anchors) > max_pairs:
        anchors = anchors[:max_pairs]

    return anchors


class MUSEAligner:
    """MUSE-style Procrustes refinement with CSLS criterion.

    Parameters
    ----------
    max_iter:
        Maximum refinement iterations.
    tol:
        Convergence tolerance.
    k_csls:
        Number of neighbours for CSLS scoring.
    normalize:
        Whether to normalise before Procrustes.
    """

    def __init__(
        self,
        max_iter: int = 50,
        tol: float = 1e-6,
        k_csls: int = 10,
        normalize: bool = True,
    ) -> None:
        self._max_iter = max_iter
        self._tol = tol
        self._k_csls = k_csls
        self._normalize = normalize
        self._W: np.ndarray | None = None
        self._history: list[dict[str, Any]] = []

    def align(
        self,
        source: EmbeddingMatrix,
        target: EmbeddingMatrix,
        anchors: list[str] | None = None,
    ) -> np.ndarray:
        """Compute alignment via Procrustes + CSLS refinement.

        Parameters
        ----------
        source:
            Source dialect embedding matrix.
        target:
            Target (reference) dialect embedding matrix.
        anchors:
            Initial seed dictionary.  Vocabulary intersection if None.

        Returns
        -------
        np.ndarray
            Orthogonal alignment matrix W.
        """
        proc = ProcrustesAligner(normalize=self._normalize)

        current_anchors = anchors
        if current_anchors is None:
            current_anchors = sorted(set(source.vocab) & set(target.vocab))

        if len(current_anchors) == 0:
            raise ValueError(
                "No seed anchors available.  MUSE requires at least "
                "some vocabulary overlap or explicit seed pairs."
            )

        prev_objective = float("inf")
        self._history = []

        for iteration in range(self._max_iter):
            # Step 1: Procrustes on current dictionary
            W = proc.align(source, target, anchors=current_anchors)

            # Step 2: translate source
            src_aligned = source.data.astype(np.float64) @ W

            # Step 3: Compute objective
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
                "MUSE iter %d: %d anchors, objective=%.6f",
                iteration, len(current_anchors), objective,
            )

            if abs(prev_objective - objective) < self._tol:
                logger.info("MUSE converged at iteration %d", iteration)
                break
            prev_objective = objective

            # Step 4: Build new dictionary via CSLS
            new_anchors = _build_dictionary_csls(
                src_aligned,
                target.data.astype(np.float64),
                source.vocab,
                target.vocab,
                k_csls=self._k_csls,
            )

            if len(new_anchors) == 0:
                logger.warning(
                    "MUSE: no CSLS mutual NN at iteration %d, "
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
        """Apply the alignment matrix to all source embeddings."""
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
