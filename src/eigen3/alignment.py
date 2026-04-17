"""Procrustes alignment of per-variety embeddings to a shared space."""

from __future__ import annotations

import logging

import numpy as np
from numpy.linalg import svd

logger = logging.getLogger(__name__)


def procrustes_align(
    source: np.ndarray,
    target: np.ndarray,
    anchor_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal Procrustes rotation: align source to target using anchor words.

    Parameters
    ----------
    source : np.ndarray
        (vocab_size, dim) source embeddings.
    target : np.ndarray
        (vocab_size, dim) target embeddings.
    anchor_indices : list[int]
        Indices of anchor words for computing the rotation.

    Returns
    -------
    R : np.ndarray
        (dim, dim) orthogonal rotation matrix.
    aligned : np.ndarray
        (vocab_size, dim) rotated source embeddings.
    """
    src_anchors = source[anchor_indices]  # (n_anchors, dim)
    tgt_anchors = target[anchor_indices]

    # SVD of cross-covariance: target^T @ source
    M = tgt_anchors.T @ src_anchors  # (dim, dim)
    U, _, Vt = svd(M)
    R = U @ Vt  # (dim, dim) orthogonal

    aligned = source @ R.T
    return R, aligned


def align_all_to_reference(
    embeddings: dict[str, np.ndarray],
    anchor_indices: list[int],
    reference: str = "ES_PEN",
) -> dict[str, np.ndarray]:
    """Align all varieties to a reference variety using Procrustes.

    Parameters
    ----------
    embeddings : dict[str, np.ndarray]
        Per-variety (vocab_size, dim) embedding matrices.
    anchor_indices : list[int]
        Indices of anchor words.
    reference : str
        The reference variety (kept unchanged).

    Returns
    -------
    dict[str, np.ndarray]
        Aligned embeddings.
    """
    ref_emb = embeddings[reference]
    aligned = {reference: ref_emb.copy()}

    for variety, emb in embeddings.items():
        if variety == reference:
            continue
        R, aligned_emb = procrustes_align(emb, ref_emb, anchor_indices)
        aligned[variety] = aligned_emb
        logger.info("Aligned %s to %s", variety, reference)

    return aligned
