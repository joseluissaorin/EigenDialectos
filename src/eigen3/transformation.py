"""W matrix computation via ridge regression: W = E_t @ E_s^T @ inv(E_s @ E_s^T + lI)."""

from __future__ import annotations

import logging

import numpy as np
from numpy.linalg import inv

from eigen3.constants import REFERENCE_VARIETY
from eigen3.types import TransformationMatrix

logger = logging.getLogger(__name__)


def compute_W(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    lambda_reg: float = 1e-4,
) -> TransformationMatrix:
    """Compute transformation matrix W mapping source → target space.

    W = E_t @ E_s^T @ inv(E_s @ E_s^T + lambda * I)

    Parameters
    ----------
    source_emb : np.ndarray
        (vocab_size, dim) source embedding matrix.
    target_emb : np.ndarray
        (vocab_size, dim) target embedding matrix.
    lambda_reg : float
        Tikhonov regularization strength.

    Returns
    -------
    TransformationMatrix with W (dim, dim).
    """
    # Transpose to (dim, vocab_size) for the formula
    E_s = source_emb.T.astype(np.float64)  # (dim, V)
    E_t = target_emb.T.astype(np.float64)  # (dim, V)

    dim = E_s.shape[0]
    gram = E_s @ E_s.T + lambda_reg * np.eye(dim)  # (dim, dim)
    W = E_t @ E_s.T @ inv(gram)  # (dim, dim)

    cond = float(np.linalg.cond(W))
    return TransformationMatrix(W=W, source="", target="", condition=cond)


def compute_all_W(
    embeddings: dict[str, np.ndarray],
    reference: str = REFERENCE_VARIETY,
    lambda_reg: float = 1e-4,
) -> dict[str, TransformationMatrix]:
    """Compute W for all varieties relative to a reference.

    W_v maps reference embeddings → variety v embeddings.
    W_reference ≈ I (identity).
    """
    ref_emb = embeddings[reference]
    result: dict[str, TransformationMatrix] = {}

    for variety, emb in embeddings.items():
        tm = compute_W(ref_emb, emb, lambda_reg)
        tm.source = reference
        tm.target = variety
        result[variety] = tm
        logger.info(
            "W[%s->%s]: condition=%.2f, ||W||_F=%.4f",
            reference, variety, tm.condition,
            float(np.linalg.norm(tm.W, "fro")),
        )

    return result
