"""Residual analysis: Delta-W SVD, per-word shifts, and PCA on embedding differences.

This module provides tools for understanding *what changed* between two
parametric operators (or between source and target embedding spaces) and
*which words were most affected*.

Typical workflow:

    >>> delta = compute_delta_W(W_a, W_b)
    >>> U, S, Vt = svd_analysis(delta)
    >>> top = top_shifted_words(W_a, vocab, embeddings=source_emb, k=20)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.linalg import svd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Delta-W computation
# ---------------------------------------------------------------------------

def compute_delta_W(W_a: np.ndarray, W_b: np.ndarray) -> np.ndarray:
    """Compute the residual (difference) between two transformation matrices.

    Parameters
    ----------
    W_a : np.ndarray
        (n, n) first transformation matrix.
    W_b : np.ndarray
        (n, n) second transformation matrix.

    Returns
    -------
    np.ndarray
        (n, n) matrix  ``W_a - W_b``.

    Raises
    ------
    ValueError
        If the shapes do not match.
    """
    if W_a.shape != W_b.shape:
        raise ValueError(
            f"Shape mismatch: W_a {W_a.shape} vs W_b {W_b.shape}"
        )
    return W_a - W_b


# ---------------------------------------------------------------------------
# SVD of Delta-W
# ---------------------------------------------------------------------------

def svd_analysis(
    delta_W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full SVD of a residual matrix.

    Parameters
    ----------
    delta_W : np.ndarray
        (n, n) residual matrix (typically from :func:`compute_delta_W`).

    Returns
    -------
    U : np.ndarray
        (n, n) left singular vectors.
    S : np.ndarray
        (n,) singular values in descending order.
    Vt : np.ndarray
        (n, n) right singular vectors (transposed).
    """
    U, S, Vt = svd(delta_W, full_matrices=True)
    return U, S, Vt


def svd_effective_rank(S: np.ndarray, threshold: float = 0.01) -> int:
    """Number of singular values above a fraction of the largest.

    Parameters
    ----------
    S : np.ndarray
        Singular values (descending).
    threshold : float
        Fraction of S[0] below which a singular value is considered
        negligible.

    Returns
    -------
    int
        Effective rank.
    """
    if len(S) == 0 or S[0] == 0:
        return 0
    cutoff = S[0] * threshold
    return int(np.sum(S >= cutoff))


def svd_energy_ratio(S: np.ndarray, k: int) -> float:
    """Fraction of total energy captured by the top-k singular values.

    Energy is measured as sum of squared singular values.

    Parameters
    ----------
    S : np.ndarray
        Singular values.
    k : int
        Number of top components.

    Returns
    -------
    float
        Ratio in [0, 1].
    """
    total = np.sum(S ** 2)
    if total == 0:
        return 0.0
    return float(np.sum(S[:k] ** 2) / total)


# ---------------------------------------------------------------------------
# Per-word shift analysis
# ---------------------------------------------------------------------------

def per_word_shifts(
    W: np.ndarray,
    vocab: list[str],
    embeddings: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute ||W*v - v|| for each word — how much W "moves" it.

    If *embeddings* is provided, ``v`` is the actual word vector.
    Otherwise, ``v`` is the standard basis vector e_i (useful when you
    only care about the action of W on each coordinate axis).

    Parameters
    ----------
    W : np.ndarray
        (n, n) transformation matrix.
    vocab : list[str]
        Vocabulary (length V).  If *embeddings* is None, only the first
        ``n`` words are used (one per dimension).
    embeddings : np.ndarray, optional
        (V, n) word embedding matrix.  If provided, shifts are computed
        for all V words.

    Returns
    -------
    dict[str, float]
        ``{word: shift_magnitude}``.
    """
    n = W.shape[0]
    shifts: dict[str, float] = {}

    if embeddings is not None:
        # Vectorised path: compute all shifts at once
        # embeddings: (V, n),  W: (n, n)
        transformed = embeddings @ W.T  # (V, n)
        deltas = transformed - embeddings  # (V, n)
        norms = np.linalg.norm(deltas, axis=1)  # (V,)
        for i, word in enumerate(vocab):
            shifts[word] = float(norms[i])
    else:
        # Standard-basis path
        I = np.eye(n, dtype=W.dtype)
        for i in range(min(n, len(vocab))):
            e_i = I[i]
            delta = W @ e_i - e_i
            shifts[vocab[i]] = float(np.linalg.norm(delta))

    return shifts


def top_shifted_words(
    W: np.ndarray,
    vocab: list[str],
    embeddings: Optional[np.ndarray] = None,
    k: int = 50,
) -> list[tuple[str, float]]:
    """Return the *k* words most shifted by the operator *W*.

    Parameters
    ----------
    W : np.ndarray
        (n, n) transformation matrix.
    vocab : list[str]
        Vocabulary.
    embeddings : np.ndarray, optional
        (V, n) embedding matrix.  See :func:`per_word_shifts`.
    k : int
        Number of top-shifted words to return.

    Returns
    -------
    list[tuple[str, float]]
        ``[(word, shift), ...]`` sorted descending by shift magnitude.
    """
    all_shifts = per_word_shifts(W, vocab, embeddings)
    sorted_shifts = sorted(all_shifts.items(), key=lambda t: t[1], reverse=True)
    return sorted_shifts[:k]


def shift_histogram(
    W: np.ndarray,
    vocab: list[str],
    embeddings: Optional[np.ndarray] = None,
    n_bins: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of per-word shift magnitudes.

    Parameters
    ----------
    W : np.ndarray
        (n, n) transformation matrix.
    vocab : list[str]
        Vocabulary.
    embeddings : np.ndarray, optional
        (V, n) embedding matrix.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    counts : np.ndarray
        (n_bins,) bin counts.
    bin_edges : np.ndarray
        (n_bins + 1,) bin edge values.
    """
    all_shifts = per_word_shifts(W, vocab, embeddings)
    values = np.array(list(all_shifts.values()), dtype=np.float64)
    counts, bin_edges = np.histogram(values, bins=n_bins)
    return counts, bin_edges


# ---------------------------------------------------------------------------
# PCA on embedding differences
# ---------------------------------------------------------------------------

def pca_embedding_shifts(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    n_components: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA on the per-word difference vectors (target - source).

    This reveals the principal directions along which the target dialect
    diverges from the source.  The leading components often correspond
    to interpretable phonological / lexical shifts.

    Parameters
    ----------
    source_emb : np.ndarray
        (V, dim) source embedding matrix.
    target_emb : np.ndarray
        (V, dim) target embedding matrix.
    n_components : int
        Number of principal components to return.

    Returns
    -------
    components : np.ndarray
        (n_components, dim) principal directions (unit vectors).
    variance_ratios : np.ndarray
        (n_components,) fraction of total variance explained by each
        component.

    Raises
    ------
    ValueError
        If the embedding matrices have different shapes or if
        *n_components* exceeds the embedding dimension.
    """
    if source_emb.shape != target_emb.shape:
        raise ValueError(
            f"Shape mismatch: source {source_emb.shape} vs "
            f"target {target_emb.shape}"
        )

    dim = source_emb.shape[1]
    if n_components > dim:
        raise ValueError(
            f"n_components ({n_components}) > embedding dim ({dim})"
        )

    # Difference matrix: (V, dim)
    diff = (target_emb - source_emb).astype(np.float64)

    # Center the differences
    mean = diff.mean(axis=0, keepdims=True)
    diff_centered = diff - mean

    # SVD is more numerically stable than eig on the covariance matrix
    # diff_centered = U @ diag(S) @ Vt,  shape (V, dim)
    # Principal directions are rows of Vt.
    _, S, Vt = svd(diff_centered, full_matrices=False)

    # Variance explained
    total_var = np.sum(S ** 2)
    if total_var == 0:
        return (
            np.zeros((n_components, dim), dtype=np.float64),
            np.zeros(n_components, dtype=np.float64),
        )

    variance_ratios = (S ** 2) / total_var

    components = Vt[:n_components]
    variance_ratios = variance_ratios[:n_components]

    return components, variance_ratios


def project_shifts(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Project per-word shifts onto a set of principal components.

    Parameters
    ----------
    source_emb : np.ndarray
        (V, dim) source embeddings.
    target_emb : np.ndarray
        (V, dim) target embeddings.
    components : np.ndarray
        (k, dim) principal component directions.

    Returns
    -------
    np.ndarray
        (V, k) projection scores for each word onto each component.
    """
    diff = (target_emb - source_emb).astype(np.float64)
    mean = diff.mean(axis=0, keepdims=True)
    diff_centered = diff - mean
    return diff_centered @ components.T
