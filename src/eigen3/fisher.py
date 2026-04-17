"""Fisher Information Matrix via LDA-like discriminant analysis.

Computes between-class and within-class scatter matrices over
dialect embedding spaces to identify maximally discriminative
directions and diagnostic vocabulary.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fisher Information Matrix
# ---------------------------------------------------------------------------

def compute_fim(
    embeddings_dict: dict[str, np.ndarray],
    vocab: list[str],
    regularisation: float = 1e-4,
) -> np.ndarray:
    """Compute Fisher Information Matrix from per-variety embeddings.

    Uses the LDA scatter-matrix formulation:

    - S_b = sum_c  n_c * (mu_c - mu)(mu_c - mu)^T   (between-class)
    - S_w = sum_c  sum_i (x_i - mu_c)(x_i - mu_c)^T  (within-class)
    - FIM = inv(S_w) @ S_b

    S_w is Tikhonov-regularised for invertibility.

    Parameters
    ----------
    embeddings_dict : dict[str, np.ndarray]
        Variety name -> (vocab_size, dim) embedding matrix.
    vocab : list[str]
        Shared vocabulary (length == vocab_size).
    regularisation : float
        Ridge added to S_w diagonal.

    Returns
    -------
    np.ndarray
        (dim, dim) Fisher Information Matrix.
    """
    varieties = sorted(embeddings_dict.keys())
    dim = embeddings_dict[varieties[0]].shape[1]
    n_vocab = len(vocab)

    # Global mean
    all_embs = np.stack([embeddings_dict[v] for v in varieties], axis=0)  # (C, V, d)
    mu_global = all_embs.mean(axis=0).mean(axis=0)  # (d,)

    # Between-class scatter
    S_b = np.zeros((dim, dim), dtype=np.float64)
    for v in varieties:
        emb = embeddings_dict[v]  # (V, d)
        mu_c = emb.mean(axis=0)   # (d,)
        diff = (mu_c - mu_global).reshape(-1, 1)
        S_b += n_vocab * (diff @ diff.T)

    # Within-class scatter
    S_w = np.zeros((dim, dim), dtype=np.float64)
    for v in varieties:
        emb = embeddings_dict[v]
        mu_c = emb.mean(axis=0)
        centered = emb - mu_c  # (V, d)
        S_w += centered.T @ centered

    # Regularise S_w
    S_w += regularisation * np.eye(dim)

    # FIM = inv(S_w) @ S_b
    fim = np.linalg.solve(S_w, S_b)

    return fim


# ---------------------------------------------------------------------------
# Diagnostic words (highest Fisher discriminant score)
# ---------------------------------------------------------------------------

def _word_scores(
    fim: np.ndarray,
    embeddings_dict: dict[str, np.ndarray],
    vocab: list[str],
) -> np.ndarray:
    """Compute Fisher discriminant score for every word.

    For each word, its score is the average across varieties of
    v^T @ FIM @ v, where v is the word's embedding vector.

    Returns
    -------
    np.ndarray
        (vocab_size,) array of scores.
    """
    varieties = sorted(embeddings_dict.keys())
    n_vocab = len(vocab)
    scores = np.zeros(n_vocab, dtype=np.float64)

    for v in varieties:
        emb = embeddings_dict[v]  # (V, d)
        # Vectorised: score_i = emb[i] @ FIM @ emb[i]
        # = diag(emb @ FIM @ emb.T)
        projected = emb @ fim  # (V, d)
        scores += np.sum(projected * emb, axis=1)  # (V,)

    scores /= len(varieties)
    return scores


def diagnostic_words(
    fim: np.ndarray,
    embeddings_dict: dict[str, np.ndarray],
    vocab: list[str],
    k: int = 50,
) -> list[tuple[str, float]]:
    """Top-k words with highest Fisher discriminant score.

    Parameters
    ----------
    fim : np.ndarray
        (dim, dim) Fisher Information Matrix.
    embeddings_dict : dict[str, np.ndarray]
        Variety name -> (vocab_size, dim) embeddings.
    vocab : list[str]
        Shared vocabulary.
    k : int
        Number of top words to return.

    Returns
    -------
    list[tuple[str, float]]
        (word, score) pairs sorted descending by score.
    """
    scores = _word_scores(fim, embeddings_dict, vocab)
    top_idx = np.argsort(-scores)[:k]
    return [(vocab[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Per-variety diagnostics
# ---------------------------------------------------------------------------

def per_variety_diagnostics(
    embeddings_dict: dict[str, np.ndarray],
    vocab: list[str],
    k: int = 20,
) -> dict[str, list[str]]:
    """Top diagnostic words per variety using one-vs-rest Fisher scores.

    For each variety *v*, a binary FIM is computed treating *v* as the
    positive class and all others as the negative class. The top-k
    words by that binary Fisher score are returned.

    Parameters
    ----------
    embeddings_dict : dict[str, np.ndarray]
        Variety name -> (vocab_size, dim) embeddings.
    vocab : list[str]
        Shared vocabulary.
    k : int
        Number of top words per variety.

    Returns
    -------
    dict[str, list[str]]
        Variety name -> list of diagnostic words.
    """
    varieties = sorted(embeddings_dict.keys())
    dim = embeddings_dict[varieties[0]].shape[1]
    n_vocab = len(vocab)
    result: dict[str, list[str]] = {}

    for target in varieties:
        # Binary split: target vs. rest
        emb_target = embeddings_dict[target]  # (V, d)
        mu_target = emb_target.mean(axis=0)

        # Rest: pool all other varieties
        rest_embs = [embeddings_dict[v] for v in varieties if v != target]
        rest_stacked = np.concatenate(rest_embs, axis=0)  # (V*(C-1), d)
        mu_rest = rest_stacked.mean(axis=0)

        # Global mean (weighted)
        n_t = emb_target.shape[0]
        n_r = rest_stacked.shape[0]
        mu_global = (n_t * mu_target + n_r * mu_rest) / (n_t + n_r)

        # Between-class scatter (2 classes)
        diff_t = (mu_target - mu_global).reshape(-1, 1)
        diff_r = (mu_rest - mu_global).reshape(-1, 1)
        S_b = n_t * (diff_t @ diff_t.T) + n_r * (diff_r @ diff_r.T)

        # Within-class scatter
        centered_t = emb_target - mu_target
        centered_r = rest_stacked - mu_rest
        S_w = centered_t.T @ centered_t + centered_r.T @ centered_r
        S_w += 1e-4 * np.eye(dim)

        fim_binary = np.linalg.solve(S_w, S_b)

        # Score each word using target embeddings only
        projected = emb_target @ fim_binary  # (V, d)
        scores = np.sum(projected * emb_target, axis=1)  # (V,)

        top_idx = np.argsort(-scores)[:k]
        result[target] = [vocab[i] for i in top_idx]

    return result
