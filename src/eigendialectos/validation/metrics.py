"""Central metrics module for EigenDialectos validation."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from eigendialectos.constants import DialectCode


# ---------------------------------------------------------------------------
# Text-generation metrics
# ---------------------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a list of tokens."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Compute a simple BLEU score (unigram to *max_n*-gram with brevity penalty).

    Parameters
    ----------
    reference : str
        Reference (ground-truth) text.
    hypothesis : str
        Hypothesis (generated) text.
    max_n : int
        Maximum n-gram order (default 4).

    Returns
    -------
    float
        BLEU score in [0, 1].
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1.0 - len(ref_tokens) / len(hyp_tokens))

    log_avg = 0.0
    effective_order = 0

    for n in range(1, max_n + 1):
        ref_ng = _ngrams(ref_tokens, n)
        hyp_ng = _ngrams(hyp_tokens, n)

        if not hyp_ng:
            break

        ref_counts: Counter[tuple[str, ...]] = Counter(ref_ng)
        hyp_counts: Counter[tuple[str, ...]] = Counter(hyp_ng)

        clipped = 0
        for ng, cnt in hyp_counts.items():
            clipped += min(cnt, ref_counts.get(ng, 0))

        precision = clipped / len(hyp_ng) if hyp_ng else 0.0
        if precision == 0.0:
            # If any precision is zero, BLEU is zero (product becomes 0)
            return 0.0

        log_avg += math.log(precision)
        effective_order += 1

    if effective_order == 0:
        return 0.0

    log_avg /= effective_order
    return bp * math.exp(log_avg)


def _char_ngrams(text: str, n: int) -> list[str]:
    """Extract character n-grams from *text*."""
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def compute_chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    """Compute character n-gram F-score (chrF).

    Parameters
    ----------
    reference : str
        Reference text.
    hypothesis : str
        Hypothesis text.
    n : int
        Maximum character n-gram order (default 6).
    beta : float
        F-score beta parameter (default 2.0 favours recall).

    Returns
    -------
    float
        chrF score in [0, 1].
    """
    if not reference or not hypothesis:
        return 0.0

    total_precision = 0.0
    total_recall = 0.0
    effective_orders = 0

    for order in range(1, n + 1):
        ref_ng = _char_ngrams(reference, order)
        hyp_ng = _char_ngrams(hypothesis, order)

        if not ref_ng or not hyp_ng:
            continue

        ref_counts: Counter[str] = Counter(ref_ng)
        hyp_counts: Counter[str] = Counter(hyp_ng)

        matched = 0
        for ng, cnt in hyp_counts.items():
            matched += min(cnt, ref_counts.get(ng, 0))

        precision = matched / len(hyp_ng) if hyp_ng else 0.0
        recall = matched / len(ref_ng) if ref_ng else 0.0

        total_precision += precision
        total_recall += recall
        effective_orders += 1

    if effective_orders == 0:
        return 0.0

    avg_p = total_precision / effective_orders
    avg_r = total_recall / effective_orders

    if avg_p + avg_r == 0.0:
        return 0.0

    beta_sq = beta ** 2
    score = (1.0 + beta_sq) * avg_p * avg_r / (beta_sq * avg_p + avg_r)
    return score


# ---------------------------------------------------------------------------
# Perplexity ratio
# ---------------------------------------------------------------------------

def compute_dialectal_perplexity_ratio(
    text: str,
    target_probs: dict[str, float],
    baseline_probs: dict[str, float],
) -> float:
    """Ratio of perplexity under target vs baseline n-gram probability dictionaries.

    Lower values mean the text is *more* characteristic of the target dialect
    than the baseline.

    Parameters
    ----------
    text : str
        Input text.
    target_probs : dict[str, float]
        Token-to-probability mapping for the target dialect.
    baseline_probs : dict[str, float]
        Token-to-probability mapping for the baseline dialect.

    Returns
    -------
    float
        PP_target / PP_baseline.  Values < 1 indicate text closer to target.
    """
    tokens = text.lower().split()
    if not tokens:
        return 1.0

    eps = 1e-10
    n = len(tokens)

    log_pp_target = 0.0
    log_pp_baseline = 0.0
    for t in tokens:
        log_pp_target -= math.log(target_probs.get(t, eps))
        log_pp_baseline -= math.log(baseline_probs.get(t, eps))

    pp_target = math.exp(log_pp_target / n)
    pp_baseline = math.exp(log_pp_baseline / n)

    if pp_baseline == 0.0:
        return float("inf")

    return pp_target / pp_baseline


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_classification_accuracy(
    predictions: list[DialectCode],
    ground_truth: list[DialectCode],
) -> float:
    """Simple accuracy: proportion of correct predictions.

    Parameters
    ----------
    predictions : list[DialectCode]
    ground_truth : list[DialectCode]

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    if not predictions or len(predictions) != len(ground_truth):
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def compute_confusion_matrix(
    predictions: list,
    ground_truth: list,
    labels: list,
) -> np.ndarray:
    """Compute a confusion matrix.

    Parameters
    ----------
    predictions : list
        Predicted labels.
    ground_truth : list
        True labels.
    labels : list
        Ordered label list; determines row/column ordering.

    Returns
    -------
    np.ndarray
        Shape ``(n_labels, n_labels)`` with ``cm[i, j]`` = count of true=i, pred=j.
    """
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for true, pred in zip(ground_truth, predictions):
        i = label_to_idx.get(true)
        j = label_to_idx.get(pred)
        if i is not None and j is not None:
            cm[i, j] += 1
    return cm


# ---------------------------------------------------------------------------
# Matrix & spectral metrics
# ---------------------------------------------------------------------------

def compute_frobenius_error(W_true: np.ndarray, W_predicted: np.ndarray) -> float:
    """Relative Frobenius error: ||W_true - W_pred||_F / ||W_true||_F.

    Parameters
    ----------
    W_true : np.ndarray
    W_predicted : np.ndarray

    Returns
    -------
    float
        Relative error (0 = perfect match).
    """
    norm_true = np.linalg.norm(W_true, "fro")
    if norm_true == 0.0:
        return float(np.linalg.norm(W_true - W_predicted, "fro"))
    return float(np.linalg.norm(W_true - W_predicted, "fro") / norm_true)


def compute_eigenspectrum_divergence(
    spec_a: np.ndarray,
    spec_b: np.ndarray,
) -> float:
    """KL divergence between two normalised eigenvalue distributions.

    Both spectra are first converted to valid probability distributions
    (absolute values, normalised, with a small epsilon for stability).

    Parameters
    ----------
    spec_a : np.ndarray
        First eigenvalue spectrum.
    spec_b : np.ndarray
        Second eigenvalue spectrum.

    Returns
    -------
    float
        KL(a || b).  Non-negative; 0 when distributions match.
    """
    eps = 1e-10
    a = np.abs(spec_a).astype(np.float64) + eps
    b = np.abs(spec_b).astype(np.float64) + eps

    a = a / a.sum()
    b = b / b.sum()

    return float(np.sum(a * np.log(a / b)))


# ---------------------------------------------------------------------------
# Inter-annotator agreement
# ---------------------------------------------------------------------------

def compute_krippendorff_alpha(ratings: np.ndarray) -> float:
    """Krippendorff's alpha for nominal data.

    Parameters
    ----------
    ratings : np.ndarray
        Shape ``(n_raters, n_items)``.  Use ``np.nan`` for missing ratings.

    Returns
    -------
    float
        Alpha in ``(-inf, 1]``.  1 = perfect agreement, 0 = chance,
        < 0 = worse than chance.
    """
    n_raters, n_items = ratings.shape

    # Collect all unique non-nan values
    all_values = ratings[~np.isnan(ratings)]
    if len(all_values) == 0:
        return 0.0
    categories = np.unique(all_values)

    # Observed coincidence matrix
    o_matrix: dict[tuple, float] = {}
    for cat_c in categories:
        for cat_k in categories:
            o_matrix[(cat_c, cat_k)] = 0.0

    for item in range(n_items):
        col = ratings[:, item]
        valid = col[~np.isnan(col)]
        m = len(valid)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o_matrix[(valid[i], valid[j])] += 1.0 / (m - 1)

    # Expected coincidence matrix (from marginal frequencies)
    n_total = 0.0
    freq: dict[float, float] = {}
    for item in range(n_items):
        col = ratings[:, item]
        valid = col[~np.isnan(col)]
        m = len(valid)
        if m < 2:
            continue
        for v in valid:
            freq[v] = freq.get(v, 0.0) + 1.0
            n_total += 1.0

    if n_total <= 1:
        return 0.0

    # Observed disagreement
    do = 0.0
    de = 0.0
    for cat_c in categories:
        for cat_k in categories:
            if cat_c != cat_k:
                do += o_matrix.get((cat_c, cat_k), 0.0)
                de += freq.get(float(cat_c), 0.0) * freq.get(float(cat_k), 0.0)

    if de == 0.0:
        return 1.0

    de = de / (n_total - 1.0)

    if de == 0.0:
        return 1.0

    return 1.0 - do / de
