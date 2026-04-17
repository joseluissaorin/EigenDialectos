"""Null model for dialect transformation noise floor.

Establishes the noise baseline by splitting a single variety's corpus
in half, training independent fastText models on each half, aligning
them via Procrustes, and computing ΔW = W - I.  The singular values
of ΔW_null define the noise floor: any real dialectal eigenvalue must
exceed this floor to be considered genuine signal.

Noise sources captured by the null model:
  1. Training stochasticity (different random seeds, different data order)
  2. Corpus sampling variation (different documents in each half)
  3. Alignment imperfection (Procrustes residual error)

The null model does NOT capture topic confounds between varieties
(which require corpus-level controls), but it quantifies the minimum
effect size detectable by the pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy.linalg import svdvals

logger = logging.getLogger(__name__)


def compute_null_model(
    corpus_reference: list[str],
    vocab: list[str],
    anchor_indices: list[int] | None = None,
    vector_size: int = 100,
    n_trials: int = 3,
    min_count: int = 3,
    epochs: int = 10,
    seed: int = 42,
    regularization: float = 0.01,
) -> dict:
    """Compute null-model ΔW singular values from split-corpus experiment.

    Splits *corpus_reference* (typically ES_PEN) into two halves,
    trains independent fastText models on each, aligns via Procrustes,
    computes W, then returns singular values of ΔW = W - I.

    Repeats *n_trials* times with different random splits and returns
    the median singular values as the noise floor.

    Parameters
    ----------
    corpus_reference:
        List of text documents from the reference variety.
    vocab:
        The filtered vocabulary to extract embeddings for.
    anchor_indices:
        Indices for anchor-only Procrustes (consistent with main pipeline).
    vector_size:
        Embedding dimensionality.
    n_trials:
        Number of random splits.  More trials → more stable estimate.
    min_count:
        Minimum word frequency for fastText training.
    epochs:
        FastText training epochs per half.
    seed:
        Base random seed.
    regularization:
        Ridge regularization λ for W computation.

    Returns
    -------
    dict with keys:
        ``"null_singular_values"``: (n_trials, dim) array of ΔW SVs per trial
        ``"median_sv"``: (dim,) median singular values across trials
        ``"mean_sv"``: (dim,) mean singular values across trials
        ``"p95_sv"``: (dim,) 95th percentile SVs (conservative threshold)
        ``"delta_w_frob_norms"``: (n_trials,) ||ΔW||_F per trial
        ``"n_trials"``: number of trials
        ``"split_sizes"``: list of (n_half_a, n_half_b) per trial
    """
    from eigendialectos.embeddings.fasttext_pipeline import (
        _extract_matrix,
        train_per_variety_fasttext,
    )
    from scipy.linalg import orthogonal_procrustes

    t0 = time.perf_counter()
    logger.info(
        "Computing null model: %d trials, %d docs, dim=%d ...",
        n_trials, len(corpus_reference), vector_size,
    )

    all_svs = []
    all_frob = []
    split_sizes = []

    for trial in range(n_trials):
        rng = np.random.RandomState(seed + trial)
        indices = rng.permutation(len(corpus_reference))
        mid = len(indices) // 2
        half_a = [corpus_reference[i] for i in indices[:mid]]
        half_b = [corpus_reference[i] for i in indices[mid:]]
        split_sizes.append((len(half_a), len(half_b)))

        # Train two independent fastText models
        models = train_per_variety_fasttext(
            corpus_by_variety={"NULL_A": half_a, "NULL_B": half_b},
            vector_size=vector_size,
            window=5,
            min_count=min_count,
            epochs=epochs,
            min_n=3,
            max_n=6,
            sg=1,
            workers=4,
            seed=seed + trial,
        )

        # Extract matrices for shared vocabulary
        X_a = _extract_matrix(models["NULL_A"], vocab, vector_size)
        X_b = _extract_matrix(models["NULL_B"], vocab, vector_size)

        # Anchor-only Procrustes alignment (same as main pipeline)
        if anchor_indices is not None and len(anchor_indices) > 0:
            X_a_proc = X_a[anchor_indices]
            X_b_proc = X_b[anchor_indices]
            mean_a = X_a_proc.mean(axis=0)
            mean_b = X_b_proc.mean(axis=0)
            R, _ = orthogonal_procrustes(
                X_b_proc - mean_b, X_a_proc - mean_a,
            )
            X_b_aligned = (X_b - mean_b) @ R + mean_a
        else:
            mean_a = X_a.mean(axis=0)
            mean_b = X_b.mean(axis=0)
            R, _ = orthogonal_procrustes(X_b - mean_b, X_a - mean_a)
            X_b_aligned = (X_b - mean_b) @ R + mean_a

        # Compute W: W @ X_a ≈ X_b_aligned (both are V×dim, need dim×V)
        E_s = X_a.T.astype(np.float64)       # (dim, V)
        E_t = X_b_aligned.T.astype(np.float64)  # (dim, V)

        gram = E_s @ E_s.T  # (dim, dim)
        gram_reg = gram + regularization * np.eye(vector_size)
        cross = E_t @ E_s.T
        W = cross @ np.linalg.inv(gram_reg)

        # ΔW = W - I
        delta_w = W - np.eye(vector_size)
        svs = svdvals(delta_w)
        frob = np.linalg.norm(delta_w, "fro")

        all_svs.append(svs)
        all_frob.append(frob)

        logger.info(
            "  Trial %d/%d: ||ΔW||_F=%.4f, top-3 SV=[%.4f, %.4f, %.4f]",
            trial + 1, n_trials, frob, svs[0], svs[1], svs[2],
        )

    all_svs_arr = np.array(all_svs)  # (n_trials, dim)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Null model complete in %.1fs.  Median ||ΔW||_F=%.4f, "
        "median top SV=%.4f",
        elapsed, np.median(all_frob), np.median(all_svs_arr[:, 0]),
    )

    return {
        "null_singular_values": all_svs_arr,
        "median_sv": np.median(all_svs_arr, axis=0),
        "mean_sv": np.mean(all_svs_arr, axis=0),
        "p95_sv": np.percentile(all_svs_arr, 95, axis=0),
        "delta_w_frob_norms": np.array(all_frob),
        "n_trials": n_trials,
        "split_sizes": split_sizes,
    }


def save_null_model(result: dict, output_path: Path) -> None:
    """Save null model results to disk."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(
        str(output_path / "null_singular_values.npy"),
        result["null_singular_values"],
    )
    np.save(
        str(output_path / "null_median_sv.npy"),
        result["median_sv"],
    )
    np.save(
        str(output_path / "null_p95_sv.npy"),
        result["p95_sv"],
    )

    meta = {
        "n_trials": result["n_trials"],
        "split_sizes": result["split_sizes"],
        "delta_w_frob_norms": result["delta_w_frob_norms"].tolist(),
        "median_top_sv": float(result["median_sv"][0]),
        "p95_top_sv": float(result["p95_sv"][0]),
    }
    (output_path / "null_model_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )
    logger.info("Null model saved to %s", output_path)


def load_null_model(output_path: Path) -> dict | None:
    """Load cached null model results, or return None if not found."""
    output_path = Path(output_path)
    sv_path = output_path / "null_median_sv.npy"
    if not sv_path.exists():
        return None

    return {
        "null_singular_values": np.load(
            str(output_path / "null_singular_values.npy"),
        ),
        "median_sv": np.load(str(output_path / "null_median_sv.npy")),
        "p95_sv": np.load(str(output_path / "null_p95_sv.npy")),
    }
