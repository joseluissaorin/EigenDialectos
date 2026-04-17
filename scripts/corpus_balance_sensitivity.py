#!/usr/bin/env python3
"""Corpus balance sensitivity analysis for EigenDialectos.

Subsamples larger dialects to match the smallest (ES_CAN ≈ 3,937),
retrains the full pipeline, and compares spectral metrics with the
full-corpus results to assess robustness to corpus imbalance.

Usage
-----
    python scripts/corpus_balance_sensitivity.py
    python scripts/corpus_balance_sensitivity.py --target-size 4000

Outputs
-------
    outputs/analysis/balance_sensitivity.json  — full comparison data
    outputs/analysis/balance_sensitivity.png   — comparison plots
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eigendialectos.constants import DIALECT_NAMES, DialectCode
from eigendialectos.types import (
    CorpusSlice,
    DialectSample,
    DialectalSpectrum,
    EmbeddingMatrix,
    TransformationMatrix,
)

LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)
logger = logging.getLogger("balance")

ALL_DIALECT_CODES = sorted(DialectCode, key=lambda c: c.value)

DIALECT_LABELS = {
    "ES_PEN": "Peninsular", "ES_AND": "Andaluz", "ES_CAN": "Canario",
    "ES_RIO": "Rioplatense", "ES_MEX": "Mexicano", "ES_CAR": "Caribeño",
    "ES_CHI": "Chileno", "ES_AND_BO": "Andino",
}

CHECKPOINTS = PROJECT_ROOT / "outputs" / "checkpoints"
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
})


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.complexfloating,)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, DialectCode):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_corpus(corpus_path: Path) -> dict[DialectCode, list[DialectSample]]:
    """Load JSONL corpus grouped by dialect."""
    samples_by_dialect: dict[DialectCode, list[DialectSample]] = {
        code: [] for code in ALL_DIALECT_CODES
    }
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                code = DialectCode(record.get("dialect", ""))
            except ValueError:
                continue
            sample = DialectSample(
                text=record.get("text", ""),
                dialect_code=code,
                source_id=record.get("source", "unknown"),
                confidence=float(record.get("confidence", 0.0)),
                metadata=record.get("metadata", {}),
            )
            samples_by_dialect[code].append(sample)
    return {k: v for k, v in samples_by_dialect.items() if v}


def run_balanced_pipeline(
    samples_by_dialect: dict[DialectCode, list[DialectSample]],
    target_size: int,
    dim: int = 100,
    epochs: int = 10,
    seed: int = 42,
) -> tuple[dict[str, float], np.ndarray, list[str], int]:
    """Subsample, train, compute spectra. Returns entropies, distance matrix, order, shared_vocab_size."""
    from eigendialectos.embeddings.subword.fasttext_model import FastTextModel
    from eigendialectos.spectral.transformation import compute_all_transforms
    from eigendialectos.spectral.eigendecomposition import eigendecompose
    from eigendialectos.spectral.eigenspectrum import compute_eigenspectrum
    from eigendialectos.spectral.distance import compute_distance_matrix

    rng = random.Random(seed)
    reference = DialectCode.ES_PEN

    # Subsample each dialect to target_size
    subsampled: dict[DialectCode, CorpusSlice] = {}
    for code, samples in samples_by_dialect.items():
        if len(samples) > target_size:
            chosen = rng.sample(samples, target_size)
        else:
            chosen = samples  # Keep as-is if already smaller
        subsampled[code] = CorpusSlice(samples=chosen, dialect_code=code)
        logger.info("  %s: %d → %d samples", code.value, len(samples), len(chosen))

    # Train FastText
    models = {}
    for code, corpus_slice in subsampled.items():
        model = FastTextModel(
            dialect_code=code, vector_size=dim, min_count=2,
            window=5, epochs=epochs, sg=1,
        )
        model.train(corpus_slice)
        models[code] = model

    # Shared vocabulary
    vocab_sets = {code: set(m._model.wv.key_to_index.keys()) for code, m in models.items()}
    shared_vocab = sorted(set.intersection(*vocab_sets.values()))
    logger.info("  Shared vocabulary (balanced): %d words", len(shared_vocab))

    # Embedding matrices
    embeddings: dict[DialectCode, EmbeddingMatrix] = {}
    for code, model in models.items():
        emb = model.encode_words(shared_vocab)
        emb.data = emb.data.T
        embeddings[code] = emb

    # Transforms + eigen
    transforms = compute_all_transforms(
        embeddings=embeddings, reference=reference,
        method="lstsq", regularization=0.01,
    )

    spectra = {}
    for code, W in transforms.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigen = eigendecompose(W)
        spectra[code] = compute_eigenspectrum(eigen)

    entropies = {code.value: float(spec.entropy) for code, spec in spectra.items()}

    D = compute_distance_matrix(
        transforms={c: transforms[c] for c in spectra},
        spectra=spectra,
        entropies={c: spectra[c].entropy for c in spectra},
        method="combined",
    )
    sorted_codes = sorted(spectra.keys(), key=lambda c: c.value)
    dialect_order = [c.value for c in sorted_codes]

    return entropies, D, dialect_order, len(shared_vocab)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Corpus balance sensitivity analysis")
    parser.add_argument("--target-size", type=int, default=4000)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(args.corpus) if args.corpus else PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"

    # Load original results
    orig_spectra_path = CHECKPOINTS / "spectra.json"
    with open(orig_spectra_path) as f:
        orig_data = json.load(f)
    orig_entropies = {k: v["entropy"] for k, v in orig_data.items()}

    orig_dist_path = CHECKPOINTS / "distance_matrix.json"
    with open(orig_dist_path) as f:
        orig_dist_data = json.load(f)
    orig_D = np.array(orig_dist_data["matrix"])
    orig_order = orig_dist_data["dialect_order"]

    logger.info("Original entropy ranking:")
    for code in sorted(orig_entropies, key=lambda c: orig_entropies[c], reverse=True):
        logger.info("  %s: %.4f", code, orig_entropies[code])

    # Load corpus and run balanced pipeline
    logger.info("\nLoading corpus from %s", corpus_path)
    samples = load_corpus(corpus_path)

    logger.info("\nRunning balanced pipeline (target=%d samples per dialect)...", args.target_size)
    t0 = time.perf_counter()
    bal_entropies, bal_D, bal_order, bal_vocab_size = run_balanced_pipeline(
        samples, args.target_size, args.dim, args.epochs, args.seed,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Balanced pipeline complete in %.1fs", elapsed)

    # Align distance matrices to same order
    bal_idx = [bal_order.index(c) for c in orig_order if c in bal_order]
    orig_idx = [orig_order.index(c) for c in orig_order if c in bal_order]
    aligned_codes = [orig_order[i] for i in orig_idx]

    bal_D_aligned = bal_D[np.ix_(bal_idx, bal_idx)]
    orig_D_aligned = orig_D[np.ix_(orig_idx, orig_idx)]

    # ---------------------------------------------------------------
    # Comparison metrics
    # ---------------------------------------------------------------
    common_codes = sorted(set(orig_entropies.keys()) & set(bal_entropies.keys()))

    # Entropy comparison
    orig_ent_vec = [orig_entropies[c] for c in common_codes]
    bal_ent_vec = [bal_entropies[c] for c in common_codes]
    ent_pearson, _ = pearsonr(orig_ent_vec, bal_ent_vec)
    ent_spearman, _ = spearmanr(orig_ent_vec, bal_ent_vec)

    # Ranking comparison
    orig_ranking = sorted(common_codes, key=lambda c: orig_entropies[c], reverse=True)
    bal_ranking = sorted(common_codes, key=lambda c: bal_entropies[c], reverse=True)
    ranking_matches = sum(1 for a, b in zip(orig_ranking, bal_ranking) if a == b)

    # Distance matrix comparison
    n = len(aligned_codes)
    idx = np.triu_indices(n, k=1)
    orig_flat = orig_D_aligned[idx]
    bal_flat = bal_D_aligned[idx]
    dist_pearson, _ = pearsonr(orig_flat, bal_flat)
    dist_spearman, _ = spearmanr(orig_flat, bal_flat)

    # Closest pair comparison
    orig_pairs = sorted(
        [(aligned_codes[i], aligned_codes[j], orig_D_aligned[i, j])
         for i, j in zip(*idx)],
        key=lambda x: x[2],
    )
    bal_pairs = sorted(
        [(aligned_codes[i], aligned_codes[j], bal_D_aligned[i, j])
         for i, j in zip(*idx)],
        key=lambda x: x[2],
    )

    orig_top5 = {(a, b) for a, b, _ in orig_pairs[:5]}
    bal_top5 = {(a, b) for a, b, _ in bal_pairs[:5]}
    top5_jaccard = len(orig_top5 & bal_top5) / len(orig_top5 | bal_top5) if orig_top5 | bal_top5 else 0

    logger.info("\n" + "=" * 60)
    logger.info("SENSITIVITY ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info("  Target corpus size: %d per dialect", args.target_size)
    logger.info("  Shared vocab: %d (balanced) vs 1623 (full)", bal_vocab_size)
    logger.info("  Entropy Pearson r: %.4f", ent_pearson)
    logger.info("  Entropy Spearman ρ: %.4f", ent_spearman)
    logger.info("  Ranking matches: %d/%d positions", ranking_matches, len(common_codes))
    logger.info("  Distance Pearson r: %.4f", dist_pearson)
    logger.info("  Distance Spearman ρ: %.4f", dist_spearman)
    logger.info("  Top-5 closest pairs Jaccard: %.2f", top5_jaccard)

    logger.info("\nEntropy comparison:")
    for code in common_codes:
        delta = bal_entropies[code] - orig_entropies[code]
        logger.info("  %s: %.4f (full) → %.4f (balanced) Δ=%.4f",
                     code, orig_entropies[code], bal_entropies[code], delta)

    logger.info("\nOriginal ranking: %s", " > ".join(orig_ranking))
    logger.info("Balanced ranking: %s", " > ".join(bal_ranking))

    logger.info("\nOriginal top-5 closest pairs:")
    for a, b, d in orig_pairs[:5]:
        logger.info("  %s ↔ %s = %.4f", a, b, d)
    logger.info("Balanced top-5 closest pairs:")
    for a, b, d in bal_pairs[:5]:
        logger.info("  %s ↔ %s = %.4f", a, b, d)

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    results = {
        "target_size": args.target_size,
        "balanced_shared_vocab_size": bal_vocab_size,
        "original_shared_vocab_size": 1623,
        "entropy_comparison": {
            code: {
                "original": orig_entropies.get(code),
                "balanced": bal_entropies.get(code),
                "delta": bal_entropies.get(code, 0) - orig_entropies.get(code, 0),
            }
            for code in common_codes
        },
        "ranking_comparison": {
            "original": orig_ranking,
            "balanced": bal_ranking,
            "positions_matching": ranking_matches,
            "total_positions": len(common_codes),
        },
        "correlation": {
            "entropy_pearson": float(ent_pearson),
            "entropy_spearman": float(ent_spearman),
            "distance_pearson": float(dist_pearson),
            "distance_spearman": float(dist_spearman),
            "top5_jaccard": float(top5_jaccard),
        },
        "distance_matrices": {
            "dialect_order": aligned_codes,
            "original": orig_D_aligned.tolist(),
            "balanced": bal_D_aligned.tolist(),
        },
    }

    with open(ANALYSIS_DIR / "balance_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info("Saved results to %s", ANALYSIS_DIR / "balance_sensitivity.json")

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Plot 1: Entropy scatter (full vs balanced)
    ax = axes[0]
    for i, code in enumerate(common_codes):
        label = DIALECT_LABELS.get(code, code)
        ax.scatter(orig_entropies[code], bal_entropies[code], s=60, zorder=5)
        ax.annotate(label, (orig_entropies[code], bal_entropies[code]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")

    mn = min(min(orig_ent_vec), min(bal_ent_vec)) - 0.05
    mx = max(max(orig_ent_vec), max(bal_ent_vec)) + 0.05
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Entropy (full corpus)")
    ax.set_ylabel(f"Entropy (balanced, N={args.target_size})")
    ax.set_title(f"Entropy: r={ent_pearson:.3f}, ρ={ent_spearman:.3f}")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_aspect("equal")

    # Plot 2: Distance scatter
    ax = axes[1]
    ax.scatter(orig_flat, bal_flat, alpha=0.7, s=40, c="#4C72B0", edgecolors="white", linewidth=0.5)
    mn = min(orig_flat.min(), bal_flat.min()) * 0.9
    mx = max(orig_flat.max(), bal_flat.max()) * 1.1
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Spectral Distance (full)")
    ax.set_ylabel(f"Spectral Distance (balanced, N={args.target_size})")
    ax.set_title(f"Distances: r={dist_pearson:.3f}, ρ={dist_spearman:.3f}")

    # Label the closest pair
    min_idx = np.argmin(orig_flat)
    i_min, j_min = idx[0][min_idx], idx[1][min_idx]
    pair_label = f"{DIALECT_LABELS.get(aligned_codes[i_min], '')}-{DIALECT_LABELS.get(aligned_codes[j_min], '')}"
    ax.annotate(pair_label, (orig_flat[min_idx], bal_flat[min_idx]),
                fontsize=7, xytext=(5, -10), textcoords="offset points")

    # Plot 3: Entropy delta bar chart
    ax = axes[2]
    deltas = [bal_entropies[c] - orig_entropies[c] for c in common_codes]
    labels = [DIALECT_LABELS.get(c, c) for c in common_codes]
    colors = ["#E74C3C" if d < 0 else "#2ECC71" for d in deltas]
    x = np.arange(len(common_codes))
    ax.bar(x, deltas, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("ΔH (balanced − full)")
    ax.set_title("Entropy Change After Balancing")
    ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle(f"Corpus Balance Sensitivity (N={args.target_size} per dialect)", fontsize=13)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "balance_sensitivity.png")
    plt.close(fig)
    logger.info("Saved plots to %s", ANALYSIS_DIR / "balance_sensitivity.png")


if __name__ == "__main__":
    main()
