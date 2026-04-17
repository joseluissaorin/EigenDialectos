#!/usr/bin/env python3
"""Bootstrap confidence intervals for EigenDialectos spectral metrics.

Resamples the corpus with replacement per dialect, retrains FastText,
recomputes transformation matrices + eigendecomposition, and collects
entropy and distance distributions to produce 95% confidence intervals.

Usage
-----
    python scripts/bootstrap_ci.py --n-bootstraps 20
    python scripts/bootstrap_ci.py --n-bootstraps 50 --dim 100

Outputs
-------
    outputs/analysis/bootstrap_ci.json        — full bootstrap results
    outputs/analysis/bootstrap_entropy_ci.png  — entropy CI plot
    outputs/analysis/bootstrap_distance_ci.png — distance CI heatmap
"""

from __future__ import annotations

import argparse
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

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eigendialectos.constants import DIALECT_NAMES, DialectCode
from eigendialectos.types import (
    CorpusSlice,
    DialectSample,
    DialectalSpectrum,
    EigenDecomposition,
    EmbeddingMatrix,
    TransformationMatrix,
)

LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)
logger = logging.getLogger("bootstrap")

ALL_DIALECT_CODES = sorted(DialectCode, key=lambda c: c.value)

DIALECT_LABELS = {
    "ES_PEN": "Peninsular", "ES_AND": "Andaluz", "ES_CAN": "Canario",
    "ES_RIO": "Rioplatense", "ES_MEX": "Mexicano", "ES_CAR": "Caribeño",
    "ES_CHI": "Chileno", "ES_AND_BO": "Andino",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
})


# ===================================================================
# JSON serialiser
# ===================================================================

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


# ===================================================================
# Load corpus (same logic as run_pipeline.py)
# ===================================================================

def load_corpus(corpus_path: Path) -> dict[DialectCode, list[DialectSample]]:
    """Load JSONL corpus and return raw samples grouped by dialect."""
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


# ===================================================================
# Single bootstrap iteration
# ===================================================================

def run_one_bootstrap(
    samples_by_dialect: dict[DialectCode, list[DialectSample]],
    dim: int,
    epochs: int,
    rng: random.Random,
    reference: DialectCode = DialectCode.ES_PEN,
) -> tuple[dict[str, float], np.ndarray, list[str]]:
    """Run one bootstrap: resample, train, compute spectra, return entropies + distances."""
    from eigendialectos.embeddings.subword.fasttext_model import FastTextModel
    from eigendialectos.spectral.transformation import compute_all_transforms
    from eigendialectos.spectral.eigendecomposition import eigendecompose
    from eigendialectos.spectral.eigenspectrum import compute_eigenspectrum
    from eigendialectos.spectral.distance import compute_distance_matrix

    # Step 1: Resample corpus with replacement per dialect
    resampled: dict[DialectCode, CorpusSlice] = {}
    for code, samples in samples_by_dialect.items():
        n = len(samples)
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        resampled_samples = [samples[i] for i in indices]
        resampled[code] = CorpusSlice(samples=resampled_samples, dialect_code=code)

    # Step 2: Train FastText per dialect
    models = {}
    for code, corpus_slice in resampled.items():
        model = FastTextModel(
            dialect_code=code,
            vector_size=dim,
            min_count=2,
            window=5,
            epochs=epochs,
            sg=1,
        )
        model.train(corpus_slice)
        models[code] = model

    # Step 3: Find shared vocabulary
    vocab_sets = {code: set(m._model.wv.key_to_index.keys()) for code, m in models.items()}
    shared_vocab = sorted(set.intersection(*vocab_sets.values()))
    if len(shared_vocab) < 50:
        raise RuntimeError(f"Shared vocabulary too small: {len(shared_vocab)}")

    # Step 4: Build embedding matrices (d × V)
    embeddings: dict[DialectCode, EmbeddingMatrix] = {}
    for code, model in models.items():
        emb = model.encode_words(shared_vocab)
        emb.data = emb.data.T  # (V, d) → (d, V)
        embeddings[code] = emb

    # Step 5: Compute transformation matrices
    transforms = compute_all_transforms(
        embeddings=embeddings,
        reference=reference,
        method="lstsq",
        regularization=0.01,
    )

    # Step 6: Eigendecompose + spectra
    spectra = {}
    decompositions = {}
    for code, W in transforms.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigen = eigendecompose(W)
        decompositions[code] = eigen
        spectra[code] = compute_eigenspectrum(eigen)

    # Step 7: Entropy
    entropies = {code.value: float(spec.entropy) for code, spec in spectra.items()}

    # Step 8: Distance matrix
    filtered_entropies = {c: spectra[c].entropy for c in spectra}
    D = compute_distance_matrix(
        transforms={c: transforms[c] for c in spectra},
        spectra=spectra,
        entropies=filtered_entropies,
        method="combined",
    )
    sorted_codes = sorted(spectra.keys(), key=lambda c: c.value)
    dialect_order = [c.value for c in sorted_codes]

    return entropies, D, dialect_order


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for spectral metrics")
    parser.add_argument("--n-bootstraps", type=int, default=20, help="Number of bootstrap iterations")
    parser.add_argument("--dim", type=int, default=100, help="Embedding dimensionality")
    parser.add_argument("--epochs", type=int, default=10, help="FastText training epochs")
    parser.add_argument("--corpus", type=str, default=None, help="Path to corpus.jsonl")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    corpus_path = Path(args.corpus) if args.corpus else PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
    output_dir = PROJECT_ROOT / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading corpus from %s", corpus_path)
    samples_by_dialect = load_corpus(corpus_path)
    for code, samples in sorted(samples_by_dialect.items(), key=lambda x: x[0].value):
        logger.info("  %s: %d samples", code.value, len(samples))

    # Also load original results for comparison
    orig_spectra_path = PROJECT_ROOT / "outputs" / "checkpoints" / "spectra.json"
    orig_entropies = {}
    if orig_spectra_path.exists():
        with open(orig_spectra_path) as f:
            orig_data = json.load(f)
        orig_entropies = {k: v["entropy"] for k, v in orig_data.items()}

    rng = random.Random(args.seed)

    # Collect bootstrap results
    all_entropies: dict[str, list[float]] = {}
    all_distances: list[np.ndarray] = []
    dialect_order = None

    logger.info("Running %d bootstrap iterations...", args.n_bootstraps)
    t0 = time.perf_counter()

    for i in range(args.n_bootstraps):
        ti = time.perf_counter()
        try:
            ent, D, d_order = run_one_bootstrap(
                samples_by_dialect, args.dim, args.epochs, rng,
            )
            if dialect_order is None:
                dialect_order = d_order
            for code_str, h in ent.items():
                all_entropies.setdefault(code_str, []).append(h)
            all_distances.append(D)
            elapsed = time.perf_counter() - ti
            logger.info(
                "  Bootstrap %d/%d done in %.1fs (vocab=%s)",
                i + 1, args.n_bootstraps, elapsed,
                "ok",
            )
        except Exception as exc:
            logger.warning("  Bootstrap %d/%d FAILED: %s", i + 1, args.n_bootstraps, exc)

    total_time = time.perf_counter() - t0
    logger.info("All bootstraps complete in %.1fs", total_time)

    n_success = len(all_distances)
    if n_success < 3:
        logger.error("Too few successful bootstraps (%d). Cannot compute CIs.", n_success)
        return

    # ---------------------------------------------------------------
    # Compute CIs
    # ---------------------------------------------------------------
    ci_results: dict[str, Any] = {
        "n_bootstraps": args.n_bootstraps,
        "n_successful": n_success,
        "dim": args.dim,
        "epochs": args.epochs,
        "seed": args.seed,
        "total_time_seconds": total_time,
    }

    # Entropy CIs
    entropy_ci: dict[str, dict] = {}
    for code_str in sorted(all_entropies.keys()):
        vals = np.array(all_entropies[code_str])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        entropy_ci[code_str] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "original": orig_entropies.get(code_str, None),
            "n_samples": len(vals),
        }
        logger.info(
            "  %s entropy: %.4f [%.4f, %.4f] (orig: %s)",
            code_str,
            np.mean(vals), lo, hi,
            f"{orig_entropies[code_str]:.4f}" if code_str in orig_entropies else "N/A",
        )
    ci_results["entropy_ci"] = entropy_ci

    # Distance matrix CIs
    D_stack = np.stack(all_distances, axis=0)  # (n_boot, n_dial, n_dial)
    D_mean = np.mean(D_stack, axis=0)
    D_std = np.std(D_stack, axis=0)
    D_lo = np.percentile(D_stack, 2.5, axis=0)
    D_hi = np.percentile(D_stack, 97.5, axis=0)

    ci_results["distance_matrix"] = {
        "dialect_order": dialect_order,
        "mean": D_mean.tolist(),
        "std": D_std.tolist(),
        "ci_lower": D_lo.tolist(),
        "ci_upper": D_hi.tolist(),
    }

    # Entropy ranking stability
    rankings = []
    for code_str in sorted(all_entropies.keys()):
        rankings.append((code_str, all_entropies[code_str]))

    # Check if ranking is stable across bootstraps
    rank_matrix = []  # (n_boot, n_dialects)
    codes_sorted = sorted(all_entropies.keys())
    for b in range(n_success):
        boot_vals = {c: all_entropies[c][b] for c in codes_sorted if b < len(all_entropies[c])}
        ranking = sorted(boot_vals.keys(), key=lambda c: boot_vals[c], reverse=True)
        rank_matrix.append(ranking)

    # Mode ranking
    from collections import Counter
    rank_by_position: dict[int, Counter] = {}
    for ranking in rank_matrix:
        for pos, code_str in enumerate(ranking):
            rank_by_position.setdefault(pos, Counter())[code_str] += 1

    modal_ranking = []
    for pos in range(len(codes_sorted)):
        if pos in rank_by_position:
            modal_ranking.append(rank_by_position[pos].most_common(1)[0][0])
    ci_results["modal_entropy_ranking"] = modal_ranking

    # Ranking stability: % of bootstraps where full ranking matches the modal one
    modal_tuple = tuple(modal_ranking)
    matching = sum(1 for r in rank_matrix if tuple(r) == modal_tuple)
    ci_results["ranking_stability"] = matching / n_success

    logger.info("Modal entropy ranking: %s", " > ".join(modal_ranking))
    logger.info("Ranking stability: %.1f%% of bootstraps match modal ranking", 100 * matching / n_success)

    # Save JSON
    json_path = output_dir / "bootstrap_ci.json"
    with open(json_path, "w") as f:
        json.dump(ci_results, f, indent=2, default=_json_default)
    logger.info("Saved bootstrap results to %s", json_path)

    # ---------------------------------------------------------------
    # Plot 1: Entropy CIs
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    codes = sorted(entropy_ci.keys())
    labels = [DIALECT_LABELS.get(c, c) for c in codes]
    means = [entropy_ci[c]["mean"] for c in codes]
    lowers = [entropy_ci[c]["ci_lower"] for c in codes]
    uppers = [entropy_ci[c]["ci_upper"] for c in codes]
    originals = [entropy_ci[c]["original"] for c in codes]

    x = np.arange(len(codes))
    errors = np.array([[m - l, u - m] for m, l, u in zip(means, lowers, uppers)]).T

    bars = ax.bar(x, means, yerr=errors, capsize=5, color="#4C72B0", alpha=0.8,
                  edgecolor="white", linewidth=0.5, error_kw={"linewidth": 1.5})

    # Overlay original values
    for i, orig in enumerate(originals):
        if orig is not None:
            ax.scatter(i, orig, color="red", zorder=5, s=40, marker="D", label="Original" if i == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Spectral Entropy (H)")
    ax.set_title(f"Spectral Entropy — 95% Bootstrap CI (N={n_success})")
    ax.legend(loc="upper right")

    # Add CI text
    for i, c in enumerate(codes):
        ci_width = uppers[i] - lowers[i]
        ax.text(i, lowers[i] - 0.01, f"±{ci_width/2:.3f}", ha="center", va="top", fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(output_dir / "bootstrap_entropy_ci.png")
    plt.close(fig)
    logger.info("Saved entropy CI plot")

    # ---------------------------------------------------------------
    # Plot 2: Distance matrix with CI widths
    # ---------------------------------------------------------------
    n = D_mean.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Mean distances
    im0 = axes[0].imshow(D_mean, cmap="YlOrRd", aspect="equal")
    axes[0].set_title("Mean Spectral Distance")
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels([DIALECT_LABELS.get(c, c) for c in dialect_order], rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([DIALECT_LABELS.get(c, c) for c in dialect_order], fontsize=8)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # CI width (upper - lower)
    ci_width_matrix = D_hi - D_lo
    im1 = axes[1].imshow(ci_width_matrix, cmap="Blues", aspect="equal")
    axes[1].set_title("95% CI Width")
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels([DIALECT_LABELS.get(c, c) for c in dialect_order], rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels([DIALECT_LABELS.get(c, c) for c in dialect_order], fontsize=8)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Coefficient of variation (std/mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_matrix = np.where(D_mean > 0, D_std / D_mean, 0)
    im2 = axes[2].imshow(cv_matrix, cmap="Purples", aspect="equal")
    axes[2].set_title("Coeff. of Variation (σ/μ)")
    axes[2].set_xticks(range(n))
    axes[2].set_xticklabels([DIALECT_LABELS.get(c, c) for c in dialect_order], rotation=45, ha="right", fontsize=8)
    axes[2].set_yticks(range(n))
    axes[2].set_yticklabels([DIALECT_LABELS.get(c, c) for c in dialect_order], fontsize=8)
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle(f"Distance Matrix — Bootstrap Statistics (N={n_success})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "bootstrap_distance_ci.png")
    plt.close(fig)
    logger.info("Saved distance CI plot")

    # Summary
    logger.info("=" * 60)
    logger.info("BOOTSTRAP SUMMARY")
    logger.info("=" * 60)
    logger.info("  Iterations: %d successful / %d total", n_success, args.n_bootstraps)
    logger.info("  Total time: %.0fs (%.1fs per iteration)", total_time, total_time / args.n_bootstraps)
    for c in sorted(entropy_ci.keys()):
        ci = entropy_ci[c]
        logger.info(
            "  %s: H=%.4f [%.4f, %.4f] width=%.4f",
            c, ci["mean"], ci["ci_lower"], ci["ci_upper"],
            ci["ci_upper"] - ci["ci_lower"],
        )


if __name__ == "__main__":
    main()
