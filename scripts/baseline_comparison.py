#!/usr/bin/env python3
"""Dialectometric baseline comparison for EigenDialectos.

Computes simple cosine-similarity centroid distances between dialects
and compares with our spectral distances. Shows whether the spectral
decomposition adds information beyond simple vector similarity.

Baselines computed:
1. Cosine distance between dialect embedding centroids
2. Euclidean distance between centroids
3. Vocabulary overlap (Jaccard index)
4. Our spectral distance (from checkpoints)

Outputs
-------
    outputs/analysis/baseline_comparison.json  — full comparison data
    outputs/analysis/baseline_comparison.png   — correlation scatter plots
    outputs/analysis/baseline_mantel.json      — Mantel test results
"""

from __future__ import annotations

import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eigendialectos.constants import DIALECT_NAMES, DialectCode

LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)
logger = logging.getLogger("baseline")

CHECKPOINTS = PROJECT_ROOT / "outputs" / "checkpoints"
EMBEDDINGS = PROJECT_ROOT / "outputs" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"

DIALECT_CODES = [
    "ES_AND", "ES_AND_BO", "ES_CAN", "ES_CAR",
    "ES_CHI", "ES_MEX", "ES_PEN", "ES_RIO",
]

DIALECT_LABELS = {
    "ES_PEN": "Peninsular", "ES_AND": "Andaluz", "ES_CAN": "Canario",
    "ES_RIO": "Rioplatense", "ES_MEX": "Mexicano", "ES_CAR": "Caribeño",
    "ES_CHI": "Chileno", "ES_AND_BO": "Andino",
}

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
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_embeddings() -> dict[str, np.ndarray]:
    """Load embedding matrices (d × V) for all dialects."""
    embeddings = {}
    for code in DIALECT_CODES:
        path = EMBEDDINGS / f"{code}.npy"
        if path.exists():
            embeddings[code] = np.load(path)
            logger.info("  Loaded %s: shape %s", code, embeddings[code].shape)
    return embeddings


def load_spectral_distances() -> tuple[np.ndarray, list[str]]:
    """Load the spectral distance matrix from checkpoints."""
    dist_json = CHECKPOINTS / "distance_matrix.json"
    with open(dist_json) as f:
        data = json.load(f)
    D = np.array(data["matrix"])
    order = data["dialect_order"]
    return D, order


def compute_centroid_distances(embeddings: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute pairwise centroid-based distance matrices.

    Returns dict with keys: 'cosine', 'euclidean'.
    Centroids are the mean embedding vector (mean across vocabulary axis).
    """
    codes = sorted(embeddings.keys())
    n = len(codes)

    # Compute centroids (mean across vocabulary dimension = axis 1)
    centroids = {}
    for code in codes:
        E = embeddings[code]  # (d, V)
        centroids[code] = np.mean(E, axis=1)  # (d,)

    D_cosine = np.zeros((n, n))
    D_euclidean = np.zeros((n, n))

    for i, ci in enumerate(codes):
        for j, cj in enumerate(codes):
            if i < j:
                d_cos = cosine_dist(centroids[ci], centroids[cj])
                d_euc = np.linalg.norm(centroids[ci] - centroids[cj])
                D_cosine[i, j] = D_cosine[j, i] = d_cos
                D_euclidean[i, j] = D_euclidean[j, i] = d_euc

    return {"cosine": D_cosine, "euclidean": D_euclidean, "dialect_order": codes}


def compute_pairwise_cosine_mean(embeddings: dict[str, np.ndarray]) -> np.ndarray:
    """Compute mean pairwise cosine distance across all shared vocab words.

    For each word w, compute cosine(E_i[:,w], E_j[:,w]), then average.
    This is more fine-grained than centroid distance.
    """
    codes = sorted(embeddings.keys())
    n = len(codes)
    D = np.zeros((n, n))

    for i, ci in enumerate(codes):
        for j, cj in enumerate(codes):
            if i < j:
                Ei = embeddings[ci]  # (d, V)
                Ej = embeddings[cj]  # (d, V)
                V = Ei.shape[1]
                dists = []
                for w in range(V):
                    vi = Ei[:, w]
                    vj = Ej[:, w]
                    n_vi = np.linalg.norm(vi)
                    n_vj = np.linalg.norm(vj)
                    if n_vi > 1e-10 and n_vj > 1e-10:
                        dists.append(1.0 - np.dot(vi, vj) / (n_vi * n_vj))
                D[i, j] = D[j, i] = np.mean(dists) if dists else 0.0

    return D


def mantel_test(D1: np.ndarray, D2: np.ndarray, n_perms: int = 9999) -> dict:
    """Mantel test for correlation between two distance matrices.

    Returns observed Pearson r, p-value (permutation test).
    """
    n = D1.shape[0]
    # Extract upper triangle
    idx = np.triu_indices(n, k=1)
    v1 = D1[idx]
    v2 = D2[idx]

    # Observed correlation
    r_obs, _ = pearsonr(v1, v2)
    rho_obs, _ = spearmanr(v1, v2)

    # Permutation test
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perms):
        perm = rng.permutation(n)
        D2_perm = D2[np.ix_(perm, perm)]
        v2_perm = D2_perm[idx]
        r_perm, _ = pearsonr(v1, v2_perm)
        if r_perm >= r_obs:
            count += 1

    p_value = (count + 1) / (n_perms + 1)

    return {
        "pearson_r": float(r_obs),
        "spearman_rho": float(rho_obs),
        "p_value": float(p_value),
        "n_permutations": n_perms,
    }


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embedding matrices...")
    embeddings = load_embeddings()
    if len(embeddings) < 2:
        logger.error("Need at least 2 embedding matrices")
        return

    logger.info("Loading spectral distance matrix...")
    D_spectral, spectral_order = load_spectral_distances()

    logger.info("Computing centroid-based distances...")
    centroid_results = compute_centroid_distances(embeddings)
    D_cosine_centroid = centroid_results["cosine"]
    D_euclidean = centroid_results["euclidean"]
    baseline_order = centroid_results["dialect_order"]

    logger.info("Computing word-level mean cosine distances...")
    D_cosine_wordlevel = compute_pairwise_cosine_mean(embeddings)

    # Align orderings — spectral and baseline may differ
    # Reorder spectral matrix to match baseline order
    spectral_idx = [spectral_order.index(c) for c in baseline_order]
    D_spectral_aligned = D_spectral[np.ix_(spectral_idx, spectral_idx)]

    # ---------------------------------------------------------------
    # Mantel tests
    # ---------------------------------------------------------------
    logger.info("Running Mantel tests...")
    baselines = {
        "cosine_centroid": D_cosine_centroid,
        "euclidean_centroid": D_euclidean,
        "cosine_wordlevel": D_cosine_wordlevel,
    }

    mantel_results = {}
    for name, D_base in baselines.items():
        result = mantel_test(D_spectral_aligned, D_base)
        mantel_results[name] = result
        logger.info(
            "  Spectral vs %s: r=%.4f, rho=%.4f, p=%.4f",
            name, result["pearson_r"], result["spearman_rho"], result["p_value"],
        )

    # Cross-baseline correlations
    for (n1, D1), (n2, D2) in combinations(baselines.items(), 2):
        result = mantel_test(D1, D2, n_perms=999)
        mantel_results[f"{n1}_vs_{n2}"] = result
        logger.info("  %s vs %s: r=%.4f", n1, n2, result["pearson_r"])

    # ---------------------------------------------------------------
    # Residual analysis: what spectral captures that baselines don't
    # ---------------------------------------------------------------
    n = len(baseline_order)
    idx = np.triu_indices(n, k=1)

    spectral_flat = D_spectral_aligned[idx]
    cosine_flat = D_cosine_wordlevel[idx]

    # Normalize both to [0, 1] for comparison
    s_norm = (spectral_flat - spectral_flat.min()) / (spectral_flat.max() - spectral_flat.min() + 1e-10)
    c_norm = (cosine_flat - cosine_flat.min()) / (cosine_flat.max() - cosine_flat.min() + 1e-10)

    residuals = s_norm - c_norm  # positive = spectral says more different than cosine

    # Find pairs with largest residuals
    pair_labels = []
    for i, j in zip(*idx):
        pair_labels.append(f"{baseline_order[i]}-{baseline_order[j]}")

    residual_ranking = sorted(zip(pair_labels, residuals), key=lambda x: abs(x[1]), reverse=True)

    logger.info("\nTop residuals (spectral vs word-level cosine):")
    for pair, res in residual_ranking[:10]:
        direction = "spectral>cosine" if res > 0 else "cosine>spectral"
        logger.info("  %s: %.4f (%s)", pair, res, direction)

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    results = {
        "dialect_order": baseline_order,
        "distance_matrices": {
            "spectral": D_spectral_aligned.tolist(),
            "cosine_centroid": D_cosine_centroid.tolist(),
            "euclidean_centroid": D_euclidean.tolist(),
            "cosine_wordlevel": D_cosine_wordlevel.tolist(),
        },
        "mantel_tests": mantel_results,
        "residual_analysis": {
            "top_residuals": [
                {"pair": p, "residual": float(r)} for p, r in residual_ranking
            ],
        },
    }

    with open(ANALYSIS_DIR / "baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    with open(ANALYSIS_DIR / "baseline_mantel.json", "w") as f:
        json.dump(mantel_results, f, indent=2, default=_json_default)

    # ---------------------------------------------------------------
    # Plot: correlation scatter grid
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = [
        ("Cosine (centroid)", D_cosine_centroid),
        ("Cosine (word-level)", D_cosine_wordlevel),
        ("Euclidean (centroid)", D_euclidean),
    ]

    for ax, (name, D_base) in zip(axes, methods):
        base_flat = D_base[idx]
        ax.scatter(base_flat, spectral_flat, alpha=0.7, s=50, c="#4C72B0", edgecolors="white", linewidth=0.5)

        # Fit and plot regression line
        if np.std(base_flat) > 1e-10:
            m, b = np.polyfit(base_flat, spectral_flat, 1)
            x_line = np.linspace(base_flat.min(), base_flat.max(), 50)
            ax.plot(x_line, m * x_line + b, "r--", linewidth=1.5, alpha=0.7)

        r, p = pearsonr(base_flat, spectral_flat)
        rho, _ = spearmanr(base_flat, spectral_flat)
        ax.set_xlabel(f"{name} Distance")
        ax.set_ylabel("Spectral Distance")
        ax.set_title(f"r={r:.3f}, ρ={rho:.3f}")

        # Label interesting points
        for k, (i_idx, j_idx) in enumerate(zip(*idx)):
            pair = f"{DIALECT_LABELS.get(baseline_order[i_idx], baseline_order[i_idx][:3])}-" \
                   f"{DIALECT_LABELS.get(baseline_order[j_idx], baseline_order[j_idx][:3])}"
            # Only label outliers
            resid = abs(spectral_flat[k] - (m * base_flat[k] + b)) if np.std(base_flat) > 1e-10 else 0
            if resid > np.std(spectral_flat) * 0.8:
                ax.annotate(pair, (base_flat[k], spectral_flat[k]),
                            fontsize=6, alpha=0.7, xytext=(5, 5),
                            textcoords="offset points")

    fig.suptitle("Spectral Distance vs. Baseline Metrics", fontsize=13)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "baseline_comparison.png")
    plt.close(fig)
    logger.info("Saved baseline comparison plot")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("BASELINE COMPARISON SUMMARY")
    logger.info("=" * 60)
    for name, result in mantel_results.items():
        if "vs" not in name or name.startswith("cosine") or name.startswith("euclidean"):
            continue
    for name in ["cosine_centroid", "euclidean_centroid", "cosine_wordlevel"]:
        r = mantel_results[name]
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        logger.info(
            "  Spectral vs %-20s: r=%.4f  ρ=%.4f  p=%.4f %s",
            name, r["pearson_r"], r["spearman_rho"], r["p_value"], sig,
        )


if __name__ == "__main__":
    main()
