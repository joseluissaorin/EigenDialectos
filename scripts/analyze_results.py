#!/usr/bin/env python3
"""Comprehensive spectral analysis of EigenDialectos pipeline results.

Loads all pipeline checkpoints and produces deep linguistic interpretation
of the spectral decomposition — eigenvector word projections, shared axes,
eigenvalue decay, dialectological validation, and SVD normality checks.

Usage
-----
    python scripts/analyze_results.py
    python scripts/analyze_results.py --top-k 10 --cosine-threshold 0.25

Outputs are written to ``outputs/analysis/`` (JSON data + PNG plots)
and simultaneously printed to stdout for the research log.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import entropy as kl_div_scipy, pearsonr, wasserstein_distance

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eigendialectos.constants import DIALECT_NAMES, DIALECT_REGIONS, DialectCode

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINTS = PROJECT_ROOT / "outputs" / "checkpoints"
EMBEDDINGS = PROJECT_ROOT / "outputs" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"

DIALECT_CODES: list[str] = [
    "ES_PEN", "ES_AND", "ES_CAN", "ES_RIO",
    "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO",
]

# Human-readable short labels used in figures
DIALECT_LABELS: dict[str, str] = {
    "ES_PEN": "Peninsular",
    "ES_AND": "Andaluz",
    "ES_CAN": "Canario",
    "ES_RIO": "Rioplatense",
    "ES_MEX": "Mexicano",
    "ES_CAR": "Caribeño",
    "ES_CHI": "Chileno",
    "ES_AND_BO": "Andino",
}

# Matplotlib style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ===================================================================
# Utility helpers
# ===================================================================

def _load_vocab() -> list[str]:
    with open(EMBEDDINGS / "vocab.json", encoding="utf-8") as f:
        return json.load(f)


def _load_embedding(dialect: str) -> npt.NDArray[np.float64]:
    """Load embedding matrix (dims x vocab)."""
    return np.load(EMBEDDINGS / f"{dialect}.npy")


def _load_eigenvalues(dialect: str) -> npt.NDArray[np.complex128]:
    return np.load(CHECKPOINTS / f"eigenvalues_{dialect}.npy")


def _load_eigenvectors(dialect: str) -> npt.NDArray[np.complex128]:
    return np.load(CHECKPOINTS / f"eigenvectors_{dialect}.npy")


def _load_W(dialect: str) -> npt.NDArray[np.float64]:
    return np.load(CHECKPOINTS / f"W_{dialect}.npy")


def _load_distance_matrix() -> tuple[npt.NDArray[np.float64], list[str]]:
    dm = np.load(CHECKPOINTS / "distance_matrix.npy")
    with open(CHECKPOINTS / "distance_matrix.json", encoding="utf-8") as f:
        meta = json.load(f)
    return dm, meta["dialect_order"]


def _load_spectra() -> dict[str, Any]:
    with open(CHECKPOINTS / "spectra.json", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data: Any, filename: str) -> Path:
    """Save JSON to ANALYSIS_DIR, return the path."""
    path = ANALYSIS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def _save_figure(fig: plt.Figure, filename: str) -> Path:
    path = ANALYSIS_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def _header(title: str) -> None:
    """Print a section header to stdout."""
    width = 78
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def _subheader(title: str) -> None:
    print(f"\n--- {title} ---\n")


def _cosine_similarity_complex(
    a: npt.NDArray[np.complex128],
    b: npt.NDArray[np.complex128],
) -> float:
    """Cosine similarity for complex vectors using Hermitian inner product.

    Returns |<a, b>| / (||a|| * ||b||) so the result is real and in [0, 1].
    """
    dot = np.abs(np.vdot(a, b))
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ===================================================================
# Analysis 1: Eigenvector linguistic interpretation
# ===================================================================

def analysis_eigenvector_words(
    vocab: list[str],
    top_k: int = 5,
    top_words: int = 20,
) -> dict[str, Any]:
    """Project vocabulary onto top eigenvectors and identify salient words."""
    _header("ANALYSIS 1: Eigenvector Linguistic Interpretation")

    results: dict[str, Any] = {}

    for dialect in DIALECT_CODES:
        _subheader(f"{dialect} ({DIALECT_LABELS[dialect]})")

        E = _load_embedding(dialect)          # (100, 1623)
        P = _load_eigenvectors(dialect)       # (100, 100) — columns are eigenvectors
        ev = _load_eigenvalues(dialect)        # (100,) complex

        # Sort eigenvectors by eigenvalue magnitude (descending)
        mag = np.abs(ev)
        order = np.argsort(mag)[::-1]

        dialect_data: list[dict[str, Any]] = []

        for rank in range(top_k):
            j = order[rank]
            v_j = P[:, j]  # eigenvector column (100,)
            lambda_j = ev[j]
            mag_j = mag[j]

            # Project each vocabulary word onto v_j:
            #   projection_i = |v_j^H @ E[:, i]|
            # v_j^H is (1, 100), E is (100, 1623) => result is (1623,)
            projections = np.abs(np.conj(v_j) @ E)  # (1623,)

            # Sort by projection magnitude
            word_order = np.argsort(projections)[::-1]
            top_word_list = []
            for w_idx in word_order[:top_words]:
                top_word_list.append({
                    "word": vocab[w_idx],
                    "projection": float(projections[w_idx]),
                })

            axis_info = {
                "rank": rank + 1,
                "eigenvector_index": int(j),
                "eigenvalue": {"real": float(lambda_j.real), "imag": float(lambda_j.imag)},
                "eigenvalue_magnitude": float(mag_j),
                "top_words": top_word_list,
            }
            dialect_data.append(axis_info)

            # Print
            print(f"  Eigenvector #{rank+1}  (index {j}, |lambda|={mag_j:.4f}, "
                  f"lambda={lambda_j.real:+.4f}{lambda_j.imag:+.4f}j)")
            words_str = ", ".join(
                f"{w['word']}({w['projection']:.3f})" for w in top_word_list[:10]
            )
            print(f"    Top words: {words_str}")
            if top_words > 10:
                words_str_2 = ", ".join(
                    f"{w['word']}({w['projection']:.3f})" for w in top_word_list[10:top_words]
                )
                print(f"               {words_str_2}")

        results[dialect] = dialect_data

        # Save per-dialect
        _save_json(dialect_data, f"eigenvector_words_{dialect}.json")

    # Save combined
    path = _save_json(results, "eigenvector_words_all.json")
    print(f"\n  Saved to {path.relative_to(PROJECT_ROOT)}")
    return results


# ===================================================================
# Analysis 2: Shared vs unique eigenvector axes
# ===================================================================

def analysis_shared_axes(
    top_n: int = 10,
    cosine_threshold: float = 0.3,
) -> dict[str, Any]:
    """Compare eigenvectors across dialect pairs — find shared vs unique axes."""
    _header("ANALYSIS 2: Shared vs Unique Eigenvector Axes")

    # Pre-load top-N eigenvectors per dialect (sorted by eigenvalue magnitude)
    top_vecs: dict[str, list[npt.NDArray[np.complex128]]] = {}
    for dialect in DIALECT_CODES:
        P = _load_eigenvectors(dialect)
        ev = _load_eigenvalues(dialect)
        mag = np.abs(ev)
        order = np.argsort(mag)[::-1]
        top_vecs[dialect] = [P[:, order[i]] for i in range(top_n)]

    # Pairwise similarity matrices
    pair_results: dict[str, Any] = {}
    shared_global: dict[int, list[str]] = {i: [] for i in range(top_n)}
    # Track which axes in each dialect are "unique" (not matched in any other)
    matched_axes: dict[str, set[int]] = {d: set() for d in DIALECT_CODES}

    for d_a, d_b in combinations(DIALECT_CODES, 2):
        pair_key = f"{d_a}_vs_{d_b}"
        sim_matrix = np.zeros((top_n, top_n))
        for i in range(top_n):
            for j in range(top_n):
                sim_matrix[i, j] = _cosine_similarity_complex(
                    top_vecs[d_a][i], top_vecs[d_b][j]
                )

        # Identify shared axes
        shared_pairs = []
        for i in range(top_n):
            for j in range(top_n):
                if sim_matrix[i, j] > cosine_threshold:
                    shared_pairs.append({
                        "axis_a": i + 1,
                        "axis_b": j + 1,
                        "cosine_similarity": float(sim_matrix[i, j]),
                    })
                    matched_axes[d_a].add(i)
                    matched_axes[d_b].add(j)

        pair_results[pair_key] = {
            "similarity_matrix": sim_matrix.tolist(),
            "shared_pairs": shared_pairs,
            "n_shared": len(shared_pairs),
        }

    # Identify universal axes (matched across ALL dialect pairs)
    # An axis k in dialect d is "universal" if it matches something in every other dialect
    universal_axes: dict[str, list[int]] = {}
    unique_axes: dict[str, list[int]] = {}

    for d in DIALECT_CODES:
        universal = []
        unique = []
        for i in range(top_n):
            # Check if axis i of dialect d matches at least one axis in every other dialect
            all_matched = True
            any_matched = False
            for d_other in DIALECT_CODES:
                if d_other == d:
                    continue
                pair_key = f"{d}_vs_{d_other}" if f"{d}_vs_{d_other}" in pair_results else f"{d_other}_vs_{d}"
                sim_data = pair_results[pair_key]
                sim_mat = np.array(sim_data["similarity_matrix"])
                # Determine orientation
                if pair_key.startswith(d):
                    row_sims = sim_mat[i, :]
                else:
                    row_sims = sim_mat[:, i]
                if np.max(row_sims) > cosine_threshold:
                    any_matched = True
                else:
                    all_matched = False
            if all_matched:
                universal.append(i + 1)
            if not any_matched:
                unique.append(i + 1)
        universal_axes[d] = universal
        unique_axes[d] = unique

    # Print summary
    print("  Pairwise shared axis counts (cosine > {:.2f}):".format(cosine_threshold))
    print()

    # Build a table
    header = f"  {'Pair':<25s} {'Shared axes':>12s}"
    print(header)
    print("  " + "-" * 40)
    for pair_key, data in sorted(pair_results.items()):
        print(f"  {pair_key:<25s} {data['n_shared']:>12d}")

    print()
    _subheader("Universal axes (shared with ALL other dialects)")
    for d in DIALECT_CODES:
        axes_str = ", ".join(str(a) for a in universal_axes[d]) if universal_axes[d] else "(none)"
        print(f"  {d:<12s}: {axes_str}")

    _subheader("Unique axes (not matched in ANY other dialect)")
    for d in DIALECT_CODES:
        axes_str = ", ".join(str(a) for a in unique_axes[d]) if unique_axes[d] else "(none)"
        print(f"  {d:<12s}: {axes_str}")

    output = {
        "parameters": {"top_n": top_n, "cosine_threshold": cosine_threshold},
        "pairwise": pair_results,
        "universal_axes": universal_axes,
        "unique_axes": unique_axes,
    }
    path = _save_json(output, "shared_axes.json")
    print(f"\n  Saved to {path.relative_to(PROJECT_ROOT)}")
    return output


# ===================================================================
# Analysis 3: Eigenvalue decay curves
# ===================================================================

def analysis_eigenvalue_decay() -> dict[str, Any]:
    """Compute eigenvalue energy curves and effective ranks."""
    _header("ANALYSIS 3: Eigenvalue Decay and Effective Rank")

    thresholds = [0.80, 0.90, 0.95]
    results: dict[str, Any] = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax_raw, ax_cum = axes

    colors = plt.cm.tab10(np.linspace(0, 1, len(DIALECT_CODES)))

    # Print header
    col_w = 12
    header_parts = [f"{'Dialect':<12s}"] + [f"{'r(' + str(int(t*100)) + '%)':>{col_w}s}" for t in thresholds]
    header_parts.append(f"{'Entropy':>{col_w}s}")
    header_parts.append(f"{'|lambda|_max':>{col_w}s}")
    header_parts.append(f"{'|lambda|_min':>{col_w}s}")
    print("  " + " ".join(header_parts))
    print("  " + "-" * (12 + (col_w + 1) * (len(thresholds) + 2)))

    spectra = _load_spectra()

    for idx, dialect in enumerate(DIALECT_CODES):
        ev = _load_eigenvalues(dialect)
        magnitudes = np.sort(np.abs(ev))[::-1]
        total_energy = np.sum(magnitudes)
        cum_energy = np.cumsum(magnitudes) / total_energy if total_energy > 0 else np.zeros_like(magnitudes)

        # Effective ranks
        effective_ranks = {}
        for t in thresholds:
            rank = int(np.searchsorted(cum_energy, t) + 1)
            effective_ranks[f"r{int(t*100)}"] = rank

        dialect_entropy = spectra.get(dialect, {}).get("entropy", 0.0)

        results[dialect] = {
            "eigenvalue_magnitudes": magnitudes.tolist(),
            "cumulative_energy": cum_energy.tolist(),
            "effective_ranks": effective_ranks,
            "entropy": dialect_entropy,
            "max_magnitude": float(magnitudes[0]),
            "min_magnitude": float(magnitudes[-1]),
        }

        # Print row
        row_parts = [f"{dialect:<12s}"]
        for t in thresholds:
            row_parts.append(f"{effective_ranks[f'r{int(t*100)}']:>{col_w}d}")
        row_parts.append(f"{dialect_entropy:>{col_w}.4f}")
        row_parts.append(f"{magnitudes[0]:>{col_w}.4f}")
        row_parts.append(f"{magnitudes[-1]:>{col_w}.6f}")
        print("  " + " ".join(row_parts))

        # Plots
        label = DIALECT_LABELS[dialect]
        ax_raw.semilogy(range(1, len(magnitudes) + 1), magnitudes,
                        color=colors[idx], label=label, linewidth=1.5, alpha=0.85)
        ax_cum.plot(range(1, len(cum_energy) + 1), cum_energy * 100,
                    color=colors[idx], label=label, linewidth=1.5, alpha=0.85)

    # Configure raw decay plot
    ax_raw.set_xlabel("Eigenvalue index (sorted by magnitude)")
    ax_raw.set_ylabel("|lambda| (log scale)")
    ax_raw.set_title("Eigenvalue magnitude decay")
    ax_raw.legend(fontsize=8, loc="upper right")
    ax_raw.grid(True, alpha=0.3)

    # Configure cumulative plot
    ax_cum.set_xlabel("Number of eigenvalues")
    ax_cum.set_ylabel("Cumulative energy (%)")
    ax_cum.set_title("Cumulative spectral energy")
    for t in thresholds:
        ax_cum.axhline(y=t * 100, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax_cum.text(1.5, t * 100 + 0.8, f"{int(t*100)}%", fontsize=7, color="gray")
    ax_cum.legend(fontsize=8, loc="lower right")
    ax_cum.grid(True, alpha=0.3)
    ax_cum.set_ylim(0, 102)

    fig.suptitle("EigenDialectos — Eigenvalue Decay Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = _save_figure(fig, "eigenvalue_decay.png")
    print(f"\n  Plot saved to {path.relative_to(PROJECT_ROOT)}")

    json_path = _save_json(results, "eigenvalue_decay.json")
    print(f"  Data saved to {json_path.relative_to(PROJECT_ROOT)}")
    return results


# ===================================================================
# Analysis 4: Dialectological validation (dendrogram + topology)
# ===================================================================

def analysis_dialectological_validation() -> dict[str, Any]:
    """Cluster dialects and compare with classical dialectological groupings."""
    _header("ANALYSIS 4: Dialectological Validation")

    dm, dialect_order = _load_distance_matrix()

    # Convert to condensed form for scipy
    condensed = squareform(dm, checks=False)

    # Ward linkage
    Z = linkage(condensed, method="ward")
    labels = [DIALECT_LABELS.get(d, d) for d in dialect_order]

    # --- Dendrogram plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    dendro = dendrogram(
        Z,
        labels=labels,
        ax=ax,
        leaf_rotation=35,
        leaf_font_size=10,
        color_threshold=0.7 * max(Z[:, 2]),
        above_threshold_color="gray",
    )
    ax.set_title("Dialectal Distance Dendrogram (Ward's method)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Distance")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = _save_figure(fig, "dendrogram_ward.png")
    print(f"  Dendrogram saved to {path.relative_to(PROJECT_ROOT)}")

    # --- Topology analysis ---
    _subheader("Tree topology analysis")

    # Extract merge order from linkage
    n = len(dialect_order)
    merge_log: list[str] = []
    cluster_members: dict[int, list[str]] = {}
    for i in range(n):
        cluster_members[i] = [dialect_order[i]]

    for step, (c1, c2, dist, count) in enumerate(Z):
        c1, c2 = int(c1), int(c2)
        new_id = n + step
        members_1 = cluster_members[c1]
        members_2 = cluster_members[c2]
        cluster_members[new_id] = members_1 + members_2
        merge_str = (f"  Step {step+1}: merge {{{', '.join(members_1)}}} + "
                     f"{{{', '.join(members_2)}}} at distance {dist:.4f}")
        merge_log.append(merge_str)
        print(merge_str)

    # --- Known dialectological checks ---
    _subheader("Dialectological hypothesis testing")

    # Build helper: for each dialect, its leaf index
    d_idx = {d: dialect_order.index(d) for d in dialect_order}

    def _pairwise_dist(a: str, b: str) -> float:
        return float(dm[d_idx[a], d_idx[b]])

    checks: list[dict[str, Any]] = []

    # Check 1: Iberian varieties grouped?
    iberian = ["ES_PEN", "ES_AND", "ES_CAN"]
    iberian_dists = [_pairwise_dist(a, b) for a, b in combinations(iberian, 2)]
    non_iberian_dists = []
    for ib in iberian:
        for am in [d for d in dialect_order if d not in iberian]:
            non_iberian_dists.append(_pairwise_dist(ib, am))
    iberian_cohesion = np.mean(iberian_dists) < np.mean(non_iberian_dists)
    check1 = {
        "hypothesis": "Iberian varieties (PEN, AND, CAN) cluster together",
        "iberian_mean_dist": float(np.mean(iberian_dists)),
        "cross_mean_dist": float(np.mean(non_iberian_dists)),
        "supported": bool(iberian_cohesion),
    }
    checks.append(check1)
    supported_str = "SUPPORTED" if check1["supported"] else "NOT SUPPORTED"
    print(f"  [1] {check1['hypothesis']}")
    print(f"      Iberian mean dist = {check1['iberian_mean_dist']:.4f}, "
          f"cross mean = {check1['cross_mean_dist']:.4f} => {supported_str}")

    # Check 2: CAN-CAR Atlantic link?
    can_car_dist = _pairwise_dist("ES_CAN", "ES_CAR")
    can_pen_dist = _pairwise_dist("ES_CAN", "ES_PEN")
    atlantic_link = can_car_dist < can_pen_dist
    check2 = {
        "hypothesis": "Canarian-Caribbean Atlantic link (CAN closer to CAR than to PEN)",
        "CAN_CAR_dist": float(can_car_dist),
        "CAN_PEN_dist": float(can_pen_dist),
        "supported": bool(atlantic_link),
    }
    checks.append(check2)
    supported_str = "SUPPORTED" if check2["supported"] else "NOT SUPPORTED"
    print(f"  [2] {check2['hypothesis']}")
    print(f"      d(CAN,CAR) = {check2['CAN_CAR_dist']:.4f}, "
          f"d(CAN,PEN) = {check2['CAN_PEN_dist']:.4f} => {supported_str}")

    # Check 3: RIO closer to AND (Andalusian migration) or CHI (neighbor)?
    rio_and_dist = _pairwise_dist("ES_RIO", "ES_AND")
    rio_chi_dist = _pairwise_dist("ES_RIO", "ES_CHI")
    check3 = {
        "hypothesis": "Rioplatense-Andalusian affinity (historical migration link)",
        "RIO_AND_dist": float(rio_and_dist),
        "RIO_CHI_dist": float(rio_chi_dist),
        "closer_to": "ES_AND" if rio_and_dist < rio_chi_dist else "ES_CHI",
        "supported": bool(rio_and_dist < rio_chi_dist),
    }
    checks.append(check3)
    supported_str = "SUPPORTED" if check3["supported"] else "NOT SUPPORTED"
    print(f"  [3] {check3['hypothesis']}")
    print(f"      d(RIO,AND) = {check3['RIO_AND_dist']:.4f}, "
          f"d(RIO,CHI) = {check3['RIO_CHI_dist']:.4f}, "
          f"closer to {check3['closer_to']} => {supported_str}")

    # Check 4: MEX-CAR proximity (shared features in tierras bajas)
    mex_car_dist = _pairwise_dist("ES_MEX", "ES_CAR")
    mex_chi_dist = _pairwise_dist("ES_MEX", "ES_CHI")
    check4 = {
        "hypothesis": "Mexican-Caribbean proximity (tierras bajas shared features)",
        "MEX_CAR_dist": float(mex_car_dist),
        "MEX_CHI_dist": float(mex_chi_dist),
        "supported": bool(mex_car_dist < mex_chi_dist),
    }
    checks.append(check4)
    supported_str = "SUPPORTED" if check4["supported"] else "NOT SUPPORTED"
    print(f"  [4] {check4['hypothesis']}")
    print(f"      d(MEX,CAR) = {check4['MEX_CAR_dist']:.4f}, "
          f"d(MEX,CHI) = {check4['MEX_CHI_dist']:.4f} => {supported_str}")

    # Check 5: Andean (AND_BO) outlier status
    andbo_dists = [_pairwise_dist("ES_AND_BO", d) for d in dialect_order if d != "ES_AND_BO"]
    other_pair_dists = [
        _pairwise_dist(a, b)
        for a, b in combinations([d for d in dialect_order if d != "ES_AND_BO"], 2)
    ]
    check5 = {
        "hypothesis": "Andean (AND_BO) is an outlier with highest mean distance to others",
        "AND_BO_mean_dist": float(np.mean(andbo_dists)),
        "other_pairs_mean_dist": float(np.mean(other_pair_dists)),
        "supported": bool(np.mean(andbo_dists) > np.mean(other_pair_dists)),
    }
    checks.append(check5)
    supported_str = "SUPPORTED" if check5["supported"] else "NOT SUPPORTED"
    print(f"  [5] {check5['hypothesis']}")
    print(f"      AND_BO mean dist = {check5['AND_BO_mean_dist']:.4f}, "
          f"other pairs mean = {check5['other_pairs_mean_dist']:.4f} => {supported_str}")

    # Check 6: Henriquez Urena zones
    _subheader("Henriquez Urena (1921) zone distances")
    hu_zones = {
        "Antillean (CAR)": ["ES_CAR"],
        "Mexican (MEX)": ["ES_MEX"],
        "Andean (AND_BO)": ["ES_AND_BO"],
        "Rioplatense (RIO)": ["ES_RIO"],
        "Chilean (CHI)": ["ES_CHI"],
    }
    zone_pairs = list(combinations(hu_zones.keys(), 2))
    for z_a, z_b in zone_pairs:
        dists = []
        for da in hu_zones[z_a]:
            for db in hu_zones[z_b]:
                dists.append(_pairwise_dist(da, db))
        mean_d = float(np.mean(dists))
        print(f"  {z_a:<25s} <-> {z_b:<25s} : {mean_d:.4f}")

    output = {
        "merge_log": merge_log,
        "dialectological_checks": checks,
        "linkage_matrix": Z.tolist(),
    }
    path = _save_json(output, "dialectological_validation.json")
    print(f"\n  Saved to {path.relative_to(PROJECT_ROOT)}")
    return output


# ===================================================================
# Analysis 5: Eigenvalue spectrum comparison (KL, Wasserstein, Pearson)
# ===================================================================

def analysis_spectrum_comparison() -> dict[str, Any]:
    """Compare eigenvalue spectra across dialect pairs using divergence metrics."""
    _header("ANALYSIS 5: Eigenvalue Spectrum Comparison")

    n = len(DIALECT_CODES)
    kl_matrix = np.zeros((n, n))
    wass_matrix = np.zeros((n, n))
    pearson_matrix = np.zeros((n, n))

    # Pre-compute sorted eigenvalue magnitudes for each dialect
    spectra: dict[str, npt.NDArray[np.float64]] = {}
    for dialect in DIALECT_CODES:
        ev = _load_eigenvalues(dialect)
        spectra[dialect] = np.sort(np.abs(ev))[::-1]

    for i, d_a in enumerate(DIALECT_CODES):
        for j, d_b in enumerate(DIALECT_CODES):
            s_a = spectra[d_a]
            s_b = spectra[d_b]

            # Normalize to probability distributions for KL
            eps = 1e-12
            p_a = s_a / (s_a.sum() + eps) + eps
            p_b = s_b / (s_b.sum() + eps) + eps
            p_a = p_a / p_a.sum()
            p_b = p_b / p_b.sum()

            # Symmetric KL divergence: (KL(P||Q) + KL(Q||P)) / 2
            kl_fwd = float(kl_div_scipy(p_a, p_b))
            kl_rev = float(kl_div_scipy(p_b, p_a))
            kl_matrix[i, j] = (kl_fwd + kl_rev) / 2.0

            # Wasserstein distance (earth mover's) on magnitude vectors
            wass_matrix[i, j] = float(wasserstein_distance(s_a, s_b))

            # Pearson correlation
            if i == j:
                pearson_matrix[i, j] = 1.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r, _ = pearsonr(s_a, s_b)
                pearson_matrix[i, j] = float(r) if np.isfinite(r) else 0.0

    # Print tables
    short_labels = [d.replace("ES_", "") for d in DIALECT_CODES]

    def _print_matrix(name: str, mat: npt.NDArray, fmt: str = ".4f") -> None:
        _subheader(name)
        col_w = 10
        header = f"  {'':>{col_w}s}" + "".join(f"{l:>{col_w}s}" for l in short_labels)
        print(header)
        for i, label in enumerate(short_labels):
            row = f"  {label:>{col_w}s}"
            for j in range(n):
                row += f"{mat[i, j]:{col_w}{fmt}}"
            print(row)

    _print_matrix("Symmetric KL Divergence", kl_matrix, ".6f")
    _print_matrix("Wasserstein Distance", wass_matrix, ".4f")
    _print_matrix("Pearson Correlation", pearson_matrix, ".4f")

    # --- Heatmap plots ---
    fig, axes_arr = plt.subplots(1, 3, figsize=(18, 5.5))
    metrics = [
        ("Symmetric KL Divergence", kl_matrix, "Oranges"),
        ("Wasserstein Distance", wass_matrix, "Reds"),
        ("Pearson Correlation", pearson_matrix, "RdYlGn"),
    ]

    for ax, (title, mat, cmap) in zip(axes_arr, metrics):
        im = ax.imshow(mat, cmap=cmap, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_labels, rotation=45, ha="right")
        ax.set_yticklabels(short_labels)
        ax.set_title(title, fontsize=11)
        # Annotate cells
        for ii in range(n):
            for jj in range(n):
                val = mat[ii, jj]
                txt_color = "white" if val > (mat.max() + mat.min()) / 2 else "black"
                if cmap == "RdYlGn":
                    txt_color = "black"
                ax.text(jj, ii, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=txt_color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("EigenDialectos — Spectral Distance Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = _save_figure(fig, "spectrum_comparison_heatmaps.png")
    print(f"\n  Heatmaps saved to {path.relative_to(PROJECT_ROOT)}")

    output = {
        "dialect_order": DIALECT_CODES,
        "kl_divergence": kl_matrix.tolist(),
        "wasserstein_distance": wass_matrix.tolist(),
        "pearson_correlation": pearson_matrix.tolist(),
    }
    path = _save_json(output, "spectrum_comparison.json")
    print(f"  Data saved to {path.relative_to(PROJECT_ROOT)}")
    return output


# ===================================================================
# Analysis 6: SVD comparison — normality of transformation matrices
# ===================================================================

def analysis_svd_comparison() -> dict[str, Any]:
    """Compare SVD singular values with eigenvalue magnitudes to assess normality."""
    _header("ANALYSIS 6: SVD Comparison — Normality of Transformation Matrices")

    print("  A normal matrix satisfies W W^H = W^H W, and its singular values equal")
    print("  the magnitudes of its eigenvalues. Departure from normality indicates")
    print("  non-trivial geometric distortion beyond pure scaling/rotation.")
    print()

    results: dict[str, Any] = {}

    col_w = 12
    header = (f"  {'Dialect':<12s}"
              f"{'||Sigma||':>{col_w}s}"
              f"{'||Lambda||':>{col_w}s}"
              f"{'Rel. Err':>{col_w}s}"
              f"{'Frobenius':>{col_w}s}"
              f"{'Cond(W)':>{col_w}s}"
              f"{'Normal?':>{col_w}s}")
    print(header)
    print("  " + "-" * (12 + col_w * 6))

    for dialect in DIALECT_CODES:
        W = _load_W(dialect)
        ev = _load_eigenvalues(dialect)

        # SVD
        U, sigma, Vt = np.linalg.svd(W)

        # Eigenvalue magnitudes sorted descending
        ev_mags = np.sort(np.abs(ev))[::-1]
        sigma_sorted = np.sort(sigma)[::-1]

        # Compare: relative error between sigma and |lambda|
        # Both are sorted descending
        norm_sigma = np.linalg.norm(sigma_sorted)
        norm_ev = np.linalg.norm(ev_mags)
        diff_norm = np.linalg.norm(sigma_sorted - ev_mags)
        relative_error = diff_norm / (norm_sigma + 1e-15)

        # Frobenius norm of [W, W^H] (commutator as normality check)
        # For real W, W^H = W^T
        commutator = W @ W.T - W.T @ W
        commutator_norm = np.linalg.norm(commutator, "fro")

        # Condition number
        cond = float(np.linalg.cond(W))

        # Normality threshold: relative error < 0.01 and commutator small
        is_normal = relative_error < 0.05 and commutator_norm < 0.1

        results[dialect] = {
            "singular_values": sigma_sorted.tolist(),
            "eigenvalue_magnitudes": ev_mags.tolist(),
            "sigma_norm": float(norm_sigma),
            "eigenvalue_norm": float(norm_ev),
            "relative_error": float(relative_error),
            "commutator_frobenius": float(commutator_norm),
            "condition_number": cond,
            "approximately_normal": bool(is_normal),
        }

        normal_str = "YES" if is_normal else "NO"
        print(f"  {dialect:<12s}"
              f"{norm_sigma:{col_w}.4f}"
              f"{norm_ev:{col_w}.4f}"
              f"{relative_error:{col_w}.6f}"
              f"{commutator_norm:{col_w}.4f}"
              f"{cond:{col_w}.2f}"
              f"{normal_str:>{col_w}s}")

    # --- Plot: SVD vs eigenvalue comparison for each dialect ---
    n_dialects = len(DIALECT_CODES)
    n_cols = 4
    n_rows = (n_dialects + n_cols - 1) // n_cols
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes_flat = axes_grid.flatten() if n_dialects > 1 else [axes_grid]

    for idx, dialect in enumerate(DIALECT_CODES):
        ax = axes_flat[idx]
        data = results[dialect]
        sigma_vals = np.array(data["singular_values"])
        ev_vals = np.array(data["eigenvalue_magnitudes"])
        x = np.arange(1, len(sigma_vals) + 1)

        ax.plot(x, sigma_vals, "b-", linewidth=1.2, alpha=0.8, label="Singular values")
        ax.plot(x, ev_vals, "r--", linewidth=1.2, alpha=0.8, label="|Eigenvalues|")
        ax.set_title(f"{DIALECT_LABELS[dialect]}  (err={data['relative_error']:.4f})",
                     fontsize=9)
        ax.set_xlabel("Index", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Turn off unused subplots
    for idx in range(n_dialects, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("SVD Singular Values vs Eigenvalue Magnitudes", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = _save_figure(fig, "svd_comparison.png")
    print(f"\n  Plot saved to {path.relative_to(PROJECT_ROOT)}")

    # Summary
    _subheader("Interpretation")
    normal_count = sum(1 for d in results.values() if d["approximately_normal"])
    print(f"  {normal_count}/{n_dialects} transformation matrices are approximately normal.")
    if normal_count == n_dialects:
        print("  All matrices are near-normal => eigenvalue magnitudes fully characterize")
        print("  the geometric action. The spectral decomposition is a faithful summary.")
    else:
        non_normal = [d for d, v in results.items() if not v["approximately_normal"]]
        print(f"  Non-normal matrices: {', '.join(non_normal)}")
        print("  For these, the eigenvalue spectrum alone may miss important geometric")
        print("  distortions (shearing). The singular values capture additional structure.")

    path = _save_json(results, "svd_comparison.json")
    print(f"\n  Saved to {path.relative_to(PROJECT_ROOT)}")
    return results


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep spectral analysis of EigenDialectos pipeline results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top eigenvectors to analyze per dialect (default: 5)",
    )
    parser.add_argument(
        "--top-words", type=int, default=20,
        help="Number of top words per eigenvector (default: 20)",
    )
    parser.add_argument(
        "--cosine-threshold", type=float, default=0.3,
        help="Cosine similarity threshold for shared axes (default: 0.3)",
    )
    parser.add_argument(
        "--top-n-axes", type=int, default=10,
        help="Number of top axes to compare across dialects (default: 10)",
    )
    args = parser.parse_args()

    # Ensure output directory
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"EigenDialectos — Comprehensive Spectral Analysis")
    print(f"Timestamp: {timestamp}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {ANALYSIS_DIR}")

    # Validate inputs exist
    missing = []
    for dialect in DIALECT_CODES:
        for kind in ["eigenvalues", "eigenvectors", "W"]:
            p = CHECKPOINTS / f"{kind}_{dialect}.npy"
            if not p.exists():
                missing.append(str(p))
        ep = EMBEDDINGS / f"{dialect}.npy"
        if not ep.exists():
            missing.append(str(ep))
    if not (EMBEDDINGS / "vocab.json").exists():
        missing.append(str(EMBEDDINGS / "vocab.json"))
    if not (CHECKPOINTS / "distance_matrix.npy").exists():
        missing.append(str(CHECKPOINTS / "distance_matrix.npy"))

    if missing:
        print("\nERROR: Missing required checkpoint files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    vocab = _load_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Dialects: {', '.join(DIALECT_CODES)}")

    # Run all analyses
    analysis_eigenvector_words(vocab, top_k=args.top_k, top_words=args.top_words)
    analysis_shared_axes(top_n=args.top_n_axes, cosine_threshold=args.cosine_threshold)
    analysis_eigenvalue_decay()
    analysis_dialectological_validation()
    analysis_spectrum_comparison()
    analysis_svd_comparison()

    # Final summary
    _header("ANALYSIS COMPLETE")
    print(f"  All results saved to: {ANALYSIS_DIR.relative_to(PROJECT_ROOT)}/")
    print()
    artifacts = sorted(ANALYSIS_DIR.glob("*"))
    for a in artifacts:
        size_kb = a.stat().st_size / 1024
        print(f"    {a.name:<45s} {size_kb:>8.1f} KB")
    print()


if __name__ == "__main__":
    main()
