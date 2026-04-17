#!/usr/bin/env python3
"""Compare spectral analysis results across all 3 transformation methods.

Runs the full spectral pipeline (transform -> eigendecompose -> spectrum ->
distance matrix) for lstsq, procrustes, and nuclear methods, then produces
a comprehensive comparison of entropy rankings, distance matrices, closest
pairs, condition numbers, and eigenvalue spectra.

Outputs
-------
- stdout : comparison tables
- outputs/analysis/method_comparison.json
- outputs/analysis/method_comparison.png

Usage
-----
    python scripts/compare_methods.py
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eigendialectos.constants import DIALECT_NAMES, DialectCode  # noqa: E402
from eigendialectos.spectral.distance import compute_distance_matrix  # noqa: E402
from eigendialectos.spectral.eigendecomposition import eigendecompose  # noqa: E402
from eigendialectos.spectral.eigenspectrum import compute_eigenspectrum  # noqa: E402
from eigendialectos.spectral.transformation import compute_all_transforms  # noqa: E402
from eigendialectos.spectral.utils import check_condition_number, is_orthogonal  # noqa: E402
from eigendialectos.types import EmbeddingMatrix  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_DIR = PROJECT_ROOT / "outputs" / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis"

DIALECT_CODES = [
    DialectCode.ES_PEN,
    DialectCode.ES_AND,
    DialectCode.ES_CAN,
    DialectCode.ES_RIO,
    DialectCode.ES_MEX,
    DialectCode.ES_CAR,
    DialectCode.ES_CHI,
    DialectCode.ES_AND_BO,
]

METHODS = ["lstsq", "procrustes", "nuclear"]
REFERENCE = DialectCode.ES_PEN
REGULARIZATION = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embeddings() -> dict[DialectCode, EmbeddingMatrix]:
    """Load embedding matrices and shared vocabulary from disk."""
    vocab_path = EMBEDDING_DIR / "vocab.json"
    with open(vocab_path) as f:
        vocab: list[str] = json.load(f)

    embeddings: dict[DialectCode, EmbeddingMatrix] = {}
    for code in DIALECT_CODES:
        npy_path = EMBEDDING_DIR / f"{code.value}.npy"
        data = np.load(str(npy_path))
        embeddings[code] = EmbeddingMatrix(data=data, vocab=vocab, dialect_code=code)

    print(f"Loaded {len(embeddings)} embeddings, shape {data.shape}, vocab size {len(vocab)}")
    return embeddings


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Extract the strict upper triangle of a square matrix as a flat vector."""
    n = matrix.shape[0]
    indices = np.triu_indices(n, k=1)
    return matrix[indices]


def run_pipeline_for_method(
    embeddings: dict[DialectCode, EmbeddingMatrix],
    method: str,
) -> dict:
    """Run the full spectral pipeline for a single transformation method.

    Returns a dict with transforms, eigendecompositions, spectra, entropies,
    distance_matrix, condition_numbers, and sorted dialect codes.
    """
    print(f"\n{'='*60}")
    print(f"  Method: {method.upper()}")
    print(f"{'='*60}")

    # 1. Compute transformation matrices
    print(f"  Computing transformations (reference={REFERENCE.value})...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transforms = compute_all_transforms(
            embeddings,
            reference=REFERENCE,
            method=method,
            regularization=REGULARIZATION,
        )
    if caught:
        for w in caught:
            print(f"    [WARN] {w.message}")

    # 2. Condition numbers and orthogonality check
    condition_numbers: dict[str, float] = {}
    orthogonality: dict[str, bool] = {}
    for code, W in transforms.items():
        if code == REFERENCE:
            continue
        cond = float(np.linalg.cond(W.data))
        condition_numbers[code.value] = cond
        orthogonality[code.value] = is_orthogonal(W.data)

    print(f"  Condition numbers: min={min(condition_numbers.values()):.2f}, "
          f"max={max(condition_numbers.values()):.2f}")
    if method == "procrustes":
        all_ortho = all(orthogonality.values())
        print(f"  All transforms orthogonal: {all_ortho}")

    # 3. Eigendecompose each transformation
    print("  Eigendecomposing...")
    eigendecompositions: dict[DialectCode, object] = {}
    for code, W in transforms.items():
        if code == REFERENCE:
            continue
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            eigendecompositions[code] = eigendecompose(W)

    # 4. Compute spectra
    print("  Computing eigenspectra...")
    spectra = {}
    entropies: dict[DialectCode, float] = {}
    for code, eigen in eigendecompositions.items():
        spec = compute_eigenspectrum(eigen)
        spectra[code] = spec
        entropies[code] = spec.entropy

    # 5. Eigenvalue magnitude statistics (for procrustes, should be ~1)
    eigenvalue_stats: dict[str, dict] = {}
    for code, eigen in eigendecompositions.items():
        mags = np.abs(eigen.eigenvalues)
        eigenvalue_stats[code.value] = {
            "mean_magnitude": float(np.mean(mags)),
            "std_magnitude": float(np.std(mags)),
            "min_magnitude": float(np.min(mags)),
            "max_magnitude": float(np.max(mags)),
        }

    if method == "procrustes":
        # Verify eigenvalues have magnitude ~1
        all_mags = np.concatenate([np.abs(e.eigenvalues) for e in eigendecompositions.values()])
        mean_dev = float(np.mean(np.abs(all_mags - 1.0)))
        print(f"  Procrustes eigenvalue |lambda|: mean deviation from 1.0 = {mean_dev:.6f}")

    # 6. Compute distance matrix
    print("  Computing pairwise distance matrix...")
    # We need transforms/spectra/entropies for all codes including reference
    # For reference, build an identity-like entry if not present
    # Actually compute_all_transforms already includes reference (identity-like)
    # But we only eigendecomposed non-reference codes. For distance, we need
    # the reference too.
    ref_W = transforms[REFERENCE]
    ref_eigen = eigendecompose(ref_W)
    ref_spec = compute_eigenspectrum(ref_eigen)
    spectra[REFERENCE] = ref_spec
    entropies[REFERENCE] = ref_spec.entropy

    dist_matrix = compute_distance_matrix(
        transforms=transforms,
        spectra=spectra,
        entropies=entropies,
        method="combined",
    )

    # Sort codes to match distance matrix ordering
    sorted_codes = sorted(transforms.keys(), key=lambda c: c.value)

    # Print entropy ranking
    ranking = sorted(entropies.items(), key=lambda x: -x[1])
    print("\n  Entropy ranking (H_i, descending):")
    for i, (code, h) in enumerate(ranking, 1):
        marker = " (ref)" if code == REFERENCE else ""
        print(f"    {i}. {code.value:10s}  H={h:.6f}{marker}")

    return {
        "transforms": transforms,
        "eigendecompositions": eigendecompositions,
        "spectra": spectra,
        "entropies": entropies,
        "distance_matrix": dist_matrix,
        "sorted_codes": sorted_codes,
        "condition_numbers": condition_numbers,
        "orthogonality": orthogonality,
        "eigenvalue_stats": eigenvalue_stats,
    }


def find_top_k_pairs(
    dist_matrix: np.ndarray,
    sorted_codes: list[DialectCode],
    k: int = 5,
) -> list[tuple[str, str, float]]:
    """Return the top-k closest pairs from a distance matrix."""
    n = dist_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((
                sorted_codes[i].value,
                sorted_codes[j].value,
                float(dist_matrix[i, j]),
            ))
    pairs.sort(key=lambda x: x[2])
    return pairs[:k]


def compare_results(results: dict[str, dict]) -> dict:
    """Compare results across all methods and return a comparison dict."""
    comparison: dict[str, object] = {}

    methods = list(results.keys())

    # ----- 1. Entropy ranking comparison (Spearman) -----
    print(f"\n{'='*60}")
    print("  ENTROPY RANKING COMPARISON")
    print(f"{'='*60}")

    # Exclude reference from ranking comparison
    non_ref_codes = [c for c in DIALECT_CODES if c != REFERENCE]

    entropy_rankings: dict[str, list[str]] = {}
    entropy_values: dict[str, list[float]] = {}
    for method in methods:
        ents = results[method]["entropies"]
        ranked = sorted(non_ref_codes, key=lambda c: -ents[c])
        entropy_rankings[method] = [c.value for c in ranked]
        entropy_values[method] = [ents[c] for c in non_ref_codes]

    # Print rankings side by side
    print(f"\n  {'Rank':<6}", end="")
    for m in methods:
        print(f"  {m:>20s}", end="")
    print()
    print("  " + "-" * (6 + 22 * len(methods)))

    max_rank = len(non_ref_codes)
    for i in range(max_rank):
        print(f"  {i+1:<6}", end="")
        for m in methods:
            code = entropy_rankings[m][i]
            h = results[m]["entropies"][DialectCode(code)]
            print(f"  {code:>10s} ({h:.4f})", end="")
        print()

    # Spearman rank correlation between each pair of methods
    spearman_corrs: dict[str, float] = {}
    print("\n  Spearman rank correlations (entropy ordering):")
    for m1, m2 in combinations(methods, 2):
        vals1 = entropy_values[m1]
        vals2 = entropy_values[m2]
        rho, pval = spearmanr(vals1, vals2)
        key = f"{m1}_vs_{m2}"
        spearman_corrs[key] = float(rho)
        print(f"    {m1} vs {m2}: rho={rho:.6f}, p={pval:.4e}")

    comparison["entropy_rankings"] = entropy_rankings
    comparison["spearman_correlations"] = spearman_corrs

    # ----- 2. Distance matrix correlation (Pearson on upper triangle) -----
    print(f"\n{'='*60}")
    print("  DISTANCE MATRIX CORRELATION")
    print(f"{'='*60}")

    dist_correlations: dict[str, float] = {}
    dist_vectors: dict[str, np.ndarray] = {}
    for m in methods:
        dist_vectors[m] = upper_triangle(results[m]["distance_matrix"])

    print("\n  Pearson correlations between distance matrices (upper triangle):")
    for m1, m2 in combinations(methods, 2):
        r, pval = pearsonr(dist_vectors[m1], dist_vectors[m2])
        key = f"{m1}_vs_{m2}"
        dist_correlations[key] = float(r)
        print(f"    {m1} vs {m2}: r={r:.6f}, p={pval:.4e}")

    comparison["distance_matrix_correlations"] = dist_correlations

    # ----- 3. Top-5 closest pairs -----
    print(f"\n{'='*60}")
    print("  TOP-5 CLOSEST DIALECT PAIRS")
    print(f"{'='*60}")

    top_pairs: dict[str, list] = {}
    for m in methods:
        pairs = find_top_k_pairs(
            results[m]["distance_matrix"],
            results[m]["sorted_codes"],
            k=5,
        )
        top_pairs[m] = [(a, b, round(d, 6)) for a, b, d in pairs]

    for m in methods:
        print(f"\n  {m}:")
        for rank, (a, b, d) in enumerate(top_pairs[m], 1):
            print(f"    {rank}. {a} <-> {b}  d={d:.6f}")

    # Check overlap between methods
    pair_sets: dict[str, set] = {}
    for m in methods:
        pair_sets[m] = {(a, b) for a, b, _ in top_pairs[m]}

    print("\n  Pair overlap (Jaccard index on top-5 sets):")
    pair_overlaps: dict[str, float] = {}
    for m1, m2 in combinations(methods, 2):
        intersection = len(pair_sets[m1] & pair_sets[m2])
        union = len(pair_sets[m1] | pair_sets[m2])
        jaccard = intersection / max(union, 1)
        key = f"{m1}_vs_{m2}"
        pair_overlaps[key] = float(jaccard)
        print(f"    {m1} vs {m2}: {intersection}/{union} = {jaccard:.2f}")

    comparison["top5_pairs"] = top_pairs
    comparison["top5_pair_overlap_jaccard"] = pair_overlaps

    # ----- 4. Condition numbers -----
    print(f"\n{'='*60}")
    print("  CONDITION NUMBERS")
    print(f"{'='*60}")

    cond_summary: dict[str, dict] = {}
    print(f"\n  {'Dialect':<12}", end="")
    for m in methods:
        print(f"  {m:>14s}", end="")
    print()
    print("  " + "-" * (12 + 16 * len(methods)))

    for code in DIALECT_CODES:
        if code == REFERENCE:
            continue
        print(f"  {code.value:<12}", end="")
        for m in methods:
            cond = results[m]["condition_numbers"].get(code.value, float("nan"))
            print(f"  {cond:>14.2f}", end="")
        print()

    for m in methods:
        conds = list(results[m]["condition_numbers"].values())
        cond_summary[m] = {
            "mean": float(np.mean(conds)),
            "median": float(np.median(conds)),
            "min": float(np.min(conds)),
            "max": float(np.max(conds)),
        }
        print(f"\n  {m}: mean={cond_summary[m]['mean']:.2f}, "
              f"median={cond_summary[m]['median']:.2f}, "
              f"range=[{cond_summary[m]['min']:.2f}, {cond_summary[m]['max']:.2f}]")

    comparison["condition_numbers"] = cond_summary

    # ----- 5. Eigenvalue spectra similarity -----
    print(f"\n{'='*60}")
    print("  EIGENVALUE SPECTRA SIMILARITY")
    print(f"{'='*60}")

    spectra_similarity: dict[str, dict] = {}
    for code in DIALECT_CODES:
        if code == REFERENCE:
            continue
        sims = {}
        # For each pair of methods, compute cosine similarity of eigenvalue spectra
        for m1, m2 in combinations(methods, 2):
            ev1 = results[m1]["spectra"][code].eigenvalues_sorted
            ev2 = results[m2]["spectra"][code].eigenvalues_sorted
            # Pad to same length
            max_len = max(len(ev1), len(ev2))
            v1 = np.zeros(max_len)
            v2 = np.zeros(max_len)
            v1[:len(ev1)] = ev1
            v2[:len(ev2)] = ev2
            # Cosine similarity
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-15 and n2 > 1e-15:
                cos_sim = float(np.dot(v1, v2) / (n1 * n2))
            else:
                cos_sim = 0.0
            sims[f"{m1}_vs_{m2}"] = cos_sim
        spectra_similarity[code.value] = sims

    print(f"\n  Cosine similarity of eigenvalue spectra per dialect:")
    print(f"  {'Dialect':<12}", end="")
    for m1, m2 in combinations(methods, 2):
        label = f"{m1[:4]}v{m2[:4]}"
        print(f"  {label:>12s}", end="")
    print()
    print("  " + "-" * (12 + 14 * len(list(combinations(methods, 2)))))

    for code in DIALECT_CODES:
        if code == REFERENCE:
            continue
        print(f"  {code.value:<12}", end="")
        for m1, m2 in combinations(methods, 2):
            key = f"{m1}_vs_{m2}"
            sim = spectra_similarity[code.value][key]
            print(f"  {sim:>12.6f}", end="")
        print()

    # Average spectra similarity per method pair
    avg_spectra_sim: dict[str, float] = {}
    for m1, m2 in combinations(methods, 2):
        key = f"{m1}_vs_{m2}"
        vals = [spectra_similarity[c.value][key] for c in DIALECT_CODES if c != REFERENCE]
        avg_spectra_sim[key] = float(np.mean(vals))
    print("\n  Average cosine similarity of eigenvalue spectra:")
    for key, val in avg_spectra_sim.items():
        print(f"    {key}: {val:.6f}")

    comparison["eigenvalue_spectra_similarity"] = spectra_similarity
    comparison["avg_eigenvalue_spectra_similarity"] = avg_spectra_sim

    # ----- 6. Procrustes eigenvalue magnitude check -----
    if "procrustes" in results:
        print(f"\n{'='*60}")
        print("  PROCRUSTES EIGENVALUE MAGNITUDE CHECK (should be ~1)")
        print(f"{'='*60}")
        for code in DIALECT_CODES:
            if code == REFERENCE:
                continue
            stats = results["procrustes"]["eigenvalue_stats"][code.value]
            print(f"  {code.value:<12}  mean|lambda|={stats['mean_magnitude']:.6f}  "
                  f"std={stats['std_magnitude']:.6f}  "
                  f"range=[{stats['min_magnitude']:.6f}, {stats['max_magnitude']:.6f}]")

    return comparison


def plot_comparison(results: dict[str, dict], output_path: Path) -> None:
    """Create side-by-side entropy bar charts for each method."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Spectral Entropy by Method", fontsize=16, fontweight="bold")

    # Consistent dialect ordering (non-reference, sorted by lstsq entropy)
    non_ref_codes = [c for c in DIALECT_CODES if c != REFERENCE]
    lstsq_ents = results["lstsq"]["entropies"]
    sorted_by_lstsq = sorted(non_ref_codes, key=lambda c: -lstsq_ents[c])
    labels = [c.value for c in sorted_by_lstsq]

    # Color map
    cmap = plt.cm.Set2
    colors = [cmap(i / len(labels)) for i in range(len(labels))]

    for ax_idx, method in enumerate(METHODS):
        ax = axes[ax_idx]
        ents = results[method]["entropies"]
        values = [ents[c] for c in sorted_by_lstsq]

        bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black",
                       linewidth=0.5)
        ax.set_title(method.upper(), fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Dialect")
        if ax_idx == 0:
            ax.set_ylabel("Spectral Entropy (H)")

        # Annotate bars with values
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {output_path}")


def serialize_for_json(obj: object) -> object:
    """Recursively convert numpy types and enums for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  EigenDialectos: Multi-Method Spectral Comparison")
    print("=" * 60)

    # Load embeddings once
    embeddings = load_embeddings()

    # Run pipeline for each method
    results: dict[str, dict] = {}
    for method in METHODS:
        results[method] = run_pipeline_for_method(embeddings, method)

    # Compare results
    comparison = compare_results(results)

    # Build summary for JSON export
    export: dict[str, object] = {
        "methods": METHODS,
        "reference": REFERENCE.value,
        "regularization": REGULARIZATION,
        "dialect_codes": [c.value for c in DIALECT_CODES],
        "comparison": comparison,
    }

    # Add per-method data
    per_method: dict[str, dict] = {}
    for method in METHODS:
        r = results[method]
        per_method[method] = {
            "entropies": {c.value: h for c, h in r["entropies"].items()},
            "condition_numbers": r["condition_numbers"],
            "orthogonality": r["orthogonality"],
            "eigenvalue_stats": r["eigenvalue_stats"],
            "distance_matrix": r["distance_matrix"].tolist(),
            "sorted_codes": [c.value for c in r["sorted_codes"]],
        }
    export["per_method"] = per_method

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "method_comparison.json"
    with open(json_path, "w") as f:
        json.dump(serialize_for_json(export), f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved to {json_path}")

    # Save plot
    png_path = OUTPUT_DIR / "method_comparison.png"
    plot_comparison(results, png_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
