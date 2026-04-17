#!/usr/bin/env python3
"""Run interpretable probing + dialectometric validation on v3_full embeddings."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eigen3.constants import ALL_VARIETIES, REFERENCE_VARIETY
from eigen3.transformation import compute_all_W
from eigen3.decomposition import eigendecompose
from eigen3.distance import spectral_distance
from eigen3.probing import probe_spectral_modes, format_probing_report
from eigen3.validation import validate_dialectometry, format_validation_report

EMB_DIR = ROOT / "outputs" / "eigen3_full"


def load_embeddings():
    embs = {}
    for v in ALL_VARIETIES:
        p = EMB_DIR / f"{v}.npy"
        e = np.load(str(p))
        if e.shape[0] < e.shape[1]:
            e = e.T
        embs[v] = e.astype(np.float64)
    vocab = json.loads((EMB_DIR / "vocab.json").read_text())
    return embs, vocab


def main():
    print("Loading embeddings...")
    embs, vocab = load_embeddings()
    print(f"  {len(embs)} varieties, {len(vocab)} words, dim={list(embs.values())[0].shape[1]}")

    # Compute W matrices
    print("\nComputing W matrices...")
    W_all = compute_all_W(embs)

    # Eigendecompose
    print("Eigendecomposing...")
    decomps = {}
    for v, tm in W_all.items():
        decomps[v] = eigendecompose(tm.W, variety=v)

    # Compute spectral distance matrix
    print("Computing distance matrix...")
    varieties = sorted(embs.keys())
    n = len(varieties)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = spectral_distance(decomps[varieties[i]].eigenvalues,
                                  decomps[varieties[j]].eigenvalues)
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    # === PROBING ===
    print("\n" + "=" * 70)
    print("Running spectral mode probing...")
    probing_result = probe_spectral_modes(embs, vocab, decomps)
    print(format_probing_report(probing_result))

    # === VALIDATION ===
    print("\n")
    validation_result = validate_dialectometry(dist_mat, varieties)
    print(format_validation_report(validation_result))

    # Save results for the explorer
    results_path = EMB_DIR / "analysis_results.json"
    results = {
        "probing": {
            "mode_feature_matrix": probing_result.mode_feature_matrix.tolist(),
            "feature_names": probing_result.feature_names,
            "best_mode_per_feature": {
                f: {"mode": k, "correlation": rho}
                for f, (k, rho) in probing_result.best_mode_per_feature.items()
            },
            "explained_features": {
                str(k): feats for k, feats in probing_result.explained_features.items()
            },
            "feature_correlations": [
                {"mode": c.mode_index, "feature": c.feature,
                 "correlation": c.correlation, "p_value": c.p_value}
                for c in probing_result.feature_correlations
            ],
        },
        "validation": {
            "similarity_correlation": validation_result.similarity_correlation,
            "similarity_p_value": validation_result.similarity_p_value,
            "geographic_correlation": validation_result.geographic_correlation,
            "geographic_p_value": validation_result.geographic_p_value,
            "constraints_satisfied": validation_result.constraints_satisfied,
            "constraints_total": validation_result.constraints_total,
            "constraint_details": [
                {"label": label, "satisfied": ok}
                for label, ok in validation_result.constraint_details
            ],
            "zone_cohesion": validation_result.zone_cohesion,
            "cohesion_ratio": validation_result.cohesion_ratio,
            "distance_matrix": dist_mat.tolist(),
            "varieties": varieties,
            "closest_pairs": [
                {"v1": v1, "v2": v2, "distance": d}
                for v1, v2, d in validation_result.closest_pairs
            ],
            "farthest_pairs": [
                {"v1": v1, "v2": v2, "distance": d}
                for v1, v2, d in validation_result.farthest_pairs
            ],
        },
        "word_probes": {},
    }

    # Add word loadings for top modes
    for v, probes in probing_result.mode_probes.items():
        results["word_probes"][v] = []
        for p in probes[:20]:  # top 20 modes
            results["word_probes"][v].append({
                "mode": p.mode_index,
                "magnitude": p.magnitude,
                "eigenvalue_real": p.eigenvalue.real,
                "eigenvalue_imag": p.eigenvalue.imag,
                "top_positive": [{"word": w, "loading": l} for w, l in p.top_words_positive[:15]],
                "top_negative": [{"word": w, "loading": l} for w, l in p.top_words_negative[:15]],
                "regionalism_overlap": p.regionalism_overlap,
            })

    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
