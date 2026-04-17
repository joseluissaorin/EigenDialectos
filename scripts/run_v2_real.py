#!/usr/bin/env python3
"""EigenDialectos v2 — Run full analysis on REAL corpus data.

Trains (or loads cached) DCL subword embeddings via the unified pipeline,
computes W matrices, eigendecompositions, and runs all v2 analyses +
experiments with real data.

Usage
-----
::

    python scripts/run_v2_real.py             # use cached embeddings if available
    python scripts/run_v2_real.py --retrain   # force full retraining
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eigendialectos.constants import (
    DIALECT_COORDINATES,
    DIALECT_FAMILIES,
    DialectCode,
    LinguisticLevel,
)
from eigendialectos.types import (
    EigenDecomposition,
    EmbeddingMatrix,
    LevelEmbedding,
    TransformationMatrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("v2_real")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2_real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR = PROJECT_ROOT / "outputs" / "embeddings"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EigenDialectos v2 real pipeline")
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force full retraining of embeddings",
    )
    parser.add_argument(
        "--method", default="subword_dcl",
        choices=["fasttext_procrustes", "subword_dcl"],
        help="Embedding method (default: subword_dcl)",
    )
    parser.add_argument(
        "--dcl-refinement", action="store_true",
        help="Apply word-level DCL refinement after fastText+Procrustes",
    )
    parser.add_argument(
        "--null-model", action="store_true",
        help="Compute null model noise floor (splits PEN corpus, ~30s extra)",
    )
    parser.add_argument(
        "--null-trials", type=int, default=3,
        help="Number of null model trials (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Train or load DCL subword embeddings
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 1: Embedding pipeline (train_or_load)")

    from eigendialectos.embeddings.pipeline import train_or_load_embeddings

    embeddings, shared_vocab = train_or_load_embeddings(
        corpus_dir=DATA_DIR,
        output_dir=EMB_DIR,
        force_retrain=args.retrain,
        method=args.method,
        dcl_refinement=getattr(args, "dcl_refinement", False),
    )

    logger.info("  Shared vocabulary: %d words", len(shared_vocab))
    for d_name, emb in sorted(embeddings.items()):
        logger.info("  %s: shape %s", d_name, emb.shape)

    dim = next(iter(embeddings.values())).shape[0]
    n_vocab = next(iter(embeddings.values())).shape[1]
    logger.info("  Embedding dim=%d, vocab=%d, dialects=%d", dim, n_vocab, len(embeddings))

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Compute stability weights + W matrices (reference = ES_PEN)
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 2: Computing transformation matrices W_i")

    from eigendialectos.spectral.transformation import compute_transformation_matrix

    ref = "ES_PEN"
    ref_emb = embeddings[ref]

    # ── Stability-based vocabulary pruning (fasttext_procrustes only) ──
    # DCL with vocab filtering already produces clean embeddings.
    # Stability pruning is only needed for fasttext where English noise
    # survives the intersection vocabulary.
    if args.method == "fasttext_procrustes":
        logger.info("  Computing per-word stability scores ...")
        non_ref_varieties = [k for k in sorted(embeddings.keys()) if k != ref]
        residual_norms = []
        for v in non_ref_varieties:
            diff = embeddings[v] - ref_emb
            norms_sq = np.sum(diff ** 2, axis=0)
            residual_norms.append(norms_sq)
        word_variance = np.mean(residual_norms, axis=0)

        variance_threshold = np.percentile(word_variance, 90)
        stable_mask = word_variance <= variance_threshold
        n_removed = int(np.sum(~stable_mask))

        noisy_idx = np.argsort(word_variance)[-15:][::-1]
        stable_idx = np.argsort(word_variance)[:10]
        logger.info(
            "  Stability pruning: removing %d/%d words (variance > %.2f)",
            n_removed, len(shared_vocab), variance_threshold,
        )
        logger.info(
            "  Most unstable (REMOVED): %s",
            ", ".join(
                f"{shared_vocab[i]}({word_variance[i]:.1f})"
                for i in noisy_idx if not stable_mask[i]
            ),
        )
        logger.info(
            "  Most stable (KEPT): %s",
            ", ".join(f"{shared_vocab[i]}({word_variance[i]:.1f})" for i in stable_idx),
        )

        keep_indices = np.where(stable_mask)[0]
        shared_vocab = [shared_vocab[i] for i in keep_indices]
        ref_emb = ref_emb[:, keep_indices]
        embeddings = {
            k: v[:, keep_indices] for k, v in embeddings.items()
        }
        n_vocab = len(shared_vocab)
        logger.info("  After stability pruning: %d words, dim=%d", n_vocab, dim)
    else:
        logger.info("  Skipping stability pruning (DCL embeddings already filtered)")
        n_vocab = len(shared_vocab)

    # ── Compute W matrices ──
    # Higher regularization (0.05) for subword_dcl which has more words
    # and lower-norm embeddings; 0.01 for fasttext_procrustes.
    reg = 0.05 if args.method == "subword_dcl" else 0.01
    W_matrices: dict[str, np.ndarray] = {}

    for dialect, emb in embeddings.items():
        src = EmbeddingMatrix(data=ref_emb, vocab=shared_vocab, dialect_code=DialectCode.ES_PEN)
        tgt = EmbeddingMatrix(
            data=emb, vocab=shared_vocab,
            dialect_code=DialectCode(dialect),
        )
        W_tm = compute_transformation_matrix(src, tgt, method="lstsq", regularization=reg)
        W_matrices[dialect] = W_tm.data
        logger.info("  W_%s: shape %s, cond=%.1f, det=%.4f",
                     dialect, W_tm.data.shape,
                     np.linalg.cond(W_tm.data),
                     np.linalg.det(W_tm.data))

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Eigendecompose all W
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 3: Eigendecomposition")

    from eigendialectos.spectral.eigendecomposition import eigendecompose

    eigendecomps: dict[str, EigenDecomposition] = {}
    for dialect, W in W_matrices.items():
        tm = TransformationMatrix(
            data=W, source_dialect=DialectCode.ES_PEN,
            target_dialect=DialectCode(dialect), regularization=0.01,
        )
        eigendecomps[dialect] = eigendecompose(tm)
        top5 = np.sort(np.abs(eigendecomps[dialect].eigenvalues))[-5:][::-1]
        logger.info("  %s: top 5 |λ| = %s", dialect, np.round(top5, 4))

    # ═══════════════════════════════════════════════════════════════
    # STEP 3.5: Null model + Residual analysis (ΔW = W - I)
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 3.5: Residual analysis (ΔW = W - I)")

    from eigendialectos.spectral.null_model import (
        compute_null_model,
        load_null_model,
        save_null_model,
    )
    from eigendialectos.spectral.residual_analysis import full_residual_analysis

    null_dir = OUTPUT_DIR / "null_model"
    null_p95_sv = None

    # Null model: compute or load
    if args.null_model:
        cached_null = load_null_model(null_dir)
        if cached_null is not None and not args.retrain:
            logger.info("  Loading cached null model from %s", null_dir)
            null_p95_sv = cached_null["p95_sv"]
        else:
            logger.info("  Computing null model (%d trials) ...", args.null_trials)
            # Load PEN corpus for splitting
            import json as _json
            pen_docs = []
            pen_path = DATA_DIR / "ES_PEN.jsonl"
            if pen_path.exists():
                with open(pen_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        text = obj.get("text", "").strip()
                        if text:
                            pen_docs.append(text)

            # Get anchor indices for consistent null model alignment
            from eigendialectos.embeddings.vocab_filter import get_anchor_indices
            try:
                anchor_idx = get_anchor_indices(shared_vocab, min_anchors=50)
            except ValueError:
                anchor_idx = None

            null_result = compute_null_model(
                corpus_reference=pen_docs,
                vocab=shared_vocab,
                anchor_indices=anchor_idx,
                vector_size=dim,
                n_trials=args.null_trials,
                seed=42,
            )
            save_null_model(null_result, null_dir)
            null_p95_sv = null_result["p95_sv"]
    else:
        # Try to load from a previous run
        cached_null = load_null_model(null_dir)
        if cached_null is not None:
            null_p95_sv = cached_null["p95_sv"]
            logger.info("  Using cached null model (p95 top SV=%.4f)", null_p95_sv[0])
        else:
            logger.info("  No null model available (use --null-model to compute)")

    # Full residual analysis: ΔW SVD + per-word PCA
    residual_result = full_residual_analysis(
        W_matrices=W_matrices,
        embeddings=embeddings,
        vocab=shared_vocab,
        reference="ES_PEN",
        null_p95_sv=null_p95_sv,
        n_pca_components=10,
        top_k_words=15,
    )

    # Save residual analysis
    residual_dir = OUTPUT_DIR / "residual_analysis"
    residual_dir.mkdir(parents=True, exist_ok=True)

    delta_w_data = {}
    for variety, dr in residual_result.delta_w_results.items():
        delta_w_data[variety] = {
            "frob_norm": dr.frob_norm,
            "n_significant": dr.n_significant,
            "top_singular_values": dr.singular_values[:10].tolist(),
            "top_word_impacts": [
                [(w, round(s, 4)) for w, s in axis_words[:10]]
                for axis_words in dr.top_word_impacts[:5]
            ],
        }
        np.save(
            str(residual_dir / f"delta_w_sv_{variety}.npy"),
            dr.singular_values,
        )

    with open(residual_dir / "delta_w_analysis.json", "w") as f:
        json.dump(delta_w_data, f, indent=2, ensure_ascii=False)

    np.save(
        str(residual_dir / "pca_components.npy"),
        residual_result.pca_components,
    )
    pca_data = {
        "explained_variance_ratio": residual_result.pca_explained_variance_ratio.tolist(),
        "explained_variance": residual_result.pca_explained_variance.tolist(),
        "word_loadings": [
            [(w, round(s, 4)) for w, s in comp_words[:10]]
            for comp_words in residual_result.pca_word_loadings[:5]
        ],
        "total_significant_axes": residual_result.total_dialectal_dimensions,
        "mean_delta_w_frob": residual_result.mean_frob_norm,
    }
    with open(residual_dir / "pca_analysis.json", "w") as f:
        json.dump(pca_data, f, indent=2, ensure_ascii=False)

    logger.info("  Residual analysis saved to %s", residual_dir)

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Multi-granularity decomposition
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 4: Multi-granularity decomposition")

    from eigendialectos.spectral.multigranularity import MultiGranularityDecomposition
    mg = MultiGranularityDecomposition()
    mg_results = mg.decompose(W_matrices)
    variance_ratios = mg.explained_variance_ratio()

    for d, ratios in sorted(variance_ratios.items()):
        logger.info("  %s: macro=%.1f%% zonal=%.1f%% dialect=%.1f%%",
                     d, ratios['macro']*100, ratios['zonal']*100, ratios['dialect']*100)

    with open(OUTPUT_DIR / "variance_ratios.json", "w") as f:
        json.dump(variance_ratios, f, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Lie algebra analysis
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 5: Lie algebra analysis")

    from eigendialectos.geometry.lie_algebra import LieAlgebraAnalysis
    lie = LieAlgebraAnalysis()
    lie_result = lie.full_analysis(W_matrices)

    bracket_matrix, bracket_labels = lie.bracket_magnitude_matrix(lie_result.generators)
    np.save(OUTPUT_DIR / "bracket_matrix.npy", bracket_matrix)

    logger.info("  Commutator norms (top 5 most non-commutative pairs):")
    sorted_pairs = sorted(lie_result.commutator_norms.items(), key=lambda x: -x[1])
    for (a, b), norm in sorted_pairs[:5]:
        logger.info("    [%s, %s] = %.4f", a, b, norm)

    logger.info("  Bottom 5 (most commutative pairs):")
    for (a, b), norm in sorted_pairs[-5:]:
        logger.info("    [%s, %s] = %.4f", a, b, norm)

    # Save commutator norms
    comm_data = {f"{a}_{b}": norm for (a, b), norm in lie_result.commutator_norms.items()}
    with open(OUTPUT_DIR / "commutator_norms.json", "w") as f:
        json.dump(comm_data, f, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # STEP 6: Riemannian geometry
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 6: Riemannian geometry analysis")

    from eigendialectos.geometry.riemannian import RiemannianDialectSpace
    riem = RiemannianDialectSpace()
    riem_result = riem.full_analysis(eigendecomps)

    np.save(OUTPUT_DIR / "geodesic_distances.npy", riem_result.geodesic_distances)

    logger.info("  Geodesic distance matrix:")
    labels = riem_result.dialect_labels
    for i, li in enumerate(labels):
        row = " ".join(f"{riem_result.geodesic_distances[i,j]:7.2f}" for j in range(len(labels)))
        logger.info("    %s: %s", li[:6], row)

    logger.info("  Ricci curvatures:")
    for name, kappa in sorted(riem_result.ricci_curvatures.items()):
        logger.info("    %s: κ = %.4f", name, kappa)

    geo_data = {
        "labels": labels,
        "distances": riem_result.geodesic_distances.tolist(),
        "curvatures": riem_result.ricci_curvatures,
    }
    with open(OUTPUT_DIR / "riemannian_results.json", "w") as f:
        json.dump(geo_data, f, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # STEP 7: Fisher Information
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 7: Fisher Information analysis")

    from eigendialectos.geometry.fisher import FisherInformationAnalysis
    # Need (n_words, dim) format — transpose from (dim, n_vocab)
    emb_transposed = {d: emb.T for d, emb in embeddings.items()}
    fisher = FisherInformationAnalysis()
    fisher_result = fisher.compute_fim(emb_transposed, vocabulary=shared_vocab)

    logger.info("  Top 10 most diagnostic words:")
    for word, score in fisher_result.most_diagnostic[:10]:
        logger.info("    %s: %.4f", word, score)

    logger.info("  Top 5 FIM eigenvalues: %s", np.round(fisher_result.fim_eigenvalues[:5], 4))

    with open(OUTPUT_DIR / "fisher_diagnostic.json", "w") as f:
        json.dump({
            "most_diagnostic": fisher_result.most_diagnostic,
            "top_eigenvalues": fisher_result.fim_eigenvalues[:20].tolist(),
        }, f, indent=2, ensure_ascii=False)

    # ═══════════════════════════════════════════════════════════════
    # STEP 8: Eigenvalue field (geographic)
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 8: Eigenvalue field estimation")

    from eigendialectos.geometry.eigenfield import EigenvalueField

    dialect_order = sorted(eigendecomps.keys())
    coords = []
    evals_list = []
    coord_labels = []
    for d in dialect_order:
        dc = DialectCode(d)
        if dc in DIALECT_COORDINATES:
            coords.append(list(DIALECT_COORDINATES[dc]))
            evals_list.append(np.abs(eigendecomps[d].eigenvalues[:20].real))
            coord_labels.append(d)

    coords_arr = np.array(coords)
    evals_arr = np.array(evals_list)
    logger.info("  %d geolocated dialects, %d eigenvalues per dialect", len(coords), evals_arr.shape[1])

    ef = EigenvalueField(kernel_lengthscale=15.0)
    ef.fit(coords_arr, evals_arr)
    field_result = ef.compute_field(resolution=40)

    np.save(OUTPUT_DIR / "eigenvalue_field.npy", field_result.eigenvalue_surfaces)
    np.save(OUTPUT_DIR / "eigenvalue_field_uncertainty.npy", field_result.uncertainties)

    logger.info("  Field shape: %s", field_result.eigenvalue_surfaces.shape)

    # ═══════════════════════════════════════════════════════════════
    # STEP 9: Persistent homology (TDA)
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 9: Persistent homology")

    from eigendialectos.topology.persistent_homology import PersistentHomologyAnalysis
    ph = PersistentHomologyAnalysis(max_dimension=2)
    eigenspectra = np.array([
        np.abs(eigendecomps[d].eigenvalues[:20].real) for d in dialect_order
    ])
    ph_result = ph.compute(eigenspectra, dialect_order)
    interp = ph.interpret(ph_result, dialect_order)

    logger.info("  %s", interp['summary'])

    with open(OUTPUT_DIR / "tda_results.json", "w") as f:
        json.dump({
            "betti_numbers": {str(k): v for k, v in ph_result.betti_numbers.items()},
            "persistence_entropy": ph_result.persistence_entropy,
            "interpretation": {k: str(v) for k, v in interp.items()},
        }, f, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # STEP 10: SDC compiler demo on real embeddings
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 10: SDC compiler demo with REAL embeddings")

    from eigendialectos.spectral.stack import SpectralStack
    from eigendialectos.compiler.sdc import SpectralDialectalCompiler

    # Build LevelEmbeddings from real data (word level = L2)
    # embeddings are (dim, vocab) — need (vocab, dim) for LevelEmbedding
    targets = ["ES_CAN", "ES_RIO", "ES_AND", "ES_MEX"]
    demo_texts = [
        "El autobús llega a las tres y media, ¿no?",
        "¿Puedes decirme dónde está la estación de tren?",
        "La chica quiere comprar un ordenador nuevo para trabajar.",
        "Vamos a coger el coche y nos vamos al centro.",
    ]

    src_le = LevelEmbedding(
        level=2,
        vectors=embeddings["ES_PEN"].T,  # (vocab, dim)
        labels=shared_vocab,
        vocabulary={w: i for i, w in enumerate(shared_vocab)},
    )

    sdc_results = []
    for target in targets:
        tgt_le = LevelEmbedding(
            level=2,
            vectors=embeddings[target].T,
            labels=shared_vocab,
            vocabulary={w: i for i, w in enumerate(shared_vocab)},
        )

        stack = SpectralStack(levels=[2])
        stack.fit_from_matrices({2: W_matrices[target]})

        compiler = SpectralDialectalCompiler(
            spectral_stack=stack,
            source_embeddings={2: src_le},
            target_embeddings={2: tgt_le},
            source_variety="ES_PEN",
            target_variety=target,
        )

        logger.info("  --- Target: %s ---", target)
        for text in demo_texts:
            result = compiler.compile(text, alphas={2: 0.9})
            logger.info("    IN:  %s", text)
            logger.info("    OUT: %s (%d changes)", result.output_text, len(result.change_log))
            for change in result.change_log:
                logger.info("      %s (eigvec=%d, λ=%s, conf=%.2f)",
                            change['change'],
                            change.get('eigenvector_idx', -1),
                            str(change.get('eigenvalue', '?'))[:20],
                            change.get('confidence', 0))
            sdc_results.append({
                "source": "ES_PEN",
                "target": target,
                "input": text,
                "output": result.output_text,
                "n_changes": len(result.change_log),
                "changes": result.change_log,
            })

    with open(OUTPUT_DIR / "sdc_demo_results.json", "w") as f:
        json.dump(sdc_results, f, indent=2, ensure_ascii=False, default=str)

    # ═══════════════════════════════════════════════════════════════
    # STEP 11: Run experiments A-G with REAL W matrices
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 11: Running experiments A-G with real data")

    # Save real W matrices + eigenvalue data to a staging directory
    # so experiments find them via their disk-loading logic.
    EXP_DATA_DIR = OUTPUT_DIR / "exp_data"
    EXP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dialect, W in W_matrices.items():
        np.save(EXP_DATA_DIR / f"W_{dialect}.npy", W)

    # Save eigenvalues + coordinates for Exp B (phase transitions)
    exp_eigenvalues = np.array([
        np.abs(eigendecomps[d].eigenvalues[:20].real) for d in dialect_order
    ])
    np.save(EXP_DATA_DIR / "eigenvalues.npy", exp_eigenvalues)

    exp_coords = []
    for d in dialect_order:
        dc = DialectCode(d)
        if dc in DIALECT_COORDINATES:
            exp_coords.append(list(DIALECT_COORDINATES[dc]))
        else:
            exp_coords.append([0.0, 0.0])
    np.save(EXP_DATA_DIR / "coordinates.npy", np.array(exp_coords))

    logger.info("  Saved %d W matrices + eigenvalues + coordinates to %s",
                len(W_matrices), EXP_DATA_DIR)

    from eigendialectos.experiments.runner import ExperimentRunner

    exp_config = {
        "seed": 42,
        "dim": dim,
        "n_dialects": len(embeddings),
        "data_dir": str(EXP_DATA_DIR),
        "output_dir": str(OUTPUT_DIR / "experiments"),
    }

    runner = ExperimentRunner(
        config=exp_config,
        data_dir=EXP_DATA_DIR,
        output_dir=OUTPUT_DIR / "experiments",
    )

    exp_ids = [
        "exp_a_dialectal_genome",
        "exp_b_phase_transitions",
        "exp_c_eigenvalue_archaeology",
        "exp_d_synthetic_dialect",
        "exp_e_code_switching",
        "exp_f_eigenvalue_microscope",
        "exp_g_cross_linguistic",
    ]

    for exp_id in exp_ids:
        try:
            result = runner.run_experiment(exp_id)
            logger.info("  %s: OK (%d metrics, %d artifacts)",
                        exp_id, len(result.metrics), len(result.artifact_paths))
        except Exception as e:
            logger.error("  %s: FAILED — %s", exp_id, e, exc_info=True)

    # ═══════════════════════════════════════════════════════════════
    # STEP 12: Save master results summary
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 12: Saving master results")

    # Save W matrices
    np.savez(OUTPUT_DIR / "W_matrices.npz", **W_matrices)

    # Save eigenvalues for all dialects
    eigenvalue_data = {}
    for d, eigen in eigendecomps.items():
        eigenvalue_data[d] = {
            "eigenvalues_real": eigen.eigenvalues.real.tolist(),
            "eigenvalues_imag": eigen.eigenvalues.imag.tolist(),
            "eigenvalues_abs": np.abs(eigen.eigenvalues).tolist(),
        }
    with open(OUTPUT_DIR / "eigenvalues_all.json", "w") as f:
        json.dump(eigenvalue_data, f, indent=2)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 70)
    logger.info("V2 REAL PIPELINE COMPLETE in %.1f seconds", elapsed)
    logger.info("All results saved to: %s", OUTPUT_DIR)
    logger.info("Files produced:")
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            logger.info("  %s (%s)", p.relative_to(OUTPUT_DIR),
                        f"{size/1024:.1f} KB" if size > 1024 else f"{size} B")


if __name__ == "__main__":
    main()
