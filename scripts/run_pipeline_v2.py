#!/usr/bin/env python3
"""EigenDialectos v2 Pipeline — Full algebraic compiler pipeline.

Steps:
 1. Load corpus + config
 2. Multi-level parsing
 3. DCL embedding training (or load existing / fallback to FastText)
 4. Per-level alignment and W_i computation
 5. Spectral stack construction
 6. Multi-granularity decomposition
 7. Lie algebra analysis
 8. Riemannian geometry analysis
 9. Fisher Information analysis
10. Eigenvalue field estimation
11. Persistent homology (TDA)
12. SDC compiler construction + demo
13. Run experiments A-G
14. Export all results

Usage:
    python scripts/run_pipeline_v2.py [--config config/default.yaml]
                                      [--skip-dcl]
                                      [--experiments A B C D E F G]
                                      [--output-dir outputs/v2]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
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
logger = logging.getLogger("pipeline_v2")


def load_corpus(data_dir: Path) -> dict[str, list[str]]:
    """Load corpus.jsonl and group by dialect."""
    corpus_path = data_dir / "corpus.jsonl"
    if not corpus_path.exists():
        logger.warning("corpus.jsonl not found at %s, using synthetic data", corpus_path)
        return _synthetic_corpus()

    corpus: dict[str, list[str]] = {}
    with open(corpus_path) as f:
        for line in f:
            sample = json.loads(line)
            dialect = sample.get("dialect", sample.get("dialect_code", "unknown"))
            text = sample.get("text", "")
            if text.strip():
                corpus.setdefault(dialect, []).append(text)

    logger.info("Loaded corpus: %s", {k: len(v) for k, v in corpus.items()})
    return corpus


def _synthetic_corpus() -> dict[str, list[str]]:
    """Generate a small synthetic corpus for pipeline testing."""
    dialects = [d.value for d in DialectCode]
    sample_texts = {
        "ES_PEN": ["El autobús llega a las tres y media.", "¿Puedes decirme dónde está la estación?"],
        "ES_AND": ["Er bú yega a lah treh y media.", "¿Puedeh desirme ónde ehtá la ehtasión?"],
        "ES_CAN": ["La guagua llega a las tres y media.", "¿Puedes decirme dónde está la estación?"],
        "ES_RIO": ["El bondi llega a las tres y media.", "¿Podés decirme dónde queda la estación?"],
        "ES_MEX": ["El camión llega a las tres y media.", "¿Puedes decirme dónde queda la estación?"],
        "ES_CAR": ["La guagua llega a las tres y media.", "¿Me puedes decir dónde está la estación?"],
        "ES_CHI": ["La micro llega a las tres y media.", "¿Puedes decirme dónde queda la estación?"],
        "ES_AND_BO": ["El bus llega a las tres y media.", "¿Puedes indicarme dónde está la estación?"],
    }
    corpus = {}
    for d in dialects:
        texts = sample_texts.get(d, ["Texto de ejemplo para esta variedad."])
        # Repeat to have some data
        corpus[d] = texts * 50
    return corpus


def load_existing_embeddings(data_dir: Path) -> dict[str, np.ndarray] | None:
    """Try to load pre-existing FastText or DCL embeddings."""
    models_dir = data_dir / "models"
    if not models_dir.exists():
        return None

    embeddings = {}
    for d in DialectCode:
        path = models_dir / f"{d.value}.npy"
        if path.exists():
            embeddings[d.value] = np.load(path)

    if embeddings:
        logger.info("Loaded %d pre-existing embedding matrices", len(embeddings))
        return embeddings
    return None


def train_fasttext_embeddings(
    corpus: dict[str, list[str]], dim: int = 100, epochs: int = 10
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Train FastText embeddings per dialect, return aligned matrices + shared vocab."""
    try:
        from gensim.models import FastText
    except ImportError:
        logger.warning("gensim not available, using random embeddings")
        return _random_embeddings(corpus, dim)

    models = {}
    for dialect, texts in corpus.items():
        sentences = [t.lower().split() for t in texts]
        model = FastText(
            sentences=sentences,
            vector_size=dim,
            window=5,
            min_count=1,
            epochs=epochs,
            workers=1,
            seed=42,
        )
        models[dialect] = model
        logger.info("Trained FastText for %s: %d words", dialect, len(model.wv))

    # Find shared vocabulary
    vocabs = [set(m.wv.key_to_index.keys()) for m in models.values()]
    shared_vocab = sorted(set.intersection(*vocabs))
    logger.info("Shared vocabulary: %d words", len(shared_vocab))

    # Build aligned embedding matrices (n_shared, dim)
    embeddings = {}
    for dialect, model in models.items():
        matrix = np.array([model.wv[w] for w in shared_vocab])
        embeddings[dialect] = matrix

    return embeddings, shared_vocab


def _random_embeddings(
    corpus: dict[str, list[str]], dim: int
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Fallback: random embeddings when gensim unavailable."""
    rng = np.random.default_rng(42)
    all_words: set[str] = set()
    for texts in corpus.values():
        for t in texts:
            all_words.update(t.lower().split())
    shared_vocab = sorted(all_words)[:500]  # Cap at 500

    embeddings = {}
    for dialect in corpus:
        embeddings[dialect] = rng.standard_normal((len(shared_vocab), dim)).astype(np.float64)

    return embeddings, shared_vocab


def compute_W_matrices(
    embeddings: dict[str, np.ndarray],
    reference: str = "ES_PEN",
    method: str = "lstsq",
    reg: float = 0.01,
) -> dict[str, np.ndarray]:
    """Compute transformation matrices W_i from reference to each dialect."""
    from eigendialectos.spectral.transformation import compute_transformation_matrix

    ref_emb = embeddings[reference]
    W_matrices = {}

    for dialect, emb in embeddings.items():
        src = EmbeddingMatrix(
            data=ref_emb.T,  # (dim, vocab)
            vocab=[],
            dialect_code=DialectCode.ES_PEN,
        )
        tgt = EmbeddingMatrix(
            data=emb.T,
            vocab=[],
            dialect_code=DialectCode(dialect) if dialect in DialectCode.__members__ else DialectCode.ES_PEN,
        )
        W_tm = compute_transformation_matrix(src, tgt, method=method, regularization=reg)
        W_matrices[dialect] = W_tm.data
        logger.info("W_%s: shape %s, cond=%.2f", dialect, W_tm.data.shape, np.linalg.cond(W_tm.data))

    return W_matrices


def eigendecompose_all(
    W_matrices: dict[str, np.ndarray],
) -> dict[str, EigenDecomposition]:
    """Eigendecompose all W matrices."""
    from eigendialectos.spectral.eigendecomposition import eigendecompose

    results = {}
    for dialect, W in W_matrices.items():
        tm = TransformationMatrix(
            data=W,
            source_dialect=DialectCode.ES_PEN,
            target_dialect=DialectCode(dialect) if dialect in DialectCode.__members__ else DialectCode.ES_PEN,
            regularization=0.0,
        )
        results[dialect] = eigendecompose(tm)
    return results


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full v2 pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"

    t0 = time.perf_counter()

    # ── Step 1: Load corpus ──
    logger.info("=" * 60)
    logger.info("STEP 1: Loading corpus")
    corpus = load_corpus(data_dir)

    # ── Step 2: Multi-level parsing ──
    logger.info("=" * 60)
    logger.info("STEP 2: Multi-level parsing")
    from eigendialectos.corpus.parsing.multi_level import MultiLevelParser
    parser = MultiLevelParser()
    parsed_samples: dict[str, list] = {}
    for dialect, texts in corpus.items():
        parsed_samples[dialect] = [parser.parse(t) for t in texts[:100]]  # Cap for speed
    logger.info("Parsed %d dialects", len(parsed_samples))

    # ── Step 3: Embeddings ──
    logger.info("=" * 60)
    logger.info("STEP 3: Training/loading embeddings")
    dim = args.dim
    embeddings, shared_vocab = train_fasttext_embeddings(corpus, dim=dim, epochs=args.epochs)

    # ── Step 4: W matrices ──
    logger.info("=" * 60)
    logger.info("STEP 4: Computing transformation matrices")
    W_matrices = compute_W_matrices(embeddings, reference="ES_PEN")

    # ── Step 5: Eigendecomposition ──
    logger.info("=" * 60)
    logger.info("STEP 5: Eigendecomposition")
    eigendecomps = eigendecompose_all(W_matrices)
    for d, eigen in eigendecomps.items():
        top3 = np.sort(np.abs(eigen.eigenvalues))[-3:][::-1]
        logger.info("  %s: top 3 |λ| = %s", d, top3)

    # ── Step 6: Multi-granularity decomposition ──
    logger.info("=" * 60)
    logger.info("STEP 6: Multi-granularity decomposition")
    from eigendialectos.spectral.multigranularity import MultiGranularityDecomposition
    mg = MultiGranularityDecomposition()
    mg_results = mg.decompose(W_matrices)
    variance_ratios = mg.explained_variance_ratio()
    for d, ratios in sorted(variance_ratios.items()):
        logger.info("  %s: macro=%.1f%% zonal=%.1f%% dialect=%.1f%%",
                     d, ratios['macro']*100, ratios['zonal']*100, ratios['dialect']*100)

    # ── Step 7: Lie algebra analysis ──
    logger.info("=" * 60)
    logger.info("STEP 7: Lie algebra analysis")
    from eigendialectos.geometry.lie_algebra import LieAlgebraAnalysis
    lie = LieAlgebraAnalysis()
    lie_result = lie.full_analysis(W_matrices)
    bracket_matrix, bracket_labels = lie.bracket_magnitude_matrix(lie_result.generators)
    logger.info("  Max commutator norm: %.4f", max(lie_result.commutator_norms.values()))
    logger.info("  Min commutator norm: %.4f", min(lie_result.commutator_norms.values()))

    # ── Step 8: Riemannian geometry ──
    logger.info("=" * 60)
    logger.info("STEP 8: Riemannian geometry analysis")
    from eigendialectos.geometry.riemannian import RiemannianDialectSpace
    riem = RiemannianDialectSpace()
    riem_result = riem.full_analysis(eigendecomps)
    logger.info("  Geodesic distance range: [%.4f, %.4f]",
                float(riem_result.geodesic_distances[riem_result.geodesic_distances > 0].min()),
                float(riem_result.geodesic_distances.max()))

    # ── Step 9: Fisher Information ──
    logger.info("=" * 60)
    logger.info("STEP 9: Fisher Information analysis")
    from eigendialectos.geometry.fisher import FisherInformationAnalysis
    fisher = FisherInformationAnalysis()
    fisher_result = fisher.compute_fim(embeddings, vocabulary=shared_vocab)
    logger.info("  Top 5 diagnostic words: %s",
                [w for w, _ in fisher_result.most_diagnostic[:5]])

    # ── Step 10: Eigenvalue field ──
    logger.info("=" * 60)
    logger.info("STEP 10: Eigenvalue field estimation")
    from eigendialectos.geometry.eigenfield import EigenvalueField
    # Collect coordinates and eigenvalues
    dialect_order = sorted(eigendecomps.keys())
    coords = []
    evals_matrix = []
    for d in dialect_order:
        dc = DialectCode(d) if d in DialectCode.__members__ else None
        if dc and dc in DIALECT_COORDINATES:
            coords.append(list(DIALECT_COORDINATES[dc]))
            evals_matrix.append(np.abs(eigendecomps[d].eigenvalues[:10].real))

    if len(coords) >= 3:
        coords_arr = np.array(coords)
        evals_arr = np.array(evals_matrix)
        ef = EigenvalueField()
        ef.fit(coords_arr, evals_arr)
        field_result = ef.compute_field(resolution=30)
        logger.info("  Eigenvalue field: %d surfaces, grid %s",
                     field_result.eigenvalue_surfaces.shape[0],
                     field_result.eigenvalue_surfaces.shape[1:])
    else:
        logger.warning("  Not enough geolocated dialects for eigenvalue field")
        field_result = None

    # ── Step 11: Persistent homology ──
    logger.info("=" * 60)
    logger.info("STEP 11: Persistent homology")
    from eigendialectos.topology.persistent_homology import PersistentHomologyAnalysis
    ph = PersistentHomologyAnalysis(max_dimension=2)
    eigenspectra = np.array([
        np.abs(eigendecomps[d].eigenvalues[:20].real) for d in dialect_order
    ])
    ph_result = ph.compute(eigenspectra, dialect_order)
    interp = ph.interpret(ph_result, dialect_order)
    logger.info("  %s", interp['summary'].replace('\n', ' | '))

    # ── Step 12: SDC compiler demo ──
    logger.info("=" * 60)
    logger.info("STEP 12: SDC compiler demo")
    # Build level-2 embeddings for compiler
    if shared_vocab and embeddings:
        ref_dialect = "ES_PEN"
        target_dialect = "ES_CAN"
        if ref_dialect in embeddings and target_dialect in embeddings:
            src_le = LevelEmbedding(
                level=2,
                vectors=embeddings[ref_dialect],
                labels=shared_vocab,
                vocabulary={w: i for i, w in enumerate(shared_vocab)},
            )
            tgt_le = LevelEmbedding(
                level=2,
                vectors=embeddings[target_dialect],
                labels=shared_vocab,
                vocabulary={w: i for i, w in enumerate(shared_vocab)},
            )
            from eigendialectos.spectral.stack import SpectralStack
            stack = SpectralStack(levels=[2])
            stack.fit_from_matrices({2: W_matrices[target_dialect]})

            from eigendialectos.compiler.sdc import SpectralDialectalCompiler
            compiler = SpectralDialectalCompiler(
                spectral_stack=stack,
                source_embeddings={2: src_le},
                target_embeddings={2: tgt_le},
                source_variety=ref_dialect,
                target_variety=target_dialect,
            )

            demo_texts = [
                "El autobús llega a las tres y media.",
                "¿Puedes decirme dónde está la estación?",
                "La chica camina por la calle principal.",
            ]
            for text in demo_texts:
                result = compiler.compile(text, alphas={2: 0.9})
                logger.info("  '%s' → '%s' (%d changes)",
                            text, result.output_text, len(result.change_log))
                for change in result.change_log:
                    logger.info("    %s (L%d, λ=%s, conf=%.2f)",
                                change['change'], change['level'],
                                change.get('eigenvalue', '?'),
                                change.get('confidence', 0))

    # ── Step 13: Run experiments ──
    if args.experiments:
        logger.info("=" * 60)
        logger.info("STEP 13: Running experiments %s", args.experiments)
        from eigendialectos.experiments.runner import ExperimentRunner

        exp_config = {
            "seed": 42,
            "dim": dim,
            "n_dialects": len(corpus),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir / "experiments"),
            "W_matrices": {k: v.tolist() for k, v in W_matrices.items()},
        }

        runner = ExperimentRunner(
            config=exp_config,
            data_dir=data_dir,
            output_dir=output_dir / "experiments",
        )

        exp_id_map = {
            "A": "exp_a_dialectal_genome",
            "B": "exp_b_phase_transitions",
            "C": "exp_c_eigenvalue_archaeology",
            "D": "exp_d_synthetic_dialect",
            "E": "exp_e_code_switching",
            "F": "exp_f_eigenvalue_microscope",
            "G": "exp_g_cross_linguistic",
        }

        for exp_letter in args.experiments:
            exp_id = exp_id_map.get(exp_letter.upper())
            if exp_id:
                try:
                    result = runner.run_experiment(exp_id)
                    logger.info("  Experiment %s completed: %d metrics",
                                exp_letter, len(result.metrics))
                except Exception as e:
                    logger.error("  Experiment %s failed: %s", exp_letter, e)

    # ── Step 14: Export results ──
    logger.info("=" * 60)
    logger.info("STEP 14: Exporting results")

    # Save key results
    np.save(output_dir / "W_matrices.npy",
            {k: v for k, v in W_matrices.items()}, allow_pickle=True)

    # Save Lie algebra bracket matrix
    np.save(output_dir / "bracket_matrix.npy", bracket_matrix)

    # Save geodesic distances
    np.save(output_dir / "geodesic_distances.npy", riem_result.geodesic_distances)

    # Save Fisher diagnostic words
    with open(output_dir / "fisher_diagnostic.json", "w") as f:
        json.dump(fisher_result.most_diagnostic, f, indent=2, ensure_ascii=False)

    # Save TDA results
    with open(output_dir / "tda_interpretation.json", "w") as f:
        json.dump({k: str(v) for k, v in interp.items()}, f, indent=2)

    # Save variance ratios
    with open(output_dir / "variance_ratios.json", "w") as f:
        json.dump(variance_ratios, f, indent=2)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("V2 PIPELINE COMPLETE in %.1f seconds", elapsed)
    logger.info("Output directory: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="EigenDialectos v2 Pipeline")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data directory (default: project/data)")
    parser.add_argument("--output-dir", type=str, default="outputs/v2",
                        help="Output directory")
    parser.add_argument("--dim", type=int, default=100,
                        help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=10,
                        help="FastText training epochs")
    parser.add_argument("--skip-dcl", action="store_true",
                        help="Skip DCL training, use FastText")
    parser.add_argument("--experiments", nargs="*",
                        default=["A", "B", "C", "D", "E", "F", "G"],
                        help="Which experiments to run (A-G)")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
