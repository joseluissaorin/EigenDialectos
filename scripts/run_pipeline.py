#!/usr/bin/env python3
"""Master orchestration script for the EigenDialectos pipeline.

Chains the full pipeline from corpus loading through spectral analysis
to experiment execution and result export.

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-training --experiments-only
    python scripts/run_pipeline.py --dim 50 --epochs 5 --method procrustes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project bootstrap — ensure src/ is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eigendialectos.constants import DIALECT_NAMES, DialectCode  # noqa: E402
from eigendialectos.types import (  # noqa: E402
    CorpusSlice,
    DialectSample,
    DialectalSpectrum,
    EigenDecomposition,
    EmbeddingMatrix,
    TransformationMatrix,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up root logger and return the pipeline-specific logger."""
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stderr,
    )
    return logging.getLogger("pipeline")


logger = _configure_logging()

# All eight dialect codes in canonical order
ALL_DIALECT_CODES: list[DialectCode] = sorted(DialectCode, key=lambda c: c.value)


# ===================================================================
# Helpers
# ===================================================================

def _elapsed(t0: float) -> str:
    """Format elapsed seconds since *t0* as a human-readable string."""
    secs = time.perf_counter() - t0
    if secs < 60:
        return f"{secs:.1f}s"
    mins = int(secs // 60)
    return f"{mins}m {secs - mins * 60:.1f}s"


def _save_json(data: Any, path: Path) -> None:
    """Atomically write *data* as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=_json_default)
    tmp.rename(path)
    logger.debug("Saved JSON: %s", path)


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy / enum types."""
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
    if hasattr(obj, "__dict__"):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _banner(title: str) -> None:
    """Print a visible stage banner."""
    width = 72
    border = "=" * width
    logger.info(border)
    logger.info("  %s", title.upper())
    logger.info(border)


# ===================================================================
# Step 1: Load corpus
# ===================================================================

def load_corpus(corpus_path: Path) -> dict[DialectCode, CorpusSlice]:
    """Load JSONL corpus and group samples by dialect code.

    Each line of the JSONL file must contain at minimum:
        text, dialect, confidence, source

    Returns a dict mapping DialectCode to CorpusSlice.
    """
    _banner("Step 1: Load corpus")
    t0 = time.perf_counter()

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    samples_by_dialect: dict[DialectCode, list[DialectSample]] = {
        code: [] for code in ALL_DIALECT_CODES
    }
    total = 0
    skipped = 0

    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at line %d: %s", line_num, exc)
                skipped += 1
                continue

            dialect_str = record.get("dialect", "")
            try:
                code = DialectCode(dialect_str)
            except ValueError:
                logger.warning(
                    "Unknown dialect code %r at line %d; skipping",
                    dialect_str,
                    line_num,
                )
                skipped += 1
                continue

            sample = DialectSample(
                text=record.get("text", ""),
                dialect_code=code,
                source_id=record.get("source", "unknown"),
                confidence=float(record.get("confidence", 0.0)),
                metadata=record.get("metadata", {}),
            )
            samples_by_dialect[code].append(sample)
            total += 1

    # Build CorpusSlice objects (only for dialects that have samples)
    corpus: dict[DialectCode, CorpusSlice] = {}
    for code, samples in samples_by_dialect.items():
        if samples:
            corpus[code] = CorpusSlice(samples=samples, dialect_code=code)

    logger.info(
        "Loaded %d samples (%d skipped) across %d dialects in %s",
        total,
        skipped,
        len(corpus),
        _elapsed(t0),
    )
    for code in ALL_DIALECT_CODES:
        count = len(samples_by_dialect[code])
        name = DIALECT_NAMES.get(code, code.value)
        status = "OK" if count > 0 else "MISSING"
        logger.info("  %-10s  %-35s  %6d samples  [%s]", code.value, name, count, status)

    if not corpus:
        raise RuntimeError("No valid samples found in corpus — cannot proceed")

    return corpus


# ===================================================================
# Step 2: Train FastText embeddings per dialect
# ===================================================================

def train_embeddings(
    corpus: dict[DialectCode, CorpusSlice],
    models_dir: Path,
    dim: int = 100,
    epochs: int = 10,
    skip_training: bool = False,
) -> dict[DialectCode, "FastTextModel"]:
    """Train (or load) a FastText model for each dialect in the corpus.

    Returns a dict mapping DialectCode to trained FastTextModel.
    """
    from eigendialectos.embeddings.subword.fasttext_model import FastTextModel

    _banner("Step 2: Train FastText embeddings")
    t0 = time.perf_counter()
    models_dir.mkdir(parents=True, exist_ok=True)

    models: dict[DialectCode, FastTextModel] = {}

    for code in ALL_DIALECT_CODES:
        model_path = models_dir / f"{code.value}.model"
        name = DIALECT_NAMES.get(code, code.value)

        # If skip_training and model already exists, load it
        if skip_training and model_path.exists():
            logger.info("Loading existing model for %s (%s)", code.value, name)
            try:
                model = FastTextModel(dialect_code=code, vector_size=dim)
                model.load(model_path)
                models[code] = model
                logger.info(
                    "  Loaded: vocab_size=%d, dim=%d",
                    model.vocab_size(),
                    model.embedding_dim(),
                )
                continue
            except Exception as exc:
                logger.warning(
                    "Failed to load model for %s: %s — will retrain",
                    code.value,
                    exc,
                )

        # Need corpus data to train
        if code not in corpus:
            logger.warning("No corpus data for %s — skipping", code.value)
            continue

        logger.info("Training FastText for %s (%s)...", code.value, name)
        try:
            model = FastTextModel(
                dialect_code=code,
                vector_size=dim,
                min_count=2,
                window=5,
                epochs=epochs,
                sg=1,
            )
            model.train(corpus[code])
            model.save(model_path)
            models[code] = model
            logger.info(
                "  Trained: vocab_size=%d, dim=%d, saved to %s",
                model.vocab_size(),
                model.embedding_dim(),
                model_path,
            )
        except Exception as exc:
            logger.error("Failed to train model for %s: %s", code.value, exc, exc_info=True)

    logger.info(
        "Embedding training complete: %d/%d models ready in %s",
        len(models),
        len(ALL_DIALECT_CODES),
        _elapsed(t0),
    )

    if len(models) < 2:
        raise RuntimeError(
            f"Need at least 2 trained models for comparison, got {len(models)}"
        )

    return models


# ===================================================================
# Step 3: Build shared vocabulary and EmbeddingMatrix objects
# ===================================================================

def build_embedding_matrices(
    models: dict[DialectCode, Any],
    output_dir: Path,
) -> dict[DialectCode, EmbeddingMatrix]:
    """Find shared vocabulary across all models, encode, transpose, and save.

    The resulting EmbeddingMatrix objects have data shaped (d, V) as required
    by compute_transformation_matrix.
    """
    _banner("Step 3: Build shared vocabulary and embedding matrices")
    t0 = time.perf_counter()

    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-model vocabularies
    vocab_sets: dict[DialectCode, set[str]] = {}
    for code, model in models.items():
        # Access gensim model vocabulary
        words = set(model._model.wv.key_to_index.keys())
        vocab_sets[code] = words
        logger.info("  %s vocabulary: %d words", code.value, len(words))

    # Shared vocabulary = intersection of ALL models
    shared_vocab_set = set.intersection(*vocab_sets.values())
    shared_vocab = sorted(shared_vocab_set)  # deterministic order

    logger.info(
        "Shared vocabulary: %d words (intersection of %d models)",
        len(shared_vocab),
        len(models),
    )

    if len(shared_vocab) == 0:
        raise RuntimeError("Shared vocabulary is empty — models have no words in common")

    # Save shared vocabulary
    vocab_path = emb_dir / "vocab.json"
    _save_json(shared_vocab, vocab_path)
    logger.info("Saved shared vocabulary to %s", vocab_path)

    # Encode and transpose for each dialect
    embeddings: dict[DialectCode, EmbeddingMatrix] = {}
    for code, model in models.items():
        emb = model.encode_words(shared_vocab)

        # CRITICAL: encode_words returns shape (V, d).
        # compute_transformation_matrix expects (d, V).
        # Transpose to get the correct layout.
        emb.data = emb.data.T
        logger.info(
            "  %s embedding matrix: shape %s (d=%d, V=%d)",
            code.value,
            emb.data.shape,
            emb.data.shape[0],
            emb.data.shape[1],
        )

        # Save as .npy
        npy_path = emb_dir / f"{code.value}.npy"
        np.save(npy_path, emb.data)
        embeddings[code] = emb

    logger.info("Embedding matrices built and saved in %s", _elapsed(t0))
    return embeddings


# ===================================================================
# Step 4: Align via Procrustes (optional)
# ===================================================================

def align_embeddings(
    embeddings: dict[DialectCode, EmbeddingMatrix],
    reference: DialectCode = DialectCode.ES_PEN,
) -> dict[DialectCode, EmbeddingMatrix]:
    """Optionally align embeddings using cross-variety Procrustes alignment.

    This step is optional when using transformation matrices directly but
    can improve quality. Returns aligned embeddings.
    """
    _banner("Step 4: Align embeddings (Procrustes)")
    t0 = time.perf_counter()

    try:
        from eigendialectos.embeddings.alignment import CrossVarietyAligner

        aligner = CrossVarietyAligner(method="procrustes", reference=reference)
        aligned = aligner.align_all(embeddings)
        W_matrices = aligner.alignment_matrices

        logger.info(
            "Alignment complete: %d matrices computed, reference=%s in %s",
            len(W_matrices),
            reference.value,
            _elapsed(t0),
        )
        return aligned

    except Exception as exc:
        logger.warning(
            "Alignment step failed (%s); proceeding with unaligned embeddings",
            exc,
        )
        return embeddings


# ===================================================================
# Step 5: Compute transformation matrices W_i
# ===================================================================

def compute_transforms(
    embeddings: dict[DialectCode, EmbeddingMatrix],
    reference: DialectCode,
    method: str = "lstsq",
    regularization: float = 0.01,
    output_dir: Path | None = None,
) -> dict[DialectCode, TransformationMatrix]:
    """Compute transformation matrices from reference dialect to every other.

    Saves checkpoint data to output_dir if provided.
    """
    _banner("Step 5: Compute transformation matrices")
    t0 = time.perf_counter()

    from eigendialectos.spectral.transformation import compute_all_transforms

    if reference not in embeddings:
        # Fallback to the first available dialect
        available = sorted(embeddings.keys(), key=lambda c: c.value)
        reference = available[0]
        logger.warning("Reference %s not in embeddings; using %s", DialectCode.ES_PEN.value, reference.value)

    transforms = compute_all_transforms(
        embeddings=embeddings,
        reference=reference,
        method=method,
        regularization=regularization,
    )

    for code, W in transforms.items():
        logger.info(
            "  W_%s: shape %s, ||W||_F = %.4f",
            code.value,
            W.shape,
            float(np.linalg.norm(W.data, "fro")),
        )

    # Save checkpoint
    if output_dir is not None:
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for code, W in transforms.items():
            np.save(ckpt_dir / f"W_{code.value}.npy", W.data)
        _save_json(
            {
                "method": method,
                "regularization": regularization,
                "reference": reference.value,
                "dialects": [c.value for c in transforms],
            },
            ckpt_dir / "transforms_meta.json",
        )
        logger.info("Saved transform checkpoints to %s", ckpt_dir)

    logger.info(
        "Computed %d transformation matrices in %s",
        len(transforms),
        _elapsed(t0),
    )
    return transforms


# ===================================================================
# Step 6: Eigendecompose
# ===================================================================

def eigendecompose_all(
    transforms: dict[DialectCode, TransformationMatrix],
    output_dir: Path | None = None,
) -> dict[DialectCode, EigenDecomposition]:
    """Eigendecompose each transformation matrix.

    Returns a dict mapping DialectCode to EigenDecomposition.
    """
    _banner("Step 6: Eigendecomposition")
    t0 = time.perf_counter()

    from eigendialectos.spectral.eigendecomposition import eigendecompose

    decompositions: dict[DialectCode, EigenDecomposition] = {}

    for code, W in transforms.items():
        try:
            eigen = eigendecompose(W)
            decompositions[code] = eigen
            logger.info(
                "  %s: rank=%d, |lambda_max|=%.4f, |lambda_min|=%.6f",
                code.value,
                eigen.rank,
                float(np.max(np.abs(eigen.eigenvalues))),
                float(np.min(np.abs(eigen.eigenvalues))),
            )
        except Exception as exc:
            logger.error(
                "Eigendecomposition failed for %s: %s",
                code.value,
                exc,
                exc_info=True,
            )

    # Save checkpoint
    if output_dir is not None:
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for code, eigen in decompositions.items():
            np.save(ckpt_dir / f"eigenvalues_{code.value}.npy", eigen.eigenvalues)
            np.save(ckpt_dir / f"eigenvectors_{code.value}.npy", eigen.eigenvectors)
        logger.info("Saved eigendecomposition checkpoints to %s", ckpt_dir)

    logger.info(
        "Eigendecomposed %d/%d transforms in %s",
        len(decompositions),
        len(transforms),
        _elapsed(t0),
    )
    return decompositions


# ===================================================================
# Step 7: Compute spectra and entropy
# ===================================================================

def compute_spectra(
    decompositions: dict[DialectCode, EigenDecomposition],
    output_dir: Path | None = None,
) -> dict[DialectCode, DialectalSpectrum]:
    """Compute eigenspectrum and entropy for each dialect.

    Returns a dict mapping DialectCode to DialectalSpectrum.
    """
    _banner("Step 7: Compute spectra and entropy")
    t0 = time.perf_counter()

    from eigendialectos.spectral.eigenspectrum import compute_eigenspectrum

    spectra: dict[DialectCode, DialectalSpectrum] = {}

    for code, eigen in decompositions.items():
        try:
            spectrum = compute_eigenspectrum(eigen)
            spectra[code] = spectrum
            logger.info(
                "  %s: entropy=%.4f, top-5 eigenvalues=%s",
                code.value,
                spectrum.entropy,
                np.array2string(spectrum.eigenvalues_sorted[:5], precision=4, separator=", "),
            )
        except Exception as exc:
            logger.error(
                "Spectrum computation failed for %s: %s",
                code.value,
                exc,
                exc_info=True,
            )

    # Save checkpoint
    if output_dir is not None:
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        spectra_data = {}
        for code, spec in spectra.items():
            spectra_data[code.value] = {
                "entropy": float(spec.entropy),
                "eigenvalues_sorted": spec.eigenvalues_sorted.tolist(),
            }
        _save_json(spectra_data, ckpt_dir / "spectra.json")
        logger.info("Saved spectra checkpoints to %s", ckpt_dir)

    logger.info(
        "Computed %d spectra in %s",
        len(spectra),
        _elapsed(t0),
    )
    return spectra


# ===================================================================
# Step 8: Compute pairwise distance matrix
# ===================================================================

def compute_distances(
    transforms: dict[DialectCode, TransformationMatrix],
    spectra: dict[DialectCode, DialectalSpectrum],
    output_dir: Path | None = None,
) -> np.ndarray:
    """Compute the full pairwise distance matrix using spectral distances.

    Returns an (n, n) symmetric distance matrix.
    """
    _banner("Step 8: Compute pairwise distance matrix")
    t0 = time.perf_counter()

    from eigendialectos.spectral.distance import compute_distance_matrix

    # Build entropies dict
    entropies: dict[DialectCode, float] = {
        code: spec.entropy for code, spec in spectra.items()
    }

    # Use only dialects present in both transforms and spectra
    common_codes = set(transforms.keys()) & set(spectra.keys())
    filtered_transforms = {c: transforms[c] for c in common_codes}
    filtered_spectra = {c: spectra[c] for c in common_codes}
    filtered_entropies = {c: entropies[c] for c in common_codes}

    D = compute_distance_matrix(
        transforms=filtered_transforms,
        spectra=filtered_spectra,
        entropies=filtered_entropies,
        method="combined",
    )

    # Pretty-print the distance matrix
    sorted_codes = sorted(common_codes, key=lambda c: c.value)
    logger.info("Distance matrix (%d x %d):", len(sorted_codes), len(sorted_codes))
    header = "          " + "  ".join(f"{c.value:>8}" for c in sorted_codes)
    logger.info(header)
    for i, ci in enumerate(sorted_codes):
        row = f"{ci.value:>8}  " + "  ".join(f"{D[i, j]:8.4f}" for j in range(len(sorted_codes)))
        logger.info(row)

    # Save checkpoint
    if output_dir is not None:
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        np.save(ckpt_dir / "distance_matrix.npy", D)
        _save_json(
            {
                "dialect_order": [c.value for c in sorted_codes],
                "matrix": D.tolist(),
            },
            ckpt_dir / "distance_matrix.json",
        )
        logger.info("Saved distance matrix to %s", ckpt_dir)

    logger.info("Distance matrix computed in %s", _elapsed(t0))
    return D


# ===================================================================
# Step 9: Build dialect tensor
# ===================================================================

def build_tensor(
    transforms: dict[DialectCode, TransformationMatrix],
    output_dir: Path | None = None,
) -> Any:
    """Stack all transformation matrices into a 3D dialect tensor.

    Returns a TensorDialectal object.
    """
    _banner("Step 9: Build dialect tensor")
    t0 = time.perf_counter()

    from eigendialectos.tensor.construction import build_dialect_tensor

    tensor = build_dialect_tensor(transforms)
    logger.info(
        "Dialect tensor: shape %s, dialects=%s",
        tensor.shape,
        [c.value for c in tensor.dialect_codes],
    )

    # Save checkpoint
    if output_dir is not None:
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        np.save(ckpt_dir / "dialect_tensor.npy", tensor.data)
        _save_json(
            {
                "shape": list(tensor.shape),
                "dialect_codes": [c.value for c in tensor.dialect_codes],
            },
            ckpt_dir / "dialect_tensor_meta.json",
        )
        logger.info("Saved dialect tensor to %s", ckpt_dir)

    logger.info("Dialect tensor built in %s", _elapsed(t0))
    return tensor


# ===================================================================
# Step 10: Run experiments
# ===================================================================

def run_experiments(
    data_dir: Path,
    output_dir: Path,
    dim: int,
) -> dict[str, Any]:
    """Run all registered experiments using the ExperimentRunner.

    Returns a dict of experiment_id to ExperimentResult.
    """
    _banner("Step 10: Run experiments")
    t0 = time.perf_counter()

    from eigendialectos.experiments.runner import ExperimentRunner

    exp_output = output_dir / "experiments"
    exp_output.mkdir(parents=True, exist_ok=True)

    config = {
        "dim": dim,
        "n_dialects": len(ALL_DIALECT_CODES),
        "seed": 42,
    }

    runner = ExperimentRunner(
        config=config,
        data_dir=data_dir,
        output_dir=exp_output,
    )

    available = runner.list_experiments()
    logger.info("Registered experiments: %s", available)

    results: dict[str, Any] = {}
    for exp_id in available:
        logger.info("Running experiment: %s ...", exp_id)
        try:
            result = runner.run_experiment(exp_id)
            results[exp_id] = result
            logger.info(
                "  %s: %d metrics, %d artifacts",
                exp_id,
                len(result.metrics),
                len(result.artifact_paths),
            )
        except Exception as exc:
            logger.error("Experiment %s failed: %s", exp_id, exc, exc_info=True)
            results[exp_id] = {"error": str(exc)}

    logger.info(
        "Experiments complete: %d/%d succeeded in %s",
        sum(1 for v in results.values() if not isinstance(v, dict) or "error" not in v),
        len(available),
        _elapsed(t0),
    )
    return results


# ===================================================================
# Step 11: Export everything
# ===================================================================

def export_results(
    output_dir: Path,
    corpus_stats: dict[str, Any],
    spectra: dict[DialectCode, DialectalSpectrum],
    distance_matrix: np.ndarray,
    transforms: dict[DialectCode, TransformationMatrix],
    experiment_results: dict[str, Any],
) -> None:
    """Export all pipeline results as JSON and numpy files.

    Attempts to use eigendialectos.export.exporter.export_all if available,
    otherwise falls back to manual serialisation.
    """
    _banner("Step 11: Export results")
    t0 = time.perf_counter()

    export_dir = output_dir / "final"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Try the dedicated exporter first
    try:
        from eigendialectos.export.exporter import export_all

        export_all(
            output_dir=export_dir,
            spectra=spectra,
            distance_matrix=distance_matrix,
            transforms=transforms,
            experiment_results=experiment_results,
        )
        logger.info("Exported via eigendialectos.export.exporter")
    except ImportError:
        logger.info("Export module not yet available; saving results manually")
    except Exception as exc:
        logger.warning("export_all raised %s; falling back to manual export", exc)

    # Always save these regardless of the exporter
    # -- Corpus statistics
    _save_json(corpus_stats, export_dir / "corpus_stats.json")

    # -- Spectral profiles
    spectra_export: dict[str, Any] = {}
    for code, spec in spectra.items():
        spectra_export[code.value] = {
            "entropy": float(spec.entropy),
            "eigenvalues_sorted": spec.eigenvalues_sorted.tolist(),
            "cumulative_energy": spec.cumulative_energy.tolist(),
        }
    _save_json(spectra_export, export_dir / "spectral_profiles.json")

    # -- Distance matrix
    sorted_codes = sorted(spectra.keys(), key=lambda c: c.value)
    np.save(export_dir / "distance_matrix.npy", distance_matrix)
    _save_json(
        {
            "dialect_order": [c.value for c in sorted_codes],
            "matrix": distance_matrix.tolist(),
        },
        export_dir / "distance_matrix.json",
    )

    # -- Transform norms
    transform_summary: dict[str, Any] = {}
    for code, W in transforms.items():
        transform_summary[code.value] = {
            "shape": list(W.shape),
            "frobenius_norm": float(np.linalg.norm(W.data, "fro")),
            "source_dialect": W.source_dialect.value,
            "target_dialect": W.target_dialect.value,
            "regularization": W.regularization,
        }
    _save_json(transform_summary, export_dir / "transforms_summary.json")

    # -- Experiment results summary
    exp_summary: dict[str, Any] = {}
    for exp_id, result in experiment_results.items():
        if isinstance(result, dict) and "error" in result:
            exp_summary[exp_id] = {"status": "failed", "error": result["error"]}
        else:
            exp_summary[exp_id] = {
                "status": "success",
                "metrics": {
                    k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in getattr(result, "metrics", {}).items()
                },
                "artifact_count": len(getattr(result, "artifact_paths", [])),
                "timestamp": getattr(result, "timestamp", ""),
            }
    _save_json(exp_summary, export_dir / "experiment_summary.json")

    # -- Pipeline metadata
    _save_json(
        {
            "pipeline_version": "1.0.0",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "n_dialects": len(spectra),
            "dialects": [c.value for c in sorted_codes],
            "export_files": [
                "corpus_stats.json",
                "spectral_profiles.json",
                "distance_matrix.json",
                "distance_matrix.npy",
                "transforms_summary.json",
                "experiment_summary.json",
            ],
        },
        export_dir / "pipeline_meta.json",
    )

    logger.info("All results exported to %s in %s", export_dir, _elapsed(t0))


# ===================================================================
# Summary
# ===================================================================

def print_summary(
    corpus: dict[DialectCode, CorpusSlice],
    spectra: dict[DialectCode, DialectalSpectrum],
    distance_matrix: np.ndarray,
    experiment_results: dict[str, Any],
    total_time: float,
) -> None:
    """Print a comprehensive summary table of all results."""
    _banner("Pipeline Summary")

    sorted_codes = sorted(spectra.keys(), key=lambda c: c.value)

    # Corpus table
    logger.info("")
    logger.info("%-12s  %-30s  %8s  %8s", "Dialect", "Name", "Samples", "Entropy")
    logger.info("-" * 72)
    for code in sorted_codes:
        name = DIALECT_NAMES.get(code, "?")
        n_samples = len(corpus[code].samples) if code in corpus else 0
        entropy = spectra[code].entropy if code in spectra else float("nan")
        logger.info("%-12s  %-30s  %8d  %8.4f", code.value, name, n_samples, entropy)

    # Distance matrix (condensed)
    logger.info("")
    logger.info("Pairwise distance matrix (combined metric):")
    header = "            " + "  ".join(f"{c.value:>8}" for c in sorted_codes)
    logger.info(header)
    for i, ci in enumerate(sorted_codes):
        row_vals = "  ".join(f"{distance_matrix[i, j]:8.4f}" for j in range(len(sorted_codes)))
        logger.info("%-10s  %s", ci.value, row_vals)

    # Experiment results
    logger.info("")
    logger.info("Experiment results:")
    logger.info("%-30s  %-10s", "Experiment", "Status")
    logger.info("-" * 42)
    for exp_id, result in experiment_results.items():
        if isinstance(result, dict) and "error" in result:
            status = "FAILED"
        else:
            status = "OK"
        logger.info("%-30s  %-10s", exp_id, status)

    # Timing
    logger.info("")
    mins = int(total_time // 60)
    secs = total_time - mins * 60
    if mins > 0:
        logger.info("Total pipeline time: %dm %.1fs", mins, secs)
    else:
        logger.info("Total pipeline time: %.1fs", secs)
    logger.info("Output directory: see --output-dir")
    logger.info("=" * 72)


# ===================================================================
# CLI
# ===================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="EigenDialectos: master orchestration pipeline from corpus to results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --dim 50 --epochs 5
    python scripts/run_pipeline.py --skip-training
    python scripts/run_pipeline.py --experiments-only
    python scripts/run_pipeline.py --method procrustes --corpus data/processed/corpus.jsonl
""",
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/corpus.jsonl"),
        help="Path to JSONL corpus file (default: data/processed/corpus.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory for all outputs (default: outputs/)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=100,
        help="Embedding dimensionality (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="FastText training epochs (default: 10)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="lstsq",
        choices=["lstsq", "procrustes", "nuclear"],
        help="Transformation matrix method (default: lstsq)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip embedding training if models already exist on disk",
    )
    parser.add_argument(
        "--experiments-only",
        action="store_true",
        help="Skip training/spectral, jump directly to experiments with existing data",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.01,
        help="Regularization strength for transformation matrices (default: 0.01)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="ES_PEN",
        help="Reference dialect code (default: ES_PEN)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args(argv)

    # Resolve paths relative to PROJECT_ROOT
    if not args.corpus.is_absolute():
        args.corpus = PROJECT_ROOT / args.corpus
    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir

    # Validate reference dialect
    try:
        args.reference = DialectCode(args.reference)
    except ValueError:
        parser.error(
            f"Unknown reference dialect: {args.reference!r}. "
            f"Valid: {[c.value for c in DialectCode]}"
        )

    return args


# ===================================================================
# Main
# ===================================================================

def main(argv: list[str] | None = None) -> int:
    """Run the full EigenDialectos pipeline.

    Returns 0 on success, 1 on failure.
    """
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    pipeline_start = time.perf_counter()

    logger.info("=" * 72)
    logger.info("  EigenDialectos Pipeline")
    logger.info("  Started: %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    logger.info("=" * 72)
    logger.info("Configuration:")
    logger.info("  corpus      = %s", args.corpus)
    logger.info("  output_dir  = %s", args.output_dir)
    logger.info("  dim         = %d", args.dim)
    logger.info("  epochs      = %d", args.epochs)
    logger.info("  method      = %s", args.method)
    logger.info("  reference   = %s", args.reference.value)
    logger.info("  regulariz.  = %.4f", args.regularization)
    logger.info("  skip_train  = %s", args.skip_training)
    logger.info("  exp_only    = %s", args.experiments_only)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = PROJECT_ROOT / "models" / "fasttext"

    # -------------------------------------------------------------------
    # Experiments-only shortcut
    # -------------------------------------------------------------------
    if args.experiments_only:
        logger.info("--experiments-only: skipping steps 1-9, running experiments directly")
        data_dir = args.output_dir / "final"
        if not data_dir.exists():
            data_dir = args.output_dir / "checkpoints"
        if not data_dir.exists():
            data_dir = PROJECT_ROOT / "data" / "processed" / "embeddings"

        experiment_results = run_experiments(
            data_dir=data_dir,
            output_dir=args.output_dir,
            dim=args.dim,
        )

        total_time = time.perf_counter() - pipeline_start
        logger.info("Experiments-only pipeline completed in %s", _elapsed(pipeline_start))
        return 0

    # -------------------------------------------------------------------
    # Full pipeline
    # -------------------------------------------------------------------
    try:
        # Step 1: Load corpus
        corpus = load_corpus(args.corpus)
        corpus_stats = {
            code.value: corpus[code].stats if code in corpus else {"count": 0}
            for code in ALL_DIALECT_CODES
        }
        _save_json(corpus_stats, args.output_dir / "checkpoints" / "corpus_stats.json")

        # Step 2: Train embeddings
        models = train_embeddings(
            corpus=corpus,
            models_dir=models_dir,
            dim=args.dim,
            epochs=args.epochs,
            skip_training=args.skip_training,
        )

        # Step 3: Build shared vocabulary and embedding matrices
        embeddings = build_embedding_matrices(
            models=models,
            output_dir=args.output_dir,
        )

        # Step 4: Align (optional, improves quality)
        aligned_embeddings = align_embeddings(
            embeddings=embeddings,
            reference=args.reference,
        )

        # Step 5: Compute transformation matrices
        transforms = compute_transforms(
            embeddings=aligned_embeddings,
            reference=args.reference,
            method=args.method,
            regularization=args.regularization,
            output_dir=args.output_dir,
        )

        # Step 6: Eigendecompose
        decompositions = eigendecompose_all(
            transforms=transforms,
            output_dir=args.output_dir,
        )

        # Step 7: Compute spectra and entropy
        spectra = compute_spectra(
            decompositions=decompositions,
            output_dir=args.output_dir,
        )

        # Step 8: Compute pairwise distance matrix
        distance_matrix = compute_distances(
            transforms=transforms,
            spectra=spectra,
            output_dir=args.output_dir,
        )

        # Step 9: Build dialect tensor
        tensor = build_tensor(
            transforms=transforms,
            output_dir=args.output_dir,
        )

        # Step 10: Run experiments
        experiment_results = run_experiments(
            data_dir=args.output_dir / "embeddings",
            output_dir=args.output_dir,
            dim=args.dim,
        )

        # Step 11: Export everything
        export_results(
            output_dir=args.output_dir,
            corpus_stats=corpus_stats,
            spectra=spectra,
            distance_matrix=distance_matrix,
            transforms=transforms,
            experiment_results=experiment_results,
        )

        # Summary
        total_time = time.perf_counter() - pipeline_start
        print_summary(
            corpus=corpus,
            spectra=spectra,
            distance_matrix=distance_matrix,
            experiment_results=experiment_results,
            total_time=total_time,
        )

        logger.info("Pipeline completed successfully.")
        return 0

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("Pipeline error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
