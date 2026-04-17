"""CLI commands for embedding training and alignment."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import click
import numpy as np

logger = logging.getLogger(__name__)


def _load_corpus_by_dialect(data_dir: Path) -> dict:
    """Load JSONL corpus files and group samples by dialect.

    Returns a dict mapping DialectCode to CorpusSlice.
    """
    from eigendialectos.constants import DialectCode
    from eigendialectos.types import CorpusSlice, DialectSample

    samples_by_dialect: dict[DialectCode, list[DialectSample]] = defaultdict(list)

    # Look for per-dialect files first, then combined corpus.jsonl
    dialect_files = sorted(data_dir.glob("ES_*.jsonl"))
    if not dialect_files:
        corpus_file = data_dir / "corpus.jsonl"
        if corpus_file.exists():
            dialect_files = [corpus_file]

    if not dialect_files:
        raise FileNotFoundError(
            f"No JSONL corpus files found in {data_dir}. "
            "Run 'eigendialectos corpus preprocess' first."
        )

    for jsonl_file in dialect_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                try:
                    dialect_code = DialectCode(record["dialect"])
                except (ValueError, KeyError):
                    continue

                samples_by_dialect[dialect_code].append(
                    DialectSample(
                        text=record.get("text", ""),
                        dialect_code=dialect_code,
                        source_id=record.get("source", "unknown"),
                        confidence=record.get("confidence", 0.5),
                        metadata=record.get("metadata", {}),
                    )
                )

    result = {}
    for dc, samples in samples_by_dialect.items():
        result[dc] = CorpusSlice(samples=samples, dialect_code=dc)

    return result


@click.command("train")
@click.option("--model", default="fasttext", help="Model type: fasttext, word2vec.")
@click.option("--dialect", default="all", help="Dialect code or 'all'.")
@click.option("--data-dir", default="data/processed", help="Input data directory.")
@click.option("--output-dir", default="models", help="Output model directory.")
@click.option("--dim", default=300, type=int, help="Embedding dimensionality.")
@click.option("--epochs", default=10, type=int, help="Training epochs.")
def embed_train(
    model: str,
    dialect: str,
    data_dir: str,
    output_dir: str,
    dim: int,
    epochs: int,
) -> None:
    """Train embedding models per dialect variety."""
    from eigendialectos.constants import DialectCode

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load corpus grouped by dialect
    click.echo(f"Loading corpus from {data_path}...")
    try:
        corpus_slices = _load_corpus_by_dialect(data_path)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        return

    # Determine which dialects to train
    if dialect == "all":
        dialects = sorted(corpus_slices.keys(), key=lambda d: d.value)
    else:
        try:
            target = DialectCode(dialect)
        except ValueError:
            click.echo(f"Error: Unknown dialect code '{dialect}'.", err=True)
            return
        if target not in corpus_slices:
            click.echo(
                f"Error: No corpus data for {target.value}. "
                f"Available: {[d.value for d in corpus_slices]}",
                err=True,
            )
            return
        dialects = [target]

    click.echo(
        f"Training {model} embeddings (dim={dim}, epochs={epochs}) "
        f"for {len(dialects)} dialect(s)..."
    )

    # Instantiate the model class
    for dc in dialects:
        corpus_slice = corpus_slices[dc]
        click.echo(
            f"  Training {model} for {dc.value} "
            f"({len(corpus_slice.samples)} samples)..."
        )

        try:
            if model == "fasttext":
                from eigendialectos.embeddings.subword.fasttext_model import FastTextModel
                emb_model = FastTextModel(
                    dialect_code=dc,
                    vector_size=dim,
                    epochs=epochs,
                )
            elif model == "word2vec":
                from eigendialectos.embeddings.word.word2vec_model import Word2VecModel
                emb_model = Word2VecModel(
                    dialect_code=dc,
                    vector_size=dim,
                    epochs=epochs,
                )
            else:
                click.echo(
                    f"Error: Unknown model type '{model}'. "
                    "Choose from: fasttext, word2vec.",
                    err=True,
                )
                return

            config = {
                "vector_size": dim,
                "epochs": epochs,
            }
            emb_model.train(corpus_slice, config=config)

            # Save model
            model_path = out_path / model / f"{dc.value}"
            model_path.mkdir(parents=True, exist_ok=True)
            save_file = model_path / f"{dc.value}.model"
            emb_model.save(save_file)

            click.echo(
                f"    Saved: {save_file} "
                f"(vocab={emb_model.vocab_size()}, dim={emb_model.embedding_dim()})"
            )

        except ImportError as e:
            click.echo(f"    Error: {e}", err=True)
        except Exception as e:
            click.echo(f"    Error training {dc.value}: {e}", err=True)
            logger.exception("Training failed for %s", dc.value)

    click.echo("Training complete.")


@click.command("align")
@click.option("--method", default="procrustes", help="Alignment method: procrustes, vecmap, muse.")
@click.option("--reference", default="ES_PEN", help="Reference dialect code.")
@click.option("--model-dir", default="models", help="Model directory.")
@click.option("--model-type", default="fasttext", help="Model type: fasttext, word2vec.")
@click.option("--output-dir", default="models/aligned", help="Output directory for alignment matrices.")
def embed_align(
    method: str,
    reference: str,
    model_dir: str,
    model_type: str,
    output_dir: str,
) -> None:
    """Align embedding spaces across varieties."""
    from eigendialectos.constants import DialectCode
    from eigendialectos.embeddings.alignment import CrossVarietyAligner
    from eigendialectos.utils.io import save_numpy

    model_path = Path(model_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Validate reference dialect
    try:
        ref_dialect = DialectCode(reference)
    except ValueError:
        click.echo(f"Error: Unknown reference dialect '{reference}'.", err=True)
        return

    # Discover trained models
    models_base = model_path / model_type
    if not models_base.exists():
        click.echo(
            f"Error: Model directory {models_base} does not exist. "
            "Train models first with 'eigendialectos embed train'.",
            err=True,
        )
        return

    # Load models for each dialect
    click.echo(f"Loading {model_type} models from {models_base}...")
    embeddings: dict[DialectCode, object] = {}
    loaded_models: dict[DialectCode, object] = {}

    for dialect_dir in sorted(models_base.iterdir()):
        if not dialect_dir.is_dir():
            continue
        try:
            dc = DialectCode(dialect_dir.name)
        except ValueError:
            continue

        model_file = dialect_dir / f"{dc.value}.model"
        if not model_file.exists():
            logger.warning("No model file found at %s", model_file)
            continue

        try:
            if model_type == "fasttext":
                from eigendialectos.embeddings.subword.fasttext_model import FastTextModel
                emb_model = FastTextModel()
            elif model_type == "word2vec":
                from eigendialectos.embeddings.word.word2vec_model import Word2VecModel
                emb_model = Word2VecModel()
            else:
                click.echo(f"Error: Unknown model type '{model_type}'.", err=True)
                return

            emb_model.load(model_file)
            loaded_models[dc] = emb_model
            click.echo(f"  Loaded {dc.value} (vocab={emb_model.vocab_size()})")
        except Exception as e:
            click.echo(f"  Failed to load {dc.value}: {e}", err=True)

    if ref_dialect not in loaded_models:
        click.echo(
            f"Error: Reference dialect {ref_dialect.value} model not found. "
            f"Available: {[d.value for d in loaded_models]}",
            err=True,
        )
        return

    if len(loaded_models) < 2:
        click.echo("Error: Need at least 2 models to perform alignment.", err=True)
        return

    # Find shared vocabulary across all models
    click.echo("Computing shared vocabulary...")
    vocab_sets = []
    for dc, m in loaded_models.items():
        # Get vocabulary from the gensim model
        if hasattr(m, '_model') and m._model is not None:
            vocab_sets.append(set(m._model.wv.key_to_index.keys()))
        else:
            vocab_sets.append(set())

    shared_vocab = sorted(set.intersection(*vocab_sets)) if vocab_sets else []
    click.echo(f"  Shared vocabulary: {len(shared_vocab)} words")

    if len(shared_vocab) < 10:
        click.echo(
            "Error: Shared vocabulary too small for meaningful alignment. "
            "Ensure models are trained on overlapping data.",
            err=True,
        )
        return

    # Encode shared vocab into EmbeddingMatrix objects
    for dc, m in loaded_models.items():
        embeddings[dc] = m.encode_words(shared_vocab)

    # Run alignment
    click.echo(f"Aligning {len(embeddings)} varieties to {ref_dialect.value} using {method}...")
    aligner = CrossVarietyAligner(method=method, reference=ref_dialect)

    try:
        aligned = aligner.align_all(embeddings, reference=ref_dialect)
    except Exception as e:
        click.echo(f"Error during alignment: {e}", err=True)
        logger.exception("Alignment failed")
        return

    # Save alignment matrices
    for dc, W in aligner.alignment_matrices.items():
        matrix_path = out_path / f"align_{dc.value}_to_{ref_dialect.value}.npy"
        save_numpy(W, matrix_path)
        click.echo(f"  Saved alignment matrix: {matrix_path.name}")

    # Save aligned embeddings
    for dc, emb_matrix in aligned.items():
        emb_path = out_path / f"aligned_{dc.value}.npy"
        save_numpy(emb_matrix.data, emb_path)

    # Save shared vocabulary
    vocab_path = out_path / "shared_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(shared_vocab, f, ensure_ascii=False, indent=2)
    click.echo(f"  Saved shared vocabulary: {vocab_path.name}")

    click.echo("Alignment complete.")
