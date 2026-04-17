"""CLI commands for DIAL generative model."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np

logger = logging.getLogger(__name__)


def _load_generation_resources(models_dir: str) -> tuple:
    """Load eigendecompositions, transforms, embeddings, and vocab needed for generation.

    Returns (transforms, eigendecomps, vocab, embeddings) or raises an error.
    """
    from eigendialectos.constants import DialectCode
    from eigendialectos.types import (
        EigenDecomposition,
        EmbeddingMatrix,
        TransformationMatrix,
    )

    base = Path(models_dir)

    # Look for spectral data
    spectral_dir = base / "spectral" if (base / "spectral").exists() else base
    eigendecomps_dir = spectral_dir / "eigendecompositions"
    transforms_dir = spectral_dir / "transforms"

    # Also check data/spectral as a fallback
    if not eigendecomps_dir.exists():
        alt_path = Path("data/spectral/eigendecompositions")
        if alt_path.exists():
            eigendecomps_dir = alt_path
            transforms_dir = Path("data/spectral/transforms")

    if not eigendecomps_dir.exists():
        raise FileNotFoundError(
            f"Eigendecompositions not found in {eigendecomps_dir}. "
            "Run 'eigendialectos spectral compute' first."
        )

    # Load metadata
    metadata_file = spectral_dir / "metadata.json"
    if not metadata_file.exists():
        metadata_file = eigendecomps_dir.parent / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    ref_str = metadata.get("reference", "ES_PEN")
    try:
        ref_dialect = DialectCode(ref_str)
    except ValueError:
        ref_dialect = DialectCode.ES_PEN

    # Load eigendecompositions
    eigendecomps: dict[DialectCode, EigenDecomposition] = {}
    for ev_file in sorted(eigendecomps_dir.glob("eigenvalues_*.npy")):
        dialect_str = ev_file.stem.replace("eigenvalues_", "")
        try:
            dc = DialectCode(dialect_str)
        except ValueError:
            continue

        eigvec_file = eigendecomps_dir / f"eigenvectors_{dialect_str}.npy"
        eigvec_inv_file = eigendecomps_dir / f"eigenvectors_inv_{dialect_str}.npy"
        if not eigvec_file.exists() or not eigvec_inv_file.exists():
            continue

        eigendecomps[dc] = EigenDecomposition(
            eigenvalues=np.load(ev_file, allow_pickle=False),
            eigenvectors=np.load(eigvec_file, allow_pickle=False),
            eigenvectors_inv=np.load(eigvec_inv_file, allow_pickle=False),
            dialect_code=dc,
        )

    if not eigendecomps:
        raise FileNotFoundError(
            "No eigendecompositions found. Run 'eigendialectos spectral compute' first."
        )

    # Load transforms
    transforms: dict[DialectCode, TransformationMatrix] = {}
    if transforms_dir.exists():
        for w_file in sorted(transforms_dir.glob("W_*.npy")):
            dialect_str = w_file.stem.replace("W_", "")
            try:
                dc = DialectCode(dialect_str)
            except ValueError:
                continue
            transforms[dc] = TransformationMatrix(
                data=np.load(w_file, allow_pickle=False),
                source_dialect=ref_dialect,
                target_dialect=dc,
                regularization=metadata.get("regularization", 0.01),
            )

    # Load vocabulary and embeddings
    aligned_dir = base / "aligned" if (base / "aligned").exists() else base
    vocab_file = aligned_dir / "shared_vocab.json"
    if not vocab_file.exists():
        vocab_file = Path("models/aligned/shared_vocab.json")
    if not vocab_file.exists():
        raise FileNotFoundError(
            "shared_vocab.json not found. Run 'eigendialectos embed align' first."
        )

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Load reference (neutral) embeddings
    ref_emb_file = aligned_dir / f"aligned_{ref_dialect.value}.npy"
    if not ref_emb_file.exists():
        ref_emb_file = Path(f"models/aligned/aligned_{ref_dialect.value}.npy")
    if not ref_emb_file.exists():
        raise FileNotFoundError(
            f"Reference embeddings for {ref_dialect.value} not found. "
            "Run 'eigendialectos embed align' first."
        )

    emb_data = np.load(ref_emb_file, allow_pickle=False)
    embeddings = EmbeddingMatrix(
        data=emb_data,
        vocab=vocab,
        dialect_code=ref_dialect,
    )

    return transforms, eigendecomps, vocab, embeddings


@click.command("generate")
@click.option("--text", required=True, help="Input text in neutral Spanish.")
@click.option("--dialect", required=True, help="Target dialect code.")
@click.option("--alpha", default=1.0, type=float, help="Dialectal intensity (0.0-1.5).")
@click.option("--method", default="algebraic", help="Generation method.")
@click.option("--models-dir", default="models", help="Directory containing trained models and spectral data.")
def dial_generate(text: str, dialect: str, alpha: float, method: str, models_dir: str) -> None:
    """Generate text in a target dialect variety."""
    from eigendialectos.constants import DialectCode, DIALECT_NAMES
    from eigendialectos.generative.generator import DialectGenerator

    # Validate dialect
    try:
        target = DialectCode(dialect)
    except ValueError:
        available = [dc.value for dc in DialectCode]
        click.echo(
            f"Error: Unknown dialect code '{dialect}'. "
            f"Available: {available}",
            err=True,
        )
        return

    click.echo(f"Generating {DIALECT_NAMES.get(target, target.value)} text at alpha={alpha}...")
    click.echo(f"Input:  {text}")

    # Load resources
    try:
        transforms, eigendecomps, vocab, embeddings = _load_generation_resources(models_dir)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        return

    if target not in eigendecomps:
        click.echo(
            f"Error: No eigendecomposition available for {target.value}. "
            f"Available: {[dc.value for dc in eigendecomps]}",
            err=True,
        )
        return

    # Create generator and generate
    try:
        generator = DialectGenerator(
            transforms=transforms,
            eigendecomps=eigendecomps,
            vocab=vocab,
            embeddings=embeddings,
        )

        result = generator.generate(
            text=text,
            target_dialect=target,
            alpha=alpha,
            method=method,
        )
        click.echo(f"Output: {result}")

    except Exception as e:
        click.echo(f"Error during generation: {e}", err=True)
        logger.exception("Generation failed")


@click.command("mix")
@click.option("--dialects", required=True, help="Dialect:weight pairs, e.g. 'ES_CAN:0.6,ES_AND:0.4'.")
@click.option("--text", required=True, help="Input text in neutral Spanish.")
@click.option("--alpha", default=1.0, type=float, help="Overall intensity.")
@click.option("--models-dir", default="models", help="Directory containing trained models and spectral data.")
def dial_mix(dialects: str, text: str, alpha: float, models_dir: str) -> None:
    """Generate text mixing multiple dialect varieties."""
    from eigendialectos.constants import DialectCode, DIALECT_NAMES
    from eigendialectos.generative.generator import DialectGenerator

    # Parse dialect:weight pairs
    dialect_weights: dict[DialectCode, float] = {}
    for pair in dialects.split(","):
        pair = pair.strip()
        if ":" not in pair:
            click.echo(
                f"Error: Invalid dialect:weight pair '{pair}'. "
                "Use format 'ES_XXX:0.5'.",
                err=True,
            )
            return
        code_str, weight_str = pair.split(":", 1)
        try:
            dc = DialectCode(code_str.strip())
        except ValueError:
            click.echo(f"Error: Unknown dialect code '{code_str}'.", err=True)
            return
        try:
            weight = float(weight_str.strip())
        except ValueError:
            click.echo(f"Error: Invalid weight '{weight_str}' for {code_str}.", err=True)
            return
        dialect_weights[dc] = weight

    # Validate weights sum to 1
    total_weight = sum(dialect_weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        click.echo(
            f"Warning: Weights sum to {total_weight:.4f}, normalizing to 1.0.",
            err=True,
        )
        for dc in dialect_weights:
            dialect_weights[dc] /= total_weight

    mix_desc = ", ".join(
        f"{DIALECT_NAMES.get(dc, dc.value)}:{w:.2f}"
        for dc, w in dialect_weights.items()
    )
    click.echo(f"Mixing dialects: {mix_desc}")
    click.echo(f"Input:  {text}")
    click.echo(f"Alpha:  {alpha}")

    # Load resources
    try:
        transforms, eigendecomps, vocab, embeddings = _load_generation_resources(models_dir)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        return

    # Verify all requested dialects have eigendecompositions
    for dc in dialect_weights:
        if dc not in eigendecomps:
            click.echo(
                f"Error: No eigendecomposition available for {dc.value}. "
                f"Available: {[d.value for d in eigendecomps]}",
                err=True,
            )
            return

    # Create generator and mix
    try:
        generator = DialectGenerator(
            transforms=transforms,
            eigendecomps=eigendecomps,
            vocab=vocab,
            embeddings=embeddings,
        )

        result = generator.generate_mixed(
            text=text,
            dialect_weights=dialect_weights,
            alpha=alpha,
        )
        click.echo(f"Output: {result}")

    except Exception as e:
        click.echo(f"Error during mixing: {e}", err=True)
        logger.exception("Dialect mixing failed")
