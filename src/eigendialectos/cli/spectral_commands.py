"""CLI commands for spectral analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np

logger = logging.getLogger(__name__)


@click.command("compute")
@click.option("--reference", default="ES_PEN", help="Reference dialect code.")
@click.option("--method", default="lstsq", help="Transform computation method: lstsq, procrustes, nuclear.")
@click.option("--regularization", default=0.01, type=float, help="Ridge regularization lambda.")
@click.option("--embeddings-dir", default="models/aligned", help="Directory containing aligned embeddings.")
@click.option("--output-dir", default="data/spectral", help="Output directory.")
def spectral_compute(
    reference: str,
    method: str,
    regularization: float,
    embeddings_dir: str,
    output_dir: str,
) -> None:
    """Compute transformation matrices and eigendecompositions."""
    from eigendialectos.constants import DialectCode
    from eigendialectos.spectral.transformation import compute_all_transforms
    from eigendialectos.spectral.eigendecomposition import eigendecompose
    from eigendialectos.types import EmbeddingMatrix
    from eigendialectos.utils.io import save_numpy, save_json

    emb_path = Path(embeddings_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Validate reference
    try:
        ref_dialect = DialectCode(reference)
    except ValueError:
        click.echo(f"Error: Unknown reference dialect '{reference}'.", err=True)
        return

    # Load aligned embeddings
    click.echo(f"Loading aligned embeddings from {emb_path}...")

    # Load shared vocabulary
    vocab_file = emb_path / "shared_vocab.json"
    if not vocab_file.exists():
        click.echo(
            f"Error: shared_vocab.json not found in {emb_path}. "
            "Run 'eigendialectos embed align' first.",
            err=True,
        )
        return

    with open(vocab_file, "r", encoding="utf-8") as f:
        shared_vocab = json.load(f)

    # Load .npy embedding files
    embeddings: dict[DialectCode, EmbeddingMatrix] = {}
    for npy_file in sorted(emb_path.glob("aligned_*.npy")):
        # Extract dialect code from filename: aligned_ES_XXX.npy
        name = npy_file.stem  # "aligned_ES_XXX"
        dialect_str = name.replace("aligned_", "")
        try:
            dc = DialectCode(dialect_str)
        except ValueError:
            logger.warning("Skipping unrecognized file: %s", npy_file.name)
            continue

        data = np.load(npy_file)
        # Transpose to (d, V) if needed -- transformation expects (d, V)
        # The aligned embeddings are stored as (V, d) from encode_words
        if data.shape[0] == len(shared_vocab):
            data = data.T  # Transpose to (d, V)

        embeddings[dc] = EmbeddingMatrix(
            data=data,
            vocab=shared_vocab,
            dialect_code=dc,
        )
        click.echo(f"  Loaded {dc.value}: shape {data.shape}")

    if ref_dialect not in embeddings:
        click.echo(
            f"Error: Reference dialect {ref_dialect.value} not found in embeddings. "
            f"Available: {[d.value for d in embeddings]}",
            err=True,
        )
        return

    if len(embeddings) < 2:
        click.echo("Error: Need at least 2 dialect embeddings.", err=True)
        return

    # Compute transformation matrices
    click.echo(
        f"Computing transformations (reference={ref_dialect.value}, "
        f"method={method}, lambda={regularization})..."
    )
    try:
        transforms = compute_all_transforms(
            embeddings=embeddings,
            reference=ref_dialect,
            method=method,
            regularization=regularization,
        )
    except Exception as e:
        click.echo(f"Error computing transforms: {e}", err=True)
        logger.exception("Transform computation failed")
        return

    # Save transformation matrices
    transforms_dir = out_path / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    for dc, W in transforms.items():
        save_numpy(W.data, transforms_dir / f"W_{dc.value}.npy")
        click.echo(f"  Saved transform W_{dc.value}: shape {W.data.shape}")

    # Compute eigendecompositions
    click.echo("Computing eigendecompositions...")
    eigendecomps_dir = out_path / "eigendecompositions"
    eigendecomps_dir.mkdir(parents=True, exist_ok=True)

    eigenvalues_summary: dict[str, dict] = {}

    for dc, W in transforms.items():
        try:
            eigen = eigendecompose(W)

            # Save eigenvalues, eigenvectors, and inverse
            save_numpy(
                eigen.eigenvalues,
                eigendecomps_dir / f"eigenvalues_{dc.value}.npy",
            )
            save_numpy(
                eigen.eigenvectors,
                eigendecomps_dir / f"eigenvectors_{dc.value}.npy",
            )
            save_numpy(
                eigen.eigenvectors_inv,
                eigendecomps_dir / f"eigenvectors_inv_{dc.value}.npy",
            )

            # Summary info
            magnitudes = np.abs(eigen.eigenvalues)
            eigenvalues_summary[dc.value] = {
                "n_eigenvalues": len(eigen.eigenvalues),
                "effective_rank": int(eigen.rank),
                "max_magnitude": float(np.max(magnitudes)),
                "min_magnitude": float(np.min(magnitudes)),
                "mean_magnitude": float(np.mean(magnitudes)),
                "top_5_magnitudes": [float(v) for v in np.sort(magnitudes)[::-1][:5]],
            }

            click.echo(
                f"  {dc.value}: rank={eigen.rank}, "
                f"max|lambda|={np.max(magnitudes):.4f}"
            )

        except Exception as e:
            click.echo(f"  Error decomposing {dc.value}: {e}", err=True)
            logger.exception("Eigendecomposition failed for %s", dc.value)

    # Save eigenvalue summary
    save_json(eigenvalues_summary, out_path / "eigenvalues_summary.json")

    # Save metadata
    metadata = {
        "reference": ref_dialect.value,
        "method": method,
        "regularization": regularization,
        "dialects": [dc.value for dc in embeddings],
        "vocab_size": len(shared_vocab),
    }
    save_json(metadata, out_path / "metadata.json")

    click.echo(f"Spectral computation complete. Output saved to {out_path}")


@click.command("analyze")
@click.option("--input-dir", default="data/spectral", help="Spectral data directory.")
@click.option("--output-dir", default="outputs/spectral", help="Output directory.")
def spectral_analyze(input_dir: str, output_dir: str) -> None:
    """Analyze eigenspectra, entropy, and distances."""
    from eigendialectos.constants import DialectCode, DIALECT_NAMES
    from eigendialectos.spectral.eigenspectrum import compute_eigenspectrum
    from eigendialectos.spectral.distance import (
        compute_distance_matrix,
        spectral_distance,
    )
    from eigendialectos.spectral.entropy import compute_dialectal_entropy, compare_entropies
    from eigendialectos.types import EigenDecomposition, TransformationMatrix
    from eigendialectos.utils.io import save_json

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Check input directory
    eigendecomps_dir = in_path / "eigendecompositions"
    transforms_dir = in_path / "transforms"

    if not eigendecomps_dir.exists():
        click.echo(
            f"Error: Eigendecompositions directory not found at {eigendecomps_dir}. "
            "Run 'eigendialectos spectral compute' first.",
            err=True,
        )
        return

    # Load metadata
    metadata_file = in_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Load eigendecompositions
    click.echo(f"Loading spectral data from {in_path}...")
    eigendecomps: dict[DialectCode, EigenDecomposition] = {}
    transforms: dict[DialectCode, TransformationMatrix] = {}

    for ev_file in sorted(eigendecomps_dir.glob("eigenvalues_*.npy")):
        dialect_str = ev_file.stem.replace("eigenvalues_", "")
        try:
            dc = DialectCode(dialect_str)
        except ValueError:
            continue

        eigenvalues = np.load(ev_file, allow_pickle=False)
        eigvec_file = eigendecomps_dir / f"eigenvectors_{dialect_str}.npy"
        eigvec_inv_file = eigendecomps_dir / f"eigenvectors_inv_{dialect_str}.npy"

        if not eigvec_file.exists() or not eigvec_inv_file.exists():
            logger.warning("Missing eigenvector files for %s", dialect_str)
            continue

        eigenvectors = np.load(eigvec_file, allow_pickle=False)
        eigenvectors_inv = np.load(eigvec_inv_file, allow_pickle=False)

        eigendecomps[dc] = EigenDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            eigenvectors_inv=eigenvectors_inv,
            dialect_code=dc,
        )
        click.echo(f"  Loaded eigendecomposition: {dc.value}")

    # Load transforms if available
    if transforms_dir.exists():
        for w_file in sorted(transforms_dir.glob("W_*.npy")):
            dialect_str = w_file.stem.replace("W_", "")
            try:
                dc = DialectCode(dialect_str)
            except ValueError:
                continue

            W_data = np.load(w_file, allow_pickle=False)
            ref_str = metadata.get("reference", "ES_PEN")
            try:
                ref_dc = DialectCode(ref_str)
            except ValueError:
                ref_dc = DialectCode.ES_PEN

            transforms[dc] = TransformationMatrix(
                data=W_data,
                source_dialect=ref_dc,
                target_dialect=dc,
                regularization=metadata.get("regularization", 0.01),
            )

    if not eigendecomps:
        click.echo("Error: No eigendecompositions found.", err=True)
        return

    # Compute eigenspectra
    click.echo("\nComputing eigenspectra...")
    spectra = {}
    for dc, eigen in eigendecomps.items():
        spectrum = compute_eigenspectrum(eigen)
        spectra[dc] = spectrum

    # Compute entropies
    entropies: dict[DialectCode, float] = {}
    for dc, spectrum in spectra.items():
        entropies[dc] = spectrum.entropy

    entropy_comparison = compare_entropies(entropies)

    # Print spectral summary table
    click.echo(f"\n{'=' * 90}")
    click.echo("SPECTRAL ANALYSIS SUMMARY")
    click.echo(f"{'=' * 90}")

    click.echo(
        f"{'Dialect':<12} {'Name':<25} {'Entropy':>8} {'Rank':>5} "
        f"{'Max |λ|':>8} {'Energy@10':>10}"
    )
    click.echo(f"{'-' * 90}")

    for dc in sorted(spectra.keys(), key=lambda d: d.value):
        spectrum = spectra[dc]
        eigen = eigendecomps[dc]
        name = DIALECT_NAMES.get(dc, dc.value)

        magnitudes = np.abs(eigen.eigenvalues)
        max_mag = float(np.max(magnitudes))

        # Cumulative energy at rank 10
        cum_energy = spectrum.cumulative_energy
        energy_at_10 = float(cum_energy[min(9, len(cum_energy) - 1)]) if len(cum_energy) > 0 else 0.0

        click.echo(
            f"{dc.value:<12} {name:<25} {spectrum.entropy:>8.4f} "
            f"{eigen.rank:>5} {max_mag:>8.4f} {energy_at_10:>10.4f}"
        )

    click.echo(f"{'-' * 90}")

    # Print entropy comparison
    click.echo(f"\nEntropy Statistics:")
    click.echo(f"  Mean: {entropy_comparison['mean']:.4f}")
    click.echo(f"  Std:  {entropy_comparison['std']:.4f}")
    click.echo(f"  Range: {entropy_comparison['range']:.4f}")
    if entropy_comparison.get("max"):
        click.echo(f"  Max: {entropy_comparison['max'][0].value} ({entropy_comparison['max'][1]:.4f})")
    if entropy_comparison.get("min"):
        click.echo(f"  Min: {entropy_comparison['min'][0].value} ({entropy_comparison['min'][1]:.4f})")
    click.echo(f"  {entropy_comparison.get('interpretation', '')}")

    # Compute pairwise distance matrix if transforms available
    results: dict[str, object] = {
        "spectra": {},
        "entropies": {},
        "entropy_comparison": {
            "mean": entropy_comparison["mean"],
            "std": entropy_comparison["std"],
            "range": entropy_comparison["range"],
            "interpretation": entropy_comparison.get("interpretation", ""),
        },
    }

    for dc, spectrum in spectra.items():
        results["spectra"][dc.value] = {
            "entropy": float(spectrum.entropy),
            "top_eigenvalues": [float(v) for v in spectrum.eigenvalues_sorted[:10]],
            "cumulative_energy_at_10": float(
                spectrum.cumulative_energy[min(9, len(spectrum.cumulative_energy) - 1)]
            ) if len(spectrum.cumulative_energy) > 0 else 0.0,
        }

    for dc, h in entropies.items():
        results["entropies"][dc.value] = float(h)

    if transforms and len(transforms) >= 2:
        click.echo("\nComputing pairwise distance matrix...")
        try:
            D = compute_distance_matrix(
                transforms=transforms,
                spectra=spectra,
                entropies=entropies,
                method="combined",
            )
            codes = sorted(transforms.keys(), key=lambda c: c.value)

            # Print distance matrix
            click.echo(f"\n{'Pairwise Distances':}")
            header = f"{'':>12}" + "".join(f"{c.value:>12}" for c in codes)
            click.echo(header)
            for i, ci in enumerate(codes):
                row = f"{ci.value:>12}" + "".join(f"{D[i, j]:>12.4f}" for j in range(len(codes)))
                click.echo(row)

            # Save distance matrix
            np.save(out_path / "distance_matrix.npy", D)
            results["distance_matrix"] = {
                "dialects": [c.value for c in codes],
                "matrix": D.tolist(),
            }

        except Exception as e:
            click.echo(f"  Warning: Could not compute distance matrix: {e}", err=True)
            logger.warning("Distance matrix computation failed", exc_info=True)

    # Save results as JSON
    save_json(results, out_path / "spectral_analysis.json")

    # Save as CSV (spectra summary)
    csv_path = out_path / "spectral_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("dialect,name,entropy,rank,max_magnitude,energy_at_10\n")
        for dc in sorted(spectra.keys(), key=lambda d: d.value):
            spectrum = spectra[dc]
            eigen = eigendecomps[dc]
            name = DIALECT_NAMES.get(dc, dc.value)
            magnitudes = np.abs(eigen.eigenvalues)
            max_mag = float(np.max(magnitudes))
            cum_energy = spectrum.cumulative_energy
            energy_at_10 = float(cum_energy[min(9, len(cum_energy) - 1)]) if len(cum_energy) > 0 else 0.0
            f.write(f"{dc.value},{name},{spectrum.entropy:.6f},{eigen.rank},{max_mag:.6f},{energy_at_10:.6f}\n")

    click.echo(f"\nResults saved to {out_path}")
    click.echo(f"  - spectral_analysis.json")
    click.echo(f"  - spectral_summary.csv")
    if (out_path / "distance_matrix.npy").exists():
        click.echo(f"  - distance_matrix.npy")
    click.echo("Analysis complete.")
