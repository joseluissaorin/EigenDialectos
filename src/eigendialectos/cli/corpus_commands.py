"""CLI commands for corpus management."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command("download")
@click.option("--source", default="all", help="Source name or 'all'.")
@click.option("--output", default="data/raw", help="Output directory.")
def corpus_download(source: str, output: str) -> None:
    """Download corpus data from configured sources."""
    from eigendialectos.corpus.registry import list_available, get_source

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    sources = list_available() if source == "all" else [source]
    for name in sources:
        click.echo(f"Downloading from {name}...")
        try:
            src = get_source(name)
            src.download(output_path / name)
            click.echo(f"  Done: {name}")
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)


@click.command("preprocess")
@click.option("--config", default="demo", help="Config profile: demo or full.")
@click.option("--input-dir", default="data/raw", help="Input directory.")
@click.option("--output-dir", default="data/processed", help="Output directory.")
def corpus_preprocess(config: str, input_dir: str, output_dir: str) -> None:
    """Preprocess raw corpus data."""
    from eigendialectos.corpus.preprocessing.noise import clean_text
    from eigendialectos.corpus.preprocessing.segmentation import split_sentences
    from eigendialectos.corpus.preprocessing.filters import apply_filters
    from eigendialectos.constants import DialectCode
    from eigendialectos.types import DialectSample

    click.echo(f"Preprocessing with config={config}...")
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Collect all JSONL files from input directory
    jsonl_files = sorted(in_path.rglob("*.jsonl"))
    if not jsonl_files:
        click.echo(f"Error: No .jsonl files found in {in_path}", err=True)
        return

    click.echo(f"Found {len(jsonl_files)} JSONL file(s) to process.")

    # Load all samples from JSONL files
    samples: list[DialectSample] = []
    for jsonl_file in jsonl_files:
        click.echo(f"  Loading {jsonl_file.name}...")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping invalid JSON at %s:%d", jsonl_file, line_num
                    )
                    continue

                text = record.get("text", "")
                dialect_str = record.get("dialect", "")
                try:
                    dialect_code = DialectCode(dialect_str)
                except ValueError:
                    logger.warning(
                        "Unknown dialect code '%s' at %s:%d, skipping",
                        dialect_str, jsonl_file, line_num,
                    )
                    continue

                samples.append(
                    DialectSample(
                        text=text,
                        dialect_code=dialect_code,
                        source_id=record.get("source", "unknown"),
                        confidence=record.get("confidence", 0.5),
                        metadata=record.get("metadata", {}),
                    )
                )

    click.echo(f"Loaded {len(samples)} raw samples.")

    # Step 1: Clean text (noise removal + normalization)
    click.echo("  Cleaning text...")
    for sample in samples:
        sample.text = clean_text(sample.text)

    # Step 2: Sentence segmentation -- split multi-sentence texts
    click.echo("  Segmenting sentences...")
    segmented_samples: list[DialectSample] = []
    for sample in samples:
        sentences = split_sentences(sample.text)
        if not sentences:
            # Keep original if segmentation yields nothing
            if sample.text.strip():
                segmented_samples.append(sample)
            continue
        for sent in sentences:
            segmented_samples.append(
                DialectSample(
                    text=sent,
                    dialect_code=sample.dialect_code,
                    source_id=sample.source_id,
                    confidence=sample.confidence,
                    metadata=sample.metadata,
                )
            )

    click.echo(f"  After segmentation: {len(segmented_samples)} samples.")

    # Step 3: Quality filtering
    click.echo("  Applying quality filters...")
    filter_config = {
        "min_length": {"min_len": 10},
        "language": {"lang": "es"},
        "dedup": {},
        "quality": {},
    }
    if config == "full":
        filter_config["near_dedup"] = {"threshold": 0.9}
        filter_config["confidence"] = {"min_confidence": 0.5}

    filtered_samples = apply_filters(segmented_samples, config=filter_config)
    click.echo(f"  After filtering: {len(filtered_samples)} samples.")

    # Step 4: Save preprocessed corpus
    # Save combined corpus
    combined_path = out_path / "corpus.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for sample in filtered_samples:
            record = {
                "text": sample.text,
                "dialect": sample.dialect_code.value,
                "confidence": sample.confidence,
                "source": sample.source_id,
                "metadata": sample.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save per-dialect files
    by_dialect: dict[DialectCode, list[DialectSample]] = defaultdict(list)
    for sample in filtered_samples:
        by_dialect[sample.dialect_code].append(sample)

    for dialect_code, dialect_samples in sorted(by_dialect.items(), key=lambda x: x[0].value):
        dialect_path = out_path / f"{dialect_code.value}.jsonl"
        with open(dialect_path, "w", encoding="utf-8") as f:
            for sample in dialect_samples:
                record = {
                    "text": sample.text,
                    "dialect": sample.dialect_code.value,
                    "confidence": sample.confidence,
                    "source": sample.source_id,
                    "metadata": sample.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        click.echo(f"  {dialect_code.value}: {len(dialect_samples)} samples -> {dialect_path.name}")

    click.echo(f"Preprocessing complete. Output saved to {out_path}")


@click.command("stats")
@click.option("--data-dir", default="data/processed", help="Data directory.")
def corpus_stats(data_dir: str) -> None:
    """Show corpus statistics."""
    from eigendialectos.constants import DialectCode, DIALECT_NAMES

    data_path = Path(data_dir)

    # Collect all JSONL files
    jsonl_files = sorted(data_path.glob("*.jsonl"))
    if not jsonl_files:
        click.echo(f"Error: No .jsonl files found in {data_path}", err=True)
        return

    # Prefer corpus.jsonl if it exists, otherwise combine all dialect files
    corpus_file = data_path / "corpus.jsonl"
    if corpus_file.exists():
        source_files = [corpus_file]
    else:
        source_files = jsonl_files

    # Load and aggregate
    dialect_stats: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "total_length": 0,
        "min_length": float("inf"),
        "max_length": 0,
        "sources": defaultdict(int),
    })

    total_samples = 0
    for jsonl_file in source_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                dialect = record.get("dialect", "UNKNOWN")
                text = record.get("text", "")
                source = record.get("source", "unknown")
                text_len = len(text)

                stats = dialect_stats[dialect]
                stats["count"] += 1
                stats["total_length"] += text_len
                stats["min_length"] = min(stats["min_length"], text_len)
                stats["max_length"] = max(stats["max_length"], text_len)

                # Extract source category (before the colon if present)
                source_category = source.split(":")[0] if ":" in source else source
                stats["sources"][source_category] += 1

                total_samples += 1

    if total_samples == 0:
        click.echo("No samples found in corpus files.")
        return

    # Print statistics
    click.echo(f"\nCorpus statistics from {data_path}")
    click.echo(f"{'=' * 80}")
    click.echo(f"Total samples: {total_samples}")
    click.echo(f"Dialects: {len(dialect_stats)}")
    click.echo(f"{'=' * 80}")

    # Header
    click.echo(
        f"{'Dialect':<12} {'Name':<30} {'Count':>7} {'Avg Len':>8} "
        f"{'Min':>5} {'Max':>6}  Sources"
    )
    click.echo(f"{'-' * 80}")

    for dialect_code_str in sorted(dialect_stats.keys()):
        stats = dialect_stats[dialect_code_str]
        count = stats["count"]
        avg_len = stats["total_length"] / count if count > 0 else 0
        min_len = stats["min_length"] if stats["min_length"] != float("inf") else 0
        max_len = stats["max_length"]

        # Get friendly name
        try:
            dc = DialectCode(dialect_code_str)
            name = DIALECT_NAMES.get(dc, dialect_code_str)
        except ValueError:
            name = dialect_code_str

        # Source breakdown
        source_parts = [
            f"{src}({n})" for src, n in sorted(stats["sources"].items(), key=lambda x: -x[1])
        ]
        sources_str = ", ".join(source_parts[:3])
        if len(source_parts) > 3:
            sources_str += f" +{len(source_parts) - 3} more"

        click.echo(
            f"{dialect_code_str:<12} {name:<30} {count:>7} {avg_len:>8.1f} "
            f"{min_len:>5} {max_len:>6}  {sources_str}"
        )

    click.echo(f"{'=' * 80}")


@click.command("generate-synthetic")
@click.option("-n", "--num-samples", default=200, help="Samples per dialect.")
@click.option("--output-dir", default="data/synthetic", help="Output directory.")
def corpus_generate_synthetic(num_samples: int, output_dir: str) -> None:
    """Generate synthetic dialect data."""
    from eigendialectos.corpus.synthetic.generator import SyntheticGenerator
    from eigendialectos.constants import DialectCode, DIALECT_NAMES
    from eigendialectos.utils.io import save_json

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gen = SyntheticGenerator()
    all_data = gen.generate_all(n_per_dialect=num_samples)

    for code, corpus_slice in all_data.items():
        click.echo(f"  {DIALECT_NAMES[code]}: {len(corpus_slice.samples)} samples")
        data = [{"text": s.text, "dialect": s.dialect_code.value} for s in corpus_slice.samples]
        save_json(data, out_path / f"{code.value}.json")

    click.echo(f"Synthetic data saved to {output_dir}")
