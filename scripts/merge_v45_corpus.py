#!/usr/bin/env python3
"""Merge all v4 + v4.5 raw shards into a single processed_v4 corpus.

Walks ``data/raw_v4/<source>/<DIALECT>.jsonl`` for every source directory
that contains per-dialect JSONL files (culturax, leipzig, opensubtitles,
tweets, coser, lanzarote, parcan, preseea, cv17, tweet_hisp), runs the
samples through ``build_filtered_corpus`` (dedup, OCR clean, language
detect, length + confidence + authenticity), and writes the cleaned
result to ``data/processed_v4/<DIALECT>.jsonl``.

A per-dialect cap is applied BEFORE filtering to keep training tractable
on a Mac. The cap is sampled uniformly across the available sources for
that dialect so each source is represented (instead of only the first
shards being kept).

Usage:
    python scripts/merge_v45_corpus.py
    python scripts/merge_v45_corpus.py --max-per-dialect 100000
    python scripts/merge_v45_corpus.py --output-dir data/processed_v45
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eigen3.constants import ALL_VARIETIES  # noqa: E402
from eigen3.corpus import build_filtered_corpus  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merge_v45")


def load_source_shards(
    raw_dir: Path,
) -> dict[str, dict[str, list[dict]]]:
    """Return {source_name: {dialect: [sample_dict, ...]}}.

    Only directories that contain per-dialect ``ES_*.jsonl`` files are
    considered (skips ``leipzig``'s ``.tar.gz``, etc.).
    """
    by_source: dict[str, dict[str, list[dict]]] = {}
    for source_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        per_dialect: dict[str, list[dict]] = {}
        for dialect in ALL_VARIETIES:
            shard = source_dir / f"{dialect}.jsonl"
            if not shard.exists():
                continue
            samples: list[dict] = []
            with shard.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if samples:
                per_dialect[dialect] = samples
        if per_dialect:
            by_source[source_dir.name] = per_dialect
            total = sum(len(v) for v in per_dialect.values())
            logger.info(
                "  source %-15s : %s dialects, %s samples",
                source_dir.name, len(per_dialect), f"{total:,}",
            )
    return by_source


def assemble_raw(
    by_source: dict[str, dict[str, list[dict]]],
    max_per_dialect: int,
    seed: int,
) -> dict[str, list[dict]]:
    """Combine sources into a single {dialect: [samples]} map.

    If ``max_per_dialect <= 0`` no cap is applied (full corpus).
    """
    rng = random.Random(seed)
    uncapped = max_per_dialect <= 0

    # Group all samples per-dialect, tagged by their source
    per_dialect: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for source_name, dialect_map in by_source.items():
        for dialect, samples in dialect_map.items():
            for s in samples:
                per_dialect[dialect].append((source_name, s))

    raw: dict[str, list[dict]] = {}
    for dialect in ALL_VARIETIES:
        items = per_dialect.get(dialect, [])
        n_total = len(items)
        if n_total == 0:
            raw[dialect] = []
            logger.warning("  %s: no samples found", dialect)
            continue

        if not uncapped and n_total > max_per_dialect:
            rng.shuffle(items)
            items = items[:max_per_dialect]

        raw[dialect] = [s for (_src, s) in items]

        # Per-source breakdown
        breakdown: dict[str, int] = defaultdict(int)
        for src, _ in items:
            breakdown[src] += 1
        breakdown_str = ", ".join(
            f"{k}={v:,}" for k, v in sorted(breakdown.items(), key=lambda x: -x[1])
        )
        cap_note = "uncapped" if uncapped else "after cap"
        logger.info(
            "  %s: %s/%s docs (%s) | %s",
            dialect, f"{len(raw[dialect]):,}", f"{n_total:,}", cap_note, breakdown_str,
        )

    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=ROOT / "data" / "raw_v4",
        help="Directory containing per-source raw JSONL shards",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "processed_v4",
        help="Output directory for cleaned per-dialect JSONL files",
    )
    parser.add_argument(
        "--max-per-dialect", type=int, default=0,
        help="Cap on raw samples per dialect before filtering. 0 = no cap (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-length", type=int, default=30,
        help="Minimum text length in characters",
    )
    parser.add_argument(
        "--max-length", type=int, default=2000,
        help="Maximum text length in characters",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.3,
        help="Minimum source confidence to keep",
    )
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MERGE v4 + v4.5 RAW SHARDS")
    logger.info("=" * 60)
    logger.info("Raw dir   : %s", raw_dir)
    logger.info("Output    : %s", out_dir)
    logger.info("Cap/dialect: %s",
                "uncapped" if args.max_per_dialect <= 0 else f"{args.max_per_dialect:,}")
    logger.info("=" * 60)

    logger.info("Loading source shards...")
    by_source = load_source_shards(raw_dir)
    logger.info("Found %d sources with per-dialect shards", len(by_source))

    logger.info("Assembling raw dialect map (with per-dialect cap)...")
    raw = assemble_raw(by_source, args.max_per_dialect, args.seed)

    pre_total = sum(len(v) for v in raw.values())
    logger.info("Pre-filter total: %s", f"{pre_total:,}")

    logger.info("Running build_filtered_corpus (dedup + OCR + lang + length + auth)...")
    cleaned = build_filtered_corpus(
        raw_samples=raw,
        output_dir=out_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        min_confidence=args.min_confidence,
    )

    post_total = sum(len(v) for v in cleaned.values())
    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info("Pre-filter : %s", f"{pre_total:,}")
    logger.info("Post-filter: %s (%.1f%% kept)", f"{post_total:,}",
                100.0 * post_total / max(pre_total, 1))
    for d in sorted(cleaned.keys()):
        logger.info("  %s: %s docs", d, f"{len(cleaned[d]):,}")
    logger.info("Output written to: %s", out_dir)


if __name__ == "__main__":
    main()
