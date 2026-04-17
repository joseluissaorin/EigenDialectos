#!/usr/bin/env python3
"""Download bulk corpus for EigenDialectos v4.

Usage:
    python scripts/download_corpus_v4.py --all
    python scripts/download_corpus_v4.py --sources leipzig,culturax
    python scripts/download_corpus_v4.py --sources tweets --max-per-dialect 100000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load .env if present
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def main():
    parser = argparse.ArgumentParser(description="Download bulk corpus for EigenDialectos v4")
    parser.add_argument("--all", action="store_true", help="Download from all sources")
    parser.add_argument("--sources", type=str, default=None,
                        help="Comma-separated sources: leipzig,culturax,tweets,reddit,opensubtitles,literary")
    parser.add_argument("--max-per-dialect", type=int, default=200_000,
                        help="Max documents per dialect per source (default: 200000)")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "data" / "processed_v4"),
                        help="Output directory for processed JSONL")
    parser.add_argument("--cache-dir", type=str, default=str(ROOT / "data" / "raw_v4"),
                        help="Cache directory for raw downloads")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Balancing temperature (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-merge", action="store_true",
                        help="Don't merge with existing v3 corpus")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download raw data, skip filtering/balancing")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    sources = None
    if args.sources:
        sources = [s.strip() for s in args.sources.split(",")]
    elif args.all:
        sources = None  # download_all uses all sources when None

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    if args.download_only:
        # Just download, don't filter/balance
        from eigen3.downloader import CorpusDownloader
        dl = CorpusDownloader(
            cache_dir=args.cache_dir,
            hf_token=hf_token,
        )
        raw = dl.download_all(sources=sources, max_per_dialect=args.max_per_dialect)

        total = sum(len(s) for s in raw.values())
        print(f"\nDownload complete: {total:,} total documents")
        for d in sorted(raw.keys()):
            print(f"  {d}: {len(raw[d]):,} docs")
    else:
        # Full pipeline: download -> filter -> balance -> blend
        from eigen3.corpus import download_and_build
        corpus = download_and_build(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            sources=sources,
            max_per_dialect=args.max_per_dialect,
            temperature=args.temperature,
            seed=args.seed,
            hf_token=hf_token,
            merge_existing=not args.no_merge,
        )

        total = sum(len(d) for d in corpus.values())
        print(f"\nCorpus build complete: {total:,} total documents")
        for d in sorted(corpus.keys()):
            print(f"  {d}: {len(corpus[d]):,} docs")
        print(f"\nOutput written to: {args.output_dir}")


if __name__ == "__main__":
    main()
