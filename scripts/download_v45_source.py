#!/usr/bin/env python3
"""Download a single v4.5 high-quality source.

Each source is runnable independently so multiple can run in parallel.

Usage:
    python scripts/download_v45_source.py --source tweet_hisp --max-per-dialect 200000
    python scripts/download_v45_source.py --source preseea
    python scripts/download_v45_source.py --source coser
    python scripts/download_v45_source.py --source lanzarote
    python scripts/download_v45_source.py --source cv17
    python scripts/download_v45_source.py --source parcan
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
    parser = argparse.ArgumentParser(description="Download a single v4.5 source")
    parser.add_argument("--source", type=str, required=True,
                        choices=["tweet_hisp", "preseea", "coser",
                                 "lanzarote", "cv17", "parcan"],
                        help="Which v4.5 source to download")
    parser.add_argument("--cache-dir", type=str,
                        default=str(ROOT / "data" / "raw_v4"),
                        help="Cache directory (default: data/raw_v4)")
    parser.add_argument("--max-per-dialect", type=int, default=50_000,
                        help="Max docs per dialect (default: 50000)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from eigen3.downloader_v45 import download_single_source

    print(f"\n=== Downloading v4.5 source: {args.source} ===")
    print(f"Cache dir: {cache_dir}")
    print(f"Max per dialect: {args.max_per_dialect}")
    print()

    result = download_single_source(
        source=args.source,
        cache_dir=cache_dir,
        hf_token=hf_token,
        max_per_dialect=args.max_per_dialect,
    )

    total = sum(len(v) for v in result.values())
    print(f"\n=== {args.source} COMPLETE: {total:,} total docs ===")
    for d in sorted(result.keys()):
        if result[d]:
            print(f"  {d}: {len(result[d]):,}")


if __name__ == "__main__":
    main()
