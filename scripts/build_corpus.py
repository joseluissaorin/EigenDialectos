#!/usr/bin/env python3
"""Build the EigenDialectos corpus from multiple real and synthetic sources.

Usage:
    python scripts/build_corpus.py                    # all sources
    python scripts/build_corpus.py --sources wiki     # Wikipedia only
    python scripts/build_corpus.py --sources synth    # enhanced synthetic only
    python scripts/build_corpus.py --sources wiki,web  # Wikipedia + web
    python scripts/build_corpus.py --stats-only       # just print stats for existing corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eigendialectos.constants import DialectCode, DIALECT_NAMES
from eigendialectos.types import CorpusSlice, DialectSample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_corpus")

# ======================================================================
# Data directories
# ======================================================================

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CORPUS_FILE = PROCESSED_DIR / "corpus.jsonl"
STATS_FILE = PROCESSED_DIR / "corpus_stats.json"


def ensure_dirs() -> None:
    for d in [RAW_DIR, PROCESSED_DIR, RAW_DIR / "wikipedia", RAW_DIR / "opus",
              RAW_DIR / "web", DATA_DIR / "synthetic"]:
        d.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Source runners
# ======================================================================

def run_wikipedia() -> dict[DialectCode, list[DialectSample]]:
    """Fetch Wikipedia articles via MediaWiki API."""
    log.info("=" * 60)
    log.info("SOURCE: Wikipedia (MediaWiki API)")
    log.info("=" * 60)
    try:
        from eigendialectos.corpus.acquisition.wikipedia_fetcher import WikipediaFetcher
        fetcher = WikipediaFetcher(cache_dir=str(RAW_DIR / "wikipedia"))
        result = fetcher.fetch_all()
        total = sum(len(v) for v in result.values())
        log.info(f"Wikipedia: fetched {total} samples across {len(result)} dialects")
        return result
    except Exception as e:
        log.error(f"Wikipedia fetch failed: {e}")
        import traceback; traceback.print_exc()
        return {}


def run_opus() -> dict[DialectCode, list[DialectSample]]:
    """Fetch OpenSubtitles data from OPUS."""
    log.info("=" * 60)
    log.info("SOURCE: OpenSubtitles (OPUS)")
    log.info("=" * 60)
    try:
        from eigendialectos.corpus.acquisition.opus_fetcher import OPUSFetcher
        fetcher = OPUSFetcher(cache_dir=str(RAW_DIR / "opus"))
        result = fetcher.fetch_all()
        total = sum(len(v) for v in result.values())
        log.info(f"OPUS: fetched {total} samples across {len(result)} dialects")
        return result
    except Exception as e:
        log.error(f"OPUS fetch failed: {e}")
        import traceback; traceback.print_exc()
        return {}


def run_web() -> dict[DialectCode, list[DialectSample]]:
    """Scrape regional web sources."""
    log.info("=" * 60)
    log.info("SOURCE: Regional Web Sources")
    log.info("=" * 60)
    try:
        from eigendialectos.corpus.acquisition.web_scraper import WebScraper
        scraper = WebScraper(cache_dir=str(RAW_DIR / "web"))
        result = scraper.fetch_all()
        total = sum(len(v) for v in result.values())
        log.info(f"Web: fetched {total} samples across {len(result)} dialects")
        return result
    except Exception as e:
        log.error(f"Web scrape failed: {e}")
        import traceback; traceback.print_exc()
        return {}


def run_enhanced_synthetic() -> dict[DialectCode, list[DialectSample]]:
    """Generate massively expanded synthetic corpus."""
    log.info("=" * 60)
    log.info("SOURCE: Enhanced Synthetic Generator")
    log.info("=" * 60)
    try:
        from eigendialectos.corpus.acquisition.enhanced_synthetic import (
            EnhancedSyntheticGenerator,
        )
        gen = EnhancedSyntheticGenerator(seed=42)
        result = gen.generate_all()
        total = sum(len(v) for v in result.values())
        log.info(f"Enhanced synthetic: generated {total} samples across {len(result)} dialects")
        return result
    except Exception as e:
        log.error(f"Enhanced synthetic failed: {e}")
        import traceback; traceback.print_exc()
        return {}


def run_opensubtitles() -> dict[DialectCode, list[DialectSample]]:
    """Fetch subtitles from OpenSubtitles API (regional films/TV)."""
    log.info("=" * 60)
    log.info("SOURCE: OpenSubtitles API (Regional Films/TV)")
    log.info("=" * 60)
    try:
        from eigendialectos.corpus.acquisition.opensubtitles_fetcher import (
            OpenSubtitlesFetcher,
        )
        fetcher = OpenSubtitlesFetcher(data_dir=str(RAW_DIR / "opensubtitles"))
        result = fetcher.fetch_all()
        total = sum(len(v) for v in result.values())
        log.info(f"OpenSubtitles: fetched {total} samples across {len(result)} dialects")
        return result
    except Exception as e:
        log.error(f"OpenSubtitles fetch failed: {e}")
        import traceback; traceback.print_exc()
        return {}


def run_lyrics() -> dict[DialectCode, list[DialectSample]]:
    """Fetch song lyrics by regional genre."""
    log.info("=" * 60)
    log.info("SOURCE: Song Lyrics (Regional Genres)")
    log.info("=" * 60)
    try:
        from eigendialectos.corpus.acquisition.lyrics_fetcher import LyricsFetcher
        fetcher = LyricsFetcher(cache_dir=str(RAW_DIR / "lyrics"))
        result = fetcher.fetch_all()
        total = sum(len(v) for v in result.values())
        log.info(f"Lyrics: fetched {total} samples across {len(result)} dialects")
        return result
    except Exception as e:
        log.error(f"Lyrics fetch failed: {e}")
        import traceback; traceback.print_exc()
        return {}


def run_reddit() -> dict[DialectCode, list[DialectSample]]:
    """Fetch posts from regional Reddit subreddits."""
    log.info("=" * 60)
    log.info("SOURCE: Reddit (Regional Subreddits)")
    log.info("=" * 60)
    try:
        from eigendialectos.corpus.acquisition.reddit_fetcher import RedditFetcher
        fetcher = RedditFetcher(cache_dir=str(RAW_DIR / "reddit"))
        result = fetcher.fetch_all()
        total = sum(len(v) for v in result.values())
        log.info(f"Reddit: fetched {total} samples across {len(result)} dialects")
        return result
    except Exception as e:
        log.error(f"Reddit fetch failed: {e}")
        import traceback; traceback.print_exc()
        return {}


SOURCE_RUNNERS = {
    "subs": ("OpenSubtitles API (Films/TV)", run_opensubtitles),
    "lyrics": ("Song Lyrics (Regional Genres)", run_lyrics),
    "reddit": ("Reddit (Regional Subreddits)", run_reddit),
    "wiki": ("Wikipedia (MediaWiki API)", run_wikipedia),
    "synth": ("Enhanced Synthetic", run_enhanced_synthetic),
    "web": ("Regional Web Sources", run_web),
    "opus": ("OpenSubtitles (OPUS bulk)", run_opus),
}

# ======================================================================
# Merge & deduplicate
# ======================================================================

def merge_results(
    *sources: dict[DialectCode, list[DialectSample]],
) -> dict[DialectCode, list[DialectSample]]:
    """Merge multiple source results, deduplicating by text hash."""
    merged: dict[DialectCode, list[DialectSample]] = defaultdict(list)
    seen: set[int] = set()
    total_dupes = 0

    for source in sources:
        for dialect, samples in source.items():
            for sample in samples:
                text_hash = hash(sample.text.strip().lower())
                if text_hash not in seen:
                    seen.add(text_hash)
                    merged[dialect].append(sample)
                else:
                    total_dupes += 1

    if total_dupes:
        log.info(f"Removed {total_dupes} duplicate samples during merge")
    return dict(merged)


# ======================================================================
# Quality filtering
# ======================================================================

def filter_samples(
    corpus: dict[DialectCode, list[DialectSample]],
    min_length: int = 30,
    max_length: int = 2000,
    min_confidence: float = 0.1,
) -> dict[DialectCode, list[DialectSample]]:
    """Apply quality filters to the merged corpus."""
    filtered: dict[DialectCode, list[DialectSample]] = {}
    total_removed = 0

    for dialect, samples in corpus.items():
        good = []
        for s in samples:
            if len(s.text) < min_length:
                total_removed += 1
                continue
            if len(s.text) > max_length:
                total_removed += 1
                continue
            if s.confidence < min_confidence:
                total_removed += 1
                continue
            # Skip if mostly non-alphabetic
            alpha_ratio = sum(c.isalpha() for c in s.text) / max(len(s.text), 1)
            if alpha_ratio < 0.5:
                total_removed += 1
                continue
            good.append(s)
        filtered[dialect] = good

    if total_removed:
        log.info(f"Quality filter removed {total_removed} samples")
    return filtered


# ======================================================================
# Save / Load
# ======================================================================

def sample_to_dict(s: DialectSample) -> dict:
    return {
        "text": s.text,
        "dialect": s.dialect_code.value,
        "confidence": s.confidence,
        "source": s.source_id,
        "metadata": s.metadata,
    }


def dict_to_sample(d: dict) -> DialectSample:
    return DialectSample(
        text=d["text"],
        dialect_code=DialectCode(d["dialect"]),
        source_id=d.get("source", "unknown"),
        confidence=d.get("confidence", 0.5),
        metadata=d.get("metadata", {}),
    )


def save_corpus(
    corpus: dict[DialectCode, list[DialectSample]],
    output_path: Path,
) -> int:
    """Save corpus to JSONL. Returns total sample count."""
    total = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for dialect in sorted(corpus.keys(), key=lambda c: c.value):
            for sample in corpus[dialect]:
                json.dump(sample_to_dict(sample), f, ensure_ascii=False)
                f.write("\n")
                total += 1
    log.info(f"Saved {total} samples to {output_path}")
    return total


def load_corpus(input_path: Path) -> dict[DialectCode, list[DialectSample]]:
    """Load corpus from JSONL."""
    corpus: dict[DialectCode, list[DialectSample]] = defaultdict(list)
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sample = dict_to_sample(d)
            corpus[sample.dialect_code].append(sample)
    return dict(corpus)


# ======================================================================
# Statistics
# ======================================================================

def compute_stats(corpus: dict[DialectCode, list[DialectSample]]) -> dict:
    """Compute detailed corpus statistics."""
    stats = {
        "total_samples": 0,
        "total_tokens_approx": 0,
        "dialects": {},
        "sources": defaultdict(int),
    }

    for dialect in sorted(corpus.keys(), key=lambda c: c.value):
        samples = corpus[dialect]
        count = len(samples)
        stats["total_samples"] += count

        if count == 0:
            stats["dialects"][dialect.value] = {
                "name": DIALECT_NAMES.get(dialect, dialect.value),
                "count": 0,
            }
            continue

        lengths = [len(s.text) for s in samples]
        tokens = [len(s.text.split()) for s in samples]
        confidences = [s.confidence for s in samples]
        source_counts: dict[str, int] = defaultdict(int)
        for s in samples:
            source_counts[s.source_id] += 1
            stats["sources"][s.source_id] += 1

        total_tok = sum(tokens)
        stats["total_tokens_approx"] += total_tok

        stats["dialects"][dialect.value] = {
            "name": DIALECT_NAMES.get(dialect, dialect.value),
            "count": count,
            "avg_length_chars": round(sum(lengths) / count, 1),
            "avg_length_tokens": round(total_tok / count, 1),
            "min_confidence": round(min(confidences), 3),
            "max_confidence": round(max(confidences), 3),
            "avg_confidence": round(sum(confidences) / count, 3),
            "sources": dict(source_counts),
        }

    stats["sources"] = dict(stats["sources"])
    return stats


def print_stats(stats: dict) -> None:
    """Pretty-print corpus statistics."""
    print("\n" + "=" * 72)
    print("  EIGENDIALECTOS CORPUS STATISTICS")
    print("=" * 72)
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Total tokens (approx): {stats['total_tokens_approx']:,}")
    print(f"  Dialects: {len(stats['dialects'])}")
    print(f"  Sources: {', '.join(stats['sources'].keys())}")
    print("-" * 72)

    # Per-dialect table
    header = f"{'Dialect':<12} {'Name':<25} {'Count':>8} {'Avg Tok':>8} {'Conf':>8} {'Sources'}"
    print(header)
    print("-" * 72)

    for code, info in stats["dialects"].items():
        if info["count"] == 0:
            print(f"{code:<12} {info['name']:<25} {'0':>8}")
            continue
        sources = ", ".join(f"{k}:{v}" for k, v in info.get("sources", {}).items())
        print(
            f"{code:<12} {info['name']:<25} {info['count']:>8,} "
            f"{info.get('avg_length_tokens', 0):>8.1f} "
            f"{info.get('avg_confidence', 0):>8.3f} "
            f"{sources}"
        )

    print("-" * 72)

    # Source breakdown
    print("\nSource breakdown:")
    for source, count in sorted(stats["sources"].items(), key=lambda x: -x[1]):
        print(f"  {source:<30} {count:>8,} samples")
    print("=" * 72)


# ======================================================================
# Main pipeline
# ======================================================================

def build(sources: list[str]) -> dict[DialectCode, list[DialectSample]]:
    """Run the full corpus construction pipeline."""
    ensure_dirs()

    t0 = time.time()
    results: list[dict[DialectCode, list[DialectSample]]] = []

    for src_key in sources:
        if src_key not in SOURCE_RUNNERS:
            log.warning(f"Unknown source: {src_key}. Skipping.")
            continue
        name, runner = SOURCE_RUNNERS[src_key]
        log.info(f"\n>>> Running source: {name}")
        t1 = time.time()
        result = runner()
        elapsed = time.time() - t1
        total = sum(len(v) for v in result.values())
        log.info(f"<<< {name}: {total} samples in {elapsed:.1f}s")
        if result:
            results.append(result)

    if not results:
        log.error("No sources produced any data!")
        return {}

    # Merge all sources
    log.info("\n>>> Merging and deduplicating...")
    corpus = merge_results(*results)

    # Quality filter
    log.info(">>> Applying quality filters...")
    corpus = filter_samples(corpus)

    # Save
    log.info(">>> Saving corpus...")
    total = save_corpus(corpus, CORPUS_FILE)

    # Also save per-dialect files for convenience
    for dialect, samples in corpus.items():
        dialect_file = PROCESSED_DIR / f"{dialect.value}.jsonl"
        with open(dialect_file, "w", encoding="utf-8") as f:
            for s in samples:
                json.dump(sample_to_dict(s), f, ensure_ascii=False)
                f.write("\n")

    # Stats
    stats = compute_stats(corpus)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    elapsed_total = time.time() - t0
    log.info(f"\n>>> Pipeline complete in {elapsed_total:.1f}s")
    print_stats(stats)

    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the EigenDialectos corpus from multiple sources."
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="subs,lyrics,wiki,synth,web",
        help=(
            "Comma-separated list of sources: "
            "subs (OpenSubtitles API), lyrics (song lyrics), "
            "wiki (Wikipedia), synth (enhanced synthetic), "
            "web (regional websites), opus (OPUS bulk download). "
            "Default: subs,lyrics,wiki,synth,web"
        ),
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print stats for existing corpus (no fetching)",
    )
    args = parser.parse_args()

    if args.stats_only:
        if not CORPUS_FILE.exists():
            log.error(f"No corpus found at {CORPUS_FILE}. Run without --stats-only first.")
            sys.exit(1)
        corpus = load_corpus(CORPUS_FILE)
        stats = compute_stats(corpus)
        print_stats(stats)
        return

    sources = [s.strip() for s in args.sources.split(",")]
    build(sources)


if __name__ == "__main__":
    main()
