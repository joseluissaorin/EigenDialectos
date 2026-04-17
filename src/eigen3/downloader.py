"""Bulk corpus downloader for dialectal Spanish acquisition.

Downloads from HuggingFace Hub (CulturaX, Arctic Shift Reddit, tweets),
Leipzig Corpora Collection, OPUS OpenSubtitles, and local literary works.
No live web scraping -- all sources are bulk downloads or streaming datasets.

Usage:
    from eigen3.downloader import CorpusDownloader
    dl = CorpusDownloader(cache_dir="data/raw_v4")
    raw = dl.download_all(max_per_dialect=200_000)
"""

from __future__ import annotations

import io
import json
import logging
import re
import tarfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from eigen3.constants import (
    ALL_VARIETIES,
    DIALECT_MARKERS,
    DIALECT_MARKERS_EXTENDED,
    GEO_BOUNDS_AND,
    GEO_BOUNDS_CAN,
    HF_ARCTIC_DATASET,
    HF_CULTURAX_DATASET,
    HF_LEIPZIG_DATASET,
    HF_TWEETS_DATASET,
    LEIPZIG_CORPORA,
    LITERARY_WORKS,
    OPUS_OPENSUB_URL,
    REGIONALISMS,
    SCRAPER_SUBREDDITS,
    SOURCE_CONFIDENCE,
    TLD_TO_DIALECT,
    TWITTER_COUNTRY_TO_DIALECT,
)

logger = logging.getLogger(__name__)

# Tweet cleaning patterns
_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_STRIP_RE = re.compile(r"#(\w+)")
_RT_RE = re.compile(r"^RT\s+@?\w+:?\s*", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _clean_tweet(text: str) -> str:
    """Clean a tweet: strip RT prefix, URLs, mentions, normalize spaces."""
    text = _RT_RE.sub("", text)  # RT first (before mentions are removed)
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = _HASHTAG_STRIP_RE.sub(r"\1", text)  # Keep hashtag text, remove #
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _extract_tld(url: str) -> str:
    """Extract the country-code TLD from a URL (e.g., '.ar', '.mx')."""
    try:
        host = urlparse(url).hostname or ""
        parts = host.rsplit(".", 2)
        if len(parts) >= 2:
            tld = "." + parts[-1]
            return tld
    except Exception:
        pass
    return ""


class CorpusDownloader:
    """Bulk dataset downloader for dialectal Spanish corpus acquisition.

    Downloads from HuggingFace Hub, Leipzig Corpora, Arctic Shift,
    OPUS, and literary text repositories. No live web scraping.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/raw_v4",
        max_total_gb: float = 60.0,
        hf_token: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_total_gb = max_total_gb
        self.hf_token = hf_token
        self._total_downloaded_bytes = 0

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def download_all(
        self,
        sources: list[str] | None = None,
        max_per_dialect: int = 200_000,
    ) -> dict[str, list[dict[str, Any]]]:
        """Download from all configured sources.

        Parameters
        ----------
        sources : list[str] | None
            Subset of sources to use. None = all available.
            Valid: "leipzig", "culturax", "tweets", "reddit", "opensubtitles", "literary"
        max_per_dialect : int
            Maximum documents per dialect per source.

        Returns
        -------
        dict[str, list[dict]]
            {dialect_code: [{text, source, confidence}, ...]}
        """
        all_sources = ["leipzig", "culturax", "tweets", "reddit", "opensubtitles", "literary"]
        if sources is None:
            sources = all_sources

        result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}

        source_methods = {
            "leipzig": self.download_leipzig,
            "culturax": self.download_culturax,
            "tweets": self.download_tweets,
            "reddit": self.download_reddit,
            "opensubtitles": self.download_opensubtitles,
            "literary": self.download_literary,
        }

        for source_name in sources:
            if source_name not in source_methods:
                logger.warning("Unknown source: %s, skipping", source_name)
                continue

            logger.info("=" * 60)
            logger.info("Downloading: %s", source_name)
            logger.info("=" * 60)

            try:
                source_data = source_methods[source_name](max_per_dialect=max_per_dialect)
                for dialect, samples in source_data.items():
                    result[dialect].extend(samples)
                self._report_progress(source_name, {d: len(s) for d, s in source_data.items()})
            except Exception as e:
                logger.error("Source %s failed: %s", source_name, e, exc_info=True)
                continue

        # Final summary
        total = sum(len(s) for s in result.values())
        logger.info("Download complete: %d total documents across %d varieties", total, len(result))
        for d in sorted(result.keys()):
            logger.info("  %s: %d docs", d, len(result[d]))

        return result

    # ------------------------------------------------------------------
    # Source: Leipzig Corpora Collection
    # ------------------------------------------------------------------

    def download_leipzig(
        self,
        max_per_dialect: int = 100_000,
    ) -> dict[str, list[dict[str, Any]]]:
        """Download Leipzig Corpora country-segmented files.

        Tries direct HTTP download first, falls back to HuggingFace dataset.
        """
        result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}

        # Try loading cached data first
        cached = self._load_cache("leipzig")
        if cached:
            logger.info("Leipzig: loaded from cache")
            return self._cap_per_dialect(cached, max_per_dialect)

        # Try direct download
        for corpus_id, (url, dialect) in LEIPZIG_CORPORA.items():
            try:
                samples = self._download_leipzig_archive(corpus_id, url, dialect,
                                                         max_sentences=max_per_dialect)
                result[dialect].extend(samples)
                logger.info("Leipzig %s: %d sentences -> %s", corpus_id, len(samples), dialect)
            except Exception as e:
                logger.warning("Leipzig %s failed: %s, trying next", corpus_id, e)
                continue

        # Fall back to HuggingFace if direct download yielded nothing
        if sum(len(s) for s in result.values()) == 0:
            logger.info("Direct Leipzig download failed, trying HuggingFace fallback")
            result = self._download_leipzig_hf(max_per_dialect)

        self._save_cache("leipzig", result)
        return result

    def _download_leipzig_archive(
        self, corpus_id: str, url: str, dialect: str, max_sentences: int
    ) -> list[dict[str, Any]]:
        """Download and parse a single Leipzig .tar.gz archive."""
        import requests

        cache_file = self.cache_dir / "leipzig" / f"{corpus_id}.tar.gz"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Download if not cached
        if not cache_file.exists():
            logger.info("Downloading %s ...", url)
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(cache_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            self._total_downloaded_bytes += cache_file.stat().st_size

        # Parse: look for *-sentences.txt inside the tar
        samples: list[dict[str, Any]] = []
        with tarfile.open(cache_file, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith("-sentences.txt"):
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    for line_bytes in f:
                        line = line_bytes.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        # Format: <id>\t<sentence>
                        parts = line.split("\t", 1)
                        if len(parts) < 2:
                            continue
                        text = parts[1].strip()
                        if len(text) >= 20:
                            samples.append({
                                "text": text,
                                "source": f"leipzig:{corpus_id}",
                                "confidence": SOURCE_CONFIDENCE["leipzig"],
                            })
                            if len(samples) >= max_sentences:
                                break
                    break  # Only process the first sentences file

        return samples

    def _download_leipzig_hf(
        self, max_per_dialect: int
    ) -> dict[str, list[dict[str, Any]]]:
        """Fallback: get Leipzig corpus links from HuggingFace and download."""
        result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}

        try:
            from datasets import load_dataset
            ds = load_dataset(HF_LEIPZIG_DATASET, "links", split="train")
        except Exception as e:
            logger.warning("Leipzig HF fallback failed: %s", e)
            return result

        # Filter for Spanish entries
        spanish_entries = [row for row in ds if "spa" in str(row.get("language", "")).lower()
                          or "spanish" in str(row.get("language", "")).lower()]

        for entry in spanish_entries[:20]:  # Cap to avoid too many downloads
            url = entry.get("download_url") or entry.get("url", "")
            name = entry.get("name", "") or entry.get("corpus_name", "")
            if not url or not name:
                continue

            # Try to infer dialect from corpus name
            dialect = "ES_PEN"  # Default
            name_lower = name.lower()
            for key, (_, d) in LEIPZIG_CORPORA.items():
                if key.split("_")[0] in name_lower:
                    dialect = d
                    break

            try:
                samples = self._download_leipzig_archive(
                    name, url, dialect, max_sentences=max_per_dialect
                )
                result[dialect].extend(samples)
            except Exception:
                continue

        return result

    # ------------------------------------------------------------------
    # Source: CulturaX (HuggingFace streaming)
    # ------------------------------------------------------------------

    def download_culturax(
        self,
        max_per_dialect: int = 100_000,
    ) -> dict[str, list[dict[str, Any]]]:
        """Download CulturaX Spanish web text, TLD-filtered.

        Streams the dataset to avoid downloading the full corpus.
        For .es domains: applies reclassification to rescue AND/CAN docs.
        """
        result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}

        cached = self._load_cache("culturax")
        if cached:
            logger.info("CulturaX: loaded from cache")
            return self._cap_per_dialect(cached, max_per_dialect)

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library required for CulturaX download")
            return result

        logger.info("Streaming CulturaX (es) from HuggingFace...")
        try:
            ds = load_dataset(
                HF_CULTURAX_DATASET,
                "es",
                split="train",
                streaming=True,
                token=self.hf_token,
            )
        except Exception as e:
            logger.error("Failed to load CulturaX: %s", e)
            return result

        counts: dict[str, int] = {v: 0 for v in ALL_VARIETIES}
        total_limit = max_per_dialect * len(ALL_VARIETIES)
        processed = 0

        for row in ds:
            processed += 1
            if processed % 50_000 == 0:
                logger.info("CulturaX: processed %dk rows, counts: %s",
                           processed // 1000,
                           {k: v for k, v in counts.items() if v > 0})

            text = row.get("text", "").strip()
            url = row.get("url", "")

            if len(text) < 30 or len(text) > 5000:
                continue

            # Classify by TLD
            tld = _extract_tld(url)
            dialect = TLD_TO_DIALECT.get(tld)

            if dialect is None:
                continue

            # For .es domains, try reclassification to AND/CAN
            if tld == ".es":
                reclassified = self._reclassify_es_domain(text, url)
                if reclassified != "ES_PEN":
                    dialect = reclassified
                    # Lower confidence for reclassified docs
                    conf = SOURCE_CONFIDENCE["culturax_reclassified"]
                else:
                    conf = SOURCE_CONFIDENCE["culturax_tld"]
            else:
                conf = SOURCE_CONFIDENCE["culturax_tld"]

            if counts[dialect] >= max_per_dialect:
                # Check if all dialects are full
                if all(c >= max_per_dialect for c in counts.values()):
                    break
                continue

            result[dialect].append({
                "text": text,
                "source": f"culturax:{tld}",
                "confidence": conf,
            })
            counts[dialect] += 1

            # Safety: stop after processing a lot of data
            if processed >= total_limit * 10:
                logger.info("CulturaX: hit safety limit at %d rows", processed)
                break

        self._save_cache("culturax", result)
        return result

    # ------------------------------------------------------------------
    # Source: Spanish tweets
    # ------------------------------------------------------------------

    def download_tweets(
        self,
        max_per_dialect: int = 200_000,
    ) -> dict[str, list[dict[str, Any]]]:
        """Download Spanish tweets from HuggingFace.

        Uses content-based classification: scans tweet text for dialect-specific
        regionalisms and markers. Assigns to the dialect with the highest score
        if it has ≥2 points and ≥2x the runner-up. This is necessary because
        pysentimiento/spanish-tweets has no location/geo fields.
        """
        result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}

        cached = self._load_cache("tweets")
        if cached:
            logger.info("Tweets: loaded from cache")
            return self._cap_per_dialect(cached, max_per_dialect)

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library required for tweet download")
            return result

        # Build word→dialect lookup from REGIONALISMS + DIALECT_MARKERS_EXTENDED
        word_to_dialect: dict[str, str] = {}
        for dialect_code, words in REGIONALISMS.items():
            for w in words:
                wl = w.lower()
                if wl not in word_to_dialect:
                    word_to_dialect[wl] = dialect_code
        for dialect_code, words in DIALECT_MARKERS_EXTENDED.items():
            for w in words:
                wl = w.lower()
                if wl not in word_to_dialect:
                    word_to_dialect[wl] = dialect_code

        logger.info("Streaming tweets from HuggingFace (content-based classification)...")
        logger.info("Built regionalism lookup: %d words across %d dialects",
                    len(word_to_dialect), len(ALL_VARIETIES))
        try:
            ds = load_dataset(
                HF_TWEETS_DATASET,
                split="train",
                streaming=True,
                token=self.hf_token,
            )
        except Exception as e:
            logger.error("Failed to load tweet dataset: %s", e)
            return result

        counts: dict[str, int] = {v: 0 for v in ALL_VARIETIES}
        processed = 0
        classified = 0

        for row in ds:
            processed += 1
            if processed % 500_000 == 0:
                total_found = sum(counts.values())
                logger.info("Tweets: processed %dk, classified %dk, counts: %s",
                           processed // 1000, total_found // 1000,
                           {k: v for k, v in counts.items() if v > 0})

            text = row.get("text", "").strip()
            if not text or len(text) < 25:
                continue

            # Tokenize into words (lowercase)
            words = set(re.findall(r"\b[a-záéíóúñü]+\b", text.lower()))
            if len(words) < 4:
                continue

            # Score each dialect by regionalism hits
            scores: dict[str, int] = {}
            for w in words:
                d = word_to_dialect.get(w)
                if d:
                    scores[d] = scores.get(d, 0) + 1

            if not scores:
                continue

            # Find best dialect
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_dialect, best_score = sorted_scores[0]
            runner_up_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

            # Require ≥2 hits AND clear winner (2x runner-up or sole scorer)
            if best_score < 2:
                continue
            if runner_up_score > 0 and best_score < runner_up_score * 2:
                continue

            if counts[best_dialect] >= max_per_dialect:
                if all(c >= max_per_dialect for c in counts.values()):
                    break
                continue

            # Clean the tweet
            cleaned = _clean_tweet(text)
            if len(cleaned) < 15:
                continue

            result[best_dialect].append({
                "text": cleaned,
                "source": f"twitter_content:{best_dialect}",
                "confidence": SOURCE_CONFIDENCE.get("twitter_content", 0.60),
            })
            counts[best_dialect] += 1
            classified += 1

            # Safety limit: stop after scanning 20M tweets
            if processed >= 20_000_000:
                logger.info("Tweets: reached 20M scan limit, stopping")
                break

        total = sum(counts.values())
        logger.info("Tweets: processed %dk, classified %d total", processed // 1000, total)
        for d in sorted(ALL_VARIETIES):
            if counts[d] > 0:
                logger.info("  %s: %d tweets", d, counts[d])

        self._save_cache("tweets", result)
        return result

    # ------------------------------------------------------------------
    # Source: Arctic Shift Reddit archives
    # ------------------------------------------------------------------

    def download_reddit(
        self,
        max_per_dialect: int = 50_000,
    ) -> dict[str, list[dict[str, Any]]]:
        """Download Reddit comments via Arctic Shift HuggingFace dataset.

        Streams from open-index/arctic, filtering by subreddit names
        from SCRAPER_SUBREDDITS.
        """
        result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}

        cached = self._load_cache("reddit")
        if cached:
            logger.info("Reddit: loaded from cache")
            return self._cap_per_dialect(cached, max_per_dialect)

        # Build subreddit -> dialect lookup
        sub_to_dialect: dict[str, str] = {}
        for dialect, subs in SCRAPER_SUBREDDITS.items():
            for sub in subs:
                sub_to_dialect[sub.lower()] = dialect

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library required for Reddit download")
            return result

        logger.info("Streaming Reddit data from Arctic Shift...")
        try:
            # Try loading comments subset
            ds = load_dataset(
                HF_ARCTIC_DATASET,
                split="train",
                streaming=True,
                token=self.hf_token,
            )
        except Exception as e:
            logger.error("Failed to load Arctic Shift dataset: %s", e)
            return result

        counts: dict[str, int] = {v: 0 for v in ALL_VARIETIES}
        processed = 0
        target_subs = set(sub_to_dialect.keys())

        for row in ds:
            processed += 1
            if processed % 500_000 == 0:
                logger.info("Reddit: scanned %dk rows, counts: %s",
                           processed // 1000,
                           {k: v for k, v in counts.items() if v > 0})

            # Check subreddit
            subreddit = str(row.get("subreddit", "")).lower()
            if subreddit not in target_subs:
                continue

            dialect = sub_to_dialect[subreddit]
            if counts[dialect] >= max_per_dialect:
                if all(c >= max_per_dialect for c in counts.values()):
                    break
                continue

            # Get text (comment body or submission selftext)
            text = row.get("body") or row.get("selftext") or row.get("text", "")
            text = str(text).strip()

            # Skip deleted/removed
            if text in ("[deleted]", "[removed]", "") or len(text) < 20:
                continue

            result[dialect].append({
                "text": text,
                "source": f"reddit:r/{subreddit}",
                "confidence": SOURCE_CONFIDENCE["reddit_sub"],
            })
            counts[dialect] += 1

            # Safety: Reddit dataset is massive, stop after scanning enough
            if processed >= 50_000_000:
                break

        self._save_cache("reddit", result)
        return result

    # ------------------------------------------------------------------
    # Source: OPUS OpenSubtitles (bulk download)
    # ------------------------------------------------------------------

    def download_opensubtitles(
        self,
        max_per_dialect: int = 50_000,
    ) -> dict[str, list[dict[str, Any]]]:
        """Download OpenSubtitles from OPUS bulk files.

        Downloads the monolingual Spanish subtitle corpus from OPUS mirrors.
        """
        result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}

        cached = self._load_cache("opensubtitles")
        if cached:
            logger.info("OpenSubtitles: loaded from cache")
            return self._cap_per_dialect(cached, max_per_dialect)

        import requests

        opus_url = OPUS_OPENSUB_URL
        cache_file = self.cache_dir / "opensubtitles" / "es.txt.gz"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if not cache_file.exists():
            logger.info("Downloading OpenSubtitles Spanish from OPUS...")
            try:
                resp = requests.get(opus_url, timeout=300, stream=True)
                resp.raise_for_status()
                with open(cache_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                self._total_downloaded_bytes += cache_file.stat().st_size
                logger.info("Downloaded OpenSubtitles: %.1f MB",
                           cache_file.stat().st_size / 1e6)
            except Exception as e:
                logger.error("OPUS download failed: %s", e)
                return result

        # Parse: monolingual text, one line per subtitle segment
        import gzip

        logger.info("Parsing OpenSubtitles...")
        # Distribute across all dialects using heuristic assignment:
        # Since monolingual OPUS doesn't have country metadata, we'll assign
        # based on content analysis (regionalisms) with a default round-robin
        dialect_cycle = ALL_VARIETIES.copy()
        dial_idx = 0
        samples_buffer: list[str] = []

        try:
            with gzip.open(cache_file, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    text = line.strip()
                    if len(text) < 20 or len(text) > 500:
                        continue
                    # Skip subtitle noise
                    if text.startswith("-") and len(text) < 30:
                        continue
                    samples_buffer.append(text)
                    if len(samples_buffer) >= max_per_dialect * len(ALL_VARIETIES) * 2:
                        break
        except Exception as e:
            logger.error("Failed to parse OpenSubtitles: %s", e)
            return result

        # Classify by regionalism content
        counts: dict[str, int] = {v: 0 for v in ALL_VARIETIES}
        rng = np.random.default_rng(42)
        rng.shuffle(samples_buffer)

        for text in samples_buffer:
            # Try to detect dialect from content
            text_lower = text.lower()
            words = set(text_lower.split())
            best_dialect = None
            best_hits = 0
            for d, regs in REGIONALISMS.items():
                hits = len(words & regs)
                if hits > best_hits:
                    best_hits = hits
                    best_dialect = d

            if best_dialect and best_hits >= 1:
                dialect = best_dialect
            else:
                # Round-robin assignment for neutral text
                dialect = dialect_cycle[dial_idx % len(dialect_cycle)]
                dial_idx += 1

            if counts[dialect] >= max_per_dialect:
                continue

            result[dialect].append({
                "text": text,
                "source": "opus:opensubtitles",
                "confidence": SOURCE_CONFIDENCE["opensubtitles"],
            })
            counts[dialect] += 1

            if all(c >= max_per_dialect for c in counts.values()):
                break

        self._save_cache("opensubtitles", result)
        return result

    # ------------------------------------------------------------------
    # Source: Literary works
    # ------------------------------------------------------------------

    def download_literary(
        self,
        max_per_dialect: int = 10_000,
    ) -> dict[str, list[dict[str, Any]]]:
        """Load literary works from local data/literary/ directory.

        Scans data/literary/{dialect}/ for .txt files placed by the user.
        Also attempts to download works from Project Gutenberg for known IDs.
        """
        result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}
        literary_dir = self.cache_dir.parent / "literary"

        # 1. Scan manual literary directory
        for dialect in ALL_VARIETIES:
            dialect_dir = literary_dir / dialect
            if not dialect_dir.exists():
                continue

            for txt_file in sorted(dialect_dir.glob("*.txt")):
                try:
                    text = txt_file.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                # Segment into paragraphs
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 30]
                for para in paragraphs[:max_per_dialect]:
                    if len(result[dialect]) >= max_per_dialect:
                        break
                    result[dialect].append({
                        "text": para[:2000],  # Cap length
                        "source": f"literary:{txt_file.stem}",
                        "confidence": SOURCE_CONFIDENCE["literary"],
                    })

                logger.info("Literary %s/%s: %d paragraphs",
                           dialect, txt_file.name, len(paragraphs))

        # 2. Try Project Gutenberg for works with known IDs
        for dialect, works in LITERARY_WORKS.items():
            for work in works:
                if work.get("source") != "gutenberg" or "id" not in work:
                    continue
                if len(result[dialect]) >= max_per_dialect:
                    break
                try:
                    paragraphs = self._fetch_gutenberg(work["id"], work.get("title", ""))
                    for para in paragraphs:
                        if len(result[dialect]) >= max_per_dialect:
                            break
                        result[dialect].append({
                            "text": para,
                            "source": f"gutenberg:{work['id']}",
                            "confidence": SOURCE_CONFIDENCE["literary"],
                        })
                    logger.info("Gutenberg %s (%s): %d paragraphs",
                               work["title"], dialect, len(paragraphs))
                except Exception as e:
                    logger.warning("Gutenberg %s failed: %s", work.get("title"), e)

        return result

    def _fetch_gutenberg(self, gutenberg_id: str, title: str) -> list[str]:
        """Download a Project Gutenberg text by ID and return paragraphs."""
        import requests

        cache_file = self.cache_dir / "gutenberg" / f"{gutenberg_id}.txt"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if not cache_file.exists():
            # Try multiple URL patterns
            urls = [
                f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
                f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
                f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
            ]
            downloaded = False
            for url in urls:
                try:
                    resp = requests.get(url, timeout=30)
                    if resp.status_code == 200:
                        cache_file.write_text(resp.text, encoding="utf-8")
                        downloaded = True
                        break
                except Exception:
                    continue
            if not downloaded:
                raise RuntimeError(f"Could not download Gutenberg {gutenberg_id}")

        text = cache_file.read_text(encoding="utf-8", errors="replace")

        # Strip Gutenberg header/footer
        start_markers = ["*** START OF", "***START OF"]
        end_markers = ["*** END OF", "***END OF"]
        for m in start_markers:
            idx = text.find(m)
            if idx >= 0:
                text = text[text.index("\n", idx) + 1:]
                break
        for m in end_markers:
            idx = text.find(m)
            if idx >= 0:
                text = text[:idx]
                break

        # Segment into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 50]
        return paragraphs[:5000]  # Cap

    # ------------------------------------------------------------------
    # .es domain reclassification
    # ------------------------------------------------------------------

    def _reclassify_es_domain(self, text: str, url: str = "") -> str:
        """Classify a .es-domain document into PEN, AND, or CAN.

        Three-tier classifier:
          1. High-precision keyword markers (2+ hits -> reclassify)
          2. Regionalism density scoring
          3. Phonological pattern detection (aspiration, seseo markers)

        Returns dialect code. Defaults to ES_PEN.
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        # Tier 1: Keyword markers (high precision)
        for dialect, markers in DIALECT_MARKERS_EXTENDED.items():
            hits = sum(1 for m in markers if m in text_lower)
            if hits >= 2:
                return dialect

        # Tier 2: Regionalism density
        scores: dict[str, int] = {}
        for dialect in ("ES_CAN", "ES_AND"):
            regs = REGIONALISMS.get(dialect, set())
            scores[dialect] = len(words & regs)

        # Need a clear winner with at least 2 hits
        if scores.get("ES_CAN", 0) >= 2 and scores.get("ES_CAN", 0) > scores.get("ES_AND", 0) * 2:
            return "ES_CAN"
        if scores.get("ES_AND", 0) >= 2 and scores.get("ES_AND", 0) > scores.get("ES_CAN", 0) * 2:
            return "ES_AND"

        # Tier 3: Phonological patterns
        # s-aspiration: words ending in -ao (cansao, helao, pescao)
        aspiration_count = len(re.findall(r"\b\w+ao\b", text_lower))
        # d-dropping: words ending in -á (verdá, ciudá)
        d_drop_count = len(re.findall(r"\b\w+á\b", text_lower))
        # Combined signal
        phon_score = aspiration_count + d_drop_count
        if phon_score >= 3:
            # These phonological features are shared by AND and CAN
            # but AND is more common on .es, so default to AND
            # unless there are CAN markers too
            if scores.get("ES_CAN", 0) >= 1:
                return "ES_CAN"
            return "ES_AND"

        # Check URL for regional clues
        url_lower = url.lower()
        can_url_markers = ["canarias", "canario", "tenerife", "grancanaria", "laspalmas"]
        and_url_markers = ["andalucia", "sevilla", "malaga", "cadiz", "granada", "cordoba"]
        if any(m in url_lower for m in can_url_markers):
            return "ES_CAN"
        if any(m in url_lower for m in and_url_markers):
            return "ES_AND"

        return "ES_PEN"

    # ------------------------------------------------------------------
    # Geo classification
    # ------------------------------------------------------------------

    def _geo_to_dialect(self, lat: float, lon: float) -> str | None:
        """Map geographic coordinates to a dialect code."""
        # Check Canary Islands
        lat_min, lat_max, lon_min, lon_max = GEO_BOUNDS_CAN
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return "ES_CAN"

        # Check Andalusia
        lat_min, lat_max, lon_min, lon_max = GEO_BOUNDS_AND
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return "ES_AND"

        # Check Spain (rest)
        if 36.0 <= lat <= 43.8 and -9.3 <= lon <= 3.3:
            return "ES_PEN"

        # Latin America by latitude/longitude ranges (rough)
        if -56.0 <= lat <= 32.0 and -118.0 <= lon <= -34.0:
            # Argentina/Uruguay
            if lat < -22.0 and lon > -70.0:
                if lon > -58.0:
                    return "ES_RIO"  # Buenos Aires / Uruguay
                return "ES_CHI" if lon < -66.0 else "ES_RIO"
            # Chile (long and thin)
            if lon < -66.0 and lat < -17.0:
                return "ES_CHI"
            # Bolivia/Peru/Ecuador
            if -22.0 <= lat <= 2.0 and -82.0 <= lon <= -58.0:
                return "ES_AND_BO"
            # Mexico
            if lat > 14.0 and lon < -86.0:
                return "ES_MEX"
            # Caribbean
            if lat > 10.0 and lon > -86.0:
                return "ES_CAR"

        return None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _save_cache(self, source: str, data: dict[str, list[dict[str, Any]]]) -> None:
        """Save downloaded data to cache as JSONL files."""
        cache_source_dir = self.cache_dir / source
        cache_source_dir.mkdir(parents=True, exist_ok=True)

        for dialect, samples in data.items():
            if not samples:
                continue
            cache_file = cache_source_dir / f"{dialect}.jsonl"
            with cache_file.open("w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info("Cached %s: %d total docs",
                    source, sum(len(s) for s in data.values()))

    def _load_cache(self, source: str) -> dict[str, list[dict[str, Any]]] | None:
        """Load cached data if available."""
        cache_source_dir = self.cache_dir / source
        if not cache_source_dir.exists():
            return None

        result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}
        found_any = False

        for dialect in ALL_VARIETIES:
            cache_file = cache_source_dir / f"{dialect}.jsonl"
            if cache_file.exists():
                found_any = True
                for line in cache_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            result[dialect].append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        if not found_any:
            return None

        total = sum(len(s) for s in result.values())
        if total == 0:
            return None

        logger.info("Cache hit: %s (%d docs)", source, total)
        return result

    def _cap_per_dialect(
        self,
        data: dict[str, list[dict[str, Any]]],
        max_per_dialect: int,
    ) -> dict[str, list[dict[str, Any]]]:
        """Cap samples per dialect."""
        return {d: samples[:max_per_dialect] for d, samples in data.items()}

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------

    def _report_progress(self, source: str, counts: dict[str, int]) -> None:
        """Log download progress for a source."""
        total = sum(counts.values())
        logger.info("--- %s: %d total ---", source, total)
        for d in sorted(counts.keys()):
            if counts[d] > 0:
                logger.info("  %s: %d", d, counts[d])
