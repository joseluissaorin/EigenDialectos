"""OPUS / OpenSubtitles fetcher for Spanish dialect corpus data.

Downloads Spanish monolingual subtitle text from the OPUS project
(https://opus.nlpl.eu/) and labels each line with a dialect code using
the project's rule-based :class:`DialectLabeler`.

Primary data source:
    https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/es.txt.gz

The fetcher streams the gzip-compressed file, applies quality filters,
samples up to a configurable number of lines, and persists labelled
results in JSONL format for subsequent runs.
"""

from __future__ import annotations

import gzip
import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from eigendialectos.constants import DialectCode
from eigendialectos.corpus.preprocessing.labeling import DialectLabeler
from eigendialectos.types import DialectSample

logger = logging.getLogger(__name__)

# ======================================================================
# Download URLs (in order of preference)
# ======================================================================

_OPUS_MONO_URLS: list[str] = [
    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/es.txt.gz",
    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2016/mono/es.txt.gz",
]

# ======================================================================
# Quality-filter constants
# ======================================================================

_MIN_LINE_LENGTH = 20

# Subtitle formatting artefacts to strip
_SUBTITLE_TAG_RE = re.compile(
    r"(?i)"
    r"\[(?:Music|Música|Applause|Laughter|Risas|♪|#)[^\]]*\]"
    r"|<[^>]+>"         # HTML-like tags (<i>, </b>, ...)
    r"|♪+"
    r"|#+"
    r"|\{[^}]*\}"       # SSA-style overrides {\\an8} etc.
    r"|- "              # leading dash (dialogue indicator)
)

_WHITESPACE_RE = re.compile(r"\s+")

# Lines that are mostly non-alphabetic are likely junk
_ALPHA_RATIO_THRESHOLD = 0.50


def _clean_subtitle_line(line: str) -> str:
    """Strip subtitle formatting tags and normalise whitespace."""
    line = _SUBTITLE_TAG_RE.sub("", line)
    line = _WHITESPACE_RE.sub(" ", line).strip()
    return line


def _is_quality_line(line: str) -> bool:
    """Return True if *line* passes basic quality gates."""
    if len(line) < _MIN_LINE_LENGTH:
        return False
    alpha_count = sum(1 for c in line if c.isalpha())
    if len(line) == 0 or alpha_count / len(line) < _ALPHA_RATIO_THRESHOLD:
        return False
    return True


# ======================================================================
# OPUSFetcher
# ======================================================================


class OPUSFetcher:
    """Fetch and label Spanish subtitle lines from OPUS / OpenSubtitles.

    Parameters
    ----------
    data_dir:
        Root directory for cached data (``data/raw/opus/`` by default).
    max_lines:
        Maximum number of lines to retain after quality filtering.
    low_confidence_threshold:
        Lines labelled with confidence below this value are re-assigned
        to ``ES_PEN`` as a generic fallback with confidence 0.2.
    seed:
        Random seed for reproducible line sampling.
    """

    _GZ_FILENAME = "es.txt.gz"
    _CACHE_FILENAME = "es_labeled.jsonl"

    def __init__(
        self,
        data_dir: str | Path | None = None,
        max_lines: int = 50_000,
        low_confidence_threshold: float = 0.3,
        seed: int = 42,
    ) -> None:
        if data_dir is None:
            # Resolve relative to this file's location -> project root
            project_root = Path(__file__).resolve().parents[4]
            data_dir = project_root / "data" / "raw" / "opus"
        self._data_dir = Path(data_dir)
        self._max_lines = max_lines
        self._low_confidence_threshold = low_confidence_threshold
        self._seed = seed
        self._labeler = DialectLabeler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self) -> dict[DialectCode, list[DialectSample]]:
        """Download, label, and return samples grouped by dialect.

        Returns a dict mapping each :class:`DialectCode` to a list of
        :class:`DialectSample` instances.  If a cached JSONL file exists
        it is loaded directly; otherwise the monolingual data is
        downloaded and processed from scratch.

        Returns an empty dict on network failure (logged as a warning).
        """
        cache_path = self._data_dir / self._CACHE_FILENAME

        if cache_path.exists():
            logger.info("Loading cached labelled data from %s", cache_path)
            samples = self._load_cache(cache_path)
        else:
            raw_lines = self.fetch_monolingual()
            if not raw_lines:
                return {}
            samples = self._label_lines(raw_lines)
            self._save_cache(samples, cache_path)

        result: dict[DialectCode, list[DialectSample]] = defaultdict(list)
        for sample in samples:
            result[sample.dialect_code].append(sample)
        return dict(result)

    def fetch_monolingual(self) -> list[str]:
        """Download the ``es.txt.gz`` monolingual file and return cleaned lines.

        The file is streamed and decompressed line by line.  Quality
        filtering is applied on the fly.  If the number of qualifying
        lines exceeds :attr:`max_lines`, a random sample is taken.

        Returns an empty list if all download URLs fail.
        """
        self._data_dir.mkdir(parents=True, exist_ok=True)
        gz_path = self._data_dir / self._GZ_FILENAME

        # Download if not already cached on disk
        if not gz_path.exists():
            downloaded = self._download_gz(gz_path)
            if not downloaded:
                return []

        # Stream-decompress and filter
        logger.info("Decompressing and filtering %s", gz_path)
        quality_lines: list[str] = []
        total_read = 0

        try:
            with gzip.open(gz_path, mode="rt", encoding="utf-8", errors="replace") as fh:
                for raw_line in fh:
                    total_read += 1
                    cleaned = _clean_subtitle_line(raw_line)
                    if _is_quality_line(cleaned):
                        quality_lines.append(cleaned)

                    # Progress logging every 500k lines
                    if total_read % 500_000 == 0:
                        logger.info(
                            "  ... read %d lines, %d passed quality filter",
                            total_read,
                            len(quality_lines),
                        )
        except (OSError, gzip.BadGzipFile) as exc:
            logger.warning("Error reading %s: %s", gz_path, exc)
            if not quality_lines:
                return []

        logger.info(
            "Read %d total lines; %d passed quality filter",
            total_read,
            len(quality_lines),
        )

        # Sample if we have more than max_lines
        if len(quality_lines) > self._max_lines:
            rng = random.Random(self._seed)
            quality_lines = rng.sample(quality_lines, self._max_lines)
            logger.info("Sampled down to %d lines", self._max_lines)

        return quality_lines

    # ------------------------------------------------------------------
    # Internal: dialect labelling
    # ------------------------------------------------------------------

    def _label_subtitle_line(self, text: str) -> tuple[DialectCode, float]:
        """Label a single subtitle line using the project DialectLabeler.

        Falls back to ``ES_PEN`` with confidence 0.2 when the labeler
        returns a confidence below :attr:`low_confidence_threshold`.
        """
        dialect_code, confidence = self._labeler.label(text)

        if confidence < self._low_confidence_threshold:
            return DialectCode.ES_PEN, 0.2

        return dialect_code, confidence

    def _label_lines(self, lines: list[str]) -> list[DialectSample]:
        """Label a batch of lines and return DialectSample instances."""
        samples: list[DialectSample] = []
        t0 = time.monotonic()

        for i, line in enumerate(lines):
            dialect_code, confidence = self._label_subtitle_line(line)
            sample = DialectSample(
                text=line,
                dialect_code=dialect_code,
                source_id="opensubtitles",
                confidence=confidence,
                metadata={"line_num": i},
            )
            samples.append(sample)

            if (i + 1) % 10_000 == 0:
                elapsed = time.monotonic() - t0
                logger.info(
                    "  labelled %d / %d lines (%.1f s)",
                    i + 1,
                    len(lines),
                    elapsed,
                )

        elapsed = time.monotonic() - t0
        logger.info("Labelled %d lines in %.1f s", len(samples), elapsed)

        # Summary counts
        counts: dict[str, int] = defaultdict(int)
        for s in samples:
            counts[s.dialect_code.value] += 1
        logger.info("Dialect distribution: %s", dict(counts))

        return samples

    # ------------------------------------------------------------------
    # Internal: download
    # ------------------------------------------------------------------

    def _download_gz(self, dest: Path) -> bool:
        """Try each URL in ``_OPUS_MONO_URLS`` until one succeeds.

        Returns True on success, False if all URLs fail.
        """
        for url in _OPUS_MONO_URLS:
            logger.info("Attempting download: %s", url)
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "EigenDialectos/0.1 (research)"},
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    total_size = resp.headers.get("Content-Length")
                    total_size = int(total_size) if total_size else None

                    downloaded = 0
                    chunk_size = 1024 * 256  # 256 KB chunks
                    tmp_path = dest.with_suffix(".tmp")

                    with open(tmp_path, "wb") as fh:
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            fh.write(chunk)
                            downloaded += len(chunk)

                            if total_size and downloaded % (chunk_size * 40) < chunk_size:
                                pct = downloaded / total_size * 100
                                logger.info(
                                    "  downloaded %.1f MB / %.1f MB (%.0f%%)",
                                    downloaded / 1e6,
                                    total_size / 1e6,
                                    pct,
                                )

                    # Atomic rename
                    tmp_path.rename(dest)
                    logger.info(
                        "Download complete: %s (%.1f MB)",
                        dest,
                        downloaded / 1e6,
                    )
                    return True

            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                logger.warning("Download failed for %s: %s", url, exc)
                # Clean up partial file
                tmp_path = dest.with_suffix(".tmp")
                if tmp_path.exists():
                    tmp_path.unlink()
                continue

        logger.warning(
            "All OPUS download URLs failed. "
            "You can manually download the Spanish monolingual file from "
            "https://opus.nlpl.eu/OpenSubtitles.php and place it at %s",
            dest,
        )
        return False

    # ------------------------------------------------------------------
    # Internal: cache persistence
    # ------------------------------------------------------------------

    def _save_cache(self, samples: list[DialectSample], path: Path) -> None:
        """Persist labelled samples as JSONL for fast reload."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for s in samples:
                record = {
                    "text": s.text,
                    "dialect_code": s.dialect_code.value,
                    "source_id": s.source_id,
                    "confidence": s.confidence,
                    "metadata": s.metadata,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved %d labelled samples to %s", len(samples), path)

    @staticmethod
    def _load_cache(path: Path) -> list[DialectSample]:
        """Load previously cached JSONL data."""
        samples: list[DialectSample] = []
        with open(path, encoding="utf-8") as fh:
            for line_num, raw_line in enumerate(fh, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                    sample = DialectSample(
                        text=obj["text"],
                        dialect_code=DialectCode(obj["dialect_code"]),
                        source_id=obj.get("source_id", "opensubtitles"),
                        confidence=obj["confidence"],
                        metadata=obj.get("metadata", {}),
                    )
                    samples.append(sample)
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    logger.debug("Skipping malformed cache line %d: %s", line_num, exc)
        logger.info("Loaded %d samples from cache %s", len(samples), path)
        return samples
