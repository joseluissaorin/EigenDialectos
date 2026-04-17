"""Geolocated tweet corpus source.

Loads tweets collected via the Twitter/X API, filtered by geographic
bounding boxes corresponding to Spanish-speaking dialect regions.
"""

from __future__ import annotations

import csv
import json
import logging
import warnings
from pathlib import Path
from typing import Iterator

from eigendialectos.constants import DialectCode
from eigendialectos.corpus.base import CorpusSource
from eigendialectos.types import DialectSample

logger = logging.getLogger(__name__)

# Bounding-box centroids used during data collection (for reference).
# Actual filtering uses the full bbox; this maps region labels to dialects.
_REGION_DIALECT_MAP: dict[str, DialectCode] = {
    "spain_central":   DialectCode.ES_PEN,
    "spain_andalucia": DialectCode.ES_AND,
    "spain_canarias":  DialectCode.ES_CAN,
    "argentina":       DialectCode.ES_RIO,
    "uruguay":         DialectCode.ES_RIO,
    "mexico":          DialectCode.ES_MEX,
    "cuba":            DialectCode.ES_CAR,
    "puerto_rico":     DialectCode.ES_CAR,
    "dominican_rep":   DialectCode.ES_CAR,
    "venezuela":       DialectCode.ES_CAR,
    "colombia":        DialectCode.ES_CAR,
    "chile":           DialectCode.ES_CHI,
    "peru":            DialectCode.ES_AND_BO,
    "bolivia":         DialectCode.ES_AND_BO,
    "ecuador":         DialectCode.ES_AND_BO,
}


class TwitterSource(CorpusSource):
    """Corpus source backed by geolocated tweets.

    The :meth:`download` method is a stub -- Twitter/X API access requires
    authentication tokens that must be configured externally.  This loader
    reads pre-collected JSONL or CSV files organised by region.

    Expected directory layout::

        tweets/
          argentina/
            *.jsonl
          mexico/
            *.jsonl
          ...
    """

    def download(self, output_dir: Path) -> Path:
        """Stub: Twitter data requires API credentials and manual collection.

        Returns *output_dir* / ``tweets`` as the expected artefact path.
        """
        target = output_dir / "tweets"
        target.mkdir(parents=True, exist_ok=True)
        warnings.warn(
            "Twitter/X data must be collected manually using the Academic "
            "Research API v2 with geographic bounding-box filters.  "
            f"Place JSONL files in subdirectories by region under {target}.",
            UserWarning,
            stacklevel=2,
        )
        return target

    def load(self, path: Path) -> Iterator[DialectSample]:
        """Yield :class:`DialectSample` instances from tweet files.

        Supports JSONL (one JSON object per line with at least a ``"text"``
        field) and CSV (with a ``text`` column).
        """
        if not path.is_dir():
            logger.warning("Twitter data path %s does not exist", path)
            return

        for region_dir in sorted(path.iterdir()):
            if not region_dir.is_dir():
                continue
            region = region_dir.name.lower()
            dialect = _REGION_DIALECT_MAP.get(region)
            if dialect is None:
                logger.debug("Skipping unknown region %s", region)
                continue

            # Process JSONL files
            for jsonl_file in sorted(region_dir.glob("*.jsonl")):
                try:
                    yield from self._parse_jsonl(jsonl_file, dialect)
                except Exception:
                    logger.warning(
                        "Failed to parse %s", jsonl_file, exc_info=True,
                    )

            # Process CSV files
            for csv_file in sorted(region_dir.glob("*.csv")):
                try:
                    yield from self._parse_csv(csv_file, dialect)
                except Exception:
                    logger.warning(
                        "Failed to parse %s", csv_file, exc_info=True,
                    )

    def dialect_codes(self) -> list[DialectCode]:
        return list(set(_REGION_DIALECT_MAP.values()))

    def citation(self) -> str:
        return (
            "Twitter/X Academic Research API v2.  "
            "Tweets collected with geographic bounding-box filters for "
            "Spanish-speaking regions.  Subject to Twitter Terms of Service."
        )

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_jsonl(
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        """Parse a JSONL file of tweet objects."""
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSON on line %d of %s", i, path)
                continue

            text = obj.get("text", "").strip()
            if len(text) < 10:
                continue

            tweet_id = obj.get("id", str(i))
            yield DialectSample(
                text=text,
                dialect_code=dialect,
                source_id=f"twitter:{tweet_id}",
                confidence=0.55,
                metadata={
                    "file": str(path),
                    "tweet_id": tweet_id,
                    "created_at": obj.get("created_at", ""),
                    "geo": obj.get("geo", {}),
                },
            )

    @staticmethod
    def _parse_csv(
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        """Parse a CSV file with at least a ``text`` column."""
        try:
            text_data = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        reader = csv.DictReader(text_data.splitlines())
        for i, row in enumerate(reader):
            text = row.get("text", "").strip()
            if len(text) < 10:
                continue

            tweet_id = row.get("id", str(i))
            yield DialectSample(
                text=text,
                dialect_code=dialect,
                source_id=f"twitter:{tweet_id}",
                confidence=0.55,
                metadata={
                    "file": str(path),
                    "tweet_id": tweet_id,
                },
            )
