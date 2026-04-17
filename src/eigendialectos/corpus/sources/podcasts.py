"""Podcast transcription corpus source.

Loads transcribed podcast episodes from Spanish-speaking regions,
providing dialectal text from spontaneous oral speech.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Iterator

from eigendialectos.constants import DialectCode
from eigendialectos.corpus.base import CorpusSource
from eigendialectos.types import DialectSample

logger = logging.getLogger(__name__)

# Mapping from podcast region tags to dialect codes.
# Transcripts are expected to carry a region tag in their metadata.
_REGION_DIALECT_MAP: dict[str, DialectCode] = {
    "spain":        DialectCode.ES_PEN,
    "andalucia":    DialectCode.ES_AND,
    "canarias":     DialectCode.ES_CAN,
    "argentina":    DialectCode.ES_RIO,
    "uruguay":      DialectCode.ES_RIO,
    "mexico":       DialectCode.ES_MEX,
    "caribbean":    DialectCode.ES_CAR,
    "cuba":         DialectCode.ES_CAR,
    "venezuela":    DialectCode.ES_CAR,
    "puerto_rico":  DialectCode.ES_CAR,
    "colombia":     DialectCode.ES_CAR,
    "chile":        DialectCode.ES_CHI,
    "peru":         DialectCode.ES_AND_BO,
    "bolivia":      DialectCode.ES_AND_BO,
    "ecuador":      DialectCode.ES_AND_BO,
}

# Supported transcript formats
_TRANSCRIPT_EXTENSIONS = {".json", ".jsonl", ".txt", ".vtt", ".srt"}


class PodcastSource(CorpusSource):
    """Corpus source backed by podcast transcriptions.

    Transcripts can be produced by Whisper, Google Speech-to-Text, or
    similar ASR systems.  The loader supports:

    - **JSON**: a single object or array with ``"text"`` (and optional
      ``"region"``, ``"podcast"``, ``"episode"`` fields).
    - **JSONL**: one JSON object per line (same fields).
    - **Plain text**: one transcript per ``.txt`` file, with the region
      inferred from the parent directory name.
    - **VTT/SRT**: basic subtitle-format transcripts (timestamps stripped).

    Expected directory layout::

        podcasts/
          argentina/
            episode_001.json
          mexico/
            episode_042.txt
          ...
    """

    def download(self, output_dir: Path) -> Path:
        """Stub: podcast transcription requires ASR processing.

        Returns *output_dir* / ``podcasts`` as the expected artefact path.
        """
        target = output_dir / "podcasts"
        target.mkdir(parents=True, exist_ok=True)
        warnings.warn(
            "Podcast data must be collected and transcribed externally.  "
            "Use Whisper or equivalent ASR on podcast audio, then place "
            f"transcript files in subdirectories by region under {target}.  "
            "Supported formats: JSON, JSONL, TXT, VTT, SRT.",
            UserWarning,
            stacklevel=2,
        )
        return target

    def load(self, path: Path) -> Iterator[DialectSample]:
        """Yield :class:`DialectSample` instances from transcript files."""
        if not path.is_dir():
            logger.warning("Podcast data path %s does not exist", path)
            return

        for region_dir in sorted(path.iterdir()):
            if not region_dir.is_dir():
                continue
            region = region_dir.name.lower()
            dialect = _REGION_DIALECT_MAP.get(region)
            if dialect is None:
                logger.debug("Skipping unknown podcast region %s", region)
                continue

            for transcript in sorted(region_dir.iterdir()):
                if not transcript.is_file():
                    continue
                suffix = transcript.suffix.lower()
                if suffix not in _TRANSCRIPT_EXTENSIONS:
                    continue
                try:
                    if suffix == ".json":
                        yield from self._parse_json(transcript, dialect)
                    elif suffix == ".jsonl":
                        yield from self._parse_jsonl(transcript, dialect)
                    elif suffix == ".txt":
                        yield from self._parse_txt(transcript, dialect)
                    elif suffix in (".vtt", ".srt"):
                        yield from self._parse_subtitle(transcript, dialect)
                except Exception:
                    logger.warning(
                        "Failed to parse %s", transcript, exc_info=True,
                    )

    def dialect_codes(self) -> list[DialectCode]:
        return list(set(_REGION_DIALECT_MAP.values()))

    def citation(self) -> str:
        return (
            "Podcast transcriptions produced via automatic speech recognition "
            "(Whisper large-v3).  Curated from publicly available Spanish-language "
            "podcasts across multiple dialect regions."
        )

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_sample(
        text: str,
        dialect: DialectCode,
        source_tag: str,
        metadata: dict[str, object] | None = None,
    ) -> DialectSample | None:
        """Create a sample if text is long enough."""
        text = text.strip()
        if len(text) < 15:
            return None
        return DialectSample(
            text=text,
            dialect_code=dialect,
            source_id=f"podcast:{source_tag}",
            confidence=0.60,
            metadata=metadata or {},
        )

    @classmethod
    def _parse_json(
        cls,
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        """Parse a JSON file (object or array of objects with ``text``)."""
        raw = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)

        if isinstance(data, dict):
            data = [data]

        for i, obj in enumerate(data):
            text = obj.get("text", "")
            sample = cls._make_sample(
                text, dialect, f"{path.stem}:{i}",
                metadata={
                    "file": str(path),
                    "podcast": obj.get("podcast", ""),
                    "episode": obj.get("episode", ""),
                },
            )
            if sample is not None:
                yield sample

    @classmethod
    def _parse_jsonl(
        cls,
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        """Parse a JSONL file (one JSON object per line)."""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            sample = cls._make_sample(
                text, dialect, f"{path.stem}:{i}",
                metadata={"file": str(path)},
            )
            if sample is not None:
                yield sample

    @classmethod
    def _parse_txt(
        cls,
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        """Parse a plain text transcript."""
        text = path.read_text(encoding="utf-8", errors="replace")
        # Treat the entire file as one sample (or split on double-newlines
        # for multi-segment files).
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        for i, para in enumerate(paragraphs):
            sample = cls._make_sample(
                para, dialect, f"{path.stem}:{i}",
                metadata={"file": str(path)},
            )
            if sample is not None:
                yield sample

    @classmethod
    def _parse_subtitle(
        cls,
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        """Parse VTT/SRT transcript files, stripping timestamps."""
        import re

        text = path.read_text(encoding="utf-8", errors="replace")
        # Remove VTT header
        text = re.sub(r'^WEBVTT\s*\n', '', text)
        # Remove timestamps (HH:MM:SS.mmm --> HH:MM:SS.mmm or similar)
        text = re.sub(r'\d{1,2}:\d{2}[:\.][\d.,]+\s*-->\s*\d{1,2}:\d{2}[:\.][\d.,]+', '', text)
        # Remove sequence numbers (lines that are just digits)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Collect non-empty lines into paragraph groups
        lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                lines.append(stripped)

        if lines:
            full_text = " ".join(lines)
            sample = cls._make_sample(
                full_text, dialect, path.stem,
                metadata={"file": str(path)},
            )
            if sample is not None:
                yield sample
