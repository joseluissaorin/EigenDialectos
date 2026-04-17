"""OpenSubtitles corpus source.

Provides an interface to download and load Spanish-language subtitle
files, segmented by country of origin to approximate dialect regions.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Iterator

from eigendialectos.constants import DialectCode
from eigendialectos.corpus.base import CorpusSource
from eigendialectos.types import DialectSample

logger = logging.getLogger(__name__)

# Mapping from OpenSubtitles country codes to our dialect codes
_COUNTRY_DIALECT_MAP: dict[str, DialectCode] = {
    "es": DialectCode.ES_PEN,
    "ar": DialectCode.ES_RIO,
    "uy": DialectCode.ES_RIO,
    "mx": DialectCode.ES_MEX,
    "cl": DialectCode.ES_CHI,
    "cu": DialectCode.ES_CAR,
    "ve": DialectCode.ES_CAR,
    "pr": DialectCode.ES_CAR,
    "do": DialectCode.ES_CAR,
    "pe": DialectCode.ES_AND_BO,
    "bo": DialectCode.ES_AND_BO,
    "ec": DialectCode.ES_AND_BO,
    "co": DialectCode.ES_CAR,
}


class SubtitlesSource(CorpusSource):
    """Corpus source backed by OpenSubtitles data.

    This source expects pre-downloaded SRT files organised in directories
    by country code.  The :meth:`download` method provides a stub that
    warns about manual download requirements (OpenSubtitles requires
    authentication).
    """

    def download(self, output_dir: Path) -> Path:
        """Stub: OpenSubtitles data requires manual download.

        Returns *output_dir* / ``subtitles`` as the expected data path.
        """
        target = output_dir / "subtitles"
        target.mkdir(parents=True, exist_ok=True)
        warnings.warn(
            "OpenSubtitles data must be downloaded manually from "
            "https://opus.nlpl.eu/OpenSubtitles.php -- "
            "place SRT files in subdirectories named by country code "
            f"(e.g. {target / 'ar'}, {target / 'mx'}).",
            UserWarning,
            stacklevel=2,
        )
        return target

    def load(self, path: Path) -> Iterator[DialectSample]:
        """Load SRT subtitle files from *path*.

        Expects subdirectories named by 2-letter country code
        (``ar/``, ``mx/``, ``es/``, etc.) containing ``.srt`` files.
        """
        if not path.is_dir():
            logger.warning("Subtitles path %s does not exist", path)
            return

        for country_dir in sorted(path.iterdir()):
            if not country_dir.is_dir():
                continue
            country = country_dir.name.lower()
            dialect = _COUNTRY_DIALECT_MAP.get(country)
            if dialect is None:
                logger.debug("Skipping unknown country code %s", country)
                continue

            for srt_file in sorted(country_dir.glob("*.srt")):
                try:
                    yield from self._parse_srt(srt_file, dialect)
                except Exception:
                    logger.warning(
                        "Failed to parse %s", srt_file, exc_info=True,
                    )

    def dialect_codes(self) -> list[DialectCode]:
        return list(set(_COUNTRY_DIALECT_MAP.values()))

    def citation(self) -> str:
        return (
            "P. Lison and J. Tiedemann, 2016. OpenSubtitles2016: "
            "Extracting Large Parallel Corpora from Movie and TV Subtitles. "
            "LREC 2016."
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_srt(
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        """Parse a single SRT file and yield samples."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        lines: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            # Skip sequence numbers, timestamps, and blank lines
            if not line:
                if lines:
                    full = " ".join(lines)
                    # Strip HTML-like tags
                    import re
                    full = re.sub(r'<[^>]+>', '', full)
                    full = full.strip()
                    if len(full) > 5:
                        yield DialectSample(
                            text=full,
                            dialect_code=dialect,
                            source_id=f"subtitles:{path.stem}",
                            confidence=0.6,
                            metadata={"file": str(path)},
                        )
                    lines = []
                continue
            if line.isdigit():
                continue
            if "-->" in line:
                continue
            lines.append(line)

        # Final block
        if lines:
            full = " ".join(lines)
            import re
            full = re.sub(r'<[^>]+>', '', full).strip()
            if len(full) > 5:
                yield DialectSample(
                    text=full,
                    dialect_code=dialect,
                    source_id=f"subtitles:{path.stem}",
                    confidence=0.6,
                    metadata={"file": str(path)},
                )
