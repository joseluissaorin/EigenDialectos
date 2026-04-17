"""CORPES XXI corpus source.

Interface to the *Corpus del Español del Siglo XXI* maintained by the
Real Academia Española (RAE).  CORPES XXI contains over 300 million
word-forms from texts produced between 2001 and the present, balanced
across Spanish-speaking regions.

Since the corpus is not freely downloadable in bulk, this loader works
with locally cached query results exported from the CORPES XXI web
interface (https://apps.rae.es/CORPES/).
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

# CORPES XXI uses zone labels; map them to our dialect codes.
_ZONE_DIALECT_MAP: dict[str, DialectCode] = {
    # Spain
    "espana":           DialectCode.ES_PEN,
    "spain":            DialectCode.ES_PEN,
    "andalucia":        DialectCode.ES_AND,
    "canarias":         DialectCode.ES_CAN,
    # Latin America
    "rio_de_la_plata":  DialectCode.ES_RIO,
    "rioplatense":      DialectCode.ES_RIO,
    "argentina":        DialectCode.ES_RIO,
    "uruguay":          DialectCode.ES_RIO,
    "mexico":           DialectCode.ES_MEX,
    "centroamerica":    DialectCode.ES_MEX,  # closest approximation
    "caribe":           DialectCode.ES_CAR,
    "antillas":         DialectCode.ES_CAR,
    "cuba":             DialectCode.ES_CAR,
    "venezuela":        DialectCode.ES_CAR,
    "colombia":         DialectCode.ES_CAR,
    "chile":            DialectCode.ES_CHI,
    "andina":           DialectCode.ES_AND_BO,
    "peru":             DialectCode.ES_AND_BO,
    "bolivia":          DialectCode.ES_AND_BO,
    "ecuador":          DialectCode.ES_AND_BO,
}


class CorpesXXISource(CorpusSource):
    """Corpus source backed by CORPES XXI (RAE) exported data.

    The loader expects JSON or JSONL files containing concordance results
    exported from the CORPES XXI web interface, organised by zone.

    Expected directory layout::

        corpes_xxi/
          espana/
            results_001.json
          argentina/
            results_001.jsonl
          ...

    Each JSON object should have at least:
    - ``"text"`` or ``"concordance"``: the text fragment.
    - Optionally ``"zone"``, ``"country"``, ``"title"``, ``"author"``,
      ``"year"``.
    """

    def download(self, output_dir: Path) -> Path:
        """Stub: CORPES XXI requires manual query and export.

        Returns *output_dir* / ``corpes_xxi`` as the expected path.
        """
        target = output_dir / "corpes_xxi"
        target.mkdir(parents=True, exist_ok=True)
        warnings.warn(
            "CORPES XXI data must be exported manually from "
            "https://apps.rae.es/CORPES/ -- query for concordances by zone "
            "and export as JSON.  Place exported files in subdirectories "
            f"named by zone/country under {target}.",
            UserWarning,
            stacklevel=2,
        )
        return target

    def load(self, path: Path) -> Iterator[DialectSample]:
        """Yield :class:`DialectSample` instances from CORPES XXI exports."""
        if not path.is_dir():
            logger.warning("CORPES XXI path %s does not exist", path)
            return

        for zone_dir in sorted(path.iterdir()):
            if not zone_dir.is_dir():
                continue
            zone = zone_dir.name.lower().replace(" ", "_")
            dialect = _ZONE_DIALECT_MAP.get(zone)
            if dialect is None:
                logger.debug("Skipping unknown CORPES zone %s", zone)
                continue

            for data_file in sorted(zone_dir.iterdir()):
                if not data_file.is_file():
                    continue
                suffix = data_file.suffix.lower()
                try:
                    if suffix == ".json":
                        yield from self._parse_json(data_file, dialect)
                    elif suffix == ".jsonl":
                        yield from self._parse_jsonl(data_file, dialect)
                except Exception:
                    logger.warning(
                        "Failed to parse %s", data_file, exc_info=True,
                    )

    def dialect_codes(self) -> list[DialectCode]:
        return list(set(_ZONE_DIALECT_MAP.values()))

    def citation(self) -> str:
        return (
            "Real Academia Española. CORPES XXI: Corpus del Español del "
            "Siglo XXI [en línea]. https://apps.rae.es/CORPES/"
        )

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(obj: dict) -> str:
        """Extract text from a CORPES concordance object."""
        # Try common field names
        for key in ("text", "concordance", "concordancia", "texto"):
            if key in obj and obj[key]:
                return str(obj[key]).strip()
        return ""

    @classmethod
    def _make_sample(
        cls,
        obj: dict,
        dialect: DialectCode,
        source_tag: str,
    ) -> DialectSample | None:
        """Build a sample from a parsed JSON object."""
        text = cls._extract_text(obj)
        if len(text) < 10:
            return None

        return DialectSample(
            text=text,
            dialect_code=dialect,
            source_id=f"corpes_xxi:{source_tag}",
            confidence=0.85,
            metadata={
                "title": obj.get("title", obj.get("titulo", "")),
                "author": obj.get("author", obj.get("autor", "")),
                "year": obj.get("year", obj.get("año", "")),
                "zone": obj.get("zone", obj.get("zona", "")),
                "country": obj.get("country", obj.get("pais", "")),
            },
        )

    @classmethod
    def _parse_json(
        cls,
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        """Parse a JSON file (object or array)."""
        raw = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
        if isinstance(data, dict):
            # Could be {"results": [...]} or a single concordance
            if "results" in data:
                data = data["results"]
            else:
                data = [data]

        for i, obj in enumerate(data):
            if not isinstance(obj, dict):
                continue
            sample = cls._make_sample(obj, dialect, f"{path.stem}:{i}")
            if sample is not None:
                yield sample

    @classmethod
    def _parse_jsonl(
        cls,
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        """Parse a JSONL file."""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            sample = cls._make_sample(obj, dialect, f"{path.stem}:{i}")
            if sample is not None:
                yield sample
