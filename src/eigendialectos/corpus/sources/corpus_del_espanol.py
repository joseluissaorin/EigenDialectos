"""Corpus del Español source (Mark Davies, BYU).

Interface to the *Corpus del Español* created by Mark Davies, available at
https://www.corpusdelespanol.org/.  The corpus exists in two versions:

- **Genre/Historical** (100 million words, 13th-20th century)
- **Web/Dialects** (2 billion words from web pages, ~2013-2014),
  balanced by country of origin.

This loader targets the Web/Dialects version, which provides texts
labelled by country, making it useful for dialect research.
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

# Country labels used in the Corpus del Español -> our dialect codes.
_COUNTRY_DIALECT_MAP: dict[str, DialectCode] = {
    "spain":               DialectCode.ES_PEN,
    "españa":              DialectCode.ES_PEN,
    "es":                  DialectCode.ES_PEN,
    "argentina":           DialectCode.ES_RIO,
    "ar":                  DialectCode.ES_RIO,
    "uruguay":             DialectCode.ES_RIO,
    "uy":                  DialectCode.ES_RIO,
    "mexico":              DialectCode.ES_MEX,
    "méxico":              DialectCode.ES_MEX,
    "mx":                  DialectCode.ES_MEX,
    "cuba":                DialectCode.ES_CAR,
    "cu":                  DialectCode.ES_CAR,
    "puerto rico":         DialectCode.ES_CAR,
    "pr":                  DialectCode.ES_CAR,
    "dominican republic":  DialectCode.ES_CAR,
    "república dominicana": DialectCode.ES_CAR,
    "do":                  DialectCode.ES_CAR,
    "venezuela":           DialectCode.ES_CAR,
    "ve":                  DialectCode.ES_CAR,
    "colombia":            DialectCode.ES_CAR,
    "co":                  DialectCode.ES_CAR,
    "chile":               DialectCode.ES_CHI,
    "cl":                  DialectCode.ES_CHI,
    "peru":                DialectCode.ES_AND_BO,
    "perú":                DialectCode.ES_AND_BO,
    "pe":                  DialectCode.ES_AND_BO,
    "bolivia":             DialectCode.ES_AND_BO,
    "bo":                  DialectCode.ES_AND_BO,
    "ecuador":             DialectCode.ES_AND_BO,
    "ec":                  DialectCode.ES_AND_BO,
}


class CorpusDelEspanolSource(CorpusSource):
    """Corpus source backed by Mark Davies' *Corpus del Español*.

    The :meth:`download` method is a stub -- the corpus requires
    institutional access or a personal license.  This loader reads
    pre-exported concordance or full-text files organised by country.

    Expected directory layout::

        corpus_del_espanol/
          spain/
            *.json | *.jsonl | *.csv | *.txt
          argentina/
            ...
          ...

    JSON/JSONL objects should have at least ``"text"`` and optionally
    ``"country"``, ``"genre"``, ``"title"``.

    CSV files should have at least a ``text`` column.

    TXT files are treated as one document per file, with the country
    inferred from the parent directory.
    """

    def download(self, output_dir: Path) -> Path:
        """Stub: Corpus del Español requires licensed access.

        Returns *output_dir* / ``corpus_del_espanol`` as the expected path.
        """
        target = output_dir / "corpus_del_espanol"
        target.mkdir(parents=True, exist_ok=True)
        warnings.warn(
            "Corpus del Español (Mark Davies, BYU) requires licensed access.  "
            "Export concordances or text samples from "
            "https://www.corpusdelespanol.org/ and place them in "
            f"subdirectories by country under {target}.",
            UserWarning,
            stacklevel=2,
        )
        return target

    def load(self, path: Path) -> Iterator[DialectSample]:
        """Yield :class:`DialectSample` instances from exported files."""
        if not path.is_dir():
            logger.warning("Corpus del Español path %s does not exist", path)
            return

        for country_dir in sorted(path.iterdir()):
            if not country_dir.is_dir():
                continue
            country = country_dir.name.lower().strip()
            dialect = _COUNTRY_DIALECT_MAP.get(country)
            if dialect is None:
                logger.debug(
                    "Skipping unknown country %s in Corpus del Español",
                    country,
                )
                continue

            for data_file in sorted(country_dir.iterdir()):
                if not data_file.is_file():
                    continue
                suffix = data_file.suffix.lower()
                try:
                    if suffix == ".json":
                        yield from self._parse_json(data_file, dialect)
                    elif suffix == ".jsonl":
                        yield from self._parse_jsonl(data_file, dialect)
                    elif suffix == ".csv":
                        yield from self._parse_csv(data_file, dialect)
                    elif suffix == ".txt":
                        yield from self._parse_txt(data_file, dialect)
                except Exception:
                    logger.warning(
                        "Failed to parse %s", data_file, exc_info=True,
                    )

    def dialect_codes(self) -> list[DialectCode]:
        return list(set(_COUNTRY_DIALECT_MAP.values()))

    def citation(self) -> str:
        return (
            "Davies, Mark. (2016-) Corpus del Español: Web/Dialects.  "
            "Available at https://www.corpusdelespanol.org/web-dial/."
        )

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(obj: dict) -> str:
        for key in ("text", "texto", "concordance", "concordancia", "context"):
            if key in obj and obj[key]:
                return str(obj[key]).strip()
        return ""

    @classmethod
    def _make_sample(
        cls,
        text: str,
        dialect: DialectCode,
        source_tag: str,
        metadata: dict[str, object] | None = None,
    ) -> DialectSample | None:
        text = text.strip()
        if len(text) < 10:
            return None
        return DialectSample(
            text=text,
            dialect_code=dialect,
            source_id=f"corpus_del_espanol:{source_tag}",
            confidence=0.80,
            metadata=metadata or {},
        )

    @classmethod
    def _parse_json(
        cls,
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("results", [data])
        for i, obj in enumerate(data):
            if not isinstance(obj, dict):
                continue
            text = cls._extract_text(obj)
            sample = cls._make_sample(
                text, dialect, f"{path.stem}:{i}",
                metadata={
                    "file": str(path),
                    "genre": obj.get("genre", obj.get("género", "")),
                    "title": obj.get("title", obj.get("titulo", "")),
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
            text = cls._extract_text(obj)
            sample = cls._make_sample(
                text, dialect, f"{path.stem}:{i}",
                metadata={"file": str(path)},
            )
            if sample is not None:
                yield sample

    @classmethod
    def _parse_csv(
        cls,
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        text_data = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(text_data.splitlines())
        for i, row in enumerate(reader):
            text = row.get("text", row.get("texto", "")).strip()
            sample = cls._make_sample(
                text, dialect, f"{path.stem}:{i}",
                metadata={
                    "file": str(path),
                    "genre": row.get("genre", row.get("género", "")),
                },
            )
            if sample is not None:
                yield sample

    @classmethod
    def _parse_txt(
        cls,
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        sample = cls._make_sample(
            text, dialect, path.stem,
            metadata={"file": str(path)},
        )
        if sample is not None:
            yield sample
