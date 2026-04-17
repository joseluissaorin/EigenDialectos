"""Wikipedia regional content extractor.

Extracts text from Spanish Wikipedia articles that are strongly
associated with specific dialect regions.  The approach uses:

1. Pre-curated lists of region-specific article titles (cities,
   cultural topics, regional vocabulary articles).
2. Downloaded Wikipedia dump excerpts or API query results.

Since Wikipedia text is formal/encyclopaedic, it is less dialectally
marked than spoken sources, but it provides:
- Regional vocabulary in context.
- Reliable, clean, long-form text.
- Good coverage of cultural and geographic topics per region.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Iterator

from eigendialectos.constants import DialectCode
from eigendialectos.corpus.base import CorpusSource
from eigendialectos.types import DialectSample

logger = logging.getLogger(__name__)

# Pre-curated topic seeds per dialect region.
# These guide which Wikipedia articles to extract.
REGION_ARTICLE_SEEDS: dict[DialectCode, list[str]] = {
    DialectCode.ES_PEN: [
        "Idioma español en España", "Castellano", "Dialecto madrileño",
        "Español de España", "Madrid", "Castilla",
    ],
    DialectCode.ES_AND: [
        "Dialecto andaluz", "Andalucía", "Flamenco", "Sevilla", "Málaga",
        "Ceceo", "Seseo en Andalucía",
    ],
    DialectCode.ES_CAN: [
        "Español canario", "Canarias", "Guanche", "Gofio",
        "Las Palmas de Gran Canaria", "Santa Cruz de Tenerife",
    ],
    DialectCode.ES_RIO: [
        "Español rioplatense", "Lunfardo", "Voseo", "Buenos Aires",
        "Tango", "Montevideo", "Mate (infusión)",
    ],
    DialectCode.ES_MEX: [
        "Español mexicano", "Nahuatlismos", "Ciudad de México",
        "Mexicanismo", "Gastronomía de México",
    ],
    DialectCode.ES_CAR: [
        "Español caribeño", "La Habana", "San Juan (Puerto Rico)",
        "Español venezolano", "Español cubano", "Salsa (género musical)",
    ],
    DialectCode.ES_CHI: [
        "Español chileno", "Chilenismo", "Santiago de Chile",
        "Mapudungún", "Jerga chilena",
    ],
    DialectCode.ES_AND_BO: [
        "Español andino", "Quechuismo", "Lima", "La Paz",
        "Quito", "Español peruano", "Español boliviano",
    ],
}

# Regex to strip MediaWiki markup residue
_WIKI_MARKUP_RE = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]')
_WIKI_REF_RE = re.compile(r'<ref[^>]*>.*?</ref>|<ref[^/]*/>', re.DOTALL)
_WIKI_TAG_RE = re.compile(r'<[^>]+>')
_WIKI_TEMPLATE_RE = re.compile(r'\{\{[^}]*\}\}')
_WIKI_HEADING_RE = re.compile(r'^={2,}.*?={2,}\s*$', re.MULTILINE)

# Country directory name -> dialect code
_COUNTRY_DIALECT_MAP: dict[str, DialectCode] = {
    "spain":        DialectCode.ES_PEN,
    "espana":       DialectCode.ES_PEN,
    "andalucia":    DialectCode.ES_AND,
    "canarias":     DialectCode.ES_CAN,
    "argentina":    DialectCode.ES_RIO,
    "uruguay":      DialectCode.ES_RIO,
    "mexico":       DialectCode.ES_MEX,
    "cuba":         DialectCode.ES_CAR,
    "venezuela":    DialectCode.ES_CAR,
    "puerto_rico":  DialectCode.ES_CAR,
    "colombia":     DialectCode.ES_CAR,
    "chile":        DialectCode.ES_CHI,
    "peru":         DialectCode.ES_AND_BO,
    "bolivia":      DialectCode.ES_AND_BO,
    "ecuador":      DialectCode.ES_AND_BO,
}


def _clean_wiki_text(text: str) -> str:
    """Strip common MediaWiki markup from extracted text."""
    text = _WIKI_REF_RE.sub('', text)
    text = _WIKI_TEMPLATE_RE.sub('', text)
    text = _WIKI_MARKUP_RE.sub(r'\1', text)
    text = _WIKI_TAG_RE.sub('', text)
    text = _WIKI_HEADING_RE.sub('', text)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


class WikipediaSource(CorpusSource):
    """Corpus source backed by Spanish Wikipedia articles.

    The :meth:`download` method is a stub -- users should download
    relevant article texts via the MediaWiki API or from Wikipedia dumps.
    This loader reads pre-extracted JSON, JSONL, or TXT files organised
    by region.

    Expected directory layout::

        wikipedia/
          spain/
            article_001.json
          argentina/
            article_042.txt
          ...

    JSON objects should have ``"text"`` (or ``"extract"``) and optionally
    ``"title"``, ``"pageid"``.
    """

    def download(self, output_dir: Path) -> Path:
        """Stub: Wikipedia data should be fetched via the MediaWiki API.

        Returns *output_dir* / ``wikipedia`` as the expected artefact path.
        """
        target = output_dir / "wikipedia"
        target.mkdir(parents=True, exist_ok=True)
        warnings.warn(
            "Wikipedia data should be extracted using the MediaWiki API "
            "(action=query&prop=extracts) or from dumps at "
            "https://dumps.wikimedia.org/eswiki/.  "
            f"Place extracted articles in subdirectories by region under {target}.",
            UserWarning,
            stacklevel=2,
        )
        return target

    def load(self, path: Path) -> Iterator[DialectSample]:
        """Yield :class:`DialectSample` instances from Wikipedia extracts."""
        if not path.is_dir():
            logger.warning("Wikipedia data path %s does not exist", path)
            return

        for region_dir in sorted(path.iterdir()):
            if not region_dir.is_dir():
                continue
            region = region_dir.name.lower().replace(" ", "_")
            dialect = _COUNTRY_DIALECT_MAP.get(region)
            if dialect is None:
                logger.debug("Skipping unknown Wikipedia region %s", region)
                continue

            for data_file in sorted(region_dir.iterdir()):
                if not data_file.is_file():
                    continue
                suffix = data_file.suffix.lower()
                try:
                    if suffix == ".json":
                        yield from self._parse_json(data_file, dialect)
                    elif suffix == ".jsonl":
                        yield from self._parse_jsonl(data_file, dialect)
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
            "Wikipedia en español. Wikimedia Foundation.  "
            "https://es.wikipedia.org/  "
            "Content available under CC BY-SA 3.0."
        )

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    @classmethod
    def _make_sample(
        cls,
        text: str,
        dialect: DialectCode,
        source_tag: str,
        metadata: dict[str, object] | None = None,
    ) -> DialectSample | None:
        text = _clean_wiki_text(text)
        if len(text) < 20:
            return None
        return DialectSample(
            text=text,
            dialect_code=dialect,
            source_id=f"wikipedia:{source_tag}",
            confidence=0.50,  # lower confidence: encyclopaedic, less dialectal
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

        # Handle single article or list of articles
        if isinstance(data, dict):
            # Could be MediaWiki API response or single article
            pages = data.get("query", {}).get("pages", {})
            if pages:
                data = list(pages.values())
            else:
                data = [data]

        for i, obj in enumerate(data):
            if not isinstance(obj, dict):
                continue
            text = obj.get("text", obj.get("extract", ""))
            title = obj.get("title", obj.get("titulo", path.stem))
            sample = cls._make_sample(
                text, dialect, f"{title}",
                metadata={
                    "file": str(path),
                    "title": title,
                    "pageid": obj.get("pageid", ""),
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
            text = obj.get("text", obj.get("extract", ""))
            title = obj.get("title", f"{path.stem}:{i}")
            sample = cls._make_sample(
                text, dialect, title,
                metadata={"file": str(path), "title": title},
            )
            if sample is not None:
                yield sample

    @classmethod
    def _parse_txt(
        cls,
        path: Path,
        dialect: DialectCode,
    ) -> Iterator[DialectSample]:
        text = path.read_text(encoding="utf-8", errors="replace")
        sample = cls._make_sample(
            text, dialect, path.stem,
            metadata={"file": str(path)},
        )
        if sample is not None:
            yield sample
