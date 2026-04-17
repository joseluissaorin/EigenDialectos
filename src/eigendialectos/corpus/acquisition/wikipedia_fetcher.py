"""Fetch real Spanish Wikipedia articles as dialect corpus data.

Each dialect variety is associated with two categories of articles:

- **Category A** (``dialect_article``): Articles *about* the dialect itself,
  which contain authentic examples of dialectal features.  Assigned a higher
  confidence of 0.7.
- **Category B** (``regional_article``): Articles about the region's culture,
  geography, cuisine, etc.  Written in standard Spanish but topically
  associated with the dialect's region.  Assigned a lower confidence of 0.4.

Articles are fetched from the Spanish Wikipedia MediaWiki API, cached locally
as JSON files under ``data/raw/wikipedia/``, and segmented into paragraph-level
:class:`~eigendialectos.types.DialectSample` instances.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests

from eigendialectos.constants import DialectCode
from eigendialectos.types import DialectSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_ENDPOINT = "https://es.wikipedia.org/w/api.php"
_USER_AGENT = "EigenDialectos/0.1 (research; dialect-corpus-builder)"
_REQUEST_DELAY_SECONDS = 1.0
_MIN_PARAGRAPH_LENGTH = 50

_CONFIDENCE_DIALECT_ARTICLE = 0.7
_CONFIDENCE_REGIONAL_ARTICLE = 0.4

# ---------------------------------------------------------------------------
# Article lists per dialect
# ---------------------------------------------------------------------------
# Each entry is a tuple (title, category) where category is one of
# "dialect_article" or "regional_article".

ArticleSpec = tuple[str, str]  # (title, category)

_DIALECT_ARTICLES: dict[DialectCode, list[ArticleSpec]] = {
    # ---- ES_PEN: Peninsular Standard ----
    DialectCode.ES_PEN: [
        # Category A: dialect articles
        ("Español peninsular", "dialect_article"),
        ("Castellano septentrional", "dialect_article"),
        ("Leísmo", "dialect_article"),
        ("Laísmo", "dialect_article"),
        ("Loísmo", "dialect_article"),
        ("Distinción (lingüística)", "dialect_article"),
        ("Dialecto castellano", "dialect_article"),
        ("Español de España", "dialect_article"),
        ("Dialecto leonés", "dialect_article"),
        ("Dialecto aragonés", "dialect_article"),
        # Category B: regional articles
        ("Madrid", "regional_article"),
        ("Gastronomía de Castilla", "regional_article"),
        ("Plaza Mayor de Madrid", "regional_article"),
        ("Castilla y León", "regional_article"),
        ("Castilla-La Mancha", "regional_article"),
        ("Salamanca", "regional_article"),
        ("Valladolid", "regional_article"),
    ],
    # ---- ES_AND: Andalusian ----
    DialectCode.ES_AND: [
        # Category A
        ("Dialecto andaluz", "dialect_article"),
        ("Seseo", "dialect_article"),
        ("Ceceo", "dialect_article"),
        ("Habla de Sevilla", "dialect_article"),
        ("Yeísmo", "dialect_article"),
        ("Aspiración de la /s/ en español", "dialect_article"),
        ("Habla de Cádiz", "dialect_article"),
        ("Habla de Málaga", "dialect_article"),
        ("Modalidades lingüísticas de Andalucía", "dialect_article"),
        ("Habla de Granada", "dialect_article"),
        # Category B
        ("Andalucía", "regional_article"),
        ("Flamenco", "regional_article"),
        ("Feria de Abril", "regional_article"),
        ("Gazpacho", "regional_article"),
        ("Sevilla", "regional_article"),
        ("Alhambra", "regional_article"),
        ("Semana Santa en Andalucía", "regional_article"),
    ],
    # ---- ES_CAN: Canarian ----
    DialectCode.ES_CAN: [
        # Category A
        ("Español canario", "dialect_article"),
        ("Silbo gomero", "dialect_article"),
        ("Isleño (español de Luisiana)", "dialect_article"),
        ("Guanchismos", "dialect_article"),
        ("Habla de Las Palmas de Gran Canaria", "dialect_article"),
        ("Dialecto canario", "dialect_article"),
        ("Seseo", "dialect_article"),
        ("Aspiración de la /s/ en español", "dialect_article"),
        ("Español de América", "dialect_article"),
        ("Español atlántico", "dialect_article"),
        # Category B
        ("Islas Canarias", "regional_article"),
        ("Papas arrugadas", "regional_article"),
        ("Carnaval de Santa Cruz de Tenerife", "regional_article"),
        ("Tenerife", "regional_article"),
        ("Gran Canaria", "regional_article"),
        ("Gastronomía de Canarias", "regional_article"),
        ("Gofio", "regional_article"),
    ],
    # ---- ES_RIO: Rioplatense ----
    DialectCode.ES_RIO: [
        # Category A
        ("Español rioplatense", "dialect_article"),
        ("Lunfardo", "dialect_article"),
        ("Voseo", "dialect_article"),
        ("Cocoliche", "dialect_article"),
        ("Rehilamiento", "dialect_article"),
        ("Sheísmo", "dialect_article"),
        ("Español de Argentina", "dialect_article"),
        ("Español uruguayo", "dialect_article"),
        ("Yeísmo", "dialect_article"),
        ("Che (interjección)", "dialect_article"),
        # Category B
        ("Buenos Aires", "regional_article"),
        ("Tango", "regional_article"),
        ("Asado", "regional_article"),
        ("Mate (infusión)", "regional_article"),
        ("Montevideo", "regional_article"),
        ("Gaucho", "regional_article"),
        ("Río de la Plata", "regional_article"),
    ],
    # ---- ES_MEX: Mexican ----
    DialectCode.ES_MEX: [
        # Category A
        ("Español mexicano", "dialect_article"),
        ("Nahuatlismos", "dialect_article"),
        ("Español yucateco", "dialect_article"),
        ("Español norteño de México", "dialect_article"),
        ("Spanglish", "dialect_article"),
        ("Caló (México)", "dialect_article"),
        ("Albur", "dialect_article"),
        ("Mexicanismo", "dialect_article"),
        ("Español de América", "dialect_article"),
        ("Diminutivo", "dialect_article"),
        # Category B
        ("México", "regional_article"),
        ("Gastronomía de México", "regional_article"),
        ("Día de Muertos", "regional_article"),
        ("Mariachi", "regional_article"),
        ("Ciudad de México", "regional_article"),
        ("Mole (salsa)", "regional_article"),
        ("Tacos", "regional_article"),
    ],
    # ---- ES_CAR: Caribbean ----
    DialectCode.ES_CAR: [
        # Category A
        ("Español caribeño", "dialect_article"),
        ("Español cubano", "dialect_article"),
        ("Español dominicano", "dialect_article"),
        ("Español puertorriqueño", "dialect_article"),
        ("Español venezolano", "dialect_article"),
        ("Lambdacismo", "dialect_article"),
        ("Seseo", "dialect_article"),
        ("Aspiración de la /s/ en español", "dialect_article"),
        ("Habla bozal", "dialect_article"),
        ("Español de América", "dialect_article"),
        # Category B
        ("La Habana", "regional_article"),
        ("Salsa (género musical)", "regional_article"),
        ("República Dominicana", "regional_article"),
        ("Puerto Rico", "regional_article"),
        ("Reguetón", "regional_article"),
        ("Son cubano", "regional_article"),
        ("Merengue (género musical)", "regional_article"),
    ],
    # ---- ES_CHI: Chilean ----
    DialectCode.ES_CHI: [
        # Category A
        ("Español chileno", "dialect_article"),
        ("Chilenismos", "dialect_article"),
        ("Coa (jerga)", "dialect_article"),
        ("Voseo", "dialect_article"),
        ("Español de América", "dialect_article"),
        ("Mapudungunismos", "dialect_article"),
        ("Yeísmo", "dialect_article"),
        ("Aspiración de la /s/ en español", "dialect_article"),
        ("Jerga chilena", "dialect_article"),
        ("Chilenismo", "dialect_article"),
        # Category B
        ("Chile", "regional_article"),
        ("Empanada chilena", "regional_article"),
        ("Fiestas Patrias en Chile", "regional_article"),
        ("Santiago de Chile", "regional_article"),
        ("Cueca", "regional_article"),
        ("Valparaíso", "regional_article"),
        ("Gastronomía de Chile", "regional_article"),
    ],
    # ---- ES_AND_BO: Andean ----
    DialectCode.ES_AND_BO: [
        # Category A
        ("Español andino", "dialect_article"),
        ("Quechuismos", "dialect_article"),
        ("Español boliviano", "dialect_article"),
        ("Español ecuatoriano", "dialect_article"),
        ("Español peruano", "dialect_article"),
        ("Voseo", "dialect_article"),
        ("Español de América", "dialect_article"),
        ("Aimara", "dialect_article"),
        ("Quechua", "dialect_article"),
        ("Bilingüismo en el Perú", "dialect_article"),
        # Category B
        ("Bolivia", "regional_article"),
        ("Perú", "regional_article"),
        ("Pachamama", "regional_article"),
        ("Carnaval de Oruro", "regional_article"),
        ("Cuzco", "regional_article"),
        ("La Paz", "regional_article"),
        ("Gastronomía de Bolivia", "regional_article"),
    ],
}

# ---------------------------------------------------------------------------
# WikipediaFetcher
# ---------------------------------------------------------------------------


class WikipediaFetcher:
    """Fetches Spanish Wikipedia articles and converts them to
    :class:`~eigendialectos.types.DialectSample` instances.

    Parameters
    ----------
    cache_dir:
        Directory where fetched articles are cached as JSON files.
        Defaults to ``data/raw/wikipedia/`` relative to the project root.
    request_delay:
        Seconds to wait between consecutive API requests.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        request_delay: float = _REQUEST_DELAY_SECONDS,
    ) -> None:
        if cache_dir is None:
            # Default: project-root/data/raw/wikipedia
            project_root = Path(__file__).resolve().parents[4]
            self._cache_dir = project_root / "data" / "raw" / "wikipedia"
        else:
            self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._request_delay = request_delay
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self) -> dict[DialectCode, list[DialectSample]]:
        """Fetch articles for all dialect varieties.

        Returns
        -------
        dict[DialectCode, list[DialectSample]]
            Mapping from dialect code to a list of paragraph-level samples.
        """
        results: dict[DialectCode, list[DialectSample]] = {}
        total_articles = sum(len(arts) for arts in _DIALECT_ARTICLES.values())
        processed = 0

        for dialect in DialectCode:
            logger.info("Fetching articles for %s ...", dialect.value)
            samples = self.fetch_dialect(dialect, _progress_offset=processed)
            results[dialect] = samples
            processed += len(_DIALECT_ARTICLES.get(dialect, []))
            logger.info(
                "  %s: %d samples from %d articles",
                dialect.value,
                len(samples),
                len(_DIALECT_ARTICLES.get(dialect, [])),
            )

        total_samples = sum(len(s) for s in results.values())
        logger.info(
            "Fetch complete: %d total samples from %d articles across %d dialects",
            total_samples,
            total_articles,
            len(results),
        )
        return results

    def fetch_dialect(
        self,
        dialect: DialectCode,
        *,
        _progress_offset: int = 0,
    ) -> list[DialectSample]:
        """Fetch articles for a single dialect variety.

        Parameters
        ----------
        dialect:
            The dialect code to fetch articles for.

        Returns
        -------
        list[DialectSample]
            Paragraph-level samples extracted from the fetched articles.
        """
        article_specs = _DIALECT_ARTICLES.get(dialect, [])
        if not article_specs:
            logger.warning("No articles configured for dialect %s", dialect.value)
            return []

        samples: list[DialectSample] = []
        for idx, (title, category) in enumerate(article_specs, start=1):
            logger.info(
                "  [%d/%d] Fetching '%s' (%s)",
                idx,
                len(article_specs),
                title,
                category,
            )
            try:
                text = self._fetch_article(title)
            except Exception:
                logger.exception("    Failed to fetch '%s', skipping.", title)
                continue

            if not text or len(text.strip()) < _MIN_PARAGRAPH_LENGTH:
                logger.warning(
                    "    Article '%s' returned empty or very short text, skipping.",
                    title,
                )
                continue

            confidence = (
                _CONFIDENCE_DIALECT_ARTICLE
                if category == "dialect_article"
                else _CONFIDENCE_REGIONAL_ARTICLE
            )
            article_samples = self._segment_article(
                text, title, dialect, confidence, category
            )
            samples.extend(article_samples)
            logger.info(
                "    Got %d paragraphs (%d chars total)",
                len(article_samples),
                sum(len(s.text) for s in article_samples),
            )

        return samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_article(self, title: str) -> str:
        """Fetch a single Wikipedia article's plaintext via the MediaWiki API.

        Articles are cached locally as JSON files.  If a cached version
        exists, it is returned without making an API request.

        Parameters
        ----------
        title:
            The exact Wikipedia article title (in Spanish).

        Returns
        -------
        str
            The article's full plaintext content.
        """
        cache_path = self._cache_path(title)

        # Return cached version if available
        if cache_path.exists():
            logger.debug("    Cache hit for '%s'", title)
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached.get("text", "")

        # Fetch from API
        params: dict[str, str | int] = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": 1,
            "format": "json",
            "formatversion": 2,
        }

        text_parts: list[str] = []
        actual_title = title

        try:
            response = self._session.get(
                _API_ENDPOINT, params=params, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", [])
            if not pages:
                logger.warning("    No pages returned for '%s'", title)
                return ""

            page = pages[0]
            if page.get("missing", False):
                logger.warning(
                    "    Article '%s' not found on es.wikipedia.org", title
                )
                return ""

            actual_title = page.get("title", title)
            extract = page.get("extract", "")
            text_parts.append(extract)

        except requests.RequestException as exc:
            logger.error("    HTTP error fetching '%s': %s", title, exc)
            raise

        # Rate limit
        time.sleep(self._request_delay)

        full_text = "\n".join(text_parts)

        # Cache the result
        cache_data = {
            "title": actual_title,
            "requested_title": title,
            "text": full_text,
            "char_count": len(full_text),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug(
            "    Cached '%s' (%d chars) -> %s",
            actual_title,
            len(full_text),
            cache_path,
        )

        return full_text

    def _segment_article(
        self,
        text: str,
        title: str,
        dialect: DialectCode,
        confidence: float,
        category: str,
    ) -> list[DialectSample]:
        """Split article text into paragraph-level DialectSample instances.

        Parameters
        ----------
        text:
            Full article plaintext.
        title:
            Article title (used in metadata).
        dialect:
            Dialect code to assign.
        confidence:
            Confidence score (0.7 for dialect articles, 0.4 for regional).
        category:
            Either ``"dialect_article"`` or ``"regional_article"``.

        Returns
        -------
        list[DialectSample]
            One sample per qualifying paragraph (>= 50 chars).
        """
        # Split on double newlines (Wikipedia paragraph separators) and also
        # on section headings (== Heading ==)
        paragraphs = re.split(r"\n{2,}", text)

        samples: list[DialectSample] = []
        for para in paragraphs:
            # Strip section heading markers
            cleaned = re.sub(r"^={2,}\s*.*?\s*={2,}$", "", para, flags=re.MULTILINE)
            cleaned = cleaned.strip()

            if len(cleaned) < _MIN_PARAGRAPH_LENGTH:
                continue

            # Skip paragraphs that are just lists of "See also" / references
            if _is_boilerplate(cleaned):
                continue

            sample = DialectSample(
                text=cleaned,
                dialect_code=dialect,
                source_id="wikipedia",
                confidence=confidence,
                metadata={
                    "article": title,
                    "category": category,
                },
            )
            samples.append(sample)

        return samples

    def _cache_path(self, title: str) -> Path:
        """Return the local cache file path for a given article title."""
        # Sanitise the title for use as a filename
        safe_name = _sanitize_filename(title)
        return self._cache_dir / f"{safe_name}.json"

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def save_samples(
        self,
        results: dict[DialectCode, list[DialectSample]],
        output_dir: str | Path | None = None,
    ) -> Path:
        """Save fetched samples to a JSON file for downstream processing.

        Parameters
        ----------
        results:
            The output of :meth:`fetch_all` or a subset thereof.
        output_dir:
            Directory to write the output file.  Defaults to
            ``data/processed/``.

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        if output_dir is None:
            project_root = Path(__file__).resolve().parents[4]
            output_dir = project_root / "data" / "processed"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        out_file = output_path / "wikipedia_corpus.json"
        serializable: dict[str, list[dict[str, Any]]] = {}
        for code, samples in results.items():
            serializable[code.value] = [asdict(s) for s in samples]

        out_file.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved corpus to %s", out_file)
        return out_file


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def _sanitize_filename(title: str) -> str:
    """Convert a Wikipedia title to a safe filename."""
    # Replace characters that are problematic in file paths
    safe = title.replace("/", "_").replace("\\", "_")
    safe = re.sub(r'[<>:"|?*]', "_", safe)
    # Collapse multiple underscores / spaces
    safe = re.sub(r"[\s_]+", "_", safe).strip("_")
    # Limit length
    if len(safe) > 200:
        safe = safe[:200]
    return safe


def _is_boilerplate(text: str) -> bool:
    """Heuristic check for boilerplate/reference paragraphs."""
    lower = text.lower()
    # Skip very short lines that are just cross-references
    boilerplate_markers = [
        "véase también",
        "referencias",
        "bibliografía",
        "enlaces externos",
        "notas y referencias",
        "fuentes",
    ]
    # If the entire paragraph is just a heading-like marker, skip it
    if any(lower.strip() == marker for marker in boilerplate_markers):
        return True
    # Skip paragraphs that are mostly ISBN / URL references
    if lower.count("isbn") > 2 or lower.count("http") > 3:
        return True
    return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the fetcher as a standalone script."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Fetch Spanish Wikipedia articles for the EigenDialectos corpus.",
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default=None,
        help=(
            "Fetch only one dialect (e.g. ES_PEN, ES_AND).  "
            "If omitted, fetches all dialects."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache raw article JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write the processed corpus JSON.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=_REQUEST_DELAY_SECONDS,
        help=f"Seconds between API requests (default: {_REQUEST_DELAY_SECONDS}).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging.",
    )
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    fetcher = WikipediaFetcher(
        cache_dir=args.cache_dir,
        request_delay=args.delay,
    )

    if args.dialect:
        try:
            dialect = DialectCode(args.dialect.upper())
        except ValueError:
            valid = ", ".join(c.value for c in DialectCode)
            logger.error("Unknown dialect '%s'. Valid: %s", args.dialect, valid)
            sys.exit(1)

        logger.info("Fetching articles for %s ...", dialect.value)
        samples = fetcher.fetch_dialect(dialect)
        results = {dialect: samples}
    else:
        logger.info("Fetching articles for all dialects ...")
        results = fetcher.fetch_all()

    # Print summary
    total = 0
    for code in sorted(results, key=lambda c: c.value):
        n = len(results[code])
        total += n
        print(f"  {code.value}: {n} samples")
    print(f"  TOTAL: {total} samples")

    # Save
    out_path = fetcher.save_samples(results, output_dir=args.output_dir)
    print(f"\nCorpus saved to: {out_path}")


if __name__ == "__main__":
    main()
