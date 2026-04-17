"""OpenSubtitles REST API fetcher for curated dialectal subtitle corpora.

Uses the OpenSubtitles.com REST API (v1) to search and download SRT
subtitle files for films and TV shows known to feature authentic
regional Spanish speech.  Each dialect variety has a hand-curated list
of titles chosen for colloquial, naturalistic dialogue.

Workflow per dialect
--------------------
1. Iterate over curated title list.
2. Search the API (``GET /subtitles``) for each title in Spanish,
   excluding AI-translated results.
3. Select the best subtitle file (highest ``download_count``).
4. Download the SRT via ``POST /download`` -> follow the returned link.
5. Parse the SRT into individual dialogue lines.
6. Wrap each line as a :class:`DialectSample` with confidence 0.85.

Caching
-------
- Search results: ``data/raw/opensubtitles/search_cache.json``
- SRT files:      ``data/raw/opensubtitles/{dialect}/{file_id}.srt``
- Download quota:  ``data/raw/opensubtitles/download_count.json``
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import requests

from eigendialectos.constants import DialectCode
from eigendialectos.types import DialectSample

logger = logging.getLogger(__name__)

# ======================================================================
# API configuration
# ======================================================================

_BASE_URL = "https://api.opensubtitles.com/api/v1"

_HEADERS = {
    "Api-Key": "2cJKSrMOejQjo1StVcxYikNJrk5Vjt6h",
    "Authorization": (
        "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9."
        "eyJpc3MiOiJ6bFVmU28xSHA1b3FodHpLMktmOFkzdEx2U2drMUsxbCIs"
        "ImV4cCI6MTc3NTM4MTI0OX0."
        "qfdfAwYMEFWyN0IfV_KEVgPvkQEKbDVjSM7DhywC9wU"
    ),
    "User-Agent": "EigenDialectos v0.1",
    "Content-Type": "application/json",
}

# ======================================================================
# Rate-limit / quota defaults
# ======================================================================

_SEARCH_DELAY_S = 1.0
_DOWNLOAD_DELAY_S = 2.0
_MAX_DOWNLOADS_PER_DIALECT = 40
_DAILY_DOWNLOAD_QUOTA = 998

# ======================================================================
# SRT parsing helpers
# ======================================================================

_SRT_INDEX_RE = re.compile(r"^\d+\s*$")
_SRT_TIMESTAMP_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}"
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SUBTITLE_NOISE_RE = re.compile(
    r"(?i)"
    r"\[(?:Music|Música|Applause|Laughter|Risas|♪|#)[^\]]*\]"
    r"|♪+"
    r"|#+"
    r"|\{[^}]*\}"        # SSA overrides {\\an8}
    r"|- +"              # leading dialogue dash
)
_WHITESPACE_RE = re.compile(r"\s+")

_MIN_LINE_LENGTH = 15
_ALPHA_RATIO_THRESHOLD = 0.50

# ======================================================================
# Curated title lists per dialect
# ======================================================================

DIALECT_TITLES: dict[DialectCode, list[str]] = {
    DialectCode.ES_PEN: [
        "volver",
        "la casa de papel",
        "ocho apellidos vascos",
        "todo sobre mi madre",
        "los lunes al sol",
        "mar adentro",
        "celda 211",
        "el buen patron",
        "campeones",
        "perfectos desconocidos",
        "tesis",
        "abre los ojos",
        "el dia de la bestia",
        "vis a vis",
        "elite",
    ],
    DialectCode.ES_AND: [
        "solas zambrano",
        "la isla minima",
        "ocho apellidos catalanes",
        "vivir es facil con los ojos cerrados",
        "la gran familia española",
        "el camino de los ingleses",
        "grupo 7",
        "alosno",
        "maria la del barrio",
        "solas",
    ],
    DialectCode.ES_CAN: [
        "mararía",
        "el rayo canarias",
        "canarias film",
        "tenerife",
        "las palmas",
        "canarias",
    ],
    DialectCode.ES_RIO: [
        "el secreto de sus ojos",
        "nueve reinas",
        "relatos salvajes",
        "el clan",
        "un cuento chino",
        "esperando la carroza",
        "pizza birra faso",
        "carancho",
        "el angel 2018",
        "el marginal",
        "okupas",
        "tiempo de valientes",
        "el aura",
        "el hijo de la novia",
        "bombón el perro",
    ],
    DialectCode.ES_MEX: [
        "amores perros",
        "y tu mama tambien",
        "roma cuaron",
        "el infierno",
        "nosotros los nobles",
        "no se aceptan devoluciones",
        "la ley de herodes",
        "el crimen del padre amaro",
        "gueros",
        "que culpa tiene el nino",
        "el tigre de santa julia",
        "club de cuervos",
        "la dictadura perfecta",
        "matando cabos",
    ],
    DialectCode.ES_CAR: [
        "fresa y chocolate",
        "habana blues",
        "juan of the dead",
        "conducta",
        "el rey de la habana",
        "lista de espera",
        "memorias del subdesarrollo",
        "los dioses rotos",
        "tres veces ana",
    ],
    DialectCode.ES_CHI: [
        "no 2012",
        "gloria 2013",
        "una mujer fantastica",
        "machuca",
        "el club",
        "tony manero",
        "stefan vs kramer",
        "que pena tu vida",
        "baby shower 2011",
        "el chacotero sentimental",
        "los 80",
        "el bosque de karadima",
    ],
    DialectCode.ES_AND_BO: [
        "la teta asustada",
        "zona sur bolivia",
        "ratas ratones rateros",
        "contracorriente peru",
        "magallanes peru",
        "madeinusa",
        "claudia llosa",
        "retablo peru",
        "juku bolivia",
        "quien mato a la llamita blanca",
    ],
}


# ======================================================================
# OpenSubtitlesFetcher
# ======================================================================


class OpenSubtitlesFetcher:
    """Fetch curated dialectal subtitles via the OpenSubtitles REST API.

    Parameters
    ----------
    data_dir:
        Root cache directory.  Defaults to ``<project>/data/raw/opensubtitles``.
    max_downloads_per_dialect:
        Maximum number of SRT files to download per dialect in one run.
    daily_quota:
        Total daily download limit to respect.
    search_delay:
        Seconds to wait between consecutive search API calls.
    download_delay:
        Seconds to wait between consecutive download API calls.
    confidence:
        Confidence score assigned to every sample produced by this
        fetcher.  Defaults to 0.85 (high: curated country-specific films).
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        max_downloads_per_dialect: int = _MAX_DOWNLOADS_PER_DIALECT,
        daily_quota: int = _DAILY_DOWNLOAD_QUOTA,
        search_delay: float = _SEARCH_DELAY_S,
        download_delay: float = _DOWNLOAD_DELAY_S,
        confidence: float = 0.85,
    ) -> None:
        if data_dir is None:
            project_root = Path(__file__).resolve().parents[4]
            data_dir = project_root / "data" / "raw" / "opensubtitles"
        self._data_dir = Path(data_dir)
        self._max_downloads_per_dialect = max_downloads_per_dialect
        self._daily_quota = daily_quota
        self._search_delay = search_delay
        self._download_delay = download_delay
        self._confidence = confidence

        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

        # Caches loaded lazily
        self._search_cache: dict[str, list[dict]] | None = None
        self._download_count: int = 0

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._load_download_count()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self) -> dict[DialectCode, list[DialectSample]]:
        """Download and parse subtitles for every dialect.

        Returns a dict mapping each :class:`DialectCode` to its list
        of :class:`DialectSample` instances.
        """
        result: dict[DialectCode, list[DialectSample]] = {}
        for dialect in DialectCode:
            samples = self.fetch_dialect(dialect)
            if samples:
                result[dialect] = samples
            logger.info(
                "Dialect %s: %d samples collected", dialect.value, len(samples)
            )
        return result

    def fetch_dialect(self, dialect: DialectCode) -> list[DialectSample]:
        """Search, download, and parse subtitles for a single dialect.

        Iterates over the curated title list for *dialect*, searches
        the API, downloads up to :attr:`max_downloads_per_dialect` SRT
        files, parses them, and returns :class:`DialectSample` instances.
        """
        titles = DIALECT_TITLES.get(dialect, [])
        if not titles:
            logger.warning("No curated titles for dialect %s", dialect.value)
            return []

        dialect_dir = self._data_dir / dialect.value
        dialect_dir.mkdir(parents=True, exist_ok=True)

        samples: list[DialectSample] = []
        downloads_this_run = 0

        for title in titles:
            if downloads_this_run >= self._max_downloads_per_dialect:
                logger.info(
                    "Reached per-dialect download cap (%d) for %s",
                    self._max_downloads_per_dialect,
                    dialect.value,
                )
                break

            if self._download_count >= self._daily_quota:
                logger.warning(
                    "Daily download quota (%d) exhausted. Stopping.",
                    self._daily_quota,
                )
                break

            # --- Search ---
            search_results = self.search_subtitles(title)
            if not search_results:
                logger.info(
                    "No results for '%s' (%s) -- skipping", title, dialect.value
                )
                continue

            # Pick the best file_id (highest download_count)
            best = self._pick_best_result(search_results)
            if best is None:
                logger.info(
                    "No suitable file found for '%s' (%s)", title, dialect.value
                )
                continue

            file_id = best["file_id"]
            feature_title = best.get("title", title)

            # --- Check SRT cache ---
            srt_path = dialect_dir / f"{file_id}.srt"
            if srt_path.exists():
                logger.info(
                    "SRT cached: %s (file_id=%d)", feature_title, file_id
                )
                srt_content = srt_path.read_text(encoding="utf-8", errors="replace")
            else:
                # --- Download ---
                srt_content = self.download_subtitle(file_id)
                if not srt_content:
                    logger.warning(
                        "Download failed for '%s' file_id=%d", feature_title, file_id
                    )
                    continue

                srt_path.write_text(srt_content, encoding="utf-8")
                downloads_this_run += 1
                self._download_count += 1
                self._save_download_count()
                logger.info(
                    "Downloaded: %s (file_id=%d, %d bytes)",
                    feature_title,
                    file_id,
                    len(srt_content),
                )

            # --- Parse ---
            lines = self.parse_srt(srt_content)
            for line in lines:
                sample = DialectSample(
                    text=line,
                    dialect_code=dialect,
                    source_id=f"opensubtitles:{file_id}",
                    confidence=self._confidence,
                    metadata={
                        "title": feature_title,
                        "file_id": file_id,
                        "fetcher": "opensubtitles_api",
                    },
                )
                samples.append(sample)

            logger.info(
                "Parsed %d lines from '%s' for %s",
                len(lines),
                feature_title,
                dialect.value,
            )

        logger.info(
            "Dialect %s complete: %d total samples from %d titles",
            dialect.value,
            len(samples),
            len(titles),
        )
        return samples

    def search_subtitles(self, query: str) -> list[dict]:
        """Search the OpenSubtitles API for Spanish subtitles.

        Parameters
        ----------
        query:
            Free-text search query (film/show title).

        Returns
        -------
        list[dict]
            Each dict contains at minimum ``file_id``, ``title``, and
            ``download_count`` keys.  Returns an empty list on failure.
        """
        # Check cache first
        cache = self._get_search_cache()
        cache_key = query.lower().strip()
        if cache_key in cache:
            logger.debug("Search cache hit for '%s'", query)
            return cache[cache_key]

        url = f"{_BASE_URL}/subtitles"
        params = {
            "languages": "es",
            "query": query,
            "ai_translated": "exclude",
        }

        results: list[dict] = []
        try:
            resp = self._api_get(url, params=params)
            if resp is None:
                return []

            data = resp.get("data", [])
            for item in data:
                attrs = item.get("attributes", {})
                files = attrs.get("files", [])
                feature = attrs.get("feature_details", {})
                download_count = attrs.get("download_count", 0)

                for f in files:
                    results.append({
                        "file_id": f.get("file_id"),
                        "title": feature.get("title", query),
                        "download_count": download_count,
                        "year": feature.get("year"),
                        "ai_translated": attrs.get("ai_translated", False),
                    })

        except Exception:
            logger.exception("Search failed for query '%s'", query)

        # Cache the results (even empty -- avoids repeated failed lookups)
        cache[cache_key] = results
        self._save_search_cache()

        time.sleep(self._search_delay)
        return results

    def download_subtitle(self, file_id: int) -> str:
        """Request a download link and fetch the SRT content.

        Parameters
        ----------
        file_id:
            The ``file_id`` obtained from a search result.

        Returns
        -------
        str
            Raw SRT file content, or an empty string on failure.
        """
        url = f"{_BASE_URL}/download"
        payload = {"file_id": file_id}

        try:
            resp = self._api_post(url, json_body=payload)
            if resp is None:
                return ""

            link = resp.get("link", "")
            if not link:
                logger.warning("No download link returned for file_id=%d", file_id)
                return ""

            # Fetch the actual SRT content from the CDN link
            srt_resp = self._session.get(
                link, timeout=30, allow_redirects=True
            )
            srt_resp.raise_for_status()

            time.sleep(self._download_delay)
            return srt_resp.text

        except Exception:
            logger.exception("Download failed for file_id=%d", file_id)
            time.sleep(self._download_delay)
            return ""

    @staticmethod
    def parse_srt(content: str) -> list[str]:
        """Extract clean dialogue lines from SRT subtitle content.

        Strips sequence numbers, timestamps, HTML tags, subtitle noise
        markers (``[Music]``, ``{\\an8}``, etc.), and joins multi-line
        subtitle blocks into single lines.

        Only lines with at least 15 characters and >= 50 %% alphabetic
        characters are retained.

        Parameters
        ----------
        content:
            Raw SRT file content.

        Returns
        -------
        list[str]
            Cleaned dialogue strings.
        """
        lines = content.replace("\r\n", "\n").split("\n")
        blocks: list[list[str]] = []
        current_block: list[str] = []

        for line in lines:
            stripped = line.strip()

            # Skip SRT index lines (bare numbers)
            if _SRT_INDEX_RE.match(stripped):
                # Flush previous block
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                continue

            # Skip timestamp lines
            if _SRT_TIMESTAMP_RE.match(stripped):
                continue

            # Empty line = block separator
            if not stripped:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                continue

            current_block.append(stripped)

        # Flush trailing block
        if current_block:
            blocks.append(current_block)

        # Join multi-line blocks and clean
        result: list[str] = []
        for block in blocks:
            joined = " ".join(block)
            # Strip HTML tags
            joined = _HTML_TAG_RE.sub("", joined)
            # Strip subtitle noise
            joined = _SUBTITLE_NOISE_RE.sub("", joined)
            # Normalise whitespace
            joined = _WHITESPACE_RE.sub(" ", joined).strip()

            if not joined:
                continue

            # Quality filters
            if len(joined) < _MIN_LINE_LENGTH:
                continue
            alpha_count = sum(1 for c in joined if c.isalpha())
            if alpha_count / max(len(joined), 1) < _ALPHA_RATIO_THRESHOLD:
                continue

            result.append(joined)

        return result

    # ------------------------------------------------------------------
    # Internal: API helpers
    # ------------------------------------------------------------------

    def _api_get(
        self, url: str, params: dict | None = None
    ) -> dict | None:
        """Perform a GET request with retry on rate-limit (HTTP 429)."""
        try:
            resp = self._session.get(
                url, params=params, timeout=30, allow_redirects=True
            )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                logger.warning(
                    "Rate limited (429). Waiting %d seconds and retrying.",
                    retry_after,
                )
                time.sleep(retry_after)
                resp = self._session.get(
                    url, params=params, timeout=30, allow_redirects=True
                )

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException:
            logger.exception("GET %s failed", url)
            return None

    def _api_post(
        self, url: str, json_body: dict | None = None
    ) -> dict | None:
        """Perform a POST request with retry on rate-limit (HTTP 429)."""
        try:
            resp = self._session.post(
                url, json=json_body, timeout=30, allow_redirects=True
            )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                logger.warning(
                    "Rate limited (429). Waiting %d seconds and retrying.",
                    retry_after,
                )
                time.sleep(retry_after)
                resp = self._session.post(
                    url, json=json_body, timeout=30, allow_redirects=True
                )

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException:
            logger.exception("POST %s failed", url)
            return None

    # ------------------------------------------------------------------
    # Internal: result selection
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_best_result(results: list[dict]) -> dict | None:
        """Select the best subtitle from search results.

        Prefers non-AI-translated results with the highest download count.
        Returns ``None`` if no valid candidates exist.
        """
        candidates = [
            r for r in results
            if r.get("file_id") and not r.get("ai_translated", False)
        ]

        if not candidates:
            # Fall back to all results with a file_id
            candidates = [r for r in results if r.get("file_id")]

        if not candidates:
            return None

        # Sort by download_count descending
        candidates.sort(key=lambda r: r.get("download_count", 0), reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # Internal: search cache
    # ------------------------------------------------------------------

    def _get_search_cache(self) -> dict[str, list[dict]]:
        """Lazily load the search cache from disk."""
        if self._search_cache is not None:
            return self._search_cache

        cache_path = self._data_dir / "search_cache.json"
        if cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as fh:
                    self._search_cache = json.load(fh)
                    logger.info(
                        "Loaded search cache with %d entries",
                        len(self._search_cache),
                    )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load search cache: %s", exc)
                self._search_cache = {}
        else:
            self._search_cache = {}

        return self._search_cache

    def _save_search_cache(self) -> None:
        """Persist the search cache to disk."""
        if self._search_cache is None:
            return
        cache_path = self._data_dir / "search_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(self._search_cache, fh, ensure_ascii=False, indent=2)
        except OSError:
            logger.exception("Failed to save search cache")

    # ------------------------------------------------------------------
    # Internal: download-count tracking
    # ------------------------------------------------------------------

    def _load_download_count(self) -> None:
        """Load the daily download counter from disk."""
        count_path = self._data_dir / "download_count.json"
        if count_path.exists():
            try:
                with open(count_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                stored_date = data.get("date", "")
                today = time.strftime("%Y-%m-%d")
                if stored_date == today:
                    self._download_count = data.get("count", 0)
                    logger.info(
                        "Resumed download count: %d (date=%s)",
                        self._download_count,
                        today,
                    )
                else:
                    # New day -- reset counter
                    self._download_count = 0
                    logger.info("New day; download counter reset to 0.")
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load download count: %s", exc)
                self._download_count = 0
        else:
            self._download_count = 0

    def _save_download_count(self) -> None:
        """Persist the daily download counter to disk."""
        count_path = self._data_dir / "download_count.json"
        count_path.parent.mkdir(parents=True, exist_ok=True)
        today = time.strftime("%Y-%m-%d")
        try:
            with open(count_path, "w", encoding="utf-8") as fh:
                json.dump({"date": today, "count": self._download_count}, fh)
        except OSError:
            logger.exception("Failed to save download count")

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def download_count(self) -> int:
        """Number of downloads performed today."""
        return self._download_count

    @property
    def remaining_quota(self) -> int:
        """Estimated remaining daily downloads."""
        return max(0, self._daily_quota - self._download_count)

    def summary(self) -> dict[str, object]:
        """Return a summary of cached data per dialect."""
        info: dict[str, object] = {
            "downloads_today": self._download_count,
            "remaining_quota": self.remaining_quota,
            "dialects": {},
        }
        for dialect in DialectCode:
            dialect_dir = self._data_dir / dialect.value
            if dialect_dir.exists():
                srt_files = list(dialect_dir.glob("*.srt"))
                info["dialects"][dialect.value] = {  # type: ignore[index]
                    "cached_srt_files": len(srt_files),
                    "curated_titles": len(DIALECT_TITLES.get(dialect, [])),
                }
        return info
