"""Fetch Reddit posts and comments as dialectal corpus data.

Reddit is an excellent source of informal, colloquial text with natural
regional variation.  Each Spanish-speaking country/region has dedicated
subreddits whose users write in their local dialect variety.

The fetcher uses Reddit's public JSON API (no authentication required) by
appending ``.json`` to standard Reddit URLs.  Raw responses are cached
locally under ``data/raw/reddit/`` to avoid redundant network requests.

Posts and comments go through quality filtering:
- Minimum length threshold (30 characters)
- Language detection heuristic (skip English-dominant text)
- Skip deleted/removed content and bot comments
- Skip link-heavy content

Each extracted text is wrapped in a
:class:`~eigendialectos.types.DialectSample` with a confidence of 0.75
for country-specific subreddits or 0.50 for more general ones.
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

_BASE_URL = "https://www.reddit.com"
_USER_AGENT = "EigenDialectos:v0.1 (research corpus)"
_REQUEST_DELAY_SECONDS = 2.0  # Reddit rate limit: 1 req per 2 seconds
_RATE_LIMIT_WAIT_SECONDS = 60
_MIN_TEXT_LENGTH = 30
_POSTS_PER_PAGE = 25
_DEFAULT_PAGES = 5

_CONFIDENCE_SPECIFIC = 0.75  # Country/region-specific subreddits
_CONFIDENCE_GENERAL = 0.50  # General or multi-country subreddits

_SORT_MODES = ["hot", "top?t=year", "top?t=month"]

# Bot usernames to skip
_BOT_AUTHORS = frozenset({
    "AutoModerator",
    "[deleted]",
    "BotDefense",
    "RemindMeBot",
    "sneakpeekbot",
    "WikiTextBot",
    "CommonMisspellingBot",
    "HelperBot_",
    "RepostSleuthBot",
    "SaveVideo",
    "VredditDownloader",
})

# Common English words for language detection
_ENGLISH_MARKERS = frozenset({
    "the", "and", "is", "for", "that", "this", "with", "are",
    "was", "have", "has", "but", "not", "you", "they", "from",
    "what", "been", "would", "could", "should", "their", "which",
})

# Common Spanish words for language detection
_SPANISH_MARKERS = frozenset({
    "que", "de", "en", "por", "para", "con", "una", "los", "las",
    "del", "como", "pero", "más", "ser", "está", "hay", "esta",
    "también", "todo", "puede", "tiene", "cuando", "muy", "sin",
    "sobre", "entre", "todos", "este", "ese", "eso", "nos",
})

# ---------------------------------------------------------------------------
# Subreddit-to-dialect mapping
# ---------------------------------------------------------------------------

# Each entry is (subreddit_name, confidence) where confidence indicates how
# strongly the subreddit is associated with a specific dialect.

SubredditSpec = tuple[str, float]  # (subreddit, confidence)

_DIALECT_SUBREDDITS: dict[DialectCode, list[SubredditSpec]] = {
    DialectCode.ES_PEN: [
        ("spain", _CONFIDENCE_SPECIFIC),
        ("SpainPolitics", _CONFIDENCE_SPECIFIC),
        ("es", _CONFIDENCE_GENERAL),
        ("preguntaleareddit", _CONFIDENCE_GENERAL),
    ],
    DialectCode.ES_AND: [
        ("andalucia", _CONFIDENCE_SPECIFIC),
        ("Sevilla", _CONFIDENCE_SPECIFIC),
        ("malaga", _CONFIDENCE_SPECIFIC),
    ],
    DialectCode.ES_CAN: [
        ("canarias", _CONFIDENCE_SPECIFIC),
        ("Tenerife", _CONFIDENCE_SPECIFIC),
        ("GranCanaria", _CONFIDENCE_SPECIFIC),
    ],
    DialectCode.ES_RIO: [
        ("argentina", _CONFIDENCE_SPECIFIC),
        ("uruguay", _CONFIDENCE_SPECIFIC),
        ("BuenosAires", _CONFIDENCE_SPECIFIC),
        ("RepublicaArgentina", _CONFIDENCE_SPECIFIC),
    ],
    DialectCode.ES_MEX: [
        ("mexico", _CONFIDENCE_SPECIFIC),
        ("Mujico", _CONFIDENCE_SPECIFIC),
        ("MexicoCity", _CONFIDENCE_SPECIFIC),
    ],
    DialectCode.ES_CAR: [
        ("cuba", _CONFIDENCE_SPECIFIC),
        ("Dominican", _CONFIDENCE_GENERAL),
        ("PuertoRico", _CONFIDENCE_GENERAL),
        ("vzla", _CONFIDENCE_SPECIFIC),
    ],
    DialectCode.ES_CHI: [
        ("chile", _CONFIDENCE_SPECIFIC),
        ("Santiago", _CONFIDENCE_SPECIFIC),
    ],
    DialectCode.ES_AND_BO: [
        ("PERU", _CONFIDENCE_SPECIFIC),
        ("Bolivia", _CONFIDENCE_SPECIFIC),
        ("Ecuador", _CONFIDENCE_SPECIFIC),
    ],
}

# ---------------------------------------------------------------------------
# RedditFetcher
# ---------------------------------------------------------------------------


class RedditFetcher:
    """Fetches Reddit posts and comments and converts them to
    :class:`~eigendialectos.types.DialectSample` instances.

    Uses Reddit's public JSON API (no authentication needed).  Subreddits
    are organized by dialect variety and fetched across multiple sort modes
    (hot, top/year, top/month) with pagination.

    Parameters
    ----------
    cache_dir:
        Directory where raw JSON responses are cached.
        Defaults to ``data/raw/reddit/`` relative to the project root.
    request_delay:
        Seconds to wait between consecutive API requests (minimum 2.0
        to respect Reddit's rate limit).
    pages:
        Number of pages to fetch per subreddit per sort mode.
    fetch_comments:
        Whether to also fetch top-level comments from each post.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        request_delay: float = _REQUEST_DELAY_SECONDS,
        pages: int = _DEFAULT_PAGES,
        fetch_comments: bool = True,
    ) -> None:
        if cache_dir is None:
            project_root = Path(__file__).resolve().parents[4]
            self._cache_dir = project_root / "data" / "raw" / "reddit"
        else:
            self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._request_delay = max(request_delay, _REQUEST_DELAY_SECONDS)
        self._pages = pages
        self._fetch_comments = fetch_comments
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self) -> dict[DialectCode, list[DialectSample]]:
        """Fetch posts and comments for all dialect varieties.

        Returns
        -------
        dict[DialectCode, list[DialectSample]]
            Mapping from dialect code to extracted text samples.
        """
        results: dict[DialectCode, list[DialectSample]] = {}

        for dialect in DialectCode:
            logger.info("Fetching Reddit data for %s ...", dialect.value)
            subreddit_specs = _DIALECT_SUBREDDITS.get(dialect, [])
            dialect_samples: list[DialectSample] = []

            for subreddit, confidence in subreddit_specs:
                try:
                    samples = self.fetch_subreddit(
                        subreddit, dialect, pages=self._pages, confidence=confidence,
                    )
                    dialect_samples.extend(samples)
                except Exception:
                    logger.exception(
                        "  Failed to fetch r/%s, skipping.", subreddit
                    )

            results[dialect] = dialect_samples
            logger.info(
                "  %s: %d samples from %d subreddits",
                dialect.value,
                len(dialect_samples),
                len(subreddit_specs),
            )

        total_samples = sum(len(s) for s in results.values())
        logger.info(
            "Fetch complete: %d total samples across %d dialects",
            total_samples,
            len(results),
        )
        return results

    def fetch_subreddit(
        self,
        subreddit: str,
        dialect: DialectCode,
        pages: int = _DEFAULT_PAGES,
        confidence: float = _CONFIDENCE_SPECIFIC,
    ) -> list[DialectSample]:
        """Fetch posts and comments from a single subreddit.

        Iterates over multiple sort modes (hot, top/year, top/month) and
        paginates through each.

        Parameters
        ----------
        subreddit:
            Subreddit name (without the ``r/`` prefix).
        dialect:
            Dialect code to assign to extracted samples.
        pages:
            Number of listing pages to fetch per sort mode.
        confidence:
            Confidence score to assign to extracted samples.

        Returns
        -------
        list[DialectSample]
            Extracted and filtered text samples.
        """
        logger.info("  Fetching r/%s ...", subreddit)
        samples: list[DialectSample] = []
        seen_ids: set[str] = set()  # Deduplicate across sort modes
        sub_cache_dir = self._cache_dir / subreddit
        sub_cache_dir.mkdir(parents=True, exist_ok=True)

        for sort_mode in _SORT_MODES:
            logger.info("    Sort: %s", sort_mode)
            after_token: str | None = None

            for page_num in range(pages):
                # Build the listing URL.
                # Reddit JSON API: insert .json before query params.
                # "hot"        -> /r/{sub}/hot.json?limit=25
                # "top?t=year" -> /r/{sub}/top.json?t=year&limit=25
                if "?" in sort_mode:
                    sort_path, sort_query = sort_mode.split("?", 1)
                    url = (
                        f"{_BASE_URL}/r/{subreddit}/{sort_path}.json"
                        f"?{sort_query}&limit={_POSTS_PER_PAGE}"
                    )
                else:
                    url = (
                        f"{_BASE_URL}/r/{subreddit}/{sort_mode}.json"
                        f"?limit={_POSTS_PER_PAGE}"
                    )

                if after_token:
                    url += f"&after={after_token}"

                # Construct cache key
                sort_key = sort_mode.replace("?", "_").replace("=", "_")
                cache_file = sub_cache_dir / f"{sort_key}_page_{page_num}.json"

                # Fetch listing data
                data = self._fetch_page_cached(url, cache_file)
                if data is None:
                    logger.warning(
                        "    Failed to fetch page %d of r/%s/%s, stopping pagination.",
                        page_num, subreddit, sort_mode,
                    )
                    break

                # Extract the listing data
                listing = data.get("data", {})
                children = listing.get("children", [])
                after_token = listing.get("after")

                if not children:
                    logger.debug("    No more posts on page %d.", page_num)
                    break

                # Process each post
                for child in children:
                    if child.get("kind") != "t3":
                        continue
                    post_data = child.get("data", {})
                    post_id = post_data.get("id", "")

                    if post_id in seen_ids:
                        continue
                    seen_ids.add(post_id)

                    # Extract selftext (post body) for text posts
                    selftext = post_data.get("selftext", "").strip()
                    if selftext and _passes_quality_filter(selftext):
                        sample = DialectSample(
                            text=selftext,
                            dialect_code=dialect,
                            source_id=f"reddit:r/{subreddit}",
                            confidence=confidence,
                            metadata={
                                "subreddit": subreddit,
                                "post_id": post_id,
                                "type": "post",
                                "title": post_data.get("title", ""),
                                "score": post_data.get("score", 0),
                                "author": post_data.get("author", ""),
                            },
                        )
                        samples.append(sample)

                    # Fetch top-level comments if enabled
                    if self._fetch_comments and post_data.get("num_comments", 0) > 0:
                        comment_samples = self._fetch_post_comments(
                            subreddit, post_id, dialect, confidence,
                        )
                        samples.extend(comment_samples)

                logger.info(
                    "    Page %d: %d posts processed, %d samples so far",
                    page_num, len(children), len(samples),
                )

                if after_token is None:
                    break

        # Cache extracted text as JSONL
        self._save_posts_jsonl(sub_cache_dir, samples)

        logger.info("  r/%s: %d total samples", subreddit, len(samples))
        return samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_page(self, url: str) -> dict[str, Any] | None:
        """Fetch a JSON page from Reddit with rate limiting and retry.

        Parameters
        ----------
        url:
            Full Reddit URL (must end with ``.json`` or have JSON parameters).

        Returns
        -------
        dict or None
            Parsed JSON response, or ``None`` on unrecoverable error.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._session.get(url, timeout=30)

                if response.status_code == 429:
                    logger.warning(
                        "    Rate limited (429). Waiting %d seconds...",
                        _RATE_LIMIT_WAIT_SECONDS,
                    )
                    time.sleep(_RATE_LIMIT_WAIT_SECONDS)
                    continue

                if response.status_code == 403:
                    logger.warning(
                        "    Subreddit is private or quarantined (403): %s", url
                    )
                    return None

                if response.status_code == 404:
                    logger.warning("    Subreddit not found (404): %s", url)
                    return None

                response.raise_for_status()
                time.sleep(self._request_delay)

                return response.json()

            except requests.exceptions.JSONDecodeError:
                logger.warning(
                    "    Invalid JSON response from %s (attempt %d/%d)",
                    url, attempt + 1, max_retries,
                )
                time.sleep(self._request_delay)
            except requests.RequestException as exc:
                logger.error(
                    "    HTTP error fetching %s (attempt %d/%d): %s",
                    url, attempt + 1, max_retries, exc,
                )
                time.sleep(self._request_delay)

        return None

    def _fetch_page_cached(
        self, url: str, cache_file: Path
    ) -> dict[str, Any] | None:
        """Fetch a Reddit JSON page, returning cached version if available.

        Parameters
        ----------
        url:
            Full Reddit URL.
        cache_file:
            Path to the local cache file.

        Returns
        -------
        dict or None
            Parsed JSON response, or ``None`` on error.
        """
        if cache_file.exists():
            logger.debug("    Cache hit: %s", cache_file.name)
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "    Corrupt cache file %s, re-fetching.", cache_file
                )

        data = self._fetch_page(url)
        if data is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(
                    json.dumps(data, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("    Failed to cache %s: %s", cache_file, exc)

        return data

    def _fetch_post_comments(
        self,
        subreddit: str,
        post_id: str,
        dialect: DialectCode,
        confidence: float,
    ) -> list[DialectSample]:
        """Fetch top-level comments from a single Reddit post.

        Parameters
        ----------
        subreddit:
            Subreddit name.
        post_id:
            Reddit post ID (the short alphanumeric string).
        dialect:
            Dialect code to assign.
        confidence:
            Confidence score to assign.

        Returns
        -------
        list[DialectSample]
            Filtered comment samples.
        """
        url = f"{_BASE_URL}/r/{subreddit}/comments/{post_id}.json"
        cache_dir = self._cache_dir / subreddit / "comments"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{post_id}.json"

        data = self._fetch_page_cached(url, cache_file)
        if data is None:
            return []

        samples: list[DialectSample] = []

        # Reddit comment pages return a list of two listings:
        # [0] = the post itself, [1] = the comments
        if not isinstance(data, list) or len(data) < 2:
            return []

        comment_listing = data[1]
        children = comment_listing.get("data", {}).get("children", [])

        for child in children:
            if child.get("kind") != "t1":
                continue
            comment_data = child.get("data", {})
            body = comment_data.get("body", "").strip()
            author = comment_data.get("author", "")

            # Skip bots
            if author in _BOT_AUTHORS:
                continue

            if body and _passes_quality_filter(body):
                sample = DialectSample(
                    text=body,
                    dialect_code=dialect,
                    source_id=f"reddit:r/{subreddit}",
                    confidence=confidence,
                    metadata={
                        "subreddit": subreddit,
                        "post_id": post_id,
                        "comment_id": comment_data.get("id", ""),
                        "type": "comment",
                        "score": comment_data.get("score", 0),
                        "author": author,
                    },
                )
                samples.append(sample)

        return samples

    @staticmethod
    def _save_posts_jsonl(
        cache_dir: Path, samples: list[DialectSample]
    ) -> None:
        """Save extracted samples to a JSONL file for quick inspection.

        Parameters
        ----------
        cache_dir:
            Subreddit-specific cache directory.
        samples:
            Samples to serialize.
        """
        out_file = cache_dir / "posts.jsonl"
        try:
            with out_file.open("w", encoding="utf-8") as fh:
                for sample in samples:
                    line = json.dumps(asdict(sample), ensure_ascii=False)
                    fh.write(line + "\n")
        except OSError as exc:
            logger.warning("Failed to write %s: %s", out_file, exc)

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

        out_file = output_path / "reddit_corpus.json"
        serializable: dict[str, list[dict[str, Any]]] = {}
        for code, samples in results.items():
            serializable[code.value] = [asdict(s) for s in samples]

        out_file.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved Reddit corpus to %s", out_file)
        return out_file


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------


def _passes_quality_filter(text: str) -> bool:
    """Apply quality heuristics to determine if text is usable.

    Checks applied:
    1. Minimum length (30 characters)
    2. Not deleted/removed content
    3. Not mostly URLs
    4. Not English-dominant (simple word frequency heuristic)

    Parameters
    ----------
    text:
        The raw post body or comment text.

    Returns
    -------
    bool
        ``True`` if the text passes all quality checks.
    """
    # Length check
    if len(text) < _MIN_TEXT_LENGTH:
        return False

    # Skip deleted/removed content
    stripped = text.strip().lower()
    if stripped in ("[deleted]", "[removed]", "[eliminado]"):
        return False

    # Skip text that is mostly URLs
    url_pattern = re.compile(r"https?://\S+")
    urls = url_pattern.findall(text)
    text_without_urls = url_pattern.sub("", text).strip()
    if len(text_without_urls) < _MIN_TEXT_LENGTH:
        return False
    if urls and len("".join(urls)) > len(text) * 0.5:
        return False

    # Language detection: skip English-dominant text
    if _is_english_dominant(text_without_urls):
        return False

    return True


def _is_english_dominant(text: str) -> bool:
    """Heuristic to detect if text is predominantly English.

    Compares the count of common English words against common Spanish
    words.  If English words outnumber Spanish ones, the text is
    considered English.

    Parameters
    ----------
    text:
        Text to analyze (ideally with URLs already removed).

    Returns
    -------
    bool
        ``True`` if the text appears to be predominantly English.
    """
    words = set(re.findall(r"\b\w+\b", text.lower()))

    english_count = len(words & _ENGLISH_MARKERS)
    spanish_count = len(words & _SPANISH_MARKERS)

    # If neither language has markers, don't filter
    if english_count == 0 and spanish_count == 0:
        return False

    # If English markers clearly dominate
    return english_count > spanish_count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Reddit fetcher as a standalone script."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Fetch Reddit posts/comments for the EigenDialectos corpus.",
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default=None,
        help=(
            "Fetch only one dialect (e.g. ES_PEN, ES_RIO).  "
            "If omitted, fetches all dialects."
        ),
    )
    parser.add_argument(
        "--subreddit",
        type=str,
        default=None,
        help=(
            "Fetch a single subreddit (e.g. 'argentina').  "
            "Requires --dialect to be set."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache raw Reddit JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write the processed corpus JSON.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=_DEFAULT_PAGES,
        help=f"Number of listing pages per sort mode (default: {_DEFAULT_PAGES}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=_REQUEST_DELAY_SECONDS,
        help=f"Seconds between requests (default: {_REQUEST_DELAY_SECONDS}).",
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Skip fetching comments (faster, fewer samples).",
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

    fetcher = RedditFetcher(
        cache_dir=args.cache_dir,
        request_delay=args.delay,
        pages=args.pages,
        fetch_comments=not args.no_comments,
    )

    if args.subreddit:
        if not args.dialect:
            logger.error("--subreddit requires --dialect to be set.")
            sys.exit(1)
        try:
            dialect = DialectCode(args.dialect.upper())
        except ValueError:
            valid = ", ".join(c.value for c in DialectCode)
            logger.error("Unknown dialect '%s'. Valid: %s", args.dialect, valid)
            sys.exit(1)
        logger.info(
            "Fetching r/%s for %s ...", args.subreddit, dialect.value
        )
        samples = fetcher.fetch_subreddit(
            args.subreddit, dialect, pages=args.pages,
        )
        results = {dialect: samples}
    elif args.dialect:
        try:
            dialect = DialectCode(args.dialect.upper())
        except ValueError:
            valid = ", ".join(c.value for c in DialectCode)
            logger.error("Unknown dialect '%s'. Valid: %s", args.dialect, valid)
            sys.exit(1)
        logger.info("Fetching Reddit data for %s ...", dialect.value)
        subreddit_specs = _DIALECT_SUBREDDITS.get(dialect, [])
        dialect_samples: list[DialectSample] = []
        for subreddit, confidence in subreddit_specs:
            try:
                s = fetcher.fetch_subreddit(
                    subreddit, dialect, pages=args.pages, confidence=confidence,
                )
                dialect_samples.extend(s)
            except Exception:
                logger.exception("  Failed to fetch r/%s, skipping.", subreddit)
        results = {dialect: dialect_samples}
    else:
        logger.info("Fetching Reddit data for all dialects ...")
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
    print(f"\nReddit corpus saved to: {out_path}")


if __name__ == "__main__":
    main()
