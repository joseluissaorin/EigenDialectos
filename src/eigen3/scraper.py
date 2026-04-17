"""Unified multi-source corpus scraper for dialectal Spanish.

Sources: Reddit, Wikipedia, OpenSubtitles, song lyrics, HuggingFace datasets,
OPUS bulk download. All scraping is automatic — no manual steps required.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests

from eigen3.constants import (
    ALL_VARIETIES,
    DIALECT_MARKERS,
    OPENSUB_API_KEY,
    OPENSUB_BASE_URL,
    SCRAPER_FILMS,
    SCRAPER_NEWS_SITES,
    SCRAPER_SONGS,
    SCRAPER_SUBREDDITS,
    SCRAPER_WIKI_ARTICLES,
    TLD_TO_DIALECT,
)

logger = logging.getLogger(__name__)

# SRT timestamp pattern: 00:01:23,456 --> 00:01:25,789
_SRT_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}")
_SRT_INDEX_RE = re.compile(r"^\d+$")
_SRT_TAGS_RE = re.compile(r"<[^>]+>|\{[^}]*\}")
_SRT_MARKERS_RE = re.compile(r"\[.*?\]|\(.*?\)")

# Bot authors to skip on Reddit
_BOT_AUTHORS = frozenset({"AutoModerator", "[deleted]", "RemindMeBot", "sneakpeekbot"})


def _sanitize_filename(s: str) -> str:
    """Convert a string to a safe filename."""
    return re.sub(r"[^\w\-.]", "_", s)[:120]


class CorpusScraper:
    """Unified scraper for dialectal Spanish from multiple web sources."""

    def __init__(
        self,
        cache_dir: str | Path = "data/raw",
        api_key: str | None = None,
        delay: float = 2.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or OPENSUB_API_KEY
        self.delay = delay
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "EigenDialectos/3.0 (research; +https://github.com/eigendialectos)",
            "Accept": "application/json",
        })
        self._last_request_time: float = 0.0
        self._osub_token: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_all(
        self, sources: list[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Scrape all sources. Returns {dialect: [sample_dicts]}."""
        sources = sources or [
            "huggingface", "reddit", "wikipedia", "opensubtitles",
            "lyrics", "opus", "news",
        ]
        result: dict[str, list[dict]] = {v: [] for v in ALL_VARIETIES}

        dispatch = {
            "reddit": self.scrape_reddit,
            "wikipedia": self.scrape_wikipedia,
            "opensubtitles": self.scrape_opensubtitles,
            "lyrics": self.scrape_lyrics,
            "huggingface": self.scrape_huggingface,
            "opus": self.scrape_opus_bulk,
            "news": self.scrape_regional_news,
        }

        for source in sources:
            fn = dispatch.get(source)
            if fn is None:
                logger.warning("Unknown source: %s", source)
                continue
            try:
                samples = fn()
                for dialect, docs in samples.items():
                    result[dialect].extend(docs)
                logger.info("Source '%s': collected %d total samples",
                            source, sum(len(d) for d in samples.values()))
            except Exception:
                logger.exception("Source '%s' failed — skipping", source)

        return result

    # ------------------------------------------------------------------
    # Reddit
    # ------------------------------------------------------------------

    def scrape_reddit(
        self, pages: int = 5, sorts: tuple[str, ...] = ("hot", "top", "new"),
    ) -> dict[str, list[dict]]:
        """Scrape Reddit comments and posts from dialect-specific subreddits."""
        result: dict[str, list[dict]] = {v: [] for v in ALL_VARIETIES}

        for dialect, subs in SCRAPER_SUBREDDITS.items():
            for sub in subs:
                for sort in sorts:
                    try:
                        posts = self._reddit_fetch_listing(sub, sort, pages)
                        for post in posts:
                            samples = self._reddit_extract_samples(post, sub, dialect)
                            result[dialect].extend(samples)
                    except Exception:
                        logger.warning("Reddit r/%s/%s failed", sub, sort, exc_info=True)

            logger.info("Reddit %s: %d samples", dialect, len(result[dialect]))

        return result

    def _reddit_fetch_listing(
        self, subreddit: str, sort: str, pages: int,
    ) -> list[dict]:
        """Fetch listing pages from a subreddit."""
        posts: list[dict] = []
        after: str | None = None

        for _ in range(pages):
            params: dict[str, Any] = {"limit": 100, "raw_json": 1}
            if after:
                params["after"] = after

            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            cache_key = f"reddit/{subreddit}/{sort}_{after or 'first'}"
            data = self._cached_get_json(url, params, cache_key)
            if data is None:
                break

            children = data.get("data", {}).get("children", [])
            if not children:
                break

            for child in children:
                posts.append(child.get("data", {}))

            after = data.get("data", {}).get("after")
            if not after:
                break

        return posts

    def _reddit_extract_samples(
        self, post: dict, subreddit: str, dialect: str,
    ) -> list[dict]:
        """Extract text samples from a Reddit post and its comments."""
        samples: list[dict] = []
        source_tag = f"reddit:r/{subreddit}"

        # Post self-text
        selftext = (post.get("selftext") or "").strip()
        if selftext and len(selftext) >= 30:
            author = post.get("author", "")
            if author not in _BOT_AUTHORS and not selftext.startswith("["):
                samples.append({
                    "text": selftext,
                    "dialect": dialect,
                    "source": source_tag,
                    "confidence": 0.7,
                })

        # Title (short, but useful for dialect markers)
        title = (post.get("title") or "").strip()
        if title and len(title) >= 20:
            samples.append({
                "text": title,
                "dialect": dialect,
                "source": source_tag,
                "confidence": 0.5,
            })

        # Fetch comments for posts with significant discussion
        n_comments = post.get("num_comments", 0)
        if n_comments >= 5:
            permalink = post.get("permalink", "")
            if permalink:
                try:
                    comments = self._reddit_fetch_comments(permalink, subreddit, dialect)
                    samples.extend(comments)
                except Exception:
                    pass

        return samples

    def _reddit_fetch_comments(
        self, permalink: str, subreddit: str, dialect: str,
    ) -> list[dict]:
        """Fetch comments from a Reddit post."""
        url = f"https://www.reddit.com{permalink}.json"
        cache_key = f"reddit/{subreddit}/comments_{_sanitize_filename(permalink)}"
        data = self._cached_get_json(url, {"limit": 200, "raw_json": 1}, cache_key)
        if data is None or not isinstance(data, list) or len(data) < 2:
            return []

        samples: list[dict] = []
        source_tag = f"reddit:r/{subreddit}"

        def _walk_comments(node: Any) -> None:
            if isinstance(node, dict):
                body = node.get("body", "").strip()
                author = node.get("author", "")
                if (body and len(body) >= 30
                        and author not in _BOT_AUTHORS
                        and not body.startswith("[")
                        and body not in ("[deleted]", "[removed]")):
                    samples.append({
                        "text": body,
                        "dialect": dialect,
                        "source": source_tag,
                        "confidence": 0.7,
                    })
                # Recurse into replies
                replies = node.get("replies")
                if isinstance(replies, dict):
                    children = replies.get("data", {}).get("children", [])
                    for child in children:
                        _walk_comments(child.get("data", {}))

        comment_listing = data[1].get("data", {}).get("children", [])
        for child in comment_listing:
            _walk_comments(child.get("data", {}))

        return samples

    # ------------------------------------------------------------------
    # Wikipedia
    # ------------------------------------------------------------------

    def scrape_wikipedia(self) -> dict[str, list[dict]]:
        """Scrape Wikipedia articles about dialect-specific topics."""
        result: dict[str, list[dict]] = {v: [] for v in ALL_VARIETIES}

        for dialect, articles in SCRAPER_WIKI_ARTICLES.items():
            for title in articles:
                try:
                    text = self._wiki_fetch_article(title)
                    if text:
                        segments = self._wiki_segment(text, title, dialect)
                        result[dialect].extend(segments)
                except Exception:
                    logger.warning("Wiki article '%s' failed", title, exc_info=True)

            logger.info("Wikipedia %s: %d samples", dialect, len(result[dialect]))

        return result

    def _wiki_fetch_article(self, title: str) -> str | None:
        """Fetch plain text of a Wikipedia article via MediaWiki API."""
        url = "https://es.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": 1,
            "format": "json",
            "formatversion": 2,
        }
        cache_key = f"wikipedia/{_sanitize_filename(title)}"
        data = self._cached_get_json(url, params, cache_key, delay=1.0)
        if data is None:
            return None

        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None
        return pages[0].get("extract", "")

    def _wiki_segment(
        self, text: str, title: str, dialect: str,
    ) -> list[dict]:
        """Split Wikipedia article into paragraph-level samples."""
        samples: list[dict] = []
        source_tag = f"wikipedia:{title}"

        # Split by double newlines (paragraph boundaries)
        paragraphs = re.split(r"\n{2,}", text)

        # Skip boilerplate sections
        skip_headers = {"Referencias", "Véase también", "Enlaces externos",
                        "Bibliografía", "Notas", "Fuentes"}

        in_skip = False
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if this is a section header to skip
            header = para.replace("=", "").strip()
            if header in skip_headers:
                in_skip = True
                continue

            # New non-skip section header resets
            if para.startswith("==") and header not in skip_headers:
                in_skip = False
                continue

            if in_skip:
                continue

            # Skip very short or list-like paragraphs
            if len(para) < 50 or para.startswith("*") or para.startswith("#"):
                continue

            samples.append({
                "text": para,
                "dialect": dialect,
                "source": source_tag,
                "confidence": 0.5,  # Wikipedia is formal, not dialectal
            })

        return samples

    # ------------------------------------------------------------------
    # OpenSubtitles
    # ------------------------------------------------------------------

    def scrape_opensubtitles(self) -> dict[str, list[dict]]:
        """Scrape subtitle text from OpenSubtitles API."""
        result: dict[str, list[dict]] = {v: [] for v in ALL_VARIETIES}

        for dialect, films in SCRAPER_FILMS.items():
            for film_title in films:
                try:
                    srt_text = self._osub_fetch_subtitle(film_title)
                    if srt_text:
                        lines = self._osub_parse_srt(srt_text)
                        samples = self._osub_lines_to_samples(lines, film_title, dialect)
                        result[dialect].extend(samples)
                except Exception:
                    logger.warning("OpenSub '%s' failed", film_title, exc_info=True)

            logger.info("OpenSubtitles %s: %d samples", dialect, len(result[dialect]))

        return result

    def _osub_fetch_subtitle(self, query: str) -> str | None:
        """Search and download a subtitle from OpenSubtitles."""
        # Search
        headers = {"Api-Key": self.api_key}
        params = {
            "languages": "es",
            "query": query,
            "ai_translated": "exclude",
            "order_by": "download_count",
            "order_direction": "desc",
        }
        cache_key = f"opensubtitles/search_{_sanitize_filename(query)}"
        search_data = self._cached_get_json(
            f"{OPENSUB_BASE_URL}/subtitles", params, cache_key,
            extra_headers=headers, delay=1.5,
        )
        if search_data is None:
            return None

        results = search_data.get("data", [])
        if not results:
            return None

        # Get file_id from first result
        file_id = None
        for item in results:
            attrs = item.get("attributes", {})
            files = attrs.get("files", [])
            if files:
                file_id = files[0].get("file_id")
                break

        if file_id is None:
            return None

        # Check cache for downloaded content
        dl_cache_key = f"opensubtitles/dl_{file_id}"
        cached = self._cache_get(dl_cache_key)
        if cached is not None:
            return cached.get("content", "")

        # Download
        dl_url = f"{OPENSUB_BASE_URL}/download"
        self._rate_limit(2.0)
        try:
            resp = self._session.post(
                dl_url,
                json={"file_id": file_id},
                headers={"Api-Key": self.api_key, "Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            dl_data = resp.json()
        except Exception:
            logger.warning("OpenSub download for file_id=%s failed", file_id)
            return None

        link = dl_data.get("link")
        if not link:
            return None

        # Fetch actual SRT content
        self._rate_limit(1.0)
        try:
            srt_resp = self._session.get(link, timeout=30)
            srt_resp.raise_for_status()
            content = srt_resp.text
        except Exception:
            return None

        self._cache_put(dl_cache_key, {"content": content})
        return content

    def _osub_parse_srt(self, content: str) -> list[str]:
        """Parse SRT content into clean dialogue lines."""
        lines: list[str] = []
        for raw_line in content.split("\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            if _SRT_INDEX_RE.match(raw_line):
                continue
            if _SRT_TIMESTAMP_RE.match(raw_line):
                continue

            # Remove HTML tags and SSA style overrides
            clean = _SRT_TAGS_RE.sub("", raw_line)
            # Remove markers like [Music], (laughing)
            clean = _SRT_MARKERS_RE.sub("", clean)
            # Remove leading dialogue dash
            clean = re.sub(r"^-\s*", "", clean)
            clean = clean.strip()

            if clean and len(clean) >= 5:
                lines.append(clean)

        return lines

    def _osub_lines_to_samples(
        self, lines: list[str], film_title: str, dialect: str,
    ) -> list[dict]:
        """Group SRT lines into multi-line samples (3-5 lines each)."""
        samples: list[dict] = []
        source_tag = f"opensubtitles:{film_title}"
        chunk_size = 4

        for i in range(0, len(lines), chunk_size):
            chunk = lines[i : i + chunk_size]
            text = " ".join(chunk)
            if len(text) >= 30:
                samples.append({
                    "text": text,
                    "dialect": dialect,
                    "source": source_tag,
                    "confidence": 0.6,
                })

        return samples

    # ------------------------------------------------------------------
    # Lyrics
    # ------------------------------------------------------------------

    def scrape_lyrics(self) -> dict[str, list[dict]]:
        """Scrape song lyrics via lyrics.ovh API."""
        result: dict[str, list[dict]] = {v: [] for v in ALL_VARIETIES}

        for dialect, songs in SCRAPER_SONGS.items():
            for artist, title in songs:
                try:
                    text = self._lyrics_fetch(artist, title)
                    if text:
                        stanzas = self._lyrics_segment(text, artist, title, dialect)
                        result[dialect].extend(stanzas)
                except Exception:
                    logger.warning("Lyrics '%s - %s' failed", artist, title, exc_info=True)

            logger.info("Lyrics %s: %d samples", dialect, len(result[dialect]))

        return result

    def _lyrics_fetch(self, artist: str, title: str) -> str | None:
        """Fetch lyrics from lyrics.ovh API."""
        cache_key = f"lyrics/{_sanitize_filename(artist)}_{_sanitize_filename(title)}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached.get("lyrics", "")

        url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
        self._rate_limit(1.5)
        try:
            resp = self._session.get(url, timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            lyrics = data.get("lyrics", "")
            self._cache_put(cache_key, {"lyrics": lyrics})
            return lyrics
        except Exception:
            return None

    def _lyrics_segment(
        self, text: str, artist: str, title: str, dialect: str,
    ) -> list[dict]:
        """Split lyrics into stanza-level samples."""
        samples: list[dict] = []
        source_tag = f"lyrics:{artist} - {title}"

        stanzas = re.split(r"\n{2,}", text)
        for stanza in stanzas:
            stanza = stanza.strip()
            if len(stanza) < 20:
                continue
            # Skip instrumental markers
            if re.match(r"^\[.*\]$", stanza):
                continue
            samples.append({
                "text": stanza,
                "dialect": dialect,
                "source": source_tag,
                "confidence": 0.75,
            })

        return samples

    # ------------------------------------------------------------------
    # HuggingFace datasets (mC4 / OSCAR by country TLD)
    # ------------------------------------------------------------------

    def scrape_huggingface(
        self,
        dataset_name: str = "mc4",
        split: str = "train",
        max_per_dialect: int = 10000,
        streaming: bool = True,
    ) -> dict[str, list[dict]]:
        """Download Spanish web text from HuggingFace, label by URL country TLD.

        Uses mC4 (Common Crawl) which includes URL metadata — we can
        automatically assign dialect based on TLD (.ar, .mx, .cl, etc.).
        Falls back to OSCAR if mC4 is unavailable.

        Streaming mode avoids downloading the entire dataset.
        """
        result: dict[str, list[dict]] = {v: [] for v in ALL_VARIETIES}
        counts: dict[str, int] = {v: 0 for v in ALL_VARIETIES}
        target = max_per_dialect

        try:
            from datasets import load_dataset

            logger.info("Loading HuggingFace dataset '%s' (streaming=%s)", dataset_name, streaming)

            if dataset_name == "mc4":
                ds = load_dataset(
                    "mc4", "es",
                    split=split,
                    streaming=streaming,
                    trust_remote_code=True,
                )
            elif dataset_name == "oscar":
                ds = load_dataset(
                    "oscar-corpus/OSCAR-2301", "es",
                    split=split,
                    streaming=streaming,
                    trust_remote_code=True,
                )
            else:
                ds = load_dataset(
                    dataset_name, "es",
                    split=split,
                    streaming=streaming,
                    trust_remote_code=True,
                )

            n_processed = 0
            for item in ds:
                # Extract text and URL
                text = item.get("text", "").strip()
                url = item.get("url", "")

                if not text or len(text) < 50:
                    continue

                # Determine dialect from URL TLD
                dialect = self._url_to_dialect(url)
                if dialect is None:
                    continue

                # Keyword-filter .es TLD: rescue CAN/AND from PEN pool
                if dialect == "ES_PEN":
                    reclassified = self._keyword_reclassify(text)
                    if reclassified is not None:
                        dialect = reclassified

                if counts[dialect] >= target:
                    # Check if all dialects are full
                    if all(c >= target for c in counts.values()):
                        break
                    continue

                # Truncate very long texts
                if len(text) > 2000:
                    text = text[:2000]

                result[dialect].append({
                    "text": text,
                    "dialect": dialect,
                    "source": f"hf:{dataset_name}",
                    "confidence": 0.7,
                })
                counts[dialect] += 1
                n_processed += 1

                if n_processed % 5000 == 0:
                    logger.info("HF progress: %s", dict(counts))

        except Exception:
            logger.exception("HuggingFace dataset '%s' failed", dataset_name)

        for dialect in ALL_VARIETIES:
            logger.info("HuggingFace %s: %d samples", dialect, len(result[dialect]))

        return result

    @staticmethod
    def _url_to_dialect(url: str) -> str | None:
        """Map a URL to a dialect code based on country TLD."""
        if not url:
            return None
        try:
            from urllib.parse import urlparse
            hostname = urlparse(url).hostname or ""
            # Extract TLD (last part after final dot)
            parts = hostname.rsplit(".", 1)
            if len(parts) >= 2:
                tld = "." + parts[-1]
                return TLD_TO_DIALECT.get(tld)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # OPUS bulk download (OpenSubtitles monolingual Spanish)
    # ------------------------------------------------------------------

    def scrape_opus_bulk(
        self, max_lines: int = 50000,
    ) -> dict[str, list[dict]]:
        """Download OPUS OpenSubtitles monolingual Spanish (bulk gzip).

        All lines assigned to ES_PEN by default (low confidence).
        This provides a large volume of general Spanish text.
        """
        import gzip
        import io

        result: dict[str, list[dict]] = {v: [] for v in ALL_VARIETIES}

        urls = [
            "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/es.txt.gz",
            "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2016/mono/es.txt.gz",
        ]

        cache_key = "opus/opensubtitles_mono_es"
        cached = self._cache_get(cache_key)
        if cached is not None:
            lines = cached.get("lines", [])
        else:
            lines = []
            for url in urls:
                try:
                    logger.info("Downloading OPUS bulk from %s", url)
                    resp = self._session.get(url, timeout=120, stream=True)
                    resp.raise_for_status()

                    content = gzip.decompress(resp.content).decode("utf-8", errors="replace")
                    lines = [l.strip() for l in content.split("\n") if l.strip()]
                    logger.info("OPUS: downloaded %d raw lines", len(lines))
                    break
                except Exception:
                    logger.warning("OPUS download from %s failed", url, exc_info=True)
                    continue

            if lines:
                # Cache only up to max_lines to avoid huge cache files
                self._cache_put(cache_key, {"lines": lines[:max_lines]})

        # Quality filter and distribute across dialects
        n_per_dialect = max_lines // len(ALL_VARIETIES)
        dialect_counts: dict[str, int] = {v: 0 for v in ALL_VARIETIES}
        variety_cycle = list(ALL_VARIETIES)
        v_idx = 0

        for line in lines:
            if len(line) < 20 or len(line) > 500:
                continue
            # Basic quality: >50% alphabetic
            alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
            if alpha_ratio < 0.5:
                continue

            # Distribute round-robin (with PEN bias for unlabeled)
            dialect = variety_cycle[v_idx % len(variety_cycle)]
            if dialect_counts[dialect] >= n_per_dialect:
                v_idx += 1
                if all(c >= n_per_dialect for c in dialect_counts.values()):
                    break
                continue

            result[dialect].append({
                "text": line,
                "dialect": dialect,
                "source": "opus:opensubtitles",
                "confidence": 0.3,
            })
            dialect_counts[dialect] += 1
            v_idx += 1

        for dialect in ALL_VARIETIES:
            logger.info("OPUS %s: %d samples", dialect, len(result[dialect]))

        return result

    # ------------------------------------------------------------------
    # Keyword-based dialect reclassification (rescues CAN/AND from .es)
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_reclassify(text: str) -> str | None:
        """Reclassify a .es-domain text to CAN or AND based on dialect markers.

        Scans text for high-precision dialect keywords. Requires ≥2 marker
        hits to reclassify (avoids false positives from single word mentions).
        Returns dialect code or None if no strong signal.
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        best_dialect: str | None = None
        best_hits = 0

        for dialect, markers in DIALECT_MARKERS.items():
            # Count single-word markers via set intersection
            single_hits = len(words & markers)

            # Count multi-word markers via substring search
            multi_hits = sum(
                1 for m in markers
                if " " in m and m in text_lower
            )

            total = single_hits + multi_hits
            if total >= 2 and total > best_hits:
                best_hits = total
                best_dialect = dialect

        return best_dialect

    # ------------------------------------------------------------------
    # Regional news (CAN + AND priority sources)
    # ------------------------------------------------------------------

    def scrape_regional_news(
        self, max_articles_per_site: int = 100,
    ) -> dict[str, list[dict]]:
        """Scrape regional news sites for Canarian and Andalusian text.

        Uses RSS feeds to discover articles, then fetches article text.
        These are the richest untapped sources for CAN and AND — written
        by locals in authentic regional Spanish.
        """
        result: dict[str, list[dict]] = {v: [] for v in ALL_VARIETIES}

        for dialect, sites in SCRAPER_NEWS_SITES.items():
            for site in sites:
                name = site["name"]
                rss_url = site["rss"]
                try:
                    articles = self._news_fetch_rss(rss_url, name, dialect, max_articles_per_site)
                    result[dialect].extend(articles)
                except Exception:
                    logger.warning("News site '%s' (%s) failed", name, dialect, exc_info=True)

            logger.info("Regional news %s: %d samples", dialect, len(result[dialect]))

        return result

    def _news_fetch_rss(
        self, rss_url: str, site_name: str, dialect: str,
        max_articles: int,
    ) -> list[dict]:
        """Fetch and parse an RSS feed, extract article text."""
        cache_key = f"news/{_sanitize_filename(site_name)}_rss"
        cached = self._cache_get(cache_key)

        if cached is not None:
            items = cached.get("items", [])
        else:
            self._rate_limit(1.5)
            try:
                resp = self._session.get(rss_url, timeout=20, headers={
                    "Accept": "application/rss+xml, application/xml, text/xml",
                })
                resp.raise_for_status()
                items = self._parse_rss_xml(resp.text)
                self._cache_put(cache_key, {"items": items[:max_articles]})
            except Exception:
                logger.warning("RSS fetch failed for %s", site_name, exc_info=True)
                return []

        samples: list[dict] = []
        source_tag = f"news:{site_name}"

        for item in items[:max_articles]:
            title = item.get("title", "").strip()
            description = item.get("description", "").strip()
            link = item.get("link", "")

            # Use description as primary text (usually contains article summary)
            text = description
            if not text or len(text) < 40:
                text = title
            if not text or len(text) < 30:
                continue

            # Try to fetch full article text for richer content
            if link and len(text) < 200:
                full_text = self._news_fetch_article(link, site_name)
                if full_text and len(full_text) > len(text):
                    text = full_text

            # Clean HTML tags
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) >= 30:
                samples.append({
                    "text": text[:2000],
                    "dialect": dialect,
                    "source": source_tag,
                    "confidence": 0.8,  # High: written by locals
                })

        return samples

    def _news_fetch_article(self, url: str, site_name: str) -> str | None:
        """Fetch full article text from a news URL."""
        cache_key = f"news/{_sanitize_filename(site_name)}_{hashlib.md5(url.encode()).hexdigest()[:12]}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached.get("text", "")

        self._rate_limit(1.5)
        try:
            resp = self._session.get(url, timeout=15, headers={
                "Accept": "text/html",
            })
            resp.raise_for_status()
            text = self._extract_article_text(resp.text)
            if text:
                self._cache_put(cache_key, {"text": text})
            return text
        except Exception:
            return None

    @staticmethod
    def _extract_article_text(html: str) -> str | None:
        """Extract article body text from HTML using simple heuristics.

        Looks for <article> or <p> tags within article containers.
        Falls back to extracting all <p> content.
        """
        # Try to find <article> content first
        article_match = re.search(
            r"<article[^>]*>(.*?)</article>", html, re.DOTALL | re.IGNORECASE,
        )
        if article_match:
            html_chunk = article_match.group(1)
        else:
            html_chunk = html

        # Extract text from <p> tags
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html_chunk, re.DOTALL | re.IGNORECASE)
        if not paragraphs:
            return None

        # Clean HTML tags from paragraphs
        clean_paras: list[str] = []
        for p in paragraphs:
            text = re.sub(r"<[^>]+>", "", p)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) >= 20:
                clean_paras.append(text)

        if not clean_paras:
            return None

        return " ".join(clean_paras)[:2000]

    @staticmethod
    def _parse_rss_xml(xml_text: str) -> list[dict[str, str]]:
        """Parse RSS/Atom XML into list of {title, description, link} dicts.

        Uses regex-based parsing to avoid requiring xml.etree for simple RSS.
        """
        items: list[dict[str, str]] = []

        # Match RSS <item> or Atom <entry> blocks
        item_pattern = re.compile(
            r"<(?:item|entry)[^>]*>(.*?)</(?:item|entry)>", re.DOTALL | re.IGNORECASE,
        )

        for match in item_pattern.finditer(xml_text):
            block = match.group(1)

            title = ""
            title_m = re.search(r"<title[^>]*>(.*?)</title>", block, re.DOTALL | re.IGNORECASE)
            if title_m:
                title = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", title_m.group(1)).strip()

            desc = ""
            for tag in ("description", "summary", "content"):
                desc_m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, re.DOTALL | re.IGNORECASE)
                if desc_m:
                    desc = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", desc_m.group(1)).strip()
                    break

            link = ""
            link_m = re.search(r"<link[^>]*>(.*?)</link>", block, re.DOTALL | re.IGNORECASE)
            if link_m:
                link = link_m.group(1).strip()
            else:
                # Atom-style: <link href="..."/>
                link_m = re.search(r'<link[^>]*href="([^"]*)"', block, re.IGNORECASE)
                if link_m:
                    link = link_m.group(1).strip()

            items.append({"title": title, "description": desc, "link": link})

        return items

    # ------------------------------------------------------------------
    # HTTP infrastructure
    # ------------------------------------------------------------------

    def _rate_limit(self, delay: float | None = None) -> None:
        """Enforce minimum delay between requests."""
        d = delay if delay is not None else self.delay
        elapsed = time.time() - self._last_request_time
        if elapsed < d:
            time.sleep(d - elapsed)
        self._last_request_time = time.time()

    def _cached_get_json(
        self,
        url: str,
        params: dict | None = None,
        cache_key: str | None = None,
        extra_headers: dict | None = None,
        delay: float | None = None,
        max_retries: int = 3,
    ) -> dict | list | None:
        """GET JSON with caching, rate limiting, and retry."""
        # Check cache first
        if cache_key:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        self._rate_limit(delay)

        headers = dict(self._session.headers)
        if extra_headers:
            headers.update(extra_headers)

        for attempt in range(max_retries):
            try:
                resp = self._session.get(url, params=params, headers=headers, timeout=30)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    logger.warning("Rate limited, waiting %ds", retry_after)
                    time.sleep(retry_after)
                    continue

                if resp.status_code in (403, 404):
                    logger.debug("HTTP %d for %s", resp.status_code, url)
                    return None

                resp.raise_for_status()
                data = resp.json()

                if cache_key:
                    self._cache_put(cache_key, data)
                return data

            except requests.exceptions.ConnectionError:
                wait = 2 ** attempt
                logger.warning("Connection error (attempt %d), retrying in %ds", attempt + 1, wait)
                time.sleep(wait)
            except Exception:
                if attempt == max_retries - 1:
                    logger.warning("Failed after %d attempts: %s", max_retries, url, exc_info=True)
                    return None
                time.sleep(2 ** attempt)

        return None

    # ------------------------------------------------------------------
    # File-based cache
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> dict | None:
        """Read from file-based JSON cache."""
        path = self.cache_dir / f"{_sanitize_filename(key)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            path.unlink(missing_ok=True)
            return None

    def _cache_put(self, key: str, data: Any) -> None:
        """Write to file-based JSON cache."""
        path = self.cache_dir / f"{_sanitize_filename(key)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError:
            logger.warning("Cache write failed for %s", key)
