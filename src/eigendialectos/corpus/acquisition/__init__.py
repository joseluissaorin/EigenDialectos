"""Corpus acquisition modules for fetching real-world dialect data."""

from eigendialectos.corpus.acquisition.enhanced_synthetic import EnhancedSyntheticGenerator
from eigendialectos.corpus.acquisition.lyrics_fetcher import LyricsFetcher
from eigendialectos.corpus.acquisition.opensubtitles_fetcher import OpenSubtitlesFetcher
from eigendialectos.corpus.acquisition.opus_fetcher import OPUSFetcher
from eigendialectos.corpus.acquisition.reddit_fetcher import RedditFetcher
from eigendialectos.corpus.acquisition.web_scraper import WebScraper
from eigendialectos.corpus.acquisition.wikipedia_fetcher import WikipediaFetcher

__all__ = [
    "EnhancedSyntheticGenerator",
    "LyricsFetcher",
    "OpenSubtitlesFetcher",
    "OPUSFetcher",
    "RedditFetcher",
    "WebScraper",
    "WikipediaFetcher",
]
