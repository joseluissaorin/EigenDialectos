"""External corpus data sources."""

from eigendialectos.corpus.sources.corpes_xxi import CorpesXXISource
from eigendialectos.corpus.sources.corpus_del_espanol import CorpusDelEspanolSource
from eigendialectos.corpus.sources.podcasts import PodcastSource
from eigendialectos.corpus.sources.subtitles import SubtitlesSource
from eigendialectos.corpus.sources.twitter import TwitterSource
from eigendialectos.corpus.sources.wikipedia import WikipediaSource

# Auto-register all sources
from eigendialectos.corpus.registry import register_source as _register

_ALL_SOURCES: dict[str, type] = {
    "subtitles": SubtitlesSource,
    "twitter": TwitterSource,
    "podcasts": PodcastSource,
    "corpes_xxi": CorpesXXISource,
    "corpus_del_espanol": CorpusDelEspanolSource,
    "wikipedia": WikipediaSource,
}

for _name, _cls in _ALL_SOURCES.items():
    try:
        _register(_name, _cls)
    except ValueError:
        pass  # already registered

__all__ = [
    "CorpesXXISource",
    "CorpusDelEspanolSource",
    "PodcastSource",
    "SubtitlesSource",
    "TwitterSource",
    "WikipediaSource",
]
