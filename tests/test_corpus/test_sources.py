"""Tests for corpus data sources: registry, base class, and concrete sources."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.corpus.base import CorpusSource
from eigendialectos.corpus.registry import (
    clear_registry,
    get_source,
    list_available,
    register_source,
)
from eigendialectos.types import DialectSample


# ======================================================================
# Dummy source for testing
# ======================================================================


class DummySource(CorpusSource):
    """Minimal concrete source for testing the base class and registry."""

    def download(self, output_dir: Path) -> Path:
        return output_dir / "dummy.jsonl"

    def load(self, path: Path) -> Iterator[DialectSample]:
        yield DialectSample(
            text="Hola mundo.",
            dialect_code=DialectCode.ES_PEN,
            source_id="dummy",
            confidence=1.0,
        )

    def dialect_codes(self) -> list[DialectCode]:
        return [DialectCode.ES_PEN, DialectCode.ES_MEX]

    def citation(self) -> str:
        return "DummySource (2024). Fictional corpus."


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


# ======================================================================
# Abstract source interface tests
# ======================================================================


class TestCorpusSourceInterface:
    """Tests for the abstract CorpusSource contract."""

    def test_instantiation(self):
        src = DummySource()
        assert isinstance(src, CorpusSource)

    def test_dialect_codes_nonempty(self):
        src = DummySource()
        codes = src.dialect_codes()
        assert len(codes) > 0
        for c in codes:
            assert isinstance(c, DialectCode)

    def test_citation_nonempty(self):
        src = DummySource()
        cit = src.citation()
        assert isinstance(cit, str)
        assert len(cit) > 0

    def test_name(self):
        src = DummySource()
        assert src.name() == "DummySource"

    def test_repr(self):
        src = DummySource()
        rep = repr(src)
        assert "DummySource" in rep
        assert "ES_PEN" in rep

    def test_load_yields_samples(self):
        src = DummySource()
        samples = list(src.load(Path("/tmp/nonexistent")))
        assert len(samples) == 1
        assert samples[0].text == "Hola mundo."

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            CorpusSource()  # type: ignore[abstract]


# ======================================================================
# Registry tests
# ======================================================================


class TestSourceRegistry:
    """Tests for the source registration system."""

    def test_register_and_get(self):
        register_source("dummy", DummySource)
        src = get_source("dummy")
        assert isinstance(src, DummySource)

    def test_list_available(self):
        register_source("dummy_a", DummySource)
        register_source("dummy_b", DummySource)
        available = list_available()
        assert "dummy_a" in available
        assert "dummy_b" in available

    def test_list_available_sorted(self):
        register_source("z_source", DummySource)
        register_source("a_source", DummySource)
        available = list_available()
        assert available == sorted(available)

    def test_unknown_source_raises(self):
        with pytest.raises(KeyError, match="Unknown source"):
            get_source("nonexistent")

    def test_duplicate_registration_raises(self):
        register_source("dup", DummySource)
        with pytest.raises(ValueError, match="already registered"):
            register_source("dup", DummySource)

    def test_non_subclass_raises(self):
        with pytest.raises(TypeError, match="CorpusSource subclass"):
            register_source("bad", int)  # type: ignore[arg-type]

    def test_clear_registry(self):
        register_source("temp", DummySource)
        assert "temp" in list_available()
        clear_registry()
        assert "temp" not in list_available()


# ======================================================================
# Concrete source instantiation tests
# ======================================================================


class TestConcreteSourcesInstantiable:
    """Test that all concrete sources can be instantiated and have valid interfaces."""

    def _get_source_classes(self):
        """Import and return all concrete source classes."""
        from eigendialectos.corpus.sources.subtitles import SubtitlesSource
        from eigendialectos.corpus.sources.twitter import TwitterSource
        from eigendialectos.corpus.sources.podcasts import PodcastSource
        from eigendialectos.corpus.sources.corpes_xxi import CorpesXXISource
        from eigendialectos.corpus.sources.corpus_del_espanol import CorpusDelEspanolSource
        from eigendialectos.corpus.sources.wikipedia import WikipediaSource

        return [
            SubtitlesSource,
            TwitterSource,
            PodcastSource,
            CorpesXXISource,
            CorpusDelEspanolSource,
            WikipediaSource,
        ]

    def test_all_sources_instantiable(self):
        for cls in self._get_source_classes():
            src = cls()
            assert isinstance(src, CorpusSource)

    def test_all_sources_have_dialect_codes(self):
        for cls in self._get_source_classes():
            src = cls()
            codes = src.dialect_codes()
            assert isinstance(codes, list)
            assert len(codes) > 0
            for c in codes:
                assert isinstance(c, DialectCode)

    def test_all_sources_have_citation(self):
        for cls in self._get_source_classes():
            src = cls()
            cit = src.citation()
            assert isinstance(cit, str)
            assert len(cit) > 10

    def test_all_sources_download_creates_dir(self):
        for cls in self._get_source_classes():
            src = cls()
            with tempfile.TemporaryDirectory() as tmpdir:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    path = src.download(Path(tmpdir))
                assert isinstance(path, Path)

    def test_all_sources_load_on_missing_dir(self):
        """Loading from a non-existent path should return empty iterator."""
        for cls in self._get_source_classes():
            src = cls()
            samples = list(src.load(Path("/tmp/nonexistent_dir_12345")))
            assert samples == []

    def test_all_sources_load_on_empty_dir(self):
        """Loading from an empty directory should return empty iterator."""
        for cls in self._get_source_classes():
            src = cls()
            with tempfile.TemporaryDirectory() as tmpdir:
                samples = list(src.load(Path(tmpdir)))
                assert samples == []


# ======================================================================
# Source-specific tests with mock data
# ======================================================================


class TestSubtitlesSource:
    """Test SubtitlesSource with mock SRT data."""

    def test_load_srt_file(self):
        from eigendialectos.corpus.sources.subtitles import SubtitlesSource

        src = SubtitlesSource()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock SRT in an 'ar' directory
            ar_dir = Path(tmpdir) / "ar"
            ar_dir.mkdir()
            srt_content = (
                "1\n"
                "00:00:01,000 --> 00:00:03,000\n"
                "Che, vos sabés que hoy es viernes.\n"
                "\n"
                "2\n"
                "00:00:04,000 --> 00:00:06,000\n"
                "Dale, vamos al bar.\n"
                "\n"
            )
            (ar_dir / "test_movie.srt").write_text(srt_content, encoding="utf-8")
            samples = list(src.load(Path(tmpdir)))
            assert len(samples) >= 1
            for s in samples:
                assert s.dialect_code == DialectCode.ES_RIO


class TestTwitterSourceMock:
    """Test TwitterSource with mock JSONL data."""

    def test_load_jsonl(self):
        from eigendialectos.corpus.sources.twitter import TwitterSource

        src = TwitterSource()
        with tempfile.TemporaryDirectory() as tmpdir:
            region_dir = Path(tmpdir) / "argentina"
            region_dir.mkdir()
            tweet = {"text": "Che boludo, hoy fui a laburar al centro.", "id": "123"}
            (region_dir / "tweets.jsonl").write_text(
                json.dumps(tweet) + "\n", encoding="utf-8",
            )
            samples = list(src.load(Path(tmpdir)))
            assert len(samples) >= 1
            for s in samples:
                assert s.dialect_code == DialectCode.ES_RIO


class TestWikipediaSourceMock:
    """Test WikipediaSource with mock pre-downloaded JSON data."""

    def test_load_json(self):
        from eigendialectos.corpus.sources.wikipedia import WikipediaSource

        src = WikipediaSource()
        with tempfile.TemporaryDirectory() as tmpdir:
            region_dir = Path(tmpdir) / "argentina"
            region_dir.mkdir()
            article = {
                "title": "Buenos Aires",
                "text": (
                    "Buenos Aires es la capital de Argentina.\n\n"
                    "La ciudad fue fundada en el siglo XVI por los colonizadores españoles."
                ),
            }
            (region_dir / "buenos_aires.json").write_text(
                json.dumps(article, ensure_ascii=False), encoding="utf-8",
            )
            samples = list(src.load(Path(tmpdir)))
            assert len(samples) >= 1
            for s in samples:
                assert s.dialect_code == DialectCode.ES_RIO


# ======================================================================
# Auto-registration test
# ======================================================================


class TestAutoRegistration:
    """Test that importing sources.__init__ auto-registers all sources."""

    def test_auto_registration(self):
        # Re-import to trigger auto-registration
        import importlib
        import eigendialectos.corpus.sources
        importlib.reload(eigendialectos.corpus.sources)

        available = list_available()
        expected_names = [
            "subtitles", "twitter", "podcasts",
            "corpes_xxi", "corpus_del_espanol", "wikipedia",
        ]
        for name in expected_names:
            assert name in available, f"Source '{name}' not auto-registered"
