"""Tests for multi-level parsing system."""

from __future__ import annotations

import pytest

from eigendialectos.corpus.parsing.morpheme_parser import parse_morphemes
from eigendialectos.corpus.parsing.phrase_parser import parse_phrases
from eigendialectos.corpus.parsing.discourse_parser import parse_discourse
from eigendialectos.corpus.parsing.multi_level import MultiLevelParser
from eigendialectos.types import ParsedText


class TestMorphemeParser:
    def test_verb_conjugation(self):
        result = parse_morphemes(["hablamos"])
        assert len(result) == 1
        assert len(result[0]) >= 2  # should split into stem + suffix

    def test_diminutive(self):
        result = parse_morphemes(["casita"])
        assert len(result[0]) >= 2
        assert any("it" in m for m in result[0])

    def test_mente_adverb(self):
        result = parse_morphemes(["rápidamente"])
        morphs = result[0]
        assert "mente" in morphs

    def test_clitic_chain(self):
        result = parse_morphemes(["dámelo"])
        morphs = result[0]
        assert len(morphs) >= 3  # stem + me + lo

    def test_short_word_unchanged(self):
        result = parse_morphemes(["a"])
        assert result[0] == ["a"]

    def test_multiple_words(self):
        result = parse_morphemes(["el", "gato", "come"])
        assert len(result) == 3


class TestPhraseParser:
    def test_basic_chunking(self):
        tokens = ["El", "gato", "come", "pescado"]
        result = parse_phrases(tokens)
        assert len(result) >= 1
        # All tokens should appear somewhere
        flat = [t for phrase in result for t in phrase]
        assert set(flat) == set(tokens)

    def test_prepositional_phrase(self):
        tokens = ["Vamos", "a", "la", "playa"]
        result = parse_phrases(tokens)
        assert len(result) >= 2

    def test_empty_input(self):
        result = parse_phrases([])
        assert result == []


class TestDiscourseParser:
    def test_basic_features(self):
        text = "¿Puedes decirme dónde está? Vamos a la playa."
        result = parse_discourse(text)
        assert "n_sentences" in result
        assert "question_ratio" in result
        assert "formality_score" in result
        assert result["n_sentences"] == 2
        assert result["question_ratio"] == 0.5

    def test_discourse_markers(self):
        text = "Bueno, pues vamos a ver. O sea, es complicado."
        result = parse_discourse(text)
        assert len(result["discourse_markers"]) >= 2

    def test_subordination(self):
        text = "Creo que el chico que vino ayer es el que buscamos."
        result = parse_discourse(text)
        assert result["subordination_ratio"] > 0


class TestMultiLevelParser:
    def test_parse_returns_parsed_text(self):
        parser = MultiLevelParser()
        result = parser.parse("El gato come pescado.")
        assert isinstance(result, ParsedText)
        assert result.original == "El gato come pescado."
        assert len(result.words) >= 4
        assert len(result.morphemes) == len(result.words)
        assert len(result.sentences) >= 1

    def test_parse_spanish_sentence(self):
        parser = MultiLevelParser()
        result = parser.parse("El autobús llega a las tres y media, ¿no?")
        assert "autobús" in result.words or "autobús" in [w for w in result.words]
        assert result.discourse["question_ratio"] > 0

    def test_empty_text(self):
        parser = MultiLevelParser()
        result = parser.parse("")
        assert result.original == ""
        assert result.words == []

    def test_consistency(self):
        parser = MultiLevelParser()
        result = parser.parse("La chica camina por la calle.")
        # Morphemes and words should have same count
        assert len(result.morphemes) == len(result.words)
