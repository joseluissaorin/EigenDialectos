"""Multi-level parser orchestrating all 5 linguistic levels.

Combines morpheme segmentation (L1), tokenization (L2), phrase chunking (L3),
sentence splitting (L4), and discourse feature extraction (L5) into a single
:class:`ParsedText` structure for downstream spectral transformation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Sequence

from eigendialectos.types import ParsedText, DialectSample
from eigendialectos.corpus.preprocessing.segmentation import split_sentences
from eigendialectos.corpus.parsing.morpheme_parser import parse_morphemes
from eigendialectos.corpus.parsing.phrase_parser import parse_phrases
from eigendialectos.corpus.parsing.discourse_parser import parse_discourse


# ======================================================================
# Tokenizer
# ======================================================================

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Split *text* into word tokens using a Unicode-aware regex."""
    return _TOKEN_RE.findall(text)


# ======================================================================
# MultiLevelParser
# ======================================================================

class MultiLevelParser:
    """Orchestrates parsing of Spanish text across 5 linguistic levels.

    Levels
    ------
    L1 -- Morpheme segmentation (per token)
    L2 -- Word tokens
    L3 -- Phrase chunks (NP / VP / PP groups)
    L4 -- Sentences
    L5 -- Discourse features (statistics, markers, formality)
    """

    def parse(self, text: str) -> ParsedText:
        """Parse *text* into all 5 linguistic levels.

        Parameters
        ----------
        text:
            Raw Spanish text (can be multi-sentence / multi-paragraph).

        Returns
        -------
        ParsedText
            Dataclass holding the original text and all 5 level
            representations.
        """
        if not text or not text.strip():
            return ParsedText(
                original=text,
                morphemes=[],
                words=[],
                phrases=[],
                sentences=[],
                discourse={
                    "n_sentences": 0,
                    "avg_sentence_length": 0.0,
                    "subordination_ratio": 0.0,
                    "question_ratio": 0.0,
                    "exclamation_ratio": 0.0,
                    "discourse_markers": [],
                    "marker_density": 0.0,
                    "formality_score": 0.5,
                },
            )

        # L4: Sentence segmentation
        sentences = split_sentences(text)

        # L2: Word tokenization (full text)
        words = _tokenize(text)

        # L1: Morpheme segmentation (over the token list)
        morphemes = parse_morphemes(words)

        # L3: Phrase chunking (over the token list)
        phrases = parse_phrases(words)

        # L5: Discourse features
        discourse = parse_discourse(text)

        return ParsedText(
            original=text,
            morphemes=morphemes,
            words=words,
            phrases=phrases,
            sentences=sentences,
            discourse=discourse,
        )

    def parse_batch(self, texts: Sequence[str]) -> list[ParsedText]:
        """Parse a batch of texts.

        Parameters
        ----------
        texts:
            Iterable of raw Spanish text strings.

        Returns
        -------
        list[ParsedText]
            One ParsedText per input string.
        """
        return [self.parse(t) for t in texts]

    def parse_corpus(
        self,
        samples: list[DialectSample],
    ) -> dict[str, list[ParsedText]]:
        """Parse a list of :class:`DialectSample` objects grouped by dialect.

        Parameters
        ----------
        samples:
            Corpus samples, each carrying ``.text`` and ``.dialect_code``.

        Returns
        -------
        dict[str, list[ParsedText]]
            Mapping from dialect code string to list of parsed texts for
            that dialect.
        """
        grouped: dict[str, list[ParsedText]] = defaultdict(list)
        for sample in samples:
            parsed = self.parse(sample.text)
            # Use the string value of the dialect code as key
            dialect_key = (
                sample.dialect_code.value
                if hasattr(sample.dialect_code, "value")
                else str(sample.dialect_code)
            )
            grouped[dialect_key].append(parsed)
        return dict(grouped)

    def summary(self, parsed: ParsedText) -> dict[str, Any]:
        """Return a compact statistical summary of a parsed text.

        Useful for quick inspection and logging.

        Parameters
        ----------
        parsed:
            A previously parsed text.

        Returns
        -------
        dict[str, Any]
            Summary statistics across all 5 levels.
        """
        n_tokens = len(parsed.words)
        n_morphemes_total = sum(len(m) for m in parsed.morphemes)
        avg_morphemes_per_token = (
            n_morphemes_total / n_tokens if n_tokens > 0 else 0.0
        )
        n_phrases = len(parsed.phrases)
        avg_phrase_length = (
            sum(len(p) for p in parsed.phrases) / n_phrases
            if n_phrases > 0
            else 0.0
        )

        return {
            "n_tokens": n_tokens,
            "n_morphemes_total": n_morphemes_total,
            "avg_morphemes_per_token": round(avg_morphemes_per_token, 3),
            "n_phrases": n_phrases,
            "avg_phrase_length": round(avg_phrase_length, 3),
            "n_sentences": parsed.discourse.get("n_sentences", 0),
            "formality_score": parsed.discourse.get("formality_score", 0.5),
            "n_discourse_markers": len(
                parsed.discourse.get("discourse_markers", [])
            ),
        }
