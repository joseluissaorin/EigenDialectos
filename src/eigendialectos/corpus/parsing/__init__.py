"""Multi-level parsing for Spanish dialectal text.

This sub-package decomposes raw text into 5 linguistic levels for
downstream spectral transformation in the EigenDialectos v2 pipeline:

    L1 -- Morpheme segmentation
    L2 -- Word tokenization
    L3 -- Phrase chunking
    L4 -- Sentence splitting
    L5 -- Discourse feature extraction
"""

from __future__ import annotations

from eigendialectos.corpus.parsing.morpheme_parser import parse_morphemes
from eigendialectos.corpus.parsing.phrase_parser import parse_phrases
from eigendialectos.corpus.parsing.discourse_parser import parse_discourse
from eigendialectos.corpus.parsing.multi_level import MultiLevelParser

__all__ = [
    "parse_morphemes",
    "parse_phrases",
    "parse_discourse",
    "MultiLevelParser",
]
