"""Text preprocessing utilities for corpus cleaning and labelling."""

from eigendialectos.corpus.preprocessing.filters import (
    apply_filters,
    confidence_filter,
    dedup_filter,
    language_filter,
    max_length_filter,
    min_length_filter,
    near_dedup_filter,
    quality_filter,
)
from eigendialectos.corpus.preprocessing.labeling import DialectLabeler
from eigendialectos.corpus.preprocessing.noise import (
    clean_text,
    collapse_whitespace,
    fix_encoding,
    handle_emojis,
    normalize_repetitions,
    normalize_unicode,
    remove_hashtags,
    remove_mentions,
    remove_urls,
)
from eigendialectos.corpus.preprocessing.segmentation import (
    segment_text,
    split_paragraphs,
    split_sentences,
)

__all__ = [
    "DialectLabeler",
    "apply_filters",
    "clean_text",
    "collapse_whitespace",
    "confidence_filter",
    "dedup_filter",
    "fix_encoding",
    "handle_emojis",
    "language_filter",
    "max_length_filter",
    "min_length_filter",
    "near_dedup_filter",
    "normalize_repetitions",
    "normalize_unicode",
    "quality_filter",
    "remove_hashtags",
    "remove_mentions",
    "remove_urls",
    "segment_text",
    "split_paragraphs",
    "split_sentences",
]
