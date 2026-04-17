"""Text segmentation utilities for splitting text into sentences, paragraphs,
or fixed-length chunks.

Designed for Spanish text with proper handling of abbreviations, numbers,
and inverted punctuation marks.
"""

from __future__ import annotations

import re

# ======================================================================
# Common Spanish abbreviations that end with a period but are NOT
# sentence boundaries.
# ======================================================================

_ABBREVIATIONS: set[str] = {
    "sr", "sra", "srta", "dr", "dra", "prof", "ing", "lic", "arq",
    "ud", "uds", "vd", "vds", "etc", "pág", "págs", "vol", "vols",
    "cap", "núm", "tel", "av", "avda", "cta", "dpto", "dept",
    "ed", "ej", "aprox", "máx", "mín", "fig", "gral", "admón",
    "ee", "uu",  # EE. UU.
}

# Sentence-ending punctuation
_SENT_END_RE = re.compile(
    r'(?<=[.!?…¡¿])'      # lookbehind: sentence-ending punct
    r'(?:\s*["\'\)\]»]*)'  # optional closing quotes / brackets
    r'\s+'                 # required whitespace between sentences
    r'(?='                 # lookahead: new sentence starts with
    r'[A-ZÁÉÍÓÚÑ¿¡"\'(\[«]'  # uppercase / opening punct
    r')',
    re.UNICODE,
)

# Paragraph boundary: two or more newlines
_PARA_RE = re.compile(r'\n\s*\n')


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences.

    Uses a regex heuristic that respects common Spanish abbreviations
    and inverted question/exclamation marks.

    Returns
    -------
    list[str]
        Non-empty sentences with leading/trailing whitespace stripped.
    """
    if not text or not text.strip():
        return []

    # Protect abbreviations by temporarily replacing their trailing dot
    protected = text
    for abbr in _ABBREVIATIONS:
        # Match abbreviation + period (case-insensitive)
        pattern = re.compile(
            r'\b(' + re.escape(abbr) + r')\.',
            re.IGNORECASE,
        )
        protected = pattern.sub(lambda m: m.group(1) + '\x00', protected)

    # Split on sentence boundaries
    raw_parts = _SENT_END_RE.split(protected)

    # Restore protected dots
    sentences = [part.replace('\x00', '.').strip() for part in raw_parts]

    # Filter empty strings
    return [s for s in sentences if s]


def split_paragraphs(text: str) -> list[str]:
    """Split *text* into paragraphs (separated by blank lines).

    Returns
    -------
    list[str]
        Non-empty paragraphs with leading/trailing whitespace stripped.
    """
    if not text or not text.strip():
        return []
    parts = _PARA_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_text(text: str, max_length: int) -> list[str]:
    """Split *text* into chunks of at most *max_length* characters.

    Tries to break at sentence boundaries first, then at word boundaries.
    """
    if len(text) <= max_length:
        return [text]

    sentences = split_sentences(text)
    if not sentences:
        # Fallback: split on word boundaries
        return _chunk_by_words(text, max_length)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        # If a single sentence exceeds max_length, split it by words
        if sent_len > max_length:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            chunks.extend(_chunk_by_words(sent, max_length))
            continue

        if current_len + sent_len + (1 if current else 0) > max_length:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))

    return chunks


def _chunk_by_words(text: str, max_length: int) -> list[str]:
    """Break text at word boundaries to fit *max_length*."""
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        if word_len > max_length:
            # Very long token -- force-split
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            for i in range(0, word_len, max_length):
                chunks.append(word[i : i + max_length])
            continue

        if current_len + word_len + (1 if current else 0) > max_length:
            chunks.append(" ".join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len += word_len + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))

    return chunks


def segment_text(
    text: str,
    method: str = "sentence",
    max_length: int = 256,
) -> list[str]:
    """Segment *text* using the specified method.

    Parameters
    ----------
    text:
        Input text to segment.
    method:
        ``"sentence"`` -- split into sentences.
        ``"paragraph"`` -- split into paragraphs.
        ``"chunk"`` -- split into chunks of at most *max_length* chars,
        breaking at sentence/word boundaries.
    max_length:
        Maximum segment length in characters (only used for ``"chunk"``).

    Returns
    -------
    list[str]
        Non-empty segments.
    """
    if method == "sentence":
        return split_sentences(text)
    if method == "paragraph":
        return split_paragraphs(text)
    if method == "chunk":
        return _chunk_text(text, max_length)
    raise ValueError(
        f"Unknown segmentation method '{method}'. "
        "Choose from: 'sentence', 'paragraph', 'chunk'."
    )
