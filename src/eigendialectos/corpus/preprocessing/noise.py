"""Text noise cleaning and normalisation utilities.

All functions operate on plain strings and use only the standard library.
"""

from __future__ import annotations

import re
import unicodedata

# ======================================================================
# Regex patterns (compiled once at module level)
# ======================================================================

_URL_RE = re.compile(
    r'https?://\S+|www\.\S+', re.IGNORECASE,
)
_MENTION_RE = re.compile(r'@\w+')
_HASHTAG_RE = re.compile(r'#\w+')
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002702-\U000027B0"
    "]+",
    flags=re.UNICODE,
)
_REPEATED_CHARS_RE = re.compile(r'(.)\1{2,}')
_MULTI_SPACE_RE = re.compile(r'[ \t]+')
_MULTI_NEWLINE_RE = re.compile(r'\n{3,}')


# ======================================================================
# Public API
# ======================================================================


def normalize_unicode(text: str) -> str:
    """Normalise to NFC form and strip non-printable control characters.

    Preserves standard whitespace (space, tab, newline).
    """
    text = unicodedata.normalize("NFC", text)
    # Remove control characters except \t, \n, \r
    text = "".join(
        ch for ch in text
        if ch in ("\t", "\n", "\r") or not unicodedata.category(ch).startswith("C")
    )
    return text


def remove_urls(text: str) -> str:
    """Replace URLs with a single space."""
    return _URL_RE.sub(" ", text)


def remove_mentions(text: str) -> str:
    """Remove @mentions."""
    return _MENTION_RE.sub("", text)


def remove_hashtags(text: str) -> str:
    """Remove #hashtags."""
    return _HASHTAG_RE.sub("", text)


def handle_emojis(text: str, mode: str = "remove") -> str:
    """Handle emoji characters.

    Parameters
    ----------
    text:
        Input string.
    mode:
        ``"remove"`` strips emojis, ``"replace"`` substitutes them with
        ``<EMOJI>``, ``"keep"`` leaves them untouched.
    """
    if mode == "keep":
        return text
    if mode == "replace":
        return _EMOJI_RE.sub("<EMOJI>", text)
    # default: remove
    return _EMOJI_RE.sub("", text)


def normalize_repetitions(text: str) -> str:
    """Collapse runs of 3+ identical characters to 2.

    E.g. ``"hoooolaaaa"`` becomes ``"hoolaa"``.
    """
    return _REPEATED_CHARS_RE.sub(r'\1\1', text)


def fix_encoding(text: str) -> str:
    """Attempt to fix common encoding artefacts.

    Handles mojibake patterns frequently seen when UTF-8 text is read as
    Latin-1 (or vice versa).
    """
    replacements: dict[str, str] = {
        "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
        "Ã±": "ñ", "Ã¼": "ü",
        "Ã\x81": "Á", "Ã\x89": "É", "Ã\x8d": "Í", "Ã\x93": "Ó",
        "Ã\x9a": "Ú", "Ã\x91": "Ñ", "Ã\x9c": "Ü",
        "â\x80\x93": "–", "â\x80\x94": "—",
        "â\x80\x98": "'", "â\x80\x99": "'",
        "â\x80\x9c": "\u201c", "â\x80\x9d": "\u201d",
        "â\x80¦": "...",
        "Â¡": "¡", "Â¿": "¿", "Â·": "·",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Try round-trip fix: encode as latin-1, decode as utf-8
    try:
        candidate = text.encode("latin-1").decode("utf-8")
        # Accept only if it reduced non-ASCII mojibake
        if sum(1 for c in candidate if ord(c) > 127) <= sum(
            1 for c in text if ord(c) > 127
        ):
            text = candidate
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    return text


def collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs to one, multiple newlines to two."""
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str, **options: object) -> str:
    """Apply a full cleaning pipeline.

    Keyword arguments toggle individual steps (all default to ``True``
    except ``emoji_mode`` which defaults to ``"remove"``):

    - ``unicode``: normalise Unicode
    - ``encoding``: fix encoding artefacts
    - ``urls``: remove URLs
    - ``mentions``: remove @mentions
    - ``hashtags``: remove #hashtags
    - ``emoji_mode``: ``"remove"`` / ``"replace"`` / ``"keep"``
    - ``repetitions``: normalise repeated characters
    - ``whitespace``: collapse whitespace
    """
    if options.get("unicode", True):
        text = normalize_unicode(text)
    if options.get("encoding", True):
        text = fix_encoding(text)
    if options.get("urls", True):
        text = remove_urls(text)
    if options.get("mentions", True):
        text = remove_mentions(text)
    if options.get("hashtags", True):
        text = remove_hashtags(text)

    emoji_mode = str(options.get("emoji_mode", "remove"))
    text = handle_emojis(text, mode=emoji_mode)

    if options.get("repetitions", True):
        text = normalize_repetitions(text)
    if options.get("whitespace", True):
        text = collapse_whitespace(text)

    return text
