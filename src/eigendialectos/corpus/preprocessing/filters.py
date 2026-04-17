"""Sample filtering utilities for corpus quality control.

All filter functions accept an iterable of :class:`DialectSample` and
return a list of samples that pass the filter.  ``apply_filters`` chains
multiple filters according to a configuration dict.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Optional

from eigendialectos.types import DialectSample

# ======================================================================
# Basic Spanish-character heuristic
# ======================================================================

# Letters common in Spanish (including accented vowels, ñ, ü)
_SPANISH_CHARS = re.compile(
    r'[a-záéíóúñü]', re.IGNORECASE | re.UNICODE,
)

# Characters unlikely in Spanish text (CJK, Cyrillic, Arabic, etc.)
_NON_LATIN_RE = re.compile(
    r'[\u0400-\u04FF'   # Cyrillic
    r'\u0600-\u06FF'    # Arabic
    r'\u3000-\u9FFF'    # CJK
    r'\uAC00-\uD7AF'    # Korean
    r']',
    re.UNICODE,
)


def min_length_filter(
    samples: Iterable[DialectSample],
    min_len: int = 10,
) -> list[DialectSample]:
    """Keep samples whose text is at least *min_len* characters.

    Parameters
    ----------
    samples:
        Input samples.
    min_len:
        Minimum text length (in characters).  Default 10.
    """
    return [s for s in samples if len(s.text.strip()) >= min_len]


def max_length_filter(
    samples: Iterable[DialectSample],
    max_len: int = 5000,
) -> list[DialectSample]:
    """Keep samples whose text is at most *max_len* characters."""
    return [s for s in samples if len(s.text.strip()) <= max_len]


def language_filter(
    samples: Iterable[DialectSample],
    lang: str = "es",
) -> list[DialectSample]:
    """Heuristic language filter.

    For ``lang="es"`` (the default), keeps samples where:
    - At least 60 % of alphabetical characters are in the Spanish set.
    - No more than 5 % of characters are from non-Latin scripts.

    For other languages a no-op passthrough is returned (future extension).
    """
    if lang != "es":
        return list(samples)

    result: list[DialectSample] = []
    for s in samples:
        text = s.text.strip()
        if not text:
            continue

        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            continue

        spanish_count = len(_SPANISH_CHARS.findall(text))
        spanish_ratio = spanish_count / len(alpha_chars)

        non_latin = len(_NON_LATIN_RE.findall(text))
        non_latin_ratio = non_latin / len(text) if text else 0

        if spanish_ratio >= 0.60 and non_latin_ratio <= 0.05:
            result.append(s)

    return result


def dedup_filter(
    samples: Iterable[DialectSample],
) -> list[DialectSample]:
    """Remove exact-duplicate texts (case-sensitive).

    The first occurrence of each unique text is kept; subsequent
    duplicates are discarded.
    """
    seen: set[str] = set()
    result: list[DialectSample] = []
    for s in samples:
        h = hashlib.md5(s.text.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(s)
    return result


def near_dedup_filter(
    samples: Iterable[DialectSample],
    threshold: float = 0.9,
) -> list[DialectSample]:
    """Remove near-duplicate texts using character-level trigram Jaccard.

    Keeps the first occurrence; drops later samples whose Jaccard
    similarity with any previously accepted sample exceeds *threshold*.
    """

    def _trigrams(text: str) -> set[str]:
        t = text.lower().strip()
        if len(t) < 3:
            return {t}
        return {t[i : i + 3] for i in range(len(t) - 2)}

    accepted: list[DialectSample] = []
    accepted_trigrams: list[set[str]] = []

    for s in samples:
        s_tri = _trigrams(s.text)
        is_dup = False
        for a_tri in accepted_trigrams:
            if not s_tri and not a_tri:
                is_dup = True
                break
            intersection = len(s_tri & a_tri)
            union = len(s_tri | a_tri)
            if union > 0 and intersection / union >= threshold:
                is_dup = True
                break
        if not is_dup:
            accepted.append(s)
            accepted_trigrams.append(s_tri)

    return accepted


def quality_filter(
    samples: Iterable[DialectSample],
) -> list[DialectSample]:
    """Apply a battery of quality heuristics.

    Removes samples that:
    - Contain fewer than 3 words.
    - Are more than 80 % non-alphabetic characters.
    - Consist entirely of uppercase text (likely OCR / spam).
    """
    result: list[DialectSample] = []
    for s in samples:
        text = s.text.strip()
        if not text:
            continue
        words = text.split()
        if len(words) < 3:
            continue
        alpha = sum(1 for c in text if c.isalpha())
        if alpha / len(text) < 0.20:
            continue
        # All-uppercase check (ignoring short texts)
        if len(text) > 20 and text == text.upper():
            continue
        result.append(s)
    return result


def confidence_filter(
    samples: Iterable[DialectSample],
    min_confidence: float = 0.5,
) -> list[DialectSample]:
    """Keep only samples with confidence >= *min_confidence*."""
    return [s for s in samples if s.confidence >= min_confidence]


# ======================================================================
# Pipeline combinator
# ======================================================================

_FILTER_REGISTRY: dict[str, object] = {
    "min_length": min_length_filter,
    "max_length": max_length_filter,
    "language": language_filter,
    "dedup": dedup_filter,
    "near_dedup": near_dedup_filter,
    "quality": quality_filter,
    "confidence": confidence_filter,
}


def apply_filters(
    samples: Iterable[DialectSample],
    config: Optional[dict[str, object]] = None,
) -> list[DialectSample]:
    """Apply a sequence of filters described by *config*.

    Parameters
    ----------
    samples:
        Input samples.
    config:
        A dict mapping filter names to their keyword arguments, e.g.::

            {
                "min_length": {"min_len": 20},
                "language": {"lang": "es"},
                "dedup": {},
                "quality": {},
            }

        Filters are applied in the order they appear.
        If *config* is ``None``, a sensible default pipeline is used.
    """
    if config is None:
        config = {
            "min_length": {"min_len": 10},
            "language": {"lang": "es"},
            "dedup": {},
            "quality": {},
        }

    current = list(samples)
    for name, kwargs in config.items():
        fn = _FILTER_REGISTRY.get(name)
        if fn is None:
            raise ValueError(
                f"Unknown filter '{name}'. Available: "
                f"{sorted(_FILTER_REGISTRY)}"
            )
        if not isinstance(kwargs, dict):
            kwargs = {}
        current = fn(current, **kwargs)  # type: ignore[operator]

    return current
