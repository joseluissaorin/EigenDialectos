"""Back-end of the SDC compiler: reassembles transformed units into text.

Handles:
- Morphological agreement (gender, number via suffix rules)
- Article agreement (el/la/los/las matching noun gender)
- Clitic repositioning
- Orthographic adjustments
"""

from __future__ import annotations

import re
from typing import Any

from eigendialectos.types import ParsedText


# Gender inference from word ending
_FEMININE_ENDINGS = {"a", "ción", "sión", "dad", "tad", "tud", "ez", "umbre"}
_MASCULINE_ENDINGS = {"o", "or", "aje", "miento", "ón"}

# Determiner gender/number forms
_DETERMINERS = {
    ("m", "s"): "el",
    ("f", "s"): "la",
    ("m", "p"): "los",
    ("f", "p"): "las",
    ("m", "s", "indef"): "un",
    ("f", "s", "indef"): "una",
    ("m", "p", "indef"): "unos",
    ("f", "p", "indef"): "unas",
}

# Known feminine nouns (exceptions to ending rules)
_KNOWN_FEMININE = {
    "mano", "radio", "foto", "moto", "guagua", "agua", "águila",
    "calle", "noche", "clase", "gente", "parte", "muerte", "suerte",
    "leche", "sangre", "llave", "torre", "nube",
}

_KNOWN_MASCULINE = {
    "día", "mapa", "problema", "sistema", "tema", "programa",
    "idioma", "diploma", "clima", "poema", "drama", "planeta",
    "sofá", "papá", "tranvía",
}

# Article forms for detection
_ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas"}


def _infer_gender(word: str) -> str:
    """Infer grammatical gender of a Spanish noun."""
    w = word.lower().rstrip("s")  # strip plural for analysis
    if w in _KNOWN_FEMININE:
        return "f"
    if w in _KNOWN_MASCULINE:
        return "m"
    for ending in sorted(_FEMININE_ENDINGS, key=len, reverse=True):
        if w.endswith(ending):
            return "f"
    for ending in sorted(_MASCULINE_ENDINGS, key=len, reverse=True):
        if w.endswith(ending):
            return "m"
    # Default: masculine
    return "m"


def _is_plural(word: str) -> bool:
    """Check if a word is likely plural."""
    w = word.lower()
    return w.endswith("s") and not w.endswith("és")


def _pluralize(word: str) -> str:
    """Simple Spanish pluralization."""
    w = word.lower()
    if w.endswith(("s", "x")):
        return word
    if w.endswith("z"):
        return word[:-1] + "ces"
    if w.endswith(("á", "é", "í", "ó", "ú")):
        return word + "s"
    if w[-1] in "aeiou":
        return word + "s"
    return word + "es"


def _singularize(word: str) -> str:
    """Simple Spanish singularization."""
    w = word.lower()
    if w.endswith("ces"):
        return word[:-3] + "z"
    if w.endswith("es") and not w.endswith("ses"):
        return word[:-2]
    if w.endswith("s"):
        return word[:-1]
    return word


class TextReconstructor:
    """Reassembles text from multi-level replacements with agreement checks."""

    def reconstruct(
        self,
        original: ParsedText,
        level_replacements: dict[int, list[tuple[str, dict[str, Any]]]],
    ) -> str:
        """Reconstruct text from multi-level replacements.

        Priority: higher levels override lower when conflicts exist.
        Level 2 (word) is the primary replacement level.

        Parameters
        ----------
        original : ParsedText
        level_replacements : dict mapping level int to list of (replacement, metadata)

        Returns
        -------
        str : Reconstructed text.
        """
        # Start with original words
        tokens = list(original.words)

        # Apply word-level (L2) replacements — primary
        word_replacements = level_replacements.get(2, [])
        for i, (replacement, meta) in enumerate(word_replacements):
            if i < len(tokens) and meta.get("changed", False):
                tokens[i] = replacement

        # Apply morpheme-level (L1) replacements where L2 didn't change
        morph_replacements = level_replacements.get(1, [])
        morph_idx = 0
        for i, morphemes in enumerate(original.morphemes):
            if i >= len(tokens):
                break
            word_meta = word_replacements[i][1] if i < len(word_replacements) else {}
            if not word_meta.get("changed", False):
                # Check if morpheme-level made changes
                new_morphemes = []
                for m_idx in range(len(morphemes)):
                    if morph_idx < len(morph_replacements):
                        m_repl, m_meta = morph_replacements[morph_idx]
                        new_morphemes.append(m_repl)
                        morph_idx += 1
                    else:
                        new_morphemes.append(morphemes[m_idx])
                        morph_idx += 1
                reconstructed = "".join(new_morphemes)
                if reconstructed.lower() != tokens[i].lower():
                    tokens[i] = self._match_case(tokens[i], reconstructed)
            else:
                morph_idx += len(morphemes)

        # Apply agreement corrections
        tokens = self._apply_agreement(tokens)

        # Reconstruct text preserving original spacing/punctuation
        return self._rejoin(original.original, original.words, tokens)

    def _apply_agreement(self, tokens: list[str]) -> list[str]:
        """Fix gender/number agreement after replacements.

        Scans for article + noun pairs and adjusts the article if the
        noun's gender/number changed.
        """
        result = list(tokens)

        for i in range(len(result) - 1):
            if result[i].lower() in _ARTICLES:
                # This is an article — check if next word (noun) changed
                noun = result[i + 1]
                gender = _infer_gender(noun)
                plural = _is_plural(noun)
                number = "p" if plural else "s"

                # Determine article type (definite vs indefinite)
                current_art = result[i].lower()
                is_indef = current_art in {"un", "una", "unos", "unas"}

                if is_indef:
                    key = (gender, number, "indef")
                else:
                    key = (gender, number)

                new_art = _DETERMINERS.get(key, result[i].lower())
                result[i] = self._match_case(result[i], new_art)

        return result

    def _rejoin(
        self,
        original_text: str,
        original_words: list[str],
        new_words: list[str],
    ) -> str:
        """Reconstruct text by replacing words while preserving original spacing.

        Uses word-boundary-aware regex matching from right to left so that
        earlier indices are not shifted by replacements.  This prevents
        partial matches (e.g., replacing "a" inside "autobús").
        """
        result = original_text
        # Replace from right to left to preserve character indices
        for old_word, new_word in reversed(list(zip(original_words, new_words))):
            if old_word != new_word:
                # Use word boundaries to avoid matching inside other words
                # \b works for ASCII word chars; for Spanish we also handle
                # punctuation boundaries via lookbehind/lookahead
                escaped = re.escape(old_word)
                pattern = re.compile(
                    r'(?<![a-záéíóúñüA-ZÁÉÍÓÚÑÜ])' + escaped + r'(?![a-záéíóúñüA-ZÁÉÍÓÚÑÜ])',
                    re.UNICODE,
                )
                result = pattern.sub(new_word, result, count=1)

        return result

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        """Preserve the casing pattern of the original."""
        if original.isupper():
            return replacement.upper()
        if original and original[0].isupper():
            return replacement[0].upper() + replacement[1:] if replacement else ""
        return replacement
