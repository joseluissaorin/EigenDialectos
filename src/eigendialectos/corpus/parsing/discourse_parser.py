"""Discourse-level feature extraction for Spanish text.

Extracts features at the discourse level (L5) including sentence statistics,
subordination ratios, question/exclamation ratios, discourse marker inventory,
and a heuristic formality score.
"""

from __future__ import annotations

import re
from typing import Any

from eigendialectos.corpus.preprocessing.segmentation import split_sentences

# ======================================================================
# Discourse markers — 80+ entries organized by pragmatic function
# ======================================================================

_DISCOURSE_MARKERS: dict[str, str] = {
    # Conversational / filler
    "bueno": "filler",
    "pues": "filler",
    "mira": "attention",
    "oye": "attention",
    "oiga": "attention",
    "hombre": "filler",
    "mujer": "filler",
    "vamos": "filler",
    "vale": "agreement",
    "venga": "filler",
    "anda": "surprise",
    "dale": "agreement",
    "claro": "agreement",
    "exacto": "agreement",
    "efectivamente": "agreement",
    "desde luego": "agreement",
    "por supuesto": "agreement",
    "faltaría más": "agreement",
    "ya": "agreement",
    # Reformulation
    "o sea": "reformulation",
    "es decir": "reformulation",
    "en otras palabras": "reformulation",
    "dicho de otro modo": "reformulation",
    "mejor dicho": "reformulation",
    "a saber": "reformulation",
    "quiero decir": "reformulation",
    "esto es": "reformulation",
    # Consequence / conclusion
    "en fin": "conclusion",
    "total": "conclusion",
    "al fin y al cabo": "conclusion",
    "en resumidas cuentas": "conclusion",
    "en definitiva": "conclusion",
    "en conclusión": "conclusion",
    "en resumen": "conclusion",
    "a fin de cuentas": "conclusion",
    "por lo tanto": "consequence",
    "por eso": "consequence",
    "así que": "consequence",
    "de modo que": "consequence",
    "de manera que": "consequence",
    "de ahí que": "consequence",
    "por consiguiente": "consequence",
    "en consecuencia": "consequence",
    # Topic shift / digression
    "por cierto": "digression",
    "a propósito": "digression",
    "a todo esto": "digression",
    "hablando de": "digression",
    "cambiando de tema": "digression",
    "dicho sea de paso": "digression",
    # Attention / engagement
    "a ver": "attention",
    "fíjate": "attention",
    "imagínate": "attention",
    "date cuenta": "attention",
    "mire usted": "attention",
    # Contrast / concession
    "sin embargo": "contrast",
    "no obstante": "contrast",
    "con todo": "contrast",
    "ahora bien": "contrast",
    "en cambio": "contrast",
    "por el contrario": "contrast",
    "de todas formas": "concession",
    "de todos modos": "concession",
    "de cualquier manera": "concession",
    "en cualquier caso": "concession",
    "aun así": "concession",
    # Addition
    "además": "addition",
    "encima": "addition",
    "aparte": "addition",
    "por añadidura": "addition",
    "es más": "addition",
    "incluso": "addition",
    "asimismo": "addition",
    "también": "addition",
    "igualmente": "addition",
    # Exemplification
    "por ejemplo": "exemplification",
    "verbigracia": "exemplification",
    "pongamos por caso": "exemplification",
    "como muestra": "exemplification",
    # Temporal
    "mientras tanto": "temporal",
    "entre tanto": "temporal",
    "a continuación": "temporal",
    "acto seguido": "temporal",
    "en primer lugar": "ordering",
    "en segundo lugar": "ordering",
    "por último": "ordering",
    "finalmente": "ordering",
    "para empezar": "ordering",
    "para terminar": "ordering",
    # Regional / colloquial (dialectally marked)
    "che": "filler_regional",       # Rioplatense
    "güey": "filler_regional",      # Mexican
    "wey": "filler_regional",       # Mexican
    "pana": "filler_regional",      # Caribbean
    "chamo": "filler_regional",     # Venezuelan
    "tío": "filler_regional",       # Peninsular
    "tía": "filler_regional",       # Peninsular
    "chaval": "filler_regional",    # Peninsular
    "chavala": "filler_regional",   # Peninsular
    "cachái": "filler_regional",    # Chilean
    "po": "filler_regional",        # Chilean
    "ñaño": "filler_regional",      # Andean
}

# Sort markers longest-first for correct matching of multi-word markers
_MARKERS_BY_LENGTH: list[tuple[str, str]] = sorted(
    _DISCOURSE_MARKERS.items(),
    key=lambda kv: len(kv[0]),
    reverse=True,
)


# ======================================================================
# Subordination cue words
# ======================================================================

_SUBORDINATORS: set[str] = {
    "que", "quien", "quienes", "cual", "cuales",
    "cuyo", "cuya", "cuyos", "cuyas",
    "donde", "adonde",
    "cuando", "mientras",
    "como", "según",
    "porque", "puesto que", "ya que", "dado que",
    "aunque", "si bien", "a pesar de que",
    "si", "siempre que", "con tal de que",
    "para que", "a fin de que",
    "de modo que", "de manera que",
}

_SUBORDINATOR_MULTIWORD: list[str] = sorted(
    [s for s in _SUBORDINATORS if " " in s],
    key=len,
    reverse=True,
)

_SUBORDINATOR_SINGLE: set[str] = {s for s in _SUBORDINATORS if " " not in s}


# ======================================================================
# Informal / slang indicators
# ======================================================================

_SLANG_WORDS: set[str] = {
    "tío", "tía", "mola", "molar", "curro", "currar", "flipar",
    "guay", "genial", "mogollón", "mazo", "chorrada", "mierda",
    "joder", "coño", "hostia", "gilipollas", "capullo", "imbécil",
    "güey", "wey", "chido", "padre", "neta", "chamba",
    "pana", "chamo", "chévere", "bacano", "chimba",
    "che", "boludo", "boluda", "pibe", "piba", "laburo", "guita",
    "pololo", "polola", "fome", "bacán", "cachar", "weón", "huevón",
    "ñaño", "ñaña", "chuta", "chucha",
    "jato", "pata", "causa", "chamba", "chela",
    "loco", "loca", "chaval", "chavala", "cachái",
    "onda", "rollo", "movida", "lio",
}


# ======================================================================
# Internal helpers
# ======================================================================

def _count_questions(text: str) -> int:
    """Count question marks (inverted or normal)."""
    return text.count("?") + text.count("¿")


def _count_exclamations(text: str) -> int:
    """Count exclamation marks (inverted or normal)."""
    return text.count("!") + text.count("¡")


def _find_discourse_markers(text: str) -> list[dict[str, str]]:
    """Find all discourse markers in text, returning list of {marker, function}."""
    lower = text.lower()
    found: list[dict[str, str]] = []
    # Track positions already consumed to avoid overlapping matches
    consumed: list[tuple[int, int]] = []

    for marker, function in _MARKERS_BY_LENGTH:
        start = 0
        while True:
            idx = lower.find(marker, start)
            if idx == -1:
                break

            end = idx + len(marker)

            # Check word boundaries
            if idx > 0 and lower[idx - 1].isalnum():
                start = end
                continue
            if end < len(lower) and lower[end].isalnum():
                start = end
                continue

            # Check overlap with already consumed spans
            overlap = False
            for cs, ce in consumed:
                if idx < ce and end > cs:
                    overlap = True
                    break
            if overlap:
                start = end
                continue

            consumed.append((idx, end))
            found.append({"marker": marker, "function": function})
            start = end

    return found


def _count_subordination(text: str) -> int:
    """Count subordination cues in text."""
    lower = text.lower()
    count = 0

    # Multi-word subordinators first
    for sub in _SUBORDINATOR_MULTIWORD:
        idx = 0
        while True:
            pos = lower.find(sub, idx)
            if pos == -1:
                break
            # Check word boundaries
            before_ok = pos == 0 or not lower[pos - 1].isalnum()
            end = pos + len(sub)
            after_ok = end >= len(lower) or not lower[end].isalnum()
            if before_ok and after_ok:
                count += 1
            idx = end

    # Single-word subordinators via token scan
    tokens = re.findall(r"\b\w+\b", lower, re.UNICODE)
    for tok in tokens:
        if tok in _SUBORDINATOR_SINGLE:
            count += 1

    return count


def _estimate_clause_count(sentences: list[str]) -> int:
    """Estimate total clause count as: n_sentences + subordination cues.

    Each sentence has at least one clause, and each subordinator
    introduces an additional subordinate clause.
    """
    n = len(sentences)
    sub = sum(_count_subordination(s) for s in sentences)
    return max(n + sub, 1)


def _count_tu_vos(text: str) -> int:
    """Count informal 2nd person markers: tú, vos, te, tú forms."""
    tokens = re.findall(r"\b\w+\b", text.lower(), re.UNICODE)
    informal_2nd = {"tú", "tu", "tus", "te", "vos", "ti", "contigo"}
    return sum(1 for t in tokens if t in informal_2nd)


def _count_usted(text: str) -> int:
    """Count formal 2nd person markers: usted, ustedes."""
    tokens = re.findall(r"\b\w+\b", text.lower(), re.UNICODE)
    formal = {"usted", "ustedes", "ud", "uds", "vd", "vds"}
    return sum(1 for t in tokens if t in formal)


def _count_slang(text: str) -> int:
    """Count slang/colloquial words."""
    tokens = re.findall(r"\b\w+\b", text.lower(), re.UNICODE)
    return sum(1 for t in tokens if t in _SLANG_WORDS)


def _formality_score(
    text: str,
    sentences: list[str],
    avg_sent_len: float,
) -> float:
    """Compute a 0-1 formality score (1 = most formal).

    Factors (weighted):
    - tu/vos vs usted ratio (informal pronouns lower score)
    - slang density (more slang = lower score)
    - average sentence length (longer = more formal)
    - exclamation density (more = less formal)
    """
    n_tokens = max(len(re.findall(r"\b\w+\b", text, re.UNICODE)), 1)
    n_sents = max(len(sentences), 1)

    # Factor 1: pronoun formality (0-1, 1 = formal)
    tu_count = _count_tu_vos(text)
    ud_count = _count_usted(text)
    total_2nd = tu_count + ud_count
    if total_2nd > 0:
        pronoun_formality = ud_count / total_2nd
    else:
        pronoun_formality = 0.5  # neutral when no 2nd person

    # Factor 2: slang density (0-1, 1 = no slang)
    slang_count = _count_slang(text)
    slang_density = slang_count / n_tokens
    slang_factor = max(0.0, 1.0 - slang_density * 10.0)  # 10% slang = 0

    # Factor 3: sentence length (0-1, scaled: 5 words = 0.2, 25+ words = 1.0)
    length_factor = min(1.0, max(0.0, (avg_sent_len - 3.0) / 22.0))

    # Factor 4: exclamation sparsity (0-1, 1 = no exclamations)
    exc_count = _count_exclamations(text) / 2.0  # pairs
    exc_ratio = exc_count / n_sents
    exc_factor = max(0.0, 1.0 - exc_ratio)

    # Weighted combination
    score = (
        0.30 * pronoun_formality
        + 0.30 * slang_factor
        + 0.20 * length_factor
        + 0.20 * exc_factor
    )
    return round(max(0.0, min(1.0, score)), 4)


# ======================================================================
# Public API
# ======================================================================

def parse_discourse(text: str) -> dict[str, Any]:
    """Extract discourse-level features from Spanish *text*.

    Parameters
    ----------
    text:
        Raw Spanish text (can be multi-sentence).

    Returns
    -------
    dict[str, Any]
        Keys:
        - ``n_sentences``: int
        - ``avg_sentence_length``: float (in tokens)
        - ``subordination_ratio``: float (subordinate clauses / total clauses)
        - ``question_ratio``: float (questions / sentences)
        - ``exclamation_ratio``: float (exclamations / sentences)
        - ``discourse_markers``: list of {marker, function} dicts
        - ``marker_density``: float (markers per sentence)
        - ``formality_score``: float in [0, 1]
    """
    if not text or not text.strip():
        return {
            "n_sentences": 0,
            "avg_sentence_length": 0.0,
            "subordination_ratio": 0.0,
            "question_ratio": 0.0,
            "exclamation_ratio": 0.0,
            "discourse_markers": [],
            "marker_density": 0.0,
            "formality_score": 0.5,
        }

    sentences = split_sentences(text)
    n_sentences = max(len(sentences), 1)

    # Average sentence length in tokens
    token_counts = [
        len(re.findall(r"\b\w+\b", s, re.UNICODE)) for s in sentences
    ]
    total_tokens = sum(token_counts)
    avg_sentence_length = total_tokens / n_sentences if n_sentences > 0 else 0.0

    # Subordination ratio
    total_sub_cues = sum(_count_subordination(s) for s in sentences)
    total_clauses = _estimate_clause_count(sentences)
    subordination_ratio = total_sub_cues / total_clauses if total_clauses > 0 else 0.0

    # Question and exclamation ratios
    # Count sentences containing questions/exclamations
    question_sents = sum(1 for s in sentences if "?" in s or "¿" in s)
    exclamation_sents = sum(1 for s in sentences if "!" in s or "¡" in s)
    question_ratio = question_sents / n_sentences
    exclamation_ratio = exclamation_sents / n_sentences

    # Discourse markers
    markers = _find_discourse_markers(text)
    marker_density = len(markers) / n_sentences

    # Formality
    formality = _formality_score(text, sentences, avg_sentence_length)

    return {
        "n_sentences": len(sentences),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "subordination_ratio": round(subordination_ratio, 4),
        "question_ratio": round(question_ratio, 4),
        "exclamation_ratio": round(exclamation_ratio, 4),
        "discourse_markers": markers,
        "marker_density": round(marker_density, 4),
        "formality_score": formality,
    }
