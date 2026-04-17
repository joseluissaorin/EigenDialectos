"""Chi-squared corpus-statistical regionalism detection + merge with LLM set.

Supplements the curated + LLM-generated regionalism lists with words
that show statistically significant variety affinity in the actual corpus.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

from eigendialectos.embeddings.dcl.regionalisms import ALL_REGIONALISMS, REGIONALISMS

logger = logging.getLogger(__name__)

# Spanish stop words (high-frequency function words to exclude)
_STOP_WORDS: frozenset[str] = frozenset({
    "de", "la", "el", "en", "y", "a", "los", "que", "del", "las",
    "un", "por", "con", "una", "su", "para", "es", "al", "lo",
    "como", "más", "pero", "sus", "le", "ya", "o", "fue", "este",
    "ha", "sí", "porque", "esta", "entre", "cuando", "muy", "sin",
    "sobre", "también", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les",
    "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e",
    "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro",
    "otras", "otra", "él", "tanto", "esa", "estos", "mucho",
    "quienes", "nada", "muchos", "cual", "poco", "ella", "estar",
    "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú",
    "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros",
    "vosotras", "os", "mío", "mía", "míos", "mías", "tuyo",
    "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas",
    "nuestro", "nuestra", "nuestros", "nuestras", "vuestro",
    "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy",
    "estás", "está", "estamos", "estáis", "están", "esté",
    "estés", "estemos", "estéis", "estén", "estaré", "estarás",
    "estará", "estaremos", "estaréis", "estarán", "estaría",
    "no", "se", "ser", "haber", "ir", "hacer", "poder", "tener",
    "decir", "ver", "dar", "saber", "querer", "llegar", "pasar",
    "deber", "poner", "parecer", "quedar", "creer", "hablar",
    "llevar", "dejar", "seguir", "encontrar", "llamar", "venir",
    "pensar", "salir", "volver", "tomar", "conocer", "vivir",
    "sentir", "tratar", "mirar", "contar", "empezar", "esperar",
    "buscar", "existir", "entrar", "trabajar", "escribir", "perder",
    "producir", "ocurrir", "entender", "pedir", "recibir", "recordar",
    "terminar", "permitir", "aparecer", "conseguir", "comenzar",
    "servir", "sacar", "necesitar", "mantener", "resultar",
    "leer", "caer", "cambiar", "presentar", "crear", "abrir",
    "considerar", "oír", "acabar", "convertir", "ganar",
    "formar", "traer", "partir", "morir", "aceptar", "realizar",
    "suponer", "comprender", "lograr", "explicar",
})


def detect_corpus_regionalisms(
    corpus_dir: Path,
    chi2_threshold: float = 10.0,
    min_count: int = 3,
    min_word_len: int = 3,
) -> dict[str, set[str]]:
    """Detect variety-associated words via chi-squared divergence.

    Parameters
    ----------
    corpus_dir:
        Directory containing per-variety JSONL corpus files.
    chi2_threshold:
        Minimum chi-squared value for a word to be considered a regionalism.
    min_count:
        Minimum occurrences in the primary variety.
    min_word_len:
        Minimum word length (characters).

    Returns
    -------
    dict mapping variety code to set of detected regionalisms.
    """
    # Load word counts per variety
    variety_counts: dict[str, Counter] = {}
    variety_totals: dict[str, int] = {}

    for jsonl_path in sorted(corpus_dir.glob("ES_*.jsonl")):
        variety = jsonl_path.stem
        counter: Counter = Counter()
        total = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = doc.get("text", "").strip()
                if not text:
                    continue
                for word in text.lower().split():
                    # Basic cleaning
                    word = word.strip(".,;:!?¿¡\"'()[]{}…-—–")
                    if len(word) >= min_word_len and word not in _STOP_WORDS:
                        counter[word] += 1
                        total += 1
        variety_counts[variety] = counter
        variety_totals[variety] = total
        logger.info("  %s: %d unique words, %d total tokens", variety, len(counter), total)

    if not variety_counts:
        logger.warning("No corpus files found in %s", corpus_dir)
        return {}

    grand_total = sum(variety_totals.values())
    varieties = sorted(variety_counts.keys())

    # Global word counts
    global_counts: Counter = Counter()
    for counter in variety_counts.values():
        global_counts.update(counter)

    # Chi-squared test per word
    detected: dict[str, set[str]] = {v: set() for v in varieties}

    for word, total_count in global_counts.items():
        if total_count < min_count:
            continue

        chi2 = 0.0
        max_ratio = 0.0
        primary_variety = varieties[0]

        for v in varieties:
            observed_count = variety_counts[v].get(word, 0)
            # Expected count under uniform distribution across varieties
            expected_count = total_count * variety_totals[v] / grand_total
            if expected_count > 0:
                chi2 += (observed_count - expected_count) ** 2 / expected_count
            # Over-representation ratio for primary variety assignment
            ratio = (observed_count / max(variety_totals[v], 1)) / max(
                total_count / grand_total, 1e-12
            )
            if ratio > max_ratio:
                max_ratio = ratio
                primary_variety = v

        if chi2 >= chi2_threshold:
            # Verify the word actually appears enough in the primary variety
            if variety_counts[primary_variety].get(word, 0) >= min_count:
                detected[primary_variety].add(word)

    for v in varieties:
        logger.info("  %s: %d corpus-detected regionalisms", v, len(detected[v]))

    return detected


def get_all_regionalisms(
    corpus_dir: Path | None = None,
) -> frozenset[str]:
    """Return the merged regionalism set (curated + LLM + corpus-detected).

    Parameters
    ----------
    corpus_dir:
        If provided, also runs corpus-statistical detection and merges.
        If None, returns just the curated + LLM set.
    """
    merged = set(ALL_REGIONALISMS)

    if corpus_dir is not None:
        corpus_path = Path(corpus_dir)
        if corpus_path.exists():
            corpus_detected = detect_corpus_regionalisms(corpus_path)
            for words in corpus_detected.values():
                merged |= words
            logger.info(
                "Merged regionalisms: %d (curated+LLM=%d, corpus-detected added %d)",
                len(merged), len(ALL_REGIONALISMS),
                len(merged) - len(ALL_REGIONALISMS),
            )

    return frozenset(merged)


def get_regionalisms_by_variety(
    corpus_dir: Path | None = None,
) -> dict[str, set[str]]:
    """Return per-variety regionalism dict (curated + LLM + corpus).

    Useful when you need per-variety breakdown rather than the union.
    """
    merged: dict[str, set[str]] = {k: set(v) for k, v in REGIONALISMS.items()}

    if corpus_dir is not None:
        corpus_path = Path(corpus_dir)
        if corpus_path.exists():
            corpus_detected = detect_corpus_regionalisms(corpus_path)
            for variety, words in corpus_detected.items():
                if variety in merged:
                    merged[variety] |= words
                else:
                    merged[variety] = words

    return merged
