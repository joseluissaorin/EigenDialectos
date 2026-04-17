"""4-layer vocabulary filtering and Procrustes anchor word detection."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional

from eigen3.constants import ALL_REGIONALISMS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer 1: Spanish alphabet regex
# ---------------------------------------------------------------------------

_SPANISH_ALPHA_RE = re.compile(
    r"^[a-záéíóúñüàèìòùâêîôûäëïöüç]+$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Layer 2: Essential short Spanish words (exceptions to min_len=3)
# ---------------------------------------------------------------------------

_SHORT_SPANISH: frozenset[str] = frozenset({
    "al", "da", "de", "di", "el", "en", "es", "fe", "ha", "he",
    "id", "ir", "la", "le", "lo", "me", "mi", "ni", "no",
    "os", "re", "se", "si", "su", "te", "ti", "un", "va", "ve",
    "vi", "ya", "yo",
})

_SPANISH_ACCENT_CHARS = set("áéíóúñüÁÉÍÓÚÑÜ")


def _has_accent(word: str) -> bool:
    return any(c in _SPANISH_ACCENT_CHARS for c in word)


# ---------------------------------------------------------------------------
# Layer 3: English blacklist (~400 words)
# ---------------------------------------------------------------------------

_ENGLISH_BLACKLIST: frozenset[str] = frozenset({
    # Morphological fragments
    "ing", "ted", "ting", "tion", "ment", "ness", "ible", "able",
    "ful", "ous", "ive", "ent", "ant", "est", "ist", "ism",
    "ize", "ise", "ised", "ized", "ling", "ster",
    "ings", "tions", "ments", "bles",
    # Articles / determiners
    "the", "this", "that", "these", "those",
    # Pronouns
    "his", "her", "she", "him", "they", "them", "their",
    "its", "you", "your", "who", "whom", "whose", "which",
    "what", "ourselves", "themselves", "yourself",
    # Prepositions / conjunctions
    "and", "for", "with", "from", "into", "about", "after",
    "before", "between", "through", "during", "without",
    "within", "along", "across", "behind", "beyond",
    "but", "yet", "nor", "either", "neither", "whether",
    "although", "though", "because", "since", "while",
    "until", "unless", "however", "therefore", "moreover",
    # Auxiliary / common verbs
    "was", "were", "are", "been", "being", "have", "had", "has",
    "would", "could", "should", "will", "shall", "may", "might",
    "must", "did", "does", "done", "got", "get", "getting",
    "went", "going", "gone", "came", "come", "coming",
    "made", "make", "making", "took", "take", "taken", "taking",
    "said", "say", "saying", "told", "tell", "telling",
    "gave", "give", "given", "giving", "kept", "keep", "keeping",
    "thought", "think", "thinking", "knew", "know", "known",
    "left", "let", "put", "run", "set", "stood", "bring",
    "brought", "built", "buy", "bought", "catch", "caught",
    # Common nouns / adjectives / adverbs
    "back", "just", "only", "also", "then", "than", "some",
    "other", "such", "even", "each", "every", "own",
    "any", "few", "many", "much", "most", "more", "less",
    "very", "too", "quite", "rather", "still", "already",
    "soon", "often", "ever", "never", "always", "here",
    "there", "where", "when", "how", "why", "now",
    "out", "off", "away", "down", "over", "under", "again",
    "people", "children", "women", "thing", "things",
    "something", "nothing", "everything", "anything",
    "someone", "nobody", "everybody", "anyone",
    "way", "day", "year", "house", "world",
    "right", "good", "great", "little", "small", "big",
    "long", "high", "old", "new", "young",
    "first", "last", "next", "same", "different",
    "help", "need", "want", "like", "look",
    "see", "seem", "saw", "seen",
    "work", "working", "worked",
    "call", "called", "calling",
    "try", "tried", "trying",
    "ask", "asked", "asking",
    # Subtitle artifacts
    "okay", "yeah", "hey", "wow", "ugh", "hmm",
    "bye", "sir", "miss", "mrs", "mister",
    "please", "thank", "thanks", "sorry",
    "yes", "yep", "yup", "nah", "nope",
    # Abbreviations
    "inc", "int", "ext", "gen", "ref",
    "pre", "sub", "non",
    "bed", "wed", "shed", "fed", "red",
    "inn", "ins", "add", "odd", "ill",
})


# ---------------------------------------------------------------------------
# Procrustes anchor words (~200 universal Spanish words)
# ---------------------------------------------------------------------------

SPANISH_ANCHOR_WORDS: frozenset[str] = frozenset({
    # Articles
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # Prepositions
    "de", "en", "con", "por", "para", "sin", "sobre", "entre",
    "hacia", "hasta", "desde", "durante", "contra", "ante", "bajo",
    # Conjunctions
    "que", "como", "pero", "porque", "aunque", "cuando", "donde",
    "mientras", "sino", "pues", "ni",
    # Pronouns
    "yo", "él", "ella", "nosotros", "ellos", "ellas",
    "me", "te", "se", "nos", "les", "lo", "le",
    # Demonstratives
    "este", "esta", "estos", "estas",
    "ese", "esa", "esos", "esas",
    # Indefinites
    "esto", "eso", "algo", "nada", "todo", "toda",
    "todos", "todas", "cada", "alguien", "nadie",
    "otro", "otra", "otros", "otras",
    "mucho", "mucha", "muchos", "muchas",
    "poco", "poca", "pocos", "pocas",
    # Possessives
    "mi", "tu", "su", "mis", "tus", "sus",
    "nuestro", "nuestra", "nuestros", "nuestras",
    # Adverbs
    "no", "ya", "más", "muy", "bien", "mal", "así",
    "también", "después", "antes", "siempre", "nunca",
    "aquí", "allí", "ahora", "hoy", "todavía", "tan",
    # Numbers
    "uno", "dos", "tres", "cuatro", "cinco",
    "seis", "siete", "ocho", "nueve", "diez",
    # Universal verbs (infinitives)
    "ser", "estar", "haber", "tener", "hacer", "poder",
    "decir", "ver", "dar", "saber", "querer", "ir",
    "venir", "llegar", "pasar", "dejar", "seguir",
    "encontrar", "poner", "parecer", "creer", "conocer",
    "sentir", "pensar", "vivir", "morir", "salir", "entrar",
    "llevar", "tomar", "escribir", "leer", "hablar",
    "llamar", "comer", "dormir", "caminar", "correr",
    "trabajar", "comprar", "abrir", "cerrar",
    "perder", "ganar", "pagar", "jugar",
    # High-freq verb forms
    "es", "son", "fue", "era", "sido",
    "está", "están", "hay",
    "tiene", "hace", "puede", "dice", "sabe",
    "hecho", "dicho", "visto",
    # Universal nouns
    "casa", "vida", "tiempo", "hombre", "mujer", "mundo",
    "año", "día", "parte", "vez", "país", "forma",
    "agua", "tierra", "ciudad", "pueblo",
    "hijo", "hija", "padre", "madre",
    "nombre", "mano", "cuerpo", "cabeza",
    "ojo", "noche", "muerte", "amor", "guerra",
    "familia", "historia", "gobierno", "trabajo",
    "ejemplo", "palabra", "cosa", "persona",
    "momento", "lugar", "problema", "verdad",
    "manera", "razón", "idea", "punto", "lado",
    "caso", "libro", "mesa", "puerta",
    "calle", "sol", "luna", "mar",
    "campo", "sangre", "hermano", "hermana",
    "amigo", "amiga", "escuela",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_vocabulary(vocab: list[str], min_len: int = 3) -> list[str]:
    """4-layer vocabulary filter: alphabetic → min-length → English blacklist.

    Keeps accented short words and essential Spanish function words.
    """
    filtered = []
    for word in vocab:
        # Layer 1: alphabetic only
        if not _SPANISH_ALPHA_RE.match(word):
            continue
        # Layer 2: minimum length (with exceptions)
        if len(word) < min_len:
            if not (_has_accent(word) or word.lower() in _SHORT_SPANISH):
                continue
        # Layer 3: English blacklist
        if word.lower() in _ENGLISH_BLACKLIST:
            continue
        filtered.append(word)

    filtered.sort()
    logger.info("Vocabulary filter: %d -> %d words", len(vocab), len(filtered))
    return filtered


def filter_by_corpus_evidence(
    vocab: list[str],
    corpus: dict[str, list[str]],
    min_count: int = 2,
    min_varieties: int = 1,
    ascii_min_total: int = 10,
) -> list[str]:
    """Layer 4: keep words with corpus evidence of being Spanish.

    Known regionalisms are always kept (they appear in 1 variety by
    design and are essential for dialect discrimination). Accented
    words pass with low threshold. ASCII-only words need moderate
    total frequency to exclude English contaminants.
    """
    # Count per-variety frequencies
    variety_counts: dict[str, Counter] = {}
    for variety, docs in corpus.items():
        counter: Counter = Counter()
        for doc in docs:
            for word in doc.strip().lower().split():
                word = word.strip(".,;:!?¿¡\"'()[]{}…-—–")
                if word:
                    counter[word] += 1
        variety_counts[variety] = counter

    filtered = []
    for word in vocab:
        # Always keep known regionalisms — they are the dialect signal
        if word.lower() in ALL_REGIONALISMS:
            filtered.append(word)
            continue

        # Always keep anchor words
        if word.lower() in SPANISH_ANCHOR_WORDS:
            filtered.append(word)
            continue

        n_varieties = sum(
            1 for counter in variety_counts.values()
            if counter.get(word, 0) >= min_count
        )
        if n_varieties < min_varieties:
            continue

        total_count = sum(counter.get(word, 0) for counter in variety_counts.values())

        if _has_accent(word):
            filtered.append(word)
        elif total_count >= ascii_min_total:
            filtered.append(word)

    filtered = sorted(set(filtered))
    logger.info("Corpus evidence filter: %d -> %d words", len(vocab), len(filtered))
    return filtered


def get_anchor_indices(vocab: list[str], min_anchors: int = 50) -> list[int]:
    """Find indices of Procrustes anchor words in the vocabulary."""
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    indices = sorted(
        word_to_idx[w] for w in SPANISH_ANCHOR_WORDS if w in word_to_idx
    )
    if len(indices) < min_anchors:
        raise ValueError(
            f"Only {len(indices)} anchor words found (need >= {min_anchors})"
        )
    logger.info("Procrustes anchors: %d/%d found", len(indices), len(SPANISH_ANCHOR_WORDS))
    return indices
