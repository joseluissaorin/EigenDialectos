"""Vocabulary filtering for clean dialectal analysis.

Removes non-Spanish tokens (symbols, numbers, phonetic notation,
English fragments) from the union vocabulary.  Defines a curated set
of Spanish function words to use as Procrustes alignment anchors —
words whose meaning and distributional properties are identical
across all 8 Spanish varieties.

The key insight: dialectal signal is inherently small (these are
dialects, not separate languages).  Noise from non-Spanish tokens
is LARGER than the dialectal signal, so it must be eliminated
before computing W.

Three-layer filtering:
  1. Alphabetic-only (removes symbols, numbers, phonetic notation)
  2. Minimum length ≥ 3 (removes 2-char fragments: st, th, ed, re)
  3. English blacklist (removes common English words/fragments that
     aren't Spanish: ing, ted, the, and, for, which, etc.)
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# ======================================================================
# Spanish-alphabet regex: keeps only letters + accented chars
# ======================================================================

# Matches strings composed entirely of Spanish-alphabet characters
_SPANISH_ALPHA_RE = re.compile(
    r"^[a-záéíóúñüàèìòùâêîôûäëïöüç]+$",
    re.IGNORECASE,
)

# ======================================================================
# English-only words to exclude
# ======================================================================
# Words that are common in English but do NOT exist as standard Spanish
# words.  Curated to avoid false positives (words like "no", "me", "once"
# that exist in both languages are NOT included).
#
# This blacklist targets the specific contamination pattern observed in
# the OpenSubtitles corpus: English movie titles, character names,
# code-switching, and English morphological fragments.

_ENGLISH_BLACKLIST: frozenset[str] = frozenset({
    # ── English morphological fragments (suffixes that appear as tokens) ──
    "ing", "ted", "ting", "tion", "ment", "ness", "ible", "able",
    "ful", "ous", "ive", "ent", "ant", "est", "ist", "ism",
    "ize", "ise", "ised", "ized", "ling", "ling", "ster",
    "ings", "ness", "tions", "ments", "bles",
    # ── English articles / determiners ──
    "the", "this", "that", "these", "those",
    # ── English pronouns ──
    "his", "her", "she", "him", "they", "them", "their",
    "its", "you", "your", "who", "whom", "whose", "which",
    "what", "ourselves", "themselves", "yourself",
    # ── English prepositions / conjunctions ──
    "and", "for", "with", "from", "into", "about", "after",
    "before", "between", "through", "during", "without",
    "within", "along", "across", "behind", "beyond",
    "but", "yet", "nor", "either", "neither", "whether",
    "although", "though", "because", "since", "while",
    "until", "unless", "however", "therefore", "moreover",
    # ── English auxiliary / common verbs ──
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
    "cut", "draw", "drew", "drawn", "drink", "drove",
    "eat", "ate", "eaten", "fall", "fell", "fallen",
    "feel", "felt", "fight", "fought", "find", "found",
    "fly", "flew", "forget", "forgot", "forgotten",
    "grow", "grew", "grown", "hang", "hear", "heard",
    "hide", "hid", "hit", "hold", "held", "hurt",
    "lay", "lead", "led", "learn", "learned", "leave",
    "lend", "lent", "lie", "light", "lit", "lose", "lost",
    "meet", "met", "pay", "paid", "prove", "proved",
    "read", "ride", "rode", "ring", "rang", "rise", "rose",
    "seek", "sell", "sold", "send", "sent", "shake", "shook",
    "shoot", "shot", "show", "showed", "shown", "shut",
    "sing", "sang", "sit", "sat", "sleep", "slept",
    "speak", "spoke", "spend", "spent", "stand", "steal",
    "stole", "stick", "stuck", "strike", "struck", "swim",
    "swam", "swing", "teach", "taught", "tear", "threw",
    "throw", "thrown", "understand", "understood",
    "wake", "woke", "wear", "wore", "win", "won",
    "wind", "write", "wrote", "written",
    # ── Common English nouns / adjectives / adverbs ──
    "back", "just", "only", "also", "then", "than", "some",
    "other", "such", "even", "each", "every", "own",
    "any", "few", "many", "much", "most", "more", "less",
    "very", "too", "quite", "rather", "still", "already",
    "soon", "often", "ever", "never", "always", "here",
    "there", "where", "when", "how", "why", "now",
    "out", "off", "away", "down", "over", "under", "again",
    "once", "twice",  # "once" = 11 in Spanish but rare as text
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
    "play", "played", "playing",
    "move", "moved", "moving",
    "live", "lived", "living",
    "believe", "believed",
    "turn", "turned", "turning",
    "start", "started", "starting",
    "show", "name", "place", "end", "hand",
    "head", "home", "room", "fact", "kind",
    # ── English words from subtitle artifacts ──
    "okay", "yeah", "hey", "wow", "ugh", "hmm",
    "bye", "sir", "miss", "mrs", "mister",
    "please", "thank", "thanks", "sorry",
    "yes", "yep", "yup", "nah", "nope",
    # ── Common English abbreviations / fragments in subtitles ──
    "inc", "int", "ext", "gen", "ref",
    "pre", "sub", "non",
    "bed", "wed", "shed", "fed", "red",
    "inn", "ins", "add", "odd", "ill",
    "tao", "zen", "chi",  # these showed up in diagnostics
})


# ======================================================================
# Procrustes anchor words
# ======================================================================
# These words have IDENTICAL meaning and very similar distributional
# properties across ALL 8 Spanish varieties.  They are the safest
# reference points for computing the Procrustes rotation.
#
# Criteria for inclusion:
#   - Function word OR universally shared concrete noun/verb
#   - NO dialectal meaning variation (excludes: coger, piso, carro, tío)
#   - NO dialect-specific morphology (excludes: vosotros, hablás)
#   - High frequency in all varieties (well-trained embeddings)
#
# ~200 words.  We want enough for a stable rotation matrix (need >> dim).

SPANISH_ANCHOR_WORDS: frozenset[str] = frozenset({
    # ── Articles ──
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # ── Prepositions ──
    "de", "en", "con", "por", "para", "sin", "sobre", "entre",
    "hacia", "hasta", "desde", "durante", "según", "contra",
    "ante", "bajo", "tras", "mediante",
    # ── Conjunctions ──
    "que", "como", "pero", "porque", "aunque", "cuando", "donde",
    "mientras", "sino", "pues", "ni", "entonces", "sin embargo",
    # ── Personal pronouns (universal) ──
    "yo", "él", "ella", "nosotros", "ellos", "ellas",
    "me", "te", "se", "nos", "les", "lo", "le",
    # ── Demonstratives ──
    "este", "esta", "estos", "estas",
    "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas",
    # ── Indefinites / quantifiers ──
    "esto", "eso", "aquello", "algo", "nada", "todo", "toda",
    "todos", "todas", "cada", "alguien", "nadie",
    "otro", "otra", "otros", "otras",
    "mucho", "mucha", "muchos", "muchas",
    "poco", "poca", "pocos", "pocas",
    "mismo", "misma", "mismos", "mismas",
    "varios", "varias", "algún", "alguno", "alguna",
    "ningún", "ninguno", "ninguna",
    # ── Possessives ──
    "mi", "tu", "su", "mis", "tus", "sus",
    "nuestro", "nuestra", "nuestros", "nuestras",
    # ── Adverbs ──
    "no", "ya", "más", "muy", "bien", "mal", "así",
    "también", "además", "después", "antes",
    "siempre", "nunca", "jamás",
    "aquí", "allí", "ahora", "hoy", "ayer",
    "todavía", "aún", "sólo", "solo", "tan", "tanto",
    "demasiado", "bastante", "casi", "apenas",
    # ── Interrogatives / relatives ──
    "quién", "qué", "cuál", "cuánto", "cuándo", "dónde", "cómo",
    "quien", "cual", "cuanto",
    # ── Numbers ──
    "uno", "dos", "tres", "cuatro", "cinco",
    "seis", "siete", "ocho", "nueve", "diez",
    "cien", "mil", "primer", "primera", "segundo", "segunda",
    # ── Universal verbs (infinitives — same across all dialects) ──
    "ser", "estar", "haber", "tener", "hacer", "poder",
    "decir", "ver", "dar", "saber", "querer", "ir",
    "venir", "llegar", "pasar", "dejar", "seguir",
    "encontrar", "poner", "parecer", "creer", "conocer",
    "sentir", "pensar", "vivir", "morir", "salir", "entrar",
    "llevar", "tomar", "escribir", "leer", "hablar",
    "llamar", "comer", "dormir", "caminar", "correr",
    "trabajar", "comprar", "vender", "abrir", "cerrar",
    "caer", "perder", "ganar", "pagar", "jugar",
    # ── High-freq verb forms (shared conjugation) ──
    "es", "son", "fue", "era", "sido",
    "está", "están", "hay",
    "tiene", "hace", "puede", "dice", "sabe",
    "quiere", "viene", "llega", "sale",
    "hecho", "dicho", "visto", "puesto",
    # ── Universal nouns (same referent in all dialects) ──
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
    "número", "caso", "libro", "mesa", "puerta",
    "calle", "sol", "luna", "mar", "río",
    "montaña", "campo", "fuego", "sangre",
    "hermano", "hermana", "amigo", "amiga",
    "rey", "iglesia", "escuela", "hospital",
})


# ======================================================================
# Public API
# ======================================================================


def filter_vocabulary(
    vocab: list[str],
    min_len: int = 3,
) -> list[str]:
    """Filter vocabulary to clean Spanish words only.

    Three-layer filter:
    1. **Alphabetic**: removes tokens with digits, symbols, phonetic
       notation, punctuation (keeps accented chars)
    2. **Minimum length**: removes 1-2 char tokens (``st``, ``th``,
       ``ed`` — English fragments that dominate noise).  Function words
       like ``de``, ``en`` are handled via the anchor list separately.
    3. **English blacklist**: removes ~400 common English words/fragments
       that appear in OpenSubtitles corpus due to code-switching, movie
       titles, and subtitle artifacts.

    Parameters
    ----------
    vocab : list[str]
        Raw union vocabulary from ``build_union_vocabulary``.
    min_len : int
        Minimum word length.  Default 3 removes 2-char noise.

    Returns
    -------
    list[str]
        Filtered, sorted vocabulary.
    """
    filtered = []
    rejected_alpha = 0
    rejected_len = 0
    rejected_english = 0

    # Essential short Spanish words that must survive min_len filtering
    _SHORT_SPANISH = frozenset({
        "al", "da", "de", "di", "el", "en", "es", "fe", "ha", "he",
        "id", "ir", "la", "le", "ll", "lo", "me", "mi", "ni", "no",
        "os", "re", "se", "si", "su", "te", "ti", "un", "va", "ve",
        "vi", "ya", "yo",
    })

    for word in vocab:
        if not _SPANISH_ALPHA_RE.match(word):
            rejected_alpha += 1
            continue
        if len(word) < min_len:
            # Keep accented short words (tú, sí, más) and essential function words
            if not (_has_spanish_accent(word) or word.lower() in _SHORT_SPANISH):
                rejected_len += 1
                continue
        if word.lower() in _ENGLISH_BLACKLIST:
            rejected_english += 1
            continue
        filtered.append(word)

    filtered.sort()

    logger.info(
        "Vocabulary filter: %d → %d words "
        "(rejected %d non-alpha, %d too short, %d English)",
        len(vocab), len(filtered),
        rejected_alpha, rejected_len, rejected_english,
    )
    return filtered


_SPANISH_ACCENT_CHARS = set("áéíóúñüÁÉÍÓÚÑÜ")


def _has_spanish_accent(word: str) -> bool:
    """Check if word contains any Spanish-specific accented character."""
    return any(c in _SPANISH_ACCENT_CHARS for c in word)


def filter_by_corpus_evidence(
    vocab: list[str],
    corpus_by_variety: dict[str, list[str]],
    min_count: int = 3,
    min_varieties: int = 2,
    ascii_min_total: int = 30,
) -> list[str]:
    """Keep only words with strong corpus evidence of being Spanish.

    Two-tier frequency threshold:
    - **Accented words** (áéíóúñü): definitely Spanish.  Require only
      ≥ min_count in ≥ min_varieties.
    - **ASCII-only words**: could be English contamination.  Require
      higher total frequency (≥ ascii_min_total across ALL varieties).
      Real Spanish ASCII words (casa=5000+, perro=200+) pass easily.
      English contaminants (wing=15, king=20, station=10) don't.

    Also requires ≥ min_varieties presence for both tiers.

    Parameters
    ----------
    vocab : list[str]
        The (already filtered) vocabulary.
    corpus_by_variety : dict[str, list[str]]
        Per-variety corpus texts.
    min_count : int
        Minimum occurrences per variety to count as "present."
    min_varieties : int
        Minimum number of varieties where the word must appear.
    ascii_min_total : int
        Minimum total corpus count for ASCII-only words (no accents).

    Returns
    -------
    list[str]
        Vocabulary filtered by corpus evidence.
    """
    # Count per-variety word frequencies
    variety_word_counts: dict[str, Counter] = {}
    for variety, docs in corpus_by_variety.items():
        counter: Counter = Counter()
        for doc in docs:
            for word in doc.strip().lower().split():
                word = word.strip(".,;:!?¿¡\"'()[]{}…-—–")
                if word:
                    counter[word] += 1
        variety_word_counts[variety] = counter

    filtered = []
    rejected_variety = 0
    rejected_freq = 0

    for word in vocab:
        # Count variety presence
        n_varieties = sum(
            1 for counter in variety_word_counts.values()
            if counter.get(word, 0) >= min_count
        )
        if n_varieties < min_varieties:
            rejected_variety += 1
            continue

        # Total frequency across all varieties
        total_count = sum(
            counter.get(word, 0) for counter in variety_word_counts.values()
        )

        # Two-tier threshold
        if _has_spanish_accent(word):
            # Accented → definitely Spanish, low threshold
            filtered.append(word)
        else:
            # ASCII-only → could be English, require high frequency
            if total_count >= ascii_min_total:
                filtered.append(word)
            else:
                rejected_freq += 1

    filtered.sort()

    logger.info(
        "Corpus evidence filter: %d → %d words "
        "(rejected %d <2 varieties, %d low-freq ASCII)",
        len(vocab), len(filtered), rejected_variety, rejected_freq,
    )
    return filtered


def get_anchor_indices(
    vocab: list[str],
    min_anchors: int = 50,
) -> list[int]:
    """Find indices of Procrustes anchor words in the vocabulary.

    Parameters
    ----------
    vocab : list[str]
        The (filtered) vocabulary.
    min_anchors : int
        Minimum number of anchors required.  Raises if fewer found.

    Returns
    -------
    list[int]
        Sorted indices into *vocab* for anchor words.

    Raises
    ------
    ValueError
        If fewer than *min_anchors* anchor words are found.
    """
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    indices = []
    found = []
    missing = []

    for anchor in sorted(SPANISH_ANCHOR_WORDS):
        idx = word_to_idx.get(anchor)
        if idx is not None:
            indices.append(idx)
            found.append(anchor)
        else:
            missing.append(anchor)

    indices.sort()

    if len(indices) < min_anchors:
        raise ValueError(
            f"Only {len(indices)} anchor words found in vocabulary "
            f"(need ≥{min_anchors}).  Missing anchors include: "
            f"{missing[:20]}"
        )

    logger.info(
        "Procrustes anchors: %d/%d found in vocabulary "
        "(missing %d: %s...)",
        len(indices), len(SPANISH_ANCHOR_WORDS),
        len(missing), ", ".join(missing[:5]),
    )
    return indices
