"""Rule-based Spanish morpheme segmentation.

Decomposes Spanish tokens into morpheme sequences using suffix-stripping
with curated tables of verb conjugation endings, derivational morphemes,
diminutive/augmentative suffixes, and clitic pronouns.
"""

from __future__ import annotations

import re
import unicodedata

# ======================================================================
# Helper: strip accents for matching (internal use only)
# ======================================================================

def _strip_accents(s: str) -> str:
    """Remove diacritics, returning ASCII-folded lowercase."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


# ======================================================================
# Clitic pronouns (post-verbal enclitic chains)
# ======================================================================

# Ordered longest-first so "les" matches before "le", etc.
_CLITICS: list[str] = [
    "melo", "mela", "melos", "melas",
    "telo", "tela", "telos", "telas",
    "selo", "sela", "selos", "selas",
    "noslo", "nosla", "noslos", "noslas",
    "oslo", "osla", "oslos", "oslas",
    "nos", "les", "los", "las",
    "me", "te", "se", "lo", "la", "le", "os",
]

# ======================================================================
# Verb conjugation suffixes organized by tense/mood
# Sorted longest-first within each group for greedy matching.
# ======================================================================

# --- Present indicative ---
_PRES_IND: list[str] = [
    "-amos", "-áis", "-emos", "-éis", "-imos",
    "-as", "-es", "-an", "-en",
    "-o", "-a", "-e",
]

# --- Preterite (pretérito indefinido) ---
_PRETERITE: list[str] = [
    "-asteis", "-isteis",
    "-amos", "-imos",
    "-aron", "-ieron",
    "-aste", "-iste",
    "-ó", "-é", "-í",
    "-io",
]

# --- Imperfect indicative (-ar verbs) ---
_IMPERF_AR: list[str] = [
    "-ábamos", "-abais", "-aban",
    "-abas", "-aba",
]

# --- Imperfect indicative (-er/-ir verbs) ---
_IMPERF_ER_IR: list[str] = [
    "-íamos", "-íais", "-ían",
    "-ías", "-ía",
]

# --- Present subjunctive ---
_PRES_SUBJ: list[str] = [
    "-emos", "-éis",
    "-amos", "-áis",
    "-es", "-en",
    "-as", "-an",
    "-e", "-a",
]

# --- Imperfect subjunctive (-ra / -se forms) ---
_IMPERF_SUBJ: list[str] = [
    "-áramos", "-ásemos", "-iéramos", "-iésemos",
    "-arais", "-aseis", "-ierais", "-ieseis",
    "-aran", "-asen", "-ieran", "-iesen",
    "-aras", "-ases", "-ieras", "-ieses",
    "-ara", "-ase", "-iera", "-iese",
]

# --- Future ---
_FUTURE: list[str] = [
    "-remos", "-réis",
    "-rás", "-rán",
    "-ré", "-rá",
]

# --- Conditional ---
_CONDITIONAL: list[str] = [
    "-ríamos", "-ríais", "-rían",
    "-rías", "-ría",
]

# --- Imperative (tú/vosotros) ---
_IMPERATIVE: list[str] = [
    "-ad", "-ed", "-id",
    "-a", "-e",
]

# --- Future subjunctive (archaic, still in legal/literary texts) ---
_FUTURE_SUBJ: list[str] = [
    "-áremos", "-iéremos",
    "-areis", "-iereis",
    "-aren", "-ieren",
    "-ares", "-ieres",
    "-are", "-iere",
]

# --- Voseo conjugation patterns (Argentina, Uruguay, Central America) ---
_VOSEO: list[str] = [
    "-ás", "-és", "-ís",             # present indicative voseo
    "-aste", "-iste",                 # (same as tú preterite, shared)
    "-á", "-é",                       # imperative voseo
]

# --- Perfect compound auxiliary suffixes (on haber) ---
_PERFECT_AUX: list[str] = [
    "-abríamos", "-abríais", "-abrían",
    "-abrías", "-abría",
]

# --- Additional imperfect variants ---
_IMPERF_EXTRA: list[str] = [
    "-ábais",                         # explicit vosotros imperfect -ar
]

# --- Non-finite forms ---
_NONFINITE: list[str] = [
    "-ando", "-iendo", "-yendo",      # gerund
    "-ado", "-ido", "-ada", "-ida",    # past participle (with gender)
    "-ados", "-idos", "-adas", "-idas",
    "-ar", "-er", "-ir",               # infinitive
    "-arse", "-erse", "-irse",         # reflexive infinitive
]

# --- Additional derived verb forms ---
_VERB_DERIVED: list[str] = [
    "-aría", "-ería", "-iría",         # conditional per conjugation class
    "-aré", "-eré", "-iré",            # future per conjugation class
    "-arás", "-erás", "-irás",
    "-ará", "-erá", "-irá",
    "-aremos", "-eremos", "-iremos",
    "-aréis", "-eréis", "-iréis",
    "-arán", "-erán", "-irán",
    "-aríamos", "-eríamos", "-iríamos",
    "-aríais", "-eríais", "-iríais",
    "-arían", "-erían", "-irían",
    "-arías", "-erías", "-irías",
]

# --- Preterite strong (irregular) endings ---
_PRETERITE_STRONG: list[str] = [
    "-uve", "-uviste", "-uvo", "-uvimos", "-uvisteis", "-uvieron",
    "-ise", "-iso",
    "-aje", "-ajo",
]

# --- Present indicative stem-changing verb patterns ---
_STEM_CHANGE: list[str] = [
    "-emos",      # shared form
    "-uelvo", "-uelves", "-uelve", "-uelven",    # o->ue pattern
    "-ierdo", "-ierdes", "-ierde", "-ierden",     # e->ie pattern
    "-ido",        # shared participle
]

# Merge all verb suffixes into a single sorted-by-length-descending list
_ALL_VERB_SUFFIXES: list[str] = sorted(
    set(
        _PRES_IND + _PRETERITE + _IMPERF_AR + _IMPERF_ER_IR +
        _PRES_SUBJ + _IMPERF_SUBJ + _FUTURE + _CONDITIONAL +
        _IMPERATIVE + _FUTURE_SUBJ + _VOSEO + _PERFECT_AUX +
        _IMPERF_EXTRA + _NONFINITE + _VERB_DERIVED +
        _PRETERITE_STRONG + _STEM_CHANGE
    ),
    key=lambda s: len(s),
    reverse=True,
)

# Strip leading dash for actual matching
_VERB_SUFFIX_STRINGS: list[str] = [s.lstrip("-") for s in _ALL_VERB_SUFFIXES]


# ======================================================================
# Diminutive / Augmentative suffixes
# ======================================================================

_DIMINUTIVE_AUGMENTATIVE: list[str] = sorted([
    # Diminutives
    "ecitos", "ecitas", "ecillo", "ecilla",
    "ecito", "ecita",
    "citos", "citas", "cillos", "cillas",
    "cito", "cita", "cillo", "cilla",
    "itos", "itas", "illos", "illas",
    "ito", "ita", "illo", "illa",
    "icos", "icas",
    "ico", "ica",
    "iños", "iñas",
    "iño", "iña",
    "uelos", "uelas",
    "uelo", "uela",
    # Augmentatives
    "azos", "azas",
    "azo", "aza",
    "ones", "onas",
    "ón", "ona",
    "otes", "otas",
    "ote", "ota",
    # Pejorative
    "uchos", "uchas",
    "ucho", "ucha",
    "acos", "acas",
    "aco", "aca",
], key=len, reverse=True)


# ======================================================================
# Derivational suffixes
# ======================================================================

_DERIVATIONAL: list[str] = sorted([
    # Adverbial
    "mente",
    # Nominal
    "ción", "sión", "amiento", "imiento",
    "miento", "encia", "ancia",
    "idad", "edad", "dad", "tud",
    "anza", "anza",
    "aje", "azgo",
    "ura", "ería", "ería",
    "ismo", "ista",
    "ero", "era", "eros", "eras",
    "dor", "dora", "dores", "doras",
    "tor", "tora", "tores", "toras",
    "ante", "ente", "iente",
    # Adjectival
    "oso", "osa", "osos", "osas",
    "ble", "bles",
    "ivo", "iva", "ivos", "ivas",
    "ente", "entes",
    "tico", "tica", "ticos", "ticas",
    "ico", "ica",
    "al", "ales",
    "ario", "aria", "arios", "arias",
    "ado", "ada", "ados", "adas",
    "ido", "ida", "idos", "idas",
    # Verb-forming
    "izar", "ificar", "ecer",
], key=len, reverse=True)


# ======================================================================
# Common irregular stems / stop words that should NOT be segmented
# ======================================================================

_STOP_WORDS: set[str] = {
    # Articles / determiners
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # Prepositions / contractions
    "de", "del", "al", "en", "por", "para", "con", "sin",
    "ante", "bajo", "contra", "entre", "hacia", "hasta", "desde",
    "sobre", "tras", "según", "durante", "mediante",
    # Conjunctions
    "a", "y", "e", "o", "u", "ni", "que", "pero", "sino",
    "como", "donde", "cuando", "porque", "aunque", "pues",
    # Clitics / pronouns (standalone)
    "se", "me", "te", "le", "lo", "nos", "os", "les",
    "su", "sus", "mi", "mis", "tu", "tus",
    "yo", "tú", "él", "ella", "ello",
    "nosotros", "nosotras", "vosotros", "vosotras",
    "ellos", "ellas", "usted", "ustedes", "vos",
    # Demonstratives
    "este", "esta", "estos", "estas", "esto",
    "ese", "esa", "esos", "esas", "eso",
    "aquel", "aquella", "aquellos", "aquellas", "aquello",
    # Indefinites / quantifiers
    "algo", "nada", "nadie", "alguien",
    "mucho", "mucha", "muchos", "muchas",
    "poco", "poca", "pocos", "pocas",
    "todo", "toda", "todos", "todas",
    "otro", "otra", "otros", "otras", "cada",
    # Adverbs / particles
    "no", "sí", "muy", "más", "menos", "tan", "ya", "aún",
    "bien", "mal", "aquí", "ahí", "allí", "acá", "allá",
    "solo", "siempre", "nunca", "también", "tampoco",
    # High-frequency irregular verbs (short forms)
    "es", "son", "fue", "ser", "hay", "ha", "he", "has",
    "era", "eras", "eran",
    "sido", "siendo",
    "ir", "va", "voy", "vas", "van", "iba", "fui",
    "dar", "doy", "das", "da", "dan", "dio",
    "ver", "veo", "ves", "ve", "ven", "vi", "vio",
    "haber", "han", "hemos",
    "poder", "puedo", "puede", "pueden",
    "saber", "sé",
    # Common short nouns / adjectives that should not be stripped
    "casa", "cosa", "vida", "vez", "día", "año", "hombre", "mujer",
    "mundo", "país", "tiempo", "parte", "lado", "modo", "tipo",
    "bueno", "buena", "buenos", "buenas", "buen",
    "malo", "mala", "malos", "malas",
    "grande", "grandes", "gran",
    "nuevo", "nueva", "nuevos", "nuevas",
    "viejo", "vieja", "viejos", "viejas",
    "largo", "larga", "largos", "largas",
    "corto", "corta", "cortos", "cortas",
    "alto", "alta", "altos", "altas",
    "bajo", "baja", "bajos", "bajas",
    "mismo", "misma", "mismos", "mismas",
    "mejor", "peor", "mayor", "menor",
    # Common 4-5 letter words misidentified by suffix rules
    "hace", "hice", "dice", "dije",
    "vale", "sale", "calle", "clase",
    "parque", "padre", "madre", "torre", "noche", "leche",
    "gente", "fuente", "puente", "suerte", "muerte",
    "nombre", "hombre", "hambre", "siempre",
    "agua", "mesa", "cara", "hora", "obra", "idea",
    "vamos", "somos", "damos", "tiene", "viene", "quiere",
    "puede", "sigue", "sabe", "cabe", "pone", "debe",
}

# Minimum stem length after stripping
_MIN_STEM: int = 2


# ======================================================================
# Clitic segmentation on verb forms
# ======================================================================

def _is_valid_enclitic_stem(stem: str) -> bool:
    """Check if *stem* could be a verb form that takes enclitics.

    In Spanish, enclitics attach to:
    - Infinitives: stem ends in -ar, -er, -ir (the 'r' is kept)
    - Gerunds: stem ends in -ando, -endo, -iendo (the vowel before 'ndo')
    - Affirmative imperatives: stem typically ends in a vowel or -d

    After stripping the clitic, the remaining stem should end in:
    a vowel, 'r', 'd', or 'n' (for gerund -ndo -> -ndo+se style).
    """
    if not stem:
        return False
    last = stem[-1]
    # Ends in a vowel (most imperative forms: da, di, pon, ten... or accent: dá, dí)
    if last in "aeiouáéíóú":
        return True
    # Ends in 'r' (infinitive: dar, decir, hacer...)
    if last == "r":
        return True
    # Ends in 'd' (vosotros imperative: hablad, comed, decid)
    if last == "d":
        return True
    # Ends in 'ndo' (gerund: hablando, comiendo -- clitic attaches after -ndo)
    if stem.endswith("ndo"):
        return True
    return False


def _split_compound_clitic(compound: str) -> list[str]:
    """Split compound clitic strings like 'melo' into ['me', 'lo'].

    Compound clitics are two single clitics fused together.
    """
    _SINGLE_CLITICS = ["me", "te", "se", "lo", "la", "le", "nos", "os", "los", "las", "les"]
    # Try splitting at each position
    for i in range(1, len(compound)):
        first = compound[:i]
        second = compound[i:]
        if first in _SINGLE_CLITICS and second in _SINGLE_CLITICS:
            return [first, second]
    # Could not split -- return as single
    return [compound]


def _strip_clitics(token: str) -> tuple[str, list[str]]:
    """Try to peel off enclitic pronouns from the end of a verb form.

    Returns (stem, list_of_clitics) where clitics are in the order they
    were attached (left to right).  E.g. "dámelo" -> ("dá", ["me", "lo"]).

    Only succeeds if the remaining stem looks like a valid verb form that
    can host enclitics (ends in vowel, -r, -d, or -ndo).
    """
    lower = token.lower()
    found_clitics: list[str] = []

    # Try peeling off up to 3 clitics from the right
    changed = True
    while changed and len(lower) > _MIN_STEM:
        changed = False
        for cl in _CLITICS:
            if lower.endswith(cl) and len(lower) - len(cl) >= _MIN_STEM:
                candidate_stem = lower[: -len(cl)]
                # Only accept if stem looks like a valid enclitic host
                # For intermediate steps (more clitics to peel), we also accept
                # stems that end in another clitic sequence
                if _is_valid_enclitic_stem(candidate_stem) or found_clitics:
                    # Expand compound clitics into individual ones
                    parts = _split_compound_clitic(cl)
                    found_clitics = parts + found_clitics
                    lower = candidate_stem
                    changed = True
                    break  # restart from longest

    # Final validation: the leftover stem must be a valid enclitic host
    if found_clitics and _is_valid_enclitic_stem(lower):
        return lower, found_clitics
    return token.lower(), []


# ======================================================================
# Core suffix-stripping algorithm
# ======================================================================

def _try_suffix_strip(token_lower: str, suffixes: list[str]) -> tuple[str, str] | None:
    """Try each suffix (longest first). Return (stem, suffix) or None."""
    for sfx in suffixes:
        if token_lower.endswith(sfx) and len(token_lower) - len(sfx) >= _MIN_STEM:
            return token_lower[: -len(sfx)], sfx
    return None


def _segment_single(token: str) -> list[str]:
    """Segment a single Spanish token into morphemes.

    Strategy:
    1. If it's a stop word or very short, return as-is.
    2. Try clitic stripping (verb + enclitics like "dámelo").
    3. Try derivational suffixes (e.g. "rápidamente" -> "rápida" + "mente").
    4. Try diminutive/augmentative (e.g. "casita" -> "cas" + "ita").
    5. Try verb conjugation suffix stripping.
    6. Fall back to returning the whole token.
    """
    lower = token.lower()

    # Very short words or stop words: no segmentation
    if len(lower) <= 3 or lower in _STOP_WORDS:
        return [token]

    # ------------------------------------------------------------------
    # Step 1: Derivational suffixes (highest static priority)
    # e.g. "rápidamente" -> ["rápida", "mente"]
    # ------------------------------------------------------------------
    result = _try_suffix_strip(lower, _DERIVATIONAL)
    if result is not None:
        stem, sfx = result
        return [stem, sfx]

    # ------------------------------------------------------------------
    # Step 2: Diminutive / Augmentative
    # e.g. "casita" -> ["cas", "ita"], "grandote" -> ["grand", "ote"]
    # ------------------------------------------------------------------
    result = _try_suffix_strip(lower, _DIMINUTIVE_AUGMENTATIVE)
    if result is not None:
        stem, sfx = result
        return [stem, sfx]

    # ------------------------------------------------------------------
    # Step 3: Clitic stripping (verb + enclitics like "dámelo")
    # Tried after derivational/dim-aug to avoid false positives.
    # ------------------------------------------------------------------
    stem_after_clitics, clitics = _strip_clitics(token)
    if clitics:
        # Also try to strip verb suffix from the remaining stem
        inner_parts = _segment_verb_stem(stem_after_clitics)
        return inner_parts + clitics

    # ------------------------------------------------------------------
    # Step 4: Verb conjugation suffix stripping
    # e.g. "hablábamos" -> ["habl", "ábamos"]
    # ------------------------------------------------------------------
    parts = _segment_verb_stem(lower)
    if len(parts) > 1:
        return parts

    # ------------------------------------------------------------------
    # Fallback: return whole token
    # ------------------------------------------------------------------
    return [token]


def _segment_verb_stem(token_lower: str) -> list[str]:
    """Try to split a verb form into stem + conjugation suffix."""
    result = _try_suffix_strip(token_lower, _VERB_SUFFIX_STRINGS)
    if result is not None:
        stem, sfx = result
        return [stem, sfx]
    return [token_lower]


# ======================================================================
# Public API
# ======================================================================

def parse_morphemes(tokens: list[str]) -> list[list[str]]:
    """Segment each token in *tokens* into a list of morphemes.

    Parameters
    ----------
    tokens:
        List of Spanish word tokens (already tokenized).

    Returns
    -------
    list[list[str]]
        For each input token, a list of morpheme strings.
        E.g. ``parse_morphemes(["hablábamos", "casita"])``
        returns ``[["habl", "ábamos"], ["cas", "ita"]]``.
    """
    return [_segment_single(tok) for tok in tokens]
