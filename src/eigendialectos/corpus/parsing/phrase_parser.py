"""Rule-based Spanish phrase chunking using word-class heuristics.

Assigns pseudo-POS tags via lookup tables and groups tokens into noun
phrases (NP), prepositional phrases (PP), and verb phrases (VP) using
greedy left-to-right chunking.
"""

from __future__ import annotations

from enum import Enum

# ======================================================================
# Pseudo-POS tag set
# ======================================================================

class POS(str, Enum):
    DET = "DET"
    NOUN = "NOUN"
    ADJ = "ADJ"
    VERB = "VERB"
    PREP = "PREP"
    CONJ = "CONJ"
    ADV = "ADV"
    PRON = "PRON"
    PUNCT = "PUNCT"
    UNKNOWN = "UNK"


# ======================================================================
# Word lookup tables (~300+ entries)
# ======================================================================

_DETERMINERS: set[str] = {
    # Definite articles
    "el", "la", "los", "las", "lo",
    # Indefinite articles
    "un", "una", "unos", "unas",
    # Demonstratives
    "este", "esta", "estos", "estas", "esto",
    "ese", "esa", "esos", "esas", "eso",
    "aquel", "aquella", "aquellos", "aquellas", "aquello",
    # Possessives (pre-nominal)
    "mi", "mis", "tu", "tus", "su", "sus",
    "nuestro", "nuestra", "nuestros", "nuestras",
    "vuestro", "vuestra", "vuestros", "vuestras",
    # Quantifiers / indefinites
    "todo", "toda", "todos", "todas",
    "otro", "otra", "otros", "otras",
    "mucho", "mucha", "muchos", "muchas",
    "poco", "poca", "pocos", "pocas",
    "algún", "alguno", "alguna", "algunos", "algunas",
    "ningún", "ninguno", "ninguna", "ningunos", "ningunas",
    "cada", "varios", "varias",
    "mismo", "misma", "mismos", "mismas",
    "cierto", "cierta", "ciertos", "ciertas",
    "cualquier", "cualquiera", "cualesquiera",
    "ambos", "ambas",
    "demás",
    "bastante", "bastantes",
    "tanto", "tanta", "tantos", "tantas",
    "cuanto", "cuanta", "cuantos", "cuantas",
    "sendos", "sendas",
    # Contracted forms
    "del", "al",
}

_PREPOSITIONS: set[str] = {
    "a", "ante", "bajo", "cabe", "con", "contra",
    "de", "desde", "durante", "en", "entre",
    "hacia", "hasta", "mediante", "para", "por",
    "según", "sin", "so", "sobre", "tras",
    # Compound prepositions (treated as single tokens if already tokenized together)
    "excepto", "salvo", "incluso",
    # Note: "del" and "al" are contractions but handled as DET above
}

_CONJUNCTIONS: set[str] = {
    # Coordinating
    "y", "e", "o", "u", "ni", "pero", "sino",
    "mas", "aunque", "mientras",
    # Subordinating
    "que", "porque", "como", "si", "cuando",
    "donde", "quien", "quienes", "cual", "cuales",
    "cuyo", "cuya", "cuyos", "cuyas",
    "pues", "puesto", "ya", "luego",
    "conque", "así",
}

_ADVERBS: set[str] = {
    # Manner
    "bien", "mal", "mejor", "peor", "así", "despacio", "deprisa",
    "rápido", "lento", "fuerte", "claro",
    # Place
    "aquí", "ahí", "allí", "acá", "allá",
    "cerca", "lejos", "dentro", "fuera",
    "arriba", "abajo", "delante", "detrás",
    "encima", "debajo", "enfrente", "alrededor",
    # Time
    "hoy", "ayer", "mañana", "ahora", "antes", "después",
    "luego", "pronto", "tarde", "temprano",
    "siempre", "nunca", "jamás", "todavía", "aún", "ya",
    "recién", "apenas", "enseguida",
    # Quantity
    "mucho", "poco", "bastante", "demasiado",
    "más", "menos", "tan", "tanto", "muy",
    "algo", "nada", "casi", "solo",
    # Affirmation/negation/doubt
    "sí", "no", "también", "tampoco",
    "quizá", "quizás", "acaso", "tal",
    # Others
    "además", "incluso", "solamente",
    "especialmente", "realmente", "simplemente",
}

_PRONOUNS: set[str] = {
    # Personal subject
    "yo", "tú", "él", "ella", "ello",
    "nosotros", "nosotras", "vosotros", "vosotras",
    "ellos", "ellas", "usted", "ustedes",
    # Reflexive / object
    "me", "te", "se", "nos", "os",
    "mí", "ti", "sí", "consigo", "conmigo", "contigo",
    # Demonstrative pronouns (same form as det but used pronominally)
    # Handled via DET
    # Relative/interrogative
    "qué", "quién", "quiénes", "cuál", "cuáles",
    "dónde", "cómo", "cuándo", "cuánto", "cuánta", "cuántos", "cuántas",
    # Indefinite pronouns
    "algo", "alguien", "nada", "nadie",
    "uno", "una",
    # Voseo
    "vos",
}

# Common adjectives for lookup (beyond suffix heuristics)
_ADJECTIVES: set[str] = {
    "bueno", "buena", "buenos", "buenas", "buen",
    "malo", "mala", "malos", "malas", "mal",
    "grande", "grandes", "gran",
    "pequeño", "pequeña", "pequeños", "pequeñas",
    "nuevo", "nueva", "nuevos", "nuevas",
    "viejo", "vieja", "viejos", "viejas",
    "largo", "larga", "largos", "largas",
    "corto", "corta", "cortos", "cortas",
    "alto", "alta", "altos", "altas",
    "bajo", "baja", "bajos", "bajas",
    "joven", "jóvenes",
    "mejor", "peor", "mayor", "menor",
    "primero", "primera", "primeros", "primeras", "primer",
    "segundo", "segunda", "segundos", "segundas",
    "tercero", "tercera", "terceros", "terceras", "tercer",
    "último", "última", "últimos", "últimas",
    "propio", "propia", "propios", "propias",
    "solo", "sola", "solos", "solas",
    "posible", "posibles",
    "importante", "importantes",
    "diferente", "diferentes",
    "libre", "libres",
    "claro", "clara", "claros", "claras",
    "distinto", "distinta", "distintos", "distintas",
    "necesario", "necesaria", "necesarios", "necesarias",
    "verdadero", "verdadera", "verdaderos", "verdaderas",
    "general", "generales",
    "público", "pública", "públicos", "públicas",
    "social", "sociales",
    "político", "política", "políticos", "políticas",
    "económico", "económica", "económicos", "económicas",
    "humano", "humana", "humanos", "humanas",
    "único", "única", "únicos", "únicas",
    "difícil", "difíciles", "fácil", "fáciles",
    "simple", "simples",
    "bonito", "bonita", "bonitos", "bonitas",
    "feo", "fea", "feos", "feas",
    "rico", "rica", "ricos", "ricas",
    "pobre", "pobres",
    "feliz", "felices",
    "triste", "tristes",
    "contento", "contenta", "contentos", "contentas",
    "rojo", "roja", "rojos", "rojas",
    "azul", "azules",
    "verde", "verdes",
    "blanco", "blanca", "blancos", "blancas",
    "negro", "negra", "negros", "negras",
    "amarillo", "amarilla", "amarillos", "amarillas",
    "español", "española", "españoles", "españolas",
}

# Common verb forms (high-frequency, irregular, or auxiliary)
_VERBS: set[str] = {
    # ser
    "ser", "soy", "eres", "es", "somos", "sois", "son",
    "era", "eras", "éramos", "erais", "eran",
    "fui", "fuiste", "fue", "fuimos", "fuisteis", "fueron",
    "seré", "serás", "será", "seremos", "seréis", "serán",
    "sería", "serías", "seríamos", "seríais", "serían",
    "sea", "seas", "seamos", "seáis", "sean",
    "fuera", "fueras", "fuéramos", "fuerais", "fueran",
    "fuese", "fueses", "fuésemos", "fueseis", "fuesen",
    "sido", "siendo", "sé",
    # estar
    "estar", "estoy", "estás", "está", "estamos", "estáis", "están",
    "estaba", "estabas", "estábamos", "estabais", "estaban",
    "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron",
    "estaré", "estarás", "estará", "estaremos", "estaréis", "estarán",
    "estaría", "estarías", "estaríamos", "estaríais", "estarían",
    "esté", "estés", "estemos", "estéis", "estén",
    "estado", "estando",
    # haber
    "haber", "he", "has", "ha", "hemos", "habéis", "han",
    "había", "habías", "habíamos", "habíais", "habían",
    "hube", "hubiste", "hubo", "hubimos", "hubisteis", "hubieron",
    "habré", "habrás", "habrá", "habremos", "habréis", "habrán",
    "habría", "habrías", "habríamos", "habríais", "habrían",
    "haya", "hayas", "hayamos", "hayáis", "hayan",
    "hubiera", "hubieras", "hubiéramos", "hubierais", "hubieran",
    "habido", "habiendo", "hay",
    # tener
    "tener", "tengo", "tienes", "tiene", "tenemos", "tenéis", "tienen",
    "tenía", "tenías", "teníamos", "teníais", "tenían",
    "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis", "tuvieron",
    "tendré", "tendrás", "tendrá", "tendremos", "tendréis", "tendrán",
    "tendría", "tendrías", "tendríamos", "tendríais", "tendrían",
    "tenga", "tengas", "tengamos", "tengáis", "tengan",
    "tenido", "teniendo",
    # hacer
    "hacer", "hago", "haces", "hace", "hacemos", "hacéis", "hacen",
    "hacía", "hacías", "hacíamos", "hacíais", "hacían",
    "hice", "hiciste", "hizo", "hicimos", "hicisteis", "hicieron",
    "haré", "harás", "hará", "haremos", "haréis", "harán",
    "haría", "harías", "haríamos", "haríais", "harían",
    "haga", "hagas", "hagamos", "hagáis", "hagan",
    "hecho", "haciendo",
    # ir
    "ir", "voy", "vas", "va", "vamos", "vais", "van",
    "iba", "ibas", "íbamos", "ibais", "iban",
    "iré", "irás", "irá", "iremos", "iréis", "irán",
    "iría", "irías", "iríamos", "iríais", "irían",
    "vaya", "vayas", "vayamos", "vayáis", "vayan",
    "ido", "yendo",
    # poder
    "poder", "puedo", "puedes", "puede", "podemos", "podéis", "pueden",
    "podía", "podías", "podíamos", "podíais", "podían",
    "pude", "pudiste", "pudo", "pudimos", "pudisteis", "pudieron",
    "podré", "podrás", "podrá", "podremos", "podréis", "podrán",
    "podría", "podrías", "podríamos", "podríais", "podrían",
    "pueda", "puedas", "podamos", "podáis", "puedan",
    "podido", "pudiendo",
    # decir
    "decir", "digo", "dices", "dice", "decimos", "decís", "dicen",
    "decía", "decías", "decíamos", "decíais", "decían",
    "dije", "dijiste", "dijo", "dijimos", "dijisteis", "dijeron",
    "diré", "dirás", "dirá", "diremos", "diréis", "dirán",
    "diría", "dirías", "diríamos", "diríais", "dirían",
    "diga", "digas", "digamos", "digáis", "digan",
    "dicho", "diciendo",
    # dar
    "dar", "doy", "das", "da", "damos", "dais", "dan",
    "daba", "dabas", "dábamos", "dabais", "daban",
    "di", "diste", "dio", "dimos", "disteis", "dieron",
    "daré", "darás", "dará", "daremos", "daréis", "darán",
    "daría", "darías", "daríamos", "daríais", "darían",
    "dé", "des", "demos", "deis", "den",
    "dado", "dando",
    # saber
    "saber", "sé", "sabes", "sabe", "sabemos", "sabéis", "saben",
    "sabía", "sabías", "sabíamos", "sabíais", "sabían",
    "supe", "supiste", "supo", "supimos", "supisteis", "supieron",
    "sabré", "sabrás", "sabrá", "sabremos", "sabréis", "sabrán",
    "sabría", "sabrías", "sabríamos", "sabríais", "sabrían",
    "sepa", "sepas", "sepamos", "sepáis", "sepan",
    "sabido", "sabiendo",
    # querer
    "querer", "quiero", "quieres", "quiere", "queremos", "queréis", "quieren",
    "quería", "querías", "queríamos", "queríais", "querían",
    "quise", "quisiste", "quiso", "quisimos", "quisisteis", "quisieron",
    "querré", "querrás", "querrá", "querremos", "querréis", "querrán",
    "querría", "querrías", "querríamos", "querríais", "querrían",
    "quiera", "quieras", "queramos", "queráis", "quieran",
    "querido", "queriendo",
    # ver
    "ver", "veo", "ves", "ve", "vemos", "veis", "ven",
    "veía", "veías", "veíamos", "veíais", "veían",
    "vi", "viste", "vio", "vimos", "visteis", "vieron",
    "veré", "verás", "verá", "veremos", "veréis", "verán",
    "vería", "verías", "veríamos", "veríais", "verían",
    "vea", "veas", "veamos", "veáis", "vean",
    "visto", "viendo",
    # Common regular forms that look like other POS
    "hablar", "comer", "vivir", "trabajar", "llamar",
    "pensar", "creer", "parecer", "quedar", "pasar",
    "llegar", "llevar", "dejar", "seguir", "encontrar",
    "venir", "salir", "poner", "tomar", "conocer",
    "sentir", "contar", "empezar", "buscar", "escribir",
    "perder", "producir", "ocurrir", "entender", "pedir",
    "recibir", "recordar", "terminar", "permitir", "aparecer",
    "conseguir", "comenzar", "servir", "sacar", "necesitar",
    "mantener", "resultar", "leer", "caer", "cambiar",
    "presentar", "crear", "abrir", "considerar", "oír",
    "acabar", "convertir", "ganar", "formar", "traer",
    "partir", "morir", "aceptar", "realizar", "suponer",
    "comprender", "lograr", "explicar", "tocar", "reconocer",
}

# Adjective suffix heuristics for unknown words
_ADJ_SUFFIXES: tuple[str, ...] = (
    "oso", "osa", "osos", "osas",
    "ivo", "iva", "ivos", "ivas",
    "ble", "bles",
    "ico", "ica", "icos", "icas",
    "ente", "entes",
    "ante", "antes",
    "ario", "aria", "arios", "arias",
    "al", "ales",
)

# Verb suffix heuristics for unknown words
_VERB_SUFFIXES_HEURISTIC: tuple[str, ...] = (
    "ar", "er", "ir",
    "ando", "iendo", "yendo",
    "ado", "ido",
    "aba", "abas", "ábamos", "aban",
    "ía", "ías", "íamos", "ían",
)

# Noun suffix heuristics
_NOUN_SUFFIXES: tuple[str, ...] = (
    "ción", "sión", "idad", "dad", "eza",
    "ismo", "ista", "miento", "amiento", "imiento",
    "ancia", "encia", "aje", "ura",
    "ero", "era", "ería",
    "dor", "dora", "tor", "tora",
)


# ======================================================================
# Pseudo-POS tagger
# ======================================================================

def _assign_pos(token: str) -> POS:
    """Assign a pseudo-POS tag to *token* via lookup + suffix heuristics."""
    lower = token.lower()

    # Punctuation
    if not any(c.isalnum() for c in token):
        return POS.PUNCT

    # Lookup tables (order matters: most specific first)
    if lower in _PREPOSITIONS:
        return POS.PREP
    if lower in _CONJUNCTIONS:
        return POS.CONJ
    if lower in _DETERMINERS:
        return POS.DET
    if lower in _PRONOUNS:
        return POS.PRON
    if lower in _VERBS:
        return POS.VERB
    if lower in _ADVERBS:
        return POS.ADV
    if lower in _ADJECTIVES:
        return POS.ADJ

    # Suffix heuristics for unknown words
    if any(lower.endswith(sfx) for sfx in _VERB_SUFFIXES_HEURISTIC):
        return POS.VERB
    if any(lower.endswith(sfx) for sfx in _NOUN_SUFFIXES):
        return POS.NOUN
    if any(lower.endswith(sfx) for sfx in _ADJ_SUFFIXES):
        return POS.ADJ

    # Default: assume noun
    return POS.NOUN


# ======================================================================
# Greedy phrase chunker
# ======================================================================

def _chunk_np(tagged: list[tuple[str, POS]], start: int) -> tuple[list[str], int]:
    """Try to consume a noun phrase starting at *start*.

    NP = Det? Adj* (Noun|Pron) Adj*

    Returns (tokens_in_chunk, next_index).
    """
    i = start
    n = len(tagged)
    tokens: list[str] = []

    # Optional determiner
    if i < n and tagged[i][1] == POS.DET:
        tokens.append(tagged[i][0])
        i += 1

    # Pre-nominal adjectives
    while i < n and tagged[i][1] == POS.ADJ:
        tokens.append(tagged[i][0])
        i += 1

    # Head noun or pronoun (required)
    if i < n and tagged[i][1] in (POS.NOUN, POS.PRON):
        tokens.append(tagged[i][0])
        i += 1
    elif not tokens:
        # No NP found
        return [], start
    else:
        # We had det/adj but no noun -- still a valid fragment
        return tokens, i

    # Post-nominal adjectives
    while i < n and tagged[i][1] == POS.ADJ:
        tokens.append(tagged[i][0])
        i += 1

    return tokens, i


def _chunk_pp(tagged: list[tuple[str, POS]], start: int) -> tuple[list[str], int]:
    """Try to consume a prepositional phrase: PP = Prep + NP.

    Returns (tokens_in_chunk, next_index).
    """
    i = start
    n = len(tagged)

    if i >= n or tagged[i][1] != POS.PREP:
        return [], start

    prep_token = tagged[i][0]
    i += 1

    # Try to get the NP complement
    np_tokens, next_i = _chunk_np(tagged, i)
    if np_tokens:
        return [prep_token] + np_tokens, next_i
    else:
        # Bare preposition, return it alone
        return [prep_token], i


def _chunk_vp(tagged: list[tuple[str, POS]], start: int) -> tuple[list[str], int]:
    """Try to consume a verb phrase: VP = Adv* Verb+ Adv*.

    Returns (tokens_in_chunk, next_index).
    """
    i = start
    n = len(tagged)
    tokens: list[str] = []

    # Pre-verbal adverbs (e.g. "no", "ya", "siempre")
    while i < n and tagged[i][1] == POS.ADV:
        tokens.append(tagged[i][0])
        i += 1

    # At least one verb required
    if i >= n or tagged[i][1] != POS.VERB:
        if not tokens:
            return [], start
        # Adverbs without a verb -- return them as a chunk anyway
        return tokens, i

    while i < n and tagged[i][1] == POS.VERB:
        tokens.append(tagged[i][0])
        i += 1

    # Post-verbal adverbs
    while i < n and tagged[i][1] == POS.ADV:
        tokens.append(tagged[i][0])
        i += 1

    return tokens, i


# ======================================================================
# Public API
# ======================================================================

def parse_phrases(tokens: list[str]) -> list[list[str]]:
    """Chunk *tokens* into phrase groups using pseudo-POS heuristics.

    The chunker attempts, at each position, to match:
    1. PP (Prep + NP)
    2. VP (Adv* + Verb+ + Adv*)
    3. NP (Det? + Adj* + Noun + Adj*)
    4. Fallback: single-token chunk

    Parameters
    ----------
    tokens:
        List of Spanish word tokens.

    Returns
    -------
    list[list[str]]
        Each inner list is a phrase chunk (group of tokens).
    """
    if not tokens:
        return []

    tagged = [(tok, _assign_pos(tok)) for tok in tokens]
    chunks: list[list[str]] = []
    i = 0
    n = len(tagged)

    while i < n:
        # Skip punctuation as its own chunk
        if tagged[i][1] == POS.PUNCT:
            chunks.append([tagged[i][0]])
            i += 1
            continue

        # Skip conjunctions as their own chunk (clause boundary)
        if tagged[i][1] == POS.CONJ:
            chunks.append([tagged[i][0]])
            i += 1
            continue

        # Try PP first (Prep + NP)
        pp_tokens, pp_end = _chunk_pp(tagged, i)
        if pp_tokens:
            chunks.append(pp_tokens)
            i = pp_end
            continue

        # Try VP
        vp_tokens, vp_end = _chunk_vp(tagged, i)
        if vp_tokens:
            chunks.append(vp_tokens)
            i = vp_end
            continue

        # Try NP
        np_tokens, np_end = _chunk_np(tagged, i)
        if np_tokens:
            chunks.append(np_tokens)
            i = np_end
            continue

        # Fallback: single-token chunk
        chunks.append([tagged[i][0]])
        i += 1

    return chunks
