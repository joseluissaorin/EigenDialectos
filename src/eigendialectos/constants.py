"""Project-wide constants for EigenDialectos."""

from __future__ import annotations

from enum import Enum, IntEnum


class DialectCode(str, Enum):
    """ISO-style codes for major Spanish dialect varieties."""

    ES_PEN = "ES_PEN"  # Peninsular Standard
    ES_AND = "ES_AND"  # Andalusian
    ES_CAN = "ES_CAN"  # Canarian
    ES_RIO = "ES_RIO"  # Rioplatense
    ES_MEX = "ES_MEX"  # Mexican
    ES_CAR = "ES_CAR"  # Caribbean
    ES_CHI = "ES_CHI"  # Chilean
    ES_AND_BO = "ES_AND_BO"  # Andean


class LinguisticLevel(IntEnum):
    """Linguistic levels for multi-level spectral analysis."""

    MORPHEME = 1
    WORD = 2
    PHRASE = 3
    SENTENCE = 4
    DISCOURSE = 5


DIALECT_NAMES: dict[DialectCode, str] = {
    DialectCode.ES_PEN: "Castellano peninsular estándar",
    DialectCode.ES_AND: "Andaluz",
    DialectCode.ES_CAN: "Canario",
    DialectCode.ES_RIO: "Rioplatense",
    DialectCode.ES_MEX: "Mexicano",
    DialectCode.ES_CAR: "Caribeño",
    DialectCode.ES_CHI: "Chileno",
    DialectCode.ES_AND_BO: "Andino",
}

DIALECT_REGIONS: dict[DialectCode, str] = {
    DialectCode.ES_PEN: "España (centro-norte)",
    DialectCode.ES_AND: "España (Andalucía)",
    DialectCode.ES_CAN: "España (Islas Canarias)",
    DialectCode.ES_RIO: "Argentina / Uruguay",
    DialectCode.ES_MEX: "México",
    DialectCode.ES_CAR: "Cuba / Puerto Rico / Rep. Dominicana / Venezuela costera",
    DialectCode.ES_CHI: "Chile",
    DialectCode.ES_AND_BO: "Perú / Bolivia / Ecuador (sierra)",
}


class FeatureCategory(str, Enum):
    """Linguistic feature categories tracked by the framework."""

    LEXICAL = "LEXICAL"
    MORPHOSYNTACTIC = "MORPHOSYNTACTIC"
    PRAGMATIC = "PRAGMATIC"
    PHONOLOGICAL = "PHONOLOGICAL"
    TEMPORAL = "TEMPORAL"


EMBEDDING_DIMS: dict[str, int] = {
    "subword": 300,
    "word": 300,
    "sentence": 768,
}

ALPHA_RANGE: tuple[float, float, float] = (0.0, 1.5, 0.1)
"""(start, stop, step) for dialectal intensity parameter alpha."""

DEFAULT_SEED: int = 42

MIN_CORPUS_SIZE: int = 1000
"""Minimum number of samples required per dialect variety."""


# Geographic coordinates for dialect centers (lat, lon) — for eigenvalue field
DIALECT_COORDINATES: dict[DialectCode, tuple[float, float]] = {
    DialectCode.ES_CAN: (28.1, -15.4),       # Las Palmas
    DialectCode.ES_AND: (37.4, -6.0),        # Sevilla
    DialectCode.ES_PEN: (40.4, -3.7),        # Madrid
    DialectCode.ES_RIO: (-34.6, -58.4),      # Buenos Aires
    DialectCode.ES_MEX: (19.4, -99.1),       # CDMX
    DialectCode.ES_CAR: (23.1, -82.4),       # La Habana
    DialectCode.ES_CHI: (-33.4, -70.6),      # Santiago
    DialectCode.ES_AND_BO: (-16.5, -68.1),   # La Paz
}

# Dialect family groupings for multi-granularity decomposition
DIALECT_FAMILIES: dict[str, list[DialectCode]] = {
    "peninsular": [DialectCode.ES_PEN, DialectCode.ES_AND, DialectCode.ES_CAN],
    "caribbean": [DialectCode.ES_CAR],
    "southern_cone": [DialectCode.ES_RIO, DialectCode.ES_CHI],
    "mesoamerican": [DialectCode.ES_MEX],
    "andean": [DialectCode.ES_AND_BO],
}

LINGUISTIC_LEVEL_NAMES: dict[LinguisticLevel, str] = {
    LinguisticLevel.MORPHEME: "Morpheme",
    LinguisticLevel.WORD: "Word/Lemma",
    LinguisticLevel.PHRASE: "Phrase/Collocation",
    LinguisticLevel.SENTENCE: "Sentence",
    LinguisticLevel.DISCOURSE: "Discourse",
}
