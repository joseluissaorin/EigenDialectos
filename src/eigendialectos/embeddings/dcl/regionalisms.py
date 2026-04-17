"""Regionalism sets for all Spanish dialect varieties.

Combines curated linguist-verified regionalisms with a larger
LLM-generated dictionary.  The DCL loss uses these to decide whether
the anchor regularisation term should apply: words in ANY variety's
regionalism set are exempted from the cross-variety anchor penalty.
"""

from __future__ import annotations

from eigendialectos.embeddings.dcl.llm_regionalisms import LLM_REGIONALISMS

# ---------------------------------------------------------------------------
# Curated per-variety regionalism sets (linguist-verified core)
# ---------------------------------------------------------------------------

_CURATED_REGIONALISMS: dict[str, set[str]] = {
    "ES_PEN": set(),
    "ES_AND": {
        "quillo", "picha", "churumbel", "chiquillo", "pisha", "chaval",
        "arsa", "illo", "compae", "bulla", "pringao", "gazpacho",
        "salmorejo", "pestorejo", "malaje", "chipén", "amoto",
        "perchelero", "chirigota", "marengo", "esnortao", "pelete",
        "mijilla", "arrecío", "tajá",
    },
    "ES_CAN": {
        "guagua", "papa", "mojo", "gofio", "pelete", "baifo",
        "guanche", "machango", "piña", "beletén", "jareas", "tunera",
        "magua", "enyesque", "lepe", "fisco", "perenquén", "bubango",
        "cherne", "sancocho", "frangollo", "cambullón", "guanajo",
        "achiperres", "jable",
    },
    "ES_RIO": {
        "che", "pibe", "piba", "mina", "laburo", "afanar", "bondi",
        "birra", "guita", "fiaca", "quilombo", "morfar", "trucho",
        "groso", "chabón", "pucho", "garpar", "laburar", "remera",
        "campera", "colectivo", "subte", "boludo", "macana", "bancar",
        "pilcha", "mate", "pochoclo", "vereda",
    },
    "ES_MEX": {
        "güey", "wey", "chido", "neta", "chamba", "chamaco", "mole",
        "órale", "naco", "fresa", "chafa", "padre", "cuate", "pinche",
        "morro", "chela", "camión", "antro", "chavo", "lana", "bronca",
        "chingar", "escuincle", "popote", "alberca", "banqueta",
        "cajuela", "chamarra",
    },
    "ES_CAR": {
        "chévere", "vaina", "chamo", "pana", "jeva", "tipo", "guagua",
        "bachata", "jíbaro", "bochinche", "disparate", "coro",
        "tripear", "china", "habichuela", "mangú", "mofongo",
        "chingar", "pargo", "plátano", "bemba", "prieto", "cocolazo",
        "asere", "ecobio", "jinetear", "yunta", "guarapo",
    },
    "ES_CHI": {
        "pololo", "polola", "fome", "cachai", "bacán", "luca",
        "al tiro", "pololear", "carrete", "cuático", "filete",
        "huevón", "weón", "gallo", "mina", "pega", "micro", "flaite",
        "copete", "guagua", "chupalla", "altiro", "palta", "once",
        "empanada", "completo", "polera", "guata", "locomoción",
    },
    "ES_AND_BO": {
        "cholo", "chompa", "pollera", "chuño", "wawa", "soroche",
        "chacra", "cancha", "calato", "cuy", "ñaño", "pata",
        "jato", "causa", "chaufa", "chicha", "charqui", "quinua",
        "papa", "yapa", "chifa", "huayno", "puna", "ayllu",
        "anticucho", "pachamanca", "lúcuma", "choclo",
    },
}

# ---------------------------------------------------------------------------
# Merged: curated + LLM-generated (union per variety)
# ---------------------------------------------------------------------------

REGIONALISMS: dict[str, set[str]] = {}
_all_keys = set(_CURATED_REGIONALISMS.keys()) | set(LLM_REGIONALISMS.keys())
for _key in _all_keys:
    REGIONALISMS[_key] = (
        _CURATED_REGIONALISMS.get(_key, set())
        | LLM_REGIONALISMS.get(_key, set())
    )

# ---------------------------------------------------------------------------
# Aggregate set for quick membership tests
# ---------------------------------------------------------------------------

ALL_REGIONALISMS: frozenset[str] = frozenset().union(*REGIONALISMS.values())
