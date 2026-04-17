"""Dialect transformation templates.

Each dialect has a set of transformation rules organised by feature type:
- **lexical**: direct word/phrase substitutions
- **morphological**: regex-based morphological pattern replacements
- **pragmatic**: discourse markers and interjections characteristic of the dialect
- **phonological**: spelling/orthographic transformations that reflect pronunciation

Rules are applied to neutral Peninsular-style Spanish to produce dialectal text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from eigendialectos.constants import DialectCode


@dataclass
class TransformationRule:
    """A single text transformation rule."""
    pattern: str          # regex pattern or literal string
    replacement: str      # replacement text (may use group refs for regex)
    is_regex: bool = False
    description: str = ""


@dataclass
class DialectTemplate:
    """Complete transformation template for a dialect."""
    lexical: dict[str, str] = field(default_factory=dict)
    morphological: list[TransformationRule] = field(default_factory=list)
    pragmatic_markers: list[str] = field(default_factory=list)
    phonological: list[TransformationRule] = field(default_factory=list)

    def apply_lexical(self, text: str) -> str:
        """Apply lexical substitutions (case-insensitive, word-boundary)."""
        result = text
        for original, replacement in self.lexical.items():
            pattern = re.compile(
                r'\b' + re.escape(original) + r'\b',
                re.IGNORECASE,
            )
            # Preserve case of first character
            def _replace(m: re.Match, repl: str = replacement) -> str:
                matched = m.group(0)
                if matched[0].isupper():
                    return repl[0].upper() + repl[1:]
                return repl
            result = pattern.sub(_replace, result)
        return result

    def apply_morphological(self, text: str) -> str:
        """Apply morphological regex transformations (case-insensitive)."""
        result = text
        for rule in self.morphological:
            if rule.is_regex:
                # Use IGNORECASE and a helper to preserve leading case
                compiled = re.compile(rule.pattern, re.IGNORECASE)
                replacement = rule.replacement

                def _repl(m: re.Match, rep: str = replacement) -> str:
                    out = m.expand(rep)
                    # Preserve capitalisation of the first character
                    if m.group(0) and m.group(0)[0].isupper() and out and out[0].islower():
                        out = out[0].upper() + out[1:]
                    return out

                result = compiled.sub(_repl, result)
            else:
                result = result.replace(rule.pattern, rule.replacement)
        return result

    def apply_phonological(self, text: str) -> str:
        """Apply phonological/orthographic transformations."""
        result = text
        for rule in self.phonological:
            if rule.is_regex:
                result = re.sub(rule.pattern, rule.replacement, result)
            else:
                result = result.replace(rule.pattern, rule.replacement)
        return result

    def apply_all(self, text: str) -> str:
        """Apply all transformations in order: lexical -> morphological -> phonological."""
        text = self.apply_lexical(text)
        text = self.apply_morphological(text)
        text = self.apply_phonological(text)
        return text


# ======================================================================
# Templates per dialect
# ======================================================================

DIALECT_TEMPLATES: dict[DialectCode, DialectTemplate] = {
    # ------------------------------------------------------------------
    # ES_PEN -- Peninsular (identity/baseline -- minimal transforms)
    # ------------------------------------------------------------------
    DialectCode.ES_PEN: DialectTemplate(
        lexical={},
        morphological=[],
        pragmatic_markers=["vale", "tío", "mola", "¿no?", "joder", "oye"],
        phonological=[],
    ),

    # ------------------------------------------------------------------
    # ES_AND -- Andalusian
    # ------------------------------------------------------------------
    DialectCode.ES_AND: DialectTemplate(
        lexical={
            "chico": "quillo",
            "chica": "quilla",
            "prisa": "bulla",
            "mucho": "musho",
        },
        morphological=[
            # ustedes replaces vosotros (but sometimes with vosotros verb endings)
            TransformationRule(
                r'\bvosotros\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotros",
            ),
            TransformationRule(
                r'\bvosotras\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotras",
            ),
            # -ado -> -ao (participio)
            TransformationRule(
                r'\b(\w+)ado\b', r'\1ao', is_regex=True,
                description="caída de -d- intervocálica en -ado",
            ),
            # -ido -> -ío
            TransformationRule(
                r'\b(\w+)ido\b', r'\1ío', is_regex=True,
                description="caída de -d- intervocálica en -ido",
            ),
        ],
        pragmatic_markers=["quillo", "quilla", "arsa", "vale", "bah", "picha"],
        phonological=[
            # aspiración de -s ante consonante: es -> eh
            TransformationRule(
                r's\b', 'h', is_regex=True,
                description="aspiración de -s implosiva a final de sílaba",
            ),
            # el -> er (lateral to rhotic before consonant in casual speech)
            TransformationRule(
                r'\bel\b', 'er', is_regex=True,
                description="rotacismo: el -> er",
            ),
            # para -> pa
            TransformationRule(
                r'\bpara\b', 'pa', is_regex=True,
                description="apócope: para -> pa",
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # ES_CAN -- Canarian
    # ------------------------------------------------------------------
    DialectCode.ES_CAN: DialectTemplate(
        lexical={
            "autobús": "guagua",
            "patata": "papa",
            "patatas": "papas",
        },
        morphological=[
            # ustedes replaces vosotros
            TransformationRule(
                r'\bvosotros\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotros",
            ),
            TransformationRule(
                r'\bvosotras\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotras",
            ),
            TransformationRule(
                r'\bhabéis\b', 'han', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\btenéis\b', 'tienen', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bqueréis\b', 'quieren', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bsabéis\b', 'saben', is_regex=True,
                description="ustedes conjugation",
            ),
        ],
        pragmatic_markers=["chacho", "chacha", "mijo", "mija", "¿verdad?"],
        phonological=[
            # seseo: z before vowel -> s
            TransformationRule(
                r'z([aeiou])', r's\1', is_regex=True,
                description="seseo: z -> s ante vocal",
            ),
            # ce, ci -> se, si
            TransformationRule(
                r'c([ei])', r's\1', is_regex=True,
                description="seseo: ce/ci -> se/si",
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # ES_RIO -- Rioplatense
    # ------------------------------------------------------------------
    DialectCode.ES_RIO: DialectTemplate(
        lexical={
            "autobús": "colectivo",
            "ordenador": "computadora",
            "computadora": "computadora",
            "coche": "auto",
            "carro": "auto",
            "gafas": "anteojos",
            "piso": "departamento",
            "apartamento": "departamento",
            "móvil": "celular",
            "teléfono móvil": "celular",
            "chico": "pibe",
            "chica": "piba",
            "trabajo": "laburo",
            "trabajar": "laburar",
            "dinero": "guita",
            "cerveza": "birra",
            "genial": "bárbaro",
        },
        morphological=[
            # tú -> vos
            TransformationRule(
                r'\btú\b', 'vos', is_regex=True,
                description="pronombre vos",
            ),
            # Voseo verbal: tienes -> tenés
            TransformationRule(
                r'\btienes\b', 'tenés', is_regex=True,
                description="voseo: tienes -> tenés",
            ),
            TransformationRule(
                r'\bquieres\b', 'querés', is_regex=True,
                description="voseo: quieres -> querés",
            ),
            TransformationRule(
                r'\bsabes\b', 'sabés', is_regex=True,
                description="voseo: sabes -> sabés",
            ),
            TransformationRule(
                r'\bpuedes\b', 'podés', is_regex=True,
                description="voseo: puedes -> podés",
            ),
            TransformationRule(
                r'\bvienes\b', 'venís', is_regex=True,
                description="voseo: vienes -> venís",
            ),
            TransformationRule(
                r'\bpiensas\b', 'pensás', is_regex=True,
                description="voseo: piensas -> pensás",
            ),
            TransformationRule(
                r'\bsientes\b', 'sentís', is_regex=True,
                description="voseo: sientes -> sentís",
            ),
            # Imperative voseante
            TransformationRule(
                r'\bmira\b', 'mirá', is_regex=True,
                description="imperativo voseante: mira -> mirá",
            ),
            TransformationRule(
                r'\bven\b', 'vení', is_regex=True,
                description="imperativo voseante: ven -> vení",
            ),
            TransformationRule(
                r'\bdi\b', 'decí', is_regex=True,
                description="imperativo voseante: di -> decí",
            ),
            # Pretérito perfecto compuesto -> indefinido
            TransformationRule(
                r'\bhe ido\b', 'fui', is_regex=True,
                description="perfecto simple: he ido -> fui",
            ),
            TransformationRule(
                r'\bhe comido\b', 'comí', is_regex=True,
                description="perfecto simple: he comido -> comí",
            ),
            TransformationRule(
                r'\bhe visto\b', 'vi', is_regex=True,
                description="perfecto simple: he visto -> vi",
            ),
        ],
        pragmatic_markers=[
            "che", "dale", "boludo", "¿viste?", "mirá", "¿entendés?",
            "re", "bárbaro",
        ],
        phonological=[
            # Yeísmo rehilado: ll/y -> sh (orthographic representation)
            # We represent it only in select words for readability
            TransformationRule(
                r'\byo\b', 'sho', is_regex=True,
                description="yeísmo rehilado: yo -> sho",
            ),
            TransformationRule(
                r'\bya\b', 'sha', is_regex=True,
                description="yeísmo rehilado: ya -> sha",
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # ES_MEX -- Mexican
    # ------------------------------------------------------------------
    DialectCode.ES_MEX: DialectTemplate(
        lexical={
            "autobús": "camión",
            "ordenador": "computadora",
            "coche": "carro",
            "gafas": "lentes",
            "piso": "departamento",
            "apartamento": "departamento",
            "móvil": "celular",
            "genial": "padrísimo",
            "trabajo": "chamba",
            "trabajar": "chambear",
            "chico": "chamaco",
            "dinero": "lana",
            "cerveza": "chela",
        },
        morphological=[
            # Diminutivos extendidos
            TransformationRule(
                r'\bahora\b', 'ahorita', is_regex=True,
                description="diminutivo: ahora -> ahorita",
            ),
            TransformationRule(
                r'\bcerca\b', 'cerquita', is_regex=True,
                description="diminutivo: cerca -> cerquita",
            ),
            # le intensivo in some expressions
            TransformationRule(
                r'\bánda\b', 'ándale', is_regex=True,
                description="le intensivo: ánda -> ándale",
            ),
        ],
        pragmatic_markers=[
            "güey", "wey", "¿va?", "órale", "ándale", "¿mande?",
            "no manches", "chido", "padre", "neta",
        ],
        phonological=[
            # seseo (same as other Latin American)
            TransformationRule(
                r'z([aeiou])', r's\1', is_regex=True,
                description="seseo: z -> s ante vocal",
            ),
            TransformationRule(
                r'c([ei])', r's\1', is_regex=True,
                description="seseo: ce/ci -> se/si",
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # ES_CAR -- Caribbean
    # ------------------------------------------------------------------
    DialectCode.ES_CAR: DialectTemplate(
        lexical={
            "autobús": "guagua",
            "ordenador": "computadora",
            "coche": "carro",
            "gafas": "gafas",
            "piso": "apartamento",
            "móvil": "celular",
            "genial": "chévere",
            "bueno": "chévere",
            "amigo": "pana",
            "cosa": "vaina",
        },
        morphological=[
            # Sujeto pronominal expreso (no inversión en preguntas)
            TransformationRule(
                r'¿qué quieres', '¿qué tú quieres', is_regex=False,
                description="sujeto pronominal: ¿qué quieres -> ¿qué tú quieres",
            ),
            TransformationRule(
                r'¿dónde vas', '¿dónde tú vas', is_regex=False,
                description="sujeto pronominal: ¿dónde vas -> ¿dónde tú vas",
            ),
            TransformationRule(
                r'¿cómo estás', '¿cómo tú estás', is_regex=False,
                description="sujeto pronominal: ¿cómo estás -> ¿cómo tú estás",
            ),
        ],
        pragmatic_markers=[
            "mijo", "mija", "¿oíste?", "chévere", "asere", "pana",
            "¿tú sabes?", "pa'",
        ],
        phonological=[
            # seseo
            TransformationRule(
                r'z([aeiou])', r's\1', is_regex=True,
                description="seseo",
            ),
            TransformationRule(
                r'c([ei])', r's\1', is_regex=True,
                description="seseo",
            ),
            # aspiración / pérdida de -s
            TransformationRule(
                r's\b', "'", is_regex=True,
                description="aspiración/elisión de -s final",
            ),
            # para -> pa'
            TransformationRule(
                r'\bpara\b', "pa'", is_regex=True,
                description="apócope: para -> pa'",
            ),
            # -ado -> -ao
            TransformationRule(
                r'\b(\w+)ado\b', r"\1ao", is_regex=True,
                description="elisión de -d- intervocálica en -ado",
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # ES_CHI -- Chilean
    # ------------------------------------------------------------------
    DialectCode.ES_CHI: DialectTemplate(
        lexical={
            "autobús": "micro",
            "ordenador": "computador",
            "coche": "auto",
            "gafas": "lentes",
            "piso": "depa",
            "apartamento": "depa",
            "móvil": "celu",
            "genial": "bacán",
            "inmediatamente": "al tiro",
            "novia": "polola",
            "novio": "pololo",
            "trabajo": "pega",
            "aburrido": "fome",
            "fiesta": "carrete",
            "cerveza": "chela",
        },
        morphological=[
            # Voseo mixto verbal chileno: tú + verb ending -ís / -ái
            TransformationRule(
                r'\bsabes\b', 'sabís', is_regex=True,
                description="voseo chileno: sabes -> sabís",
            ),
            TransformationRule(
                r'\bquieres\b', 'querís', is_regex=True,
                description="voseo chileno: quieres -> querís",
            ),
            TransformationRule(
                r'\btienes\b', 'tenís', is_regex=True,
                description="voseo chileno: tienes -> tenís",
            ),
            TransformationRule(
                r'\bpuedes\b', 'podís', is_regex=True,
                description="voseo chileno: puedes -> podís",
            ),
            TransformationRule(
                r'\bvienes\b', 'venís', is_regex=True,
                description="voseo chileno: vienes -> venís",
            ),
            TransformationRule(
                r'\bpiensas\b', 'pensái', is_regex=True,
                description="voseo chileno: piensas -> pensái",
            ),
            TransformationRule(
                r'\bentiendes\b', 'entendís', is_regex=True,
                description="voseo chileno: entiendes -> entendís",
            ),
        ],
        pragmatic_markers=[
            "¿cachai?", "hueón", "weón", "po", "sí po", "no po",
            "ya po", "la raja", "al tiro", "caleta",
        ],
        phonological=[
            # seseo
            TransformationRule(
                r'z([aeiou])', r's\1', is_regex=True,
                description="seseo",
            ),
            TransformationRule(
                r'c([ei])', r's\1', is_regex=True,
                description="seseo",
            ),
            # para -> pa'
            TransformationRule(
                r'\bpara\b', "pa'", is_regex=True,
                description="apócope: para -> pa'",
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # ES_AND_BO -- Andean (Peru / Bolivia / Ecuador highlands)
    # ------------------------------------------------------------------
    DialectCode.ES_AND_BO: DialectTemplate(
        lexical={
            "autobús": "bus",
            "coche": "carro",
            "ordenador": "computadora",
            "gafas": "lentes",
            "piso": "departamento",
            "apartamento": "departamento",
            "móvil": "celular",
        },
        morphological=[
            # nomás usage
            TransformationRule(
                r'\bpase\b', 'pase nomás', is_regex=True,
                description="atenuación andina: pase -> pase nomás",
            ),
            TransformationRule(
                r'\bsírvase\b', 'sírvase nomás', is_regex=True,
                description="atenuación andina: sírvase -> sírvase nomás",
            ),
            # doble posesivo
            TransformationRule(
                r'la casa de mi (\w+)', r'su casa de mi \1', is_regex=True,
                description="doble posesivo andino",
            ),
        ],
        pragmatic_markers=[
            "¿ya?", "pues", "pe", "nomás", "oye", "oiga",
            "sí pues", "ya pues",
        ],
        phonological=[
            # seseo
            TransformationRule(
                r'z([aeiou])', r's\1', is_regex=True,
                description="seseo",
            ),
            TransformationRule(
                r'c([ei])', r's\1', is_regex=True,
                description="seseo",
            ),
        ],
    ),
}


def get_template(dialect_code: DialectCode) -> DialectTemplate:
    """Return the transformation template for the given dialect.

    Raises
    ------
    KeyError
        If no template is defined for *dialect_code*.
    """
    if dialect_code not in DIALECT_TEMPLATES:
        label = dialect_code.value if hasattr(dialect_code, "value") else dialect_code
        raise KeyError(f"No template for dialect {label}")
    return DIALECT_TEMPLATES[dialect_code]


def list_templates() -> list[DialectCode]:
    """Return dialect codes that have transformation templates."""
    return list(DIALECT_TEMPLATES.keys())
