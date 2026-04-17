"""Rule-based dialect labelling using feature detection heuristics.

The labeler checks for the presence of distinctive dialect markers
(lexical items, morphological patterns, pragmatic discourse markers,
and phonological-orthographic cues) to assign a dialect label and
confidence score.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from eigendialectos.constants import DialectCode, FeatureCategory

# ======================================================================
# Feature detection rules
# ======================================================================

# Each rule: (compiled_regex, weight)
# Higher weight = more distinctive for the dialect

_FeatureRule = tuple[re.Pattern[str], float, str]  # (pattern, weight, description)


def _compile(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    return re.compile(pattern, flags)


_DIALECT_RULES: dict[DialectCode, dict[FeatureCategory, list[_FeatureRule]]] = {
    # ------------------------------------------------------------------
    DialectCode.ES_PEN: {
        FeatureCategory.LEXICAL: [
            (_compile(r'\bordenador\b'), 2.0, "ordenador"),
            (_compile(r'\bmóvil\b'), 1.0, "móvil"),
            (_compile(r'\bgafas\b'), 1.0, "gafas"),
            (_compile(r'\bpiso\b'), 0.5, "piso (vivienda)"),
            (_compile(r'\bmola\b'), 3.0, "mola"),
            (_compile(r'\bflipa\b'), 2.5, "flipa"),
            (_compile(r'\btío\b'), 1.5, "tío (vocativo)"),
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            (_compile(r'\bvosotros\b'), 4.0, "vosotros"),
            (_compile(r'\bvosotras\b'), 4.0, "vosotras"),
            (_compile(r'\bhabéis\b'), 3.0, "habéis"),
            (_compile(r'\btenéis\b'), 3.0, "tenéis"),
            (_compile(r'\bqueréis\b'), 3.0, "queréis"),
            (_compile(r'\bsabéis\b'), 3.0, "sabéis"),
            (_compile(r'\bhemos\b.*\bhoy\b'), 1.0, "pretérito perfecto reciente"),
        ],
        FeatureCategory.PRAGMATIC: [
            (_compile(r',?\s*¿vale\?'), 2.0, "vale (tag)"),
            (_compile(r'\bjoder\b'), 1.5, "joder"),
        ],
        FeatureCategory.PHONOLOGICAL: [
            # Distinction /θ/ is hard to detect in text, but we can note
            # consistent use of z/c before e/i (no seseo)
        ],
    },
    # ------------------------------------------------------------------
    DialectCode.ES_AND: {
        FeatureCategory.LEXICAL: [
            (_compile(r'\bquillo\b'), 5.0, "quillo"),
            (_compile(r'\bquilla\b'), 5.0, "quilla"),
            (_compile(r'\bpicha\b'), 4.0, "picha (vocativo)"),
            (_compile(r'\bbulla\b'), 2.0, "bulla"),
            (_compile(r'\bpechá\b'), 3.0, "pechá"),
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            # -ado > -ao, -ido > -ío patterns
            (_compile(r'\b\w+ao\b'), 1.5, "-ado > -ao"),
            (_compile(r'\b\w+ío\b'), 1.5, "-ido > -ío"),
        ],
        FeatureCategory.PRAGMATIC: [
            (_compile(r'\barsa\b'), 4.0, "arsa"),
        ],
        FeatureCategory.PHONOLOGICAL: [
            (_compile(r'\beh\w*\b'), 1.0, "aspiración s -> h"),
            (_compile(r'\ber\b'), 1.5, "rotacismo el -> er"),
            (_compile(r'\bvamoh\b'), 3.0, "aspiración: vamos -> vamoh"),
            (_compile(r'\blah\b'), 1.5, "aspiración: las -> lah"),
            (_compile(r'\bmusho\b'), 3.0, "aspiración: mucho -> musho"),
            (_compile(r'\bpa\b'), 1.0, "apócope: para -> pa"),
        ],
    },
    # ------------------------------------------------------------------
    DialectCode.ES_CAN: {
        FeatureCategory.LEXICAL: [
            (_compile(r'\bguagua\b'), 3.0, "guagua"),
            (_compile(r'\bgofio\b'), 5.0, "gofio"),
            (_compile(r'\bpapa(?:s)?\b'), 1.0, "papa(s)"),
            (_compile(r'\bbarraquito\b'), 5.0, "barraquito"),
            (_compile(r'\bmachango\b'), 4.0, "machango"),
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            (_compile(r'\bustedes\b.*\bsaben\b'), 2.0, "ustedes saben"),
            (_compile(r'\bustedes\b.*\btienen\b'), 2.0, "ustedes tienen"),
        ],
        FeatureCategory.PRAGMATIC: [
            (_compile(r'\bchacho\b'), 4.0, "chacho"),
            (_compile(r'\bchacha\b'), 4.0, "chacha"),
            (_compile(r'\bmijo\b'), 1.5, "mijo"),
            (_compile(r'\bmija\b'), 1.5, "mija"),
        ],
        FeatureCategory.PHONOLOGICAL: [],
    },
    # ------------------------------------------------------------------
    DialectCode.ES_RIO: {
        FeatureCategory.LEXICAL: [
            (_compile(r'\bcolectivo\b'), 3.0, "colectivo"),
            (_compile(r'\bbondi\b'), 5.0, "bondi"),
            (_compile(r'\bpibe\b'), 4.0, "pibe"),
            (_compile(r'\bpiba\b'), 4.0, "piba"),
            (_compile(r'\blaburar\b'), 5.0, "laburar"),
            (_compile(r'\blaburo\b'), 5.0, "laburo"),
            (_compile(r'\bmorfar\b'), 5.0, "morfar"),
            (_compile(r'\bbirra\b'), 3.0, "birra"),
            (_compile(r'\bguita\b'), 4.0, "guita"),
            (_compile(r'\bbárbaro\b'), 2.0, "bárbaro"),
            (_compile(r'\bfernet\b'), 3.0, "fernet"),
            (_compile(r'\bmina\b'), 2.0, "mina (mujer)"),
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            (_compile(r'\bvos\b'), 3.0, "vos (pronombre)"),
            (_compile(r'\btenés\b'), 4.0, "voseo: tenés"),
            (_compile(r'\bquerés\b'), 4.0, "voseo: querés"),
            (_compile(r'\bsabés\b'), 4.0, "voseo: sabés"),
            (_compile(r'\bpodés\b'), 4.0, "voseo: podés"),
            (_compile(r'\bvenís\b'), 3.0, "voseo: venís"),
            (_compile(r'\bmirá\b'), 3.0, "imperativo voseante: mirá"),
            (_compile(r'\bvení\b'), 3.5, "imperativo voseante: vení"),
            (_compile(r'\bdecí\b'), 3.5, "imperativo voseante: decí"),
        ],
        FeatureCategory.PRAGMATIC: [
            (_compile(r'\bche\b'), 4.0, "che"),
            (_compile(r'\bdale\b'), 2.0, "dale"),
            (_compile(r'\bboludo\b'), 5.0, "boludo"),
            (_compile(r'\bboluda\b'), 5.0, "boluda"),
            (_compile(r'¿viste\?'), 3.0, "¿viste?"),
            (_compile(r'\bre\s'), 2.0, "re (intensificador)"),
        ],
        FeatureCategory.PHONOLOGICAL: [
            (_compile(r'\bsho\b'), 4.0, "yeísmo rehilado: sho"),
        ],
    },
    # ------------------------------------------------------------------
    DialectCode.ES_MEX: {
        FeatureCategory.LEXICAL: [
            (_compile(r'\bcamión\b'), 1.5, "camión (bus)"),
            (_compile(r'\bchamba\b'), 4.0, "chamba"),
            (_compile(r'\bchamaco\b'), 4.0, "chamaco"),
            (_compile(r'\bchela\b'), 3.0, "chela"),
            (_compile(r'\blana\b'), 2.0, "lana (dinero)"),
            (_compile(r'\bferia\b'), 1.5, "feria (dinero)"),
            (_compile(r'\bcuate\b'), 4.0, "cuate"),
            (_compile(r'\bchido\b'), 5.0, "chido"),
            (_compile(r'\bpadre\b'), 1.0, "padre (bueno)"),
            (_compile(r'\bpadrísim[oa]\b'), 5.0, "padrísimo/a"),
            (_compile(r'\bneta\b'), 3.0, "neta"),
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            (_compile(r'\bahorita\b'), 3.0, "ahorita"),
            (_compile(r'\bcerquita\b'), 2.0, "cerquita"),
            (_compile(r'\btodito\b'), 2.0, "todito"),
        ],
        FeatureCategory.PRAGMATIC: [
            (_compile(r'\bgüey\b'), 5.0, "güey"),
            (_compile(r'\bwey\b'), 5.0, "wey"),
            (_compile(r'\bórale\b'), 5.0, "órale"),
            (_compile(r'\bándale\b'), 4.0, "ándale"),
            (_compile(r'\bmande\b'), 3.0, "mande"),
            (_compile(r'no manches'), 5.0, "no manches"),
            (_compile(r',?\s*¿va\?'), 2.0, "¿va? (confirmación)"),
        ],
        FeatureCategory.PHONOLOGICAL: [],
    },
    # ------------------------------------------------------------------
    DialectCode.ES_CAR: {
        FeatureCategory.LEXICAL: [
            (_compile(r'\bguagua\b'), 2.5, "guagua"),
            (_compile(r'\bvaina\b'), 4.0, "vaina"),
            (_compile(r'\bchévere\b'), 4.0, "chévere"),
            (_compile(r'\bpana\b'), 3.0, "pana"),
            (_compile(r'\bchamo\b'), 5.0, "chamo"),
            (_compile(r'\bchama\b'), 5.0, "chama"),
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            (_compile(r'¿qué tú\b'), 4.0, "no inversión: ¿qué tú...?"),
            (_compile(r'¿cómo tú\b'), 4.0, "no inversión: ¿cómo tú...?"),
            (_compile(r'¿dónde tú\b'), 4.0, "no inversión: ¿dónde tú...?"),
        ],
        FeatureCategory.PRAGMATIC: [
            (_compile(r'\bmijo\b'), 2.0, "mijo"),
            (_compile(r'\bmija\b'), 2.0, "mija"),
            (_compile(r'\basere\b'), 5.0, "asere"),
            (_compile(r'¿oíste\?'), 4.0, "¿oíste?"),
            (_compile(r"pa'"), 1.5, "pa' (apócope)"),
        ],
        FeatureCategory.PHONOLOGICAL: [
            (_compile(r"na'"), 2.0, "elisión: nada -> na'"),
            (_compile(r"to'"), 2.0, "elisión: todo -> to'"),
            (_compile(r'\b\w+ao\b'), 1.0, "-ado > -ao"),
        ],
    },
    # ------------------------------------------------------------------
    DialectCode.ES_CHI: {
        FeatureCategory.LEXICAL: [
            (_compile(r'\bmicro\b'), 2.0, "micro (bus)"),
            (_compile(r'\bpolol[oa]\b'), 5.0, "pololo/a"),
            (_compile(r'\bpega\b'), 2.5, "pega (trabajo)"),
            (_compile(r'\bfome\b'), 5.0, "fome"),
            (_compile(r'\bbacán\b'), 3.0, "bacán"),
            (_compile(r'\bluca\b'), 4.0, "luca"),
            (_compile(r'\bcarrete\b'), 4.0, "carrete"),
            (_compile(r'\bcopete\b'), 3.0, "copete"),
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            (_compile(r'\bsabís\b'), 5.0, "voseo chileno: sabís"),
            (_compile(r'\bquerís\b'), 5.0, "voseo chileno: querís"),
            (_compile(r'\btenís\b'), 5.0, "voseo chileno: tenís"),
            (_compile(r'\bpodís\b'), 5.0, "voseo chileno: podís"),
            (_compile(r'\bvenís\b'), 3.0, "voseo: venís"),
            (_compile(r'\bpensái\b'), 5.0, "voseo chileno: pensái"),
            (_compile(r'\bentendís\b'), 4.0, "voseo chileno: entendís"),
        ],
        FeatureCategory.PRAGMATIC: [
            (_compile(r'¿cachai\?'), 5.0, "¿cachai?"),
            (_compile(r'\bcachai\b'), 5.0, "cachai"),
            (_compile(r'\bcacharon\b'), 4.0, "cacharon"),
            (_compile(r'\bhueón\b'), 5.0, "hueón"),
            (_compile(r'\bweón\b'), 5.0, "weón"),
            (_compile(r'\bpo\b'), 4.0, "po (pues)"),
            (_compile(r'\bla raja\b'), 4.0, "la raja"),
            (_compile(r'\bal tiro\b'), 3.0, "al tiro"),
            (_compile(r'\bcaleta\b'), 3.0, "caleta"),
            (_compile(r'\btinca\b'), 4.0, "tinca"),
        ],
        FeatureCategory.PHONOLOGICAL: [],
    },
    # ------------------------------------------------------------------
    DialectCode.ES_AND_BO: {
        FeatureCategory.LEXICAL: [
            (_compile(r'\bcaserito\b'), 5.0, "caserito"),
            (_compile(r'\bchacra\b'), 4.0, "chacra"),
            (_compile(r'\bcombo\b'), 1.5, "combo (menú)"),
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            (_compile(r'\bnomás\b'), 4.0, "nomás"),
            (_compile(r'\bno más\b'), 2.0, "no más (atenuación)"),
            (_compile(r'su\s+\w+\s+de\s+(?:mi|tu|su)\b'), 4.0, "doble posesivo"),
        ],
        FeatureCategory.PRAGMATIC: [
            (_compile(r',?\s*¿ya\?'), 3.0, "¿ya? (confirmación)"),
            (_compile(r'\bpues\b'), 1.5, "pues"),
            (_compile(r'\bpe\b'), 3.5, "pe (pues apocopado)"),
            (_compile(r',?\s*pues\s*[.,]'), 2.0, "pues (muletilla)"),
            (_compile(r'\boiga\b'), 1.5, "oiga"),
        ],
        FeatureCategory.PHONOLOGICAL: [],
    },
}


class DialectLabeler:
    """Rule-based dialect labeller using weighted feature detection.

    The labeller scores each dialect by summing the weights of detected
    features, then picks the dialect with the highest score.  Confidence
    is derived from the margin between the top two scores.
    """

    def __init__(self) -> None:
        self._rules = _DIALECT_RULES

    def detect_features(
        self,
        text: str,
    ) -> dict[FeatureCategory, list[str]]:
        """Detect all dialectal features present in *text*.

        Returns a dict mapping feature categories to lists of human-readable
        descriptions of detected features.
        """
        detected: dict[FeatureCategory, list[str]] = defaultdict(list)
        for _code, categories in self._rules.items():
            for cat, rules in categories.items():
                for pattern, _weight, description in rules:
                    if pattern.search(text):
                        if description not in detected[cat]:
                            detected[cat].append(description)
        return dict(detected)

    def label(self, text: str) -> tuple[DialectCode, float]:
        """Predict the dialect of *text* and return ``(code, confidence)``.

        Confidence is in [0, 1].  A score of 0.0 means no dialect features
        were detected (falls back to ``ES_PEN`` as the unmarked default).
        """
        scores: dict[DialectCode, float] = {code: 0.0 for code in DialectCode}

        for code, categories in self._rules.items():
            for _cat, rules in categories.items():
                for pattern, weight, _desc in rules:
                    matches = pattern.findall(text)
                    if matches:
                        scores[code] += weight * len(matches)

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_code, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_score == 0.0:
            # No features detected -- default to peninsular
            return DialectCode.ES_PEN, 0.0

        # Confidence: margin normalised by best score
        total = sum(s for _, s in ranked)
        if total > 0:
            confidence = best_score / total
        else:
            confidence = 0.0

        # Boost confidence when margin is large
        if best_score > 0 and second_score >= 0:
            margin = (best_score - second_score) / best_score
            confidence = min(1.0, confidence * (0.5 + 0.5 * margin))

        return best_code, round(confidence, 4)

    def label_detailed(
        self,
        text: str,
    ) -> dict[str, object]:
        """Return detailed labelling information.

        Includes per-dialect scores, detected features, and the final
        prediction.
        """
        scores: dict[DialectCode, float] = {code: 0.0 for code in DialectCode}
        per_dialect_features: dict[DialectCode, list[str]] = defaultdict(list)

        for code, categories in self._rules.items():
            for _cat, rules in categories.items():
                for pattern, weight, desc in rules:
                    matches = pattern.findall(text)
                    if matches:
                        scores[code] += weight * len(matches)
                        per_dialect_features[code].append(
                            f"{desc} (x{len(matches)}, w={weight})"
                        )

        code, confidence = self.label(text)
        return {
            "prediction": code,
            "confidence": confidence,
            "scores": {k.value: v for k, v in scores.items()},
            "features": {k.value: v for k, v in per_dialect_features.items()},
        }
