"""Linguistic quality survey generator for dialect evaluation."""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from eigendialectos.constants import DialectCode, DIALECT_NAMES
from eigendialectos.validation.metrics import compute_krippendorff_alpha


# ======================================================================
# HTML template fragments
# ======================================================================

_CSS = """\
<style>
  :root { --accent: #2c5282; --bg: #f7fafc; --card: #ffffff; --border: #e2e8f0; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: #1a202c; line-height: 1.6; padding: 2rem; }
  h1 { color: var(--accent); margin-bottom: .5rem; }
  h2 { color: var(--accent); font-size: 1.1rem; margin: 1.5rem 0 .75rem; }
  .instructions { background: #ebf8ff; border-left: 4px solid var(--accent); padding: 1rem; margin-bottom: 2rem; border-radius: 4px; }
  .sample-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
  .sample-text { background: #edf2f7; padding: .75rem 1rem; border-radius: 4px; font-style: italic; margin: .5rem 0 1rem; white-space: pre-wrap; }
  .question { margin-bottom: 1rem; }
  .question label { display: block; font-weight: 600; margin-bottom: .25rem; }
  .likert { display: flex; gap: 1rem; align-items: center; }
  .likert label { font-weight: 400; }
  select, input[type=radio] { accent-color: var(--accent); }
  select { padding: .35rem .5rem; border: 1px solid var(--border); border-radius: 4px; }
  button { background: var(--accent); color: #fff; border: none; padding: .6rem 1.5rem; border-radius: 4px; cursor: pointer; font-size: 1rem; margin-top: 1rem; }
  button:hover { opacity: .9; }
  .metadata { display: none; }
</style>
"""

_SCRIPT_TEMPLATE = """\
<script>
document.getElementById('survey-form').addEventListener('submit', function(e) {
  e.preventDefault();
  var data = {};
  var formData = new FormData(this);
  for (var pair of formData.entries()) {
    data[pair[0]] = pair[1];
  }
  data['_metadata'] = JSON.parse(document.getElementById('survey-metadata').textContent);
  var blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'survey_responses.json';
  a.click();
});
</script>
"""


# ======================================================================
# SurveyGenerator
# ======================================================================

class SurveyGenerator:
    """Generate and analyse linguistic quality surveys.

    Parameters
    ----------
    config : dict
        Configuration dictionary.  Recognised keys:

        - ``seed`` (int): random seed (default 42).
        - ``max_samples_per_dialect`` (int): cap on samples shown (default 10).
        - ``title`` (str): survey page title.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed: int = config.get("seed", 42)
        self.max_per_dialect: int = config.get("max_samples_per_dialect", 10)
        self.title: str = config.get("title", "Evaluacion de Variedad Dialectal")

    # ---------------------------------------------------------
    # Survey creation
    # ---------------------------------------------------------

    def create_survey(
        self,
        real_samples: dict[DialectCode, list[str]],
        generated_samples: dict[DialectCode, list[str]],
    ) -> str:
        """Create an HTML survey mixing real and generated samples.

        Questions per sample:
        (a) Which dialect variety is this?
        (b) Naturalness rating (1--5 Likert).
        (c) Dialectal identity strength (1--5 Likert).

        Parameters
        ----------
        real_samples : dict[DialectCode, list[str]]
        generated_samples : dict[DialectCode, list[str]]

        Returns
        -------
        str
            Complete HTML document.
        """
        rng = random.Random(self.seed)

        # Build item list: (text, dialect, origin)
        items: list[dict[str, Any]] = []
        for dialect in sorted(set(real_samples) | set(generated_samples), key=lambda d: d.value):
            for text in real_samples.get(dialect, [])[: self.max_per_dialect]:
                items.append({
                    "id": str(uuid.uuid4())[:8],
                    "text": text,
                    "dialect": dialect.value,
                    "origin": "real",
                })
            for text in generated_samples.get(dialect, [])[: self.max_per_dialect]:
                items.append({
                    "id": str(uuid.uuid4())[:8],
                    "text": text,
                    "dialect": dialect.value,
                    "origin": "generated",
                })

        rng.shuffle(items)

        # Build HTML
        dialect_options = "\n".join(
            f'          <option value="{d.value}">{DIALECT_NAMES.get(d, d.value)}</option>'
            for d in DialectCode
        )

        cards_html = []
        for idx, item in enumerate(items, 1):
            card = f"""\
    <div class="sample-card" data-item-id="{item['id']}">
      <h2>Muestra {idx}</h2>
      <div class="sample-text">{_escape_html(item['text'])}</div>

      <div class="question">
        <label for="variety_{item['id']}">a) &iquest;Qu&eacute; variedad dialectal es esta?</label>
        <select name="variety_{item['id']}" id="variety_{item['id']}" required>
          <option value="">-- Seleccione --</option>
{dialect_options}
        </select>
      </div>

      <div class="question">
        <label>b) Naturalidad (1 = muy artificial, 5 = completamente natural)</label>
        <div class="likert">
{_likert_radios(f"naturalness_{item['id']}", 5)}
        </div>
      </div>

      <div class="question">
        <label>c) Identidad dialectal (1 = nada marcada, 5 = muy marcada)</label>
        <div class="likert">
{_likert_radios(f"identity_{item['id']}", 5)}
        </div>
      </div>
    </div>"""
            cards_html.append(card)

        metadata = json.dumps(
            [{"id": it["id"], "dialect": it["dialect"], "origin": it["origin"]} for it in items]
        )

        html = f"""\
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_escape_html(self.title)}</title>
{_CSS}
</head>
<body>
  <h1>{_escape_html(self.title)}</h1>
  <div class="instructions">
    <p>A continuaci&oacute;n se presentan textos en distintas variedades del espa&ntilde;ol.
    Para cada muestra, responda las tres preguntas.</p>
  </div>
  <form id="survey-form">
{"".join(cards_html)}
    <button type="submit">Enviar respuestas</button>
  </form>
  <script id="survey-metadata" type="application/json">{metadata}</script>
{_SCRIPT_TEMPLATE}
</body>
</html>"""
        return html

    # ---------------------------------------------------------
    # Response parsing
    # ---------------------------------------------------------

    def parse_responses(self, response_file: Path) -> dict[str, Any]:
        """Parse a JSON response file produced by the survey.

        Parameters
        ----------
        response_file : Path

        Returns
        -------
        dict
            Parsed response dictionary.
        """
        with open(response_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    # ---------------------------------------------------------
    # Response analysis
    # ---------------------------------------------------------

    def analyze_responses(self, responses: dict[str, Any]) -> dict[str, Any]:
        """Analyse a parsed response dictionary.

        Parameters
        ----------
        responses : dict
            Keys include ``variety_<id>``, ``naturalness_<id>``,
            ``identity_<id>``, and ``_metadata`` (list of item dicts).

        Returns
        -------
        dict
            ``naturalness_mean``, ``identity_mean``, ``identification_accuracy``,
            ``agreement_alpha``, per-origin breakdown.
        """
        metadata = responses.get("_metadata", [])
        item_map = {it["id"]: it for it in metadata}

        naturalness_scores: list[float] = []
        identity_scores: list[float] = []
        correct_identifications = 0
        total_identifications = 0

        origin_stats: dict[str, dict[str, list[float]]] = {
            "real": {"naturalness": [], "identity": []},
            "generated": {"naturalness": [], "identity": []},
        }

        for item_id, info in item_map.items():
            variety_key = f"variety_{item_id}"
            nat_key = f"naturalness_{item_id}"
            ident_key = f"identity_{item_id}"

            guessed_variety = responses.get(variety_key)
            nat_score_raw = responses.get(nat_key)
            ident_score_raw = responses.get(ident_key)

            origin = info.get("origin", "unknown")

            if nat_score_raw is not None:
                nat = float(nat_score_raw)
                naturalness_scores.append(nat)
                if origin in origin_stats:
                    origin_stats[origin]["naturalness"].append(nat)

            if ident_score_raw is not None:
                ident = float(ident_score_raw)
                identity_scores.append(ident)
                if origin in origin_stats:
                    origin_stats[origin]["identity"].append(ident)

            if guessed_variety is not None:
                total_identifications += 1
                if guessed_variety == info["dialect"]:
                    correct_identifications += 1

        # Inter-annotator agreement (single-rater degenerates to trivial)
        # Build a ratings matrix for naturalness (1 rater x N items)
        n_items = len(metadata)
        if naturalness_scores and n_items > 0:
            ratings = np.full((1, n_items), np.nan)
            for i, item in enumerate(metadata):
                val = responses.get(f"naturalness_{item['id']}")
                if val is not None:
                    ratings[0, i] = float(val)
            alpha = compute_krippendorff_alpha(ratings)
        else:
            alpha = float("nan")

        def _safe_mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        per_origin = {}
        for origin, stats in origin_stats.items():
            per_origin[origin] = {
                "naturalness_mean": _safe_mean(stats["naturalness"]),
                "identity_mean": _safe_mean(stats["identity"]),
                "count": len(stats["naturalness"]),
            }

        return {
            "naturalness_mean": _safe_mean(naturalness_scores),
            "identity_mean": _safe_mean(identity_scores),
            "identification_accuracy": (
                correct_identifications / total_identifications
                if total_identifications > 0
                else 0.0
            ),
            "total_items": n_items,
            "agreement_alpha": alpha,
            "per_origin": per_origin,
        }


# ======================================================================
# Helpers
# ======================================================================

def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _likert_radios(name: str, n: int) -> str:
    parts = []
    for i in range(1, n + 1):
        parts.append(
            f'          <label><input type="radio" name="{name}" value="{i}" required> {i}</label>'
        )
    return "\n".join(parts)
