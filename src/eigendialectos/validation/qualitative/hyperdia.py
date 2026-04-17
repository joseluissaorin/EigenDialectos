"""Hyperdialectal evaluation: assess naturalness across alpha intensities."""

from __future__ import annotations

import json
import random
import uuid
from typing import Any

from eigendialectos.constants import DialectCode, DIALECT_NAMES


# ======================================================================
# CSS
# ======================================================================

_CSS = """\
<style>
  :root { --accent: #6b46c1; --bg: #faf5ff; --card: #fff; --border: #e9d8fd; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: #1a202c; line-height: 1.6; padding: 2rem; max-width: 900px; margin: 0 auto; }
  h1 { color: var(--accent); margin-bottom: .5rem; }
  h2 { color: var(--accent); font-size: 1.05rem; margin: 1rem 0 .5rem; }
  .instructions { background: #faf5ff; border-left: 4px solid var(--accent); padding: 1rem; margin-bottom: 2rem; border-radius: 4px; }
  .alpha-group { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
  .alpha-badge { display: inline-block; background: var(--accent); color: #fff; padding: 2px 10px; border-radius: 12px; font-size: .85rem; margin-bottom: .5rem; }
  .sample-text { background: #f0e6ff; padding: .75rem 1rem; border-radius: 4px; font-style: italic; margin: .5rem 0; white-space: pre-wrap; }
  .question { margin-top: .75rem; margin-bottom: .5rem; }
  .question label { font-weight: 600; }
  .likert { display: flex; gap: 1rem; align-items: center; margin-top: .25rem; }
  .likert label { font-weight: 400; }
  input[type=radio] { accent-color: var(--accent); }
  select { padding: .35rem .5rem; border: 1px solid var(--border); border-radius: 4px; }
  button { background: var(--accent); color: #fff; border: none; padding: .6rem 1.5rem; border-radius: 4px; cursor: pointer; font-size: 1rem; margin-top: 1rem; }
  button:hover { opacity: .9; }
</style>
"""


# ======================================================================
# HyperdialectalEvaluator
# ======================================================================

class HyperdialectalEvaluator:
    """Evaluate the naturalness boundary of hyperdialectal text generation.

    Texts are generated at various dialectal-intensity levels (alpha values).
    Evaluators rate whether the text sounds exaggerated, natural, and at which
    point it becomes unnatural.

    Parameters
    ----------
    alpha_values : list[float]
        The alpha intensities to evaluate (default ``[1.0, 1.2, 1.5]``).
    seed : int
        Random seed (default 42).
    """

    def __init__(
        self,
        alpha_values: list[float] | None = None,
        seed: int = 42,
    ) -> None:
        self.alpha_values = alpha_values if alpha_values is not None else [1.0, 1.2, 1.5]
        self.seed = seed

    # ---------------------------------------------------------
    # Evaluation creation
    # ---------------------------------------------------------

    def create_evaluation(
        self,
        texts_by_alpha: dict[float, dict[DialectCode, list[str]]],
    ) -> str:
        """Create an HTML evaluation page for hyperdialectal texts.

        For each text the evaluator answers:
        - Does this sound exaggerated? (yes/no)
        - Naturalness (1--5 Likert)
        - At what alpha value does it become unnatural? (select)

        Parameters
        ----------
        texts_by_alpha : dict[float, dict[DialectCode, list[str]]]
            ``{alpha: {dialect: [texts...]}}``.

        Returns
        -------
        str
            Complete HTML document.
        """
        rng = random.Random(self.seed)

        # Flatten into items
        items: list[dict[str, Any]] = []
        for alpha in sorted(texts_by_alpha.keys()):
            dialect_texts = texts_by_alpha[alpha]
            for dialect in sorted(dialect_texts.keys(), key=lambda d: d.value):
                for text in dialect_texts[dialect]:
                    items.append({
                        "id": str(uuid.uuid4())[:8],
                        "alpha": alpha,
                        "dialect": dialect.value,
                        "text": text,
                    })

        rng.shuffle(items)

        alpha_options = "\n".join(
            f'          <option value="{a}">&alpha; = {a}</option>'
            for a in sorted(self.alpha_values)
        )

        cards: list[str] = []
        for idx, item in enumerate(items, 1):
            dialect_name = DIALECT_NAMES.get(DialectCode(item["dialect"]), item["dialect"])
            card = f"""\
    <div class="alpha-group" data-item-id="{item['id']}">
      <h2>Muestra {idx} &mdash; {_escape_html(dialect_name)}</h2>
      <span class="alpha-badge">&alpha; = {item['alpha']}</span>
      <div class="sample-text">{_escape_html(item['text'])}</div>

      <div class="question">
        <label>&iquest;Suena exagerado?</label><br>
        <label><input type="radio" name="exaggerated_{item['id']}" value="yes" required> S&iacute;</label>
        <label><input type="radio" name="exaggerated_{item['id']}" value="no" required> No</label>
      </div>

      <div class="question">
        <label>Naturalidad (1 = muy artificial, 5 = completamente natural)</label>
        <div class="likert">
{_likert_radios(f"naturalness_{item['id']}", 5)}
        </div>
      </div>

      <div class="question">
        <label>&iquest;A partir de qu&eacute; &alpha; deja de ser natural?</label>
        <select name="threshold_{item['id']}">
          <option value="">-- No aplica --</option>
{alpha_options}
        </select>
      </div>
    </div>"""
            cards.append(card)

        metadata = json.dumps(items)

        script = """\
<script>
document.getElementById('hyper-form').addEventListener('submit', function(e) {
  e.preventDefault();
  var data = {};
  var fd = new FormData(this);
  for (var pair of fd.entries()) { data[pair[0]] = pair[1]; }
  data['_metadata'] = JSON.parse(document.getElementById('hyper-metadata').textContent);
  var blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'hyperdialectal_responses.json';
  a.click();
});
</script>"""

        html = f"""\
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Evaluaci&oacute;n Hiperdialectal</title>
{_CSS}
</head>
<body>
  <h1>Evaluaci&oacute;n Hiperdialectal</h1>
  <div class="instructions">
    <p>Evaluar textos generados a distintas intensidades dialectales (&alpha;).
    Para cada muestra, indique si suena exagerado, su naturalidad, y
    a partir de qu&eacute; nivel de &alpha; deja de sonar natural.</p>
  </div>
  <form id="hyper-form">
{"".join(cards)}
    <button type="submit">Enviar respuestas</button>
  </form>
  <script id="hyper-metadata" type="application/json">{metadata}</script>
{script}
</body>
</html>"""
        return html

    # ---------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------

    def analyze(self, responses: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyse hyperdialectal evaluation responses.

        Parameters
        ----------
        responses : list[dict]
            Each dict is one evaluator's response set.

        Returns
        -------
        dict
            ``mean_naturalness_by_alpha``, ``exaggeration_rate_by_alpha``,
            ``threshold_distribution``, ``per_dialect`` breakdown.
        """
        naturalness_by_alpha: dict[float, list[float]] = {}
        exaggerated_by_alpha: dict[float, list[bool]] = {}
        thresholds: list[float] = []
        per_dialect: dict[str, dict[float, list[float]]] = {}

        for resp in responses:
            metadata = resp.get("_metadata", [])
            for item in metadata:
                item_id = item["id"]
                alpha = item["alpha"]
                dialect = item["dialect"]

                nat_raw = resp.get(f"naturalness_{item_id}")
                exag_raw = resp.get(f"exaggerated_{item_id}")
                thr_raw = resp.get(f"threshold_{item_id}")

                if nat_raw is not None:
                    nat = float(nat_raw)
                    naturalness_by_alpha.setdefault(alpha, []).append(nat)
                    per_dialect.setdefault(dialect, {}).setdefault(alpha, []).append(nat)

                if exag_raw is not None:
                    exaggerated_by_alpha.setdefault(alpha, []).append(exag_raw == "yes")

                if thr_raw is not None and thr_raw != "":
                    thresholds.append(float(thr_raw))

        def _mean(vals: list) -> float:
            if not vals:
                return 0.0
            return sum(float(v) for v in vals) / len(vals)

        mean_nat: dict[float, float] = {
            a: _mean(vals) for a, vals in sorted(naturalness_by_alpha.items())
        }
        exag_rate: dict[float, float] = {
            a: _mean(vals) for a, vals in sorted(exaggerated_by_alpha.items())
        }

        # Threshold distribution
        threshold_dist: dict[float, int] = {}
        for t in thresholds:
            threshold_dist[t] = threshold_dist.get(t, 0) + 1

        # Per-dialect summary
        per_dialect_summary: dict[str, dict[str, Any]] = {}
        for dialect, alpha_map in per_dialect.items():
            per_dialect_summary[dialect] = {
                "mean_naturalness_by_alpha": {
                    a: _mean(vals) for a, vals in sorted(alpha_map.items())
                },
            }

        return {
            "mean_naturalness_by_alpha": mean_nat,
            "exaggeration_rate_by_alpha": exag_rate,
            "threshold_distribution": threshold_dist,
            "mean_threshold": _mean(thresholds) if thresholds else None,
            "per_dialect": per_dialect_summary,
            "n_evaluators": len(responses),
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
