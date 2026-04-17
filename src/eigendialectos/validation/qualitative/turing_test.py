"""Dialectal Turing test: can evaluators distinguish real from generated text?"""

from __future__ import annotations

import json
import random
import uuid
from typing import Any

from eigendialectos.constants import DialectCode, DIALECT_NAMES


# ======================================================================
# CSS (shared style)
# ======================================================================

_CSS = """\
<style>
  :root { --accent: #2b6cb0; --bg: #f7fafc; --card: #fff; --border: #e2e8f0; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: #1a202c; line-height: 1.6; padding: 2rem; max-width: 900px; margin: 0 auto; }
  h1 { color: var(--accent); margin-bottom: .5rem; }
  .instructions { background: #ebf8ff; border-left: 4px solid var(--accent); padding: 1rem; margin-bottom: 2rem; border-radius: 4px; }
  .pair-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
  .pair-card h2 { color: var(--accent); font-size: 1.05rem; margin-bottom: .75rem; }
  .text-box { background: #edf2f7; padding: .75rem 1rem; border-radius: 4px; font-style: italic; margin-bottom: .75rem; white-space: pre-wrap; }
  .text-label { font-weight: 600; margin-bottom: .25rem; }
  .choice { margin-top: .5rem; }
  .choice label { margin-right: 1.5rem; font-weight: 400; }
  input[type=radio] { accent-color: var(--accent); }
  button { background: var(--accent); color: #fff; border: none; padding: .6rem 1.5rem; border-radius: 4px; cursor: pointer; font-size: 1rem; margin-top: 1rem; }
  button:hover { opacity: .9; }
</style>
"""


# ======================================================================
# DialectalTuringTest
# ======================================================================

class DialectalTuringTest:
    """Build and evaluate a Turing-test style evaluation for dialectal texts.

    For each dialect, *n_pairs_per_dialect* pairs of (real, generated) texts
    are presented in randomised order.  Evaluators must identify which text
    in each pair is machine-generated.

    Parameters
    ----------
    n_pairs_per_dialect : int
        Number of pairs to present per dialect (default 15).
    seed : int
        Random seed for reproducibility (default 42).
    """

    def __init__(self, n_pairs_per_dialect: int = 15, seed: int = 42) -> None:
        self.n_pairs_per_dialect = n_pairs_per_dialect
        self.seed = seed

    # ---------------------------------------------------------
    # Test creation
    # ---------------------------------------------------------

    def create_test(
        self,
        real: dict[DialectCode, list[str]],
        generated: dict[DialectCode, list[str]],
    ) -> str:
        """Create an HTML page with randomised real/generated text pairs.

        Parameters
        ----------
        real : dict[DialectCode, list[str]]
            Real texts per dialect.
        generated : dict[DialectCode, list[str]]
            Generated texts per dialect.

        Returns
        -------
        str
            Complete HTML document.
        """
        rng = random.Random(self.seed)

        # Build pairs
        pairs: list[dict[str, Any]] = []
        for dialect in sorted(set(real) | set(generated), key=lambda d: d.value):
            r_texts = list(real.get(dialect, []))
            g_texts = list(generated.get(dialect, []))
            n_pairs = min(self.n_pairs_per_dialect, len(r_texts), len(g_texts))
            rng.shuffle(r_texts)
            rng.shuffle(g_texts)
            for i in range(n_pairs):
                pair_id = str(uuid.uuid4())[:8]
                # Randomly assign which position is real vs generated
                if rng.random() < 0.5:
                    text_a, text_b = r_texts[i], g_texts[i]
                    generated_position = "B"
                else:
                    text_a, text_b = g_texts[i], r_texts[i]
                    generated_position = "A"
                pairs.append({
                    "id": pair_id,
                    "dialect": dialect.value,
                    "text_a": text_a,
                    "text_b": text_b,
                    "generated_position": generated_position,
                })

        rng.shuffle(pairs)

        # Render HTML
        cards: list[str] = []
        for idx, pair in enumerate(pairs, 1):
            dialect_name = DIALECT_NAMES.get(DialectCode(pair["dialect"]), pair["dialect"])
            card = f"""\
    <div class="pair-card" data-pair-id="{pair['id']}">
      <h2>Par {idx} &mdash; {_escape_html(dialect_name)}</h2>
      <div class="text-label">Texto A:</div>
      <div class="text-box">{_escape_html(pair['text_a'])}</div>
      <div class="text-label">Texto B:</div>
      <div class="text-box">{_escape_html(pair['text_b'])}</div>
      <div class="choice">
        <label>&iquest;Cu&aacute;l es generado por m&aacute;quina?</label><br>
        <label><input type="radio" name="pair_{pair['id']}" value="A" required> A</label>
        <label><input type="radio" name="pair_{pair['id']}" value="B" required> B</label>
      </div>
    </div>"""
            cards.append(card)

        metadata = json.dumps(
            [{"id": p["id"], "dialect": p["dialect"], "generated_position": p["generated_position"]}
             for p in pairs]
        )

        script = """\
<script>
document.getElementById('turing-form').addEventListener('submit', function(e) {
  e.preventDefault();
  var data = {};
  var fd = new FormData(this);
  for (var pair of fd.entries()) { data[pair[0]] = pair[1]; }
  data['_metadata'] = JSON.parse(document.getElementById('turing-metadata').textContent);
  var blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'turing_test_responses.json';
  a.click();
});
</script>"""

        html = f"""\
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Test de Turing Dialectal</title>
{_CSS}
</head>
<body>
  <h1>Test de Turing Dialectal</h1>
  <div class="instructions">
    <p>Para cada par de textos, identifique cu&aacute;l fue generado por una m&aacute;quina.</p>
  </div>
  <form id="turing-form">
{"".join(cards)}
    <button type="submit">Enviar respuestas</button>
  </form>
  <script id="turing-metadata" type="application/json">{metadata}</script>
{script}
</body>
</html>"""
        return html

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    def evaluate(self, responses: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate Turing-test responses.

        Parameters
        ----------
        responses : list[dict]
            Each dict is one evaluator's response set with keys
            ``pair_<id>`` -> ``'A'``/``'B'`` and ``_metadata`` (the pairs list).

        Returns
        -------
        dict
            ``success_rate`` (% correctly identified as generated),
            ``per_dialect`` breakdown, ``n_evaluators``.
        """
        total_correct = 0
        total_pairs = 0
        per_dialect_correct: dict[str, int] = {}
        per_dialect_total: dict[str, int] = {}

        for resp in responses:
            metadata = resp.get("_metadata", [])
            for pair_info in metadata:
                pair_id = pair_info["id"]
                dialect = pair_info["dialect"]
                truth = pair_info["generated_position"]
                answer = resp.get(f"pair_{pair_id}")
                if answer is None:
                    continue

                per_dialect_total[dialect] = per_dialect_total.get(dialect, 0) + 1
                total_pairs += 1
                if answer == truth:
                    total_correct += 1
                    per_dialect_correct[dialect] = per_dialect_correct.get(dialect, 0) + 1

        per_dialect: dict[str, dict[str, Any]] = {}
        for dialect in per_dialect_total:
            c = per_dialect_correct.get(dialect, 0)
            t = per_dialect_total[dialect]
            per_dialect[dialect] = {
                "correct": c,
                "total": t,
                "success_rate": c / t if t > 0 else 0.0,
            }

        return {
            "success_rate": total_correct / total_pairs if total_pairs > 0 else 0.0,
            "total_correct": total_correct,
            "total_pairs": total_pairs,
            "n_evaluators": len(responses),
            "per_dialect": per_dialect,
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
