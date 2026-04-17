"""Hold-out evaluation for dialect generation and classification."""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np

from eigendialectos.constants import DialectCode, DIALECT_NAMES
from eigendialectos.types import CorpusSlice
from eigendialectos.validation.metrics import (
    compute_bleu,
    compute_chrf,
    compute_classification_accuracy,
    compute_confusion_matrix,
)


class HoldoutEvaluator:
    """Evaluate generated text or classifiers against held-out test data.

    Parameters
    ----------
    test_data : dict[DialectCode, CorpusSlice]
        Held-out test slices keyed by dialect.
    """

    def __init__(self, test_data: dict[DialectCode, CorpusSlice]) -> None:
        self.test_data = test_data
        self._results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Generation evaluation
    # ------------------------------------------------------------------

    def evaluate_generation(
        self,
        generated: dict[DialectCode, list[str]],
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare generated texts against reference test samples.

        For each dialect the generated texts are paired with reference
        texts (order-aligned, truncated to the shorter list) and the
        requested metrics are averaged.

        Parameters
        ----------
        generated : dict[DialectCode, list[str]]
            Generated texts keyed by dialect.
        metrics : list[str] or None
            Metric names to compute (default ``['bleu', 'chrf']``).

        Returns
        -------
        dict
            ``{dialect: {metric: avg_score}}``, plus a ``'macro_avg'`` entry.
        """
        if metrics is None:
            metrics = ["bleu", "chrf"]

        metric_fn = {
            "bleu": compute_bleu,
            "chrf": compute_chrf,
        }

        results: dict[str, Any] = {}
        all_scores: dict[str, list[float]] = {m: [] for m in metrics}

        for dialect, corpus_slice in self.test_data.items():
            refs = [s.text for s in corpus_slice.samples]
            hyps = generated.get(dialect, [])

            n_pairs = min(len(refs), len(hyps))
            if n_pairs == 0:
                results[dialect.value] = {m: 0.0 for m in metrics}
                continue

            dialect_scores: dict[str, float] = {}
            for m in metrics:
                fn = metric_fn.get(m)
                if fn is None:
                    continue
                scores = [fn(refs[i], hyps[i]) for i in range(n_pairs)]
                avg = sum(scores) / len(scores) if scores else 0.0
                dialect_scores[m] = avg
                all_scores[m].append(avg)
            results[dialect.value] = dialect_scores

        macro_avg = {}
        for m in metrics:
            vals = all_scores[m]
            macro_avg[m] = sum(vals) / len(vals) if vals else 0.0
        results["macro_avg"] = macro_avg

        self._results["generation"] = results
        return results

    # ------------------------------------------------------------------
    # Classification evaluation
    # ------------------------------------------------------------------

    def evaluate_classification(
        self,
        classifier: Any,
        test_data: dict[DialectCode, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run *classifier* on test data and compute classification metrics.

        The *classifier* must expose a ``predict(features)`` method.  If
        *test_data* is ``None`` the evaluator falls back to dummy feature
        extraction from the stored corpus slices (one-hot bag-of-chars).

        Parameters
        ----------
        classifier
            Object with ``predict(features: np.ndarray) -> list[DialectCode]``.
        test_data : dict[DialectCode, np.ndarray] or None
            Pre-extracted feature matrices keyed by dialect.

        Returns
        -------
        dict
            ``accuracy``, ``per_class_f1``, ``confusion_matrix`` (as list).
        """
        all_features: list[np.ndarray] = []
        all_labels: list[DialectCode] = []

        if test_data is not None:
            for dialect, feats in test_data.items():
                all_features.append(feats)
                all_labels.extend([dialect] * feats.shape[0])
            X = np.vstack(all_features)
        else:
            # Fallback: extract trivial character-frequency features
            X_list: list[np.ndarray] = []
            for dialect, corpus_slice in self.test_data.items():
                for sample in corpus_slice.samples:
                    vec = _char_freq_vector(sample.text)
                    X_list.append(vec)
                    all_labels.append(dialect)
            X = np.vstack(X_list)

        predictions = classifier.predict(X)

        labels = sorted(set(all_labels), key=lambda d: d.value)
        acc = compute_classification_accuracy(predictions, all_labels)
        cm = compute_confusion_matrix(predictions, all_labels, labels)

        # Per-class F1
        per_class_f1: dict[str, float] = {}
        for idx, lbl in enumerate(labels):
            tp = cm[idx, idx]
            fp = cm[:, idx].sum() - tp
            fn = cm[idx, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class_f1[lbl.value] = float(f1)

        result = {
            "accuracy": acc,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm.tolist(),
        }
        self._results["classification"] = result
        return result

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self) -> str:
        """Render a human-readable plaintext report of all accumulated results."""
        lines: list[str] = ["=" * 60, "HoldoutEvaluator Report", "=" * 60]

        gen = self._results.get("generation")
        if gen:
            lines.append("\n--- Generation Metrics ---")
            for key, scores in gen.items():
                if key == "macro_avg":
                    continue
                name = DIALECT_NAMES.get(DialectCode(key), key) if key in [d.value for d in DialectCode] else key
                lines.append(f"  {name}:")
                for m, v in scores.items():
                    lines.append(f"    {m}: {v:.4f}")
            macro = gen.get("macro_avg", {})
            if macro:
                lines.append("  Macro average:")
                for m, v in macro.items():
                    lines.append(f"    {m}: {v:.4f}")

        cls = self._results.get("classification")
        if cls:
            lines.append("\n--- Classification Metrics ---")
            lines.append(f"  Accuracy: {cls['accuracy']:.4f}")
            lines.append("  Per-class F1:")
            for lbl, f1 in cls["per_class_f1"].items():
                lines.append(f"    {lbl}: {f1:.4f}")

        if not self._results:
            lines.append("\n  No evaluations have been run yet.")

        lines.append("=" * 60)
        return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _char_freq_vector(text: str, size: int = 256) -> np.ndarray:
    """Return a character-frequency feature vector of fixed *size*."""
    vec = np.zeros(size, dtype=np.float64)
    for ch in text:
        idx = ord(ch) % size
        vec[idx] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec.reshape(1, -1)
