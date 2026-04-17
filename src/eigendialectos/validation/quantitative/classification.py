"""Dialect classification based on eigenvalue features."""

from __future__ import annotations

from typing import Any

import numpy as np

from eigendialectos.constants import DialectCode
from eigendialectos.types import DialectalSpectrum
from eigendialectos.validation.metrics import (
    compute_classification_accuracy,
    compute_confusion_matrix,
)


# ======================================================================
# Simple logistic-regression classifier (numpy-only, with sklearn fallback)
# ======================================================================

class _SoftmaxClassifierNumpy:
    """Multi-class logistic regression trained with mini-batch gradient descent.

    This is a pure-numpy implementation used when scikit-learn is not available.
    """

    def __init__(
        self,
        lr: float = 0.1,
        max_iter: int = 500,
        batch_size: int = 64,
        reg: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.reg = reg
        self.seed = seed
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape
        n_classes = int(y.max()) + 1

        self.W = rng.standard_normal((n_features, n_classes)) * 0.01
        self.b = np.zeros(n_classes)

        for _ in range(self.max_iter):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                idx = indices[start:end]
                X_batch = X[idx]
                y_batch = y[idx]

                logits = X_batch @ self.W + self.b
                probs = self._softmax(logits)

                # One-hot targets
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(len(y_batch)), y_batch] = 1.0

                grad = probs - one_hot  # (batch, n_classes)
                dW = (X_batch.T @ grad) / len(y_batch) + self.reg * self.W
                db = grad.mean(axis=0)

                self.W -= self.lr * dW
                self.b -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Classifier has not been trained yet.")
        logits = X @ self.W + self.b
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ======================================================================
# DialectClassifier
# ======================================================================

class DialectClassifier:
    """Dialect classifier with pluggable backend.

    Parameters
    ----------
    method : str
        ``'logistic_regression'`` (default).  Uses scikit-learn if available,
        otherwise falls back to a pure-numpy implementation.
    """

    def __init__(self, method: str = "logistic_regression") -> None:
        self.method = method
        self._label_to_idx: dict[DialectCode, int] = {}
        self._idx_to_label: dict[int, DialectCode] = {}
        self._model: Any = None

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def train(self, features: np.ndarray, labels: list[DialectCode]) -> None:
        """Train the classifier.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(n_samples, n_features)``.
        labels : list[DialectCode]
            One label per sample.
        """
        unique_labels = sorted(set(labels), key=lambda d: d.value)
        self._label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        self._idx_to_label = {idx: lbl for lbl, idx in self._label_to_idx.items()}

        y = np.array([self._label_to_idx[lbl] for lbl in labels])

        if self.method == "logistic_regression":
            self._model = self._train_logistic(features, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _train_logistic(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Try sklearn first; fall back to numpy implementation."""
        try:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf.fit(X, y)
            return clf
        except ImportError:
            clf = _SoftmaxClassifierNumpy(max_iter=800)
            clf.fit(X, y)
            return clf

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(self, features: np.ndarray) -> list[DialectCode]:
        """Predict dialect codes for *features*.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(n_samples, n_features)``.

        Returns
        -------
        list[DialectCode]
        """
        if self._model is None:
            raise RuntimeError("Classifier has not been trained yet.")
        raw = self._model.predict(features)
        return [self._idx_to_label[int(idx)] for idx in raw]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities for *features*.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_classes)``.
        """
        if self._model is None:
            raise RuntimeError("Classifier has not been trained yet.")
        return self._model.predict_proba(features)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    def evaluate(
        self,
        test_features: np.ndarray,
        test_labels: list[DialectCode],
    ) -> dict[str, Any]:
        """Evaluate the classifier on held-out test data.

        Parameters
        ----------
        test_features : np.ndarray
        test_labels : list[DialectCode]

        Returns
        -------
        dict
            ``accuracy``, ``per_class_f1``, ``confusion_matrix`` (as list).
        """
        predictions = self.predict(test_features)
        labels = sorted(set(test_labels) | set(predictions), key=lambda d: d.value)

        acc = compute_classification_accuracy(predictions, test_labels)
        cm = compute_confusion_matrix(predictions, test_labels, labels)

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

        return {
            "accuracy": acc,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm.tolist(),
        }


# ======================================================================
# Feature extraction from DialectalSpectrum
# ======================================================================

def extract_eigenvalue_features(
    spectra: dict[DialectCode, DialectalSpectrum],
) -> tuple[np.ndarray, list[DialectCode]]:
    """Convert a mapping of dialectal spectra into a feature matrix.

    Each spectrum is converted into a fixed-size feature vector containing:

    - The sorted eigenvalues (zero-padded/truncated to a common length).
    - The entropy value.
    - Cumulative-energy statistics (25th, 50th, 75th percentiles).

    Parameters
    ----------
    spectra : dict[DialectCode, DialectalSpectrum]

    Returns
    -------
    tuple[np.ndarray, list[DialectCode]]
        Feature matrix of shape ``(n_samples, n_features)`` and corresponding
        dialect labels.
    """
    if not spectra:
        return np.empty((0, 0)), []

    # Determine max eigenvalue vector length
    max_len = max(len(s.eigenvalues_sorted) for s in spectra.values())

    features_list: list[np.ndarray] = []
    labels: list[DialectCode] = []

    for dialect, spectrum in spectra.items():
        eig = np.abs(spectrum.eigenvalues_sorted).astype(np.float64)
        # Pad or truncate to max_len
        padded = np.zeros(max_len, dtype=np.float64)
        n = min(len(eig), max_len)
        padded[:n] = eig[:n]

        # Cumulative energy percentiles
        cum_energy = spectrum.cumulative_energy
        if len(cum_energy) > 0:
            p25 = float(np.interp(0.25, np.linspace(0, 1, len(cum_energy)), cum_energy))
            p50 = float(np.interp(0.50, np.linspace(0, 1, len(cum_energy)), cum_energy))
            p75 = float(np.interp(0.75, np.linspace(0, 1, len(cum_energy)), cum_energy))
        else:
            p25 = p50 = p75 = 0.0

        extra = np.array([spectrum.entropy, p25, p50, p75])
        feat = np.concatenate([padded, extra])
        features_list.append(feat)
        labels.append(dialect)

    return np.vstack(features_list), labels
