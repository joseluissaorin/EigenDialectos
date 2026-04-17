"""Validation module for EigenDialectos."""

from eigendialectos.validation.metrics import (
    compute_bleu,
    compute_chrf,
    compute_classification_accuracy,
    compute_confusion_matrix,
    compute_dialectal_perplexity_ratio,
    compute_eigenspectrum_divergence,
    compute_frobenius_error,
    compute_krippendorff_alpha,
)

__all__ = [
    "compute_bleu",
    "compute_chrf",
    "compute_classification_accuracy",
    "compute_confusion_matrix",
    "compute_dialectal_perplexity_ratio",
    "compute_eigenspectrum_divergence",
    "compute_frobenius_error",
    "compute_krippendorff_alpha",
]
