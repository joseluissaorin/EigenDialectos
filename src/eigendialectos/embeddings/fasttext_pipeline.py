"""Per-variety fastText training with Procrustes alignment.

Trains independent fastText models per dialect variety using Gensim,
then aligns all embedding spaces to a reference variety (ES_PEN) via
orthogonal Procrustes.  Produces per-variety word embedding matrices
in the same ``(dim, vocab_size)`` format the spectral pipeline expects.

This replaces the subword DCL pipeline with a simpler, faster, and
more effective approach:

- fastText captures variety-specific co-occurrence patterns natively
- Character n-grams handle OOV words and morphological variants
- Procrustes alignment makes spaces comparable while preserving
  variety-specific structure
- Total runtime: ~3-5 minutes on CPU (vs 2.5 hours on MPS for subword DCL)
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

logger = logging.getLogger(__name__)


class _GensimCorpus:
    """Iterable wrapper for gensim's streaming corpus interface."""

    def __init__(self, docs: list[str]) -> None:
        self._docs = docs

    def __iter__(self):
        for doc in self._docs:
            tokens = doc.strip().split()
            if tokens:
                yield tokens


def train_per_variety_fasttext(
    corpus_by_variety: dict[str, list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 3,
    epochs: int = 10,
    min_n: int = 3,
    max_n: int = 6,
    sg: int = 1,
    workers: int = 4,
    seed: int = 42,
) -> dict[str, object]:
    """Train independent fastText models per variety using Gensim.

    Returns dict mapping variety code to trained gensim FastText model.
    """
    from gensim.models import FastText

    models: dict[str, object] = {}

    for variety in sorted(corpus_by_variety.keys()):
        docs = corpus_by_variety[variety]
        logger.info("Training fastText for %s (%d docs) ...", variety, len(docs))
        t0 = time.perf_counter()

        corpus = _GensimCorpus(docs)
        model = FastText(
            sentences=corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            min_n=min_n,
            max_n=max_n,
            sg=sg,
            workers=workers,
            seed=seed,
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "  %s: vocab=%d, vectors=%d (%.1fs)",
            variety, len(model.wv), model.wv.vectors.shape[0], elapsed,
        )
        models[variety] = model

    return models


def _extract_matrix(
    model, vocab: list[str], vector_size: int,
) -> np.ndarray:
    """Extract embedding matrix for a vocabulary from a gensim model.

    For OOV words, gensim's FastText composes vectors from character
    n-grams automatically via ``model.wv.get_vector(word)``.

    Returns (len(vocab), vector_size) float32 matrix.
    """
    matrix = np.zeros((len(vocab), vector_size), dtype=np.float32)
    for i, word in enumerate(vocab):
        try:
            matrix[i] = model.wv.get_vector(word)
        except KeyError:
            # Extremely rare: word has no n-gram overlap at all
            pass
    return matrix


def align_varieties_procrustes(
    models: dict[str, object],
    vocab: list[str],
    reference_variety: str = "ES_PEN",
    vector_size: int = 100,
    anchor_indices: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Align all variety embeddings to a reference via orthogonal Procrustes.

    For each non-reference variety, finds the orthogonal matrix R that
    minimizes ``||X_var @ R - X_ref||_F``.  This preserves inner products
    (angles, distances) within each variety while making the spaces
    comparable across varieties.

    Parameters
    ----------
    anchor_indices:
        If provided, the Procrustes rotation R is computed using ONLY
        these vocabulary indices (typically function words with stable
        meaning across all dialects).  The rotation is then applied to
        the FULL vocabulary.  This prevents dialectally divergent words
        (e.g. "coger", "guagua") from corrupting the alignment.

    Returns dict mapping variety code to aligned (vocab_size, dim) matrices.
    """
    anchor_mode = anchor_indices is not None and len(anchor_indices) > 0
    logger.info(
        "Aligning %d varieties to %s via Procrustes (%s) ...",
        len(models), reference_variety,
        f"{len(anchor_indices)} anchors" if anchor_mode else "full vocab",
    )

    # Extract reference matrix — full vocabulary
    ref_model = models[reference_variety]
    X_ref = _extract_matrix(ref_model, vocab, vector_size)

    # Compute centering from anchors (if provided) or full vocab
    if anchor_mode:
        X_ref_anchor = X_ref[anchor_indices]
        ref_mean = X_ref_anchor.mean(axis=0)
        X_ref_for_procrustes = X_ref_anchor - ref_mean
        anchor_cos_ref = _mean_cosine(X_ref_anchor, X_ref_anchor)
        logger.info(
            "  Reference %s: %d anchor words (mean norm=%.3f)",
            reference_variety, len(anchor_indices),
            np.linalg.norm(X_ref_anchor, axis=1).mean(),
        )
    else:
        ref_mean = X_ref.mean(axis=0)
        X_ref_for_procrustes = X_ref - ref_mean

    aligned: dict[str, np.ndarray] = {}

    for variety in sorted(models.keys()):
        X_var = _extract_matrix(models[variety], vocab, vector_size)

        if variety == reference_variety:
            aligned[variety] = X_var
            logger.info("  %s: reference (no alignment)", variety)
            continue

        # Compute rotation from anchors or full vocab
        if anchor_mode:
            X_var_anchor = X_var[anchor_indices]
            var_mean = X_var_anchor.mean(axis=0)
            X_var_for_procrustes = X_var_anchor - var_mean
        else:
            var_mean = X_var.mean(axis=0)
            X_var_for_procrustes = X_var - var_mean

        # Orthogonal Procrustes: find R minimizing ||X_var @ R - X_ref||
        R, scale = orthogonal_procrustes(X_var_for_procrustes, X_ref_for_procrustes)

        # Apply rotation to ALL words using anchor-derived centering
        # Center → rotate → re-center to reference anchor centroid
        X_aligned = (X_var - var_mean) @ R + ref_mean

        # Alignment quality metrics
        cos_before = _mean_cosine(X_var, X_ref)
        cos_after = _mean_cosine(X_aligned, X_ref)

        if anchor_mode:
            # Also report anchor-specific alignment quality
            X_aligned_anchor = X_aligned[anchor_indices]
            anchor_cos = _mean_cosine(X_aligned_anchor, X_ref[anchor_indices])
            logger.info(
                "  %s: cos_before=%.3f, cos_after=%.3f "
                "(anchor_cos=%.3f, scale=%.3f)",
                variety, cos_before, cos_after, anchor_cos, scale,
            )
        else:
            logger.info(
                "  %s: cos_before=%.3f, cos_after=%.3f (scale=%.3f)",
                variety, cos_before, cos_after, scale,
            )

        aligned[variety] = X_aligned.astype(np.float32)

    return aligned


def _mean_cosine(A: np.ndarray, B: np.ndarray) -> float:
    """Mean row-wise cosine similarity between two matrices."""
    norms_a = np.linalg.norm(A, axis=1, keepdims=True).clip(1e-8)
    norms_b = np.linalg.norm(B, axis=1, keepdims=True).clip(1e-8)
    cos = ((A / norms_a) * (B / norms_b)).sum(axis=1)
    return float(cos.mean())


def save_embeddings(
    aligned: dict[str, np.ndarray],
    vocab: list[str],
    output_dir: Path,
) -> None:
    """Save aligned embeddings in the format the spectral pipeline expects.

    Writes ``{variety}.npy`` with shape ``(dim, vocab_size)`` and
    ``vocab.json`` with the shared vocabulary list.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for variety, matrix in sorted(aligned.items()):
        # matrix is (vocab_size, dim) → transpose to (dim, vocab_size)
        npy_path = output_dir / f"{variety}.npy"
        np.save(str(npy_path), matrix.T.astype(np.float32))
        logger.info(
            "  %s: (%d, %d) saved to %s",
            variety, matrix.T.shape[0], matrix.T.shape[1], npy_path,
        )

    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("  Vocabulary (%d words) saved to %s", len(vocab), vocab_path)
