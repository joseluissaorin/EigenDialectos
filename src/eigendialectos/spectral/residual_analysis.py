"""Residual analysis of dialect transformation matrices.

The key insight: for dialects of the same language, W ≈ I.  The
dialectal information lives in the RESIDUAL ΔW = W - I.  This module
provides tools to:

1. **SVD of ΔW** — decompose dialectal deviation into orthogonal axes,
   ranked by magnitude.  Compare against the null model noise floor to
   identify significant dialectal dimensions.

2. **Per-word residual PCA** — for each word, compute the shift vector
   r_v(w) = embedding_v(w) - embedding_ref(w) after alignment.  PCA on
   these shift vectors reveals the principal directions of dialectal
   variation at the word level.

3. **Axis interpretation** — project vocabulary onto each dialectal axis
   to find which words are most affected.  This gives linguistic
   interpretability to the eigenstructure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import svdvals

logger = logging.getLogger(__name__)


@dataclass
class DeltaWResult:
    """SVD results for ΔW = W - I for a single variety."""

    variety: str
    singular_values: np.ndarray       # (dim,) — magnitude of each axis
    left_vectors: np.ndarray          # (dim, dim) — U from SVD
    right_vectors: np.ndarray         # (dim, dim) — Vt from SVD
    frob_norm: float                  # ||ΔW||_F — total dialectal shift
    n_significant: int                # axes above null threshold
    top_word_impacts: list[list[tuple[str, float]]]  # per-axis top words


@dataclass
class ResidualAnalysisResult:
    """Complete residual analysis results."""

    # Per-variety ΔW analysis
    delta_w_results: dict[str, DeltaWResult]

    # Null model comparison
    null_median_sv: np.ndarray | None       # (dim,) noise floor
    null_p95_sv: np.ndarray | None          # (dim,) conservative threshold

    # Per-word residual PCA (across all varieties)
    pca_components: np.ndarray              # (n_components, dim)
    pca_explained_variance: np.ndarray      # (n_components,)
    pca_explained_variance_ratio: np.ndarray  # (n_components,)
    pca_word_loadings: list[list[tuple[str, float]]]  # per-component top words

    # Summary statistics
    total_dialectal_dimensions: int         # sum of significant axes
    mean_frob_norm: float                   # mean ||ΔW||_F across varieties


# ======================================================================
# ΔW SVD analysis
# ======================================================================


def analyze_delta_w(
    W_matrices: dict[str, np.ndarray],
    vocab: list[str],
    embeddings: dict[str, np.ndarray],
    reference: str = "ES_PEN",
    null_p95_sv: np.ndarray | None = None,
    top_k_words: int = 15,
) -> dict[str, DeltaWResult]:
    """Compute and analyze ΔW = W - I for each variety.

    Parameters
    ----------
    W_matrices:
        Dict mapping variety name to (dim, dim) W matrix.
    vocab:
        Shared vocabulary list.
    embeddings:
        Dict mapping variety name to (dim, V) embedding matrix.
    reference:
        Reference variety (its ΔW should be ~zero).
    null_p95_sv:
        95th-percentile singular values from null model.
        Axes with SV above this are considered significant.
    top_k_words:
        Number of top-impact words to report per axis.

    Returns
    -------
    Dict mapping variety name to DeltaWResult.
    """
    dim = next(iter(W_matrices.values())).shape[0]
    I = np.eye(dim)
    E_ref = embeddings.get(reference)
    results = {}

    for variety in sorted(W_matrices.keys()):
        W = W_matrices[variety]
        delta_w = W - I

        # Full SVD
        U, s, Vt = np.linalg.svd(delta_w, full_matrices=True)
        frob = np.linalg.norm(delta_w, "fro")

        # Count significant axes
        if null_p95_sv is not None:
            # An axis is significant if its SV exceeds the null threshold
            n_sig = int(np.sum(s > null_p95_sv[:len(s)]))
        else:
            # Without null model, use heuristic: SV > 5% of top SV
            threshold = 0.05 * s[0] if s[0] > 0 else 0
            n_sig = int(np.sum(s > threshold))

        # Per-axis word impact: for axis k, impact(w) = |u_k^T · ΔW · e_ref(w)|
        # Simplification: ΔW = U @ diag(s) @ Vt, so ΔW · e = U @ diag(s) @ Vt @ e
        # impact_k(w) = s_k * |vt_k · e_ref(w)|
        top_word_impacts: list[list[tuple[str, float]]] = []

        n_axes_to_analyze = min(10, n_sig if n_sig > 0 else 5)
        if E_ref is not None:
            for k in range(n_axes_to_analyze):
                # Vt[k] is (dim,) — project all ref embeddings onto it
                # E_ref is (dim, V), so projections = Vt[k] @ E_ref → (V,)
                projections = np.abs(Vt[k] @ E_ref) * s[k]
                top_indices = np.argsort(projections)[-top_k_words:][::-1]
                axis_words = [
                    (vocab[idx], float(projections[idx]))
                    for idx in top_indices
                    if idx < len(vocab)
                ]
                top_word_impacts.append(axis_words)
        else:
            for k in range(n_axes_to_analyze):
                top_word_impacts.append([])

        results[variety] = DeltaWResult(
            variety=variety,
            singular_values=s,
            left_vectors=U,
            right_vectors=Vt,
            frob_norm=frob,
            n_significant=n_sig,
            top_word_impacts=top_word_impacts,
        )

        if variety == reference:
            logger.info(
                "  %s (reference): ||ΔW||_F=%.6f (should be ~0)",
                variety, frob,
            )
        else:
            sig_str = f"{n_sig} significant" if null_p95_sv is not None else f"{n_sig} above heuristic"
            logger.info(
                "  %s: ||ΔW||_F=%.4f, top-3 SV=[%.4f, %.4f, %.4f], %s axes",
                variety, frob, s[0], s[1], s[2], sig_str,
            )
            if top_word_impacts and top_word_impacts[0]:
                top3 = ", ".join(
                    f"{w}({score:.3f})"
                    for w, score in top_word_impacts[0][:3]
                )
                logger.info("    Axis 0 top words: %s", top3)

    return results


# ======================================================================
# Per-word residual PCA
# ======================================================================


def per_word_residual_pca(
    embeddings: dict[str, np.ndarray],
    reference: str = "ES_PEN",
    vocab: list[str] | None = None,
    n_components: int = 10,
    top_k_words: int = 15,
) -> dict:
    """PCA on per-word shift vectors across all non-reference varieties.

    For each word w in each variety v ≠ reference:
        r_v(w) = E_v(:, w) - E_ref(:, w)

    Stack all residuals into matrix R of shape (n_words × n_varieties, dim),
    then PCA to find principal directions of dialectal shift.

    Parameters
    ----------
    embeddings:
        Dict mapping variety name to (dim, V) embedding matrix.
    reference:
        Reference variety.
    vocab:
        Vocabulary list (for word impact reporting).
    n_components:
        Number of PCA components.
    top_k_words:
        Number of top words per component.

    Returns
    -------
    dict with PCA results.
    """
    E_ref = embeddings[reference]  # (dim, V)
    dim, V = E_ref.shape
    varieties = sorted(k for k in embeddings if k != reference)

    logger.info(
        "Per-word residual PCA: %d varieties × %d words = %d residual vectors",
        len(varieties), V, len(varieties) * V,
    )

    # Build residual matrix: (n_varieties * V, dim)
    residuals = []
    variety_labels = []
    for v in varieties:
        E_v = embeddings[v]  # (dim, V)
        R_v = (E_v - E_ref).T  # (V, dim)
        residuals.append(R_v)
        variety_labels.extend([v] * V)

    R = np.vstack(residuals).astype(np.float64)  # (n_var * V, dim)

    # Center
    R_mean = R.mean(axis=0)
    R_centered = R - R_mean

    # SVD (more efficient than forming covariance for tall-skinny matrix)
    # R_centered is (N, dim) with N >> dim, so use economy SVD
    # Covariance = R^T @ R / N  → eigendecompose (dim, dim)
    cov = R_centered.T @ R_centered / R_centered.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order][:n_components]
    eigenvectors = eigenvectors[:, order][:, :n_components]  # (dim, n_components)

    total_var = np.sum(np.linalg.eigvalsh(cov))
    explained_ratio = eigenvalues / total_var if total_var > 0 else eigenvalues

    # Per-component word loadings: which words shift most along each PC?
    word_loadings: list[list[tuple[str, float]]] = []

    if vocab is not None:
        # For each component, compute per-word impact across all varieties
        for k in range(n_components):
            pc = eigenvectors[:, k]  # (dim,)
            # Project each word's residual onto this PC, average across varieties
            word_impacts = np.zeros(V)
            for v_idx, v in enumerate(varieties):
                R_v = residuals[v_idx]  # (V, dim)
                projections = np.abs(R_v @ pc)  # (V,)
                word_impacts += projections
            word_impacts /= len(varieties)

            top_indices = np.argsort(word_impacts)[-top_k_words:][::-1]
            component_words = [
                (vocab[idx], float(word_impacts[idx]))
                for idx in top_indices
                if idx < len(vocab)
            ]
            word_loadings.append(component_words)
    else:
        for _ in range(n_components):
            word_loadings.append([])

    logger.info("  PCA explained variance ratio: %s",
                np.round(explained_ratio[:5], 4))
    logger.info("  Cumulative: %.1f%% in %d components",
                100 * np.sum(explained_ratio), n_components)

    for k in range(min(3, n_components)):
        if word_loadings[k]:
            top3 = ", ".join(
                f"{w}({s:.3f})" for w, s in word_loadings[k][:3]
            )
            logger.info("  PC%d (%.1f%%): %s", k, 100 * explained_ratio[k], top3)

    return {
        "components": eigenvectors.T,           # (n_components, dim)
        "explained_variance": eigenvalues,      # (n_components,)
        "explained_variance_ratio": explained_ratio,  # (n_components,)
        "word_loadings": word_loadings,
        "residual_mean": R_mean,
        "n_varieties": len(varieties),
        "n_words": V,
    }


# ======================================================================
# Per-variety residual analysis (lighter-weight)
# ======================================================================


def per_variety_word_shifts(
    embeddings: dict[str, np.ndarray],
    reference: str = "ES_PEN",
    vocab: list[str] | None = None,
    top_k: int = 20,
) -> dict[str, list[tuple[str, float]]]:
    """Find the most-shifted words per variety (largest ||r_v(w)||).

    Returns dict mapping variety to list of (word, shift_magnitude).
    """
    E_ref = embeddings[reference]  # (dim, V)
    result = {}

    for variety in sorted(embeddings.keys()):
        if variety == reference:
            continue
        E_v = embeddings[variety]
        # Per-word L2 shift
        shifts = np.linalg.norm(E_v - E_ref, axis=0)  # (V,)
        top_indices = np.argsort(shifts)[-top_k:][::-1]

        if vocab is not None:
            words = [
                (vocab[i], float(shifts[i]))
                for i in top_indices if i < len(vocab)
            ]
        else:
            words = [
                (f"word_{i}", float(shifts[i]))
                for i in top_indices
            ]

        result[variety] = words
        logger.info(
            "  %s: top shifted — %s",
            variety,
            ", ".join(f"{w}({s:.2f})" for w, s in words[:5]),
        )

    return result


# ======================================================================
# Full residual analysis pipeline
# ======================================================================


def full_residual_analysis(
    W_matrices: dict[str, np.ndarray],
    embeddings: dict[str, np.ndarray],
    vocab: list[str],
    reference: str = "ES_PEN",
    null_p95_sv: np.ndarray | None = None,
    n_pca_components: int = 10,
    top_k_words: int = 15,
) -> ResidualAnalysisResult:
    """Run the complete residual analysis pipeline.

    1. ΔW = W - I SVD per variety (with null model comparison)
    2. Per-word residual PCA across all varieties
    3. Summary statistics

    Parameters
    ----------
    W_matrices:
        Dict mapping variety name to (dim, dim) W matrix.
    embeddings:
        Dict mapping variety name to (dim, V) embedding matrix.
    vocab:
        Shared vocabulary.
    reference:
        Reference variety.
    null_p95_sv:
        95th-percentile null singular values for significance testing.
    n_pca_components:
        Number of PCA components.
    top_k_words:
        Top words per axis/component.

    Returns
    -------
    ResidualAnalysisResult with all analysis outputs.
    """
    logger.info("=" * 60)
    logger.info("RESIDUAL ANALYSIS: ΔW = W - I")
    logger.info("=" * 60)

    # ΔW SVD
    logger.info("--- ΔW SVD per variety ---")
    delta_results = analyze_delta_w(
        W_matrices=W_matrices,
        vocab=vocab,
        embeddings=embeddings,
        reference=reference,
        null_p95_sv=null_p95_sv,
        top_k_words=top_k_words,
    )

    # Per-word residual PCA
    logger.info("--- Per-word residual PCA ---")
    pca_result = per_word_residual_pca(
        embeddings=embeddings,
        reference=reference,
        vocab=vocab,
        n_components=n_pca_components,
        top_k_words=top_k_words,
    )

    # Per-variety top shifted words
    logger.info("--- Per-variety most-shifted words ---")
    per_variety_word_shifts(
        embeddings=embeddings,
        reference=reference,
        vocab=vocab,
        top_k=10,
    )

    # Summary
    non_ref = {k: v for k, v in delta_results.items() if k != reference}
    total_sig = sum(r.n_significant for r in non_ref.values())
    mean_frob = np.mean([r.frob_norm for r in non_ref.values()])

    logger.info("--- Summary ---")
    logger.info(
        "  Total significant dialectal axes: %d (across %d varieties)",
        total_sig, len(non_ref),
    )
    logger.info("  Mean ||ΔW||_F: %.4f", mean_frob)
    if null_p95_sv is not None:
        logger.info("  Null model p95 top SV: %.4f", null_p95_sv[0])

    null_median = null_p95_sv  # reuse for the result
    null_p95 = null_p95_sv

    return ResidualAnalysisResult(
        delta_w_results=delta_results,
        null_median_sv=null_median,
        null_p95_sv=null_p95,
        pca_components=pca_result["components"],
        pca_explained_variance=pca_result["explained_variance"],
        pca_explained_variance_ratio=pca_result["explained_variance_ratio"],
        pca_word_loadings=pca_result["word_loadings"],
        total_dialectal_dimensions=total_sig,
        mean_frob_norm=mean_frob,
    )
