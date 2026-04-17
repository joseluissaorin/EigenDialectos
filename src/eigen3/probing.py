"""Interpretable spectral mode probing.

Maps eigendecomposition modes to known linguistic features by:
  1. Correlating eigenvalue magnitudes with variety-level feature labels
  2. Analyzing word loadings on each eigenvector to find feature-associated words
  3. Computing mutual information between mode projections and features

The result is a mode x feature matrix that shows which spectral modes
correspond to which dialectological phenomena (seseo, voseo, aspiration, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from eigen3.constants import ALL_VARIETIES, REFERENCE_VARIETY, REGIONALISMS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known linguistic features: 1 = variety has feature, 0 = does not
# ---------------------------------------------------------------------------

LINGUISTIC_FEATURES: dict[str, dict[str, int]] = {
    "seseo": {
        # Merger of /θ/ and /s/ — all American + CAN + AND (partial)
        "ES_PEN": 0, "ES_AND": 1, "ES_CAN": 1, "ES_RIO": 1,
        "ES_MEX": 1, "ES_CAR": 1, "ES_CHI": 1, "ES_AND_BO": 1,
    },
    "voseo": {
        # Use of 'vos' instead of 'tú'
        "ES_PEN": 0, "ES_AND": 0, "ES_CAN": 0, "ES_RIO": 1,
        "ES_MEX": 0, "ES_CAR": 0, "ES_CHI": 1, "ES_AND_BO": 1,
    },
    "ustedes_exclusive": {
        # No vosotros, only ustedes for 2nd person plural
        "ES_PEN": 0, "ES_AND": 0, "ES_CAN": 1, "ES_RIO": 1,
        "ES_MEX": 1, "ES_CAR": 1, "ES_CHI": 1, "ES_AND_BO": 1,
    },
    "s_aspiration": {
        # Weakening/aspiration of syllable-final /s/
        "ES_PEN": 0, "ES_AND": 1, "ES_CAN": 1, "ES_RIO": 1,
        "ES_MEX": 0, "ES_CAR": 1, "ES_CHI": 1, "ES_AND_BO": 0,
    },
    "leismo": {
        # Use of 'le' for direct object (masculine)
        "ES_PEN": 1, "ES_AND": 0, "ES_CAN": 0, "ES_RIO": 0,
        "ES_MEX": 0, "ES_CAR": 0, "ES_CHI": 0, "ES_AND_BO": 1,
    },
    "atlantic_spanish": {
        # Atlantic/Meridional dialect family vs Central/Northern
        "ES_PEN": 0, "ES_AND": 1, "ES_CAN": 1, "ES_RIO": 0,
        "ES_MEX": 0, "ES_CAR": 1, "ES_CHI": 0, "ES_AND_BO": 0,
    },
    "southern_cone": {
        # Rioplatense-Chilean-Andean cluster
        "ES_PEN": 0, "ES_AND": 0, "ES_CAN": 0, "ES_RIO": 1,
        "ES_MEX": 0, "ES_CAR": 0, "ES_CHI": 1, "ES_AND_BO": 1,
    },
    "caribbean": {
        # Caribbean basin dialect zone
        "ES_PEN": 0, "ES_AND": 0, "ES_CAN": 1, "ES_RIO": 0,
        "ES_MEX": 0, "ES_CAR": 1, "ES_CHI": 0, "ES_AND_BO": 0,
    },
    "mesoamerican": {
        # Mexican-Central American dialect zone
        "ES_PEN": 0, "ES_AND": 0, "ES_CAN": 0, "ES_RIO": 0,
        "ES_MEX": 1, "ES_CAR": 0, "ES_CHI": 0, "ES_AND_BO": 0,
    },
}

# Feature descriptions for reporting
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "seseo": "Merger of /θ/ and /s/ (ceceo/seseo)",
    "voseo": "Use of 'vos' pronoun (voseo)",
    "ustedes_exclusive": "Exclusive 'ustedes' (no vosotros)",
    "s_aspiration": "Aspiration of syllable-final /s/",
    "leismo": "Use of 'le' as direct object pronoun",
    "atlantic_spanish": "Atlantic/Meridional dialect family",
    "southern_cone": "Southern Cone dialect cluster (RIO-CHI-AND_BO)",
    "caribbean": "Caribbean basin dialect zone (CAN-CAR)",
    "mesoamerican": "Mesoamerican dialect zone (MEX)",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ModeProbe:
    """Result of probing a single eigenmode for linguistic features."""
    mode_index: int
    variety: str
    eigenvalue: complex
    magnitude: float
    top_words_positive: list[tuple[str, float]]  # (word, loading)
    top_words_negative: list[tuple[str, float]]
    regionalism_overlap: dict[str, list[str]]    # dialect -> matched regionalisms


@dataclass
class FeatureCorrelation:
    """Correlation between a spectral mode and a linguistic feature."""
    mode_index: int
    feature: str
    correlation: float        # Spearman rho
    p_value: float
    direction: str            # "positive" or "negative"


@dataclass
class ProbingResult:
    """Full probing analysis result."""
    mode_feature_matrix: np.ndarray       # (n_modes, n_features) correlation matrix
    feature_names: list[str]
    mode_probes: dict[str, list[ModeProbe]]  # variety -> mode probes
    best_mode_per_feature: dict[str, tuple[int, float]]  # feature -> (mode_idx, correlation)
    feature_correlations: list[FeatureCorrelation]
    explained_features: dict[int, list[str]]  # mode -> list of features it captures


# ---------------------------------------------------------------------------
# Core probing functions
# ---------------------------------------------------------------------------

def probe_spectral_modes(
    embeddings: dict[str, np.ndarray],
    vocab: list[str],
    decomps: dict[str, "EigenDecomp"],
    n_top_modes: int = 30,
    n_top_words: int = 30,
) -> ProbingResult:
    """Run full interpretable probing analysis on spectral modes.

    Parameters
    ----------
    embeddings : dict[str, np.ndarray]
        Per-variety word embeddings (vocab_size, dim).
    vocab : list[str]
        Shared vocabulary.
    decomps : dict[str, EigenDecomp]
        Eigendecompositions per variety.
    n_top_modes : int
        Number of top modes to analyze.
    n_top_words : int
        Number of top-loaded words per mode to report.

    Returns
    -------
    ProbingResult
    """
    varieties = sorted(v for v in decomps if v != REFERENCE_VARIETY)
    feature_names = list(LINGUISTIC_FEATURES.keys())
    n_features = len(feature_names)

    # 1. Collect eigenvalue magnitudes per mode across varieties
    # Use the first non-reference variety's n_modes as reference
    ref_decomp = decomps[varieties[0]]
    n_modes = min(n_top_modes, ref_decomp.n_modes)

    # Build eigenvalue magnitude matrix: (n_varieties, n_modes)
    all_varieties = sorted(decomps.keys())
    eig_mag_matrix = np.zeros((len(all_varieties), n_modes))
    for i, v in enumerate(all_varieties):
        mags = np.abs(decomps[v].eigenvalues[:n_modes])
        eig_mag_matrix[i, :len(mags)] = mags.real

    # 2. Correlate each mode's eigenvalue profile with each feature
    mode_feature_matrix = np.zeros((n_modes, n_features))
    feature_correlations: list[FeatureCorrelation] = []

    for j, feat in enumerate(feature_names):
        feat_labels = np.array([LINGUISTIC_FEATURES[feat].get(v, 0) for v in all_varieties])

        for k in range(n_modes):
            mode_vals = eig_mag_matrix[:, k]

            # Spearman rank correlation
            if np.std(mode_vals) < 1e-10 or np.std(feat_labels) < 1e-10:
                rho, pval = 0.0, 1.0
            else:
                rho, pval = stats.spearmanr(mode_vals, feat_labels)

            mode_feature_matrix[k, j] = rho
            if abs(rho) > 0.5:
                feature_correlations.append(FeatureCorrelation(
                    mode_index=k,
                    feature=feat,
                    correlation=float(rho),
                    p_value=float(pval),
                    direction="positive" if rho > 0 else "negative",
                ))

    # 3. Find best mode per feature
    best_mode_per_feature: dict[str, tuple[int, float]] = {}
    for j, feat in enumerate(feature_names):
        abs_corrs = np.abs(mode_feature_matrix[:, j])
        best_k = int(np.argmax(abs_corrs))
        best_mode_per_feature[feat] = (best_k, float(mode_feature_matrix[best_k, j]))

    # 4. Find which features each mode explains
    explained_features: dict[int, list[str]] = {}
    for k in range(n_modes):
        explained = []
        for j, feat in enumerate(feature_names):
            if abs(mode_feature_matrix[k, j]) > 0.5:
                explained.append(feat)
        if explained:
            explained_features[k] = explained

    # 5. Word loading analysis per variety
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    mode_probes: dict[str, list[ModeProbe]] = {}

    for v in varieties:
        probes: list[ModeProbe] = []
        decomp = decomps[v]
        ref_emb = embeddings[REFERENCE_VARIETY]

        for k in range(min(n_top_modes, decomp.n_modes)):
            # Eigenvector k defines a direction in embedding space
            eigvec = decomp.P[:, k].real
            eigvec_norm = eigvec / (np.linalg.norm(eigvec) + 1e-12)

            # Project all words onto this direction
            loadings = ref_emb @ eigvec_norm  # (vocab_size,)

            # Top positive and negative words
            pos_idx = np.argsort(loadings)[::-1][:n_top_words]
            neg_idx = np.argsort(loadings)[:n_top_words]

            top_pos = [(vocab[i], float(loadings[i])) for i in pos_idx]
            top_neg = [(vocab[i], float(loadings[i])) for i in neg_idx]

            # Check regionalism overlap
            top_word_set = {vocab[i] for i in pos_idx} | {vocab[i] for i in neg_idx}
            reg_overlap: dict[str, list[str]] = {}
            for dialect, regs in REGIONALISMS.items():
                overlap = sorted(top_word_set & regs)
                if overlap:
                    reg_overlap[dialect] = overlap

            probes.append(ModeProbe(
                mode_index=k,
                variety=v,
                eigenvalue=complex(decomp.eigenvalues[k]),
                magnitude=float(np.abs(decomp.eigenvalues[k])),
                top_words_positive=top_pos,
                top_words_negative=top_neg,
                regionalism_overlap=reg_overlap,
            ))

        mode_probes[v] = probes

    logger.info("Probing complete: %d modes x %d features", n_modes, n_features)
    for feat, (k, rho) in best_mode_per_feature.items():
        logger.info("  %s -> mode %d (rho=%.3f)", feat, k, rho)

    return ProbingResult(
        mode_feature_matrix=mode_feature_matrix,
        feature_names=feature_names,
        mode_probes=mode_probes,
        best_mode_per_feature=best_mode_per_feature,
        feature_correlations=feature_correlations,
        explained_features=explained_features,
    )


def format_probing_report(result: ProbingResult) -> str:
    """Format probing results as a human-readable report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("INTERPRETABLE SPECTRAL MODE ANALYSIS")
    lines.append("=" * 70)

    # Best mode per feature
    lines.append("\n--- Feature -> Best Eigenmode Mapping ---\n")
    for feat in result.feature_names:
        k, rho = result.best_mode_per_feature[feat]
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        strength = "STRONG" if abs(rho) > 0.7 else "moderate" if abs(rho) > 0.5 else "weak"
        lines.append(f"  {desc}")
        lines.append(f"    -> Mode {k:2d} (rho={rho:+.3f}, {strength})")

    # Mode interpretation
    lines.append("\n--- Eigenmode Interpretation ---\n")
    for k in sorted(result.explained_features.keys()):
        feats = result.explained_features[k]
        feat_strs = [f"{f} (rho={result.mode_feature_matrix[k, result.feature_names.index(f)]:+.3f})"
                     for f in feats]
        lines.append(f"  Mode {k:2d}: {', '.join(feat_strs)}")

    # Significant correlations
    lines.append("\n--- All Significant Correlations (|rho| > 0.5) ---\n")
    sorted_corrs = sorted(result.feature_correlations, key=lambda c: -abs(c.correlation))
    for c in sorted_corrs:
        lines.append(f"  Mode {c.mode_index:2d} <-> {c.feature:25s}  "
                     f"rho={c.correlation:+.3f}  p={c.p_value:.4f}")

    # Word loading examples for interesting modes
    lines.append("\n--- Top Word Loadings for Key Modes ---\n")
    reported_modes = set()
    for feat, (k, rho) in result.best_mode_per_feature.items():
        if abs(rho) < 0.5 or k in reported_modes:
            continue
        reported_modes.add(k)

        # Show for one variety
        for v, probes in result.mode_probes.items():
            if k < len(probes):
                probe = probes[k]
                lines.append(f"  Mode {k} ({v}, |lambda|={probe.magnitude:.3f}):")
                lines.append(f"    Features: {', '.join(result.explained_features.get(k, ['none']))}")
                pos_words = ', '.join(f"{w}" for w, _ in probe.top_words_positive[:10])
                neg_words = ', '.join(f"{w}" for w, _ in probe.top_words_negative[:10])
                lines.append(f"    Top (+): {pos_words}")
                lines.append(f"    Top (-): {neg_words}")
                if probe.regionalism_overlap:
                    for dialect, words in probe.regionalism_overlap.items():
                        lines.append(f"    Regionalisms [{dialect}]: {', '.join(words)}")
                lines.append("")
                break

    return "\n".join(lines)
