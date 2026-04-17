"""Eigenmode linguistic analysis: interpret, name, compare, and stress-test eigenmodes.

Provides tools to understand *what* each eigenmode captures linguistically
(top loading words, auto-names), *how* modes relate across dialects (shared
vs. unique axes, pairwise similarity), and *how robust* the decomposition is
to perturbation (stability analysis, sparsity).
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np
from scipy import linalg

from eigen3.types import AnalysisResult, EigenDecomp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenization (shared with scorer)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-záéíóúüñ]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer with basic punctuation stripping."""
    return _WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Cosine similarity (internal)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two real-valued vectors.

    Returns 0.0 when either vector has zero norm.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Eigenvector interpretation
# ---------------------------------------------------------------------------

def interpret_eigenvector(
    P_col: np.ndarray,
    vocab: list[str],
    top_k: int = 20,
    embeddings: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    """Identify the top words by eigenvector loading magnitude.

    Eigenvectors may be complex-valued (from non-symmetric W matrices).

    When the eigenvector lives in the embedding space (dim-sized, not
    vocab-sized), an ``embeddings`` matrix ``(vocab_size, dim)`` must be
    provided so that per-word loadings can be computed as the dot-product
    of each word vector with the eigenvector.

    Parameters
    ----------
    P_col : np.ndarray
        A single column of the eigenvector matrix P, shape ``(dim,)``
        or ``(vocab_size,)``.
    vocab : list[str]
        Vocabulary aligned with the embedding rows.
    top_k : int
        Number of top-loading words to return.
    embeddings : np.ndarray, optional
        (vocab_size, dim) embedding matrix.  Required when
        ``len(P_col) != len(vocab)`` so that per-word loadings can be
        projected.

    Returns
    -------
    list[tuple[str, float]]
        Pairs of ``(word, loading)`` sorted by absolute loading descending.
    """
    col_real = np.real(P_col).astype(np.float64)

    if len(col_real) == len(vocab):
        # P_col is vocab-sized — direct interpretation
        real_loadings = col_real
    elif embeddings is not None:
        # P_col is dim-sized — project through embeddings
        # loading_i = embedding_i · eigenvector
        real_loadings = (embeddings.astype(np.float64) @ col_real)
    else:
        raise ValueError(
            f"P_col length ({len(P_col)}) != vocab length ({len(vocab)}). "
            f"Pass embeddings=(vocab_size, dim) to project dim-sized eigenvectors."
        )

    # Sort by absolute magnitude descending
    abs_loadings = np.abs(real_loadings)
    top_indices = np.argsort(-abs_loadings)[:top_k]

    return [(vocab[i], float(real_loadings[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Mode naming
# ---------------------------------------------------------------------------

def name_mode(
    P_col: np.ndarray,
    vocab: list[str],
    top_k: int = 5,
    embeddings: np.ndarray | None = None,
) -> str:
    """Auto-generate a human-readable name for an eigenmode.

    The name is formed by joining the top-k loading words with hyphens.
    This provides a quick mnemonic for what the mode captures.

    Parameters
    ----------
    P_col : np.ndarray
        Eigenvector column of shape ``(dim,)`` or ``(vocab_size,)``.
    vocab : list[str]
        Shared vocabulary.
    top_k : int
        Number of words to include in the name.
    embeddings : np.ndarray, optional
        (vocab_size, dim) embedding matrix (required for dim-sized P_col).

    Returns
    -------
    str
        Hyphen-joined top loading words, e.g. ``"che-pibe-boludo-laburar-bondi"``.
    """
    top_words = interpret_eigenvector(P_col, vocab, top_k=top_k, embeddings=embeddings)
    return "-".join(word for word, _ in top_words)


def name_all_modes(
    decomp: EigenDecomp,
    vocab: list[str],
    top_k: int = 5,
    embeddings: np.ndarray | None = None,
) -> dict[int, str]:
    """Auto-name every eigenmode in a decomposition.

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition of a dialect's W matrix.
    vocab : list[str]
        Shared vocabulary.
    top_k : int
        Number of words per mode name.
    embeddings : np.ndarray, optional
        (vocab_size, dim) embedding matrix (required for dim-sized P).

    Returns
    -------
    dict[int, str]
        Mapping from mode index to auto-generated name.
    """
    names: dict[int, str] = {}
    for k in range(decomp.n_modes):
        names[k] = name_mode(decomp.P[:, k], vocab, top_k=top_k, embeddings=embeddings)
    return names


# ---------------------------------------------------------------------------
# Cross-dialect mode comparison
# ---------------------------------------------------------------------------

def compare_eigenvectors(
    decomps: dict[str, EigenDecomp],
    mode_idx: int,
) -> np.ndarray:
    """Pairwise cosine similarity matrix between the same mode across dialects.

    For a given mode index ``k``, extracts column ``P[:, k]`` from each
    dialect's eigenvector matrix and computes all-pairs cosine similarity.

    Parameters
    ----------
    decomps : dict[str, EigenDecomp]
        Per-variety eigendecompositions.
    mode_idx : int
        Index of the eigenmode to compare.

    Returns
    -------
    np.ndarray
        Similarity matrix of shape ``(n_dialects, n_dialects)`` with values
        in [-1, 1].  Dialect ordering follows sorted dict keys.
    """
    varieties = sorted(decomps.keys())
    n = len(varieties)
    sim_matrix = np.eye(n, dtype=np.float64)

    # Extract real part of eigenvector columns for the target mode
    columns: list[np.ndarray] = []
    for v in varieties:
        col = np.real(decomps[v].P[:, mode_idx]).astype(np.float64)
        columns.append(col)

    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(columns[i], columns[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix


def find_shared_axes(
    decomps: dict[str, EigenDecomp],
    threshold: float = 0.8,
) -> list[int]:
    """Find eigenmodes that are shared (highly correlated) across all dialects.

    A mode is "shared" if the minimum pairwise cosine similarity of its
    eigenvector across all dialect pairs exceeds the threshold.

    Parameters
    ----------
    decomps : dict[str, EigenDecomp]
        Per-variety eigendecompositions.
    threshold : float
        Minimum cosine similarity to consider a mode shared (default 0.8).

    Returns
    -------
    list[int]
        Sorted list of mode indices that are shared across all dialects.
    """
    if len(decomps) < 2:
        # With fewer than 2 dialects, all modes are trivially shared.
        first = next(iter(decomps.values()))
        return list(range(first.n_modes))

    # Determine the minimum number of modes across all decompositions
    n_modes = min(d.n_modes for d in decomps.values())

    shared: list[int] = []
    for k in range(n_modes):
        sim_matrix = compare_eigenvectors(decomps, k)
        # Minimum off-diagonal similarity
        n = sim_matrix.shape[0]
        min_sim = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] < min_sim:
                    min_sim = sim_matrix[i, j]
        if min_sim >= threshold:
            shared.append(k)

    logger.info(
        "Found %d shared axes (threshold=%.2f) out of %d modes.",
        len(shared), threshold, n_modes,
    )
    return shared


def find_unique_axes(
    decomps: dict[str, EigenDecomp],
    threshold: float = 0.3,
) -> dict[str, list[int]]:
    """Find eigenmodes that are unique to specific dialects.

    A mode is "unique" to a dialect if its maximum cosine similarity with
    the same mode in any other dialect is below the threshold.

    Parameters
    ----------
    decomps : dict[str, EigenDecomp]
        Per-variety eigendecompositions.
    threshold : float
        Maximum cosine similarity to consider a mode unique (default 0.3).

    Returns
    -------
    dict[str, list[int]]
        Mapping from dialect code to list of mode indices unique to that
        dialect.
    """
    varieties = sorted(decomps.keys())
    n_modes = min(d.n_modes for d in decomps.values())

    unique: dict[str, list[int]] = {v: [] for v in varieties}

    for k in range(n_modes):
        sim_matrix = compare_eigenvectors(decomps, k)

        for i, v in enumerate(varieties):
            # Max similarity with any *other* dialect
            max_other = 0.0
            for j in range(len(varieties)):
                if j != i:
                    max_other = max(max_other, abs(sim_matrix[i, j]))
            if max_other < threshold:
                unique[v].append(k)

    for v in varieties:
        if unique[v]:
            logger.info(
                "Dialect %s has %d unique modes (threshold=%.2f).",
                v, len(unique[v]), threshold,
            )

    return unique


# ---------------------------------------------------------------------------
# Mode stability analysis
# ---------------------------------------------------------------------------

def mode_stability(
    decomp: EigenDecomp,
    n_perturbations: int = 10,
    noise_scale: float = 1e-4,
) -> dict[int, float]:
    """Measure eigenmode stability under small perturbations to W.

    For each perturbation trial:
        1. Add Gaussian noise to W_original.
        2. Re-eigendecompose.
        3. Match each original eigenvector to the closest perturbed
           eigenvector (by cosine similarity).
        4. Record the similarity.

    Stability for each mode is the mean matched similarity across trials,
    clamped to [0, 1].

    Parameters
    ----------
    decomp : EigenDecomp
        Original eigendecomposition.
    n_perturbations : int
        Number of noise trials (default 10).
    noise_scale : float
        Standard deviation of Gaussian noise added to W (default 1e-4).

    Returns
    -------
    dict[int, float]
        Mapping from mode index to stability score in [0, 1].
        1.0 = perfectly stable, 0.0 = completely unstable.
    """
    W = decomp.W_original.astype(np.float64)
    n = decomp.n_modes
    original_cols = [np.real(decomp.P[:, k]).astype(np.float64) for k in range(n)]

    # Accumulate similarity per mode across trials
    sim_accum = np.zeros(n, dtype=np.float64)

    rng = np.random.default_rng()

    for trial in range(n_perturbations):
        # Perturbed W
        noise = rng.normal(0.0, noise_scale, size=W.shape)
        W_noisy = W + noise

        # Eigendecompose the perturbed matrix
        eigenvalues_p, P_p = linalg.eig(W_noisy)
        order = np.argsort(-np.abs(eigenvalues_p))
        P_p = P_p[:, order]

        # For each original mode, find best-matching perturbed mode
        for k in range(n):
            orig_col = original_cols[k]
            best_sim = 0.0
            for j in range(min(n, P_p.shape[1])):
                pert_col = np.real(P_p[:, j]).astype(np.float64)
                # Eigenvectors can flip sign; take absolute cosine
                sim = abs(_cosine_similarity(orig_col, pert_col))
                if sim > best_sim:
                    best_sim = sim
            sim_accum[k] += best_sim

    # Average over trials, clamp to [0, 1]
    stability = np.clip(sim_accum / max(n_perturbations, 1), 0.0, 1.0)

    return {k: float(stability[k]) for k in range(n)}


# ---------------------------------------------------------------------------
# Mode sparsity
# ---------------------------------------------------------------------------

def mode_sparsity(P_col: np.ndarray) -> float:
    """Compute the Gini coefficient of eigenvector loadings.

    The Gini coefficient measures inequality among loading magnitudes.
    A coefficient of 0 means perfectly uniform loadings (all words
    contribute equally); 1 means maximally sparse (a single word
    dominates).

    Parameters
    ----------
    P_col : np.ndarray
        Eigenvector column of shape ``(vocab_size,)``.

    Returns
    -------
    float
        Gini coefficient in [0, 1].
    """
    # Take absolute real-part loadings
    values = np.abs(np.real(P_col)).astype(np.float64)
    n = len(values)
    if n == 0:
        return 0.0

    total = values.sum()
    if total < 1e-15:
        return 0.0

    # Sort ascending
    sorted_vals = np.sort(values)

    # Gini formula: G = (2 * sum_i (i+1)*x_i) / (n * sum_i x_i) - (n+1)/n
    index = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * np.dot(index, sorted_vals)) / (n * total) - (n + 1.0) / n

    return float(np.clip(gini, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Full text analysis
# ---------------------------------------------------------------------------

def analyze_text(
    text: str,
    embeddings: dict[str, np.ndarray],
    vocab: list[str],
    decomps: dict[str, EigenDecomp],
) -> AnalysisResult:
    """Full per-mode decomposition of a text.

    For each word in the text that exists in the vocabulary:
        1. Look up its embedding in the reference embedding space.
        2. Project it onto every eigenmode of every available decomposition.
        3. Record per-word mode activation vectors.

    Also computes mode names and aggregate mode strengths across all
    words in the text.

    Parameters
    ----------
    text : str
        Raw input text.
    embeddings : dict[str, np.ndarray]
        Per-variety embedding matrices keyed by dialect code.
    vocab : list[str]
        Shared vocabulary.
    decomps : dict[str, EigenDecomp]
        Per-variety eigendecompositions.

    Returns
    -------
    AnalysisResult
        Contains ``mode_names`` (per-mode auto-name), ``mode_strengths``
        (aggregate activation per mode), and ``per_word_modes`` (per-word
        activation vectors).
    """
    word2idx = {w: i for i, w in enumerate(vocab)}
    tokens = _tokenize(text)

    # Use the first available decomposition as the analysis target
    # (typically the reference variety comes first when sorted)
    if not decomps:
        raise ValueError("No eigendecompositions provided.")

    # Pick a representative decomposition — prefer alphabetically first
    variety = sorted(decomps.keys())[0]
    decomp = decomps[variety]
    n_modes = decomp.n_modes

    # Determine embedding matrix: use the same variety as the decomposition
    if variety not in embeddings:
        # Fallback to first available
        variety = next(iter(embeddings))
    emb_matrix = embeddings[variety]

    # Auto-name modes
    mode_names = name_all_modes(decomp, vocab, top_k=5, embeddings=emb_matrix)

    # Per-word mode activations
    per_word_modes: dict[str, np.ndarray] = {}
    all_activations: list[np.ndarray] = []

    for token in tokens:
        if token in per_word_modes:
            # Already computed for this word — reuse
            all_activations.append(per_word_modes[token])
            continue

        idx = word2idx.get(token)
        if idx is None:
            continue

        word_vec = emb_matrix[idx].astype(np.float64)

        # Project word vector onto eigenmodes: activations = P_inv @ word_vec
        activations = np.abs(decomp.P_inv @ word_vec.astype(np.complex128))
        activations = activations.astype(np.float64)
        per_word_modes[token] = activations
        all_activations.append(activations)

    # Aggregate mode strengths: mean activation across all text words
    if all_activations:
        mode_strengths = np.mean(np.stack(all_activations), axis=0)
    else:
        mode_strengths = np.zeros(n_modes, dtype=np.float64)

    return AnalysisResult(
        mode_names=mode_names,
        mode_strengths=mode_strengths,
        per_word_modes=per_word_modes,
    )


# ---------------------------------------------------------------------------
# Eigenvalue spectrum comparison
# ---------------------------------------------------------------------------

def compare_spectra(
    decomps: dict[str, EigenDecomp],
) -> dict[str, np.ndarray]:
    """Extract eigenvalue magnitude spectra for comparison.

    Parameters
    ----------
    decomps : dict[str, EigenDecomp]
        Per-variety eigendecompositions.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from dialect code to sorted magnitude spectrum (descending).
    """
    return {v: d.magnitudes for v, d in decomps.items()}


def spectral_distance(
    decomp_a: EigenDecomp,
    decomp_b: EigenDecomp,
) -> float:
    """L2 distance between two eigenvalue magnitude spectra.

    Spectra are zero-padded to equal length if they differ in dimension.

    Parameters
    ----------
    decomp_a, decomp_b : EigenDecomp
        Two eigendecompositions to compare.

    Returns
    -------
    float
        Euclidean distance between sorted magnitude vectors.
    """
    mag_a = decomp_a.magnitudes
    mag_b = decomp_b.magnitudes
    max_len = max(len(mag_a), len(mag_b))

    # Zero-pad to equal length
    padded_a = np.zeros(max_len, dtype=np.float64)
    padded_b = np.zeros(max_len, dtype=np.float64)
    padded_a[:len(mag_a)] = mag_a
    padded_b[:len(mag_b)] = mag_b

    return float(np.linalg.norm(padded_a - padded_b))


# ---------------------------------------------------------------------------
# Mode energy / dominance
# ---------------------------------------------------------------------------

def mode_energy(decomp: EigenDecomp) -> np.ndarray:
    """Fractional energy (eigenvalue magnitude) per mode.

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition of a W matrix.

    Returns
    -------
    np.ndarray
        Normalized magnitude vector that sums to 1.  Modes are ordered
        by magnitude descending (matching the sort order in EigenDecomp).
    """
    magnitudes = decomp.magnitudes.astype(np.float64)
    total = magnitudes.sum()
    if total < 1e-15:
        return np.zeros_like(magnitudes)
    return magnitudes / total


def cumulative_energy(decomp: EigenDecomp) -> np.ndarray:
    """Cumulative energy curve.

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition of a W matrix.

    Returns
    -------
    np.ndarray
        Cumulative sum of ``mode_energy``, so the last element is 1.0.
    """
    return np.cumsum(mode_energy(decomp))


def effective_rank(decomp: EigenDecomp) -> int:
    """Number of modes needed to capture 90% of spectral energy.

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition of a W matrix.

    Returns
    -------
    int
        Smallest k such that the top-k modes account for >= 90% of total
        eigenvalue magnitude.
    """
    cum = cumulative_energy(decomp)
    indices = np.where(cum >= 0.90)[0]
    if len(indices) == 0:
        return decomp.n_modes
    return int(indices[0]) + 1


# ---------------------------------------------------------------------------
# Mode clustering across dialects
# ---------------------------------------------------------------------------

def mode_similarity_matrix(
    decomps: dict[str, EigenDecomp],
) -> tuple[np.ndarray, list[str], int]:
    """Full similarity tensor collapsed into a block matrix.

    For each pair of dialects, computes the cosine similarity between
    every pair of eigenvector columns.  The result is a block matrix
    of shape ``(n_dialects * n_modes, n_dialects * n_modes)``.

    Parameters
    ----------
    decomps : dict[str, EigenDecomp]
        Per-variety eigendecompositions.

    Returns
    -------
    tuple of (np.ndarray, list[str], int)
        - Block similarity matrix.
        - Sorted list of variety codes (row/column block order).
        - Number of modes per dialect (minimum across all decompositions).
    """
    varieties = sorted(decomps.keys())
    n_modes = min(d.n_modes for d in decomps.values())
    n_vars = len(varieties)
    total = n_vars * n_modes

    sim = np.zeros((total, total), dtype=np.float64)

    # Precompute real eigenvector columns
    cols: dict[str, list[np.ndarray]] = {}
    for v in varieties:
        cols[v] = [
            np.real(decomps[v].P[:, k]).astype(np.float64)
            for k in range(n_modes)
        ]

    for i, v_i in enumerate(varieties):
        for j, v_j in enumerate(varieties):
            for ki in range(n_modes):
                for kj in range(n_modes):
                    row = i * n_modes + ki
                    col = j * n_modes + kj
                    sim[row, col] = abs(
                        _cosine_similarity(cols[v_i][ki], cols[v_j][kj])
                    )

    return sim, varieties, n_modes


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def summarize_decomposition(
    decomp: EigenDecomp,
    vocab: list[str],
    top_modes: int = 5,
    top_words: int = 10,
) -> str:
    """Generate a human-readable summary of an eigendecomposition.

    Parameters
    ----------
    decomp : EigenDecomp
        Eigendecomposition to summarize.
    vocab : list[str]
        Shared vocabulary.
    top_modes : int
        Number of leading modes to describe.
    top_words : int
        Number of top-loading words per mode.

    Returns
    -------
    str
        Multi-line text summary.
    """
    lines: list[str] = []
    lines.append(f"Eigendecomposition summary: variety={decomp.variety!r}")
    lines.append(f"  Total modes: {decomp.n_modes}")
    lines.append(f"  Effective rank (90% energy): {effective_rank(decomp)}")
    lines.append("")

    for k in range(min(top_modes, decomp.n_modes)):
        mag = float(decomp.magnitudes[k])
        energy_frac = float(mode_energy(decomp)[k])
        sparsity = mode_sparsity(decomp.P[:, k])
        mode_label = name_mode(decomp.P[:, k], vocab, top_k=5)

        lines.append(f"  Mode {k}: |lambda|={mag:.4f}  energy={energy_frac:.2%}  "
                      f"sparsity={sparsity:.3f}")
        lines.append(f"    Name: {mode_label}")

        top = interpret_eigenvector(decomp.P[:, k], vocab, top_k=top_words)
        for word, loading in top:
            lines.append(f"      {word:>20s}  {loading:+.4f}")
        lines.append("")

    return "\n".join(lines)
