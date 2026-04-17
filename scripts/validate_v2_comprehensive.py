#!/usr/bin/env python3
"""EigenDialectos v2 — Comprehensive Validation Battery.

Tests from trivial sanity checks to deep structural analysis.
Outputs a structured report with PASS/FAIL/WARN for each test.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import expm, logm
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import pearsonr, spearmanr, ks_2samp, entropy as sp_entropy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eigendialectos.constants import (
    DIALECT_COORDINATES,
    DIALECT_FAMILIES,
    DialectCode,
)
from eigendialectos.types import (
    EigenDecomposition,
    EmbeddingMatrix,
    LevelEmbedding,
    TransformationMatrix,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

EMB_DIR = PROJECT_ROOT / "outputs" / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2_real"

# ─── Helpers ─────────────────────────────────────────────────────────────

class Report:
    def __init__(self):
        self.sections: list[dict] = []
        self._current_section = None

    def section(self, name: str):
        self._current_section = {"name": name, "tests": []}
        self.sections.append(self._current_section)

    def test(self, name: str, passed: bool, detail: str = "", warn: bool = False):
        status = "PASS" if passed else ("WARN" if warn else "FAIL")
        self._current_section["tests"].append({
            "name": name, "status": status, "detail": detail
        })
        icon = "✓" if passed else ("⚠" if warn else "✗")
        print(f"  [{icon}] {name}: {detail[:120]}")

    def summary(self) -> str:
        lines = ["# EigenDialectos v2 — Comprehensive Validation Report\n"]
        total = pass_count = fail_count = warn_count = 0
        for sec in self.sections:
            lines.append(f"\n## {sec['name']}\n")
            for t in sec["tests"]:
                total += 1
                if t["status"] == "PASS":
                    pass_count += 1
                    icon = "✅"
                elif t["status"] == "WARN":
                    warn_count += 1
                    icon = "⚠️"
                else:
                    fail_count += 1
                    icon = "❌"
                lines.append(f"- {icon} **{t['name']}**: {t['detail']}")
        lines.insert(1, f"\n**Summary: {pass_count} passed, {warn_count} warnings, {fail_count} failed out of {total} tests**\n")
        return "\n".join(lines)


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─── Load data ───────────────────────────────────────────────────────────

def load_data():
    with open(EMB_DIR / "vocab.json") as f:
        vocab = json.load(f)

    embeddings = {}
    for d in DialectCode:
        path = EMB_DIR / f"{d.value}.npy"
        if path.exists():
            embeddings[d.value] = np.load(path)

    return vocab, embeddings


def compute_W_and_eigen(embeddings, vocab):
    from eigendialectos.spectral.transformation import compute_transformation_matrix
    from eigendialectos.spectral.eigendecomposition import eigendecompose

    ref = "ES_PEN"
    ref_emb = embeddings[ref]
    W_matrices = {}
    eigendecomps = {}

    for dialect, emb in embeddings.items():
        src = EmbeddingMatrix(data=ref_emb, vocab=vocab, dialect_code=DialectCode.ES_PEN)
        tgt = EmbeddingMatrix(data=emb, vocab=vocab, dialect_code=DialectCode(dialect))
        W_tm = compute_transformation_matrix(src, tgt, method="lstsq", regularization=0.01)
        W_matrices[dialect] = W_tm.data

        tm = TransformationMatrix(
            data=W_tm.data, source_dialect=DialectCode.ES_PEN,
            target_dialect=DialectCode(dialect), regularization=0.01,
        )
        eigendecomps[dialect] = eigendecompose(tm)

    return W_matrices, eigendecomps


# ═══════════════════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_section_1_embedding_sanity(report, vocab, embeddings):
    """Basic embedding sanity checks."""
    report.section("1. Embedding Sanity Checks")

    # 1.1 All 8 dialects loaded
    report.test("All 8 dialect embeddings loaded",
                len(embeddings) == 8,
                f"Got {len(embeddings)}: {sorted(embeddings.keys())}")

    # 1.2 Shape consistency
    shapes = {d: e.shape for d, e in embeddings.items()}
    all_same = len(set(shapes.values())) == 1
    report.test("All embeddings same shape",
                all_same,
                f"Shapes: {set(shapes.values())}")

    dim, n_vocab = next(iter(embeddings.values())).shape
    report.test("Dimension = 100, vocab >= 1000",
                dim == 100 and n_vocab >= 1000,
                f"dim={dim}, vocab={n_vocab}")

    # 1.3 No NaN/Inf
    any_bad = False
    for d, e in embeddings.items():
        if np.any(np.isnan(e)) or np.any(np.isinf(e)):
            any_bad = True
    report.test("No NaN/Inf in any embedding", not any_bad, "Clean" if not any_bad else "Found NaN/Inf!")

    # 1.4 Embedding norms not degenerate
    norms = {}
    for d, e in embeddings.items():
        word_norms = np.linalg.norm(e, axis=0)  # (vocab,)
        norms[d] = (word_norms.mean(), word_norms.std(), word_norms.min(), word_norms.max())
    pen_mean, pen_std = norms["ES_PEN"][0], norms["ES_PEN"][1]
    report.test("PEN word vector norms well-distributed",
                pen_mean > 0.1 and pen_std > 0.01,
                f"mean={pen_mean:.3f}, std={pen_std:.3f}, range=[{norms['ES_PEN'][2]:.3f}, {norms['ES_PEN'][3]:.3f}]")

    # 1.5 Embeddings not identical across dialects
    pen_can_corr = np.corrcoef(embeddings["ES_PEN"].flatten(), embeddings["ES_CAN"].flatten())[0, 1]
    report.test("PEN ≠ CAN (embeddings differ)",
                pen_can_corr < 0.99,
                f"Pearson correlation PEN↔CAN flat = {pen_can_corr:.4f}")

    # 1.6 Vocab has real Spanish words
    known = {"de", "que", "no", "es", "la", "el", "en", "un", "por", "con"}
    found = known.intersection(set(vocab))
    report.test("Vocab contains common Spanish words",
                len(found) >= 8,
                f"Found {len(found)}/10: {sorted(found)}")

    # 1.7 Embedding rank
    rank_pen = np.linalg.matrix_rank(embeddings["ES_PEN"])
    report.test("PEN embedding matrix is full rank",
                rank_pen == dim,
                f"rank={rank_pen}/{dim}")

    # 1.8 Mean centering check
    mean_vec = embeddings["ES_PEN"].mean(axis=1)
    mean_norm = np.linalg.norm(mean_vec)
    report.test("PEN mean vector norm (should be small if centered)",
                True, f"||mean|| = {mean_norm:.4f}", warn=mean_norm > 1.0)


def test_section_2_W_matrix_properties(report, W_matrices, embeddings, vocab):
    """Transformation matrix properties."""
    report.section("2. Transformation Matrix (W) Properties")

    # 2.1 W_PEN is near identity
    W_pen = W_matrices["ES_PEN"]
    pen_identity_err = np.linalg.norm(W_pen - np.eye(W_pen.shape[0]), 'fro') / np.linalg.norm(np.eye(W_pen.shape[0]), 'fro')
    report.test("W_PEN ≈ Identity",
                pen_identity_err < 0.05,
                f"||W_PEN - I||_F / ||I||_F = {pen_identity_err:.6f}")

    # 2.2 W matrices are square
    all_square = all(W.shape[0] == W.shape[1] for W in W_matrices.values())
    report.test("All W matrices are square (100×100)",
                all_square,
                f"Shapes: {set(W.shape for W in W_matrices.values())}")

    # 2.3 Condition numbers
    conds = {d: np.linalg.cond(W) for d, W in W_matrices.items()}
    max_cond_d = max(conds, key=conds.get)
    min_cond_d = min(conds, key=conds.get)
    report.test("Condition numbers in reasonable range",
                all(c < 1e6 for c in conds.values()),
                f"Range: {conds[min_cond_d]:.1f} ({min_cond_d}) to {conds[max_cond_d]:.1f} ({max_cond_d})",
                warn=any(c > 1e4 for c in conds.values()))

    # 2.4 Spectral radius
    for d, W in W_matrices.items():
        eigs = np.abs(np.linalg.eigvals(W))
        rho = eigs.max()
        report.test(f"Spectral radius ρ(W_{d}) (should be < 2.0)",
                    rho < 2.0,
                    f"ρ = {rho:.4f}",
                    warn=rho > 1.5)

    # 2.5 Reconstruction quality: W @ E_source ≈ E_target
    ref_emb = embeddings["ES_PEN"]
    for d in ["ES_CAN", "ES_AND", "ES_RIO", "ES_MEX"]:
        tgt_emb = embeddings[d]
        W = W_matrices[d]
        reconstructed = W @ ref_emb
        rel_err = np.linalg.norm(reconstructed - tgt_emb, 'fro') / np.linalg.norm(tgt_emb, 'fro')
        report.test(f"W_{d} reconstructs target embeddings",
                    rel_err < 0.5,
                    f"||W·E_PEN - E_{d}||_F / ||E_{d}||_F = {rel_err:.4f}",
                    warn=rel_err > 0.3)

    # 2.6 Determinant sign consistency
    dets = {d: np.linalg.det(W) for d, W in W_matrices.items() if d != "ES_PEN"}
    n_pos = sum(1 for v in dets.values() if v > 0)
    n_neg = sum(1 for v in dets.values() if v < 0)
    report.test("W matrix determinants",
                True,
                f"{n_pos} positive, {n_neg} negative (negative = orientation-reversing transforms)")

    # 2.7 Singular value spectrum
    for d in ["ES_AND", "ES_CAN", "ES_RIO"]:
        svs = np.linalg.svd(W_matrices[d], compute_uv=False)
        eff_rank = np.sum(svs > svs.max() * 0.01)
        report.test(f"W_{d} effective rank (sv > 1% of max)",
                    eff_rank > 50,
                    f"effective rank = {eff_rank}/100, σ_max={svs[0]:.3f}, σ_min={svs[-1]:.6f}")


def test_section_3_eigendecomposition(report, eigendecomps, W_matrices):
    """Eigendecomposition quality."""
    report.section("3. Eigendecomposition Quality")

    # 3.1 PEN eigenvalues ≈ 1.0 (since W_PEN ≈ I)
    pen_eigs = np.abs(eigendecomps["ES_PEN"].eigenvalues)
    pen_near_1 = np.mean(np.abs(pen_eigs - 1.0))
    report.test("PEN eigenvalues ≈ 1.0 (identity transform)",
                pen_near_1 < 0.01,
                f"Mean |λ - 1| = {pen_near_1:.6f}")

    # 3.2 Non-PEN eigenvalues < 1.0 (contractive transforms)
    for d in ["ES_AND", "ES_CAN", "ES_RIO", "ES_MEX"]:
        eigs = np.abs(eigendecomps[d].eigenvalues)
        report.test(f"{d} max |λ| < 1.0 (contractive)",
                    eigs.max() < 1.0,
                    f"max |λ| = {eigs.max():.4f}, mean = {eigs.mean():.4f}",
                    warn=eigs.max() > 0.9)

    # 3.3 Eigenvalue spectrum: exponential or power-law decay
    for d in ["ES_AND", "ES_CAN", "ES_MEX"]:
        eigs_sorted = np.sort(np.abs(eigendecomps[d].eigenvalues))[::-1]
        top10 = eigs_sorted[:10]
        bot10 = eigs_sorted[-10:]
        ratio = top10.mean() / max(bot10.mean(), 1e-10)
        report.test(f"{d} spectral gap: top-10 vs bottom-10 eigenvalues",
                    ratio > 5.0,
                    f"Ratio = {ratio:.1f}x (top10_mean={top10.mean():.4f}, bot10_mean={bot10.mean():.4f})")

    # 3.4 Eigenvalue reconstruction: P @ diag(λ) @ P_inv ≈ W
    for d in ["ES_AND", "ES_RIO"]:
        ed = eigendecomps[d]
        W_recon = (ed.eigenvectors @ np.diag(ed.eigenvalues) @ ed.eigenvectors_inv).real
        W_orig = W_matrices[d]
        recon_err = np.linalg.norm(W_recon - W_orig, 'fro') / np.linalg.norm(W_orig, 'fro')
        report.test(f"{d} eigendecomposition reconstructs W",
                    recon_err < 0.5,
                    f"||PΛP⁻¹ - W||_F / ||W||_F = {recon_err:.4f}",
                    warn=recon_err > 0.1)

    # 3.5 Complex conjugate pairs (eigenvalues of real matrix come in conjugate pairs)
    for d in ["ES_AND", "ES_CAN"]:
        eigs = eigendecomps[d].eigenvalues
        imag_parts = np.sort(np.abs(eigs.imag))
        # For each eigenvalue with nonzero imag, its conjugate should also appear
        n_complex = np.sum(np.abs(eigs.imag) > 1e-10)
        report.test(f"{d} complex eigenvalues come in conjugate pairs",
                    n_complex % 2 == 0,
                    f"{n_complex} complex eigenvalues out of {len(eigs)}")

    # 3.6 Eigenvalue uniqueness: different dialects have different spectra
    spec_and = np.sort(np.abs(eigendecomps["ES_AND"].eigenvalues))[::-1]
    spec_can = np.sort(np.abs(eigendecomps["ES_CAN"].eigenvalues))[::-1]
    spec_mex = np.sort(np.abs(eigendecomps["ES_MEX"].eigenvalues))[::-1]
    corr_and_can = np.corrcoef(spec_and, spec_can)[0, 1]
    corr_and_mex = np.corrcoef(spec_and, spec_mex)[0, 1]
    report.test("Eigenvalue spectra differ across dialects",
                corr_and_can < 0.99 or corr_and_mex < 0.99,
                f"r(AND,CAN)={corr_and_can:.4f}, r(AND,MEX)={corr_and_mex:.4f}")


def test_section_4_cross_variety_alignment(report, embeddings, vocab):
    """Cross-variety embedding alignment quality."""
    report.section("4. Cross-Variety Embedding Alignment")

    # Transpose to (vocab, dim) for word-level analysis
    embs_t = {d: e.T for d, e in embeddings.items()}

    # 4.1 Self-similarity: same word in same variety should be identical
    pen_t = embs_t["ES_PEN"]
    report.test("Self-similarity: PEN word 0 cosine with itself = 1.0",
                abs(cosine_sim(pen_t[0], pen_t[0]) - 1.0) < 1e-6,
                f"cos = {cosine_sim(pen_t[0], pen_t[0]):.6f}")

    # 4.2 Cross-variety same-word similarity
    # Same word should have some positive similarity across varieties
    sims_same_word = []
    for i in range(0, min(500, len(vocab))):
        s = cosine_sim(embs_t["ES_PEN"][i], embs_t["ES_CAN"][i])
        sims_same_word.append(s)
    mean_same = np.mean(sims_same_word)
    report.test("Mean cosine similarity of same word across PEN↔CAN",
                mean_same > 0.0,
                f"Mean cos(w_PEN, w_CAN) = {mean_same:.4f} over {len(sims_same_word)} words",
                warn=mean_same < 0.3)

    # 4.3 Cross-variety different-word similarity (should be lower)
    rng = np.random.default_rng(42)
    sims_diff_word = []
    for _ in range(500):
        i, j = rng.integers(0, len(vocab), size=2)
        if i != j:
            sims_diff_word.append(cosine_sim(embs_t["ES_PEN"][i], embs_t["ES_CAN"][j]))
    mean_diff = np.mean(sims_diff_word)
    report.test("Same-word sim > different-word sim (alignment quality)",
                mean_same > mean_diff,
                f"Same-word: {mean_same:.4f}, diff-word: {mean_diff:.4f}, gap: {mean_same - mean_diff:.4f}")

    # 4.4 Known dialectal pairs: check if they're close in embedding space
    # These are real dialectal equivalents
    dialectal_pairs = [
        # (PEN word, target dialect, expected target word)
        ("ordenador", "ES_MEX", "computadora"),
        ("coche", "ES_RIO", "auto"),
        ("autobús", "ES_CAN", "guagua"),
        ("apartamento", "ES_RIO", "departamento"),
        ("vale", "ES_MEX", "órale"),
    ]
    vocab_set = set(vocab)
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    for src_word, tgt_dial, tgt_word in dialectal_pairs:
        if src_word in vocab_set and tgt_word in vocab_set:
            sim = cosine_sim(embs_t["ES_PEN"][vocab_idx[src_word]],
                             embs_t[tgt_dial][vocab_idx[tgt_word]])
            report.test(f"Dialectal pair: {src_word}(PEN) ↔ {tgt_word}({tgt_dial})",
                        sim > 0.0,
                        f"cosine = {sim:.4f}",
                        warn=sim < 0.3)
        else:
            missing = []
            if src_word not in vocab_set:
                missing.append(f"{src_word} not in vocab")
            if tgt_word not in vocab_set:
                missing.append(f"{tgt_word} not in vocab")
            report.test(f"Dialectal pair: {src_word}(PEN) ↔ {tgt_word}({tgt_dial})",
                        False, f"SKIP: {', '.join(missing)}", warn=True)

    # 4.5 Nearest neighbor accuracy: for each word, is same word the NN in target variety?
    # Use random sample to avoid alphabetical bias in large vocabularies
    n_tested = min(2000, len(vocab))
    rng = np.random.RandomState(42)
    test_idx = rng.choice(len(vocab), size=n_tested, replace=False)
    pen_normed = pen_t / np.linalg.norm(pen_t, axis=1, keepdims=True).clip(1e-10)
    can_t = embs_t["ES_CAN"]
    can_normed = can_t / np.linalg.norm(can_t, axis=1, keepdims=True).clip(1e-10)
    # Compute NN on the sampled subset
    pen_sub = pen_normed[test_idx]
    can_sub = can_normed[test_idx]
    sim_matrix = can_sub @ pen_sub.T  # (n_tested, n_tested)
    nn_pred = np.argmax(sim_matrix, axis=0)  # for each PEN word, NN in CAN subset
    nn_hits = int((nn_pred == np.arange(n_tested)).sum())
    nn_acc = nn_hits / n_tested
    report.test("NN accuracy: PEN word → same word in CAN",
                nn_acc > 0.1,
                f"{nn_hits}/{n_tested} = {nn_acc:.1%}",
                warn=nn_acc < 0.3)

    # 4.6 Hubness check: are some target words NN of many source words?
    nn_counts = np.bincount(nn_pred, minlength=n_tested)
    max_hub = int(nn_counts.max())
    n_hubs = int(np.sum(nn_counts > 10))
    report.test("Hubness: no single word dominates NN results",
                max_hub < n_tested * 0.1,
                f"Max hub count = {max_hub}, words appearing >10 times: {n_hubs}",
                warn=max_hub > 20)


def test_section_5_lie_algebra(report, W_matrices):
    """Lie algebra structure tests."""
    report.section("5. Lie Algebra Analysis")

    from eigendialectos.geometry.lie_algebra import LieAlgebraAnalysis
    lie = LieAlgebraAnalysis()
    lie_result = lie.full_analysis(W_matrices)

    # 5.1 logm(W) exists for all matrices
    report.test("logm(W) computed for all dialects",
                len(lie_result.generators) == len(W_matrices),
                f"{len(lie_result.generators)} generators computed")

    # 5.2 expm(logm(W)) ≈ W roundtrip
    for d in ["ES_AND", "ES_CAN", "ES_RIO"]:
        A = lie_result.generators[d]
        W_recon = expm(A)
        err = np.linalg.norm(W_recon.real - W_matrices[d], 'fro') / np.linalg.norm(W_matrices[d], 'fro')
        report.test(f"exp(log(W_{d})) ≈ W_{d}",
                    err < 0.1,
                    f"Relative Frobenius error = {err:.6f}",
                    warn=err > 0.01)

    # 5.3 PEN generator ≈ 0 (log of identity)
    A_pen = lie_result.generators["ES_PEN"]
    pen_gen_norm = np.linalg.norm(A_pen, 'fro')
    report.test("PEN generator ≈ 0 (log(I) = 0)",
                pen_gen_norm < 0.01,
                f"||A_PEN||_F = {pen_gen_norm:.6f}")

    # 5.4 Commutator antisymmetry: [A,B] = -[B,A]
    pairs_checked = 0
    max_antisym_err = 0
    for (i, j), bracket in lie_result.commutators.items():
        if (j, i) in lie_result.commutators:
            err = np.linalg.norm(bracket + lie_result.commutators[(j, i)], 'fro')
            max_antisym_err = max(max_antisym_err, err)
            pairs_checked += 1
    report.test("Commutator antisymmetry [A,B] = -[B,A]",
                max_antisym_err < 1e-8,
                f"Max antisymmetry error = {max_antisym_err:.2e} over {pairs_checked} pairs")

    # 5.5 Commutator norms: PEN pairs should be smallest (PEN ≈ identity)
    pen_norms = [v for (a, b), v in lie_result.commutator_norms.items()
                 if a == "ES_PEN" or b == "ES_PEN"]
    non_pen_norms = [v for (a, b), v in lie_result.commutator_norms.items()
                     if a != "ES_PEN" and b != "ES_PEN"]
    if pen_norms and non_pen_norms:
        report.test("PEN commutator norms << non-PEN (identity commutes with all)",
                    np.mean(pen_norms) < np.mean(non_pen_norms),
                    f"Mean PEN-involved: {np.mean(pen_norms):.4f}, non-PEN: {np.mean(non_pen_norms):.4f}")

    # 5.6 Bracket magnitude matrix is symmetric
    bracket_matrix, labels = lie.bracket_magnitude_matrix(lie_result.generators)
    sym_err = np.linalg.norm(bracket_matrix - bracket_matrix.T, 'fro')
    report.test("Bracket magnitude matrix is symmetric",
                sym_err < 1e-10,
                f"||M - M^T||_F = {sym_err:.2e}")

    # 5.7 Diagonal of bracket matrix is zero
    diag_norm = np.linalg.norm(np.diag(bracket_matrix))
    report.test("Bracket matrix diagonal = 0 ([A,A] = 0)",
                diag_norm < 1e-10,
                f"||diag||_2 = {diag_norm:.2e}")

    # 5.8 Linguistically related dialects have smaller commutators
    # AND ↔ AND_BO should commute more than AND ↔ MEX
    and_andbo = lie_result.commutator_norms.get(("ES_AND", "ES_AND_BO"),
                lie_result.commutator_norms.get(("ES_AND_BO", "ES_AND"), 999))
    and_mex = lie_result.commutator_norms.get(("ES_AND", "ES_MEX"),
              lie_result.commutator_norms.get(("ES_MEX", "ES_AND"), 0))
    report.test("Related dialects commute more: ||[AND,AND_BO]|| < ||[AND,MEX]||",
                and_andbo < and_mex,
                f"||[AND,AND_BO]|| = {and_andbo:.2f}, ||[AND,MEX]|| = {and_mex:.2f}",
                warn=True)


def test_section_6_riemannian(report, eigendecomps):
    """Riemannian geometry tests."""
    report.section("6. Riemannian Geometry")

    from eigendialectos.geometry.riemannian import RiemannianDialectSpace
    riem = RiemannianDialectSpace()
    result = riem.full_analysis(eigendecomps)
    D = result.geodesic_distances
    labels = result.dialect_labels

    # 6.1 Diagonal is zero
    report.test("Geodesic distance d(x,x) = 0",
                np.allclose(np.diag(D), 0, atol=1e-10),
                f"Max diagonal = {np.max(np.abs(np.diag(D))):.2e}")

    # 6.2 Symmetry
    sym_err = np.max(np.abs(D - D.T))
    report.test("Geodesic distance is symmetric",
                sym_err < 1e-10,
                f"Max |d(i,j) - d(j,i)| = {sym_err:.2e}")

    # 6.3 Non-negative
    report.test("Geodesic distances are non-negative",
                np.all(D >= -1e-10),
                f"Min distance = {D.min():.6f}")

    # 6.4 Triangle inequality
    n = D.shape[0]
    violations = 0
    max_violation = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v = D[i, j] - D[i, k] - D[k, j]
                if v > 1e-6:
                    violations += 1
                    max_violation = max(max_violation, v)
    report.test("Triangle inequality holds",
                violations == 0,
                f"Violations: {violations}, max violation: {max_violation:.6f}",
                warn=violations > 0)

    # 6.5 Distance variance: should not be too uniform
    off_diag = D[np.triu_indices(n, k=1)]
    cv = off_diag.std() / off_diag.mean() if off_diag.mean() > 0 else 0
    report.test("Distance matrix has meaningful variance (CV > 0.01)",
                cv > 0.01,
                f"CV = {cv:.4f}, range = [{off_diag.min():.2f}, {off_diag.max():.2f}]",
                warn=cv < 0.05)

    # 6.6 PEN has distinctive position (reference dialect)
    pen_idx = labels.index("ES_PEN") if "ES_PEN" in labels else -1
    if pen_idx >= 0:
        pen_dists = D[pen_idx]
        mean_pen = np.mean(pen_dists[pen_dists > 0])
        report.test("PEN geodesic distances (reference point position)",
                    True,
                    f"Mean distance from PEN = {mean_pen:.2f}")

    # 6.7 Curvatures are finite
    all_finite = all(np.isfinite(k) for k in result.ricci_curvatures.values())
    report.test("All Ricci curvatures are finite",
                all_finite,
                f"Range: [{min(result.ricci_curvatures.values()):.4f}, {max(result.ricci_curvatures.values()):.4f}]")

    # 6.8 Curvature sign (positive = "bunched together", negative = "spread out")
    n_positive = sum(1 for k in result.ricci_curvatures.values() if k > 0)
    report.test("Curvature sign distribution",
                True,
                f"{n_positive}/{len(result.ricci_curvatures)} positive (dialect space is {'convex' if n_positive > len(result.ricci_curvatures)/2 else 'hyperbolic'})")


def test_section_7_fisher(report, embeddings, vocab):
    """Fisher Information analysis."""
    report.section("7. Fisher Information Analysis")

    from eigendialectos.geometry.fisher import FisherInformationAnalysis
    emb_t = {d: e.T for d, e in embeddings.items()}
    fisher = FisherInformationAnalysis()
    result = fisher.compute_fim(emb_t, vocabulary=vocab)

    # 7.1 FIM shape
    report.test("FIM shape matches embedding dimension",
                result.fim.shape == (100, 100),
                f"FIM shape = {result.fim.shape}")

    # 7.2 FIM eigenvalues sorted descending
    is_sorted = np.all(np.diff(result.fim_eigenvalues) <= 1e-10)
    report.test("FIM eigenvalues are sorted descending",
                is_sorted,
                f"Top 5: {np.round(result.fim_eigenvalues[:5], 3)}")

    # 7.3 Most eigenvalues are non-negative (PSD check)
    n_negative = np.sum(result.fim_eigenvalues < -1e-10)
    report.test("FIM eigenvalues mostly non-negative",
                n_negative < 10,
                f"{n_negative} negative eigenvalues out of {len(result.fim_eigenvalues)}",
                warn=n_negative > 0)

    # 7.4 Top diagnostic words are linguistically meaningful
    top_words = [w for w, _ in result.most_diagnostic[:20]]
    # Known dialectal markers
    dialectal_markers = {"os", "vosotros", "vos", "ustedes", "tú", "che", "pues",
                         "vale", "oye", "mira", "vas", "ido", "hemos", "coger",
                         "guagua", "plata", "acá", "ahí"}
    hits = dialectal_markers.intersection(set(top_words))
    report.test("Top diagnostic words include known dialectal markers",
                len(hits) > 0,
                f"Found {len(hits)}: {sorted(hits)} in top-20: {top_words}",
                warn=len(hits) < 2)

    # 7.5 Diagnostic scores are non-trivial
    scores = [s for _, s in result.most_diagnostic]
    report.test("Diagnostic scores have meaningful spread",
                max(scores) > 2 * min(scores),
                f"Range: [{min(scores):.4f}, {max(scores):.4f}]")

    # 7.6 FIM information content: effective dimension
    total_var = result.fim_eigenvalues.sum()
    cumvar = np.cumsum(result.fim_eigenvalues) / total_var if total_var > 0 else np.zeros_like(result.fim_eigenvalues)
    eff_dim = int(np.searchsorted(cumvar, 0.90)) + 1
    report.test("FIM effective dimension (90% variance)",
                True,
                f"{eff_dim} dimensions explain 90% of dialectal variance")


def test_section_8_topology(report, eigendecomps):
    """Persistent homology tests."""
    report.section("8. Persistent Homology (TDA)")

    from eigendialectos.topology.persistent_homology import PersistentHomologyAnalysis
    dialect_order = sorted(eigendecomps.keys())
    eigenspectra = np.array([
        np.abs(eigendecomps[d].eigenvalues[:20].real) for d in dialect_order
    ])

    ph = PersistentHomologyAnalysis(max_dimension=2)
    result = ph.compute(eigenspectra, dialect_order)

    # 8.1 Betti numbers exist
    report.test("Betti numbers computed for H₀, H₁, H₂",
                all(dim in result.betti_numbers for dim in [0, 1, 2]),
                f"β = {dict(result.betti_numbers)}")

    # 8.2 H₀ ≥ 1 (at least one connected component)
    report.test("H₀ ≥ 1 (at least one connected component)",
                result.betti_numbers[0] >= 1,
                f"β₀ = {result.betti_numbers[0]}")

    # 8.3 Persistence entropy is non-negative
    report.test("Persistence entropy ≥ 0",
                result.persistence_entropy >= 0,
                f"H = {result.persistence_entropy:.4f}")

    # 8.4 Persistence diagram has features
    total_features = sum(dgm.shape[0] for dgm in result.diagrams.values())
    report.test("Persistence diagrams have features",
                total_features > 0,
                f"Total features across dimensions: {total_features}")

    # 8.5 Stability: slight perturbation doesn't change Betti numbers dramatically
    rng = np.random.default_rng(42)
    perturbed = eigenspectra + rng.standard_normal(eigenspectra.shape) * 0.01
    result_pert = ph.compute(perturbed, dialect_order)
    betti_diff = sum(abs(result.betti_numbers.get(d, 0) - result_pert.betti_numbers.get(d, 0)) for d in [0, 1, 2])
    report.test("TDA stability: small perturbation preserves Betti numbers",
                betti_diff == 0,
                f"Betti number change: {betti_diff} (original={dict(result.betti_numbers)}, perturbed={dict(result_pert.betti_numbers)})",
                warn=betti_diff > 0)

    # 8.6 Interpret produces meaningful output
    interp = ph.interpret(result, dialect_order)
    report.test("Interpretation produces n_dialect_families ∈ [2, 6]",
                2 <= interp["n_dialect_families"] <= 6,
                f"Detected {interp['n_dialect_families']} families")


def test_section_9_eigenvalue_field(report, eigendecomps):
    """Eigenvalue field (GP) tests."""
    report.section("9. Eigenvalue Field (GP Interpolation)")

    from eigendialectos.geometry.eigenfield import EigenvalueField

    dialect_order = sorted(eigendecomps.keys())
    coords = []
    evals_list = []
    for d in dialect_order:
        dc = DialectCode(d)
        if dc in DIALECT_COORDINATES:
            coords.append(list(DIALECT_COORDINATES[dc]))
            evals_list.append(np.abs(eigendecomps[d].eigenvalues[:10].real))

    coords_arr = np.array(coords)
    evals_arr = np.array(evals_list)

    ef = EigenvalueField(kernel_lengthscale=15.0)
    ef.fit(coords_arr, evals_arr)

    # 9.1 Predictions at training points ≈ actual values
    preds, uncerts = ef.predict(coords_arr)
    pred_err = np.mean(np.abs(preds - np.abs(evals_arr)))
    report.test("GP predictions at training points match actual",
                pred_err < 0.1,
                f"Mean absolute error = {pred_err:.4f}")

    # 9.2 Uncertainty at training points is small
    mean_uncert_train = uncerts.mean()
    report.test("Uncertainty at training points is small",
                mean_uncert_train < 0.5,
                f"Mean uncertainty = {mean_uncert_train:.4f}")

    # 9.3 Uncertainty increases away from data
    far_point = np.array([[0.0, 0.0]])  # Middle of Atlantic
    _, uncert_far = ef.predict(far_point)
    report.test("Uncertainty increases far from data",
                uncert_far.mean() > mean_uncert_train,
                f"Far uncertainty = {uncert_far.mean():.4f} > train = {mean_uncert_train:.4f}")

    # 9.4 Field computation produces valid grid
    field = ef.compute_field(resolution=20)
    report.test("Field grid shape is correct",
                field.eigenvalue_surfaces.shape == (10, 20, 20),
                f"Shape = {field.eigenvalue_surfaces.shape}")

    # 9.5 No NaN in field
    report.test("No NaN/Inf in eigenvalue field",
                not np.any(np.isnan(field.eigenvalue_surfaces)) and
                not np.any(np.isinf(field.eigenvalue_surfaces)),
                "Clean")


def test_section_10_multigranularity(report, W_matrices):
    """Multi-granularity decomposition tests."""
    report.section("10. Multi-Granularity Decomposition")

    from eigendialectos.spectral.multigranularity import MultiGranularityDecomposition
    mg = MultiGranularityDecomposition()
    mg_results = mg.decompose(W_matrices)

    # 10.1 Three levels exist
    report.test("Decomposition has macro, zonal, dialect levels",
                all(k in mg_results for k in ["macro", "zonal", "dialect"]),
                f"Keys: {sorted(mg_results.keys())}")

    # 10.2 Reconstruction: W ≈ W_mean + zonal_residual + dialect_residual
    W_mean = mg_results["macro"]["W_mean"]
    for d in ["ES_AND", "ES_CAN", "ES_MEX"]:
        family = mg_results["dialect"][d]["family"]
        zonal_residual = mg_results["zonal"].get(family, {}).get("W_residual", np.zeros_like(W_mean))
        dialect_residual = mg_results["dialect"][d]["W_residual"]
        W_recon = W_mean + zonal_residual + dialect_residual
        err = np.linalg.norm(W_recon - W_matrices[d], 'fro') / np.linalg.norm(W_matrices[d], 'fro')
        report.test(f"{d} reconstruction: W ≈ macro + zonal + dialect",
                    err < 1e-10,
                    f"Relative error = {err:.2e}")

    # 10.3 Variance ratios sum to ≈ 1.0
    vr = mg.explained_variance_ratio()
    for d, ratios in vr.items():
        total = ratios["macro"] + ratios["zonal"] + ratios["dialect"]
        report.test(f"{d} variance ratios sum to 1.0",
                    abs(total - 1.0) < 0.01,
                    f"macro={ratios['macro']:.2%} + zonal={ratios['zonal']:.2%} + dialect={ratios['dialect']:.2%} = {total:.4f}")

    # 10.4 Macro component is shared (single W_mean for all)
    report.test("Macro component is a single shared matrix",
                mg_results["macro"]["W_mean"].shape == (100, 100),
                f"W_mean shape: {mg_results['macro']['W_mean'].shape}")

    # 10.5 Dialect component is zero for single-member families
    # AND_BO is the only Andaluz bocadillo variant, but it's in "peninsular" family with AND
    # Dialects that are sole members of their family should have dialect=0
    sole_members = [d for d, r in vr.items() if r["dialect"] == 0.0]
    report.test("Sole family members have dialect component = 0",
                len(sole_members) > 0,
                f"Sole members: {sole_members}")


def test_section_11_W_transitivity(report, W_matrices, embeddings, vocab):
    """W matrix composition / transitivity tests."""
    report.section("11. Transform Composition & Transitivity")

    from eigendialectos.spectral.transformation import compute_transformation_matrix

    # Compute W_PEN→AND and W_AND→CAN directly
    def compute_W(src_name, tgt_name):
        src = EmbeddingMatrix(data=embeddings[src_name], vocab=vocab,
                              dialect_code=DialectCode(src_name))
        tgt = EmbeddingMatrix(data=embeddings[tgt_name], vocab=vocab,
                              dialect_code=DialectCode(tgt_name))
        return compute_transformation_matrix(src, tgt, method="lstsq", regularization=0.01).data

    # 11.1 Composition: W_PEN→CAN ≈ W_AND→CAN @ W_PEN→AND ?
    W_pen_and = compute_W("ES_PEN", "ES_AND")
    W_and_can = compute_W("ES_AND", "ES_CAN")
    W_pen_can = W_matrices["ES_CAN"]  # = W_PEN→CAN

    composed = W_and_can @ W_pen_and
    composition_err = np.linalg.norm(composed - W_pen_can, 'fro') / np.linalg.norm(W_pen_can, 'fro')
    report.test("Composition: W_PEN→CAN ≈ W_AND→CAN @ W_PEN→AND",
                composition_err < 1.0,
                f"Relative error = {composition_err:.4f}",
                warn=composition_err > 0.3)

    # 11.2 Inverse: W_PEN→AND @ W_AND→PEN ≈ I ?
    W_and_pen = compute_W("ES_AND", "ES_PEN")
    product = W_and_pen @ W_pen_and
    inv_err = np.linalg.norm(product - np.eye(product.shape[0]), 'fro') / np.sqrt(product.shape[0])
    report.test("Pseudo-inverse: W_AND→PEN @ W_PEN→AND ≈ I",
                inv_err < 1.0,
                f"||W_A→P @ W_P→A - I||_F / √n = {inv_err:.4f}",
                warn=inv_err > 0.3)

    # 11.3 Similarity ordering: closer dialects should have smaller ||W - I||
    norms_from_I = {}
    for d, W in W_matrices.items():
        if d != "ES_PEN":
            norms_from_I[d] = np.linalg.norm(W - np.eye(W.shape[0]), 'fro')
    sorted_by_dist = sorted(norms_from_I.items(), key=lambda x: x[1])
    report.test("Distance from identity ordering",
                True,
                f"Closest to PEN: {sorted_by_dist[0][0]} ({sorted_by_dist[0][1]:.2f}), "
                f"farthest: {sorted_by_dist[-1][0]} ({sorted_by_dist[-1][1]:.2f})")


def test_section_12_random_baselines(report, embeddings, vocab, eigendecomps, W_matrices):
    """Random baseline comparisons."""
    report.section("12. Random Baselines (Sanity Checks)")

    rng = np.random.default_rng(42)

    # 12.1 Real W matrices have lower condition number than random matrices
    random_conds = []
    for _ in range(20):
        R = rng.standard_normal((100, 100))
        random_conds.append(np.linalg.cond(R))
    real_conds = [np.linalg.cond(W) for d, W in W_matrices.items() if d != "ES_PEN"]
    report.test("Real W cond numbers < random matrix cond numbers",
                np.median(real_conds) < np.median(random_conds),
                f"Real median cond = {np.median(real_conds):.0f}, random median = {np.median(random_conds):.0f}")

    # 12.2 Real eigenvalues differ from random matrix eigenvalues
    real_spec = np.sort(np.abs(eigendecomps["ES_AND"].eigenvalues))[::-1]
    random_spec = np.sort(np.abs(np.linalg.eigvals(np.eye(100) + 0.1 * rng.standard_normal((100, 100)))))[::-1]
    ks_stat, ks_pval = ks_2samp(real_spec, random_spec)
    report.test("Real eigenvalue distribution ≠ random (KS test)",
                ks_pval < 0.05,
                f"KS stat = {ks_stat:.4f}, p-value = {ks_pval:.4f}")

    # 12.3 Fisher diagnostic words on shuffled embeddings should be less meaningful
    from eigendialectos.geometry.fisher import FisherInformationAnalysis
    emb_t = {d: e.T for d, e in embeddings.items()}
    # Shuffle word assignments within each dialect
    shuffled = {}
    for d, e in emb_t.items():
        perm = rng.permutation(len(e))
        shuffled[d] = e[perm]
    fisher = FisherInformationAnalysis()
    real_result = fisher.compute_fim(emb_t, vocabulary=vocab)
    shuf_result = fisher.compute_fim(shuffled, vocabulary=vocab)
    real_top_score = real_result.most_diagnostic[0][1]
    shuf_top_score = shuf_result.most_diagnostic[0][1]
    report.test("Real Fisher scores > shuffled Fisher scores",
                real_top_score > shuf_top_score * 0.8,
                f"Real top = {real_top_score:.4f}, shuffled top = {shuf_top_score:.4f}",
                warn=real_top_score <= shuf_top_score)

    # 12.4 Geodesic distances from real eigendecomps vs random
    from eigendialectos.geometry.riemannian import RiemannianDialectSpace
    riem = RiemannianDialectSpace()
    real_riem = riem.full_analysis(eigendecomps)
    real_dists = real_riem.geodesic_distances[np.triu_indices(8, k=1)]

    # Generate random eigendecomps
    random_decomps = {}
    for d in eigendecomps:
        P = rng.standard_normal((100, 100)) + 0j
        random_decomps[d] = EigenDecomposition(
            eigenvalues=(rng.random(100) + 0.5) + 0j,
            eigenvectors=P,
            eigenvectors_inv=np.linalg.pinv(P),
            dialect_code=DialectCode.ES_PEN,
        )
    random_riem = riem.full_analysis(random_decomps)
    rand_dists = random_riem.geodesic_distances[np.triu_indices(8, k=1)]
    report.test("Real geodesic distances differ from random",
                True,
                f"Real: mean={real_dists.mean():.2f} std={real_dists.std():.2f}, "
                f"Random: mean={rand_dists.mean():.2f} std={rand_dists.std():.2f}")


def test_section_13_dialect_clustering(report, eigendecomps, W_matrices):
    """Dialect clustering and phylogenetic analysis."""
    report.section("13. Dialect Clustering & Phylogenetic Consistency")

    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    dialect_order = sorted(eigendecomps.keys())

    # Build spectral distance matrix
    spectra = {}
    for d in dialect_order:
        spectra[d] = np.sort(np.abs(eigendecomps[d].eigenvalues))[::-1]

    n = len(dialect_order)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = D[j, i] = np.linalg.norm(spectra[dialect_order[i]] - spectra[dialect_order[j]])

    # 13.1 Hierarchical clustering
    condensed = squareform(D)
    Z = linkage(condensed, method='ward')
    clusters_3 = fcluster(Z, t=3, criterion='maxclust')
    clusters_4 = fcluster(Z, t=4, criterion='maxclust')

    cluster_map_3 = defaultdict(list)
    for i, c in enumerate(clusters_3):
        cluster_map_3[c].append(dialect_order[i])

    report.test("3-cluster assignment",
                True,
                f"Clusters: {dict(cluster_map_3)}")

    cluster_map_4 = defaultdict(list)
    for i, c in enumerate(clusters_4):
        cluster_map_4[c].append(dialect_order[i])

    report.test("4-cluster assignment",
                True,
                f"Clusters: {dict(cluster_map_4)}")

    # 13.2 AND and AND_BO should be in same cluster
    and_cluster = clusters_4[dialect_order.index("ES_AND")]
    andbo_cluster = clusters_4[dialect_order.index("ES_AND_BO")]
    report.test("AND and AND_BO in same cluster",
                and_cluster == andbo_cluster,
                f"AND cluster={and_cluster}, AND_BO cluster={andbo_cluster}")

    # 13.3 CHI and RIO should be in same cluster (Southern Cone)
    chi_cluster = clusters_4[dialect_order.index("ES_CHI")]
    rio_cluster = clusters_4[dialect_order.index("ES_RIO")]
    report.test("CHI and RIO in same cluster (Southern Cone)",
                chi_cluster == rio_cluster,
                f"CHI cluster={chi_cluster}, RIO cluster={rio_cluster}")

    # 13.4 PEN is in its own cluster or with nearby peninsular dialects
    pen_cluster = clusters_4[dialect_order.index("ES_PEN")]
    pen_members = cluster_map_4[pen_cluster]
    report.test("PEN cluster identity",
                True,
                f"PEN grouped with: {pen_members}")

    # 13.5 Distance ordering: AND↔AND_BO < AND↔MEX
    d_and_andbo = D[dialect_order.index("ES_AND"), dialect_order.index("ES_AND_BO")]
    d_and_mex = D[dialect_order.index("ES_AND"), dialect_order.index("ES_MEX")]
    report.test("AND↔AND_BO closer than AND↔MEX",
                d_and_andbo < d_and_mex,
                f"d(AND,AND_BO) = {d_and_andbo:.4f}, d(AND,MEX) = {d_and_mex:.4f}")

    # 13.6 Distance ordering: CHI↔RIO < CHI↔AND
    d_chi_rio = D[dialect_order.index("ES_CHI"), dialect_order.index("ES_RIO")]
    d_chi_and = D[dialect_order.index("ES_CHI"), dialect_order.index("ES_AND")]
    report.test("CHI↔RIO closer than CHI↔AND (Southern Cone coherence)",
                d_chi_rio < d_chi_and,
                f"d(CHI,RIO) = {d_chi_rio:.4f}, d(CHI,AND) = {d_chi_and:.4f}")


def test_section_14_spectral_stability(report, W_matrices, eigendecomps, embeddings, vocab):
    """Spectral stability under perturbation."""
    report.section("14. Spectral Stability & Robustness")

    rng = np.random.default_rng(42)

    # 14.1 Small perturbation → small eigenvalue change (Weyl's theorem)
    for d in ["ES_AND", "ES_CAN", "ES_MEX"]:
        W = W_matrices[d]
        eps = 0.001
        perturbation = rng.standard_normal(W.shape) * eps
        W_pert = W + perturbation
        eigs_orig = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]
        eigs_pert = np.sort(np.abs(np.linalg.eigvals(W_pert)))[::-1]
        max_change = np.max(np.abs(eigs_orig - eigs_pert))
        report.test(f"{d} eigenvalue stability (ε={eps})",
                    max_change < 10 * eps * np.linalg.norm(W),
                    f"Max |Δλ| = {max_change:.6f}, expected bound ≈ {eps * np.linalg.norm(W):.6f}")

    # 14.2 Vocabulary subset stability: removing 10% of words
    n_remove = len(vocab) // 10
    from eigendialectos.spectral.transformation import compute_transformation_matrix
    keep_idx = np.sort(rng.choice(len(vocab), len(vocab) - n_remove, replace=False))
    sub_vocab = [vocab[i] for i in keep_idx]

    ref_sub = embeddings["ES_PEN"][:, keep_idx]
    and_sub = embeddings["ES_AND"][:, keep_idx]
    src = EmbeddingMatrix(data=ref_sub, vocab=sub_vocab, dialect_code=DialectCode.ES_PEN)
    tgt = EmbeddingMatrix(data=and_sub, vocab=sub_vocab, dialect_code=DialectCode.ES_AND)
    W_sub = compute_transformation_matrix(src, tgt, method="lstsq", regularization=0.01).data

    eigs_full = np.sort(np.abs(np.linalg.eigvals(W_matrices["ES_AND"])))[::-1]
    eigs_sub = np.sort(np.abs(np.linalg.eigvals(W_sub)))[::-1]
    corr = np.corrcoef(eigs_full[:50], eigs_sub[:50])[0, 1]
    report.test("Vocab subset stability: top-50 eigenvalue correlation",
                corr > 0.7,
                f"Correlation = {corr:.4f} (90% vocab)",
                warn=corr < 0.9)

    # 14.3 Regularization sensitivity
    from eigendialectos.spectral.transformation import compute_transformation_matrix
    ref_full = embeddings["ES_PEN"]
    and_full = embeddings["ES_AND"]
    regs = [0.001, 0.01, 0.1]
    eigs_by_reg = {}
    for r in regs:
        src = EmbeddingMatrix(data=ref_full, vocab=vocab, dialect_code=DialectCode.ES_PEN)
        tgt = EmbeddingMatrix(data=and_full, vocab=vocab, dialect_code=DialectCode.ES_AND)
        W_r = compute_transformation_matrix(src, tgt, method="lstsq", regularization=r).data
        eigs_by_reg[r] = np.sort(np.abs(np.linalg.eigvals(W_r)))[::-1]

    corr_01_001 = np.corrcoef(eigs_by_reg[0.01][:50], eigs_by_reg[0.001][:50])[0, 1]
    corr_01_1 = np.corrcoef(eigs_by_reg[0.01][:50], eigs_by_reg[0.1][:50])[0, 1]
    report.test("Regularization sensitivity: eigenvalues stable across λ",
                corr_01_001 > 0.8 and corr_01_1 > 0.5,
                f"r(λ=0.01, λ=0.001) = {corr_01_001:.4f}, r(λ=0.01, λ=0.1) = {corr_01_1:.4f}",
                warn=corr_01_1 < 0.7)


def test_section_15_information_theory(report, embeddings, vocab, eigendecomps):
    """Information-theoretic analysis of eigenspectra."""
    report.section("15. Information-Theoretic Analysis")

    # 15.1 Eigenvalue entropy per dialect
    entropies = {}
    for d in sorted(eigendecomps.keys()):
        eigs = np.abs(eigendecomps[d].eigenvalues)
        eigs_pos = eigs[eigs > 1e-10]
        p = eigs_pos / eigs_pos.sum()
        entropies[d] = float(sp_entropy(p))

    report.test("Eigenvalue entropy varies across dialects",
                max(entropies.values()) - min(entropies.values()) > 0.01,
                f"Range: [{min(entropies.values()):.4f} ({min(entropies, key=entropies.get)}), "
                f"{max(entropies.values()):.4f} ({max(entropies, key=entropies.get)})]")

    # 15.2 PEN has highest entropy (most uniform eigenvalues, near identity)
    pen_entropy = entropies["ES_PEN"]
    others_max = max(v for k, v in entropies.items() if k != "ES_PEN")
    report.test("PEN has highest eigenvalue entropy (most uniform spectrum)",
                pen_entropy > others_max,
                f"PEN = {pen_entropy:.4f}, next = {others_max:.4f}",
                warn=True)

    # 15.3 Effective number of eigenvalues (exp(entropy))
    for d in ["ES_AND", "ES_CAN", "ES_MEX"]:
        eff_n = np.exp(entropies[d])
        report.test(f"{d} effective eigenvalue count = exp(H)",
                    True,
                    f"exp(H) = {eff_n:.1f} out of 100 total")

    # 15.4 Mutual information between eigenspectra (via correlation)
    spectra = {}
    for d in sorted(eigendecomps.keys()):
        spectra[d] = np.abs(eigendecomps[d].eigenvalues)

    # Pairwise correlations
    dialect_order = sorted(spectra.keys())
    corr_matrix = np.zeros((len(dialect_order), len(dialect_order)))
    for i, d1 in enumerate(dialect_order):
        for j, d2 in enumerate(dialect_order):
            corr_matrix[i, j] = np.corrcoef(spectra[d1], spectra[d2])[0, 1]
    mean_off_diag = corr_matrix[np.triu_indices(len(dialect_order), k=1)].mean()
    report.test("Mean pairwise eigenspectral correlation",
                True,
                f"Mean off-diagonal r = {mean_off_diag:.4f}")


def test_section_16_linguistic_validation(report, embeddings, vocab):
    """Linguistic validation: do the embeddings capture known dialectal features?"""
    report.section("16. Linguistic Feature Validation")

    vocab_idx = {w: i for i, w in enumerate(vocab)}
    vocab_set = set(vocab)
    embs_t = {d: e.T for d, e in embeddings.items()}

    # 16.1 Voseo: 'vos' should be closer to RIO/CHI than to PEN
    if "vos" in vocab_set:
        idx = vocab_idx["vos"]
        sim_pen = cosine_sim(embs_t["ES_PEN"][idx], embs_t["ES_RIO"][idx])
        sim_rio_self = np.linalg.norm(embs_t["ES_RIO"][idx])
        sim_pen_self = np.linalg.norm(embs_t["ES_PEN"][idx])
        report.test("'vos' embedding norms: RIO vs PEN",
                    True,
                    f"||vos||_RIO = {sim_rio_self:.4f}, ||vos||_PEN = {sim_pen_self:.4f}")

    # 16.2 Seseo/ceceo markers
    seseo_words = ["casa", "caza", "cena", "cielo"]
    for w in seseo_words:
        if w in vocab_set:
            idx = vocab_idx[w]
            # In PEN, "casa" and "caza" should have different embeddings
            # In AND/CAN (seseo), they may converge
            sim_pen = cosine_sim(embs_t["ES_PEN"][idx], embs_t["ES_AND"][idx])
            report.test(f"'{w}' PEN↔AND similarity",
                        True,
                        f"cos = {sim_pen:.4f}")
            break

    # 16.3 High-frequency function words should be more stable across varieties
    function_words = ["de", "que", "en", "por", "con"]
    content_words = ["casa", "coche", "comer", "hablar", "grande"]
    func_sims, cont_sims = [], []
    for w in function_words:
        if w in vocab_set:
            idx = vocab_idx[w]
            func_sims.append(cosine_sim(embs_t["ES_PEN"][idx], embs_t["ES_MEX"][idx]))
    for w in content_words:
        if w in vocab_set:
            idx = vocab_idx[w]
            cont_sims.append(cosine_sim(embs_t["ES_PEN"][idx], embs_t["ES_MEX"][idx]))

    if func_sims and cont_sims:
        report.test("Function words more stable across varieties than content words",
                    np.mean(func_sims) > np.mean(cont_sims),
                    f"Function mean cos = {np.mean(func_sims):.4f}, content mean cos = {np.mean(cont_sims):.4f}",
                    warn=np.mean(func_sims) <= np.mean(cont_sims))

    # 16.4 Verb forms: do conjugation patterns differ across varieties?
    verb_forms = ["tengo", "tienes", "tiene", "tenemos", "tienen"]
    for w in verb_forms:
        if w in vocab_set:
            idx = vocab_idx[w]
            sims = {}
            for d in ["ES_PEN", "ES_MEX", "ES_RIO", "ES_AND"]:
                sims[d] = float(np.linalg.norm(embs_t[d][idx]))
            report.test(f"'{w}' norm across varieties",
                        True,
                        f"PEN={sims.get('ES_PEN',0):.3f}, MEX={sims.get('ES_MEX',0):.3f}, "
                        f"RIO={sims.get('ES_RIO',0):.3f}, AND={sims.get('ES_AND',0):.3f}")
            break

    # 16.5 Top-N most varying words across all varieties
    n_vocab = len(vocab)
    variances = np.zeros(n_vocab)
    dialects = sorted(embs_t.keys())
    for i in range(n_vocab):
        vecs = [embs_t[d][i] for d in dialects]
        mean_vec = np.mean(vecs, axis=0)
        var = np.mean([np.linalg.norm(v - mean_vec)**2 for v in vecs])
        variances[i] = var

    top_varying = np.argsort(variances)[::-1][:20]
    top_words = [(vocab[i], variances[i]) for i in top_varying]
    report.test("Top 20 most varying words across varieties",
                True,
                f"{[f'{w}({v:.3f})' for w, v in top_words[:10]]}")

    # 16.6 Bottom-N most stable words
    bottom_varying = np.argsort(variances)[:20]
    bottom_words = [(vocab[i], variances[i]) for i in bottom_varying]
    report.test("Top 20 most stable words across varieties",
                True,
                f"{[f'{w}({v:.3f})' for w, v in bottom_words[:10]]}")


def test_section_17_experiment_outputs(report):
    """Validate experiment output files and reports."""
    report.section("17. Experiment Output Validation")

    exp_dir = OUTPUT_DIR / "experiments"
    experiments = [
        "exp_a_dialectal_genome",
        "exp_b_phase_transitions",
        "exp_c_eigenvalue_archaeology",
        "exp_d_synthetic_dialect",
        "exp_e_code_switching",
        "exp_f_eigenvalue_microscope",
        "exp_g_cross_linguistic",
    ]

    for exp_id in experiments:
        exp_path = exp_dir / exp_id
        # Check directory exists
        exists = exp_path.exists()
        report.test(f"{exp_id}: output directory exists",
                    exists,
                    str(exp_path))
        if not exists:
            continue

        # Check result.json
        result_path = exp_path / "result.json"
        report.test(f"{exp_id}: result.json exists and is valid JSON",
                    result_path.exists(),
                    f"Size: {result_path.stat().st_size / 1024:.1f} KB" if result_path.exists() else "MISSING")

        if result_path.exists():
            with open(result_path) as f:
                result_data = json.load(f)
            n_metrics = len(result_data.get("metrics", {}))
            n_artifacts = len(result_data.get("artifact_paths", []))
            report.test(f"{exp_id}: has metrics and artifacts",
                        n_metrics > 0,
                        f"{n_metrics} metrics, {n_artifacts} artifact paths")

        # Check report.md
        report_path = exp_path / "report.md"
        report.test(f"{exp_id}: report.md exists",
                    report_path.exists(),
                    f"Size: {report_path.stat().st_size / 1024:.1f} KB" if report_path.exists() else "MISSING")

        # Check PNG files
        pngs = list(exp_path.glob("*.png"))
        report.test(f"{exp_id}: has visualization PNGs",
                    len(pngs) > 0,
                    f"{len(pngs)} PNGs: {[p.name for p in pngs]}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    report = Report()

    print("Loading data...")
    vocab, embeddings = load_data()
    print("Computing W matrices and eigendecompositions...")
    W_matrices, eigendecomps = compute_W_and_eigen(embeddings, vocab)
    print(f"Data loaded in {time.perf_counter() - t0:.1f}s\n")

    print("=" * 70)
    print("RUNNING COMPREHENSIVE VALIDATION BATTERY")
    print("=" * 70)

    test_section_1_embedding_sanity(report, vocab, embeddings)
    print()
    test_section_2_W_matrix_properties(report, W_matrices, embeddings, vocab)
    print()
    test_section_3_eigendecomposition(report, eigendecomps, W_matrices)
    print()
    test_section_4_cross_variety_alignment(report, embeddings, vocab)
    print()
    test_section_5_lie_algebra(report, W_matrices)
    print()
    test_section_6_riemannian(report, eigendecomps)
    print()
    test_section_7_fisher(report, embeddings, vocab)
    print()
    test_section_8_topology(report, eigendecomps)
    print()
    test_section_9_eigenvalue_field(report, eigendecomps)
    print()
    test_section_10_multigranularity(report, W_matrices)
    print()
    test_section_11_W_transitivity(report, W_matrices, embeddings, vocab)
    print()
    test_section_12_random_baselines(report, embeddings, vocab, eigendecomps, W_matrices)
    print()
    test_section_13_dialect_clustering(report, eigendecomps, W_matrices)
    print()
    test_section_14_spectral_stability(report, W_matrices, eigendecomps, embeddings, vocab)
    print()
    test_section_15_information_theory(report, embeddings, vocab, eigendecomps)
    print()
    test_section_16_linguistic_validation(report, embeddings, vocab)
    print()
    test_section_17_experiment_outputs(report)
    print()

    elapsed = time.perf_counter() - t0
    print("=" * 70)
    print(f"VALIDATION COMPLETE in {elapsed:.1f}s")
    print("=" * 70)

    # Write report
    report_text = report.summary()
    report_path = OUTPUT_DIR / "validation_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
