#!/usr/bin/env python3
"""Validate v3_full embeddings quality and compare vs v2 and v3 first run."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eigen3.constants import ALL_VARIETIES, REGIONALISMS, REFERENCE_VARIETY
from eigen3.transformation import compute_W
from eigen3.decomposition import eigendecompose
from eigen3.distance import spectral_distance

DIRS = {
    "v2": ROOT / "outputs" / "embeddings",
    "v3_first": ROOT / "outputs" / "eigen3",
    "v3_full": ROOT / "outputs" / "eigen3_full",
}


def load_embeddings(emb_dir: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load embedding matrices and vocab."""
    embs = {}
    for v in ALL_VARIETIES:
        p = emb_dir / f"{v}.npy"
        if p.exists():
            e = np.load(str(p))
            if e.shape[0] < e.shape[1]:
                e = e.T  # (vocab, dim)
            embs[v] = e.astype(np.float64)

    vocab_path = emb_dir / "vocab.json"
    vocab = json.loads(vocab_path.read_text()) if vocab_path.exists() else []
    return embs, vocab


def check_basic_quality(name: str, embs: dict, vocab: list[str]):
    """Basic shape, NaN, rank checks."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if not embs:
        print("  NO EMBEDDINGS FOUND")
        return

    v0 = list(embs.values())[0]
    print(f"  Varieties: {len(embs)}")
    print(f"  Vocab size: {len(vocab)} words")
    print(f"  Shape: {v0.shape}")

    # NaN/Inf check
    nan_count = sum(1 for e in embs.values() if np.any(np.isnan(e)))
    inf_count = sum(1 for e in embs.values() if np.any(np.isinf(e)))
    print(f"  NaN matrices: {nan_count}, Inf matrices: {inf_count}")

    # Rank check
    for v, e in sorted(embs.items()):
        r = np.linalg.matrix_rank(e)
        print(f"    {v}: rank={r}/{min(e.shape)}")


def check_alignment(name: str, embs: dict):
    """Cross-variety alignment via mean cosine of shared words."""
    if len(embs) < 2:
        return

    print(f"\n  Cross-variety alignment (mean cosine, first 1000 words):")
    varieties = sorted(embs.keys())
    ref = embs.get(REFERENCE_VARIETY)
    if ref is None:
        return

    ref_norm = ref[:1000] / (np.linalg.norm(ref[:1000], axis=1, keepdims=True) + 1e-10)
    for v in varieties:
        if v == REFERENCE_VARIETY:
            continue
        e = embs[v]
        e_norm = e[:1000] / (np.linalg.norm(e[:1000], axis=1, keepdims=True) + 1e-10)
        cos = np.mean(np.sum(ref_norm * e_norm, axis=1))
        print(f"    {REFERENCE_VARIETY}-{v}: {cos:.4f}")


def check_regionalism_discrimination(name: str, embs: dict, vocab: list[str]):
    """Check if regionalisms are more discriminative than common words."""
    if not vocab or not embs:
        return

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    varieties = sorted(embs.keys())

    print(f"\n  Regionalism discrimination:")
    for dialect, reg_words in sorted(REGIONALISMS.items()):
        if dialect not in embs:
            continue
        reg_indices = [word_to_idx[w] for w in reg_words if w in word_to_idx]
        if len(reg_indices) < 3:
            print(f"    {dialect}: too few regionalisms ({len(reg_indices)})")
            continue

        # Variance of regionalism embeddings across varieties
        reg_vecs = []
        for v in varieties:
            reg_vecs.append(embs[v][reg_indices])
        stacked = np.stack(reg_vecs, axis=0)  # (n_var, n_reg, dim)
        reg_var = np.mean(np.var(stacked, axis=0))

        # Variance of first 100 common words
        common_indices = list(range(min(100, len(vocab))))
        com_vecs = []
        for v in varieties:
            com_vecs.append(embs[v][common_indices])
        stacked_com = np.stack(com_vecs, axis=0)
        com_var = np.mean(np.var(stacked_com, axis=0))

        ratio = reg_var / (com_var + 1e-10)
        status = "PASS" if ratio > 1.0 else "FAIL"
        print(f"    {dialect}: reg_var={reg_var:.4f}, com_var={com_var:.4f}, ratio={ratio:.2f} [{status}]")


def check_dialect_classification(name: str, embs: dict, vocab: list[str]):
    """Simple nearest-centroid dialect classification using regionalisms."""
    if not vocab or not embs:
        return

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    varieties = sorted(embs.keys())
    correct = 0
    total = 0

    print(f"\n  Dialect classification (regionalism centroid):")
    for true_dialect in varieties:
        if true_dialect not in REGIONALISMS:
            continue

        reg_words = [w for w in REGIONALISMS[true_dialect] if w in word_to_idx]
        if len(reg_words) < 2:
            continue

        reg_indices = [word_to_idx[w] for w in reg_words]

        # Centroid of regionalisms in this variety's embedding
        centroid = np.mean(embs[true_dialect][reg_indices], axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)

        # Which variety's mean embedding is closest?
        best_v, best_sim = None, -1
        for v in varieties:
            v_mean = np.mean(embs[v][reg_indices], axis=0)
            v_mean_norm = v_mean / (np.linalg.norm(v_mean) + 1e-10)
            sim = float(np.dot(centroid_norm, v_mean_norm))
            if sim > best_sim:
                best_sim = sim
                best_v = v

        match = best_v == true_dialect
        if match:
            correct += 1
        total += 1
        status = "PASS" if match else "FAIL"
        print(f"    {true_dialect}: predicted={best_v} (sim={best_sim:.4f}) [{status}]")

    if total > 0:
        print(f"    Overall: {correct}/{total} ({100*correct/total:.0f}%)")


def check_spectral_properties(name: str, embs: dict):
    """Compute W matrices and eigenvalues, check spectral distance ordering."""
    if len(embs) < 2:
        return

    print(f"\n  Spectral analysis:")
    ref_emb = embs.get(REFERENCE_VARIETY)
    if ref_emb is None:
        return

    decomps = {}
    for v, e in embs.items():
        try:
            W_obj = compute_W(ref_emb, e)
            W = W_obj.W.copy()
            decomp = eigendecompose(W, variety=v)
            decomps[v] = decomp
            top5 = decomp.eigenvalues[:5]
            print(f"    {v}: top-5 eigenvalues = [{', '.join(f'{x:.3f}' for x in top5)}]")
        except Exception as ex:
            print(f"    {v}: FAILED ({ex})")

    # Distance matrix
    if len(decomps) >= 2:
        print(f"\n  Spectral distances (closest pairs):")
        pairs = []
        varieties = sorted(decomps.keys())
        for i, v1 in enumerate(varieties):
            for v2 in varieties[i+1:]:
                try:
                    d = spectral_distance(decomps[v1].eigenvalues, decomps[v2].eigenvalues)
                    pairs.append((v1, v2, d))
                except Exception:
                    pass
        pairs.sort(key=lambda x: x[2])
        for v1, v2, d in pairs[:5]:
            print(f"    {v1}-{v2}: {d:.4f}")
        print(f"    ...")
        for v1, v2, d in pairs[-3:]:
            print(f"    {v1}-{v2}: {d:.4f}")


def main():
    for name, emb_dir in DIRS.items():
        if not emb_dir.exists():
            print(f"\n{'='*60}")
            print(f"  {name}: NOT FOUND ({emb_dir})")
            continue

        embs, vocab = load_embeddings(emb_dir)
        check_basic_quality(name, embs, vocab)
        check_alignment(name, embs)
        check_regionalism_discrimination(name, embs, vocab)
        check_dialect_classification(name, embs, vocab)
        check_spectral_properties(name, embs)

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
