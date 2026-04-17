#!/usr/bin/env python3
"""FastAPI backend for the EigenDialectos interactive explorer.

Serves the analysis API and static frontend.
Run: uvicorn explorer.server:app --reload --port 8400
Or:  python explorer/server.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eigen3.constants import (
    ALL_VARIETIES, DIALECT_COORDINATES, REFERENCE_VARIETY, REGIONALISMS,
)
from eigen3.transformation import compute_all_W
from eigen3.decomposition import eigendecompose
from eigen3.distance import spectral_distance
from eigen3.scorer import DialectScorer

logger = logging.getLogger(__name__)

app = FastAPI(title="EigenDialectos Explorer", version="3.0")

# ---------------------------------------------------------------------------
# Global state (loaded at startup)
# ---------------------------------------------------------------------------

EMB_DIR = ROOT / "outputs" / "eigen3_full"
_state: dict = {}


@app.on_event("startup")
async def load_data():
    """Load embeddings, compute decompositions, initialize scorer."""
    logger.info("Loading embeddings from %s", EMB_DIR)

    # Load embeddings
    embs: dict[str, np.ndarray] = {}
    for v in ALL_VARIETIES:
        p = EMB_DIR / f"{v}.npy"
        if not p.exists():
            continue
        e = np.load(str(p))
        if e.shape[0] < e.shape[1]:
            e = e.T
        embs[v] = e.astype(np.float64)

    vocab = json.loads((EMB_DIR / "vocab.json").read_text())

    # W matrices + eigendecompositions
    W_all = compute_all_W(embs)
    decomps = {v: eigendecompose(tm.W, variety=v) for v, tm in W_all.items()}

    # Distance matrix
    varieties = sorted(embs.keys())
    n = len(varieties)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = spectral_distance(decomps[varieties[i]].eigenvalues,
                                  decomps[varieties[j]].eigenvalues)
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    # Scorer
    scorer = DialectScorer(embs, vocab, decomps)

    # Precomputed analysis
    analysis = {}
    analysis_path = EMB_DIR / "analysis_results.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())

    # 2D PCA projection for eigenspace view
    from sklearn.decomposition import PCA
    ref_emb = embs[REFERENCE_VARIETY]
    # Sample words for visualization
    n_viz = min(2000, len(vocab))
    rng = np.random.default_rng(42)
    # Include all regionalisms + random sample
    reg_indices = set()
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    for regs in REGIONALISMS.values():
        for w in regs:
            if w in word_to_idx:
                reg_indices.add(word_to_idx[w])
    remaining = list(set(range(len(vocab))) - reg_indices)
    sample_indices = sorted(reg_indices | set(rng.choice(remaining, min(n_viz - len(reg_indices), len(remaining)), replace=False).tolist()))

    pca = PCA(n_components=2)
    # Stack all variety embeddings for shared PCA
    all_points = []
    all_labels = []
    all_words = []
    for v in varieties:
        sub = embs[v][sample_indices]
        all_points.append(sub)
        all_labels.extend([v] * len(sample_indices))
        all_words.extend([vocab[i] for i in sample_indices])

    stacked = np.vstack(all_points)
    coords_2d = pca.fit_transform(stacked)

    eigenspace_data = []
    for i in range(len(all_labels)):
        eigenspace_data.append({
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "variety": all_labels[i],
            "word": all_words[i],
        })

    _state.update({
        "embs": embs,
        "vocab": vocab,
        "word_to_idx": word_to_idx,
        "decomps": decomps,
        "dist_mat": dist_mat,
        "varieties": varieties,
        "scorer": scorer,
        "analysis": analysis,
        "eigenspace": eigenspace_data,
        "sample_indices": sample_indices,
    })
    logger.info("Explorer ready: %d varieties, %d words", len(embs), len(vocab))


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/score")
async def score_text(text: str = Query(...), temperature: float = Query(1.0)):
    """Score text for dialect probabilities."""
    scorer: DialectScorer = _state["scorer"]
    result = scorer.score(text, temperature=temperature)

    # Per-word scoring
    from eigen3.scorer import _tokenize
    words = _tokenize(text)
    word_scores = []
    vocab_set = set(_state["vocab"])
    for w in words:
        if w in vocab_set:
            single = scorer.score(w, temperature=temperature)
            word_scores.append({
                "word": w,
                "top_dialect": single.top_dialect,
                "probabilities": single.probabilities,
                "in_vocab": True,
            })
        else:
            word_scores.append({
                "word": w,
                "top_dialect": "",
                "probabilities": {},
                "in_vocab": False,
            })

    return {
        "probabilities": result.probabilities,
        "top_dialect": result.top_dialect,
        "mode_activations": result.mode_activations.tolist() if result.mode_activations is not None else [],
        "word_scores": word_scores,
    }


@app.get("/api/distance_matrix")
async def get_distance_matrix():
    """Return the full spectral distance matrix."""
    return {
        "matrix": _state["dist_mat"].tolist(),
        "varieties": _state["varieties"],
        "coordinates": {v: list(c) for v, c in DIALECT_COORDINATES.items()},
    }


@app.get("/api/analysis")
async def get_analysis():
    """Return precomputed probing + validation results."""
    return _state.get("analysis", {})


@app.get("/api/eigenspace")
async def get_eigenspace(variety: str = Query(None)):
    """Return 2D PCA points for eigenspace visualization."""
    data = _state["eigenspace"]
    if variety:
        data = [p for p in data if p["variety"] == variety]
    return {"points": data}


@app.get("/api/mode/{mode_id}")
async def get_mode_detail(mode_id: int, variety: str = Query("ES_AND")):
    """Return word loadings for a specific eigenmode."""
    decomps = _state["decomps"]
    embs = _state["embs"]
    vocab = _state["vocab"]

    if variety not in decomps:
        return {"error": f"Unknown variety: {variety}"}

    decomp = decomps[variety]
    if mode_id >= decomp.n_modes:
        return {"error": f"Mode {mode_id} out of range (max {decomp.n_modes - 1})"}

    ref_emb = embs[REFERENCE_VARIETY]
    eigvec = decomp.P[:, mode_id].real
    eigvec_norm = eigvec / (np.linalg.norm(eigvec) + 1e-12)
    loadings = ref_emb @ eigvec_norm

    # Top and bottom words
    n_words = 30
    pos_idx = np.argsort(loadings)[::-1][:n_words]
    neg_idx = np.argsort(loadings)[:n_words]

    return {
        "mode": mode_id,
        "variety": variety,
        "eigenvalue_real": float(decomp.eigenvalues[mode_id].real),
        "eigenvalue_imag": float(decomp.eigenvalues[mode_id].imag),
        "magnitude": float(np.abs(decomp.eigenvalues[mode_id])),
        "top_positive": [{"word": vocab[i], "loading": float(loadings[i])} for i in pos_idx],
        "top_negative": [{"word": vocab[i], "loading": float(loadings[i])} for i in neg_idx],
    }


@app.get("/api/regionalisms")
async def get_regionalisms():
    """Return regionalism word lists."""
    return {v: sorted(words) for v, words in REGIONALISMS.items()}


@app.get("/api/varieties")
async def get_varieties():
    """Return variety metadata."""
    return {
        "varieties": _state["varieties"],
        "coordinates": {v: list(c) for v, c in DIALECT_COORDINATES.items()},
        "regionalisms_count": {v: len(words) for v, words in REGIONALISMS.items()},
    }


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8400)
