#!/usr/bin/env python3
"""Classify text by dialect using projection-space centroids.

The ArcFace classifier head collapsed during training (all cosines = -0.9999),
but the contrastive projection head learned real dialect structure (SupCon
loss converged to 2.65 with meaningful separation). This script builds
per-variety centroids from corpus texts in projection space, then classifies
new texts by nearest centroid (cosine similarity).

Usage:
    # Build centroids first (one-time, ~2 min):
    python scripts/classify_projection.py --build-centroids

    # Classify text:
    python scripts/classify_projection.py "Che boludo qué decís"

    # Interactive mode:
    python scripts/classify_projection.py --interactive
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CHECKPOINT = ROOT / "outputs" / "eigen3" / "transformer" / "checkpoints" / "step_00162000.pt"
CORPUS_DIR = ROOT / "data" / "processed_v4"
CENTROIDS_PATH = ROOT / "outputs" / "eigen3" / "projection_centroids.npz"

VARIETIES = [
    "ES_PEN", "ES_AND", "ES_CAN", "ES_RIO",
    "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO",
]

VARIETY_NAMES = {
    "ES_PEN": "Peninsular (Spain)",
    "ES_AND": "Andalusian",
    "ES_CAN": "Canarian",
    "ES_RIO": "Rioplatense (Argentina)",
    "ES_MEX": "Mexican",
    "ES_CAR": "Caribbean",
    "ES_CHI": "Chilean",
    "ES_AND_BO": "Andean-Bolivian",
}

DOCS_PER_VARIETY = 200  # For centroid computation


def load_model():
    from eigen3.model import DialectTransformer

    model = DialectTransformer(
        model_name="dccuchile/bert-base-spanish-wwm-cased",
        proj_dim=384,
    )
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, step={ckpt['global_step']}")
    del ckpt

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    return model, device


def encode_texts(model, device, texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Encode texts to L2-normalized projection embeddings."""
    tokenizer = model.tokenizer
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            _, _, proj_emb = model(input_ids, attention_mask, labels=None)

        all_embs.append(proj_emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def build_centroids(model, device):
    """Build per-variety centroids from corpus texts."""
    print(f"\nBuilding centroids from {DOCS_PER_VARIETY} docs per variety...")
    centroids = {}

    for variety in VARIETIES:
        path = CORPUS_DIR / f"{variety}.jsonl"
        if not path.exists():
            print(f"  WARNING: Missing corpus {path}")
            continue

        # Load texts
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "").strip()
                if text and len(text) > 20:  # Skip very short texts
                    texts.append(text)
                if len(texts) >= DOCS_PER_VARIETY:
                    break

        if not texts:
            print(f"  WARNING: No valid texts for {variety}")
            continue

        # Encode
        embs = encode_texts(model, device, texts)

        # Centroid = mean of L2-normalized embeddings, then re-normalize
        centroid = embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids[variety] = centroid

        print(f"  {variety}: {len(texts)} texts -> centroid norm={np.linalg.norm(centroid):.3f}")

    # Save
    CENTROIDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(CENTROIDS_PATH), **centroids)
    print(f"\nCentroids saved to {CENTROIDS_PATH}")
    return centroids


def load_centroids() -> dict[str, np.ndarray]:
    """Load precomputed centroids."""
    data = np.load(str(CENTROIDS_PATH))
    return {k: data[k] for k in data.files}


def classify(model, device, text: str, centroids: dict[str, np.ndarray]):
    """Classify text by nearest centroid in projection space."""
    emb = encode_texts(model, device, [text])[0]  # (384,)

    # Cosine similarity to each centroid
    sims = {}
    for variety, centroid in centroids.items():
        sims[variety] = float(np.dot(emb, centroid))

    # Softmax over similarities (temperature-scaled for sharper distribution)
    temp = 0.1
    vals = np.array([sims[v] for v in VARIETIES if v in sims])
    keys = [v for v in VARIETIES if v in sims]
    exp_vals = np.exp((vals - vals.max()) / temp)
    probs = exp_vals / exp_vals.sum()

    result = {k: float(p) for k, p in zip(keys, probs)}
    ranked = sorted(result.items(), key=lambda x: -x[1])
    return ranked, sims


def main():
    args = sys.argv[1:]

    if "--build-centroids" in args:
        model, device = load_model()
        build_centroids(model, device)
        return

    # Load centroids
    if not CENTROIDS_PATH.exists():
        print("No centroids found. Building them first...")
        model, device = load_model()
        centroids = build_centroids(model, device)
    else:
        model, device = load_model()
        centroids = load_centroids()
        print(f"Loaded centroids for {len(centroids)} varieties")

    if args and args[0] != "--interactive":
        text = " ".join(args)
        ranked, sims = classify(model, device, text, centroids)
        code = ranked[0][0]
        name = VARIETY_NAMES[code]
        print(f'\n  "{text}"')
        print(f"  -> {code} ({name})")
        print()
        for variety, prob in ranked:
            name = VARIETY_NAMES[variety]
            cos = sims[variety]
            bar = "#" * int(prob * 40)
            print(f"  {variety:>10} ({name:>25})  {prob:5.1%}  cos={cos:+.3f}  {bar}")
        return

    # Interactive mode
    print("\nProjection-Space Dialect Classifier (type 'quit' to exit)")
    print("-" * 60)
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ("quit", "exit", "q"):
            break

        ranked, sims = classify(model, device, text, centroids)
        code = ranked[0][0]
        name = VARIETY_NAMES[code]
        print(f"  -> {code} ({name})")
        for variety, prob in ranked[:4]:
            cos = sims[variety]
            bar = "#" * int(prob * 40)
            print(f"     {variety:>10}  {prob:5.1%}  cos={cos:+.3f}  {bar}")


if __name__ == "__main__":
    main()
