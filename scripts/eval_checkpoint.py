#!/usr/bin/env python3
"""Quick evaluation of a trained checkpoint without full corpus extraction.

Loads a checkpoint, extracts embeddings for a small corpus sample,
runs alignment + W computation + eigendecomposition + dialect scoring.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_checkpoint")

CORPUS_DIR = ROOT / "data" / "processed_v4"
OUTPUT_DIR = ROOT / "outputs" / "eigen3"
CHECKPOINT = OUTPUT_DIR / "transformer" / "checkpoints" / "step_00162000.pt"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"

MAX_DOCS_PER_VARIETY = 500  # Small sample for speed
MAX_VOCAB = 5000  # Most frequent words only

VARIETIES = [
    "ES_PEN", "ES_AND", "ES_CAN", "ES_RIO",
    "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO",
]

# Test sentences with known dialectal markers
TEST_SENTENCES = {
    "ES_RIO": [
        "che boludo vos sabés que el pibe se fue al laburo temprano",
        "dale que la mina esa es re copada y no te va a bardear",
        "vamos a tomar unos mates en la vereda esta tarde",
    ],
    "ES_MEX": [
        "orale güey ya mero llegamos a la chamba no mames",
        "ándale pues vamos por unas chelas al tianguis",
        "qué onda carnal nos vemos en la esquina del barrio",
    ],
    "ES_PEN": [
        "oye tío que el chaval ha ido al curro esta mañana temprano",
        "vale pues vamos a tomar unas cañas en el bar de la esquina",
        "mira que el chico ese es majo pero un poco borde a veces",
    ],
    "ES_CAN": [
        "mira mi niño ven acá que vamos pa la guagua ahora mismo",
        "qué es lo que hay mi socio nos vemos en la placita",
    ],
    "ES_CHI": [
        "oye huevón cachai que la weá está terrible fome hoy día",
        "ya po cabro chico anda a comprar unas sopaipillas altiro",
    ],
    "ES_CAR": [
        "mira chamo esa vaina está demasiado chimba vale",
        "hey pana vamos a la rumba que hoy hay tremenda fiesta",
    ],
    "ES_AND": [
        "quillo mira que eze pisha se ha ido pa la feria esta mañana",
        "venga arma que vamos a tomarnos unas tapitas por el centro",
    ],
    "ES_AND_BO": [
        "mira la wawa está llorando y hay que ir al mercado rápido",
        "oye hermano vamos nomás a la fiesta del pueblo esta noche",
    ],
}


def load_corpus_sample():
    """Load a small sample of the corpus for quick evaluation."""
    corpus = {}
    for variety in VARIETIES:
        path = CORPUS_DIR / f"{variety}.jsonl"
        if not path.exists():
            logger.warning("Missing corpus: %s", path)
            continue
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= MAX_DOCS_PER_VARIETY:
                    break
                data = json.loads(line)
                text = data.get("text", "")
                if text.strip():
                    docs.append(text)
        corpus[variety] = docs
        logger.info("Loaded %d docs for %s", len(docs), variety)
    return corpus


def build_small_vocab(corpus, max_words=MAX_VOCAB):
    """Build vocabulary from the most frequent words in the sample."""
    from collections import Counter
    counts = Counter()
    for docs in corpus.values():
        for doc in docs:
            for word in doc.lower().split():
                word = word.strip(".,;:!?\"'()[]{}")
                if word and len(word) >= 2:
                    counts[word] += 1
    # Take most common
    vocab = [w for w, _ in counts.most_common(max_words)]
    logger.info("Built sample vocab: %d words", len(vocab))
    return vocab


def main():
    t0 = time.time()

    # Load model
    logger.info("Loading model...")
    from eigen3.model import DialectTransformer

    model = DialectTransformer(
        model_name="dccuchile/bert-base-spanish-wwm-cased",
        proj_dim=384,
    )

    # Load checkpoint
    logger.info("Loading checkpoint: %s", CHECKPOINT)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    logger.info("Checkpoint: epoch=%d, step=%d", ckpt["epoch"], ckpt["global_step"])
    del ckpt

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Load corpus sample
    corpus = load_corpus_sample()

    # Build small vocab from sample
    vocab = build_small_vocab(corpus)

    # Extract embeddings
    logger.info("Extracting word embeddings...")
    from eigen3.composer import TransformerWordComposer

    composer = TransformerWordComposer(model, device=device, batch_size=32, max_length=256)
    word_embs = composer.compose_vocabulary(vocab, corpus)

    # Alignment
    logger.info("Aligning to reference (ES_PEN)...")
    from eigen3.alignment import align_all_to_reference
    from eigen3.vocab import get_anchor_indices

    anchor_indices = get_anchor_indices(vocab, min_anchors=20)
    logger.info("Found %d anchor words", len(anchor_indices))
    aligned = align_all_to_reference(word_embs, anchor_indices, "ES_PEN")

    # Basic embedding stats
    print("\n" + "=" * 60)
    print("EMBEDDING STATISTICS")
    print("=" * 60)
    for v in sorted(aligned.keys()):
        emb = aligned[v]
        norms = np.linalg.norm(emb, axis=1)
        nonzero = np.count_nonzero(norms > 1e-6)
        print(f"  {v}: shape={emb.shape}  nonzero={nonzero}/{emb.shape[0]}  "
              f"mean_norm={norms[norms > 1e-6].mean():.3f}")

    # Cross-variety cosine similarities
    print("\n" + "=" * 60)
    print("CROSS-VARIETY MEAN COSINE (higher = more similar)")
    print("=" * 60)

    # Only use words that have nonzero embeddings in ALL varieties
    valid_mask = np.ones(len(vocab), dtype=bool)
    for v in aligned:
        norms = np.linalg.norm(aligned[v], axis=1)
        valid_mask &= (norms > 1e-6)
    n_valid = valid_mask.sum()
    print(f"  Words with embeddings in all varieties: {n_valid}/{len(vocab)}")

    if n_valid > 0:
        # Normalize
        normed = {}
        for v in sorted(aligned.keys()):
            emb = aligned[v][valid_mask]
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            normed[v] = emb / np.maximum(norms, 1e-8)

        varieties = sorted(normed.keys())
        print(f"\n  {'':>10}", end="")
        for v in varieties:
            print(f"  {v:>8}", end="")
        print()

        cos_matrix = np.zeros((len(varieties), len(varieties)))
        for i, v1 in enumerate(varieties):
            print(f"  {v1:>10}", end="")
            for j, v2 in enumerate(varieties):
                cos = np.mean(np.sum(normed[v1] * normed[v2], axis=1))
                cos_matrix[i, j] = cos
                print(f"  {cos:>8.3f}", end="")
            print()

        # Summary stats
        triu_idx = np.triu_indices_from(cos_matrix, k=1)
        print(f"\n  Mean pairwise cosine: {cos_matrix[triu_idx].mean():.3f}")
        print(f"  Min pairwise cosine:  {cos_matrix[triu_idx].min():.3f}")
        print(f"  Max pairwise cosine:  {cos_matrix[triu_idx].max():.3f}")

    # W matrices + eigendecomposition
    print("\n" + "=" * 60)
    print("W TRANSFORMATION MATRICES & EIGENDECOMPOSITION")
    print("=" * 60)

    from eigen3.transformation import compute_all_W
    from eigen3.decomposition import eigendecompose

    W_dict = compute_all_W(aligned, "ES_PEN")
    decomps = {}
    for v, tm in W_dict.items():
        decomps[v] = eigendecompose(tm.W, variety=v)
        mags = decomps[v].magnitudes
        eff_rank = np.searchsorted(np.cumsum(mags / mags.sum()), 0.9) + 1
        print(f"  {v}: top eigenvalue |λ|={mags[0]:.3f}  "
              f"effective_rank(90%)={eff_rank}  "
              f"spectral_gap={mags[0] - mags[1]:.4f}")

    # Distance matrix
    print("\n" + "=" * 60)
    print("FROBENIUS DISTANCE MATRIX")
    print("=" * 60)

    from eigen3.distance import distance_matrix
    D, d_varieties = distance_matrix({v: tm.W for v, tm in W_dict.items()}, "frobenius")

    print(f"  {'':>10}", end="")
    for v in d_varieties:
        print(f"  {v:>8}", end="")
    print()
    for i, v1 in enumerate(d_varieties):
        print(f"  {v1:>10}", end="")
        for j in range(len(d_varieties)):
            print(f"  {D[i,j]:>8.3f}", end="")
        print()

    # Dialect scoring on test sentences
    print("\n" + "=" * 60)
    print("DIALECT SCORING ON TEST SENTENCES")
    print("=" * 60)

    from eigen3.scorer import DialectScorer

    scorer = DialectScorer(aligned, vocab, decomps, reference="ES_PEN")

    correct = 0
    total = 0
    for true_variety, sentences in TEST_SENTENCES.items():
        for sent in sentences:
            result = scorer.score(sent)
            pred = result.top_dialect
            is_correct = pred == true_variety
            correct += int(is_correct)
            total += 1
            mark = "+" if is_correct else "X"
            top3 = sorted(result.probabilities.items(), key=lambda x: -x[1])[:3]
            top3_str = ", ".join(f"{v}={p:.2f}" for v, p in top3)
            print(f"  [{mark}] True={true_variety:>10}  Pred={pred:>10}  | {top3_str}")
            print(f"       \"{sent[:60]}...\"" if len(sent) > 60 else f"       \"{sent}\"")

    accuracy = correct / max(total, 1)
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.1%}")

    elapsed = time.time() - t0
    print(f"\n  Evaluation completed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
