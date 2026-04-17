"""Unified embedding pipeline: trains or loads per-variety word embeddings.

Single entry point that wires together embedding components.  Called
from ``scripts/run_v2_real.py`` to produce per-variety word-level
embedding matrices in the same format the spectral pipeline expects.

Supports two methods:

- ``"fasttext_procrustes"`` (default): Per-variety fastText → Procrustes
  alignment → optional word-level DCL refinement.  Fast (~5 min on CPU).
- ``"subword_dcl"``: Shared BPE tokenizer → subword DCL training → word
  composition.  Slower (~2.5 hours on MPS).

Usage
-----
::

    from eigendialectos.embeddings.pipeline import train_or_load_embeddings

    embeddings, vocab = train_or_load_embeddings(
        corpus_dir=Path("data/processed"),
        output_dir=Path("outputs/embeddings"),
        force_retrain=False,
    )
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_corpus(corpus_dir: Path) -> dict[str, list[str]]:
    """Load per-variety text corpus from JSONL files."""
    corpus: dict[str, list[str]] = {}
    for jsonl_path in sorted(corpus_dir.glob("ES_*.jsonl")):
        variety = jsonl_path.stem
        docs: list[str] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "").strip()
                if text:
                    docs.append(text)
        corpus[variety] = docs
        logger.info("  %s: %d documents", variety, len(docs))
    return corpus


def _load_cached(output_dir: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load cached embeddings and vocabulary."""
    with open(output_dir / "vocab.json", encoding="utf-8") as f:
        vocab: list[str] = json.load(f)

    embeddings: dict[str, np.ndarray] = {}
    for npy_path in sorted(output_dir.glob("ES_*.npy")):
        # Skip subword files
        if "_subword" in npy_path.stem:
            continue
        variety = npy_path.stem
        embeddings[variety] = np.load(npy_path)
        logger.info("  Loaded %s: shape %s", variety, embeddings[variety].shape)

    return embeddings, vocab


# ======================================================================
# Default method: fastText + Procrustes
# ======================================================================


def _run_fasttext_procrustes(
    corpus: dict[str, list[str]],
    balanced: dict[str, list[str]],
    output_dir: Path,
    embedding_dim: int,
    word_min_count: int,
    seed: int,
    fasttext_epochs: int,
    fasttext_workers: int,
    dcl_refinement: bool,
    dcl_refinement_epochs: int,
    dcl_refinement_lr: float,
    balance_temperature: float,
) -> None:
    """Run the fastText + Procrustes pipeline."""
    from eigendialectos.embeddings.dcl.word_composer import build_union_vocabulary
    from eigendialectos.embeddings.fasttext_pipeline import (
        align_varieties_procrustes,
        save_embeddings,
        train_per_variety_fasttext,
    )
    from eigendialectos.embeddings.vocab_filter import (
        filter_by_corpus_evidence,
        filter_vocabulary,
        get_anchor_indices,
    )

    # ------------------------------------------------------------------
    # Step 1: Build union vocabulary + MULTI-LAYER FILTER
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Building and filtering union vocabulary")
    raw_vocab = build_union_vocabulary(corpus, min_count=word_min_count)
    logger.info("  Raw union vocabulary: %d words", len(raw_vocab))

    # Layer 1+2+3: alphabetic, min_len=3, English blacklist
    clean_vocab = filter_vocabulary(raw_vocab, min_len=3)

    # Layer 4: corpus evidence filter
    #   - Require presence in ≥2 varieties
    #   - ASCII-only words need total freq ≥30 (filters English contaminants
    #     like "wing", "king", "station" which are mid-frequency)
    #   - Accented words pass with lower threshold (definitely Spanish)
    union_vocab = filter_by_corpus_evidence(
        clean_vocab, corpus,
        min_count=word_min_count,
        min_varieties=2,
        ascii_min_total=30,
    )
    logger.info("  Final vocabulary: %d words (removed %d total)",
                len(union_vocab), len(raw_vocab) - len(union_vocab))

    # ------------------------------------------------------------------
    # Step 1b: Identify Procrustes anchor words
    # ------------------------------------------------------------------
    anchor_indices = get_anchor_indices(union_vocab, min_anchors=50)
    logger.info("  Procrustes anchors: %d function/universal words", len(anchor_indices))

    # ------------------------------------------------------------------
    # Step 2: Train per-variety fastText
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Training per-variety fastText models")
    models = train_per_variety_fasttext(
        corpus_by_variety=balanced,
        vector_size=embedding_dim,
        window=5,
        min_count=word_min_count,
        epochs=fasttext_epochs,
        min_n=3,
        max_n=6,
        sg=1,
        workers=fasttext_workers,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Step 3: Anchor-only Procrustes alignment
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Aligning varieties via anchor-only Procrustes")
    aligned = align_varieties_procrustes(
        models=models,
        vocab=union_vocab,
        reference_variety="ES_PEN",
        vector_size=embedding_dim,
        anchor_indices=anchor_indices,
    )

    # ------------------------------------------------------------------
    # Step 4 (optional): Word-level DCL refinement
    # ------------------------------------------------------------------
    if dcl_refinement:
        logger.info("=" * 60)
        logger.info("STEP 4: Word-level DCL refinement (%d epochs)", dcl_refinement_epochs)
        aligned = _run_dcl_refinement(
            aligned=aligned,
            vocab=union_vocab,
            corpus_by_variety=balanced,
            corpus_dir=None,
            epochs=dcl_refinement_epochs,
            lr=dcl_refinement_lr,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Step 5: Save
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Saving embeddings")
    save_embeddings(aligned, union_vocab, output_dir)

    # Mark as trained
    (output_dir / ".fasttext_trained").touch()


def _run_dcl_refinement(
    aligned: dict[str, np.ndarray],
    vocab: list[str],
    corpus_by_variety: dict[str, list[str]],
    corpus_dir: Path | None,
    epochs: int = 5,
    lr: float = 0.0001,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Word-level DCL refinement on top of aligned fastText embeddings."""
    import torch

    from eigendialectos.embeddings.dcl.dataset import DCLDataset
    from eigendialectos.embeddings.dcl.loss import DialectContrastiveLoss
    from eigendialectos.embeddings.dcl.model import DCLEmbeddingModel
    from eigendialectos.embeddings.dcl.regionalism_expansion import get_all_regionalisms

    regionalisms = get_all_regionalisms(corpus_dir)
    variety_names = sorted(aligned.keys())
    variety_to_idx = {v: i for i, v in enumerate(variety_names)}
    vocab_size = len(vocab)
    embedding_dim = next(iter(aligned.values())).shape[1]

    # Build word-level DCL dataset
    logger.info("  Building word-level DCL dataset ...")
    dataset = DCLDataset(
        corpus_by_variety=corpus_by_variety,
        window_size=5,
        neg_samples=5,
        regionalism_set=regionalisms,
        min_count=2,
        seed=seed,
    )

    # Create model and initialize from aligned embeddings
    model = DCLEmbeddingModel(
        vocab_size=dataset.vocab_size,
        embedding_dim=embedding_dim,
        n_varieties=len(variety_names),
    )

    # Map dataset vocab to aligned vocab for initialization
    aligned_word2idx = {w: i for i, w in enumerate(vocab)}
    with torch.no_grad():
        for v_name in variety_names:
            v_idx = variety_to_idx[v_name]
            ds_v_idx = dataset.variety_to_idx.get(v_name)
            if ds_v_idx is None:
                continue
            src_matrix = aligned[v_name]  # (vocab_size_aligned, dim)
            for ds_word_idx, word in enumerate(dataset.vocab):
                aligned_idx = aligned_word2idx.get(word)
                if aligned_idx is not None:
                    flat_idx = v_idx * dataset.vocab_size + ds_word_idx
                    vec = torch.from_numpy(src_matrix[aligned_idx])
                    model.word_emb.weight[flat_idx] = vec
                    model.ctx_emb.weight[flat_idx] = vec

    # Training loop
    criterion = DialectContrastiveLoss(lambda_anchor=0.01)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2048, shuffle=True, num_workers=0,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            word_idx, ctx_same, ctx_other, va, vb, is_reg = batch
            w_a, c_a, c_b, w_b = model(word_idx, ctx_same, ctx_other, va, vb)
            loss = criterion(w_a, c_a, c_b, w_b, is_reg)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1
        logger.info(
            "  DCL refinement epoch %d/%d  loss=%.6f",
            epoch, epochs, epoch_loss / max(n_batches, 1),
        )

    # Extract refined embeddings back into aligned format
    model.eval()
    refined: dict[str, np.ndarray] = {}
    for v_name in variety_names:
        ds_v_idx = dataset.variety_to_idx.get(v_name)
        if ds_v_idx is None:
            refined[v_name] = aligned[v_name]
            continue
        # Start from aligned embeddings
        result = aligned[v_name].copy()
        weight = model.get_word_embeddings(ds_v_idx).cpu().numpy()
        # Overwrite words that were in the DCL vocabulary
        for ds_word_idx, word in enumerate(dataset.vocab):
            aligned_idx = aligned_word2idx.get(word)
            if aligned_idx is not None:
                result[aligned_idx] = weight[ds_word_idx]
        refined[v_name] = result.astype(np.float32)

    return refined


# ======================================================================
# Legacy method: subword DCL
# ======================================================================


def _run_subword_dcl(
    corpus: dict[str, list[str]],
    balanced: dict[str, list[str]],
    output_dir: Path,
    embedding_dim: int,
    bpe_vocab_size: int,
    morpheme_aware: bool,
    dcl_epochs: int,
    dcl_lr: float,
    dcl_lambda_anchor: float,
    dcl_batch_size: int,
    dcl_window_size: int,
    dcl_neg_samples: int,
    word_min_count: int,
    seed: int,
) -> None:
    """Run the legacy subword DCL pipeline."""
    from eigendialectos.embeddings.dcl.regionalism_expansion import get_all_regionalisms

    # Step 2: Regionalisms
    logger.info("=" * 60)
    logger.info("STEP 2: Loading expanded regionalisms")
    corpus_dir_for_reg = None  # Already loaded
    regionalisms = get_all_regionalisms(corpus_dir_for_reg)
    logger.info("  Total regionalisms: %d", len(regionalisms))

    # Step 3: BPE tokenizer
    logger.info("=" * 60)
    logger.info("STEP 3: Training shared BPE tokenizer (vocab=%d)", bpe_vocab_size)
    tokenizer_dir = output_dir / "tokenizer"
    from eigendialectos.embeddings.subword.shared_tokenizer import SharedSubwordTokenizer
    tokenizer = SharedSubwordTokenizer.train(
        corpus_texts=balanced,
        output_path=tokenizer_dir,
        vocab_size=bpe_vocab_size,
        morpheme_aware=morpheme_aware,
        seed=seed,
    )

    # Step 4: Subword DCL
    logger.info("=" * 60)
    logger.info("STEP 4: Training subword DCL embeddings")
    from eigendialectos.embeddings.dcl.trainer import SubwordDCLTrainer
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass

    trainer = SubwordDCLTrainer(
        tokenizer=tokenizer,
        embedding_dim=embedding_dim,
        epochs=dcl_epochs,
        lr=dcl_lr,
        lambda_anchor=dcl_lambda_anchor,
        batch_size=dcl_batch_size,
        window_size=dcl_window_size,
        neg_samples=dcl_neg_samples,
        seed=seed,
        device=device,
    )
    subword_embs = trainer.train(balanced, regionalism_set=regionalisms)
    trainer.save(output_dir / "dcl_subword")

    # Step 5: Word composition
    logger.info("=" * 60)
    logger.info("STEP 5: Composing word-level embeddings")
    from eigendialectos.embeddings.dcl.word_composer import (
        SubwordToWordComposer,
        build_union_vocabulary,
    )
    from eigendialectos.embeddings.vocab_filter import filter_vocabulary

    raw_vocab = build_union_vocabulary(corpus, min_count=word_min_count)
    union_vocab = filter_vocabulary(raw_vocab, min_len=3)
    logger.info("  Filtered vocabulary: %d → %d words", len(raw_vocab), len(union_vocab))

    composer = SubwordToWordComposer(tokenizer, subword_embs)

    for variety in sorted(subword_embs.keys()):
        word_emb = composer.compose_vocabulary(union_vocab, variety)
        npy_path = output_dir / f"{variety}.npy"
        np.save(str(npy_path), word_emb.T.astype(np.float32))
        logger.info("  %s: (%d, %d) saved", variety, word_emb.T.shape[0], word_emb.T.shape[1])

    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(
        json.dumps(union_vocab, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output_dir / ".dcl_trained").touch()


# ======================================================================
# Variety blending for high-affinity pairs
# ======================================================================

# Variety pairs to blend, with blend fraction.
# For CAN-CAR (0.92 affinity), blend 20% of each into the other.
_BLEND_PAIRS: list[tuple[str, str, float]] = [
    ("ES_CAN", "ES_CAR", 0.20),
    ("ES_AND", "ES_AND_BO", 0.15),
]


def _blend_affine_varieties(
    balanced: dict[str, list[str]],
    seed: int = 42,
) -> dict[str, list[str]]:
    """Blend high-affinity variety corpora to boost content overlap.

    For each blend pair (A, B, frac), adds frac * len(A) random docs
    from B into A's corpus, and vice versa.  This makes the skip-gram
    contexts for A and B more similar, so the DCL training produces
    closer embeddings.

    Combined with affinity-weighted negative sampling (A and B rarely
    appear as each other's negatives), this directly makes high-affinity
    pairs converge.
    """
    import random
    rng = random.Random(seed)
    result = {v: list(docs) for v, docs in balanced.items()}

    for va, vb, frac in _BLEND_PAIRS:
        if va not in result or vb not in result:
            continue

        docs_a = balanced[va]
        docs_b = balanced[vb]
        n_blend_a = int(len(docs_a) * frac)
        n_blend_b = int(len(docs_b) * frac)

        # Sample from B into A (with replacement if needed)
        if n_blend_a > 0 and docs_b:
            blend_into_a = rng.choices(docs_b, k=n_blend_a)
            result[va].extend(blend_into_a)

        if n_blend_b > 0 and docs_a:
            blend_into_b = rng.choices(docs_a, k=n_blend_b)
            result[vb].extend(blend_into_b)

        logger.info(
            "  Blended %s ↔ %s: +%d docs into %s, +%d docs into %s",
            va, vb, n_blend_a, va, n_blend_b, vb,
        )

    return result


# ======================================================================
# Main entry point
# ======================================================================


def train_or_load_embeddings(
    corpus_dir: Path,
    output_dir: Path,
    force_retrain: bool = False,
    method: str = "fasttext_procrustes",
    embedding_dim: int = 100,
    word_min_count: int = 3,
    balance_temperature: float = 0.7,
    seed: int = 42,
    # fastText + Procrustes parameters
    fasttext_epochs: int = 10,
    fasttext_workers: int = 4,
    dcl_refinement: bool = False,
    dcl_refinement_epochs: int = 5,
    dcl_refinement_lr: float = 0.0001,
    # Legacy subword DCL parameters
    bpe_vocab_size: int = 8000,
    morpheme_aware: bool = True,
    dcl_epochs: int = 30,
    dcl_lr: float = 0.001,
    dcl_lambda_anchor: float = 0.05,
    dcl_batch_size: int = 8192,
    dcl_window_size: int = 5,
    dcl_neg_samples: int = 5,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Train or load per-variety word embeddings.

    Parameters
    ----------
    corpus_dir:
        Directory with per-variety JSONL files (``ES_*.jsonl``).
    output_dir:
        Directory for all outputs.
    force_retrain:
        If True, retrain even if cached results exist.
    method:
        ``"fasttext_procrustes"`` (default) or ``"subword_dcl"`` (legacy).

    Returns
    -------
    tuple of (embeddings_dict, vocab_list)
        ``embeddings_dict``: maps variety code to ``(dim, vocab_size)`` array.
        ``vocab_list``: list of words in the shared vocabulary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    if method == "fasttext_procrustes":
        cache_marker = output_dir / ".fasttext_trained"
    else:
        cache_marker = output_dir / ".dcl_trained"

    if cache_marker.exists() and not force_retrain:
        logger.info("Loading cached embeddings from %s (method=%s)", output_dir, method)
        return _load_cached(output_dir)

    t0 = time.perf_counter()

    # Load & balance corpus (shared by both methods)
    logger.info("=" * 60)
    logger.info("STEP 0: Loading and balancing corpus from %s", corpus_dir)
    corpus = _load_corpus(corpus_dir)

    from eigendialectos.corpus.preprocessing.balancing import balance_corpus
    balanced = balance_corpus(corpus, temperature=balance_temperature, seed=seed)
    for v in sorted(balanced):
        logger.info("  %s: %d -> %d docs", v, len(corpus[v]), len(balanced[v]))

    # Blend high-affinity variety corpora for DCL
    if method == "subword_dcl":
        balanced = _blend_affine_varieties(balanced, seed=seed)

    # Dispatch
    if method == "fasttext_procrustes":
        _run_fasttext_procrustes(
            corpus=corpus,
            balanced=balanced,
            output_dir=output_dir,
            embedding_dim=embedding_dim,
            word_min_count=word_min_count,
            seed=seed,
            fasttext_epochs=fasttext_epochs,
            fasttext_workers=fasttext_workers,
            dcl_refinement=dcl_refinement,
            dcl_refinement_epochs=dcl_refinement_epochs,
            dcl_refinement_lr=dcl_refinement_lr,
            balance_temperature=balance_temperature,
        )
    elif method == "subword_dcl":
        _run_subword_dcl(
            corpus=corpus,
            balanced=balanced,
            output_dir=output_dir,
            embedding_dim=embedding_dim,
            bpe_vocab_size=bpe_vocab_size,
            morpheme_aware=morpheme_aware,
            dcl_epochs=dcl_epochs,
            dcl_lr=dcl_lr,
            dcl_lambda_anchor=dcl_lambda_anchor,
            dcl_batch_size=dcl_batch_size,
            dcl_window_size=dcl_window_size,
            dcl_neg_samples=dcl_neg_samples,
            word_min_count=word_min_count,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown embedding method: {method!r}")

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("Embedding pipeline (%s) complete in %.1f seconds", method, elapsed)

    return _load_cached(output_dir)
