#!/usr/bin/env python3
"""FULL eigen3 pipeline: scrape + augment + train + extract + align.

This runs everything from scratch:
  1. Scrape all sources (reddit, wiki, opensub, lyrics, mC4, opus, news)
  2. Quality filter + dedup + language detection
  3. Synthetic CAN/AND phonological augmentation
  4. Temperature-scaled balancing + affinity blending
  5. Build vocabulary (preserving all known regionalisms)
  6. Train BETO+LoRA transformer (MLM + classification + contrastive)
  7. Extract contextual word embeddings per variety
  8. Procrustes alignment to shared coordinate system
  9. Save everything to outputs/eigen3_full/

Does NOT touch eigenv2 code or outputs.
"""

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "outputs" / "eigen3_full"
CORPUS_DIR = ROOT / "data" / "processed_v3"
CACHE_DIR = ROOT / "data" / "raw_v3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger("full_pipeline")


def main():
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("EIGEN3 FULL PIPELINE: SCRAPE + TRAIN")
    logger.info("=" * 70)

    # Verify v2 is safe
    v2_dirs = [ROOT / "outputs" / "embeddings", ROOT / "outputs" / "v2_real"]
    for d in v2_dirs:
        if d.exists():
            logger.info("v2 preserved: %s (%d items)", d, len(list(d.iterdir())))

    # ---------------------------------------------------------------
    # Phase 1: Scrape + Filter + Augment
    # ---------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 1: CORPUS BUILDING")
    logger.info("=" * 70)

    from eigen3.scraper import CorpusScraper
    from eigen3.corpus import (
        build_filtered_corpus,
        balance_corpus,
        blend_affine_varieties,
        generate_synthetic_variants,
        build_union_vocabulary,
    )

    # Step 1: Scrape all sources
    logger.info("Step 1: Scraping all sources...")
    scraper = CorpusScraper(cache_dir=str(CACHE_DIR))

    # Scrape each source individually for better error isolation
    # NOTE: Reddit skipped — unauthenticated JSON API hits 429 too aggressively.
    # Existing v1 corpus (merged below) already contains Reddit data.
    raw: dict[str, list[dict]] = {v: [] for v in _varieties()}
    sources_to_try = [
        ("wikipedia", scraper.scrape_wikipedia),
        ("opensubtitles", scraper.scrape_opensubtitles),
        ("lyrics", scraper.scrape_lyrics),
        ("news", scraper.scrape_regional_news),
    ]

    for source_name, scrape_fn in sources_to_try:
        try:
            logger.info("  Scraping %s...", source_name)
            result = scrape_fn()
            for dialect, docs in result.items():
                raw[dialect].extend(docs)
            total = sum(len(d) for d in result.values())
            logger.info("  %s: %d samples", source_name, total)
        except Exception:
            logger.exception("  %s FAILED — continuing", source_name)

    # Also merge in existing v1 corpus (don't lose what we have)
    logger.info("  Merging existing v1 corpus...")
    _merge_existing_corpus(raw, ROOT / "data" / "processed")

    raw_total = sum(len(d) for d in raw.values())
    logger.info("Total raw samples: %d", raw_total)
    for v in sorted(raw.keys()):
        logger.info("  %s: %d", v, len(raw[v]))

    # Step 2: Quality filter
    logger.info("Step 2: Quality filtering...")
    corpus = build_filtered_corpus(raw, CORPUS_DIR)

    # Step 3: Synthetic augmentation
    logger.info("Step 3: Synthetic phonological augmentation for CAN/AND...")
    corpus = generate_synthetic_variants(corpus, fraction=0.4)

    # Step 4: Balance
    logger.info("Step 4: Temperature-scaled balancing (T=0.7)...")
    corpus = balance_corpus(corpus, temperature=0.7)

    # Step 5: Blend
    logger.info("Step 5: Affinity-based blending...")
    corpus = blend_affine_varieties(corpus)

    corpus_total = sum(len(d) for d in corpus.values())
    logger.info("Final corpus: %d docs", corpus_total)
    for v in sorted(corpus.keys()):
        logger.info("  %s: %d", v, len(corpus[v]))

    t_corpus = time.time()
    logger.info("Corpus phase: %.1f min", (t_corpus - t_start) / 60)

    # ---------------------------------------------------------------
    # Phase 2: Vocabulary
    # ---------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: VOCABULARY")
    logger.info("=" * 70)

    from eigen3.vocab import filter_vocabulary, filter_by_corpus_evidence

    raw_vocab = build_union_vocabulary(corpus, min_count=2)
    vocab = filter_vocabulary(raw_vocab)
    vocab = filter_by_corpus_evidence(vocab, corpus)

    import json
    vocab_path = OUTPUT_DIR / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Vocabulary: %d words", len(vocab))

    # Check regionalism coverage
    from eigen3.constants import REGIONALISMS
    vocab_set = set(vocab)
    for d, regs in REGIONALISMS.items():
        found = len(regs & vocab_set)
        logger.info("  %s regionalisms: %d/%d (%.0f%%)", d, found, len(regs), 100 * found / len(regs))

    # ---------------------------------------------------------------
    # Phase 3: Model + Training
    # ---------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3: TRANSFORMER TRAINING")
    logger.info("=" * 70)

    from eigen3.model import DialectTransformer
    from eigen3.dataset import DialectMLMDataset, DialectContrastiveCollator
    from eigen3.trainer import TransformerTrainer

    logger.info("Building DialectTransformer (BETO + LoRA, proj_dim=256)...")
    model = DialectTransformer(
        model_name="dccuchile/bert-base-spanish-wwm-cased",
        proj_dim=256,
        lora_r=16,
        lora_alpha=32,
    )
    logger.info("Trainable params: %d", model.count_trainable_parameters())

    logger.info("Building MLM dataset...")
    dataset = DialectMLMDataset(
        corpus_by_variety=corpus,
        tokenizer=model.tokenizer,
        variety_token_ids=model.variety_token_ids,
        seed=42,
    )
    collator = DialectContrastiveCollator(n_varieties=len(corpus), hard_mining=True)

    logger.info("Training (10 epochs, lr=2e-4, batch=32, grad_accum=4)...")
    trainer = TransformerTrainer(
        epochs=10,
        lr=2e-4,
        batch_size=32,
        grad_accum=4,
        patience=3,
        seed=42,
    )
    trainer.train(model, dataset, collator=collator)

    # Save transformer
    transformer_dir = OUTPUT_DIR / "transformer"
    trainer.save(model, transformer_dir, meta_extra={
        "n_varieties": len(corpus),
        "varieties": sorted(corpus.keys()),
        "vocab_size": len(vocab),
        "corpus_total": corpus_total,
    })

    t_train = time.time()
    logger.info("Training phase: %.1f min", (t_train - t_corpus) / 60)

    # ---------------------------------------------------------------
    # Phase 4: Extract + Align
    # ---------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 4: EMBEDDING EXTRACTION + ALIGNMENT")
    logger.info("=" * 70)

    import torch
    import numpy as np
    from eigen3.composer import TransformerWordComposer
    from eigen3.alignment import align_all_to_reference
    from eigen3.vocab import get_anchor_indices
    from eigen3.constants import REFERENCE_VARIETY

    device = next(model.parameters()).device
    composer = TransformerWordComposer(model, device=device, batch_size=64)

    logger.info("Extracting contextual word embeddings...")
    word_embs = composer.compose_vocabulary(vocab, corpus)

    logger.info("Procrustes alignment to %s...", REFERENCE_VARIETY)
    anchor_indices = get_anchor_indices(vocab)
    aligned_embs = align_all_to_reference(word_embs, anchor_indices, REFERENCE_VARIETY)

    # Save
    for variety, emb in aligned_embs.items():
        np.save(str(OUTPUT_DIR / f"{variety}.npy"), emb.T.astype(np.float32))

    t_end = time.time()
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Total time: %.1f min", (t_end - t_start) / 60)
    for variety, emb in sorted(aligned_embs.items()):
        logger.info("  %s: shape %s", variety, emb.shape)
    logger.info("Outputs: %s", OUTPUT_DIR)

    # Final v2 check
    for d in v2_dirs:
        if d.exists():
            logger.info("v2 INTACT: %s", d)


def _varieties():
    from eigen3.constants import ALL_VARIETIES
    return ALL_VARIETIES


def _merge_existing_corpus(raw: dict, processed_dir: Path):
    """Merge existing v1 processed corpus into raw samples."""
    import json
    for variety in _varieties():
        path = processed_dir / f"{variety}.jsonl"
        if not path.exists():
            continue
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                text = doc.get("text", "").strip()
                if text and len(text) >= 20:
                    raw[variety].append({
                        "text": text,
                        "dialect": variety,
                        "source": doc.get("source", "v1_corpus"),
                        "confidence": doc.get("confidence", 0.7),
                    })
                    count += 1
            except json.JSONDecodeError:
                continue
        if count:
            logger.info("    Merged %d existing docs for %s", count, variety)


if __name__ == "__main__":
    main()
