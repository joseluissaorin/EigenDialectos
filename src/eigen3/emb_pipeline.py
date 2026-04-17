"""Full embedding pipeline: corpus -> train -> extract -> align -> save.

Two training methods:
  "transformer" — BETO + LoRA + variety tokens (recommended)
  "skipgram"    — Legacy BPE + DCL skip-gram (backward compat)

Optionally scrapes corpus automatically before training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from eigen3.alignment import align_all_to_reference
from eigen3.constants import (
    ALL_VARIETIES,
    BPE_VOCAB_SIZE,
    DEFAULT_SEED,
    EMBEDDING_DIM,
    REFERENCE_VARIETY,
)
from eigen3.corpus import (
    balance_corpus,
    blend_affine_varieties,
    build_union_vocabulary,
    generate_synthetic_variants,
    load_corpus,
)
from eigen3.vocab import filter_by_corpus_evidence, filter_vocabulary, get_anchor_indices

logger = logging.getLogger(__name__)


def run_embedding_pipeline(
    corpus_dir: str | Path,
    output_dir: str | Path,
    method: str = "transformer",
    embedding_dim: int = 384,
    model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
    bpe_vocab_size: int = BPE_VOCAB_SIZE,
    temperature: float = 0.7,
    epochs: int = 10,
    lr: float = 2e-4,
    batch_size: int = 32,
    seed: int = DEFAULT_SEED,
    device: Optional[str] = None,
    scrape: bool = False,
    scrape_sources: list[str] | None = None,
    download: bool = False,
    download_sources: list[str] | None = None,
    max_per_dialect: int = 200_000,
    hf_token: Optional[str] = None,
    log_every_steps: int = 50,
    checkpoint_every_steps: int = 0,
    keep_last_checkpoints: int = 3,
    max_length: int = 256,
    samples_per_variety: int = 4,
    queue_size: int = 4096,
    moco_momentum: float = 0.999,
    moco_start_epoch: int = 4,
    queue_ramp_steps: int = 5000,
    supcon_temperature: float = 0.07,
    pretrain_epochs: int = 2,
    proj_lr_mult: float = 10.0,
    warmup_steps: int = 5000,
    center_warmup_steps: int = 2000,
    resume_from: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Run the full embedding pipeline end-to-end.

    Parameters
    ----------
    method : str
        "transformer" — BETO + LoRA (recommended)
        "skipgram"    — Legacy BPE + DCL
    scrape : bool
        If True, automatically scrape corpus before training (legacy).
    download : bool
        If True, download bulk corpus before training (recommended).

    Returns aligned word-level embeddings per variety.
    """
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Optional: download bulk corpus (recommended path)
    if download:
        logger.info("Step 0: Downloading bulk corpus")
        from eigen3.corpus import download_and_build
        download_and_build(
            output_dir=corpus_dir,
            sources=download_sources,
            max_per_dialect=max_per_dialect,
            temperature=temperature,
            seed=seed,
            hf_token=hf_token,
        )
    # 0b. Legacy: scrape corpus
    elif scrape:
        logger.info("Step 0: Scraping corpus automatically")
        from eigen3.corpus import scrape_and_build
        scrape_and_build(
            output_dir=corpus_dir,
            sources=scrape_sources,
            temperature=temperature,
            seed=seed,
        )

    # 1. Load
    logger.info("Step 1: Loading corpus from %s", corpus_dir)
    corpus = load_corpus(corpus_dir)

    # 2. Synthetic augmentation for underrepresented dialects (CAN, AND)
    logger.info("Step 2: Synthetic phonological augmentation for CAN/AND")
    corpus = generate_synthetic_variants(corpus, seed=seed)

    # 3. Balance
    logger.info("Step 3: Balancing corpus (T=%.1f)", temperature)
    corpus = balance_corpus(corpus, temperature=temperature, seed=seed)

    # 4. Blend
    logger.info("Step 4: Blending affine varieties")
    corpus = blend_affine_varieties(corpus, seed=seed)

    # 5. Vocabulary
    logger.info("Step 5: Building and filtering vocabulary")
    raw_vocab = build_union_vocabulary(corpus, min_count=2)
    vocab = filter_vocabulary(raw_vocab)
    vocab = filter_by_corpus_evidence(vocab, corpus)

    # Save vocab
    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Vocabulary: %d words saved to %s", len(vocab), vocab_path)

    # Dispatch to method-specific pipeline
    if method == "transformer":
        aligned_embs = _run_transformer_pipeline(
            corpus=corpus,
            vocab=vocab,
            output_dir=output_dir,
            model_name=model_name,
            embedding_dim=embedding_dim,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
            device=device,
            log_every_steps=log_every_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            keep_last_checkpoints=keep_last_checkpoints,
            max_length=max_length,
            samples_per_variety=samples_per_variety,
            queue_size=queue_size,
            moco_momentum=moco_momentum,
            moco_start_epoch=moco_start_epoch,
            queue_ramp_steps=queue_ramp_steps,
            supcon_temperature=supcon_temperature,
            pretrain_epochs=pretrain_epochs,
            proj_lr_mult=proj_lr_mult,
            warmup_steps=warmup_steps,
            center_warmup_steps=center_warmup_steps,
            resume_from=resume_from,
        )
    elif method == "skipgram":
        aligned_embs = _run_skipgram_pipeline(
            corpus=corpus,
            vocab=vocab,
            output_dir=output_dir,
            embedding_dim=embedding_dim or EMBEDDING_DIM,
            bpe_vocab_size=bpe_vocab_size,
            epochs=epochs if epochs != 10 else 30,
            lr=lr if lr != 2e-4 else 0.001,
            batch_size=batch_size if batch_size != 32 else 8192,
            seed=seed,
            device=device,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'transformer' or 'skipgram'.")

    # Save aligned embeddings
    for variety, emb in aligned_embs.items():
        np.save(str(output_dir / f"{variety}.npy"), emb.T.astype(np.float32))

    logger.info("Pipeline complete. Outputs in %s", output_dir)
    return aligned_embs


def _run_transformer_pipeline(
    corpus: dict[str, list[str]],
    vocab: list[str],
    output_dir: Path,
    model_name: str,
    embedding_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: str | None,
    log_every_steps: int = 50,
    checkpoint_every_steps: int = 0,
    keep_last_checkpoints: int = 3,
    max_length: int = 256,
    samples_per_variety: int = 4,
    queue_size: int = 4096,
    moco_momentum: float = 0.999,
    moco_start_epoch: int = 4,
    queue_ramp_steps: int = 5000,
    supcon_temperature: float = 0.07,
    pretrain_epochs: int = 2,
    proj_lr_mult: float = 10.0,
    warmup_steps: int = 5000,
    center_warmup_steps: int = 2000,
    resume_from: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Transformer path: BETO + LoRA -> contextual extraction -> Procrustes."""
    from eigen3.composer import TransformerWordComposer
    from eigen3.dataset import DialectBatchCollator, DialectMLMDataset
    from eigen3.model import DialectTransformer
    from eigen3.trainer import TransformerTrainer

    # 6. Build model
    logger.info(
        "Step 6: Building dialect transformer (proj_dim=%d, max_length=%d)",
        embedding_dim, max_length,
    )
    model = DialectTransformer(
        model_name=model_name,
        proj_dim=embedding_dim,
    )
    logger.info("Trainable params: %d", model.count_trainable_parameters())

    # 7. Build dataset
    logger.info("Step 7: Building MLM dataset")
    dataset = DialectMLMDataset(
        corpus_by_variety=corpus,
        tokenizer=model.tokenizer,
        variety_token_ids=model.variety_token_ids,
        max_length=max_length,
        seed=seed,
    )
    collator = DialectBatchCollator(n_varieties=len(corpus))

    # 8. Train
    logger.info("Step 8: Training transformer")
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    trainer = TransformerTrainer(
        epochs=epochs,
        lr=lr,
        samples_per_variety=samples_per_variety,
        seed=seed,
        device=device,
        log_every_steps=log_every_steps,
        checkpoint_every_steps=checkpoint_every_steps,
        checkpoint_dir=transformer_dir / "checkpoints",
        step_log_path=output_dir / "step_log.jsonl",
        keep_last_checkpoints=keep_last_checkpoints,
        queue_size=queue_size,
        moco_momentum=moco_momentum,
        moco_start_epoch=moco_start_epoch,
        queue_ramp_steps=queue_ramp_steps,
        supcon_temperature=supcon_temperature,
        pretrain_epochs=pretrain_epochs,
        proj_lr_mult=proj_lr_mult,
        warmup_steps=warmup_steps,
        center_warmup_steps=center_warmup_steps,
        resume_from=resume_from,
    )
    trainer.train(model, dataset, collator=collator)

    # Save transformer artifacts
    trainer.save(model, transformer_dir, meta_extra={
        "n_varieties": len(corpus),
        "varieties": sorted(corpus.keys()),
        "vocab_size": len(vocab),
    })

    # 9. Extract word embeddings
    logger.info("Step 9: Extracting contextual word embeddings")
    import torch
    dev = torch.device(device) if device else next(model.parameters()).device
    composer_batch = max(1, samples_per_variety * len(corpus) * 2)
    composer = TransformerWordComposer(
        model,
        device=dev,
        batch_size=composer_batch,
        max_length=max_length,
    )
    word_embs = composer.compose_vocabulary(vocab, corpus)

    # 10. Procrustes alignment
    logger.info("Step 10: Aligning to reference (%s)", REFERENCE_VARIETY)
    anchor_indices = get_anchor_indices(vocab)
    aligned_embs = align_all_to_reference(word_embs, anchor_indices, REFERENCE_VARIETY)

    return aligned_embs


def _run_skipgram_pipeline(
    corpus: dict[str, list[str]],
    vocab: list[str],
    output_dir: Path,
    embedding_dim: int,
    bpe_vocab_size: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: str | None,
) -> dict[str, np.ndarray]:
    """Legacy skip-gram path: BPE + DCL -> subword composition -> Procrustes."""
    from eigen3.composer import SubwordToWordComposer
    from eigen3.dataset import SubwordDCLDataset
    from eigen3.tokenizer import Tokenizer, train_tokenizer
    from eigen3.trainer import DCLTrainer

    # 6. Tokenizer
    logger.info("Step 6: Training BPE tokenizer (vocab_size=%d)", bpe_vocab_size)
    all_texts = [doc for docs in corpus.values() for doc in docs]
    tok_dir = output_dir / "tokenizer"
    tok_path = train_tokenizer(all_texts, vocab_size=bpe_vocab_size, output_dir=tok_dir)
    tokenizer = Tokenizer(tok_path)

    # 7. DCL training
    logger.info("Step 7: Training DCL embeddings")
    dataset = SubwordDCLDataset(
        corpus_by_variety=corpus,
        tokenizer=tokenizer,
        seed=seed,
    )
    trainer = DCLTrainer(
        embedding_dim=embedding_dim,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    subword_embs = trainer.train(dataset)

    # Save subword embeddings
    dcl_dir = output_dir / "dcl_subword"
    trainer.save(subword_embs, dcl_dir, meta_extra={
        "bpe_vocab_size": bpe_vocab_size,
        "n_varieties": len(subword_embs),
        "varieties": sorted(subword_embs.keys()),
    })

    # 8. Compose word-level embeddings
    logger.info("Step 8: Composing word-level embeddings")
    word_embs: dict[str, np.ndarray] = {}
    for variety, sub_emb in subword_embs.items():
        composer = SubwordToWordComposer(tokenizer, sub_emb)
        word_embs[variety] = composer.compose_vocabulary(vocab)
        logger.info("%s: composed %d word vectors", variety, len(vocab))

    # 9. Procrustes alignment
    logger.info("Step 9: Aligning to reference (%s)", REFERENCE_VARIETY)
    anchor_indices = get_anchor_indices(vocab)
    aligned_embs = align_all_to_reference(word_embs, anchor_indices, REFERENCE_VARIETY)

    return aligned_embs
