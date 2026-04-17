#!/usr/bin/env python3
"""Train eigen3 transformer pipeline end-to-end.

Loads an existing corpus directory, applies synthetic augmentation for
CAN/AND, trains BETO+LoRA, extracts contextual word embeddings, aligns
via Procrustes, and saves everything to ``--output-dir``.

Does NOT modify eigenv2 outputs or code.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir", type=Path, default=ROOT / "data" / "processed",
        help="Directory containing per-variety JSONL files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "eigen3",
        help="Directory to write transformer + embeddings + vocab",
    )
    parser.add_argument(
        "--log-file", type=Path, default=ROOT / "outputs" / "eigen3_train.log",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Legacy flag. v2 derives batch = samples_per_variety * n_varieties. "
             "If set, must equal samples_per_variety * 8 or the run aborts.",
    )
    parser.add_argument(
        "--samples-per-variety", type=int, default=4,
        help="Per-variety samples in every balanced batch (k). "
             "Effective batch size = k * n_varieties.",
    )
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default=None,
        help="torch device override (default: auto MPS/CUDA/CPU)",
    )
    parser.add_argument(
        "--model-name", type=str,
        default="dccuchile/bert-base-spanish-wwm-cased",
    )
    parser.add_argument(
        "--log-every-steps", type=int, default=50,
        help="Log a rich progress line every N optimizer steps",
    )
    parser.add_argument(
        "--checkpoint-every-steps", type=int, default=0,
        help="Save a checkpoint every N optimizer steps. 0 = epoch-only",
    )
    parser.add_argument(
        "--keep-last-checkpoints", type=int, default=3,
        help="Number of mid-epoch checkpoints to retain",
    )
    parser.add_argument(
        "--queue-size", type=int, default=4096,
        help="MoCo queue size (0 disables momentum contrastive)",
    )
    parser.add_argument(
        "--moco-momentum", type=float, default=0.999,
        help="EMA momentum coefficient for momentum encoder",
    )
    parser.add_argument(
        "--moco-start-epoch", type=int, default=4,
        help="Epoch when MoCo queue starts being used (1-indexed)",
    )
    parser.add_argument(
        "--queue-ramp-steps", type=int, default=5000,
        help="Optimizer steps to ramp queue from 128 to full size",
    )
    parser.add_argument(
        "--supcon-temperature", type=float, default=0.07,
        help="Temperature for the supervised contrastive loss",
    )
    parser.add_argument(
        "--pretrain-epochs", type=int, default=2,
        help="Epochs of MLM+CLS only before adding SupCon",
    )
    parser.add_argument(
        "--proj-lr-mult", type=float, default=10.0,
        help="LR multiplier for projection head parameters",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=5000,
        help="Optimizer steps for linear warmup",
    )
    parser.add_argument(
        "--center-warmup-steps", type=int, default=2000,
        help="Contrastive opt steps with center loss active",
    )
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Checkpoint .pt file to resume training from",
    )
    args = parser.parse_args()

    # Backstop: the legacy --batch-size flag must agree with the derived
    # v2 batch. If the user passes an incompatible value we abort loudly
    # instead of silently ignoring it.
    n_var_assumed = 8
    derived = args.samples_per_variety * n_var_assumed
    if args.batch_size is not None and args.batch_size != derived:
        raise SystemExit(
            f"--batch-size={args.batch_size} conflicts with derived v2 batch "
            f"({args.samples_per_variety} * {n_var_assumed} = {derived}). "
            f"Drop --batch-size or set --samples-per-variety={args.batch_size // n_var_assumed}."
        )
    args.batch_size = derived
    return args


def main() -> None:
    args = _parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file, mode="w"),
        ],
    )
    logger = logging.getLogger("train_eigen3")

    from eigen3.emb_pipeline import run_embedding_pipeline

    logger.info("=" * 60)
    logger.info("EIGEN3 TRANSFORMER TRAINING PIPELINE (v2 SupCon+MoCo+DCL)")
    logger.info("=" * 60)
    logger.info("Corpus     : %s", args.corpus_dir)
    logger.info("Output     : %s", args.output_dir)
    logger.info("Method     : transformer (BETO + LoRA, LoRA ⊇ attn+FFN)")
    logger.info("Proj dim   : %d", args.embedding_dim)
    logger.info("Max length : %d", args.max_length)
    logger.info("Epochs     : %d  LR: %.1e", args.epochs, args.lr)
    logger.info("k/var      : %d  (batch = %d)", args.samples_per_variety, args.batch_size)
    logger.info("MoCo       : queue=%d  m=%.4f  start_ep=%d  ramp=%d  τ=%.3f",
                args.queue_size, args.moco_momentum, args.moco_start_epoch,
                args.queue_ramp_steps, args.supcon_temperature)
    logger.info("Phases     : pretrain=%d  proj_lr=%.1fx  warmup=%d  center=%d",
                args.pretrain_epochs, args.proj_lr_mult, args.warmup_steps,
                args.center_warmup_steps)
    logger.info("Balance T  : %.2f  Seed: %d", args.temperature, args.seed)
    logger.info("=" * 60)

    # Verify eigenv2 outputs are safe
    v2_dirs = [
        ROOT / "outputs" / "embeddings",
        ROOT / "outputs" / "v2_real",
        ROOT / "outputs" / "v2_test",
    ]
    for d in v2_dirs:
        if d.exists():
            logger.info("v2 output preserved: %s (%d files)", d, len(list(d.iterdir())))

    embeddings = run_embedding_pipeline(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        method="transformer",
        embedding_dim=args.embedding_dim,
        model_name=args.model_name,
        temperature=args.temperature,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        scrape=False,
        log_every_steps=args.log_every_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        keep_last_checkpoints=args.keep_last_checkpoints,
        max_length=args.max_length,
        samples_per_variety=args.samples_per_variety,
        queue_size=args.queue_size,
        moco_momentum=args.moco_momentum,
        moco_start_epoch=args.moco_start_epoch,
        queue_ramp_steps=args.queue_ramp_steps,
        supcon_temperature=args.supcon_temperature,
        pretrain_epochs=args.pretrain_epochs,
        proj_lr_mult=args.proj_lr_mult,
        warmup_steps=args.warmup_steps,
        center_warmup_steps=args.center_warmup_steps,
        resume_from=args.resume_from,
    )

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    for variety, emb in sorted(embeddings.items()):
        logger.info("  %s: shape %s", variety, emb.shape)

    for d in v2_dirs:
        if d.exists():
            logger.info("v2 output still intact: %s", d)

    logger.info("All outputs saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
