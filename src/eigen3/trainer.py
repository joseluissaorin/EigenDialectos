"""Training loops for both transformer and legacy skip-gram models.

TransformerTrainer — AdamW + linear warmup + cosine decay + checkpointing
                     with balanced per-variety sampling, SupCon, and MoCo.
DCLTrainer         — Adam + cosine annealing (legacy, kept for backward compat)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    """Format a duration as ``HH:MM:SS`` (or ``Dd HH:MM:SS`` past 24h)."""
    if seconds < 0 or not math.isfinite(seconds):
        return "??:??:??"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _mps_memory_mb() -> Optional[float]:
    """Return current MPS allocated memory in MB if available, else ``None``."""
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / (1024 * 1024)
    except Exception:
        pass
    return None


def _process_rss_mb() -> float:
    """Resident set size of the current process in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KB. Heuristic: if value is huge it
        # is bytes, otherwise KB. macOS is what we care about here.
        if usage > 10_000_000_000:
            return usage / (1024 * 1024)
        return usage / 1024
    except Exception:
        return 0.0


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Transformer trainer
# ---------------------------------------------------------------------------

class TransformerTrainer:
    """Trains dialect transformer with AdamW, warmup+cosine decay, checkpointing.

    Verbose inner-loop telemetry is emitted every ``log_every_steps`` optimizer
    steps: rolling loss components, throughput (samples/sec), live LR,
    epoch + total ETA, and MPS/RSS memory. The same numbers are appended as
    one JSON line per logged window to ``step_log_path`` (default
    ``<output_dir>/step_log.jsonl`` when ``output_dir`` is supplied) so a
    long-running training can be tail'd, grep'd or post-hoc plotted without
    re-running anything.

    Periodic full snapshots can also be written to ``checkpoint_dir`` every
    ``checkpoint_every_steps`` optimizer steps so a multi-day run survives
    crashes / power loss / accidental kills.
    """

    def __init__(
        self,
        epochs: int = 10,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 5000,
        samples_per_variety: int = 4,
        grad_accum: int = 4,
        grad_clip: float = 1.0,
        patience: int = 3,
        seed: int = 42,
        device: Optional[str] = None,
        log_every_steps: int = 50,
        checkpoint_every_steps: int = 0,
        checkpoint_dir: str | Path | None = None,
        step_log_path: str | Path | None = None,
        keep_last_checkpoints: int = 3,
        queue_size: int = 4096,
        moco_momentum: float = 0.999,
        moco_start_epoch: int = 4,
        queue_ramp_steps: int = 5000,
        supcon_temperature: float = 0.07,
        pretrain_epochs: int = 2,
        proj_lr_mult: float = 10.0,
        center_warmup_steps: int = 2000,
        resume_from: str | Path | None = None,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = max(1, int(warmup_steps))
        self.samples_per_variety = samples_per_variety
        self.grad_accum = grad_accum
        self.grad_clip = grad_clip
        self.patience = patience
        self.seed = seed
        self.device = torch.device(device) if device else _detect_device()
        self.loss_history: list[dict[str, float]] = []

        self.log_every_steps = max(1, int(log_every_steps))
        self.checkpoint_every_steps = max(0, int(checkpoint_every_steps))
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.step_log_path = Path(step_log_path) if step_log_path else None
        self.keep_last_checkpoints = max(0, int(keep_last_checkpoints))

        self.queue_size = max(0, int(queue_size))
        self.moco_momentum = float(moco_momentum)
        self.moco_start_epoch = max(1, int(moco_start_epoch))
        self.queue_ramp_steps = max(1, int(queue_ramp_steps))
        self.supcon_temperature = float(supcon_temperature)
        self.pretrain_epochs = max(0, int(pretrain_epochs))
        self.proj_lr_mult = float(proj_lr_mult)
        self.center_warmup_steps = max(0, int(center_warmup_steps))
        self.resume_from = Path(resume_from) if resume_from else None
        # Derived batch size: k × n_varieties; filled in at train() once we
        # know the dataset's variety count.
        self.batch_size: int = 0

    def train(
        self,
        model,
        dataset,
        collator=None,
    ) -> None:
        """Train the transformer model in-place.

        v2 multi-task objective: MLM + dialect CE + SupCon (+ MoCo + DCL).
        Batches are balanced per variety by ``BalancedVarietySampler`` so
        every batch carries real same-label positives for SupCon. Best
        checkpoint (by epoch-mean total loss) is restored at the end.
        Inner-loop telemetry is emitted every ``self.log_every_steps``
        optimizer steps; see ``__init__``.
        """
        from eigen3.dataset import BalancedVarietySampler
        from eigen3.loss import DialectMultiTaskLoss

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        model = model.to(self.device)
        model.train()

        n_varieties = len(dataset.variety_names)
        # Resolve the derived batch size now that we know n_varieties.
        self.batch_size = self.samples_per_variety * n_varieties

        # Build loss (v3: τ=0.07, center loss, front-loaded w_con)
        criterion = DialectMultiTaskLoss(
            n_varieties=n_varieties,
            proj_dim=model.proj_dim,
            temperature=self.supcon_temperature,
        )
        criterion = criterion.to(self.device)

        # MLM head: share BETO's word embeddings for MLM prediction
        mlm_head = self._build_mlm_head(model).to(self.device)

        # Optimizer: separate param groups (projection gets 10x LR)
        proj_params = list(model.projection.parameters())
        proj_param_ids = {id(p) for p in proj_params}
        other_model_params = [p for p in model.parameters()
                              if p.requires_grad and id(p) not in proj_param_ids]
        mlm_params = [p for p in mlm_head.parameters() if p.requires_grad]
        criterion_params = [p for p in criterion.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": other_model_params + mlm_params + criterion_params,
             "lr": self.lr},
            {"params": proj_params, "lr": self.lr * self.proj_lr_mult},
        ], weight_decay=self.weight_decay)
        trainable_params = (other_model_params + mlm_params
                            + criterion_params + proj_params)

        # Balanced per-variety batch sampler — SupCon needs real positives
        variety_ids_np = np.array(
            [s[1] for s in dataset._samples], dtype=np.int64,
        )
        sampler = BalancedVarietySampler(
            variety_ids=variety_ids_np,
            samples_per_variety=self.samples_per_variety,
            n_varieties=n_varieties,
            seed=self.seed,
        )

        # DataLoader: batch_sampler path (no shuffle / batch_size here)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        # MoCo momentum encoder + queue
        from eigen3.moco import MoCoQueue, MomentumEncoder

        proj_dim = model.proj_dim
        if self.queue_size > 0:
            momentum_encoder = MomentumEncoder(model, momentum=self.moco_momentum)
            momentum_encoder.to(self.device)
            queue = MoCoQueue(self.queue_size, proj_dim, self.device)
        else:
            momentum_encoder = None
            queue = None

        # LR scheduler: linear warmup (5K steps) + cosine decay
        batches_per_epoch = len(loader)
        opt_steps_per_epoch = max(1, batches_per_epoch // self.grad_accum)
        total_steps = opt_steps_per_epoch * self.epochs
        warmup_steps = min(self.warmup_steps, total_steps // 2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: self._lr_schedule(step, warmup_steps, total_steps),
        )

        # Resume from checkpoint if requested
        start_epoch = 1
        start_batch_idx = 0
        if self.resume_from is not None:
            logger.info("Loading checkpoint: %s", self.resume_from)
            ckpt = torch.load(self.resume_from, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            mlm_head.load_state_dict(ckpt["mlm_head"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            self.loss_history = ckpt.get("loss_history", [])
            start_epoch = ckpt["epoch"]
            resumed_global_step = ckpt["global_step"]
            # Which batch_idx to skip to within the resumed epoch
            steps_into_epoch = resumed_global_step - (start_epoch - 1) * opt_steps_per_epoch
            start_batch_idx = steps_into_epoch * self.grad_accum
            # Restore MoCo state (graceful fallback for old checkpoints)
            if momentum_encoder is not None:
                mom_state = ckpt.get("momentum_encoder")
                if mom_state is not None:
                    momentum_encoder.load_state_dict(mom_state)
                    logger.info("Restored momentum encoder from checkpoint")
                else:
                    # Old checkpoint without MoCo — re-init from current model
                    logger.info("No momentum_encoder in checkpoint; initializing from model")
                    momentum_encoder = MomentumEncoder(model, momentum=self.moco_momentum)
                    momentum_encoder.to(self.device)
            if queue is not None:
                q_state = ckpt.get("queue")
                if q_state is not None:
                    queue.load_state_dict(q_state)
                    logger.info("Restored MoCo queue from checkpoint (filled=%d)", queue.filled)
                else:
                    logger.info("No queue state in checkpoint; starting fresh queue")
            logger.info(
                "Resumed: epoch=%d, global_step=%d, skipping %d batches in epoch %d",
                start_epoch, resumed_global_step, start_batch_idx, start_epoch,
            )
            del ckpt

        # Open the JSONL step log (line-buffered append)
        step_log_fp = None
        if self.step_log_path is not None:
            self.step_log_path.parent.mkdir(parents=True, exist_ok=True)
            step_log_fp = open(self.step_log_path, "a", buffering=1, encoding="utf-8")

        # One-time training-config banner
        n_samples = len(dataset)
        eff_batch = self.batch_size * self.grad_accum
        logger.info(
            "Trainer config: device=%s  samples=%d  n_var=%d  k=%d  batch=%d  "
            "grad_accum=%d  eff_batch=%d  batches/epoch=%d  opt_steps/epoch=%d  "
            "total_opt_steps=%d  warmup_steps=%d  log_every=%d  ckpt_every=%d  "
            "queue_size=%d  moco_m=%.4f  moco_start_ep=%d  queue_ramp=%d  supcon_temp=%.3f  "
            "proj_dim=%d  pretrain_ep=%d  proj_lr_mult=%.1f  center_warmup=%d",
            self.device.type, n_samples, n_varieties, self.samples_per_variety,
            self.batch_size, self.grad_accum, eff_batch,
            batches_per_epoch, opt_steps_per_epoch, total_steps,
            warmup_steps, self.log_every_steps, self.checkpoint_every_steps,
            self.queue_size, self.moco_momentum, self.moco_start_epoch,
            self.queue_ramp_steps, self.supcon_temperature, proj_dim,
            self.pretrain_epochs, self.proj_lr_mult, self.center_warmup_steps,
        )

        # Training loop
        best_loss = float("inf")
        best_state: dict | None = None
        no_improve = 0
        global_step = resumed_global_step if self.resume_from is not None else 0
        use_mps = self.device.type == "mps"

        # Rolling window for inner-loop telemetry
        window_losses: deque[dict[str, float]] = deque(maxlen=self.log_every_steps * 4)
        window_t0 = time.time()
        window_samples = 0

        run_t0 = time.time()
        saved_checkpoints: list[Path] = []

        # PCA-initialize the first projection layer (skip on resume)
        if self.resume_from is None:
            self._pca_init_projection(model, dataset, collator)

        contrastive_opt_steps = 0
        queue_ramp_start_step: int | None = None

        for epoch in range(start_epoch, self.epochs + 1):
            # Two-phase training: MLM+CLS only for pretrain_epochs
            contrastive_active = epoch > self.pretrain_epochs
            if not contrastive_active:
                criterion.w_mlm = 0.5
                criterion.w_cls = 0.5
                criterion.w_con = 0.0
                criterion.w_center = 0.0
            else:
                contrastive_epoch = epoch - self.pretrain_epochs
                contrastive_total = max(1, self.epochs - self.pretrain_epochs)
                criterion.update_curriculum(contrastive_epoch, contrastive_total)
            epoch_losses: list[dict[str, float]] = []
            optimizer.zero_grad(set_to_none=True)

            if use_mps:
                torch.mps.synchronize()
            epoch_t0 = time.time()

            logger.info(
                "Epoch %d/%d started  (curriculum updated)  lr=%.2e",
                epoch, self.epochs, optimizer.param_groups[0]["lr"],
            )

            for batch_idx, batch in enumerate(loader):
                # Skip already-processed batches when resuming
                if epoch == start_epoch and batch_idx < start_batch_idx:
                    continue

                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                mlm_labels = batch["mlm_labels"].to(self.device)
                variety_ids = batch["variety_ids"].to(self.device)

                # Sanity assertion on the very first batch — catches sampler
                # regressions loudly instead of letting SupCon run on broken
                # label distributions for days.
                if global_step == 0 and batch_idx == 0:
                    bincount = torch.bincount(
                        variety_ids, minlength=n_varieties,
                    )
                    expected = torch.full(
                        (n_varieties,), self.samples_per_variety,
                        dtype=bincount.dtype, device=bincount.device,
                    )
                    assert torch.equal(bincount, expected), (
                        f"BalancedVarietySampler broken: got {bincount.tolist()}, "
                        f"expected {expected.tolist()}"
                    )

                # Forward (3 outputs: hidden, cls, proj_emb — proj_emb is unit-norm)
                hidden_states, cls_logits, proj_emb = model(
                    input_ids, attention_mask, labels=variety_ids,
                )

                # MLM logits from shared head
                mlm_logits = mlm_head(hidden_states)

                # Momentum encoder forward (for MoCo contrastive keys)
                moco_keys = None
                if momentum_encoder is not None and contrastive_active:
                    moco_keys = momentum_encoder.forward(input_ids, attention_mask)

                # MoCo queue view: only after moco_start_epoch, with gradual ramp
                queue_emb = None
                queue_lab = None
                if (
                    queue is not None
                    and epoch >= self.moco_start_epoch
                    and queue.filled > 0
                ):
                    if queue_ramp_start_step is None:
                        queue_ramp_start_step = global_step
                    ramp_progress = min(
                        (global_step - queue_ramp_start_step)
                        / max(self.queue_ramp_steps, 1),
                        1.0,
                    )
                    effective_size = int(
                        128 + ramp_progress * (self.queue_size - 128),
                    )
                    result = queue.get(max_entries=effective_size)
                    if result is not None:
                        queue_emb, queue_lab = result

                # Loss
                loss, loss_dict = criterion(
                    mlm_logits=mlm_logits,
                    mlm_labels=mlm_labels,
                    cls_logits=cls_logits,
                    cls_labels=variety_ids,
                    proj_emb=proj_emb,
                    variety_ids=variety_ids,
                    moco_keys=moco_keys,
                    queue_emb=queue_emb,
                    queue_labels=queue_lab,
                )

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.grad_accum
                scaled_loss.backward()

                # Enqueue momentum keys (not training embeddings) AFTER backward
                if queue is not None and moco_keys is not None:
                    queue.enqueue(moco_keys.detach(), variety_ids.detach())

                epoch_losses.append(loss_dict)
                window_losses.append(loss_dict)
                window_samples += int(input_ids.shape[0])

                if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    # MoCo EMA update (after optimizer.step updates training encoder)
                    if momentum_encoder is not None:
                        momentum_encoder.update(model)

                    # Center loss warmup: active for first N contrastive opt steps
                    if contrastive_active:
                        contrastive_opt_steps += 1
                        if contrastive_opt_steps <= self.center_warmup_steps:
                            criterion.w_center = 0.1
                        else:
                            criterion.w_center = 0.0

                    # ----------------------------------------------------
                    # Inner-loop telemetry
                    # ----------------------------------------------------
                    if global_step % self.log_every_steps == 0:
                        now = time.time()
                        window_dt = max(now - window_t0, 1e-6)
                        samples_per_sec = window_samples / window_dt
                        steps_per_sec = self.log_every_steps / window_dt

                        # Rolling means over window
                        win_avg = {
                            k: float(np.mean([d.get(k, 0.0) for d in window_losses]))
                            for k in window_losses[0]
                        }

                        lr_now = optimizer.param_groups[0]["lr"]
                        epoch_elapsed = now - epoch_t0
                        run_elapsed = now - run_t0

                        steps_done_this_epoch = global_step - (epoch - 1) * opt_steps_per_epoch
                        steps_left_this_epoch = max(0, opt_steps_per_epoch - steps_done_this_epoch)
                        steps_left_total = max(0, total_steps - global_step)

                        sec_per_step = window_dt / max(self.log_every_steps, 1)
                        eta_epoch = steps_left_this_epoch * sec_per_step
                        eta_total = steps_left_total * sec_per_step

                        rss_mb = _process_rss_mb()
                        mps_mb = _mps_memory_mb()
                        mps_str = f" mps={mps_mb:.0f}MB" if mps_mb is not None else ""

                        logger.info(
                            "ep %d/%d step %d/%d (%5.2f%%)  loss=%.4f "
                            "(mlm=%.4f cls=%.4f con=%.4f ctr=%.4f)  "
                            "lr=%.2e  %.1f sm/s  %.2f st/s  ep %s/%s  "
                            "total %s/%s  rss=%.0fMB%s",
                            epoch, self.epochs,
                            steps_done_this_epoch, opt_steps_per_epoch,
                            100.0 * global_step / max(total_steps, 1),
                            win_avg.get("total", 0.0),
                            win_avg.get("mlm", 0.0),
                            win_avg.get("cls", 0.0),
                            win_avg.get("contrastive", 0.0),
                            win_avg.get("center", 0.0),
                            lr_now, samples_per_sec, steps_per_sec,
                            _format_duration(epoch_elapsed), _format_duration(epoch_elapsed + eta_epoch),
                            _format_duration(run_elapsed), _format_duration(run_elapsed + eta_total),
                            rss_mb, mps_str,
                        )

                        if step_log_fp is not None:
                            record = {
                                "ts": now,
                                "epoch": epoch,
                                "epoch_step": steps_done_this_epoch,
                                "epoch_steps_total": opt_steps_per_epoch,
                                "global_step": global_step,
                                "global_steps_total": total_steps,
                                "progress_pct": 100.0 * global_step / max(total_steps, 1),
                                "lr": lr_now,
                                "loss": win_avg,
                                "samples_per_sec": samples_per_sec,
                                "steps_per_sec": steps_per_sec,
                                "epoch_elapsed_s": epoch_elapsed,
                                "epoch_eta_s": eta_epoch,
                                "run_elapsed_s": run_elapsed,
                                "run_eta_s": eta_total,
                                "rss_mb": rss_mb,
                                "mps_mb": mps_mb,
                            }
                            step_log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

                        window_t0 = now
                        window_samples = 0

                    # ----------------------------------------------------
                    # Periodic checkpoint
                    # ----------------------------------------------------
                    if (
                        self.checkpoint_every_steps > 0
                        and self.checkpoint_dir is not None
                        and global_step % self.checkpoint_every_steps == 0
                    ):
                        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        ckpt_path = self.checkpoint_dir / f"step_{global_step:08d}.pt"
                        tmp_path = ckpt_path.with_suffix(".pt.tmp")
                        ckpt_data = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "mlm_head": mlm_head.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "loss_history": self.loss_history,
                        }
                        if momentum_encoder is not None:
                            ckpt_data["momentum_encoder"] = momentum_encoder.state_dict()
                        if queue is not None:
                            ckpt_data["queue"] = queue.state_dict()
                        torch.save(ckpt_data, tmp_path)
                        os.replace(tmp_path, ckpt_path)
                        saved_checkpoints.append(ckpt_path)
                        logger.info("Checkpoint saved: %s", ckpt_path.name)

                        # Rotate
                        if self.keep_last_checkpoints > 0:
                            while len(saved_checkpoints) > self.keep_last_checkpoints:
                                old = saved_checkpoints.pop(0)
                                try:
                                    old.unlink()
                                except FileNotFoundError:
                                    pass

            if use_mps:
                torch.mps.synchronize()

            # After first resumed epoch completes, reset skip so next epochs run fully
            start_batch_idx = 0

            # Compute epoch averages
            avg_losses = {
                k: float(np.mean([d.get(k, 0) for d in epoch_losses]))
                for k in epoch_losses[0]
            }
            self.loss_history.append(avg_losses)

            avg_total = avg_losses["total"]
            lr_now = optimizer.param_groups[0]["lr"]
            epoch_dt = time.time() - epoch_t0
            run_elapsed = time.time() - run_t0
            epochs_left = self.epochs - epoch
            run_eta = epochs_left * epoch_dt
            logger.info(
                "Epoch %d/%d  loss=%.4f (mlm=%.4f cls=%.4f con=%.4f ctr=%.4f)  "
                "lr=%.2e  epoch=%s  run=%s/%s",
                epoch, self.epochs, avg_total,
                avg_losses.get("mlm", 0), avg_losses.get("cls", 0),
                avg_losses.get("contrastive", 0), avg_losses.get("center", 0),
                lr_now,
                _format_duration(epoch_dt),
                _format_duration(run_elapsed),
                _format_duration(run_elapsed + run_eta),
            )

            # Checkpointing
            if avg_total < best_loss:
                best_loss = avg_total
                best_state = {
                    "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "mlm_head": {k: v.cpu().clone() for k, v in mlm_head.state_dict().items()},
                }
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.patience)
                break

        # Restore best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state["model"])
            logger.info("Restored best model (loss=%.4f)", best_loss)

        if step_log_fp is not None:
            step_log_fp.close()

        model.eval()

    def _build_mlm_head(self, model) -> torch.nn.Module:
        """Build MLM prediction head reusing BETO's word embeddings."""
        vocab_size = len(model.tokenizer)
        hidden_dim = model.hidden_dim
        return torch.nn.Linear(hidden_dim, vocab_size, bias=False)

    def _pca_init_projection(
        self, model, dataset, collator, n_samples: int = 2048,
    ) -> None:
        """Initialize first projection layer with PCA of pooled hidden states."""
        model.eval()
        all_pooled: list[torch.Tensor] = []
        loader = DataLoader(
            dataset, batch_size=64, shuffle=True,
            collate_fn=collator, num_workers=0,
        )
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = model._base_model(
                    input_ids=input_ids, attention_mask=attention_mask,
                )
                pooled = model.attention_pool(
                    outputs.last_hidden_state, attention_mask,
                )
                all_pooled.append(pooled.cpu())
                if sum(p.shape[0] for p in all_pooled) >= n_samples:
                    break

        pooled_cat = torch.cat(all_pooled, dim=0)[:n_samples]
        mean = pooled_cat.mean(dim=0)
        centered = pooled_cat - mean

        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        proj_dim = model.proj_dim
        pca_weight = Vh[:proj_dim]  # (proj_dim, hidden_dim)

        with torch.no_grad():
            model.projection[0].weight.copy_(
                pca_weight.to(model.projection[0].weight.device),
            )
            model.projection[0].bias.zero_()

        model.train()
        logger.info(
            "PCA-initialized projection[0] from %d samples", pooled_cat.shape[0],
        )

    @staticmethod
    def _lr_schedule(step: int, warmup: int, total: int) -> float:
        """Linear warmup + cosine decay."""
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def save(
        self,
        model,
        output_dir: str | Path,
        meta_extra: dict | None = None,
    ) -> None:
        """Save model state, loss history, and metadata."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save model state (LoRA adapters + heads)
        torch.save(model.state_dict(), out / "model_state.pt")

        # Save loss history
        (out / "loss_history.json").write_text(
            json.dumps(self.loss_history, indent=2), encoding="utf-8",
        )

        # Save metadata
        meta: dict[str, Any] = {
            "loss_version": "v3_supcon_moco_dcl_arcface_center",
            "epochs": self.epochs,
            "lr": self.lr,
            "samples_per_variety": self.samples_per_variety,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "seed": self.seed,
            "model_name": model.model_name,
            "proj_dim": model.proj_dim,
            "queue_size": self.queue_size,
            "moco_momentum": self.moco_momentum,
            "moco_start_epoch": self.moco_start_epoch,
            "queue_ramp_steps": self.queue_ramp_steps,
            "supcon_temperature": self.supcon_temperature,
            "pretrain_epochs": self.pretrain_epochs,
            "proj_lr_mult": self.proj_lr_mult,
            "center_warmup_steps": self.center_warmup_steps,
            "warmup_steps": self.warmup_steps,
            "lora_targets": [
                "attention.self.query", "attention.self.key",
                "attention.self.value", "attention.output.dense",
                "intermediate.dense", "output.dense",
            ],
            "n_epochs_trained": len(self.loss_history),
            "final_loss": self.loss_history[-1]["total"] if self.loss_history else None,
        }
        if meta_extra:
            meta.update(meta_extra)
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info("Saved transformer artifacts to %s", out)


# ---------------------------------------------------------------------------
# Legacy DCL trainer (kept for backward compatibility)
# ---------------------------------------------------------------------------

class DCLTrainer:
    """Trains DCL embeddings with cosine annealing and best-model checkpointing."""

    def __init__(
        self,
        embedding_dim: int = 100,
        epochs: int = 30,
        lr: float = 0.001,
        lr_min: float = 1e-5,
        lambda_anchor: float = 0.05,
        batch_size: int = 8192,
        grad_clip: float = 1.0,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.lr_min = lr_min
        self.lambda_anchor = lambda_anchor
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.seed = seed
        self.device = torch.device(device) if device else _detect_device()

        self.model = None
        self.loss_history: list[float] = []

    def train(self, dataset) -> dict[str, np.ndarray]:
        """Train on a SubwordDCLDataset. Returns per-variety embedding dicts."""
        from eigen3.loss import DialectContrastiveLoss
        from eigen3.model import DCLModel

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        vocab_size = dataset.vocab_size
        n_varieties = len(dataset.variety_names)

        self.model = DCLModel(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            n_varieties=n_varieties,
        ).to(self.device)

        criterion = DialectContrastiveLoss(self.lambda_anchor).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr_min,
        )

        data_tensor = torch.from_numpy(dataset._data)
        n_total = data_tensor.shape[0]
        bs = self.batch_size

        self.loss_history = []
        self.model.train()
        best_loss = float("inf")
        best_state = None
        use_mps = self.device.type == "mps"

        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(n_total)
            epoch_loss = 0.0
            n_batches = 0

            if use_mps:
                torch.mps.synchronize()
            t0 = time.time()

            for start in range(0, n_total, bs):
                idx = perm[start:start + bs]
                chunk = data_tensor[idx]

                word_idx = chunk[:, 0].to(self.device)
                ctx_same = chunk[:, 1].to(self.device)
                ctx_other = chunk[:, 2].to(self.device)
                variety_a = chunk[:, 3].to(self.device)
                variety_b = chunk[:, 4].to(self.device)
                is_reg = chunk[:, 5].to(self.device).bool()

                w_a, c_a, c_b, w_b = self.model(
                    word_idx, ctx_same, ctx_other, variety_a, variety_b,
                )
                loss = criterion(w_a, c_a, c_b, w_b, is_reg)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if use_mps:
                torch.mps.synchronize()
            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)

            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d  loss=%.6f  lr=%.6f  (%.1fs)",
                epoch, self.epochs, avg_loss, lr_now, time.time() - t0,
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # Restore best checkpoint
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("Restored best model (loss=%.6f)", best_loss)

        # Extract embeddings
        self.model.eval()
        result: dict[str, np.ndarray] = {}
        for name in dataset.variety_names:
            v_idx = dataset.variety_to_idx[name]
            result[name] = self.model.extract_variety_embeddings(v_idx).astype(np.float32)

        return result

    def save(self, embeddings: dict[str, np.ndarray], output_dir: str | Path, meta_extra: dict = None):
        """Save embeddings, loss history, and metadata."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for name, emb in embeddings.items():
            np.save(str(out / f"{name}_subword.npy"), emb)

        meta: dict[str, Any] = {
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "lambda_anchor": self.lambda_anchor,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
        }
        if meta_extra:
            meta.update(meta_extra)
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (out / "loss_history.json").write_text(json.dumps(self.loss_history), encoding="utf-8")

        logger.info("Saved DCL artifacts to %s", out)
