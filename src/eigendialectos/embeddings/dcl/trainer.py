"""End-to-end DCL training orchestrator.

Wires together :class:`DCLDataset`, :class:`DCLEmbeddingModel`, and
:class:`DialectContrastiveLoss` into a complete training loop.  Produces
per-variety embedding matrices saved as ``.npy`` files with an
accompanying ``vocab.json``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from eigendialectos.embeddings.dcl.dataset import DCLDataset
from eigendialectos.embeddings.dcl.loss import DialectContrastiveLoss
from eigendialectos.embeddings.dcl.model import DCLEmbeddingModel
from eigendialectos.embeddings.dcl.regionalisms import ALL_REGIONALISMS

logger = logging.getLogger(__name__)


class DCLTrainer:
    """Trains DCL embeddings end-to-end.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of each embedding vector.
    epochs:
        Number of training epochs.
    lr:
        Learning rate for Adam optimiser.
    lambda_anchor:
        Weight for the anchor regularisation term in DCL loss.
    batch_size:
        Mini-batch size for the DataLoader.
    window_size:
        Skip-gram context window radius.
    neg_samples:
        Number of cross-variety negative samples per positive pair.
    min_count:
        Minimum token frequency for vocabulary inclusion.
    seed:
        Random seed for reproducibility.
    device:
        PyTorch device string (``"cpu"`` or ``"cuda"``).  Defaults to
        CPU; the caller can pass ``"cuda"`` if a GPU is available.
    num_workers:
        Number of DataLoader worker processes.
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        epochs: int = 20,
        lr: float = 0.001,
        lambda_anchor: float = 0.1,
        batch_size: int = 512,
        window_size: int = 5,
        neg_samples: int = 5,
        min_count: int = 2,
        seed: int = 42,
        device: str = "cpu",
        num_workers: int = 0,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.lambda_anchor = lambda_anchor
        self.batch_size = batch_size
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.min_count = min_count
        self.seed = seed
        self.device = torch.device(device)
        self.num_workers = num_workers

        # Populated during training
        self._model: DCLEmbeddingModel | None = None
        self._dataset: DCLDataset | None = None
        self._loss_history: list[float] = []

    def train(
        self,
        corpus_by_variety: dict[str, list[str]],
        regionalism_set: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Train DCL embeddings and return per-variety matrices.

        Parameters
        ----------
        corpus_by_variety:
            Mapping from variety name (e.g. ``"ES_RIO"``) to a list of
            text documents (each a whitespace-separated token string).
        regionalism_set:
            Optional custom regionalism set.  Defaults to
            :data:`ALL_REGIONALISMS`.

        Returns
        -------
        dict[str, np.ndarray]
            ``{variety_name: (vocab_size, embedding_dim)}`` embedding
            matrices as float32 numpy arrays.
        """
        if regionalism_set is None:
            regionalism_set = ALL_REGIONALISMS

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # ------------------------------------------------------------------
        # 1. Build dataset
        # ------------------------------------------------------------------
        logger.info("Building DCL dataset (window=%d, neg=%d, min_count=%d) ...",
                     self.window_size, self.neg_samples, self.min_count)
        t0 = time.time()

        self._dataset = DCLDataset(
            corpus_by_variety=corpus_by_variety,
            window_size=self.window_size,
            neg_samples=self.neg_samples,
            regionalism_set=regionalism_set,
            min_count=self.min_count,
            seed=self.seed,
        )

        vocab_size = self._dataset.vocab_size
        n_varieties = len(self._dataset.variety_names)
        logger.info(
            "Dataset ready: vocab=%d, varieties=%d, samples=%d (%.1fs)",
            vocab_size, n_varieties, len(self._dataset), time.time() - t0,
        )

        if vocab_size == 0:
            raise ValueError(
                "Empty vocabulary after filtering.  Check corpus content "
                "and min_count setting."
            )

        # ------------------------------------------------------------------
        # 2. Create model and loss
        # ------------------------------------------------------------------
        self._model = DCLEmbeddingModel(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            n_varieties=n_varieties,
        ).to(self.device)

        criterion = DialectContrastiveLoss(
            lambda_anchor=self.lambda_anchor,
        ).to(self.device)

        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        # ------------------------------------------------------------------
        # 3. DataLoader
        # ------------------------------------------------------------------
        loader = DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type != "cpu"),
            drop_last=False,
        )

        # ------------------------------------------------------------------
        # 4. Training loop
        # ------------------------------------------------------------------
        logger.info(
            "Starting DCL training: dim=%d, epochs=%d, lr=%.4f, "
            "lambda=%.3f, batch=%d",
            self.embedding_dim, self.epochs, self.lr,
            self.lambda_anchor, self.batch_size,
        )

        self._loss_history = []
        self._model.train()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            t_epoch = time.time()

            for batch in loader:
                (word_idx, ctx_idx_same, ctx_idx_other,
                 variety_a, variety_b, is_regionalism) = batch

                # Move to device
                word_idx = word_idx.to(self.device)
                ctx_idx_same = ctx_idx_same.to(self.device)
                ctx_idx_other = ctx_idx_other.to(self.device)
                variety_a = variety_a.to(self.device)
                variety_b = variety_b.to(self.device)
                is_regionalism = is_regionalism.to(self.device)

                # Forward
                word_emb_a, ctx_emb_a, ctx_emb_b, word_emb_b = self._model(
                    word_idx, ctx_idx_same, ctx_idx_other,
                    variety_a, variety_b,
                )

                loss = criterion(
                    word_emb_a, ctx_emb_a, ctx_emb_b,
                    word_emb_b, is_regionalism,
                    variety_a=variety_a, variety_b=variety_b,
                )

                # Backward
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self._loss_history.append(avg_loss)
            elapsed = time.time() - t_epoch

            logger.info(
                "Epoch %d/%d  loss=%.6f  (%.1fs, %d batches)",
                epoch, self.epochs, avg_loss, elapsed, n_batches,
            )

        # ------------------------------------------------------------------
        # 5. Extract final embeddings as numpy arrays
        # ------------------------------------------------------------------
        self._model.eval()
        result: dict[str, np.ndarray] = {}
        for variety_name in self._dataset.variety_names:
            v_idx = self._dataset.variety_to_idx[variety_name]
            weight = self._model.get_word_embeddings(v_idx)
            result[variety_name] = weight.cpu().numpy().astype(np.float32)

        logger.info("Training complete. Final loss: %.6f", self._loss_history[-1])
        return result

    def save(self, output_dir: Path) -> None:
        """Save trained embeddings and vocabulary to disk.

        Creates one ``.npy`` file per variety plus a ``vocab.json``
        file, a ``meta.json`` with training config, and a
        ``loss_history.json`` for diagnostics.

        Parameters
        ----------
        output_dir:
            Directory to write files into (created if needed).
        """
        if self._model is None or self._dataset is None:
            raise RuntimeError("No trained model.  Call train() first.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Vocabulary
        vocab_path = out / "vocab.json"
        vocab_path.write_text(
            json.dumps(self._dataset.get_vocab(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved vocabulary (%d words) to %s",
                     self._dataset.vocab_size, vocab_path)

        # Per-variety embedding matrices
        self._model.eval()
        for variety_name in self._dataset.variety_names:
            v_idx = self._dataset.variety_to_idx[variety_name]
            weight = self._model.get_word_embeddings(v_idx)
            npy_path = out / f"{variety_name}.npy"
            np.save(str(npy_path), weight.cpu().numpy().astype(np.float32))
            logger.info("Saved %s embeddings to %s", variety_name, npy_path)

        # Training metadata
        meta: dict[str, Any] = {
            "embedding_dim": self.embedding_dim,
            "vocab_size": self._dataset.vocab_size,
            "n_varieties": len(self._dataset.variety_names),
            "varieties": self._dataset.variety_names,
            "epochs": self.epochs,
            "lr": self.lr,
            "lambda_anchor": self.lambda_anchor,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "neg_samples": self.neg_samples,
            "min_count": self.min_count,
            "seed": self.seed,
            "final_loss": self._loss_history[-1] if self._loss_history else None,
        }
        meta_path = out / "meta.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Loss history
        loss_path = out / "loss_history.json"
        loss_path.write_text(
            json.dumps(self._loss_history),
            encoding="utf-8",
        )

        logger.info("All DCL artifacts saved to %s", out)

    @property
    def loss_history(self) -> list[float]:
        """Per-epoch average loss values from the most recent training run."""
        return list(self._loss_history)

    @property
    def model(self) -> DCLEmbeddingModel | None:
        """The underlying PyTorch model (``None`` before training)."""
        return self._model

    @property
    def dataset(self) -> DCLDataset | None:
        """The dataset used in the most recent training run."""
        return self._dataset


class SubwordDCLTrainer(DCLTrainer):
    """DCL trainer optimized for MPS (Apple Silicon GPU) throughput.

    Uses :class:`SubwordDCLDataset` with pre-materialized samples
    converted to a ``TensorDataset`` for maximum DataLoader speed.

    Parameters
    ----------
    tokenizer:
        A trained :class:`SharedSubwordTokenizer`.
    **kwargs:
        Forwarded to :class:`DCLTrainer`.  ``min_count`` is ignored.
    """

    def __init__(self, tokenizer, grad_clip: float = 1.0, **kwargs) -> None:
        kwargs.pop("min_count", None)
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.grad_clip = grad_clip

    def train(
        self,
        corpus_by_variety: dict[str, list[str]],
        regionalism_set: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Train subword-level DCL embeddings on MPS/CUDA/CPU."""
        from eigendialectos.embeddings.dcl.subword_dataset import SubwordDCLDataset

        if regionalism_set is None:
            regionalism_set = ALL_REGIONALISMS

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Build subword dataset (pre-materializes all samples)
        logger.info(
            "Building SubwordDCL dataset (window=%d, neg=%d) ...",
            self.window_size, self.neg_samples,
        )
        t0 = time.time()

        dataset = SubwordDCLDataset(
            corpus_by_variety=corpus_by_variety,
            tokenizer=self.tokenizer,
            window_size=self.window_size,
            neg_samples=self.neg_samples,
            regionalism_set=regionalism_set,
            seed=self.seed,
        )

        vocab_size = dataset.get_vocab_size()
        n_varieties = len(dataset.variety_names)
        logger.info(
            "SubwordDCL dataset ready: vocab=%d, varieties=%d, samples=%d (%.1fs)",
            vocab_size, n_varieties, len(dataset), time.time() - t0,
        )

        if vocab_size == 0:
            raise ValueError("Empty BPE vocabulary.")

        # Convert pre-materialized data to a single contiguous tensor.
        # Training loop uses direct tensor slicing (no DataLoader overhead).
        data_tensor = torch.from_numpy(dataset._data)  # (N, 6) int64
        n_total = data_tensor.shape[0]

        # Create model and loss
        self._model = DCLEmbeddingModel(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            n_varieties=n_varieties,
        ).to(self.device)

        criterion = DialectContrastiveLoss(
            lambda_anchor=self.lambda_anchor,
        ).to(self.device)

        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        # Cosine annealing: decay LR from self.lr to 1e-5 over all epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.epochs, eta_min=1e-5,
        )

        bs = self.batch_size
        n_batches_per_epoch = (n_total + bs - 1) // bs

        logger.info(
            "Starting SubwordDCL training: dim=%d, epochs=%d, lr=%.4f→1e-5, "
            "lambda=%.3f, batch=%d, device=%s, batches/epoch=%d, grad_clip=%.1f",
            self.embedding_dim, self.epochs, self.lr,
            self.lambda_anchor, bs, self.device, n_batches_per_epoch,
            self.grad_clip,
        )

        self._loss_history = []
        self._model.train()
        use_mps = self.device.type == "mps"

        # Best-model checkpointing
        best_loss = float("inf")
        best_state = None

        for epoch in range(1, self.epochs + 1):
            # Shuffle via index permutation (no 8.7GB copy)
            perm = torch.randperm(n_total)

            epoch_loss = 0.0
            n_batches = 0
            if use_mps:
                torch.mps.synchronize()
            t_epoch = time.time()

            for start in range(0, n_total, bs):
                idx = perm[start:start + bs]
                chunk = data_tensor[idx]

                word_idx = chunk[:, 0].to(self.device, non_blocking=True)
                ctx_idx_same = chunk[:, 1].to(self.device, non_blocking=True)
                ctx_idx_other = chunk[:, 2].to(self.device, non_blocking=True)
                variety_a = chunk[:, 3].to(self.device, non_blocking=True)
                variety_b = chunk[:, 4].to(self.device, non_blocking=True)
                is_regionalism = chunk[:, 5].to(self.device, non_blocking=True).bool()

                word_emb_a, ctx_emb_a, ctx_emb_b, word_emb_b = self._model(
                    word_idx, ctx_idx_same, ctx_idx_other,
                    variety_a, variety_b,
                )

                loss = criterion(
                    word_emb_a, ctx_emb_a, ctx_emb_b,
                    word_emb_b, is_regionalism,
                    variety_a=variety_a, variety_b=variety_b,
                )

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), self.grad_clip,
                )
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if use_mps:
                torch.mps.synchronize()
            avg_loss = epoch_loss / max(n_batches, 1)
            self._loss_history.append(avg_loss)
            elapsed = time.time() - t_epoch

            current_lr = optimiser.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d  loss=%.6f  lr=%.6f  (%.1fs, %d batches)",
                epoch, self.epochs, avg_loss, current_lr, elapsed, n_batches,
            )

            # Checkpoint best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self._model.state_dict().items()
                }
                logger.info("  ↳ New best loss: %.6f (epoch %d)", best_loss, epoch)

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict(best_state)
            logger.info(
                "Restored best model (loss=%.6f) from training run",
                best_loss,
            )

        # Extract per-variety subword embeddings
        self._model.eval()
        result: dict[str, np.ndarray] = {}
        for variety_name in dataset.variety_names:
            v_idx = dataset.variety_to_idx[variety_name]
            weight = self._model.get_word_embeddings(v_idx)
            result[variety_name] = weight.cpu().numpy().astype(np.float32)

        logger.info("SubwordDCL training complete. Final loss: %.6f", self._loss_history[-1])

        self._subword_dataset = dataset
        return result

    def save(self, output_dir: Path) -> None:
        """Save subword embeddings and metadata."""
        if self._model is None:
            raise RuntimeError("No trained model. Call train() first.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        dataset = self._subword_dataset

        self._model.eval()
        for variety_name in dataset.variety_names:
            v_idx = dataset.variety_to_idx[variety_name]
            weight = self._model.get_word_embeddings(v_idx)
            npy_path = out / f"{variety_name}_subword.npy"
            np.save(str(npy_path), weight.cpu().numpy().astype(np.float32))
            logger.info("Saved %s subword embeddings to %s", variety_name, npy_path)

        meta = {
            "embedding_dim": self.embedding_dim,
            "bpe_vocab_size": dataset.get_vocab_size(),
            "n_varieties": len(dataset.variety_names),
            "varieties": dataset.variety_names,
            "epochs": self.epochs,
            "lr": self.lr,
            "lambda_anchor": self.lambda_anchor,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "neg_samples": self.neg_samples,
            "seed": self.seed,
            "final_loss": self._loss_history[-1] if self._loss_history else None,
        }
        meta_path = out / "meta.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        loss_path = out / "loss_history.json"
        loss_path.write_text(json.dumps(self._loss_history), encoding="utf-8")

        logger.info("SubwordDCL artifacts saved to %s", out)
