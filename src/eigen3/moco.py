"""Momentum Contrast (MoCo) for dialect contrastive learning.

Provides a momentum encoder (EMA of the training model) and a FIFO queue
for storing momentum-encoded embeddings. This replaces the broken XBM
(cross-batch memory) which stored stale training-encoder embeddings that
are incompatible with softmax-family losses (SupCon/DCL).

Reference: He et al. "Momentum Contrast for Unsupervised Visual
Representation Learning" (CVPR 2020).
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class MomentumEncoder:
    """EMA copy of DialectTransformer for MoCo contrastive learning.

    Maintains a slowly-updating copy of the training model so that queue
    entries remain mutually consistent. Only trainable parameters (LoRA +
    attention_pool + projection) are EMA-updated; frozen BETO weights are
    identical in both encoders by construction.

    Parameters
    ----------
    model : DialectTransformer
        The training encoder to copy.
    momentum : float
        EMA coefficient. Higher = slower updates = more consistent queue.
        Default 0.999 (standard MoCo value).
    """

    def __init__(self, model: torch.nn.Module, momentum: float = 0.999) -> None:
        self.momentum = momentum
        self.encoder = copy.deepcopy(model)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        # Cache trainable parameter pairs for fast EMA loop
        self._trainable_pairs: list[tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
        for (name, p_train), (_, p_mom) in zip(
            model.named_parameters(), self.encoder.named_parameters(),
        ):
            if p_train.requires_grad:
                self._trainable_pairs.append((p_train, p_mom))

        # Cache buffer pairs for BN running stats EMA
        self._buffer_pairs: list[tuple[torch.Tensor, torch.Tensor, bool]] = []
        for (name, b_train), (_, b_mom) in zip(
            model.named_buffers(), self.encoder.named_buffers(),
        ):
            is_float = b_train.is_floating_point()
            self._buffer_pairs.append((b_train, b_mom, is_float))

        logger.info(
            "MomentumEncoder: %d trainable param pairs, %d buffer pairs, m=%.4f",
            len(self._trainable_pairs), len(self._buffer_pairs), momentum,
        )

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """EMA update: p_mom = m * p_mom + (1 - m) * p_train."""
        m = self.momentum
        for p_train, p_mom in self._trainable_pairs:
            p_mom.data.mul_(m).add_(p_train.data, alpha=1.0 - m)
        for b_train, b_mom, is_float in self._buffer_pairs:
            if is_float:
                b_mom.data.mul_(m).add_(b_train.data, alpha=1.0 - m)
            else:
                b_mom.data.copy_(b_train.data)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through momentum encoder.

        Returns only the L2-normalized projection embedding (no cls_logits,
        no hidden_states — those are only needed by the training encoder).
        """
        _, _, proj_emb = self.encoder(input_ids, attention_mask, labels=None)
        return proj_emb

    def to(self, device: torch.device) -> "MomentumEncoder":
        self.encoder = self.encoder.to(device)
        return self

    def state_dict(self) -> dict:
        return self.encoder.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.encoder.load_state_dict(state_dict)


class MoCoQueue:
    """FIFO circular buffer for momentum-encoded embeddings.

    Stores L2-normalized embeddings produced by the momentum encoder along
    with their variety labels. All entries come from the same slowly-updating
    encoder, so they remain mutually consistent (unlike XBM which stored
    stale training-encoder snapshots).

    Parameters
    ----------
    size : int
        Maximum queue capacity.
    dim : int
        Embedding dimension (e.g. 384).
    device : torch.device
        Device for the buffer tensors.
    """

    def __init__(self, size: int, dim: int, device: torch.device) -> None:
        self.size = size
        self.dim = dim
        self.buffer = torch.zeros(size, dim, device=device)
        self.labels = torch.full((size,), -1, dtype=torch.long, device=device)
        self.ptr = 0
        self.filled = 0

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor, key_labels: torch.Tensor) -> None:
        """Add momentum-encoded keys to the queue (FIFO)."""
        B = keys.shape[0]
        if B == 0:
            return
        end = self.ptr + B
        if end <= self.size:
            self.buffer[self.ptr:end] = keys
            self.labels[self.ptr:end] = key_labels
        else:
            first = self.size - self.ptr
            self.buffer[self.ptr:] = keys[:first]
            self.labels[self.ptr:] = key_labels[:first]
            rem = B - first
            self.buffer[:rem] = keys[first:]
            self.labels[:rem] = key_labels[first:]
        self.ptr = (self.ptr + B) % self.size
        self.filled = min(self.filled + B, self.size)

    def get(
        self, max_entries: Optional[int] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Get the most recent queue entries.

        Returns (embeddings, labels) or None if the queue is empty.
        """
        if self.filled == 0:
            return None
        n = self.filled if max_entries is None else min(self.filled, max_entries)
        if self.filled < self.size:
            # Buffer hasn't wrapped yet — simple slice
            return self.buffer[:n], self.labels[:n]
        # Circular buffer wrapped — get most recent N entries
        start = (self.ptr - n) % self.size
        if start + n <= self.size:
            return self.buffer[start:start + n], self.labels[start:start + n]
        first = self.size - start
        emb = torch.cat([self.buffer[start:], self.buffer[:n - first]])
        lab = torch.cat([self.labels[start:], self.labels[:n - first]])
        return emb, lab

    def state_dict(self) -> dict:
        return {
            "buffer": self.buffer.clone(),
            "labels": self.labels.clone(),
            "ptr": self.ptr,
            "filled": self.filled,
        }

    def load_state_dict(self, d: dict) -> None:
        self.buffer.copy_(d["buffer"])
        self.labels.copy_(d["labels"])
        self.ptr = d["ptr"]
        self.filled = d["filled"]
