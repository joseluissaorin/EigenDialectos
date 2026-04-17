"""Dialect-conditioned transformer model: BETO + LoRA + variety tokens.

Architecture (v3):
    BETO (frozen) + LoRA adapters on q,k,v,attn_out,FFN (trainable)
    + 8 variety special tokens [VAR_ES_PEN] ... [VAR_ES_AND_BO]
    + Dialect-aware attention pooling (learned weighted average of all tokens)
    + Projection head: 2-layer MLP + BatchNorm (768->384->384)
    + ArcFace classifier: angular-margin on projected features (384->8)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eigen3.constants import ALL_VARIETIES

logger = logging.getLogger(__name__)

# Default BETO model
DEFAULT_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"


# ---------------------------------------------------------------------------
# Dialect-Aware Attention Pooling — learns which tokens matter most
# for dialect distinction (replaces naive [CLS]-only approach).
# ---------------------------------------------------------------------------

class DialectAttentionPooling(nn.Module):
    """Learned weighted average over all token positions.

    Instead of using only [CLS], this learns attention weights over
    all token positions to create a richer sentence representation.
    Dialect-significant tokens (regionalisms, morphological markers)
    get higher attention weight through end-to-end learning.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : (batch, seq_len, hidden_dim)
        attention_mask : (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns
        -------
        pooled : (batch, hidden_dim) — weighted average of all token positions
        """
        # Compute attention scores
        scores = self.query(hidden_states).squeeze(-1)  # (batch, seq_len)

        # Mask padding positions
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Softmax over sequence
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden_dim)
        return pooled


# ---------------------------------------------------------------------------
# ArcFace angular margin classifier
# ---------------------------------------------------------------------------

class ArcFaceClassifier(nn.Module):
    """ArcFace additive angular margin classifier (Deng et al. 2019).

    Encourages angular separation in the pooled feature space, which
    directly benefits the contrastive projection downstream.
    """

    def __init__(
        self, in_features: int, n_classes: int, s: float = 30.0, m: float = 0.3,
    ) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scaled cosine logits, with angular margin on target class during training."""
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        cosine = x_norm @ w_norm.t()

        if labels is not None and self.training:
            theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
            target_logit = torch.cos(theta + self.m)
            one_hot = torch.zeros_like(cosine, dtype=torch.bool)
            one_hot.scatter_(1, labels.unsqueeze(1), True)
            cosine = torch.where(one_hot, target_logit, cosine)

        return cosine * self.s


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DialectTransformer(nn.Module):
    """BETO + LoRA + dialect-aware attention pooling.

    v3 architecture: 2-layer MLP+BatchNorm projection, ArcFace
    classifier, expanded LoRA (attn+FFN). The projection directly
    produces dialect-discriminative embeddings for SupCon+MoCo+DCL.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        n_varieties: int = 8,
        proj_dim: int = 384,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.n_varieties = n_varieties
        self.proj_dim = proj_dim
        self._hidden_dim: int = 0

        # Load tokenizer and model
        self._tokenizer, self._base_model = self._load_base_model(
            model_name, lora_r, lora_alpha, lora_dropout,
        )

        # Dialect-aware attention pooling (replaces [CLS]-only)
        self.attention_pool = DialectAttentionPooling(self._hidden_dim)

        # Heads
        self.projection = nn.Sequential(
            nn.Linear(self._hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.classifier = ArcFaceClassifier(proj_dim, n_varieties)

    def _load_base_model(
        self,
        model_name: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> tuple:
        """Load BETO, add variety tokens, apply LoRA on attention + FFN."""
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add variety tokens
        variety_tokens = [f"[VAR_{v}]" for v in ALL_VARIETIES]
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": variety_tokens,
        })
        logger.info("Added %d variety tokens to tokenizer", num_added)

        # Store variety token IDs
        self._variety_token_ids = {
            v: tokenizer.convert_tokens_to_ids(f"[VAR_{v}]")
            for v in ALL_VARIETIES
        }

        # Load base model and resize embeddings
        base_model = AutoModel.from_pretrained(model_name)
        base_model.resize_token_embeddings(len(tokenizer))
        self._hidden_dim = base_model.config.hidden_size

        # Freeze all parameters
        for param in base_model.parameters():
            param.requires_grad = False

        # Expanded LoRA coverage: attention q/k/v + attention output + FFN.
        # Dotted-suffix anchors avoid matching the projection/classifier
        # heads that already live in plain float space.
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
                "intermediate.dense",
                "output.dense",
            ],
            bias="none",
        )
        peft_model = get_peft_model(base_model, lora_config)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        logger.info("LoRA: %d trainable / %d total params (%.2f%%)",
                     trainable, total, 100 * trainable / total)

        return tokenizer, peft_model

    @property
    def tokenizer(self):
        """Access the underlying tokenizer."""
        return self._tokenizer

    @property
    def variety_token_ids(self) -> dict[str, int]:
        """Mapping from variety code to special token ID."""
        return self._variety_token_ids

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        labels : optional variety IDs for ArcFace angular margin.

        Returns
        -------
        hidden_states : (batch, seq_len, hidden_dim) — for MLM head
        cls_logits    : (batch, n_varieties)       — ArcFace scaled cosine
        proj_emb      : (batch, proj_dim)          — L2-normalized, SupCon input
        """
        outputs = self._base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Dialect-aware attention pooling (uses ALL tokens, not just [CLS])
        pooled = self.attention_pool(hidden_states, attention_mask)  # (batch, hidden)

        # Projection head — shared feature space for both CLS and contrastive
        projected = self.projection(pooled)  # (batch, proj_dim)

        # ArcFace classification — operates on projected features (same space
        # as contrastive), not raw pooled. This prevents the angular collapse
        # seen when ArcFace operated on the 768-dim pooled representation.
        cls_logits = self.classifier(projected, labels)  # (batch, n_varieties)

        # L2-normalize for SupCon / MoCo contrastive loss
        proj_emb = F.normalize(projected, dim=-1)

        return hidden_states, cls_logits, proj_emb

    def get_projected_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get projected token embeddings for word extraction.

        Returns
        -------
        projected : (batch, seq_len, proj_dim)

        NOTE: this path is *not* L2-normalized. SupCon operates on unit
        vectors during training, but downstream spectral analysis treats
        word-level vectors as Euclidean with norms that encode frequency
        weight, so composition must keep the raw projection.
        """
        with torch.no_grad():
            outputs = self._base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden = outputs.last_hidden_state

        batch, seq_len, _ = hidden.shape
        flat = hidden.reshape(-1, self._hidden_dim)
        projected = self.projection(flat)
        return projected.reshape(batch, seq_len, self.proj_dim)

    def encode_text(
        self,
        text: str,
        variety: str | None = None,
        max_length: int = 256,
    ) -> np.ndarray:
        """Encode a single text to a projected pooled embedding."""
        if variety:
            text = f"[VAR_{variety}] {text}"
        encoding = self._tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        device = next(self.parameters()).device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            outputs = self._base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            pooled = self.attention_pool(outputs.last_hidden_state, attention_mask)
            projected = self.projection(pooled)
        if was_training:
            self.train()

        return projected.squeeze(0).cpu().numpy()

    def count_trainable_parameters(self) -> int:
        """Count trainable parameters (LoRA + heads + pooling)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Legacy skip-gram model (kept for backward compatibility)
# ---------------------------------------------------------------------------

class DCLModel(nn.Module):
    """Single flat nn.Embedding for all varieties (legacy skip-gram).

    Layout: variety v, token t -> global index = v * vocab_size + t.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 100, n_varieties: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_varieties = n_varieties

        total = n_varieties * vocab_size
        self.word_emb = nn.Embedding(total, embedding_dim)
        self.ctx_emb = nn.Embedding(total, embedding_dim)

        nn.init.xavier_uniform_(self.word_emb.weight)
        nn.init.xavier_uniform_(self.ctx_emb.weight)

    def _lookup(self, emb: nn.Embedding, tokens: torch.Tensor, varieties: torch.Tensor) -> torch.Tensor:
        flat_idx = varieties * self.vocab_size + tokens
        return emb(flat_idx)

    def forward(
        self,
        word_idx: torch.Tensor,
        ctx_same: torch.Tensor,
        ctx_other: torch.Tensor,
        variety_a: torch.Tensor,
        variety_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        word_emb_a = self._lookup(self.word_emb, word_idx, variety_a)
        ctx_emb_a = self._lookup(self.ctx_emb, ctx_same, variety_a)
        ctx_emb_b = self._lookup(self.ctx_emb, ctx_other, variety_b)
        word_emb_b = self._lookup(self.word_emb, word_idx, variety_b)
        return word_emb_a, ctx_emb_a, ctx_emb_b, word_emb_b

    def extract_variety_embeddings(self, variety_idx: int) -> np.ndarray:
        start = variety_idx * self.vocab_size
        end = start + self.vocab_size
        return self.word_emb.weight[start:end].detach().cpu().numpy()
