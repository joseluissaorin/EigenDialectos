"""Dialect-Contrastive Loss (DCL) for training variety-aware embeddings.

The loss combines three terms:

    L_DCL = -log sigma(e_w^A . e_{c_A}^A)       [same-variety attraction]
            -log sigma(-e_w^A . e_{c_B}^B)       [cross-variety repulsion]
            + lambda ||e_w^A - e_w^B||^2 * 1[w not in R]  [anchor]

where R is the set of known regionalisms and sigma is the sigmoid function.

Variety affinity (CAN-CAR closer than CAN-MEX) is encoded in the dataset's
negative sampling distribution, not in the loss function.  This keeps the
loss numerically stable on MPS while achieving the same effect: similar
varieties generate fewer cross-variety negative pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DialectContrastiveLoss(nn.Module):
    """Three-term Dialect-Contrastive Loss.

    Parameters
    ----------
    lambda_anchor:
        Weight for the anchor regularisation term that keeps shared
        (non-regionalism) words aligned across varieties.
    """

    def __init__(self, lambda_anchor: float = 0.1) -> None:
        super().__init__()
        self.lambda_anchor = lambda_anchor

    def forward(
        self,
        word_emb_a: torch.Tensor,
        ctx_emb_a: torch.Tensor,
        ctx_emb_b: torch.Tensor,
        word_emb_b: torch.Tensor,
        is_regionalism: torch.Tensor,
        variety_a: torch.Tensor | None = None,
        variety_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the DCL loss over a mini-batch.

        Parameters
        ----------
        word_emb_a:
            ``(batch, dim)`` -- word embeddings in variety A.
        ctx_emb_a:
            ``(batch, dim)`` -- positive context embeddings in variety A.
        ctx_emb_b:
            ``(batch, dim)`` -- negative context embeddings in variety B.
        word_emb_b:
            ``(batch, dim)`` -- the *same* word's embedding in variety B
            (used for anchor regularisation).
        is_regionalism:
            ``(batch,)`` bool tensor -- ``True`` for words that are
            regionalisms (exempted from the anchor penalty).
        variety_a, variety_b:
            ``(batch,)`` int tensors -- variety indices (accepted for
            API compatibility with the trainer, not used in loss).
        """
        # Term 1: same-variety skip-gram attraction
        pos_dot = (word_emb_a * ctx_emb_a).sum(dim=-1)
        term_attraction = -F.logsigmoid(pos_dot)

        # Term 2: cross-variety repulsion
        neg_dot = (word_emb_a * ctx_emb_b).sum(dim=-1)
        term_repulsion = -F.logsigmoid(-neg_dot)

        # Term 3: anchor regularisation for shared words
        diff = word_emb_a - word_emb_b
        l2_sq = (diff * diff).sum(dim=-1)
        anchor_mask = (~is_regionalism).float()
        term_anchor = self.lambda_anchor * l2_sq * anchor_mask

        loss = (term_attraction + term_repulsion + term_anchor).mean()
        return loss
