"""Multi-task training loss for the dialect transformer (v3).

L = w_mlm * L_mlm + w_cls * L_cls + w_con * L_supcon + w_center * L_center

v3 uses supervised contrastive learning (Khosla et al. 2020) with
MoCo momentum-encoded queue (He et al. 2020), the decoupled contrastive
objective (Yeh et al. 2022), center loss warmup (Wen et al. 2016),
and ArcFace-compatible classification (Deng et al. 2019). Temperature
lowered to 0.07, contrastive weight front-loaded to 0.4, with
two-phase training and gradual queue ramp.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Center loss (Wen et al. 2016) for class-conditional clustering.

    Pulls projected embeddings toward learned class centroids during
    the contrastive warmup phase, helping break the initial symmetry
    before SupCon gradients become dominant.
    """

    def __init__(self, n_classes: int, feat_dim: int) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_classes, feat_dim) * 0.01)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute mean squared distance to class centers."""
        batch_centers = self.centers[labels]
        return ((x - batch_centers) ** 2).sum(dim=-1).mean() / 2


class DialectMultiTaskLoss(nn.Module):
    """Multi-task loss: MLM + ArcFace classification + SupCon + center (+ MoCo + DCL).

    v3 loss: temperature 0.07, front-loaded w_con=0.4, center loss
    warmup for initial contrastive steps. DCL drops positives from
    the denominator (Yeh et al. 2022). MoCo momentum queue replaces
    the original XBM cross-batch memory.
    """

    def __init__(
        self,
        n_varieties: int = 8,
        proj_dim: int = 384,
        temperature: float = 0.07,
        w_mlm_init: float = 0.3,
        w_mlm_final: float = 0.15,
        w_cls: float = 0.3,
        w_con_init: float = 0.4,
        w_con_final: float = 0.55,
        use_dcl: bool = True,
        w_center: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_varieties = n_varieties
        self.temperature = temperature
        self.w_mlm_init = w_mlm_init
        self.w_mlm_final = w_mlm_final
        self.w_con_init = w_con_init
        self.w_con_final = w_con_final
        self.use_dcl = use_dcl

        # Curriculum weights (updated per epoch or by trainer phase logic)
        self.w_mlm = w_mlm_init
        self.w_cls = w_cls
        self.w_con = w_con_init
        self.w_center = w_center

        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.center_loss_fn = CenterLoss(n_varieties, proj_dim)

    def update_curriculum(self, epoch: int, total_epochs: int) -> None:
        """Shift emphasis from MLM to contrastive over training.

        Early: MLM=0.3, cls=0.3, con=0.4
        Late:  MLM=0.15, cls=0.3, con=0.55
        """
        denom = max(total_epochs - 1, 1)
        progress = max(0, epoch - 1) / denom
        progress = min(max(progress, 0.0), 1.0)
        self.w_mlm = self.w_mlm_init - (self.w_mlm_init - self.w_mlm_final) * progress
        self.w_con = self.w_con_init + (self.w_con_final - self.w_con_init) * progress
        # w_cls stays constant

    def forward(
        self,
        mlm_logits: torch.Tensor | None = None,
        mlm_labels: torch.Tensor | None = None,
        cls_logits: torch.Tensor | None = None,
        cls_labels: torch.Tensor | None = None,
        proj_emb: torch.Tensor | None = None,
        variety_ids: torch.Tensor | None = None,
        moco_keys: torch.Tensor | None = None,
        queue_emb: torch.Tensor | None = None,
        queue_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute multi-task loss.

        Parameters
        ----------
        proj_emb : (B, D) unit-norm current-batch projections (grad-carrying).
        variety_ids : (B,) dialect labels for the current batch.
        moco_keys : (B, D) unit-norm momentum-encoder keys (detached), optional.
            When provided, these are used as in-batch candidates instead of
            proj_emb itself. When None, falls back to in-batch-only mode
            using proj_emb as both queries and candidates.
        queue_emb : (Q, D) unit-norm queue projections (detached), optional.
        queue_labels : (Q,) dialect labels aligned to queue_emb, optional.

        Returns total loss and a dict of component losses for logging.
        """
        losses: dict[str, float] = {}
        total = torch.tensor(0.0, device=self._get_device(mlm_logits, cls_logits, proj_emb))

        # 1. MLM loss
        if mlm_logits is not None and mlm_labels is not None:
            l_mlm = self.mlm_criterion(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_labels.view(-1),
            )
            total = total + self.w_mlm * l_mlm
            losses["mlm"] = l_mlm.item()

        # 2. Classification loss (ArcFace logits)
        if cls_logits is not None and cls_labels is not None:
            l_cls = self.cls_criterion(cls_logits, cls_labels)
            total = total + self.w_cls * l_cls
            losses["cls"] = l_cls.item()

        # 3. Center loss (warmup clustering objective)
        if self.w_center > 0 and proj_emb is not None and variety_ids is not None:
            l_center = self.center_loss_fn(proj_emb, variety_ids)
            total = total + self.w_center * l_center
            losses["center"] = l_center.item()

        # 4. Supervised contrastive loss with optional MoCo queue and DCL.
        if self.w_con > 0 and proj_emb is not None and variety_ids is not None and proj_emb.size(0) > 1:
            l_con = self._supcon_moco_dcl(
                queries=proj_emb,
                query_labels=variety_ids,
                keys=moco_keys,
                queue_keys=queue_emb,
                queue_labels=queue_labels,
            )
            total = total + self.w_con * l_con
            losses["contrastive"] = l_con.item()

        losses["total"] = total.item()
        return total, losses

    def _supcon_moco_dcl(
        self,
        queries: torch.Tensor,
        query_labels: torch.Tensor,
        keys: torch.Tensor | None = None,
        queue_keys: torch.Tensor | None = None,
        queue_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Supervised contrastive loss with MoCo momentum queue and DCL.

        Parameters
        ----------
        queries      : (B, D) unit-norm training-encoder projections (grad-carrying).
        query_labels : (B,)   variety IDs for the current batch.
        keys         : (B, D) unit-norm momentum-encoder projections (detached), optional.
                       When None, ``queries`` are used as both queries and candidates
                       (in-batch-only fallback, equivalent to the pre-MoCo behavior).
        queue_keys   : (Q, D) unit-norm queue entries from momentum encoder, optional.
        queue_labels : (Q,)   variety IDs aligned to queue_keys.

        With MoCo, candidates = keys ∪ queue (all from the momentum encoder).
        Queries come from the training encoder. Gradients flow through queries
        only. The self-mask excludes position i in keys (same sample as query i,
        just from a different encoder).

        DCL (Yeh et al. 2022) removes positives from the denominator.
        """
        B, D = queries.shape
        device = queries.device

        # Fallback: no momentum encoder → in-batch-only (queries are candidates)
        if keys is None:
            keys = queries
            in_batch_self = True  # query i and candidate i are the same tensor
        else:
            in_batch_self = False

        # Assemble candidates: in-batch keys + queue
        if queue_keys is not None and queue_keys.numel() > 0:
            cand = torch.cat([keys, queue_keys], dim=0)
            cand_labels = torch.cat([query_labels, queue_labels], dim=0)
        else:
            cand = keys
            cand_labels = query_labels

        N = cand.shape[0]

        # Cosine / temperature
        logits = queries @ cand.t() / self.temperature  # (B, N)

        # Self-mask: position i in queries maps to position i in keys
        # (same sample, possibly different encoder view)
        self_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        diag_idx = torch.arange(B, device=device)
        self_mask[diag_idx, diag_idx] = True

        # Positive mask: same label AND not self
        pos_mask = (query_labels.unsqueeze(1) == cand_labels.unsqueeze(0)) & ~self_mask

        # Numerical stability — subtract the max over non-self positions.
        logits_for_max = logits.masked_fill(self_mask, float("-inf"))
        logits_max, _ = logits_for_max.max(dim=1, keepdim=True)
        shifted = logits - logits_max.detach()

        # exp over all positions, then zero the self position for the sums
        exp_logits = torch.exp(shifted)
        valid_mask = (~self_mask).float()
        exp_logits = exp_logits * valid_mask

        if self.use_dcl:
            # Decoupled: denominator = negatives only (drop positives)
            neg_mask = (~pos_mask) & (~self_mask)
            denom = (exp_logits * neg_mask.float()).sum(dim=1, keepdim=True)
        else:
            # Vanilla SupCon: denominator = all non-self candidates
            denom = exp_logits.sum(dim=1, keepdim=True)

        log_denom = torch.log(denom.clamp(min=1e-12)).squeeze(1)  # (B,)

        pos_count = pos_mask.sum(dim=1)
        safe_pos_count = pos_count.clamp(min=1).float()

        # mean over positives of (shifted[i,j] - log_denom[i])
        sum_shifted_pos = (shifted * pos_mask.float()).sum(dim=1)  # (B,)
        mean_log_prob_pos = sum_shifted_pos / safe_pos_count - log_denom

        has_pos = (pos_count > 0).float()
        n_valid = has_pos.sum().clamp(min=1)
        return -(mean_log_prob_pos * has_pos).sum() / n_valid

    @staticmethod
    def _get_device(*tensors: torch.Tensor | None) -> torch.device:
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Legacy DCL loss (kept for backward compatibility with skip-gram path)
# ---------------------------------------------------------------------------

class DialectContrastiveLoss(nn.Module):
    """Three-term DCL: attraction + repulsion + anchor regularization."""

    def __init__(self, lambda_anchor: float = 0.05) -> None:
        super().__init__()
        self.lambda_anchor = lambda_anchor

    def forward(
        self,
        word_emb_a: torch.Tensor,
        ctx_emb_a: torch.Tensor,
        ctx_emb_b: torch.Tensor,
        word_emb_b: torch.Tensor,
        is_regionalism: torch.Tensor,
    ) -> torch.Tensor:
        # Term 1: same-variety attraction (skip-gram)
        pos_dot = (word_emb_a * ctx_emb_a).sum(dim=-1)
        term_attract = -F.logsigmoid(pos_dot)

        # Term 2: cross-variety repulsion
        neg_dot = (word_emb_a * ctx_emb_b).sum(dim=-1)
        term_repel = -F.logsigmoid(-neg_dot)

        # Term 3: anchor regularization for non-regionalism words
        diff = word_emb_a - word_emb_b
        l2_sq = (diff * diff).sum(dim=-1)
        anchor_mask = (~is_regionalism).float()
        term_anchor = self.lambda_anchor * l2_sq * anchor_mask

        return (term_attract + term_repel + term_anchor).mean()
