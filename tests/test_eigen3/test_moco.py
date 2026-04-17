"""Tests for MoCo momentum encoder and queue."""

from __future__ import annotations

import torch
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_model():
    """Create a minimal model with trainable + frozen params and BatchNorm."""
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 16, bias=False),   # "frozen" layer
        torch.nn.BatchNorm1d(16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 8),                # "trainable" layer
    )
    # Freeze the first linear layer (simulates frozen BETO)
    model[0].weight.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# MomentumEncoder tests
# ---------------------------------------------------------------------------

class TestMomentumEncoder:
    def test_init_matches_model(self):
        """Momentum encoder should start as an exact copy of the model."""
        from eigen3.moco import MomentumEncoder

        model = _make_simple_model()
        me = MomentumEncoder(model, momentum=0.999)

        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), me.encoder.named_parameters(),
        ):
            assert torch.equal(p1.data, p2.data), f"Mismatch at {n1}"

        for (n1, b1), (n2, b2) in zip(
            model.named_buffers(), me.encoder.named_buffers(),
        ):
            assert torch.equal(b1.data, b2.data), f"Buffer mismatch at {n1}"

    def test_encoder_in_eval_mode(self):
        from eigen3.moco import MomentumEncoder

        model = _make_simple_model()
        me = MomentumEncoder(model, momentum=0.999)
        assert not me.encoder.training

    def test_encoder_no_grad(self):
        from eigen3.moco import MomentumEncoder

        model = _make_simple_model()
        me = MomentumEncoder(model, momentum=0.999)
        for p in me.encoder.parameters():
            assert not p.requires_grad

    def test_ema_moves_toward_model(self):
        """After perturbing training model and updating, momentum params should move."""
        from eigen3.moco import MomentumEncoder

        model = _make_simple_model()
        me = MomentumEncoder(model, momentum=0.9)

        # Perturb trainable params
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * 0.5)

        # Snapshot before update
        old_params = {n: p.data.clone() for n, p in me.encoder.named_parameters()}

        me.update(model)

        # Trainable params should have moved; frozen params should not
        for (name, p_train), (_, p_mom) in zip(
            model.named_parameters(), me.encoder.named_parameters(),
        ):
            if p_train.requires_grad:
                assert not torch.equal(p_mom.data, old_params[name]), \
                    f"Trainable param {name} didn't move after EMA update"
            else:
                assert torch.equal(p_mom.data, old_params[name]), \
                    f"Frozen param {name} moved after EMA update"

    def test_ema_update_only_trainable(self):
        """Frozen params must remain identical after EMA update."""
        from eigen3.moco import MomentumEncoder

        model = _make_simple_model()
        frozen_before = model[0].weight.data.clone()

        me = MomentumEncoder(model, momentum=0.999)

        # Perturb trainable params
        with torch.no_grad():
            model[3].weight.add_(torch.randn_like(model[3].weight))

        me.update(model)

        # Frozen layer in momentum encoder should equal the original
        assert torch.equal(me.encoder[0].weight.data, frozen_before)

    def test_ema_updates_bn_buffers(self):
        """BatchNorm running_mean and running_var should be EMA-updated."""
        from eigen3.moco import MomentumEncoder

        model = _make_simple_model()
        model.train()

        # Run a forward pass to update BN running stats
        x = torch.randn(8, 16)
        model(x)

        me = MomentumEncoder(model, momentum=0.5)

        # Run another forward to change BN stats
        model(torch.randn(8, 16) * 3)

        old_mean = me.encoder[1].running_mean.clone()
        me.update(model)

        # running_mean should have changed
        assert not torch.equal(me.encoder[1].running_mean, old_mean)

    def test_ema_convergence(self):
        """With m=0 (no momentum), encoder should match model exactly after update."""
        from eigen3.moco import MomentumEncoder

        model = _make_simple_model()
        me = MomentumEncoder(model, momentum=0.0)

        # Perturb model
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p))

        me.update(model)

        for (_, p_train), (_, p_mom) in zip(
            model.named_parameters(), me.encoder.named_parameters(),
        ):
            if p_train.requires_grad:
                assert torch.allclose(p_mom.data, p_train.data, atol=1e-6)

    def test_state_dict_roundtrip(self):
        from eigen3.moco import MomentumEncoder

        model = _make_simple_model()
        me = MomentumEncoder(model, momentum=0.999)

        # Perturb and update
        with torch.no_grad():
            model[3].weight.add_(torch.ones_like(model[3].weight))
        me.update(model)

        sd = me.state_dict()

        # Create fresh and load
        me2 = MomentumEncoder(model, momentum=0.999)
        me2.load_state_dict(sd)

        for (_, p1), (_, p2) in zip(
            me.encoder.named_parameters(), me2.encoder.named_parameters(),
        ):
            assert torch.equal(p1.data, p2.data)


# ---------------------------------------------------------------------------
# MoCoQueue tests
# ---------------------------------------------------------------------------

class TestMoCoQueue:
    def test_empty_queue_returns_none(self):
        from eigen3.moco import MoCoQueue

        q = MoCoQueue(size=64, dim=8, device=torch.device("cpu"))
        assert q.get() is None

    def test_enqueue_and_get(self):
        from eigen3.moco import MoCoQueue

        q = MoCoQueue(size=64, dim=8, device=torch.device("cpu"))
        keys = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 3])

        q.enqueue(keys, labels)
        assert q.filled == 4

        emb, lab = q.get()
        assert emb.shape == (4, 8)
        assert torch.equal(lab, labels)
        assert torch.equal(emb, keys)

    def test_fifo_order(self):
        from eigen3.moco import MoCoQueue

        q = MoCoQueue(size=8, dim=2, device=torch.device("cpu"))

        # Enqueue 3 batches of 4 each → fills 12, but queue size=8, wraps
        for i in range(3):
            keys = torch.full((4, 2), float(i))
            labels = torch.full((4,), i, dtype=torch.long)
            q.enqueue(keys, labels)

        assert q.filled == 8
        # Most recent 4 entries should be from batch 2
        emb, lab = q.get(max_entries=4)
        assert emb.shape == (4, 2)
        assert torch.all(lab == 2)

    def test_circular_wrap(self):
        from eigen3.moco import MoCoQueue

        q = MoCoQueue(size=6, dim=2, device=torch.device("cpu"))

        # Fill with 3 batches of 4 → wraps around
        for i in range(3):
            keys = torch.full((4, 2), float(i))
            labels = torch.full((4,), i, dtype=torch.long)
            q.enqueue(keys, labels)

        assert q.filled == 6  # capped at size
        # Get all 6 — should be the most recent 6 entries
        emb, lab = q.get()
        assert emb.shape == (6, 2)

    def test_get_max_entries(self):
        from eigen3.moco import MoCoQueue

        q = MoCoQueue(size=32, dim=4, device=torch.device("cpu"))
        q.enqueue(torch.randn(16, 4), torch.zeros(16, dtype=torch.long))

        emb, lab = q.get(max_entries=8)
        assert emb.shape == (8, 4)

    def test_empty_enqueue(self):
        from eigen3.moco import MoCoQueue

        q = MoCoQueue(size=32, dim=4, device=torch.device("cpu"))
        q.enqueue(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long))
        assert q.filled == 0
        assert q.get() is None

    def test_state_dict_roundtrip(self):
        from eigen3.moco import MoCoQueue

        q = MoCoQueue(size=16, dim=4, device=torch.device("cpu"))
        q.enqueue(torch.randn(8, 4), torch.arange(8))

        sd = q.state_dict()

        q2 = MoCoQueue(size=16, dim=4, device=torch.device("cpu"))
        q2.load_state_dict(sd)

        assert q2.filled == q.filled
        assert q2.ptr == q.ptr
        emb1, lab1 = q.get()
        emb2, lab2 = q2.get()
        assert torch.equal(emb1, emb2)
        assert torch.equal(lab1, lab2)


# ---------------------------------------------------------------------------
# Loss integration test
# ---------------------------------------------------------------------------

class TestMoCoLossIntegration:
    def test_gradient_flows_through_queries_only(self):
        """proj_emb should get gradients; moco_keys and queue should not."""
        from eigen3.loss import DialectMultiTaskLoss

        criterion = DialectMultiTaskLoss(n_varieties=4, proj_dim=8, temperature=0.1)
        criterion.w_mlm = 0.0
        criterion.w_cls = 0.0
        criterion.w_con = 1.0
        criterion.w_center = 0.0

        queries = torch.randn(8, 8, requires_grad=True)
        queries_norm = torch.nn.functional.normalize(queries, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        keys = torch.randn(8, 8)  # no grad
        keys_norm = torch.nn.functional.normalize(keys, dim=1)

        queue_keys = torch.randn(16, 8)
        queue_labels = torch.randint(0, 4, (16,))

        loss, _ = criterion(
            proj_emb=queries_norm,
            variety_ids=labels,
            moco_keys=keys_norm,
            queue_emb=queue_keys,
            queue_labels=queue_labels,
        )

        loss.backward()
        assert queries.grad is not None
        assert keys.grad is None

    def test_fallback_without_moco_keys(self):
        """When moco_keys=None, should fall back to in-batch-only mode."""
        from eigen3.loss import DialectMultiTaskLoss

        criterion = DialectMultiTaskLoss(n_varieties=4, proj_dim=8, temperature=0.1)
        criterion.w_mlm = 0.0
        criterion.w_cls = 0.0
        criterion.w_con = 1.0
        criterion.w_center = 0.0

        queries = torch.nn.functional.normalize(torch.randn(8, 8), dim=1)
        queries.requires_grad_(True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        loss, loss_dict = criterion(
            proj_emb=queries,
            variety_ids=labels,
            moco_keys=None,
            queue_emb=None,
            queue_labels=None,
        )

        assert "contrastive" in loss_dict
        assert loss.item() > 0
        loss.backward()
        assert queries.grad is not None

    def test_moco_keys_change_loss(self):
        """Adding MoCo keys should change the contrastive loss value."""
        from eigen3.loss import DialectMultiTaskLoss

        criterion = DialectMultiTaskLoss(n_varieties=4, proj_dim=8, temperature=0.1)
        criterion.w_mlm = 0.0
        criterion.w_cls = 0.0
        criterion.w_con = 1.0
        criterion.w_center = 0.0

        torch.manual_seed(42)
        queries = torch.nn.functional.normalize(torch.randn(8, 8), dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        # Without MoCo keys
        _, d1 = criterion(proj_emb=queries, variety_ids=labels)

        # With MoCo keys (different embeddings → different loss)
        keys = torch.nn.functional.normalize(torch.randn(8, 8), dim=1)
        _, d2 = criterion(proj_emb=queries, variety_ids=labels, moco_keys=keys)

        assert d1["contrastive"] != d2["contrastive"]
