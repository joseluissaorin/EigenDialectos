"""Tests for Dialect-Contrastive Loss training pipeline."""

from __future__ import annotations

import pytest
import torch

from eigendialectos.embeddings.dcl.loss import DialectContrastiveLoss
from eigendialectos.embeddings.dcl.model import DCLEmbeddingModel
from eigendialectos.embeddings.dcl.regionalisms import ALL_REGIONALISMS, REGIONALISMS


class TestDialectContrastiveLoss:
    def test_loss_computation(self):
        loss_fn = DialectContrastiveLoss(lambda_anchor=0.1)
        batch, dim = 16, 50
        word_a = torch.randn(batch, dim)
        ctx_a = torch.randn(batch, dim)
        ctx_b = torch.randn(batch, dim)
        word_b = torch.randn(batch, dim)
        is_reg = torch.zeros(batch, dtype=torch.bool)
        loss = loss_fn(word_a, ctx_a, ctx_b, word_b, is_reg)
        assert loss.shape == ()
        assert loss.item() > 0  # Loss should be positive

    def test_anchor_term_zero_for_regionalisms(self):
        loss_fn = DialectContrastiveLoss(lambda_anchor=1.0)
        batch, dim = 16, 50
        word_a = torch.randn(batch, dim)
        ctx_a = torch.randn(batch, dim)
        ctx_b = torch.randn(batch, dim)
        word_b = torch.randn(batch, dim)

        # All regionalisms: anchor term should be zero
        is_reg_all = torch.ones(batch, dtype=torch.bool)
        loss_reg = loss_fn(word_a, ctx_a, ctx_b, word_b, is_reg_all)

        # No regionalisms: anchor term should be non-zero
        is_reg_none = torch.zeros(batch, dtype=torch.bool)
        loss_no_reg = loss_fn(word_a, ctx_a, ctx_b, word_b, is_reg_none)

        # Loss with no regionalisms should differ from all-regionalisms
        assert loss_reg.item() != loss_no_reg.item()

    def test_gradient_flow(self):
        loss_fn = DialectContrastiveLoss()
        batch, dim = 8, 30
        word_a = torch.randn(batch, dim, requires_grad=True)
        ctx_a = torch.randn(batch, dim)
        ctx_b = torch.randn(batch, dim)
        word_b = torch.randn(batch, dim)
        is_reg = torch.zeros(batch, dtype=torch.bool)
        loss = loss_fn(word_a, ctx_a, ctx_b, word_b, is_reg)
        loss.backward()
        assert word_a.grad is not None
        assert not torch.all(word_a.grad == 0)


class TestDCLModel:
    def test_forward_shapes(self):
        model = DCLEmbeddingModel(vocab_size=100, embedding_dim=50, n_varieties=4)
        batch = 8
        word_idx = torch.randint(0, 100, (batch,))
        ctx_a = torch.randint(0, 100, (batch,))
        ctx_b = torch.randint(0, 100, (batch,))
        va = torch.zeros(batch, dtype=torch.long)
        vb = torch.ones(batch, dtype=torch.long)

        w_a, c_a, c_b, w_b = model(word_idx, ctx_a, ctx_b, va, vb)
        assert w_a.shape == (batch, 50)
        assert c_a.shape == (batch, 50)
        assert c_b.shape == (batch, 50)
        assert w_b.shape == (batch, 50)

    def test_different_varieties_give_different_embeddings(self):
        model = DCLEmbeddingModel(vocab_size=100, embedding_dim=50, n_varieties=4)
        word_idx = torch.tensor([5])
        ctx = torch.tensor([10])

        emb_0 = model.get_word_embeddings(0)[5]
        emb_1 = model.get_word_embeddings(1)[5]
        # Different varieties should have different embeddings (Xavier init)
        assert not torch.allclose(emb_0, emb_1)

    def test_training_step(self):
        model = DCLEmbeddingModel(vocab_size=100, embedding_dim=50, n_varieties=4)
        loss_fn = DialectContrastiveLoss()
        opt = torch.optim.Adam(model.parameters())

        batch = 8
        word_idx = torch.randint(0, 100, (batch,))
        ctx_a = torch.randint(0, 100, (batch,))
        ctx_b = torch.randint(0, 100, (batch,))
        va = torch.zeros(batch, dtype=torch.long)
        vb = torch.ones(batch, dtype=torch.long)
        is_reg = torch.zeros(batch, dtype=torch.bool)

        w_a, c_a, c_b, w_b = model(word_idx, ctx_a, ctx_b, va, vb)
        loss = loss_fn(w_a, c_a, c_b, w_b, is_reg)

        loss.backward()
        opt.step()
        # Should not error


class TestRegionalisms:
    def test_all_varieties_present(self):
        for code in ["ES_AND", "ES_CAN", "ES_RIO", "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO"]:
            assert code in REGIONALISMS
            assert len(REGIONALISMS[code]) >= 20, f"{code} has only {len(REGIONALISMS[code])} words"

    def test_pen_is_empty(self):
        assert len(REGIONALISMS.get("ES_PEN", set())) == 0

    def test_all_regionalisms_union(self):
        assert len(ALL_REGIONALISMS) >= 150

    def test_no_overlap_with_common(self):
        # Common words should not be in regionalism sets
        common = {"de", "la", "el", "en", "que", "y", "a", "los", "se"}
        overlap = ALL_REGIONALISMS & common
        assert len(overlap) == 0
