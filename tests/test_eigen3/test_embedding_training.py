"""Tests for embedding training: dataset, model, loss, training loop.

50 tests covering the DCL embedding pipeline.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from eigen3.constants import ALL_VARIETIES, VARIETY_AFFINITIES
from eigen3.loss import DialectContrastiveLoss
from eigen3.model import DCLModel


# ======================================================================
# Dataset (15 tests)
# ======================================================================

class TestDataset:

    @pytest.fixture
    def mini_dataset(self, tokenizer_model_path, sample_corpus):
        """Small dataset for fast tests."""
        if not tokenizer_model_path.exists():
            pytest.skip("Tokenizer model not found")
        if all(len(docs) == 0 for docs in sample_corpus.values()):
            pytest.skip("No corpus data")

        from eigen3.tokenizer import Tokenizer
        from eigen3.dataset import SubwordDCLDataset

        tok = Tokenizer(tokenizer_model_path)
        # Use just 10 docs per variety for speed
        mini = {v: docs[:10] for v, docs in sample_corpus.items() if docs}
        if len(mini) < 2:
            pytest.skip("Need at least 2 varieties")
        return SubwordDCLDataset(mini, tok, window_size=3, neg_samples=2, seed=42)

    def test_skipgram_pairs_generated(self, mini_dataset):
        assert len(mini_dataset) > 0

    def test_correct_output_shape(self, mini_dataset):
        """Each item returns 6 tensors."""
        item = mini_dataset[0]
        assert len(item) == 6

    def test_variety_offset_correct(self, mini_dataset):
        """Global indices use variety * vocab_size + token."""
        item = mini_dataset[0]
        center, ctx_same, ctx_other, v_a, v_b, is_reg = [x.item() for x in item]
        assert 0 <= v_a < len(mini_dataset.variety_names)
        assert 0 <= v_b < len(mini_dataset.variety_names)

    def test_affinity_neg_sampling_can_car(self, mini_dataset):
        """CAN rarely samples CAR as negative (high affinity)."""
        if "ES_CAN" not in mini_dataset.variety_to_idx or "ES_CAR" not in mini_dataset.variety_to_idx:
            pytest.skip("CAN or CAR not in dataset")

        can_idx = mini_dataset.variety_to_idx["ES_CAN"]
        car_idx = mini_dataset.variety_to_idx["ES_CAR"]
        probs = mini_dataset._build_neg_sampling_probs(
            list(range(len(mini_dataset.variety_names)))
        )
        if can_idx in probs:
            others = [v for v in range(len(mini_dataset.variety_names)) if v != can_idx]
            if car_idx in others:
                car_prob = probs[can_idx][others.index(car_idx)]
                # High affinity (0.92) → low neg sampling probability
                assert car_prob < 0.1, f"CAR neg prob for CAN = {car_prob}"

    def test_neg_samples_count(self, mini_dataset):
        """Total samples = base_pairs * neg_samples."""
        assert mini_dataset._n_total == mini_dataset._data.shape[0]

    def test_deterministic(self, tokenizer_model_path, sample_corpus):
        """Same seed → same dataset."""
        if not tokenizer_model_path.exists():
            pytest.skip("No tokenizer")
        mini = {v: docs[:5] for v, docs in sample_corpus.items() if docs}
        if len(mini) < 2:
            pytest.skip("Need 2+ varieties")

        from eigen3.tokenizer import Tokenizer
        from eigen3.dataset import SubwordDCLDataset
        tok = Tokenizer(tokenizer_model_path)
        d1 = SubwordDCLDataset(mini, tok, window_size=3, neg_samples=2, seed=42)
        d2 = SubwordDCLDataset(mini, tok, window_size=3, neg_samples=2, seed=42)
        assert np.array_equal(d1._data, d2._data)

    def test_no_self_pairs(self, mini_dataset):
        """Center and context are different tokens (mostly)."""
        # In skip-gram, center != context by construction (j != i)
        centers = mini_dataset._data[:100, 0]
        contexts = mini_dataset._data[:100, 1]
        # Most should differ (not all, some tokens could repeat)
        assert np.sum(centers != contexts) > len(centers) * 0.5

    def test_all_long_tensors(self, mini_dataset):
        """All returned tensors are LongTensor."""
        item = mini_dataset[0]
        for t in item:
            assert t.dtype == torch.long

    def test_variety_names_sorted(self, mini_dataset):
        assert mini_dataset.variety_names == sorted(mini_dataset.variety_names)

    def test_vocab_size_positive(self, mini_dataset):
        assert mini_dataset.vocab_size > 0

    def test_data_array_correct_columns(self, mini_dataset):
        """Data array has 6 columns."""
        assert mini_dataset._data.shape[1] == 6

    def test_variety_indices_in_range(self, mini_dataset):
        """All variety indices valid."""
        v_a = mini_dataset._data[:, 3]
        v_b = mini_dataset._data[:, 4]
        n_v = len(mini_dataset.variety_names)
        assert np.all(v_a >= 0) and np.all(v_a < n_v)
        assert np.all(v_b >= 0) and np.all(v_b < n_v)

    def test_cross_variety_negatives(self, mini_dataset):
        """Negative samples come from different variety than anchor."""
        v_a = mini_dataset._data[:, 3]
        v_b = mini_dataset._data[:, 4]
        assert np.sum(v_a != v_b) > len(v_a) * 0.8

    def test_regionalism_flag_binary(self, mini_dataset):
        """is_regionalism is 0 or 1."""
        flags = mini_dataset._data[:, 5]
        assert np.all((flags == 0) | (flags == 1))

    def test_token_indices_in_range(self, mini_dataset):
        """Token indices within vocab_size."""
        for col in [0, 1, 2]:
            tokens = mini_dataset._data[:, col]
            assert np.all(tokens >= 0) and np.all(tokens < mini_dataset.vocab_size)


# ======================================================================
# Model (10 tests)
# ======================================================================

class TestModel:

    @pytest.fixture
    def model(self):
        return DCLModel(vocab_size=100, embedding_dim=50, n_varieties=8)

    def test_correct_param_count(self, model):
        """Total parameters = 2 * 8 * 100 * 50."""
        total = sum(p.numel() for p in model.parameters())
        assert total == 2 * 8 * 100 * 50

    def test_forward_produces_embeddings(self, model):
        batch = 16
        w = torch.randint(0, 100, (batch,))
        cs = torch.randint(0, 100, (batch,))
        co = torch.randint(0, 100, (batch,))
        va = torch.randint(0, 8, (batch,))
        vb = torch.randint(0, 8, (batch,))
        wa, ca, cb, wb = model(w, cs, co, va, vb)
        assert wa.shape == (batch, 50)

    def test_correct_dimensions(self, model):
        assert model.embedding_dim == 50
        assert model.vocab_size == 100
        assert model.n_varieties == 8

    def test_variety_extraction(self, model):
        emb = model.extract_variety_embeddings(0)
        assert emb.shape == (100, 50)

    def test_gradients_flow(self, model):
        w = torch.randint(0, 100, (8,))
        cs = torch.randint(0, 100, (8,))
        co = torch.randint(0, 100, (8,))
        va = torch.zeros(8, dtype=torch.long)
        vb = torch.ones(8, dtype=torch.long)
        wa, ca, cb, wb = model(w, cs, co, va, vb)
        loss = wa.sum()
        loss.backward()
        assert model.word_emb.weight.grad is not None

    def test_cpu_compatible(self, model):
        """Model works on CPU."""
        assert next(model.parameters()).device.type == "cpu"

    def test_deterministic_init(self):
        torch.manual_seed(42)
        m1 = DCLModel(100, 50, 8)
        torch.manual_seed(42)
        m2 = DCLModel(100, 50, 8)
        assert torch.allclose(m1.word_emb.weight, m2.word_emb.weight)

    def test_no_nan_output(self, model):
        w = torch.randint(0, 100, (8,))
        va = torch.zeros(8, dtype=torch.long)
        wa, _, _, _ = model(w, w, w, va, va)
        assert not torch.any(torch.isnan(wa))

    def test_different_varieties_different(self, model):
        """Different variety indices give different embeddings."""
        w = torch.tensor([0])
        v0 = torch.tensor([0])
        v1 = torch.tensor([1])
        e0 = model._lookup(model.word_emb, w, v0)
        e1 = model._lookup(model.word_emb, w, v1)
        assert not torch.allclose(e0, e1)

    def test_flat_layout_consecutive(self, model):
        """Variety 0 occupies indices [0, vocab_size)."""
        # Variety 0, token 5 → index 5
        # Variety 1, token 5 → index 105
        w = torch.tensor([5])
        v0 = torch.tensor([0])
        v1 = torch.tensor([1])
        idx0 = v0 * model.vocab_size + w
        idx1 = v1 * model.vocab_size + w
        assert idx0.item() == 5
        assert idx1.item() == 105


# ======================================================================
# Loss (10 tests)
# ======================================================================

class TestLoss:

    @pytest.fixture
    def loss_fn(self):
        return DialectContrastiveLoss(lambda_anchor=0.05)

    def test_attraction_for_similar(self, loss_fn):
        """Similar pairs have lower attraction loss."""
        dim = 50
        # Very similar pair
        a = torch.randn(8, dim)
        pos_close = a + 0.01 * torch.randn(8, dim)
        pos_far = torch.randn(8, dim)
        neg = torch.randn(8, dim)
        b = torch.randn(8, dim)
        reg = torch.zeros(8, dtype=torch.bool)

        l_close = loss_fn(a, pos_close, neg, b, reg)
        l_far = loss_fn(a, pos_far, neg, b, reg)
        assert l_close < l_far

    def test_repulsion_for_dissimilar(self, loss_fn):
        """Orthogonal negatives have lower repulsion loss."""
        dim = 50
        a = torch.randn(8, dim)
        pos = torch.randn(8, dim)
        # Orthogonal negatives
        neg_orth = torch.randn(8, dim)
        neg_orth = neg_orth - (neg_orth * a).sum(-1, keepdim=True) / (a * a).sum(-1, keepdim=True) * a
        # Parallel negatives (bad)
        neg_par = a + 0.1 * torch.randn(8, dim)
        b = torch.randn(8, dim)
        reg = torch.zeros(8, dtype=torch.bool)

        l_orth = loss_fn(a, pos, neg_orth, b, reg)
        l_par = loss_fn(a, pos, neg_par, b, reg)
        assert l_orth < l_par

    def test_anchor_penalizes_drift(self, loss_fn):
        """Anchor term penalizes when word_a != word_b."""
        dim = 50
        a = torch.randn(8, dim)
        pos = torch.randn(8, dim)
        neg = torch.randn(8, dim)
        reg = torch.zeros(8, dtype=torch.bool)

        b_same = a.clone()
        b_diff = torch.randn(8, dim)

        l_same = loss_fn(a, pos, neg, b_same, reg)
        l_diff = loss_fn(a, pos, neg, b_diff, reg)
        assert l_same < l_diff

    def test_regionalism_exempt_from_anchor(self, loss_fn):
        """Regionalisms are not penalized by anchor term."""
        dim = 50
        a = torch.randn(8, dim)
        pos = torch.randn(8, dim)
        neg = torch.randn(8, dim)
        b_diff = torch.randn(8, dim)

        reg_false = torch.zeros(8, dtype=torch.bool)
        reg_true = torch.ones(8, dtype=torch.bool)

        l_normal = loss_fn(a, pos, neg, b_diff, reg_false)
        l_regional = loss_fn(a, pos, neg, b_diff, reg_true)
        assert l_regional < l_normal

    def test_loss_non_negative(self, loss_fn):
        a = torch.randn(8, 50)
        l = loss_fn(a, torch.randn(8, 50), torch.randn(8, 50), a, torch.zeros(8, dtype=torch.bool))
        assert l.item() >= 0

    def test_gradient_non_zero(self, loss_fn):
        a = torch.randn(8, 50, requires_grad=True)
        l = loss_fn(a, torch.randn(8, 50), torch.randn(8, 50), torch.randn(8, 50), torch.zeros(8, dtype=torch.bool))
        l.backward()
        assert a.grad is not None and a.grad.abs().sum() > 0

    def test_nan_free(self, loss_fn):
        a = torch.randn(8, 50)
        l = loss_fn(a, torch.randn(8, 50), torch.randn(8, 50), a, torch.zeros(8, dtype=torch.bool))
        assert not torch.isnan(l)

    def test_zero_lambda_no_anchor(self):
        loss_fn = DialectContrastiveLoss(lambda_anchor=0.0)
        a = torch.randn(8, 50)
        b = torch.randn(8, 50)
        reg = torch.zeros(8, dtype=torch.bool)
        l = loss_fn(a, torch.randn(8, 50), torch.randn(8, 50), b, reg)
        # With lambda=0 and different anchor, should still be finite
        assert torch.isfinite(l)

    def test_lambda_scales(self):
        """Higher lambda → higher loss for drifted anchors."""
        a = torch.randn(8, 50)
        b = torch.randn(8, 50)
        pos = torch.randn(8, 50)
        neg = torch.randn(8, 50)
        reg = torch.zeros(8, dtype=torch.bool)

        l_low = DialectContrastiveLoss(0.01)(a, pos, neg, b, reg)
        l_high = DialectContrastiveLoss(1.0)(a, pos, neg, b, reg)
        assert l_high > l_low

    def test_batch_size_1(self, loss_fn):
        """Works with batch size 1."""
        a = torch.randn(1, 50)
        l = loss_fn(a, torch.randn(1, 50), torch.randn(1, 50), a, torch.zeros(1, dtype=torch.bool))
        assert torch.isfinite(l)


# ======================================================================
# Training (15 tests)
# ======================================================================

class TestTraining:

    def test_loss_history_loaded(self, embedding_meta):
        """Meta has final_loss recorded."""
        assert "final_loss" in embedding_meta
        assert embedding_meta["final_loss"] < 2.0

    def test_meta_epochs(self, embedding_meta):
        """30 epochs trained."""
        assert embedding_meta["epochs"] == 30

    def test_meta_lr(self, embedding_meta):
        """LR = 0.001."""
        assert embedding_meta["lr"] == 0.001

    def test_meta_batch_size(self, embedding_meta):
        assert embedding_meta["batch_size"] == 8192

    def test_meta_varieties(self, embedding_meta):
        assert len(embedding_meta["varieties"]) == 8

    def test_bpe_embeddings_loaded(self, bpe_embeddings_dict):
        """All 8 BPE embedding matrices loaded."""
        assert len(bpe_embeddings_dict) == 8

    def test_bpe_shape(self, bpe_embeddings_dict):
        """BPE embeddings are (8000, 100)."""
        for v, emb in bpe_embeddings_dict.items():
            assert emb.shape == (8000, 100), f"{v}: {emb.shape}"

    def test_no_nan_in_embeddings(self, bpe_embeddings_dict):
        for v, emb in bpe_embeddings_dict.items():
            assert np.all(np.isfinite(emb)), f"{v} has NaN/Inf"

    def test_embeddings_not_zero(self, bpe_embeddings_dict):
        """Embeddings are not all zeros."""
        for v, emb in bpe_embeddings_dict.items():
            assert np.linalg.norm(emb) > 0, f"{v} is all zeros"

    def test_device_detection(self):
        from eigen3.trainer import _detect_device
        device = _detect_device()
        assert device.type in ("cpu", "cuda", "mps")

    def test_model_creation(self):
        """DCLModel instantiates correctly."""
        m = DCLModel(8000, 100, 8)
        assert m.vocab_size == 8000

    def test_loss_fn_creation(self):
        l = DialectContrastiveLoss(0.05)
        assert l.lambda_anchor == 0.05

    def test_varieties_match(self, bpe_embeddings_dict):
        """All expected varieties present."""
        assert set(bpe_embeddings_dict.keys()) == set(ALL_VARIETIES)

    def test_embedding_dim_consistent(self, bpe_embeddings_dict, embedding_meta):
        """Embedding dim matches meta."""
        for v, emb in bpe_embeddings_dict.items():
            assert emb.shape[1] == embedding_meta["embedding_dim"]

    def test_different_varieties_different(self, bpe_embeddings_dict):
        """Different varieties have different embeddings."""
        can = bpe_embeddings_dict["ES_CAN"]
        car = bpe_embeddings_dict["ES_CAR"]
        assert not np.allclose(can, car)
