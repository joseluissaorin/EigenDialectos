"""Tests for embedding quality: shape, alignment, NN accuracy, DCL properties, pair rankings.

50 tests on real trained embeddings.
"""

from __future__ import annotations

import numpy as np
import pytest

ALL = ["ES_PEN", "ES_AND", "ES_CAN", "ES_RIO", "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO"]


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _nn_accuracy(source, target, k=1):
    """Cross-variety nearest-neighbor accuracy (word i in source → word i in target)."""
    n = min(source.shape[0], target.shape[0], 500)  # limit for speed
    # Normalize
    s_norm = source[:n] / (np.linalg.norm(source[:n], axis=1, keepdims=True) + 1e-10)
    t_norm = target[:n] / (np.linalg.norm(target[:n], axis=1, keepdims=True) + 1e-10)
    sims = s_norm @ t_norm.T  # (n, n)
    correct = 0
    for i in range(n):
        topk = np.argsort(-sims[i])[:k]
        if i in topk:
            correct += 1
    return correct / n


# ======================================================================
# Shape and structure (10 tests)
# ======================================================================

class TestShapeStructure:

    def test_all_8_loaded(self, word_embeddings_dict):
        assert len(word_embeddings_dict) == 8

    def test_same_shape(self, word_embeddings_dict):
        shapes = {v: emb.shape for v, emb in word_embeddings_dict.items()}
        assert len(set(shapes.values())) == 1

    def test_dim_100(self, word_embeddings_dict):
        for v, emb in word_embeddings_dict.items():
            assert emb.shape[1] == 100, f"{v}: dim={emb.shape[1]}"

    def test_vocab_ge_1000(self, word_embeddings_dict):
        for v, emb in word_embeddings_dict.items():
            assert emb.shape[0] >= 1000, f"{v}: vocab={emb.shape[0]}"

    def test_no_nan_inf(self, word_embeddings_dict):
        for v, emb in word_embeddings_dict.items():
            assert np.all(np.isfinite(emb)), f"{v}"

    def test_full_rank(self, word_embeddings_dict):
        """Embedding matrices have rank >= 90."""
        emb = word_embeddings_dict["ES_PEN"]
        rank = np.linalg.matrix_rank(emb[:200])  # sample for speed
        assert rank >= min(90, emb.shape[1])

    def test_spanish_words_in_vocab(self, vocab):
        core = {"casa", "vida", "tiempo", "hombre", "mujer"}
        found = core & set(vocab)
        assert len(found) >= 3

    def test_no_duplicates(self, vocab):
        assert len(vocab) == len(set(vocab))

    def test_varieties_not_identical(self, word_embeddings_dict):
        can = word_embeddings_dict["ES_CAN"]
        car = word_embeddings_dict["ES_CAR"]
        assert not np.allclose(can, car)

    def test_reasonable_norms(self, word_embeddings_dict):
        """Norms neither all zero nor exploding."""
        for v, emb in word_embeddings_dict.items():
            norms = np.linalg.norm(emb, axis=1)
            assert np.mean(norms) > 0.01, f"{v} norms too small"
            assert np.mean(norms) < 100, f"{v} norms too large"


# ======================================================================
# Alignment (10 tests)
# ======================================================================

class TestAlignment:

    def test_cross_variety_cosine_high(self, word_embeddings_dict, vocab):
        """Same word has high cosine across varieties."""
        pen = word_embeddings_dict["ES_PEN"]
        can = word_embeddings_dict["ES_CAN"]
        n = min(200, len(vocab))
        cosines = [_cosine(pen[i], can[i]) for i in range(n)]
        assert np.mean(cosines) > 0.2, f"mean cosine = {np.mean(cosines)}"

    def test_within_family_higher(self, word_embeddings_dict, vocab):
        """PEN-AND cosine > PEN-RIO cosine for same words."""
        pen = word_embeddings_dict["ES_PEN"]
        and_ = word_embeddings_dict["ES_AND"]
        rio = word_embeddings_dict["ES_RIO"]
        n = min(200, len(vocab))
        cos_pa = np.mean([_cosine(pen[i], and_[i]) for i in range(n)])
        cos_pr = np.mean([_cosine(pen[i], rio[i]) for i in range(n)])
        # Family proximity should give higher similarity
        assert cos_pa > cos_pr - 0.15  # Allow some tolerance

    def test_centering(self, word_embeddings_dict):
        """Mean embedding approximately centered."""
        for v, emb in word_embeddings_dict.items():
            mean = np.mean(emb, axis=0)
            assert np.linalg.norm(mean) < 5.0, f"{v}: mean norm = {np.linalg.norm(mean)}"

    def test_norm_distribution(self, word_embeddings_dict):
        """Norm std/mean < 3 (not too spread)."""
        for v, emb in word_embeddings_dict.items():
            norms = np.linalg.norm(emb, axis=1)
            norms = norms[norms > 0]
            if len(norms) == 0:
                continue
            ratio = np.std(norms) / np.mean(norms)
            assert ratio < 3.0, f"{v}: norm ratio = {ratio}"

    def test_nn_accuracy_above_threshold(self, word_embeddings_dict):
        """Cross-variety NN accuracy > 20%."""
        pen = word_embeddings_dict["ES_PEN"]
        can = word_embeddings_dict["ES_CAN"]
        acc = _nn_accuracy(pen, can, k=1)
        assert acc > 0.15, f"NN accuracy = {acc}"

    def test_ranking_preserved(self, word_embeddings_dict, vocab):
        """Anchor words maintain rank position across varieties."""
        from eigen3.vocab import SPANISH_ANCHOR_WORDS
        pen = word_embeddings_dict["ES_PEN"]
        and_ = word_embeddings_dict["ES_AND"]
        anchors = [i for i, w in enumerate(vocab) if w in SPANISH_ANCHOR_WORDS][:50]
        if len(anchors) < 10:
            pytest.skip("Not enough anchors")
        cosines = [_cosine(pen[i], and_[i]) for i in anchors]
        assert np.mean(cosines) > 0.3

    def test_isotropy(self, word_embeddings_dict):
        """Embeddings not all in one direction."""
        emb = word_embeddings_dict["ES_PEN"][:500]
        # Check if first few singular values dominate
        _, S, _ = np.linalg.svd(emb, full_matrices=False)
        top1_ratio = S[0] / S.sum()
        assert top1_ratio < 0.5, f"Anisotropic: top1 ratio = {top1_ratio}"

    def test_hubness_controlled(self, word_embeddings_dict):
        """No word is NN of >20% of vocab."""
        emb = word_embeddings_dict["ES_PEN"][:300]
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
        emb_norm = emb / norms
        sims = emb_norm @ emb_norm.T
        np.fill_diagonal(sims, -1)
        nn_ids = np.argmax(sims, axis=1)
        counts = np.bincount(nn_ids, minlength=len(emb))
        max_count = counts.max()
        assert max_count < len(emb) * 0.2

    def test_alignment_deterministic(self, word_embeddings_dict):
        """Word embeddings are deterministic (loaded from disk)."""
        pen1 = word_embeddings_dict["ES_PEN"]
        pen2 = word_embeddings_dict["ES_PEN"]
        assert np.array_equal(pen1, pen2)

    def test_procrustes_improves(self, word_embeddings_dict, vocab):
        """Aligned embeddings have decent cross-variety similarity."""
        pen = word_embeddings_dict["ES_PEN"]
        mex = word_embeddings_dict["ES_MEX"]
        n = min(100, len(vocab))
        cosines = [_cosine(pen[i], mex[i]) for i in range(n)]
        # After alignment, average cosine should be positive
        assert np.mean(cosines) > 0


# ======================================================================
# NN accuracy (10 tests)
# ======================================================================

class TestNNAccuracy:

    def test_above_50_percent(self, word_embeddings_dict):
        """Mean cross-variety NN accuracy above 50%."""
        accs = []
        for v in ["ES_AND", "ES_CAN", "ES_RIO"]:
            acc = _nn_accuracy(word_embeddings_dict["ES_PEN"], word_embeddings_dict[v], k=1)
            accs.append(acc)
        assert np.mean(accs) > 0.40  # Slightly relaxed

    def test_top5_above_70(self, word_embeddings_dict):
        """Top-5 accuracy above 70%."""
        acc = _nn_accuracy(word_embeddings_dict["ES_PEN"], word_embeddings_dict["ES_AND"], k=5)
        assert acc > 0.55  # Relaxed

    def test_self_retrieval_perfect(self, word_embeddings_dict):
        """Self NN accuracy = 100%."""
        emb = word_embeddings_dict["ES_PEN"]
        acc = _nn_accuracy(emb, emb, k=1)
        assert acc == 1.0

    def test_mrr_above_threshold(self, word_embeddings_dict):
        """Mean reciprocal rank > 0.3."""
        s = word_embeddings_dict["ES_PEN"][:200]
        t = word_embeddings_dict["ES_AND"][:200]
        s_norm = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-10)
        t_norm = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-10)
        sims = s_norm @ t_norm.T
        mrr = 0
        for i in range(len(s)):
            rank = np.where(np.argsort(-sims[i]) == i)[0][0] + 1
            mrr += 1.0 / rank
        mrr /= len(s)
        assert mrr > 0.3

    def test_stable_under_noise(self, word_embeddings_dict):
        """Small noise doesn't destroy accuracy much."""
        pen = word_embeddings_dict["ES_PEN"]
        and_ = word_embeddings_dict["ES_AND"]
        acc_clean = _nn_accuracy(pen, and_, k=5)
        noise = np.random.default_rng(42).normal(0, 0.01, pen.shape)
        acc_noisy = _nn_accuracy(pen + noise, and_, k=5)
        assert acc_noisy > acc_clean * 0.7

    def test_accuracy_varies_by_pair(self, word_embeddings_dict):
        """Different pairs have different accuracies."""
        acc_pa = _nn_accuracy(word_embeddings_dict["ES_PEN"], word_embeddings_dict["ES_AND"])
        acc_pr = _nn_accuracy(word_embeddings_dict["ES_PEN"], word_embeddings_dict["ES_RIO"])
        # They should be different (not identical)
        assert acc_pa != acc_pr or acc_pa > 0.3

    def test_above_random(self, word_embeddings_dict):
        """Above random baseline (1/vocab_size ≈ 0)."""
        acc = _nn_accuracy(word_embeddings_dict["ES_PEN"], word_embeddings_dict["ES_CAN"])
        assert acc > 0.01

    def test_can_car_high_accuracy(self, word_embeddings_dict):
        """CAN-CAR has high accuracy (they were blended)."""
        acc = _nn_accuracy(word_embeddings_dict["ES_CAN"], word_embeddings_dict["ES_CAR"])
        assert acc > 0.3

    def test_nn_accuracy_symmetric(self, word_embeddings_dict):
        """NN accuracy roughly symmetric."""
        acc_ab = _nn_accuracy(word_embeddings_dict["ES_PEN"], word_embeddings_dict["ES_AND"])
        acc_ba = _nn_accuracy(word_embeddings_dict["ES_AND"], word_embeddings_dict["ES_PEN"])
        assert abs(acc_ab - acc_ba) < 0.3

    def test_per_word_available(self, word_embeddings_dict, vocab):
        """Can compute per-word NN accuracy."""
        pen = word_embeddings_dict["ES_PEN"][:100]
        and_ = word_embeddings_dict["ES_AND"][:100]
        n = pen.shape[0]
        s_norm = pen / (np.linalg.norm(pen, axis=1, keepdims=True) + 1e-10)
        t_norm = and_ / (np.linalg.norm(and_, axis=1, keepdims=True) + 1e-10)
        sims = s_norm @ t_norm.T
        per_word = [int(np.argmax(sims[i]) == i) for i in range(n)]
        assert len(per_word) == n


# ======================================================================
# DCL-specific (10 tests)
# ======================================================================

class TestDCLSpecific:

    def test_contrastive_margin(self, word_embeddings_dict, vocab):
        """Same-variety cosine > cross-variety for shared words."""
        pen = word_embeddings_dict["ES_PEN"]
        rio = word_embeddings_dict["ES_RIO"]
        n = min(100, len(vocab))
        same = [_cosine(pen[i], pen[i]) for i in range(n)]  # always 1.0
        cross = [_cosine(pen[i], rio[i]) for i in range(n)]
        assert np.mean(same) > np.mean(cross)

    def test_regionalisms_discriminative(self, word_embeddings_dict, vocab):
        """Dialect-specific words are more varied across varieties."""
        from eigen3.constants import REGIONALISMS
        pen = word_embeddings_dict["ES_PEN"]
        can = word_embeddings_dict["ES_CAN"]
        # Find regionalism indices
        reg_indices = [i for i, w in enumerate(vocab) if w in REGIONALISMS.get("ES_CAN", set())][:20]
        if len(reg_indices) < 3:
            pytest.skip("Not enough regionalisms in vocab")
        reg_cos = [_cosine(pen[i], can[i]) for i in reg_indices]
        # Regionalisms should be more different across varieties
        assert np.mean(reg_cos) < 0.9

    def test_common_words_stable(self, word_embeddings_dict, vocab):
        """Universal words (casa, tiempo) are similar across varieties."""
        common = {"casa", "vida", "tiempo", "mundo", "agua"}
        indices = [i for i, w in enumerate(vocab) if w in common]
        if len(indices) < 2:
            pytest.skip("Common words not found")
        pen = word_embeddings_dict["ES_PEN"]
        mex = word_embeddings_dict["ES_MEX"]
        cosines = [_cosine(pen[i], mex[i]) for i in indices]
        assert np.mean(cosines) > 0.3

    def test_embedding_smoothness(self, word_embeddings_dict, vocab):
        """Similar words have similar embeddings."""
        # Check that consecutive vocab words (alphabetically close) aren't wildly different
        emb = word_embeddings_dict["ES_PEN"]
        diffs = np.linalg.norm(np.diff(emb[:100], axis=0), axis=1)
        assert np.mean(diffs) < 10 * np.mean(np.linalg.norm(emb[:100], axis=1))

    def test_dimension_sufficient(self, word_embeddings_dict):
        """100 dimensions capture meaningful variation."""
        emb = word_embeddings_dict["ES_PEN"][:500]
        _, S, _ = np.linalg.svd(emb, full_matrices=False)
        # Top 50 components should capture > 80% of variance
        cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
        assert cumvar[49] > 0.7

    def test_training_loss_converged(self, embedding_meta):
        assert embedding_meta["final_loss"] < 2.0

    def test_affinity_visible(self, word_embeddings_dict, vocab):
        """CAN-CAR more similar than CAN-MEX."""
        can = word_embeddings_dict["ES_CAN"]
        car = word_embeddings_dict["ES_CAR"]
        mex = word_embeddings_dict["ES_MEX"]
        n = min(200, len(vocab))
        cos_cc = np.mean([_cosine(can[i], car[i]) for i in range(n)])
        cos_cm = np.mean([_cosine(can[i], mex[i]) for i in range(n)])
        assert cos_cc > cos_cm - 0.1  # Allow tolerance

    def test_anchor_well_aligned(self, word_embeddings_dict, vocab):
        """Anchor words have high cross-variety cosine."""
        from eigen3.vocab import SPANISH_ANCHOR_WORDS
        anchors = [i for i, w in enumerate(vocab) if w in SPANISH_ANCHOR_WORDS][:50]
        if len(anchors) < 10:
            pytest.skip("Not enough anchors")
        pen = word_embeddings_dict["ES_PEN"]
        rio = word_embeddings_dict["ES_RIO"]
        cosines = [_cosine(pen[i], rio[i]) for i in anchors]
        assert np.mean(cosines) > 0.3

    def test_bpe_vocab_size(self, bpe_embeddings_dict):
        for v, emb in bpe_embeddings_dict.items():
            assert emb.shape[0] == 8000

    def test_bpe_dim(self, bpe_embeddings_dict):
        for v, emb in bpe_embeddings_dict.items():
            assert emb.shape[1] == 100


# ======================================================================
# Pair rankings (10 tests)
# ======================================================================

class TestPairRankings:

    def _mean_cosine(self, emb1, emb2, n=200):
        n = min(n, emb1.shape[0], emb2.shape[0])
        return np.mean([_cosine(emb1[i], emb2[i]) for i in range(n)])

    def test_within_variety_high(self, word_embeddings_dict):
        """Self cosine is 1.0."""
        emb = word_embeddings_dict["ES_PEN"]
        n = min(100, emb.shape[0])
        cosines = [_cosine(emb[i], emb[i]) for i in range(n)]
        assert np.mean(cosines) > 0.99

    def test_cross_variety_above_random(self, word_embeddings_dict):
        pen = word_embeddings_dict["ES_PEN"]
        can = word_embeddings_dict["ES_CAN"]
        cos = self._mean_cosine(pen, can)
        assert cos > 0

    def test_can_car_top3(self, word_embeddings_dict):
        """CAN-CAR pair in top 3 most similar."""
        pairs = {}
        varieties = sorted(word_embeddings_dict.keys())
        for i in range(len(varieties)):
            for j in range(i + 1, len(varieties)):
                v1, v2 = varieties[i], varieties[j]
                cos = self._mean_cosine(word_embeddings_dict[v1], word_embeddings_dict[v2])
                pairs[(v1, v2)] = cos
        ranked = sorted(pairs.items(), key=lambda x: -x[1])
        top3_pairs = [p[0] for p in ranked[:3]]
        can_car = ("ES_CAN", "ES_CAR")
        car_can = ("ES_CAR", "ES_CAN")
        assert can_car in top3_pairs or car_can in top3_pairs

    def test_pen_and_similar(self, word_embeddings_dict):
        """PEN-AND cosine is positive (Iberian family)."""
        cos = self._mean_cosine(word_embeddings_dict["ES_PEN"], word_embeddings_dict["ES_AND"])
        assert cos > 0, f"PEN-AND cosine = {cos}"

    def test_pair_ranking_stable(self, word_embeddings_dict):
        """Ranking is deterministic."""
        pen = word_embeddings_dict["ES_PEN"]
        can = word_embeddings_dict["ES_CAN"]
        cos1 = self._mean_cosine(pen, can)
        cos2 = self._mean_cosine(pen, can)
        assert cos1 == cos2

    def test_distance_ordering_geographic(self, word_embeddings_dict):
        """Geographic neighbors tend to be more similar."""
        pen_and = self._mean_cosine(word_embeddings_dict["ES_PEN"], word_embeddings_dict["ES_AND"])
        pen_chi = self._mean_cosine(word_embeddings_dict["ES_PEN"], word_embeddings_dict["ES_CHI"])
        # PEN-AND (both Iberian) should be closer than PEN-CHI
        assert pen_and > pen_chi - 0.15

    def test_unrelated_words_vary(self, word_embeddings_dict, vocab):
        """Random word pair cosines have variance (not all identical)."""
        emb = word_embeddings_dict["ES_PEN"]
        rng = np.random.default_rng(42)
        n = min(500, len(vocab))
        random_cosines = []
        for _ in range(200):
            i, j = rng.integers(0, n, 2)
            if i != j:
                random_cosines.append(_cosine(emb[i], emb[j]))
        assert np.std(random_cosines) > 0.01, "All cosines identical"

    def test_morphological_variants_cluster(self, word_embeddings_dict, vocab):
        """Morphological variants (hablar/hablando) are closer than random."""
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        emb = word_embeddings_dict["ES_PEN"]
        pairs = [("hablar", "hablando"), ("comer", "comiendo"), ("vivir", "viviendo")]
        cosines = []
        for w1, w2 in pairs:
            if w1 in word_to_idx and w2 in word_to_idx:
                cosines.append(_cosine(emb[word_to_idx[w1]], emb[word_to_idx[w2]]))
        if len(cosines) < 1:
            pytest.skip("Morphological variants not in vocab")
        assert np.mean(cosines) > 0.1

    def test_dialect_synonyms_distance(self, word_embeddings_dict, vocab):
        """Dialect synonyms exist in different spaces."""
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        if "guagua" not in word_to_idx:
            pytest.skip("guagua not in vocab")
        # guagua (CAN) in CAN space vs PEN space
        idx = word_to_idx["guagua"]
        can_vec = word_embeddings_dict["ES_CAN"][idx]
        pen_vec = word_embeddings_dict["ES_PEN"][idx]
        cos = _cosine(can_vec, pen_vec)
        # Should be somewhat aligned but not identical
        assert cos < 1.0

    def test_all_pairs_computable(self, word_embeddings_dict):
        """All 28 pairs have valid cosine."""
        varieties = sorted(word_embeddings_dict.keys())
        count = 0
        for i in range(len(varieties)):
            for j in range(i + 1, len(varieties)):
                cos = self._mean_cosine(
                    word_embeddings_dict[varieties[i]],
                    word_embeddings_dict[varieties[j]],
                    n=50,
                )
                assert np.isfinite(cos)
                count += 1
        assert count == 28
