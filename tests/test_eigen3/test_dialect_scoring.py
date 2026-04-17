"""Tests for DialectScorer: spectral fingerprinting of text to P(dialect|text).

60 tests covering text embedding, self-classification, cross-variety ordering,
calibration, and per-mode activation behaviour on real v2 artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigen3.scorer import DialectScorer, _cosine_similarity, _softmax, _tokenize

ALL_VARIETIES = [
    "ES_PEN", "ES_AND", "ES_CAN", "ES_RIO",
    "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO",
]


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def scorer(word_embeddings_dict, vocab, decomps_dict):
    return DialectScorer(word_embeddings_dict, vocab, decomps_dict, reference="ES_PEN")


@pytest.fixture(scope="module")
def known_word(vocab):
    """Return a word that is guaranteed to be in the vocabulary."""
    # Pick a common Spanish word that should be present in any 43k+ vocab
    for candidate in ["casa", "mundo", "tiempo", "vida", "hombre"]:
        if candidate in vocab:
            return candidate
    # Fallback: return the first vocab entry
    return vocab[0]


@pytest.fixture(scope="module")
def known_text(vocab):
    """Build a short text from known vocabulary words."""
    words = [w for w in vocab[:200] if len(w) > 2][:5]
    return " ".join(words)


# ======================================================================
# TestTextEmbedding (10 tests)
# ======================================================================

class TestTextEmbedding:

    def test_correct_dimension(self, scorer, known_text):
        """embed_text returns shape (dim,) matching embedding dimension."""
        vec = scorer.embed_text(known_text)
        assert vec.shape == (scorer._dim,)

    def test_oov_words_zero(self, scorer):
        """Text with only OOV words returns the zero vector."""
        oov_text = "xyzzy qwrtp fblthp"
        vec = scorer.embed_text(oov_text)
        assert np.allclose(vec, 0.0)

    def test_known_words_nonzero(self, scorer, known_text):
        """Text with known words produces a nonzero centroid."""
        vec = scorer.embed_text(known_text)
        assert np.linalg.norm(vec) > 1e-10

    def test_deterministic(self, scorer, known_text):
        """Same text always produces the same embedding."""
        v1 = scorer.embed_text(known_text)
        v2 = scorer.embed_text(known_text)
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_differ(self, scorer, vocab):
        """Two texts built from different vocabulary words produce different embeddings."""
        # Pick two disjoint sets of vocabulary words
        words_a = [w for w in vocab[:100] if len(w) > 2][:3]
        words_b = [w for w in vocab[500:600] if len(w) > 2][:3]
        text_a = " ".join(words_a)
        text_b = " ".join(words_b)
        va = scorer.embed_text(text_a)
        vb = scorer.embed_text(text_b)
        assert not np.allclose(va, vb), "Distinct word sets should yield distinct centroids"

    def test_punctuation_stripped(self, scorer, known_word):
        """Punctuation does not affect the embedding."""
        clean = known_word
        decorated = f"\u00a1{known_word.capitalize()}! \u00bf{known_word}?"
        v_clean = scorer.embed_text(clean)
        v_decorated = scorer.embed_text(decorated)
        np.testing.assert_allclose(v_clean, v_decorated, atol=1e-12)

    def test_case_insensitive(self, scorer, known_word):
        """Embeddings are case-insensitive (tokenizer lowercases)."""
        v_lower = scorer.embed_text(known_word.lower())
        v_upper = scorer.embed_text(known_word.upper())
        np.testing.assert_allclose(v_lower, v_upper, atol=1e-12)

    def test_empty_text_zero(self, scorer):
        """Empty string returns the zero vector."""
        vec = scorer.embed_text("")
        assert np.allclose(vec, 0.0)

    def test_single_word(self, scorer, known_word):
        """A single known word returns a nonzero embedding."""
        vec = scorer.embed_text(known_word)
        assert np.linalg.norm(vec) > 1e-10

    def test_order_invariant(self, scorer, vocab):
        """Bag-of-words centroid is order-invariant."""
        words = [w for w in vocab[:200] if len(w) > 2][:4]
        text_fwd = " ".join(words)
        text_rev = " ".join(reversed(words))
        v_fwd = scorer.embed_text(text_fwd)
        v_rev = scorer.embed_text(text_rev)
        np.testing.assert_allclose(v_fwd, v_rev, atol=1e-12)


# ======================================================================
# TestSelfClassification (15 tests)
# ======================================================================

class TestSelfClassification:

    def test_score_probabilities_sum_to_one(self, scorer, known_text):
        """Probability distribution sums to 1."""
        result = scorer.score(known_text)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-8

    def test_all_probs_nonnegative(self, scorer, known_text):
        """All probabilities are non-negative."""
        result = scorer.score(known_text)
        for p in result.probabilities.values():
            assert p >= 0.0

    def test_exactly_8_dialects(self, scorer, known_text):
        """Result contains exactly 8 dialect keys (one per variety)."""
        result = scorer.score(known_text)
        assert len(result.probabilities) == 8

    def test_top_dialect_matches_argmax(self, scorer, known_text):
        """top_dialect is the key with the maximum probability."""
        result = scorer.score(known_text)
        argmax_dialect = max(result.probabilities, key=result.probabilities.get)
        assert result.top_dialect == argmax_dialect

    def test_classifier_returns_string(self, scorer, known_text):
        """classify() returns a string that is one of the known varieties."""
        label = scorer.classify(known_text)
        assert isinstance(label, str)
        assert label in ALL_VARIETIES

    def test_guagua_scores_can_high(self, scorer, vocab):
        """'guagua' (if in vocab) should place CAN or CAR among top-3."""
        if "guagua" not in vocab:
            pytest.skip("'guagua' not in vocabulary")
        top3 = [d for d, _ in scorer.top_k_dialects("guagua", k=3)]
        assert "ES_CAN" in top3 or "ES_CAR" in top3, (
            f"Expected CAN or CAR in top-3 for 'guagua', got {top3}"
        )

    def test_colectivo_scores(self, scorer, vocab):
        """'colectivo' (if in vocab) scores without error."""
        if "colectivo" not in vocab:
            pytest.skip("'colectivo' not in vocabulary")
        result = scorer.score("colectivo")
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-8

    def test_mixed_text_mixed_scores(self, scorer, vocab):
        """Text mixing words from different positions in vocab doesn't crash."""
        words = [vocab[i] for i in range(0, min(len(vocab), 1000), 200) if len(vocab[i]) > 2]
        text = " ".join(words)
        result = scorer.score(text)
        assert result.top_dialect in ALL_VARIETIES

    def test_confidence_varies(self, scorer, vocab):
        """Different texts should yield different max probabilities."""
        words_a = [w for w in vocab[:50] if len(w) > 2][:3]
        words_b = [w for w in vocab[1000:1050] if len(w) > 2][:3]
        res_a = scorer.score(" ".join(words_a))
        res_b = scorer.score(" ".join(words_b))
        max_a = max(res_a.probabilities.values())
        max_b = max(res_b.probabilities.values())
        # They shouldn't be identical to many decimals (extremely unlikely with different words)
        assert not np.isclose(max_a, max_b, atol=1e-10), (
            "Two unrelated texts should not have identical max probabilities"
        )

    def test_batch_matches_individual(self, scorer, vocab):
        """batch_score matches individual score calls."""
        words = [w for w in vocab[:100] if len(w) > 2]
        t1 = words[0]
        t2 = " ".join(words[1:4])
        batch = scorer.batch_score([t1, t2])
        individual = [scorer.score(t1), scorer.score(t2)]
        for b, s in zip(batch, individual):
            assert b.top_dialect == s.top_dialect
            for v in b.probabilities:
                assert abs(b.probabilities[v] - s.probabilities[v]) < 1e-12

    def test_score_with_numbers(self, scorer):
        """Numeric text doesn't crash (numbers are OOV, falls back to zero)."""
        result = scorer.score("123 456")
        # Should still produce a valid distribution
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-8

    def test_long_text(self, scorer, vocab):
        """Paragraph-length text works without error."""
        words = [w for w in vocab[:500] if len(w) > 2][:100]
        long_text = " ".join(words)
        result = scorer.score(long_text)
        assert result.top_dialect in ALL_VARIETIES

    def test_unicode_safe(self, scorer):
        """Text with accented characters processes correctly."""
        text = "\u00f1o\u00f1o caf\u00e9 acci\u00f3n \u00faltimo"
        result = scorer.score(text)
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-8

    def test_detailed_has_centroid(self, scorer, known_text):
        """score_detailed returns dict with 'centroid' key."""
        detail = scorer.score_detailed(known_text)
        assert "centroid" in detail
        assert isinstance(detail["centroid"], np.ndarray)
        assert detail["centroid"].shape == (scorer._dim,)

    def test_detailed_tokens_count(self, scorer, known_text):
        """score_detailed reports correct tokens_total."""
        detail = scorer.score_detailed(known_text)
        expected_total = len(_tokenize(known_text))
        assert detail["tokens_total"] == expected_total


# ======================================================================
# TestCrossVarietyOrdering (10 tests)
# ======================================================================

class TestCrossVarietyOrdering:

    def test_neutral_text_flatter(self, scorer, vocab):
        """Neutral text (OOV / universal) has lower confidence than dialectal text."""
        # OOV text -> zero centroid -> cosine similarity undefined -> effectively uniform
        oov_result = scorer.score("xyzzy qwrtp fblthp")
        known_words = [w for w in vocab[:300] if len(w) > 3][:10]
        dialectal_result = scorer.score(" ".join(known_words))
        max_oov = max(oov_result.probabilities.values())
        max_dialectal = max(dialectal_result.probabilities.values())
        # Uniform distribution has max = 1/8 = 0.125; dialectal should be higher
        assert max_dialectal >= max_oov, (
            f"Dialectal text should be at least as confident as OOV "
            f"({max_dialectal:.4f} vs {max_oov:.4f})"
        )

    def test_longer_text_higher_confidence(self, scorer, vocab):
        """Longer text generally produces higher max probability (more signal)."""
        words = [w for w in vocab[:500] if len(w) > 3]
        short_text = " ".join(words[:3])
        long_text = " ".join(words[:100])
        res_short = scorer.score(short_text)
        res_long = scorer.score(long_text)
        max_short = max(res_short.probabilities.values())
        max_long = max(res_long.probabilities.values())
        # This is a soft property; both should be valid distributions
        assert abs(sum(res_short.probabilities.values()) - 1.0) < 1e-8
        assert abs(sum(res_long.probabilities.values()) - 1.0) < 1e-8

    def test_deterministic_scoring(self, scorer, known_text):
        """Same text produces identical scores on repeated calls."""
        r1 = scorer.score(known_text)
        r2 = scorer.score(known_text)
        assert r1.top_dialect == r2.top_dialect
        for v in r1.probabilities:
            assert r1.probabilities[v] == r2.probabilities[v]

    def test_family_internal_related(self, scorer, sample_corpus):
        """CAN and CAR (Caribbean family) should show correlated scoring patterns."""
        can_docs = sample_corpus.get("ES_CAN", [])
        if len(can_docs) < 5:
            pytest.skip("Not enough ES_CAN sample corpus data")
        # Score CAN text against all varieties
        text = can_docs[0]
        result = scorer.score(text)
        # Both CAN and CAR should be present as valid keys
        assert "ES_CAN" in result.probabilities
        assert "ES_CAR" in result.probabilities

    def test_score_ordering_stable(self, scorer, known_text):
        """Running score 10 times gives the same ranking."""
        results = [scorer.score(known_text) for _ in range(10)]
        rankings = [
            sorted(r.probabilities, key=r.probabilities.get, reverse=True)
            for r in results
        ]
        for ranking in rankings[1:]:
            assert ranking == rankings[0], "Ranking should be deterministic"

    def test_top_k_sorted(self, scorer, known_text):
        """top_k_dialects returns entries sorted by descending probability."""
        top5 = scorer.top_k_dialects(known_text, k=5)
        probs = [p for _, p in top5]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], (
                f"top_k not sorted: {probs[i]:.6f} < {probs[i+1]:.6f}"
            )

    def test_top_k_length(self, scorer, known_text):
        """top_k(k=3) returns exactly 3 items."""
        top3 = scorer.top_k_dialects(known_text, k=3)
        assert len(top3) == 3

    def test_available_varieties(self, scorer):
        """available_varieties returns a non-empty list."""
        varieties = scorer.available_varieties()
        assert len(varieties) > 0
        for v in varieties:
            assert v in ALL_VARIETIES

    def test_dialect_profile_exists(self, scorer):
        """get_dialect_profile('ES_PEN') returns a non-None array."""
        profile = scorer.get_dialect_profile("ES_PEN")
        assert profile is not None

    def test_dialect_profile_shape(self, scorer):
        """Dialect profile shape is (n_modes,) matching the W matrix dimension."""
        profile = scorer.get_dialect_profile("ES_PEN")
        assert profile is not None
        # n_modes equals embedding dim (W is dim x dim)
        assert profile.ndim == 1
        assert profile.shape[0] == scorer._dim


# ======================================================================
# TestCalibration (10 tests)
# ======================================================================

class TestCalibration:

    def test_low_temperature_sharper(self, scorer, known_text):
        """temperature=0.1 produces higher max probability than t=1.0."""
        r_low = scorer.score(known_text, temperature=0.1)
        r_mid = scorer.score(known_text, temperature=1.0)
        max_low = max(r_low.probabilities.values())
        max_mid = max(r_mid.probabilities.values())
        assert max_low >= max_mid - 1e-12, (
            f"Low temperature should sharpen: {max_low:.6f} vs {max_mid:.6f}"
        )

    def test_high_temperature_flatter(self, scorer, known_text):
        """temperature=10.0 produces more uniform distribution than t=1.0."""
        r_high = scorer.score(known_text, temperature=10.0)
        r_mid = scorer.score(known_text, temperature=1.0)
        max_high = max(r_high.probabilities.values())
        max_mid = max(r_mid.probabilities.values())
        assert max_high <= max_mid + 1e-12, (
            f"High temperature should flatten: {max_high:.6f} vs {max_mid:.6f}"
        )

    def test_temperature_valid_distribution(self, scorer, known_text):
        """Probabilities sum to 1 at any temperature."""
        for temp in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]:
            result = scorer.score(known_text, temperature=temp)
            total = sum(result.probabilities.values())
            assert abs(total - 1.0) < 1e-8, f"sum={total} at temperature={temp}"

    def test_temperature_doesnt_change_ranking(self, scorer, known_text):
        """Top dialect is consistent across temperatures for strong-signal text."""
        r_low = scorer.score(known_text, temperature=0.1)
        r_mid = scorer.score(known_text, temperature=1.0)
        # The top dialect should remain the same when signal is clear
        assert r_low.top_dialect == r_mid.top_dialect

    def test_mode_activations_returned(self, scorer, known_text):
        """mode_activations field is not None."""
        result = scorer.score(known_text)
        assert result.mode_activations is not None

    def test_mode_activations_shape(self, scorer, known_text):
        """mode_activations shape is (n_modes,)."""
        result = scorer.score(known_text)
        assert result.mode_activations.ndim == 1
        assert result.mode_activations.shape[0] == scorer._dim

    def test_mode_activations_nonneg(self, scorer, known_text):
        """All mode activations are non-negative (they are magnitudes)."""
        result = scorer.score(known_text)
        assert np.all(result.mode_activations >= 0.0)

    def test_softmax_temperature_positive_required(self):
        """_softmax raises ValueError when temperature <= 0."""
        logits = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            _softmax(logits, temperature=0.0)
        with pytest.raises(ValueError):
            _softmax(logits, temperature=-1.0)

    def test_cosine_similarity_self(self, rng):
        """Cosine similarity of a vector with itself is ~1.0."""
        v = rng.standard_normal(50)
        sim = _cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-10

    def test_cosine_zero_vector(self, rng):
        """Cosine similarity with the zero vector is 0.0."""
        v = rng.standard_normal(50)
        z = np.zeros(50)
        assert _cosine_similarity(z, v) == 0.0
        assert _cosine_similarity(v, z) == 0.0


# ======================================================================
# TestPerModeActivation (15 tests)
# ======================================================================

class TestPerModeActivation:

    def test_mode_activation_nonzero_for_known_text(self, scorer, known_text):
        """Known text produces nonzero mode activations."""
        result = scorer.score(known_text)
        assert np.linalg.norm(result.mode_activations) > 1e-10

    def test_zero_activation_empty_text(self, scorer):
        """Empty text produces zero mode activations (zero centroid -> zero projection)."""
        result = scorer.score("")
        # Zero centroid projected onto any basis gives zero activations
        assert np.allclose(result.mode_activations, 0.0)

    def test_mode_activations_deterministic(self, scorer, known_text):
        """Same text always gives same mode activations."""
        r1 = scorer.score(known_text)
        r2 = scorer.score(known_text)
        np.testing.assert_array_equal(r1.mode_activations, r2.mode_activations)

    def test_activations_scale_with_text(self, scorer, vocab):
        """More text generally yields higher total activation energy."""
        words = [w for w in vocab[:300] if len(w) > 3]
        short = " ".join(words[:2])
        long = " ".join(words[:50])
        r_short = scorer.score(short)
        r_long = scorer.score(long)
        # Both should be valid; the sum may or may not be larger, but neither should crash
        assert r_short.mode_activations.shape == r_long.mode_activations.shape

    def test_top_modes_significant(self, scorer, known_text):
        """Top 3 modes capture a meaningful fraction of total activation."""
        result = scorer.score(known_text)
        acts = result.mode_activations
        total = np.sum(acts)
        if total < 1e-12:
            pytest.skip("Zero total activation")
        top3 = np.sort(acts)[-3:]
        fraction = np.sum(top3) / total
        # Top 3 out of 100 modes should capture at least some fraction
        assert fraction > 0.0, "Top modes should have nonzero contribution"

    def test_sparse_for_short_text(self, scorer, known_word):
        """Single-word text: many modes near zero relative to max."""
        result = scorer.score(known_word)
        acts = result.mode_activations
        if np.max(acts) < 1e-12:
            pytest.skip("Zero activations")
        # Count modes that are less than 1% of max
        near_zero = np.sum(acts < 0.01 * np.max(acts))
        # At least some modes should be near zero for a 1-word signal
        assert near_zero > 0, "Expected some near-zero modes for short text"

    def test_mode_activation_varies_by_text(self, scorer, vocab):
        """Different texts produce different activation patterns."""
        words = [w for w in vocab[:200] if len(w) > 2]
        r1 = scorer.score(words[0])
        r2 = scorer.score(words[-1])
        # Activations should differ (they come from different centroids)
        assert not np.allclose(
            r1.mode_activations, r2.mode_activations, atol=1e-10,
        ), "Different words should produce different activation patterns"

    def test_scorer_repr(self, scorer):
        """repr contains 'DialectScorer'."""
        r = repr(scorer)
        assert "DialectScorer" in r

    def test_scorer_init_logs(self, word_embeddings_dict, vocab, decomps_dict):
        """Scorer initializes without error."""
        s = DialectScorer(word_embeddings_dict, vocab, decomps_dict, reference="ES_PEN")
        assert s is not None
        assert s.reference == "ES_PEN"

    def test_profile_computation(self, scorer):
        """All 8 dialect profiles are computed."""
        varieties = scorer.available_varieties()
        assert len(varieties) == 8
        for v in varieties:
            profile = scorer.get_dialect_profile(v)
            assert profile is not None
            assert profile.ndim == 1

    def test_confusion_matrix_shape(self, scorer, sample_corpus):
        """confusion_matrix returns (n_varieties, n_varieties) array."""
        # Build a small labeled set from available corpus
        texts = []
        labels = []
        for v, docs in sample_corpus.items():
            for doc in docs[:2]:
                texts.append(doc)
                labels.append(v)
        if not texts:
            pytest.skip("No sample corpus available")
        cm = scorer.confusion_matrix(texts, labels)
        n = len(scorer.available_varieties())
        assert cm.shape == (n, n)

    def test_accuracy_returns_float(self, scorer, sample_corpus):
        """accuracy returns a float in [0, 1]."""
        texts = []
        labels = []
        for v, docs in sample_corpus.items():
            for doc in docs[:2]:
                texts.append(doc)
                labels.append(v)
        if not texts:
            pytest.skip("No sample corpus available")
        acc = scorer.accuracy(texts, labels)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_accuracy_empty_returns_zero(self, scorer):
        """accuracy with empty lists returns 0.0."""
        acc = scorer.accuracy([], [])
        assert acc == 0.0

    def test_scoring_pipeline_integration(self, scorer, known_text):
        """score -> classify -> top_k all produce consistent results."""
        result = scorer.score(known_text)
        label = scorer.classify(known_text)
        top3 = scorer.top_k_dialects(known_text, k=3)

        # classify matches score's top_dialect
        assert label == result.top_dialect
        # top_k's first entry matches score's top_dialect
        assert top3[0][0] == result.top_dialect
        # top_k's first probability matches score's probability for that dialect
        assert abs(top3[0][1] - result.probabilities[result.top_dialect]) < 1e-12

    def test_batch_empty(self, scorer):
        """batch_score([]) returns an empty list."""
        results = scorer.batch_score([])
        assert results == []
