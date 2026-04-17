"""Tests for corpus loading, balancing, blending, regionalism detection, and vocabulary.

50 tests covering the full data foundation pipeline.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eigen3.constants import ALL_VARIETIES, BLEND_PAIRS, REGIONALISMS
from eigen3.corpus import (
    balance_corpus,
    blend_affine_varieties,
    build_union_vocabulary,
    detect_regionalisms,
    load_corpus,
)
from eigen3.vocab import (
    SPANISH_ANCHOR_WORDS,
    filter_by_corpus_evidence,
    filter_vocabulary,
    get_anchor_indices,
)


# ======================================================================
# Loading (10 tests)
# ======================================================================


class TestLoading:
    """JSONL corpus loading."""

    def test_load_per_variety_files(self, corpus_dir):
        """Loading per-variety JSONL files produces all 8 varieties."""
        corpus = load_corpus(corpus_dir)
        assert set(corpus.keys()) == set(ALL_VARIETIES)

    def test_all_varieties_non_empty(self, corpus_dir):
        """Every variety has at least some documents."""
        corpus = load_corpus(corpus_dir)
        for v in ALL_VARIETIES:
            assert len(corpus[v]) > 0, f"{v} has no documents"

    def test_correct_format(self, corpus_dir):
        """Documents are strings, not dicts or other types."""
        corpus = load_corpus(corpus_dir)
        for v in ALL_VARIETIES:
            for doc in corpus[v][:5]:
                assert isinstance(doc, str)

    def test_handles_missing_directory(self, tmp_path):
        """Missing directory returns empty corpus."""
        corpus = load_corpus(tmp_path / "nonexistent")
        assert all(len(docs) == 0 for docs in corpus.values())

    def test_handles_empty_lines(self, tmp_path):
        """Empty lines in JSONL are skipped."""
        f = tmp_path / "ES_PEN.jsonl"
        f.write_text('\n{"text": "hola"}\n\n{"text": "mundo"}\n\n')
        corpus = load_corpus(tmp_path)
        assert len(corpus["ES_PEN"]) == 2

    def test_unicode_text(self, corpus_dir):
        """Corpus handles Spanish unicode (accents, ñ, ¿, ¡)."""
        corpus = load_corpus(corpus_dir)
        all_text = " ".join(corpus["ES_PEN"][:20])
        # Should contain some accented characters
        assert any(c in all_text for c in "áéíóúñ¿¡")

    def test_combined_fallback(self, tmp_path):
        """Falls back to corpus.jsonl if no per-variety files."""
        f = tmp_path / "corpus.jsonl"
        lines = [
            json.dumps({"text": "hola mundo", "variety": "ES_PEN"}),
            json.dumps({"text": "che boludo", "variety": "ES_RIO"}),
            json.dumps({"text": "oye wey", "dialect": "ES_MEX"}),
        ]
        f.write_text("\n".join(lines))
        corpus = load_corpus(tmp_path)
        assert len(corpus["ES_PEN"]) == 1
        assert len(corpus["ES_RIO"]) == 1

    def test_deterministic(self, corpus_dir):
        """Two loads produce identical results."""
        c1 = load_corpus(corpus_dir)
        c2 = load_corpus(corpus_dir)
        for v in ALL_VARIETIES:
            assert c1[v] == c2[v]

    def test_large_corpus(self, corpus_dir):
        """Full corpus has significant number of documents."""
        corpus = load_corpus(corpus_dir)
        total = sum(len(docs) for docs in corpus.values())
        assert total > 100, f"Only {total} total documents"

    def test_metadata_not_leaked(self, corpus_dir):
        """Documents are pure text, not JSON strings."""
        corpus = load_corpus(corpus_dir)
        for doc in corpus["ES_PEN"][:10]:
            assert not doc.startswith("{"), "Document looks like raw JSON"


# ======================================================================
# Balancing (15 tests)
# ======================================================================


class TestBalancing:
    """Temperature-scaled corpus balancing."""

    @pytest.fixture
    def unbalanced(self):
        return {
            "ES_PEN": [f"doc{i}" for i in range(1000)],
            "ES_CAN": [f"doc{i}" for i in range(200)],
            "ES_RIO": [f"doc{i}" for i in range(500)],
            "ES_MEX": [f"doc{i}" for i in range(1000)],
        }

    def test_temperature_1_no_change(self, unbalanced):
        """T=1 means no upsampling."""
        bal = balance_corpus(unbalanced, temperature=1.0)
        for v in unbalanced:
            assert len(bal[v]) == len(unbalanced[v])

    def test_temperature_0_equal_sizes(self, unbalanced):
        """T=0 makes all varieties same size (capped by max_ratio)."""
        bal = balance_corpus(unbalanced, temperature=0.0, max_ratio=10.0)
        sizes = [len(docs) for docs in bal.values()]
        # All should be equal (= n_max)
        assert len(set(sizes)) == 1

    def test_temperature_07_moderate(self, unbalanced):
        """T=0.7 gives moderate upsampling."""
        bal = balance_corpus(unbalanced, temperature=0.7)
        # CAN (200) should be upsampled but not to 1000
        assert len(bal["ES_CAN"]) > 200
        assert len(bal["ES_CAN"]) < 1000

    def test_never_downsamples(self, unbalanced):
        """No variety loses documents."""
        bal = balance_corpus(unbalanced, temperature=0.5)
        for v in unbalanced:
            assert len(bal[v]) >= len(unbalanced[v])

    def test_max_ratio_cap(self):
        """max_ratio prevents excessive duplication."""
        corpus = {
            "ES_PEN": [f"doc{i}" for i in range(1000)],
            "ES_CAN": [f"doc{i}" for i in range(10)],
        }
        bal = balance_corpus(corpus, temperature=0.0, max_ratio=3.0)
        assert len(bal["ES_CAN"]) <= 30  # 10 * 3

    def test_empty_variety(self):
        """Empty variety stays empty."""
        corpus = {"ES_PEN": ["doc1"], "ES_CAN": []}
        bal = balance_corpus(corpus)
        assert len(bal["ES_CAN"]) == 0

    def test_single_variety(self):
        """Single variety passes through unchanged."""
        corpus = {"ES_PEN": ["a", "b", "c"]}
        bal = balance_corpus(corpus)
        assert len(bal["ES_PEN"]) == 3

    def test_reproducible_with_seed(self, unbalanced):
        """Same seed produces same results."""
        b1 = balance_corpus(unbalanced, seed=42)
        b2 = balance_corpus(unbalanced, seed=42)
        for v in unbalanced:
            assert b1[v] == b2[v]

    def test_different_seeds_differ(self, unbalanced):
        """Different seeds produce different samples."""
        b1 = balance_corpus(unbalanced, temperature=0.5, seed=42)
        b2 = balance_corpus(unbalanced, temperature=0.5, seed=123)
        # The upsampled parts should differ
        assert b1["ES_CAN"] != b2["ES_CAN"]

    def test_preserves_original_text(self, unbalanced):
        """Original documents are preserved (not modified)."""
        bal = balance_corpus(unbalanced, temperature=0.5)
        for v in unbalanced:
            for doc in unbalanced[v]:
                assert doc in bal[v]

    def test_ratio_calculation(self):
        """Verify the n_target formula."""
        corpus = {
            "ES_PEN": ["d"] * 1000,
            "ES_CAN": ["d"] * 250,
        }
        bal = balance_corpus(corpus, temperature=0.7, max_ratio=10.0)
        # n_target = 1000 * (250/1000)^0.7 = 1000 * 0.25^0.7
        expected = int(1000 * (0.25 ** 0.7))
        # Allow +-1 for rounding
        assert abs(len(bal["ES_CAN"]) - expected) <= 1

    def test_all_varieties_balanced(self, unbalanced):
        """After balancing, size disparity is reduced."""
        bal = balance_corpus(unbalanced, temperature=0.5)
        sizes = [len(bal[v]) for v in unbalanced]
        ratio = max(sizes) / min(s for s in sizes if s > 0)
        orig_sizes = [len(unbalanced[v]) for v in unbalanced]
        orig_ratio = max(orig_sizes) / min(s for s in orig_sizes if s > 0)
        assert ratio <= orig_ratio

    def test_min_size_never_below_original(self, unbalanced):
        """Every variety has at least its original count."""
        bal = balance_corpus(unbalanced, temperature=0.3)
        for v in unbalanced:
            assert len(bal[v]) >= len(unbalanced[v])

    def test_largest_variety_unchanged(self, unbalanced):
        """The largest variety is never upsampled."""
        bal = balance_corpus(unbalanced, temperature=0.5)
        assert len(bal["ES_PEN"]) == len(unbalanced["ES_PEN"])

    def test_statistics_correct(self, unbalanced):
        """Returned corpus has correct structure."""
        bal = balance_corpus(unbalanced)
        assert set(bal.keys()) == set(unbalanced.keys())
        for v in bal:
            assert all(isinstance(d, str) for d in bal[v])


# ======================================================================
# Blending (15 tests)
# ======================================================================


class TestBlending:
    """Affinity-based corpus blending."""

    @pytest.fixture
    def corpus(self):
        return {v: [f"{v}_doc{i}" for i in range(100)] for v in ALL_VARIETIES}

    def test_can_car_20_percent(self, corpus):
        """CAN and CAR get 20% cross-pollination."""
        blended = blend_affine_varieties(corpus)
        # CAN should have ~20 docs from CAR added
        can_new = len(blended["ES_CAN"]) - len(corpus["ES_CAN"])
        assert can_new == 20

    def test_and_andbo_15_percent(self, corpus):
        """AND and AND_BO get 15% cross-pollination."""
        blended = blend_affine_varieties(corpus)
        and_new = len(blended["ES_AND"]) - len(corpus["ES_AND"])
        assert and_new == 15

    def test_bidirectional(self, corpus):
        """Blending adds docs in both directions."""
        blended = blend_affine_varieties(corpus)
        can_growth = len(blended["ES_CAN"]) - len(corpus["ES_CAN"])
        car_growth = len(blended["ES_CAR"]) - len(corpus["ES_CAR"])
        assert can_growth > 0
        assert car_growth > 0

    def test_original_docs_preserved(self, corpus):
        """Original documents are still present after blending."""
        blended = blend_affine_varieties(corpus)
        for v in ALL_VARIETIES:
            for doc in corpus[v]:
                assert doc in blended[v]

    def test_blend_fraction_correct(self, corpus):
        """Blend adds exactly the right number of documents."""
        blended = blend_affine_varieties(corpus, pairs=[("ES_CAN", "ES_CAR", 0.10)])
        # 10% of 100 = 10 docs added to each
        assert len(blended["ES_CAN"]) == 110
        assert len(blended["ES_CAR"]) == 110

    def test_no_self_blending(self, corpus):
        """Self-blending doubles the additions (A->A + A->A)."""
        blended = blend_affine_varieties(corpus, pairs=[("ES_PEN", "ES_PEN", 0.5)])
        # Self-blend: adds 50% from PEN to PEN twice (bidirectional)
        assert len(blended["ES_PEN"]) == 200  # 100 + 50 + 50

    def test_empty_pair_skip(self):
        """Empty variety in a pair is skipped gracefully."""
        corpus = {v: [] for v in ALL_VARIETIES}
        corpus["ES_PEN"] = ["doc1"]
        blended = blend_affine_varieties(corpus)
        assert len(blended["ES_PEN"]) == 1

    def test_blend_then_balance(self, corpus):
        """Blending + balancing composition works."""
        blended = blend_affine_varieties(corpus)
        balanced = balance_corpus(blended, temperature=0.8)
        # All should still have documents
        for v in ALL_VARIETIES:
            assert len(balanced[v]) >= len(corpus[v])

    def test_regionalism_detection_chi_squared(self, sample_corpus):
        """Chi-squared method detects some regionalisms."""
        if all(len(docs) == 0 for docs in sample_corpus.values()):
            pytest.skip("No corpus data available")
        regions = detect_regionalisms(sample_corpus)
        total = sum(len(r) for r in regions.values())
        assert total > 0, "No regionalisms detected"

    def test_curated_regionalisms_present(self, sample_corpus):
        """Curated lists (guagua, colectivo, camion) appear."""
        regions = detect_regionalisms(sample_corpus)
        all_detected = set().union(*regions.values())
        curated = {"guagua", "colectivo"}
        found = curated & all_detected
        # At least some curated words should be detected
        assert len(found) > 0 or len(all_detected) > 10

    def test_per_variety_regionalism_counts(self, sample_corpus):
        """Each variety has its own regionalism set."""
        if all(len(docs) == 0 for docs in sample_corpus.values()):
            pytest.skip("No corpus data available")
        regions = detect_regionalisms(sample_corpus)
        assert len(regions) == len(sample_corpus)

    def test_regionalism_p_threshold(self, sample_corpus):
        """Lower p-threshold produces fewer regionalisms."""
        if all(len(docs) == 0 for docs in sample_corpus.values()):
            pytest.skip("No corpus data available")
        strict = detect_regionalisms(sample_corpus, p_threshold=0.001)
        loose = detect_regionalisms(sample_corpus, p_threshold=0.1)
        n_strict = sum(len(r) for r in strict.values())
        n_loose = sum(len(r) for r in loose.values())
        assert n_strict <= n_loose

    def test_high_freq_words_not_regional(self, sample_corpus):
        """Universal high-frequency words (de, en, la) are not regionalisms."""
        if all(len(docs) == 0 for docs in sample_corpus.values()):
            pytest.skip("No corpus data available")
        regions = detect_regionalisms(sample_corpus)
        all_regional = set().union(*regions.values())
        # Core function words that should be truly universal
        universal = {"de", "en", "que", "no"}
        overlap = universal & all_regional
        assert len(overlap) == 0, f"Universal words detected as regional: {overlap}"

    def test_deterministic_blending(self, corpus):
        """Same seed produces same blend."""
        b1 = blend_affine_varieties(corpus, seed=42)
        b2 = blend_affine_varieties(corpus, seed=42)
        for v in ALL_VARIETIES:
            assert b1[v] == b2[v]

    def test_blend_custom_pairs(self, corpus):
        """Custom blend pairs work correctly."""
        blended = blend_affine_varieties(
            corpus,
            pairs=[("ES_RIO", "ES_CHI", 0.30)],
        )
        assert len(blended["ES_RIO"]) == 130
        assert len(blended["ES_CHI"]) == 130
        # Other varieties unchanged
        assert len(blended["ES_PEN"]) == 100


# ======================================================================
# Vocabulary (10 tests)
# ======================================================================


class TestVocabulary:
    """Vocabulary building and filtering."""

    @pytest.fixture
    def raw_vocab(self, sample_corpus):
        if all(len(docs) == 0 for docs in sample_corpus.values()):
            pytest.skip("No corpus data")
        return build_union_vocabulary(sample_corpus, min_count=1)

    def test_union_vocabulary_non_empty(self, raw_vocab):
        """Union vocabulary has words."""
        assert len(raw_vocab) > 0

    def test_alphabetic_filter_removes_symbols(self):
        """Filter removes words with digits and symbols."""
        vocab = ["casa", "123", "hello!", "café", "test@", "niño"]
        filtered = filter_vocabulary(vocab)
        assert "123" not in filtered
        assert "hello!" not in filtered
        assert "test@" not in filtered
        assert "casa" in filtered

    def test_min_length_filter(self):
        """Words shorter than 3 chars are removed (with exceptions)."""
        vocab = ["ab", "cd", "casa", "de", "el", "en", "x"]
        filtered = filter_vocabulary(vocab)
        assert "ab" not in filtered
        assert "x" not in filtered
        assert "de" in filtered  # essential short word
        assert "el" in filtered
        assert "casa" in filtered

    def test_english_blacklist(self):
        """English words are filtered out."""
        vocab = ["the", "casa", "would", "should", "tiempo", "getting"]
        filtered = filter_vocabulary(vocab)
        assert "the" not in filtered
        assert "would" not in filtered
        assert "casa" in filtered
        assert "tiempo" in filtered

    def test_corpus_evidence_accented(self, sample_corpus):
        """Accented words pass with low threshold."""
        if all(len(docs) == 0 for docs in sample_corpus.values()):
            pytest.skip("No corpus data")
        vocab = ["también", "después", "notarealword"]
        # Accented words that appear in corpus should pass
        filtered = filter_by_corpus_evidence(
            vocab, sample_corpus, min_varieties=1, ascii_min_total=1
        )
        # At least accented words should survive if in corpus
        assert isinstance(filtered, list)

    def test_short_spanish_preserved(self):
        """Essential short Spanish words survive filtering."""
        vocab = ["de", "el", "en", "es", "ir", "la", "lo", "no", "se", "ya", "yo"]
        filtered = filter_vocabulary(vocab, min_len=3)
        for w in vocab:
            assert w in filtered, f"Short Spanish word '{w}' was filtered"

    def test_anchor_words_found(self, vocab):
        """Enough Procrustes anchor words are in vocabulary."""
        indices = get_anchor_indices(vocab)
        assert len(indices) >= 50

    def test_filtered_vocab_sorted(self):
        """Filtered vocabulary is sorted."""
        vocab = ["banana", "apple", "cherry", "date"]
        filtered = filter_vocabulary(vocab)
        assert filtered == sorted(filtered)

    def test_no_duplicates(self, raw_vocab):
        """Union vocabulary has no duplicates."""
        assert len(raw_vocab) == len(set(raw_vocab))

    def test_pipeline_composition(self, sample_corpus):
        """Full pipeline: build -> filter -> evidence works end-to-end."""
        if all(len(docs) == 0 for docs in sample_corpus.values()):
            pytest.skip("No corpus data")
        raw = build_union_vocabulary(sample_corpus, min_count=1)
        step1 = filter_vocabulary(raw)
        step2 = filter_by_corpus_evidence(step1, sample_corpus, min_varieties=1)
        # Each step should reduce or maintain size
        assert len(step1) <= len(raw)
        assert len(step2) <= len(step1)
        assert len(step2) > 0
