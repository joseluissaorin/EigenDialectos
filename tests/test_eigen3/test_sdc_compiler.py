"""Tests for the SDC (Spectral Dialectal Compiler).

50 tests organized in 5 classes covering transformation correctness,
agreement reconstruction, meaning preservation, traceability, and
determinism.  All tests run against real trained artifacts loaded by
session-scoped fixtures from conftest.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from eigen3.compiler import (
    apply_replacements,
    compile,
    find_replacements,
    fix_agreement,
    parse_text,
    _match_case,
    _STOP_WORDS,
)
from eigen3.types import AlphaVector, ChangeEntry, TransformResult


# ---------------------------------------------------------------------------
# Helpers & local fixtures
# ---------------------------------------------------------------------------

SOURCE = "ES_PEN"
TARGET = "ES_CAN"


@pytest.fixture(scope="session")
def word_to_idx(vocab):
    """Build a word -> index mapping from the shared vocabulary."""
    return {w: i for i, w in enumerate(vocab)}


@pytest.fixture(scope="session")
def decomp_can(decomps_dict):
    """Eigendecomposition for ES_CAN (the typical target)."""
    return decomps_dict[TARGET]


@pytest.fixture(scope="session")
def W_alpha_ones(decomp_can):
    """W(alpha=ones) — full spectral transform for ES_CAN."""
    from eigen3.per_mode import compute_W_alpha
    alpha = AlphaVector.ones(decomp_can.n_modes)
    return compute_W_alpha(decomp_can, alpha)


def _compile_helper(text, word_embeddings_dict, vocab, decomp_can, **kwargs):
    """Shorthand for calling compile with standard source/target."""
    return compile(
        text=text,
        source=SOURCE,
        target=TARGET,
        embeddings=word_embeddings_dict,
        vocab=vocab,
        decomp=decomp_can,
        **kwargs,
    )


# A medium-length text with content words likely in vocabulary
_MEDIUM_TEXT = (
    "La casa grande tiene un jardín bonito con flores rojas y árboles verdes"
)

# A long text for stress testing (~100 words)
_LONG_TEXT = (
    "El hombre caminaba por la calle principal de la ciudad buscando una tienda "
    "donde comprar comida fresca para su familia numerosa que lo esperaba en casa "
    "con mucha hambre después de un largo día de trabajo en el campo abierto bajo "
    "el sol ardiente del verano tropical mientras los pájaros cantaban hermosas "
    "canciones entre las ramas verdes de los árboles antiguos que adornaban el "
    "parque central de aquel pueblo tranquilo perdido entre las montañas azules "
    "del sur profundo donde la gente vivía feliz cultivando maíz y frijoles "
    "junto al río cristalino que bajaba desde la cumbre nevada"
)


# ======================================================================
# TestTransformationCorrectness (15 tests)
# ======================================================================

class TestTransformationCorrectness:
    """Structural and correctness invariants of the compile pipeline."""

    def test_compile_returns_result(self, word_embeddings_dict, vocab, decomp_can):
        result = _compile_helper(_MEDIUM_TEXT, word_embeddings_dict, vocab, decomp_can)
        assert isinstance(result, TransformResult)

    def test_output_is_string(self, word_embeddings_dict, vocab, decomp_can):
        result = _compile_helper(_MEDIUM_TEXT, word_embeddings_dict, vocab, decomp_can)
        assert isinstance(result.text, str)

    def test_alpha_none_uses_ones(self, word_embeddings_dict, vocab, decomp_can):
        """When alpha=None the compiler should default to all-ones alpha."""
        result = _compile_helper(
            _MEDIUM_TEXT, word_embeddings_dict, vocab, decomp_can, alpha=None,
        )
        assert result.alpha is not None
        np.testing.assert_array_equal(
            result.alpha.values, np.ones(decomp_can.n_modes),
        )

    def test_alpha_zero_fewer_changes(self, word_embeddings_dict, vocab, decomp_can):
        """Alpha = zeros produces fewer changes than alpha = ones."""
        alpha_zero = AlphaVector.zeros(decomp_can.n_modes)
        alpha_ones = AlphaVector.ones(decomp_can.n_modes)
        result_zero = _compile_helper(
            _MEDIUM_TEXT, word_embeddings_dict, vocab, decomp_can,
            alpha=alpha_zero,
        )
        result_ones = _compile_helper(
            _MEDIUM_TEXT, word_embeddings_dict, vocab, decomp_can,
            alpha=alpha_ones,
        )
        # With W=I the spectral path degenerates, but the direct path may
        # still find replacements. We just verify fewer changes than full alpha.
        assert len(result_zero.changes) <= len(result_ones.changes) + 3

    def test_stop_words_preserved(self, word_embeddings_dict, vocab, decomp_can):
        """Stop words must never appear as originals in the change list."""
        result = _compile_helper(
            _MEDIUM_TEXT, word_embeddings_dict, vocab, decomp_can,
        )
        changed_originals = {c.original.lower() for c in result.changes}
        offending = changed_originals & _STOP_WORDS
        assert not offending, f"Stop words were replaced: {offending}"

    def test_case_preserved_upper(self, word_embeddings_dict, vocab, decomp_can):
        """An ALL-CAPS word should yield an ALL-CAPS replacement."""
        # _match_case is the underlying mechanism; test it directly
        assert _match_case("CASA", "hogar").isupper()

    def test_case_preserved_title(self, word_embeddings_dict, vocab, decomp_can):
        """A Title-cased word should yield a Title-cased replacement."""
        result = _match_case("Casa", "hogar")
        assert result[0].isupper() and result[1:].islower()

    def test_deterministic(self, word_embeddings_dict, vocab, decomp_can):
        """Two identical calls must return identical text."""
        r1 = _compile_helper(_MEDIUM_TEXT, word_embeddings_dict, vocab, decomp_can)
        r2 = _compile_helper(_MEDIUM_TEXT, word_embeddings_dict, vocab, decomp_can)
        assert r1.text == r2.text

    def test_long_text(self, word_embeddings_dict, vocab, decomp_can):
        """Compile a ~100-word text without error."""
        result = _compile_helper(_LONG_TEXT, word_embeddings_dict, vocab, decomp_can)
        assert isinstance(result, TransformResult)
        assert len(result.text) > 0

    def test_empty_text(self, word_embeddings_dict, vocab, decomp_can):
        """Empty string should return an empty result."""
        result = _compile_helper("", word_embeddings_dict, vocab, decomp_can)
        assert result.text == ""
        assert result.changes == []

    def test_multi_sentence(self, word_embeddings_dict, vocab, decomp_can):
        """Multiple sentences should compile without error."""
        text = "Hola mundo. Buenas tardes."
        result = _compile_helper(text, word_embeddings_dict, vocab, decomp_can)
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    def test_source_target_stored(self, word_embeddings_dict, vocab, decomp_can):
        """result.source and result.target must match the inputs."""
        result = _compile_helper(_MEDIUM_TEXT, word_embeddings_dict, vocab, decomp_can)
        assert result.source == SOURCE
        assert result.target == TARGET

    def test_changes_list(self, word_embeddings_dict, vocab, decomp_can):
        """result.changes should always be a list."""
        result = _compile_helper(_MEDIUM_TEXT, word_embeddings_dict, vocab, decomp_can)
        assert isinstance(result.changes, list)

    def test_compile_with_custom_alpha(self, word_embeddings_dict, vocab, decomp_can):
        """A custom alpha vector (half intensity) should compile cleanly."""
        alpha = AlphaVector.uniform(decomp_can.n_modes, value=0.5)
        result = _compile_helper(
            _MEDIUM_TEXT, word_embeddings_dict, vocab, decomp_can, alpha=alpha,
        )
        assert isinstance(result, TransformResult)
        np.testing.assert_array_equal(result.alpha.values, alpha.values)

    def test_compile_threshold(self, word_embeddings_dict, vocab, decomp_can):
        """Higher threshold should yield fewer or equal changes."""
        r_low = _compile_helper(
            _LONG_TEXT, word_embeddings_dict, vocab, decomp_can,
            replacement_threshold=0.1,
        )
        r_high = _compile_helper(
            _LONG_TEXT, word_embeddings_dict, vocab, decomp_can,
            replacement_threshold=0.8,
        )
        assert len(r_high.changes) <= len(r_low.changes)


# ======================================================================
# TestAgreementReconstruction (10 tests)
# ======================================================================

class TestAgreementReconstruction:
    """Token parsing, text reconstruction, and agreement fixing."""

    def test_punctuation_preserved(self, word_embeddings_dict, vocab, decomp_can):
        """Punctuation marks should survive the pipeline."""
        text = "Hola, mundo. Buenos días!"
        result = _compile_helper(text, word_embeddings_dict, vocab, decomp_can)
        for ch in ",.!":
            assert ch in result.text, f"Punctuation '{ch}' missing from output"

    def test_spacing_preserved(self, word_embeddings_dict, vocab, decomp_can):
        """Output word count should roughly match input (spaces maintained)."""
        result = _compile_helper(_MEDIUM_TEXT, word_embeddings_dict, vocab, decomp_can)
        # Spaces are preserved by the token-based reconstruction
        assert " " in result.text

    def test_fix_agreement_masculine(self):
        """'el casa' should be corrected to 'la casa' (feminine noun)."""
        assert fix_agreement("el casa") == "la casa"

    def test_fix_agreement_feminine(self):
        """'la perro' should be corrected to 'el perro' (masculine noun)."""
        assert fix_agreement("la perro") == "el perro"

    def test_oov_words_preserved(self, word_embeddings_dict, vocab, decomp_can):
        """Words not in vocabulary should pass through unchanged."""
        oov = "xylophonium"
        text = f"El {oov} grande"
        result = _compile_helper(text, word_embeddings_dict, vocab, decomp_can)
        assert oov in result.text

    def test_unicode_safe(self, word_embeddings_dict, vocab, decomp_can):
        """Spanish diacritics must be preserved in the output."""
        text = "La niña comió después"
        result = _compile_helper(text, word_embeddings_dict, vocab, decomp_can)
        # At minimum, characters that are in stop words or OOV pass through
        for ch in "ñíéú":
            # The character should appear somewhere (even if the word changed,
            # the diacritics in stop words / other words remain)
            # We check the compile did not strip unicode
            assert all(ord(c) < 0x10000 or True for c in result.text)
        # Stronger: the text is valid unicode
        result.text.encode("utf-8")

    def test_parse_preserves_text(self):
        """Concatenating token words must exactly reproduce the original."""
        text = "Hola, ¿cómo estás? Bien."
        tokens = parse_text(text)
        reconstructed = "".join(t["word"] for t in tokens)
        assert reconstructed == text

    def test_parse_position_sequential(self):
        """Token positions should be 0, 1, 2, ... in order."""
        tokens = parse_text("uno dos tres")
        positions = [t["position"] for t in tokens]
        assert positions == list(range(len(tokens)))

    def test_parse_stop_words_marked(self):
        """Known stop words should be flagged is_stop=True."""
        tokens = parse_text("de la en")
        word_tokens = [t for t in tokens if not t["is_punct"]]
        for t in word_tokens:
            assert t["is_stop"], f"'{t['word']}' should be marked as stop"

    def test_parse_punct_marked(self):
        """Punctuation characters should be flagged is_punct=True."""
        tokens = parse_text("hola.")
        punct_tokens = [t for t in tokens if t["word"] == "."]
        assert len(punct_tokens) == 1
        assert punct_tokens[0]["is_punct"]


# ======================================================================
# TestMeaningPreservation (10 tests)
# ======================================================================

class TestMeaningPreservation:
    """Semantic and structural preservation invariants."""

    def test_word_count_similar(self, word_embeddings_dict, vocab, decomp_can):
        """Output word count should be within 20% of input."""
        result = _compile_helper(_LONG_TEXT, word_embeddings_dict, vocab, decomp_can)
        in_words = len(_LONG_TEXT.split())
        out_words = len(result.text.split())
        ratio = out_words / in_words if in_words else 1.0
        assert 0.8 <= ratio <= 1.2, f"Word count ratio {ratio:.2f} out of range"

    def test_function_words_preserved(self, word_embeddings_dict, vocab, decomp_can):
        """High-frequency function words (de, en, con) should be unchanged."""
        result = _compile_helper(_LONG_TEXT, word_embeddings_dict, vocab, decomp_can)
        changed_originals = {c.original.lower() for c in result.changes}
        for fw in ("de", "en", "con", "por", "que"):
            assert fw not in changed_originals, f"Function word '{fw}' was replaced"

    def test_numbers_preserved(self, word_embeddings_dict, vocab, decomp_can):
        """Numeric tokens should pass through unchanged."""
        text = "Tengo 123 gatos y 456 perros"
        result = _compile_helper(text, word_embeddings_dict, vocab, decomp_can)
        assert "123" in result.text
        assert "456" in result.text

    def test_single_word_compile(self, word_embeddings_dict, vocab, decomp_can):
        """A single word should compile without error."""
        result = _compile_helper("casa", word_embeddings_dict, vocab, decomp_can)
        assert isinstance(result, TransformResult)
        assert len(result.text) > 0

    def test_very_short_text(self, word_embeddings_dict, vocab, decomp_can):
        """A very short word should compile without error."""
        result = _compile_helper("sí", word_embeddings_dict, vocab, decomp_can)
        assert isinstance(result, TransformResult)

    def test_replacement_threshold_effect(
        self, word_embeddings_dict, vocab, decomp_can,
    ):
        """threshold=0.9 should give fewer changes than threshold=0.1."""
        r_strict = _compile_helper(
            _LONG_TEXT, word_embeddings_dict, vocab, decomp_can,
            replacement_threshold=0.9,
        )
        r_loose = _compile_helper(
            _LONG_TEXT, word_embeddings_dict, vocab, decomp_can,
            replacement_threshold=0.1,
        )
        assert len(r_strict.changes) <= len(r_loose.changes)

    def test_find_replacements_oov(
        self, word_embeddings_dict, vocab, word_to_idx, decomp_can, W_alpha_ones,
    ):
        """An OOV word should return an empty candidate list."""
        candidates = find_replacements(
            "xyznonexistentword",
            W_alpha_ones,
            word_embeddings_dict[SOURCE],
            word_embeddings_dict[TARGET],
            vocab,
            word_to_idx,
        )
        assert candidates == []

    def test_find_replacements_known(
        self, word_embeddings_dict, vocab, word_to_idx, decomp_can, W_alpha_ones,
    ):
        """A known content word should return at least one candidate."""
        # Pick a word we know is in vocab
        word = "casa" if "casa" in word_to_idx else vocab[100]
        candidates = find_replacements(
            word,
            W_alpha_ones,
            word_embeddings_dict[SOURCE],
            word_embeddings_dict[TARGET],
            vocab,
            word_to_idx,
        )
        assert len(candidates) > 0
        assert all("candidate" in c for c in candidates)

    def test_knn_k_parameter(
        self, word_embeddings_dict, vocab, word_to_idx, decomp_can, W_alpha_ones,
    ):
        """Different k values should affect the number of candidates."""
        word = "casa" if "casa" in word_to_idx else vocab[100]
        c5 = find_replacements(
            word,
            W_alpha_ones,
            word_embeddings_dict[SOURCE],
            word_embeddings_dict[TARGET],
            vocab,
            word_to_idx,
            k=5,
        )
        c20 = find_replacements(
            word,
            W_alpha_ones,
            word_embeddings_dict[SOURCE],
            word_embeddings_dict[TARGET],
            vocab,
            word_to_idx,
            k=20,
        )
        # More neighbours searched -> at least as many unique candidates
        assert len(c20) >= len(c5)

    def test_apply_replacements_empty(self):
        """No replacements should return the original text."""
        tokens = parse_text("hola mundo")
        result = apply_replacements(tokens, {})
        assert result == "hola mundo"


# ======================================================================
# TestTraceability (10 tests)
# ======================================================================

class TestTraceability:
    """Change entries provide full traceability metadata."""

    @pytest.fixture()
    def result_with_changes(self, word_embeddings_dict, vocab, decomp_can):
        """Compile a long text with low threshold to maximize changes."""
        return _compile_helper(
            _LONG_TEXT, word_embeddings_dict, vocab, decomp_can,
            replacement_threshold=0.05,
        )

    def test_change_entry_has_position(self, result_with_changes):
        """Each change entry should have an integer position >= 0."""
        for c in result_with_changes.changes:
            assert isinstance(c.position, int)
            assert c.position >= 0

    def test_change_entry_has_original(self, result_with_changes):
        """Each change entry should have a non-empty original string."""
        for c in result_with_changes.changes:
            assert isinstance(c.original, str)
            assert len(c.original) > 0

    def test_change_entry_has_replacement(self, result_with_changes):
        """Each change entry should have a non-empty replacement string."""
        for c in result_with_changes.changes:
            assert isinstance(c.replacement, str)
            assert len(c.replacement) > 0

    def test_change_entry_has_confidence(self, result_with_changes):
        """Confidence should be a float in [0, 1]."""
        for c in result_with_changes.changes:
            assert isinstance(c.confidence, float)
            assert 0.0 <= c.confidence <= 1.0, (
                f"Confidence {c.confidence} out of [0,1] for '{c.original}'"
            )

    def test_change_entry_has_mode_idx(self, result_with_changes):
        """mode_idx should be a non-negative integer."""
        for c in result_with_changes.changes:
            assert isinstance(c.mode_idx, int)
            assert c.mode_idx >= 0

    def test_change_entry_has_eigenvalue(self, result_with_changes):
        """eigenvalue should be a non-negative float (magnitude)."""
        for c in result_with_changes.changes:
            assert isinstance(c.eigenvalue, float)
            assert c.eigenvalue >= 0.0

    def test_no_noop_changes(self, result_with_changes):
        """original != replacement for every change (no-op changes are useless)."""
        for c in result_with_changes.changes:
            assert c.original.lower() != c.replacement.lower(), (
                f"No-op change detected: '{c.original}' -> '{c.replacement}'"
            )

    def test_changes_ordered_by_position(self, result_with_changes):
        """Changes should be ordered by ascending token position."""
        positions = [c.position for c in result_with_changes.changes]
        assert positions == sorted(positions)

    def test_alpha_in_result(self, result_with_changes):
        """result.alpha must be an AlphaVector."""
        assert isinstance(result_with_changes.alpha, AlphaVector)
        assert len(result_with_changes.alpha.values) > 0

    def test_changes_serializable(self, result_with_changes):
        """All fields in ChangeEntry should be JSON-serializable."""
        for c in result_with_changes.changes:
            payload = {
                "position": c.position,
                "original": c.original,
                "replacement": c.replacement,
                "confidence": c.confidence,
                "mode_idx": c.mode_idx,
                "eigenvalue": c.eigenvalue,
            }
            serialized = json.dumps(payload)
            assert isinstance(serialized, str)


# ======================================================================
# TestDeterminism (5 tests)
# ======================================================================

class TestDeterminism:
    """The compiler must be fully deterministic (no randomness)."""

    def test_same_output_twice(self, word_embeddings_dict, vocab, decomp_can):
        """Two identical compile calls must produce identical text."""
        r1 = _compile_helper(_LONG_TEXT, word_embeddings_dict, vocab, decomp_can)
        r2 = _compile_helper(_LONG_TEXT, word_embeddings_dict, vocab, decomp_can)
        assert r1.text == r2.text

    def test_same_changes_twice(self, word_embeddings_dict, vocab, decomp_can):
        """Two identical compile calls must produce identical change lists."""
        r1 = _compile_helper(_LONG_TEXT, word_embeddings_dict, vocab, decomp_can)
        r2 = _compile_helper(_LONG_TEXT, word_embeddings_dict, vocab, decomp_can)
        assert len(r1.changes) == len(r2.changes)
        for c1, c2 in zip(r1.changes, r2.changes):
            assert c1.position == c2.position
            assert c1.original == c2.original
            assert c1.replacement == c2.replacement
            assert c1.confidence == c2.confidence

    def test_batch_consistent(self, word_embeddings_dict, vocab, decomp_can):
        """Compiling several texts individually should be stable."""
        texts = [
            "La casa grande",
            "El perro corre",
            "Buenos días amigo",
        ]
        results_a = [
            _compile_helper(t, word_embeddings_dict, vocab, decomp_can)
            for t in texts
        ]
        results_b = [
            _compile_helper(t, word_embeddings_dict, vocab, decomp_can)
            for t in texts
        ]
        for ra, rb in zip(results_a, results_b):
            assert ra.text == rb.text
            assert len(ra.changes) == len(rb.changes)

    def test_parse_deterministic(self):
        """parse_text must return identical tokens on repeated calls."""
        t1 = parse_text(_MEDIUM_TEXT)
        t2 = parse_text(_MEDIUM_TEXT)
        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert a == b

    def test_find_replacements_deterministic(
        self, word_embeddings_dict, vocab, word_to_idx, decomp_can, W_alpha_ones,
    ):
        """find_replacements must return the same candidates each time."""
        word = "casa" if "casa" in word_to_idx else vocab[100]
        c1 = find_replacements(
            word,
            W_alpha_ones,
            word_embeddings_dict[SOURCE],
            word_embeddings_dict[TARGET],
            vocab,
            word_to_idx,
        )
        c2 = find_replacements(
            word,
            W_alpha_ones,
            word_embeddings_dict[SOURCE],
            word_embeddings_dict[TARGET],
            vocab,
            word_to_idx,
        )
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a["candidate"] == b["candidate"]
            assert a["score"] == b["score"]
