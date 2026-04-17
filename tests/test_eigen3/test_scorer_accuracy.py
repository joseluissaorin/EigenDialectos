"""Accuracy & regression tests for DialectScorer on real v3_full embeddings.

Tests the scorer against:
  - Curated dialectal texts (pure, moderate, subtle)
  - Single-word regionalism classification
  - Morphological matching (plurals)
  - Real corpus documents (labeled evaluation)
  - Confidence calibration
  - Adversarial / tricky inputs
  - Regression benchmarks (per-dialect accuracy floors)
  - Confusion matrix analysis

Unlike test_dialect_scoring.py (which tests structural properties on v2
fixtures), this file validates actual classification quality on the
production v3_full embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# v3_full fixtures (session-scoped, loaded once)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]
_V3_DIR = _ROOT / "outputs" / "eigen3_full"

_SKIP_REASON = "v3_full embeddings not found (run full_pipeline.py first)"


def _v3_available() -> bool:
    return (_V3_DIR / "vocab.json").exists() and (_V3_DIR / "ES_PEN.npy").exists()


pytestmark = pytest.mark.skipif(not _v3_available(), reason=_SKIP_REASON)


@pytest.fixture(scope="module")
def v3_scorer():
    """Build DialectScorer from v3_full production embeddings."""
    import sys
    sys.path.insert(0, str(_ROOT / "src"))

    from eigen3.constants import ALL_VARIETIES
    from eigen3.transformation import compute_all_W
    from eigen3.decomposition import eigendecompose
    from eigen3.scorer import DialectScorer

    embs: dict[str, np.ndarray] = {}
    for v in ALL_VARIETIES:
        p = _V3_DIR / f"{v}.npy"
        if not p.exists():
            continue
        e = np.load(str(p))
        if e.shape[0] < e.shape[1]:
            e = e.T
        embs[v] = e.astype(np.float64)

    vocab = json.loads((_V3_DIR / "vocab.json").read_text())
    W_all = compute_all_W(embs)
    decomps = {v: eigendecompose(tm.W, variety=v) for v, tm in W_all.items()}
    return DialectScorer(embs, vocab, decomps)


@pytest.fixture(scope="module")
def v3_vocab():
    return json.loads((_V3_DIR / "vocab.json").read_text())


@pytest.fixture(scope="module")
def corpus_by_variety() -> dict[str, list[str]]:
    """Load first 100 docs per variety from processed corpus for evaluation."""
    from eigen3.constants import ALL_VARIETIES
    data_dir = _ROOT / "data" / "processed"
    corpus: dict[str, list[str]] = {}
    for v in ALL_VARIETIES:
        path = data_dir / f"{v}.jsonl"
        docs: list[str] = []
        if path.exists():
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if i >= 100:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    text = doc.get("text", "").strip()
                    if text and len(text) > 30:
                        docs.append(text)
                except json.JSONDecodeError:
                    continue
        corpus[v] = docs
    return corpus


# ======================================================================
# 1. PURE DIALECT CLASSIFICATION (heavy markers)
# ======================================================================

_PURE_TEXTS: list[tuple[str, str]] = [
    # --- ES_RIO ---
    ("ES_RIO", "Che boludo, ayer me tomé el bondi y me fui al laburo. Después nos juntamos con los pibes a tomar unos mates en la vereda."),
    ("ES_RIO", "La piba esa es muy grosa, labura todo el día y después se toma el colectivo con los chabones del barrio."),
    ("ES_RIO", "El chabón se morfó tres empanadas y después se tomó un mate con los pibes en el bondi."),
    # --- ES_MEX ---
    ("ES_MEX", "Órale güey, qué chido estuvo la fiesta. Los chavos se pusieron bien fresones con la chela y el mole."),
    ("ES_MEX", "El morro ese es bien naco pero le echa ganas a la chamba. Los cuates lo invitaron a la alberca."),
    ("ES_MEX", "La neta güey, esos tacos estuvieron bien chidos. El chamaco trajo unas chelas y nos pusimos a cotorrear."),
    # --- ES_PEN ---
    ("ES_PEN", "Tío, mola mogollón este piso. Vosotros flipáis con el curro que lleva el chaval. Es muy guay."),
    ("ES_PEN", "Los chavales del barrio quedaron para tomar zumo y comer patatas bravas. Menudo curro llevan los tíos."),
    ("ES_PEN", "El tío ese es un gilipollas, mola que le echen del curro. El chaval flipó con la quedada."),
    # --- ES_CHI ---
    ("ES_CHI", "Cachai weón, fui a un carrete terrible bacán. Los gallos estaban cuáticos tomando copete en la micro."),
    ("ES_CHI", "La polola del gallo es fome pero la luca que gana en la pega le alcanza para ir a carretes bacanes."),
    ("ES_CHI", "El huevón se fue a la pega altiro y después se tomó un copete con los cabros. Cachai que estaba cuático."),
    # --- ES_CAN ---
    ("ES_CAN", "Mira niño, coge la guagua hasta el centro y compra gofio y mojo para el sancocho. El pelete no para."),
    ("ES_CAN", "El machango ese cogió la guagua y se fue a comprar gofio y tuneras al mercado de Las Palmas."),
    # --- ES_CAR ---
    ("ES_CAR", "Chévere pana, esa vaina estuvo buenísima. El chamo se puso a bailar bachata y a tomar guarapo."),
    ("ES_CAR", "Asere, el bochinche con la jeva del chamo estuvo tremendo. Se tomaron unos guarapos después de la bachata."),
    # --- ES_AND_BO ---
    ("ES_AND_BO", "Causa, vamos a comer cuy con chicha y después nos tomamos un anticucho. Lleva tu chompa por el soroche."),
    ("ES_AND_BO", "Mi ñaño se fue a la chacra a buscar quinua y chuño. Después se tomó un anticucho con chicha en el mercado."),
    # --- ES_AND ---
    ("ES_AND", "Quillo, ayer estuve con el compae en la chirigota. Qué pisha más graciosa. Nos comimos un gazpacho y un salmorejo."),
    ("ES_AND", "Illo que bulla tienen estos chiquillos, parecen churumbeles. Nos comimos una mijilla de pescaíto frito."),
]


class TestPureDialect:
    """Heavy-marker texts: each should classify as the expected dialect with high confidence."""

    @pytest.mark.parametrize("expected,text", _PURE_TEXTS, ids=[
        f"{e}-{i}" for i, (e, _) in enumerate(_PURE_TEXTS)
    ])
    def test_pure_classification(self, v3_scorer, expected, text):
        result = v3_scorer.score(text)
        assert result.top_dialect == expected, (
            f"Expected {expected}, got {result.top_dialect} "
            f"(prob={result.probabilities.get(expected, 0):.4f})"
        )

    @pytest.mark.parametrize("expected,text", _PURE_TEXTS, ids=[
        f"{e}-{i}-conf" for i, (e, _) in enumerate(_PURE_TEXTS)
    ])
    def test_pure_high_confidence(self, v3_scorer, expected, text):
        result = v3_scorer.score(text)
        prob = result.probabilities[result.top_dialect]
        assert prob > 0.7, (
            f"Expected >70% confidence for {expected}, got {prob:.4f}"
        )


# ======================================================================
# 2. MODERATE DIALECT (fewer markers)
# ======================================================================

_MODERATE_TEXTS: list[tuple[str, str]] = [
    ("ES_RIO", "Ayer fuimos con los pibes al centro. Tomamos unos mates y después laburamos un rato en la oficina."),
    ("ES_MEX", "Los chavos fueron a la chamba temprano. Después se tomaron unas chelas en el antro de siempre."),
    ("ES_PEN", "El chaval vino al piso con su coche nuevo. Estuvimos currando toda la tarde y después tomamos un zumo."),
    ("ES_CHI", "El gallo fue a buscar pega en Santiago. Después se tomó un copete con los amigos en el carrete."),
    ("ES_CAN", "Cogimos la guagua hasta Las Palmas. Compramos gofio y papas para la cena. Hacía un pelete tremendo."),
    ("ES_CAR", "El chamo dijo que la vaina estaba chévere. Fuimos a bailar bachata con los panas del barrio."),
    ("ES_AND_BO", "Mi causa me llevó a comer cuy en el restaurante de la esquina. La chicha estaba bien rica."),
    ("ES_AND", "El quillo se fue a Sevilla a ver la feria. Se comió un gazpacho y vio una chirigota muy buena."),
]


class TestModerateDialect:
    """Texts with fewer markers: should still classify correctly."""

    @pytest.mark.parametrize("expected,text", _MODERATE_TEXTS, ids=[
        f"{e}" for e, _ in _MODERATE_TEXTS
    ])
    def test_moderate_classification(self, v3_scorer, expected, text):
        result = v3_scorer.score(text)
        assert result.top_dialect == expected, (
            f"Expected {expected}, got {result.top_dialect}"
        )

    @pytest.mark.parametrize("expected,text", _MODERATE_TEXTS, ids=[
        f"{e}-conf" for e, _ in _MODERATE_TEXTS
    ])
    def test_moderate_minimum_confidence(self, v3_scorer, expected, text):
        result = v3_scorer.score(text)
        assert result.probabilities.get(expected, 0) > 0.3, (
            f"Expected >30% for {expected}"
        )


# ======================================================================
# 3. SINGLE REGIONALISM WORDS
# ======================================================================

_SINGLE_WORDS: list[tuple[str, list[str]]] = [
    ("che", ["ES_RIO"]),
    ("boludo", ["ES_RIO"]),
    ("bondi", ["ES_RIO"]),
    ("güey", ["ES_MEX"]),
    ("chido", ["ES_MEX"]),
    ("chamba", ["ES_MEX"]),
    ("mola", ["ES_PEN"]),
    ("chaval", ["ES_PEN"]),
    ("flipar", ["ES_PEN"]),
    ("cachai", ["ES_CHI"]),
    ("bacán", ["ES_CHI"]),
    ("polola", ["ES_CHI"]),
    ("gofio", ["ES_CAN"]),
    ("guagua", ["ES_CAN", "ES_CAR"]),  # shared regionalism
    ("chévere", ["ES_CAR"]),
    ("vaina", ["ES_CAR"]),
    ("cuy", ["ES_AND_BO"]),
    ("chicha", ["ES_AND_BO"]),
    ("causa", ["ES_AND_BO"]),
    ("quillo", ["ES_AND"]),
    ("pisha", ["ES_AND"]),
    ("gazpacho", ["ES_AND", "ES_RIO"]),  # widely known food term
]


class TestSingleWord:
    """Single regionalism words should classify to their dialect."""

    @pytest.mark.parametrize("word,expected_dialects", _SINGLE_WORDS, ids=[
        w for w, _ in _SINGLE_WORDS
    ])
    def test_single_word_top_dialect(self, v3_scorer, v3_vocab, word, expected_dialects):
        if word not in v3_vocab:
            pytest.skip(f"'{word}' not in v3 vocabulary")
        result = v3_scorer.score(word)
        assert result.top_dialect in expected_dialects, (
            f"'{word}' → {result.top_dialect}, expected one of {expected_dialects}"
        )


# ======================================================================
# 4. MORPHOLOGICAL MATCHING (plurals)
# ======================================================================

_PLURAL_FORMS: list[tuple[str, str, str]] = [
    # (plural, expected_base, expected_dialect)
    ("pibes", "pibe", "ES_RIO"),
    ("boludos", "boludo", "ES_RIO"),
    ("chavos", "chavo", "ES_MEX"),
    ("tíos", "tío", "ES_PEN"),
    ("gallos", "gallo", "ES_CHI"),
    ("chavales", "chaval", "ES_PEN"),    # -es suffix
    ("pringaos", "pringao", "ES_AND"),   # -s suffix on already-reduced form
]


class TestMorphologicalMatching:
    """Plurals of known regionalisms should still match via stemming."""

    @pytest.mark.parametrize("plural,base,expected", _PLURAL_FORMS, ids=[
        p for p, _, _ in _PLURAL_FORMS
    ])
    def test_plural_resolves_to_dialect(self, v3_scorer, plural, base, expected):
        result = v3_scorer.score(plural)
        prob = result.probabilities.get(expected, 0)
        # Either top dialect matches OR expected is in top-3 with reasonable prob
        sorted_p = sorted(result.probabilities.items(), key=lambda x: -x[1])
        top3 = [v for v, _ in sorted_p[:3]]
        assert expected in top3, (
            f"'{plural}' (→{base}): expected {expected} in top-3, "
            f"got {top3} (prob={prob:.4f})"
        )

    def test_no_false_positive_stems(self, v3_scorer):
        """Common words ending in -s shouldn't falsely match regionalisms."""
        # "más", "tres", "nos" end in -s but their stems aren't regionalisms
        for word in ["más", "tres", "después"]:
            result = v3_scorer.score(word)
            # Should not have any single variety dominate suspiciously
            max_prob = max(result.probabilities.values())
            assert max_prob < 0.95, (
                f"'{word}' classified with {max_prob:.2%} confidence — "
                f"possible false stem match"
            )


# ======================================================================
# 5. CORPUS-BASED EVALUATION
# ======================================================================

class TestCorpusEvaluation:
    """Score real labeled corpus documents.

    Note: Most corpus docs are subtitle / Wikipedia text with few or no
    dialectal markers.  The scorer identifies dialect through markers, so
    accuracy on neutral text is fundamentally limited.  Random baseline
    is 12.5% (1/8).  These tests verify the system is above random and
    that no dialect is *systematically* misclassified.
    """

    def test_marker_rich_dialects_above_baseline(self, v3_scorer, corpus_by_variety):
        """Dialects with colloquial corpora (MEX, PEN, RIO) should beat random."""
        marker_rich = ["ES_MEX", "ES_PEN", "ES_RIO"]
        for variety in marker_rich:
            docs = corpus_by_variety.get(variety, [])
            if len(docs) < 10:
                continue
            n = min(len(docs), 50)
            correct = sum(1 for doc in docs[:n] if v3_scorer.classify(doc) == variety)
            acc = correct / n
            assert acc > 0.15, (
                f"{variety}: accuracy {acc:.1%} on own corpus (expected >15%)"
            )

    def test_overall_accuracy_beats_random(self, v3_scorer, corpus_by_variety):
        """Overall accuracy should be above random chance (12.5%)."""
        correct = 0
        total = 0
        for variety, docs in corpus_by_variety.items():
            for doc in docs[:30]:
                if v3_scorer.classify(doc) == variety:
                    correct += 1
                total += 1
        if total == 0:
            pytest.skip("No corpus data")
        acc = correct / total
        assert acc > 0.125, f"Overall accuracy {acc:.1%} not above random (12.5%)"

    def test_no_systematic_misclassification(self, v3_scorer, corpus_by_variety):
        """No dialect should classify as a single OTHER dialect >70% of the time."""
        from eigen3.constants import ALL_VARIETIES
        varieties = [v for v in ALL_VARIETIES if len(corpus_by_variety.get(v, [])) >= 10]

        for variety in varieties:
            docs = corpus_by_variety[variety][:30]
            if not docs:
                continue
            pred_counts: dict[str, int] = {}
            for doc in docs:
                pred = v3_scorer.classify(doc)
                pred_counts[pred] = pred_counts.get(pred, 0) + 1

            for pred_v, count in pred_counts.items():
                if pred_v == variety:
                    continue
                ratio = count / len(docs)
                assert ratio < 0.70, (
                    f"{variety} systematically misclassified as {pred_v} "
                    f"({ratio:.0%} of docs)"
                )

    def test_dialectal_docs_classified_better_than_neutral(
        self, v3_scorer, corpus_by_variety
    ):
        """Docs containing known regionalisms should classify more accurately
        than docs without any markers."""
        from eigen3.constants import REGIONALISMS
        from eigen3.scorer import _tokenize

        correct_with_markers = 0
        total_with_markers = 0
        correct_without = 0
        total_without = 0

        for variety, docs in corpus_by_variety.items():
            regs = REGIONALISMS.get(variety, set())
            for doc in docs[:50]:
                tokens = set(_tokenize(doc))
                has_marker = bool(tokens & regs)
                is_correct = v3_scorer.classify(doc) == variety

                if has_marker:
                    total_with_markers += 1
                    if is_correct:
                        correct_with_markers += 1
                else:
                    total_without += 1
                    if is_correct:
                        correct_without += 1

        if total_with_markers < 5:
            pytest.skip("Too few docs with regionalisms")

        acc_markers = correct_with_markers / total_with_markers
        acc_neutral = correct_without / total_without if total_without > 0 else 0
        assert acc_markers >= acc_neutral, (
            f"Docs with markers ({acc_markers:.1%}) should be classified "
            f"better than neutral docs ({acc_neutral:.1%})"
        )


# ======================================================================
# 6. CONFIDENCE CALIBRATION
# ======================================================================

class TestConfidenceCalibration:

    def test_strong_markers_high_confidence(self, v3_scorer):
        """Text with many markers → top probability > 90%."""
        text = "Che boludo, ayer me fui con los pibes al laburo en el bondi."
        result = v3_scorer.score(text)
        assert max(result.probabilities.values()) > 0.90

    def test_neutral_text_lower_confidence(self, v3_scorer):
        """Text with no markers → top probability < 70%."""
        text = "La casa es grande y tiene un jardín bonito con flores."
        result = v3_scorer.score(text)
        assert max(result.probabilities.values()) < 0.70

    def test_more_markers_higher_confidence(self, v3_scorer):
        """Adding more markers should increase (or maintain) confidence."""
        text_1 = "Fuimos a tomar mate ayer."
        text_3 = "Che boludo, fuimos a tomar mate con los pibes en la vereda."

        r1 = v3_scorer.score(text_1)
        r3 = v3_scorer.score(text_3)

        max1 = max(r1.probabilities.values())
        max3 = max(r3.probabilities.values())
        assert max3 >= max1, (
            f"More markers should increase confidence: "
            f"{max3:.4f} < {max1:.4f}"
        )

    def test_oov_returns_uniform(self, v3_scorer):
        """Completely OOV text should return near-uniform distribution."""
        result = v3_scorer.score("xyzzy qwrtp fblthp")
        probs = list(result.probabilities.values())
        # All should be approximately 1/8
        assert all(abs(p - 0.125) < 0.01 for p in probs)

    @pytest.mark.parametrize("temp", [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
    def test_temperature_valid_distribution(self, v3_scorer, temp):
        """Probabilities sum to 1 at any temperature."""
        result = v3_scorer.score("Che boludo laburo pibes mates", temperature=temp)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-8, f"sum={total} at temperature={temp}"

    def test_low_temp_sharper_than_high(self, v3_scorer):
        """Low temperature yields sharper distribution than high temperature."""
        text = "Che boludo, nos juntamos con los pibes a tomar mates."
        r_low = v3_scorer.score(text, temperature=0.1)
        r_high = v3_scorer.score(text, temperature=10.0)
        assert max(r_low.probabilities.values()) > max(r_high.probabilities.values())

    def test_temperature_preserves_ranking(self, v3_scorer):
        """Top dialect should be the same at different temperatures."""
        text = "Órale güey, qué chido. Los chavos fueron a la chamba."
        r_low = v3_scorer.score(text, temperature=0.1)
        r_mid = v3_scorer.score(text, temperature=1.0)
        r_high = v3_scorer.score(text, temperature=10.0)
        assert r_low.top_dialect == r_mid.top_dialect == r_high.top_dialect


# ======================================================================
# 7. ADVERSARIAL / TRICKY INPUTS
# ======================================================================

class TestAdversarial:

    def test_mixed_dialect_text(self, v3_scorer):
        """Text mixing markers from two dialects should classify as one of them."""
        text = "Che boludo, esa vaina está muy chida güey"
        result = v3_scorer.score(text)
        assert result.top_dialect in ("ES_RIO", "ES_MEX", "ES_CAR")

    def test_opposite_iberian_markers(self, v3_scorer):
        """Mixing PEN and AND markers: top should be one of them."""
        text = "Tío, el quillo ese mola mogollón. Qué pisha más majo."
        result = v3_scorer.score(text)
        assert result.top_dialect in ("ES_PEN", "ES_AND")

    def test_single_marker_in_long_neutral(self, v3_scorer):
        """One marker buried in a long neutral text should still register."""
        text = (
            "Ayer por la mañana fui a la tienda del centro a comprar "
            "cosas para la casa. Había mucha gente en la calle y los "
            "precios estaban bastante altos. Me tomé un café con leche "
            "en un bar muy bonito de la plaza. El boludo del camarero "
            "se equivocó con la cuenta pero al final todo bien."
        )
        result = v3_scorer.score(text)
        # RIO should at least be in top-3 thanks to "boludo"
        sorted_p = sorted(result.probabilities.items(), key=lambda x: -x[1])
        top3 = [v for v, _ in sorted_p[:3]]
        assert "ES_RIO" in top3

    def test_repeated_word_not_overweighted(self, v3_scorer):
        """Repeating a word shouldn't change classification vs saying it once."""
        r1 = v3_scorer.score("boludo laburo mate")
        r5 = v3_scorer.score("boludo boludo boludo boludo boludo laburo mate")
        # Both should classify as RIO
        assert r1.top_dialect == "ES_RIO"
        assert r5.top_dialect == "ES_RIO"

    def test_all_caps(self, v3_scorer):
        """ALL CAPS text should classify correctly (case-insensitive tokenizer)."""
        result = v3_scorer.score("CHE BOLUDO, NOS JUNTAMOS CON LOS PIBES")
        assert result.top_dialect == "ES_RIO"

    def test_heavy_punctuation(self, v3_scorer):
        """Punctuation should not break classification."""
        result = v3_scorer.score("¡¡¡Che!!! ¿¿¿Boludo??? ...laburo... ¡¡¡pibes!!!")
        assert result.top_dialect == "ES_RIO"

    def test_unicode_accents(self, v3_scorer):
        """Accented characters should work correctly."""
        result = v3_scorer.score("órale güey qué chido está la neta")
        assert result.top_dialect == "ES_MEX"

    def test_numbers_mixed_in(self, v3_scorer):
        """Numbers interspersed with dialect words shouldn't break things."""
        result = v3_scorer.score("Che boludo, son las 3 de la tarde y hay 25 pibes")
        assert result.top_dialect == "ES_RIO"

    def test_empty_string(self, v3_scorer):
        """Empty string should return valid uniform distribution."""
        result = v3_scorer.score("")
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-8
        assert np.allclose(result.mode_activations, 0.0)

    def test_only_stopwords(self, v3_scorer):
        """Text with only universal function words should not be very confident."""
        result = v3_scorer.score("de la en el que los las del por con una para")
        max_prob = max(result.probabilities.values())
        # Should not be super confident about any dialect
        assert max_prob < 0.85, (
            f"Stopword-only text classified with {max_prob:.2%} confidence"
        )


# ======================================================================
# 8. DIALECT FAMILY PROXIMITY
# ======================================================================

class TestDialectFamilies:
    """Verify that linguistically related dialects show expected patterns."""

    def test_can_car_atlantic_proximity(self, v3_scorer):
        """CAN text should have CAR among top-3 (Atlantic family)."""
        result = v3_scorer.score(
            "Cogimos la guagua y compramos gofio y mojo para el sancocho."
        )
        sorted_p = sorted(result.probabilities.items(), key=lambda x: -x[1])
        top3 = [v for v, _ in sorted_p[:3]]
        assert "ES_CAN" in top3, f"Expected CAN in top-3, got {top3}"
        # CAR should also be present due to shared "guagua"
        assert "ES_CAR" in [v for v, _ in sorted_p[:4]]

    def test_mex_distinguishable_from_car(self, v3_scorer):
        """MEX and CAR should be distinguished despite both being American."""
        r_mex = v3_scorer.score(
            "Los morros fueron a la chamba y después comieron mole con chela."
        )
        r_car = v3_scorer.score(
            "Los chamos fueron al bochinche y después comieron habichuelas con guarapo."
        )
        assert r_mex.top_dialect == "ES_MEX"
        assert r_car.top_dialect == "ES_CAR"

    def test_pen_distinguishable_from_and(self, v3_scorer):
        """PEN and AND should be distinguished despite geographic proximity."""
        r_pen = v3_scorer.score(
            "Tío, vosotros flipáis con lo guay que es currar en este piso."
        )
        r_and = v3_scorer.score(
            "Quillo, illo, ayer el pringao ese me dijo una bulla tremenda."
        )
        assert r_pen.top_dialect == "ES_PEN"
        assert r_and.top_dialect == "ES_AND"

    def test_rio_distinguishable_from_andbo(self, v3_scorer):
        """RIO and AND_BO should be distinguished (both South American)."""
        r_rio = v3_scorer.score(
            "Che boludo, vamos a morfar unas empanadas con birra en la vereda."
        )
        r_bo = v3_scorer.score(
            "Causa, vamos a comer cuy con quinua y chicha en la chacra."
        )
        assert r_rio.top_dialect == "ES_RIO"
        assert r_bo.top_dialect == "ES_AND_BO"

    def test_chi_distinguishable_from_rio(self, v3_scorer):
        """CHI and RIO should be distinguished (both Southern Cone)."""
        r_chi = v3_scorer.score(
            "Cachai weón, la polola del gallo es fome pero el carrete estuvo bacán."
        )
        r_rio = v3_scorer.score(
            "Che boludo, la mina del chabón es muy grosa pero el quilombo fue terrible."
        )
        assert r_chi.top_dialect == "ES_CHI"
        assert r_rio.top_dialect == "ES_RIO"

    def test_all_eight_dialects_distinguishable(self, v3_scorer):
        """One text per dialect: all 8 should classify correctly."""
        texts = {
            "ES_RIO": "Che boludo, tomamos unos mates con los pibes en la vereda.",
            "ES_MEX": "Órale güey, los chavos fueron a la chamba y comieron chelas.",
            "ES_PEN": "Tío, el chaval flipó con el piso nuevo. Mola mogollón el curro.",
            "ES_CHI": "Cachai weón, el carrete estuvo bacán. Los gallos tomaron copete.",
            "ES_CAN": "La guagua nos llevó al mercado. Compramos gofio y mojo.",
            "ES_CAR": "Chévere pana, el chamo se fue al bochinche a bailar bachata.",
            "ES_AND_BO": "Causa, comimos cuy con chicha y quinua en la chacra.",
            "ES_AND": "Quillo compae, nos comimos un gazpacho y vimos la chirigota.",
        }
        for expected, text in texts.items():
            result = v3_scorer.score(text)
            assert result.top_dialect == expected, (
                f"{expected} text classified as {result.top_dialect}"
            )


# ======================================================================
# 9. REAL-WORLD SENTENCES (no artificial stacking)
# ======================================================================

_REALISTIC_TEXTS: list[tuple[str, str]] = [
    ("ES_RIO", "Ayer a la noche nos juntamos con los pibes a mirar el partido. Después fuimos a tomar algo por el barrio y terminamos morfando en el bodegón de siempre."),
    ("ES_MEX", "Fíjate que ayer fui a la taquería de la esquina y pedí unos tacos al pastor bien chidos. El güero del mostrador me dijo que el mole estaba recién hecho."),
    ("ES_PEN", "Quedamos ayer los colegas para tomar unas cañas en el bar de siempre. El tío del mostrador nos dijo que había patatas bravas nuevas. Mola el sitio."),
    ("ES_CHI", "Ayer fuimos a un carrete en Viña. Había caleta de gente y harto copete. Un gallo se puso a cantar y el ambiente estuvo bacán toda la noche."),
    ("ES_CAN", "Bajamos al puerto a comer unas papas arrugadas con mojo picón. La guagua tardó media hora pero el paseo valió la pena. Hacía un pelete tremendo."),
    ("ES_CAR", "El pana llegó diciendo que la vaina estaba buenísima por el malecón. Nos fuimos a bailar salsa y a tomar unos tragos. El chamo ese sí sabe gozar."),
    ("ES_AND_BO", "Fuimos al mercado de La Paz con mi causa a buscar quinua y chuño para la sopa. El anticucho de la señora estaba riquísimo. La chicha de jora también."),
    ("ES_AND", "Ayer me fui a Triana a comerme un pescaíto frito con mi compae. El quillo del bar nos puso un rebujito que estaba de chipén. Qué arte tiene el pisha."),
]


class TestRealisticTexts:
    """Naturalistic texts that don't just stack regionalisms."""

    @pytest.mark.parametrize("expected,text", _REALISTIC_TEXTS, ids=[
        e for e, _ in _REALISTIC_TEXTS
    ])
    def test_realistic_classification(self, v3_scorer, expected, text):
        result = v3_scorer.score(text)
        assert result.top_dialect == expected, (
            f"Expected {expected}, got {result.top_dialect}"
        )


# ======================================================================
# 10. CONSISTENCY & INVARIANTS
# ======================================================================

class TestConsistency:

    def test_deterministic(self, v3_scorer):
        """Same text → identical results every time."""
        text = "Che boludo laburo pibes mates"
        r1 = v3_scorer.score(text)
        r2 = v3_scorer.score(text)
        assert r1.top_dialect == r2.top_dialect
        for v in r1.probabilities:
            assert r1.probabilities[v] == r2.probabilities[v]

    def test_word_order_invariant(self, v3_scorer):
        """Bag-of-words model: order shouldn't matter."""
        r_a = v3_scorer.score("che boludo laburo pibes")
        r_b = v3_scorer.score("pibes laburo boludo che")
        assert r_a.top_dialect == r_b.top_dialect

    def test_classify_matches_score(self, v3_scorer):
        """classify() must agree with score().top_dialect."""
        for text in ["Che boludo", "Güey chido", "Tío mola", "Cachai weón"]:
            result = v3_scorer.score(text)
            assert v3_scorer.classify(text) == result.top_dialect

    def test_top_k_matches_score(self, v3_scorer):
        """top_k_dialects ordering must match score probabilities."""
        text = "Órale güey chido neta chamba"
        result = v3_scorer.score(text)
        top5 = v3_scorer.top_k_dialects(text, k=5)
        sorted_p = sorted(result.probabilities.items(), key=lambda x: -x[1])[:5]
        for i in range(5):
            assert top5[i][0] == sorted_p[i][0]

    def test_batch_matches_individual(self, v3_scorer):
        """batch_score must produce identical results to individual calls."""
        texts = ["Che boludo laburo", "Güey chido chamba", "Tío mola curro"]
        batch = v3_scorer.batch_score(texts)
        for text, batch_result in zip(texts, batch):
            ind_result = v3_scorer.score(text)
            assert batch_result.top_dialect == ind_result.top_dialect
            for v in batch_result.probabilities:
                assert abs(batch_result.probabilities[v] - ind_result.probabilities[v]) < 1e-12

    def test_all_probs_nonnegative(self, v3_scorer):
        """No probability should ever be negative."""
        for text in ["Che boludo", "casa grande", "", "xyzzy", "123"]:
            result = v3_scorer.score(text)
            for v, p in result.probabilities.items():
                assert p >= 0.0, f"Negative prob {p} for {v} on '{text}'"

    def test_all_probs_sum_to_one(self, v3_scorer):
        """Probabilities must sum to 1.0 for any input."""
        for text in ["Che boludo laburo", "", "xyzzy", "casa", "!!!???..."]:
            result = v3_scorer.score(text)
            total = sum(result.probabilities.values())
            assert abs(total - 1.0) < 1e-8, f"sum={total} for '{text}'"

    def test_exactly_8_varieties(self, v3_scorer):
        """Result should contain exactly 8 dialect probabilities."""
        result = v3_scorer.score("casa bonita jardín")
        assert len(result.probabilities) == 8

    def test_mode_activations_shape(self, v3_scorer):
        """Mode activations must be 1-D with correct length."""
        result = v3_scorer.score("Che boludo laburo pibes")
        assert result.mode_activations.ndim == 1
        assert result.mode_activations.shape[0] == v3_scorer._dim

    def test_mode_activations_nonnegative(self, v3_scorer):
        """Mode activations are magnitudes, must be non-negative."""
        result = v3_scorer.score("Che boludo laburo pibes")
        assert np.all(result.mode_activations >= 0.0)


# ======================================================================
# 11. REGRESSION BENCHMARKS
# ======================================================================

class TestRegressionBenchmarks:
    """Hard-coded minimum accuracy thresholds — if these drop, something broke."""

    def test_pure_dialect_perfect_score(self, v3_scorer):
        """All 20 pure-dialect texts must classify correctly (100%)."""
        correct = 0
        for expected, text in _PURE_TEXTS:
            if v3_scorer.classify(text) == expected:
                correct += 1
        assert correct == len(_PURE_TEXTS), (
            f"Pure dialect: {correct}/{len(_PURE_TEXTS)} "
            f"(expected 100%, got {correct/len(_PURE_TEXTS):.0%})"
        )

    def test_moderate_dialect_all_correct(self, v3_scorer):
        """All 8 moderate-dialect texts must classify correctly."""
        correct = 0
        for expected, text in _MODERATE_TEXTS:
            if v3_scorer.classify(text) == expected:
                correct += 1
        assert correct == len(_MODERATE_TEXTS), (
            f"Moderate: {correct}/{len(_MODERATE_TEXTS)}"
        )

    def test_realistic_texts_all_correct(self, v3_scorer):
        """All 8 realistic texts must classify correctly."""
        correct = 0
        for expected, text in _REALISTIC_TEXTS:
            if v3_scorer.classify(text) == expected:
                correct += 1
        assert correct == len(_REALISTIC_TEXTS), (
            f"Realistic: {correct}/{len(_REALISTIC_TEXTS)}"
        )

    def test_single_word_accuracy_above_80pct(self, v3_scorer, v3_vocab):
        """At least 80% of single-word tests should classify correctly."""
        correct = 0
        tested = 0
        for word, expected_dialects in _SINGLE_WORDS:
            if word not in v3_vocab:
                continue
            tested += 1
            if v3_scorer.classify(word) in expected_dialects:
                correct += 1
        if tested == 0:
            pytest.skip("No single-word tests possible")
        acc = correct / tested
        assert acc >= 0.80, (
            f"Single-word accuracy: {acc:.0%} ({correct}/{tested}), expected ≥80%"
        )

    def test_corpus_accuracy_report(self, v3_scorer, corpus_by_variety):
        """Print per-dialect corpus accuracy (informational + above random)."""
        results: dict[str, dict] = {}
        total_correct = 0
        total_tested = 0

        for variety, docs in corpus_by_variety.items():
            n = min(len(docs), 50)
            if n < 5:
                continue
            correct = sum(
                1 for doc in docs[:n] if v3_scorer.classify(doc) == variety
            )
            acc = correct / n
            results[variety] = {"correct": correct, "total": n, "accuracy": acc}
            total_correct += correct
            total_tested += n

        # Print report for visibility in pytest -s output
        print("\n--- Corpus accuracy per dialect ---")
        for v in sorted(results):
            r = results[v]
            print(f"  {v}: {r['accuracy']:6.1%} ({r['correct']}/{r['total']})")

        if total_tested > 0:
            overall = total_correct / total_tested
            print(f"  OVERALL: {overall:.1%} ({total_correct}/{total_tested})")
            # Must beat random baseline (12.5%)
            assert overall > 0.125, (
                f"Overall accuracy {overall:.1%} below random baseline (12.5%)"
            )
