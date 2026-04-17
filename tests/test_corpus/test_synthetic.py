"""Tests for corpus.synthetic: fixtures, templates, and generator."""

from __future__ import annotations

import pytest

from eigendialectos.constants import DialectCode, FeatureCategory
from eigendialectos.types import CorpusSlice, DialectSample

from eigendialectos.corpus.synthetic.fixtures import (
    DIALECT_FEATURES,
    get_dialect_features,
    get_fixtures,
)
from eigendialectos.corpus.synthetic.templates import (
    DIALECT_TEMPLATES,
    DialectTemplate,
    get_template,
    list_templates,
)
from eigendialectos.corpus.synthetic.generator import (
    BASE_SENTENCES,
    SyntheticGenerator,
)


# ======================================================================
# Fixtures tests
# ======================================================================


class TestFixtures:
    """Tests for hand-crafted dialect fixture sentences."""

    def test_all_dialects_present(self):
        fixtures = get_fixtures()
        for code in DialectCode:
            assert code in fixtures, f"Missing dialect {code.value} in fixtures"

    def test_minimum_sentences_per_dialect(self):
        fixtures = get_fixtures()
        for code, samples in fixtures.items():
            assert len(samples) >= 15, (
                f"Dialect {code.value} has only {len(samples)} fixtures, "
                "expected at least 15"
            )

    def test_fixture_sample_types(self):
        fixtures = get_fixtures()
        for code, samples in fixtures.items():
            for s in samples:
                assert isinstance(s, DialectSample)
                assert s.dialect_code == code
                assert s.source_id == "synthetic_fixture"
                assert s.confidence == 1.0
                assert len(s.text) > 0

    def test_dialect_features_all_present(self):
        features = get_dialect_features()
        for code in DialectCode:
            assert code in features, f"Missing features for {code.value}"

    def test_dialect_features_categories(self):
        for code, cats in DIALECT_FEATURES.items():
            for cat in cats:
                assert isinstance(cat, FeatureCategory)
                assert len(cats[cat]) > 0, (
                    f"Empty feature list for {code.value}/{cat.value}"
                )

    def test_peninsular_has_vosotros(self):
        fixtures = get_fixtures()
        pen_texts = " ".join(s.text for s in fixtures[DialectCode.ES_PEN])
        assert "vosotros" in pen_texts.lower() or "habéis" in pen_texts.lower()

    def test_rioplatense_has_voseo(self):
        fixtures = get_fixtures()
        rio_texts = " ".join(s.text for s in fixtures[DialectCode.ES_RIO])
        rio_lower = rio_texts.lower()
        has_voseo = any(
            marker in rio_lower
            for marker in ["vos ", "tenés", "querés", "sabés", "che"]
        )
        assert has_voseo, "Rioplatense fixtures lack voseo markers"

    def test_canarian_has_guagua(self):
        fixtures = get_fixtures()
        can_texts = " ".join(s.text for s in fixtures[DialectCode.ES_CAN])
        assert "guagua" in can_texts.lower()

    def test_chilean_has_cachai(self):
        fixtures = get_fixtures()
        chi_texts = " ".join(s.text for s in fixtures[DialectCode.ES_CHI])
        assert "cachai" in chi_texts.lower()

    def test_mexican_has_guey(self):
        fixtures = get_fixtures()
        mex_texts = " ".join(s.text for s in fixtures[DialectCode.ES_MEX])
        mex_lower = mex_texts.lower()
        assert "güey" in mex_lower or "wey" in mex_lower

    def test_andalusian_has_aspiration(self):
        fixtures = get_fixtures()
        and_texts = " ".join(s.text for s in fixtures[DialectCode.ES_AND])
        and_lower = and_texts.lower()
        has_aspiration = any(
            marker in and_lower
            for marker in ["vamoh", "ehta", "lah", "quillo"]
        )
        assert has_aspiration, "Andalusian fixtures lack aspiration markers"

    def test_andean_has_pues(self):
        fixtures = get_fixtures()
        and_bo_texts = " ".join(s.text for s in fixtures[DialectCode.ES_AND_BO])
        and_bo_lower = and_bo_texts.lower()
        assert "pues" in and_bo_lower or "pe" in and_bo_lower

    def test_caribbean_has_markers(self):
        fixtures = get_fixtures()
        car_texts = " ".join(s.text for s in fixtures[DialectCode.ES_CAR])
        car_lower = car_texts.lower()
        has_marker = any(
            m in car_lower for m in ["chévere", "mijo", "pa'", "asere"]
        )
        assert has_marker, "Caribbean fixtures lack distinctive markers"

    def test_conftest_tiny_corpus_compatibility(self, tiny_corpus):
        for code in DialectCode:
            assert code in tiny_corpus
            assert len(tiny_corpus[code].samples) == 5


# ======================================================================
# Templates tests
# ======================================================================


class TestTemplates:
    """Tests for dialect transformation templates."""

    def test_all_dialects_have_templates(self):
        codes = list_templates()
        for code in DialectCode:
            assert code in codes, f"No template for {code.value}"

    def test_template_structure(self):
        for code in DialectCode:
            template = get_template(code)
            assert isinstance(template, DialectTemplate)
            assert isinstance(template.lexical, dict)
            assert isinstance(template.morphological, list)
            assert isinstance(template.pragmatic_markers, list)
            assert isinstance(template.phonological, list)

    def test_all_templates_have_pragmatic_markers(self):
        for code, template in DIALECT_TEMPLATES.items():
            assert len(template.pragmatic_markers) > 0, (
                f"No pragmatic markers for {code.value}"
            )

    def test_lexical_substitution_rioplatense(self):
        template = get_template(DialectCode.ES_RIO)
        result = template.apply_lexical("Vamos a coger el autobús.")
        assert "colectivo" in result
        assert "autobús" not in result

    def test_morphological_voseo(self):
        template = get_template(DialectCode.ES_RIO)
        result = template.apply_morphological("Tú tienes razón.")
        assert "tenés" in result

    def test_andalusian_phonological(self):
        template = get_template(DialectCode.ES_AND)
        text = "Vamos para el centro."
        result = template.apply_all(text)
        assert "pa" in result.lower() or "er" in result.lower()

    def test_chilean_voseo_mixto(self):
        template = get_template(DialectCode.ES_CHI)
        result = template.apply_morphological("Tú sabes la verdad.")
        assert "sabís" in result

    def test_canarian_ustedes(self):
        template = get_template(DialectCode.ES_CAN)
        result = template.apply_morphological("Vosotros habéis llegado.")
        assert "ustedes" in result.lower()
        assert "han" in result.lower()

    def test_apply_all_transforms_non_peninsular(self):
        base = "Vamos a coger el autobús para ir al centro."
        for code in DialectCode:
            if code == DialectCode.ES_PEN:
                continue
            template = get_template(code)
            result = template.apply_all(base)
            assert result != base, (
                f"Template for {code.value} did not transform the text"
            )

    def test_case_preservation(self):
        template = get_template(DialectCode.ES_RIO)
        result = template.apply_lexical("Autobús de la ciudad.")
        assert result.startswith("Colectivo")

    def test_get_template_invalid(self):
        with pytest.raises(KeyError):
            get_template("INVALID")  # type: ignore[arg-type]

    def test_template_apply_all_returns_string(self):
        test_input = "Vamos a coger el autobús para ir al centro."
        for code, template in DIALECT_TEMPLATES.items():
            result = template.apply_all(test_input)
            assert isinstance(result, str)
            assert len(result) > 0


# ======================================================================
# Generator tests
# ======================================================================


class TestSyntheticGenerator:
    """Tests for the SyntheticGenerator class."""

    @pytest.fixture
    def generator(self):
        return SyntheticGenerator(seed=42)

    def test_generate_returns_correct_count(self, generator):
        samples = generator.generate(10, DialectCode.ES_MEX)
        assert len(samples) == 10

    def test_generate_correct_dialect(self, generator):
        samples = generator.generate(5, DialectCode.ES_CHI)
        for s in samples:
            assert s.dialect_code == DialectCode.ES_CHI

    def test_generate_sample_type(self, generator):
        samples = generator.generate(3, DialectCode.ES_RIO)
        for s in samples:
            assert isinstance(s, DialectSample)
            assert s.source_id == "synthetic_generator"
            assert 0.0 < s.confidence <= 1.0
            assert len(s.text) > 0

    def test_generate_metadata(self, generator):
        samples = generator.generate(2, DialectCode.ES_AND)
        for s in samples:
            assert "base_sentence" in s.metadata
            assert "generation_index" in s.metadata

    def test_generate_all_dialects(self, generator):
        result = generator.generate_all(n_per_dialect=5)
        assert len(result) == len(DialectCode)
        for code in DialectCode:
            assert code in result
            assert isinstance(result[code], CorpusSlice)
            assert len(result[code].samples) == 5
            assert result[code].dialect_code == code

    def test_reproducibility(self):
        gen1 = SyntheticGenerator(seed=123)
        gen2 = SyntheticGenerator(seed=123)
        s1 = gen1.generate(10, DialectCode.ES_MEX)
        s2 = gen2.generate(10, DialectCode.ES_MEX)
        for a, b in zip(s1, s2):
            assert a.text == b.text

    def test_different_seeds_produce_different_output(self):
        gen1 = SyntheticGenerator(seed=1)
        gen2 = SyntheticGenerator(seed=2)
        s1 = gen1.generate(20, DialectCode.ES_CAR)
        s2 = gen2.generate(20, DialectCode.ES_CAR)
        texts1 = [s.text for s in s1]
        texts2 = [s.text for s in s2]
        assert texts1 != texts2

    def test_base_sentence_count(self, generator):
        assert generator.base_sentence_count >= 50

    def test_add_base_sentences(self, generator):
        before = generator.base_sentence_count
        generator.add_base_sentences(["Frase nueva para probar."])
        assert generator.base_sentence_count == before + 1

    def test_generate_more_than_base(self, generator):
        n = generator.base_sentence_count + 10
        samples = generator.generate(n, DialectCode.ES_PEN)
        assert len(samples) == n

    def test_invalid_dialect_raises(self, generator):
        with pytest.raises((ValueError, KeyError)):
            generator.generate(5, "INVALID")  # type: ignore[arg-type]
