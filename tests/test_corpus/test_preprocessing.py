"""Tests for corpus.preprocessing: noise cleaning, labeling, segmentation, filters."""

from __future__ import annotations

import pytest

from eigendialectos.constants import DialectCode, FeatureCategory
from eigendialectos.types import DialectSample

from eigendialectos.corpus.preprocessing.noise import (
    clean_text,
    collapse_whitespace,
    fix_encoding,
    handle_emojis,
    normalize_repetitions,
    normalize_unicode,
    remove_mentions,
    remove_urls,
)
from eigendialectos.corpus.preprocessing.labeling import DialectLabeler
from eigendialectos.corpus.preprocessing.segmentation import (
    segment_text,
    split_paragraphs,
    split_sentences,
)
from eigendialectos.corpus.preprocessing.filters import (
    apply_filters,
    confidence_filter,
    dedup_filter,
    language_filter,
    min_length_filter,
    quality_filter,
)


# ======================================================================
# Noise cleaning tests
# ======================================================================


class TestNoiseCleaning:
    """Tests for text normalisation and cleaning functions."""

    def test_normalize_unicode_nfc(self):
        decomposed = "n\u0303"
        result = normalize_unicode(decomposed)
        assert result == "\u00f1"

    def test_normalize_unicode_removes_control_chars(self):
        text = "Hola\x00 mundo\x01"
        result = normalize_unicode(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "Hola" in result

    def test_remove_urls(self):
        text = "Mira esto https://example.com/path y más"
        result = remove_urls(text)
        assert "https://" not in result
        assert "Mira esto" in result
        assert "y más" in result

    def test_remove_urls_www(self):
        text = "Visita www.example.org para más info"
        result = remove_urls(text)
        assert "www." not in result

    def test_remove_mentions(self):
        text = "Hola @usuario ¿cómo estás?"
        result = remove_mentions(text)
        assert "@usuario" not in result
        assert "Hola" in result

    def test_handle_emojis_remove(self):
        text = "Hola mundo \U0001F600 genial"
        result = handle_emojis(text, mode="remove")
        assert "\U0001F600" not in result

    def test_handle_emojis_replace(self):
        text = "Hola \U0001F600"
        result = handle_emojis(text, mode="replace")
        assert "<EMOJI>" in result

    def test_handle_emojis_keep(self):
        text = "Hola \U0001F600"
        result = handle_emojis(text, mode="keep")
        assert "\U0001F600" in result

    def test_normalize_repetitions(self):
        assert normalize_repetitions("hoooolaaaa") == "hoolaa"
        assert normalize_repetitions("aaa") == "aa"
        assert normalize_repetitions("ab") == "ab"

    def test_fix_encoding_mojibake(self):
        text = "Ã¡rbol de Navidad"
        result = fix_encoding(text)
        assert "árbol" in result

    def test_collapse_whitespace(self):
        text = "Hola    mundo   \n\n\n\n  fin"
        result = collapse_whitespace(text)
        assert "Hola mundo" in result
        assert "\n\n\n" not in result

    def test_clean_text_full_pipeline(self):
        text = "Hola @user https://t.co/abc \U0001F600 hoooola   mundo"
        result = clean_text(text)
        assert "@user" not in result
        assert "https://" not in result
        assert "\U0001F600" not in result

    def test_clean_text_options_disable(self):
        text = "Hola @user https://t.co/abc"
        result = clean_text(text, urls=False, mentions=False)
        assert "@user" in result
        assert "https://" in result

    def test_clean_text_preserves_spanish(self):
        text = "El niño comió ñoquis con aceituñas en la cañada."
        result = clean_text(text)
        assert "niño" in result
        assert "ñoquis" in result

    def test_clean_text_handles_empty(self):
        assert clean_text("") == ""


# ======================================================================
# Labeling tests
# ======================================================================


class TestDialectLabeler:
    """Tests for rule-based dialect labelling."""

    @pytest.fixture
    def labeler(self):
        return DialectLabeler()

    def test_rioplatense_detection(self, labeler):
        text = "Che, vos sabés que el pibe está laburando en el centro, ¿viste?"
        code, conf = labeler.label(text)
        assert code == DialectCode.ES_RIO
        assert conf > 0.0

    def test_chilean_detection(self, labeler):
        text = "Hueón, ¿cachai? La micro está la raja, po."
        code, conf = labeler.label(text)
        assert code == DialectCode.ES_CHI
        assert conf > 0.0

    def test_mexican_detection(self, labeler):
        text = "No manches, güey, esa chamba está bien chido, órale."
        code, conf = labeler.label(text)
        assert code == DialectCode.ES_MEX
        assert conf > 0.0

    def test_peninsular_detection(self, labeler):
        text = "Tío, ¿habéis visto el ordenador nuevo? Mola mogollón."
        code, conf = labeler.label(text)
        assert code == DialectCode.ES_PEN
        assert conf > 0.0

    def test_caribbean_detection(self, labeler):
        text = "Asere, ¿qué tú quieres? Esa vaina está chévere, ¿oíste?"
        code, conf = labeler.label(text)
        assert code == DialectCode.ES_CAR
        assert conf > 0.0

    def test_andalusian_detection(self, labeler):
        text = "Quillo, vamoh pa er centro, que eso ehtá mu lejos."
        code, conf = labeler.label(text)
        assert code == DialectCode.ES_AND
        assert conf > 0.0

    def test_andean_detection(self, labeler):
        text = "Pase nomás, caserito, sírvase nomás la sopita, pues."
        code, conf = labeler.label(text)
        assert code == DialectCode.ES_AND_BO
        assert conf > 0.0

    def test_canarian_detection(self, labeler):
        text = "Chacho, vamos a coger la guagua para comer gofio."
        code, conf = labeler.label(text)
        assert code == DialectCode.ES_CAN
        assert conf > 0.0

    def test_neutral_text_defaults_to_peninsular(self, labeler):
        text = "El clima es agradable en esta región del mundo."
        code, conf = labeler.label(text)
        assert code == DialectCode.ES_PEN
        assert conf == 0.0

    def test_detect_features_returns_dict(self, labeler):
        text = "Che, boludo, ¿vos tenés el laburo nuevo?"
        features = labeler.detect_features(text)
        assert isinstance(features, dict)
        all_features = []
        for cat_features in features.values():
            all_features.extend(cat_features)
        assert len(all_features) > 0

    def test_label_detailed(self, labeler):
        text = "Güey, esa chamba está padrísima, no manches."
        result = labeler.label_detailed(text)
        assert "prediction" in result
        assert "confidence" in result
        assert "scores" in result
        assert "features" in result
        assert result["prediction"] == DialectCode.ES_MEX

    def test_confidence_range(self, labeler):
        text = "Che, vos sabés que esto está bárbaro, ¿viste?"
        _, conf = labeler.label(text)
        assert 0.0 <= conf <= 1.0


# ======================================================================
# Segmentation tests
# ======================================================================


class TestSegmentation:
    """Tests for text segmentation utilities."""

    def test_split_sentences_basic(self):
        text = "Primera oración. Segunda oración. Tercera oración."
        sentences = split_sentences(text)
        assert len(sentences) >= 2

    def test_split_sentences_question_marks(self):
        text = "¿Cómo estás? Bien, gracias. ¿Y tú?"
        sentences = split_sentences(text)
        assert len(sentences) >= 2

    def test_split_sentences_abbreviations(self):
        text = "El Dr. García y la Sra. López fueron al hospital."
        sentences = split_sentences(text)
        assert len(sentences) == 1

    def test_split_sentences_empty(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_split_paragraphs(self):
        text = "Primer párrafo.\n\nSegundo párrafo.\n\nTercer párrafo."
        paragraphs = split_paragraphs(text)
        assert len(paragraphs) == 3

    def test_split_paragraphs_empty(self):
        assert split_paragraphs("") == []

    def test_segment_text_sentence_mode(self):
        text = "Hola mundo. Adiós mundo."
        segments = segment_text(text, method="sentence")
        assert len(segments) >= 1

    def test_segment_text_paragraph_mode(self):
        text = "Párrafo uno.\n\nPárrafo dos."
        segments = segment_text(text, method="paragraph")
        assert len(segments) == 2

    def test_segment_text_chunk_mode(self):
        text = "Esta es una oración. " * 20
        segments = segment_text(text, method="chunk", max_length=50)
        for chunk in segments:
            # Each chunk should be at most max_length, unless it is a single word
            assert len(chunk) <= 50 or len(chunk.split()) == 1

    def test_segment_text_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown segmentation method"):
            segment_text("text", method="invalid")


# ======================================================================
# Filters tests
# ======================================================================


def _make_sample(
    text: str,
    code: DialectCode = DialectCode.ES_PEN,
    conf: float = 1.0,
) -> DialectSample:
    return DialectSample(
        text=text,
        dialect_code=code,
        source_id="test",
        confidence=conf,
    )


class TestFilters:
    """Tests for sample filtering functions."""

    def test_min_length_filter(self):
        samples = [
            _make_sample("Hola"),
            _make_sample("Esta es una oración más larga para el test."),
        ]
        result = min_length_filter(samples, min_len=10)
        assert len(result) == 1
        assert "más larga" in result[0].text

    def test_language_filter_spanish(self):
        samples = [
            _make_sample("Esta oración está completamente en español."),
            _make_sample("这是一个完全用中文写的句子不是西班牙语的。"),
        ]
        result = language_filter(samples, lang="es")
        assert len(result) == 1
        assert "español" in result[0].text

    def test_language_filter_non_es_passthrough(self):
        samples = [_make_sample("Hello world")]
        result = language_filter(samples, lang="en")
        assert len(result) == 1

    def test_dedup_filter(self):
        samples = [
            _make_sample("Texto duplicado exacto."),
            _make_sample("Texto duplicado exacto."),
            _make_sample("Texto único diferente."),
        ]
        result = dedup_filter(samples)
        assert len(result) == 2

    def test_quality_filter_short(self):
        samples = [
            _make_sample("Sí"),
            _make_sample("Esta oración tiene suficientes palabras para pasar el filtro."),
        ]
        result = quality_filter(samples)
        assert len(result) == 1

    def test_quality_filter_non_alpha(self):
        samples = [
            _make_sample("123 456 789 000 111 222 333 444"),
            _make_sample("Una oración normal con palabras reales y suficientes."),
        ]
        result = quality_filter(samples)
        assert len(result) == 1

    def test_confidence_filter(self):
        samples = [
            _make_sample("Texto con baja confianza aqui presente.", conf=0.3),
            _make_sample("Texto con alta confianza aqui presente.", conf=0.8),
        ]
        result = confidence_filter(samples, min_confidence=0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.8

    def test_apply_filters_default(self):
        samples = [
            _make_sample("Corta"),
            _make_sample("Esta es una oración válida en español para el corpus grande."),
            _make_sample("Esta es una oración válida en español para el corpus grande."),
        ]
        result = apply_filters(samples)
        assert len(result) == 1

    def test_apply_filters_custom(self):
        samples = [
            _make_sample("Hola mundo, esta oración es corta pero válida para el corpus."),
            _make_sample(
                "Esta oración muy larga cumple con todos los criterios de "
                "calidad del filtro y debería ser aceptada."
            ),
        ]
        config = {
            "min_length": {"min_len": 70},
        }
        result = apply_filters(samples, config=config)
        assert len(result) == 1

    def test_apply_filters_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown filter"):
            apply_filters([], config={"nonexistent": {}})
