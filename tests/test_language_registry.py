"""Unit tests for LanguageRegistry.
Test & Validation Plan Section 2 (Language Registry).
"""

import pytest

from src.registry.language_registry import LanguageRegistry


def test_primary_source():
    assert LanguageRegistry.get_primary_source() == "en"


def test_supported_targets():
    targets = LanguageRegistry.get_supported_targets()
    assert set(targets) == {"hi", "mr", "fr", "de"}


def test_supported_directions():
    directions = LanguageRegistry.get_supported_directions()
    assert "en-hi" in directions
    assert "en-mr" in directions
    assert "en-fr" in directions
    assert "en-de" in directions


def test_nllb_codes():
    assert LanguageRegistry.get_nllb_code("en") == "eng_Latn"
    assert LanguageRegistry.get_nllb_code("hi") == "hin_Deva"
    assert LanguageRegistry.get_nllb_code("mr") == "mar_Deva"
    assert LanguageRegistry.get_nllb_code("fr") == "fra_Latn"
    assert LanguageRegistry.get_nllb_code("de") == "deu_Latn"


def test_mbart_codes():
    assert LanguageRegistry.get_mbart_code("en") == "en_XX"
    assert LanguageRegistry.get_mbart_code("hi") == "hi_IN"
    assert LanguageRegistry.get_mbart_code("mr") == "mr_IN"
    assert LanguageRegistry.get_mbart_code("fr") == "fr_XX"
    assert LanguageRegistry.get_mbart_code("de") == "de_DE"


def test_unsupported_direction():
    assert not LanguageRegistry.is_direction_supported("hi", "en")
    assert not LanguageRegistry.is_direction_supported("en", "es")
    with pytest.raises(ValueError):
        LanguageRegistry.validate_direction("hi", "en")
