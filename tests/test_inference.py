"""Inference Engine Unit Tests.
Test & Validation Plan Section 2 (Inference).
"""

import pytest

from src.inference.engine import TranslationInferenceEngine


def test_inference_engine_empty_input():
    engine = TranslationInferenceEngine(device="cpu")
    res = engine.translate_single(text="", src_lang="en", tgt_lang="hi")
    assert res["translated_text"] == ""
    assert res["latency_ms"] == 0.0


def test_inference_engine_unsupported_direction():
    engine = TranslationInferenceEngine(device="cpu")
    with pytest.raises(ValueError):
        engine.translate_single(text="Hello", src_lang="hi", tgt_lang="en")
