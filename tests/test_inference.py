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


def test_inference_engine_interpretability_empty():
    engine = TranslationInferenceEngine(device="cpu")
    res = engine.translate_with_interpretability(text="", src_lang="en", tgt_lang="hi")
    assert res["translated_text"] == ""
    assert res["src_tokens"] == []
    assert res["tgt_tokens"] == []
    assert res["token_probabilities"] == []


def test_get_finetuned_checkpoint_path_resolution():
    engine = TranslationInferenceEngine(device="cpu")
    # Should resolve Marian fine-tuned checkpoint for EN-HI if present on disk
    path = engine.get_finetuned_checkpoint_path("marian", "en", "hi")
    assert path is None or "experiments" in path


def test_cross_attention_matrix_shape_alignment():
    engine = TranslationInferenceEngine(device="cpu")
    res = engine.translate_with_interpretability(
        text="The model translates text.",
        src_lang="en",
        tgt_lang="hi",
        model_id="marian",
    )
    if res["cross_attention_matrix"] is not None:
        # Number of rows must equal number of target tokens
        assert res["cross_attention_matrix"].shape[0] == len(res["tgt_tokens"])
        # Number of columns must equal number of source tokens
        assert res["cross_attention_matrix"].shape[1] == len(res["src_tokens"])


