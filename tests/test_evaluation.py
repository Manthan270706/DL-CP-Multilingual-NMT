"""Unit tests for Evaluation Metrics and Analyzers.
Test & Validation Plan Section 2 (Metrics).
"""

import pandas as pd

from src.evaluation.bleu import BleuEvaluator
from src.evaluation.error_analyzer import ErrorAnalyzer
from src.evaluation.meteor import MeteorEvaluator
from src.evaluation.sentence_length_analyzer import SentenceLengthAnalyzer


def test_bleu_computation():
    predictions = ["The cat is on the mat."]
    references = ["The cat is on the mat."]
    res = BleuEvaluator.compute(predictions, references)
    assert res["score"] > 95.0
    assert "version" in res


def test_meteor_computation():
    predictions = ["The cat is sitting on the mat."]
    references = ["The cat is on the mat."]
    res = MeteorEvaluator.compute(predictions, references)
    assert res["score"] > 0.0
    assert "version" in res


def test_error_analyzer():
    source = "The quick brown fox jumps over the lazy dog."
    reference = "Le renard brun rapide saute par-dessus le chien paresseux."
    pred_repetition = "Le renard renard renard saute."
    cat = ErrorAnalyzer.classify_error(source, reference, pred_repetition, "en", "fr")
    assert cat == "repetition"


def test_sentence_length_analyzer():
    df = pd.DataFrame(
        {
            "source_text": [
                "Short sentence",
                "This is a medium length sentence with several words in it.",
            ],
            "target_text": ["Phrase courte", "Ceci est une phrase de longueur moyenne."],
            "prediction": ["Phrase courte", "Ceci est une phrase de longueur moyenne."],
        }
    )
    res = SentenceLengthAnalyzer.analyze(df)
    assert "buckets" in res
    assert "short" in res["buckets"]
    assert "medium" in res["buckets"]
    assert res["buckets"]["short"]["sample_count"] == 1
    assert res["buckets"]["short"]["small_sample_flag"] is True
