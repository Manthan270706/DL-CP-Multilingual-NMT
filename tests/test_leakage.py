"""Unit tests for Leakage Detector.
Test & Validation Plan Section 5 (Leakage Tests).
"""

import pandas as pd
import pytest

from src.data.leakage import LeakageDetector


def test_leakage_detector_catches_overlap():
    train_df = pd.DataFrame(
        {
            "source_text": ["The cat sat on the mat.", "Good morning."],
            "target_text": ["Le chat était assis sur le tapis.", "Bonjour."],
        }
    )
    val_df = pd.DataFrame(
        {
            "source_text": ["How are you?"],
            "target_text": ["Comment allez-vous?"],
        }
    )
    # test_df contains an exact duplicate of a training pair
    test_df = pd.DataFrame(
        {
            "source_text": ["Good morning."],
            "target_text": ["Bonjour."],
        }
    )

    report = LeakageDetector.check_splits_leakage(train_df, val_df, test_df)
    assert report["has_leakage"] is True
    assert report["pair_overlaps"]["train_test"] == 1

    with pytest.raises(AssertionError):
        LeakageDetector.assert_no_leakage(train_df, val_df, test_df)


def test_leakage_detector_passes_on_disjoint_sets():
    train_df = pd.DataFrame(
        {
            "source_text": ["Sentence one.", "Sentence two."],
            "target_text": ["Phrase un.", "Phrase deux."],
        }
    )
    val_df = pd.DataFrame(
        {
            "source_text": ["Sentence three."],
            "target_text": ["Phrase trois."],
        }
    )
    test_df = pd.DataFrame(
        {
            "source_text": ["Sentence four."],
            "target_text": ["Phrase quatre."],
        }
    )

    report = LeakageDetector.check_splits_leakage(train_df, val_df, test_df)
    assert report["has_leakage"] is False
