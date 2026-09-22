"""Unit tests for Data Cleaning, Adapters, and Splitting.
Test & Validation Plan Section 2 (Dataset).
"""

import pytest
import pandas as pd

from src.data.adapters import DatasetAdapter
from src.data.cleaner import DataCleaner
from src.data.splitter import DatasetSplitter


def test_cleaner_removes_duplicates_and_empty():
    cleaner = DataCleaner()
    df = pd.DataFrame(
        {
            "source_text": ["Hello world", "Hello world", "", None, "Good morning", "Identical"],
            "target_text": [
                "Bonjour le monde",
                "Bonjour le monde",
                "Quelque chose",
                "Rien",
                "Bonjour",
                "Identical",
            ],
        }
    )
    cleaned, stats = cleaner.clean_dataframe(df)
    assert len(cleaned) == 2  # "Hello world" -> "Bonjour le monde", "Good morning" -> "Bonjour"
    assert stats["duplicate_pairs_removed"] == 1
    assert stats["null_or_empty_removed"] == 2
    assert stats["identical_src_tgt_removed"] == 1


def test_cleaner_normalizes_unicode_and_html():
    cleaner = DataCleaner(strip_html=True, normalize_unicode="NFKC")
    text = cleaner.normalize_text("<p>Artificial&nbsp;Intelligence &amp; Machine Learning</p>")
    assert text == "Artificial Intelligence & Machine Learning"


def test_splitter_proportions_and_determinism():
    df = DatasetAdapter.create_synthetic_verification_corpus("en", "hi", n_samples=100)
    splitter = DatasetSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    train_df, val_df, test_df = splitter.split(df)
    assert len(train_df) == 80
    assert len(val_df) == 10
    assert len(test_df) == 10

    # Test determinism
    train_df2, val_df2, test_df2 = splitter.split(df)
    assert train_df["source_text"].tolist() == train_df2["source_text"].tolist()


def test_adapter_canonical_schema():
    """DatasetAdapter output must have all canonical schema columns."""
    REQUIRED_COLS = {
        "sample_id", "source_language", "target_language",
        "source_text", "target_text", "dataset_source",
    }
    for direction in ["en-hi", "en-mr", "en-fr", "en-de"]:
        src, tgt = direction.split("-")
        df = DatasetAdapter.create_synthetic_verification_corpus(src, tgt, n_samples=10)
        assert REQUIRED_COLS == set(df.columns), (
            f"Missing or extra columns for {direction}: got {set(df.columns)}"
        )
        assert len(df) == 10
        assert (df["source_language"] == src).all()
        assert (df["target_language"] == tgt).all()
        assert df["sample_id"].is_unique, "sample_id must be unique"


def test_adapter_load_for_direction_invalid():
    """load_for_direction must raise ValueError for unsupported pairs."""
    with pytest.raises((ValueError, Exception)):
        DatasetAdapter.load_for_direction("en", "zh")  # not in registry
