"""Cross-split Leakage Detection Utility.
Master Decision Register Section 5 & Spec Section 2 (TASK-006).
"""

import hashlib
from typing import Dict, Set

import pandas as pd


class LeakageDetector:
    """Detects and reports data leakage between train, validation, and test splits."""

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    @classmethod
    def check_splits_leakage(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        src_col: str = "source_text",
        tgt_col: str = "target_text",
    ) -> Dict[str, any]:
        """Perform exhaustive leakage detection across splits.

        Returns:
            leakage_report dictionary with overlap counts and indices.
        """

        def get_pair_hashes(df: pd.DataFrame) -> Set[str]:
            return {
                cls._hash_text(f"{row[src_col]} ||| {row[tgt_col]}") for _, row in df.iterrows()
            }

        def get_source_hashes(df: pd.DataFrame) -> Set[str]:
            return {cls._hash_text(str(text)) for text in df[src_col]}

        train_pair_hashes = get_pair_hashes(train_df)
        val_pair_hashes = get_pair_hashes(val_df)
        test_pair_hashes = get_pair_hashes(test_df)

        train_src_hashes = get_source_hashes(train_df)
        val_src_hashes = get_source_hashes(val_df)
        test_src_hashes = get_source_hashes(test_df)

        # Pair overlaps (Exact parallel duplicate across splits)
        train_val_pair_overlap = train_pair_hashes.intersection(val_pair_hashes)
        train_test_pair_overlap = train_pair_hashes.intersection(test_pair_hashes)
        val_test_pair_overlap = val_pair_hashes.intersection(test_pair_hashes)

        # Source text overlaps
        train_val_src_overlap = train_src_hashes.intersection(val_src_hashes)
        train_test_src_overlap = train_src_hashes.intersection(test_src_hashes)
        val_test_src_overlap = val_src_hashes.intersection(test_src_hashes)

        has_pair_leakage = (
            len(train_val_pair_overlap) > 0
            or len(train_test_pair_overlap) > 0
            or len(val_test_pair_overlap) > 0
        )
        has_source_leakage = (
            len(train_val_src_overlap) > 0
            or len(train_test_src_overlap) > 0
            or len(val_test_src_overlap) > 0
        )

        return {
            "has_leakage": has_pair_leakage or has_source_leakage,
            "has_pair_leakage": has_pair_leakage,
            "has_source_leakage": has_source_leakage,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "pair_overlaps": {
                "train_val": len(train_val_pair_overlap),
                "train_test": len(train_test_pair_overlap),
                "val_test": len(val_test_pair_overlap),
            },
            "source_overlaps": {
                "train_val": len(train_val_src_overlap),
                "train_test": len(train_test_src_overlap),
                "val_test": len(val_test_src_overlap),
            },
        }

    @classmethod
    def purge_source_leakage(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        src_col: str = "source_text",
    ) -> pd.DataFrame:
        """Purge any training rows whose source sentence appears in val or test splits."""
        protected_hashes = {cls._hash_text(str(text)) for text in val_df[src_col]}.union(
            {cls._hash_text(str(text)) for text in test_df[src_col]}
        )
        keep_mask = [
            cls._hash_text(str(text)) not in protected_hashes for text in train_df[src_col]
        ]
        return train_df[keep_mask].reset_index(drop=True)

    @classmethod
    def assert_no_leakage(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        src_col: str = "source_text",
        tgt_col: str = "target_text",
    ) -> None:
        """Raise AssertionError if ANY pair-level OR source-sentence cross-split leakage is detected."""
        report = cls.check_splits_leakage(train_df, val_df, test_df, src_col, tgt_col)
        if report["has_pair_leakage"]:
            raise AssertionError(
                f"Pair-level data leakage detected between splits! Overlaps: {report['pair_overlaps']}"
            )
        if report["has_source_leakage"]:
            raise AssertionError(
                f"Source-sentence data leakage detected between splits! Overlaps: {report['source_overlaps']}"
            )
