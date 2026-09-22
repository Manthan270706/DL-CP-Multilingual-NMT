"""Deterministic Dataset Splitting Module.
Master Decision Register Section 5 & Spec Section 2 (TASK-007).
"""

from typing import Tuple

import numpy as np
import pandas as pd

from src.data.leakage import LeakageDetector


class DatasetSplitter:
    """Splits cleaned datasets deterministically into 80/10/10 train/val/test splits."""

    def __init__(
        self,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42,
    ):
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {train_ratio} + {val_ratio} + {test_ratio})"
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(
        self,
        df: pd.DataFrame,
        src_col: str = "source_text",
        tgt_col: str = "target_text",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split a DataFrame into train, val, test with zero pair AND zero source leakage."""
        if len(df) < 10:
            raise ValueError(f"Dataset too small to split meaningfully ({len(df)} rows)")

        # Group by unique normalized source sentence hash to guarantee 0 source overlap across splits
        df = df.copy().reset_index(drop=True)
        df["_src_hash"] = [LeakageDetector._hash_text(str(s)) for s in df[src_col]]
        unique_hashes = df["_src_hash"].drop_duplicates().to_numpy()

        rng = np.random.RandomState(self.seed)
        shuffled_hashes = unique_hashes[rng.permutation(len(unique_hashes))]

        n_total_srcs = len(shuffled_hashes)
        n_train_srcs = int(n_total_srcs * self.train_ratio)
        n_val_srcs = int(n_total_srcs * self.val_ratio)

        train_hash_set = set(shuffled_hashes[:n_train_srcs])
        val_hash_set = set(shuffled_hashes[n_train_srcs : n_train_srcs + n_val_srcs])
        test_hash_set = set(shuffled_hashes[n_train_srcs + n_val_srcs :])

        train_df = df[df["_src_hash"].isin(train_hash_set)].drop(columns=["_src_hash"]).reset_index(drop=True)
        val_df = df[df["_src_hash"].isin(val_hash_set)].drop(columns=["_src_hash"]).reset_index(drop=True)
        test_df = df[df["_src_hash"].isin(test_hash_set)].drop(columns=["_src_hash"]).reset_index(drop=True)

        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        # Assert zero leakage at both pair level and source level
        LeakageDetector.assert_no_leakage(train_df, val_df, test_df, src_col, tgt_col)

        return train_df, val_df, test_df
