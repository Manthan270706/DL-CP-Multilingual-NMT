"""Data Cleaning and Normalization Pipeline.
Master Decision Register Section 5, PRD Section 14, TDD Section 6.
"""

import html
import re
import unicodedata
from typing import Dict, Optional, Tuple

import pandas as pd


class DataCleaner:
    """Deterministic parallel corpus cleaner and normalizer."""

    def __init__(
        self,
        min_chars: int = 1,
        max_chars: int = 1000,
        min_words: int = 1,
        max_words: int = 100,
        max_word_length_ratio: float = 3.5,
        normalize_unicode: str = "NFKC",
        strip_html: bool = True,
        remove_duplicate_pairs: bool = True,
        remove_empty: bool = True,
        remove_identical_src_tgt: bool = True,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_words = min_words
        self.max_words = max_words
        self.max_word_length_ratio = max_word_length_ratio
        self.normalize_unicode = normalize_unicode
        self.strip_html = strip_html
        self.remove_duplicate_pairs = remove_duplicate_pairs
        self.remove_empty = remove_empty
        self.remove_identical_src_tgt = remove_identical_src_tgt

        self._html_tag_regex = re.compile(r"<[^>]+>")
        self._whitespace_regex = re.compile(r"\s+")

    def normalize_text(self, text: Optional[str]) -> str:
        """Normalize a single text string."""
        if text is None or not isinstance(text, str):
            return ""

        # Unescape HTML entities
        text = html.unescape(text)

        # Strip HTML tags if enabled
        if self.strip_html:
            text = self._html_tag_regex.sub(" ", text)

        # Unicode normalization (NFKC)
        if self.normalize_unicode:
            text = unicodedata.normalize(self.normalize_unicode, text)

        # Normalize whitespaces
        text = self._whitespace_regex.sub(" ", text).strip()
        return text

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        src_col: str = "source_text",
        tgt_col: str = "target_text",
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Clean and filter a parallel translation DataFrame deterministically.

        Returns:
            (cleaned_df, stats_dict)
        """
        initial_count = len(df)
        stats = {
            "initial_rows": initial_count,
            "null_or_empty_removed": 0,
            "duplicate_pairs_removed": 0,
            "identical_src_tgt_removed": 0,
            "length_filtered": 0,
            "final_rows": 0,
        }

        if initial_count == 0:
            stats["final_rows"] = 0
            return df.copy(), stats

        # Work on a copy
        df_clean = df.copy()

        # Step 1: Normalize text columns
        df_clean[src_col] = df_clean[src_col].astype(str).apply(self.normalize_text)
        df_clean[tgt_col] = df_clean[tgt_col].astype(str).apply(self.normalize_text)

        # Step 2: Remove empty/null strings
        if self.remove_empty:
            valid_mask = (df_clean[src_col].str.len() >= self.min_chars) & (
                df_clean[tgt_col].str.len() >= self.min_chars
            )
            empty_count = (~valid_mask).sum()
            stats["null_or_empty_removed"] = int(empty_count)
            df_clean = df_clean[valid_mask]

        # Step 3: Remove identical source and target (if source != target language)
        if self.remove_identical_src_tgt:
            non_identical_mask = df_clean[src_col] != df_clean[tgt_col]
            identical_count = (~non_identical_mask).sum()
            stats["identical_src_tgt_removed"] = int(identical_count)
            df_clean = df_clean[non_identical_mask]

        # Step 4: Remove duplicate (source, target) pairs
        if self.remove_duplicate_pairs:
            before_dedup = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=[src_col, tgt_col], keep="first")
            stats["duplicate_pairs_removed"] = before_dedup - len(df_clean)

        # Step 5: Length and length-ratio filtering
        src_char_len = df_clean[src_col].str.len()
        tgt_char_len = df_clean[tgt_col].str.len()

        src_word_count = df_clean[src_col].str.split().apply(len)
        tgt_word_count = df_clean[tgt_col].str.split().apply(len)

        length_mask = (
            (src_char_len >= self.min_chars)
            & (src_char_len <= self.max_chars)
            & (tgt_char_len >= self.min_chars)
            & (tgt_char_len <= self.max_chars)
            & (src_word_count >= self.min_words)
            & (src_word_count <= self.max_words)
            & (tgt_word_count >= self.min_words)
            & (tgt_word_count <= self.max_words)
        )

        # Length ratio filtering
        ratio = src_word_count / tgt_word_count.clip(lower=1)
        ratio_mask = (ratio <= self.max_word_length_ratio) & (
            ratio >= (1.0 / self.max_word_length_ratio)
        )

        final_valid = length_mask & ratio_mask
        stats["length_filtered"] = int((~final_valid).sum())
        df_clean = df_clean[final_valid].reset_index(drop=True)

        stats["final_rows"] = len(df_clean)
        return df_clean, stats
