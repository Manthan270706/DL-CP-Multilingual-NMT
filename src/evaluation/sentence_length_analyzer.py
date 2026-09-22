"""Sentence Length Breakdown and Analysis.
Master Decision Register Section 8 & Spec Section 2 (TASK-018).
"""

from typing import Any, Dict

import pandas as pd

from src.evaluation.bleu import BleuEvaluator
from src.evaluation.meteor import MeteorEvaluator


class SentenceLengthAnalyzer:
    """Analyzes translation performance segmented across sentence length buckets."""

    BUCKETS = {
        "short": (1, 10),
        "medium": (11, 25),
        "long": (26, 50),
        "very_long": (51, 9999),
    }

    SMALL_SAMPLE_THRESHOLD = 30

    @classmethod
    def categorize_length(cls, text: str) -> str:
        """Determine bucket key from word count."""
        words = len(text.strip().split())
        for name, (low, high) in cls.BUCKETS.items():
            if low <= words <= high:
                return name
        return "very_long"

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        src_col: str = "source_text",
        ref_col: str = "target_text",
        pred_col: str = "prediction",
    ) -> Dict[str, Any]:
        """Compute BLEU and METEOR broken down by sentence length bucket."""
        df_eval = df.copy()
        df_eval["bucket"] = df_eval[src_col].apply(cls.categorize_length)

        bucket_results = {}
        for bucket_name, (low, high) in cls.BUCKETS.items():
            subset = df_eval[df_eval["bucket"] == bucket_name]
            count = len(subset)
            is_small = count < cls.SMALL_SAMPLE_THRESHOLD

            if count > 0:
                preds = subset[pred_col].tolist()
                refs = subset[ref_col].tolist()
                bleu_res = BleuEvaluator.compute(preds, refs)
                meteor_res = MeteorEvaluator.compute(preds, refs)
                bleu_score = bleu_res["score"]
                meteor_score = meteor_res["score"]
            else:
                bleu_score = 0.0
                meteor_score = 0.0

            bucket_results[bucket_name] = {
                "range_words": f"{low}-{high if high < 9999 else '+'}",
                "sample_count": count,
                "small_sample_flag": is_small,
                "bleu": bleu_score,
                "meteor": meteor_score,
            }

        return {
            "total_samples": len(df_eval),
            "buckets": bucket_results,
        }
