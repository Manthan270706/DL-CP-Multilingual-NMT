"""Structured Error Analysis Module.
Master Decision Register Section 8, PRD Section 20, TDD Section 21.
"""

import re
from typing import Dict

import pandas as pd


class ErrorAnalyzer:
    """Classifies translation errors into predefined categories."""

    CATEGORIES = [
        "grammar",
        "word_order",
        "named_entity",
        "ambiguity",
        "missing_words",
        "extra_words",
        "repetition",
        "untranslated_words",
        "long_sentence_failure",
    ]

    @classmethod
    def classify_error(
        cls,
        source: str,
        reference: str,
        prediction: str,
        src_lang: str,
        tgt_lang: str,
    ) -> str:
        """Classify the primary translation error using linguistic heuristics."""
        src_tokens = source.strip().split()
        ref_tokens = reference.strip().split()
        pred_tokens = prediction.strip().split()

        # 1. Long sentence failure (>30 words with high token mismatch)
        if len(src_tokens) >= 30 and abs(len(pred_tokens) - len(ref_tokens)) > (
            0.4 * len(ref_tokens)
        ):
            return "long_sentence_failure"

        # 2. Untranslated words (source words appearing verbatim in prediction if scripts differ)
        if tgt_lang in ["hi", "mr"]:
            latin_words_in_pred = [w for w in pred_tokens if re.match(r"^[A-Za-z]+$", w)]
            if len(latin_words_in_pred) >= 2:
                return "untranslated_words"

        # 3. Repetition error (same token repeated consecutively 3+ times)
        for i in range(len(pred_tokens) - 2):
            if pred_tokens[i] == pred_tokens[i + 1] == pred_tokens[i + 2]:
                return "repetition"

        # 4. Missing words (prediction significantly shorter than reference)
        if len(pred_tokens) < (0.6 * max(len(ref_tokens), 1)):
            return "missing_words"

        # 5. Extra words (prediction significantly longer than reference)
        if len(pred_tokens) > (1.5 * max(len(ref_tokens), 1)):
            return "extra_words"

        # 6. Word order (similar token set but different n-gram ordering)
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        if len(pred_set.intersection(ref_set)) >= 0.7 * max(len(ref_set), 1):
            return "word_order"

        # 7. Named entity (capitalized words in source missing/mangled)
        src_capitals = [w for w in src_tokens if w and w[0].isupper() and len(w) > 2]
        if len(src_capitals) > 0 and len(pred_set.intersection(ref_set)) < 0.3 * max(
            len(ref_set), 1
        ):
            return "named_entity"

        # 8. Ambiguity (general lexical substitution mismatch)
        if 0.3 <= len(pred_set.intersection(ref_set)) / max(len(ref_set), 1) < 0.6:
            return "ambiguity"

        # 9. Grammar
        return "grammar"

    @classmethod
    def analyze_predictions(
        cls,
        df: pd.DataFrame,
        src_lang: str,
        tgt_lang: str,
        model_id: str,
        src_col: str = "source_text",
        ref_col: str = "target_text",
        pred_col: str = "prediction",
    ) -> pd.DataFrame:
        """Annotate each prediction with its classified error category and generate error statistics."""
        error_records = []
        for idx, row in df.iterrows():
            src = str(row[src_col])
            ref = str(row[ref_col])
            pred = str(row[pred_col])

            # If identical or near-identical, no error
            if pred.strip() == ref.strip():
                category = "none"
            else:
                category = cls.classify_error(src, ref, pred, src_lang, tgt_lang)

            error_records.append(
                {
                    "sample_id": row.get("sample_id", f"sample_{idx}"),
                    "source_text": src,
                    "reference_text": ref,
                    "prediction": pred,
                    "model_id": model_id,
                    "source_language": src_lang,
                    "target_language": tgt_lang,
                    "error_category": category,
                }
            )

        return pd.DataFrame(error_records)

    @classmethod
    def summarize_errors(cls, error_df: pd.DataFrame) -> Dict[str, any]:
        """Produce error category distribution counts and percentages."""
        non_zero_errors = error_df[error_df["error_category"] != "none"]
        counts = non_zero_errors["error_category"].value_counts().to_dict()
        total_errors = len(non_zero_errors)

        distribution = {}
        for cat in cls.CATEGORIES:
            c = counts.get(cat, 0)
            pct = round((c / total_errors * 100.0), 2) if total_errors > 0 else 0.0
            distribution[cat] = {"count": c, "percentage": pct}

        return {
            "total_samples": len(error_df),
            "perfect_translations": len(error_df) - total_errors,
            "total_error_samples": total_errors,
            "distribution": distribution,
        }
