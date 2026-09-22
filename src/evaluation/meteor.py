"""METEOR Evaluation Metric Wrapper.
Master Decision Register Section 6 & PRD Section 17.2.
"""

from typing import Dict, List

import nltk
from nltk.translate.meteor_score import meteor_score


def _ensure_nltk_resources():
    """Ensure WordNet and tokenizers are downloaded."""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


class MeteorEvaluator:
    """Calculates corpus and sentence-level METEOR scores using NLTK."""

    def __init__(self):
        _ensure_nltk_resources()

    @staticmethod
    def compute(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, any]:
        """Calculate mean corpus METEOR score.

        Args:
            predictions: List of hypothesis strings
            references: List of reference strings

        Returns:
            Dict containing mean score, sentence scores, and version.
        """
        _ensure_nltk_resources()

        if not predictions or not references:
            return {"score": 0.0, "version": nltk.__version__}

        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.strip().split()
            ref_tokens = ref.strip().split()
            score = meteor_score([ref_tokens], pred_tokens)
            scores.append(score)

        mean_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "score": round(mean_score * 100.0, 4),  # Scaled to 0-100 to match BLEU scale
            "raw_score": round(mean_score, 6),  # 0-1 scale
            "num_samples": len(scores),
            "version": f"nltk-{nltk.__version__}",
        }
