"""BLEU Evaluation Metric Wrapper.
Master Decision Register Section 6 & PRD Section 17.1.
"""

from typing import Dict, List

import sacrebleu


class BleuEvaluator:
    """Calculates corpus and sentence-level BLEU scores using SacreBLEU."""

    @staticmethod
    def compute(
        predictions: List[str],
        references: List[str],
        tokenize: str = "13a",
        lowercase: bool = False,
    ) -> Dict[str, any]:
        """Calculate SacreBLEU score.

        Args:
            predictions: List of hypothesis strings
            references: List of reference strings (1 per hypothesis)
            tokenize: Tokenizer for SacreBLEU ('13a', 'intl', 'zh', 'ja-mecab')
            lowercase: Whether to lowercase before computation

        Returns:
            Dict containing score, precisions, bp, sys_len, ref_len, and version.
        """
        if not predictions or not references:
            return {"score": 0.0, "precisions": [], "bp": 0.0, "version": sacrebleu.__version__}

        # sacrebleu expects references as list of reference streams: [ [ref_1, ref_2, ...], [ref2_1, ...] ]
        ref_list = [references]
        bleu = sacrebleu.corpus_bleu(
            predictions,
            ref_list,
            tokenize=tokenize,
            lowercase=lowercase,
        )

        return {
            "score": round(bleu.score, 4),
            "precisions": [round(p, 2) for p in bleu.precisions],
            "bp": round(bleu.bp, 4),
            "sys_len": bleu.sys_len,
            "ref_len": bleu.ref_len,
            "signature": bleu.format(),
            "version": sacrebleu.__version__,
        }
