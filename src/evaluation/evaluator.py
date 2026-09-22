"""Unified Evaluation Engine.
Master Decision Register Section 6, 7, 8, 9 & Spec Section 2 (TASK-015 to TASK-019).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.evaluation.bleu import BleuEvaluator
from src.evaluation.comet_eval import CometEvaluator
from src.evaluation.error_analyzer import ErrorAnalyzer
from src.evaluation.meteor import MeteorEvaluator
from src.evaluation.sentence_length_analyzer import SentenceLengthAnalyzer
from src.utils.logger import setup_logger

logger = setup_logger("evaluator")


class Evaluator:
    """Evaluates NMT models on datasets and persists structured evaluation artifacts."""

    def __init__(
        self,
        comet_model_name: str = "Unbabel/wmt22-comet-da",
        device: Optional[str] = None,
    ):
        self.bleu_evaluator = BleuEvaluator()
        self.meteor_evaluator = MeteorEvaluator()
        self.comet_evaluator = CometEvaluator(model_name=comet_model_name, device=device)

    def evaluate_predictions(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str],
        src_lang: str,
        tgt_lang: str,
        model_id: str,
        sample_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute all mandatory metrics and analyses for a set of predictions."""
        logger.info(
            f"Computing BLEU, METEOR, and COMET for {len(predictions)} samples ({src_lang}->{tgt_lang})..."
        )

        # 1. BLEU
        bleu_res = self.bleu_evaluator.compute(predictions, references)
        logger.info(f"BLEU: {bleu_res['score']:.2f}")

        # 2. METEOR
        meteor_res = self.meteor_evaluator.compute(predictions, references)
        logger.info(f"METEOR: {meteor_res['score']:.2f}")

        # 3. COMET
        comet_res = self.comet_evaluator.compute(sources, predictions, references)
        logger.info(f"COMET: {comet_res['score']:.2f}")

        # Create DataFrame for structured analysis
        eval_df = pd.DataFrame(
            {
                "sample_id": sample_ids or [f"sample_{i}" for i in range(len(sources))],
                "source_text": sources,
                "target_text": references,
                "prediction": predictions,
            }
        )

        # 4. Error Analysis
        error_df = ErrorAnalyzer.analyze_predictions(
            eval_df,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            model_id=model_id,
        )
        error_summary = ErrorAnalyzer.summarize_errors(error_df)

        # 5. Sentence Length Analysis
        length_summary = SentenceLengthAnalyzer.analyze(eval_df)

        metrics_report = {
            "model_id": model_id,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "sample_count": len(predictions),
            "bleu": bleu_res,
            "meteor": meteor_res,
            "comet": comet_res,
            "summary_scores": {
                "bleu": bleu_res["score"],
                "meteor": meteor_res["score"],
                "comet": comet_res["score"],
            },
            "error_analysis_summary": error_summary,
            "sentence_length_analysis": length_summary,
        }

        return {
            "metrics": metrics_report,
            "predictions_df": eval_df,
            "error_df": error_df,
        }

    def evaluate_and_export(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str],
        src_lang: str,
        tgt_lang: str,
        model_id: str,
        export_dir: str,
        sample_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate and persist predictions.csv, metrics.json, and error_analysis.csv."""
        res = self.evaluate_predictions(
            sources=sources,
            predictions=predictions,
            references=references,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            model_id=model_id,
            sample_ids=sample_ids,
        )

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        # Export predictions.csv
        preds_path = export_path / "predictions.csv"
        res["predictions_df"].to_csv(preds_path, index=False, encoding="utf-8")

        # Export error_analysis.csv
        errors_path = export_path / "error_analysis.csv"
        res["error_df"].to_csv(errors_path, index=False, encoding="utf-8")

        # Export metrics.json
        metrics_path = export_path / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(res["metrics"], f, indent=2)

        logger.info(f"Saved evaluation artifacts to {export_dir}")
        return res["metrics"]
