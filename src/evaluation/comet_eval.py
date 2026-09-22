"""COMET Semantic Evaluation Metric Wrapper.
Master Decision Register Section 6 & PRD Section 17.3.
"""

from typing import Dict, List, Optional

import torch

from src.utils.logger import setup_logger

logger = setup_logger("comet_eval")


class CometEvaluator:
    """Calculates COMET semantic translation quality score using Unbabel COMET."""

    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-comet-da",
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from comet import download_model, load_from_checkpoint

                logger.info(f"Downloading/loading COMET model '{self.model_name}'...")
                model_path = download_model(self.model_name)
                self._model = load_from_checkpoint(model_path)
                logger.info("COMET model loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"Could not load official COMET model ({e}). Using semantic similarity fallback."
                )
                self._model = "fallback"

    def _get_gpu_count(self) -> int:
        """Return number of GPUs to use for COMET inference.

        Checks whether the current GPU compute capability is supported by the
        installed PyTorch build. If not (e.g., RTX 5050 sm_120 with cu124), we
        force CPU inference so that real COMET scores are produced rather than
        the Jaccard fallback.
        """
        if not torch.cuda.is_available():
            return 0
        if "cuda" not in self.device:
            return 0
        # Probe whether a trivial CUDA kernel actually executes on this GPU.
        # RTX 5050 is sm_120 (Blackwell) — not supported by PyTorch cu124
        # which caps at sm_90. The probe catches that silently.
        try:
            major, minor = torch.cuda.get_device_capability(0)
            _ = torch.zeros(1, device="cuda") + 1  # lightweight kernel probe
            logger.info(f"GPU probe succeeded (sm_{major}{minor}). Using GPU for COMET.")
            return 1
        except Exception as probe_err:
            major, minor = torch.cuda.get_device_capability(0)
            logger.info(
                f"GPU kernel probe failed (sm_{major}{minor}: {probe_err}). "
                "Using CPU for COMET inference to guarantee real COMET scores."
            )
            return 0

    def compute(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, any]:
        """Calculate system-level and sentence-level COMET scores."""
        if not predictions or not references or not sources:
            return {"score": 0.0, "version": "N/A", "model": self.model_name}

        self._load_model()

        if self._model != "fallback":
            try:
                import comet

                data = [
                    {"src": s, "mt": p, "ref": r}
                    for s, p, r in zip(sources, predictions, references)
                ]
                gpus = self._get_gpu_count()
                model_output = self._model.predict(
                    data,
                    batch_size=self.batch_size,
                    gpus=gpus,
                    progress_bar=False,
                )
                system_score = float(model_output.system_score)
                seg_scores = [float(s) for s in model_output.scores]

                return {
                    "score": round(system_score * 100.0, 4),  # 0-100 scale for consistency
                    "raw_score": round(system_score, 6),
                    "segment_scores": seg_scores,
                    "model": self.model_name,
                    "version": getattr(comet, "__version__", "2.x"),
                    "device_used": "gpu" if gpus > 0 else "cpu",
                }
            except Exception as e:
                logger.warning(
                    f"COMET prediction execution failed ({e}). Calculating token semantic approximation."
                )

        # Fallback semantic overlap calculation if COMET neural weights fail to download
        scores = []
        for s, p, r in zip(sources, predictions, references):
            p_words = set(p.strip().lower().split())
            r_words = set(r.strip().lower().split())
            overlap = len(p_words.intersection(r_words)) / max(len(p_words.union(r_words)), 1)
            scores.append(overlap)
        mean_s = sum(scores) / len(scores) if scores else 0.0

        return {
            "score": round(mean_s * 100.0, 4),
            "raw_score": round(mean_s, 6),
            "segment_scores": scores,
            "model": "fallback_jaccard_semantic",
            "version": "local-fallback",
        }
