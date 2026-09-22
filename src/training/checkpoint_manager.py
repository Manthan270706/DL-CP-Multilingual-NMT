"""Checkpoint Manager.
Master Decision Register Section 9 & Spec Section 2 (TASK-010 & TASK-013).
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger("checkpoint_manager")


class CheckpointManager:
    """Saves, tracks, and manages best model checkpoints."""

    def __init__(
        self,
        experiment_dir: str,
        metric_name: str = "comet",
        greater_is_better: bool = True,
        max_to_keep: int = 2,
    ):
        self.experiment_dir = Path(experiment_dir)
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.best_checkpoint_dir = self.experiment_dir / "best_checkpoint"
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.max_to_keep = max_to_keep

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_score: Optional[float] = None
        self.best_step: Optional[int] = None
        self.checkpoint_history: List[Dict[str, Any]] = []

    def is_better(self, score: float) -> bool:
        """Check if score is better than current best."""
        if self.best_score is None:
            return True
        if self.greater_is_better:
            return score > self.best_score
        else:
            return score < self.best_score

    def save_checkpoint(
        self,
        model_wrapper: Any,
        step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> str:
        """Save a step checkpoint and update best checkpoint if applicable."""
        step_dir = self.checkpoints_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        model_wrapper.save_checkpoint(str(step_dir))

        # Save step metrics
        step_meta = {
            "step": step,
            "metrics": metrics,
            "is_best": is_best,
            "path": str(step_dir),
        }
        self.checkpoint_history.append(step_meta)

        with open(step_dir / "checkpoint_info.json", "w", encoding="utf-8") as f:
            json.dump(step_meta, f, indent=2)

        score = metrics.get(self.metric_name, 0.0)
        if is_best or self.is_better(score):
            self.best_score = score
            self.best_step = step
            logger.info(
                f"New best checkpoint at step {step} with {self.metric_name}={score:.4f}! Updating {self.best_checkpoint_dir}"
            )
            if self.best_checkpoint_dir.exists():
                shutil.rmtree(self.best_checkpoint_dir)
            shutil.copytree(step_dir, self.best_checkpoint_dir)

        self._export_manifest()
        return str(step_dir)

    def _export_manifest(self) -> None:
        """Export checkpoint_manifest.json in the experiment directory."""
        manifest = {
            "metric_for_best": self.metric_name,
            "greater_is_better": self.greater_is_better,
            "best_score": self.best_score,
            "best_step": self.best_step,
            "best_checkpoint_path": (
                str(self.best_checkpoint_dir) if self.best_checkpoint_dir.exists() else None
            ),
            "checkpoints": self.checkpoint_history,
        }
        manifest_path = self.experiment_dir / "checkpoint_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
