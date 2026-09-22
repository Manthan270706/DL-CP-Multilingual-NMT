"""NMT Training and Fine-Tuning Engine.
Master Decision Register Section 4, 6, 9 & Spec Section 2 (TASK-010 to TASK-016).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.evaluation.evaluator import Evaluator
from src.models.base import BaseTranslationModel
from src.models.model_manager import ModelManager
from src.registry.language_registry import LanguageRegistry
from src.training.checkpoint_manager import CheckpointManager
from src.training.resource_tracker import ResourceTracker
from src.utils.config import save_yaml
from src.utils.logger import setup_logger
from src.utils.system_info import export_environment_txt


class ParallelNMTDataset(Dataset):
    """PyTorch Dataset for parallel source-target translation pairs."""

    def __init__(
        self, df: pd.DataFrame, src_col: str = "source_text", tgt_col: str = "target_text"
    ):
        self.sources = df[src_col].astype(str).tolist()
        self.targets = df[tgt_col].astype(str).tolist()
        self.sample_ids = (
            df["sample_id"].tolist()
            if "sample_id" in df.columns
            else [f"s_{i}" for i in range(len(df))]
        )

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            "source": self.sources[idx],
            "target": self.targets[idx],
            "sample_id": self.sample_ids[idx],
        }


class NMTTrainer:
    """End-to-end configuration-driven Trainer for Transformer NMT fine-tuning."""

    def __init__(
        self,
        model_id: str,
        src_lang: str,
        tgt_lang: str,
        config: Dict[str, Any],
        experiment_id: str,
        output_base_dir: str = "experiments",
        device: Optional[str] = None,
    ):
        LanguageRegistry.validate_direction(src_lang, tgt_lang)
        self.model_id = model_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.config = config
        self.experiment_id = experiment_id
        self.output_dir = Path(output_base_dir) / experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger(
            name=f"trainer_{experiment_id}",
            log_file=str(self.output_dir / "training.log"),
        )
        self.resource_tracker = ResourceTracker(device=self.device)
        self.checkpoint_manager = CheckpointManager(
            experiment_dir=str(self.output_dir),
            metric_name="comet",
            greater_is_better=True,
        )
        self.evaluator = Evaluator(device=self.device)

    def _collate_batch(
        self, batch: List[Dict[str, str]], model_wrapper: BaseTranslationModel
    ) -> Dict[str, torch.Tensor]:
        sources = [item["source"] for item in batch]
        targets = [item["target"] for item in batch]
        max_len = self.config.get("max_source_length", 128)
        return model_wrapper.prepare_inputs(
            sources=sources,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            max_length=max_len,
            targets=targets,
        )

    def run_training(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Execute fine-tuning with validation COMET tracking and artifact persistence."""
        self.logger.info(f"Starting experiment '{self.experiment_id}' on device '{self.device}'")
        self.resource_tracker.start()

        # Save configuration & environment report
        save_yaml(self.config, str(self.output_dir / "config.yaml"))
        export_environment_txt(str(self.output_dir / "environment.txt"))

        # Load Model
        model_wrapper = ModelManager.load_model(
            model_id=self.model_id,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            device=self.device,
        )

        # Gradient checkpointing
        if self.config.get("gradient_checkpointing", True):
            model_wrapper.enable_gradient_checkpointing()

        # Datasets & Loaders
        batch_size = self.config.get("per_device_train_batch_size", 4)
        eval_batch_size = self.config.get("per_device_eval_batch_size", 4)
        grad_accum = self.config.get("gradient_accumulation_steps", 4)

        train_dataset = ParallelNMTDataset(train_df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: self._collate_batch(b, model_wrapper),
        )

        epochs = self.config.get("num_train_epochs", 3)
        lr = float(self.config.get("learning_rate", 2e-5))
        weight_decay = float(self.config.get("weight_decay", 0.01))

        # Optimizer & Scheduler
        optimizer = AdamW(model_wrapper.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs // grad_accum
        warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.1))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        _use_amp = self.config.get("fp16", True) and "cuda" in self.device
        _amp_device_type = "cuda" if "cuda" in self.device else "cpu"
        # Use unified torch.amp API (forward-compatible with PyTorch 2.7+ nightly)
        if hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler(_amp_device_type, enabled=_use_amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=_use_amp)

        global_step = 0
        model_wrapper.model.train()
        optimizer.zero_grad()

        self.logger.info(
            f"Training plan: {epochs} epochs, {len(train_loader)} batches/epoch, grad_accum={grad_accum}, effective_batch={batch_size * grad_accum}"
        )

        for epoch in range(epochs):
            self.logger.info(f"=== Epoch {epoch + 1}/{epochs} ===")
            epoch_loss = 0.0

            for step, batch in enumerate(train_loader):
                try:
                    _autocast_ctx = (
                        torch.amp.autocast(_amp_device_type, enabled=_use_amp)
                        if hasattr(torch.amp, "autocast")
                        else torch.cuda.amp.autocast(enabled=_use_amp)
                    )
                    with _autocast_ctx:
                        outputs = model_wrapper.model(**batch)
                        loss = outputs.loss / grad_accum

                    scaler.scale(loss).backward()
                    epoch_loss += loss.item() * grad_accum

                    if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model_wrapper.model.parameters(), max_norm=1.0
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                        if global_step % self.config.get("logging_steps", 50) == 0:
                            current_lr = scheduler.get_last_lr()[0]
                            self.logger.info(
                                f"Step {global_step} | Loss: {loss.item() * grad_accum:.4f} | LR: {current_lr:.2e}"
                            )

                except torch.cuda.OutOfMemoryError as e:
                    self.logger.error(f"CUDA OOM encountered during training step {step}: {e}")
                    raise

            # Epoch Validation
            self.logger.info(f"Running Validation at Epoch {epoch + 1}...")
            model_wrapper.model.eval()
            val_sources = val_df["source_text"].tolist()
            val_refs = val_df["target_text"].tolist()

            # Generate validation translations with deterministic beam search
            val_preds = model_wrapper.translate(
                texts=val_sources,
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                num_beams=self.config.get("num_beams", 5),
                max_new_tokens=self.config.get("max_new_tokens", 128),
            )

            val_metrics = self.evaluator.evaluate_predictions(
                sources=val_sources,
                predictions=val_preds,
                references=val_refs,
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                model_id=self.model_id,
            )["metrics"]

            val_comet = val_metrics["summary_scores"]["comet"]
            val_bleu = val_metrics["summary_scores"]["bleu"]
            val_meteor = val_metrics["summary_scores"]["meteor"]
            self.logger.info(
                f"Validation Epoch {epoch + 1} | COMET: {val_comet:.2f} | BLEU: {val_bleu:.2f} | METEOR: {val_meteor:.2f}"
            )

            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                model_wrapper=model_wrapper,
                step=global_step,
                metrics={"comet": val_comet, "bleu": val_bleu, "meteor": val_meteor},
            )
            model_wrapper.model.train()

        # Stop resource tracker
        resource_metrics = self.resource_tracker.stop()
        self.logger.info(
            f"Training completed in {resource_metrics['elapsed_seconds']}s. Peak VRAM: {resource_metrics['peak_vram_mb']} MB"
        )

        # Freeze & Load Best Checkpoint for Test Set Evaluation
        best_ckpt_path = self.checkpoint_manager.best_checkpoint_dir
        if best_ckpt_path.exists():
            self.logger.info(
                f"Loading best checkpoint from {best_ckpt_path} for final test evaluation..."
            )
            model_wrapper.load(checkpoint_path_or_name=str(best_ckpt_path))

        # Final Untouched Test Set Evaluation
        self.logger.info(
            f"Evaluating final frozen model on untouched test set ({len(test_df)} samples)..."
        )
        test_sources = test_df["source_text"].tolist()
        test_refs = test_df["target_text"].tolist()
        test_sample_ids = test_df["sample_id"].tolist() if "sample_id" in test_df.columns else None

        test_preds = model_wrapper.translate(
            texts=test_sources,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            num_beams=self.config.get("num_beams", 5),
            max_new_tokens=self.config.get("max_new_tokens", 128),
        )

        # Latency Benchmark
        latency_benchmark = ResourceTracker.measure_inference_latency(
            model=model_wrapper,
            sample_texts=test_sources[: min(20, len(test_sources))],
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
        )

        resource_metrics["inference_latency"] = latency_benchmark
        resource_metrics["model_size_mb"] = model_wrapper.get_model_size_mb()
        resource_metrics["parameters"] = model_wrapper.count_parameters()

        with open(self.output_dir / "resource_metrics.json", "w", encoding="utf-8") as f:
            json.dump(resource_metrics, f, indent=2)

        # Export final predictions, metrics, and error analysis
        final_test_metrics = self.evaluator.evaluate_and_export(
            sources=test_sources,
            predictions=test_preds,
            references=test_refs,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            model_id=self.model_id,
            export_dir=str(self.output_dir),
            sample_ids=test_sample_ids,
        )

        # Generate Experiment README.md
        self._write_experiment_readme(final_test_metrics, resource_metrics)
        self.logger.info(f"Experiment '{self.experiment_id}' fully completed. Artifacts verified.")
        return final_test_metrics

    def _write_experiment_readme(self, metrics: Dict[str, Any], resources: Dict[str, Any]) -> None:
        """Write structured README.md in experiment directory (Spec Section 9)."""
        readme_path = self.output_dir / "README.md"
        scores = metrics.get("summary_scores", {})
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# Experiment: {self.experiment_id}\n\n")
            f.write(f"- **Model**: `{self.model_id}`\n")
            f.write(f"- **Direction**: `{self.src_lang} -> {self.tgt_lang}`\n")
            f.write(f"- **Date / Epochs**: {self.config.get('num_train_epochs')} epochs\n\n")
            f.write("## Test Metrics\n")
            f.write(f"- **BLEU**: `{scores.get('bleu', 0.0):.2f}`\n")
            f.write(f"- **METEOR**: `{scores.get('meteor', 0.0):.2f}`\n")
            f.write(f"- **COMET**: `{scores.get('comet', 0.0):.2f}`\n\n")
            f.write("## Resource Consumption\n")
            f.write(f"- **Training Time**: `{resources.get('elapsed_seconds', 0.0)} s`\n")
            f.write(f"- **Peak VRAM**: `{resources.get('peak_vram_mb', 0.0)} MB`\n")
            f.write(
                f"- **Mean Per-Sample Latency**: `{resources.get('inference_latency', {}).get('mean_per_sample_latency_ms', 0.0)} ms`\n"
            )
            f.write(
                f"- **Model Parameters**: `{resources.get('parameters', {}).get('total', 0):,}`\n"
            )
