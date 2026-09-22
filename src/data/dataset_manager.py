import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.data.cleaner import DataCleaner
from src.data.leakage import LeakageDetector
from src.data.splitter import DatasetSplitter
from src.registry.language_registry import LanguageRegistry
from src.utils.logger import setup_logger

logger = setup_logger("dataset_manager")


class DatasetManager:
    """Manages raw dataset acquisition, deterministic cleaning, leakage auditing, and splitting."""

    def __init__(
        self,
        data_dir: str = "data",
        cleaner: Optional[DataCleaner] = None,
        splitter: Optional[DatasetSplitter] = None,
    ):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.cleaner = cleaner or DataCleaner()
        self.splitter = splitter or DatasetSplitter(seed=42)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _compute_file_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def prepare_dataset_pipeline(
        self,
        raw_df: pd.DataFrame,
        src_lang: str,
        tgt_lang: str,
        dataset_name: str,
        provenance_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Run complete deterministic pipeline:
        1. Adapt to canonical schema
        2. Clean and filter (Unicode NFKC, HTML removal, length/ratio rules)
        3. Split 80/10/10 deterministically by unique source sentence
        4. Audit dual leakage (pair-level AND source-level)
        5. Persist splits to disk
        6. Compute SHA-256 checksums
        7. Generate freeze_manifest.json with full provenance
        """
        LanguageRegistry.validate_direction(src_lang, tgt_lang)
        pair_key = f"{src_lang}_{tgt_lang}"
        logger.info(
            f"Starting dataset pipeline for direction '{src_lang}->{tgt_lang}' ({len(raw_df)} initial rows)"
        )

        # Clean
        clean_df, cleaning_stats = self.cleaner.clean_dataframe(raw_df)
        logger.info(f"Cleaning finished: {cleaning_stats['final_rows']} valid rows retained")

        # Split deterministically by source sentence
        train_df, val_df, test_df = self.splitter.split(clean_df)
        logger.info(
            f"Splits created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

        # Audit Dual Leakage
        leakage_report = LeakageDetector.check_splits_leakage(train_df, val_df, test_df)
        if leakage_report["has_leakage"]:
            raise AssertionError(f"Leakage detected in {pair_key}: {leakage_report}")

        # Save to processed directory
        pair_dir = self.processed_dir / pair_key
        pair_dir.mkdir(parents=True, exist_ok=True)

        train_path = pair_dir / "train.csv"
        val_path = pair_dir / "val.csv"
        test_path = pair_dir / "test.csv"
        manifest_path = pair_dir / "dataset_manifest.json"
        freeze_manifest_path = pair_dir / "freeze_manifest.json"

        train_df.to_csv(train_path, index=False, encoding="utf-8")
        val_df.to_csv(val_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")

        # Compute SHA-256 checksums
        train_sha256 = self._compute_file_sha256(train_path)
        val_sha256 = self._compute_file_sha256(val_path)
        test_sha256 = self._compute_file_sha256(test_path)

        manifest = {
            "source_language": src_lang,
            "target_language": tgt_lang,
            "direction": f"{src_lang}-{tgt_lang}",
            "dataset_name": dataset_name,
            "provenance": provenance_metadata or {},
            "cleaning_stats": cleaning_stats,
            "split_seed": self.splitter.seed,
            "split_ratios": {
                "train": self.splitter.train_ratio,
                "val": self.splitter.val_ratio,
                "test": self.splitter.test_ratio,
            },
            "split_counts": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
                "total_cleaned": len(clean_df),
            },
            "leakage_audit": leakage_report,
            "checksums_sha256": {
                "train_csv": train_sha256,
                "val_csv": val_sha256,
                "test_csv": test_sha256,
            },
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "paths": {
                "train": str(train_path),
                "val": str(val_path),
                "test": str(test_path),
            },
        }

        # Write both dataset_manifest.json and freeze_manifest.json
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with open(freeze_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"Dataset pipeline completed successfully for {pair_key}. Freeze manifest saved to {freeze_manifest_path}"
        )
        return train_df, val_df, test_df, manifest

    def load_processed_splits(
        self,
        src_lang: str,
        tgt_lang: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load existing processed splits for a language pair."""
        pair_key = f"{src_lang}_{tgt_lang}"
        pair_dir = self.processed_dir / pair_key

        train_path = pair_dir / "train.csv"
        val_path = pair_dir / "val.csv"
        test_path = pair_dir / "test.csv"

        if not (train_path.exists() and val_path.exists() and test_path.exists()):
            raise FileNotFoundError(f"Processed dataset splits not found in {pair_dir}")

        train_df = pd.read_csv(train_path, encoding="utf-8")
        val_df = pd.read_csv(val_path, encoding="utf-8")
        test_df = pd.read_csv(test_path, encoding="utf-8")

        return train_df, val_df, test_df
