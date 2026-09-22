"""Configuration loader and manager."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigManager:
    """Centralized configuration loader."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a single YAML configuration file."""
        file_path = self.config_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_base_config(self) -> Dict[str, Any]:
        return self.load_yaml("base.yaml")

    def get_languages_config(self) -> Dict[str, Any]:
        return self.load_yaml("languages.yaml")

    def get_models_config(self) -> Dict[str, Any]:
        return self.load_yaml("models.yaml")

    def get_datasets_config(self) -> Dict[str, Any]:
        return self.load_yaml("datasets.yaml")

    def get_training_config(self) -> Dict[str, Any]:
        return self.load_yaml("training.yaml")

    def get_evaluation_config(self) -> Dict[str, Any]:
        return self.load_yaml("evaluation.yaml")

    def get_full_config(self) -> Dict[str, Any]:
        """Merge all configuration sections into a consolidated dictionary."""
        return {
            "base": self.get_base_config(),
            "languages": self.get_languages_config(),
            "models": self.get_models_config(),
            "datasets": self.get_datasets_config(),
            "training": self.get_training_config(),
            "evaluation": self.get_evaluation_config(),
        }


def save_yaml(data: Dict[str, Any], filepath: str) -> None:
    """Save a dictionary to a YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
