"""Model Manager Factory.
TDD Section 9 & Spec Section 2 (TASK-008).
"""

from typing import Optional

from src.models.base import BaseTranslationModel
from src.models.marian import MarianTranslationModel
from src.models.mbart import MBartTranslationModel
from src.models.nllb import NllbTranslationModel
from src.registry.model_registry import ModelRegistry
from src.utils.logger import setup_logger

logger = setup_logger("model_manager")


class ModelManager:
    """Factory and manager for instantiating and loading translation models."""

    @staticmethod
    def create_model(
        model_id: str,
        src_lang: str = "en",
        tgt_lang: str = "hi",
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ) -> BaseTranslationModel:
        """Create and return a model wrapper instance."""
        model_id = model_id.lower().strip()
        model_meta = ModelRegistry.get_model(model_id)

        if model_meta.family == "marian":
            return MarianTranslationModel(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                checkpoint=checkpoint,
                device=device,
            )
        elif model_meta.family == "mbart50":
            return MBartTranslationModel(checkpoint=checkpoint, device=device)
        elif model_meta.family == "nllb":
            return NllbTranslationModel(checkpoint=checkpoint, device=device)
        else:
            raise ValueError(f"Unsupported model family '{model_meta.family}'")

    @staticmethod
    def load_model(
        model_id: str,
        src_lang: str = "en",
        tgt_lang: str = "hi",
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ) -> BaseTranslationModel:
        """Instantiate and load weights for a model."""
        model = ModelManager.create_model(
            model_id=model_id,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            checkpoint=checkpoint,
            device=device,
        )
        model.load(checkpoint_path_or_name=checkpoint)
        return model
