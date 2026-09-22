"""Authoritative Model Registry for Multilingual NMT.
Master Decision Register Section 3 & Manifest Section 1.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    family: str  # 'marian', 'mbart50', 'nllb'
    display_name: str
    checkpoint: str
    role: str  # 'baseline', 'multilingual', 'primary_multilingual'
    is_multilingual: bool
    supported_directions: List[str]
    default_generation_params: Dict[str, any] = field(
        default_factory=lambda: {
            "num_beams": 5,
            "max_new_tokens": 128,
            "min_length": 1,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "do_sample": False,
        }
    )


class ModelRegistry:
    """Authoritative registry for all translation models."""

    _MODELS: Dict[str, ModelMetadata] = {
        "marian_en_hi": ModelMetadata(
            model_id="marian_en_hi",
            family="marian",
            display_name="MarianMT (EN-HI)",
            checkpoint="Helsinki-NLP/opus-mt-en-hi",
            role="baseline",
            is_multilingual=False,
            supported_directions=["en-hi"],
        ),
        "marian_en_mr": ModelMetadata(
            model_id="marian_en_mr",
            family="marian",
            display_name="MarianMT (EN-MR)",
            checkpoint="Helsinki-NLP/opus-mt-en-mr",
            role="baseline",
            is_multilingual=False,
            supported_directions=["en-mr"],
        ),
        "marian_en_fr": ModelMetadata(
            model_id="marian_en_fr",
            family="marian",
            display_name="MarianMT (EN-FR)",
            checkpoint="Helsinki-NLP/opus-mt-en-fr",
            role="baseline",
            is_multilingual=False,
            supported_directions=["en-fr"],
        ),
        "marian_en_de": ModelMetadata(
            model_id="marian_en_de",
            family="marian",
            display_name="MarianMT (EN-DE)",
            checkpoint="Helsinki-NLP/opus-mt-en-de",
            role="baseline",
            is_multilingual=False,
            supported_directions=["en-de"],
        ),
        "mbart50": ModelMetadata(
            model_id="mbart50",
            family="mbart50",
            display_name="mBART-50 (Many-to-Many)",
            checkpoint="facebook/mbart-large-50-many-to-many-mmt",
            role="multilingual",
            is_multilingual=True,
            supported_directions=["en-hi", "en-mr", "en-fr", "en-de"],
        ),
        "nllb200": ModelMetadata(
            model_id="nllb200",
            family="nllb",
            display_name="NLLB-200 (distilled-600M)",
            checkpoint="facebook/nllb-200-distilled-600M",
            role="primary_multilingual",
            is_multilingual=True,
            supported_directions=["en-hi", "en-mr", "en-fr", "en-de"],
        ),
    }

    @classmethod
    def get_model(cls, model_id: str) -> ModelMetadata:
        """Get model metadata by ID."""
        model_id = model_id.lower().strip()
        if model_id not in cls._MODELS:
            raise ValueError(
                f"Unknown model_id '{model_id}'. Available: {list(cls._MODELS.keys())}"
            )
        return cls._MODELS[model_id]

    @classmethod
    def get_marian_for_direction(cls, src: str, tgt: str) -> ModelMetadata:
        """Get the specific MarianMT model for a given direction."""
        direction = f"{src.lower()}_{tgt.lower()}"
        model_id = f"marian_{direction}"
        if model_id not in cls._MODELS:
            raise ValueError(f"No MarianMT model registered for direction '{src}->{tgt}'")
        return cls._MODELS[model_id]

    @classmethod
    def get_all_models(cls) -> List[ModelMetadata]:
        """Return all registered models."""
        return list(cls._MODELS.values())

    @classmethod
    def get_models_for_direction(cls, src: str, tgt: str) -> List[ModelMetadata]:
        """Return all models capable of translating src->tgt."""
        direction = f"{src.lower().strip()}-{tgt.lower().strip()}"
        return [m for m in cls._MODELS.values() if direction in m.supported_directions]
