"""Production Translation Inference Engine.
Master Decision Register Section 10, PRD Section 24, TDD Section 24, Spec Section 2 (TASK-020).
"""

import time
from typing import Any, Dict, List, Optional

import torch

from src.models.base import BaseTranslationModel
from src.models.model_manager import ModelManager
from src.registry.language_registry import LanguageRegistry
from src.utils.logger import setup_logger

logger = setup_logger("inference_engine")


class TranslationInferenceEngine:
    """Production translation engine managing model caching, device allocation, and generation."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_models: Dict[str, BaseTranslationModel] = {}
        logger.info(f"Initialized TranslationInferenceEngine on device '{self.device}'")

    def get_or_load_model(
        self,
        model_id: str,
        src_lang: str = "en",
        tgt_lang: str = "hi",
        checkpoint_path: Optional[str] = None,
    ) -> BaseTranslationModel:
        """Retrieve model from cache or load it once."""
        cache_key = f"{model_id}_{src_lang}_{tgt_lang}_{checkpoint_path or 'default'}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        logger.info(f"Loading and caching model '{model_id}' ({src_lang}->{tgt_lang})...")
        model = ModelManager.load_model(
            model_id=model_id,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            checkpoint=checkpoint_path,
            device=self.device,
        )
        self._loaded_models[cache_key] = model
        return model

    def translate_single(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        model_id: str = "nllb200",
        checkpoint_path: Optional[str] = None,
        num_beams: int = 5,
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Translate a single input string with timing and metadata."""
        if not text or not text.strip():
            return {
                "source_text": text,
                "translated_text": "",
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "model_id": model_id,
                "latency_ms": 0.0,
            }

        LanguageRegistry.validate_direction(src_lang, tgt_lang)

        # For pair-specific MarianMT, resolve appropriate pair model
        if model_id.startswith("marian"):
            resolved_model_id = f"marian_{src_lang}_{tgt_lang}"
        else:
            resolved_model_id = model_id

        model = self.get_or_load_model(
            model_id=resolved_model_id,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            checkpoint_path=checkpoint_path,
        )

        t0 = time.perf_counter()
        translations = model.translate(
            texts=[text.strip()],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 2)

        translated = translations[0] if translations else ""

        return {
            "source_text": text.strip(),
            "translated_text": translated,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "model_id": resolved_model_id,
            "latency_ms": latency_ms,
        }

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        model_id: str = "nllb200",
        checkpoint_path: Optional[str] = None,
        num_beams: int = 5,
        max_new_tokens: int = 128,
    ) -> List[Dict[str, Any]]:
        """Translate a batch of texts."""
        if not texts:
            return []

        LanguageRegistry.validate_direction(src_lang, tgt_lang)
        if model_id.startswith("marian"):
            resolved_model_id = f"marian_{src_lang}_{tgt_lang}"
        else:
            resolved_model_id = model_id

        model = self.get_or_load_model(
            model_id=resolved_model_id,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            checkpoint_path=checkpoint_path,
        )

        t0 = time.perf_counter()
        translations = model.translate(
            texts=texts,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        t1 = time.perf_counter()
        total_latency_ms = (t1 - t0) * 1000.0
        per_sample_ms = round(total_latency_ms / len(texts), 2)

        results = []
        for src, trans in zip(texts, translations):
            results.append(
                {
                    "source_text": src,
                    "translated_text": trans,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "model_id": resolved_model_id,
                    "latency_ms": per_sample_ms,
                }
            )
        return results

    def clear_cache(self) -> None:
        """Clear cached models and free VRAM."""
        self._loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared inference model cache and emptied CUDA cache.")
