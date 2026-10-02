import gc
import os
from pathlib import Path
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
        """Retrieve model from cache or load it once with proactive VRAM headroom management."""
        cache_key = f"{model_id}_{src_lang}_{tgt_lang}_{checkpoint_path or 'default'}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        # VRAM protection for consumer GPUs (< 12GB VRAM):
        # mBART-50 (~3.2 GB) and NLLB-200 (~3.2 GB) cannot both fit simultaneously on an 8GB GPU.
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                is_large = any(k in model_id.lower() for k in ["mbart", "nllb"])
                has_large_cached = any(any(k in m.lower() for k in ["mbart", "nllb"]) for m in self._loaded_models)
                free_bytes, _ = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024**3)

                if (is_large and has_large_cached) or (is_large and free_gb < 5.0 and len(self._loaded_models) > 0) or (free_gb < 1.8):
                    logger.info(f"VRAM protection triggered (free: {free_gb:.2f} GB). Evicting cached models...")
                    self.clear_cache()
            except Exception as e:
                logger.warning(f"Failed to check CUDA memory info: {e}")

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

    @staticmethod
    def get_finetuned_checkpoint_path(model_id: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        """Check if a validated fine-tuned checkpoint exists on disk."""
        base_dir = Path("experiments") / "finetuned"
        if not base_dir.exists():
            return None

        ds_tag = "IITB" if tgt_lang == "hi" else "OPUS"
        m_family = "marian" if model_id.startswith("marian") else model_id
        exp_id = f"{m_family.upper()}_{src_lang.upper()}_{tgt_lang.upper()}_{ds_tag}_FT_001"
        candidate_dir = base_dir / exp_id / "best_checkpoint"

        if candidate_dir.exists() and (candidate_dir / "config.json").exists():
            return str(candidate_dir)
        return None

    def translate_single(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        model_id: str = "nllb200",
        checkpoint_path: Optional[str] = None,
        checkpoint_mode: str = "zero_shot",
        num_beams: int = 5,
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Translate a single input string with timing, throughput, and metadata."""
        if not text or not text.strip():
            return {
                "source_text": text,
                "translated_text": "",
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "model_id": model_id,
                "checkpoint_mode": checkpoint_mode,
                "checkpoint_path": checkpoint_path,
                "latency_ms": 0.0,
                "tokens_per_sec": 0.0,
                "token_count": 0,
            }

        LanguageRegistry.validate_direction(src_lang, tgt_lang)

        # Resolve checkpoint if mode is finetuned
        resolved_ckpt = checkpoint_path
        if checkpoint_mode == "finetuned" and not resolved_ckpt:
            resolved_ckpt = self.get_finetuned_checkpoint_path(model_id, src_lang, tgt_lang)

        # For pair-specific MarianMT, resolve appropriate pair model
        if model_id.startswith("marian"):
            resolved_model_id = f"marian_{src_lang}_{tgt_lang}"
        else:
            resolved_model_id = model_id

        model = self.get_or_load_model(
            model_id=resolved_model_id,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            checkpoint_path=resolved_ckpt,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        translations = model.translate(
            texts=[text.strip()],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        t1 = time.perf_counter()
        elapsed_sec = t1 - t0
        latency_ms = round(elapsed_sec * 1000.0, 2)

        translated = translations[0] if translations else ""
        token_count = len(translated.split())
        tokens_per_sec = round(token_count / max(elapsed_sec, 0.001), 1)

        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1)

        return {
            "source_text": text.strip(),
            "translated_text": translated,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "model_id": resolved_model_id,
            "checkpoint_mode": checkpoint_mode if resolved_ckpt else "zero_shot",
            "checkpoint_path": resolved_ckpt or "base_pretrained",
            "latency_ms": latency_ms,
            "tokens_per_sec": tokens_per_sec,
            "token_count": token_count,
            "peak_vram_mb": peak_vram_mb,
        }

    def translate_with_interpretability(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        model_id: str = "marian",
        checkpoint_path: Optional[str] = None,
        checkpoint_mode: str = "zero_shot",
        num_beams: int = 1,
        max_new_tokens: int = 64,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate translation and extract cross-attention weights and token probabilities."""
        if not text or not text.strip():
            return {
                "source_text": text,
                "translated_text": "",
                "src_tokens": [],
                "tgt_tokens": [],
                "token_probabilities": [],
                "cross_attention_matrix": None,
                "num_layers": 0,
                "num_heads": 0,
            }

        LanguageRegistry.validate_direction(src_lang, tgt_lang)

        resolved_ckpt = checkpoint_path
        if checkpoint_mode == "finetuned" and not resolved_ckpt:
            resolved_ckpt = self.get_finetuned_checkpoint_path(model_id, src_lang, tgt_lang)

        if model_id.startswith("marian"):
            resolved_model_id = f"marian_{src_lang}_{tgt_lang}"
        else:
            resolved_model_id = model_id

        # To ensure cross-attentions are captured, model must use eager attention
        model_wrapper = self.get_or_load_model(
            model_id=resolved_model_id,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            checkpoint_path=resolved_ckpt,
        )

        tokenizer = model_wrapper.tokenizer
        raw_model = model_wrapper.model

        # Ensure eager attention is enabled on config if needed
        if hasattr(raw_model.config, "_attn_implementation"):
            raw_model.config._attn_implementation = "eager"

        # Prepare inputs
        inputs = tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        if model_id == "nllb200":
            nllb_tgt = LanguageRegistry.get_nllb_code(tgt_lang)
            if hasattr(tokenizer, "lang_code_to_id") and nllb_tgt in tokenizer.lang_code_to_id:
                gen_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id[nllb_tgt]
            else:
                gen_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(nllb_tgt)
        elif model_id == "mbart50":
            mbart_tgt = LanguageRegistry.get_mbart_code(tgt_lang)
            gen_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id[mbart_tgt]

        t0 = time.perf_counter()
        with torch.no_grad():
            gen_out = raw_model.generate(
                **inputs,
                **gen_kwargs,
            )
            gen_tokens = gen_out.sequences

            # Forward pass with decoder_input_ids to extract cross-attention maps and logits
            fwd_out = raw_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                decoder_input_ids=gen_tokens,
                output_attentions=True,
            )
        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 2)

        # Extract cross attentions
        cross_attns = fwd_out.cross_attentions  # tuple of (num_layers), each (batch, heads, tgt_len, src_len)
        num_layers = len(cross_attns) if cross_attns else 0
        num_heads = 0

        attn_matrix = None
        if cross_attns:
            # Handle layer index bounds
            actual_layer_idx = layer_idx if (0 <= layer_idx < num_layers) else -1
            selected_layer = cross_attns[actual_layer_idx][0]  # (heads, tgt_len, src_len)
            num_heads = selected_layer.shape[0]
            if head_idx is not None and 0 <= head_idx < num_heads:
                attn_matrix = selected_layer[head_idx].cpu().numpy()  # (tgt_len, src_len)
            else:
                attn_matrix = selected_layer.mean(dim=0).cpu().numpy()  # average over heads -> (tgt_len, src_len)

        # Calculate softmax token probabilities
        logits = fwd_out.logits  # (1, tgt_len, vocab_size)
        probs = torch.softmax(logits, dim=-1)

        token_probs = []
        for i in range(gen_tokens.shape[1] - 1):
            next_token_id = gen_tokens[0, i + 1].item()
            p = probs[0, i, next_token_id].item()
            token_probs.append(round(p, 4))

        src_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        tgt_tokens = tokenizer.convert_ids_to_tokens(gen_tokens[0][1:])  # exclude initial token

        # Slice attention matrix: decoder step i produces tgt_token[i] (which is gen_tokens[0][i+1])
        if attn_matrix is not None:
            attn_matrix = attn_matrix[: len(tgt_tokens), : len(src_tokens)]

        translated_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
        if "\u094D" in translated_text or tgt_lang in ["hi", "mr"]:
            import re
            translated_text = re.sub(r"\s*्\s*", "्", translated_text)

        return {
            "source_text": text.strip(),
            "translated_text": translated_text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "model_id": resolved_model_id,
            "checkpoint_mode": checkpoint_mode if resolved_ckpt else "zero_shot",
            "checkpoint_path": resolved_ckpt or "base_pretrained",
            "src_tokens": src_tokens,
            "tgt_tokens": tgt_tokens,
            "token_probabilities": token_probs,
            "cross_attention_matrix": attn_matrix,
            "num_layers": num_layers,
            "num_heads": num_heads,
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
        """Clear cached models and reclaim full GPU memory."""
        for k in list(self._loaded_models.keys()):
            m = self._loaded_models.pop(k, None)
            if m is not None:
                try:
                    if hasattr(m, "model") and m.model is not None:
                        m.model.to("cpu")
                    del m
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Cleared inference model cache and reclaimed CUDA memory.")
