"""NLLB-200 Model Implementation.
Master Decision Register Section 3, PRD Section 7.3, Manifest Section 1.
"""

from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.models.base import BaseTranslationModel
from src.registry.language_registry import LanguageRegistry
from src.registry.model_registry import ModelRegistry
from src.utils.logger import setup_logger

logger = setup_logger("nllb200_model")


class NllbTranslationModel(BaseTranslationModel):
    """Wrapper for facebook/nllb-200-distilled-600M."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        model_meta = ModelRegistry.get_model("nllb200")
        super().__init__(model_id="nllb200", device=device)
        self.default_checkpoint = checkpoint or model_meta.checkpoint

    def load(self, checkpoint_path_or_name: Optional[str] = None) -> None:
        ckpt = checkpoint_path_or_name or self.default_checkpoint
        logger.info(f"Loading NLLB-200 model from '{ckpt}' on device '{self.device}'")
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
        self.model.to(self.device)
        self.model.eval()

    def prepare_inputs(
        self,
        sources: List[str],
        src_lang: str,
        tgt_lang: str,
        max_length: int = 128,
        targets: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")

        nllb_src = LanguageRegistry.get_nllb_code(src_lang)
        nllb_tgt = LanguageRegistry.get_nllb_code(tgt_lang)
        self.tokenizer.src_lang = nllb_src

        inputs = self.tokenizer(
            sources,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        model_inputs = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
        }

        if targets is not None:
            self.tokenizer.src_lang = nllb_tgt
            labels = self.tokenizer(
                targets,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            labels[labels == self.tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels.to(self.device)
            # Reset back to source lang
            self.tokenizer.src_lang = nllb_src

        return model_inputs

    def translate(
        self,
        texts: Union[str, List[str]],
        src_lang: str = "en",
        tgt_lang: str = "hi",
        num_beams: int = 5,
        max_new_tokens: int = 128,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        do_sample: bool = False,
        batch_size: int = 8,
        **kwargs,
    ) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded. Call load() first.")

        nllb_src = LanguageRegistry.get_nllb_code(src_lang)
        nllb_tgt = LanguageRegistry.get_nllb_code(tgt_lang)
        self.tokenizer.src_lang = nllb_src

        # Get target language token ID
        if (
            hasattr(self.tokenizer, "lang_code_to_id")
            and nllb_tgt in self.tokenizer.lang_code_to_id
        ):
            forced_bos_token_id = self.tokenizer.lang_code_to_id[nllb_tgt]
        else:
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(nllb_tgt)

        all_decoded: List[str] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=early_stopping,
                    do_sample=do_sample,
                    **kwargs,
                )

            decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            all_decoded.extend([t.strip() for t in decoded])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_decoded
