"""Base Translation Model Interface.
TDD Section 10 & Spec Section 2 (TASK-008).
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn


class BaseTranslationModel(ABC):
    """Unified abstraction for all translation model families."""

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None

    @abstractmethod
    def load(self, checkpoint_path_or_name: Optional[str] = None) -> None:
        """Load the tokenizer and model weights."""
        pass

    @abstractmethod
    def prepare_inputs(
        self,
        sources: List[str],
        src_lang: str,
        tgt_lang: str,
        max_length: int = 128,
        targets: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize input sources (and optionally targets for training)."""
        pass

    @abstractmethod
    def translate(
        self,
        texts: Union[str, List[str]],
        src_lang: str,
        tgt_lang: str,
        num_beams: int = 5,
        max_new_tokens: int = 128,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        do_sample: bool = False,
        **kwargs,
    ) -> List[str]:
        """Generate translations for source texts."""
        pass

    def count_parameters(self) -> Dict[str, int]:
        """Count total, trainable, and frozen parameters."""
        if self.model is None:
            return {"total": 0, "trainable": 0, "frozen": 0}
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def get_model_size_mb(self) -> float:
        """Estimate model memory size in MB."""
        if self.model is None:
            return 0.0
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024**2)

    def to_device(self, device: str) -> None:
        """Move model to specified device."""
        self.device = device
        if self.model is not None:
            self.model.to(device)

    def save_checkpoint(self, save_dir: str) -> None:
        """Save model and tokenizer to directory."""
        os.makedirs(save_dir, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(save_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for VRAM efficiency."""
        if self.model is not None and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
