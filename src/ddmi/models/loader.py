"""Model loading with multiple quantization backends.

Extends the refusal-suppression loader pattern with INT8, GPTQ, AWQ support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch


@dataclass(frozen=True)
class ModelLoadConfig:
    """Configuration for loading a model with optional quantization."""

    model_id: str
    load_in_4bit: bool = False       # BitsAndBytes NF4
    load_in_8bit: bool = False       # BitsAndBytes INT8
    torch_dtype: str = "auto"        # 'auto', 'float16', 'bfloat16', 'float32'
    device_map: str = "auto"
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None  # 'eager', 'sdpa', 'flash_attention_2'

    def validate(self) -> None:
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot specify both load_in_4bit and load_in_8bit")


def load_model_and_tokenizer(
    config: ModelLoadConfig,
) -> Tuple[Any, Any]:
    """Load a model and tokenizer from HuggingFace Hub.

    Returns (model, tokenizer) tuple. Model is set to eval mode.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config.validate()

    kwargs: dict[str, Any] = {
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
    }

    # Resolve dtype
    dtype = resolve_torch_dtype(config.torch_dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    # Attention implementation
    if config.attn_implementation:
        kwargs["attn_implementation"] = config.attn_implementation

    # Quantization config
    if config.load_in_4bit or config.load_in_8bit:
        from transformers import BitsAndBytesConfig

        if config.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

    model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def resolve_torch_dtype(torch_dtype: str) -> Optional[torch.dtype]:
    """Resolve a string dtype to a torch.dtype."""
    if torch_dtype == "auto":
        return None
    return getattr(torch, torch_dtype)


def resolve_model_device(model: Any) -> torch.device:
    """Resolve the device a model is on."""
    if hasattr(model, "device"):
        return model.device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def count_parameters(model: Any) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def model_memory_mb(model: Any) -> float:
    """Estimate model memory usage in MB."""
    total_bytes = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 * 1024)
