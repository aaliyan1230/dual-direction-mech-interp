"""Text generation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch


@dataclass(frozen=True)
class TextGenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False
    max_input_length: Optional[int] = 512


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: Optional[TextGenerationConfig] = None,
    chat_template: bool = True,
) -> str:
    """Generate text from a prompt. Returns only the generated portion."""
    if config is None:
        config = TextGenerationConfig()

    if chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_input_length,
    )
    device = _resolve_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature if config.do_sample else None,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output[0, input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    config: Optional[TextGenerationConfig] = None,
    chat_template: bool = True,
) -> List[str]:
    """Generate text for multiple prompts (sequentially for memory safety)."""
    return [
        generate_text(model, tokenizer, p, config=config, chat_template=chat_template)
        for p in prompts
    ]


def _resolve_device(model: Any) -> torch.device:
    """Resolve model device."""
    if hasattr(model, "device"):
        return model.device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")
