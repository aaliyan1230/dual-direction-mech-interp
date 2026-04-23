"""Activation recording via raw PyTorch forward hooks.

No TransformerLens dependency — pure register_forward_hook.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass
class ActivationRecorder:
    """Attach forward hooks to named modules and record their outputs.

    Records the last-token hidden state from each specified module.
    """

    module_names: List[str]
    token_position: str = "last"  # 'last' | 'all'
    outputs: Dict[str, List[List[float]]] = field(default_factory=dict)
    _hooks: List[Any] = field(default_factory=list)

    def attach(self, model: Any) -> "ActivationRecorder":
        """Register forward hooks on the specified modules."""
        modules = dict(model.named_modules())
        for module_name in self.module_names:
            if module_name not in modules:
                raise KeyError(
                    f"Module '{module_name}' not found in model. "
                    f"Available: {[n for n, _ in model.named_modules() if '.' in n][:20]}..."
                )
            module = modules[module_name]

            def hook(_module: Any, _inputs: Any, output: Any, name: str = module_name) -> None:
                tensor = _unwrap_output(output)
                vec = extract_last_token_vector(tensor)
                self.outputs.setdefault(name, []).append(vec)

            self._hooks.append(module.register_forward_hook(hook))
        return self

    def clear(self) -> None:
        """Clear recorded activations."""
        self.outputs.clear()

    def close(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def _unwrap_output(output: Any) -> torch.Tensor:
    """Unwrap tuple outputs (common in transformer layers)."""
    if isinstance(output, tuple):
        return output[0]
    return output


def extract_last_token_vector(tensor: torch.Tensor) -> List[float]:
    """Extract the last-token hidden state as a Python list."""
    # tensor shape: (batch, seq_len, hidden_dim)
    return tensor[0, -1, :].detach().float().cpu().tolist()


def get_residual_stream_names(model: Any, prefix: str = "model.layers") -> List[str]:
    """Auto-discover residual stream (layer output) module names.

    Returns names like 'model.layers.0', 'model.layers.1', etc.
    """
    names = []
    for name, _ in model.named_modules():
        if name.startswith(prefix) and name.count(".") == 2:
            # e.g., 'model.layers.14' (exactly 2 dots after 'model')
            names.append(name)
    # Sort by layer index
    names.sort(key=lambda n: int(n.split(".")[-1]))
    return names


def collect_activations_batched(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layer_names: Sequence[str],
    max_input_length: Optional[int] = 512,
    chat_template: bool = True,
) -> Dict[str, List[List[float]]]:
    """Collect last-token activations for each prompt at each layer.

    Returns: {layer_name: [[float, ...], ...]} where outer list aligns with prompts.
    """
    recorder = ActivationRecorder(module_names=list(layer_names))
    recorder.attach(model)

    device = _resolve_device(model)

    try:
        for prompt in prompts:
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
                max_length=max_input_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                model(**inputs, use_cache=False)
    finally:
        recorder.close()

    return recorder.outputs


def _resolve_device(model: Any) -> torch.device:
    """Resolve the device a model is on."""
    if hasattr(model, "device"):
        return model.device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")
