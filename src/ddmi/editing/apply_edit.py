"""Directional ablation applied to model weights (tensor operations).

Follows the same API as refusal-suppression's apply_edit.py:
  - apply_directional_ablation_tensor for single weight matrices
  - apply_direction_to_model for full model surgery
  - find_editable_modules for architecture-agnostic module discovery
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# Architecture-agnostic module name patterns
ATTN_OUT_PATTERNS = (
    ".self_attn.o_proj",
    ".self_attn.out_proj",
    ".attention.o_proj",
    ".attention.out_proj",
    ".attn.out_proj",
    ".attn.c_proj",
    ".attention.wo",
)

MLP_DOWN_PATTERNS = (
    ".mlp.down_proj",
    ".mlp.fc2",
    ".mlp.c_proj",
    ".feed_forward.w2",
    ".feed_forward.down_proj",
)

INPUT_AXIS = "input"
OUTPUT_AXIS = "output"
AUTO_AXIS = "auto"


@dataclass(frozen=True)
class EditSpec:
    """Specification for a directional ablation edit."""

    strength: float = 1.0
    norm_preserving: bool = False
    axis: str = AUTO_AXIS

    def validate(self) -> None:
        if self.strength < 0.0:
            raise ValueError(f"strength must be >= 0, got {self.strength}")
        if self.axis not in (INPUT_AXIS, OUTPUT_AXIS, AUTO_AXIS):
            raise ValueError(f"axis must be one of input/output/auto, got {self.axis}")


@dataclass(frozen=True)
class ModelEditTarget:
    """A discovered editable module in a model."""

    module_name: str
    module_type: str  # 'attn_out' | 'mlp_down'
    layer_index: Optional[int]
    input_dim: int
    output_dim: int


@dataclass
class ActivationDirectionAblator:
    """Apply directional ablation to module activations via forward hooks."""

    module_names: Sequence[str]
    direction: Sequence[float]
    spec: EditSpec = field(default_factory=EditSpec)
    _hooks: List[Any] = field(default_factory=list)

    def attach(self, model: Any) -> "ActivationDirectionAblator":
        modules = dict(model.named_modules())
        for module_name in self.module_names:
            if module_name not in modules:
                raise KeyError(f"Module '{module_name}' not found in model")
            module = modules[module_name]

            def hook(_module: Any, _inputs: Any, output: Any, name: str = module_name) -> Any:
                del name
                return apply_directional_ablation_output(output, self.direction, self.spec)

            self._hooks.append(module.register_forward_hook(hook))
        return self

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


@dataclass
class LayerwiseActivationDirectionAblator:
    """Apply per-layer directional ablation to module activations via forward hooks."""

    directions_by_module: Dict[str, Sequence[float]]
    spec: EditSpec = field(default_factory=EditSpec)
    _hooks: List[Any] = field(default_factory=list)

    def attach(self, model: Any) -> "LayerwiseActivationDirectionAblator":
        modules = dict(model.named_modules())
        for module_name, direction in self.directions_by_module.items():
            if module_name not in modules:
                raise KeyError(f"Module '{module_name}' not found in model")
            module = modules[module_name]

            def hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                layer_direction: Sequence[float] = direction,
                name: str = module_name,
            ) -> Any:
                del name
                return apply_directional_ablation_output(output, layer_direction, self.spec)

            self._hooks.append(module.register_forward_hook(hook))
        return self

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def apply_directional_ablation_tensor(
    matrix: torch.Tensor,
    direction: Sequence[float],
    spec: EditSpec = EditSpec(),
) -> torch.Tensor:
    """Apply directional ablation to a weight matrix (tensor path).

    INPUT_AXIS:  W' = W - α * (W @ d̂) ⊗ d̂   (removes direction from input space)
    OUTPUT_AXIS: W' = W - α * d̂ ⊗ (d̂ᵀ @ W)  (removes direction from output space)
    """
    spec.validate()
    device = matrix.device
    dtype = matrix.dtype

    d = torch.tensor(direction, device=device, dtype=torch.float32)
    d = d / d.norm()

    # Resolve axis
    axis = spec.axis
    if axis == AUTO_AXIS:
        if len(direction) == matrix.shape[1]:
            axis = INPUT_AXIS
        elif len(direction) == matrix.shape[0]:
            axis = OUTPUT_AXIS
        else:
            raise ValueError(
                f"Direction dim {len(direction)} matches neither input ({matrix.shape[1]}) "
                f"nor output ({matrix.shape[0]}) of weight matrix"
            )

    W = matrix.float()

    if axis == INPUT_AXIS:
        # W' = W - α * (W d̂) d̂ᵀ
        Wd = W @ d  # (out_dim,)
        edit = spec.strength * torch.outer(Wd, d)  # (out_dim, in_dim)
    else:
        # W' = W - α * d̂ (d̂ᵀ W)
        dW = d @ W  # (in_dim,)
        edit = spec.strength * torch.outer(d, dW)  # (out_dim, in_dim)

    W_edited = W - edit

    if spec.norm_preserving:
        if axis == INPUT_AXIS:
            # Preserve column norms
            orig_norms = W.norm(dim=0, keepdim=True)
            new_norms = W_edited.norm(dim=0, keepdim=True).clamp(min=1e-12)
            W_edited = W_edited * (orig_norms / new_norms)
        else:
            # Preserve row norms
            orig_norms = W.norm(dim=1, keepdim=True)
            new_norms = W_edited.norm(dim=1, keepdim=True).clamp(min=1e-12)
            W_edited = W_edited * (orig_norms / new_norms)

    return W_edited.to(dtype)


def apply_directional_ablation_activation(
    tensor: torch.Tensor,
    direction: Sequence[float],
    spec: EditSpec = EditSpec(),
) -> torch.Tensor:
    """Remove a direction component from the last dimension of an activation tensor."""
    spec.validate()

    d = torch.tensor(direction, device=tensor.device, dtype=torch.float32)
    d = d / d.norm().clamp(min=1e-12)

    activations = tensor.float()
    projection = torch.einsum("...d,d->...", activations, d)
    edited = activations - spec.strength * projection.unsqueeze(-1) * d

    if spec.norm_preserving:
        orig_norms = activations.norm(dim=-1, keepdim=True)
        new_norms = edited.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        edited = edited * (orig_norms / new_norms)

    return edited.to(tensor.dtype)


def apply_directional_ablation_output(
    output: Any,
    direction: Sequence[float],
    spec: EditSpec = EditSpec(),
) -> Any:
    """Apply activation ablation to a module output, preserving tuple structure."""
    if isinstance(output, tuple):
        edited_first = apply_directional_ablation_activation(output[0], direction, spec)
        return (edited_first, *output[1:])
    return apply_directional_ablation_activation(output, direction, spec)


def find_editable_modules(
    model: Any,
    target_module_types: Optional[Sequence[str]] = None,
    layers: Optional[Sequence[int]] = None,
) -> List[ModelEditTarget]:
    """Walk model.named_modules() and find weight matrices matching known patterns."""
    if target_module_types is None:
        target_module_types = ["attn_out", "mlp_down"]

    targets: List[ModelEditTarget] = []

    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            continue

        module_type = infer_module_type(name)
        if module_type is None or module_type not in target_module_types:
            continue

        layer_idx = extract_layer_index(name)
        if layers is not None and layer_idx not in layers:
            continue

        targets.append(ModelEditTarget(
            module_name=name,
            module_type=module_type,
            layer_index=layer_idx,
            input_dim=weight.shape[1],
            output_dim=weight.shape[0],
        ))

    return targets


def infer_module_type(module_name: str) -> Optional[str]:
    """Infer module type from name patterns."""
    for pattern in ATTN_OUT_PATTERNS:
        if module_name.endswith(pattern):
            return "attn_out"
    for pattern in MLP_DOWN_PATTERNS:
        if module_name.endswith(pattern):
            return "mlp_down"
    return None


def extract_layer_index(module_name: str) -> Optional[int]:
    """Extract layer index from module name."""
    match = re.search(r"(?:layers|layer|h|blocks|block)\.(\d+)", module_name)
    if match:
        return int(match.group(1))
    return None


def snapshot_module_weights(
    model: Any, targets: Sequence[ModelEditTarget],
) -> Dict[str, torch.Tensor]:
    """Snapshot current weights for later restoration."""
    modules = dict(model.named_modules())
    snapshots: Dict[str, torch.Tensor] = {}
    for target in targets:
        module = modules[target.module_name]
        snapshots[target.module_name] = module.weight.data.detach().clone()
    return snapshots


def restore_module_weights(
    model: Any, snapshots: Dict[str, torch.Tensor],
) -> None:
    """Restore weights from snapshot."""
    modules = dict(model.named_modules())
    for name, weight in snapshots.items():
        modules[name].weight.data.copy_(weight)


def apply_direction_to_model(
    model: Any,
    direction: Sequence[float],
    targets: Sequence[ModelEditTarget],
    spec: EditSpec = EditSpec(),
) -> List[ModelEditTarget]:
    """Apply directional ablation to all target modules in a model."""
    modules = dict(model.named_modules())
    applied: List[ModelEditTarget] = []

    for target in targets:
        module = modules[target.module_name]
        edited = apply_directional_ablation_tensor(module.weight.data, direction, spec)
        module.weight.data.copy_(edited)
        applied.append(target)

    return applied


def serialize_targets(targets: Sequence[ModelEditTarget]) -> List[Dict[str, Any]]:
    """Serialize targets for JSON output."""
    return [
        {
            "module_name": t.module_name,
            "module_type": t.module_type,
            "layer_index": t.layer_index,
            "input_dim": t.input_dim,
            "output_dim": t.output_dim,
        }
        for t in targets
    ]
