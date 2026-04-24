"""Unit tests for activation-space directional ablation."""
from __future__ import annotations

import torch

from ddmi.editing.apply_edit import (
    ActivationDirectionAblator,
    EditSpec,
    LayerwiseActivationDirectionAblator,
    apply_directional_ablation_activation,
)


def test_apply_directional_ablation_activation_removes_component() -> None:
    tensor = torch.tensor([[[3.0, 4.0], [1.0, 2.0]]])
    direction = [1.0, 0.0]

    edited = apply_directional_ablation_activation(tensor, direction, EditSpec(strength=1.0))

    assert torch.allclose(edited[..., 0], torch.zeros_like(edited[..., 0]), atol=1e-6)
    assert torch.allclose(edited[..., 1], tensor[..., 1], atol=1e-6)


def test_activation_direction_ablator_edits_module_output() -> None:
    model = torch.nn.Sequential(torch.nn.Identity())
    ablator = ActivationDirectionAblator(["0"], [1.0, 0.0], EditSpec(strength=1.0)).attach(model)

    try:
        output = model(torch.tensor([[5.0, 7.0]]))
    finally:
        ablator.close()

    assert torch.allclose(output[..., 0], torch.zeros_like(output[..., 0]), atol=1e-6)
    assert torch.allclose(output[..., 1], torch.tensor([7.0]), atol=1e-6)


def test_layerwise_activation_direction_ablator_uses_per_layer_directions() -> None:
    class TupleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Identity()
            self.second = torch.nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.first(x)
            return self.second(x)

    model = TupleModel()
    ablator = LayerwiseActivationDirectionAblator(
        {
            "first": [1.0, 0.0],
            "second": [0.0, 1.0],
        },
        EditSpec(strength=1.0),
    ).attach(model)

    try:
        output = model(torch.tensor([[5.0, 7.0]]))
    finally:
        ablator.close()

    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)