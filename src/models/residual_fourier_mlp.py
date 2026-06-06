"""Parameter-matched residual Fourier MLP for shared PINN comparisons."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class ResidualTanhBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(width, width)
        self.linear_2 = nn.Linear(width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.linear_1(inputs))
        return inputs + torch.tanh(self.linear_2(hidden))


class ResidualFourierMLP(nn.Module):
    """Fixed Fourier features, wall distances, and residual tanh blocks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        blocks: int,
        frequencies: Iterable[float],
        input_lower: Iterable[float],
        input_upper: Iterable[float],
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.frequencies = tuple(float(value) for value in frequencies)
        self.register_buffer("input_lower", torch.tensor(list(input_lower), dtype=torch.float32))
        self.register_buffer("input_upper", torch.tensor(list(input_upper), dtype=torch.float32))
        feature_dim = self.in_dim + 2 * self.in_dim * len(self.frequencies) + 4
        self.input_layer = nn.Linear(feature_dim, int(width))
        self.blocks = nn.ModuleList([ResidualTanhBlock(int(width)) for _ in range(int(blocks))])
        self.output_layer = nn.Linear(int(width), int(out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def normalize_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        lower = self.input_lower.to(device=inputs.device, dtype=inputs.dtype)
        upper = self.input_upper.to(device=inputs.device, dtype=inputs.dtype)
        return 2.0 * (inputs - lower) / (upper - lower).clamp_min(1e-12) - 1.0

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_inputs(inputs)
        fourier = []
        for frequency in self.frequencies:
            phase = math.pi * frequency * normalized
            fourier.extend([torch.sin(phase), torch.cos(phase)])
        x = normalized[:, 0:1]
        y = normalized[:, 1:2]
        wall_distances = torch.cat(
            [
                0.5 * (x + 1.0),
                0.5 * (1.0 - x),
                0.5 * (y + 1.0),
                0.5 * (1.0 - y),
            ],
            dim=1,
        )
        return torch.cat([normalized, *fourier, wall_distances], dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.input_layer(self.features(inputs)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_layer(hidden)


def parameter_matched_width(
    input_dim: int,
    output_dim: int,
    legacy_hidden_layers: Iterable[int],
    frequencies: Iterable[float],
    blocks: int = 4,
) -> tuple[int, int, int]:
    """Return width, enhanced parameter count, and legacy parameter count."""
    hidden = [int(value) for value in legacy_hidden_layers]
    sizes = [int(input_dim), *hidden, int(output_dim)]
    legacy_count = sum((sizes[i] + 1) * sizes[i + 1] for i in range(len(sizes) - 1))
    feature_dim = int(input_dim) + 2 * int(input_dim) * len(tuple(frequencies)) + 4

    def count(width: int) -> int:
        return (
            (feature_dim + 1) * width
            + int(blocks) * 2 * (width + 1) * width
            + (width + 1) * int(output_dim)
        )

    candidates = [(abs(count(width) - legacy_count), width, count(width)) for width in range(4, 513)]
    _difference, width, enhanced_count = min(candidates)
    return int(width), int(enhanced_count), int(legacy_count)
