"""Optional physics-aware output wrappers shared by every PINN method."""

from __future__ import annotations

import torch
import torch.nn as nn


class CavityHardBoundaryWrapper(nn.Module):
    """Enforce cavity velocity walls with a corner-smoothed moving lid.

    The lid discontinuity cannot be represented continuously at the two top
    corners. The configured corner width makes that regularization explicit;
    all remaining wall values are enforced exactly by construction.
    """

    def __init__(
        self,
        base: nn.Module,
        bounds: tuple[float, float, float, float],
        lid_velocity: float = 1.0,
        corner_width: float = 0.02,
    ) -> None:
        super().__init__()
        self.base = base
        self.bounds = tuple(float(value) for value in bounds)
        self.lid_velocity = float(lid_velocity)
        self.corner_width = max(float(corner_width), 1e-6)
        self.physics_formulation = "cavity_hard_boundary"

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        raw = self.base(coords)
        x0, x1, y0, y1 = self.bounds
        x = (coords[:, 0:1] - x0) / max(x1 - x0, 1e-12)
        y = (coords[:, 1:2] - y0) / max(y1 - y0, 1e-12)
        left = _smoothstep01(x / self.corner_width)
        right = _smoothstep01((1.0 - x) / self.corner_width)
        lid_extension = self.lid_velocity * y * left * right
        envelope = 16.0 * x * (1.0 - x) * y * (1.0 - y)
        u = lid_extension + envelope * raw[:, 0:1]
        v = envelope * raw[:, 1:2]
        p = raw[:, 2:3]
        return torch.cat([u, v, p], dim=1)


class StreamfunctionPressureWrapper(nn.Module):
    """Convert a two-output ``(psi, p)`` network into divergence-free ``u,v,p``."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.physics_formulation = "streamfunction_pressure"

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            working = coords if coords.requires_grad else coords.clone().detach().requires_grad_(True)
            output = self.base(working)
            psi = output[:, 0:1]
            p = output[:, 1:2]
            gradient = torch.autograd.grad(
                psi,
                working,
                grad_outputs=torch.ones_like(psi),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            u = gradient[:, 1:2]
            v = -gradient[:, 0:1]
            return torch.cat([u, v, p], dim=1)


def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)
