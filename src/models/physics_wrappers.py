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
        lid_lifting: str = "linear",
        lid_vertical_power: int = 6,
    ) -> None:
        super().__init__()
        self.base = base
        self.bounds = tuple(float(value) for value in bounds)
        self.lid_velocity = float(lid_velocity)
        self.corner_width = max(float(corner_width), 1e-6)
        self.lid_lifting = str(lid_lifting).lower()
        self.lid_vertical_power = max(2, int(lid_vertical_power))
        if self.lid_lifting not in {"linear", "harmonic", "divergence_free"}:
            raise ValueError(
                "cavity hard-boundary lid_lifting must be 'linear', "
                "'harmonic', or 'divergence_free'."
            )
        self.physics_formulation = "cavity_hard_boundary"

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        raw = self.base(coords)
        x0, x1, y0, y1 = self.bounds
        x = (coords[:, 0:1] - x0) / max(x1 - x0, 1e-12)
        y = (coords[:, 1:2] - y0) / max(y1 - y0, 1e-12)
        left_arg = x / self.corner_width
        right_arg = (1.0 - x) / self.corner_width
        left = _smoothstep01(left_arg)
        right = _smoothstep01(right_arg)
        horizontal_lift = left * right
        if self.lid_lifting == "divergence_free":
            # Derive the lifting from
            #   psi = U * Ly * eta^m(eta - 1) * h(xi).
            # It gives the regularized moving lid at eta=1, stationary other
            # walls, and exactly zero divergence without prescribing a CFD
            # solution or a Couette-like interior profile. m localizes the
            # geometric lift near the moving lid so corner smoothing does not
            # create artificial side jets deep inside the cavity.
            left_derivative = (
                _smoothstep01_derivative(left_arg) / self.corner_width
            )
            right_derivative = (
                -_smoothstep01_derivative(right_arg) / self.corner_width
            )
            horizontal_derivative = (
                left_derivative * right + left * right_derivative
            )
            power = self.lid_vertical_power
            y_power = y.pow(power)
            vertical_streamfunction = y_power * (y - 1.0)
            vertical_derivative = (
                power * y.pow(power - 1) * (y - 1.0) + y_power
            )
            aspect = (y1 - y0) / max(x1 - x0, 1e-12)
            u_lift = (
                self.lid_velocity
                * vertical_derivative
                * horizontal_lift
            )
            v_lift = (
                -self.lid_velocity
                * aspect
                * vertical_streamfunction
                * horizontal_derivative
            )
        elif self.lid_lifting == "harmonic":
            # A reference-free, rapidly decaying lifting avoids imposing a
            # Couette-like horizontal flow through the whole cavity. It still
            # satisfies the regularized lid and all stationary walls exactly.
            aspect_wave = torch.as_tensor(
                torch.pi * max((y1 - y0) / max(x1 - x0, 1e-12), 1e-6),
                dtype=coords.dtype,
                device=coords.device,
            )
            vertical_lift = torch.sinh(aspect_wave * y) / torch.sinh(aspect_wave)
            u_lift = self.lid_velocity * vertical_lift * horizontal_lift
            v_lift = torch.zeros_like(u_lift)
        else:
            vertical_lift = y
            u_lift = self.lid_velocity * vertical_lift * horizontal_lift
            v_lift = torch.zeros_like(u_lift)
        envelope = 16.0 * x * (1.0 - x) * y * (1.0 - y)
        u = u_lift + envelope * raw[:, 0:1]
        v = v_lift + envelope * raw[:, 1:2]
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


def _smoothstep01_derivative(value: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(value, 0.0, 1.0)
    derivative = 6.0 * clipped * (1.0 - clipped)
    return torch.where(
        (value > 0.0) & (value < 1.0),
        derivative,
        torch.zeros_like(derivative),
    )
