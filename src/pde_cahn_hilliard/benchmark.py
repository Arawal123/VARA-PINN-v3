"""Manufactured moving-interface Cahn--Hilliard benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .autograd import gradient, spatial_laplacian


@dataclass
class CahnHilliardBenchmark:
    """Forced split-form Cahn--Hilliard problem with exact u and mu."""

    config: dict[str, Any]

    @property
    def epsilon(self) -> float:
        return float(self.config.get("epsilon", 0.04))

    @property
    def mobility(self) -> float:
        return float(self.config.get("mobility", 1.0))

    @property
    def delta(self) -> float:
        return float(self.config.get("delta", 1e-6))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        values = self.config.get("bounds", [0.0, 1.0, 0.0, 1.0])
        return tuple(float(value) for value in values)  # type: ignore[return-value]

    @property
    def t_bounds(self) -> tuple[float, float]:
        values = self.config.get("t_bounds", [0.0, 1.0])
        return float(values[0]), float(values[1])

    def center_and_radius(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the prescribed moving center and radius."""
        xc = 0.5 + 0.12 * torch.sin(2.0 * torch.pi * t)
        yc = 0.5 + 0.10 * torch.cos(2.0 * torch.pi * t)
        radius = 0.22 + 0.03 * torch.sin(2.0 * torch.pi * t)
        return xc, yc, radius

    def u_reference(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Evaluate the exact phase field without differentiating the model."""
        _validate_coordinates(coordinates)
        x, y, t = coordinates[:, :1], coordinates[:, 1:2], coordinates[:, 2:3]
        xc, yc, radius = self.center_and_radius(t)
        radial = torch.sqrt(
            (x - xc).square() + (y - yc).square() + self.delta**2
        )
        width = (2.0**0.5) * self.epsilon
        return torch.tanh((radial - radius) / width)

    def mu_reference(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Compute exact chemical potential consistently from exact u."""
        work, detached_input = _differentiable_coordinates(coordinates)
        u = self.u_reference(work)
        mu = -self.epsilon**2 * spatial_laplacian(u, work) + (u.pow(3) - u)
        return mu.detach() if detached_input else mu

    def exact(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return exact columns ordered as (u, mu)."""
        work, detached_input = _differentiable_coordinates(coordinates)
        u = self.u_reference(work)
        mu = -self.epsilon**2 * spatial_laplacian(u, work) + (u.pow(3) - u)
        values = torch.cat((u, mu), dim=1)
        return values.detach() if detached_input else values

    def forcing(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Compute detached f_ch = u_t - M Laplacian(mu) for supplied points."""
        work = coordinates.detach().clone().requires_grad_(True)
        u = self.u_reference(work)
        mu = -self.epsilon**2 * spatial_laplacian(u, work) + (u.pow(3) - u)
        u_t = gradient(u, work)[:, 2:3]
        forcing = u_t - self.mobility * spatial_laplacian(mu, work)
        return forcing.detach()

    def boundary_values(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.exact(coordinates)

    def initial_values(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.exact(coordinates)

    @staticmethod
    def interface_band(reference_u: torch.Tensor) -> torch.Tensor:
        """Evaluation-only interface band; never sent to the controller."""
        return reference_u.abs() < 0.8

    @staticmethod
    def interface_core(reference_u: torch.Tensor) -> torch.Tensor:
        """Sharper evaluation-only core around the zero level set."""
        return reference_u.abs() < 0.4


def _differentiable_coordinates(
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    _validate_coordinates(coordinates)
    if coordinates.requires_grad:
        return coordinates, False
    return coordinates.detach().clone().requires_grad_(True), True


def _validate_coordinates(coordinates: torch.Tensor) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Cahn--Hilliard coordinates must have shape [N, 3].")
