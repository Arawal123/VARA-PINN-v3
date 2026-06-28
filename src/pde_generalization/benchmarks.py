"""Manufactured references and forcing terms for three independent PDEs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from .autograd import gradient, spatial_laplacian


@dataclass
class ManufacturedBenchmark(ABC):
    """Common interface for exact manufactured PDE benchmarks."""

    params: dict[str, Any]

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable benchmark name."""

    @property
    @abstractmethod
    def output_names(self) -> tuple[str, ...]:
        """Names of model output components."""

    @abstractmethod
    def exact(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Evaluate the analytical manufactured field."""

    @abstractmethod
    def forcing(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Evaluate detached forcing columns for the governing PDE."""

    @abstractmethod
    def hard_region_mask(
        self,
        coordinates: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return an evaluation-only localized hard-region mask."""

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        values = self.params.get("bounds", [0.0, 1.0, 0.0, 1.0])
        return tuple(float(value) for value in values)  # type: ignore[return-value]

    @property
    def t_bounds(self) -> tuple[float, float]:
        values = self.params.get("t_bounds", [0.0, 1.0])
        return float(values[0]), float(values[1])

    def boundary_values(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.exact(coordinates)

    def initial_values(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.exact(coordinates)


class Burgers2DBenchmark(ManufacturedBenchmark):
    """Coupled viscous 2D Burgers system with two moving structures."""

    name = "burgers2d"
    output_names = ("u", "v")

    @property
    def nu(self) -> float:
        return float(self.params.get("nu", 0.01))

    def exact(self, coordinates: torch.Tensor) -> torch.Tensor:
        x, y, t = _columns(coordinates)
        amplitude = float(self.params.get("amplitude", 0.75))
        sigma = float(self.params.get("sigma", 0.11))
        pi = torch.pi
        base = torch.sin(pi * x) * torch.sin(pi * y) * torch.exp(-t)
        xc1 = 0.22 + 0.48 * t
        yc1 = 0.32 + 0.12 * torch.sin(2.0 * pi * t)
        xc2 = 0.78 - 0.42 * t
        yc2 = 0.68 - 0.10 * torch.cos(2.0 * pi * t)
        g1 = torch.exp(-((x - xc1).square() + (y - yc1).square()) / sigma**2)
        g2 = torch.exp(-((x - xc2).square() + (y - yc2).square()) / sigma**2)
        u = base + amplitude * g1 - 0.20 * amplitude * g2
        v = -base + 0.85 * amplitude * g2 - 0.15 * amplitude * g1
        return torch.cat((u, v), dim=1)

    def forcing(self, coordinates: torch.Tensor) -> torch.Tensor:
        coords = coordinates.detach().clone().requires_grad_(True)
        values = self.exact(coords)
        u, v = values[:, :1], values[:, 1:2]
        grad_u = gradient(u, coords)
        grad_v = gradient(v, coords)
        f_u = (
            grad_u[:, 2:3]
            + u * grad_u[:, 0:1]
            + v * grad_u[:, 1:2]
            - self.nu * spatial_laplacian(u, coords)
        )
        f_v = (
            grad_v[:, 2:3]
            + u * grad_v[:, 0:1]
            + v * grad_v[:, 1:2]
            - self.nu * spatial_laplacian(v, coords)
        )
        return torch.cat((f_u, f_v), dim=1).detach()

    def hard_region_mask(
        self,
        coordinates: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        del reference
        x, y, t = _columns(coordinates)
        sigma = float(self.params.get("sigma", 0.11))
        xc1 = 0.22 + 0.48 * t
        yc1 = 0.32 + 0.12 * torch.sin(2.0 * torch.pi * t)
        xc2 = 0.78 - 0.42 * t
        yc2 = 0.68 - 0.10 * torch.cos(2.0 * torch.pi * t)
        d1 = (x - xc1).square() + (y - yc1).square()
        d2 = (x - xc2).square() + (y - yc2).square()
        return ((d1 <= (1.8 * sigma) ** 2) | (d2 <= (1.8 * sigma) ** 2)).squeeze(1)


class AllenCahnBenchmark(ManufacturedBenchmark):
    """Allen--Cahn equation with a moving circular diffuse interface."""

    name = "allen_cahn"
    output_names = ("u",)

    @property
    def eps(self) -> float:
        return float(self.params.get("eps", 0.04))

    def exact(self, coordinates: torch.Tensor) -> torch.Tensor:
        x, y, t = _columns(coordinates)
        pi = torch.pi
        xc = 0.5 + 0.12 * torch.sin(2.0 * pi * t)
        yc = 0.5 + 0.10 * torch.cos(2.0 * pi * t)
        radius = 0.22 + 0.03 * torch.sin(2.0 * pi * t)
        radial = torch.sqrt((x - xc).square() + (y - yc).square() + 1e-10)
        return torch.tanh((radial - radius) / ((2.0**0.5) * self.eps))

    def forcing(self, coordinates: torch.Tensor) -> torch.Tensor:
        coords = coordinates.detach().clone().requires_grad_(True)
        u = self.exact(coords)
        grad_u = gradient(u, coords)
        forcing = (
            grad_u[:, 2:3]
            - self.eps**2 * spatial_laplacian(u, coords)
            + (u.pow(3) - u)
        )
        return forcing.detach()

    def hard_region_mask(
        self,
        coordinates: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        del coordinates
        return (reference[:, 0].abs() < 0.8)


class AdvectionDiffusionBenchmark(ManufacturedBenchmark):
    """Transport-dominated scalar equation with a moving Gaussian layer."""

    name = "advection_diffusion"
    output_names = ("u",)

    @property
    def kappa(self) -> float:
        return float(self.params.get("kappa", 0.01))

    @property
    def velocity(self) -> tuple[float, float]:
        values = self.params.get("advection_velocity", [1.0, 0.5])
        return float(values[0]), float(values[1])

    def center(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 0.20 + 0.60 * t, 0.32 + 0.18 * torch.sin(torch.pi * t)

    def exact(self, coordinates: torch.Tensor) -> torch.Tensor:
        x, y, t = _columns(coordinates)
        sigma = float(self.params.get("sigma", 0.09))
        xc, yc = self.center(t)
        layer = torch.exp(-((x - xc).square() + (y - yc).square()) / sigma**2)
        background = 0.25 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.exp(-t)
        return layer + background

    def forcing(self, coordinates: torch.Tensor) -> torch.Tensor:
        coords = coordinates.detach().clone().requires_grad_(True)
        u = self.exact(coords)
        grad_u = gradient(u, coords)
        ax, ay = self.velocity
        forcing = (
            grad_u[:, 2:3]
            + ax * grad_u[:, 0:1]
            + ay * grad_u[:, 1:2]
            - self.kappa * spatial_laplacian(u, coords)
        )
        return forcing.detach()

    def hard_region_mask(
        self,
        coordinates: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        del reference
        x, y, t = _columns(coordinates)
        sigma = float(self.params.get("sigma", 0.09))
        xc, yc = self.center(t)
        radius = torch.sqrt((x - xc).square() + (y - yc).square())
        return ((radius >= 0.35 * sigma) & (radius <= 1.85 * sigma)).squeeze(1)


def build_benchmark(config: dict[str, Any]) -> ManufacturedBenchmark:
    """Construct a supported manufactured benchmark and reject unknown names."""
    name = str(config.get("benchmark", "")).lower().replace("-", "_")
    params = dict(config.get("benchmark_params", {}))
    classes: dict[str, type[ManufacturedBenchmark]] = {
        "burgers2d": Burgers2DBenchmark,
        "burgers_2d": Burgers2DBenchmark,
        "allen_cahn": AllenCahnBenchmark,
        "advection_diffusion": AdvectionDiffusionBenchmark,
    }
    if name not in classes:
        raise ValueError(
            f"Unsupported PDE generalization benchmark {name!r}; "
            f"expected one of {sorted(set(classes))}."
        )
    return classes[name](params)


def _columns(coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("PDE coordinates must have shape [N, 3] ordered as (x, y, t).")
    return coordinates[:, :1], coordinates[:, 1:2], coordinates[:, 2:3]
