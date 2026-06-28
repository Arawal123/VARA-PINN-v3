"""Pointwise residuals for the isolated PDE generalization benchmarks."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .autograd import gradient, spatial_laplacian
from .benchmarks import (
    AdvectionDiffusionBenchmark,
    AllenCahnBenchmark,
    Burgers2DBenchmark,
    ManufacturedBenchmark,
)


def compute_residuals(
    model: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    benchmark: ManufacturedBenchmark,
) -> dict[str, torch.Tensor]:
    """Return named signed PDE residuals plus a scalar residual magnitude."""
    if not coordinates.requires_grad:
        coordinates = coordinates.detach().clone().requires_grad_(True)
    prediction = model(coordinates)
    if isinstance(benchmark, Burgers2DBenchmark):
        if prediction.shape[1] != 2:
            raise ValueError("The Burgers model must output exactly (u, v).")
        u, v = prediction[:, :1], prediction[:, 1:2]
        grad_u = gradient(u, coordinates)
        grad_v = gradient(v, coordinates)
        forcing = benchmark.forcing(coordinates).to(prediction)
        f_u = (
            grad_u[:, 2:3]
            + u * grad_u[:, 0:1]
            + v * grad_u[:, 1:2]
            - benchmark.nu * spatial_laplacian(u, coordinates)
            - forcing[:, :1]
        )
        f_v = (
            grad_v[:, 2:3]
            + u * grad_v[:, 0:1]
            + v * grad_v[:, 1:2]
            - benchmark.nu * spatial_laplacian(v, coordinates)
            - forcing[:, 1:2]
        )
        magnitude = torch.sqrt(0.5 * (f_u.square() + f_v.square()) + 1e-20)
        return {"f_u": f_u, "f_v": f_v, "pde_residual": magnitude}

    if prediction.shape[1] != 1:
        raise ValueError(f"The {benchmark.name} model must output exactly one scalar field.")
    u = prediction[:, :1]
    grad_u = gradient(u, coordinates)
    forcing = benchmark.forcing(coordinates).to(prediction)
    if isinstance(benchmark, AllenCahnBenchmark):
        f_ac = (
            grad_u[:, 2:3]
            - benchmark.eps**2 * spatial_laplacian(u, coordinates)
            + (u.pow(3) - u)
            - forcing
        )
        return {"f_ac": f_ac, "pde_residual": f_ac.abs()}
    if isinstance(benchmark, AdvectionDiffusionBenchmark):
        ax, ay = benchmark.velocity
        f_advdiff = (
            grad_u[:, 2:3]
            + ax * grad_u[:, 0:1]
            + ay * grad_u[:, 1:2]
            - benchmark.kappa * spatial_laplacian(u, coordinates)
            - forcing
        )
        return {"f_advdiff": f_advdiff, "pde_residual": f_advdiff.abs()}
    raise TypeError(f"Unsupported benchmark type: {type(benchmark).__name__}")
