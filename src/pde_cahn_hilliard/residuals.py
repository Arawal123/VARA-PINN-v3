"""Split-form Cahn--Hilliard pointwise residuals."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .autograd import gradient, spatial_laplacian
from .benchmark import CahnHilliardBenchmark


def compute_cahn_hilliard_residuals(
    model: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    benchmark: CahnHilliardBenchmark,
    forcing: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return phase and chemical-potential residual columns.

    The learned model is differentiated only to second order. The manufactured
    forcing may involve higher exact derivatives, but it is detached before it
    enters optimization and can be cached for a fixed training batch.
    """
    if not coordinates.requires_grad:
        coordinates = coordinates.detach().clone().requires_grad_(True)
    prediction = model(coordinates)
    if prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("Split-form Cahn--Hilliard models must output (u, mu).")
    u, mu = prediction[:, :1], prediction[:, 1:2]
    u_t = gradient(u, coordinates)[:, 2:3]
    laplacian_mu = spatial_laplacian(mu, coordinates)
    laplacian_u = spatial_laplacian(u, coordinates)
    if forcing is None:
        forcing = benchmark.forcing(coordinates)
    forcing = forcing.detach().to(device=u.device, dtype=u.dtype)
    r_ch = u_t - benchmark.mobility * laplacian_mu - forcing
    r_mu = mu + benchmark.epsilon**2 * laplacian_u - (u.pow(3) - u)
    magnitude = torch.sqrt(0.5 * (r_ch.square() + r_mu.square()) + 1e-20)
    return {
        "r_ch": r_ch,
        "r_mu": r_mu,
        "pde_residual": magnitude,
    }
