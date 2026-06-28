"""Cahn--Hilliard losses with optional VARA-local multipliers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .benchmark import CahnHilliardBenchmark
from .residuals import compute_cahn_hilliard_residuals


@dataclass
class LossResult:
    """Differentiable loss result and pointwise controller channels."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]
    channels: dict[str, tuple[torch.Tensor, torch.Tensor]]


def compute_training_loss(
    model: nn.Module,
    benchmark: CahnHilliardBenchmark,
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    patch_grid: Any,
    allocation_state: Any | None = None,
) -> LossResult:
    """Assemble split residual, condition, sparse, and optional proxy losses."""
    interior = batch["interior"].detach().clone().requires_grad_(True)
    residuals = compute_cahn_hilliard_residuals(
        model,
        interior,
        benchmark,
        batch.get("forcing"),
    )
    ch_values = residuals["r_ch"].square().squeeze(1)
    mu_values = residuals["r_mu"].square().squeeze(1)
    ch_loss = _localized_mean(
        ch_values, interior, "ch_residual", patch_grid, allocation_state
    )
    mu_loss = _localized_mean(
        mu_values,
        interior,
        "chemical_potential_residual",
        patch_grid,
        allocation_state,
    )

    boundary = batch["boundary"]
    boundary_error = (model(boundary) - batch["boundary_target"]).square()
    bc_u = _localized_mean(
        boundary_error[:, 0], boundary, "bc_u", patch_grid, allocation_state
    )
    bc_mu = _localized_mean(
        boundary_error[:, 1], boundary, "bc_mu", patch_grid, allocation_state
    )

    initial = batch["initial"]
    initial_error = (model(initial) - batch["initial_target"]).square()
    ic_u = _localized_mean(
        initial_error[:, 0], initial, "ic_u", patch_grid, allocation_state
    )
    ic_mu = _localized_mean(
        initial_error[:, 1], initial, "ic_mu", patch_grid, allocation_state
    )

    sparse = batch.get("sparse")
    sparse_target = batch.get("sparse_target")
    if sparse is not None and sparse_target is not None and sparse.numel():
        sparse_error = (model(sparse) - sparse_target).square()
        sparse_u = _localized_mean(
            sparse_error[:, 0],
            sparse,
            "sparse_u_mse",
            patch_grid,
            allocation_state,
        )
        sparse_mu = _localized_mean(
            sparse_error[:, 1],
            sparse,
            "sparse_mu_mse",
            patch_grid,
            allocation_state,
        )
    else:
        sparse_error = interior.new_empty((0, 2))
        sparse_u = ch_loss.new_zeros(())
        sparse_mu = ch_loss.new_zeros(())

    phase_range_weight = float(weights.get("phase_range_penalty", 0.0))
    proxy_weight = float(weights.get("interface_proxy_regularization", 0.0))
    if phase_range_weight or proxy_weight:
        u_pred = model(interior)[:, 0]
        phase_range_values = (
            torch.relu(u_pred - 1.0).square()
            + torch.relu(-1.0 - u_pred).square()
        )
        phase_range_penalty = _localized_mean(
            phase_range_values,
            interior,
            "phase_range_penalty",
            patch_grid,
            allocation_state,
        )
        proxy_regularization = phase_range_values.mean()
    else:
        phase_range_values = interior.new_zeros(interior.shape[0])
        phase_range_penalty = ch_loss.new_zeros(())
        proxy_regularization = ch_loss.new_zeros(())

    components = {
        "ch_residual": ch_loss,
        "chemical_potential_residual": mu_loss,
        "bc_u": bc_u,
        "bc_mu": bc_mu,
        "ic_u": ic_u,
        "ic_mu": ic_mu,
        "sparse_u_mse": sparse_u,
        "sparse_mu_mse": sparse_mu,
        "phase_range_penalty": phase_range_penalty,
        "interface_proxy_regularization": proxy_regularization,
    }
    total = sum(
        float(weights.get(name, 0.0)) * value
        for name, value in components.items()
    )
    channels = {
        "ch_residual": (ch_values, interior),
        "chemical_potential_residual": (mu_values, interior),
        "pde_residual": (0.5 * (ch_values + mu_values), interior),
        "boundary_violation": (boundary_error.mean(dim=1), boundary),
        "bc_u": (boundary_error[:, 0], boundary),
        "bc_mu": (boundary_error[:, 1], boundary),
        "initial_condition_violation": (initial_error.mean(dim=1), initial),
        "ic_u": (initial_error[:, 0], initial),
        "ic_mu": (initial_error[:, 1], initial),
        "phase_range_violation": (phase_range_values, interior),
    }
    if sparse_error.numel():
        channels.update(
            {
                "sparse_u_mismatch": (sparse_error[:, 0], sparse),
                "sparse_mu_mismatch": (sparse_error[:, 1], sparse),
            }
        )
    return LossResult(total=total, components=components, channels=channels)


def _localized_mean(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    loss_name: str,
    patch_grid: Any,
    allocation_state: Any | None,
) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    if allocation_state is None:
        return values.mean()
    multipliers = getattr(allocation_state, "loss_multipliers", {}).get(loss_name)
    if multipliers is None:
        return values.mean()
    patch_ids = patch_grid.assign_torch(coordinates.detach())
    local = torch.as_tensor(multipliers, device=values.device, dtype=values.dtype)
    return (values * local[patch_ids]).mean()
