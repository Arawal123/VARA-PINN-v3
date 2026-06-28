"""Loss assembly independent of the legacy Navier--Stokes loss path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .benchmarks import Burgers2DBenchmark, ManufacturedBenchmark
from .residuals import compute_residuals


@dataclass
class LossResult:
    """Differentiable total loss plus scalar and pointwise diagnostics."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]
    channels: dict[str, tuple[torch.Tensor, torch.Tensor]]


def compute_training_loss(
    model: nn.Module,
    benchmark: ManufacturedBenchmark,
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    patch_grid: Any,
    allocation_state: Any | None = None,
) -> LossResult:
    """Compute PDE, boundary, initial, and sparse-supervision objectives."""
    interior = batch["interior"].detach().clone().requires_grad_(True)
    residuals = compute_residuals(model, interior, benchmark)
    channels: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    if isinstance(benchmark, Burgers2DBenchmark):
        fu2 = residuals["f_u"].square().squeeze(1)
        fv2 = residuals["f_v"].square().squeeze(1)
        pde_loss = 0.5 * (
            _localized_mean(fu2, interior, "momentum_u", patch_grid, allocation_state)
            + _localized_mean(fv2, interior, "momentum_v", patch_grid, allocation_state)
        )
        channels["momentum_u_residual"] = (fu2, interior)
        channels["momentum_v_residual"] = (fv2, interior)
        channels["pde_residual"] = (0.5 * (fu2 + fv2), interior)
    else:
        signed_name = "f_ac" if "f_ac" in residuals else "f_advdiff"
        pde_values = residuals[signed_name].square().squeeze(1)
        pde_loss = _localized_mean(
            pde_values,
            interior,
            "pde",
            patch_grid,
            allocation_state,
        )
        channels["pde_residual"] = (pde_values, interior)

    boundary = batch["boundary"]
    boundary_target = batch["boundary_target"]
    boundary_pointwise = (model(boundary) - boundary_target).square().mean(dim=1)
    boundary_loss = _localized_mean(
        boundary_pointwise,
        boundary,
        "bc",
        patch_grid,
        allocation_state,
    )
    channels["boundary_mismatch"] = (boundary_pointwise, boundary)

    initial = batch["initial"]
    initial_target = batch["initial_target"]
    initial_pointwise = (model(initial) - initial_target).square().mean(dim=1)
    initial_loss = _localized_mean(
        initial_pointwise,
        initial,
        "ic",
        patch_grid,
        allocation_state,
    )
    channels["initial_condition_mismatch"] = (initial_pointwise, initial)

    sparse = batch.get("sparse")
    sparse_target = batch.get("sparse_target")
    if sparse is not None and sparse_target is not None and sparse.numel() > 0:
        sparse_component = (model(sparse) - sparse_target).square()
        sparse_pointwise = sparse_component.mean(dim=1)
        sparse_loss = _localized_mean(
            sparse_pointwise,
            sparse,
            "sparse_data",
            patch_grid,
            allocation_state,
        )
        for index, name in enumerate(benchmark.output_names):
            channels[f"sparse_{name}_mismatch"] = (sparse_component[:, index], sparse)
        channels["sparse_data_mismatch"] = (sparse_pointwise, sparse)
    else:
        sparse_loss = pde_loss.new_zeros(())

    components = {
        "pde": pde_loss,
        "bc": boundary_loss,
        "ic": initial_loss,
        "sparse_data": sparse_loss,
    }
    total = sum(
        float(weights.get(name, 0.0)) * value
        for name, value in components.items()
    )
    if allocation_state is not None:
        for name, multiplier in dict(
            getattr(allocation_state, "global_multipliers", {})
        ).items():
            if name in components:
                total = total + (
                    float(multiplier) - 1.0
                ) * float(weights.get(name, 0.0)) * components[name]
    return LossResult(total=total, components=components, channels=channels)


def unweighted_objective(result: LossResult) -> torch.Tensor:
    """Reference-free guard objective with no adaptive weights."""
    return sum(result.components.values())


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
    tensor = torch.as_tensor(multipliers, device=values.device, dtype=values.dtype)
    return (values * tensor[patch_ids]).mean()
