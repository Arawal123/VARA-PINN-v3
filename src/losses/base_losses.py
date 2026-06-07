"""Global PINN and supervised losses."""

from __future__ import annotations

from typing import Any

import torch

from src.physics.kovasznay import center_pressure
from src.physics.navier_stokes import navier_stokes_residuals


def mse(x: torch.Tensor) -> torch.Tensor:
    """Mean squared value with empty-tensor safety."""
    if x.numel() == 0:
        return x.new_tensor(0.0)
    return torch.mean(x * x)


def reduce_pointwise_loss(values: torch.Tensor, reduction: str = "legacy_mse") -> torch.Tensor:
    """Reduce pointwise loss values.

    Pointwise entries produced by ``compute_pointwise_losses`` are already
    squared errors. ``mean`` therefore gives the conventional mean-squared
    objective. ``legacy_mse`` preserves the historical fourth-power behavior
    for exact reproduction of earlier experiments.
    """
    if values.numel() == 0:
        return values.new_tensor(0.0)
    mode = str(reduction).lower()
    if mode in {"mean", "mean_squared", "mse"}:
        return torch.mean(values)
    if mode in {"legacy_mse", "legacy_fourth_power"}:
        return mse(values)
    raise ValueError(f"Unsupported pointwise loss reduction: {reduction}")


def reduce_weighted_pointwise_loss(
    values: torch.Tensor,
    weights: torch.Tensor,
    reduction: str = "legacy_mse",
) -> torch.Tensor:
    """Apply point weights without changing the configured loss definition."""
    if values.numel() == 0:
        return values.new_tensor(0.0)
    mode = str(reduction).lower()
    if mode in {"mean", "mean_squared", "mse"}:
        return torch.mean(weights * values)
    if mode in {"legacy_mse", "legacy_fourth_power"}:
        return torch.mean(weights * values * values)
    raise ValueError(f"Unsupported pointwise loss reduction: {reduction}")


def compute_pointwise_losses(
    model: torch.nn.Module,
    batch: dict[str, Any],
    benchmark: Any,
    steady: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute pointwise losses used by global and local objectives."""
    xy_f = batch["xy_f"]
    xy_bc = batch["xy_bc"]
    xy_data = batch.get("xy_data")
    targets = batch.get("targets_data")

    residuals = navier_stokes_residuals(model, xy_f, nu=benchmark.nu, steady=steady)
    pointwise: dict[str, torch.Tensor] = {
        "momentum_u": residuals["f_u"].pow(2),
        "momentum_v": residuals["f_v"].pow(2),
        "continuity": residuals["f_c"].pow(2),
        "pde": residuals["f_u"].pow(2) + residuals["f_v"].pow(2) + residuals["f_c"].pow(2),
    }

    bc_pred = model(xy_bc)
    bc_ref = benchmark.exact_torch(xy_bc)
    pointwise["bc"] = (bc_pred[:, 0:1] - bc_ref["u"]).pow(2) + (bc_pred[:, 1:2] - bc_ref["v"]).pow(2)

    if xy_data is not None and targets is not None and xy_data.shape[0] > 0 and getattr(benchmark, "has_reference", True):
        data_pred = model(xy_data)
        omega_pred = navier_stokes_residuals(model, xy_data, nu=benchmark.nu, steady=steady)["omega"]
        p_pred_c = center_pressure(data_pred[:, 2:3])
        p_true_c = center_pressure(targets["p"])
        pointwise["u"] = (data_pred[:, 0:1] - targets["u"]).pow(2)
        pointwise["v"] = (data_pred[:, 1:2] - targets["v"]).pow(2)
        pointwise["p"] = (p_pred_c - p_true_c).pow(2)
        pointwise["omega"] = (omega_pred - targets["omega"]).pow(2)
        p_grad = navier_stokes_residuals(model, xy_data, nu=benchmark.nu, steady=steady)
        pointwise["pressure_gradient"] = (p_grad["p_x"] - targets["p_x"]).pow(2) + (p_grad["p_y"] - targets["p_y"]).pow(2)
    return pointwise


def compute_global_losses(
    pointwise: dict[str, torch.Tensor],
    reduction: str = "legacy_mse",
) -> dict[str, torch.Tensor]:
    """Reduce pointwise losses."""
    return {
        name: reduce_pointwise_loss(values, reduction=reduction)
        for name, values in pointwise.items()
    }


def weighted_sum(losses: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    """Weighted sum over known losses."""
    if not losses:
        raise ValueError("No losses were provided.")
    total = next(iter(losses.values())).new_tensor(0.0)
    for name, loss in losses.items():
        total = total + float(weights.get(name, 0.0)) * loss
    return total
