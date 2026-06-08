"""Global PINN and supervised losses."""

from __future__ import annotations

from typing import Any

import torch

from src.physics.kovasznay import center_pressure
from src.physics.navier_stokes import gradients, navier_stokes_residuals


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


def pseudo_huber_from_squared_residual(
    squared_residual: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Pseudo-Huber loss from r^2 without needing the residual sign."""
    delta = max(float(delta), 1e-12)
    return delta * delta * (torch.sqrt(1.0 + squared_residual / (delta * delta)) - 1.0)


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
    residual_loss_mode: str = "mse",
    pseudo_huber_delta: float = 1.0,
    regularization_config: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute pointwise losses used by global and local objectives."""
    xy_f = batch["xy_f"]
    xy_bc = batch["xy_bc"]
    xy_data = batch.get("xy_data")
    targets = batch.get("targets_data")

    residuals = navier_stokes_residuals(model, xy_f, nu=benchmark.nu, steady=steady)
    momentum_u_mse = residuals["f_u"].pow(2)
    momentum_v_mse = residuals["f_v"].pow(2)
    continuity_mse = residuals["f_c"].pow(2)
    mode = str(residual_loss_mode).lower()
    if mode == "pseudo_huber":
        momentum_u_obj = pseudo_huber_from_squared_residual(
            momentum_u_mse,
            pseudo_huber_delta,
        )
        momentum_v_obj = pseudo_huber_from_squared_residual(
            momentum_v_mse,
            pseudo_huber_delta,
        )
    elif mode in {"mse", "mean_squared"}:
        momentum_u_obj = momentum_u_mse
        momentum_v_obj = momentum_v_mse
    else:
        raise ValueError(f"Unsupported residual_loss_mode: {residual_loss_mode}")
    pointwise: dict[str, torch.Tensor] = {
        "momentum_u": momentum_u_obj,
        "momentum_v": momentum_v_obj,
        "continuity": continuity_mse,
        "pde": momentum_u_obj + momentum_v_obj + continuity_mse,
        "raw_momentum_u_mse": momentum_u_mse,
        "raw_momentum_v_mse": momentum_v_mse,
        "raw_continuity_mse": continuity_mse,
    }
    _add_reference_free_regularizers(
        pointwise,
        model,
        xy_f,
        residuals,
        regularization_config or {},
    )

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


def _add_reference_free_regularizers(
    pointwise: dict[str, torch.Tensor],
    model: torch.nn.Module,
    xy_f: torch.Tensor,
    residuals: dict[str, torch.Tensor],
    cfg: dict[str, Any],
) -> None:
    speed_cfg = dict(cfg.get("speed_cap", {}))
    if bool(speed_cfg.get("enabled", False)):
        cap = float(speed_cfg.get("cap", 2.0))
        speed = torch.sqrt(residuals["u"].pow(2) + residuals["v"].pow(2) + 1e-18)
        pointwise["speed_cap"] = torch.relu(speed - cap).pow(2)

    psi_cfg = dict(cfg.get("raw_psi_l2", {}))
    auxiliary = None
    if bool(psi_cfg.get("enabled", False)) and hasattr(model, "streamfunction_auxiliary"):
        auxiliary = model.streamfunction_auxiliary(xy_f)
        pointwise["raw_psi_l2"] = auxiliary["raw_psi"].pow(2)

    pressure_cfg = dict(cfg.get("pressure_gradient_l2", {}))
    if bool(pressure_cfg.get("enabled", False)):
        pointwise["pressure_gradient_l2"] = (
            residuals["p_x"].pow(2) + residuals["p_y"].pow(2)
        )

    vort_cfg = dict(cfg.get("vorticity_smoothness", {}))
    if bool(vort_cfg.get("enabled", False)):
        grad_omega = gradients(residuals["omega"], residuals["coords"])
        pointwise["vorticity_smoothness"] = (
            grad_omega[:, 0:1].pow(2) + grad_omega[:, 1:2].pow(2)
        )


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
