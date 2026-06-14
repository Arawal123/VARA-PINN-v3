"""Global PINN and supervised losses."""

from __future__ import annotations

import time
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
    compute_boundary_loss: bool = True,
    runtime_profile: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute pointwise losses used by global and local objectives."""
    loss_start = time.perf_counter()
    xy_f = batch["xy_f"]
    xy_bc = batch["xy_bc"]
    xy_data = batch.get("xy_data")
    targets = batch.get("targets_data")

    residuals = navier_stokes_residuals(
        model,
        xy_f,
        nu=benchmark.nu,
        steady=steady,
        runtime_profile=runtime_profile,
    )
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
    momentum_u_obj, momentum_v_obj, near_wall_logs = _apply_near_wall_curriculum(
        momentum_u_obj,
        momentum_v_obj,
        residuals["coords"],
        benchmark,
        regularization_config or {},
    )
    pointwise: dict[str, torch.Tensor] = {
        "momentum_u": momentum_u_obj,
        "momentum_v": momentum_v_obj,
        "continuity": continuity_mse,
        "pde": momentum_u_obj + momentum_v_obj + continuity_mse,
        "raw_momentum_u_mse": momentum_u_mse,
        "raw_momentum_v_mse": momentum_v_mse,
        "raw_continuity_mse": continuity_mse,
    }
    pointwise.update(near_wall_logs)
    _add_reference_free_regularizers(
        pointwise,
        model,
        xy_f,
        residuals,
        regularization_config or {},
    )

    if compute_boundary_loss:
        bc_pred = model(xy_bc)
        bc_ref = benchmark.exact_torch(xy_bc)
        u_error = (bc_pred[:, 0:1] - bc_ref["u"]).pow(2)
        v_error = (bc_pred[:, 1:2] - bc_ref["v"]).pow(2)
        pointwise["bc"] = u_error + v_error
        if getattr(model, "physics_formulation", "") in {
            "cavity_uvp_soft_bc",
            "cavity_uvp_velocity_lift",
        }:
            x0, x1, y0, y1 = benchmark.bounds
            x = xy_bc[:, 0]
            y = xy_bc[:, 1]
            tolerance = 1e-6 * max(float(x1 - x0), float(y1 - y0), 1.0)
            wall_masks = {
                "top": torch.isclose(y, y.new_tensor(y1), atol=tolerance, rtol=0.0),
                "bottom": torch.isclose(y, y.new_tensor(y0), atol=tolerance, rtol=0.0),
                "left": torch.isclose(x, x.new_tensor(x0), atol=tolerance, rtol=0.0),
                "right": torch.isclose(x, x.new_tensor(x1), atol=tolerance, rtol=0.0),
            }
            for wall, mask in wall_masks.items():
                pointwise[f"bc_{wall}_u"] = u_error[mask]
                pointwise[f"bc_{wall}_v"] = v_error[mask]
            balance_cfg = dict(
                (regularization_config or {}).get("uvp_boundary_balance", {})
            )
            if bool(balance_cfg.get("enabled", False)):
                relative_weights = dict(balance_cfg.get("relative_weights", {}))
                weighted = u_error.new_tensor(0.0)
                weight_sum = 0.0
                for wall in ("top", "bottom", "left", "right"):
                    for component in ("u", "v"):
                        name = f"bc_{wall}_{component}"
                        values = pointwise[name]
                        relative = float(relative_weights.get(name, 1.0))
                        if values.numel() > 0 and relative > 0.0:
                            weighted = weighted + relative * torch.mean(values)
                            weight_sum += relative
                pointwise["bc_uvp_balanced"] = weighted / max(weight_sum, 1e-12)
    else:
        pointwise["bc"] = xy_bc.new_zeros((xy_bc.shape[0], 1))

    if xy_data is not None and targets is not None and xy_data.shape[0] > 0 and getattr(benchmark, "has_reference", True):
        data_pred = model(xy_data)
        data_residuals = navier_stokes_residuals(
            model,
            xy_data,
            nu=benchmark.nu,
            steady=steady,
            runtime_profile=runtime_profile,
        )
        omega_pred = data_residuals["omega"]
        p_pred_c = center_pressure(data_pred[:, 2:3])
        p_true_c = center_pressure(targets["p"])
        pointwise["u"] = (data_pred[:, 0:1] - targets["u"]).pow(2)
        pointwise["v"] = (data_pred[:, 1:2] - targets["v"]).pow(2)
        pointwise["p"] = (p_pred_c - p_true_c).pow(2)
        pointwise["omega"] = (omega_pred - targets["omega"]).pow(2)
        pointwise["pressure_gradient"] = (
            data_residuals["p_x"] - targets["p_x"]
        ).pow(2) + (data_residuals["p_y"] - targets["p_y"]).pow(2)
    if runtime_profile is not None:
        runtime_profile["loss_construction_sec"] = runtime_profile.get(
            "loss_construction_sec", 0.0
        ) + (time.perf_counter() - loss_start)
    return pointwise


def _apply_near_wall_curriculum(
    momentum_u: torch.Tensor,
    momentum_v: torch.Tensor,
    coords: torch.Tensor,
    benchmark: Any,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    wall_cfg = dict(cfg.get("near_wall_momentum", {}))
    if not bool(wall_cfg.get("enabled", False)):
        return momentum_u, momentum_v, {}
    band = max(float(wall_cfg.get("band_width", 0.08)), 0.0)
    maximum_weight = max(float(wall_cfg.get("max_weight", 3.0)), 1.0)
    near_wall_weight = min(
        max(float(wall_cfg.get("weight", 1.0)), 1e-12),
        maximum_weight,
    )
    distance = _normalized_wall_distance(coords, benchmark.bounds)
    near_wall = distance < band
    weights = torch.where(
        near_wall,
        torch.full_like(distance, near_wall_weight),
        torch.ones_like(distance),
    )
    if bool(wall_cfg.get("normalize_mean", True)):
        weights = weights / weights.mean().clamp_min(1e-12)
    return (
        momentum_u * weights,
        momentum_v * weights,
        {
            "near_wall_momentum_weight_mean": weights,
        },
    )


def _add_reference_free_regularizers(
    pointwise: dict[str, torch.Tensor],
    model: torch.nn.Module,
    xy_f: torch.Tensor,
    residuals: dict[str, torch.Tensor],
    cfg: dict[str, Any],
) -> None:
    regional_cfg = dict(cfg.get("uvp_regional_residuals", {}))
    if (
        getattr(model, "physics_formulation", "")
        == "cavity_uvp_velocity_lift"
        and bool(regional_cfg.get("enabled", False))
    ):
        x0, x1, y0, y1 = tuple(
            cfg.get("domain_bounds", (0.0, 1.0, 0.0, 1.0))
        )
        xi = (residuals["coords"][:, 0:1] - x0) / max(float(x1 - x0), 1e-12)
        eta = (residuals["coords"][:, 1:2] - y0) / max(float(y1 - y0), 1e-12)
        raw_u = residuals["f_u"].pow(2)
        raw_v = residuals["f_v"].pow(2)
        raw_c = residuals["f_c"].pow(2)
        raw_pde = raw_u + raw_v + raw_c
        masks = {
            "top_band": (
                (eta >= 0.70)
                & (eta <= 0.98)
                & (xi >= 0.05)
                & (xi <= 0.95)
            ),
            "upper_core": (
                (eta >= 0.55)
                & (eta <= 0.90)
                & (xi >= 0.15)
                & (xi <= 0.85)
            ),
            "right_wall_interior": (
                (xi >= 0.78)
                & (xi <= 0.98)
                & (eta >= 0.15)
                & (eta <= 0.85)
            ),
            "lower_core": (
                (eta >= 0.05)
                & (eta <= 0.35)
                & (xi >= 0.15)
                & (xi <= 0.85)
            ),
        }
        pointwise["top_band_pde"] = raw_pde[masks["top_band"]]
        pointwise["top_band_momentum_u"] = raw_u[masks["top_band"]]
        pointwise["top_band_continuity"] = raw_c[masks["top_band"]]
        pointwise["upper_core_pde"] = raw_pde[masks["upper_core"]]
        pointwise["right_wall_interior_pde"] = raw_pde[
            masks["right_wall_interior"]
        ]
        pointwise["lower_core_pde"] = raw_pde[masks["lower_core"]]

    wall_tail_cfg = dict(cfg.get("uvp_wall_residual_tail", {}))
    if (
        getattr(model, "physics_formulation", "")
        == "cavity_uvp_velocity_lift"
        and bool(wall_tail_cfg.get("enabled", False))
    ):
        bounds = cfg.get("domain_bounds", (0.0, 1.0, 0.0, 1.0))
        distance = _normalized_wall_distance(residuals["coords"], bounds)
        x0, x1, y0, y1 = tuple(bounds)
        xi = (residuals["coords"][:, 0:1] - x0) / max(float(x1 - x0), 1e-12)
        eta = (residuals["coords"][:, 1:2] - y0) / max(float(y1 - y0), 1e-12)
        band = max(float(wall_tail_cfg.get("band_width", 0.10)), 0.0)
        corner_width = max(
            float(wall_tail_cfg.get("corner_width", 0.12)),
            0.0,
        )
        near_wall = distance < band
        corner = (
            ((xi < corner_width) | (xi > 1.0 - corner_width))
            & ((eta < corner_width) | (eta > 1.0 - corner_width))
        )
        selected = near_wall | corner
        residual_magnitude = torch.sqrt(
            residuals["f_u"].pow(2)
            + residuals["f_v"].pow(2)
            + residuals["f_c"].pow(2)
            + 1e-18
        )
        threshold = max(float(wall_tail_cfg.get("threshold", 0.60)), 0.0)
        delta = max(float(wall_tail_cfg.get("pseudo_huber_delta", 0.50)), 1e-12)
        excess_squared = torch.relu(residual_magnitude - threshold).pow(2)
        pointwise["uvp_wall_residual_tail"] = pseudo_huber_from_squared_residual(
            excess_squared[selected],
            delta,
        )

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

    bubble_cfg = dict(cfg.get("correction_bubble", {}))
    if bool(bubble_cfg.get("enabled", False)) and hasattr(
        model, "streamfunction_auxiliary"
    ):
        if auxiliary is None:
            auxiliary = model.streamfunction_auxiliary(xy_f)
        raw_mean_l2 = auxiliary["raw_psi"].mean().pow(2)
        correction = auxiliary["scaled_correction"]
        correction_mean_l2 = correction.mean().pow(2)
        correction_cap = max(float(bubble_cfg.get("abs_max_cap", 0.30)), 0.0)
        correction_excess = torch.relu(
            correction.abs().max() - correction_cap
        ).pow(2)
        pointwise["raw_psi_mean_l2"] = torch.ones_like(
            auxiliary["raw_psi"]
        ) * raw_mean_l2
        pointwise["scaled_correction_mean_l2"] = (
            torch.ones_like(correction) * correction_mean_l2
        )
        pointwise["scaled_correction_abs_max_hinge"] = (
            torch.ones_like(correction) * correction_excess
        )

    shear_cfg = dict(cfg.get("lid_shear_direction", {}))
    if bool(shear_cfg.get("enabled", False)):
        x0, x1, y0, y1 = tuple(
            cfg.get("domain_bounds", (0.0, 1.0, 0.0, 1.0))
        )
        width = max(float(x1 - x0), 1e-12)
        height = max(float(y1 - y0), 1e-12)
        xi = (residuals["coords"][:, 0:1] - x0) / width
        eta = (residuals["coords"][:, 1:2] - y0) / height
        band = min(max(float(shear_cfg.get("band_width", 0.08)), 0.0), 0.49)
        corner_width = min(
            max(float(shear_cfg.get("corner_width", 0.08)), 0.0),
            0.49,
        )
        away_from_corners = (xi >= corner_width) & (xi <= 1.0 - corner_width)
        top = away_from_corners & (eta >= 1.0 - band)
        bottom = away_from_corners & (eta <= band)
        u = residuals["u"]
        bottom_tolerance = float(shear_cfg.get("bottom_u_tolerance", 0.075))
        pointwise["top_reverse_u"] = top.to(u.dtype) * torch.relu(-u).pow(2)
        pointwise["bottom_positive_u"] = bottom.to(u.dtype) * torch.relu(
            u - bottom_tolerance
        ).pow(2)

    tail_cfg = dict(cfg.get("raw_residual_tail", {}))
    if bool(tail_cfg.get("enabled", False)):
        threshold = max(float(tail_cfg.get("threshold", 0.5)), 0.0)
        core_margin = min(
            max(float(tail_cfg.get("core_margin", 0.08)), 0.0),
            0.49,
        )
        core_emphasis = max(float(tail_cfg.get("core_emphasis", 2.0)), 1.0)
        x0, x1, y0, y1 = tuple(
            cfg.get("domain_bounds", (0.0, 1.0, 0.0, 1.0))
        )
        width = max(float(x1 - x0), 1e-12)
        height = max(float(y1 - y0), 1e-12)
        xi = (residuals["coords"][:, 0:1] - x0) / width
        eta = (residuals["coords"][:, 1:2] - y0) / height
        core = (
            (xi >= core_margin)
            & (xi <= 1.0 - core_margin)
            & (eta >= core_margin)
            & (eta <= 1.0 - core_margin)
        )
        emphasis = torch.where(
            core,
            torch.full_like(xi, core_emphasis),
            torch.ones_like(xi),
        )
        u_tail = torch.relu(residuals["f_u"].abs() - threshold).pow(2)
        v_tail = torch.relu(residuals["f_v"].abs() - threshold).pow(2)
        pointwise["raw_pde_tail"] = emphasis * (u_tail + v_tail)

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

    near_wall_vort_cfg = dict(cfg.get("near_wall_vorticity_l2", {}))
    if bool(near_wall_vort_cfg.get("enabled", False)):
        band = max(float(near_wall_vort_cfg.get("band_width", 0.10)), 0.0)
        quantile = min(
            max(float(near_wall_vort_cfg.get("quantile", 0.95)), 0.0),
            1.0,
        )
        distance = _normalized_wall_distance(
            residuals["coords"],
            cfg.get("domain_bounds", (0.0, 1.0, 0.0, 1.0)),
        )
        near_wall = distance < band
        omega_abs = residuals["omega"].abs()
        threshold_source = omega_abs[near_wall]
        if threshold_source.numel() == 0:
            threshold_source = omega_abs.reshape(-1)
        threshold = torch.quantile(threshold_source.detach(), quantile)
        mask = near_wall.to(dtype=omega_abs.dtype)
        pointwise["near_wall_vorticity_l2"] = (
            mask * torch.relu(omega_abs - threshold).pow(2)
        )


def _normalized_wall_distance(
    coords: torch.Tensor,
    bounds: tuple[float, float, float, float] | list[float] | None,
) -> torch.Tensor:
    x0, x1, y0, y1 = tuple(bounds or (0.0, 1.0, 0.0, 1.0))
    width = max(float(x1 - x0), 1e-12)
    height = max(float(y1 - y0), 1e-12)
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    return torch.minimum(
        torch.minimum((x - x0) / width, (x1 - x) / width),
        torch.minimum((y - y0) / height, (y1 - y) / height),
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
