"""Global model metrics."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from src.physics.kovasznay import center_pressure
from src.physics.navier_stokes import navier_stokes_residuals
from src.visualization.streamlines import (
    detect_vortices,
    lid_cavity_topology_metrics,
    reconstruct_streamfunction,
)


def relative_l2(pred: np.ndarray, true: np.ndarray, min_reference_norm: float = 1e-8) -> float:
    """Relative L2, undefined when the reference field is effectively zero."""
    ref_norm = float(np.linalg.norm(true))
    if ref_norm < min_reference_norm:
        return float("nan")
    return float(np.linalg.norm(pred - true) / ref_norm)


def vector_relative_l2(
    pred_components: tuple[np.ndarray, ...],
    true_components: tuple[np.ndarray, ...],
    min_reference_norm: float = 1e-8,
) -> float:
    """Relative L2 of a vector field, preserving directional error."""
    pred = np.concatenate(
        [np.asarray(component).reshape(-1, 1) for component in pred_components],
        axis=1,
    )
    true = np.concatenate(
        [np.asarray(component).reshape(-1, 1) for component in true_components],
        axis=1,
    )
    return relative_l2(pred, true, min_reference_norm=min_reference_norm)


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def _finite_sum(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(sum(finite)) if finite else float("nan")


def evaluate_on_grid(
    model: torch.nn.Module,
    benchmark: Any,
    coords_np: np.ndarray,
    device: torch.device,
    steady: bool = True,
    residual_interior_only: bool = False,
    include_reference_metrics: bool = True,
    include_streamfunction_metrics: bool = True,
    runtime_profile: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute global evaluation metrics without using them for adaptation."""
    start = time.time()
    coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
    model.eval()
    residuals = navier_stokes_residuals(
        model,
        coords,
        nu=benchmark.nu,
        steady=steady,
        runtime_profile=runtime_profile,
    )
    pred = torch.cat(
        [residuals["u"], residuals["v"], residuals["p"]],
        dim=1,
    )
    has_reference = bool(include_reference_metrics) and bool(
        getattr(benchmark, "has_reference", True)
    )
    ref = benchmark.exact_np(coords_np) if has_reference else None

    u = pred[:, 0:1].detach().cpu().numpy()
    v = pred[:, 1:2].detach().cpu().numpy()
    p = pred[:, 2:3].detach().cpu().numpy()
    omega = residuals["omega"].detach().cpu().numpy()
    p_c = center_pressure(p)
    speed = np.sqrt(u * u + v * v)
    pde = residuals["pde_residual"].detach().cpu().numpy()
    pde_loss_all = (
        residuals["f_u"].detach().cpu().numpy() ** 2
        + residuals["f_v"].detach().cpu().numpy() ** 2
        + residuals["f_c"].detach().cpu().numpy() ** 2
    )
    momentum_u_abs = np.abs(residuals["f_u"].detach().cpu().numpy())
    momentum_v_abs = np.abs(residuals["f_v"].detach().cpu().numpy())
    p_grad_abs = np.sqrt(
        residuals["p_x"].detach().cpu().numpy() ** 2
        + residuals["p_y"].detach().cpu().numpy() ** 2
    )
    continuity_all = np.abs(residuals["f_c"].detach().cpu().numpy())
    momentum_all = np.sqrt(
        residuals["f_u"].detach().cpu().numpy() ** 2
        + residuals["f_v"].detach().cpu().numpy() ** 2
    )
    residual_mask = _residual_mask(benchmark, coords_np, residual_interior_only)
    pde_eval = _masked_values(pde, residual_mask)
    pde_loss = _masked_values(pde_loss_all, residual_mask)
    div = _masked_values(continuity_all, residual_mask)
    momentum = _masked_values(momentum_all, residual_mask)
    boundary_metrics = _boundary_metrics(model, benchmark, coords_np, device)
    unweighted_bc_loss = boundary_metrics["unweighted_bc_loss"]
    metrics = {
        "u_rel_l2": float("nan"),
        "v_rel_l2": float("nan"),
        "p_rel_l2_centered": float("nan"),
        "speed_rel_l2": float("nan"),
        "omega_rel_l2": float("nan"),
        "u_full_rel_l2": float("nan"),
        "v_full_rel_l2": float("nan"),
        "velocity_full_rel_l2": float("nan"),
        "p_full_rel_l2_centered": float("nan"),
        "omega_full_rel_l2": float("nan"),
        "u_rmse": float("nan"),
        "v_rmse": float("nan"),
        "p_rmse_centered": float("nan"),
        "omega_rmse": float("nan"),
        "velocity_mag_rmse": float("nan"),
        "velocity_mag_mae": float("nan"),
        "u_mae": float("nan"),
        "v_mae": float("nan"),
        "p_mae_centered": float("nan"),
        "omega_mae": float("nan"),
        "has_p_full_field_reference": False,
        "has_omega_full_field_reference": False,
        "omega_full_field_reference_source": "",
        "full_field_reference_path": "",
        "u_reference_norm": float("nan"),
        "v_reference_norm": float("nan"),
        "p_reference_norm": float("nan"),
        "omega_reference_norm": float("nan"),
        "u_pred_mean": float(np.mean(u)),
        "v_pred_mean": float(np.mean(v)),
        "p_pred_std_centered": float(np.std(p_c)),
        "speed_pred_mean": float(np.mean(speed)),
        "speed_pred_max": float(np.max(speed)),
        "omega_pred_abs_mean": float(np.mean(np.abs(omega))),
        "omega_pred_abs_95p": _finite_percentile(np.abs(omega), 95.0),
        "omega_pred_abs_max": float(np.max(np.abs(omega))),
        "omega_abs_mean": float(np.mean(np.abs(omega))),
        "omega_abs_95p": _finite_percentile(np.abs(omega), 95.0),
        "omega_abs_max": float(np.max(np.abs(omega))),
        "momentum_u_mean": _finite_mean(_masked_values(momentum_u_abs, residual_mask)),
        "momentum_v_mean": _finite_mean(_masked_values(momentum_v_abs, residual_mask)),
        "momentum_u_max": _finite_max(_masked_values(momentum_u_abs, residual_mask)),
        "momentum_v_max": _finite_max(_masked_values(momentum_v_abs, residual_mask)),
        "p_grad_mean": _finite_mean(_masked_values(p_grad_abs, residual_mask)),
        "p_grad_max": _finite_max(_masked_values(p_grad_abs, residual_mask)),
        "psi_min": float("nan"),
        "psi_max": float("nan"),
        "psi_abs_max": float("nan"),
        "pressure_gradient_error": float("nan"),
        "divergence_norm": _finite_mean(div),
        "continuity_residual_mean": _finite_mean(div),
        "momentum_residual_mean": _finite_mean(momentum),
        "pde_residual_mean": _finite_mean(pde_eval),
        "pde_residual_max": _finite_max(pde_eval),
        "residual_interior_only": bool(residual_interior_only),
        "num_residual_eval_points": int(np.count_nonzero(residual_mask)),
        "boundary_condition_error": _boundary_error(model, benchmark, coords_np, device),
        "u_boundary_rmse": boundary_metrics["u_boundary_rmse"],
        "v_boundary_rmse": boundary_metrics["v_boundary_rmse"],
        "boundary_speed_rmse": boundary_metrics["boundary_speed_rmse"],
        "centerline_pde_residual_mean": float("nan"),
        "centerline_continuity_residual_mean": float("nan"),
        "corner_pde_residual_mean": float("nan"),
        "corner_boundary_error": float("nan"),
        "u_centerline_rmse": float("nan"),
        "v_centerline_rmse": float("nan"),
        "u_centerline_rel_l2": float("nan"),
        "v_centerline_rel_l2": float("nan"),
        "centerline_profile_score": float("nan"),
        "cavity_benchmark_score": float("nan"),
        "cavity_profile_reference_source": "",
        "lid_cavity_expected_primary_x": float("nan"),
        "lid_cavity_expected_primary_y": float("nan"),
        "lid_cavity_primary_center_error": float("nan"),
        "lid_cavity_topology_score": float("nan"),
        "lid_cavity_topology_aligned": float("nan"),
        "unweighted_data_loss": float("nan"),
        "unweighted_pde_loss": _finite_mean(pde_loss),
        "unweighted_bc_loss": unweighted_bc_loss,
        "unweighted_physics_validation_loss": float("nan"),
        "unweighted_reference_evaluation_loss": float("nan"),
        "unweighted_validation_loss": float("nan"),
        "wall_clock_eval_sec": time.time() - start,
        "num_eval_points": int(coords_np.shape[0]),
    }
    if include_streamfunction_metrics:
        streamfunction_start = time.perf_counter()
        metrics.update(_streamfunction_metrics(benchmark, coords_np, u, v))
        if runtime_profile is not None:
            runtime_profile["streamfunction_diagnostics_sec"] = (
                runtime_profile.get("streamfunction_diagnostics_sec", 0.0)
                + time.perf_counter()
                - streamfunction_start
            )
    metrics.update(_direct_streamfunction_diagnostics(model, coords_np, device))
    if (
        include_reference_metrics
        and include_streamfunction_metrics
        and benchmark.__class__.__name__.lower().startswith("liddrivencavity")
    ):
        topology_start = time.perf_counter()
        metrics.update(_lid_cavity_topology_metrics(benchmark, coords_np, u, v))
        if runtime_profile is not None:
            runtime_profile["topology_diagnostics_sec"] = (
                runtime_profile.get("topology_diagnostics_sec", 0.0)
                + time.perf_counter()
                - topology_start
            )
    if has_reference and ref is not None:
        has_p_ref = bool(ref.get("has_p_reference", np.isfinite(ref.get("p", np.array([]))).any()))
        has_omega_ref = bool(ref.get("has_omega_reference", np.isfinite(ref.get("omega", np.array([]))).any()))
        p_ref_c = center_pressure(ref["p"])
        p_grad_err = (
            np.sqrt(
                (residuals["p_x"].detach().cpu().numpy() - ref.get("p_x", 0.0)) ** 2
                + (residuals["p_y"].detach().cpu().numpy() - ref.get("p_y", 0.0)) ** 2
            )
            if has_p_ref
            else np.full_like(u, np.nan)
        )
        speed_ref = ref.get("speed", np.sqrt(ref["u"] ** 2 + ref["v"] ** 2))
        u_mse = float(np.mean((u - ref["u"]) ** 2))
        v_mse = float(np.mean((v - ref["v"]) ** 2))
        p_mse = float(np.mean((p_c - p_ref_c) ** 2)) if has_p_ref else float("nan")
        omega_mse = float(np.mean((omega - ref["omega"]) ** 2)) if has_omega_ref else float("nan")
        data_loss = _finite_sum([u_mse, v_mse, p_mse, omega_mse])
        metrics.update(
            {
                "u_rel_l2": relative_l2(u, ref["u"]),
                "v_rel_l2": relative_l2(v, ref["v"]),
                "p_rel_l2_centered": relative_l2(p_c, p_ref_c) if has_p_ref else float("nan"),
                "speed_rel_l2": relative_l2(speed, speed_ref),
                "omega_rel_l2": relative_l2(omega, ref["omega"]) if has_omega_ref else float("nan"),
                "u_full_rel_l2": relative_l2(u, ref["u"]),
                "v_full_rel_l2": relative_l2(v, ref["v"]),
                "velocity_full_rel_l2": vector_relative_l2(
                    (u, v),
                    (ref["u"], ref["v"]),
                ),
                "p_full_rel_l2_centered": relative_l2(p_c, p_ref_c) if has_p_ref else float("nan"),
                "omega_full_rel_l2": relative_l2(omega, ref["omega"]) if has_omega_ref else float("nan"),
                "u_rmse": rmse(u, ref["u"]),
                "v_rmse": rmse(v, ref["v"]),
                "p_rmse_centered": rmse(p_c, p_ref_c) if has_p_ref else float("nan"),
                "omega_rmse": rmse(omega, ref["omega"]) if has_omega_ref else float("nan"),
                "velocity_mag_rmse": rmse(speed, speed_ref),
                "velocity_mag_mae": mae(speed, speed_ref),
                "u_mae": mae(u, ref["u"]),
                "v_mae": mae(v, ref["v"]),
                "p_mae_centered": mae(p_c, p_ref_c) if has_p_ref else float("nan"),
                "omega_mae": mae(omega, ref["omega"]) if has_omega_ref else float("nan"),
                "has_p_full_field_reference": has_p_ref,
                "has_omega_full_field_reference": has_omega_ref,
                "omega_full_field_reference_source": str(ref.get("omega_reference_source", "")),
                "full_field_reference_path": str(ref.get("source_path", "")),
                "u_reference_norm": float(np.linalg.norm(ref["u"])),
                "v_reference_norm": float(np.linalg.norm(ref["v"])),
                "p_reference_norm": float(np.linalg.norm(p_ref_c)) if has_p_ref else float("nan"),
                "omega_reference_norm": float(np.linalg.norm(ref["omega"])) if has_omega_ref else float("nan"),
                "pressure_gradient_error": float(np.mean(p_grad_err)) if has_p_ref else float("nan"),
                "unweighted_data_loss": data_loss,
            }
        )
    if include_reference_metrics:
        metrics.update(_cavity_profile_metrics(model, benchmark, device))
    if benchmark.__class__.__name__.lower().startswith("liddrivencavity"):
        metrics.update(
            _cavity_residual_geometry_metrics(
                model=model,
                benchmark=benchmark,
                coords_np=coords_np,
                device=device,
                pde=_masked_values(pde, residual_mask),
                continuity=_masked_values(continuity_all, residual_mask),
                momentum_u=_masked_values(momentum_u_abs, residual_mask),
                momentum_v=_masked_values(momentum_v_abs, residual_mask),
                p_grad=_masked_values(p_grad_abs, residual_mask),
                speed=speed,
            )
        )
    metrics["unweighted_physics_validation_loss"] = _finite_sum(
        [metrics["unweighted_pde_loss"], metrics["unweighted_bc_loss"]]
    )
    metrics["unweighted_reference_evaluation_loss"] = (
        metrics["unweighted_data_loss"] if has_reference else float("nan")
    )
    metrics["unweighted_validation_loss"] = metrics["unweighted_physics_validation_loss"]
    if benchmark.__class__.__name__.lower().startswith("liddrivencavity"):
        metrics["cavity_benchmark_score"] = _finite_sum(
            [
                metrics["centerline_profile_score"],
                metrics["pde_residual_mean"],
                metrics["continuity_residual_mean"],
                metrics["momentum_residual_mean"],
                metrics["boundary_condition_error"],
            ]
        )
    return metrics


def _residual_mask(
    benchmark: Any,
    coords_np: np.ndarray,
    interior_only: bool,
) -> np.ndarray:
    mask = np.ones(coords_np.shape[0], dtype=bool)
    if interior_only and hasattr(benchmark, "boundary_mask_np"):
        mask &= ~np.asarray(benchmark.boundary_mask_np(coords_np), dtype=bool)
    if not np.any(mask):
        raise ValueError("Residual evaluation mask contains no points.")
    return mask


def _masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).reshape(-1, 1).copy()
    out[~np.asarray(mask, dtype=bool).reshape(-1), :] = np.nan
    return out


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else float("nan")


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _direct_streamfunction_diagnostics(
    model: torch.nn.Module,
    coords_np: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    if not hasattr(model, "streamfunction_auxiliary"):
        return {}
    coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        auxiliary = model.streamfunction_auxiliary(coords)
    out: dict[str, float] = {}
    mapping = {
        "raw_psi": "raw_psi",
        "scaled_correction": "scaled_correction",
        "psi_total": "psi",
    }
    for source, prefix in mapping.items():
        if source not in auxiliary:
            continue
        values = auxiliary[source].detach().cpu().numpy()
        out[f"{prefix}_mean"] = float(np.mean(values))
        out[f"{prefix}_std"] = float(np.std(values))
        out[f"{prefix}_abs_max"] = float(np.max(np.abs(values)))
        if prefix == "psi":
            out["psi_min"] = float(np.min(values))
            out["psi_max"] = float(np.max(values))
            out["psi_abs_max"] = float(np.max(np.abs(values)))
    return out


def _streamfunction_metrics(
    benchmark: Any,
    coords_np: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> dict[str, float]:
    x_values = np.unique(coords_np[:, 0])
    y_values = np.unique(coords_np[:, 1])
    if len(x_values) * len(y_values) != len(coords_np):
        return {
            "streamfunction_consistency_rmse": float("nan"),
            "primary_vortex_center_x": float("nan"),
            "primary_vortex_center_y": float("nan"),
            "primary_vortex_y": float("nan"),
            "primary_streamfunction_abs": float("nan"),
            "detected_vortex_count": 0,
            "secondary_vortex_count": 0,
            "weak_secondary_vortex_count": 0,
            "primary_vortex_wall_distance": float("nan"),
        }
    shape = (len(y_values), len(x_values))
    X, Y = np.meshgrid(x_values, y_values)
    psi, consistency = reconstruct_streamfunction(
        X,
        Y,
        np.asarray(u).reshape(shape),
        np.asarray(v).reshape(shape),
        closed_boundary=benchmark.__class__.__name__.lower()
        == "liddrivencavityqualitative",
    )
    vortices = detect_vortices(X, Y, psi)
    weak_vortices = detect_vortices(
        X,
        Y,
        psi,
        minimum_strength_fraction=0.005,
        minimum_prominence_fraction=0.0005,
    )
    if vortices:
        primary = vortices[0]
    else:
        interior = np.abs(psi).copy()
        if min(shape) > 2:
            interior[[0, -1], :] = -np.inf
            interior[:, [0, -1]] = -np.inf
        index = np.unravel_index(int(np.argmax(interior)), shape)
        primary = {
            "x": float(X[index]),
            "y": float(Y[index]),
            "strength": float(abs(psi[index])),
        }
    x_span = max(float(np.max(X) - np.min(X)), 1e-12)
    y_span = max(float(np.max(Y) - np.min(Y)), 1e-12)
    wall_distance = min(
        (float(primary["x"]) - float(np.min(X))) / x_span,
        (float(np.max(X)) - float(primary["x"])) / x_span,
        (float(primary["y"]) - float(np.min(Y))) / y_span,
        (float(np.max(Y)) - float(primary["y"])) / y_span,
    )
    return {
        "streamfunction_consistency_rmse": float(consistency),
        "primary_vortex_center_x": float(primary["x"]),
        "primary_vortex_center_y": float(primary["y"]),
        "primary_vortex_y": float(primary["y"]),
        "primary_streamfunction_abs": float(primary["strength"]),
        "detected_vortex_count": int(len(vortices)),
        "secondary_vortex_count": int(max(0, len(vortices) - 1)),
        "weak_secondary_vortex_count": int(max(0, len(weak_vortices) - 1)),
        "primary_vortex_wall_distance": float(wall_distance),
    }


def _lid_cavity_topology_metrics(
    benchmark: Any,
    coords_np: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> dict[str, float]:
    x_values = np.unique(coords_np[:, 0])
    y_values = np.unique(coords_np[:, 1])
    if len(x_values) * len(y_values) != len(coords_np):
        return {}
    shape = (len(y_values), len(x_values))
    X, Y = np.meshgrid(x_values, y_values)
    return lid_cavity_topology_metrics(
        X,
        Y,
        np.asarray(u).reshape(shape),
        np.asarray(v).reshape(shape),
        reynolds=float(getattr(benchmark, "reynolds", 100.0)),
    )


def _cavity_residual_geometry_metrics(
    model: torch.nn.Module,
    benchmark: Any,
    coords_np: np.ndarray,
    device: torch.device,
    pde: np.ndarray,
    continuity: np.ndarray,
    momentum_u: np.ndarray,
    momentum_v: np.ndarray,
    p_grad: np.ndarray,
    speed: np.ndarray,
) -> dict[str, float]:
    x0, x1, y0, y1 = benchmark.bounds
    width = max(float(x1 - x0), 1e-12)
    height = max(float(y1 - y0), 1e-12)
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    sigma_x = max(width / 10.0, 1e-8)
    sigma_y = max(height / 10.0, 1e-8)
    wx = np.exp(-((coords_np[:, 0:1] - x_mid) / sigma_x) ** 2)
    wy = np.exp(-((coords_np[:, 1:2] - y_mid) / sigma_y) ** 2)
    centerline_weight = np.maximum(wx, wy)
    corner_width = 0.12 * min(width, height)
    left = coords_np[:, 0:1] <= x0 + corner_width
    right = coords_np[:, 0:1] >= x1 - corner_width
    bottom = coords_np[:, 1:2] <= y0 + corner_width
    top = coords_np[:, 1:2] >= y1 - corner_width
    corner_mask = (left | right) & (bottom | top)
    wall_margin = 0.08 * min(width, height)
    core_mask = (
        (coords_np[:, 0:1] >= x0 + wall_margin)
        & (coords_np[:, 0:1] <= x1 - wall_margin)
        & (coords_np[:, 1:2] >= y0 + wall_margin)
        & (coords_np[:, 1:2] <= y1 - wall_margin)
    )
    near_wall_mask = ~core_mask
    top_band_floor = y0 + 0.72 * height
    top_band_ceiling = y1 - 1e-6 * height
    top_corner_strip = 0.05 * width
    top_band_mask = (
        (coords_np[:, 1:2] >= top_band_floor)
        & (coords_np[:, 1:2] < top_band_ceiling)
        & (coords_np[:, 0:1] >= x0 + top_corner_strip)
        & (coords_np[:, 0:1] <= x1 - top_corner_strip)
    )
    upper_core_mask = core_mask & (coords_np[:, 1:2] >= y0 + 0.50 * height)
    side_wall_mask = left | right
    top_wall_mask = top
    boundary_mask = benchmark.boundary_mask_np(coords_np)[:, None] if hasattr(benchmark, "boundary_mask_np") else np.zeros_like(corner_mask)
    corner_boundary_mask = corner_mask & boundary_mask
    return {
        "centerline_pde_residual_mean": _weighted_mean(pde, centerline_weight),
        "centerline_continuity_residual_mean": _weighted_mean(continuity, centerline_weight),
        "core_pde_residual_mean": _masked_mean(pde, core_mask),
        "near_wall_pde_residual_mean": _masked_mean(pde, near_wall_mask),
        "corner_pde_residual_mean": _masked_mean(pde, corner_mask),
        "core_momentum_u_mean": _masked_mean(momentum_u, core_mask),
        "core_momentum_v_mean": _masked_mean(momentum_v, core_mask),
        "near_wall_momentum_u_mean": _masked_mean(momentum_u, near_wall_mask),
        "near_wall_momentum_v_mean": _masked_mean(momentum_v, near_wall_mask),
        "near_wall_momentum_v_max": _masked_max(momentum_v, near_wall_mask),
        "top_band_pde_residual_mean": _masked_mean(pde, top_band_mask),
        "top_band_momentum_u_mean": _masked_mean(momentum_u, top_band_mask),
        "top_band_momentum_v_mean": _masked_mean(momentum_v, top_band_mask),
        "top_band_continuity_residual_mean": _masked_mean(
            continuity, top_band_mask
        ),
        "upper_core_pde_residual_mean": _masked_mean(pde, upper_core_mask),
        "core_speed_mean": _masked_mean(speed, core_mask),
        "upper_core_speed_mean": _masked_mean(speed, upper_core_mask),
        "core_p_grad_mean": _masked_mean(p_grad, core_mask),
        "near_wall_p_grad_mean": _masked_mean(p_grad, near_wall_mask),
        "near_wall_p_grad_max": _masked_max(p_grad, near_wall_mask),
        "side_wall_p_grad_mean": _masked_mean(p_grad, side_wall_mask),
        "side_wall_p_grad_max": _masked_max(p_grad, side_wall_mask),
        "top_wall_p_grad_mean": _masked_mean(p_grad, top_wall_mask),
        "top_wall_p_grad_max": _masked_max(p_grad, top_wall_mask),
        "corner_p_grad_mean": _masked_mean(p_grad, corner_mask),
        "corner_boundary_error": _corner_boundary_error(model, benchmark, coords_np, corner_boundary_mask[:, 0], device),
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    weights = np.asarray(weights, dtype=float).reshape(-1, 1)
    finite = np.isfinite(values) & np.isfinite(weights)
    if not np.any(finite):
        return float("nan")
    weighted = np.where(finite, values * weights, 0.0)
    effective_weights = np.where(finite, weights, 0.0)
    denom = float(np.sum(effective_weights))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(weighted) / denom)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    flat_mask = np.asarray(mask).reshape(-1).astype(bool)
    flat_values = np.asarray(values, dtype=float).reshape(-1)
    selected = flat_values[flat_mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return float("nan")
    return float(np.mean(selected))


def _masked_max(values: np.ndarray, mask: np.ndarray) -> float:
    flat_mask = np.asarray(mask).reshape(-1).astype(bool)
    flat_values = np.asarray(values, dtype=float).reshape(-1)
    selected = flat_values[flat_mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return float("nan")
    return float(np.max(selected))


def _corner_boundary_error(
    model: torch.nn.Module,
    benchmark: Any,
    coords_np: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> float:
    if not np.any(mask):
        return float("nan")
    coords = torch.tensor(coords_np[mask], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(coords)
        ref = benchmark.exact_torch(coords)
        err = torch.sqrt((pred[:, 0:1] - ref["u"]).pow(2) + (pred[:, 1:2] - ref["v"]).pow(2))
    return float(torch.mean(err).detach().cpu())


def _cavity_profile_metrics(
    model: torch.nn.Module,
    benchmark: Any,
    device: torch.device,
) -> dict[str, float | str]:
    out: dict[str, float | str] = {
        "u_centerline_rmse": float("nan"),
        "v_centerline_rmse": float("nan"),
        "u_centerline_rel_l2": float("nan"),
        "v_centerline_rel_l2": float("nan"),
        "centerline_profile_score": float("nan"),
        "cavity_profile_reference_source": "",
    }
    if not bool(getattr(benchmark, "has_profile_reference", False)):
        return out
    profile = benchmark.profile_reference_np()
    out["cavity_profile_reference_source"] = str(profile.get("source", ""))
    pieces = []
    if "u_xy" in profile and "u_ref" in profile:
        pred = _predict_field(model, np.asarray(profile["u_xy"], dtype=float), device)[:, 0:1]
        ref = np.asarray(profile["u_ref"], dtype=float)
        out["u_centerline_rmse"] = rmse(pred, ref)
        out["u_centerline_rel_l2"] = relative_l2(pred, ref)
        pieces.append(float(out["u_centerline_rmse"]))
    if "v_xy" in profile and "v_ref" in profile:
        pred = _predict_field(model, np.asarray(profile["v_xy"], dtype=float), device)[:, 1:2]
        ref = np.asarray(profile["v_ref"], dtype=float)
        out["v_centerline_rmse"] = rmse(pred, ref)
        out["v_centerline_rel_l2"] = relative_l2(pred, ref)
        pieces.append(float(out["v_centerline_rmse"]))
    out["centerline_profile_score"] = float(sum(pieces)) if pieces else float("nan")
    return out


def _predict_field(model: torch.nn.Module, coords_np: np.ndarray, device: torch.device) -> np.ndarray:
    coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(coords).detach().cpu().numpy()


def _boundary_error(
    model: torch.nn.Module,
    benchmark: Any,
    coords_np: np.ndarray,
    device: torch.device,
) -> float:
    if not hasattr(benchmark, "boundary_mask_np"):
        return float("nan")
    mask = benchmark.boundary_mask_np(coords_np)
    if not np.any(mask):
        return float("nan")
    coords = torch.tensor(coords_np[mask], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(coords)
        ref = benchmark.exact_torch(coords)
        err = torch.sqrt((pred[:, 0:1] - ref["u"]).pow(2) + (pred[:, 1:2] - ref["v"]).pow(2))
    return float(torch.mean(err).detach().cpu())


def _boundary_metrics(
    model: torch.nn.Module,
    benchmark: Any,
    coords_np: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    empty = {
        "u_boundary_rmse": float("nan"),
        "v_boundary_rmse": float("nan"),
        "boundary_speed_rmse": float("nan"),
        "unweighted_bc_loss": float("nan"),
    }
    if not hasattr(benchmark, "boundary_mask_np"):
        return empty
    mask = benchmark.boundary_mask_np(coords_np)
    if not np.any(mask):
        return empty
    coords = torch.tensor(coords_np[mask], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(coords)
        ref = benchmark.exact_torch(coords)
        u_err2 = (pred[:, 0:1] - ref["u"]).pow(2)
        v_err2 = (pred[:, 1:2] - ref["v"]).pow(2)
        err2 = u_err2 + v_err2
    return {
        "u_boundary_rmse": float(torch.sqrt(torch.mean(u_err2)).detach().cpu()),
        "v_boundary_rmse": float(torch.sqrt(torch.mean(v_err2)).detach().cpu()),
        "boundary_speed_rmse": float(torch.sqrt(torch.mean(err2)).detach().cpu()),
        "unweighted_bc_loss": float(torch.mean(err2).detach().cpu()),
    }
