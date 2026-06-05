"""Global model metrics."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from src.physics.kovasznay import center_pressure
from src.physics.navier_stokes import navier_stokes_residuals


def relative_l2(pred: np.ndarray, true: np.ndarray, min_reference_norm: float = 1e-8) -> float:
    """Relative L2, undefined when the reference field is effectively zero."""
    ref_norm = float(np.linalg.norm(true))
    if ref_norm < min_reference_norm:
        return float("nan")
    return float(np.linalg.norm(pred - true) / ref_norm)


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
) -> dict[str, float]:
    """Compute global evaluation metrics without using them for adaptation."""
    start = time.time()
    coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred = model(coords)
    residuals = navier_stokes_residuals(model, coords, nu=benchmark.nu, steady=steady)
    has_reference = bool(getattr(benchmark, "has_reference", True))
    ref = benchmark.exact_np(coords_np) if has_reference else None

    u = pred[:, 0:1].detach().cpu().numpy()
    v = pred[:, 1:2].detach().cpu().numpy()
    p = pred[:, 2:3].detach().cpu().numpy()
    omega = residuals["omega"].detach().cpu().numpy()
    p_c = center_pressure(p)
    speed = np.sqrt(u * u + v * v)
    pde = residuals["pde_residual"].detach().cpu().numpy()
    pde_loss = (
        residuals["f_u"].detach().cpu().numpy() ** 2
        + residuals["f_v"].detach().cpu().numpy() ** 2
        + residuals["f_c"].detach().cpu().numpy() ** 2
    )
    div = np.abs(residuals["f_c"].detach().cpu().numpy())
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
        "omega_pred_abs_max": float(np.max(np.abs(omega))),
        "pressure_gradient_error": float("nan"),
        "divergence_norm": float(np.mean(div)),
        "continuity_residual_mean": float(np.mean(np.abs(residuals["f_c"].detach().cpu().numpy()))),
        "momentum_residual_mean": float(
            np.mean(
                np.sqrt(
                    residuals["f_u"].detach().cpu().numpy() ** 2
                    + residuals["f_v"].detach().cpu().numpy() ** 2
                )
            )
        ),
        "pde_residual_mean": float(np.mean(pde)),
        "pde_residual_max": float(np.max(pde)),
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
        "unweighted_data_loss": float("nan"),
        "unweighted_pde_loss": float(np.mean(pde_loss)),
        "unweighted_bc_loss": unweighted_bc_loss,
        "unweighted_validation_loss": float("nan"),
        "wall_clock_eval_sec": time.time() - start,
        "num_eval_points": int(coords_np.shape[0]),
    }
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
                "velocity_full_rel_l2": relative_l2(speed, speed_ref),
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
    metrics.update(_cavity_profile_metrics(model, benchmark, device))
    if benchmark.__class__.__name__.lower().startswith("liddrivencavity"):
        metrics.update(
            _cavity_residual_geometry_metrics(
                model=model,
                benchmark=benchmark,
                coords_np=coords_np,
                device=device,
                pde=pde,
                continuity=np.abs(residuals["f_c"].detach().cpu().numpy()),
            )
        )
    metrics["unweighted_validation_loss"] = _finite_sum(
        [
            metrics["unweighted_pde_loss"],
            metrics["unweighted_bc_loss"],
            0.0 if not has_reference else metrics["unweighted_data_loss"],
        ]
    )
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


def _cavity_residual_geometry_metrics(
    model: torch.nn.Module,
    benchmark: Any,
    coords_np: np.ndarray,
    device: torch.device,
    pde: np.ndarray,
    continuity: np.ndarray,
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
    boundary_mask = benchmark.boundary_mask_np(coords_np)[:, None] if hasattr(benchmark, "boundary_mask_np") else np.zeros_like(corner_mask)
    corner_boundary_mask = corner_mask & boundary_mask
    return {
        "centerline_pde_residual_mean": _weighted_mean(pde, centerline_weight),
        "centerline_continuity_residual_mean": _weighted_mean(continuity, centerline_weight),
        "corner_pde_residual_mean": _masked_mean(pde, corner_mask),
        "corner_boundary_error": _corner_boundary_error(model, benchmark, coords_np, corner_boundary_mask[:, 0], device),
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    weights = np.asarray(weights, dtype=float).reshape(-1, 1)
    denom = float(np.sum(weights))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(values * weights) / denom)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    flat_mask = np.asarray(mask).reshape(-1).astype(bool)
    if not np.any(flat_mask):
        return float("nan")
    flat_values = np.asarray(values, dtype=float).reshape(-1)
    return float(np.mean(flat_values[flat_mask]))


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
