"""Evaluation-only Cahn--Hilliard metrics and deterministic grids."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .benchmark import CahnHilliardBenchmark
from .residuals import compute_cahn_hilliard_residuals


def make_evaluation_grid(
    benchmark: CahnHilliardBenchmark,
    nx: int,
    ny: int,
    nt: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a fixed full-field reporting grid."""
    x0, x1, y0, y1 = benchmark.bounds
    t0, t1 = benchmark.t_bounds
    x = torch.linspace(x0, x1, int(nx), device=device, dtype=dtype)
    y = torch.linspace(y0, y1, int(ny), device=device, dtype=dtype)
    t = torch.linspace(t0, t1, int(nt), device=device, dtype=dtype)
    mesh = torch.meshgrid(x, y, t, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, 3)


def evaluate_cahn_hilliard(
    model: nn.Module,
    benchmark: CahnHilliardBenchmark,
    evaluation_coordinates: torch.Tensor,
    boundary_coordinates: torch.Tensor,
    boundary_targets: torch.Tensor,
    initial_coordinates: torch.Tensor,
    initial_targets: torch.Tensor,
    sparse_coordinates: torch.Tensor,
    sparse_targets: torch.Tensor,
    *,
    sparse_fraction: float,
    sparse_seed: int,
    sparse_hash: str,
    chunk_size: int = 1024,
) -> dict[str, Any]:
    """Compute reporting metrics after all controller decisions are complete."""
    predictions: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    ch_residuals: list[torch.Tensor] = []
    mu_residuals: list[torch.Tensor] = []
    pde_residuals: list[torch.Tensor] = []
    model.eval()
    for chunk in evaluation_coordinates.split(max(1, int(chunk_size))):
        reference = benchmark.exact(chunk).detach()
        with torch.no_grad():
            prediction = model(chunk)
        coords = chunk.detach().clone().requires_grad_(True)
        forcing = benchmark.forcing(coords)
        residuals = compute_cahn_hilliard_residuals(
            model, coords, benchmark, forcing
        )
        predictions.append(prediction.detach().cpu())
        references.append(reference.detach().cpu())
        ch_residuals.append(residuals["r_ch"].detach().abs().cpu())
        mu_residuals.append(residuals["r_mu"].detach().abs().cpu())
        pde_residuals.append(residuals["pde_residual"].detach().cpu())

    prediction = torch.cat(predictions)
    reference = torch.cat(references)
    ch_values = torch.cat(ch_residuals).reshape(-1)
    mu_values = torch.cat(mu_residuals).reshape(-1)
    pde_values = torch.cat(pde_residuals).reshape(-1)
    eval_cpu = evaluation_coordinates.detach().cpu()

    u_pred, mu_pred = prediction[:, 0], prediction[:, 1]
    u_ref, mu_ref = reference[:, 0], reference[:, 1]
    interface = benchmark.interface_band(u_ref)
    interface_core = benchmark.interface_core(u_ref)
    final_mask = torch.isclose(
        eval_cpu[:, 2],
        torch.tensor(benchmark.t_bounds[1], dtype=eval_cpu.dtype),
        rtol=0.0,
        atol=1e-7,
    )

    with torch.no_grad():
        boundary_prediction = model(boundary_coordinates).detach().cpu()
        initial_prediction = model(initial_coordinates).detach().cpu()
        if sparse_coordinates.numel():
            sparse_prediction = model(sparse_coordinates).detach().cpu()
            sparse_reference = sparse_targets.detach().cpu()
            sparse_mse = (sparse_prediction - sparse_reference).square().mean(dim=0)
        else:
            sparse_mse = torch.full((2,), float("nan"))

    boundary_reference = boundary_targets.detach().cpu()
    initial_reference = initial_targets.detach().cpu()
    metrics: dict[str, Any] = {
        "cahn_hilliard_u_rel_l2": _relative_l2(u_pred, u_ref),
        "cahn_hilliard_u_rmse": float(torch.sqrt((u_pred - u_ref).square().mean())),
        "cahn_hilliard_u_mae": float((u_pred - u_ref).abs().mean()),
        "cahn_hilliard_mu_rel_l2": _relative_l2(mu_pred, mu_ref),
        "cahn_hilliard_mu_rmse": float(torch.sqrt((mu_pred - mu_ref).square().mean())),
        "cahn_hilliard_sparse_u_mse": float(sparse_mse[0]),
        "cahn_hilliard_sparse_mu_mse": float(sparse_mse[1]),
        "cahn_hilliard_sparse_sample_count": int(sparse_coordinates.shape[0]),
        "cahn_hilliard_sparse_fraction": float(sparse_fraction),
        "cahn_hilliard_sparse_seed": int(sparse_seed),
        "cahn_hilliard_sparse_hash": sparse_hash,
        "cahn_hilliard_pde_residual_mean": float(pde_values.mean()),
        "cahn_hilliard_ch_residual_mean": float(ch_values.mean()),
        "cahn_hilliard_mu_residual_mean": float(mu_values.mean()),
        "cahn_hilliard_pde_residual_95p": float(torch.quantile(pde_values, 0.95)),
        "cahn_hilliard_ch_residual_95p": float(torch.quantile(ch_values, 0.95)),
        "cahn_hilliard_mu_residual_95p": float(torch.quantile(mu_values, 0.95)),
        "cahn_hilliard_interface_band_rel_l2": _masked_relative_l2(u_pred, u_ref, interface),
        "cahn_hilliard_interface_band_mse": _masked_mean((u_pred - u_ref).square(), interface),
        "cahn_hilliard_interface_band_mae": _masked_mean((u_pred - u_ref).abs(), interface),
        "cahn_hilliard_interface_band_sample_count": int(interface.sum()),
        "cahn_hilliard_interface_core_rel_l2": _masked_relative_l2(u_pred, u_ref, interface_core),
        "cahn_hilliard_boundary_u_error": float((boundary_prediction[:, 0] - boundary_reference[:, 0]).square().mean()),
        "cahn_hilliard_boundary_mu_error": float((boundary_prediction[:, 1] - boundary_reference[:, 1]).square().mean()),
        "cahn_hilliard_initial_u_error": float((initial_prediction[:, 0] - initial_reference[:, 0]).square().mean()),
        "cahn_hilliard_initial_mu_error": float((initial_prediction[:, 1] - initial_reference[:, 1]).square().mean()),
        "cahn_hilliard_mass_ref": float(u_ref[final_mask].mean()),
        "cahn_hilliard_mass_pred": float(u_pred[final_mask].mean()),
        "cahn_hilliard_mass_error": float((u_pred[final_mask].mean() - u_ref[final_mask].mean()).abs()),
        "cahn_hilliard_phase_range_min": float(u_pred.min()),
        "cahn_hilliard_phase_range_max": float(u_pred.max()),
        "cahn_hilliard_overshoot_above_one": float(torch.relu(u_pred.max() - 1.0)),
        "cahn_hilliard_overshoot_below_minus_one": float(torch.relu(-1.0 - u_pred.min())),
    }
    return metrics


def metric_groups() -> dict[str, list[str]]:
    """Publication table groups."""
    return {
        "full_field_reconstruction": [
            "cahn_hilliard_u_rel_l2",
            "cahn_hilliard_u_rmse",
            "cahn_hilliard_mu_rel_l2",
        ],
        "sparse_data_fit": ["cahn_hilliard_sparse_u_mse"],
        "interface_band_recovery": ["cahn_hilliard_interface_band_rel_l2"],
        "physics_residuals": ["cahn_hilliard_pde_residual_mean"],
        "boundary_initial_consistency": [
            "cahn_hilliard_boundary_u_error",
            "cahn_hilliard_initial_u_error",
        ],
        "runtime_compute": ["optimization_wall_clock_sec", "applied_optimizer_steps"],
        "controller_behavior": [
            "accepted_interventions",
            "rejected_interventions",
            "rollback_count",
            "accepted_u_interface_interventions",
            "accepted_mu_interventions",
            "rejected_due_to_sparse_u_guard",
            "rejected_due_to_ic_u_guard",
            "rejected_due_to_bc_u_guard",
            "rejected_due_to_phase_range_guard",
            "accepted_pareto_safe_interventions",
            "rejected_hard_guard_pde",
            "rejected_hard_guard_ch",
            "rejected_hard_guard_mass",
            "rejected_hard_guard_phase",
            "rejected_hard_guard_sparse_u",
            "rejected_mu_only",
            "post_block_rollbacks",
            "accepted_interface_targets",
            "accepted_sparse_u_targets",
        ],
    }


def _relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(prediction - reference)
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return float(numerator / denominator)


def _masked_relative_l2(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    if not bool(mask.any()):
        return float("nan")
    return _relative_l2(prediction[mask], reference[mask])


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    return float(values[mask].mean()) if bool(mask.any()) else float("nan")
