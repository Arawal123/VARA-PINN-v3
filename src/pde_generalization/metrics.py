"""Evaluation-only metrics for manufactured full-field references."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .benchmarks import Burgers2DBenchmark, ManufacturedBenchmark
from .residuals import compute_residuals


def make_evaluation_grid(
    benchmark: ManufacturedBenchmark,
    nx: int,
    ny: int,
    nt: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a deterministic Cartesian x-y-t test grid."""
    x0, x1, y0, y1 = benchmark.bounds
    t0, t1 = benchmark.t_bounds
    x = torch.linspace(x0, x1, int(nx), device=device, dtype=dtype)
    y = torch.linspace(y0, y1, int(ny), device=device, dtype=dtype)
    t = torch.linspace(t0, t1, int(nt), device=device, dtype=dtype)
    mesh = torch.meshgrid(x, y, t, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, 3)


def evaluate_model(
    model: nn.Module,
    benchmark: ManufacturedBenchmark,
    evaluation_coordinates: torch.Tensor,
    boundary_coordinates: torch.Tensor,
    initial_coordinates: torch.Tensor,
    sparse_coordinates: torch.Tensor,
    sparse_targets: torch.Tensor,
    *,
    residual_chunk_size: int = 2048,
) -> dict[str, float]:
    """Compute full-field, localized, physics, and prescribed-data errors."""
    model.eval()
    with torch.no_grad():
        reference = benchmark.exact(evaluation_coordinates)
        prediction = model(evaluation_coordinates)
        boundary_error = (
            model(boundary_coordinates) - benchmark.boundary_values(boundary_coordinates)
        ).square().mean()
        initial_error = (
            model(initial_coordinates) - benchmark.initial_values(initial_coordinates)
        ).square().mean()
        if sparse_coordinates.numel():
            sparse_component = (model(sparse_coordinates) - sparse_targets).square().mean(dim=0)
        else:
            sparse_component = torch.full(
                (len(benchmark.output_names),),
                float("nan"),
                device=evaluation_coordinates.device,
            )

    component_rel = []
    for index in range(reference.shape[1]):
        component_rel.append(_relative_l2(prediction[:, index], reference[:, index]))
    all_rel = _relative_l2(prediction.reshape(-1), reference.reshape(-1))
    hard_mask = benchmark.hard_region_mask(evaluation_coordinates, reference)
    if bool(hard_mask.any()):
        hard_rel = _relative_l2(prediction[hard_mask].reshape(-1), reference[hard_mask].reshape(-1))
        hard_mse = float((prediction[hard_mask] - reference[hard_mask]).square().mean().cpu())
    else:
        hard_rel = float("nan")
        hard_mse = float("nan")

    residual_values: dict[str, list[torch.Tensor]] = {}
    for chunk in evaluation_coordinates.split(max(1, int(residual_chunk_size))):
        coords = chunk.detach().clone().requires_grad_(True)
        result = compute_residuals(model, coords, benchmark)
        for name, values in result.items():
            residual_values.setdefault(name, []).append(values.detach().abs().cpu())
    residual_means = {
        name: float(torch.cat(values).mean())
        for name, values in residual_values.items()
    }

    common = {
        "boundary_error": float(boundary_error.cpu()),
        "initial_condition_error": float(initial_error.cpu()),
        "pde_residual_mean": residual_means["pde_residual"],
        "hard_region_rel_l2": hard_rel,
        "hard_region_mse": hard_mse,
    }
    if isinstance(benchmark, Burgers2DBenchmark):
        return {
            "burgers_u_rel_l2": component_rel[0],
            "burgers_v_rel_l2": component_rel[1],
            "burgers_velocity_rel_l2": all_rel,
            "burgers_u_mse_sparse": float(sparse_component[0].cpu()),
            "burgers_v_mse_sparse": float(sparse_component[1].cpu()),
            "burgers_velocity_mse_sparse": float(torch.nanmean(sparse_component).cpu()),
            "burgers_pde_residual_mean": common["pde_residual_mean"],
            "burgers_fu_residual_mean": residual_means["f_u"],
            "burgers_fv_residual_mean": residual_means["f_v"],
            "burgers_boundary_error": common["boundary_error"],
            "burgers_initial_condition_error": common["initial_condition_error"],
            "burgers_localized_band_rel_l2": hard_rel,
            "burgers_localized_band_mse": hard_mse,
        }
    if benchmark.name == "allen_cahn":
        return {
            "allen_cahn_u_rel_l2": component_rel[0],
            "allen_cahn_u_mse_sparse": float(sparse_component[0].cpu()),
            "allen_cahn_pde_residual_mean": common["pde_residual_mean"],
            "allen_cahn_interface_band_rel_l2": hard_rel,
            "allen_cahn_interface_band_mse": hard_mse,
            "allen_cahn_boundary_error": common["boundary_error"],
            "allen_cahn_initial_condition_error": common["initial_condition_error"],
        }
    return {
        "advdiff_u_rel_l2": component_rel[0],
        "advdiff_u_mse_sparse": float(sparse_component[0].cpu()),
        "advdiff_pde_residual_mean": common["pde_residual_mean"],
        "advdiff_layer_band_rel_l2": hard_rel,
        "advdiff_layer_band_mse": hard_mse,
        "advdiff_boundary_error": common["boundary_error"],
        "advdiff_initial_condition_error": common["initial_condition_error"],
    }


def primary_metric_name(benchmark_name: str) -> str:
    """Return the principal reconstruction metric for comparisons."""
    return {
        "burgers2d": "burgers_velocity_rel_l2",
        "allen_cahn": "allen_cahn_u_rel_l2",
        "advection_diffusion": "advdiff_u_rel_l2",
    }[benchmark_name]


def metric_groups(benchmark_name: str) -> dict[str, list[str]]:
    """Group metrics for tables and publication plots."""
    if benchmark_name == "burgers2d":
        return {
            "full_field_reconstruction": ["burgers_velocity_rel_l2", "burgers_u_rel_l2", "burgers_v_rel_l2"],
            "sparse_data_fit": ["burgers_velocity_mse_sparse"],
            "physics_consistency": ["burgers_pde_residual_mean"],
            "localized_hard_region": ["burgers_localized_band_rel_l2"],
        }
    if benchmark_name == "allen_cahn":
        return {
            "full_field_reconstruction": ["allen_cahn_u_rel_l2"],
            "sparse_data_fit": ["allen_cahn_u_mse_sparse"],
            "physics_consistency": ["allen_cahn_pde_residual_mean"],
            "localized_hard_region": ["allen_cahn_interface_band_rel_l2"],
        }
    return {
        "full_field_reconstruction": ["advdiff_u_rel_l2"],
        "sparse_data_fit": ["advdiff_u_mse_sparse"],
        "physics_consistency": ["advdiff_pde_residual_mean"],
        "localized_hard_region": ["advdiff_layer_band_rel_l2"],
    }


def _relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(prediction - reference)
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return float((numerator / denominator).detach().cpu())
