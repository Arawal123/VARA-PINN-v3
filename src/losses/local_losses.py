"""Variable-region local loss terms."""

from __future__ import annotations

from typing import Any

import torch

from .base_losses import mse


LOSS_COORD_SOURCE = {
    "u": "xy_data",
    "v": "xy_data",
    "p": "xy_data",
    "omega": "xy_data",
    "pressure_gradient": "xy_data",
    "momentum_u": "xy_f",
    "momentum_v": "xy_f",
    "continuity": "xy_f",
    "pde": "xy_f",
    "pressure_poisson": "xy_f",
    "vorticity_transport": "xy_f",
    "bc": "xy_bc",
}


def compute_budgeted_patch_losses(
    pointwise: dict[str, torch.Tensor],
    batch: dict[str, Any],
    patch_grid: Any,
    multipliers: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Reduce pointwise losses with mean-one patch multipliers.

    Unlike the V1 additive local loss, this replaces the ordinary mean for an
    affected loss component. A multiplier field with mean one therefore
    redistributes emphasis without increasing total configured loss mass.
    """
    reduced: dict[str, torch.Tensor] = {}
    for name, values in pointwise.items():
        if name not in multipliers:
            reduced[name] = mse(values)
            continue
        coord_name = LOSS_COORD_SOURCE.get(name)
        if coord_name is None or coord_name not in batch:
            reduced[name] = mse(values)
            continue
        coords = batch[coord_name]
        patch_ids = patch_grid.assign_torch(coords).to(device=values.device)
        patch_values = torch.as_tensor(
            multipliers[name],
            dtype=values.dtype,
            device=values.device,
        )
        weights = patch_values[patch_ids].reshape(values.shape)
        weights = weights / weights.mean().clamp_min(1e-12)
        reduced[name] = torch.mean(weights * values)
    return reduced


def compute_local_weighted_loss(
    pointwise: dict[str, torch.Tensor],
    batch: dict[str, Any],
    patch_grid: Any,
    local_weights: dict[str, dict[int, float]],
    entropy_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute sum_j,r w[j,r] L[j,r] with empty-mask protection."""
    device = next(iter(pointwise.values())).device
    total = torch.tensor(0.0, device=device)
    logs: dict[str, float] = {}
    active_weights = []
    for variable, patch_weights in local_weights.items():
        if variable not in pointwise:
            continue
        coord_name = LOSS_COORD_SOURCE.get(variable)
        if coord_name is None or coord_name not in batch:
            continue
        coords = batch[coord_name]
        patch_ids = patch_grid.assign_torch(coords).to(device=device)
        values = pointwise[variable]
        for patch_id, weight in patch_weights.items():
            mask = patch_ids == int(patch_id)
            if not torch.any(mask):
                local_loss = values.new_tensor(0.0)
            else:
                local_loss = mse(values[mask])
            key = f"local/{variable}/patch_{patch_id}"
            logs[key] = float(local_loss.detach().cpu())
            total = total + float(weight) * local_loss
            active_weights.append(float(weight))
    if entropy_weight > 0.0 and active_weights:
        w = torch.tensor(active_weights, dtype=torch.float32, device=device)
        prob = w / w.sum().clamp_min(1e-12)
        entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum()
        total = total - float(entropy_weight) * entropy
        logs["local/entropy"] = float(entropy.detach().cpu())
    return total, logs
