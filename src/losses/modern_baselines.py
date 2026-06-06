"""Loss utilities for independent modern PINN baselines."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.losses.base_losses import compute_pointwise_losses
from src.physics.navier_stokes import gradients, navier_stokes_residuals


INTERIOR_LOSSES = {"pde", "momentum_u", "momentum_v", "continuity"}


def positive_attention(logits: torch.Tensor, maximum: float = 20.0) -> torch.Tensor:
    """Positive, mean-one attention weights with a finite upper guard."""
    values = F.softplus(logits) + 1e-6
    values = values / values.mean().clamp_min(1e-8)
    return values.clamp(max=float(maximum))


def self_adaptive_objective(
    model: torch.nn.Module,
    batch: dict[str, Any],
    benchmark: Any,
    steady: bool,
    scalar_weights: dict[str, float],
    interior_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    maximum_attention: float = 20.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    """Soft-attention PINN objective with adversarially trained point weights."""
    pointwise = compute_pointwise_losses(model, batch, benchmark, steady)
    interior_attention = positive_attention(interior_logits, maximum_attention)
    boundary_attention = positive_attention(boundary_logits, maximum_attention)
    reduced: dict[str, torch.Tensor] = {}
    total = next(iter(pointwise.values())).new_tensor(0.0)
    for name, values in pointwise.items():
        if name in INTERIOR_LOSSES:
            loss = torch.mean(interior_attention * values.reshape(-1))
        elif name == "bc":
            loss = torch.mean(boundary_attention * values.reshape(-1))
        else:
            loss = torch.mean(values)
        reduced[name] = loss
        total = total + float(scalar_weights.get(name, 0.0)) * loss
    logs = {
        "attention_interior_mean": float(interior_attention.detach().mean().cpu()),
        "attention_interior_max": float(interior_attention.detach().max().cpu()),
        "attention_boundary_mean": float(boundary_attention.detach().mean().cpu()),
        "attention_boundary_max": float(boundary_attention.detach().max().cpu()),
    }
    return total, reduced, logs


def gradient_enhanced_pointwise_losses(
    model: torch.nn.Module,
    batch: dict[str, Any],
    benchmark: Any,
    steady: bool,
) -> dict[str, torch.Tensor]:
    """Standard PINN losses plus spatial gradients of governing residuals."""
    pointwise = compute_pointwise_losses(model, batch, benchmark, steady)
    residuals = navier_stokes_residuals(model, batch["xy_f"], benchmark.nu, steady=steady)
    coords = residuals["coords"]
    gradient_terms = []
    for name in ("f_u", "f_v", "f_c"):
        grad = gradients(residuals[name], coords)
        spatial = grad[:, :2]
        gradient_terms.append(torch.sum(spatial * spatial, dim=1, keepdim=True))
    pointwise["pde_gradient"] = sum(gradient_terms)
    return pointwise


def loss_gradient_norm(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> float:
    """L2 norm of one loss component's model gradient."""
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    total = loss.new_tensor(0.0)
    for grad in grads:
        if grad is not None:
            total = total + torch.sum(grad.detach() ** 2)
    return float(torch.sqrt(total + 1e-24).cpu())
