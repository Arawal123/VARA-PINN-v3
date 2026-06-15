"""Loss utilities for independent modern PINN baselines."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.losses.base_losses import compute_pointwise_losses
from src.physics.navier_stokes import gradients, navier_stokes_residuals


INTERIOR_LOSSES = {"pde", "momentum_u", "momentum_v", "continuity"}


class ReLoBRaLoWeights:
    """Relative-loss balancing with deterministic random lookback."""

    def __init__(
        self,
        names: list[str],
        temperature: float = 0.1,
        alpha: float = 0.999,
        random_lookback_probability: float = 0.999,
        seed: int = 0,
    ) -> None:
        self.names = list(names)
        self.temperature = max(float(temperature), 1e-6)
        self.alpha = float(alpha)
        self.random_lookback_probability = float(random_lookback_probability)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.initial: torch.Tensor | None = None
        self.previous: torch.Tensor | None = None
        self.weights = torch.ones(len(self.names), dtype=torch.float32)

    def update(self, losses: dict[str, torch.Tensor]) -> dict[str, float]:
        values = torch.stack([losses[name].detach().float().cpu() for name in self.names])
        if self.initial is None:
            self.initial = values.clamp_min(1e-12)
            self.previous = values.clamp_min(1e-12)
            return self.as_dict()
        use_previous = bool(
            torch.rand((), generator=self.generator).item() < self.random_lookback_probability
        )
        reference = self.previous if use_previous else self.initial
        ratios = values / reference.clamp_min(1e-12)
        balanced = len(self.names) * torch.softmax(ratios / self.temperature, dim=0)
        self.weights = self.alpha * self.weights + (1.0 - self.alpha) * balanced
        self.weights = len(self.names) * self.weights / self.weights.sum().clamp_min(1e-12)
        self.previous = values.clamp_min(1e-12)
        return self.as_dict()

    def as_dict(self) -> dict[str, float]:
        return {name: float(self.weights[i]) for i, name in enumerate(self.names)}


class ResidualAttentionState:
    """Residual-based attention with bounded mean-one point weights."""

    def __init__(self, size: int, decay: float = 0.999, eta: float = 0.01, maximum: float = 10.0) -> None:
        self.values = torch.ones(int(size), dtype=torch.float32)
        self.decay = float(decay)
        self.eta = float(eta)
        self.maximum = float(maximum)

    def update(self, residual: torch.Tensor) -> torch.Tensor:
        detached = residual.detach().reshape(-1).float().cpu().abs()
        normalized = detached / detached.max().clamp_min(1e-12)
        self.values = self.decay * self.values + self.eta * normalized
        self.values = self.values / self.values.mean().clamp_min(1e-12)
        self.values.clamp_(min=1e-3, max=self.maximum)
        return self.values.to(device=residual.device, dtype=residual.dtype)


def causal_temporal_objective(
    pointwise: dict[str, torch.Tensor],
    times: torch.Tensor,
    scalar_weights: dict[str, float],
    chunks: int = 16,
    epsilon: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    """Causal PINN weighting over ordered temporal residual chunks."""
    if times.ndim > 1:
        times = times.reshape(-1)
    edges = torch.linspace(
        float(times.min().detach()),
        float(times.max().detach()) + 1e-7,
        int(chunks) + 1,
        device=times.device,
        dtype=times.dtype,
    )
    interior_names = [name for name in INTERIOR_LOSSES if name in pointwise]
    reduced: dict[str, torch.Tensor] = {}
    cumulative = times.new_tensor(0.0)
    chunk_weights = []
    per_name = {name: [] for name in interior_names}
    for index in range(int(chunks)):
        mask = (times >= edges[index]) & (times < edges[index + 1])
        if not torch.any(mask):
            continue
        weight = torch.exp(-float(epsilon) * cumulative.detach())
        chunk_weights.append(weight)
        chunk_total = times.new_tensor(0.0)
        for name in interior_names:
            value = pointwise[name].reshape(-1)[mask].mean()
            per_name[name].append(weight * value)
            chunk_total = chunk_total + value
        cumulative = cumulative + chunk_total
    for name in interior_names:
        reduced[name] = torch.stack(per_name[name]).mean() if per_name[name] else pointwise[name].mean()
    for name, values in pointwise.items():
        if name not in reduced:
            reduced[name] = values.mean()
    total = sum(float(scalar_weights.get(name, 0.0)) * value for name, value in reduced.items())
    logs = {
        "causal_weight_min": float(torch.stack(chunk_weights).min().detach().cpu()) if chunk_weights else 1.0,
        "causal_weight_mean": float(torch.stack(chunk_weights).mean().detach().cpu()) if chunk_weights else 1.0,
    }
    return total, reduced, logs


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
    pointwise: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    """Soft-attention PINN objective with adversarially trained point weights."""
    if pointwise is None:
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
    **loss_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Standard PINN losses plus spatial gradients of governing residuals."""
    pointwise = compute_pointwise_losses(
        model,
        batch,
        benchmark,
        steady,
        **loss_kwargs,
    )
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
