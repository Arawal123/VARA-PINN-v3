"""Reference-free spatiotemporal Cahn--Hilliard diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .autograd import gradient
from .benchmark import CahnHilliardBenchmark
from .residuals import compute_cahn_hilliard_residuals


@dataclass(frozen=True)
class CahnHilliardPatch:
    """One regular x-y-t controller patch."""

    patch_id: int
    ix: int
    iy: int
    it: int
    bounds: tuple[float, float, float, float, float, float]


class CahnHilliardPatchGrid:
    """Regular patch grid with no dependency on prior PDE suites."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        t_bounds: tuple[float, float],
        nx: int,
        ny: int,
        nt: int,
    ) -> None:
        self.bounds = bounds
        self.t_bounds = t_bounds
        self.nx = int(nx)
        self.ny = int(ny)
        self.nt = int(nt)
        if min(self.nx, self.ny, self.nt) <= 0:
            raise ValueError("All Cahn--Hilliard patch counts must be positive.")
        self.num_patches = self.nx * self.ny * self.nt
        self._patches = self._build()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        benchmark: CahnHilliardBenchmark,
    ) -> "CahnHilliardPatchGrid":
        patch_cfg = dict(config.get("patches", {}))
        return cls(
            benchmark.bounds,
            benchmark.t_bounds,
            int(patch_cfg.get("nx_patches", 5)),
            int(patch_cfg.get("ny_patches", 5)),
            int(patch_cfg.get("nt_patches", 4)),
        )

    def get_patch(self, patch_id: int) -> CahnHilliardPatch:
        return self._patches[int(patch_id)]

    def assign_numpy(self, coordinates: np.ndarray) -> np.ndarray:
        coords = np.asarray(coordinates)
        x0, x1, y0, y1 = self.bounds
        t0, t1 = self.t_bounds
        ix = np.clip(((coords[:, 0] - x0) / (x1 - x0) * self.nx).astype(int), 0, self.nx - 1)
        iy = np.clip(((coords[:, 1] - y0) / (y1 - y0) * self.ny).astype(int), 0, self.ny - 1)
        it = np.clip(((coords[:, 2] - t0) / (t1 - t0) * self.nt).astype(int), 0, self.nt - 1)
        return ix + self.nx * (iy + self.ny * it)

    def assign_torch(self, coordinates: torch.Tensor) -> torch.Tensor:
        x0, x1, y0, y1 = self.bounds
        t0, t1 = self.t_bounds
        ix = torch.clamp(((coordinates[:, 0] - x0) / (x1 - x0) * self.nx).long(), 0, self.nx - 1)
        iy = torch.clamp(((coordinates[:, 1] - y0) / (y1 - y0) * self.ny).long(), 0, self.ny - 1)
        it = torch.clamp(((coordinates[:, 2] - t0) / (t1 - t0) * self.nt).long(), 0, self.nt - 1)
        return ix + self.nx * (iy + self.ny * it)

    def _build(self) -> list[CahnHilliardPatch]:
        x0, x1, y0, y1 = self.bounds
        t0, t1 = self.t_bounds
        xs = np.linspace(x0, x1, self.nx + 1)
        ys = np.linspace(y0, y1, self.ny + 1)
        ts = np.linspace(t0, t1, self.nt + 1)
        patches = []
        for it in range(self.nt):
            for iy in range(self.ny):
                for ix in range(self.nx):
                    patch_id = ix + self.nx * (iy + self.ny * it)
                    patches.append(
                        CahnHilliardPatch(
                            patch_id,
                            ix,
                            iy,
                            it,
                            (
                                float(xs[ix]),
                                float(xs[ix + 1]),
                                float(ys[iy]),
                                float(ys[iy + 1]),
                                float(ts[it]),
                                float(ts[it + 1]),
                            ),
                        )
                    )
        return patches


@dataclass
class DiagnosticSnapshot:
    """All diagnostic maps plus the subset eligible for adaptation."""

    names: list[str]
    raw_scores: np.ndarray
    normalized_scores: np.ndarray
    priority_scores: np.ndarray
    adaptation_names: list[str]
    scalar_metrics: dict[str, float]

    def adaptation_scores(self) -> tuple[np.ndarray, np.ndarray]:
        indexes = [self.names.index(name) for name in self.adaptation_names]
        return self.raw_scores[indexes], self.priority_scores[indexes]


def build_diagnostic_snapshot(
    model: nn.Module,
    benchmark: CahnHilliardBenchmark,
    patch_grid: CahnHilliardPatchGrid,
    batch: dict[str, torch.Tensor],
    *,
    percentile: float,
    variable_awareness: bool,
    interface_tau: float,
    interface_focus_strength: float,
    interface_threshold: float = 0.8,
    priority_weights: dict[str, float] | None = None,
    mass_baseline: float = 0.0,
) -> DiagnosticSnapshot:
    """Build percentile maps without using any full-field reference error."""
    channels, scalar_metrics = compute_reference_free_pointwise_diagnostics(
        model,
        benchmark,
        batch,
        variable_awareness=variable_awareness,
        interface_tau=interface_tau,
        interface_beta=interface_focus_strength,
        interface_threshold=interface_threshold,
        mass_baseline=mass_baseline,
    )
    names = list(channels)
    raw = np.vstack(
        [
            aggregate_by_patch(values, coordinates, patch_grid, percentile)
            for values, coordinates in channels.values()
        ]
    )
    normalized = np.vstack([_normalize(row) for row in raw])
    configured_weights = dict(priority_weights or {})
    priority = normalized.copy()
    for index, name in enumerate(names):
        priority[index] *= max(0.0, float(configured_weights.get(name, 1.0)))
    adaptation_names = list(names)
    return DiagnosticSnapshot(
        names,
        raw,
        normalized,
        priority,
        adaptation_names,
        scalar_metrics,
    )


def compute_reference_free_pointwise_diagnostics(
    model: nn.Module,
    benchmark: CahnHilliardBenchmark,
    batch: dict[str, torch.Tensor],
    *,
    variable_awareness: bool,
    interface_tau: float,
    interface_beta: float,
    interface_threshold: float,
    mass_baseline: float,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, float]]:
    """Return prediction/training-signal diagnostics without full-field truth.

    Exact values are accessed only at prescribed boundary, initial, and shared
    sparse-supervision points already present in ``batch``.
    """
    interior = batch["interior"].detach().clone().requires_grad_(True)
    residuals = compute_cahn_hilliard_residuals(
        model, interior, benchmark, batch.get("forcing")
    )
    prediction = model(interior)
    u = prediction[:, :1]
    grad_u = gradient(u, interior)
    predicted_proxy = torch.exp(-u.detach().abs() / max(interface_tau, 1e-6)).squeeze(1)
    predicted_gradient = torch.sqrt(
        grad_u[:, 0].detach().square() + grad_u[:, 1].detach().square() + 1e-20
    )
    predicted_mask = (u.detach().abs() < float(interface_threshold)).float().squeeze(1)
    focus = 1.0 + float(interface_beta) * predicted_proxy
    phase_range = (
        torch.relu(u.detach().squeeze(1) - 1.0).square()
        + torch.relu(-1.0 - u.detach().squeeze(1)).square()
    )
    mass_value = float(u.detach().mean().cpu())
    mass_violation_value = (mass_value - float(mass_baseline)) ** 2
    mass_violation = torch.full_like(predicted_proxy, mass_violation_value)

    channels: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if variable_awareness:
        channels["ch_residual"] = (
            residuals["r_ch"].detach().abs().squeeze(1) * focus,
            interior,
        )
        # Mu remains observable but does not receive interface amplification.
        channels["chemical_potential_residual"] = (
            residuals["r_mu"].detach().abs().squeeze(1),
            interior,
        )
    channels["pde_residual"] = (
        residuals["pde_residual"].detach().squeeze(1),
        interior,
    )

    with torch.no_grad():
        boundary = batch["boundary"]
        boundary_prediction = model(boundary)
        boundary_error = (boundary_prediction - batch["boundary_target"]).abs()
        boundary_proxy = torch.exp(
            -boundary_prediction[:, 0].abs() / max(interface_tau, 1e-6)
        )
        boundary_focus = 1.0 + float(interface_beta) * boundary_proxy
        initial = batch["initial"]
        initial_prediction = model(initial)
        initial_error = (initial_prediction - batch["initial_target"]).abs()
        initial_proxy = torch.exp(
            -initial_prediction[:, 0].abs() / max(interface_tau, 1e-6)
        )
        initial_focus = 1.0 + float(interface_beta) * initial_proxy
        if variable_awareness:
            channels["bc_u_violation"] = (
                boundary_error[:, 0] * boundary_focus,
                boundary,
            )
            channels["bc_mu_violation"] = (boundary_error[:, 1], boundary)
            channels["ic_u_violation"] = (
                initial_error[:, 0] * initial_focus,
                initial,
            )
            channels["ic_mu_violation"] = (initial_error[:, 1], initial)
        else:
            channels["boundary_violation"] = (
                boundary_error.mean(dim=1),
                boundary,
            )
            channels["initial_condition_violation"] = (
                initial_error.mean(dim=1),
                initial,
            )
        sparse = batch.get("sparse")
        sparse_target = batch.get("sparse_target")
        if sparse is not None and sparse_target is not None and sparse.numel():
            sparse_prediction = model(sparse)
            mismatch = (sparse_prediction - sparse_target).abs()
            sparse_proxy = torch.exp(
                -sparse_prediction[:, 0].abs() / max(interface_tau, 1e-6)
            )
            sparse_focus = 1.0 + float(interface_beta) * sparse_proxy
            if variable_awareness:
                channels["sparse_u_mismatch"] = (
                    mismatch[:, 0] * sparse_focus,
                    sparse,
                )
                channels["sparse_mu_mismatch"] = (mismatch[:, 1], sparse)
            else:
                channels["sparse_data_mismatch"] = (mismatch.mean(dim=1), sparse)
        else:
            sparse_prediction = None
            mismatch = None

    channels["predicted_interface_proxy"] = (predicted_proxy, interior)
    channels["predicted_interface_mask"] = (predicted_mask, interior)
    channels["predicted_gradient_norm"] = (predicted_gradient, interior)
    channels["phase_range_violation"] = (phase_range, interior)
    channels["mass_proxy_violation"] = (mass_violation, interior)
    scalar_metrics = {
        "ch_residual_mean": float(residuals["r_ch"].detach().abs().mean().cpu()),
        "mu_residual_mean": float(residuals["r_mu"].detach().abs().mean().cpu()),
        "pde_residual_mean": float(residuals["pde_residual"].detach().mean().cpu()),
        "bc_u_violation": float(boundary_error[:, 0].square().mean().cpu()),
        "bc_mu_violation": float(boundary_error[:, 1].square().mean().cpu()),
        "ic_u_violation": float(initial_error[:, 0].square().mean().cpu()),
        "ic_mu_violation": float(initial_error[:, 1].square().mean().cpu()),
        "phase_range_violation": float(phase_range.mean().cpu()),
        "mass_proxy_value": mass_value,
        "mass_proxy_violation": float(mass_violation_value),
        "interface_proxy_mean": float(predicted_proxy.mean().cpu()),
    }
    if mismatch is not None:
        scalar_metrics["sparse_u_mse"] = float(mismatch[:, 0].square().mean().cpu())
        scalar_metrics["sparse_mu_mse"] = float(mismatch[:, 1].square().mean().cpu())
    else:
        scalar_metrics["sparse_u_mse"] = 0.0
        scalar_metrics["sparse_mu_mse"] = 0.0
    return channels, scalar_metrics


def aggregate_by_patch(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    patch_grid: CahnHilliardPatchGrid,
    percentile: float,
) -> np.ndarray:
    value_array = values.detach().cpu().numpy().reshape(-1)
    patch_ids = patch_grid.assign_numpy(coordinates.detach().cpu().numpy())
    scores = np.zeros(patch_grid.num_patches, dtype=float)
    for patch_id in range(patch_grid.num_patches):
        selected = value_array[patch_ids == patch_id]
        if selected.size:
            scores[patch_id] = float(np.percentile(selected, percentile))
    return scores


def _normalize(values: np.ndarray) -> np.ndarray:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if not positive.size:
        return np.zeros_like(values)
    scale = max(float(np.median(positive)), 1e-12)
    return np.nan_to_num(values / scale, nan=0.0, posinf=1e6, neginf=0.0)
