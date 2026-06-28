"""Three-dimensional patch diagnostics built from reference-free signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .benchmarks import Burgers2DBenchmark, ManufacturedBenchmark
from .residuals import compute_residuals


@dataclass(frozen=True)
class PDEPatch:
    """One axis-aligned x-y-t patch."""

    patch_id: int
    ix: int
    iy: int
    it: int
    bounds: tuple[float, float, float, float, float, float]


class PDEPatchGrid:
    """Regular spatiotemporal patch grid used only by this suite."""

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
            raise ValueError("Patch counts must be positive.")
        self.num_patches = self.nx * self.ny * self.nt
        self._patches = self._build()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        benchmark: ManufacturedBenchmark,
    ) -> "PDEPatchGrid":
        cfg = dict(config.get("patches", {}))
        return cls(
            benchmark.bounds,
            benchmark.t_bounds,
            int(cfg.get("nx_patches", 4)),
            int(cfg.get("ny_patches", 4)),
            int(cfg.get("nt_patches", 3)),
        )

    def get_patch(self, patch_id: int) -> PDEPatch:
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

    def _build(self) -> list[PDEPatch]:
        x0, x1, y0, y1 = self.bounds
        t0, t1 = self.t_bounds
        xs = np.linspace(x0, x1, self.nx + 1)
        ys = np.linspace(y0, y1, self.ny + 1)
        ts = np.linspace(t0, t1, self.nt + 1)
        patches: list[PDEPatch] = []
        for it in range(self.nt):
            for iy in range(self.ny):
                for ix in range(self.nx):
                    pid = ix + self.nx * (iy + self.ny * it)
                    patches.append(
                        PDEPatch(
                            pid,
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
    """Patch-aggregated diagnostic scores for one controller decision."""

    names: list[str]
    raw_scores: np.ndarray
    normalized_scores: np.ndarray


def build_diagnostic_snapshot(
    model: nn.Module,
    benchmark: ManufacturedBenchmark,
    patch_grid: PDEPatchGrid,
    diagnostic_batch: dict[str, torch.Tensor],
    *,
    percentile: float = 90.0,
    variable_awareness: bool = True,
) -> DiagnosticSnapshot:
    """Aggregate residual, prescribed-condition, and sparse-training signals."""
    channels: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    interior = diagnostic_batch["interior"].detach().clone().requires_grad_(True)
    residuals = compute_residuals(model, interior, benchmark)
    if isinstance(benchmark, Burgers2DBenchmark) and variable_awareness:
        channels.extend(
            [
                ("momentum_u_residual", residuals["f_u"].detach().abs().squeeze(1), interior),
                ("momentum_v_residual", residuals["f_v"].detach().abs().squeeze(1), interior),
            ]
        )
    else:
        channels.append(
            ("pde_residual", residuals["pde_residual"].detach().squeeze(1), interior)
        )

    with torch.no_grad():
        boundary = diagnostic_batch["boundary"]
        boundary_values = (model(boundary) - diagnostic_batch["boundary_target"]).abs()
        channels.append(("boundary_mismatch", boundary_values.mean(dim=1), boundary))

        initial = diagnostic_batch["initial"]
        initial_values = (model(initial) - diagnostic_batch["initial_target"]).abs()
        channels.append(("initial_condition_mismatch", initial_values.mean(dim=1), initial))

        sparse = diagnostic_batch.get("sparse")
        sparse_target = diagnostic_batch.get("sparse_target")
        if sparse is not None and sparse_target is not None and sparse.numel() > 0:
            mismatch = (model(sparse) - sparse_target).abs()
            if variable_awareness:
                for index, output_name in enumerate(benchmark.output_names):
                    channels.append((f"sparse_{output_name}_mismatch", mismatch[:, index], sparse))
            else:
                channels.append(("sparse_data_mismatch", mismatch.mean(dim=1), sparse))

    names = [name for name, _, _ in channels]
    raw = np.vstack(
        [
            aggregate_by_patch(values, coords, patch_grid, percentile)
            for _, values, coords in channels
        ]
    )
    normalized = np.vstack([_normalize_scores(row) for row in raw])
    return DiagnosticSnapshot(names, raw, normalized)


def aggregate_by_patch(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    patch_grid: PDEPatchGrid,
    percentile: float,
) -> np.ndarray:
    """Return a robust percentile for each patch, with zero for empty patches."""
    value_array = values.detach().cpu().numpy().reshape(-1)
    patch_ids = patch_grid.assign_numpy(coordinates.detach().cpu().numpy())
    result = np.zeros(patch_grid.num_patches, dtype=float)
    for patch_id in range(patch_grid.num_patches):
        selected = value_array[patch_ids == patch_id]
        if selected.size:
            result[patch_id] = float(np.percentile(selected, percentile))
    return result


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    finite_positive = values[np.isfinite(values) & (values > 0.0)]
    if not finite_positive.size:
        return np.zeros_like(values)
    scale = float(np.median(finite_positive))
    return np.nan_to_num(values / max(scale, 1e-12), nan=0.0, posinf=1e6, neginf=0.0)
