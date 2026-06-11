"""Pointwise diagnostic map builder."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.physics.kovasznay import center_pressure
from src.physics.navier_stokes import navier_stokes_residuals
from src.physics.vorticity import vorticity_transport_residual


class DiagnosticMapBuilder:
    """Build pointwise diagnostic arrays for VARA diagnosis."""

    def __init__(
        self,
        model: torch.nn.Module,
        benchmark: Any,
        device: torch.device,
        steady: bool = True,
        residual_interior_only: bool = False,
    ) -> None:
        self.model = model
        self.benchmark = benchmark
        self.device = device
        self.steady = steady
        self.residual_interior_only = bool(residual_interior_only)

    def build(
        self,
        coords_np: np.ndarray,
        mode: str = "full_reference",
        boundary_coords_np: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute diagnostics in full_reference, sparse_data, or residual_only mode."""
        mode = mode.lower()
        coords = torch.tensor(coords_np, dtype=torch.float32, device=self.device)
        self.model.eval()
        residuals = navier_stokes_residuals(self.model, coords, nu=self.benchmark.nu, steady=self.steady)
        pred = torch.cat(
            [residuals["u"], residuals["v"], residuals["p"]],
            dim=1,
        )

        u_pred = pred[:, 0:1].detach().cpu().numpy()
        v_pred = pred[:, 1:2].detach().cpu().numpy()
        p_pred = pred[:, 2:3].detach().cpu().numpy()
        omega_pred = residuals["omega"].detach().cpu().numpy()
        p_x = residuals["p_x"].detach().cpu().numpy()
        p_y = residuals["p_y"].detach().cpu().numpy()
        speed_pred = np.sqrt(u_pred * u_pred + v_pred * v_pred)

        maps: dict[str, np.ndarray] = {
            "u_pred": u_pred,
            "v_pred": v_pred,
            "p_pred": center_pressure(p_pred),
            "omega_pred": omega_pred,
            "omega_abs": np.abs(omega_pred),
            "speed_pred": speed_pred,
            "pressure_gradient_norm": np.sqrt(p_x * p_x + p_y * p_y),
            "continuity_residual": np.abs(residuals["f_c"].detach().cpu().numpy()),
            "momentum_u_residual": np.abs(residuals["f_u"].detach().cpu().numpy()),
            "momentum_v_residual": np.abs(residuals["f_v"].detach().cpu().numpy()),
            "pde_residual": residuals["pde_residual"].detach().cpu().numpy(),
        }
        if self.residual_interior_only and hasattr(self.benchmark, "boundary_mask_np"):
            boundary_mask = np.asarray(self.benchmark.boundary_mask_np(coords_np), dtype=bool)
            for name in (
                "continuity_residual",
                "momentum_u_residual",
                "momentum_v_residual",
                "pde_residual",
            ):
                values = maps[name].copy()
                values[boundary_mask] = np.nan
                maps[name] = values

        if not self.steady:
            vt = vorticity_transport_residual(self.model, coords, nu=self.benchmark.nu, steady=False)
            maps["vorticity_transport_residual"] = np.abs(vt.detach().cpu().numpy())

        has_reference = bool(getattr(self.benchmark, "has_reference", True))
        if mode in {"full_reference", "sparse_data"} and has_reference:
            ref = self.benchmark.exact_np(coords_np)
            p_ref_c = center_pressure(ref["p"])
            p_pred_c = center_pressure(p_pred)
            p_grad_err = np.sqrt((p_x - ref.get("p_x", 0.0)) ** 2 + (p_y - ref.get("p_y", 0.0)) ** 2)
            maps.update(
                {
                    "u_ref": ref["u"],
                    "v_ref": ref["v"],
                    "p_ref": p_ref_c,
                    "omega_ref": ref["omega"],
                    "speed_ref": ref.get("speed", np.sqrt(ref["u"] ** 2 + ref["v"] ** 2)),
                    "u_error": np.abs(u_pred - ref["u"]),
                    "v_error": np.abs(v_pred - ref["v"]),
                    "p_error_mean_centered": np.abs(p_pred_c - p_ref_c),
                    "pressure_gradient_error": np.abs(p_grad_err),
                    "speed_error": np.abs(speed_pred - ref.get("speed", np.sqrt(ref["u"] ** 2 + ref["v"] ** 2))),
                    "omega_error": np.abs(omega_pred - ref["omega"]),
                }
            )
        else:
            zeros = np.zeros_like(u_pred)
            maps.update(
                {
                    "u_error": zeros,
                    "v_error": zeros,
                    "p_error_mean_centered": zeros,
                    "pressure_gradient_error": zeros,
                    "speed_error": zeros,
                    "omega_error": zeros,
                }
            )

        maps["aggregate_pde_residual"] = maps["pde_residual"]
        maps["boundary_violation"] = self._boundary_violation(coords_np, boundary_coords_np)
        if self._is_lid_driven_cavity():
            centerline_weight = self._centerline_weight(coords_np)
            maps["centerline_pde_residual"] = maps["pde_residual"] * centerline_weight
            maps["centerline_continuity_residual"] = maps["continuity_residual"] * centerline_weight
            corner_weight = self._corner_weight(coords_np)
            maps["corner_pde_residual"] = maps["pde_residual"] * corner_weight
            maps["corner_boundary_error"] = maps["boundary_violation"] * corner_weight
        return maps

    def _is_lid_driven_cavity(self) -> bool:
        return self.benchmark.__class__.__name__.lower() == "liddrivencavityqualitative"

    def _centerline_weight(self, coords_np: np.ndarray) -> np.ndarray:
        x0, x1, y0, y1 = self.benchmark.bounds
        x_mid = 0.5 * (x0 + x1)
        y_mid = 0.5 * (y0 + y1)
        sigma_x = max((x1 - x0) / 10.0, 1e-8)
        sigma_y = max((y1 - y0) / 10.0, 1e-8)
        wx = np.exp(-((coords_np[:, 0:1] - x_mid) / sigma_x) ** 2)
        wy = np.exp(-((coords_np[:, 1:2] - y_mid) / sigma_y) ** 2)
        return np.maximum(wx, wy)

    def _corner_weight(self, coords_np: np.ndarray) -> np.ndarray:
        x0, x1, y0, y1 = self.benchmark.bounds
        width = max(float(x1 - x0), 1e-12)
        height = max(float(y1 - y0), 1e-12)
        corner_width = 0.12 * min(width, height)
        left = coords_np[:, 0:1] <= x0 + corner_width
        right = coords_np[:, 0:1] >= x1 - corner_width
        bottom = coords_np[:, 1:2] <= y0 + corner_width
        top = coords_np[:, 1:2] >= y1 - corner_width
        return ((left | right) & (bottom | top)).astype(float)

    def _boundary_violation(self, coords_np: np.ndarray, boundary_coords_np: np.ndarray | None) -> np.ndarray:
        x0, x1, y0, y1 = self.benchmark.bounds
        tol = 1e-6
        boundary_mask = (
            np.isclose(coords_np[:, 0], x0, atol=tol)
            | np.isclose(coords_np[:, 0], x1, atol=tol)
            | np.isclose(coords_np[:, 1], y0, atol=tol)
            | np.isclose(coords_np[:, 1], y1, atol=tol)
        )
        out = np.zeros((coords_np.shape[0], 1), dtype=float)
        if not np.any(boundary_mask):
            return out
        coords = torch.tensor(coords_np[boundary_mask], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.model(coords)
            ref = self.benchmark.exact_torch(coords)
            viol = torch.sqrt((pred[:, 0:1] - ref["u"]).pow(2) + (pred[:, 1:2] - ref["v"]).pow(2))
        out[boundary_mask] = viol.cpu().numpy()
        return out
