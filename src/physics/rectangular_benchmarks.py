"""Rectangular benchmark definitions for professor-facing PINN comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
import torch

from src.physics.cavity_reference import (
    interpolate_full_field,
    load_full_field_reference,
    load_lid_cavity_profile_reference,
)


@dataclass(frozen=True)
class RectangularBenchmarkBase:
    """Common rectangular benchmark interface used by the existing trainer."""

    reynolds: float = 40.0
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    amplitude: float = 1.0
    reference_kind: str = "manufactured"
    has_reference: bool = True

    @property
    def nu(self) -> float:
        return 1.0 / self.reynolds

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def grid(self, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(self.x_min, self.x_max, nx)
        y = np.linspace(self.y_min, self.y_max, ny)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        return X, Y, xy

    def boundary_mask_np(self, xy: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        return (
            np.isclose(xy[:, 0], self.x_min, atol=tol)
            | np.isclose(xy[:, 0], self.x_max, atol=tol)
            | np.isclose(xy[:, 1], self.y_min, atol=tol)
            | np.isclose(xy[:, 1], self.y_max, atol=tol)
        )

    def exact_np(self, xy: np.ndarray) -> dict[str, np.ndarray]:
        x = torch.tensor(xy[:, 0:1], dtype=torch.float64)
        y = torch.tensor(xy[:, 1:2], dtype=torch.float64)
        return {k: v.detach().cpu().numpy().astype(float) for k, v in self._exact_torch_xy(x, y).items()}

    def exact_torch(self, xy: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._exact_torch_xy(xy[:, 0:1], xy[:, 1:2])

    def _exact_torch_xy(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


@dataclass(frozen=True)
class PoiseuilleChannelFlow(RectangularBenchmarkBase):
    """Analytical pressure-driven channel with inflow/outflow boundaries."""

    reference_kind: str = "analytical"
    has_reference: bool = True

    def _exact_torch_xy(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        eta = (y - self.y_min) / self.height
        u = 4.0 * self.amplitude * eta * (1.0 - eta)
        v = torch.zeros_like(u)
        dpdx = -8.0 * self.nu * self.amplitude / (self.height * self.height)
        p = dpdx * (x - self.x_min)
        omega = -4.0 * self.amplitude * (1.0 - 2.0 * eta) / self.height
        px = torch.full_like(u, dpdx)
        py = torch.zeros_like(u)
        speed = torch.sqrt(u * u + v * v)
        return {"u": u, "v": v, "p": p, "omega": omega, "p_x": px, "p_y": py, "speed": speed}


@dataclass(frozen=True)
class DoubleVortexBoxFlow(RectangularBenchmarkBase):
    """Manufactured two-cell incompressible vortex field."""

    reference_kind: str = "manufactured"
    has_reference: bool = True

    def _exact_torch_xy(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        xi = (x - self.x_min) / self.width
        eta = (y - self.y_min) / self.height
        pi_t = torch.as_tensor(pi, dtype=x.dtype, device=x.device)
        a = self.amplitude
        u = a * torch.sin(2.0 * pi_t * xi).pow(2) * torch.sin(2.0 * pi_t * eta) * pi_t / self.height
        v = -a * torch.sin(4.0 * pi_t * xi) * torch.sin(pi_t * eta).pow(2) * pi_t / self.width
        p = 0.25 * a * (torch.cos(4.0 * pi_t * xi) + torch.cos(2.0 * pi_t * eta))
        px = -a * pi_t * torch.sin(4.0 * pi_t * xi) / self.width
        py = -0.5 * a * pi_t * torch.sin(2.0 * pi_t * eta) / self.height
        dv_dx = -4.0 * a * pi_t * pi_t * torch.cos(4.0 * pi_t * xi) * torch.sin(pi_t * eta).pow(2) / (
            self.width * self.width
        )
        du_dy = 2.0 * a * pi_t * pi_t * torch.sin(2.0 * pi_t * xi).pow(2) * torch.cos(2.0 * pi_t * eta) / (
            self.height * self.height
        )
        omega = dv_dx - du_dy
        speed = torch.sqrt(u * u + v * v)
        return {"u": u, "v": v, "p": p, "omega": omega, "p_x": px, "p_y": py, "speed": speed}


@dataclass(frozen=True)
class BoundaryStressBoxFlow(RectangularBenchmarkBase):
    """Manufactured rectangular case with prescribed u/v boundary variation."""

    reference_kind: str = "manufactured"
    has_reference: bool = True

    def _exact_torch_xy(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        xi = (x - self.x_min) / self.width
        eta = (y - self.y_min) / self.height
        pi_t = torch.as_tensor(pi, dtype=x.dtype, device=x.device)
        a = self.amplitude
        u = a * (1.0 - eta) + 0.25 * a * torch.sin(pi_t * xi) * torch.sin(pi_t * eta)
        v = 0.25 * a * torch.sin(2.0 * pi_t * xi) * eta * (1.0 - eta)
        p = 0.1 * a * (1.0 - xi)
        px = torch.full_like(u, -0.1 * a / self.width)
        py = torch.zeros_like(u)
        dv_dx = 0.5 * a * pi_t * torch.cos(2.0 * pi_t * xi) * eta * (1.0 - eta) / self.width
        du_dy = -a / self.height + 0.25 * a * pi_t * torch.sin(pi_t * xi) * torch.cos(pi_t * eta) / self.height
        omega = dv_dx - du_dy
        speed = torch.sqrt(u * u + v * v)
        return {"u": u, "v": v, "p": p, "omega": omega, "p_x": px, "p_y": py, "speed": speed}


@dataclass(frozen=True)
class LidDrivenCavityQualitative(RectangularBenchmarkBase):
    """Boundary/residual-only lid-driven cavity benchmark without fake interior truth."""

    lid_velocity: float = 1.0
    lid_corner_regularization_width: float = 0.0
    reference: str = "none"
    reference_path: str | None = None
    full_field_reference_path: str | None = None
    profile_only: bool = True
    reference_kind: str = "residual_only"
    has_reference: bool = False

    @property
    def has_profile_reference(self) -> bool:
        return self.reference.lower() != "none" or self.reference_path is not None

    def profile_reference_np(self) -> dict[str, np.ndarray | str]:
        profile = load_lid_cavity_profile_reference(self.reference, self.reynolds, self.reference_path)
        x_mid = 0.5 * (self.x_min + self.x_max)
        y_mid = 0.5 * (self.y_min + self.y_max)
        out: dict[str, np.ndarray | str] = {"source": profile.source}
        if profile.has_u:
            y_norm = profile.u_profile["y"].to_numpy(dtype=float)
            out["u_xy"] = np.column_stack([np.full_like(y_norm, x_mid), self.y_min + y_norm * self.height])
            out["u_ref"] = profile.u_profile["u_ref"].to_numpy(dtype=float).reshape(-1, 1)
        if profile.has_v:
            x_norm = profile.v_profile["x"].to_numpy(dtype=float)
            out["v_xy"] = np.column_stack([self.x_min + x_norm * self.width, np.full_like(x_norm, y_mid)])
            out["v_ref"] = profile.v_profile["v_ref"].to_numpy(dtype=float).reshape(-1, 1)
        return out

    def exact_np(self, xy: np.ndarray) -> dict[str, np.ndarray]:
        if self.has_reference and self.full_field_reference_path is not None:
            return interpolate_full_field(load_full_field_reference(self.full_field_reference_path), xy)
        return super().exact_np(xy)

    def _exact_torch_xy(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        tol = 1e-5
        u = torch.zeros_like(x)
        v = torch.zeros_like(x)
        top = torch.isclose(y, torch.full_like(y, self.y_max), atol=tol, rtol=0.0)
        lid = torch.full_like(u, self.lid_velocity)
        if self.lid_corner_regularization_width > 0.0:
            xi = (x - self.x_min) / max(self.width, 1e-12)
            width = max(float(self.lid_corner_regularization_width), 1e-8)
            left = _smoothstep01(xi / width)
            right = _smoothstep01((1.0 - xi) / width)
            lid = lid * left * right
        u = torch.where(top, lid, u)
        p = torch.zeros_like(x)
        omega = torch.zeros_like(x)
        px = torch.zeros_like(x)
        py = torch.zeros_like(x)
        speed = torch.sqrt(u * u + v * v)
        return {"u": u, "v": v, "p": p, "omega": omega, "p_x": px, "p_y": py, "speed": speed}


def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)
