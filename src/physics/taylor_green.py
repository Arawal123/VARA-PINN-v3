"""Analytical time-dependent Taylor-Green vortex benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
import torch


@dataclass(frozen=True)
class TaylorGreenVortex:
    reynolds: float = 100.0
    x_min: float = 0.0
    x_max: float = 2.0 * pi
    y_min: float = 0.0
    y_max: float = 2.0 * pi
    t_min: float = 0.0
    t_max: float = 1.0
    evaluation_time: float = 1.0
    amplitude: float = 1.0
    reference_kind: str = "analytical"
    has_reference: bool = True

    @property
    def nu(self) -> float:
        return 1.0 / self.reynolds

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x_min, self.x_max, self.y_min, self.y_max

    @property
    def t_bounds(self) -> tuple[float, float]:
        return self.t_min, self.t_max

    def grid(self, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(self.x_min, self.x_max, nx)
        y = np.linspace(self.y_min, self.y_max, ny)
        x_grid, y_grid = np.meshgrid(x, y)
        time = np.full(x_grid.size, float(self.evaluation_time))
        coords = np.column_stack([x_grid.reshape(-1), y_grid.reshape(-1), time])
        return x_grid, y_grid, coords

    def boundary_mask_np(self, coords: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        return (
            np.isclose(coords[:, 0], self.x_min, atol=tol)
            | np.isclose(coords[:, 0], self.x_max, atol=tol)
            | np.isclose(coords[:, 1], self.y_min, atol=tol)
            | np.isclose(coords[:, 1], self.y_max, atol=tol)
            | np.isclose(coords[:, 2], self.t_min, atol=tol)
        )

    def exact_np(self, coords: np.ndarray) -> dict[str, np.ndarray]:
        tensor = torch.tensor(coords, dtype=torch.float64)
        return {
            name: value.detach().cpu().numpy().astype(float)
            for name, value in self.exact_torch(tensor).items()
        }

    def exact_torch(self, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        amplitude = float(self.amplitude)
        decay = torch.exp(-2.0 * self.nu * t)
        u = -amplitude * torch.cos(x) * torch.sin(y) * decay
        v = amplitude * torch.sin(x) * torch.cos(y) * decay
        p = -0.25 * amplitude * amplitude * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * decay * decay
        omega = 2.0 * amplitude * torch.sin(x) * torch.sin(y) * decay
        p_x = 0.5 * amplitude * amplitude * torch.sin(2.0 * x) * decay * decay
        p_y = 0.5 * amplitude * amplitude * torch.sin(2.0 * y) * decay * decay
        speed = torch.sqrt(u * u + v * v)
        return {
            "u": u,
            "v": v,
            "p": p,
            "omega": omega,
            "p_x": p_x,
            "p_y": p_y,
            "speed": speed,
        }
