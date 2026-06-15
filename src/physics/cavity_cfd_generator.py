"""Deterministic finite-difference CFD references for lid-driven cavities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_cavity_cfd_reference(
    reynolds: float,
    output_path: str | Path,
    *,
    resolution: int = 65,
    max_steps: int = 12_000,
    pressure_iterations: int = 40,
    tolerance: float = 2e-7,
    lid_velocity: float = 1.0,
) -> Path:
    """Generate and cache a deterministic steady cavity velocity field."""
    reynolds = float(reynolds)
    resolution = int(resolution)
    if not np.isfinite(reynolds) or reynolds <= 0.0:
        raise ValueError(f"Reynolds number must be positive and finite, got {reynolds!r}.")
    if resolution < 17:
        raise ValueError("CFD reference resolution must be at least 17.")
    if max_steps < 1 or pressure_iterations < 1:
        raise ValueError("CFD reference iteration counts must be positive.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dx = 1.0 / (resolution - 1)
    dy = dx
    viscosity = lid_velocity / reynolds
    convective_dt = 0.02 * dx / max(lid_velocity, 1e-12)
    diffusive_dt = 0.10 * dx * dx / max(viscosity, 1e-12)
    dt = min(convective_dt, diffusive_dt)

    u = np.zeros((resolution, resolution), dtype=np.float64)
    v = np.zeros_like(u)
    pressure = np.zeros_like(u)
    u[-1, 1:-1] = lid_velocity
    previous_u = u.copy()
    previous_v = v.copy()
    converged = False
    completed_steps = max_steps

    for step in range(1, max_steps + 1):
        old_u = u.copy()
        old_v = v.copy()
        source = _pressure_source(old_u, old_v, dx, dy, dt)
        pressure = _solve_pressure_poisson(
            pressure,
            source,
            dx,
            dy,
            pressure_iterations,
        )
        u[1:-1, 1:-1] = (
            old_u[1:-1, 1:-1]
            - old_u[1:-1, 1:-1]
            * dt
            / dx
            * (old_u[1:-1, 1:-1] - old_u[1:-1, :-2])
            - old_v[1:-1, 1:-1]
            * dt
            / dy
            * (old_u[1:-1, 1:-1] - old_u[:-2, 1:-1])
            - dt
            / (2.0 * dx)
            * (pressure[1:-1, 2:] - pressure[1:-1, :-2])
            + viscosity
            * dt
            * (
                (old_u[1:-1, 2:] - 2.0 * old_u[1:-1, 1:-1] + old_u[1:-1, :-2])
                / (dx * dx)
                + (old_u[2:, 1:-1] - 2.0 * old_u[1:-1, 1:-1] + old_u[:-2, 1:-1])
                / (dy * dy)
            )
        )
        v[1:-1, 1:-1] = (
            old_v[1:-1, 1:-1]
            - old_u[1:-1, 1:-1]
            * dt
            / dx
            * (old_v[1:-1, 1:-1] - old_v[1:-1, :-2])
            - old_v[1:-1, 1:-1]
            * dt
            / dy
            * (old_v[1:-1, 1:-1] - old_v[:-2, 1:-1])
            - dt
            / (2.0 * dy)
            * (pressure[2:, 1:-1] - pressure[:-2, 1:-1])
            + viscosity
            * dt
            * (
                (old_v[1:-1, 2:] - 2.0 * old_v[1:-1, 1:-1] + old_v[1:-1, :-2])
                / (dx * dx)
                + (old_v[2:, 1:-1] - 2.0 * old_v[1:-1, 1:-1] + old_v[:-2, 1:-1])
                / (dy * dy)
            )
        )
        _apply_velocity_boundaries(u, v, lid_velocity)
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise RuntimeError(
                f"CFD reference generation became non-finite at Re={reynolds:g}, step={step}."
            )

        if step % 100 == 0:
            delta = np.sqrt(
                np.mean((u - previous_u) ** 2)
                + np.mean((v - previous_v) ** 2)
            )
            if delta <= tolerance:
                converged = True
                completed_steps = step
                break
            previous_u = u.copy()
            previous_v = v.copy()

    axis = np.linspace(0.0, 1.0, resolution)
    x, y = np.meshgrid(axis, axis)
    pressure -= np.mean(pressure)
    omega = np.gradient(v, axis, axis=1, edge_order=1) - np.gradient(
        u, axis, axis=0, edge_order=1
    )
    np.savez_compressed(
        output,
        x=x.reshape(-1),
        y=y.reshape(-1),
        u=u.reshape(-1),
        v=v.reshape(-1),
        p=pressure.reshape(-1),
        omega=omega.reshape(-1),
        reynolds=np.asarray(reynolds),
        resolution=np.asarray(resolution),
        generator=np.asarray("deterministic_pressure_projection"),
        generator_steps=np.asarray(completed_steps),
        generator_converged=np.asarray(converged),
        generator_dt=np.asarray(dt),
    )
    return output.resolve()


def _pressure_source(
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
) -> np.ndarray:
    source = np.zeros_like(u)
    du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dx)
    dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dy)
    du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dy)
    dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dx)
    source[1:-1, 1:-1] = (
        (du_dx + dv_dy) / dt
        - du_dx * du_dx
        - 2.0 * du_dy * dv_dx
        - dv_dy * dv_dy
    )
    return source


def _solve_pressure_poisson(
    pressure: np.ndarray,
    source: np.ndarray,
    dx: float,
    dy: float,
    iterations: int,
) -> np.ndarray:
    denominator = 2.0 * (dx * dx + dy * dy)
    for _ in range(iterations):
        old = pressure.copy()
        pressure[1:-1, 1:-1] = (
            (
                (old[1:-1, 2:] + old[1:-1, :-2]) * dy * dy
                + (old[2:, 1:-1] + old[:-2, 1:-1]) * dx * dx
            )
            / denominator
            - source[1:-1, 1:-1] * dx * dx * dy * dy / denominator
        )
        pressure[:, -1] = pressure[:, -2]
        pressure[:, 0] = pressure[:, 1]
        pressure[0, :] = pressure[1, :]
        pressure[-1, :] = 0.0
    return pressure


def _apply_velocity_boundaries(
    u: np.ndarray,
    v: np.ndarray,
    lid_velocity: float,
) -> None:
    u[0, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    u[-1, :] = 0.0
    u[-1, 1:-1] = lid_velocity
    v[0, :] = 0.0
    v[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reynolds", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=65)
    parser.add_argument("--max_steps", type=int, default=12_000)
    parser.add_argument("--pressure_iterations", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=2e-7)
    parser.add_argument("--lid_velocity", type=float, default=1.0)
    args = parser.parse_args()
    path = generate_cavity_cfd_reference(
        args.reynolds,
        args.output,
        resolution=args.resolution,
        max_steps=args.max_steps,
        pressure_iterations=args.pressure_iterations,
        tolerance=args.tolerance,
        lid_velocity=args.lid_velocity,
    )
    print(path)


if __name__ == "__main__":
    main()
