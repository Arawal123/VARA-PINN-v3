"""Streamline plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def save_streamlines(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    path: str | Path,
    *,
    closed_boundary: bool = False,
    annotate_vortices: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5.2), constrained_layout=True)
    speed = np.sqrt(U * U + V * V)
    psi, consistency = reconstruct_streamfunction(
        X,
        Y,
        U,
        V,
        closed_boundary=closed_boundary,
    )
    vortices = detect_vortices(X, Y, psi) if closed_boundary else []
    linewidth = 0.7 + 1.3 * speed / max(float(np.nanmax(speed)), 1e-12)
    ax.streamplot(
        X,
        Y,
        U,
        V,
        color=speed,
        linewidth=linewidth,
        cmap="plasma",
        density=1.55,
        integration_direction="both",
        broken_streamlines=True,
    )
    if annotate_vortices:
        for index, vortex in enumerate(vortices):
            marker = "*" if index == 0 else "o"
            ax.scatter(
                vortex["x"],
                vortex["y"],
                marker=marker,
                s=65 if index == 0 else 28,
                facecolors="none",
                edgecolors="black",
                linewidths=1.0,
                zorder=5,
            )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(X)), float(np.max(X)))
    ax.set_ylim(float(np.min(Y)), float(np.max(Y)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    topology = f", detected vortices={len(vortices)}" if closed_boundary else ""
    ax.set_title(
        "Predicted velocity streamlines\n"
        f"closed-path consistency RMSE={consistency:.3e}{topology}"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def reconstruct_streamfunction(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    *,
    closed_boundary: bool = False,
) -> tuple[np.ndarray, float]:
    """Reconstruct psi from u=psi_y and v=-psi_x.

    Generic fields use independent bottom and left integration paths. Closed
    no-penetration domains additionally use the top and right walls, reducing
    directional integration bias and making the reported disagreement a
    nonlocal incompressibility diagnostic.
    """
    x = np.asarray(X[0, :], dtype=float)
    y = np.asarray(Y[:, 0], dtype=float)
    u = np.asarray(U, dtype=float)
    v = np.asarray(V, dtype=float)
    psi_from_u = np.zeros_like(u)
    psi_from_v = np.zeros_like(v)
    if len(y) > 1:
        dy = np.diff(y)[:, None]
        psi_from_u[1:, :] = np.cumsum(0.5 * (u[1:, :] + u[:-1, :]) * dy, axis=0)
    if len(x) > 1:
        dx = np.diff(x)[None, :]
        psi_from_v[:, 1:] = np.cumsum(-0.5 * (v[:, 1:] + v[:, :-1]) * dx, axis=1)
    psi_from_u -= float(psi_from_u[0, 0])
    psi_from_v -= float(psi_from_v[0, 0])
    paths = [psi_from_u, psi_from_v]
    if closed_boundary:
        psi_from_top = np.zeros_like(u)
        psi_from_right = np.zeros_like(v)
        if len(y) > 1:
            dy = np.diff(y)[:, None]
            increments = 0.5 * (u[1:, :] + u[:-1, :]) * dy
            psi_from_top[:-1, :] = -np.flip(
                np.cumsum(np.flip(increments, axis=0), axis=0),
                axis=0,
            )
        if len(x) > 1:
            dx = np.diff(x)[None, :]
            increments = 0.5 * (v[:, 1:] + v[:, :-1]) * dx
            psi_from_right[:, :-1] = np.flip(
                np.cumsum(np.flip(increments, axis=1), axis=1),
                axis=1,
            )
        paths.extend([psi_from_top, psi_from_right])
    stacked = np.stack(paths, axis=0)
    psi = np.mean(stacked, axis=0)
    consistency = float(np.sqrt(np.mean((stacked - psi[None, :, :]) ** 2)))
    return psi, consistency


def detect_vortices(
    X: np.ndarray,
    Y: np.ndarray,
    psi: np.ndarray,
    *,
    minimum_strength_fraction: float = 0.02,
    minimum_prominence_fraction: float = 0.002,
    separation_fraction: float = 0.06,
) -> list[dict[str, float]]:
    """Detect robust streamfunction extrema without assuming expected vortices."""
    values = _smooth_scalar_field(np.asarray(psi, dtype=float))
    if min(values.shape) < 5 or not np.isfinite(values).any():
        return []
    interior = values[1:-1, 1:-1]
    neighbors = [
        values[dy : dy + interior.shape[0], dx : dx + interior.shape[1]]
        for dy in range(3)
        for dx in range(3)
        if not (dy == 1 and dx == 1)
    ]
    local_min = np.logical_and.reduce([interior < neighbor for neighbor in neighbors])
    local_max = np.logical_and.reduce([interior > neighbor for neighbor in neighbors])
    indices = np.argwhere(local_min | local_max) + 1
    global_strength = max(float(np.nanmax(np.abs(values))), 1e-12)
    global_range = max(
        float(np.nanmax(values) - np.nanmin(values)),
        1e-12,
    )
    prominence_radius = max(2, int(round(0.025 * min(values.shape))))
    candidates: list[dict[str, float]] = []
    for iy, ix in indices:
        value = float(values[iy, ix])
        if abs(value) < minimum_strength_fraction * global_strength:
            continue
        y0 = max(0, iy - prominence_radius)
        y1 = min(values.shape[0], iy + prominence_radius + 1)
        x0 = max(0, ix - prominence_radius)
        x1 = min(values.shape[1], ix + prominence_radius + 1)
        neighborhood = values[y0:y1, x0:x1]
        ring = np.concatenate(
            [
                neighborhood[0, :],
                neighborhood[-1, :],
                neighborhood[1:-1, 0],
                neighborhood[1:-1, -1],
            ]
        )
        ring_level = float(np.nanmedian(ring))
        prominence = (
            ring_level - value
            if local_min[iy - 1, ix - 1]
            else value - ring_level
        )
        if prominence < minimum_prominence_fraction * global_range:
            continue
        candidates.append(
            {
                "x": float(X[iy, ix]),
                "y": float(Y[iy, ix]),
                "psi": value,
                "strength": abs(value),
                "prominence": float(prominence),
                "rotation": -1.0 if value < 0.0 else 1.0,
            }
        )
    candidates.sort(key=lambda item: item["strength"], reverse=True)
    x_span = max(float(np.max(X) - np.min(X)), 1e-12)
    y_span = max(float(np.max(Y) - np.min(Y)), 1e-12)
    kept: list[dict[str, float]] = []
    for candidate in candidates:
        if any(
            np.hypot(
                (candidate["x"] - existing["x"]) / x_span,
                (candidate["y"] - existing["y"]) / y_span,
            )
            < separation_fraction
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _smooth_scalar_field(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, 1, mode="edge")
    total = np.zeros_like(values, dtype=float)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
    return total / 9.0


def save_streamfunction_contours(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    path: str | Path,
    *,
    closed_boundary: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    psi, consistency = reconstruct_streamfunction(
        X,
        Y,
        U,
        V,
        closed_boundary=closed_boundary,
    )
    vortices = detect_vortices(X, Y, psi) if closed_boundary else []
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    levels = np.linspace(float(np.min(psi)), float(np.max(psi)), 31)
    if np.allclose(levels[0], levels[-1]):
        levels = 31
    contours = ax.contour(X, Y, psi, levels=levels, cmap="plasma", linewidths=1.0)
    ax.clabel(contours, inline=True, fontsize=6, fmt="%.3f")
    for index, vortex in enumerate(vortices):
        ax.scatter(
            vortex["x"],
            vortex["y"],
            marker="*" if index == 0 else "o",
            s=70 if index == 0 else 32,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "Reconstructed streamfunction\n"
        f"consistency RMSE={consistency:.3e}, detected vortices={len(vortices)}"
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)
