"""Streamline plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def save_streamlines(X: np.ndarray, Y: np.ndarray, U: np.ndarray, V: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    speed = np.sqrt(U * U + V * V)
    _psi, consistency = reconstruct_streamfunction(X, Y, U, V)
    ax.streamplot(X, Y, U, V, color=speed, cmap="plasma", density=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(X)), float(np.max(X)))
    ax.set_ylim(float(np.min(Y)), float(np.max(Y)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Velocity streamlines\nstreamfunction consistency RMSE={consistency:.3e}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def reconstruct_streamfunction(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Reconstruct psi from u=psi_y and v=-psi_x in two independent ways."""
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
    consistency = float(np.sqrt(np.mean((psi_from_u - psi_from_v) ** 2)))
    return 0.5 * (psi_from_u + psi_from_v), consistency


def save_streamfunction_contours(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    psi, consistency = reconstruct_streamfunction(X, Y, U, V)
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    levels = np.linspace(float(np.min(psi)), float(np.max(psi)), 31)
    if np.allclose(levels[0], levels[-1]):
        levels = 31
    contours = ax.contour(X, Y, psi, levels=levels, cmap="plasma", linewidths=1.0)
    ax.clabel(contours, inline=True, fontsize=6, fmt="%.3f")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Reconstructed streamfunction\nconsistency RMSE={consistency:.3e}")
    fig.savefig(path, dpi=200)
    plt.close(fig)
