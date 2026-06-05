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
    ax.streamplot(X, Y, U, V, color=speed, cmap="plasma", density=1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Streamlines")
    fig.savefig(path, dpi=180)
    plt.close(fig)
