"""Field comparison plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def save_field_panel(
    X: np.ndarray,
    Y: np.ndarray,
    fields: dict[str, np.ndarray],
    path: str | Path,
    cmap: str = "viridis",
) -> None:
    """Save a compact multi-field panel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (name, values) in zip(axes, fields.items()):
        im = ax.pcolormesh(X, Y, values, shading="auto", cmap=cmap)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)
