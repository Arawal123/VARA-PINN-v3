"""Paper-oriented heatmap plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def save_heatmap(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    path: str | Path,
    title: str,
    cmap: str = "magma",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    im = ax.pcolormesh(x, y, values, shading="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)
