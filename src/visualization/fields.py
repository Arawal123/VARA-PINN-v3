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


def save_prediction_reference_error_panel(
    X: np.ndarray,
    Y: np.ndarray,
    field_triplets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    path: str | Path,
) -> None:
    """Save prediction/reference/error columns for several fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(field_triplets)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12.6, 3.2 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.asarray([axes])
    for row, (name, (pred, ref, err)) in enumerate(field_triplets.items()):
        for col, (title, values, cmap) in enumerate(
            [
                (f"{name} pred", pred, "viridis"),
                (f"{name} ref", ref, "viridis"),
                (f"{name} abs error", err, "magma"),
            ]
        ):
            ax = axes[row, col]
            im = ax.pcolormesh(X, Y, values, shading="auto", cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)
