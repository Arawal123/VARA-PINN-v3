"""Ablation comparison plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


def save_ablation_bar(table: pd.DataFrame, metric: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(table["mode"], table[metric], color="#3a7d44")
    ax.set_ylabel(metric)
    ax.set_xticklabels(table["mode"], rotation=30, ha="right")
    fig.savefig(path, dpi=180)
    plt.close(fig)
