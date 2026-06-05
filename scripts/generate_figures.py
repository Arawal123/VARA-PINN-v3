"""Regenerate summary figures from an experiment directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    args = parser.parse_args()
    exp = Path(args.experiment_dir)
    metrics = pd.read_csv(exp / "metrics.csv")
    out = exp / "metric_curves.png"
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for col in ["u_rel_l2", "v_rel_l2", "p_rel_l2_centered", "omega_rel_l2"]:
        if col in metrics:
            rows = metrics[metrics["cycle"] != "final_test"]
            ax.plot(rows["cycle"], rows[col], marker="o", label=col)
    ax.set_yscale("log")
    ax.set_xlabel("cycle")
    ax.legend()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
