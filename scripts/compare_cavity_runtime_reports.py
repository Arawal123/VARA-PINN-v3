"""Compare two cavity summary.json files without rerunning experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FIELDS = [
    "training_wall_clock_sec",
    "optimizer_steps",
    "seconds_per_optimizer_step",
    "collocation_evaluations",
    "runtime_validation_sec",
    "runtime_plotting_sec",
    "pde_residual_mean",
    "momentum_residual_mean",
    "near_wall_pde_residual_mean",
    "near_wall_momentum_v_mean",
    "speed_pred_max",
    "detected_vortex_count",
    "lid_cavity_topology_aligned",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", default="runtime_before_after.csv")
    args = parser.parse_args()

    rows = []
    for label, path in (("before", args.before), ("after", args.after)):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append({"run": label, **{field: data.get(field) for field in FIELDS}})
    table = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
