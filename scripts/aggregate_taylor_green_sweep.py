"""Combine repaired Taylor--Green comparison runs into paper-ready CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


METRICS = [
    "taylor_green_mean_time_velocity_rel_l2",
    "taylor_green_worst_time_velocity_rel_l2",
    "taylor_green_mean_time_pressure_rel_l2_centered",
    "taylor_green_worst_time_pressure_rel_l2_centered",
    "taylor_green_mean_time_omega_rel_l2",
    "taylor_green_worst_time_omega_rel_l2",
    "taylor_green_mean_time_pde_residual_mean",
    "taylor_green_worst_time_pde_residual_mean",
    "taylor_green_mean_time_continuity_residual_mean",
    "taylor_green_total_runtime_sec",
    "optimization_wall_clock_sec",
    "applied_optimizer_steps",
    "objective_evaluations",
    "accepted_interventions",
    "rejected_interventions",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="experiments/taylor_green_stress_sweep",
    )
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "summary"
    csv_paths = sorted(
        input_dir.glob("*/summary/taylor_green_repaired_comparison_raw.csv")
    )
    if not csv_paths:
        raise SystemExit(f"No Taylor-Green comparison CSVs found under {input_dir}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        setting = csv_path.parents[1].name
        config = load_config(csv_path.parent / "resolved_base_config.yaml")
        frame = pd.read_csv(csv_path)
        frame.insert(0, "setting", setting)
        frame.insert(1, "reynolds", float(config["benchmark_params"]["reynolds"]))
        frame.insert(2, "total_steps", int(config["controller_v2"]["total_steps"]))
        frame.insert(3, "n_collocation", int(config["training"]["n_collocation"]))
        frame.insert(4, "n_boundary", int(config["training"]["n_boundary"]))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "taylor_green_stress_sweep_combined.csv"
    combined.to_csv(combined_path, index=False)

    available_metrics = [name for name in METRICS if name in combined.columns]
    index_columns = [
        "setting",
        "reynolds",
        "total_steps",
        "n_collocation",
        "n_boundary",
        "seed",
    ]
    table = combined.pivot_table(
        index=index_columns,
        columns="method",
        values=available_metrics,
        aggfunc="first",
    ).reset_index()
    table.columns = [
        "__".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in table.columns
    ]
    for metric in available_metrics:
        vanilla = f"{metric}__vanilla"
        vara = f"{metric}__vara_v2"
        if vanilla in table and vara in table:
            table[f"{metric}__vara_minus_vanilla"] = table[vara] - table[vanilla]
    table_path = output_dir / "taylor_green_stress_sweep_comparison_table.csv"
    table.to_csv(table_path, index=False)
    print(f"Combined results: {combined_path}")
    print(f"Comparison table: {table_path}")


if __name__ == "__main__":
    main()
