"""Run same-schedule and equal-compute comparisons across PINN methods."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_modern_baselines import DEFAULT_METHODS, METHODS, _quick_config
from src.training.vara_trainer import VARATrainer
from src.utils.config import deep_update, load_config, save_config
from src.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lid_driven_cavity.yaml")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["same_schedule", "same_steps", "same_collocation", "same_wall_clock"],
        default=["same_schedule", "same_steps", "same_collocation", "same_wall_clock"],
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--collocation_evaluations", type=int, default=819200)
    parser.add_argument("--wall_clock_sec", type=float, default=600.0)
    parser.add_argument("--output_dir", default="experiments/cavity_equal_compute")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    unknown = sorted(set(args.methods).difference(METHODS))
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Available: {sorted(METHODS)}")

    base = load_config(args.config)
    output = Path(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            for method in args.methods:
                mode, overlay_path = METHODS[method]
                config = deepcopy(base)
                if overlay_path:
                    config = deep_update(config, load_config(ROOT / overlay_path))
                config["seed"] = int(seed)
                config["run_type"] = f"equal_compute_{scenario}"
                scenario_root = output / scenario
                config["experiments"] = {**config.get("experiments", {}), "root": str(scenario_root)}
                if args.device:
                    config["device"] = args.device
                # Guarded LBFGS uses a variable number of closure evaluations,
                # so equal-compute studies use Adam-only optimization for every method.
                config = deep_update(config, {"optimizer": {"final_repair": {"enabled": False}}})
                if args.quick:
                    config = _quick_config(config)
                config["compute_budget"] = _budget_for(scenario, args)
                trainer = VARATrainer(config, mode=mode)
                metrics = trainer.run()
                row = {
                    **metrics,
                    "scenario": scenario,
                    "method": method,
                    "mode": mode,
                    "seed": int(seed),
                    "run_dir": str(trainer.run_dir),
                }
                save_json(row, trainer.run_dir / "summary.json")
                all_rows.append(row)
                print(f"scenario={scenario} seed={seed} method={method}: {trainer.run_dir}")

    summary = output / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(all_rows)
    raw.to_csv(summary / "equal_compute_raw.csv", index=False)
    _aggregate(raw).to_csv(summary / "equal_compute_mean_std.csv", index=False)
    _fairness_table(raw).to_csv(summary / "equal_compute_fairness_check.csv", index=False)
    save_config(base, summary / "base_config_snapshot.yaml")
    print(f"Saved: {summary}")


def _budget_for(scenario: str, args: argparse.Namespace) -> dict[str, Any]:
    if scenario == "same_schedule":
        return {"enabled": False}
    if scenario == "same_steps":
        return {"enabled": True, "type": "optimizer_steps", "value": int(args.steps)}
    if scenario == "same_collocation":
        return {
            "enabled": True,
            "type": "collocation_evaluations",
            "value": int(args.collocation_evaluations),
        }
    return {"enabled": True, "type": "wall_clock_sec", "value": float(args.wall_clock_sec)}


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "cavity_benchmark_score",
        "centerline_profile_score",
        "u_centerline_rmse",
        "v_centerline_rmse",
        "pde_residual_mean",
        "continuity_residual_mean",
        "momentum_residual_mean",
        "boundary_condition_error",
        "unweighted_validation_loss",
        "final_total_loss",
        "training_wall_clock_sec",
        "optimizer_steps",
        "objective_evaluations",
        "collocation_evaluations",
        "controller_and_io_overhead_percent",
    ]
    rows = []
    for (scenario, method), group in df.groupby(["scenario", "method"]):
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(),
                    "count": len(values),
                }
            )
    return pd.DataFrame(rows)


def _fairness_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "training_wall_clock_sec",
        "optimizer_steps",
        "objective_evaluations",
        "collocation_evaluations",
        "boundary_evaluations",
    ]
    rows = []
    for (scenario, method), group in df.groupby(["scenario", "method"]):
        row = {"scenario": scenario, "method": method, "count": len(group)}
        for column in columns:
            if column in group:
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"{column}_mean"] = values.mean()
                row[f"{column}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
