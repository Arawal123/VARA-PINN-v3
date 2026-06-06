"""Run a one-factor-at-a-time VARA sensitivity study."""

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

from scripts.run_modern_baselines import _quick_config
from src.training.vara_trainer import VARATrainer
from src.utils.config import deep_update, load_config, save_config
from src.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lid_driven_cavity.yaml")
    parser.add_argument("--study", default="configs/studies/lid_cavity_sensitivity.yaml")
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output_dir", default="experiments/cavity_sensitivity")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plan_only", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    study = load_config(args.study)
    available = dict(study.get("variants", {}))
    selected = list(available) if args.variants == ["all"] else args.variants
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Available: {sorted(available)}")

    output = Path(args.output_dir)
    manifest = []
    for variant in selected:
        manifest.append({"variant": variant, "override": available[variant]})
        resolved = deep_update(base, available[variant])
        resolved["experiments"] = {**resolved.get("experiments", {}), "root": str(output / variant)}
        save_config(resolved, output / "resolved_configs" / f"{variant}.yaml")
    save_json(manifest, output / "sensitivity_manifest.json")
    if args.plan_only:
        print(f"Saved sensitivity plan under {output}")
        return

    rows = []
    for variant in selected:
        for seed in args.seeds:
            config = deep_update(deepcopy(base), available[variant])
            config["seed"] = int(seed)
            config["run_type"] = "sensitivity_quick" if args.quick else "sensitivity"
            config["experiments"] = {**config.get("experiments", {}), "root": str(output / variant)}
            if args.device:
                config["device"] = args.device
            if args.quick:
                config = _quick_config(config)
            trainer = VARATrainer(config, mode="local_constrained_vara")
            metrics = trainer.run()
            row = {
                **metrics,
                "variant": variant,
                "seed": int(seed),
                "run_dir": str(trainer.run_dir),
            }
            save_json(row, trainer.run_dir / "summary.json")
            rows.append(row)
            print(f"variant={variant} seed={seed}: {trainer.run_dir}")

    summary = output / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(summary / "sensitivity_raw.csv", index=False)
    _aggregate(raw).to_csv(summary / "sensitivity_mean_std.csv", index=False)
    print(f"Saved: {summary}")


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
        "accepted_interventions",
        "rejected_interventions",
        "rollback_count",
        "training_wall_clock_sec",
    ]
    rows = []
    for variant, group in df.groupby("variant"):
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(),
                    "count": len(values),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
