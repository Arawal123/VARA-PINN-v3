"""Run independent modern PINN baselines alongside Vanilla, RAR, and VARA."""

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

from src.training.vara_trainer import VARATrainer
from src.utils.config import deep_update, load_config, save_config
from src.utils.io import save_json


METHODS = {
    "vanilla": ("vanilla_pinn", None),
    "rar": ("rar_pinn", "configs/ablation_rar.yaml"),
    "self_adaptive_attention": (
        "self_adaptive_attention_pinn",
        "configs/baselines/self_adaptive_attention.yaml",
    ),
    "gradient_balanced": (
        "gradient_balanced_pinn",
        "configs/baselines/gradient_balanced.yaml",
    ),
    "gradient_enhanced": (
        "gradient_enhanced_pinn",
        "configs/baselines/gradient_enhanced.yaml",
    ),
    "relobralo": (
        "relobralo_pinn",
        "configs/baselines/relobralo.yaml",
    ),
    "residual_attention": (
        "residual_attention_pinn",
        "configs/baselines/residual_attention.yaml",
    ),
    "causal": (
        "causal_pinn",
        "configs/baselines/causal.yaml",
    ),
    "vara": ("local_constrained_vara", None),
}

DEFAULT_METHODS = [name for name in METHODS if name != "causal"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lid_driven_cavity.yaml")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output_dir", default="experiments/cavity_modern_baselines")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include_final_repair", action="store_true")
    args = parser.parse_args()

    unknown = sorted(set(args.methods).difference(METHODS))
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Available: {sorted(METHODS)}")

    base = load_config(args.config)
    output = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for method in args.methods:
            mode, overlay_path = METHODS[method]
            config = deepcopy(base)
            if overlay_path:
                config = deep_update(config, load_config(ROOT / overlay_path))
            config["seed"] = int(seed)
            config["run_type"] = "modern_baseline_quick" if args.quick else "modern_baseline"
            config["experiments"] = {**config.get("experiments", {}), "root": str(output)}
            if args.device:
                config["device"] = args.device
            if not args.include_final_repair:
                config = deep_update(config, {"optimizer": {"final_repair": {"enabled": False}}})
            if args.quick:
                config = _quick_config(config)
            trainer = VARATrainer(config, mode=mode)
            metrics = trainer.run()
            row = {
                **metrics,
                "method": method,
                "mode": mode,
                "seed": int(seed),
                "run_dir": str(trainer.run_dir),
            }
            save_json(row, trainer.run_dir / "summary.json")
            rows.append(row)
            print(f"seed={seed} method={method}: {trainer.run_dir}")

    summary = output / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(summary / "modern_baselines_raw.csv", index=False)
    _aggregate(raw).to_csv(summary / "modern_baselines_mean_std.csv", index=False)
    save_config(base, summary / "base_config_snapshot.yaml")
    print(f"Saved: {summary}")


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = {"seed", "run_dir"}
    numeric = [
        col
        for col in df.columns
        if col not in id_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    rows = []
    for (method, mode), group in df.groupby(["method", "mode"], dropna=False):
        for metric in numeric:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "method": method,
                    "mode": mode,
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(),
                    "count": len(values),
                }
            )
    return pd.DataFrame(rows)


def _quick_config(config: dict[str, Any]) -> dict[str, Any]:
    return deep_update(
        config,
        {
            "model": {"hidden_layers": [16, 16]},
            "training": {
                "adaptive_cycles": 1,
                "epochs_per_cycle": 2,
                "log_every": 1,
                "n_collocation": 32,
                "n_boundary": 24,
                "n_data": 0,
            },
            "local_controller": {
                "trial_epochs": 1,
                "warmup_cycles": 0,
                "max_actions_per_cycle": 1,
                "rejection_recovery_epochs": 0,
            },
            "gradient_balancing": {"update_every": 1},
            "validation": {"nx": 8, "ny": 8},
            "test": {"nx": 8, "ny": 8},
        },
    )


if __name__ == "__main__":
    main()
