"""Run repaired Taylor--Green Vanilla and VARA V2 on matched wiring."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.taylor_green_vanilla_trainer import TaylorGreenVanillaTrainer
from src.training.taylor_green_vara_v2_trainer import TaylorGreenVARAV2Trainer
from src.utils.config import deep_update, load_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/taylor_green_repaired.yaml",
    )
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["vanilla", "vara", "vara_v2"],
        default=["vanilla", "vara_v2"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output_dir",
        default="experiments/vara_v2/taylor_green_repaired_comparison",
    )
    args = parser.parse_args()

    base = load_config(args.config)
    for overlay in args.overlay:
        base = deep_update(base, load_config(overlay))
    schedule = dict(base.get("controller_v2", {}))
    base["taylor_green_comparison"] = {
        "warmup_steps": int(schedule.get("warmup_steps", 1000)),
        "block_steps": int(schedule.get("block_steps", 500)),
    }

    methods = list(dict.fromkeys(
        "vara_v2" if method == "vara" else method for method in args.methods
    ))
    rows: list[dict[str, object]] = []
    output = Path(args.output_dir)
    for seed in args.seeds:
        for method in methods:
            config = deepcopy(base)
            config["seed"] = int(seed)
            config["experiments"] = {
                **config.get("experiments", {}),
                "root": str(output / method),
            }
            config["experiments"].pop("run_id", None)
            if args.device:
                config["device"] = args.device
            trainer = (
                TaylorGreenVanillaTrainer(config)
                if method == "vanilla"
                else TaylorGreenVARAV2Trainer(config)
            )
            metrics = trainer.run()
            row = {
                **metrics,
                "method": method,
                "seed": int(seed),
                "run_dir": str(trainer.run_dir),
            }
            rows.append(row)
            print(f"Taylor-Green seed={seed} method={method}: {trainer.run_dir}")

    summary = output / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        summary / "taylor_green_repaired_comparison_raw.csv",
        index=False,
    )
    save_config(base, summary / "resolved_base_config.yaml")


if __name__ == "__main__":
    main()
