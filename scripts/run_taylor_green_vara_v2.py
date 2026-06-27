"""Run the isolated repaired Taylor--Green VARA V2 experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.taylor_green_vara_v2_trainer import TaylorGreenVARAV2Trainer
from src.utils.config import deep_update, load_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/taylor_green_repaired.yaml",
    )
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output_dir",
        default="experiments/vara_v2/taylor_green_repaired",
    )
    args = parser.parse_args()

    base = load_config(args.config)
    for overlay in args.overlay:
        base = deep_update(base, load_config(overlay))
    rows = []
    for seed in args.seeds:
        config = deepcopy(base)
        config["seed"] = int(seed)
        config["experiments"] = {
            **config.get("experiments", {}),
            "root": args.output_dir,
        }
        if args.device:
            config["device"] = args.device
        trainer = TaylorGreenVARAV2Trainer(config)
        metrics = trainer.run()
        rows.append(
            {
                **metrics,
                "seed": int(seed),
                "run_dir": str(trainer.run_dir),
            }
        )
        print(f"seed={seed}: {trainer.run_dir}")

    summary = Path(args.output_dir) / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary / "taylor_green_vara_v2_raw.csv", index=False)
    save_config(base, summary / "resolved_base_config.yaml")


if __name__ == "__main__":
    main()
