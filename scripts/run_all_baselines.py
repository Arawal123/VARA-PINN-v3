"""Run the standard baseline suite sequentially."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.baseline_trainer import BaselineTrainer
from src.utils.config import load_config


MODES = [
    "vanilla_pinn",
    "global_adaptive_loss",
    "region_only_adaptive",
    "variable_only_adaptive",
    "residual_adaptive_sampling",
    "rar_pinn",
    "global_vorticity_loss",
    "global_pressure_correction",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    base = load_config(args.config)
    for mode in MODES:
        cfg = dict(base)
        trainer = BaselineTrainer(cfg, mode=mode)
        trainer.run()
        print(f"{mode}: {trainer.run_dir}")


if __name__ == "__main__":
    main()
