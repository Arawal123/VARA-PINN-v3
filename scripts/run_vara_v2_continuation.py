"""Run isolated multi-method Reynolds continuation for the V2 study."""

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

from scripts.run_lid_cavity_re_continuation import (
    _comparison_rows,
    _full_field_reference_for_re,
    _load_full_field_reference_map,
    _reference_for_re,
    _save_per_re_comparison,
    _save_image_grid,
    _montage_label,
    _save_summary_bar_plots,
    _save_summary_montages,
    _wide_improvement,
)
from scripts.run_modern_baselines import METHODS as BASELINE_METHODS
from src.training.vara_trainer import VARATrainer
from src.training.vara_v2_trainer import VARAV2Trainer
from src.utils.config import deep_update, load_config, save_config
from src.utils.io import save_json


DEFAULT_REYNOLDS = [100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3200]
EXPLORATORY_REYNOLDS = [4000, 5000, 7500, 10000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vara_v2/lid_driven_cavity.yaml")
    parser.add_argument("--methods", nargs="+", default=["vanilla", "vara_v1", "vara_v2"])
    parser.add_argument("--reynolds", nargs="+", type=float, default=DEFAULT_REYNOLDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--full_field_reference_map",
        default="data/references/lid_driven_cavity/full_field/reference_map.csv",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="experiments/vara_v2/re_continuation")
    parser.add_argument("--enhanced_backbone", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args)


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    base = load_config(args.config)
    base = deep_update(base, load_config("configs/vara_v2/controller.yaml"))
    base = deep_update(base, load_config("configs/vara_v2/continuation.yaml"))
    if args.enhanced_backbone:
        base = deep_update(base, load_config("configs/vara_v2/enhanced_backbone.yaml"))
    if args.quick:
        base = deep_update(
            base,
            {
                "model": {"hidden_layers": [16, 16]},
                "training": {
                    "adaptive_cycles": 1,
                    "epochs_per_cycle": 4,
                    "n_collocation": 32,
                    "n_boundary": 24,
                    "n_data": 0,
                    "log_every": 1,
                },
                "validation": {"nx": 8, "ny": 8},
                "test": {"nx": 8, "ny": 8},
                "patches": {"nx_patches": 2, "ny_patches": 2},
                "controller_v2": {
                    "total_steps": 4,
                    "warmup_steps": 1,
                    "control_blocks": 1,
                    "block_steps": 3,
                    "probe_steps": 1,
                    "gradient_probe_interior": 12,
                    "gradient_probe_boundary": 8,
                },
                "compute_budget": {"enabled": True, "type": "optimizer_steps", "value": 4},
            },
        )
    if args.device:
        base["device"] = args.device
    reference_map = _load_full_field_reference_map(args.full_field_reference_map)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        previous: dict[str, Path | None] = {method: None for method in args.methods}
        for reynolds in args.reynolds:
            per_method: dict[str, dict[str, Any]] = {}
            re_name = f"re_{int(round(reynolds)):04d}"
            for method in args.methods:
                method_dir = output / f"seed_{seed}" / re_name / method
                config = deepcopy(base)
                config["seed"] = int(seed)
                config["experiments"] = {
                    **config.get("experiments", {}),
                    "root": str(method_dir),
                    "flat_layout": True,
                }
                reference = _reference_for_re(float(reynolds), "ghia", None)
                full_field = _full_field_reference_for_re(float(reynolds), reference_map)
                config["benchmark_params"] = {
                    **config.get("benchmark_params", {}),
                    "reynolds": float(reynolds),
                    "reference": reference["reference"],
                    "reference_path": reference.get("reference_path"),
                    "full_field_reference_path": str(full_field) if full_field is not None else None,
                    "profile_only": full_field is None,
                }
                config["warm_start_checkpoint"] = str(previous[method]) if previous[method] else None
                config["warm_start"] = {"load_optimizer": False}
                if previous[method] is None:
                    config["continuation_anchor"] = {
                        **config.get("continuation_anchor", {}),
                        "enabled": False,
                    }
                    config["continuation_replay"] = {
                        **config.get("continuation_replay", {}),
                        "enabled": False,
                    }
                trainer = _trainer_for(method, config)
                metrics = trainer.run()
                checkpoint = trainer.checkpoint_dir / "final.pt"
                previous[method] = checkpoint
                row = {
                    **metrics,
                    "method": method,
                    "seed": int(seed),
                    "reynolds": float(reynolds),
                    "run_dir": str(trainer.run_dir),
                    "checkpoint": str(checkpoint),
                    "method_dir": str(method_dir),
                }
                save_json(row, trainer.run_dir / "summary.json")
                rows.append(row)
                per_method[method] = row
                print(f"seed={seed} Re={reynolds:g} method={method}: {trainer.run_dir}")

            if "vanilla" in per_method and "vara_v2" in per_method:
                comparison_rows.extend(
                    _comparison_rows(seed, reynolds, per_method["vanilla"], per_method["vara_v2"])
                )
                # Reuse the established Vanilla/VARA comparison renderer.
                comparison_dir = output / f"seed_{seed}" / re_name
                _save_per_re_comparison(
                    comparison_dir,
                    per_method["vanilla"],
                    per_method["vara_v2"],
                )

    summary = output / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    comparisons = pd.DataFrame(comparison_rows)
    raw.to_csv(summary / "continuation_results_long.csv", index=False)
    comparisons.to_csv(summary / "vara_v2_vs_vanilla_by_re.csv", index=False)
    _wide_improvement(comparisons).to_csv(summary / "improvement_percent_by_re.csv", index=False)
    _save_summary_bar_plots(comparisons, summary)
    _save_summary_montages(output, summary)
    v2_items = [
        (path, _montage_label(path, "vara_v2"))
        for path in sorted(output.glob("seed_*/re_*/vara_v2/figures/streamlines.png"))
    ]
    _save_image_grid(
        v2_items,
        summary / "streamline_montage_vara_v2.png",
        cols=4,
        title="VARA V2 continuation streamlines",
    )
    save_config(base, summary / "resolved_base_config.yaml")
    return {"raw": raw, "comparisons": comparisons}


def _trainer_for(method: str, config: dict[str, Any]) -> Any:
    if method == "vara_v2":
        return VARAV2Trainer(config)
    if method == "vara_v1":
        return VARATrainer(config, mode="local_constrained_vara")
    if method not in BASELINE_METHODS:
        raise ValueError(
            f"Unknown continuation method {method!r}; choose vara_v1, vara_v2, "
            f"or one of {sorted(BASELINE_METHODS)}."
        )
    mode, overlay = BASELINE_METHODS[method]
    resolved = deep_update(config, load_config(overlay)) if overlay else config
    return VARATrainer(resolved, mode=mode)


if __name__ == "__main__":
    main()
