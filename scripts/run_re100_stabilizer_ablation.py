"""Separate cavity solver stabilization from VARA controller contribution."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_vara_v2_continuation import run as run_continuation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--preset",
        choices=["fast_screen", "diagnostic", "reliable", "final"],
        default="fast_screen",
    )
    parser.add_argument(
        "--config",
        default="configs/vara_v2/lid_cavity_continuation_reliable.yaml",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/vara_v2/re100_stabilizer_ablation",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_dir)
    rows: list[pd.DataFrame] = []
    for label, disabled in (("with_stabilizers", False), ("without_stabilizers", True)):
        destination = root / label
        run_args = argparse.Namespace(
            config=args.config,
            methods=["vanilla", "vara_v2"],
            reynolds=[100.0],
            seeds=[int(args.seed)],
            full_field_reference_map=None,
            device=args.device,
            output_dir=str(destination),
            enhanced_backbone=False,
            reliable=True,
            preset=args.preset,
            disable_stabilizers=disabled,
            gate_vara_on_vanilla=False,
            continue_on_invalid=True,
            quick=False,
            overwrite=bool(args.overwrite),
        )
        result = run_continuation(run_args)["raw"].copy()
        result["stabilizer_setting"] = label
        rows.append(result)

    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(rows, ignore_index=True)
    columns = [
        column
        for column in (
            "stabilizer_setting",
            "method",
            "seed",
            "continuation_stage_valid",
            "pde_residual_mean",
            "momentum_residual_mean",
            "near_wall_pde_residual_mean",
            "near_wall_momentum_v_mean",
            "boundary_condition_error",
            "speed_pred_max",
            "detected_vortex_count",
            "lid_cavity_topology_aligned",
            "optimizer_steps",
            "collocation_evaluations",
            "training_wall_clock_sec",
        )
        if column in raw.columns
    ]
    raw.to_csv(summary / "stabilizer_ablation_full.csv", index=False)
    raw[columns].to_csv(summary / "stabilizer_ablation_compact.csv", index=False)
    print(raw[columns].to_string(index=False))
    print(
        "Interpretation: compare VARA only against Vanilla within the same "
        "stabilizer_setting. Stabilizer gains are solver-repair evidence, not VARA gains."
    )


if __name__ == "__main__":
    main()
