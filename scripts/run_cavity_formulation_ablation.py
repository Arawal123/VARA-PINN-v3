"""Vanilla-only Re=100 cavity formulation ablation.

This runner is intentionally narrow: it screens hard-boundary streamfunction
parameters before VARA is evaluated. It uses no CFD/Ghia data in training;
centerline metrics are evaluation-only fields in the output table.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_vara_v2_continuation import _load_base_config
from src.training.vara_trainer import VARATrainer
from src.utils.config import deep_update, load_config, save_config
from src.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="experiments/vara_v2/cavity_formulation_ablation_re100")
    parser.add_argument("--screening_steps", type=int, default=1200)
    parser.add_argument("--epochs_per_cycle", type=int, default=200)
    parser.add_argument("--corner_widths", nargs="+", type=float, default=[0.04, 0.06, 0.08])
    parser.add_argument("--lid_vertical_powers", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--correction_scales", nargs="+", type=float, default=[32.0, 64.0, 128.0])
    parser.add_argument("--speed_kill_step", type=int, default=400)
    parser.add_argument("--speed_kill_max", type=float, default=3.0)
    parser.add_argument("--pde_kill_step", type=int, default=800)
    parser.add_argument("--pde_kill_max", type=float, default=20.0)
    parser.add_argument("--vortex_kill_max", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


def run(args: argparse.Namespace) -> pd.DataFrame:
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        if not bool(args.overwrite):
            raise SystemExit(f"Output directory is not empty: {output}. Use --overwrite.")
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    base = _load_base_config(args.config)
    base = deep_update(base, load_config("configs/vara_v2/controller.yaml"))
    base = _screening_config(base, args)
    save_config(base, output / "resolved_screening_base_config.yaml")

    rows: list[dict[str, Any]] = []
    for corner_width in args.corner_widths:
        for lid_power in args.lid_vertical_powers:
            for correction_scale in args.correction_scales:
                row = _run_candidate(
                    base,
                    output,
                    seed=int(args.seed),
                    corner_width=float(corner_width),
                    lid_vertical_power=int(lid_power),
                    correction_scale=float(correction_scale),
                    screening_steps=int(args.screening_steps),
                    epochs_per_cycle=int(args.epochs_per_cycle),
                    args=args,
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(output / "ablation_raw_results.csv", index=False)

    results = pd.DataFrame(rows)
    ranked = _rank(results)
    ranked.to_csv(output / "ablation_ranked_results.csv", index=False)
    _write_report(output, ranked)
    print(f"Saved cavity formulation ablation to: {output}")
    print(ranked.head(10).to_string(index=False))
    return ranked


def _screening_config(base: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cycles = max(1, int(np.ceil(args.screening_steps / max(args.epochs_per_cycle, 1))))
    overrides: dict[str, Any] = {
        "seed": int(args.seed),
        "run_type": "cavity_formulation_ablation",
        "model": {
            "physics_formulation": "hard_boundary_streamfunction_pressure",
            "output_dim": 2,
        },
        "benchmark_params": {
            "reynolds": 100.0,
            "reference": "ghia",
            "reference_path": None,
            "full_field_reference_path": None,
            "profile_only": True,
        },
        "training": {
            "adaptive_cycles": cycles,
            "epochs_per_cycle": int(args.epochs_per_cycle),
            "log_every": 50,
        },
        "controller_v2": {
            "total_steps": int(args.screening_steps),
            "warmup_steps": min(300, int(args.screening_steps)),
        },
        "optimizer": {
            "scheduler": {
                "enabled": True,
                "total_steps": int(args.screening_steps),
                "warmup_steps": min(200, int(args.screening_steps)),
                "warmup_start_ratio": 0.2,
                "min_lr_ratio": 0.10,
            },
            "final_repair": {"enabled": False},
        },
        "cavity_curriculum": {"enabled": False},
        "convergence_early_stopping": {"enabled": False},
        "evaluation": {
            "controller_reference_metrics_enabled": False,
            "checkpoint_reference_metrics_enabled": False,
        },
    }
    if args.device:
        overrides["device"] = args.device
    return deep_update(base, overrides)


def _run_candidate(
    base: dict[str, Any],
    output: Path,
    *,
    seed: int,
    corner_width: float,
    lid_vertical_power: int,
    correction_scale: float,
    screening_steps: int,
    epochs_per_cycle: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = (
        f"cw_{corner_width:.3f}_p_{lid_vertical_power}_"
        f"scale_{correction_scale:g}"
    ).replace(".", "p")
    run_dir = output / "candidates" / name
    config = deepcopy(base)
    config["seed"] = seed
    config["experiments"] = {
        **config.get("experiments", {}),
        "root": str(run_dir),
        "flat_layout": True,
    }
    config["model"] = {
        **config.get("model", {}),
        "hard_boundary_corner_width": float(corner_width),
        "hard_boundary_lid_vertical_power": int(lid_vertical_power),
        "hard_boundary_correction_scale": float(correction_scale),
    }
    config["benchmark_params"] = {
        **config.get("benchmark_params", {}),
        "lid_corner_regularization_width": float(corner_width),
    }
    trainer = VARATrainer(config, mode="vanilla_pinn")
    batch = trainer.initial_batch()
    cycles = max(1, int(np.ceil(screening_steps / max(epochs_per_cycle, 1))))
    early_stop_reason = ""
    last_metrics: dict[str, Any] = {}
    for cycle in range(cycles):
        trainer.train_epochs(
            batch,
            None,
            cycle=cycle,
            epochs_override=epochs_per_cycle,
            log_prefix="vanilla_formulation_ablation",
        )
        maps, scores, names, _weak_regions, _x, _y, coords = trainer.diagnose()
        metrics = trainer._validation_metrics(coords)
        last_metrics = dict(metrics)
        trainer.metrics_logger.log({"cycle": cycle, "phase": "ablation_screen", **metrics})
        trainer.score_logger.log({"cycle": cycle, "diagnostics": names, "scores": scores})
        trainer.maybe_checkpoint(cycle, metrics)
        step = trainer.global_step
        if step >= int(args.speed_kill_step) and float(metrics.get("speed_pred_max", 0.0)) > float(args.speed_kill_max):
            early_stop_reason = f"speed_pred_max>{args.speed_kill_max:g}@{step}"
            break
        if step >= int(args.pde_kill_step) and float(metrics.get("pde_residual_mean", 0.0)) > float(args.pde_kill_max):
            early_stop_reason = f"pde_residual_mean>{args.pde_kill_max:g}@{step}"
            break
        if step >= screening_steps and int(metrics.get("detected_vortex_count", 0)) > int(args.vortex_kill_max):
            early_stop_reason = f"detected_vortex_count>{args.vortex_kill_max}@{step}"
            break
        batch = trainer.resample_batch(batch, maps, coords, [], None, adaptive=False)

    final_metrics = trainer.evaluate_and_save_final()
    row = {
        **final_metrics,
        "seed": int(seed),
        "candidate": name,
        "corner_width": float(corner_width),
        "lid_vertical_power": int(lid_vertical_power),
        "correction_scale": float(correction_scale),
        "screening_steps_requested": int(screening_steps),
        "screening_steps_completed": int(trainer.global_step),
        "early_stop_reason": early_stop_reason,
        "candidate_dir": str(run_dir),
    }
    if last_metrics:
        row["last_screen_pde_residual_mean"] = last_metrics.get("pde_residual_mean")
        row["last_screen_speed_pred_max"] = last_metrics.get("speed_pred_max")
    save_json(row, run_dir / "ablation_summary.json")
    return row


def _rank(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    ranked = results.copy()
    topology = (
        pd.to_numeric(ranked["lid_cavity_topology_aligned"], errors="coerce")
        if "lid_cavity_topology_aligned" in ranked.columns
        else pd.Series(0.0, index=ranked.index)
    )
    ranked["topology_bonus"] = -topology.fillna(0.0)
    sort_cols = [
        "topology_bonus",
        "pde_residual_mean",
        "momentum_residual_mean",
        "speed_pred_max",
        "omega_pred_abs_95p",
        "lid_cavity_primary_center_error",
        "detected_vortex_count",
        "centerline_profile_score",
    ]
    available = [column for column in sort_cols if column in ranked.columns]
    for column in available:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked = ranked.sort_values(available, ascending=True, na_position="last")
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked.drop(columns=["topology_bonus"], errors="ignore")


def _write_report(output: Path, ranked: pd.DataFrame) -> None:
    lines = [
        "# Re=100 Vanilla Cavity Formulation Ablation",
        "",
        "Training is reference-free. Ghia centerline fields in the CSV are evaluation-only.",
        "",
    ]
    if ranked.empty:
        lines.append("No candidates completed.")
    else:
        best = ranked.iloc[0]
        lines.extend(
            [
                "## Best Candidate",
                "",
                f"- candidate: `{best['candidate']}`",
                f"- pde_residual_mean: {best.get('pde_residual_mean', np.nan):.6g}",
                f"- momentum_residual_mean: {best.get('momentum_residual_mean', np.nan):.6g}",
                f"- speed_pred_max: {best.get('speed_pred_max', np.nan):.6g}",
                f"- omega_pred_abs_95p: {best.get('omega_pred_abs_95p', np.nan):.6g}",
                f"- topology_aligned: {best.get('lid_cavity_topology_aligned', np.nan)}",
                f"- primary_center_error: {best.get('lid_cavity_primary_center_error', np.nan):.6g}",
                f"- detected_vortex_count: {best.get('detected_vortex_count', np.nan)}",
                "",
            ]
        )
    (output / "repair_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
