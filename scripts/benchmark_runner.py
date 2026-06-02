"""Shared helpers for lightweight professor benchmark runners."""

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


METHOD_TO_MODE = {"vanilla": "vanilla_pinn", "vara": "local_constrained_vara"}


BENCHMARK_DEFAULTS: dict[str, dict[str, Any]] = {
    "channel_inflow_outflow": {
        "benchmark": "channel_inflow_outflow",
        "benchmark_params": {"reynolds": 40.0, "x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 1.0, "amplitude": 1.0},
        "diagnostics": {"mode": "full_reference"},
        "local_controller": {
            "objective_weights": {
                "u": 1.0,
                "v": 1.0,
                "p": 1.0,
                "omega": 0.5,
                "residual": 0.25,
                "continuity": 1.0,
                "momentum": 0.5,
                "boundary": 1.0,
                "unweighted_validation": 0.25,
            },
            "continuity_collateral_tolerance": 0.05,
            "boundary_collateral_tolerance": 0.05,
            "validation_loss_tolerance": 0.02,
        },
    },
    "lid_driven_cavity": {
        "benchmark": "lid_driven_cavity",
        "benchmark_params": {
            "reynolds": 100.0,
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "lid_velocity": 1.0,
            "reference": "ghia",
            "profile_only": True,
        },
        "diagnostics": {
            "mode": "residual_only",
            "variables": [
                "continuity_residual",
                "momentum_u_residual",
                "momentum_v_residual",
                "aggregate_pde_residual",
                "centerline_pde_residual",
                "centerline_continuity_residual",
                "corner_pde_residual",
                "corner_boundary_error",
                "boundary_violation",
            ],
        },
        "training": {"n_data": 0},
        "optimizer": {
            "name": "adam",
            "lr": 0.001,
            "final_repair": {
                "enabled": True,
                "score_metric": "cavity_benchmark_score",
                "acceptance_tolerance": 0.0,
                "epochs": 8,
                "lr": 0.5,
                "max_iter": 5,
                "history_size": 20,
                "line_search_fn": "strong_wolfe",
                "batch_multiplier": 2.0,
                "residual_fraction": 0.30,
                "collateral_tolerances": {
                    "pde_residual_mean": 0.10,
                    "momentum_residual_mean": 0.18,
                    "boundary_condition_error": 0.02,
                },
                "weights": {
                    "pde": 0.5,
                    "momentum_u": 1.0,
                    "momentum_v": 1.0,
                    "continuity": 1.5,
                    "bc": 6.0,
                    "u": 0.0,
                    "v": 0.0,
                    "p": 0.0,
                    "omega": 0.0,
                    "pressure_gradient": 0.0,
                },
            },
        },
        "local_controller": {
            "objective_weights": {
                "u": 0.0,
                "v": 0.0,
                "p": 0.0,
                "omega": 0.0,
                "residual": 0.5,
                "continuity": 1.75,
                "momentum": 0.75,
                "boundary": 3.0,
                "unweighted_validation": 0.75,
                "centerline_pde": 0.75,
                "centerline_continuity": 0.50,
                "corner_pde": 0.25,
                "corner_boundary": 1.00,
                "u_boundary": 0.50,
                "v_boundary": 0.75,
            },
            "continuity_collateral_tolerance": 0.05,
            "boundary_collateral_tolerance": 0.005,
            "validation_loss_tolerance": 0.010,
            "pde_hard_tolerance": 0.005,
            "boundary_hard_tolerance": 0.002,
            "validation_loss_hard_tolerance": 0.005,
            "centerline_pde_hard_tolerance": 0.025,
            "corner_boundary_hard_tolerance": 0.005,
            "u_boundary_hard_tolerance": 0.005,
            "v_boundary_hard_tolerance": 0.005,
            "initial_strength": 0.30,
            "warmup_cycles": 2,
            "wall_patch_strength_factor": 0.70,
            "corner_patch_strength_factor": 0.50,
            "boundary_patch_margin": 1.0e-9,
            "rejection_recovery_epochs": 10,
            "local_velocity_weight_max": 1.5,
            "local_pde_weight_max": 1.5,
            "local_bc_weight_max": 5.0,
        },
        "sampling": {
            "mixture": {
                "uniform": 0.45,
                "pde_residual": 0.25,
                "pressure_vorticity": 0.08,
                "velocity": 0.10,
                "boundary": 0.12,
            },
            "cavity_boundary": {
                "enabled": True,
                "lid_fraction": 0.45,
                "corner_fraction": 0.25,
                "corner_width": 0.12,
            },
        },
    },
    "double_vortex_box": {
        "benchmark": "double_vortex_box",
        "benchmark_params": {"reynolds": 40.0, "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "amplitude": 1.0},
        "diagnostics": {"mode": "full_reference"},
        "local_controller": {
            "objective_weights": {
                "u": 1.0,
                "v": 1.0,
                "p": 0.5,
                "omega": 1.0,
                "residual": 0.25,
                "continuity": 1.0,
                "momentum": 0.5,
                "boundary": 1.0,
                "unweighted_validation": 0.25,
            },
            "continuity_collateral_tolerance": 0.05,
            "boundary_collateral_tolerance": 0.05,
            "validation_loss_tolerance": 0.02,
        },
    },
    "boundary_condition_stress_test": {
        "benchmark": "boundary_condition_stress_test",
        "benchmark_params": {"reynolds": 40.0, "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "amplitude": 1.0},
        "diagnostics": {"mode": "full_reference"},
        "local_controller": {
            "objective_weights": {
                "u": 1.0,
                "v": 1.0,
                "p": 0.5,
                "omega": 0.5,
                "residual": 0.25,
                "continuity": 1.0,
                "momentum": 0.5,
                "boundary": 1.5,
                "unweighted_validation": 0.25,
            },
            "continuity_collateral_tolerance": 0.05,
            "boundary_collateral_tolerance": 0.05,
            "validation_loss_tolerance": 0.02,
        },
    },
    "rectangular_aspect_ratio": {
        "benchmark": "rectangular_aspect_ratio",
        "benchmark_params": {"reynolds": 40.0, "x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 1.0, "amplitude": 1.0},
        "diagnostics": {"mode": "full_reference"},
        "local_controller": {
            "objective_weights": {
                "u": 1.0,
                "v": 1.0,
                "p": 1.0,
                "omega": 0.5,
                "residual": 0.25,
                "continuity": 1.0,
                "momentum": 0.5,
                "boundary": 1.0,
                "unweighted_validation": 0.25,
            },
            "continuity_collateral_tolerance": 0.05,
            "boundary_collateral_tolerance": 0.05,
            "validation_loss_tolerance": 0.02,
        },
    },
}


def parser_for(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--method", choices=["vanilla", "vara", "both"], default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--config", default="configs/kovasznay_debug.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--reynolds", type=float, default=None)
    parser.add_argument("--reference", choices=["ghia", "external", "none"], default=None)
    parser.add_argument("--reference_path", default=None)
    parser.add_argument("--full_field_reference_path", default=None)
    parser.add_argument("--profile_only", action="store_true")
    return parser


def run_named_benchmark(
    benchmark_name: str,
    args: argparse.Namespace,
    aspect_ratio: float | None = None,
) -> list[dict[str, Any]]:
    base = load_config(args.config)
    config = deep_update(base, BENCHMARK_DEFAULTS[benchmark_name])
    config["run_type"] = "smoke" if args.quick else "benchmark"
    config["experiments"] = dict(config.get("experiments", {}))
    config["experiments"]["root"] = args.output_dir
    if args.device:
        config["device"] = args.device
    if benchmark_name == "lid_driven_cavity":
        params = dict(config.get("benchmark_params", {}))
        if args.reynolds is not None:
            params["reynolds"] = float(args.reynolds)
        params["reference"] = args.reference or params.get("reference", "ghia")
        if args.reference_path:
            params["reference_path"] = args.reference_path
        if args.full_field_reference_path:
            params["full_field_reference_path"] = args.full_field_reference_path
        params["profile_only"] = bool(args.profile_only or not args.full_field_reference_path)
        config["benchmark_params"] = params
    if aspect_ratio is not None:
        params = dict(config.get("benchmark_params", {}))
        params["x_min"] = 0.0
        params["y_min"] = 0.0
        params["y_max"] = 1.0
        params["x_max"] = float(aspect_ratio)
        config["benchmark_params"] = params
        config["benchmark"] = "rectangular_aspect_ratio"
    if config["benchmark"] == "lid_driven_cavity":
        config["training"] = {**config.get("training", {}), "n_data": 0}
        config["diagnostics"] = {
            **config.get("diagnostics", {}),
            "mode": "residual_only",
            "variables": BENCHMARK_DEFAULTS["lid_driven_cavity"]["diagnostics"]["variables"],
        }
    if args.quick:
        is_cavity = config["benchmark"] == "lid_driven_cavity"
        config = deep_update(
            config,
            {
                "model": {"hidden_layers": [32, 32]},
                "training": {
                    "adaptive_cycles": 2 if is_cavity else 1,
                    "epochs_per_cycle": 5,
                    "log_every": 5,
                    "n_collocation": 128,
                    "n_boundary": 96 if is_cavity else 64,
                    "n_data": 0 if is_cavity else 64,
                },
                "local_controller": {
                    "trial_epochs": 2,
                    "max_actions_per_cycle": 2,
                    "warmup_cycles": 1 if is_cavity else 0,
                    "rejection_recovery_epochs": 2 if is_cavity else 0,
                },
                "optimizer": {
                    "final_repair": {
                        "epochs": 1 if is_cavity else 0,
                        "max_iter": 1,
                        "history_size": 5,
                        "batch_multiplier": 1.0,
                    },
                },
                "validation": {"nx": 16, "ny": 16},
                "test": {"nx": 20, "ny": 20},
            },
        )

    methods = ["vanilla", "vara"] if args.method == "both" else [args.method]
    rows = []
    for seed in args.seeds:
        for method in methods:
            run_config = deepcopy(config)
            run_config["seed"] = int(seed)
            mode = METHOD_TO_MODE[method]
            trainer = VARATrainer(run_config, mode=mode)
            metrics = trainer.run()
            metrics["benchmark"] = run_config["benchmark"]
            metrics["method"] = method
            metrics["mode"] = mode
            metrics["seed"] = int(seed)
            metrics["run_dir"] = str(trainer.run_dir)
            save_json(metrics, trainer.run_dir / "summary.json")
            pd.DataFrame([metrics]).to_csv(trainer.run_dir / "summary_table.csv", index=False)
            rows.append(metrics)
            print(f"benchmark={run_config['benchmark']} seed={seed} method={method}: {trainer.run_dir}")
    out = Path(args.output_dir) / "benchmark_runs"
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_ar{aspect_ratio:g}" if aspect_ratio is not None else ""
    pd.DataFrame(rows).to_csv(out / f"{benchmark_name}{suffix}_latest_results.csv", index=False)
    save_config(config, out / f"{benchmark_name}{suffix}_resolved_config.yaml")
    return rows
