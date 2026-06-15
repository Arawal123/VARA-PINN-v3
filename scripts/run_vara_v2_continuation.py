"""Run isolated multi-method Reynolds continuation for the V2 study."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
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
    _wide_improvement,
)
from scripts.run_modern_baselines import METHODS as BASELINE_METHODS
from src.physics.cavity_cfd_generator import generate_cavity_cfd_reference
from src.physics.cavity_reference import load_full_field_reference
from src.training.vara_trainer import VARATrainer
from src.training.vara_v2_trainer import VARAV2Trainer
from src.utils.config import deep_update, load_config, save_config
from src.utils.io import save_json


DEFAULT_REYNOLDS = [100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3200]
EXPLORATORY_REYNOLDS = [4000, 5000, 7500, 10000]
PRESET_PATHS = {
    "fast_screen": "configs/vara_v2/presets/fast_screen.yaml",
    "diagnostic": "configs/vara_v2/presets/diagnostic.yaml",
    "reliable": "configs/vara_v2/presets/reliable.yaml",
    "final": "configs/vara_v2/presets/final.yaml",
}
PRESET_STEPS = {
    "fast_screen": 1200,
    "diagnostic": 2000,
    "reliable": 4000,
    "final": 6000,
}


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
    parser.add_argument(
        "--data_supervision",
        choices=[
            "pure_pinn",
            "sparse_cfd",
            "sparse_cfd_polish",
            "full_cfd_oracle",
        ],
        default="pure_pinn",
    )
    parser.add_argument("--cfd_sample_fraction", type=float, default=0.01)
    parser.add_argument("--cfd_sample_count", type=int, default=None)
    parser.add_argument("--cfd_seed", type=int, default=None)
    parser.add_argument("--cfd_include_pressure", action="store_true")
    parser.add_argument("--cfd_include_vorticity", action="store_true")
    parser.add_argument(
        "--cfd_reference_dir",
        default="data/cavity_cfd",
        help="Deterministic cache root for generated per-Re CFD references.",
    )
    parser.add_argument(
        "--generate_missing_cfd_reference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Generate missing deterministic CFD references. Defaults on for "
            "sparse_cfd_polish and off for other supervision modes."
        ),
    )
    parser.add_argument("--output_dir", default="experiments/vara_v2/re_continuation")
    parser.add_argument("--enhanced_backbone", action="store_true")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_PATHS),
        default=None,
        help="Shared runtime/accuracy preset applied identically to all methods.",
    )
    parser.add_argument(
        "--reliable",
        action="store_true",
        help="Use the physically guarded cavity continuation protocol.",
    )
    parser.add_argument(
        "--cavity_base_formulation",
        choices=[
            "uvp_velocity_lift",
            "uvp_soft_bc",
            "hard_boundary_streamfunction_pressure",
        ],
        default=None,
        help="Override the shared cavity formulation for both Vanilla and VARA.",
    )
    parser.add_argument(
        "--continue_on_invalid",
        action="store_true",
        help="Continue a method chain after a stage fails reference-free validity checks.",
    )
    parser.add_argument(
        "--gate_vara_on_vanilla",
        action="store_true",
        help="At each Re, skip VARA V2 unless the matched Vanilla stage is valid.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--disable_stabilizers",
        action="store_true",
        help="Ablation: retain the formulation/budget but disable numerical stabilizers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate output_dir. Otherwise non-empty outputs are rejected.",
    )
    args = parser.parse_args()
    run(args)


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    base = _load_base_config(args.config)
    base = deep_update(base, load_config("configs/vara_v2/controller.yaml"))
    base = deep_update(base, load_config("configs/vara_v2/continuation.yaml"))
    if args.reliable:
        base = deep_update(base, load_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml"))
    preset = getattr(args, "preset", None)
    if preset:
        base = deep_update(base, load_config(PRESET_PATHS[str(preset)]))
    cavity_base_formulation = getattr(args, "cavity_base_formulation", None)
    if cavity_base_formulation:
        base["cavity_base_formulation"] = str(cavity_base_formulation)
    base = _apply_data_supervision_settings(
        base,
        mode=str(getattr(args, "data_supervision", "pure_pinn")),
        sample_fraction=float(getattr(args, "cfd_sample_fraction", 0.01)),
        sample_count=getattr(args, "cfd_sample_count", None),
        cfd_seed=getattr(args, "cfd_seed", None),
        include_pressure=bool(
            getattr(args, "cfd_include_pressure", False)
        ),
        include_vorticity=bool(
            getattr(args, "cfd_include_vorticity", False)
        ),
        formulation_explicit=bool(cavity_base_formulation),
    )
    if bool(getattr(args, "disable_stabilizers", False)):
        base = _without_cavity_stabilizers(base)
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
                "compute_budget": {
                    "enabled": True,
                    "type": "applied_optimizer_steps",
                    "value": 4,
                },
            },
        )
    if args.reliable and not args.quick:
        for reynolds in args.reynolds:
            _validate_reliable_config(
                _apply_re_aware_cavity_settings(base, float(reynolds)),
                preset,
                require_materialized=True,
            )
    if args.device:
        base["device"] = args.device
    reference_map = _load_full_field_reference_map(args.full_field_reference_map)
    supervision_mode = str(
        base.get("data_supervision", {}).get("mode", "pure_pinn")
    )
    generate_missing = getattr(args, "generate_missing_cfd_reference", None)
    if generate_missing is None:
        generate_missing = supervision_mode == "sparse_cfd_polish"
    resolved_references = {
        float(reynolds): _resolve_cfd_reference_for_re(
            float(reynolds),
            mode=supervision_mode,
            reference_map=reference_map,
            reference_dir=getattr(args, "cfd_reference_dir", "data/cavity_cfd"),
            generate_missing=bool(generate_missing),
        )
        for reynolds in args.reynolds
    }
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        if not bool(getattr(args, "overwrite", False)):
            raise SystemExit(
                f"Output directory already exists and is not empty: {output}. "
                "Use --overwrite or choose a new --output_dir."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        previous: dict[str, Path | None] = {method: None for method in args.methods}
        failed_methods: set[str] = set()
        for reynolds in args.reynolds:
            per_method: dict[str, dict[str, Any]] = {}
            re_name = f"re_{int(round(reynolds)):04d}"
            re_base = _apply_re_aware_cavity_settings(base, float(reynolds))
            cfd_reference = resolved_references[float(reynolds)]
            for method in args.methods:
                if method in failed_methods:
                    continue
                if (
                    method == "vara_v2"
                    and bool(getattr(args, "gate_vara_on_vanilla", False))
                    and (
                        "vanilla" not in per_method
                        or not bool(
                            per_method["vanilla"].get(
                                "continuation_stage_valid", False
                            )
                        )
                    )
                ):
                    print(
                        f"seed={seed} Re={reynolds:g}: skipping VARA V2 because "
                        "matched Vanilla did not pass."
                    )
                    continue
                method_dir = output / f"seed_{seed}" / re_name / method
                config = deepcopy(re_base)
                config["seed"] = int(seed)
                config["experiments"] = {
                    **config.get("experiments", {}),
                    "root": str(method_dir),
                    "flat_layout": True,
                }
                reference = _reference_for_re(float(reynolds), "ghia", None)
                full_field = cfd_reference["path"]
                config["benchmark_params"] = {
                    **config.get("benchmark_params", {}),
                    "reynolds": float(reynolds),
                    "reference": reference["reference"],
                    "reference_path": reference.get("reference_path"),
                    "full_field_reference_path": str(full_field) if full_field is not None else None,
                    "profile_only": full_field is None,
                }
                config["data_supervision"] = {
                    **config.get("data_supervision", {}),
                    "reference_path": (
                        str(full_field) if full_field is not None else None
                    ),
                    "reference_reynolds": float(reynolds),
                    "reference_generated": bool(cfd_reference["generated"]),
                    "reference_resolution": cfd_reference["resolution"],
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
                validity = _continuation_validity(metrics, config)
                metrics.update(validity)
                checkpoint = trainer.checkpoint_dir / "final.pt"
                if validity["continuation_stage_valid"] or args.continue_on_invalid or not args.reliable:
                    previous[method] = checkpoint
                else:
                    failed_methods.add(method)
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
                if not validity["continuation_stage_valid"]:
                    print(
                        f"  INVALID continuation stage: {validity['continuation_invalid_reasons']}"
                    )

            if "vanilla" in per_method and "vara_v2" in per_method:
                pair_rows = _comparison_rows(
                    seed,
                    reynolds,
                    per_method["vanilla"],
                    per_method["vara_v2"],
                )
                vanilla_valid = bool(per_method["vanilla"]["continuation_stage_valid"])
                vara_valid = bool(per_method["vara_v2"]["continuation_stage_valid"])
                pair_valid = vanilla_valid and vara_valid
                for pair_row in pair_rows:
                    pair_row.update(
                        {
                            "vanilla_stage_valid": vanilla_valid,
                            "vara_stage_valid": vara_valid,
                            "comparison_stage_valid": pair_valid,
                        }
                    )
                comparison_rows.extend(pair_rows)
                # Reuse the established Vanilla/VARA comparison renderer.
                comparison_dir = output / f"seed_{seed}" / re_name
                _save_per_re_comparison(
                    comparison_dir,
                    per_method["vanilla"],
                    per_method["vara_v2"],
                )
                save_json(
                    {
                        "comparison_stage_valid": pair_valid,
                        "vanilla_stage_valid": vanilla_valid,
                        "vara_stage_valid": vara_valid,
                        "vanilla_invalid_reasons": per_method["vanilla"][
                            "continuation_invalid_reasons"
                        ],
                        "vara_invalid_reasons": per_method["vara_v2"][
                            "continuation_invalid_reasons"
                        ],
                    },
                    comparison_dir / "comparison" / "continuation_validity.json",
                )

    summary = output / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    comparisons_all = pd.DataFrame(comparison_rows)
    if "comparison_stage_valid" in comparisons_all:
        comparisons = comparisons_all[
            comparisons_all["comparison_stage_valid"].fillna(False).astype(bool)
        ].copy()
    else:
        comparisons = comparisons_all.copy()
    raw.to_csv(summary / "continuation_results_long.csv", index=False)
    comparisons_all.to_csv(
        summary / "vara_v2_vs_vanilla_by_re_all_stages.csv",
        index=False,
    )
    comparisons.to_csv(summary / "vara_v2_vs_vanilla_by_re.csv", index=False)
    _wide_improvement(comparisons).to_csv(summary / "improvement_percent_by_re.csv", index=False)
    _save_summary_bar_plots(comparisons, summary)
    _save_validity_aware_montages(output, raw, summary)
    save_config(base, summary / "resolved_base_config.yaml")
    return {"raw": raw, "comparisons": comparisons}


def _without_cavity_stabilizers(config: dict[str, Any]) -> dict[str, Any]:
    weights = dict(config.get("training", {}).get("weights", {}))
    for name in (
        "speed_cap",
        "raw_psi_l2",
        "raw_psi_mean_l2",
        "scaled_correction_mean_l2",
        "scaled_correction_abs_max_hinge",
        "top_reverse_u",
        "bottom_positive_u",
        "raw_pde_tail",
        "pressure_gradient_l2",
        "vorticity_smoothness",
        "near_wall_vorticity_l2",
    ):
        weights[name] = 0.0
    return deep_update(
        config,
        {
            "training": {
                "residual_loss_mode": "mse",
                "weights": weights,
            },
            "losses": {
                "speed_cap": {"enabled": False},
                "raw_psi_l2": {"enabled": False},
                "correction_bubble": {"enabled": False},
                "lid_shear_direction": {"enabled": False},
                "raw_residual_tail": {"enabled": False},
                "pressure_gradient_l2": {"enabled": False},
                "vorticity_smoothness": {"enabled": False},
                "near_wall_vorticity_l2": {"enabled": False},
                "near_wall_momentum": {"enabled": False},
            },
            "cavity_curriculum": {"enabled": False},
        },
    )


def _load_base_config(config_path: str | Path) -> dict[str, Any]:
    """Load continuation config, accepting overlay-only YAML files.

    The reliable cavity protocol is intentionally an overlay so it cannot
    disturb the base lid-cavity config. In notebooks it is easy to pass that
    overlay through --config, so merge benchmark-free configs onto the default
    lid-driven-cavity base instead of failing later during model construction.
    """
    config = load_config(config_path)
    if "benchmark" in config:
        return config
    default_base = load_config("configs/vara_v2/lid_driven_cavity.yaml")
    return deep_update(default_base, config)


def _apply_data_supervision_settings(
    config: dict[str, Any],
    *,
    mode: str,
    sample_fraction: float = 0.01,
    sample_count: int | None = None,
    cfd_seed: int | None = None,
    include_pressure: bool = False,
    include_vorticity: bool = False,
    formulation_explicit: bool = False,
) -> dict[str, Any]:
    mode = str(mode).lower()
    if mode not in {
        "pure_pinn",
        "sparse_cfd",
        "sparse_cfd_polish",
        "full_cfd_oracle",
    }:
        raise ValueError(f"Unsupported data supervision mode: {mode}")
    resolved = deep_update(
        config,
        {
            "data_supervision": {
                "mode": mode,
                "sample_fraction": float(sample_fraction),
                "sample_count": (
                    None if sample_count is None else int(sample_count)
                ),
                "seed": int(
                    config.get("seed", 0) if cfd_seed is None else cfd_seed
                ),
                "include_pressure": bool(include_pressure),
                "include_vorticity": bool(include_vorticity),
                "formulation_explicit_override": bool(
                    formulation_explicit
                ),
                "oracle": mode == "full_cfd_oracle",
                "oracle_label": (
                    "upper_bound_only"
                    if mode == "full_cfd_oracle"
                    else ""
                ),
                "sparse_cfd_polish_enabled": (
                    mode == "sparse_cfd_polish"
                ),
            }
        },
    )
    if mode == "sparse_cfd_polish":
        resolved = deep_update(
            resolved,
            {
                "data_supervision": {
                    "cfd": {"sampling": {"mode": "mixed"}}
                }
            },
        )
    if mode in {"sparse_cfd", "sparse_cfd_polish"} and not formulation_explicit:
        resolved["cavity_base_formulation"] = "uvp_soft_bc"
    return resolved


def _apply_re_aware_cavity_settings(
    config: dict[str, Any],
    reynolds: float,
) -> dict[str, Any]:
    schedule = dict(config.get("re_aware_cavity", {}))
    regimes = list(schedule.get("regimes", []))
    if not bool(schedule.get("enabled", False)) or not regimes:
        return deepcopy(config)
    if not np.isfinite(reynolds) or reynolds <= 0.0:
        raise ValueError(f"Reynolds number must be positive and finite, got {reynolds!r}.")

    regime = regimes[-1]
    for candidate in regimes:
        if reynolds <= float(candidate.get("max_re", np.inf)):
            regime = candidate
            break

    band_cfg = dict(schedule.get("near_wall_band", {}))
    band = float(band_cfg.get("scale", 1.2)) / math.sqrt(reynolds)
    band = min(
        max(band, float(band_cfg.get("min", 0.025))),
        float(band_cfg.get("max", 0.12)),
    )
    total_steps = int(config.get("controller_v2", {}).get("total_steps", 4000))
    cavity_until = [
        max(1, int(round(total_steps * fraction)))
        for fraction in (0.25, 0.75, 1.0)
    ]
    wall_until = [
        max(1, int(round(total_steps * fraction)))
        for fraction in (0.20, 0.45, 0.75, 1.0)
    ]
    corner_widths = [float(value) for value in regime["corner_widths"]]
    lid_powers = [int(value) for value in regime["lid_vertical_powers"]]
    correction_scales = [float(value) for value in regime["correction_scales"]]
    wall_weights = [float(value) for value in regime["near_wall_momentum_weights"]]
    topology = dict(regime.get("topology", {}))

    resolved = deep_update(
        config,
        {
            "benchmark_params": {
                "reynolds": float(reynolds),
                "lid_corner_regularization_width": corner_widths[-1],
            },
            "model": {
                "hard_boundary_corner_width": corner_widths[-1],
                "hard_boundary_lid_vertical_power": lid_powers[-1],
                "hard_boundary_correction_scale": correction_scales[-1],
            },
            "training": {
                "residual_loss_mode": {
                    "switch_step": max(1, int(round(total_steps * 0.95))),
                },
                "weights": {
                    "speed_cap": float(regime["speed_cap_weight"]),
                    "raw_psi_l2": float(regime["raw_psi_l2_weight"]),
                    "raw_psi_mean_l2": float(
                        regime["raw_psi_mean_l2_weight"]
                    ),
                    "scaled_correction_mean_l2": float(
                        regime["scaled_correction_mean_l2_weight"]
                    ),
                    "scaled_correction_abs_max_hinge": float(
                        regime["scaled_correction_abs_max_hinge_weight"]
                    ),
                    "top_reverse_u": float(regime["top_reverse_u_weight"]),
                    "bottom_positive_u": float(
                        regime["bottom_positive_u_weight"]
                    ),
                    "raw_pde_tail": float(regime["raw_pde_tail_weight"]),
                    "near_wall_vorticity_l2": float(
                        regime["near_wall_vorticity_l2_weight"]
                    ),
                },
            },
            "losses": {
                "correction_bubble": {
                    "enabled": True,
                    "abs_max_cap": float(
                        regime["scaled_correction_abs_max_cap"]
                    ),
                },
                "lid_shear_direction": {
                    "enabled": bool(
                        float(regime["top_reverse_u_weight"]) > 0.0
                        or float(regime["bottom_positive_u_weight"]) > 0.0
                    ),
                    "band_width": band,
                    "corner_width": corner_widths[-1],
                    "bottom_u_tolerance": (
                        0.04 if float(reynolds) <= 200.0 else 0.075
                    ),
                },
                "raw_residual_tail": {
                    "enabled": True,
                    "threshold": float(regime["raw_pde_tail_threshold"]),
                    "core_margin": 0.08,
                    "core_emphasis": float(
                        regime["raw_pde_tail_core_emphasis"]
                    ),
                },
                "near_wall_momentum": {
                    "stages": [
                        {
                            "until_step": until_step,
                            "band_width": band,
                            "weight": weight,
                        }
                        for until_step, weight in zip(wall_until, wall_weights)
                    ],
                },
                "near_wall_vorticity_l2": {"band_width": band},
            },
            "cavity_curriculum": {
                "stages": [
                    {
                        "until_step": until_step,
                        "corner_width": corner_width,
                        "lid_vertical_power": lid_power,
                        "correction_scale": correction_scale,
                    }
                    for until_step, corner_width, lid_power, correction_scale in zip(
                        cavity_until,
                        corner_widths,
                        lid_powers,
                        correction_scales,
                    )
                ],
            },
            "sampling": {
                "cavity_boundary": {"corner_width": corner_widths[-1]},
                "cavity_near_wall": {
                    "enabled": True,
                    "fraction": float(regime["near_wall_fraction"]),
                    "band_width": band,
                },
            },
            "evaluation": {
                "controller_streamfunction_metrics": bool(reynolds <= 200.0),
            },
            "continuation_validity": {
                "max_lid_cavity_primary_center_error": float(
                    topology["max_primary_center_error"]
                ),
                "max_lid_cavity_topology_score": float(
                    topology["max_topology_score"]
                ),
                "require_lid_cavity_topology_alignment": bool(
                    topology["require_alignment"]
                ),
                "max_detected_vortices": int(topology["max_detected_vortices"]),
            },
        },
    )
    selected_formulation = str(
        resolved.get("cavity_base_formulation", "")
    ).lower()
    if selected_formulation not in {"uvp_soft_bc", "uvp_velocity_lift"}:
        return resolved
    lifted = selected_formulation == "uvp_velocity_lift"
    uvp_cfg = dict(
        resolved.get(
            "uvp_velocity_lift" if lifted else "uvp_soft_bc",
            {},
        )
    )
    diagnostic_budget = total_steps <= PRESET_STEPS["diagnostic"]
    boundary_key = (
        "diagnostic_boundary_samples"
        if diagnostic_budget
        else "reliable_boundary_samples"
    )
    boundary_count = int(uvp_cfg.get(boundary_key, 384 if diagnostic_budget else 512))
    validity_key = "diagnostic_validity" if diagnostic_budget else "reliable_validity"
    uvp_validity = dict(uvp_cfg.get(validity_key, {}))
    checkpoint_weights = dict(uvp_cfg.get("checkpoint_metric_weights", {}))
    component_relative_weights = {
        str(name): float(weight)
        for name, weight in dict(
            uvp_cfg.get("boundary_component_relative_weights", {})
        ).items()
    }
    uvp_widths = dict(uvp_cfg.get("lid_corner_regularization_widths", {}))
    uvp_corner_width = float(
        uvp_widths.get(
            "low_re" if reynolds <= 200.0 else "mid_re" if reynolds <= 600.0 else "high_re",
            0.05 if reynolds <= 200.0 else 0.04 if reynolds <= 600.0 else 0.03,
        )
    )
    repair_epochs = int(
        uvp_cfg.get(
            "diagnostic_polish_steps" if diagnostic_budget else "reliable_polish_steps",
            15 if diagnostic_budget else 40,
        )
    )
    collocation_stages = []
    for stage in resolved.get("training", {}).get("collocation_curriculum", {}).get(
        "stages", []
    ):
        collocation_stages.append({**stage, "n_boundary": boundary_count})
    materialized = deep_update(
        resolved,
        {
            "model": {
                "physics_formulation": (
                    "cavity_uvp_velocity_lift"
                    if lifted
                    else "cavity_uvp_soft_bc"
                ),
                "output_dim": 3,
                "uvp_velocity_lift_corner_width": uvp_corner_width,
                "uvp_velocity_lift_scale": float(
                    uvp_cfg.get("lift_scale", 16.0)
                ),
                "uvp_velocity_lift_vertical_power": int(
                    uvp_cfg.get("lift_vertical_power", 3)
                ),
            },
            "training": {
                "n_boundary": boundary_count,
                "skip_boundary_loss_if_hard_enforced": False,
                "collocation_curriculum": {"stages": collocation_stages},
                "weights": {
                    "pde": 0.0,
                    "momentum_u": 1.0,
                    "momentum_v": 1.0,
                    "continuity": float(uvp_cfg.get("continuity_weight", 5.0)),
                    "bc": float(uvp_cfg.get("boundary_weight", 25.0)),
                    "bc_uvp_balanced": float(
                        uvp_cfg.get("balanced_boundary_weight", 30.0)
                    ),
                    "top_band_pde": float(
                        uvp_cfg.get("top_band_pde_weight", 0.0)
                    ),
                    "top_band_momentum_u": float(
                        uvp_cfg.get("top_band_momentum_u_weight", 0.0)
                    ),
                    "top_band_continuity": float(
                        uvp_cfg.get("top_band_continuity_weight", 0.0)
                    ),
                    "upper_core_pde": float(
                        uvp_cfg.get("upper_core_pde_weight", 0.0)
                    ),
                    "right_wall_interior_pde": float(
                        uvp_cfg.get("right_wall_interior_pde_weight", 0.0)
                    ),
                    "lower_core_pde": float(
                        uvp_cfg.get("lower_core_pde_weight", 0.0)
                    ),
                    "speed_cap": 0.0,
                    "raw_psi_l2": 0.0,
                    "raw_psi_mean_l2": 0.0,
                    "scaled_correction_mean_l2": 0.0,
                    "scaled_correction_abs_max_hinge": 0.0,
                    "top_reverse_u": 0.0,
                    "bottom_positive_u": 0.0,
                    "raw_pde_tail": 0.07,
                    "pressure_gradient_l2": 0.0,
                    "near_wall_vorticity_l2": 0.0,
                    **{name: 0.0 for name in component_relative_weights},
                },
            },
            "pressure_gauge": {"weight": 0.001},
            "losses": {
                "speed_cap": {"enabled": False},
                "raw_psi_l2": {"enabled": False},
                "correction_bubble": {"enabled": False},
                "lid_shear_direction": {"enabled": False},
                "pressure_gradient_l2": {"enabled": False},
                "near_wall_momentum": {"enabled": False},
                "near_wall_vorticity_l2": {"enabled": False},
                "uvp_boundary_balance": {
                    "enabled": not lifted,
                    "relative_weights": component_relative_weights,
                },
                "uvp_regional_residuals": {"enabled": lifted},
                "raw_residual_tail": {
                    "enabled": True,
                    "threshold": 0.525,
                    "core_margin": 0.08,
                    "core_emphasis": 1.5,
                },
            },
            "cavity_curriculum": {"enabled": False},
            "checkpoint": {
                "require_final_cavity_stage": False,
                "reference_free_metrics": [
                    "pde_residual_mean",
                    "momentum_residual_mean",
                    "core_pde_residual_mean",
                    "continuity_residual_mean",
                    "boundary_condition_error",
                    "u_boundary_rmse",
                    "near_wall_pde_residual_mean",
                    "top_band_pde_residual_mean",
                    "top_band_momentum_u_mean",
                    "upper_core_pde_residual_mean",
                    *(
                        []
                        if lifted
                        else [
                            "near_wall_momentum_v_mean",
                            "top_band_continuity_residual_mean",
                        ]
                    ),
                ],
                "metric_weights": checkpoint_weights,
                "low_re_vortex_tiebreaker": {"enabled": False},
            },
            "optimizer": {
                "final_repair": {
                    "enabled": True,
                    "score_metric": "uvp_reference_free_score",
                    "epochs": repair_epochs,
                    "lr": 0.08 if diagnostic_budget else 0.10,
                    "max_iter": 5,
                    "max_eval": 10,
                    "batch_multiplier": 1.5,
                    "residual_fraction": 0.25,
                    "acceptance_tolerance": 0.0,
                    "pareto_guard": {
                        "enabled": True,
                        "relative_tolerance": 0.05,
                        "metrics": [
                            "pde_residual_mean",
                            "momentum_residual_mean",
                            "core_pde_residual_mean",
                            "continuity_residual_mean",
                            "speed_pred_max",
                            "top_band_pde_residual_mean",
                            "top_band_momentum_u_mean",
                            "upper_core_pde_residual_mean",
                            "near_wall_pde_residual_mean",
                        ],
                        "validity_gate_metrics": [
                            "pde_residual_mean",
                            "momentum_residual_mean",
                            "core_pde_residual_mean",
                            "continuity_residual_mean",
                        ],
                    },
                },
            },
            "benchmark_params": {
                "lid_corner_regularization_width": uvp_corner_width,
            },
            "sampling": {
                "cavity_boundary": {
                    "mode": "focused",
                    "lid_fraction": 0.31,
                    "corner_fraction": 0.10,
                    "corner_width": uvp_corner_width,
                },
                "cavity_lid_interior_band": {
                    "enabled": bool(not lifted and reynolds <= 200.0),
                    "fraction": 0.20 if reynolds <= 200.0 else 0.0,
                    "y_min": 0.72,
                    "y_max": 0.98,
                    "corner_width": uvp_corner_width,
                    "minimum_core_fraction": 0.35,
                },
                "cavity_circulation_bands": {
                    "enabled": bool(lifted and reynolds <= 200.0),
                    "top_band_fraction": 0.20,
                    "right_band_fraction": 0.15,
                    "lower_band_fraction": 0.10,
                    "left_band_fraction": 0.10,
                },
                "cavity_near_wall": {
                    "enabled": not lifted,
                    "fraction": (
                        0.0
                        if lifted
                        else float(regime["near_wall_fraction"])
                    ),
                    "band_width": band,
                },
            },
            "evaluation": {"controller_streamfunction_metrics": False},
            "continuation_validity": uvp_validity,
        },
    )
    supervision = dict(materialized.get("data_supervision", {}))
    supervision_mode = str(supervision.get("mode", "pure_pinn"))
    if supervision_mode in {
        "sparse_cfd",
        "sparse_cfd_polish",
        "full_cfd_oracle",
    }:
        include_pressure = bool(supervision.get("include_pressure", False))
        include_vorticity = bool(
            supervision.get("include_vorticity", False)
        )
        sparse_metrics = [
            "pde_residual_mean",
            "momentum_residual_mean",
            "continuity_residual_mean",
            "core_pde_residual_mean",
            "boundary_condition_error",
            "cfd_velocity_mse_sparse",
        ]
        materialized = deep_update(
            materialized,
            {
                "training": {
                    "weights": {
                        "continuity": float(
                            supervision.get(
                                "sparse_continuity_weight", 6.0
                            )
                        ),
                        "cfd_velocity_mse": float(
                            supervision.get(
                                "sparse_cfd_velocity_weight", 3.0
                            )
                        ),
                        "cfd_u_mse": 0.0,
                        "cfd_v_mse": 0.0,
                        "cfd_p_mse": 1.0 if include_pressure else 0.0,
                        "cfd_omega_mse": (
                            1.0 if include_vorticity else 0.0
                        ),
                        "raw_pde_tail": float(
                            supervision.get(
                                "sparse_raw_pde_tail_weight", 0.04
                            )
                        ),
                    }
                },
                "checkpoint": {
                    "reference_free_metrics": sparse_metrics,
                    "metric_weights": {
                        "pde_residual_mean": 1.5,
                        "momentum_residual_mean": 1.5,
                        "continuity_residual_mean": 1.5,
                        "core_pde_residual_mean": 1.5,
                        "boundary_condition_error": 0.5,
                        "cfd_velocity_mse_sparse": 1.0,
                    },
                    "low_re_vortex_tiebreaker": {"enabled": False},
                },
                "optimizer": {
                    "final_repair": {
                        "pareto_guard": {
                            "metrics": sparse_metrics,
                        }
                    }
                },
            },
        )
    if supervision_mode == "sparse_cfd_polish":
        polish = dict(supervision.get("polish", {}))
        boundary_multiplier = float(
            polish.get("balanced_boundary_multiplier", 1.12)
        )
        top_u_multiplier = float(
            polish.get("top_u_relative_multiplier", 1.12)
        )
        continuity_multiplier = float(
            polish.get("continuity_multiplier", 1.12)
        )
        relative_weights = dict(
            materialized.get("losses", {})
            .get("uvp_boundary_balance", {})
            .get("relative_weights", {})
        )
        relative_weights["bc_top_u"] = float(
            relative_weights.get("bc_top_u", 1.0)
        ) * top_u_multiplier
        guard_metrics = [
            "pde_residual_mean",
            "momentum_residual_mean",
            "continuity_residual_mean",
            "boundary_condition_error",
            "cfd_velocity_mse_sparse",
            "speed_pred_max",
        ]
        checkpoint_metrics = [
            "pde_residual_mean",
            "momentum_residual_mean",
            "continuity_residual_mean",
            "core_pde_residual_mean",
            "boundary_condition_error",
            "top_band_continuity_residual_mean",
            "top_band_pde_residual_mean",
            "upper_core_pde_residual_mean",
            "cfd_velocity_mse_sparse",
        ]
        materialized = deep_update(
            materialized,
            {
                "training": {
                    "weights": {
                        "continuity": float(
                            materialized["training"]["weights"][
                                "continuity"
                            ]
                        )
                        * continuity_multiplier,
                        "bc_uvp_balanced": float(
                            materialized["training"]["weights"][
                                "bc_uvp_balanced"
                            ]
                        )
                        * boundary_multiplier,
                        "cfd_velocity_mse": 0.0,
                        "cfd_u_mse": float(
                            polish.get("cfd_u_weight", 3.0)
                        ),
                        "cfd_v_mse": float(
                            polish.get("cfd_v_weight", 3.9)
                        ),
                        "top_band_continuity": float(
                            polish.get(
                                "top_band_continuity_weight", 0.35
                            )
                        ),
                        "top_band_pde": float(
                            polish.get("top_band_pde_weight", 0.15)
                        ),
                        "upper_core_pde": float(
                            polish.get("upper_core_pde_weight", 0.10)
                        ),
                    }
                },
                "losses": {
                    "uvp_boundary_balance": {
                        "relative_weights": relative_weights,
                    },
                    "uvp_regional_residuals": {"enabled": True},
                },
                "checkpoint": {
                    "reference_free_metrics": checkpoint_metrics,
                    "metric_weights": {
                        "pde_residual_mean": 1.5,
                        "momentum_residual_mean": 1.5,
                        "continuity_residual_mean": 2.0,
                        "core_pde_residual_mean": 1.25,
                        "boundary_condition_error": 1.0,
                        "top_band_continuity_residual_mean": 1.25,
                        "top_band_pde_residual_mean": 0.5,
                        "upper_core_pde_residual_mean": 0.25,
                        "cfd_velocity_mse_sparse": 1.0,
                    },
                    "low_re_vortex_tiebreaker": {"enabled": False},
                },
                "optimizer": {
                    "final_repair": {
                        "enabled": True,
                        "epochs": int(
                            polish.get("final_repair_steps", 4)
                        ),
                        "weights": {
                            "pde": 0.0,
                            "momentum_u": 1.0,
                            "momentum_v": 1.0,
                            "continuity": float(
                                materialized["training"]["weights"][
                                    "continuity"
                                ]
                            ),
                            "bc": 0.0,
                            "bc_uvp_balanced": float(
                                materialized["training"]["weights"][
                                    "bc_uvp_balanced"
                                ]
                            ),
                            "cfd_velocity_mse": 0.0,
                            "cfd_u_mse": float(
                                polish.get("cfd_u_weight", 3.0)
                            ),
                            "cfd_v_mse": float(
                                polish.get("cfd_v_weight", 3.9)
                            ),
                            "raw_pde_tail": float(
                                materialized["training"]["weights"].get(
                                    "raw_pde_tail", 0.04
                                )
                            ),
                            "top_band_pde": float(
                                polish.get("top_band_pde_weight", 0.15)
                            ),
                            "top_band_momentum_u": 0.0,
                            "top_band_continuity": float(
                                polish.get(
                                    "top_band_continuity_weight", 0.35
                                )
                            ),
                            "upper_core_pde": float(
                                polish.get("upper_core_pde_weight", 0.10)
                            ),
                            "right_wall_interior_pde": 0.0,
                            "lower_core_pde": 0.0,
                            "speed_cap": 0.0,
                            "top_reverse_u": 0.0,
                            "bottom_positive_u": 0.0,
                            "pressure_gradient_l2": 0.0,
                            "near_wall_vorticity_l2": 0.0,
                        },
                        "pareto_guard": {
                            "enabled": True,
                            "relative_tolerance": 0.05,
                            "metrics": guard_metrics,
                            "validity_gate_metrics": [
                                "pde_residual_mean",
                                "momentum_residual_mean",
                                "continuity_residual_mean",
                            ],
                        },
                    }
                },
            },
        )
    return materialized


def _validate_reliable_config(
    config: dict[str, Any],
    preset: str | None,
    *,
    require_materialized: bool = False,
) -> None:
    expected_steps = PRESET_STEPS.get(str(preset), 4000)
    model_cfg = dict(config.get("model", {}))
    train_cfg = dict(config.get("training", {}))
    controller_cfg = dict(config.get("controller_v2", {}))
    scheduler_cfg = dict(config.get("optimizer", {}).get("scheduler", {}))
    failures = []
    formulation = str(model_cfg.get("physics_formulation", ""))
    allowed_formulations = {
        "cavity_uvp_velocity_lift",
        "cavity_uvp_soft_bc",
        "hard_boundary_streamfunction_pressure",
    }
    if formulation not in allowed_formulations:
        failures.append(
            "physics_formulation must be cavity_uvp_velocity_lift, "
            "cavity_uvp_soft_bc, or "
            "hard_boundary_streamfunction_pressure"
        )
    if int(train_cfg.get("n_data", -1)) != 0:
        failures.append("training.n_data must be 0")
    skip_boundary = bool(train_cfg.get("skip_boundary_loss_if_hard_enforced", False))
    if formulation == "hard_boundary_streamfunction_pressure" and not skip_boundary:
        failures.append("hard-enforced boundary loss must be skipped")
    if formulation == "cavity_uvp_soft_bc":
        if int(model_cfg.get("output_dim", -1)) != 3:
            failures.append("cavity_uvp_soft_bc requires model.output_dim = 3")
        if skip_boundary:
            failures.append("cavity_uvp_soft_bc boundary loss must not be skipped")
        boundary_weights = dict(train_cfg.get("weights", {}))
        if not any(
            float(weight) > 0.0
            for name, weight in boundary_weights.items()
            if name == "bc" or name == "bc_uvp_balanced" or name.startswith("bc_")
        ):
            failures.append("cavity_uvp_soft_bc requires a positive boundary objective")
        if float(config.get("pressure_gauge", {}).get("weight", 0.0)) <= 0.0:
            failures.append("cavity_uvp_soft_bc requires a positive pressure gauge")
    if formulation == "cavity_uvp_velocity_lift":
        if int(model_cfg.get("output_dim", -1)) != 3:
            failures.append("cavity_uvp_velocity_lift requires model.output_dim = 3")
        if float(config.get("pressure_gauge", {}).get("weight", 0.0)) <= 0.0:
            failures.append("cavity_uvp_velocity_lift requires a positive pressure gauge")
    if int(controller_cfg.get("total_steps", -1)) != expected_steps:
        failures.append(f"controller_v2.total_steps must be {expected_steps}")
    if int(scheduler_cfg.get("total_steps", -1)) != expected_steps:
        failures.append(f"optimizer.scheduler.total_steps must be {expected_steps}")
    planned_steps = int(train_cfg.get("adaptive_cycles", 0)) * int(
        train_cfg.get("epochs_per_cycle", 0)
    )
    if planned_steps != expected_steps:
        failures.append(f"training cycle budget must equal {expected_steps}")
    schedule = dict(config.get("re_aware_cavity", {}))
    if not bool(schedule.get("enabled", False)) or not schedule.get("regimes"):
        failures.append("re_aware_cavity schedule must be enabled")
    near_wall = dict(config.get("sampling", {}).get("cavity_near_wall", {}))
    if require_materialized and formulation != "cavity_uvp_velocity_lift" and (
        not bool(near_wall.get("enabled", False))
        or float(near_wall.get("fraction", 0.0)) <= 0.0
        or float(near_wall.get("band_width", 0.0)) <= 0.0
    ):
        failures.append("Re-aware near-wall sampling must be materialized")
    if failures:
        raise ValueError("Invalid reliable cavity configuration: " + "; ".join(failures))


def _save_validity_aware_montages(
    output: Path,
    raw: pd.DataFrame,
    summary: Path,
) -> None:
    if raw.empty:
        return
    validity = raw.get(
        "continuation_stage_valid",
        pd.Series(True, index=raw.index),
    ).fillna(False).astype(bool)
    valid = raw[validity].copy()
    invalid = raw[~validity].copy()
    methods = sorted(str(value) for value in raw["method"].dropna().unique())
    for method in methods:
        items = []
        subset = valid[valid["method"] == method]
        for row in subset.to_dict("records"):
            path = Path(row["method_dir"]) / "figures" / "streamlines.png"
            items.append((path, _montage_label(path, method)))
        items.sort(key=lambda item: _continuation_sort_key(item[0]))
        _save_image_grid(
            items,
            summary / f"streamline_montage_{method}.png",
            cols=4,
            title=f"{method} valid continuation streamlines",
        )

    valid_pairs = set()
    for (seed, reynolds), group in valid.groupby(["seed", "reynolds"]):
        present = set(str(value) for value in group["method"])
        if {"vanilla", "vara_v2"}.issubset(present):
            valid_pairs.add((int(seed), float(reynolds)))
    comparison_items = []
    for seed, reynolds in sorted(valid_pairs):
        re_name = f"re_{int(round(reynolds)):04d}"
        path = output / f"seed_{seed}" / re_name / "comparison" / "streamlines_side_by_side.png"
        comparison_items.append((path, f"seed_{seed} {re_name} comparison"))
    _save_image_grid(
        comparison_items,
        summary / "streamline_montage_side_by_side.png",
        cols=2,
        title="Valid Vanilla vs VARA V2 streamlines",
    )

    invalid_items = []
    for row in invalid.to_dict("records"):
        path = Path(row["method_dir"]) / "figures" / "streamlines.png"
        label = (
            f"INVALID seed_{int(row['seed'])} "
            f"Re={float(row['reynolds']):g} {row['method']}"
        )
        invalid_items.append((path, label))
    invalid_items.sort(key=lambda item: _continuation_sort_key(item[0]))
    _save_image_grid(
        invalid_items,
        summary / "streamline_montage_invalid_stages.png",
        cols=4,
        title="Invalid stages - diagnostic use only",
    )


def _continuation_sort_key(path: Path) -> tuple[int, float, str]:
    seed_part = next((part for part in path.parts if part.startswith("seed_")), "seed_0")
    re_part = next((part for part in path.parts if part.startswith("re_")), "re_0")
    try:
        seed = int(seed_part.split("_", 1)[1])
    except ValueError:
        seed = 0
    try:
        reynolds = float(re_part.split("_", 1)[1].replace("p", "."))
    except ValueError:
        reynolds = float("inf")
    return seed, reynolds, str(path)


def _continuation_validity(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("continuation_validity", {}))
    if not bool(cfg.get("enabled", False)):
        return {
            "continuation_stage_valid": True,
            "continuation_invalid_reasons": "",
        }
    formulation = str(config.get("model", {}).get("physics_formulation", ""))
    checks = {
        "pde_residual_mean": float(cfg.get("max_pde_residual_mean", np.inf)),
        "continuity_residual_mean": float(cfg.get("max_continuity_residual_mean", np.inf)),
        "momentum_residual_mean": float(cfg.get("max_momentum_residual_mean", np.inf)),
        "boundary_condition_error": float(cfg.get("max_boundary_condition_error", np.inf)),
        "speed_pred_max": float(cfg.get("max_speed_pred", np.inf)),
        "lid_cavity_primary_center_error": float(
            cfg.get("max_lid_cavity_primary_center_error", np.inf)
        ),
        "lid_cavity_topology_score": float(
            cfg.get("max_lid_cavity_topology_score", np.inf)
        ),
        "near_wall_pde_residual_mean": float(
            cfg.get("max_near_wall_pde_residual_mean", np.inf)
        ),
        "near_wall_momentum_v_mean": float(
            cfg.get("max_near_wall_momentum_v_mean", np.inf)
        ),
        "core_pde_residual_mean": float(
            cfg.get("max_core_pde_residual_mean", np.inf)
        ),
    }
    if formulation not in {"cavity_uvp_soft_bc", "cavity_uvp_velocity_lift"}:
        checks["streamfunction_consistency_rmse"] = float(
            cfg.get("max_streamfunction_consistency_rmse", np.inf)
        )
    reasons = []
    for name, maximum in checks.items():
        try:
            value = float(metrics.get(name, np.nan))
        except (TypeError, ValueError):
            value = float("nan")
        if not np.isfinite(value):
            reasons.append(f"{name}=nonfinite")
        elif value > maximum:
            reasons.append(f"{name}={value:.4g}>{maximum:.4g}")
    if bool(cfg.get("require_lid_cavity_topology_alignment", False)):
        aligned = float(metrics.get("lid_cavity_topology_aligned", 0.0))
        if not np.isfinite(aligned) or aligned < 0.5:
            reasons.append("lid_cavity_topology_aligned=false")
    minimum_psi = float(cfg.get("min_primary_streamfunction_abs", 0.0))
    psi = float(metrics.get("primary_streamfunction_abs", np.nan))
    if not np.isfinite(psi):
        reasons.append("primary_streamfunction_abs=nonfinite")
    elif psi < minimum_psi:
        reasons.append(f"primary_streamfunction_abs={psi:.4g}<{minimum_psi:.4g}")
    minimum_speed = float(cfg.get("min_speed_pred_mean", 0.0))
    speed_mean = float(metrics.get("speed_pred_mean", np.nan))
    if not np.isfinite(speed_mean):
        reasons.append("speed_pred_mean=nonfinite")
    elif speed_mean < minimum_speed:
        reasons.append(f"speed_pred_mean={speed_mean:.4g}<{minimum_speed:.4g}")
    minimum_vortices = int(cfg.get("min_detected_vortices", 1))
    vortex_count = int(metrics.get("detected_vortex_count", 0))
    if vortex_count < minimum_vortices:
        reasons.append(f"detected_vortex_count={vortex_count}<{minimum_vortices}")
    maximum_vortices = int(cfg.get("max_detected_vortices", 10**9))
    if vortex_count > maximum_vortices:
        reasons.append(f"detected_vortex_count={vortex_count}>{maximum_vortices}")
    minimum_wall_distance = float(cfg.get("min_primary_vortex_wall_distance", 0.0))
    x = float(metrics.get("primary_vortex_center_x", np.nan))
    y = float(metrics.get("primary_vortex_center_y", np.nan))
    bounds = (
        float(config.get("benchmark_params", {}).get("x_min", 0.0)),
        float(config.get("benchmark_params", {}).get("x_max", 1.0)),
        float(config.get("benchmark_params", {}).get("y_min", 0.0)),
        float(config.get("benchmark_params", {}).get("y_max", 1.0)),
    )
    if not np.isfinite(x) or not np.isfinite(y):
        reasons.append("primary_vortex_center=nonfinite")
    else:
        x0, x1, y0, y1 = bounds
        scale = max(min(x1 - x0, y1 - y0), 1e-12)
        wall_distance = min(x - x0, x1 - x, y - y0, y1 - y) / scale
        if wall_distance < minimum_wall_distance:
            reasons.append(
                f"primary_vortex_wall_distance={wall_distance:.4g}<{minimum_wall_distance:.4g}"
            )
    return {
        "continuation_stage_valid": not reasons,
        "continuation_invalid_reasons": ";".join(reasons),
    }


def _resolve_cfd_reference_for_re(
    reynolds: float,
    *,
    mode: str,
    reference_map: dict[float, Path],
    reference_dir: str | Path,
    generate_missing: bool,
) -> dict[str, Any]:
    mapped = _full_field_reference_for_re(float(reynolds), reference_map)
    if mapped is not None:
        return _cfd_reference_info(mapped, reynolds, generated=False)

    requires_reference = mode in {
        "sparse_cfd",
        "sparse_cfd_polish",
        "full_cfd_oracle",
    }
    if not requires_reference:
        return {
            "path": None,
            "reynolds": float(reynolds),
            "generated": False,
            "resolution": None,
        }

    cache_path = _generated_cfd_reference_path(reference_dir, reynolds)
    if cache_path.exists():
        return _cfd_reference_info(cache_path, reynolds, generated=False)

    if not generate_missing:
        raise ValueError(_missing_cfd_reference_message(reynolds, cache_path))
    try:
        generated = generate_cavity_cfd_reference(reynolds, cache_path)
    except Exception as exc:
        raise RuntimeError(
            f"{_missing_cfd_reference_message(reynolds, cache_path)} "
            f"Automatic generation failed: {exc}"
        ) from exc
    return _cfd_reference_info(generated, reynolds, generated=True)


def _generated_cfd_reference_path(
    reference_dir: str | Path,
    reynolds: float,
) -> Path:
    root = Path(reference_dir)
    if not root.is_absolute():
        root = ROOT / root
    label = int(round(float(reynolds)))
    return root / f"re_{label:04d}" / "cavity_reference.npz"


def _cfd_reference_info(
    path: str | Path,
    reynolds: float,
    *,
    generated: bool,
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    reference = load_full_field_reference(resolved)
    nx = int(np.unique(reference["x"]).size)
    ny = int(np.unique(reference["y"]).size)
    return {
        "path": resolved,
        "reynolds": float(reynolds),
        "generated": bool(generated),
        "resolution": f"{nx}x{ny}",
    }


def _missing_cfd_reference_message(
    reynolds: float,
    expected_path: Path,
) -> str:
    return (
        f"No full-field CFD reference is available for requested Re={reynolds:g}. "
        f"Expected reference path: {expected_path}. Generate it with: "
        f"python -m src.physics.cavity_cfd_generator --reynolds {reynolds:g} "
        f"--output \"{expected_path}\"; or rerun continuation with "
        "--generate_missing_cfd_reference."
    )


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
