"""Reproducible publication suite for VARA Controller V2."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_runner import BENCHMARK_DEFAULTS
from scripts.run_modern_baselines import METHODS as MODERN_METHODS
from src.evaluation.statistical_tests import (
    holm_adjust,
    paired_bootstrap_improvement,
    paired_effect_size,
    wilcoxon_signed_rank,
)
from src.diagnostics import DiagnosticMapBuilder
from src.training.vara_trainer import VARATrainer
from src.training.vara_v2_trainer import VARAV2Trainer
from src.utils.config import deep_update, load_config, save_config
from src.utils.io import save_json


DEFAULT_METHODS = [
    "vanilla",
    "rar",
    "self_adaptive_attention",
    "gradient_balanced",
    "gradient_enhanced",
    "relobralo",
    "residual_attention",
    "vara_v1",
    "vara_v2",
]

REPORT_METRICS = [
    "velocity_full_rel_l2",
    "u_rel_l2",
    "v_rel_l2",
    "centerline_profile_score",
    "cavity_benchmark_score",
    "pde_residual_mean",
    "continuity_residual_mean",
    "momentum_residual_mean",
    "boundary_condition_error",
    "unweighted_validation_loss",
    "worst_patch_pde_residual",
    "worst_patch_continuity_residual",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=["core", "ablation", "generalization", "wall_clock", "physics_modules", "all"],
        default="core",
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--heldout_seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output_dir", default="experiments/vara_v2/publication_suite")
    parser.add_argument("--wall_clock_fraction", type=float, default=0.75)
    args = parser.parse_args()
    run_suite(args)


def run_suite(args: argparse.Namespace) -> pd.DataFrame:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    studies = {args.study} if args.study != "all" else {"core", "ablation", "generalization", "wall_clock"}

    if "core" in studies:
        for backbone in ("legacy_mlp", "residual_fourier_mlp"):
            case = _case_config("lid_cavity_re100", quick=args.quick)
            if backbone == "residual_fourier_mlp":
                case = deep_update(case, load_config("configs/vara_v2/enhanced_backbone.yaml"))
            rows.extend(
                _run_case(
                    case,
                    case_name="lid_cavity_re100",
                    study="core",
                    backbone=backbone,
                    methods=args.methods,
                    seeds=args.seeds,
                    output=output,
                    device=args.device,
                )
            )

    if "ablation" in studies:
        ablations = load_config("configs/vara_v2/ablations.yaml")["variants"]
        case = _case_config("lid_cavity_re100", quick=args.quick)
        methods = ["vanilla", "vara_v1", *list(ablations)]
        rows.extend(
            _run_case(
                case,
                case_name="lid_cavity_re100",
                study="ablation",
                backbone="legacy_mlp",
                methods=methods,
                seeds=args.seeds,
                output=output,
                device=args.device,
                ablations=ablations,
            )
        )

    if "generalization" in studies:
        cases = [
            "kovasznay_re20",
            "kovasznay_re40",
            "kovasznay_re80",
            "channel_re40",
            "double_vortex_re40",
            "taylor_green_re100",
            "lid_cavity_re100",
            "lid_cavity_re400",
            "lid_cavity_re1000",
            "lid_cavity_re1600",
            "lid_cavity_re3200",
        ]
        for case_name in cases:
            methods = list(args.methods)
            if case_name.startswith("taylor_green") and "causal" not in methods:
                methods.append("causal")
            if not case_name.startswith("taylor_green"):
                methods = [method for method in methods if method != "causal"]
            rows.extend(
                _run_case(
                    _case_config(case_name, quick=args.quick),
                    case_name=case_name,
                    study="generalization",
                    backbone="legacy_mlp",
                    methods=methods,
                    seeds=args.heldout_seeds,
                    output=output,
                    device=args.device,
                )
            )

    if "wall_clock" in studies:
        pilot_config = _case_config("lid_cavity_re100", quick=args.quick)
        pilot_rows = _run_case(
            pilot_config,
            case_name="lid_cavity_re100",
            study="wall_clock_pilot",
            backbone="legacy_mlp",
            methods=["vanilla"],
            seeds=args.seeds[: min(3, len(args.seeds))],
            output=output,
            device=args.device,
            budget_type="disabled",
        )
        rows.extend(pilot_rows)
        pilot_times = [float(row["training_wall_clock_sec"]) for row in pilot_rows]
        wall_budget = float(args.wall_clock_fraction) * min(pilot_times)
        wall_config = _case_config("lid_cavity_re100", quick=args.quick)
        rows.extend(
            _run_case(
                wall_config,
                case_name="lid_cavity_re100",
                study="wall_clock",
                backbone="legacy_mlp",
                methods=args.methods,
                seeds=args.seeds,
                output=output,
                device=args.device,
                budget_type="wall_clock_sec",
                budget_value=wall_budget,
            )
        )
        save_json(
            {
                "pilot_times_sec": pilot_times,
                "fraction": float(args.wall_clock_fraction),
                "binding_budget_sec": wall_budget,
            },
            output / "wall_clock" / "budget_selection.json",
        )

    if "physics_modules" in studies:
        formulations = {
            "direct": {},
            "cavity_hard_boundary": load_config("configs/vara_v2/physics_hard_boundary.yaml"),
            "streamfunction_pressure": load_config("configs/vara_v2/physics_streamfunction.yaml"),
        }
        for formulation, overlay in formulations.items():
            case = deep_update(_case_config("lid_cavity_re100", quick=args.quick), overlay)
            rows.extend(
                _run_case(
                    case,
                    case_name="lid_cavity_re100",
                    study="physics_modules",
                    backbone=f"legacy_mlp_{formulation}",
                    methods=args.methods,
                    seeds=args.seeds,
                    output=output,
                    device=args.device,
                )
            )

    raw = pd.DataFrame(rows)
    summary = output / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    raw.to_csv(summary / "publication_raw_results.csv", index=False)
    _mean_std(raw).to_csv(summary / "publication_mean_std.csv", index=False)
    paired = _paired_statistics(raw)
    paired.to_csv(summary / "paired_statistics.csv", index=False)
    _paired_metric_improvements(raw).to_csv(summary / "paired_metric_improvements.csv", index=False)
    ranks = _method_ranks(raw)
    ranks.to_csv(summary / "method_ranks.csv", index=False)
    _mechanism_table(raw).to_csv(summary / "vara_v2_mechanism_summary.csv", index=False)
    gates = _success_gates(raw, paired, ranks)
    pd.DataFrame(gates).to_csv(summary / "success_gates.csv", index=False)
    save_json({"gates": gates}, summary / "success_gates.json")
    _save_plots(raw, ranks, summary)
    _save_mechanism_plots(raw, summary)
    print(f"Saved publication suite: {summary}")
    return raw


def _run_case(
    base: dict[str, Any],
    case_name: str,
    study: str,
    backbone: str,
    methods: list[str],
    seeds: list[int],
    output: Path,
    device: str | None,
    ablations: dict[str, Any] | None = None,
    budget_type: str = "optimizer_steps",
    budget_value: float | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        for method in methods:
            config = deepcopy(base)
            config["seed"] = int(seed)
            config["device"] = device or config.get("device", "auto")
            run_root = output / study / case_name / backbone / method
            config["experiments"] = {**config.get("experiments", {}), "root": str(run_root)}
            step_budget = int(config.get("controller_v2", {}).get("total_steps", 400))
            if budget_type == "disabled":
                config["compute_budget"] = {"enabled": False}
            else:
                config["compute_budget"] = {
                    "enabled": True,
                    "type": budget_type,
                    "value": float(step_budget if budget_value is None else budget_value),
                }
            config = deep_update(config, {"optimizer": {"final_repair": {"enabled": False}}})
            if ablations and method in ablations:
                config = deep_update(config, ablations[method])
            trainer, resolved_method = _make_trainer(method, config)
            metrics = trainer.run()
            metrics.update(_worst_patch_metrics(trainer))
            primary_name, primary_value = _primary_metric(metrics)
            row = {
                **metrics,
                "study": study,
                "benchmark_case": case_name,
                "backbone": backbone,
                "method": resolved_method,
                "seed": int(seed),
                "primary_metric_name": primary_name,
                "primary_metric_value": primary_value,
                "run_dir": str(trainer.run_dir),
            }
            save_json(row, trainer.run_dir / "publication_row.json")
            rows.append(row)
            print(f"{study} {case_name} {backbone} seed={seed} method={method}")
    return rows


def _make_trainer(method: str, config: dict[str, Any]) -> tuple[Any, str]:
    if method == "vara_v2" or method.startswith("v2_"):
        # _case_config already installs the V2 defaults. Reapplying them here
        # would overwrite quick schedules and, more seriously, controller
        # ablation overlays applied by _run_case.
        return VARAV2Trainer(config), method
    if method == "vara_v1":
        return VARATrainer(config, mode="local_constrained_vara"), method
    if method not in MODERN_METHODS:
        raise ValueError(f"Unknown method {method!r}.")
    mode, overlay = MODERN_METHODS[method]
    if overlay:
        config = deep_update(config, load_config(overlay))
    return VARATrainer(config, mode=mode), method


def _case_config(name: str, quick: bool) -> dict[str, Any]:
    controller = load_config("configs/vara_v2/controller.yaml")
    if name.startswith("lid_cavity"):
        config = load_config("configs/lid_driven_cavity.yaml")
        reynolds = float(name.rsplit("re", 1)[1])
        reference_map = pd.read_csv("data/references/lid_driven_cavity/full_field/reference_map.csv")
        match = reference_map[np.isclose(reference_map["re"].astype(float), reynolds)]
        full_field = None if match.empty else str(match.iloc[0]["full_field_reference_path"])
        profile_reference = "ghia" if any(
            np.isclose(reynolds, available) for available in (100.0, 400.0, 1000.0)
        ) else "none"
        config["benchmark_params"] = {
            **config.get("benchmark_params", {}),
            "reynolds": reynolds,
            "reference": profile_reference,
            "full_field_reference_path": full_field,
            "profile_only": full_field is None,
        }
    elif name.startswith("kovasznay"):
        config = load_config("configs/kovasznay_debug.yaml")
        config["benchmark_params"]["reynolds"] = float(name.rsplit("re", 1)[1])
    elif name.startswith("channel"):
        config = deep_update(load_config("configs/kovasznay_debug.yaml"), BENCHMARK_DEFAULTS["channel_inflow_outflow"])
    elif name.startswith("double_vortex"):
        config = deep_update(load_config("configs/kovasznay_debug.yaml"), BENCHMARK_DEFAULTS["double_vortex_box"])
    elif name.startswith("taylor_green"):
        config = load_config("configs/taylor_green.yaml")
    else:
        raise ValueError(f"Unknown benchmark case {name!r}.")
    config = deep_update(config, controller)
    config["training"] = {
        **config.get("training", {}),
        "adaptive_cycles": 4,
        "epochs_per_cycle": 100,
        "log_every": 25,
    }
    if quick:
        config = deep_update(
            config,
            {
                "model": {"hidden_layers": [16, 16]},
                "training": {
                    "adaptive_cycles": 1,
                    "epochs_per_cycle": 4,
                    "n_collocation": 32,
                    "n_boundary": 24,
                    "n_data": 16 if config.get("benchmark") != "lid_driven_cavity" else 0,
                    "log_every": 1,
                },
                "validation": {"nx": 8, "ny": 8},
                "test": {"nx": 8, "ny": 8},
                "patches": {"nx_patches": 2, "ny_patches": 2, "nt_patches": 2 if config.get("benchmark") == "taylor_green" else 1},
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
    return config


def _primary_metric(metrics: dict[str, Any]) -> tuple[str, float]:
    value = _finite(metrics.get("velocity_full_rel_l2"))
    if value is not None:
        return "velocity_full_rel_l2", value
    profile = _finite(metrics.get("centerline_profile_score"))
    if profile is not None:
        return "centerline_profile_score", profile
    u = _finite(metrics.get("u_rel_l2"))
    v = _finite(metrics.get("v_rel_l2"))
    if u is not None and v is not None:
        return "mean_velocity_rel_l2", 0.5 * (u + v)
    fallback = _finite(metrics.get("unweighted_validation_loss"))
    return "unweighted_validation_loss", float("nan") if fallback is None else fallback


def _mean_std(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["study", "benchmark_case", "backbone", "method"]
    metrics = ["primary_metric_value", *REPORT_METRICS, "training_wall_clock_sec", "optimizer_steps"]
    for group_key, group in raw.groupby(keys, dropna=False):
        identity = dict(zip(keys, group_key))
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({**identity, "metric": metric, "mean": values.mean(), "std": values.std(), "count": len(values)})
    return pd.DataFrame(rows)


def _paired_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouping = ["study", "benchmark_case", "backbone"]
    for group_key, group in raw.groupby(grouping, dropna=False):
        v2 = group[group["method"] == "vara_v2"][["seed", "primary_metric_value"]].rename(
            columns={"primary_metric_value": "v2"}
        )
        if v2.empty:
            continue
        for method, method_group in group.groupby("method"):
            if method == "vara_v2":
                continue
            other = method_group[["seed", "primary_metric_value"]].rename(
                columns={"primary_metric_value": "baseline"}
            )
            paired = other.merge(v2, on="seed", how="inner").dropna()
            if paired.empty:
                continue
            bootstrap = paired_bootstrap_improvement(paired["baseline"], paired["v2"], seed=1729)
            wilcoxon = wilcoxon_signed_rank(paired["baseline"], paired["v2"])
            rows.append(
                {
                    **dict(zip(grouping, group_key)),
                    "baseline_method": method,
                    **bootstrap,
                    **wilcoxon,
                    "paired_effect_size_dz": paired_effect_size(paired["baseline"], paired["v2"]),
                    "v2_wins": int((paired["v2"] < paired["baseline"]).sum()),
                    "paired_seeds": int(len(paired)),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_adjusted_p"] = holm_adjust(result["p_value"].fillna(1.0).tolist())
    return result


def _paired_metric_improvements(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouping = ["study", "benchmark_case", "backbone"]
    for group_key, group in raw.groupby(grouping, dropna=False):
        v2_group = group[group["method"] == "vara_v2"]
        if v2_group.empty:
            continue
        for metric in REPORT_METRICS:
            if metric not in group:
                continue
            v2 = v2_group[["seed", metric]].rename(columns={metric: "v2"})
            for method, method_group in group.groupby("method"):
                if method == "vara_v2":
                    continue
                baseline = method_group[["seed", metric]].rename(columns={metric: "baseline"})
                paired = baseline.merge(v2, on="seed").dropna()
                if paired.empty:
                    continue
                result = paired_bootstrap_improvement(paired["baseline"], paired["v2"], seed=2718)
                rows.append(
                    {
                        **dict(zip(grouping, group_key)),
                        "metric": metric,
                        "baseline_method": method,
                        **result,
                        "v2_wins": int((paired["v2"] < paired["baseline"]).sum()),
                    }
                )
    return pd.DataFrame(rows)


def _method_ranks(raw: pd.DataFrame) -> pd.DataFrame:
    metric_rows = []
    keys = ["study", "benchmark_case", "backbone", "seed"]
    for key, group in raw.groupby(keys, dropna=False):
        values = group[["method", "primary_metric_value"]].dropna().copy()
        if values.empty:
            continue
        values["rank"] = values["primary_metric_value"].rank(method="average", ascending=True)
        for _, row in values.iterrows():
            metric_rows.append({**dict(zip(keys, key)), "method": row["method"], "rank": row["rank"]})
    ranks = pd.DataFrame(metric_rows)
    if ranks.empty:
        return ranks
    return (
        ranks.groupby(["study", "benchmark_case", "backbone", "method"], as_index=False)
        .agg(mean_rank=("rank", "mean"), rank_std=("rank", "std"), count=("rank", "size"))
    )


def _mechanism_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, run in raw[raw["method"].str.startswith("vara_v2")].iterrows():
        path = Path(run["run_dir"]) / "vara_v2_decisions.csv"
        if not path.exists():
            continue
        decisions = pd.read_csv(path)
        rows.append(
            {
                "benchmark_case": run["benchmark_case"],
                "backbone": run["backbone"],
                "method": run["method"],
                "seed": run["seed"],
                "candidates": len(decisions),
                "accepted": int(decisions.get("accepted", False).fillna(False).astype(bool).sum()),
                "prefiltered": int(decisions.get("prefiltered", False).fillna(False).astype(bool).sum()),
                "mean_reward_ratio": _numeric_column_mean(decisions, "reward_ratio"),
                "mean_predicted_guard_damage": _numeric_column_mean(decisions, "predicted_guard_damage"),
            }
        )
    return pd.DataFrame(rows)


def _success_gates(raw: pd.DataFrame, paired: pd.DataFrame, ranks: pd.DataFrame) -> list[dict[str, Any]]:
    gates = []
    if paired.empty:
        return [
            {"gate": "v2_beats_vanilla_7_of_10_with_positive_ci", "passed": False},
            {"gate": "v2_beats_vara_v1_7_of_10_with_positive_ci", "passed": False},
            {"gate": "v2_best_or_tied_average_rank", "passed": False},
            {"gate": "mean_boundary_degradation_at_most_2_percent", "passed": False},
            {"gate": "positive_direction_across_three_families", "passed": False},
        ]
    core = paired[
        (paired["study"] == "core")
        & (paired["benchmark_case"] == "lid_cavity_re100")
        & (paired["backbone"] == "legacy_mlp")
    ]
    for baseline in ("vanilla", "vara_v1"):
        row = core[core["baseline_method"] == baseline]
        passed = bool(
            not row.empty
            and int(row.iloc[0]["v2_wins"]) >= 7
            and float(row.iloc[0]["ci_low"]) > 0.0
        )
        gates.append({"gate": f"v2_beats_{baseline}_7_of_10_with_positive_ci", "passed": passed})
    v2_ranks = ranks[ranks["method"] == "vara_v2"]
    gates.append(
        {
            "gate": "v2_best_or_tied_average_rank",
            "passed": bool(not v2_ranks.empty and float(v2_ranks["mean_rank"].mean()) <= 1.5),
        }
    )
    cavity = raw[(raw["method"] == "vara_v2") & raw["benchmark_case"].str.startswith("lid_cavity")]
    vanilla = raw[(raw["method"] == "vanilla") & raw["benchmark_case"].str.startswith("lid_cavity")]
    paired_boundary = vanilla[["benchmark_case", "seed", "boundary_condition_error"]].merge(
        cavity[["benchmark_case", "seed", "boundary_condition_error"]],
        on=["benchmark_case", "seed"],
        suffixes=("_vanilla", "_v2"),
    )
    degradation = (
        100.0
        * (paired_boundary["boundary_condition_error_v2"] - paired_boundary["boundary_condition_error_vanilla"])
        / paired_boundary["boundary_condition_error_vanilla"].abs().clip(lower=1e-12)
        if not paired_boundary.empty
        else pd.Series(dtype=float)
    )
    gates.append({"gate": "mean_boundary_degradation_at_most_2_percent", "passed": bool(not degradation.empty and degradation.mean() <= 2.0)})
    improved_families = 0
    general = paired[(paired["study"] == "generalization") & (paired["baseline_method"] == "vanilla")]
    for family, family_rows in general.groupby(general["benchmark_case"].map(_family)):
        if family_rows["mean_improvement_percent"].mean() > 0.0:
            improved_families += 1
    gates.append({"gate": "positive_direction_across_three_families", "passed": improved_families >= 3})
    return gates


def _save_plots(raw: pd.DataFrame, ranks: pd.DataFrame, summary: Path) -> None:
    core = raw[(raw["study"] == "core") & (raw["backbone"] == "legacy_mlp")]
    if not core.empty:
        means = core.groupby("method")["primary_metric_value"].agg(["mean", "std"]).sort_values("mean")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(means.index, means["mean"], yerr=means["std"], color="#32667a", capsize=3)
        ax.set_ylabel("Primary metric (lower is better)")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(summary / "core_primary_metric_errorbar.png", dpi=220)
        plt.close(fig)
    if not ranks.empty:
        average = ranks.groupby("method")["mean_rank"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(average.index, average.values, color="#7a5232")
        ax.set_ylabel("Average rank (lower is better)")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(summary / "average_method_rank.png", dpi=220)
        plt.close(fig)


def _save_mechanism_plots(raw: pd.DataFrame, summary: Path) -> None:
    frames = []
    for _, run in raw[raw["method"].str.startswith("vara_v2")].iterrows():
        path = Path(run["run_dir"]) / "vara_v2_decisions.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["benchmark_case"] = run["benchmark_case"]
            frames.append(frame)
    if not frames:
        return
    decisions = pd.concat(frames, ignore_index=True)
    accepted = decisions.get("accepted", pd.Series(False, index=decisions.index)).fillna(False).astype(bool)
    prefiltered = decisions.get("prefiltered", pd.Series(False, index=decisions.index)).fillna(False).astype(bool)
    labels = ["accepted", "rolled back", "prefiltered"]
    counts = [
        int(accepted.sum()),
        int((~accepted & ~prefiltered).sum()),
        int(prefiltered.sum()),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, counts, color=["#2d7d46", "#b8463f", "#777777"])
    ax.set_ylabel("Candidate decisions")
    fig.tight_layout()
    fig.savefig(summary / "v2_decision_outcomes.png", dpi=220)
    plt.close(fig)

    predicted = pd.to_numeric(decisions.get("predicted_target_improvement"), errors="coerce")
    observed = pd.to_numeric(decisions.get("observed_target_improvement"), errors="coerce")
    valid = predicted.notna() & observed.notna()
    if valid.any():
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(predicted[valid], observed[valid], c=np.where(accepted[valid], "#2d7d46", "#b8463f"), alpha=0.75)
        low = min(float(predicted[valid].min()), float(observed[valid].min()))
        high = max(float(predicted[valid].max()), float(observed[valid].max()))
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted target improvement")
        ax.set_ylabel("Observed target improvement")
        fig.tight_layout()
        fig.savefig(summary / "v2_predicted_vs_observed_improvement.png", dpi=220)
        plt.close(fig)


def _family(case: str) -> str:
    if case.startswith("lid_cavity"):
        return "cavity"
    if case.startswith("kovasznay"):
        return "kovasznay"
    if case.startswith("taylor_green"):
        return "taylor_green"
    if case.startswith("channel"):
        return "channel"
    return "manufactured_vortex"


def _finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _worst_patch_metrics(trainer: Any) -> dict[str, float]:
    _x, _y, coords = trainer.test_grid()
    builder = trainer.diagnostic_builder()
    maps = builder.build(coords, mode="residual_only")
    patch_ids = trainer.patch_grid.assign_numpy(coords)
    result = {}
    for source, target in (
        ("pde_residual", "worst_patch_pde_residual"),
        ("continuity_residual", "worst_patch_continuity_residual"),
    ):
        values = np.asarray(maps[source], dtype=float).reshape(-1)
        patch_means = [
            float(np.nanmean(values[patch_ids == patch_id]))
            for patch_id in range(trainer.patch_grid.num_patches)
            if np.any(np.isfinite(values[patch_ids == patch_id]))
        ]
        result[target] = max(patch_means) if patch_means else float("nan")
    return result


def _numeric_column_mean(frame: pd.DataFrame, name: str) -> float:
    if name not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[name], errors="coerce").mean())


if __name__ == "__main__":
    main()
