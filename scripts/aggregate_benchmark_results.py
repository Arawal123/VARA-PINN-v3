"""Aggregate professor benchmark outputs into readable tables and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METRICS = [
    "u_full_rel_l2",
    "v_full_rel_l2",
    "velocity_full_rel_l2",
    "p_full_rel_l2_centered",
    "omega_full_rel_l2",
    "velocity_mag_rmse",
    "velocity_mag_mae",
    "u_rel_l2",
    "v_rel_l2",
    "p_rel_l2_centered",
    "omega_rel_l2",
    "u_rmse",
    "v_rmse",
    "p_rmse_centered",
    "omega_rmse",
    "u_pred_mean",
    "v_pred_mean",
    "p_pred_std_centered",
    "speed_pred_mean",
    "speed_pred_max",
    "omega_pred_abs_mean",
    "omega_pred_abs_max",
    "pde_residual_mean",
    "centerline_pde_residual_mean",
    "centerline_continuity_residual_mean",
    "corner_pde_residual_mean",
    "corner_boundary_error",
    "continuity_residual_mean",
    "momentum_residual_mean",
    "boundary_condition_error",
    "u_boundary_rmse",
    "v_boundary_rmse",
    "boundary_speed_rmse",
    "u_centerline_rmse",
    "v_centerline_rmse",
    "u_centerline_rel_l2",
    "v_centerline_rel_l2",
    "centerline_profile_score",
    "cavity_benchmark_score",
    "unweighted_data_loss",
    "unweighted_pde_loss",
    "unweighted_bc_loss",
    "unweighted_validation_loss",
    "final_total_loss",
    "J_score",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="experiments/logs")
    parser.add_argument("--output_dir", default="experiments/benchmark_summary")
    args = parser.parse_args()

    rows = collect_rows(Path(args.results_dir))
    if not rows:
        raise SystemExit(f"No summary.json files found under {args.results_dir}")
    df = pd.DataFrame(rows)
    out = Path(args.output_dir)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "combined_results.csv", index=False)
    methodwise = build_methodwise(df)
    methodwise.to_csv(out / "methodwise_mean_std_table.csv", index=False)
    methodwise.to_csv(out / "final_summary_table.csv", index=False)
    collapse = build_collapse(df)
    collapse.to_csv(out / "collapse_rate_table.csv", index=False)
    seedwise = build_seedwise(df)
    seedwise.to_csv(out / "seedwise_comparison_table.csv", index=False)
    cavity = build_cavity_table(methodwise)
    cavity.to_csv(out / "cavity_profile_summary_table.csv", index=False)
    build_readable(df).to_csv(out / "combined_results_readable.csv", index=False)
    build_readable(methodwise).to_csv(out / "methodwise_mean_std_readable.csv", index=False)
    build_readable(seedwise).to_csv(out / "seedwise_comparison_readable.csv", index=False)
    build_readable(cavity).to_csv(out / "cavity_profile_summary_readable.csv", index=False)
    save_bar_plots(methodwise, fig_dir)
    save_seedwise_scatter(df, fig_dir)

    print("Saved:")
    for name in [
        "combined_results.csv",
        "final_summary_table.csv",
        "collapse_rate_table.csv",
        "seedwise_comparison_table.csv",
        "methodwise_mean_std_table.csv",
        "cavity_profile_summary_table.csv",
        "combined_results_readable.csv",
        "methodwise_mean_std_readable.csv",
        "seedwise_comparison_readable.csv",
        "cavity_profile_summary_readable.csv",
    ]:
        print(f"  {out / name}")
    print(f"  {fig_dir}")


def collect_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*/summary.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cfg_path = path.parent / "config_snapshot.yaml"
        cfg = {}
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        mode = data.get("mode") or path.parent.name.split("_seed")[0].split("_", 1)[-1]
        row = {
            "run_id": path.parent.name,
            "run_dir": str(path.parent),
            "benchmark": data.get("benchmark", cfg.get("benchmark", "unknown")),
            "mode": data.get("mode", mode),
            "method": data.get("method", "vara" if "vara" in mode else "vanilla"),
            "seed": int(data.get("seed", cfg.get("seed", -1))),
            "reference_kind": data.get("reference_kind", "unknown"),
            "has_reference": data.get("has_reference", None),
            "run_type": data.get("run_type", cfg.get("run_type", "unknown")),
            "reportable": bool(data.get("reportable", data.get("run_type", cfg.get("run_type", "")) != "smoke")),
            "collapse_evaluated": bool(data.get("collapse_evaluated", data.get("run_type", cfg.get("run_type", "")) != "smoke")),
            "accepted_interventions": data.get("accepted_interventions", 0),
            "rejected_interventions": data.get("rejected_interventions", 0),
            "rollback_count": data.get("rollback_count", 0),
            "number_of_active_patches": data.get("number_of_active_patches", 0),
            "most_frequently_targeted_variable": data.get("most_frequently_targeted_variable"),
            "most_frequently_targeted_patch": data.get("most_frequently_targeted_patch"),
            "collapsed": bool(data.get("collapsed", False)),
            "cavity_profile_reference_source": data.get("cavity_profile_reference_source", ""),
            "full_field_reference_path": data.get("full_field_reference_path", ""),
            "has_p_full_field_reference": data.get("has_p_full_field_reference", False),
            "has_omega_full_field_reference": data.get("has_omega_full_field_reference", False),
            "omega_full_field_reference_source": data.get("omega_full_field_reference_source", ""),
        }
        for metric in METRICS:
            row[metric] = data.get(metric, np.nan)
        rows.append(row)
    return rows


def build_methodwise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["benchmark", "run_type", "method"] if "run_type" in df.columns else ["benchmark", "method"]
    for key, group in df.groupby(group_keys):
        if len(group_keys) == 3:
            benchmark, run_type, method = key
            row = {"benchmark": benchmark, "run_type": run_type, "method": method, "n_runs": len(group)}
        else:
            benchmark, method = key
            row = {"benchmark": benchmark, "method": method, "n_runs": len(group)}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
    sort_cols = ["benchmark", "run_type", "method"] if rows and "run_type" in rows[0] else ["benchmark", "method"]
    return pd.DataFrame(rows).sort_values(sort_cols)


def build_readable(df: pd.DataFrame) -> pd.DataFrame:
    readable = df.copy()
    for col in readable.columns:
        if pd.api.types.is_float_dtype(readable[col]):
            readable[col] = readable[col].map(lambda x: "N/A" if pd.isna(x) else f"{float(x):.6g}")
        else:
            readable[col] = readable[col].where(pd.notna(readable[col]), "N/A")
    return readable


def build_cavity_table(methodwise: pd.DataFrame) -> pd.DataFrame:
    if methodwise.empty or "benchmark" not in methodwise:
        return pd.DataFrame()
    cavity = methodwise[methodwise["benchmark"] == "lid_driven_cavity"].copy()
    keep = [
        "benchmark",
        "run_type",
        "method",
        "n_runs",
        "u_full_rel_l2_mean",
        "u_full_rel_l2_std",
        "v_full_rel_l2_mean",
        "v_full_rel_l2_std",
        "velocity_full_rel_l2_mean",
        "velocity_full_rel_l2_std",
        "p_full_rel_l2_centered_mean",
        "p_full_rel_l2_centered_std",
        "omega_full_rel_l2_mean",
        "omega_full_rel_l2_std",
        "velocity_mag_rmse_mean",
        "velocity_mag_rmse_std",
        "velocity_mag_mae_mean",
        "velocity_mag_mae_std",
        "u_centerline_rmse_mean",
        "u_centerline_rmse_std",
        "v_centerline_rmse_mean",
        "v_centerline_rmse_std",
        "centerline_profile_score_mean",
        "centerline_profile_score_std",
        "pde_residual_mean_mean",
        "centerline_pde_residual_mean_mean",
        "centerline_continuity_residual_mean_mean",
        "corner_pde_residual_mean_mean",
        "corner_boundary_error_mean",
        "continuity_residual_mean_mean",
        "momentum_residual_mean_mean",
        "boundary_condition_error_mean",
        "unweighted_validation_loss_mean",
        "cavity_benchmark_score_mean",
    ]
    return cavity[[col for col in keep if col in cavity.columns]]


def build_collapse(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["benchmark", "run_type", "method"] if "run_type" in df.columns else ["benchmark", "method"]
    for key, group in df.groupby(group_keys):
        if len(group_keys) == 3:
            benchmark, run_type, method = key
        else:
            benchmark, method = key
            run_type = "unknown"
        evaluated = group[group["collapse_evaluated"].astype(bool)] if "collapse_evaluated" in group else group
        n = len(evaluated)
        c = int(evaluated["collapsed"].sum()) if n else 0
        rows.append(
            {
                "benchmark": benchmark,
                "run_type": run_type,
                "method": method,
                "collapse_rate_percent": 100.0 * c / n if n else np.nan,
                "collapsed": c,
                "evaluated_runs": n,
                "total_runs": len(group),
                "smoke_or_not_evaluated_runs": len(group) - n,
            }
        )
    return pd.DataFrame(rows).sort_values(["benchmark", "run_type", "method"])


def build_seedwise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["benchmark", "run_type", "seed"] if "run_type" in df.columns else ["benchmark", "seed"]
    for key, group in df.groupby(group_keys):
        if len(group_keys) == 3:
            benchmark, run_type, seed = key
        else:
            benchmark, seed = key
            run_type = "unknown"
        vanilla = group[group["method"] == "vanilla"]
        vara = group[group["method"] == "vara"]
        if vanilla.empty or vara.empty:
            continue
        v = vanilla.iloc[-1]
        a = vara.iloc[-1]
        row = {"benchmark": benchmark, "run_type": run_type, "seed": seed}
        for metric in METRICS:
            v_val = pd.to_numeric(pd.Series([v[metric]]), errors="coerce").iloc[0]
            a_val = pd.to_numeric(pd.Series([a[metric]]), errors="coerce").iloc[0]
            row[f"{metric}_vanilla"] = v_val
            row[f"{metric}_vara"] = a_val
            row[f"{metric}_improvement_percent"] = (v_val - a_val) / v_val * 100.0 if pd.notna(v_val) and v_val != 0 and pd.notna(a_val) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["benchmark", "run_type", "seed"]) if rows else pd.DataFrame()


def save_bar_plots(methodwise: pd.DataFrame, fig_dir: Path) -> None:
    group_keys = ["benchmark", "run_type"] if "run_type" in methodwise.columns else ["benchmark"]
    for key, group in methodwise.groupby(group_keys):
        if isinstance(key, tuple):
            benchmark, run_type = key
            stem_prefix = f"{benchmark}_{run_type}"
            title_prefix = f"{benchmark} ({run_type})"
        else:
            benchmark = key
            stem_prefix = benchmark
            title_prefix = benchmark
        labels = []
        vanilla = []
        vara = []
        for metric in [
            "u_full_rel_l2",
            "v_full_rel_l2",
            "velocity_full_rel_l2",
            "p_full_rel_l2_centered",
            "omega_full_rel_l2",
            "velocity_mag_rmse",
            "velocity_mag_mae",
            "u_rmse",
            "v_rmse",
            "p_rmse_centered",
            "omega_rmse",
            "pde_residual_mean",
            "unweighted_validation_loss",
            "boundary_condition_error",
            "boundary_speed_rmse",
            "centerline_profile_score",
            "cavity_benchmark_score",
        ]:
            v = group[group["method"] == "vanilla"]
            a = group[group["method"] == "vara"]
            if v.empty or a.empty:
                continue
            labels.append(metric)
            vanilla.append(v.iloc[0][f"{metric}_mean"])
            vara.append(a.iloc[0][f"{metric}_mean"])
        if not labels:
            continue
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        ax.bar(x - 0.18, vanilla, 0.36, label="Vanilla")
        ax.bar(x + 0.18, vara, 0.36, label="VARA")
        ax.set_title(f"{title_prefix}: VARA vs Vanilla")
        ax.set_ylabel("mean metric, lower is better")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(fig_dir / f"{stem_prefix}_vara_vs_vanilla_bar.png", dpi=220)
        plt.close(fig)


def save_seedwise_scatter(df: pd.DataFrame, fig_dir: Path) -> None:
    group_keys = ["benchmark", "run_type"] if "run_type" in df.columns else ["benchmark"]
    for key, group in df.groupby(group_keys):
        if isinstance(key, tuple):
            benchmark, run_type = key
            stem_prefix = f"{benchmark}_{run_type}"
            title_prefix = f"{benchmark} ({run_type})"
        else:
            benchmark = key
            stem_prefix = benchmark
            title_prefix = benchmark
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for method, sub in group.groupby("method"):
            ax.scatter(sub["seed"], pd.to_numeric(sub["pde_residual_mean"], errors="coerce"), label=method)
        ax.set_title(f"{title_prefix}: seedwise PDE residual")
        ax.set_xlabel("seed")
        ax.set_ylabel("PDE residual mean")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.savefig(fig_dir / f"{stem_prefix}_seedwise_pde_residual.png", dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    main()
