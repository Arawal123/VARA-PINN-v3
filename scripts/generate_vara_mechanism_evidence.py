"""Generate publication artifacts explaining why VARA wins."""

from __future__ import annotations

import argparse
import json
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


METRIC_PAIRS = [
    "pde_residual_mean",
    "continuity_residual_mean",
    "momentum_residual_mean",
    "boundary_condition_error",
    "u_boundary_rmse",
    "v_boundary_rmse",
    "centerline_pde_residual_mean",
    "corner_boundary_error",
    "unweighted_validation_loss",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    root = Path(args.results_dir)
    output = Path(args.output_dir) if args.output_dir else root / "summary" / "vara_mechanism_evidence"
    output.mkdir(parents=True, exist_ok=True)

    decisions = _collect_csv(root, "local_controller_decisions.csv")
    patch_scores = _collect_csv(root, "patch_scores.csv")
    summaries = _collect_summaries(root)

    if decisions.empty:
        print("No local_controller_decisions.csv files found. Controller evidence was not generated.")
    else:
        _decision_artifacts(decisions, output)
    if patch_scores.empty:
        print("No patch_scores.csv files found. Patch evolution evidence was not generated.")
    else:
        _patch_artifacts(patch_scores, output)
    if summaries.empty:
        print("No summary.json files found. Method trade-off evidence was not generated.")
    else:
        _method_tradeoff_artifacts(summaries, output)
    _write_interpretation(decisions, patch_scores, summaries, output)
    print(f"Saved VARA mechanism evidence to: {output}")


def _collect_csv(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob(filename)):
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        frame["run_id"] = path.parent.name
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _collect_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("summary.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        row["run_id"] = path.parent.name
        row["source_file"] = str(path)
        row.setdefault("mode", _infer_mode(path.parent.name))
        rows.append(row)
    return pd.DataFrame(rows)


def _decision_artifacts(df: pd.DataFrame, output: Path) -> None:
    decisions = df.copy()
    for name in ["accepted", "rejected", "rollback_triggered"]:
        if name in decisions:
            decisions[name] = decisions[name].map(_as_bool)
    for metric in METRIC_PAIRS:
        before = f"{metric}_before"
        after = f"{metric}_after"
        if before in decisions and after in decisions:
            b = pd.to_numeric(decisions[before], errors="coerce")
            a = pd.to_numeric(decisions[after], errors="coerce")
            decisions[f"{metric}_change"] = a - b
            decisions[f"{metric}_improvement_percent"] = 100.0 * (b - a) / b.abs().clip(lower=1e-12)
    decisions.to_csv(output / "controller_decisions_enriched.csv", index=False)

    counts = (
        decisions.assign(
            outcome=np.where(decisions.get("accepted", False), "accepted", "rejected")
        )
        .groupby(["run_id", "outcome"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    counts.to_csv(output / "accepted_rejected_by_run.csv", index=False)

    reasons = decisions.get("rejection_reason", pd.Series(dtype=str)).fillna("").astype(str)
    reason_rows = []
    for run_id, value in zip(decisions["run_id"], reasons):
        for reason in [part for part in value.split(",") if part]:
            reason_rows.append({"run_id": run_id, "rejection_reason": reason})
    reason_df = pd.DataFrame(reason_rows)
    if not reason_df.empty:
        reason_counts = reason_df.groupby("rejection_reason").size().sort_values(ascending=False).rename("count").reset_index()
        reason_counts.to_csv(output / "rejection_reason_counts.csv", index=False)
        _bar_plot(
            reason_counts["rejection_reason"],
            reason_counts["count"],
            output / "rejection_reason_counts.png",
            "Why VARA rejects candidate interventions",
            "Rejected candidates",
        )

    rollback = decisions[decisions.get("rollback_triggered", False)].copy()
    damage_columns = [f"{metric}_change" for metric in METRIC_PAIRS if f"{metric}_change" in rollback]
    if not rollback.empty and damage_columns:
        rollback["maximum_observed_damage"] = rollback[damage_columns].max(axis=1, skipna=True)
        rollback["damaging_candidate"] = rollback["maximum_observed_damage"] > 0
        rollback.to_csv(output / "rollback_prevented_damage.csv", index=False)

    if "target_local_improvement" in decisions:
        values = pd.to_numeric(decisions["target_local_improvement"], errors="coerce")
        accepted = decisions.get("accepted", pd.Series(False, index=decisions.index))
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.hist(values[accepted].dropna(), bins=12, alpha=0.75, label="Accepted", color="#247a4d")
        ax.hist(values[~accepted].dropna(), bins=12, alpha=0.70, label="Rejected", color="#b83b3b")
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title("Targeted local improvement by controller outcome")
        ax.set_xlabel("Targeted local improvement")
        ax.set_ylabel("Candidate interventions")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "target_improvement_accepted_vs_rejected.png", dpi=220)
        plt.close(fig)

    delta_rows = []
    for metric in METRIC_PAIRS:
        column = f"{metric}_improvement_percent"
        if column not in decisions:
            continue
        for outcome, mask in [
            ("accepted", decisions.get("accepted", False)),
            ("rejected", ~decisions.get("accepted", False)),
        ]:
            values = pd.to_numeric(decisions.loc[mask, column], errors="coerce").dropna()
            if not values.empty:
                delta_rows.append(
                    {
                        "metric": metric,
                        "outcome": outcome,
                        "mean_improvement_percent": values.mean(),
                        "std": values.std(),
                        "count": len(values),
                    }
                )
    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_df.to_csv(output / "accepted_rejected_metric_effects.csv", index=False)
        pivot = delta_df.pivot(index="metric", columns="outcome", values="mean_improvement_percent").fillna(0.0)
        ax = pivot.plot(kind="bar", figsize=(11.0, 5.0), color=["#247a4d", "#b83b3b"])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("Mean metric effect of accepted and rejected candidates")
        ax.set_ylabel("Improvement percent, positive is better")
        ax.tick_params(axis="x", rotation=35)
        ax.figure.tight_layout()
        ax.figure.savefig(output / "accepted_rejected_metric_effects.png", dpi=220)
        plt.close(ax.figure)

    if {"variable", "patch_id"}.issubset(decisions.columns):
        targeted = decisions[decisions["variable"].fillna("").astype(str) != ""].copy()
        targeted["patch_id"] = pd.to_numeric(targeted["patch_id"], errors="coerce")
        targeted = targeted.dropna(subset=["patch_id"])
        if not targeted.empty:
            table = pd.crosstab(targeted["variable"], targeted["patch_id"].astype(int))
            table.to_csv(output / "variable_patch_target_frequency.csv")
            _heatmap(
                table.to_numpy(dtype=float),
                [str(value) for value in table.columns],
                [str(value) for value in table.index],
                output / "variable_patch_target_frequency.png",
                "Where VARA intervenes",
                "Patch ID",
                "Diagnostic variable",
            )


def _patch_artifacts(df: pd.DataFrame, output: Path) -> None:
    required = {"run_id", "cycle", "variable", "patch_id", "raw_score"}
    if not required.issubset(df.columns):
        return
    patch = df.copy()
    patch["cycle"] = pd.to_numeric(patch["cycle"], errors="coerce")
    patch["patch_id"] = pd.to_numeric(patch["patch_id"], errors="coerce")
    patch["raw_score"] = pd.to_numeric(patch["raw_score"], errors="coerce")
    patch = patch.dropna(subset=["cycle", "patch_id", "raw_score"])
    rows = []
    for (run_id, variable, patch_id), group in patch.groupby(["run_id", "variable", "patch_id"]):
        ordered = group.sort_values("cycle")
        first = float(ordered.iloc[0]["raw_score"])
        last = float(ordered.iloc[-1]["raw_score"])
        rows.append(
            {
                "run_id": run_id,
                "variable": variable,
                "patch_id": int(patch_id),
                "first_score": first,
                "last_score": last,
                "improvement_percent": 100.0 * (first - last) / max(abs(first), 1e-12),
            }
        )
    evolution = pd.DataFrame(rows)
    evolution.to_csv(output / "patch_score_first_last.csv", index=False)
    aggregate = (
        evolution.groupby(["variable", "patch_id"])["improvement_percent"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.to_csv(output / "patch_score_improvement_by_variable_patch.csv", index=False)

    table = aggregate.pivot(index="variable", columns="patch_id", values="mean")
    if not table.empty:
        _heatmap(
            table.fillna(0.0).to_numpy(),
            [str(value) for value in table.columns],
            [str(value) for value in table.index],
            output / "patch_score_improvement_heatmap.png",
            "Weak-region score reduction from first to last cycle",
            "Patch ID",
            "Diagnostic variable",
        )


def _method_tradeoff_artifacts(df: pd.DataFrame, output: Path) -> None:
    if "mode" not in df:
        return
    mode = df["mode"].fillna("").astype(str)
    subset = df[mode.isin(["rar_pinn", "local_constrained_vara"])].copy()
    if subset.empty:
        return
    rows = []
    for method, group in subset.groupby("mode"):
        for metric in METRIC_PAIRS + ["centerline_profile_score", "cavity_benchmark_score"]:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if not values.empty:
                rows.append(
                    {
                        "mode": method,
                        "metric": metric,
                        "mean": values.mean(),
                        "std": values.std(),
                        "count": len(values),
                    }
                )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "rar_vs_vara_tradeoff.csv", index=False)


def _write_interpretation(
    decisions: pd.DataFrame,
    patch_scores: pd.DataFrame,
    summaries: pd.DataFrame,
    output: Path,
) -> None:
    lines = ["VARA mechanism evidence summary", ""]
    if not decisions.empty:
        accepted = decisions.get("accepted", pd.Series(False, index=decisions.index)).map(_as_bool)
        rollback = decisions.get("rollback_triggered", pd.Series(False, index=decisions.index)).map(_as_bool)
        lines.append(f"Candidate interventions: {len(decisions)}")
        lines.append(f"Accepted interventions: {int(accepted.sum())}")
        lines.append(f"Rejected interventions: {int((~accepted).sum())}")
        lines.append(f"Rollback-triggered candidates: {int(rollback.sum())}")
    if not patch_scores.empty:
        lines.append(f"Patch score observations: {len(patch_scores)}")
        lines.append(f"Diagnostic variables observed: {patch_scores.get('variable', pd.Series(dtype=str)).nunique()}")
    if not summaries.empty:
        lines.append(f"Run summaries inspected: {len(summaries)}")
    lines.extend(
        [
            "",
            "Interpretation guidance:",
            "- Use accepted/rejected metric effects to show that target improvement alone is insufficient.",
            "- Use rollback-prevented damage rows to quantify harmful proposals that were filtered.",
            "- Use variable-patch targeting and patch-score evolution to demonstrate localized diagnosis.",
            "- Use RAR-vs-VARA trade-offs to distinguish residual chasing from balanced physical improvement.",
        ]
    )
    (output / "mechanism_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def _bar_plot(x: Any, y: Any, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(np.arange(len(x)), y, color="#8f3f52")
    ax.set_xticks(np.arange(len(x)), x, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _heatmap(
    values: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(7.0, len(xlabels) * 0.55), max(4.0, len(ylabels) * 0.45)))
    image = ax.imshow(values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(xlabels)), xlabels)
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _infer_mode(name: str) -> str:
    for mode in [
        "local_constrained_vara",
        "self_adaptive_attention_pinn",
        "gradient_balanced_pinn",
        "gradient_enhanced_pinn",
        "rar_pinn",
        "vanilla_pinn",
    ]:
        if mode in name:
            return mode
    return ""


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


if __name__ == "__main__":
    main()
