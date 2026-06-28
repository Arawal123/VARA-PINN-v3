"""Cahn--Hilliard paired summaries, win rates, and tables."""

from __future__ import annotations

import json
from math import comb
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = [
    "benchmark",
    "method",
    "seed",
    "run_dir",
    "git_commit",
    "initial_model_parameter_hash",
    "sparse_hash",
]
PRIMARY_METRIC = "cahn_hilliard_u_rel_l2"


def collect_summaries(input_dirs: Iterable[str | Path]) -> pd.DataFrame:
    """Recursively collect valid per-run summaries."""
    records = []
    seen: set[Path] = set()
    for input_dir in input_dirs:
        path = Path(input_dir)
        candidates = [path] if path.name == "summary.json" else path.rglob("summary.json")
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or candidate.parent.name == "summary":
                continue
            seen.add(resolved)
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("benchmark") != "cahn_hilliard" or "metrics" not in data:
                continue
            records.append(
                {
                    **{key: data.get(key) for key in IDENTITY_COLUMNS},
                    **dict(data["metrics"]),
                }
            )
    return pd.DataFrame(records)


def paired_comparisons(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each method to its same-seed vanilla baseline."""
    pairs, improvements = [], []
    metrics = [
        column
        for column in raw.columns
        if column not in IDENTITY_COLUMNS and _is_lower_better_metric(column)
    ]
    for seed, group in raw.groupby("seed"):
        vanilla = group[group["method"] == "vanilla"]
        if vanilla.empty:
            continue
        baseline = vanilla.iloc[0]
        for _, method_row in group[group["method"] != "vanilla"].iterrows():
            for metric in metrics:
                left = pd.to_numeric(pd.Series([baseline.get(metric)]), errors="coerce").iloc[0]
                right = pd.to_numeric(pd.Series([method_row.get(metric)]), errors="coerce").iloc[0]
                if not np.isfinite(left) or not np.isfinite(right):
                    continue
                pair = {
                    "seed": int(seed),
                    "method": method_row["method"],
                    "metric": metric,
                    "vanilla_value": float(left),
                    "method_value": float(right),
                }
                pairs.append(pair)
                improvements.append(
                    {
                        "seed": int(seed),
                        "method": method_row["method"],
                        "metric": metric,
                        "improvement_percent": 100.0
                        * (float(left) - float(right))
                        / max(abs(float(left)), 1e-12),
                    }
                )
    return pd.DataFrame(pairs), pd.DataFrame(improvements)


def aggregate_summary(raw: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in raw.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    if not numeric or raw.empty:
        return pd.DataFrame()
    grouped = raw.groupby("method")[numeric].agg(["mean", "std", "median", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    index_frame = grouped.index.to_frame(index=False)
    value_frame = grouped.reset_index(drop=True).copy()
    return pd.concat([index_frame, value_frame], axis=1)


def winrate_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or PRIMARY_METRIC not in raw:
        return pd.DataFrame()
    baseline = raw[raw["method"] == "vanilla"].set_index("seed")[PRIMARY_METRIC]
    records = []
    for method, group in raw[raw["method"] != "vanilla"].groupby("method"):
        values = group.set_index("seed")[PRIMARY_METRIC]
        common = baseline.index.intersection(values.index)
        if common.empty:
            continue
        wins = int((values.loc[common] < baseline.loc[common]).sum())
        ties = int(np.isclose(values.loc[common], baseline.loc[common]).sum())
        records.append(
            {
                "method": method,
                "metric": PRIMARY_METRIC,
                "paired_seeds": len(common),
                "wins": wins,
                "ties": ties,
                "win_rate_percent": 100.0 * wins / len(common),
            }
        )
    return pd.DataFrame(records)


def significance_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compute exact sign tests and optional Wilcoxon tests."""
    if pairs.empty:
        return pd.DataFrame()
    records = []
    for (method, metric), group in pairs.groupby(["method", "metric"]):
        differences = group["vanilla_value"].to_numpy(float) - group["method_value"].to_numpy(float)
        nonzero = differences[~np.isclose(differences, 0.0)]
        wins, count = int((nonzero > 0).sum()), len(nonzero)
        tail = min(wins, count - wins) if count else 0
        sign_p = min(1.0, 2.0 * sum(comb(count, k) for k in range(tail + 1)) / (2**count)) if count else 1.0
        wilcoxon_p = np.nan
        try:
            from scipy.stats import wilcoxon

            if count:
                wilcoxon_p = float(wilcoxon(nonzero).pvalue)
        except (ImportError, ValueError):
            pass
        records.append(
            {
                "method": method,
                "metric": metric,
                "paired_seeds": len(group),
                "median_absolute_improvement": float(np.median(differences)),
                "sign_test_p": sign_p,
                "wilcoxon_p": wilcoxon_p,
            }
        )
    return pd.DataFrame(records)


def write_runner_outputs(raw: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Write runner-level files and publication figures."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairs, improvements = paired_comparisons(raw)
    aggregate = aggregate_summary(raw)
    winrates = winrate_summary(raw)
    raw.to_csv(output / "cahn_hilliard_results_long.csv", index=False)
    pairs.to_csv(output / "vara_v2_vs_vanilla_by_seed.csv", index=False)
    improvements.to_csv(output / "improvement_percent_by_seed.csv", index=False)
    aggregate.to_csv(output / "aggregate_summary.csv", index=False)
    winrates.to_csv(output / "winrate_summary.csv", index=False)
    _write_text_products(raw, winrates, output)
    from .plots import save_aggregate_plots

    save_aggregate_plots(raw, output)
    return {
        "raw": raw,
        "pairs": pairs,
        "improvements": improvements,
        "aggregate": aggregate,
        "winrates": winrates,
    }


def write_aggregate_outputs(raw: pd.DataFrame, output_dir: str | Path) -> None:
    """Write the externally aggregated file naming contract."""
    output = Path(output_dir)
    products = write_runner_outputs(raw, output)
    significance = significance_summary(products["pairs"])
    raw.to_csv(output / "cahn_hilliard_combined_results.csv", index=False)
    products["aggregate"].to_csv(output / "cahn_hilliard_mean_std.csv", index=False)
    products["winrates"].to_csv(output / "cahn_hilliard_winrates.csv", index=False)
    improvement_summary = (
        products["improvements"]
        .groupby(["method", "metric"])["improvement_percent"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_improvement_percent",
                "std": "std_improvement_percent",
                "median": "median_improvement_percent",
                "count": "paired_seed_count",
            }
        )
    )
    merged = improvement_summary.merge(
        significance,
        on=["method", "metric"],
        how="left",
    )
    merged.to_csv(output / "cahn_hilliard_improvement_summary.csv", index=False)


def _write_text_products(raw: pd.DataFrame, winrates: pd.DataFrame, output: Path) -> None:
    lines = [
        "% Auto-generated Cahn--Hilliard reconstruction table.",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Method & Mean $u$ relative $L_2$ & Std. \\\\",
        "\\midrule",
    ]
    for method, group in raw.groupby("method"):
        values = pd.to_numeric(group[PRIMARY_METRIC], errors="coerce").dropna()
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        lines.append(f"{method.replace('_', ' ')} & {values.mean():.4e} & {std:.4e} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output / "cahn_hilliard_latex_tables.tex").write_text("\n".join(lines), encoding="utf-8")

    markdown = [
        "# Cahn–Hilliard benchmark summary",
        "",
        f"Runs collected: {len(raw)}",
        "",
        "Full-field exact references are reporting-only; adaptation uses residual, prescribed-condition, shared sparse-training, and prediction-derived signals.",
        "",
    ]
    if not winrates.empty:
        columns = list(winrates.columns)
        markdown.extend(
            [
                "## Paired reconstruction win rates",
                "",
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join(["---"] * len(columns)) + " |",
            ]
        )
        for row in winrates.itertuples(index=False, name=None):
            markdown.append("| " + " | ".join(str(value) for value in row) + " |")
        markdown.append("")
    (output / "cahn_hilliard_summary.md").write_text("\n".join(markdown), encoding="utf-8")


def _is_lower_better_metric(name: str) -> bool:
    tokens = ("rel_l2", "rmse", "mae", "mse", "error", "residual", "wall_clock")
    return any(token in name for token in tokens)
