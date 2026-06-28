"""Cross-seed summaries and publication table generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .metrics import primary_metric_name


IDENTITY_COLUMNS = [
    "benchmark",
    "method",
    "seed",
    "run_dir",
    "git_commit",
    "initial_model_parameter_hash",
    "sparse_sample_hash",
]


def collect_run_summaries(input_dirs: Iterable[str | Path]) -> pd.DataFrame:
    """Recursively load per-run summary JSON files into one long table."""
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for input_dir in input_dirs:
        path = Path(input_dir)
        candidates = [path] if path.name == "summary.json" else path.rglob("summary.json")
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or "summary" in candidate.parent.parts[-1:]:
                continue
            seen.add(resolved)
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not {"benchmark", "method", "seed", "metrics"}.issubset(data):
                continue
            records.append(
                {
                    **{key: data.get(key) for key in IDENTITY_COLUMNS},
                    **dict(data.get("metrics", {})),
                }
            )
    return pd.DataFrame(records)


def paired_comparisons(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paired values and percent improvements relative to vanilla."""
    paired_rows: list[dict[str, Any]] = []
    improvement_rows: list[dict[str, Any]] = []
    metric_columns = [column for column in raw.columns if column not in IDENTITY_COLUMNS]
    for (benchmark, seed), group in raw.groupby(["benchmark", "seed"]):
        vanilla = group[group["method"] == "vanilla"]
        if vanilla.empty:
            continue
        baseline = vanilla.iloc[0]
        for _, method_row in group[group["method"] != "vanilla"].iterrows():
            for metric in metric_columns:
                left = pd.to_numeric(pd.Series([baseline.get(metric)]), errors="coerce").iloc[0]
                right = pd.to_numeric(pd.Series([method_row.get(metric)]), errors="coerce").iloc[0]
                if not np.isfinite(left) or not np.isfinite(right):
                    continue
                paired_rows.append(
                    {
                        "benchmark": benchmark,
                        "seed": int(seed),
                        "method": method_row["method"],
                        "metric": metric,
                        "vanilla_value": float(left),
                        "method_value": float(right),
                    }
                )
                improvement_rows.append(
                    {
                        "benchmark": benchmark,
                        "seed": int(seed),
                        "method": method_row["method"],
                        "metric": metric,
                        "improvement_percent": 100.0 * (float(left) - float(right)) / max(abs(float(left)), 1e-12),
                    }
                )
    return pd.DataFrame(paired_rows), pd.DataFrame(improvement_rows)


def aggregate_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, sample standard deviation, and count by PDE/method."""
    numeric = raw.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [column for column in numeric if column != "seed"]
    if not numeric or raw.empty:
        return pd.DataFrame()
    grouped = raw.groupby(["benchmark", "method"])[numeric].agg(["mean", "std", "median", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def winrate_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute paired reconstruction win rates for each non-vanilla method."""
    records: list[dict[str, Any]] = []
    for benchmark, group in raw.groupby("benchmark"):
        metric = primary_metric_name(str(benchmark))
        if metric not in group:
            continue
        baseline = group[group["method"] == "vanilla"].set_index("seed")[metric]
        for method, method_group in group[group["method"] != "vanilla"].groupby("method"):
            values = method_group.set_index("seed")[metric]
            common = baseline.index.intersection(values.index)
            if common.empty:
                continue
            wins = int((values.loc[common] < baseline.loc[common]).sum())
            ties = int(np.isclose(values.loc[common], baseline.loc[common]).sum())
            records.append(
                {
                    "benchmark": benchmark,
                    "method": method,
                    "metric": metric,
                    "paired_seeds": len(common),
                    "wins": wins,
                    "ties": ties,
                    "win_rate_percent": 100.0 * wins / len(common),
                }
            )
    return pd.DataFrame(records)


def write_standard_outputs(raw: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Write runner-level CSVs, figures, Markdown, and a compact LaTeX table."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paired, improvements = paired_comparisons(raw)
    aggregate = aggregate_summary(raw)
    winrates = winrate_summary(raw)
    raw.to_csv(output / "raw_results_long.csv", index=False)
    paired.to_csv(output / "vara_v2_vs_vanilla_by_seed.csv", index=False)
    improvements.to_csv(output / "improvement_percent_by_seed.csv", index=False)
    aggregate.to_csv(output / "aggregate_summary.csv", index=False)
    winrates.to_csv(output / "winrate_summary.csv", index=False)
    (output / "pde_generalization_latex_tables.tex").write_text(
        _latex_table(raw), encoding="utf-8"
    )
    (output / "pde_generalization_summary.md").write_text(
        _markdown_summary(raw, winrates), encoding="utf-8"
    )
    from .plots import save_aggregate_plots

    save_aggregate_plots(raw, winrates, output)
    return {
        "raw": raw,
        "paired": paired,
        "improvements": improvements,
        "aggregate": aggregate,
        "winrates": winrates,
    }


def _latex_table(raw: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated PDE generalization reconstruction table.",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "PDE & Method & Mean relative $L_2$ & Std. \\\\",
        "\\midrule",
    ]
    for (benchmark, method), group in raw.groupby(["benchmark", "method"]):
        metric = primary_metric_name(str(benchmark))
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        lines.append(f"{benchmark.replace('_', ' ')} & {method.replace('_', ' ')} & {values.mean():.4e} & {std:.4e} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _markdown_summary(raw: pd.DataFrame, winrates: pd.DataFrame) -> str:
    lines = ["# PDE generalization summary", "", f"Runs collected: {len(raw)}", ""]
    if not winrates.empty:
        columns = list(winrates.columns)
        lines.extend(
            [
                "## Paired reconstruction win rates",
                "",
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join(["---"] * len(columns)) + " |",
            ]
        )
        for row in winrates.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        lines.append("")
    lines.extend(
        [
            "Full-field references were used only for final evaluation; controller decisions used physics, prescribed-condition, and sparse-training signals.",
            "",
        ]
    )
    return "\n".join(lines)
