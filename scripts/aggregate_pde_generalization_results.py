"""Aggregate one or more isolated PDE generalization experiment folders."""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pde_generalization.summary import (
    collect_run_summaries,
    paired_comparisons,
    write_standard_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = collect_run_summaries(args.input_dirs)
    if raw.empty:
        raise FileNotFoundError("No PDE generalization run summaries were found.")
    products = write_standard_outputs(raw, output)
    raw.to_csv(output / "combined_pde_generalization_results.csv", index=False)
    products["winrates"].to_csv(output / "pde_generalization_winrates.csv", index=False)
    products["aggregate"].to_csv(output / "pde_generalization_mean_std.csv", index=False)
    significance = _paired_significance(raw)
    significance.to_csv(output / "pde_generalization_significance.csv", index=False)
    print(f"Collected {len(raw)} runs into {output}")


def _paired_significance(raw: pd.DataFrame) -> pd.DataFrame:
    paired, _ = paired_comparisons(raw)
    records = []
    for (benchmark, method, metric), group in paired.groupby(["benchmark", "method", "metric"]):
        differences = group["vanilla_value"].to_numpy(float) - group["method_value"].to_numpy(float)
        nonzero = differences[~np.isclose(differences, 0.0)]
        wins = int((nonzero > 0).sum())
        n = len(nonzero)
        sign_p = min(1.0, 2.0 * sum(comb(n, k) for k in range(0, min(wins, n - wins) + 1)) / (2**n)) if n else 1.0
        wilcoxon_p = np.nan
        try:
            from scipy.stats import wilcoxon

            if n:
                wilcoxon_p = float(wilcoxon(nonzero).pvalue)
        except (ImportError, ValueError):
            pass
        records.append({
            "benchmark": benchmark,
            "method": method,
            "metric": metric,
            "paired_seeds": len(group),
            "median_improvement": float(np.median(differences)),
            "sign_test_p": sign_p,
            "wilcoxon_p": wilcoxon_p,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    main()
