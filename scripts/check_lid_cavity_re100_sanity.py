"""Strict Re=100 lid-driven cavity sanity gate for reliable V2 studies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_vara_v2_continuation import run as run_v2_continuation
from src.utils.config import load_config
from src.utils.io import save_json


RAW_LOSS_COLUMNS = ("momentum_u", "momentum_v", "continuity", "bc")
VANILLA_REQUIRED = (
    "boundary_condition_error",
    "pde_residual_mean",
    "continuity_residual_mean",
    "momentum_residual_mean",
    "streamfunction_consistency_rmse",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["vanilla", "vara_v2", "both"], default="vanilla")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional multi-seed gate. If omitted, --seed is used.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default="configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    parser.add_argument("--output_dir", default="experiments/vara_v2/re100_sanity_gate")
    parser.add_argument("--results_dir", default=None, help="Validate an existing run instead of training.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report_path", default=None)
    parser.add_argument(
        "--full_field_reference_map",
        default=None,
        help="Optional CSV/JSON map. Full-field gates are applied only when a Re=100 reference is supplied.",
    )
    args = parser.parse_args()
    seeds = _requested_seeds(args)

    output_dir = Path(args.results_dir or args.output_dir)
    if args.results_dir is None:
        _run_required_methods(args, output_dir, seeds)

    report = build_combined_report(output_dir, args.method, seeds)
    report_path = Path(args.report_path) if args.report_path else output_dir / "summary" / "re100_sanity_report.json"
    save_json(report, report_path)
    print(f"Saved sanity report: {report_path}")

    if not report["passed"]:
        for failure in report["failures"]:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("PASS: Re=100 sanity gates satisfied.")


def _requested_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    return [int(args.seed)]


def _run_required_methods(args: argparse.Namespace, output_dir: Path, seeds: list[int]) -> None:
    """Run Vanilla first; VARA is evaluated only if Vanilla passes."""
    if args.method == "vanilla":
        _run_methods(args, output_dir, ["vanilla"], seeds=seeds, overwrite=True)
        return
    vanilla_dir = output_dir / "_vanilla_gate"
    _run_methods(args, vanilla_dir, ["vanilla"], seeds=seeds, overwrite=True)
    vanilla_report = build_combined_report(vanilla_dir, "vanilla", seeds)
    if not vanilla_report["passed"]:
        save_json(
            vanilla_report,
            output_dir / "summary" / "re100_sanity_report.json",
        )
        raise SystemExit(
            "Vanilla Re=100 failed sanity gates; VARA was not evaluated."
        )
    _run_methods(args, output_dir, ["vanilla", "vara_v2"], seeds=seeds, overwrite=True)


def _run_methods(
    args: argparse.Namespace,
    output_dir: Path,
    methods: list[str],
    *,
    seeds: list[int],
    overwrite: bool,
) -> None:
    run_args = argparse.Namespace(
        config=args.config,
        methods=methods,
        reynolds=[100.0],
        seeds=seeds,
        full_field_reference_map=args.full_field_reference_map,
        device=args.device,
        output_dir=str(output_dir),
        enhanced_backbone=False,
        reliable=True,
        continue_on_invalid=True,
        quick=False,
        overwrite=overwrite or bool(args.overwrite),
    )
    run_v2_continuation(run_args)


def build_combined_report(results_dir: Path, method: str, seeds: list[int]) -> dict[str, Any]:
    if len(seeds) == 1:
        return build_report(results_dir, method, int(seeds[0]))
    seed_reports = [build_report(results_dir, method, int(seed)) for seed in seeds]
    failures = [
        f"seed {seed}: {failure}"
        for seed, report in zip(seeds, seed_reports)
        for failure in report["failures"]
    ]
    return {
        "results_dir": str(results_dir),
        "seeds": [int(seed) for seed in seeds],
        "reynolds": 100.0,
        "method_requested": method,
        "seed_reports": seed_reports,
        "failures": failures,
        "passed": not failures,
    }


def build_report(results_dir: Path, method: str, seed: int) -> dict[str, Any]:
    summary = results_dir / "summary" / "continuation_results_long.csv"
    if not summary.exists():
        raise FileNotFoundError(f"Missing continuation summary: {summary}")
    df = pd.read_csv(summary)
    failures: list[str] = []
    report: dict[str, Any] = {
        "results_dir": str(results_dir),
        "seed": int(seed),
        "reynolds": 100.0,
        "method_requested": method,
        "methods": {},
        "failures": failures,
    }
    config = _resolved_config(results_dir)
    thresholds = dict(config.get("continuation_validity", {}))
    requested_methods = ["vanilla"] if method == "vanilla" else ["vanilla", "vara_v2"]
    for current in requested_methods:
        row = _row_for(df, seed, current)
        if row is None:
            failures.append(f"{current}: missing summary row")
            continue
        current_report = _method_report(results_dir, current, row, thresholds)
        report["methods"][current] = current_report
        failures.extend(f"{current}: {reason}" for reason in current_report["failures"])

    if method in {"vara_v2", "both"} and {"vanilla", "vara_v2"}.issubset(report["methods"]):
        failures.extend(_compare_vara_to_vanilla(report["methods"]["vanilla"], report["methods"]["vara_v2"], thresholds))

    report["passed"] = len(failures) == 0
    if method in {"vara_v2", "both"} and {"vanilla", "vara_v2"}.issubset(report["methods"]):
        report["vara_dominance"] = _dominance_label(report["methods"]["vanilla"], report["methods"]["vara_v2"])
    return report


def _method_report(
    results_dir: Path,
    method: str,
    row: pd.Series,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    method_dir = Path(str(row["method_dir"]))
    if not method_dir.is_absolute():
        method_dir = results_dir / f"seed_{int(row['seed'])}" / "re_0100" / method
    losses = _raw_loss_snapshot(method_dir / "logs" / "losses.csv")
    best_metrics, final_checkpoint_metrics = _checkpoint_metrics(method_dir)
    metrics = {key: _to_float(row.get(key)) for key in row.index}
    physics_loss = metrics.get("unweighted_physics_validation_loss")
    if physics_loss is None or not np.isfinite(physics_loss):
        physics_loss = _finite_sum([metrics.get("unweighted_pde_loss"), metrics.get("unweighted_bc_loss")])
    report = {
        "metrics": metrics,
        "raw_training_losses_last": losses,
        "best_checkpoint_metrics": best_metrics,
        "final_checkpoint_metrics": final_checkpoint_metrics,
        "unweighted_physics_validation_loss": physics_loss,
        "unweighted_reference_evaluation_loss": metrics.get("unweighted_reference_evaluation_loss"),
        "figures": _required_figures(
            method_dir,
            has_reference=_to_bool(row.get("has_reference", False)),
        ),
        "failures": [],
    }
    failures = report["failures"]
    if not _to_bool(row.get("continuation_stage_valid", False)):
        failures.append(f"continuation_stage_valid=false ({row.get('continuation_invalid_reasons', '')})")
    for metric in VANILLA_REQUIRED:
        _check_max(metrics, metric, thresholds, failures)
    _check_min(metrics, "speed_pred_mean", float(thresholds.get("min_speed_pred_mean", 0.05)), failures)
    _check_min(metrics, "primary_streamfunction_abs", float(thresholds.get("min_primary_streamfunction_abs", 0.015)), failures)
    _check_min(metrics, "detected_vortex_count", int(thresholds.get("min_detected_vortices", 1)), failures)
    _check_max_value(
        metrics,
        "detected_vortex_count",
        int(thresholds.get("max_detected_vortices", 10**9)),
        failures,
    )
    _check_exact(metrics, "lid_cavity_topology_aligned", 1.0, failures)
    _check_max(metrics, "lid_cavity_primary_center_error", thresholds, failures)
    if "velocity_full_rel_l2" in metrics and np.isfinite(metrics["velocity_full_rel_l2"]):
        _check_max(metrics, "velocity_full_rel_l2", thresholds, failures)
    _check_final_vs_best(report, float(thresholds.get("max_final_checkpoint_physics_degradation", 0.05)), failures)
    for name, exists in report["figures"].items():
        if not exists:
            failures.append(f"missing figure {name}")
    return report


def _check_max(metrics: dict[str, float], metric: str, thresholds: dict[str, Any], failures: list[str]) -> None:
    key = f"max_{metric}"
    maximum = float(thresholds.get(key, np.inf))
    value = metrics.get(metric, float("nan"))
    if not np.isfinite(value):
        failures.append(f"{metric}=nonfinite")
    elif value > maximum:
        failures.append(f"{metric}={value:.6g}>{maximum:.6g}")


def _check_min(metrics: dict[str, float], metric: str, minimum: float, failures: list[str]) -> None:
    value = metrics.get(metric, float("nan"))
    if not np.isfinite(value):
        failures.append(f"{metric}=nonfinite")
    elif value < minimum:
        failures.append(f"{metric}={value:.6g}<{minimum:.6g}")


def _check_max_value(
    metrics: dict[str, float],
    metric: str,
    maximum: float,
    failures: list[str],
) -> None:
    value = metrics.get(metric, float("nan"))
    if not np.isfinite(value):
        failures.append(f"{metric}=nonfinite")
    elif value > maximum:
        failures.append(f"{metric}={value:.6g}>{maximum:.6g}")


def _check_exact(metrics: dict[str, float], metric: str, expected: float, failures: list[str]) -> None:
    value = metrics.get(metric, float("nan"))
    if not np.isfinite(value) or abs(value - expected) > 1e-12:
        failures.append(f"{metric}={value!r}!={expected!r}")


def _check_final_vs_best(report: dict[str, Any], tolerance: float, failures: list[str]) -> None:
    final_value = report.get("unweighted_physics_validation_loss", float("nan"))
    best_metrics = report.get("best_checkpoint_metrics", {})
    best_value = _finite_sum(
        [
            best_metrics.get("unweighted_pde_loss"),
            best_metrics.get("unweighted_bc_loss"),
        ]
    )
    if not np.isfinite(best_value):
        best_value = best_metrics.get("unweighted_physics_validation_loss", float("nan"))
    if np.isfinite(final_value) and np.isfinite(best_value) and final_value > best_value * (1.0 + tolerance):
        failures.append(
            f"final physics loss {final_value:.6g} worse than best checkpoint {best_value:.6g} by > {tolerance:.3g}"
        )


def _compare_vara_to_vanilla(
    vanilla: dict[str, Any],
    vara: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    tolerance = float(thresholds.get("max_vara_vs_vanilla_physics_degradation", 0.02))
    vanilla_metrics = vanilla["metrics"]
    vara_metrics = vara["metrics"]
    for metric in ["boundary_condition_error", "pde_residual_mean", "momentum_residual_mean"]:
        v0 = vanilla_metrics.get(metric, float("nan"))
        v1 = vara_metrics.get(metric, float("nan"))
        if np.isfinite(v0) and np.isfinite(v1) and v1 > v0 * (1.0 + tolerance):
            failures.append(f"vara_v2 degraded {metric}: {v1:.6g}>{v0:.6g}*(1+{tolerance:.3g})")
    if vara_metrics.get("lid_cavity_topology_aligned", 0.0) < vanilla_metrics.get("lid_cavity_topology_aligned", 0.0):
        failures.append("vara_v2 degraded topology alignment")
    return failures


def _dominance_label(vanilla: dict[str, Any], vara: dict[str, Any]) -> str:
    metrics = ["pde_residual_mean", "momentum_residual_mean", "boundary_condition_error", "velocity_full_rel_l2"]
    wins = 0
    total = 0
    for metric in metrics:
        v0 = vanilla["metrics"].get(metric, float("nan"))
        v1 = vara["metrics"].get(metric, float("nan"))
        if np.isfinite(v0) and np.isfinite(v1):
            total += 1
            wins += int(v1 <= v0)
    return "VARA dominant" if total and wins == total else "VARA not dominant"


def _required_figures(method_dir: Path, *, has_reference: bool) -> dict[str, bool]:
    figures = method_dir / "figures"
    required = {
        "streamlines.png": (figures / "streamlines.png").exists(),
        "streamfunction_contours.png": (figures / "streamfunction_contours.png").exists(),
        "predicted_fields.png": (figures / "predicted_fields.png").exists(),
    }
    if has_reference:
        required["reference_streamlines.png"] = (figures / "reference_streamlines.png").exists()
    return required


def _row_for(df: pd.DataFrame, seed: int, method: str) -> pd.Series | None:
    subset = df[
        (pd.to_numeric(df["seed"], errors="coerce") == int(seed))
        & (pd.to_numeric(df["reynolds"], errors="coerce") == 100.0)
        & (df["method"].astype(str) == method)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def _resolved_config(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "summary" / "resolved_base_config.yaml"
    if path.exists():
        return load_config(path)
    return load_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")


def _raw_loss_snapshot(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[-1]
    out = {}
    for column in RAW_LOSS_COLUMNS:
        if column in row:
            out[f"raw_{column}"] = _to_float(row[column])
    for column in row.index:
        if str(column).startswith("loss_scale_"):
            out[str(column)] = _to_float(row[column])
    return out


def _checkpoint_metrics(method_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    best = _load_checkpoint_metrics(method_dir / "checkpoints" / "best.pt")
    final = _load_checkpoint_metrics(method_dir / "checkpoints" / "final.pt")
    return best, final


def _load_checkpoint_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return {
        str(key): _to_float(value)
        for key, value in dict(payload.get("metrics", {})).items()
        if isinstance(value, (int, float, np.floating, np.integer, bool))
    }


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            return False
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "pass", "passed"}


def _finite_sum(values: list[Any]) -> float:
    finite = [_to_float(value) for value in values]
    finite = [value for value in finite if np.isfinite(value)]
    return float(sum(finite)) if finite else float("nan")


if __name__ == "__main__":
    main()
