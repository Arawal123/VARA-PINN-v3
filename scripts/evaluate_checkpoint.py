"""Evaluate saved checkpoints, including post-hoc lid-cavity CFD references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_lid_cavity_re_continuation import (
    FULL_FIELD_METRICS,
    METRICS as CONTINUATION_METRICS,
    _comparison_rows,
    _full_field_reference_for_re,
    _has_builtin_ghia,
    _load_full_field_reference_map,
    _save_improvement_by_re_bar,
    _save_metric_comparison_bar,
    _wide_improvement,
)
from src.diagnostics import DiagnosticMapBuilder
from src.evaluation.metrics import evaluate_on_grid
from src.models import build_mlp_from_config
from src.physics.cavity_reference import load_full_field_reference, validate_full_field_against_ghia
from src.physics.kovasznay import KovasznayFlow
from src.physics.rectangular_benchmarks import LidDrivenCavityQualitative
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.io import save_json
from src.visualization.fields import save_field_panel, save_prediction_reference_error_panel


CFD_OUTPUT_METRICS = [
    "u_full_rel_l2",
    "v_full_rel_l2",
    "velocity_full_rel_l2",
    "p_full_rel_l2_centered",
    "omega_full_rel_l2",
    "velocity_mag_rmse",
    "velocity_mag_mae",
    "has_p_full_field_reference",
    "has_omega_full_field_reference",
    "omega_full_field_reference_source",
    "full_field_reference_path",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints without retraining.")
    parser.add_argument("--checkpoint", default=None, help="Single checkpoint path.")
    parser.add_argument("--config", default=None, help="Config snapshot for --checkpoint.")
    parser.add_argument("--results_dir", default=None, help="Continuation output folder containing seed_*/re_* runs.")
    parser.add_argument("--full_field_reference_map", default=None, help="CSV/JSON map with re,full_field_reference_path.")
    parser.add_argument("--full_field_reference_path", default=None, help="Single full-field reference for --checkpoint.")
    parser.add_argument("--output_dir", default=None, help="Directory for combined post-hoc tables.")
    parser.add_argument("--merge_into_summary", action="store_true", help="Append CFD metrics into existing summary JSON/CSV files.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.results_dir:
        evaluate_results_dir(args)
        return
    if not args.checkpoint or not args.config:
        raise SystemExit("Use either --results_dir or both --checkpoint and --config.")
    metrics = evaluate_single_checkpoint(
        checkpoint=Path(args.checkpoint),
        config_path=Path(args.config),
        full_field_reference_path=Path(args.full_field_reference_path) if args.full_field_reference_path else None,
        device_override=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True, default=str))


def evaluate_results_dir(args: argparse.Namespace) -> pd.DataFrame:
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    reference_map = _load_full_field_reference_map(args.full_field_reference_map)
    if not reference_map:
        raise SystemExit("--full_field_reference_map is required for CFD reference evaluation.")

    rows: list[dict[str, Any]] = []
    for checkpoint in sorted(results_dir.glob("seed_*/re_*/*/checkpoints/final.pt")):
        if checkpoint.parents[1].name not in {"vanilla", "vara", "rar"}:
            continue
        rows.extend(_evaluate_checkpoint_path(checkpoint, reference_map, args.device, args.merge_into_summary))

    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cfd_reference_evaluation_long.csv", index=False)
    _update_continuation_outputs(results_dir, df, reference_map, bool(args.merge_into_summary))
    print(f"Saved CFD reference evaluation rows: {out_dir / 'cfd_reference_evaluation_long.csv'}")
    return df


def _evaluate_checkpoint_path(
    checkpoint: Path,
    reference_map: dict[float, Path],
    device_override: str | None,
    merge_into_summary: bool,
) -> list[dict[str, Any]]:
    method_dir = checkpoint.parents[1]
    config_path = method_dir / "logs" / "config_snapshot.yaml"
    if not config_path.exists():
        return []
    config = load_config(config_path)
    reynolds = float(config.get("benchmark_params", {}).get("reynolds", _re_from_path(method_dir)))
    reference_path = _full_field_reference_for_re(reynolds, reference_map)
    if reference_path is None:
        return []
    metrics = evaluate_single_checkpoint(
        checkpoint=checkpoint,
        config_path=config_path,
        full_field_reference_path=reference_path,
        device_override=device_override,
        method_dir=method_dir,
    )
    logs_dir = method_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    cfd_metrics = {key: metrics.get(key, np.nan) for key in CFD_OUTPUT_METRICS}
    cfd_metrics.update(
        {
            "seed": metrics.get("seed", config.get("seed")),
            "reynolds": reynolds,
            "method": metrics.get("method", method_dir.name),
            "mode": metrics.get("mode", ""),
            "checkpoint": str(checkpoint),
        }
    )
    save_json(cfd_metrics, logs_dir / "cfd_reference_metrics.json")
    pd.DataFrame([cfd_metrics]).to_csv(logs_dir / "cfd_reference_metrics.csv", index=False)
    if merge_into_summary:
        _merge_metrics_into_run_summary(logs_dir, metrics)
    return [metrics]


def evaluate_single_checkpoint(
    checkpoint: Path,
    config_path: Path,
    full_field_reference_path: Path | None,
    device_override: str | None = None,
    method_dir: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if full_field_reference_path is not None:
        params = dict(config.get("benchmark_params", {}))
        params["full_field_reference_path"] = str(full_field_reference_path)
        params["profile_only"] = False
        config["benchmark_params"] = params
    device = get_device(device_override or config.get("device", "auto"))
    benchmark = _build_benchmark(config)
    model = build_mlp_from_config(config, benchmark.bounds).to(device)
    payload = load_checkpoint(checkpoint, model, optimizer=None)
    _, _, coords = benchmark.grid(int(config.get("test", {}).get("nx", 64)), int(config.get("test", {}).get("ny", 64)))
    metrics = evaluate_on_grid(model, benchmark, coords, device, steady=bool(config.get("pde", {}).get("steady", True)))
    prior_metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    metrics.update(
        {
            "benchmark": config.get("benchmark", "unknown"),
            "method": prior_metrics.get("method", method_dir.name if method_dir else ""),
            "mode": prior_metrics.get("mode", ""),
            "seed": int(config.get("seed", prior_metrics.get("seed", -1))),
            "reynolds": float(config.get("benchmark_params", {}).get("reynolds", np.nan)),
            "checkpoint": str(checkpoint),
            "run_dir": str(method_dir / "logs") if method_dir else "",
            "method_dir": str(method_dir) if method_dir else "",
            "reference_kind": getattr(benchmark, "reference_kind", "unknown"),
            "has_reference": bool(getattr(benchmark, "has_reference", False)),
        }
    )
    if method_dir is not None and getattr(benchmark, "has_reference", False):
        _save_cfd_plots(model, benchmark, config, device, method_dir / "figures")
    return metrics


def _build_benchmark(config: dict[str, Any]) -> Any:
    name = str(config.get("benchmark", "kovasznay")).lower()
    cfg = config.get("benchmark_params", {})
    if name == "kovasznay":
        return KovasznayFlow(reynolds=float(cfg.get("reynolds", 40.0)))
    if name in {"lid_driven_cavity", "cavity"}:
        full_field_reference_path = cfg.get("full_field_reference_path")
        return LidDrivenCavityQualitative(
            reynolds=float(cfg.get("reynolds", 100.0)),
            x_min=float(cfg.get("x_min", 0.0)),
            x_max=float(cfg.get("x_max", 1.0)),
            y_min=float(cfg.get("y_min", 0.0)),
            y_max=float(cfg.get("y_max", 1.0)),
            amplitude=float(cfg.get("amplitude", 1.0)),
            lid_velocity=float(cfg.get("lid_velocity", 1.0)),
            reference=str(cfg.get("reference", "none")),
            reference_path=cfg.get("reference_path"),
            full_field_reference_path=full_field_reference_path,
            profile_only=bool(cfg.get("profile_only", full_field_reference_path is None)),
            has_reference=bool(full_field_reference_path) and not bool(cfg.get("profile_only", False)),
            reference_kind="full_field_cfd" if full_field_reference_path else str(cfg.get("reference", "residual_only")),
        )
    raise NotImplementedError(f"evaluate_checkpoint.py currently supports kovasznay and lid_driven_cavity, got {name}.")


def _save_cfd_plots(model: torch.nn.Module, benchmark: Any, config: dict[str, Any], device: torch.device, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    _, _, coords = benchmark.grid(int(config.get("test", {}).get("nx", 64)), int(config.get("test", {}).get("ny", 64)))
    X, Y, _ = benchmark.grid(int(config.get("test", {}).get("nx", 64)), int(config.get("test", {}).get("ny", 64)))
    builder = DiagnosticMapBuilder(model, benchmark, device, steady=bool(config.get("pde", {}).get("steady", True)))
    maps = builder.build(coords, mode="full_reference")
    ref = benchmark.exact_np(coords)
    shape = X.shape
    reference_fields = {
        "u ref": maps["u_ref"].reshape(shape),
        "v ref": maps["v_ref"].reshape(shape),
        "speed ref": maps["speed_ref"].reshape(shape),
    }
    error_fields = {
        "u error": maps["u_error"].reshape(shape),
        "v error": maps["v_error"].reshape(shape),
        "speed error": maps["speed_error"].reshape(shape),
    }
    triplets = {
        "u": (maps["u_pred"].reshape(shape), maps["u_ref"].reshape(shape), maps["u_error"].reshape(shape)),
        "v": (maps["v_pred"].reshape(shape), maps["v_ref"].reshape(shape), maps["v_error"].reshape(shape)),
        "speed": (
            maps["speed_pred"].reshape(shape),
            maps["speed_ref"].reshape(shape),
            maps["speed_error"].reshape(shape),
        ),
    }
    if bool(ref.get("has_p_reference", False)):
        reference_fields["p ref centered"] = maps["p_ref"].reshape(shape)
        error_fields["p error centered"] = maps["p_error_mean_centered"].reshape(shape)
        triplets["p"] = (
            maps["p_pred"].reshape(shape),
            maps["p_ref"].reshape(shape),
            maps["p_error_mean_centered"].reshape(shape),
        )
    if bool(ref.get("has_omega_reference", False)):
        reference_fields["omega ref"] = maps["omega_ref"].reshape(shape)
        error_fields["omega error"] = maps["omega_error"].reshape(shape)
        triplets["omega"] = (
            maps["omega_pred"].reshape(shape),
            maps["omega_ref"].reshape(shape),
            maps["omega_error"].reshape(shape),
        )
    save_field_panel(X, Y, reference_fields, fig_dir / "cfd_reference_fields.png")
    save_field_panel(X, Y, error_fields, fig_dir / "cfd_prediction_error_fields.png", cmap="magma")
    save_prediction_reference_error_panel(X, Y, triplets, fig_dir / "cfd_prediction_reference_error.png")


def _merge_metrics_into_run_summary(logs_dir: Path, metrics: dict[str, Any]) -> None:
    summary_path = logs_dir / "summary.json"
    table_path = logs_dir / "summary_table.csv"
    existing: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.update({key: _jsonable(value) for key, value in metrics.items()})
    save_json(existing, summary_path)
    pd.DataFrame([existing]).to_csv(table_path, index=False)


def _update_continuation_outputs(
    results_dir: Path,
    cfd_df: pd.DataFrame,
    reference_map: dict[float, Path],
    merge_into_summary: bool,
) -> None:
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if cfd_df.empty:
        return
    long_path = summary_dir / "continuation_results_long.csv"
    if long_path.exists():
        long_df = pd.read_csv(long_path)
        metric_cols = [col for col in CFD_OUTPUT_METRICS if col in cfd_df.columns]
        merge_cols = ["seed", "reynolds", "method", *metric_cols]
        long_df = long_df.merge(cfd_df[merge_cols], on=["seed", "reynolds", "method"], how="left", suffixes=("", "_cfd_new"))
        for col in metric_cols:
            new_col = f"{col}_cfd_new"
            if new_col in long_df.columns:
                if col in long_df.columns:
                    long_df[col] = long_df[new_col].combine_first(long_df[col])
                else:
                    long_df[col] = long_df[new_col]
                long_df = long_df.drop(columns=[new_col])
    else:
        long_df = cfd_df.copy()
    long_df.to_csv(long_path, index=False)

    comparison_rows = []
    for (seed, reynolds), group in long_df.groupby(["seed", "reynolds"]):
        vanilla = group[group["method"] == "vanilla"]
        vara = group[group["method"] == "vara"]
        if vanilla.empty or vara.empty:
            continue
        comparison_rows.extend(_comparison_rows(int(seed), float(reynolds), vanilla.iloc[-1].to_dict(), vara.iloc[-1].to_dict()))
        _update_per_re_comparison(results_dir / f"seed_{int(seed)}" / f"re_{_re_label(float(reynolds))}", vanilla.iloc[-1].to_dict(), vara.iloc[-1].to_dict())
    compare_df = pd.DataFrame(comparison_rows)
    compare_df.to_csv(summary_dir / "vara_vs_vanilla_by_re.csv", index=False)
    _wide_improvement(compare_df).to_csv(summary_dir / "improvement_percent_by_re.csv", index=False)
    _save_improvement_by_re_bar(compare_df, CONTINUATION_METRICS, summary_dir / "metric_improvement_by_re_bar.png", "VARA improvement over Vanilla by Reynolds number")
    _save_improvement_by_re_bar(compare_df, FULL_FIELD_METRICS, summary_dir / "full_field_metric_improvement_by_re_bar.png", "Full-field CFD metric improvement over Vanilla by Reynolds number")
    _save_improvement_by_re_bar(compare_df, FULL_FIELD_METRICS, summary_dir / "cfd_metric_improvement_by_re_bar.png", "CFD metric improvement over Vanilla by Reynolds number")
    _update_reference_availability(summary_dir, reference_map)


def _update_per_re_comparison(re_dir: Path, vanilla: dict[str, Any], vara: dict[str, Any]) -> None:
    comparison_dir = re_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = _comparison_rows(int(vanilla["seed"]), float(vanilla["reynolds"]), vanilla, vara)
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(comparison_dir / "metrics_comparison.csv", index=False)
    _save_metric_comparison_bar(
        comparison_df,
        CONTINUATION_METRICS,
        comparison_dir / "metric_comparison_bar.png",
        f"Re={float(vanilla['reynolds']):g} Vanilla vs VARA metrics (lower is better)",
    )
    _save_metric_comparison_bar(
        comparison_df,
        FULL_FIELD_METRICS,
        comparison_dir / "full_field_metric_comparison_bar.png",
        f"Re={float(vanilla['reynolds']):g} full-field CFD metrics (lower is better)",
    )
    _save_metric_comparison_bar(
        comparison_df,
        FULL_FIELD_METRICS,
        comparison_dir / "cfd_metric_comparison_bar.png",
        f"Re={float(vanilla['reynolds']):g} CFD reference metrics (lower is better)",
    )


def _update_reference_availability(summary_dir: Path, reference_map: dict[float, Path]) -> None:
    rows = []
    for reynolds, path in sorted(reference_map.items()):
        ref = load_full_field_reference(path)
        profile_available = _has_builtin_ghia(reynolds)
        rows.append(
            {
                "reynolds": float(reynolds),
                "ghia_profile_available": profile_available,
                "profile_reference_available": profile_available,
                "full_field_reference_available": True,
                "full_field_reference_path": str(path),
                "has_p_reference": bool(ref.get("has_p_reference", False)),
                "has_omega_reference": bool(ref.get("has_omega_reference", False)),
                "omega_reference_source": str(ref.get("omega_reference_source", "")),
                "quantitative_reference_level": "profile+full_field" if profile_available else "full_field_only",
            }
        )
    if rows:
        updates = pd.DataFrame(rows)
        availability_path = summary_dir / "available_reference_metrics_by_re.csv"
        if availability_path.exists():
            existing = pd.read_csv(availability_path)
            combined = existing.merge(updates, on="reynolds", how="outer", suffixes=("", "_cfd_new"))
            for col in updates.columns:
                if col == "reynolds":
                    continue
                new_col = f"{col}_cfd_new"
                if new_col in combined.columns:
                    if col in combined.columns:
                        combined[col] = combined[new_col].combine_first(combined[col])
                    else:
                        combined[col] = combined[new_col]
                    combined = combined.drop(columns=[new_col])
            combined.to_csv(availability_path, index=False)
        else:
            updates.to_csv(availability_path, index=False)
    ghia_rows = [validate_full_field_against_ghia(path, reynolds) for reynolds, path in sorted(reference_map.items()) if _has_builtin_ghia(reynolds)]
    if ghia_rows:
        pd.DataFrame(ghia_rows).to_csv(summary_dir / "cfd_reference_vs_ghia_validation.csv", index=False)


def _re_from_path(path: Path) -> float:
    for part in path.parts:
        if part.startswith("re_"):
            return float(part.replace("re_", ""))
    return float("nan")


def _re_label(reynolds: float) -> str:
    value = float(reynolds)
    if value.is_integer():
        return f"{int(value):04d}"
    return f"{value:g}".replace(".", "p")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
