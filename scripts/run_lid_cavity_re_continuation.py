"""Run lid-driven cavity Reynolds continuation with Vanilla/VARA comparisons."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
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

from scripts.benchmark_runner import BENCHMARK_DEFAULTS, METHOD_TO_MODE
from src.physics.cavity_reference import load_full_field_reference, validate_full_field_against_ghia
from src.training.vara_trainer import VARATrainer
from src.utils.config import deep_update, load_config, save_config
from src.utils.io import save_json


DEFAULT_REYNOLDS = [100, 150, 200, 300, 400, 600, 800, 1000]
GHIA_REYNOLDS = {100.0, 400.0, 1000.0}
METRICS = [
    "cavity_benchmark_score",
    "centerline_profile_score",
    "u_centerline_rmse",
    "v_centerline_rmse",
    "pde_residual_mean",
    "continuity_residual_mean",
    "momentum_residual_mean",
    "boundary_condition_error",
    "unweighted_validation_loss",
    "final_total_loss",
]
FULL_FIELD_METRICS = [
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
    "unweighted_data_loss",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Lid-driven cavity Reynolds continuation comparison.")
    parser.add_argument("--config", default="configs/lid_driven_cavity_continuation.yaml")
    parser.add_argument("--output_dir", default="experiments/lid_cavity_re_continuation_comparison")
    parser.add_argument("--reynolds", nargs="+", type=float, default=DEFAULT_REYNOLDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--method", choices=["vanilla", "vara", "both"], default="both")
    parser.add_argument("--reference", choices=["ghia", "external", "none"], default="ghia")
    parser.add_argument("--reference_path", default=None)
    parser.add_argument("--full_field_reference_map", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_continuation(args)


def run_continuation(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not bool(args.overwrite):
        raise SystemExit(f"Output directory already exists and is not empty: {out}. Use --overwrite or choose a new --output_dir.")
    if out.exists() and bool(args.overwrite):
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    base = deep_update(BENCHMARK_DEFAULTS["lid_driven_cavity"], load_config(args.config))
    base["run_type"] = "smoke" if args.quick else "benchmark"
    base["benchmark"] = "lid_driven_cavity"
    if args.device:
        base["device"] = args.device
    if args.quick:
        base = _quick_config(base)

    full_field_reference_map = _load_full_field_reference_map(getattr(args, "full_field_reference_map", None))
    methods = ["vanilla", "vara"] if args.method == "both" else [args.method]
    long_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        previous_checkpoint: dict[str, Path | None] = {method: None for method in methods}
        for reynolds in args.reynolds:
            re_dir = _re_dir(out, seed, reynolds)
            reference_info = _reference_for_re(float(reynolds), args.reference, args.reference_path)
            full_field_reference_path = _full_field_reference_for_re(float(reynolds), full_field_reference_map)
            reference_info["full_field_reference_path"] = full_field_reference_path
            profile_available = _has_profile_reference(float(reynolds), reference_info)
            full_field_available = full_field_reference_path is not None
            full_field_meta = _full_field_reference_metadata(full_field_reference_path) if full_field_available else {}
            reference_rows.append(
                {
                    "seed": int(seed),
                    "reynolds": float(reynolds),
                    "ghia_profile_available": _has_builtin_ghia(float(reynolds)),
                    "profile_reference_available": profile_available,
                    "full_field_reference_available": full_field_available,
                    "full_field_reference_path": str(full_field_reference_path) if full_field_reference_path else None,
                    "has_p_reference": full_field_meta.get("has_p_reference", False),
                    "has_omega_reference": full_field_meta.get("has_omega_reference", False),
                    "omega_reference_source": full_field_meta.get("omega_reference_source", ""),
                    "quantitative_reference_level": _quantitative_reference_level(profile_available, full_field_available),
                    "reference": reference_info["reference"],
                    "reference_path": reference_info.get("reference_path"),
                }
            )
            method_metrics: dict[str, dict[str, Any]] = {}

            for method in methods:
                method_dir = re_dir / method
                run_config = _config_for_run(
                    base=base,
                    seed=seed,
                    reynolds=reynolds,
                    method=method,
                    method_dir=method_dir,
                    reference_info=reference_info,
                    warm_start_checkpoint=previous_checkpoint[method],
                )
                mode = METHOD_TO_MODE[method]
                trainer = VARATrainer(run_config, mode=mode)
                metrics = trainer.run()
                metrics.update(
                    {
                        "benchmark": "lid_driven_cavity",
                        "method": method,
                        "mode": mode,
                        "seed": int(seed),
                        "reynolds": float(reynolds),
                        "run_dir": str(trainer.run_dir),
                        "checkpoint": str(trainer.checkpoint_dir / "final.pt"),
                        "method_dir": str(method_dir),
                    }
                )
                save_json(metrics, trainer.run_dir / "summary.json")
                pd.DataFrame([metrics]).to_csv(trainer.run_dir / "summary_table.csv", index=False)
                long_rows.append(metrics)
                method_metrics[method] = metrics
                previous_checkpoint[method] = trainer.checkpoint_dir / "final.pt"
                print(f"seed={seed} Re={reynolds:g} method={method}: {method_dir}")

            if {"vanilla", "vara"}.issubset(method_metrics):
                comparison_rows.extend(_comparison_rows(seed, reynolds, method_metrics["vanilla"], method_metrics["vara"]))
                _save_per_re_comparison(re_dir, method_metrics["vanilla"], method_metrics["vara"])

    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    long_df = pd.DataFrame(long_rows)
    compare_df = pd.DataFrame(comparison_rows)
    reference_df = pd.DataFrame(reference_rows).drop_duplicates()
    wide_df = _wide_improvement(compare_df)
    long_df.to_csv(summary_dir / "continuation_results_long.csv", index=False)
    compare_df.to_csv(summary_dir / "vara_vs_vanilla_by_re.csv", index=False)
    wide_df.to_csv(summary_dir / "improvement_percent_by_re.csv", index=False)
    reference_df.to_csv(summary_dir / "available_reference_metrics_by_re.csv", index=False)
    _save_cfd_reference_validation(full_field_reference_map, summary_dir)
    _save_summary_montages(out, summary_dir)
    _save_summary_bar_plots(compare_df, summary_dir)
    return {
        "continuation_results_long": long_df,
        "vara_vs_vanilla_by_re": compare_df,
        "improvement_percent_by_re": wide_df,
        "available_reference_metrics_by_re": reference_df,
    }


def _quick_config(config: dict[str, Any]) -> dict[str, Any]:
    return deep_update(
        config,
        {
            "model": {"hidden_layers": [16]},
            "training": {
                "adaptive_cycles": 1,
                "epochs_per_cycle": 1,
                "log_every": 1,
                "n_collocation": 16,
                "n_boundary": 16,
                "n_data": 0,
            },
            "local_controller": {
                "trial_epochs": 1,
                "warmup_cycles": 0,
                "max_actions_per_cycle": 1,
                "rejection_recovery_epochs": 0,
            },
            "optimizer": {
                "final_repair": {
                    "enabled": False,
                    "epochs": 0,
                },
            },
            "validation": {"nx": 6, "ny": 6},
            "test": {"nx": 8, "ny": 8},
            "patches": {"nx_patches": 2, "ny_patches": 2, "nt_patches": 1},
        },
    )


def _config_for_run(
    base: dict[str, Any],
    seed: int,
    reynolds: float,
    method: str,
    method_dir: Path,
    reference_info: dict[str, Any],
    warm_start_checkpoint: Path | None,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["seed"] = int(seed)
    cfg["benchmark_params"] = dict(cfg.get("benchmark_params", {}))
    cfg["benchmark_params"].update(
        {
            "reynolds": float(reynolds),
            "reference": reference_info["reference"],
            "reference_path": reference_info.get("reference_path"),
            "profile_only": reference_info.get("full_field_reference_path") is None,
            "full_field_reference_path": (
                str(reference_info["full_field_reference_path"]) if reference_info.get("full_field_reference_path") else None
            ),
        }
    )
    cfg["experiments"] = {
        "root": str(method_dir),
        "flat_layout": True,
        "run_id": f"seed{seed}_re{_re_label(reynolds)}_{method}",
    }
    if warm_start_checkpoint is not None:
        cfg["warm_start_checkpoint"] = str(warm_start_checkpoint)
        cfg["warm_start"] = {"load_optimizer": False}
    else:
        cfg.pop("warm_start_checkpoint", None)
        cfg.pop("warm_start", None)
    return cfg


def _reference_for_re(reynolds: float, reference: str, reference_path: str | None) -> dict[str, Any]:
    if reference == "none":
        return {"reference": "none", "reference_path": None}
    if reference == "external":
        return {"reference": "external", "reference_path": reference_path}
    if _has_builtin_ghia(reynolds):
        return {"reference": "ghia", "reference_path": None}
    return {"reference": "none", "reference_path": None}


def _load_full_field_reference_map(path: str | None) -> dict[float, Path]:
    if not path:
        return {}
    map_path = Path(path)
    if not map_path.exists() and not map_path.is_absolute():
        repo_candidate = ROOT / map_path
        if repo_candidate.exists():
            map_path = repo_candidate
    if not map_path.exists():
        raise FileNotFoundError(f"Full-field reference map not found: {path}")

    suffix = map_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(map_path)
        required = {"re", "full_field_reference_path"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Full-field reference map is missing columns: {sorted(missing)}")
        rows = df[["re", "full_field_reference_path"]].to_dict("records")
    elif suffix == ".json":
        with map_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = _json_reference_rows(payload)
    else:
        raise ValueError("--full_field_reference_map must be a CSV or JSON file.")

    out: dict[float, Path] = {}
    for row in rows:
        reynolds = float(row["re"])
        raw_path = row["full_field_reference_path"]
        resolved = _resolve_reference_path(raw_path, map_path.parent)
        if not resolved.exists():
            raise FileNotFoundError(f"Full-field reference for Re={reynolds:g} not found: {resolved}")
        out[reynolds] = resolved
    return out


def _json_reference_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "references" in payload:
        payload = payload["references"]
    if isinstance(payload, dict):
        return [{"re": key, "full_field_reference_path": value} for key, value in payload.items()]
    if isinstance(payload, list):
        rows = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("JSON full-field reference map list entries must be objects.")
            path = item.get("full_field_reference_path", item.get("path"))
            if "re" not in item or path is None:
                raise ValueError("JSON full-field reference map entries must include re and full_field_reference_path.")
            rows.append({"re": item["re"], "full_field_reference_path": path})
        return rows
    raise ValueError("JSON full-field reference map must be an object, a list, or an object with a references list.")


def _resolve_reference_path(raw_path: str | Path, map_dir: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    candidates = [ROOT / path, map_dir / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return ROOT / path


def _full_field_reference_for_re(reynolds: float, reference_map: dict[float, Path]) -> Path | None:
    for mapped_re, path in reference_map.items():
        if np.isclose(float(reynolds), float(mapped_re)):
            return path
    return None


def _full_field_reference_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    reference = load_full_field_reference(path)
    return {
        "has_p_reference": bool(reference.get("has_p_reference", False)),
        "has_omega_reference": bool(reference.get("has_omega_reference", False)),
        "omega_reference_source": str(reference.get("omega_reference_source", "")),
    }


def _save_cfd_reference_validation(reference_map: dict[float, Path], summary_dir: Path) -> None:
    rows = []
    for reynolds, path in sorted(reference_map.items()):
        if _has_builtin_ghia(reynolds):
            rows.append(validate_full_field_against_ghia(path, reynolds))
    if rows:
        pd.DataFrame(rows).to_csv(summary_dir / "cfd_reference_vs_ghia_validation.csv", index=False)


def _has_profile_reference(reynolds: float, reference_info: dict[str, Any]) -> bool:
    reference = str(reference_info.get("reference", "none")).lower()
    if reference == "ghia":
        return _has_builtin_ghia(reynolds)
    if reference == "external":
        return reference_info.get("reference_path") is not None
    return False


def _quantitative_reference_level(profile_available: bool, full_field_available: bool) -> str:
    if profile_available and full_field_available:
        return "profile+full_field"
    if profile_available:
        return "profile_only"
    if full_field_available:
        return "full_field_only"
    return "residual_only"


def _has_builtin_ghia(reynolds: float) -> bool:
    return any(np.isclose(float(reynolds), value) for value in GHIA_REYNOLDS)


def _comparison_rows(seed: int, reynolds: float, vanilla: dict[str, Any], vara: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    metric_names = list(METRICS)
    for metric in FULL_FIELD_METRICS:
        v = _to_float(vanilla.get(metric))
        a = _to_float(vara.get(metric))
        if np.isfinite(v) or np.isfinite(a):
            metric_names.append(metric)
    for metric in metric_names:
        v = _to_float(vanilla.get(metric))
        a = _to_float(vara.get(metric))
        improvement = (v - a) / abs(v) * 100.0 if np.isfinite(v) and v != 0.0 and np.isfinite(a) else np.nan
        rows.append(
            {
                "seed": int(seed),
                "reynolds": float(reynolds),
                "metric": metric,
                "vanilla": v,
                "vara": a,
                "improvement_percent": improvement,
            }
        )
    return rows


def _wide_improvement(compare_df: pd.DataFrame) -> pd.DataFrame:
    if compare_df.empty:
        return pd.DataFrame()
    return (
        compare_df.pivot_table(
            index=["seed", "reynolds"],
            columns="metric",
            values="improvement_percent",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )


def _save_per_re_comparison(re_dir: Path, vanilla: dict[str, Any], vara: dict[str, Any]) -> None:
    comparison_dir = re_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = _comparison_rows(int(vanilla["seed"]), float(vanilla["reynolds"]), vanilla, vara)
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(comparison_dir / "metrics_comparison.csv", index=False)
    _save_metric_comparison_bar(
        comparison_df,
        METRICS,
        comparison_dir / "metric_comparison_bar.png",
        f"Re={float(vanilla['reynolds']):g} Vanilla vs VARA metrics (lower is better)",
    )
    if comparison_df["metric"].isin(FULL_FIELD_METRICS).any():
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
    _save_image_grid(
        [
            (Path(vanilla["method_dir"]) / "figures" / "streamlines.png", "Vanilla streamlines"),
            (Path(vara["method_dir"]) / "figures" / "streamlines.png", "VARA streamlines"),
        ],
        comparison_dir / "streamlines_side_by_side.png",
        cols=2,
        title=f"Re={float(vanilla['reynolds']):g} streamlines",
    )
    _save_image_grid(
        [
            (Path(vanilla["method_dir"]) / "figures" / "pde_residual.png", "Vanilla PDE"),
            (Path(vara["method_dir"]) / "figures" / "pde_residual.png", "VARA PDE"),
            (Path(vanilla["method_dir"]) / "figures" / "continuity_residual.png", "Vanilla continuity"),
            (Path(vara["method_dir"]) / "figures" / "continuity_residual.png", "VARA continuity"),
            (Path(vanilla["method_dir"]) / "figures" / "momentum_residual.png", "Vanilla momentum"),
            (Path(vara["method_dir"]) / "figures" / "momentum_residual.png", "VARA momentum"),
        ],
        comparison_dir / "residuals_side_by_side.png",
        cols=2,
        title=f"Re={float(vanilla['reynolds']):g} residuals",
    )
    error_items = [
        (Path(vanilla["method_dir"]) / "figures" / "cfd_prediction_error_fields.png", "Vanilla CFD errors"),
        (Path(vara["method_dir"]) / "figures" / "cfd_prediction_error_fields.png", "VARA CFD errors"),
    ]
    if any(path.exists() for path, _ in error_items):
        _save_image_grid(
            error_items,
            comparison_dir / "full_field_error_side_by_side.png",
            cols=2,
            title=f"Re={float(vanilla['reynolds']):g} full-field errors",
        )
        _save_image_grid(
            error_items,
            comparison_dir / "cfd_error_side_by_side.png",
            cols=2,
            title=f"Re={float(vanilla['reynolds']):g} CFD full-field errors",
        )


def _save_summary_montages(out: Path, summary_dir: Path) -> None:
    for method in ["vanilla", "vara"]:
        items = [
            (path, _montage_label(path, method))
            for path in sorted(out.glob(f"seed_*/re_*/{method}/figures/streamlines.png"))
        ]
        _save_image_grid(items, summary_dir / f"streamline_montage_{method}.png", cols=4, title=f"{method.upper()} continuation streamlines")
    side_by_side = [
        (path, _montage_label(path, "comparison"))
        for path in sorted(out.glob("seed_*/re_*/comparison/streamlines_side_by_side.png"))
    ]
    _save_image_grid(side_by_side, summary_dir / "streamline_montage_side_by_side.png", cols=2, title="Vanilla vs VARA streamlines")


def _save_summary_bar_plots(compare_df: pd.DataFrame, summary_dir: Path) -> None:
    _save_improvement_by_re_bar(
        compare_df,
        METRICS,
        summary_dir / "metric_improvement_by_re_bar.png",
        "VARA improvement over Vanilla by Reynolds number",
    )
    if not compare_df.empty and compare_df["metric"].isin(FULL_FIELD_METRICS).any():
        _save_improvement_by_re_bar(
            compare_df,
            FULL_FIELD_METRICS,
            summary_dir / "full_field_metric_improvement_by_re_bar.png",
            "Full-field CFD metric improvement over Vanilla by Reynolds number",
        )
        _save_improvement_by_re_bar(
            compare_df,
            FULL_FIELD_METRICS,
            summary_dir / "cfd_metric_improvement_by_re_bar.png",
            "CFD metric improvement over Vanilla by Reynolds number",
        )


def _save_metric_comparison_bar(metric_df: pd.DataFrame, metrics: list[str], path: Path, title: str) -> None:
    subset = metric_df[metric_df["metric"].isin(metrics)].copy()
    if subset.empty:
        return
    subset["vanilla"] = pd.to_numeric(subset["vanilla"], errors="coerce")
    subset["vara"] = pd.to_numeric(subset["vara"], errors="coerce")
    subset["improvement_percent"] = pd.to_numeric(subset["improvement_percent"], errors="coerce")
    subset = subset[np.isfinite(subset["vanilla"]) & np.isfinite(subset["vara"])]
    if subset.empty:
        return

    x = np.arange(len(subset))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(10.0, 1.2 * len(subset)), 5.5), constrained_layout=True)
    ax.bar(x - width / 2, subset["vanilla"], width, label="Vanilla", color="#6b7280")
    ax.bar(x + width / 2, subset["vara"], width, label="VARA", color="#16a34a")
    for i, row in enumerate(subset.itertuples(index=False)):
        improvement = getattr(row, "improvement_percent")
        if np.isfinite(improvement):
            top = max(float(getattr(row, "vanilla")), float(getattr(row, "vara")))
            ax.text(i, top * 1.03 if top != 0.0 else 0.03, f"{improvement:+.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("metric value")
    ax.set_xticks(x)
    ax.set_xticklabels([str(metric).replace("_", "\n") for metric in subset["metric"]], rotation=0, fontsize=8)
    ax.legend()
    ax.text(
        0.0,
        -0.18,
        "Positive annotation means VARA is lower than Vanilla.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_improvement_by_re_bar(compare_df: pd.DataFrame, metrics: list[str], path: Path, title: str) -> None:
    if compare_df.empty:
        return
    subset = compare_df[compare_df["metric"].isin(metrics)].copy()
    if subset.empty:
        return
    subset["improvement_percent"] = pd.to_numeric(subset["improvement_percent"], errors="coerce")
    table = subset.pivot_table(index="reynolds", columns="metric", values="improvement_percent", aggfunc="mean")
    metric_names = [metric for metric in metrics if metric in table.columns and np.isfinite(table[metric]).any()]
    if not metric_names:
        return

    cols = 2
    rows = int(np.ceil(len(metric_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 3.6 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    re_labels = [f"{float(re):g}" for re in table.index]
    for ax, metric in zip(axes_arr, metric_names):
        values = pd.to_numeric(table[metric], errors="coerce").to_numpy(dtype=float)
        colors = ["#16a34a" if np.isfinite(value) and value >= 0 else "#dc2626" for value in values]
        ax.bar(re_labels, values, color=colors)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("Re")
        ax.set_ylabel("VARA improvement %")
        ax.tick_params(axis="x", rotation=45)
    for ax in axes_arr[len(metric_names) :]:
        ax.axis("off")
    fig.suptitle(f"{title}\nPositive means lower error/loss/residual for VARA.", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_image_grid(items: list[tuple[Path, str]], path: Path, cols: int, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        return
    cols = max(1, int(cols))
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (image_path, label) in zip(axes_arr, items):
        ax.axis("off")
        if image_path.exists():
            ax.imshow(plt.imread(image_path))
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_title(label, fontsize=10)
    for ax in axes_arr[len(items) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _montage_label(path: Path, method: str) -> str:
    parts = path.parts
    seed = next((part for part in parts if part.startswith("seed_")), "")
    re = next((part for part in parts if part.startswith("re_")), "")
    return f"{seed} {re} {method}".strip()


def _re_dir(out: Path, seed: int, reynolds: float) -> Path:
    return out / f"seed_{int(seed)}" / f"re_{_re_label(reynolds)}"


def _re_label(reynolds: float) -> str:
    value = float(reynolds)
    if value.is_integer():
        return f"{int(value):04d}"
    return f"{value:g}".replace(".", "p")


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()
