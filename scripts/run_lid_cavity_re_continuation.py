"""Run lid-driven cavity Reynolds continuation with Vanilla/VARA comparisons."""

from __future__ import annotations

import argparse
from copy import deepcopy
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Lid-driven cavity Reynolds continuation comparison.")
    parser.add_argument("--config", default="configs/lid_driven_cavity_continuation.yaml")
    parser.add_argument("--output_dir", default="experiments/lid_cavity_re_continuation_comparison")
    parser.add_argument("--reynolds", nargs="+", type=float, default=DEFAULT_REYNOLDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--method", choices=["vanilla", "vara", "both"], default="both")
    parser.add_argument("--reference", choices=["ghia", "external", "none"], default="ghia")
    parser.add_argument("--reference_path", default=None)
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

    methods = ["vanilla", "vara"] if args.method == "both" else [args.method]
    long_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        previous_checkpoint: dict[str, Path | None] = {method: None for method in methods}
        for reynolds in args.reynolds:
            re_dir = _re_dir(out, seed, reynolds)
            reference_info = _reference_for_re(float(reynolds), args.reference, args.reference_path)
            reference_rows.append(
                {
                    "seed": int(seed),
                    "reynolds": float(reynolds),
                    "ghia_profile_available": _has_builtin_ghia(float(reynolds)),
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
    _save_summary_montages(out, summary_dir)
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
            "profile_only": True,
            "full_field_reference_path": None,
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


def _has_builtin_ghia(reynolds: float) -> bool:
    return any(np.isclose(float(reynolds), value) for value in GHIA_REYNOLDS)


def _comparison_rows(seed: int, reynolds: float, vanilla: dict[str, Any], vara: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric in METRICS:
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
    pd.DataFrame(rows).to_csv(comparison_dir / "metrics_comparison.csv", index=False)
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
