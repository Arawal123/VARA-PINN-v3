"""Muted, publication-safe figures for PDE generalization experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from .benchmarks import Burgers2DBenchmark, ManufacturedBenchmark


PALETTE = ["#20242b", "#66717e", "#a6a9ad", "#514d4a", "#858079"]


def save_run_plots(
    model: nn.Module,
    benchmark: ManufacturedBenchmark,
    run_dir: Path,
    loss_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    allocation_history: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Save field, loss, allocation, and decision views for one run."""
    figures = Path(run_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _field_comparison(model, benchmark, figures, device=device, dtype=dtype)
    _loss_history(loss_rows, figures)
    if allocation_history:
        _allocation_heatmap(allocation_history, figures)
    if decision_rows:
        _decision_timeline(decision_rows, figures)


def save_aggregate_plots(raw: pd.DataFrame, winrates: pd.DataFrame, output_dir: Path) -> None:
    """Save the six cross-method plots required by the experiment brief."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "reconstruction_comparison": {
            "burgers2d": "burgers_velocity_rel_l2",
            "allen_cahn": "allen_cahn_u_rel_l2",
            "advection_diffusion": "advdiff_u_rel_l2",
        },
        "sparse_mse_comparison": {
            "burgers2d": "burgers_velocity_mse_sparse",
            "allen_cahn": "allen_cahn_u_mse_sparse",
            "advection_diffusion": "advdiff_u_mse_sparse",
        },
        "residual_comparison": {
            "burgers2d": "burgers_pde_residual_mean",
            "allen_cahn": "allen_cahn_pde_residual_mean",
            "advection_diffusion": "advdiff_pde_residual_mean",
        },
        "localized_hard_region_comparison": {
            "burgers2d": "burgers_localized_band_rel_l2",
            "allen_cahn": "allen_cahn_interface_band_rel_l2",
            "advection_diffusion": "advdiff_layer_band_rel_l2",
        },
    }
    for filename, mapping in metrics.items():
        _cross_pde_bar(raw, mapping, output_dir, filename)
    _runtime_bar(raw, output_dir)
    _winrate_plot(winrates, output_dir)


def _field_comparison(
    model: nn.Module,
    benchmark: ManufacturedBenchmark,
    figures: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    x0, x1, y0, y1 = benchmark.bounds
    final_t = benchmark.t_bounds[1]
    x = torch.linspace(x0, x1, 100, device=device, dtype=dtype)
    y = torch.linspace(y0, y1, 100, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack((xx, yy, torch.full_like(xx, final_t)), dim=-1).reshape(-1, 3)
    with torch.no_grad():
        reference = benchmark.exact(coords)
        prediction = model(coords)
        if isinstance(benchmark, Burgers2DBenchmark):
            reference_field = torch.linalg.vector_norm(reference, dim=1)
            prediction_field = torch.linalg.vector_norm(prediction, dim=1)
            label = "velocity magnitude"
        else:
            reference_field = reference[:, 0]
            prediction_field = prediction[:, 0]
            label = "u"
        error = (prediction_field - reference_field).abs()
    arrays = [
        reference_field.reshape(100, 100).cpu().numpy(),
        prediction_field.reshape(100, 100).cpu().numpy(),
        error.reshape(100, 100).cpu().numpy(),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    titles = [f"Reference {label}", f"PINN {label}", "Absolute error"]
    for axis, values, title in zip(axes, arrays, titles):
        image = axis.imshow(
            values.T,
            origin="lower",
            extent=[x0, x1, y0, y1],
            cmap="Greys" if title != "Absolute error" else "magma",
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, shrink=0.78)
    _save_both(fig, figures / "final_field_comparison")


def _loss_history(rows: list[dict[str, Any]], figures: Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    fig, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for index, column in enumerate(["loss_total", "loss_pde", "loss_bc", "loss_ic", "loss_sparse_data"]):
        if column in frame:
            axis.plot(frame["step"], frame[column].clip(lower=1e-16), label=column.removeprefix("loss_"), color=PALETTE[index])
    axis.set_yscale("log")
    axis.set_xlabel("Applied optimizer step")
    axis.set_ylabel("Loss")
    axis.set_title("Training objective history")
    axis.legend(frameon=False, ncol=3)
    axis.grid(alpha=0.18)
    _save_both(fig, figures / "loss_history")


def _allocation_heatmap(history: list[dict[str, Any]], figures: Path) -> None:
    mass = np.asarray(history[-1]["sampling_mass"], dtype=float)
    shape = history[-1].get("patch_grid_shape")
    if shape and int(np.prod(shape)) == mass.size:
        image = mass.reshape(tuple(int(value) for value in shape)).sum(axis=0)
    else:
        image = mass.reshape(1, -1)
    fig, axis = plt.subplots(figsize=(6.3, 3.8), constrained_layout=True)
    plotted = axis.imshow(image, cmap="Greys", aspect="auto")
    axis.set_title("Final VARA sampling allocation")
    axis.set_xlabel("x patch")
    axis.set_ylabel("y patch (time-aggregated)")
    fig.colorbar(plotted, ax=axis, label="Probability mass")
    _save_both(fig, figures / "vara_allocation_heatmap")


def _decision_timeline(rows: list[dict[str, Any]], figures: Path) -> None:
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    states = []
    for _, row in frame.iterrows():
        if bool(row.get("prefiltered", False)):
            states.append("prefiltered")
        elif bool(row.get("accepted", False)):
            states.append("accepted")
        elif row.get("rollback_reason", ""):
            states.append("rollback/rejected")
        else:
            states.append("rejected")
    mapping = {"accepted": 3, "rejected": 2, "prefiltered": 1, "rollback/rejected": 0}
    colors = {"accepted": PALETTE[0], "rejected": PALETTE[1], "prefiltered": PALETTE[2], "rollback/rejected": PALETTE[3]}
    for state in mapping:
        indexes = [index for index, value in enumerate(states) if value == state]
        axis.scatter(indexes, [mapping[state]] * len(indexes), label=state, color=colors[state], s=48)
    axis.set_yticks(list(mapping.values()), list(mapping.keys()))
    axis.set_xlabel("Decision index")
    axis.set_title("VARA controller decision timeline")
    axis.grid(axis="x", alpha=0.18)
    _save_both(fig, figures / "vara_decision_timeline")


def _cross_pde_bar(
    raw: pd.DataFrame,
    mapping: dict[str, str],
    output_dir: Path,
    filename: str,
) -> None:
    records: list[dict[str, Any]] = []
    for benchmark, metric in mapping.items():
        if metric not in raw:
            continue
        subset = raw[raw["benchmark"] == benchmark]
        for method, values in subset.groupby("method")[metric]:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            if numeric.empty:
                continue
            records.append({"benchmark": benchmark, "method": method, "mean": numeric.mean(), "std": numeric.std(ddof=1) if len(numeric) > 1 else 0.0})
    _grouped_bar(pd.DataFrame(records), output_dir / filename, filename.replace("_", " ").title(), "Error (lower is better)")


def _runtime_bar(raw: pd.DataFrame, output_dir: Path) -> None:
    records = []
    for (benchmark, method), values in raw.groupby(["benchmark", "method"])["optimization_wall_clock_sec"]:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        records.append({"benchmark": benchmark, "method": method, "mean": numeric.mean(), "std": numeric.std(ddof=1) if len(numeric) > 1 else 0.0})
    _grouped_bar(pd.DataFrame(records), output_dir / "runtime_comparison", "Optimization runtime", "Seconds")


def _winrate_plot(winrates: pd.DataFrame, output_dir: Path) -> None:
    if winrates.empty:
        return
    fig, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    labels = [f"{row.benchmark}\n{row.method}" for row in winrates.itertuples()]
    axis.bar(labels, winrates["win_rate_percent"], color=PALETTE[: len(labels)] if len(labels) <= len(PALETTE) else PALETTE[1])
    axis.axhline(50.0, color="#777777", linestyle="--", linewidth=1.0)
    axis.set_ylim(0.0, 105.0)
    axis.set_ylabel("Paired seed win rate (%)")
    axis.set_title("VARA reconstruction win rate")
    _save_both(fig, output_dir / "winrate_summary")


def _grouped_bar(frame: pd.DataFrame, path: Path, title: str, ylabel: str) -> None:
    if frame.empty:
        return
    benchmarks = list(dict.fromkeys(frame["benchmark"]))
    methods = list(dict.fromkeys(frame["method"]))
    x = np.arange(len(benchmarks), dtype=float)
    width = 0.75 / max(1, len(methods))
    fig, axis = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    for index, method in enumerate(methods):
        means, errors = [], []
        for benchmark in benchmarks:
            selected = frame[(frame["benchmark"] == benchmark) & (frame["method"] == method)]
            means.append(float(selected["mean"].iloc[0]) if not selected.empty else np.nan)
            errors.append(float(selected["std"].iloc[0]) if not selected.empty else 0.0)
        axis.bar(x + (index - (len(methods) - 1) / 2) * width, means, width, yerr=errors, capsize=3, color=PALETTE[index % len(PALETTE)], label=method)
    axis.set_xticks(x, [value.replace("_", "\n") for value in benchmarks])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(frameon=False)
    axis.margins(y=0.15)
    axis.grid(axis="y", alpha=0.18)
    _save_both(fig, path)


def _save_both(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
