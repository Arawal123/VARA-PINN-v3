"""Publication-safe Cahn--Hilliard figures."""

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

from .benchmark import CahnHilliardBenchmark


PALETTE = ["#20242b", "#68727e", "#a5a7aa", "#514d4a", "#858079"]


def save_run_plots(
    model: nn.Module,
    benchmark: CahnHilliardBenchmark,
    run_dir: Path,
    loss_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    allocation_history: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Save fixed-scale fields and run-specific controller evidence."""
    figures = Path(run_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _field_panels(model, benchmark, figures, device=device, dtype=dtype)
    _loss_history(loss_rows, figures)
    if decision_rows:
        _decision_timeline(decision_rows, figures)
    if allocation_history:
        _allocation_heatmap(allocation_history, figures)


def save_aggregate_plots(raw: pd.DataFrame, output_dir: Path) -> None:
    """Save cross-method reconstruction, sparse, physics, interface, and runtime bars."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "global_u_relative_l2": "cahn_hilliard_u_rel_l2",
        "sparse_u_mse": "cahn_hilliard_sparse_u_mse",
        "pde_residual": "cahn_hilliard_pde_residual_mean",
        "interface_band_relative_l2": "cahn_hilliard_interface_band_rel_l2",
        "runtime_comparison": "optimization_wall_clock_sec",
    }
    for filename, metric in metrics.items():
        _metric_bar(raw, metric, output / filename)


def _field_panels(
    model: nn.Module,
    benchmark: CahnHilliardBenchmark,
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
    coordinates = torch.stack(
        (xx, yy, torch.full_like(xx, final_t)), dim=-1
    ).reshape(-1, 3)
    references, predictions = [], []
    for chunk in coordinates.split(1000):
        references.append(benchmark.exact(chunk).detach().cpu())
        with torch.no_grad():
            predictions.append(model(chunk).detach().cpu())
    reference = torch.cat(references)[:, 0].reshape(100, 100).numpy()
    prediction = torch.cat(predictions)[:, 0].reshape(100, 100).numpy()
    error = np.abs(prediction - reference)
    interface_error = np.where(np.abs(reference) < 0.8, error, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.6), constrained_layout=True)
    panels = [reference, prediction, error]
    titles = ["Reference phase field", "PINN phase field", "Absolute error"]
    for axis, values, title in zip(axes, panels, titles):
        is_error = title == "Absolute error"
        image = axis.imshow(
            values.T,
            origin="lower",
            extent=[x0, x1, y0, y1],
            cmap="magma" if is_error else "Greys",
            vmin=0.0 if is_error else -1.05,
            vmax=2.1 if is_error else 1.05,
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, shrink=0.8)
    _save_both(fig, figures / "reference_prediction_error")

    fig, axis = plt.subplots(figsize=(5.3, 4.2), constrained_layout=True)
    image = axis.imshow(
        interface_error.T,
        origin="lower",
        extent=[x0, x1, y0, y1],
        cmap="magma",
        vmin=0.0,
        vmax=2.1,
        aspect="equal",
    )
    axis.set_title("Interface-band absolute error")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    fig.colorbar(image, ax=axis, label="Absolute error")
    _save_both(fig, figures / "interface_band_error")


def _loss_history(rows: list[dict[str, Any]], figures: Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    columns = [
        "loss_total",
        "loss_ch_residual",
        "loss_chemical_potential_residual",
        "loss_sparse_u_mse",
    ]
    fig, axis = plt.subplots(figsize=(7.1, 4.0), constrained_layout=True)
    for index, column in enumerate(columns):
        if column in frame:
            axis.plot(
                frame["step"],
                frame[column].clip(lower=1e-16),
                color=PALETTE[index],
                label=column.removeprefix("loss_"),
            )
    axis.set_yscale("log")
    axis.set_xlabel("Applied optimizer step")
    axis.set_ylabel("Loss")
    axis.set_title("Cahn–Hilliard training history")
    axis.legend(frameon=False)
    axis.grid(alpha=0.18)
    _save_both(fig, figures / "loss_history")


def _decision_timeline(rows: list[dict[str, Any]], figures: Path) -> None:
    frame = pd.DataFrame(rows)
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
    positions = {"accepted": 3, "rejected": 2, "prefiltered": 1, "rollback/rejected": 0}
    fig, axis = plt.subplots(figsize=(7.1, 3.6), constrained_layout=True)
    for index, state in enumerate(positions):
        selected = [i for i, value in enumerate(states) if value == state]
        axis.scatter(
            selected,
            [positions[state]] * len(selected),
            color=PALETTE[index],
            label=state,
            s=48,
        )
    axis.set_yticks(list(positions.values()), list(positions.keys()))
    axis.set_xlabel("Decision index")
    axis.set_title("VARA decision timeline")
    axis.grid(axis="x", alpha=0.18)
    _save_both(fig, figures / "vara_decision_timeline")


def _allocation_heatmap(history: list[dict[str, Any]], figures: Path) -> None:
    mass = np.asarray(history[-1]["sampling_mass"], dtype=float)
    shape = tuple(int(value) for value in history[-1]["patch_grid_shape"])
    spatial = mass.reshape(shape).sum(axis=0)
    fig, axis = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    image = axis.imshow(spatial, cmap="Greys", origin="lower", aspect="equal")
    axis.set_title("Final time-aggregated sampling allocation")
    axis.set_xlabel("x patch")
    axis.set_ylabel("y patch")
    fig.colorbar(image, ax=axis, label="Probability mass")
    _save_both(fig, figures / "vara_patch_allocation")


def _metric_bar(raw: pd.DataFrame, metric: str, path: Path) -> None:
    if metric not in raw or raw.empty:
        return
    records = []
    for method, values in raw.groupby("method")[metric]:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            continue
        records.append(
            (
                method,
                float(numeric.mean()),
                float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
            )
        )
    if not records:
        return
    fig, axis = plt.subplots(figsize=(6.4, 4.1), constrained_layout=True)
    labels = [record[0] for record in records]
    means = [record[1] for record in records]
    errors = [record[2] for record in records]
    axis.bar(
        labels,
        means,
        yerr=errors,
        capsize=4,
        color=[PALETTE[index % len(PALETTE)] for index in range(len(records))],
    )
    axis.set_ylabel(metric.replace("cahn_hilliard_", "").replace("_", " "))
    axis.set_title(metric.replace("cahn_hilliard_", "").replace("_", " ").title())
    axis.margins(y=0.15)
    axis.grid(axis="y", alpha=0.18)
    _save_both(fig, path)


def _save_both(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
