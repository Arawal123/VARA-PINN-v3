"""Controller diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def save_patch_score_map(scores: np.ndarray, names: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(names))), constrained_layout=True)
    im = ax.imshow(scores, aspect="auto", cmap="inferno")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Patch id")
    ax.set_title("Variable-region severity S[j, r]")
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_intervention_timeline(action_records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for record in action_records:
        for action in record.get("actions", []):
            counts[action.get("action", "unknown")] = counts.get(action.get("action", "unknown"), 0) + 1
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    if counts:
        ax.bar(range(len(counts)), list(counts.values()), color="#2a6fbb")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(list(counts.keys()), rotation=35, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Intervention distribution")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_patch_grid_overlay(patch_grid: object, path: str | Path) -> None:
    """Plot fixed patch grid over the domain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x0, x1, y0, y1 = patch_grid.bounds
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    for patch in patch_grid.patches:
        px0, px1, py0, py1, _, _ = patch.bounds
        ax.plot([px0, px1, px1, px0, px0], [py0, py0, py1, py1, py0], color="black", lw=0.8)
        ax.text((px0 + px1) / 2, (py0 + py1) / 2, str(patch.patch_id), ha="center", va="center", fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Patch grid")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_patch_scalar_heatmap(values: np.ndarray, patch_grid: object, path: str | Path, title: str, cmap: str = "magma") -> None:
    """Plot one scalar value per patch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype=float).reshape(patch_grid.nt_patches, patch_grid.ny_patches, patch_grid.nx_patches)[0]
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(arr, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(patch_grid.nx_patches))
    ax.set_yticks(np.arange(patch_grid.ny_patches))
    ax.set_xlabel("patch x-index")
    ax.set_ylabel("patch y-index")
    ax.set_title(title)
    for iy in range(patch_grid.ny_patches):
        for ix in range(patch_grid.nx_patches):
            pid = iy * patch_grid.nx_patches + ix
            ax.text(ix, iy, str(pid), ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_active_intervention_map(interventions: list[dict], patch_grid: object, path: str | Path, title: str) -> None:
    """Visualize which patches received local interventions."""
    labels = np.full(patch_grid.num_patches, "", dtype=object)
    values = np.zeros(patch_grid.num_patches, dtype=float)
    for i, action in enumerate(interventions, start=1):
        pid = int(action["patch_id"])
        values[pid] = i
        label = _short_label(action.get("variable", ""))
        labels[pid] = f"{labels[pid]},{label}".strip(",")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = values.reshape(patch_grid.nt_patches, patch_grid.ny_patches, patch_grid.nx_patches)[0]
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(arr, origin="lower", cmap="tab20", aspect="auto", vmin=0)
    ax.set_title(title)
    ax.set_xlabel("patch x-index")
    ax.set_ylabel("patch y-index")
    for iy in range(patch_grid.ny_patches):
        for ix in range(patch_grid.nx_patches):
            pid = iy * patch_grid.nx_patches + ix
            text = labels[pid] if labels[pid] else str(pid)
            ax.text(ix, iy, text, ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_local_weight_evolution(records: list[dict], path: str | Path) -> None:
    """Plot total local weight mass over cycles by variable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    series: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        cycle = int(record.get("cycle", 0))
        weights = record.get("local_weights", {})
        for variable, patch_weights in weights.items():
            series.setdefault(variable, []).append((cycle, sum(float(v) for v in patch_weights.values())))
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for variable, points in sorted(series.items()):
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", label=variable)
    ax.set_xlabel("cycle")
    ax.set_ylabel("local weight mass")
    ax.set_title("Local weight evolution")
    if series:
        ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_accept_reject_counts(decisions: list[dict], path: str | Path) -> None:
    """Plot accepted versus rejected local interventions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    accepted = sum(1 for row in decisions if row.get("accepted"))
    rejected = sum(1 for row in decisions if row.get("rejected"))
    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    ax.bar(["accepted", "rejected"], [accepted, rejected], color=["#26734d", "#b43c3c"])
    ax.set_ylabel("count")
    ax.set_title("Local intervention decisions")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_targeted_patch_improvement(decisions: list[dict], path: str | Path) -> None:
    """Plot targeted local improvement for every candidate action."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [float(row.get("target_local_improvement", 0.0)) for row in decisions]
    colors = ["#26734d" if row.get("accepted") else "#b43c3c" for row in decisions]
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    if values:
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("candidate intervention")
    ax.set_ylabel("targeted local improvement")
    ax.set_title("Per-action targeted patch improvement")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_collateral_damage_timeline(decisions: list[dict], path: str | Path) -> None:
    """Plot max non-target collateral damage over candidate actions."""
    _save_metric_timeline(
        decisions,
        path,
        metric="max_collateral_damage",
        ylabel="max collateral damage",
        title="Collateral damage timeline",
        color="#8a4fbf",
    )


def save_pressure_collateral_timeline(decisions: list[dict], path: str | Path) -> None:
    """Plot pressure-specific collateral damage over candidate actions."""
    _save_metric_timeline(
        decisions,
        path,
        metric="pressure_collateral_damage",
        ylabel="pressure collateral damage",
        title="Pressure collateral timeline",
        color="#c15b30",
    )


def _save_metric_timeline(
    decisions: list[dict],
    path: str | Path,
    metric: str,
    ylabel: str,
    title: str,
    color: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [float(row.get(metric, 0.0)) for row in decisions]
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    if values:
        ax.plot(np.arange(len(values)), values, marker="o", color=color)
        for idx, row in enumerate(decisions):
            if row.get("rejected"):
                ax.scatter([idx], [values[idx]], color="#b43c3c", s=32, zorder=3)
    ax.set_xlabel("candidate intervention")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _short_label(variable: str) -> str:
    if "p_error" in variable or "pressure" in variable:
        return "p"
    if "omega" in variable:
        return "om"
    if "u_error" in variable:
        return "u"
    if "v_error" in variable:
        return "v"
    if "continuity" in variable:
        return "div"
    if "momentum" in variable:
        return "mom"
    if "boundary" in variable:
        return "bc"
    return "res"
