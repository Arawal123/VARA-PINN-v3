"""Checkpoint helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    metrics: dict[str, float],
    epoch: int,
    cycle: int,
) -> None:
    """Save a full training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime_state = _model_runtime_state(model)
    checkpoint_config = deepcopy(config)
    if runtime_state:
        checkpoint_config.setdefault("model", {}).update(
            {
                "hard_boundary_corner_width": runtime_state.get(
                    "corner_width",
                    checkpoint_config.get("model", {}).get(
                        "hard_boundary_corner_width"
                    ),
                ),
                "hard_boundary_lid_vertical_power": runtime_state.get(
                    "lid_vertical_power",
                    checkpoint_config.get("model", {}).get(
                        "hard_boundary_lid_vertical_power"
                    ),
                ),
                "hard_boundary_correction_scale": runtime_state.get(
                    "correction_scale",
                    checkpoint_config.get("model", {}).get(
                        "hard_boundary_correction_scale"
                    ),
                ),
            }
        )
        if "corner_width" in runtime_state:
            checkpoint_config.setdefault("benchmark_params", {})[
                "lid_corner_regularization_width"
            ] = runtime_state["corner_width"]
    payload = {
        "model_state": model.state_dict(),
        "model_runtime_state": runtime_state,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": checkpoint_config,
        "metrics": metrics,
        "epoch": epoch,
        "cycle": cycle,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    """Load a checkpoint into model and optionally optimizer."""
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    _restore_model_runtime_state(model, payload.get("model_runtime_state", {}))
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _model_runtime_state(model: torch.nn.Module) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in ("corner_width", "lid_vertical_power", "correction_scale"):
        if hasattr(model, name):
            state[name] = getattr(model, name)
    return state


def _restore_model_runtime_state(
    model: torch.nn.Module,
    state: dict[str, Any],
) -> None:
    for name, value in dict(state).items():
        if hasattr(model, name):
            setattr(model, name, value)
