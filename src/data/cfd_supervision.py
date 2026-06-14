"""Deterministic sparse CFD supervision for lid-driven cavity studies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.physics.cavity_reference import load_full_field_reference


@dataclass(frozen=True)
class CFDSupervisionPool:
    mode: str
    coords: torch.Tensor
    targets: dict[str, torch.Tensor]
    source_path: str
    seed: int
    sample_fraction: float
    pool_hash: str
    oracle: bool
    sampling_mode: str

    @property
    def sample_count(self) -> int:
        return int(self.coords.shape[0])


def build_cavity_cfd_supervision(
    config: dict[str, Any],
    bounds: tuple[float, float, float, float],
    device: torch.device,
) -> CFDSupervisionPool | None:
    cfg = dict(config.get("data_supervision", {}))
    mode = str(cfg.get("mode", "pure_pinn")).lower()
    if mode == "pure_pinn":
        return None
    if mode not in {
        "sparse_cfd",
        "sparse_cfd_polish",
        "full_cfd_oracle",
    }:
        raise ValueError(
            "data_supervision.mode must be pure_pinn, sparse_cfd, "
            "sparse_cfd_polish, or full_cfd_oracle."
        )
    path = cfg.get("reference_path") or config.get("benchmark_params", {}).get(
        "full_field_reference_path"
    )
    if not path:
        raise ValueError(f"{mode} requires a full-field CFD reference path.")
    reference = load_full_field_reference(Path(path))
    coords = np.column_stack([reference["x"], reference["y"]])
    eligible = _eligible_mask(coords, bounds, cfg)
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        raise ValueError("No eligible interior CFD points remain after filtering.")
    seed = int(cfg.get("seed", config.get("seed", 0)))
    sampling_cfg = dict(cfg.get("cfd", {}).get("sampling", {}))
    sampling_mode = str(sampling_cfg.get("mode", "uniform")).lower()
    if sampling_mode not in {"uniform", "mixed"}:
        raise ValueError("CFD sampling mode must be uniform or mixed.")
    if mode == "full_cfd_oracle":
        selected = eligible_indices
    else:
        requested_count = cfg.get("sample_count")
        if requested_count is None:
            fraction = min(max(float(cfg.get("sample_fraction", 0.01)), 0.0), 1.0)
            count = max(1, int(round(eligible_indices.size * fraction)))
        else:
            count = max(1, int(requested_count))
        count = min(count, eligible_indices.size)
        rng = np.random.default_rng(seed)
        if sampling_mode == "mixed":
            selected = _mixed_sample_indices(
                coords,
                eligible_indices,
                count,
                bounds,
                sampling_cfg,
                rng,
            )
        else:
            selected = np.sort(
                rng.choice(eligible_indices, size=count, replace=False)
            )
    selected_coords = np.asarray(coords[selected], dtype=np.float32)
    targets: dict[str, torch.Tensor] = {
        "u": torch.tensor(
            np.asarray(reference["u"][selected], dtype=np.float32).reshape(-1, 1),
            device=device,
        ),
        "v": torch.tensor(
            np.asarray(reference["v"][selected], dtype=np.float32).reshape(-1, 1),
            device=device,
        ),
    }
    if bool(cfg.get("include_pressure", False)):
        if not bool(reference.get("has_p_reference", False)):
            raise ValueError("CFD pressure supervision requested but unavailable.")
        targets["p"] = torch.tensor(
            np.asarray(reference["p"][selected], dtype=np.float32).reshape(-1, 1),
            device=device,
        )
    if bool(cfg.get("include_vorticity", False)):
        if not bool(reference.get("has_omega_reference", False)):
            raise ValueError("CFD vorticity supervision requested but unavailable.")
        targets["omega"] = torch.tensor(
            np.asarray(reference["omega"][selected], dtype=np.float32).reshape(-1, 1),
            device=device,
        )
    digest = hashlib.sha256()
    digest.update(selected.astype(np.int64).tobytes())
    digest.update(str(Path(path).resolve()).encode("utf-8"))
    return CFDSupervisionPool(
        mode=mode,
        coords=torch.tensor(selected_coords, device=device),
        targets=targets,
        source_path=str(path),
        seed=seed,
        sample_fraction=float(selected.size / eligible_indices.size),
        pool_hash=digest.hexdigest(),
        oracle=mode == "full_cfd_oracle",
        sampling_mode=sampling_mode,
    )


def _eligible_mask(
    coords: np.ndarray,
    bounds: tuple[float, float, float, float],
    cfg: dict[str, Any],
) -> np.ndarray:
    x0, x1, y0, y1 = bounds
    width = max(float(x1 - x0), 1e-12)
    height = max(float(y1 - y0), 1e-12)
    xi = (coords[:, 0] - x0) / width
    eta = (coords[:, 1] - y0) / height
    margin = max(float(cfg.get("boundary_margin", 1e-6)), 0.0)
    mask = (
        (xi > margin)
        & (xi < 1.0 - margin)
        & (eta > margin)
        & (eta < 1.0 - margin)
    )
    if bool(cfg.get("exclude_near_corners", True)):
        corner = min(max(float(cfg.get("corner_margin", 0.05)), 0.0), 0.49)
        near_x_corner = (xi < corner) | (xi > 1.0 - corner)
        near_y_corner = (eta < corner) | (eta > 1.0 - corner)
        mask &= ~(near_x_corner & near_y_corner)
    return mask


def _mixed_sample_indices(
    coords: np.ndarray,
    eligible_indices: np.ndarray,
    count: int,
    bounds: tuple[float, float, float, float],
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    x0, x1, y0, y1 = bounds
    xi = (coords[:, 0] - x0) / max(float(x1 - x0), 1e-12)
    eta = (coords[:, 1] - y0) / max(float(y1 - y0), 1e-12)
    fractions = {
        "top_band": float(cfg.get("top_band_fraction", 0.25)),
        "right_band": float(cfg.get("right_band_fraction", 0.12)),
        "lower_core": float(cfg.get("lower_core_fraction", 0.10)),
        "left_band": float(cfg.get("left_band_fraction", 0.08)),
    }
    masks = {
        "top_band": (eta >= 0.70) & (eta <= 0.98),
        "right_band": (
            (xi >= 0.78)
            & (xi <= 0.98)
            & (eta >= 0.15)
            & (eta <= 0.85)
        ),
        "lower_core": (
            (xi >= 0.15)
            & (xi <= 0.85)
            & (eta >= 0.05)
            & (eta <= 0.35)
        ),
        "left_band": (
            (xi >= 0.02)
            & (xi <= 0.22)
            & (eta >= 0.20)
            & (eta <= 0.85)
        ),
    }
    available = set(int(index) for index in eligible_indices)
    selected: list[int] = []
    for name in ("top_band", "right_band", "lower_core", "left_band"):
        requested = int(round(count * max(fractions[name], 0.0)))
        candidates = np.asarray(
            sorted(index for index in available if masks[name][index]),
            dtype=int,
        )
        take = min(requested, candidates.size)
        if take > 0:
            chosen = rng.choice(candidates, size=take, replace=False)
            selected.extend(int(index) for index in chosen)
            available.difference_update(int(index) for index in chosen)
    remaining = count - len(selected)
    if remaining > 0:
        candidates = np.asarray(sorted(available), dtype=int)
        chosen = rng.choice(candidates, size=remaining, replace=False)
        selected.extend(int(index) for index in chosen)
    return np.sort(np.asarray(selected, dtype=int))
