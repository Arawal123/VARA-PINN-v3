"""Patch-focused sampling."""

from __future__ import annotations

import numpy as np
import torch


class RegionSampler:
    """Uniformly sample inside selected patches."""

    def __init__(self, patch_grid: object, device: torch.device, seed: int | None = None) -> None:
        self.patch_grid = patch_grid
        self.device = device
        self.rng = np.random.default_rng(seed)

    def sample_numpy(self, patch_ids: list[int], n: int, weights: np.ndarray | None = None) -> np.ndarray:
        if n <= 0 or not patch_ids:
            return np.zeros((0, 2), dtype=float)
        if weights is None:
            weights = np.ones(len(patch_ids), dtype=float)
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        chosen = self.rng.choice(patch_ids, size=n, replace=True, p=weights)
        pts = []
        for pid in chosen:
            patch = self.patch_grid.get_patch(int(pid))
            x0, x1, y0, y1, t0, t1 = patch.bounds
            point = [self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)]
            if t0 is not None and t1 is not None:
                point.append(self.rng.uniform(t0, t1))
            pts.append(point)
        return np.asarray(pts, dtype=float)

    def sample(self, patch_ids: list[int], n: int, weights: np.ndarray | None = None) -> torch.Tensor:
        return torch.tensor(self.sample_numpy(patch_ids, n, weights), dtype=torch.float32, device=self.device)
