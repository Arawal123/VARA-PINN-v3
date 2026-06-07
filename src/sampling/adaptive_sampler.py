"""Mixed variable-region adaptive sampling."""

from __future__ import annotations

import numpy as np
import torch

from .region_sampler import RegionSampler
from .residual_sampler import sample_from_score_grid
from .uniform_sampler import UniformSampler


class MixedAdaptiveSampler:
    """Budgeted mixture sampler with minimum global coverage."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        patch_grid: object,
        device: torch.device,
        seed: int | None = None,
        mixture: dict[str, float] | None = None,
        uniform_engine: str = "random",
    ) -> None:
        self.bounds = bounds
        self.patch_grid = patch_grid
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.uniform = UniformSampler(bounds, device, seed, engine=uniform_engine)
        self.region_sampler = RegionSampler(patch_grid, device, seed)
        self.mixture = mixture or {
            "uniform": 0.50,
            "pde_residual": 0.25,
            "pressure_vorticity": 0.10,
            "velocity": 0.10,
            "boundary": 0.05,
        }

    def sample_interior(
        self,
        n: int,
        diagnostic_maps: dict[str, np.ndarray] | None = None,
        grid_coords: np.ndarray | None = None,
        weak_regions: list[object] | None = None,
        sampling_priorities: dict[int, float] | None = None,
    ) -> torch.Tensor:
        if n <= 0:
            return torch.zeros((0, 2), dtype=torch.float32, device=self.device)
        counts = self._counts(n)
        pieces: list[np.ndarray] = [self.uniform.sample_numpy(counts.get("uniform", 0))]

        if diagnostic_maps is not None and grid_coords is not None:
            score = diagnostic_maps.get("pde_residual", diagnostic_maps.get("aggregate_pde_residual"))
            if score is not None:
                pieces.append(sample_from_score_grid(grid_coords, score, counts.get("pde_residual", 0), self.rng))
        else:
            pieces.append(self.uniform.sample_numpy(counts.get("pde_residual", 0)))

        weak_regions = weak_regions or []
        pv_ids = [wr.patch_id for wr in weak_regions if "p" in wr.variable or "omega" in wr.variable or "vorticity" in wr.variable]
        vel_ids = [wr.patch_id for wr in weak_regions if "u_error" in wr.variable or "v_error" in wr.variable or "speed" in wr.variable]
        priority_ids = sorted((sampling_priorities or {}).keys())
        pieces.append(self._sample_regions(pv_ids or priority_ids, counts.get("pressure_vorticity", 0), sampling_priorities))
        pieces.append(self._sample_regions(vel_ids or priority_ids, counts.get("velocity", 0), sampling_priorities))
        pieces.append(self._sample_regions(priority_ids, counts.get("boundary", 0), sampling_priorities))

        xy = np.vstack([p for p in pieces if p.size]) if pieces else self.uniform.sample_numpy(n)
        if xy.shape[0] < n:
            xy = np.vstack([xy, self.uniform.sample_numpy(n - xy.shape[0])])
        elif xy.shape[0] > n:
            xy = xy[:n]
        self.rng.shuffle(xy)
        return torch.tensor(xy, dtype=torch.float32, device=self.device)

    def _sample_regions(self, patch_ids: list[int], n: int, priorities: dict[int, float] | None) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 2), dtype=float)
        if not patch_ids:
            return self.uniform.sample_numpy(n)
        unique = sorted(set(int(pid) for pid in patch_ids))
        weights = np.array([max((priorities or {}).get(pid, 1.0), 1e-6) for pid in unique], dtype=float)
        return self.region_sampler.sample_numpy(unique, n, weights)

    def _counts(self, n: int) -> dict[str, int]:
        raw = {k: int(np.floor(v * n)) for k, v in self.mixture.items()}
        missing = n - sum(raw.values())
        keys = list(self.mixture)
        for i in range(missing):
            raw[keys[i % len(keys)]] += 1
        return raw
