"""Boundary condition sampling."""

from __future__ import annotations

import numpy as np
import torch


class BoundarySampler:
    """Uniform sampler over the rectangular boundary."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        device: torch.device,
        seed: int | None = None,
        t_bounds: tuple[float, float] | None = None,
    ) -> None:
        self.bounds = bounds
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.t_bounds = t_bounds

    def sample_numpy(self, n: int) -> np.ndarray:
        x0, x1, y0, y1 = self.bounds
        counts = [n // 4] * 4
        for i in range(n - sum(counts)):
            counts[i] += 1
        y_left = self.rng.uniform(y0, y1, counts[0])
        y_right = self.rng.uniform(y0, y1, counts[1])
        x_bottom = self.rng.uniform(x0, x1, counts[2])
        x_top = self.rng.uniform(x0, x1, counts[3])
        pts = [
            np.column_stack([np.full(counts[0], x0), y_left]),
            np.column_stack([np.full(counts[1], x1), y_right]),
            np.column_stack([x_bottom, np.full(counts[2], y0)]),
            np.column_stack([x_top, np.full(counts[3], y1)]),
        ]
        out = np.vstack([p for p in pts if len(p)])
        self.rng.shuffle(out)
        if self.t_bounds is not None:
            times = self.rng.uniform(self.t_bounds[0], self.t_bounds[1], out.shape[0])
            out = np.column_stack([out, times])
        return out

    def sample(self, n: int) -> torch.Tensor:
        return torch.tensor(self.sample_numpy(n), dtype=torch.float32, device=self.device)

    def sample_lid_cavity_numpy(
        self,
        n: int,
        lid_fraction: float = 0.45,
        corner_fraction: float = 0.25,
        corner_width: float = 0.12,
    ) -> np.ndarray:
        """Sample cavity walls with extra mass on the moving lid and lid-corner discontinuities."""
        if n <= 0:
            return np.zeros((0, 2), dtype=float)
        lid_fraction = float(np.clip(lid_fraction, 0.0, 1.0))
        corner_fraction = float(np.clip(corner_fraction, 0.0, 1.0))
        if lid_fraction + corner_fraction > 0.95:
            scale = 0.95 / (lid_fraction + corner_fraction)
            lid_fraction *= scale
            corner_fraction *= scale

        n_lid = int(round(n * lid_fraction))
        n_corner = int(round(n * corner_fraction))
        n_uniform = max(0, n - n_lid - n_corner)
        pieces = [self.sample_numpy(n_uniform), self._sample_lid(n_lid), self._sample_lid_corners(n_corner, corner_width)]
        out = np.vstack([p for p in pieces if p.size])
        if out.shape[0] < n:
            out = np.vstack([out, self.sample_numpy(n - out.shape[0])])
        elif out.shape[0] > n:
            out = out[:n]
        self.rng.shuffle(out)
        return out

    def sample_lid_cavity(
        self,
        n: int,
        lid_fraction: float = 0.45,
        corner_fraction: float = 0.25,
        corner_width: float = 0.12,
    ) -> torch.Tensor:
        return torch.tensor(
            self.sample_lid_cavity_numpy(n, lid_fraction, corner_fraction, corner_width),
            dtype=torch.float32,
            device=self.device,
        )

    def sample_patch_numpy(self, patch_grid: object, patch_ids: list[int], n: int) -> np.ndarray:
        """Sample rectangular boundary points restricted to selected patch spans."""
        if n <= 0 or not patch_ids:
            return np.zeros((0, 2), dtype=float)
        x0, x1, y0, y1 = self.bounds
        pts = []
        for _ in range(n):
            patch = patch_grid.get_patch(int(self.rng.choice(patch_ids)))
            px0, px1, py0, py1, _, _ = patch.bounds
            candidates = []
            if np.isclose(px0, x0):
                candidates.append(("left", px0, max(py0, y0), min(py1, y1)))
            if np.isclose(px1, x1):
                candidates.append(("right", px1, max(py0, y0), min(py1, y1)))
            if np.isclose(py0, y0):
                candidates.append(("bottom", py0, max(px0, x0), min(px1, x1)))
            if np.isclose(py1, y1):
                candidates.append(("top", py1, max(px0, x0), min(px1, x1)))
            if not candidates:
                side = self.rng.choice(["left", "right", "bottom", "top"])
                if side == "left":
                    candidates.append(("left", x0, max(py0, y0), min(py1, y1)))
                elif side == "right":
                    candidates.append(("right", x1, max(py0, y0), min(py1, y1)))
                elif side == "bottom":
                    candidates.append(("bottom", y0, max(px0, x0), min(px1, x1)))
                else:
                    candidates.append(("top", y1, max(px0, x0), min(px1, x1)))
            kind, fixed, a, b = candidates[int(self.rng.integers(0, len(candidates)))]
            if kind in {"left", "right"}:
                pts.append([fixed, self.rng.uniform(a, b)])
            else:
                pts.append([self.rng.uniform(a, b), fixed])
        return np.asarray(pts, dtype=float)

    def sample_patch(self, patch_grid: object, patch_ids: list[int], n: int) -> torch.Tensor:
        """Torch wrapper for patch-restricted boundary sampling."""
        return torch.tensor(self.sample_patch_numpy(patch_grid, patch_ids, n), dtype=torch.float32, device=self.device)

    def _sample_lid(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 2), dtype=float)
        x0, x1, _y0, y1 = self.bounds
        x = self.rng.uniform(x0, x1, n)
        return np.column_stack([x, np.full(n, y1)])

    def _sample_lid_corners(self, n: int, corner_width: float) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 2), dtype=float)
        x0, x1, y0, y1 = self.bounds
        span = min(max(x1 - x0, 1e-12), max(y1 - y0, 1e-12))
        width = max(float(corner_width) * span, 1e-9)
        width_x = min(width, x1 - x0)
        width_y = min(width, y1 - y0)
        pts = np.zeros((n, 2), dtype=float)
        choices = self.rng.integers(0, 4, n)
        for i, choice in enumerate(choices):
            if choice == 0:
                pts[i] = [self.rng.uniform(x0, x0 + width_x), y1]
            elif choice == 1:
                pts[i] = [self.rng.uniform(x1 - width_x, x1), y1]
            elif choice == 2:
                pts[i] = [x0, self.rng.uniform(y1 - width_y, y1)]
            else:
                pts[i] = [x1, self.rng.uniform(y1 - width_y, y1)]
        return pts
