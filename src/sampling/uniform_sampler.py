"""Uniform domain sampling."""

from __future__ import annotations

import numpy as np
import torch


class UniformSampler:
    """Uniform interior sampler."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        device: torch.device,
        seed: int | None = None,
        t_bounds: tuple[float, float] | None = None,
        engine: str = "random",
    ) -> None:
        self.bounds = bounds
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.t_bounds = t_bounds
        self.engine = str(engine).lower()
        if self.engine not in {"random", "sobol"}:
            raise ValueError("UniformSampler engine must be 'random' or 'sobol'.")
        dimension = 3 if t_bounds is not None else 2
        self.sobol = (
            torch.quasirandom.SobolEngine(
                dimension,
                scramble=True,
                seed=0 if seed is None else int(seed),
            )
            if self.engine == "sobol"
            else None
        )

    def sample_numpy(self, n: int) -> np.ndarray:
        x0, x1, y0, y1 = self.bounds
        if n <= 0:
            dimension = 3 if self.t_bounds is not None else 2
            return np.zeros((0, dimension), dtype=float)
        if self.sobol is not None:
            unit = self.sobol.draw(int(n)).cpu().numpy()
            lower = [x0, y0]
            upper = [x1, y1]
            if self.t_bounds is not None:
                lower.append(self.t_bounds[0])
                upper.append(self.t_bounds[1])
            lower_np = np.asarray(lower, dtype=float)
            upper_np = np.asarray(upper, dtype=float)
            return lower_np + unit * (upper_np - lower_np)
        columns = [self.rng.uniform(x0, x1, n), self.rng.uniform(y0, y1, n)]
        if self.t_bounds is not None:
            columns.append(self.rng.uniform(self.t_bounds[0], self.t_bounds[1], n))
        return np.column_stack(columns)

    def sample(self, n: int) -> torch.Tensor:
        return torch.tensor(self.sample_numpy(n), dtype=torch.float32, device=self.device)

    def snapshot(self) -> dict[str, object]:
        return {
            "rng": self.rng.bit_generator.state,
            "sobol_num_generated": (
                int(self.sobol.num_generated) if self.sobol is not None else 0
            ),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self.rng.bit_generator.state = snapshot["rng"]
        if self.sobol is not None:
            self.sobol.reset()
            generated = int(snapshot.get("sobol_num_generated", 0))
            if generated > 0:
                self.sobol.fast_forward(generated)
