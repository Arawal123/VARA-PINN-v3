"""Taylor--Green-only model constraints.

This module is intentionally not imported by the shared model factory.  The
dedicated Taylor--Green runner opts into it explicitly so existing benchmark
and checkpoint wiring remains byte-for-byte unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TaylorGreenHardInitialCondition(nn.Module):
    """Impose the analytical Taylor--Green state exactly at ``t=t_min``.

    The wrapped network learns the time-scaled correction away from the known
    initial state.  This removes the spurious low-residual branch that appears
    when the transient Navier--Stokes equations are trained without an initial
    condition.
    """

    physics_formulation = "taylor_green_hard_initial"

    def __init__(self, base_model: nn.Module, benchmark: object) -> None:
        super().__init__()
        self.base_model = base_model
        self.benchmark = benchmark

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                "TaylorGreenHardInitialCondition expects coordinates shaped (N, 3)."
            )
        raw = self.base_model(coords)
        t_min = float(self.benchmark.t_min)
        duration = max(float(self.benchmark.t_max) - t_min, 1e-12)
        tau = (coords[:, 2:3] - t_min) / duration
        initial_coords = torch.cat(
            [
                coords[:, 0:2],
                torch.full_like(coords[:, 2:3], t_min),
            ],
            dim=1,
        )
        initial = self.benchmark.exact_torch(initial_coords)
        initial_state = torch.cat(
            [initial["u"], initial["v"], initial["p"]],
            dim=1,
        )
        return initial_state + tau * raw
