"""Corrected analytical diagnostics for the isolated Taylor--Green runner."""

from __future__ import annotations

import numpy as np
import torch

from src.physics.taylor_green import TaylorGreenVortex


class RepairedTaylorGreenVortex(TaylorGreenVortex):
    """Taylor--Green reference with vorticity consistent with ``dv/dx-du/dy``."""

    def exact_torch(self, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        values = super().exact_torch(coords)
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        decay = torch.exp(-2.0 * self.nu * t)
        values["omega"] = (
            2.0
            * float(self.amplitude)
            * torch.cos(x)
            * torch.cos(y)
            * decay
        )
        return values

    def vorticity_reference_sanity(
        self,
        resolution: int = 48,
        time: float | None = None,
    ) -> float:
        """Compare closed-form vorticity with autograd derivatives of exact u,v."""
        resolution = max(8, int(resolution))
        value = self.evaluation_time if time is None else float(time)
        x_grid, y_grid = np.meshgrid(
            np.linspace(self.x_min, self.x_max, resolution),
            np.linspace(self.y_min, self.y_max, resolution),
        )
        coords = torch.tensor(
            np.column_stack(
                [
                    x_grid.reshape(-1),
                    y_grid.reshape(-1),
                    np.full(x_grid.size, value),
                ]
            ),
            dtype=torch.float64,
            requires_grad=True,
        )
        exact = self.exact_torch(coords)
        grad_u = torch.autograd.grad(
            exact["u"],
            coords,
            grad_outputs=torch.ones_like(exact["u"]),
            create_graph=False,
            retain_graph=True,
        )[0]
        grad_v = torch.autograd.grad(
            exact["v"],
            coords,
            grad_outputs=torch.ones_like(exact["v"]),
            create_graph=False,
        )[0]
        differentiated = grad_v[:, 0:1] - grad_u[:, 1:2]
        closed_form = exact["omega"]
        relative = torch.linalg.vector_norm(differentiated - closed_form) / torch.linalg.vector_norm(
            closed_form
        ).clamp_min(1e-14)
        return float(relative.detach().cpu())
