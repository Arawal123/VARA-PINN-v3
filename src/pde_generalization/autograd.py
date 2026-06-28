"""Small autograd helpers for scalar PDE fields."""

from __future__ import annotations

import torch


def gradient(
    field: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """Differentiate a scalar column field with respect to all coordinates."""
    return torch.autograd.grad(
        field,
        coordinates,
        grad_outputs=torch.ones_like(field),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0]


def spatial_laplacian(field: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Return d2/dx2 + d2/dy2 for coordinates ordered as (x, y, t)."""
    first = gradient(field, coordinates)
    terms: list[torch.Tensor] = []
    for axis in (0, 1):
        second = gradient(first[:, axis : axis + 1], coordinates)
        terms.append(second[:, axis : axis + 1])
    return terms[0] + terms[1]
