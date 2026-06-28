"""Autograd helpers for split-form Cahn--Hilliard equations."""

from __future__ import annotations

import torch


def gradient(field: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Differentiate a scalar column with respect to (x, y, t)."""
    return torch.autograd.grad(
        field,
        coordinates,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


def spatial_laplacian(field: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Return the x-y Laplacian of a scalar column."""
    first = gradient(field, coordinates)
    xx = gradient(first[:, 0:1], coordinates)[:, 0:1]
    yy = gradient(first[:, 1:2], coordinates)[:, 1:2]
    return xx + yy
