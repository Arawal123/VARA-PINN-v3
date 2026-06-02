"""LBFGS helpers for guarded optimizer stages."""

from __future__ import annotations

from collections.abc import Callable

import torch


def make_lbfgs_closure(
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[], torch.Tensor],
) -> Callable[[], torch.Tensor]:
    """Create a PyTorch LBFGS closure from a zero-argument loss function."""

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        return loss

    return closure
