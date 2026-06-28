"""Residual and manufactured-forcing checks for the isolated PDE suite."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.pde_generalization.benchmarks import build_benchmark
from src.pde_generalization.residuals import compute_residuals


class ExactReference(nn.Module):
    def __init__(self, benchmark) -> None:
        super().__init__()
        self.benchmark = benchmark

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.benchmark.exact(coordinates)


@pytest.mark.parametrize(
    ("name", "params", "expected"),
    [
        ("burgers2d", {"nu": 0.01, "sigma": 0.11}, {"f_u", "f_v", "pde_residual"}),
        ("allen_cahn", {"eps": 0.04}, {"f_ac", "pde_residual"}),
        (
            "advection_diffusion",
            {"kappa": 0.01, "advection_velocity": [1.0, 0.5], "sigma": 0.09},
            {"f_advdiff", "pde_residual"},
        ),
    ],
)
def test_residual_shapes_and_exact_manufactured_solution(name, params, expected) -> None:
    torch.manual_seed(7)
    benchmark = build_benchmark(
        {
            "benchmark": name,
            "benchmark_params": {
                "bounds": [0.0, 1.0, 0.0, 1.0],
                "t_bounds": [0.0, 1.0],
                **params,
            },
        }
    )
    coordinates = (0.05 + 0.90 * torch.rand(24, 3, dtype=torch.float64)).requires_grad_(True)
    residuals = compute_residuals(ExactReference(benchmark), coordinates, benchmark)
    assert set(residuals) == expected
    for values in residuals.values():
        assert values.shape == (24, 1)
        assert torch.isfinite(values).all()
    signed = [values for key, values in residuals.items() if key != "pde_residual"]
    assert max(float(value.detach().abs().mean()) for value in signed) < 1e-9


def test_unknown_benchmark_fails_loudly() -> None:
    with pytest.raises(ValueError, match="Unsupported PDE generalization benchmark"):
        build_benchmark({"benchmark": "not_a_pde"})
