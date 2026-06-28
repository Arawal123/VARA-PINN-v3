"""Split-form residual and manufactured forcing tests."""

from __future__ import annotations

import torch
from torch import nn

from src.pde_cahn_hilliard.benchmark import CahnHilliardBenchmark
from src.pde_cahn_hilliard.residuals import compute_cahn_hilliard_residuals


class ExactCahnHilliard(nn.Module):
    def __init__(self, benchmark: CahnHilliardBenchmark) -> None:
        super().__init__()
        self.benchmark = benchmark

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.benchmark.exact(coordinates)


def test_split_residual_shapes_and_exact_solution() -> None:
    torch.manual_seed(11)
    benchmark = CahnHilliardBenchmark(
        {
            "bounds": [0.0, 1.0, 0.0, 1.0],
            "t_bounds": [0.0, 1.0],
            "epsilon": 0.04,
            "mobility": 1.0,
            "delta": 1e-6,
        }
    )
    coordinates = (0.08 + 0.84 * torch.rand(10, 3, dtype=torch.float64)).requires_grad_(True)
    forcing = benchmark.forcing(coordinates)
    residuals = compute_cahn_hilliard_residuals(
        ExactCahnHilliard(benchmark), coordinates, benchmark, forcing
    )
    assert set(residuals) == {"r_ch", "r_mu", "pde_residual"}
    for values in residuals.values():
        assert values.shape == (10, 1)
        assert torch.isfinite(values).all()
    assert float(residuals["r_ch"].detach().abs().mean()) < 1e-8
    assert float(residuals["r_mu"].detach().abs().mean()) < 1e-10


def test_reference_has_two_fields_and_interface_band() -> None:
    benchmark = CahnHilliardBenchmark({"epsilon": 0.04})
    coordinates = torch.rand(12, 3, dtype=torch.float64)
    reference = benchmark.exact(coordinates)
    assert reference.shape == (12, 2)
    assert benchmark.interface_band(reference[:, 0]).dtype == torch.bool
