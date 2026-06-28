"""Isolated manufactured-PDE generalization suite for VARA Controller V2."""

from .benchmarks import (
    AdvectionDiffusionBenchmark,
    AllenCahnBenchmark,
    Burgers2DBenchmark,
    ManufacturedBenchmark,
    build_benchmark,
)
from .models import build_pde_model, model_parameter_hash
from .residuals import compute_residuals

__all__ = [
    "AdvectionDiffusionBenchmark",
    "AllenCahnBenchmark",
    "Burgers2DBenchmark",
    "ManufacturedBenchmark",
    "build_benchmark",
    "build_pde_model",
    "compute_residuals",
    "model_parameter_hash",
]
