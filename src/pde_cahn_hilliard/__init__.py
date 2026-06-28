"""Isolated split-form Cahn--Hilliard benchmark suite."""

from .benchmark import CahnHilliardBenchmark
from .models import build_cahn_hilliard_model, model_parameter_hash
from .residuals import compute_cahn_hilliard_residuals

__all__ = [
    "CahnHilliardBenchmark",
    "build_cahn_hilliard_model",
    "compute_cahn_hilliard_residuals",
    "model_parameter_hash",
]
