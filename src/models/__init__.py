"""Neural network models."""

from .mlp import MLP, build_mlp_from_config
from .residual_fourier_mlp import ResidualFourierMLP, parameter_matched_width
from .physics_wrappers import (
    CavityHardBoundaryWrapper,
    HardBoundaryStreamfunctionPressureWrapper,
    StreamfunctionPressureWrapper,
)

__all__ = [
    "MLP",
    "ResidualFourierMLP",
    "CavityHardBoundaryWrapper",
    "HardBoundaryStreamfunctionPressureWrapper",
    "StreamfunctionPressureWrapper",
    "parameter_matched_width",
    "build_mlp_from_config",
]

