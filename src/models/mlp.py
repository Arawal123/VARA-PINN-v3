"""Configurable tanh MLP for velocity-pressure PINNs."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected MLP with optional affine input normalization."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: Iterable[int] = (96, 96, 96, 96),
        activation: str = "tanh",
        input_lower: Iterable[float] | None = None,
        input_upper: Iterable[float] | None = None,
    ) -> None:
        super().__init__()
        sizes = [in_dim, *list(hidden_layers), out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )
        self.activation_name = activation
        self.activation = _activation(activation)
        if input_lower is None:
            input_lower = [-1.0] * in_dim
        if input_upper is None:
            input_upper = [1.0] * in_dim
        self.register_buffer("input_lower", torch.tensor(list(input_lower), dtype=torch.float32))
        self.register_buffer("input_upper", torch.tensor(list(input_upper), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        lower = self.input_lower.to(device=x.device, dtype=x.dtype)
        upper = self.input_upper.to(device=x.device, dtype=x.dtype)
        return 2.0 * (x - lower) / (upper - lower).clamp_min(1e-12) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.normalize_inputs(x)
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))
        return self.layers[-1](z)


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp_from_config(config: dict, bounds: tuple[float, float, float, float]) -> nn.Module:
    """Build the configured velocity-pressure network."""
    model_cfg = config.get("model", {})
    x0, x1, y0, y1 = bounds
    in_dim = int(model_cfg.get("input_dim", 2))
    lower = [x0, y0] if in_dim == 2 else [x0, y0, model_cfg.get("t_min", 0.0)]
    upper = [x1, y1] if in_dim == 2 else [x1, y1, model_cfg.get("t_max", 1.0)]
    architecture = str(model_cfg.get("architecture", "mlp")).lower()
    formulation = str(model_cfg.get("physics_formulation", "direct")).lower()
    requested_out_dim = int(model_cfg.get("output_dim", 3))
    streamfunction_formulations = {
        "streamfunction_pressure",
        "hard_boundary_streamfunction_pressure",
    }
    internal_out_dim = 2 if formulation in streamfunction_formulations else requested_out_dim
    if architecture == "residual_fourier_mlp":
        from .residual_fourier_mlp import ResidualFourierMLP, parameter_matched_width

        frequencies = tuple(model_cfg.get("frequencies", [1.0, 2.0, 4.0, 8.0]))
        blocks = int(model_cfg.get("residual_blocks", 4))
        hidden_layers = model_cfg.get("comparison_hidden_layers", model_cfg.get("hidden_layers", [96, 96, 96, 96]))
        matched_width, enhanced_count, legacy_count = parameter_matched_width(
            in_dim,
            internal_out_dim,
            hidden_layers,
            frequencies,
            blocks,
        )
        width = int(model_cfg.get("width", matched_width))
        if "width" not in model_cfg:
            mismatch = abs(enhanced_count - legacy_count) / max(legacy_count, 1)
            if mismatch > 0.05:
                raise ValueError(
                    "Could not parameter-match residual_fourier_mlp within 5%; "
                    f"legacy={legacy_count}, enhanced={enhanced_count}."
                )
        model: nn.Module = ResidualFourierMLP(
            in_dim=in_dim,
            out_dim=internal_out_dim,
            width=width,
            blocks=blocks,
            frequencies=frequencies,
            input_lower=lower,
            input_upper=upper,
        )
    else:
        model = MLP(
            in_dim=in_dim,
            out_dim=internal_out_dim,
            hidden_layers=model_cfg.get("hidden_layers", [96, 96, 96, 96]),
            activation=model_cfg.get("activation", "tanh"),
            input_lower=lower,
            input_upper=upper,
        )
    if formulation == "streamfunction_pressure":
        from .physics_wrappers import StreamfunctionPressureWrapper

        return StreamfunctionPressureWrapper(model)
    if formulation in {"cavity_hard_boundary", "hard_boundary_streamfunction_pressure"}:
        cavity_names = {
            "lid_driven_cavity",
            "lid-driven-cavity",
            "lid_cavity",
            "lid-cavity",
            "cavity",
        }
        if str(config.get("benchmark", "")).lower() not in cavity_names:
            raise ValueError(
                f"{formulation} is only valid for the lid-driven cavity benchmark."
            )
        from .physics_wrappers import (
            CavityHardBoundaryWrapper,
            HardBoundaryStreamfunctionPressureWrapper,
        )

        benchmark_cfg = config.get("benchmark_params", {})
        if formulation == "hard_boundary_streamfunction_pressure":
            return HardBoundaryStreamfunctionPressureWrapper(
                model,
                bounds,
                lid_velocity=float(benchmark_cfg.get("lid_velocity", 1.0)),
                corner_width=float(model_cfg.get("hard_boundary_corner_width", 0.02)),
                lid_vertical_power=int(
                    model_cfg.get("hard_boundary_lid_vertical_power", 6)
                ),
                correction_scale=float(
                    model_cfg.get("hard_boundary_correction_scale", 64.0)
                ),
            )
        return CavityHardBoundaryWrapper(
            model,
            bounds,
            lid_velocity=float(benchmark_cfg.get("lid_velocity", 1.0)),
            corner_width=float(model_cfg.get("hard_boundary_corner_width", 0.02)),
            lid_lifting=str(model_cfg.get("hard_boundary_lid_lifting", "linear")),
            lid_vertical_power=int(
                model_cfg.get("hard_boundary_lid_vertical_power", 6)
            ),
        )
    if formulation != "direct":
        raise ValueError(f"Unsupported model.physics_formulation: {formulation}")
    return model

