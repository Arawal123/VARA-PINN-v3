"""Model construction local to the Cahn--Hilliard suite."""

from __future__ import annotations

import hashlib
from typing import Any

from torch import nn

from src.models.mlp import MLP


def build_cahn_hilliard_model(config: dict[str, Any]) -> nn.Module:
    """Build the two-output (u, mu) normalized MLP."""
    model_cfg = dict(config.get("model", {}))
    benchmark_cfg = dict(config.get("benchmark", {}))
    bounds = benchmark_cfg.get("bounds", [0.0, 1.0, 0.0, 1.0])
    t_bounds = benchmark_cfg.get("t_bounds", [0.0, 1.0])
    input_dim = int(model_cfg.get("input_dim", 3))
    output_dim = int(model_cfg.get("output_dim", 2))
    if input_dim != 3 or output_dim != 2:
        raise ValueError("Split-form Cahn--Hilliard requires input_dim=3 and output_dim=2.")
    return MLP(
        in_dim=3,
        out_dim=2,
        hidden_layers=model_cfg.get("hidden_layers", [96, 96, 96, 96, 96]),
        activation=str(model_cfg.get("activation", "tanh")),
        input_lower=[bounds[0], bounds[2], t_bounds[0]],
        input_upper=[bounds[1], bounds[3], t_bounds[1]],
    )


def model_parameter_hash(model: nn.Module) -> str:
    """Hash initial parameters for paired-run reproducibility evidence."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
