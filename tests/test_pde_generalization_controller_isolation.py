"""Static isolation and controller-signal guard checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.controllers.v2_controller import VARAV2Controller
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_new_trainer_does_not_import_legacy_ns_training_path() -> None:
    source = (ROOT / "src" / "pde_generalization" / "trainer.py").read_text(encoding="utf-8")
    forbidden_imports = [
        "src.training.trainer",
        "src.losses.base_losses",
        "src.diagnostics.diagnostic_maps",
        "src.evaluation.metrics",
    ]
    assert all(name not in source for name in forbidden_imports)


@pytest.mark.parametrize("benchmark", ["burgers2d", "allen_cahn", "advection_diffusion"])
def test_configs_forbid_controller_reference_metrics(benchmark: str) -> None:
    config = load_config(ROOT / "configs" / "pde_generalization" / f"{benchmark}.yaml")
    assert config["evaluation"]["controller_reference_metrics_enabled"] is False
    VARAV2Controller.assert_reference_free(config["controller_v2"]["guard_metrics"])


def test_controller_rejects_full_field_reference_signal() -> None:
    with pytest.raises(ValueError, match="evaluation-only signals"):
        VARAV2Controller.assert_reference_free(["full_field_rel_l2_error"])
