"""Static isolation and reference-signal safety checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.controllers.v2_controller import VARAV2Controller
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_trainer_has_no_legacy_or_prior_pde_trainer_imports() -> None:
    source = (ROOT / "src" / "pde_cahn_hilliard" / "trainer.py").read_text(encoding="utf-8")
    forbidden = [
        "src.training",
        "src.losses",
        "src.evaluation",
        "src.pde_generalization",
    ]
    assert all(name not in source for name in forbidden)


def test_config_disables_reference_metrics_for_controller() -> None:
    config = load_config(ROOT / "configs" / "cahn_hilliard" / "base.yaml")
    assert config["evaluation"]["controller_reference_metrics_enabled"] is False
    VARAV2Controller.assert_reference_free(config["controller_v2"]["guard_metrics"])


def test_controller_rejects_interface_reference_error() -> None:
    with pytest.raises(ValueError, match="evaluation-only signals"):
        VARAV2Controller.assert_reference_free(["interface_reference_rel_l2_error"])


def test_new_files_live_only_in_isolated_roots() -> None:
    assert (ROOT / "src" / "pde_cahn_hilliard").is_dir()
    assert (ROOT / "configs" / "cahn_hilliard").is_dir()
    assert (ROOT / "scripts" / "run_vara_v2_cahn_hilliard.py").is_file()
