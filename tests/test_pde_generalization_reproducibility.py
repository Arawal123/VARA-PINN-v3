"""Paired initialization and sparse-sample reproducibility checks."""

from __future__ import annotations

from tests.test_pde_generalization_smoke import _tiny_config
from src.pde_generalization.trainer import PDEGeneralizationTrainer


def test_vanilla_and_vara_share_initial_model_and_sparse_samples(tmp_path) -> None:
    config = _tiny_config("burgers2d")
    vanilla = PDEGeneralizationTrainer(config, "vanilla", tmp_path / "vanilla")
    vara = PDEGeneralizationTrainer(config, "vara_v2", tmp_path / "vara")
    assert vanilla.initial_model_parameter_hash == vara.initial_model_parameter_hash
    assert vanilla.sparse_sample_hash == vara.sparse_sample_hash
