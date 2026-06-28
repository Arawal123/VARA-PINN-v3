"""Paired Cahn--Hilliard initialization and sparse-data determinism."""

from __future__ import annotations

from src.pde_cahn_hilliard.trainer import CahnHilliardTrainer
from test_cahn_hilliard_smoke import tiny_cahn_hilliard_config


def test_vanilla_and_vara_pair_hashes_match(tmp_path) -> None:
    config = tiny_cahn_hilliard_config()
    vanilla = CahnHilliardTrainer(config, "vanilla", tmp_path / "vanilla")
    vara = CahnHilliardTrainer(config, "vara_v2", tmp_path / "vara")
    assert vanilla.initial_model_parameter_hash == vara.initial_model_parameter_hash
    assert vanilla.sparse_hash == vara.sparse_hash
