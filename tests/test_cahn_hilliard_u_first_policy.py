"""Focused tests for the Cahn--Hilliard u-first VARA policy."""

from __future__ import annotations

import torch
import numpy as np

from src.controllers.v2_controller import V2Candidate
from src.pde_cahn_hilliard.diagnostics import (
    compute_reference_free_pointwise_diagnostics,
)
from src.pde_cahn_hilliard.trainer import CahnHilliardTrainer
from src.utils.config import deep_update, load_config
from test_cahn_hilliard_smoke import tiny_cahn_hilliard_config


def test_diagnostic_priority_config_loads_u_first_defaults() -> None:
    config = load_config("configs/cahn_hilliard/base.yaml")
    priorities = config["diagnostics"]["priority_weights"]
    assert priorities["sparse_u_mismatch"] > priorities["chemical_potential_residual"]
    assert priorities["predicted_interface_proxy"] > priorities["sparse_mu_mismatch"]
    assert config["controller_v2"]["cahn_hilliard_u_first_policy"] is True


def test_prediction_only_interface_diagnostics_are_finite_and_shape_correct(tmp_path) -> None:
    trainer = CahnHilliardTrainer(
        tiny_cahn_hilliard_config(), "vara_v2", tmp_path / "run"
    )

    def forbidden_exact(_coordinates: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Full-field exact reference entered controller diagnostics")

    trainer.benchmark.exact = forbidden_exact  # type: ignore[method-assign]
    channels, _ = compute_reference_free_pointwise_diagnostics(
        trainer.model,
        trainer.benchmark,
        trainer.diagnostic_batch,
        variable_awareness=True,
        interface_tau=0.25,
        interface_beta=2.5,
        interface_threshold=0.8,
        mass_baseline=trainer.mass_proxy_baseline,
    )
    count = trainer.diagnostic_batch["interior"].shape[0]
    for name in (
        "predicted_interface_proxy",
        "predicted_interface_mask",
        "predicted_gradient_norm",
        "phase_range_violation",
        "mass_proxy_violation",
    ):
        values, coordinates = channels[name]
        assert values.shape == (count,)
        assert coordinates.shape == (count, 3)
        assert torch.isfinite(values).all()


def test_mu_support_only_retains_mu_residual_at_lower_priority(tmp_path) -> None:
    config = deep_update(
        tiny_cahn_hilliard_config(),
        {
            "losses": {"mu_support_only": True, "mu_priority_max": 0.5},
            "diagnostics": {
                "priority_weights": {
                    "ch_residual": 1.5,
                    "chemical_potential_residual": 0.5,
                }
            },
        },
    )
    trainer = CahnHilliardTrainer(config, "vara_v2", tmp_path / "run")
    snapshot = trainer._diagnose()
    assert "chemical_potential_residual" in snapshot.names
    ch_index = snapshot.names.index("ch_residual")
    mu_index = snapshot.names.index("chemical_potential_residual")
    assert torch.as_tensor(snapshot.priority_scores[mu_index]).max() <= (
        0.5 * torch.as_tensor(snapshot.normalized_scores[mu_index]).max() + 1e-8
    )
    assert snapshot.priority_scores[ch_index].max() > 0.0


def test_u_first_policy_ranks_primary_candidate_before_mu_candidate(tmp_path) -> None:
    trainer = CahnHilliardTrainer(
        tiny_cahn_hilliard_config(), "vara_v2", tmp_path / "run"
    )
    primary = V2Candidate(
        variable="sparse_u_mismatch",
        patch_id=1,
        action_type="local_loss",
        loss_names=["sparse_u_mse"],
        severity=1.0,
        persistence=1,
        trend=0.0,
        rank_score=0.1,
    )
    auxiliary = V2Candidate(
        variable="chemical_potential_residual",
        patch_id=0,
        action_type="local_loss",
        loss_names=["chemical_potential_residual"],
        severity=10.0,
        persistence=1,
        trend=0.0,
        rank_score=100.0,
    )
    ranked = trainer._rank_u_first_candidates([auxiliary, primary])
    assert ranked[0].variable == "sparse_u_mismatch"
    assert sum(trainer._is_mu_only_candidate(item) for item in ranked) <= 1


def test_mu_support_only_caps_local_multiplier_without_losing_mass(tmp_path) -> None:
    config = deep_update(
        tiny_cahn_hilliard_config(),
        {"losses": {"mu_support_only": True, "mu_local_multiplier_max": 1.25}},
    )
    trainer = CahnHilliardTrainer(config, "vara_v2", tmp_path / "run")
    assert trainer.controller is not None
    values = np.full(trainer.patch_grid.num_patches, 6.0 / 7.0)
    values[0] = 2.0
    trainer.controller.state.loss_multipliers[
        "chemical_potential_residual"
    ] = values
    trainer._enforce_mu_support_caps()
    capped = trainer.controller.state.loss_multipliers[
        "chemical_potential_residual"
    ]
    assert float(capped.max()) <= 1.25 + 1e-10
    assert np.isclose(float(capped.mean()), 1.0)


def test_mu_improvement_is_rejected_when_sparse_u_guard_is_damaged(tmp_path) -> None:
    trainer = CahnHilliardTrainer(
        tiny_cahn_hilliard_config(), "vara_v2", tmp_path / "run"
    )
    candidate = V2Candidate(
        variable="chemical_potential_residual",
        patch_id=0,
        action_type="local_loss",
        loss_names=["chemical_potential_residual"],
        severity=1.0,
        persistence=1,
        trend=0.0,
        predicted_target_improvement=0.5,
    )
    before = {
        "pde_residual_mean": 1.0,
        "boundary_condition_error": 1.0,
        "unweighted_validation_loss": 1.0,
        "unweighted_physics_validation_loss": 1.0,
        "sparse_u_mse": 1.0,
        "sparse_mu_mse": 1.0,
        "ic_u_violation": 1.0,
        "bc_u_violation": 1.0,
        "phase_range_violation": 1.0,
        "mass_proxy_violation": 1.0,
        "interface_proxy_mean": 1.0,
        "ch_residual_mean": 1.0,
        "mu_residual_mean": 1.0,
    }
    after = {**before, "sparse_u_mse": 1.10, "mu_residual_mean": 0.5}
    accepted, decision = trainer._evaluate_with_u_guards(
        candidate,
        1.0,
        0.5,
        before,
        after,
        target_threshold=0.0,
        guard_threshold=0.20,
        comparison_mode="test_counterfactual",
    )
    assert accepted is False
    assert decision["rollback_reason"] == "sparse_u_guard"
    assert trainer.rejected_due_to_sparse_u_guard == 1
