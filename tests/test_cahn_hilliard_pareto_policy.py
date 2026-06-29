"""Pareto acceptance and delayed-drift tests for Cahn--Hilliard VARA."""

from __future__ import annotations

from src.controllers.v2_controller import V2Candidate
from src.pde_cahn_hilliard.trainer import CahnHilliardTrainer
from src.utils.config import deep_update, load_config
from test_cahn_hilliard_smoke import tiny_cahn_hilliard_config


def test_pareto_policy_config_loads() -> None:
    config = load_config("configs/cahn_hilliard/base.yaml")
    policy = config["controller_v2"]["cahn_hilliard_pareto_policy"]
    assert policy["enabled"] is True
    assert policy["hard_guards"]["pde_residual_mean"] == 0.060
    assert policy["reject_if_mu_only_improves"] is True


def test_hard_guard_rejects_pde_worsening(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    before, after = _safe_metrics()
    after.update({"sparse_u_mse": 0.9, "pde_residual_mean": 1.07})
    accepted, reason, details = trainer._pareto_decision(
        _candidate("sparse_u_mismatch"), before, after
    )
    assert accepted is False
    assert reason == "hard_guard_pde"
    assert details["primary_reward"] > 0.0


def test_hard_guard_rejects_phase_overshoot(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    before, after = _safe_metrics()
    after.update({"sparse_u_mse": 0.9, "phase_overshoot": 0.0106})
    accepted, reason, _ = trainer._pareto_decision(
        _candidate("sparse_u_mismatch"), before, after
    )
    assert accepted is False
    assert reason == "hard_guard_phase"


def test_hard_guard_rejects_mass_proxy_worsening(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    before, after = _safe_metrics()
    after.update({"sparse_u_mse": 0.9, "mass_proxy_error": 0.108})
    accepted, reason, _ = trainer._pareto_decision(
        _candidate("sparse_u_mismatch"), before, after
    )
    assert accepted is False
    assert reason == "hard_guard_mass"


def test_mu_only_improvement_is_rejected(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    before, after = _safe_metrics()
    after["mu_residual_mean"] = 0.5
    accepted, reason, details = trainer._pareto_decision(
        _candidate("chemical_potential_residual"), before, after
    )
    assert accepted is False
    assert reason == "mu_only"
    assert details["primary_reward"] == 0.0


def test_sparse_u_improvement_without_guard_damage_is_accepted(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    before, after = _safe_metrics()
    after["sparse_u_mse"] = 0.9
    accepted, reason, details = trainer._pareto_decision(
        _candidate("sparse_u_mismatch"), before, after
    )
    assert accepted is True
    assert reason == ""
    assert details["pareto_score"] > 0.0


def test_post_block_guard_detects_artificial_drift(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    before, after = _safe_metrics()
    after["pde_residual_mean"] = 1.08
    safe, reason, changes = trainer._post_block_guard_decision(before, after)
    assert safe is False
    assert reason == "post_block_guard_pde"
    assert changes["pde_residual_mean"] > 0.075


def test_candidate_log_records_hard_guard_rejection_reason(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    before, after = _safe_metrics()
    after.update({"sparse_u_mse": 0.9, "pde_residual_mean": 1.07})
    accepted, decision = trainer._evaluate_with_u_guards(
        _candidate("sparse_u_mismatch"),
        1.0,
        0.5,
        before,
        after,
        target_threshold=0.0,
        guard_threshold=0.2,
        comparison_mode="pareto_test",
    )
    assert accepted is False
    assert decision["rejection_reason"] == "hard_guard_pde"
    assert decision["violated_guard_names"] == ["pde_residual_mean"]
    assert decision["min_required_primary_reward"] == 0.0015
    assert decision["pde_before"] == 1.0
    assert decision["pde_after"] == 1.07
    assert trainer.rejected_hard_guard_pde == 1


def test_adaptive_guard_tolerance_expands_for_noisy_metric(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    trainer.pareto_metric_history["pde_residual_mean"] = [1.0, 1.3, 0.9, 1.25]
    assert trainer._effective_guard_tolerance("pde_residual_mean", 0.06) > 0.06


def test_near_zero_change_does_not_create_fake_guard_violation(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    change = trainer._guard_worsening_ratio(
        "mass_proxy_error", 1.0e-9, 2.0e-6
    )
    assert change == 0.0


def test_positive_reward_with_two_mild_guard_changes_is_accepted(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    before, after = _safe_metrics()
    after.update(
        {
            "sparse_u_mse": 0.99,
            "pde_residual_mean": 1.02,
            "mass_proxy_error": 0.103,
        }
    )
    accepted, reason, details = trainer._pareto_decision(
        _candidate("sparse_u_mismatch"), before, after
    )
    assert accepted is True
    assert reason == ""
    assert set(details["mild_guard_names"]) == {
        "pde_residual_mean",
        "mass_proxy_error",
    }
    assert details["guard_penalty"] == 0.0


def test_large_physics_and_conservation_damage_is_rejected(tmp_path) -> None:
    for metric, damaged_value, expected_reason in (
        ("pde_residual_mean", 1.07, "hard_guard_pde"),
        ("ch_residual_mean", 1.07, "hard_guard_ch"),
        ("mass_proxy_error", 0.108, "hard_guard_mass"),
        ("phase_overshoot", 0.0106, "hard_guard_phase"),
    ):
        trainer = _pareto_trainer(tmp_path / metric)
        before, after = _safe_metrics()
        after.update({"sparse_u_mse": 0.9, metric: damaged_value})
        accepted, reason, _ = trainer._pareto_decision(
            _candidate("sparse_u_mismatch"), before, after
        )
        assert accepted is False
        assert reason == expected_reason


def test_best_safe_candidate_activates_after_rejected_block_streak(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    trainer.consecutive_all_rejected_blocks = 2
    before, after = _safe_metrics()
    after["sparse_u_mse"] = 0.9998
    accepted, decision = trainer._evaluate_with_u_guards(
        _candidate("sparse_u_mismatch"),
        1.0,
        0.9998,
        before,
        after,
        comparison_mode="fallback_test",
    )
    assert accepted is True
    assert decision["acceptance_mode"] == "best_safe_fallback"
    assert trainer.best_safe_candidate_activations == 1


def test_inactivity_recovery_waits_for_configured_trigger(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    assert trainer.controller is not None
    initial_trust = trainer.controller.trust_radius
    trainer._register_block_outcome(False)
    assert trainer.controller.trust_radius == initial_trust
    assert trainer.inactivity_recovery_activations == 0
    trainer._register_block_outcome(False)
    assert trainer.controller.trust_radius > initial_trust
    assert trainer.controller.trust_radius <= 0.12
    assert trainer.inactivity_recovery_activations == 1
    assert trainer._temporary_reward_multiplier == 0.5


def test_post_block_guard_ignores_tiny_noise_near_zero(tmp_path) -> None:
    trainer = _pareto_trainer(tmp_path)
    before, after = _safe_metrics()
    before["mass_proxy_error"] = 1.0e-9
    after["mass_proxy_error"] = 2.0e-6
    safe, reason, _ = trainer._post_block_guard_decision(before, after)
    assert safe is True
    assert reason == ""


def _pareto_trainer(tmp_path) -> CahnHilliardTrainer:
    base = load_config("configs/cahn_hilliard/base.yaml")
    controller = base["controller_v2"]
    config = deep_update(
        tiny_cahn_hilliard_config(),
        {
            "controller_v2": {
                "cahn_hilliard_pareto_policy": controller[
                    "cahn_hilliard_pareto_policy"
                ],
                "cahn_hilliard_pareto_score": controller[
                    "cahn_hilliard_pareto_score"
                ],
                "cahn_hilliard_mass_guard": controller[
                    "cahn_hilliard_mass_guard"
                ],
                "cahn_hilliard_phase_guard": controller[
                    "cahn_hilliard_phase_guard"
                ],
                "cahn_hilliard_post_block_guard": controller[
                    "cahn_hilliard_post_block_guard"
                ],
                "cahn_hilliard_best_safe_candidate": controller[
                    "cahn_hilliard_best_safe_candidate"
                ],
                "cahn_hilliard_inactivity_recovery": controller[
                    "cahn_hilliard_inactivity_recovery"
                ],
            }
        },
    )
    return CahnHilliardTrainer(config, "vara_v2", tmp_path / "run")


def _safe_metrics() -> tuple[dict[str, float], dict[str, float]]:
    before = {
        "pde_residual_mean": 1.0,
        "ch_residual_mean": 1.0,
        "mu_residual_mean": 1.0,
        "mass_proxy_error": 0.1,
        "phase_overshoot": 0.01,
        "sparse_u_mse": 1.0,
        "sparse_mu_mse": 1.0,
        "ic_u_error": 1.0,
        "bc_u_error": 1.0,
        "interface_proxy_error": 1.0,
        "boundary_condition_error": 1.0,
        "unweighted_validation_loss": 1.0,
        "unweighted_physics_validation_loss": 1.0,
    }
    return before, dict(before)


def _candidate(variable: str) -> V2Candidate:
    loss_name = (
        "chemical_potential_residual"
        if variable == "chemical_potential_residual"
        else "sparse_u_mse"
    )
    return V2Candidate(
        variable=variable,
        patch_id=0,
        action_type="local_loss",
        loss_names=[loss_name],
        severity=1.0,
        persistence=1,
        trend=0.0,
        predicted_target_improvement=0.5,
    )
