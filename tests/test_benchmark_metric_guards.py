import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controllers.local_controller import LocalControllerConfig, LocalIntervention, LocalVARAController
from src.training.trainer import ExperimentTrainer


def test_smoke_runs_are_not_collapse_evaluated():
    trainer = object.__new__(ExperimentTrainer)
    trainer.config = {"collapse_thresholds": {}}
    metrics = {
        "run_type": "smoke",
        "reportable": False,
        "collapse_evaluated": False,
        "has_reference": True,
        "u_rel_l2": 100.0,
        "unweighted_validation_loss": 100.0,
        "final_total_loss": 100.0,
    }
    assert trainer._collapsed(metrics) is False


def test_weighted_final_loss_alone_does_not_mark_collapse():
    trainer = object.__new__(ExperimentTrainer)
    trainer.config = {"collapse_thresholds": {}}
    metrics = {
        "collapse_evaluated": True,
        "has_reference": True,
        "u_rel_l2": 0.1,
        "v_rel_l2": math.nan,
        "p_rel_l2_centered": 0.1,
        "omega_rel_l2": 0.1,
        "u_rmse": 0.1,
        "v_rmse": 0.1,
        "p_rmse_centered": 0.1,
        "omega_rmse": 0.1,
        "pde_residual_mean": 0.1,
        "continuity_residual_mean": 0.1,
        "momentum_residual_mean": 0.1,
        "unweighted_validation_loss": 0.1,
        "boundary_condition_error": 0.1,
        "final_total_loss": 1.0e6,
    }
    assert trainer._collapsed(metrics) is False


def test_local_objective_uses_rmse_fallback_for_zero_reference_fields():
    controller = LocalVARAController(
        initial_weights={},
        config=LocalControllerConfig.from_dict({"objective_weights": {"v": 1.0}}),
    )
    assert controller.objective({"v_rel_l2": math.nan, "v_rmse": 0.25}) == 0.25


def test_local_acceptance_rejects_continuity_collateral_damage():
    controller = LocalVARAController(
        initial_weights={},
        config=LocalControllerConfig.from_dict(
            {
                "min_improvement": 0.01,
                "continuity_collateral_tolerance": 0.05,
                "objective_weights": {"u": 1.0, "continuity": 1.0},
            }
        ),
    )
    intervention = LocalIntervention(
        variable="u_error",
        patch_id=0,
        action="increase_local_velocity",
        loss_variables=["u"],
        strength=1.0,
        severity=1.0,
        confidence=1.0,
        bounds=(0.0, 1.0, 0.0, 1.0, None, None),
    )
    before_scores = np.array([[1.0]])
    after_scores = np.array([[0.8]])
    before_metrics = {
        "u_rel_l2": 1.0,
        "u_rmse": 1.0,
        "continuity_residual_mean": 1.0,
        "unweighted_validation_loss": 1.0,
    }
    after_metrics = {
        "u_rel_l2": 0.8,
        "u_rmse": 0.8,
        "continuity_residual_mean": 1.2,
        "unweighted_validation_loss": 1.0,
    }
    accepted, decision = controller.evaluate_acceptance(
        [intervention],
        before_scores,
        after_scores,
        ["u_error"],
        before_metrics,
        after_metrics,
        constrained=True,
    )
    assert accepted is False
    assert decision["continuity_collateral_damage"] > 0.05


def test_local_acceptance_rejects_boundary_hard_damage_even_for_boundary_target():
    controller = LocalVARAController(
        initial_weights={},
        config=LocalControllerConfig.from_dict(
            {
                "min_improvement": 0.01,
                "boundary_hard_tolerance": 0.001,
                "objective_weights": {"residual": 1.0},
            }
        ),
    )
    intervention = LocalIntervention(
        variable="boundary_violation",
        patch_id=0,
        action="increase_local_boundary",
        loss_variables=["bc"],
        strength=1.0,
        severity=1.0,
        confidence=1.0,
        bounds=(0.0, 1.0, 0.0, 1.0, None, None),
    )
    before_scores = np.array([[1.0]])
    after_scores = np.array([[0.8]])
    before_metrics = {
        "pde_residual_mean": 1.0,
        "boundary_condition_error": 1.0,
        "unweighted_validation_loss": 1.0,
    }
    after_metrics = {
        "pde_residual_mean": 0.9,
        "boundary_condition_error": 1.02,
        "unweighted_validation_loss": 0.9,
    }
    accepted, decision = controller.evaluate_acceptance(
        [intervention],
        before_scores,
        after_scores,
        ["boundary_violation"],
        before_metrics,
        after_metrics,
        constrained=True,
    )
    assert accepted is False
    assert decision["boundary_hard_damage"] > 0.001


def test_cavity_wall_and_corner_strength_scaling():
    controller = LocalVARAController(
        initial_weights={},
        config=LocalControllerConfig.from_dict(
            {
                "initial_strength": 1.0,
                "wall_patch_strength_factor": 0.5,
                "corner_patch_strength_factor": 0.5,
                "boundary_patch_margin": 1.0e-9,
            }
        ),
    )
    corner = LocalIntervention(
        variable="corner_pde_residual",
        patch_id=0,
        action="increase_local_pde",
        loss_variables=["pde"],
        strength=1.0,
        severity=1.0,
        confidence=1.0,
        bounds=(0.0, 0.25, 0.0, 0.25, None, None),
    )
    from src.diagnostics import WeakRegion

    wr = WeakRegion(corner.patch_id, corner.variable, 1.0, 1.0, corner.bounds, "corner")
    action = controller.propose([wr])[0]
    assert action.strength == 0.25


def test_patch_type_config_fields_load():
    config = LocalControllerConfig.from_dict(
        {
            "benchmark": "lid_driven_cavity",
            "patch_type_aware": True,
            "domain_bounds": [0.0, 1.0, 0.0, 1.0],
            "interior_trial_epochs": 20,
            "wall_trial_epochs": 10,
            "corner_trial_epochs": 5,
            "sampling_only_trial_epochs": 7,
            "near_wall_width": 0.08,
            "centerline_band_width": 0.04,
        }
    )
    assert config.patch_type_aware is True
    assert config.benchmark == "lid_driven_cavity"
    assert config.interior_trial_epochs == 20
    assert config.wall_trial_epochs == 10
    assert config.corner_trial_epochs == 5
    assert config.sampling_only_trial_epochs == 7
    assert config.near_wall_width == 0.08
    assert config.centerline_band_width == 0.04


def test_cavity_corner_patch_prefers_sampling_and_damps_strength():
    from src.diagnostics import PatchGrid, WeakRegion

    grid = PatchGrid(bounds=(0, 1, 0, 1), nx_patches=4, ny_patches=4)
    controller = LocalVARAController(
        initial_weights={},
        config=LocalControllerConfig.from_dict(
            {
                "benchmark": "lid_driven_cavity",
                "patch_type_aware": True,
                "initial_strength": 1.0,
                "wall_patch_strength_factor": 0.5,
                "corner_patch_strength_factor": 0.4,
            }
        ),
    )
    wr = WeakRegion(0, "corner_pde_residual", 1.0, 1.0, grid.get_patch(0).bounds, "corner")
    action = controller.propose([wr])[0]
    assert action.patch_type == "corner"
    assert action.action == "increase_local_corner_sampling"
    assert action.sampling_only is True
    assert action.strength == 0.2


def test_cavity_wall_patch_rejects_boundary_worsening():
    controller = LocalVARAController(
        initial_weights={},
        config=LocalControllerConfig.from_dict(
            {
                "benchmark": "lid_driven_cavity",
                "patch_type_aware": True,
                "min_improvement": 0.01,
                "objective_weights": {"residual": 1.0, "boundary": 0.0, "u_boundary": 0.0},
                "strict_wall_boundary_acceptance": True,
                "wall_boundary_worsen_tolerance": 0.0,
            }
        ),
    )
    intervention = LocalIntervention(
        variable="boundary_violation",
        patch_id=0,
        action="increase_local_boundary",
        loss_variables=["bc"],
        strength=0.5,
        severity=1.0,
        confidence=1.0,
        bounds=(0.75, 1.0, 0.75, 1.0, None, None),
        patch_type="lid",
        action_family="boundary",
    )
    before_scores = np.array([[1.0]])
    after_scores = np.array([[0.8]])
    before_metrics = {
        "pde_residual_mean": 1.0,
        "boundary_condition_error": 1.0,
        "u_boundary_rmse": 1.0,
        "v_boundary_rmse": 1.0,
        "unweighted_validation_loss": 1.0,
    }
    after_metrics = {
        "pde_residual_mean": 0.8,
        "boundary_condition_error": 1.01,
        "u_boundary_rmse": 1.0,
        "v_boundary_rmse": 1.0,
        "unweighted_validation_loss": 0.8,
    }
    accepted, decision = controller.evaluate_acceptance(
        [intervention],
        before_scores,
        after_scores,
        ["boundary_violation"],
        before_metrics,
        after_metrics,
        constrained=True,
    )
    assert accepted is False
    assert decision["strict_wall_boundary_damage"] > 0.0
