from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_vara_v2_continuation import (
    _apply_re_aware_cavity_settings,
    _continuation_validity,
    _load_base_config,
    _validate_reliable_config,
    _without_cavity_stabilizers,
)
from scripts.check_lid_cavity_re100_sanity import build_report
from src.controllers import V2ControllerConfig, VARAV2Controller
from src.evaluation.metrics import evaluate_on_grid
from src.evaluation.statistical_tests import (
    holm_adjust,
    paired_bootstrap_improvement,
    wilcoxon_signed_rank,
)
from src.losses.base_losses import (
    _add_reference_free_regularizers,
    compute_global_losses,
    compute_pointwise_losses,
)
from src.losses.local_losses import compute_budgeted_patch_losses
from src.models import (
    CavityHardBoundaryWrapper,
    HardBoundaryStreamfunctionPressureWrapper,
    StreamfunctionPressureWrapper,
    build_mlp_from_config,
    parameter_matched_width,
)
from src.physics.navier_stokes import navier_stokes_residuals
from src.physics.rectangular_benchmarks import LidDrivenCavityQualitative
from src.sampling.boundary_sampler import BoundarySampler, boundary_side_fractions
from src.physics.taylor_green import TaylorGreenVortex
from src.training.checkpointing import save_checkpoint
from src.training.checkpointing import load_checkpoint
from src.training.vara_trainer import VARATrainer
from src.training.vara_v2_trainer import VARAV2Trainer
from src.utils.config import deep_update, load_config
from src.utils.logging import CSVLogger
from src.visualization.streamlines import (
    detect_vortices,
    lid_cavity_topology_metrics,
    reconstruct_streamfunction,
)


def test_v2_allocation_conserves_sampling_and_loss_mass():
    controller = VARAV2Controller(V2ControllerConfig(num_patches=16))

    class Region:
        variable = "continuity_residual"
        patch_id = 15
        severity = 1.0
        confidence = 0.8
        persistence = 2

    candidate = controller.candidates([Region()])[2]
    controller.apply(candidate)
    controller.validate_state()
    assert np.isclose(controller.state.sampling_mass.sum(), 1.0)
    assert controller.state.sampling_mass.max() <= 0.25
    for values in controller.state.loss_multipliers.values():
        assert np.isclose(values.mean(), 1.0)
        assert values.min() >= 0.5
        assert values.max() <= 2.0


def test_mean_pointwise_reduction_does_not_square_squared_losses_again():
    pointwise = {"pde": torch.tensor([[1.0], [4.0]])}
    corrected = compute_global_losses(pointwise, reduction="mean")
    legacy = compute_global_losses(pointwise, reduction="legacy_mse")
    assert corrected["pde"].item() == pytest.approx(2.5)
    assert legacy["pde"].item() == pytest.approx(8.5)


def test_loss_normalization_equalizes_selected_terms(tmp_path):
    config = load_config("configs/vara_v2/lid_driven_cavity.yaml")
    config = deep_update(
        config,
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "experiments": {"root": str(tmp_path)},
            "loss_normalization": {
                "enabled": True,
                "names": ["momentum_u", "continuity"],
                "ema": 0.5,
                "min_scale": 1.0e-6,
            },
        },
    )
    trainer = VARATrainer(config, mode="vanilla")
    losses = {
        "momentum_u": torch.tensor(100.0),
        "continuity": torch.tensor(0.25),
        "bc": torch.tensor(1.0),
    }
    normalized, logs = trainer.normalize_training_losses(losses)
    assert normalized["momentum_u"].item() == pytest.approx(1.0)
    assert normalized["continuity"].item() == pytest.approx(1.0)
    assert normalized["bc"].item() == pytest.approx(1.0)
    assert logs["loss_scale_momentum_u"] == pytest.approx(100.0)


def test_v2_budgeted_reduction_has_same_definition_before_and_after_action():
    class Grid:
        @staticmethod
        def assign_torch(coords):
            return torch.tensor([0, 1], device=coords.device)

    values = torch.tensor([[1.0], [4.0]])
    batch = {"xy_f": torch.tensor([[0.0, 0.0], [1.0, 1.0]])}
    before = compute_budgeted_patch_losses(
        {"pde": values},
        batch,
        Grid(),
        {},
        reduction="mean",
    )
    after = compute_budgeted_patch_losses(
        {"pde": values},
        batch,
        Grid(),
        {"pde": np.ones(2)},
        reduction="mean",
    )
    assert before["pde"].item() == pytest.approx(2.5)
    assert after["pde"].item() == pytest.approx(2.5)

    legacy_before = compute_budgeted_patch_losses(
        {"pde": values},
        batch,
        Grid(),
        {},
        reduction="legacy_mse",
    )
    legacy_after = compute_budgeted_patch_losses(
        {"pde": values},
        batch,
        Grid(),
        {"pde": np.ones(2)},
        reduction="legacy_mse",
    )
    assert legacy_before["pde"].item() == pytest.approx(8.5)
    assert legacy_after["pde"].item() == pytest.approx(8.5)


def test_v2_rejects_reference_or_test_signal_names():
    controller = VARAV2Controller(V2ControllerConfig(num_patches=4))
    for name in [
        "velocity_full_rel_l2",
        "ghia_profile_score",
        "cfd_reference_error",
        "test_rmse",
        "lid_cavity_topology_score",
    ]:
        with pytest.raises(ValueError):
            controller.assert_reference_free([name])


def test_v2_publication_config_hides_reference_metrics_from_legacy_controller(
    tmp_path,
    monkeypatch,
):
    config = load_config("configs/vara_v2/lid_driven_cavity.yaml")
    config = deep_update(config, load_config("configs/vara_v2/controller.yaml"))
    config = deep_update(
        config,
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "benchmark_params": {
                "reference": "ghia",
                "profile_only": True,
            },
            "validation": {"nx": 6, "ny": 6},
            "experiments": {"root": str(tmp_path)},
        },
    )
    trainer = VARAV2Trainer(config)
    _, _, coords = trainer.validation_grid()
    reporting = trainer.evaluate_metrics(coords)
    monkeypatch.setattr(
        type(trainer.benchmark),
        "exact_np",
        lambda _self, _coords: (_ for _ in ()).throw(
            AssertionError("controller path loaded evaluation reference")
        ),
    )
    controller = trainer.controller_metrics(coords)
    assert np.isfinite(reporting["centerline_profile_score"])
    assert np.isnan(controller["centerline_profile_score"])
    assert controller["unweighted_validation_loss"] == pytest.approx(
        controller["unweighted_pde_loss"] + controller["unweighted_bc_loss"]
    )


def test_v2_trust_radius_stays_inside_bounds():
    controller = VARAV2Controller(V2ControllerConfig(num_patches=16))

    class Region:
        variable = "aggregate_pde_residual"
        patch_id = 3
        severity = 1.0
        confidence = 0.8
        persistence = 2

    candidate = controller.candidates([Region()])[0]
    for _ in range(20):
        controller.record_prefilter(candidate)
    assert controller.trust_radius == pytest.approx(controller.config.trust_radius_min)


def test_v2_noise_units_are_bounded_against_nonstationary_training_drift():
    controller = VARAV2Controller(
        V2ControllerConfig(
            num_patches=4,
            target_noise_ceiling=0.25,
            guard_noise_ceiling=0.10,
        )
    )
    controller.score_history[("pde_residual", 0)] = [1.0, 4.0, 0.5, 3.0]
    controller.metric_history["pde_residual_mean"] = [1.0, 3.0, 0.4, 2.0]
    assert controller.target_noise("pde_residual", 0) <= 0.25
    assert controller.metric_noise("pde_residual_mean") <= 0.10


def test_residual_fourier_backbone_is_parameter_matched():
    width, enhanced, legacy = parameter_matched_width(2, 3, [64, 64, 64], [1, 2, 4, 8], 4)
    assert width > 0
    assert abs(enhanced - legacy) / legacy <= 0.05
    config = {
        "model": {
            "architecture": "residual_fourier_mlp",
            "input_dim": 2,
            "output_dim": 3,
            "hidden_layers": [64, 64, 64],
            "comparison_hidden_layers": [64, 64, 64],
            "frequencies": [1, 2, 4, 8],
            "residual_blocks": 4,
        }
    }
    model = build_mlp_from_config(config, (0.0, 1.0, 0.0, 1.0))
    assert sum(parameter.numel() for parameter in model.parameters()) == enhanced


def test_cavity_hard_boundary_wrapper_enforces_regularized_walls():
    class ConstantBase(torch.nn.Module):
        def forward(self, coords):
            return torch.ones((coords.shape[0], 3), dtype=coords.dtype, device=coords.device)

    model = CavityHardBoundaryWrapper(
        ConstantBase(),
        (0.0, 1.0, 0.0, 1.0),
        lid_velocity=1.0,
        corner_width=0.1,
    )
    points = torch.tensor(
        [
            [0.0, 0.4],
            [1.0, 0.4],
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    velocity = model(points)[:, :2]
    assert torch.allclose(velocity[:3], torch.zeros_like(velocity[:3]), atol=1e-12)
    assert velocity[3, 0].item() == pytest.approx(1.0)
    assert velocity[3, 1].item() == pytest.approx(0.0)
    assert torch.allclose(velocity[4:], torch.zeros_like(velocity[4:]), atol=1e-12)


def test_continuation_overlay_inherits_lid_cavity_base_config():
    config = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    assert config["benchmark"] == "lid_driven_cavity"
    assert config["model"]["physics_formulation"] == "hard_boundary_streamfunction_pressure"
    assert config["model"]["output_dim"] == 2
    assert config["model"]["hard_boundary_corner_width"] == pytest.approx(0.08)
    assert config["model"]["hard_boundary_lid_vertical_power"] == 2
    assert config["model"]["hard_boundary_correction_scale"] == pytest.approx(24.0)
    assert config["training"]["adaptive_cycles"] == 20
    assert config["controller_v2"]["total_steps"] == 4000
    assert config["training"]["residual_loss_mode"]["initial"] == "pseudo_huber"
    assert config["checkpoint"]["score_mode"] == "sum"
    assert config["loss_normalization"]["enabled"] is False
    assert config["evaluation"]["controller_reference_metrics_enabled"] is False
    assert config["evaluation"]["checkpoint_reference_metrics_enabled"] is False
    assert "unweighted_physics_validation_loss" in config["controller_v2"]["guard_metrics"]


def test_uvp_soft_bc_formulation_returns_raw_fields_and_residuals():
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    config = _apply_re_aware_cavity_settings(base, 100.0)
    model = build_mlp_from_config(config, (0.0, 1.0, 0.0, 1.0))
    points = torch.tensor(
        [[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]],
        dtype=torch.float32,
        requires_grad=True,
    )
    prediction = model(points)
    residuals = navier_stokes_residuals(model, points, nu=0.01, steady=True)
    assert model.physics_formulation == "cavity_uvp_soft_bc"
    assert prediction.shape == (3, 3)
    assert all(torch.isfinite(residuals[name]).all() for name in ("f_u", "f_v", "f_c"))


def test_uvp_soft_boundary_loss_covers_all_walls_and_pressure_gauge(tmp_path):
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    config = deep_update(
        _apply_re_aware_cavity_settings(base, 100.0),
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "training": {
                "n_collocation": 8,
                "n_boundary": 16,
                "n_data": 0,
                "collocation_curriculum": {"enabled": False},
            },
            "experiments": {"root": str(tmp_path), "flat_layout": True},
            "continuation_replay": {"enabled": False},
        },
    )
    trainer = VARATrainer(config, mode="vanilla")
    walls = trainer._sample_boundary(16)
    fractions = boundary_side_fractions(walls.detach().cpu().numpy(), trainer.benchmark.bounds)
    assert all(
        fractions[name] > 0.0
        for name in (
            "boundary_fraction_left",
            "boundary_fraction_right",
            "boundary_fraction_bottom",
            "boundary_fraction_top",
        )
    )
    assert trainer._compute_boundary_training_loss(config["training"]["weights"])
    with torch.no_grad():
        trainer.model.layers[-1].bias[2] = 1.0
    assert trainer.pressure_gauge_loss().item() > 0.0


def test_uvp_component_boundary_losses_are_wall_specific():
    config = {
        "benchmark": "lid_driven_cavity",
        "model": {
            "physics_formulation": "cavity_uvp_soft_bc",
            "input_dim": 2,
            "output_dim": 3,
            "hidden_layers": [8, 8],
        },
    }
    model = build_mlp_from_config(config, (0.0, 1.0, 0.0, 1.0))
    benchmark = LidDrivenCavityQualitative(
        reynolds=100.0,
        lid_corner_regularization_width=0.05,
    )
    xy_f = torch.tensor(
        [[0.25, 0.25], [0.75, 0.75]],
        dtype=torch.float32,
        requires_grad=True,
    )
    walls = torch.tensor(
        [[0.5, 1.0], [0.5, 0.0], [0.0, 0.5], [1.0, 0.5]],
        dtype=torch.float32,
    )
    batch = {"xy_f": xy_f, "xy_bc": walls, "xy_data": None, "targets_data": None}
    pointwise = compute_pointwise_losses(model, batch, benchmark, True)
    names = [
        f"bc_{wall}_{component}"
        for wall in ("top", "bottom", "left", "right")
        for component in ("u", "v")
    ]
    assert all(pointwise[name].numel() == 1 for name in names)

    extra_bottom = torch.tensor([[0.2, 0.0], [0.8, 0.0]], dtype=torch.float32)
    expanded = {**batch, "xy_bc": torch.cat([walls, extra_bottom], dim=0)}
    expanded_pointwise = compute_pointwise_losses(model, expanded, benchmark, True)
    assert torch.allclose(
        pointwise["bc_top_u"],
        expanded_pointwise["bc_top_u"],
    )


def test_uvp_balanced_boundary_is_normalized_component_average():
    config = {
        "benchmark": "lid_driven_cavity",
        "model": {
            "physics_formulation": "cavity_uvp_soft_bc",
            "input_dim": 2,
            "output_dim": 3,
            "hidden_layers": [8, 8],
        },
    }
    model = build_mlp_from_config(config, (0.0, 1.0, 0.0, 1.0))
    benchmark = LidDrivenCavityQualitative(reynolds=100.0)
    xy_f = torch.tensor(
        [[0.25, 0.25], [0.75, 0.75]],
        dtype=torch.float32,
        requires_grad=True,
    )
    walls = torch.tensor(
        [[0.5, 1.0], [0.5, 0.0], [0.0, 0.5], [1.0, 0.5]],
        dtype=torch.float32,
    )
    relative = {
        "bc_top_u": 1.3,
        "bc_top_v": 1.0,
        "bc_bottom_u": 1.3,
        "bc_bottom_v": 1.0,
        "bc_left_u": 1.1,
        "bc_left_v": 1.1,
        "bc_right_u": 1.1,
        "bc_right_v": 1.1,
    }
    pointwise = compute_pointwise_losses(
        model,
        {"xy_f": xy_f, "xy_bc": walls, "xy_data": None, "targets_data": None},
        benchmark,
        True,
        regularization_config={
            "uvp_boundary_balance": {
                "enabled": True,
                "relative_weights": relative,
            }
        },
    )
    expected = sum(
        weight * pointwise[name].mean() for name, weight in relative.items()
    ) / sum(relative.values())
    assert pointwise["bc_uvp_balanced"].detach().item() == pytest.approx(
        expected.detach().item()
    )
    assert pointwise["bc_uvp_balanced"].detach().item() < sum(
        pointwise[name].mean() for name in relative
    ).detach().item()


def test_reliable_uvp_defaults_preserve_budgets_and_reference_free_checkpointing():
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    for preset, expected in (("diagnostic", 2000), ("reliable", 4000)):
        config = deep_update(
            base,
            load_config(f"configs/vara_v2/presets/{preset}.yaml"),
        )
        config = _apply_re_aware_cavity_settings(config, 100.0)
        _validate_reliable_config(config, preset, require_materialized=True)
        assert config["model"]["physics_formulation"] == "cavity_uvp_soft_bc"
        assert config["controller_v2"]["total_steps"] == expected
        assert config["optimizer"]["scheduler"]["total_steps"] == expected
        assert config["training"]["weights"]["continuity"] == pytest.approx(8.0)
        assert config["training"]["weights"]["bc"] == pytest.approx(0.0)
        assert config["training"]["weights"]["bc_uvp_balanced"] == pytest.approx(30.0)
        assert all(
            config["training"]["weights"][name] == pytest.approx(0.0)
            for name in config["uvp_soft_bc"]["boundary_component_relative_weights"]
        )
        assert config["training"]["weights"]["raw_pde_tail"] == pytest.approx(0.04)
        assert config["training"]["n_boundary"] == (384 if preset == "diagnostic" else 512)
        relative = config["losses"]["uvp_boundary_balance"]["relative_weights"]
        assert relative["bc_top_u"] > relative["bc_top_v"]
        assert relative["bc_bottom_u"] > relative["bc_bottom_v"]
        boundary_cfg = config["sampling"]["cavity_boundary"]
        assert boundary_cfg["mode"] == "focused"
        assert boundary_cfg["lid_fraction"] == pytest.approx(0.25)
        assert boundary_cfg["corner_fraction"] == pytest.approx(0.08)
        assert config["benchmark_params"][
            "lid_corner_regularization_width"
        ] == pytest.approx(0.05)
        repair = config["optimizer"]["final_repair"]
        assert repair["enabled"]
        assert repair["epochs"] == (6 if preset == "diagnostic" else 30)
        assert repair["lr"] == pytest.approx(0.08 if preset == "diagnostic" else 0.10)
        assert repair["score_metric"] == "uvp_reference_free_score"
        assert config["continuation_validity"][
            "max_continuity_residual_mean"
        ] == pytest.approx(0.02 if preset == "diagnostic" else 0.01)
        assert config["continuation_validity"][
            "max_boundary_condition_error"
        ] == pytest.approx(0.06 if preset == "diagnostic" else 0.03)
        metrics = config["checkpoint"]["reference_free_metrics"]
        assert "continuity_residual_mean" in metrics
        assert "boundary_condition_error" in metrics
        assert "u_boundary_rmse" in metrics
        assert not any("rel_l2" in name for name in metrics)
        assert not config["checkpoint"]["low_re_vortex_tiebreaker"]["enabled"]
        trainer = object.__new__(VARATrainer)
        trainer.config = config
        assert trainer._compute_boundary_training_loss(
            config["training"]["weights"]
        )


def test_uvp_formulation_is_identical_for_vanilla_and_vara(tmp_path):
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    config = deep_update(
        _apply_re_aware_cavity_settings(base, 100.0),
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "training": {
                "n_collocation": 8,
                "n_boundary": 16,
                "n_data": 0,
                "collocation_curriculum": {"enabled": False},
            },
            "experiments": {"root": str(tmp_path), "flat_layout": True},
            "continuation_replay": {"enabled": False},
        },
    )
    vanilla = VARATrainer(config, mode="vanilla")
    vara = VARAV2Trainer(config)
    assert vanilla.model.physics_formulation == "cavity_uvp_soft_bc"
    assert vara.model.physics_formulation == vanilla.model.physics_formulation
    assert vara.config["training"]["weights"] == vanilla.config["training"]["weights"]
    assert vara.config["continuation_validity"] == vanilla.config["continuation_validity"]
    assert vara.config["checkpoint"] == vanilla.config["checkpoint"]
    assert vara.config["optimizer"]["final_repair"] == vanilla.config["optimizer"][
        "final_repair"
    ]


def test_uvp_corner_width_is_decoupled_from_hard_boundary_schedule():
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    for reynolds, uvp_width in ((100.0, 0.05), (400.0, 0.04), (1600.0, 0.03)):
        config = _apply_re_aware_cavity_settings(base, reynolds)
        assert config["benchmark_params"][
            "lid_corner_regularization_width"
        ] == pytest.approx(uvp_width)
        assert config["model"]["hard_boundary_corner_width"] != pytest.approx(
            uvp_width
        )


def test_uvp_polish_score_is_reference_free_checkpoint_score(tmp_path):
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    config = deep_update(
        _apply_re_aware_cavity_settings(base, 100.0),
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "experiments": {"root": str(tmp_path), "flat_layout": True},
        },
    )
    trainer = VARATrainer(config, mode="vanilla")
    metrics = {
        "pde_residual_mean": 0.1,
        "momentum_residual_mean": 0.1,
        "core_pde_residual_mean": 0.1,
        "continuity_residual_mean": 0.01,
        "boundary_condition_error": 0.02,
        "u_boundary_rmse": 0.03,
        "near_wall_pde_residual_mean": 0.2,
        "near_wall_momentum_v_mean": 0.1,
        "speed_pred_max": 1.0,
        "u_rel_l2": 1000.0,
        "velocity_full_rel_l2": 1000.0,
        "lid_cavity_primary_center_error": 1000.0,
    }
    name, score = trainer._repair_score(metrics)
    assert name == "uvp_reference_free_score"
    assert score == pytest.approx(trainer._checkpoint_score(metrics))


def test_uvp_repair_pareto_guard_rejects_boundary_only_improvement(tmp_path):
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    config = deep_update(
        _apply_re_aware_cavity_settings(base, 100.0),
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "experiments": {"root": str(tmp_path), "flat_layout": True},
        },
    )
    trainer = VARATrainer(config, mode="vanilla")
    before = {
        "pde_residual_mean": 0.20,
        "momentum_residual_mean": 0.20,
        "core_pde_residual_mean": 0.20,
        "continuity_residual_mean": 0.01,
        "boundary_condition_error": 0.05,
        "u_boundary_rmse": 0.08,
        "speed_pred_max": 1.0,
    }
    after = {
        **before,
        "pde_residual_mean": 0.24,
        "core_pde_residual_mean": 0.24,
        "boundary_condition_error": 0.02,
        "u_boundary_rmse": 0.04,
    }
    safe, reason = trainer._repair_pareto_safe(
        before,
        after,
        config["optimizer"]["final_repair"],
    )
    assert not safe
    assert reason.startswith("pareto_worsened_")


def test_uvp_validity_is_soft_bc_aware_but_keeps_physics_gates():
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    diagnostic = deep_update(
        base,
        load_config("configs/vara_v2/presets/diagnostic.yaml"),
    )
    diagnostic = _apply_re_aware_cavity_settings(diagnostic, 100.0)
    valid = {
        "pde_residual_mean": 0.20,
        "continuity_residual_mean": 0.015,
        "momentum_residual_mean": 0.20,
        "boundary_condition_error": 0.05,
        "speed_pred_max": 1.2,
        "streamfunction_consistency_rmse": 99.0,
        "lid_cavity_primary_center_error": 0.05,
        "lid_cavity_topology_score": 0.05,
        "lid_cavity_topology_aligned": 1.0,
        "primary_streamfunction_abs": 0.02,
        "speed_pred_mean": 0.2,
        "detected_vortex_count": 1,
        "primary_vortex_center_x": 0.6,
        "primary_vortex_center_y": 0.7,
        "near_wall_pde_residual_mean": 0.2,
        "near_wall_momentum_v_mean": 0.2,
        "core_pde_residual_mean": 0.2,
    }
    assert _continuation_validity(valid, diagnostic)["continuation_stage_valid"]
    for metric, value in (
        ("pde_residual_mean", 0.31),
        ("momentum_residual_mean", 0.31),
        ("core_pde_residual_mean", 0.36),
        ("near_wall_pde_residual_mean", 3.1),
        ("speed_pred_max", 2.3),
    ):
        assert not _continuation_validity(
            {**valid, metric: value}, diagnostic
        )["continuation_stage_valid"]


def test_hard_boundary_validity_gates_remain_exact():
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    base["cavity_base_formulation"] = "hard_boundary_streamfunction_pressure"
    hard = _apply_re_aware_cavity_settings(base, 100.0)
    assert hard["continuation_validity"]["max_boundary_condition_error"] == pytest.approx(
        1.0e-8
    )
    assert hard["continuation_validity"]["max_continuity_residual_mean"] == pytest.approx(
        0.001
    )
    assert hard["continuation_validity"][
        "max_streamfunction_consistency_rmse"
    ] == pytest.approx(0.001)


def test_re_aware_cavity_settings_cover_low_mid_and_high_reynolds():
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    base["cavity_base_formulation"] = "hard_boundary_streamfunction_pressure"
    low = _apply_re_aware_cavity_settings(base, 100.0)
    mid = _apply_re_aware_cavity_settings(base, 400.0)
    high = _apply_re_aware_cavity_settings(base, 1600.0)

    assert low["model"]["hard_boundary_corner_width"] == pytest.approx(0.09)
    assert low["model"]["hard_boundary_correction_scale"] == pytest.approx(22.0)
    assert low["training"]["residual_loss_mode"]["switch_step"] == 3800
    assert low["losses"]["near_wall_momentum"]["stages"][-1]["band_width"] == pytest.approx(
        0.12
    )
    assert low["sampling"]["cavity_near_wall"]["fraction"] == pytest.approx(0.20)
    assert [
        stage["weight"]
        for stage in low["losses"]["near_wall_momentum"]["stages"]
    ] == pytest.approx([1.0, 1.15, 1.25, 1.25])
    assert low["training"]["weights"]["raw_psi_mean_l2"] == pytest.approx(0.002)
    assert low["training"]["weights"]["scaled_correction_mean_l2"] == pytest.approx(
        0.02
    )
    assert low["losses"]["correction_bubble"]["abs_max_cap"] == pytest.approx(0.30)
    assert low["training"]["weights"]["top_reverse_u"] == pytest.approx(0.07)
    assert low["training"]["weights"]["bottom_positive_u"] == pytest.approx(0.04)
    assert low["training"]["weights"]["raw_pde_tail"] == pytest.approx(0.08)
    assert low["losses"]["raw_residual_tail"]["threshold"] == pytest.approx(0.50)
    assert low["evaluation"]["controller_streamfunction_metrics"] is True
    assert low["losses"]["lid_shear_direction"]["bottom_u_tolerance"] == pytest.approx(
        0.04
    )
    assert low["training"]["residual_loss_mode"]["final"] == "pseudo_huber"
    assert low["continuation_validity"]["max_detected_vortices"] == 2
    assert low["continuation_validity"]["require_lid_cavity_topology_alignment"] is True

    assert mid["model"]["hard_boundary_corner_width"] == pytest.approx(0.07)
    assert mid["model"]["hard_boundary_lid_vertical_power"] == 3
    assert mid["model"]["hard_boundary_correction_scale"] == pytest.approx(28.0)
    assert mid["losses"]["near_wall_momentum"]["stages"][-1]["band_width"] == pytest.approx(
        0.06
    )
    assert mid["sampling"]["cavity_near_wall"]["fraction"] == pytest.approx(0.40)
    assert mid["training"]["weights"]["top_reverse_u"] == pytest.approx(0.015)
    assert mid["training"]["weights"]["raw_pde_tail"] == pytest.approx(0.05)
    assert mid["continuation_validity"]["max_detected_vortices"] == 4

    assert high["model"]["hard_boundary_corner_width"] == pytest.approx(0.05)
    assert high["model"]["hard_boundary_correction_scale"] == pytest.approx(32.0)
    assert high["losses"]["near_wall_momentum"]["stages"][-1]["band_width"] == pytest.approx(
        0.03
    )
    assert high["sampling"]["cavity_near_wall"]["fraction"] == pytest.approx(0.50)
    assert high["training"]["weights"]["top_reverse_u"] == pytest.approx(0.0)
    assert high["training"]["weights"]["bottom_positive_u"] == pytest.approx(0.0)
    assert high["training"]["weights"]["raw_pde_tail"] == pytest.approx(0.02)
    assert high["evaluation"]["controller_streamfunction_metrics"] is False
    assert high["losses"]["lid_shear_direction"]["enabled"] is False
    assert high["continuation_validity"]["max_detected_vortices"] == 8
    assert high["continuation_validity"]["require_lid_cavity_topology_alignment"] is False


def test_high_re_validity_allows_physical_secondary_vortices_without_reference_gate():
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    metrics = {
        "has_reference": True,
        "pde_residual_mean": 0.1,
        "continuity_residual_mean": 1e-8,
        "momentum_residual_mean": 0.1,
        "boundary_condition_error": 0.0,
        "speed_pred_max": 1.5,
        "streamfunction_consistency_rmse": 1e-4,
        "lid_cavity_primary_center_error": 0.20,
        "lid_cavity_topology_score": 0.35,
        "velocity_full_rel_l2": 10.0,
        "lid_cavity_topology_aligned": 0.0,
        "primary_streamfunction_abs": 0.02,
        "speed_pred_mean": 0.2,
        "detected_vortex_count": 5,
        "primary_vortex_center_x": 0.6,
        "primary_vortex_center_y": 0.7,
        "near_wall_pde_residual_mean": 0.2,
        "near_wall_momentum_v_mean": 0.2,
        "core_pde_residual_mean": 0.1,
    }
    high = _apply_re_aware_cavity_settings(base, 1600.0)
    low = _apply_re_aware_cavity_settings(base, 100.0)

    assert _continuation_validity(metrics, high)["continuation_stage_valid"]
    assert not _continuation_validity(metrics, low)["continuation_stage_valid"]


def test_cavity_hard_boundary_accepts_lid_cavity_alias():
    config = {
        "benchmark": "lid_cavity",
        "model": {
            "physics_formulation": "cavity_hard_boundary",
            "input_dim": 2,
            "output_dim": 3,
            "hidden_layers": [8, 8],
        },
    }
    assert isinstance(
        build_mlp_from_config(config, (0.0, 1.0, 0.0, 1.0)),
        CavityHardBoundaryWrapper,
    )


def test_hard_boundary_streamfunction_pressure_builds_for_lid_cavity_alias():
    config = {
        "benchmark": "lid_cavity",
        "model": {
            "physics_formulation": "hard_boundary_streamfunction_pressure",
            "input_dim": 2,
            "output_dim": 2,
            "hidden_layers": [8, 8],
        },
    }
    assert isinstance(
        build_mlp_from_config(config, (0.0, 1.0, 0.0, 1.0)),
        HardBoundaryStreamfunctionPressureWrapper,
    )


def test_hard_boundary_streamfunction_correction_scale_is_configurable():
    class UnitPsi(torch.nn.Module):
        def forward(self, coords):
            raw_psi = torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
            raw_p = torch.zeros_like(raw_psi)
            return torch.cat([raw_psi, raw_p], dim=1)

    coords = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    small = HardBoundaryStreamfunctionPressureWrapper(
        UnitPsi(),
        (0.0, 1.0, 0.0, 1.0),
        correction_scale=32.0,
    )
    large = HardBoundaryStreamfunctionPressureWrapper(
        UnitPsi(),
        (0.0, 1.0, 0.0, 1.0),
        correction_scale=128.0,
    )
    small_correction = small.streamfunction_auxiliary(coords)["scaled_correction"]
    large_correction = large.streamfunction_auxiliary(coords)["scaled_correction"]
    assert large_correction.item() == pytest.approx(4.0 * small_correction.item())


def test_reference_free_regularizers_and_pseudo_huber_are_available():
    class SmoothBase(torch.nn.Module):
        def forward(self, coords):
            x = coords[:, 0:1]
            y = coords[:, 1:2]
            return torch.cat([x * y, x + y], dim=1)

    benchmark = LidDrivenCavityQualitative(
        reynolds=100.0,
        lid_corner_regularization_width=0.05,
    )
    model = HardBoundaryStreamfunctionPressureWrapper(
        SmoothBase(),
        benchmark.bounds,
        corner_width=0.05,
    )
    batch = {
        "xy_f": torch.tensor(
            [[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]],
            dtype=torch.float64,
        ),
        "xy_bc": torch.tensor(
            [[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]],
            dtype=torch.float64,
        ),
    }
    mse_pointwise = compute_pointwise_losses(model, batch, benchmark, True)
    robust_pointwise = compute_pointwise_losses(
        model,
        batch,
        benchmark,
        True,
        residual_loss_mode="pseudo_huber",
        pseudo_huber_delta=1.0,
        regularization_config={
            "speed_cap": {"enabled": True, "cap": 0.01},
            "raw_psi_l2": {"enabled": True},
            "pressure_gradient_l2": {"enabled": True},
        },
    )
    assert torch.mean(robust_pointwise["momentum_u"]) <= torch.mean(
        mse_pointwise["momentum_u"]
    )
    assert "speed_cap" in robust_pointwise
    assert "raw_psi_l2" in robust_pointwise
    assert "pressure_gradient_l2" in robust_pointwise


def test_near_wall_momentum_curriculum_masks_only_training_objective():
    class SmoothBase(torch.nn.Module):
        def forward(self, coords):
            x = coords[:, 0:1]
            y = coords[:, 1:2]
            return torch.cat([x * y, x + y], dim=1)

    benchmark = LidDrivenCavityQualitative(reynolds=100.0)
    model = HardBoundaryStreamfunctionPressureWrapper(
        SmoothBase(),
        benchmark.bounds,
        corner_width=0.08,
        lid_vertical_power=2,
        correction_scale=32.0,
    )
    batch = {
        "xy_f": torch.tensor(
            [[0.01, 0.5], [0.5, 0.5]],
            dtype=torch.float64,
        ),
        "xy_bc": torch.empty((0, 2), dtype=torch.float64),
    }
    pointwise = compute_pointwise_losses(
        model,
        batch,
        benchmark,
        True,
        regularization_config={
            "domain_bounds": benchmark.bounds,
            "near_wall_momentum": {
                "enabled": True,
                "band_width": 0.1,
                "weight": 2.0,
                "normalize_mean": True,
            },
        },
        compute_boundary_loss=False,
    )
    weights = pointwise["near_wall_momentum_weight_mean"].reshape(-1)
    assert weights[0].item() > weights[1].item()
    assert weights.mean().item() == pytest.approx(1.0)
    assert torch.isfinite(pointwise["raw_momentum_v_mse"]).all()


def test_near_wall_vorticity_penalizes_only_quantile_excess():
    coords = torch.tensor(
        [[0.01, 0.5], [0.02, 0.5], [0.03, 0.5], [0.50, 0.50]],
        dtype=torch.float64,
    )
    omega = torch.tensor([[1.0], [2.0], [20.0], [100.0]], dtype=torch.float64)
    pointwise = {}
    _add_reference_free_regularizers(
        pointwise,
        torch.nn.Identity(),
        coords,
        {
            "coords": coords,
            "omega": omega,
            "u": torch.zeros_like(omega),
            "v": torch.zeros_like(omega),
            "p_x": torch.zeros_like(omega),
            "p_y": torch.zeros_like(omega),
        },
        {
            "domain_bounds": (0.0, 1.0, 0.0, 1.0),
            "near_wall_vorticity_l2": {
                "enabled": True,
                "band_width": 0.1,
                "quantile": 0.5,
            },
        },
    )
    penalty = pointwise["near_wall_vorticity_l2"].reshape(-1)
    assert penalty[0].item() == pytest.approx(0.0)
    assert penalty[1].item() == pytest.approx(0.0)
    assert penalty[2].item() > 0.0
    assert penalty[3].item() == pytest.approx(0.0)


def test_correction_bubble_losses_target_global_mean_not_local_variation():
    class AuxiliaryModel(torch.nn.Module):
        def __init__(self, raw, correction):
            super().__init__()
            self.raw = raw
            self.correction = correction

        def streamfunction_auxiliary(self, coords):
            return {
                "raw_psi": self.raw.to(coords),
                "scaled_correction": self.correction.to(coords),
            }

    coords = torch.tensor([[0.25, 0.25], [0.75, 0.75]], dtype=torch.float64)
    residuals = {
        "coords": coords,
        "u": torch.zeros((2, 1), dtype=torch.float64),
        "v": torch.zeros((2, 1), dtype=torch.float64),
        "omega": torch.zeros((2, 1), dtype=torch.float64),
        "p_x": torch.zeros((2, 1), dtype=torch.float64),
        "p_y": torch.zeros((2, 1), dtype=torch.float64),
    }
    cfg = {
        "correction_bubble": {
            "enabled": True,
            "abs_max_cap": 0.30,
        }
    }
    bubble = {}
    _add_reference_free_regularizers(
        bubble,
        AuxiliaryModel(
            torch.tensor([[2.0], [2.0]]),
            torch.tensor([[0.20], [0.20]]),
        ),
        coords,
        residuals,
        cfg,
    )
    assert bubble["raw_psi_mean_l2"].mean().item() > 0.0
    assert bubble["scaled_correction_mean_l2"].mean().item() > 0.0

    local = {}
    _add_reference_free_regularizers(
        local,
        AuxiliaryModel(
            torch.tensor([[2.0], [-2.0]]),
            torch.tensor([[0.20], [-0.20]]),
        ),
        coords,
        residuals,
        cfg,
    )
    assert local["raw_psi_mean_l2"].mean().item() == pytest.approx(0.0)
    assert local["scaled_correction_mean_l2"].mean().item() == pytest.approx(0.0)
    assert local["scaled_correction_abs_max_hinge"].mean().item() == pytest.approx(
        0.0
    )


def test_correction_bubble_abs_max_uses_hinge_cap():
    class AuxiliaryModel(torch.nn.Module):
        def streamfunction_auxiliary(self, coords):
            return {
                "raw_psi": torch.tensor([[1.0], [-1.0]], dtype=coords.dtype),
                "scaled_correction": torch.tensor(
                    [[0.20], [-0.50]], dtype=coords.dtype
                ),
            }

    coords = torch.tensor([[0.25, 0.25], [0.75, 0.75]], dtype=torch.float64)
    pointwise = {}
    zeros = torch.zeros((2, 1), dtype=torch.float64)
    _add_reference_free_regularizers(
        pointwise,
        AuxiliaryModel(),
        coords,
        {
            "coords": coords,
            "u": zeros,
            "v": zeros,
            "omega": zeros,
            "p_x": zeros,
            "p_y": zeros,
        },
        {"correction_bubble": {"enabled": True, "abs_max_cap": 0.30}},
    )
    assert pointwise["scaled_correction_abs_max_hinge"].mean().item() == pytest.approx(
        0.04
    )


def test_low_re_lid_shear_guards_detect_wrong_direction_and_exclude_corners():
    coords = torch.tensor(
        [
            [0.50, 0.95],
            [0.50, 0.05],
            [0.02, 0.95],
            [0.98, 0.05],
            [0.50, 0.50],
        ],
        dtype=torch.float64,
    )
    u = torch.tensor([[-0.4], [0.4], [-0.4], [0.4], [-0.4]], dtype=torch.float64)
    zeros = torch.zeros_like(u)
    pointwise = {}
    _add_reference_free_regularizers(
        pointwise,
        torch.nn.Identity(),
        coords,
        {
            "coords": coords,
            "u": u,
            "v": zeros,
            "omega": zeros,
            "p_x": zeros,
            "p_y": zeros,
        },
        {
            "domain_bounds": (0.0, 1.0, 0.0, 1.0),
            "lid_shear_direction": {
                "enabled": True,
                "band_width": 0.10,
                "corner_width": 0.08,
                "bottom_u_tolerance": 0.075,
            },
        },
    )
    top = pointwise["top_reverse_u"].reshape(-1)
    bottom = pointwise["bottom_positive_u"].reshape(-1)
    assert top[0].item() > 0.0
    assert bottom[1].item() > 0.0
    assert top[2].item() == pytest.approx(0.0)
    assert bottom[3].item() == pytest.approx(0.0)
    assert top[4].item() == pytest.approx(0.0)
    assert bottom[4].item() == pytest.approx(0.0)


def test_raw_residual_tail_thresholds_both_momentum_terms_and_emphasizes_core():
    coords = torch.tensor(
        [[0.50, 0.50], [0.50, 0.50], [0.01, 0.50]],
        dtype=torch.float64,
    )
    f_u = torch.tensor([[0.40], [1.00], [1.00]], dtype=torch.float64)
    f_v = torch.tensor([[0.20], [0.80], [0.80]], dtype=torch.float64)
    zeros = torch.zeros_like(f_u)
    pointwise = {}
    _add_reference_free_regularizers(
        pointwise,
        torch.nn.Identity(),
        coords,
        {
            "coords": coords,
            "u": zeros,
            "v": zeros,
            "omega": zeros,
            "p_x": zeros,
            "p_y": zeros,
            "f_u": f_u,
            "f_v": f_v,
        },
        {
            "domain_bounds": (0.0, 1.0, 0.0, 1.0),
            "raw_residual_tail": {
                "enabled": True,
                "threshold": 0.50,
                "core_margin": 0.08,
                "core_emphasis": 2.0,
            },
        },
    )
    tail = pointwise["raw_pde_tail"].reshape(-1)
    assert tail[0].item() == pytest.approx(0.0)
    assert tail[1].item() == pytest.approx(2.0 * (0.50**2 + 0.30**2))
    assert tail[2].item() == pytest.approx(0.50**2 + 0.30**2)


def test_reliable_near_wall_sampling_preserves_budget_and_avoids_corners(tmp_path):
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    for reynolds, fraction, band in [
        (100.0, 0.20, 0.12),
        (400.0, 0.40, 0.06),
        (1600.0, 0.50, 0.03),
    ]:
        config = _apply_re_aware_cavity_settings(base, reynolds)
        config = deep_update(
            config,
            {
                "device": "cpu",
                "model": {"hidden_layers": [8, 8]},
                "experiments": {
                    "root": str(tmp_path / f"re_{int(reynolds)}"),
                    "flat_layout": True,
                },
            },
        )
        trainer = VARATrainer(config, mode="vanilla")
        points = trainer._sample_interior_numpy(100)
        n_wall = trainer._near_wall_sample_count(100)
        guaranteed = points[-n_wall:]
        distance = np.min(
            np.column_stack(
                [
                    guaranteed[:, 0],
                    1.0 - guaranteed[:, 0],
                    guaranteed[:, 1],
                    1.0 - guaranteed[:, 1],
                ]
            ),
            axis=1,
        )
        assert points.shape == (100, 2)
        assert n_wall == round(100 * fraction)
        assert np.all(distance < band)
        assert np.all(guaranteed[:, 0] > 0.0)
        assert np.all(guaranteed[:, 0] < 1.0)
        assert np.all(guaranteed[:, 1] > 0.0)
        assert np.all(guaranteed[:, 1] < 1.0)
        assert np.all(
            np.maximum(
                np.minimum(guaranteed[:, 0], 1.0 - guaranteed[:, 0]),
                np.minimum(guaranteed[:, 1], 1.0 - guaranteed[:, 1]),
            )
            >= 0.02 * band
        )


def test_vanilla_and_vara_share_near_wall_sampling_budget(tmp_path):
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    config = _apply_re_aware_cavity_settings(base, 400.0)
    common = {
        "device": "cpu",
        "model": {"hidden_layers": [8, 8]},
        "training": {
            "n_collocation": 40,
            "n_boundary": 4,
            "n_data": 0,
            "collocation_curriculum": {"enabled": False},
        },
        "continuation_replay": {"enabled": False},
    }
    vanilla = VARATrainer(
        deep_update(
            config,
            {
                **common,
                "experiments": {
                    "root": str(tmp_path / "vanilla"),
                    "flat_layout": True,
                },
            },
        ),
        mode="vanilla",
    )
    vara = VARAV2Trainer(
        deep_update(
            config,
            {
                **common,
                "experiments": {
                    "root": str(tmp_path / "vara"),
                    "flat_layout": True,
                },
            },
        )
    )
    assert vanilla._near_wall_sample_count(40) == 16
    assert vara._near_wall_sample_count(40) == 16
    assert vanilla.initial_batch()["xy_f"].shape[0] == 40
    assert vara.initial_batch()["xy_f"].shape[0] == 40
    assert (
        vanilla._active_loss_config()["correction_bubble"]
        == vara._active_loss_config()["correction_bubble"]
    )
    assert (
        vanilla._active_loss_config()["lid_shear_direction"]
        == vara._active_loss_config()["lid_shear_direction"]
    )
    assert (
        vanilla._active_loss_config()["raw_residual_tail"]
        == vara._active_loss_config()["raw_residual_tail"]
    )
    for name in (
        "raw_psi_mean_l2",
        "scaled_correction_mean_l2",
        "scaled_correction_abs_max_hinge",
        "top_reverse_u",
        "bottom_positive_u",
        "raw_pde_tail",
    ):
        assert (
            vanilla.config["training"]["weights"][name]
            == vara.config["training"]["weights"][name]
        )


def test_hard_boundary_correction_boost_preserves_walls_and_near_wall_authority():
    class UnitPsi(torch.nn.Module):
        def forward(self, coords):
            return torch.cat(
                [torch.ones_like(coords[:, 0:1]), torch.zeros_like(coords[:, 0:1])],
                dim=1,
            )

    plain = HardBoundaryStreamfunctionPressureWrapper(
        UnitPsi(),
        (0.0, 1.0, 0.0, 1.0),
        correction_scale=1.0,
        correction_wall_boost=0.0,
    )
    boosted = HardBoundaryStreamfunctionPressureWrapper(
        UnitPsi(),
        (0.0, 1.0, 0.0, 1.0),
        correction_scale=1.0,
        correction_wall_boost=8.0,
    )
    walls = torch.tensor(
        [[0.0, 0.4], [1.0, 0.4], [0.5, 0.0], [0.5, 1.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(
        plain(walls)[:, :2],
        boosted(walls)[:, :2],
        atol=1e-12,
    )
    near_wall = torch.tensor([[0.05, 0.5]], dtype=torch.float64, requires_grad=True)
    plain_correction = plain.streamfunction_auxiliary(near_wall)["scaled_correction"]
    boosted_correction = boosted.streamfunction_auxiliary(near_wall)["scaled_correction"]
    assert boosted_correction.item() > plain_correction.item()
    plain_gradient = torch.autograd.grad(
        plain_correction,
        near_wall,
        grad_outputs=torch.ones_like(plain_correction),
    )[0]
    boosted_gradient = torch.autograd.grad(
        boosted_correction,
        near_wall,
        grad_outputs=torch.ones_like(boosted_correction),
    )[0]
    assert abs(boosted_gradient[0, 0].item()) > abs(plain_gradient[0, 0].item())


def test_reliable_config_guard_rejects_legacy_formulation_and_budget_mismatch():
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    resolved = _apply_re_aware_cavity_settings(base, 100.0)
    _validate_reliable_config(resolved, "reliable", require_materialized=True)
    for preset, expected in [
        ("fast_screen", 1200),
        ("diagnostic", 2000),
        ("reliable", 4000),
    ]:
        preset_config = deep_update(
            base,
            load_config(f"configs/vara_v2/presets/{preset}.yaml"),
        )
        preset_config = _apply_re_aware_cavity_settings(preset_config, 100.0)
        _validate_reliable_config(
            preset_config,
            preset,
            require_materialized=True,
        )
        assert preset_config["controller_v2"]["total_steps"] == expected
    with pytest.raises(ValueError, match="physics_formulation"):
        _validate_reliable_config(
            deep_update(resolved, {"model": {"physics_formulation": "cavity_hard_boundary"}}),
            "reliable",
            require_materialized=True,
        )
    with pytest.raises(ValueError, match="total_steps"):
        _validate_reliable_config(
            deep_update(resolved, {"optimizer": {"scheduler": {"total_steps": 2000}}}),
            "reliable",
            require_materialized=True,
        )


def test_stabilizer_ablation_keeps_formulation_and_budget_but_disables_losses():
    config = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    ablated = _without_cavity_stabilizers(config)
    assert (
        ablated["model"]["physics_formulation"]
        == config["model"]["physics_formulation"]
    )
    assert ablated["controller_v2"]["total_steps"] == config["controller_v2"]["total_steps"]
    assert ablated["training"]["n_collocation"] == config["training"]["n_collocation"]
    assert ablated["training"]["residual_loss_mode"] == "mse"
    assert ablated["losses"]["near_wall_momentum"]["enabled"] is False
    assert ablated["training"]["weights"]["speed_cap"] == 0.0


def test_vara_v2_inherits_reliable_cavity_stabilizers_identically(tmp_path):
    config = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    config = deep_update(
        config,
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "training": {"n_collocation": 8, "n_boundary": 4, "n_data": 0},
            "experiments": {"root": str(tmp_path), "flat_layout": True},
            "continuation_replay": {"enabled": False},
        },
    )
    trainer = VARAV2Trainer(config)
    active = trainer._active_loss_config()
    assert trainer.model.physics_formulation == "hard_boundary_streamfunction_pressure"
    assert active["speed_cap"]["enabled"] is True
    assert active["near_wall_momentum"]["enabled"] is True
    assert active["near_wall_momentum"]["weight"] == pytest.approx(1.0)
    assert trainer._residual_loss_settings()[0] == "pseudo_huber"
    assert trainer._compute_boundary_training_loss(
        config["training"]["weights"]
    ) is False


def test_checkpoint_restores_live_cavity_curriculum_state(tmp_path):
    config = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    model = build_mlp_from_config(config, (0.0, 1.0, 0.0, 1.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.corner_width = 0.10
    model.lid_vertical_power = 2
    model.correction_scale = 32.0
    path = tmp_path / "cavity.pt"
    save_checkpoint(path, model, optimizer, config, {}, 10, 1)
    model.corner_width = 0.06
    model.lid_vertical_power = 3
    model.correction_scale = 64.0
    load_checkpoint(path, model)
    assert model.corner_width == pytest.approx(0.10)
    assert model.lid_vertical_power == 2
    assert model.correction_scale == pytest.approx(32.0)


def test_reliable_checkpoint_eligibility_requires_final_window_and_stage(tmp_path):
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    config = _apply_re_aware_cavity_settings(base, 100.0)
    config = deep_update(
        config,
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "experiments": {"root": str(tmp_path), "flat_layout": True},
        },
    )
    trainer = VARATrainer(config, mode="vanilla")
    assert trainer._checkpoint_min_restore_step() == 3000
    trainer.global_step = 600
    trainer.model.corner_width = 0.10
    trainer.model.lid_vertical_power = 2
    trainer.model.correction_scale = 18.0
    trainer.maybe_checkpoint(0, {"pde_residual_mean": 0.01})
    assert not (trainer.checkpoint_dir / "best.pt").exists()

    trainer.global_step = 3200
    trainer.model.corner_width = 0.09
    trainer.model.lid_vertical_power = 2
    trainer.model.correction_scale = 22.0
    trainer.maybe_checkpoint(
        1,
        {
            "pde_residual_mean": 1.0,
            "momentum_residual_mean": 1.0,
            "core_pde_residual_mean": 1.0,
            "near_wall_pde_residual_mean": 1.0,
            "near_wall_momentum_v_mean": 1.0,
            "omega_abs_95p": 1.0,
            "speed_pred_max": 1.0,
        },
    )
    assert (trainer.checkpoint_dir / "best.pt").exists()

    diagnostic = deep_update(
        base,
        load_config("configs/vara_v2/presets/diagnostic.yaml"),
    )
    diagnostic = _apply_re_aware_cavity_settings(diagnostic, 100.0)
    diagnostic = deep_update(
        diagnostic,
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "experiments": {
                "root": str(tmp_path / "diagnostic"),
                "flat_layout": True,
            },
        },
    )
    assert VARATrainer(
        diagnostic,
        mode="vanilla",
    )._checkpoint_min_restore_step() == 1500


def test_checkpoint_speed_score_is_hinged_and_core_weighted(tmp_path):
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    config = deep_update(
        _apply_re_aware_cavity_settings(base, 100.0),
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "experiments": {"root": str(tmp_path), "flat_layout": True},
        },
    )
    trainer = VARATrainer(config, mode="vanilla")
    metrics = {
        "pde_residual_mean": 1.0,
        "momentum_residual_mean": 1.0,
        "core_pde_residual_mean": 1.0,
        "continuity_residual_mean": 1.0,
        "boundary_condition_error": 0.0,
        "near_wall_pde_residual_mean": 1.0,
        "near_wall_momentum_v_mean": 1.0,
        "omega_abs_95p": 1.0,
        "speed_pred_max": 1.0,
    }
    below_gate = trainer._checkpoint_score(metrics)
    assert trainer._checkpoint_score({**metrics, "speed_pred_max": 2.0}) == pytest.approx(
        below_gate
    )
    assert trainer._checkpoint_score({**metrics, "speed_pred_max": 3.2}) > below_gate
    assert trainer._checkpoint_score(
        {**metrics, "core_pde_residual_mean": 2.0}
    ) == pytest.approx(below_gate + 1.5)
    assert trainer._checkpoint_score(
        {**metrics, "continuity_residual_mean": 2.0}
    ) == pytest.approx(below_gate + 1.25)
    assert trainer._checkpoint_score(
        {**metrics, "boundary_condition_error": 1.0}
    ) == pytest.approx(below_gate + 0.75)
    assert trainer._checkpoint_score(
        {**metrics, "u_rel_l2": 1000.0, "velocity_full_rel_l2": 1000.0}
    ) == pytest.approx(below_gate)


def test_checkpoint_score_does_not_prioritize_direction_losses(tmp_path):
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    config = deep_update(
        _apply_re_aware_cavity_settings(base, 100.0),
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "experiments": {"root": str(tmp_path), "flat_layout": True},
        },
    )
    trainer = VARATrainer(config, mode="vanilla")
    metrics = {
        "pde_residual_mean": 1.0,
        "momentum_residual_mean": 1.0,
        "core_pde_residual_mean": 1.0,
        "boundary_condition_error": 0.0,
        "near_wall_pde_residual_mean": 1.0,
        "near_wall_momentum_v_mean": 1.0,
        "omega_abs_95p": 1.0,
        "speed_pred_max": 1.0,
    }
    trainer.last_losses = {"top_reverse_u": 0.0, "bottom_positive_u": 0.0}
    baseline = trainer._checkpoint_score(metrics)
    trainer.last_losses = {"top_reverse_u": 0.2, "bottom_positive_u": 0.4}
    assert trainer._checkpoint_score(metrics) == pytest.approx(baseline)


def test_checkpoint_vortex_tiebreaker_is_low_re_only(tmp_path):
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    base["cavity_base_formulation"] = "hard_boundary_streamfunction_pressure"
    common = {
        "device": "cpu",
        "model": {"hidden_layers": [8, 8]},
    }
    metrics = {
        "pde_residual_mean": 1.0,
        "momentum_residual_mean": 1.0,
        "core_pde_residual_mean": 1.0,
        "boundary_condition_error": 0.0,
        "near_wall_pde_residual_mean": 1.0,
        "near_wall_momentum_v_mean": 1.0,
        "omega_abs_95p": 1.0,
        "speed_pred_max": 1.0,
        "detected_vortex_count": 4,
        "secondary_vortex_count": 3,
    }
    low = VARATrainer(
        deep_update(
            _apply_re_aware_cavity_settings(base, 100.0),
            {
                **common,
                "experiments": {
                    "root": str(tmp_path / "low"),
                    "flat_layout": True,
                },
            },
        ),
        mode="vanilla",
    )
    high = VARATrainer(
        deep_update(
            _apply_re_aware_cavity_settings(base, 1600.0),
            {
                **common,
                "experiments": {
                    "root": str(tmp_path / "high"),
                    "flat_layout": True,
                },
            },
        ),
        mode="vanilla",
    )
    clean = {
        **metrics,
        "detected_vortex_count": 2,
        "secondary_vortex_count": 1,
    }
    assert low._checkpoint_score(metrics) == pytest.approx(
        low._checkpoint_score(clean) + 0.20
    )
    assert high._checkpoint_score(metrics) == pytest.approx(
        high._checkpoint_score(clean)
    )


def test_reliable_final_state_guard_rejects_early_curriculum_values(tmp_path):
    base = _load_base_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    base["cavity_base_formulation"] = "hard_boundary_streamfunction_pressure"
    config = deep_update(
        _apply_re_aware_cavity_settings(base, 100.0),
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "experiments": {"root": str(tmp_path), "flat_layout": True},
        },
    )
    trainer = VARATrainer(config, mode="vanilla")
    trainer.model.corner_width = 0.10
    trainer.model.correction_scale = 18.0
    with pytest.raises(ValueError, match="final cavity curriculum stage"):
        trainer._validate_final_cavity_state()


def test_missing_full_field_reference_does_not_invalidate_reliable_stage():
    metrics = {
        "has_reference": False,
        "pde_residual_mean": 0.1,
        "continuity_residual_mean": 1e-8,
        "momentum_residual_mean": 0.1,
        "boundary_condition_error": 0.0,
        "speed_pred_max": 1.0,
        "streamfunction_consistency_rmse": 1e-4,
        "lid_cavity_primary_center_error": 0.05,
        "lid_cavity_topology_score": 0.05,
        "velocity_full_rel_l2": float("nan"),
        "lid_cavity_topology_aligned": 1.0,
        "primary_streamfunction_abs": 0.02,
        "speed_pred_mean": 0.2,
        "detected_vortex_count": 1,
        "primary_vortex_center_x": 0.6,
        "primary_vortex_center_y": 0.7,
        "near_wall_pde_residual_mean": 0.2,
        "near_wall_momentum_v_mean": 0.2,
        "core_pde_residual_mean": 0.1,
    }
    config = load_config("configs/vara_v2/lid_cavity_continuation_reliable.yaml")
    validity = _continuation_validity(metrics, config)
    assert validity["continuation_stage_valid"]


def test_balanced_lid_cavity_boundary_sampler_reports_side_fractions():
    sampler = BoundarySampler((0.0, 1.0, 0.0, 1.0), torch.device("cpu"), seed=7)
    points = sampler.sample_lid_cavity_numpy(
        400,
        mode="balanced",
    )
    fractions = boundary_side_fractions(points, (0.0, 1.0, 0.0, 1.0))
    assert sum(
        fractions[name]
        for name in [
            "boundary_fraction_left",
            "boundary_fraction_right",
            "boundary_fraction_bottom",
            "boundary_fraction_top",
        ]
    ) == pytest.approx(1.0)
    for name in [
        "boundary_fraction_left",
        "boundary_fraction_right",
        "boundary_fraction_bottom",
        "boundary_fraction_top",
    ]:
        assert 0.15 <= fractions[name] <= 0.35


def test_re100_sanity_report_fails_broken_physical_behavior(tmp_path):
    result_root = tmp_path / "broken_re100"
    summary = result_root / "summary"
    method_dir = result_root / "seed_0" / "re_0100" / "vanilla"
    summary.mkdir(parents=True)
    (method_dir / "logs").mkdir(parents=True)
    (method_dir / "checkpoints").mkdir(parents=True)
    (method_dir / "figures").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "seed": 0,
                "reynolds": 100.0,
                "method": "vanilla",
                "method_dir": str(method_dir),
                "continuation_stage_valid": False,
                "continuation_invalid_reasons": "known broken fixture",
                "boundary_condition_error": 0.4,
                "pde_residual_mean": 0.9,
                "continuity_residual_mean": 0.3,
                "momentum_residual_mean": 0.7,
                "speed_pred_mean": 0.01,
                "speed_pred_max": 0.02,
                "primary_streamfunction_abs": 0.001,
                "detected_vortex_count": 0,
                "lid_cavity_topology_aligned": 0.0,
                "lid_cavity_primary_center_error": 0.5,
                "streamfunction_consistency_rmse": 0.5,
                "unweighted_physics_validation_loss": 1.3,
                "has_reference": False,
            }
        ]
    ).to_csv(summary / "continuation_results_long.csv", index=False)
    pd.DataFrame(
        [{"momentum_u": 0.7, "momentum_v": 0.7, "continuity": 0.3, "bc": 0.4}]
    ).to_csv(method_dir / "logs" / "losses.csv", index=False)
    report = build_report(result_root, "vanilla", 0)
    assert not report["passed"]
    assert any("continuation_stage_valid=false" in item for item in report["failures"])
    assert any("detected_vortex_count" in item for item in report["failures"])
    assert "raw_momentum_u" in report["methods"]["vanilla"]["raw_training_losses_last"]


def test_harmonic_cavity_lifting_reduces_couette_interior_bias():
    class ZeroBase(torch.nn.Module):
        def forward(self, coords):
            return torch.zeros((coords.shape[0], 3), dtype=coords.dtype, device=coords.device)

    linear = CavityHardBoundaryWrapper(
        ZeroBase(),
        (0.0, 1.0, 0.0, 1.0),
        lid_velocity=1.0,
        corner_width=0.02,
        lid_lifting="linear",
    )
    harmonic = CavityHardBoundaryWrapper(
        ZeroBase(),
        (0.0, 1.0, 0.0, 1.0),
        lid_velocity=1.0,
        corner_width=0.02,
        lid_lifting="harmonic",
    )
    center = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    lid = torch.tensor([[0.5, 1.0]], dtype=torch.float64)
    assert harmonic(center)[0, 0].item() < linear(center)[0, 0].item()
    assert harmonic(lid)[0, 0].item() == pytest.approx(1.0)


def test_divergence_free_cavity_lifting_satisfies_walls_and_continuity():
    class ZeroBase(torch.nn.Module):
        def forward(self, coords):
            return torch.zeros(
                (coords.shape[0], 3),
                dtype=coords.dtype,
                device=coords.device,
            )

    model = CavityHardBoundaryWrapper(
        ZeroBase(),
        (0.0, 1.0, 0.0, 1.0),
        lid_velocity=1.0,
        corner_width=0.05,
        lid_lifting="divergence_free",
    )
    interior = torch.tensor(
        [[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    velocity = model(interior)[:, :2]
    du_dx = torch.autograd.grad(
        velocity[:, 0].sum(),
        interior,
        create_graph=True,
    )[0][:, 0]
    dv_dy = torch.autograd.grad(
        velocity[:, 1].sum(),
        interior,
        create_graph=True,
    )[0][:, 1]
    assert torch.max(torch.abs(du_dx + dv_dy)).item() < 1e-10

    walls = torch.tensor(
        [[0.0, 0.4], [1.0, 0.4], [0.5, 0.0], [0.5, 1.0]],
        dtype=torch.float64,
    )
    wall_velocity = model(walls)[:, :2]
    assert torch.allclose(
        wall_velocity[:3],
        torch.zeros_like(wall_velocity[:3]),
        atol=1e-12,
    )
    assert wall_velocity[3, 0].item() == pytest.approx(1.0)
    assert wall_velocity[3, 1].item() == pytest.approx(0.0)


def test_hard_boundary_streamfunction_pressure_is_divergence_free_and_residual_safe():
    class SmoothBase(torch.nn.Module):
        def forward(self, coords):
            x = coords[:, 0:1]
            y = coords[:, 1:2]
            raw_psi = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
            raw_p = x + y
            return torch.cat([raw_psi, raw_p], dim=1)

    model = HardBoundaryStreamfunctionPressureWrapper(
        SmoothBase(),
        (0.0, 1.0, 0.0, 1.0),
        lid_velocity=1.0,
        corner_width=0.05,
    )
    interior = torch.tensor(
        [[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    velocity = model(interior)[:, :2]
    du_dx = torch.autograd.grad(
        velocity[:, 0].sum(),
        interior,
        create_graph=True,
    )[0][:, 0]
    dv_dy = torch.autograd.grad(
        velocity[:, 1].sum(),
        interior,
        create_graph=True,
    )[0][:, 1]
    assert torch.max(torch.abs(du_dx + dv_dy)).item() < 1e-8

    walls = torch.tensor(
        [
            [0.0, 0.4],
            [1.0, 0.4],
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    wall_velocity = model(walls)[:, :2]
    assert torch.max(torch.abs(wall_velocity[:3])).item() < 1e-8
    assert wall_velocity[3, 0].item() == pytest.approx(1.0, abs=1e-8)
    assert wall_velocity[3, 1].item() == pytest.approx(0.0, abs=1e-8)
    assert torch.max(torch.abs(wall_velocity[4:])).item() < 1e-8

    with torch.no_grad():
        prediction = model(interior.detach())
    assert prediction.shape == (3, 3)
    assert torch.isfinite(prediction).all()

    residuals = navier_stokes_residuals(model, interior.detach(), nu=0.01, steady=True)
    assert torch.max(torch.abs(residuals["f_c"])).item() < 1e-8
    assert torch.isfinite(residuals["f_u"]).all()
    assert torch.isfinite(residuals["f_v"]).all()


def test_regularized_cavity_target_matches_hard_boundary_corners():
    benchmark = LidDrivenCavityQualitative(lid_corner_regularization_width=0.1)
    points = torch.tensor(
        [[0.0, 1.0], [0.05, 1.0], [0.5, 1.0], [0.95, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    target = benchmark.exact_torch(points)["u"].reshape(-1)
    assert target[0].item() == pytest.approx(0.0)
    assert 0.0 < target[1].item() < 1.0
    assert target[2].item() == pytest.approx(1.0)
    assert 0.0 < target[3].item() < 1.0
    assert target[4].item() == pytest.approx(0.0)


def test_streamfunction_wrapper_is_divergence_free_and_evaluable_without_grad():
    class PolynomialPsi(torch.nn.Module):
        def forward(self, coords):
            x = coords[:, 0:1]
            y = coords[:, 1:2]
            psi = x * x * y + x * y * y
            pressure = 0.0 * x
            return torch.cat([psi, pressure], dim=1)

    model = StreamfunctionPressureWrapper(PolynomialPsi())
    coords = torch.tensor(
        [[0.2, 0.3], [0.6, 0.4], [0.8, 0.9]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        prediction = model(coords)
    assert prediction.shape == (3, 3)
    residuals = navier_stokes_residuals(model, coords, nu=0.01, steady=True)
    assert torch.max(torch.abs(residuals["f_c"])).item() < 1e-10


def test_streamfunction_reconstruction_is_consistent_for_divergence_free_field():
    x = np.linspace(0.0, 1.0, 101)
    y = np.linspace(0.0, 1.0, 101)
    X, Y = np.meshgrid(x, y)
    U = X * X + 2.0 * X * Y
    V = -(2.0 * X * Y + Y * Y)
    psi, consistency = reconstruct_streamfunction(X, Y, U, V)
    exact = X * X * Y + X * Y * Y
    exact -= exact[0, 0]
    assert consistency < 1e-4
    assert np.sqrt(np.mean((psi - exact) ** 2)) < 1e-4


def test_closed_cavity_reconstruction_detects_physical_vortex():
    x = np.linspace(0.0, 1.0, 101)
    y = np.linspace(0.0, 1.0, 101)
    X, Y = np.meshgrid(x, y)
    exact = -X * (1.0 - X) * Y * (1.0 - Y)
    U = -X * (1.0 - X) * (1.0 - 2.0 * Y)
    V = (1.0 - 2.0 * X) * Y * (1.0 - Y)
    psi, consistency = reconstruct_streamfunction(
        X,
        Y,
        U,
        V,
        closed_boundary=True,
    )
    vortices = detect_vortices(X, Y, psi)
    assert consistency < 1e-4
    assert np.sqrt(np.mean((psi - exact) ** 2)) < 1e-4
    assert vortices
    assert vortices[0]["x"] == pytest.approx(0.5, abs=0.02)
    assert vortices[0]["y"] == pytest.approx(0.5, abs=0.02)


def test_lid_cavity_topology_metric_accepts_expected_primary_vortex():
    x = np.linspace(0.0, 1.0, 80)
    y = np.linspace(0.0, 1.0, 80)
    X, Y = np.meshgrid(x, y)
    cx, cy = 0.6172, 0.7344
    sigma = 0.22
    envelope = X * (1.0 - X) * Y * (1.0 - Y)
    gaussian = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / sigma**2)
    exact = -envelope * gaussian
    dy = y[1] - y[0]
    dx = x[1] - x[0]
    u = np.gradient(exact, dy, axis=0)
    v = -np.gradient(exact, dx, axis=1)
    metrics = lid_cavity_topology_metrics(X, Y, u, v, reynolds=100.0)
    assert metrics["lid_cavity_topology_aligned"] == 1.0
    assert metrics["lid_cavity_primary_center_error"] < 0.05


def test_cavity_residual_metrics_can_exclude_singular_boundary_points():
    benchmark = LidDrivenCavityQualitative(reynolds=100.0)

    class BoundarySingularModel(torch.nn.Module):
        def forward(self, coords):
            x = coords[:, 0:1]
            y = coords[:, 1:2]
            u = x * (1.0 - x) * y * (1.0 - y)
            v = -u
            p = x + y
            return torch.cat([u, v, p], dim=1)

    _, _, coords = benchmark.grid(8, 8)
    all_points = evaluate_on_grid(
        BoundarySingularModel(),
        benchmark,
        coords,
        torch.device("cpu"),
        residual_interior_only=False,
    )
    interior = evaluate_on_grid(
        BoundarySingularModel(),
        benchmark,
        coords,
        torch.device("cpu"),
        residual_interior_only=True,
    )
    assert all_points["num_residual_eval_points"] == 64
    assert interior["num_residual_eval_points"] == 36
    assert interior["residual_interior_only"]


def test_taylor_green_exact_field_satisfies_transient_navier_stokes():
    benchmark = TaylorGreenVortex(reynolds=100.0)

    class ExactModel(torch.nn.Module):
        def forward(self, coords):
            exact = benchmark.exact_torch(coords)
            return torch.cat([exact["u"], exact["v"], exact["p"]], dim=1)

    coords = torch.tensor(
        [[0.2, 0.4, 0.1], [1.1, 2.3, 0.5], [4.0, 5.0, 0.9]],
        dtype=torch.float64,
    )
    residuals = navier_stokes_residuals(ExactModel(), coords, benchmark.nu, steady=False)
    assert torch.max(torch.abs(residuals["f_u"])).item() < 1e-9
    assert torch.max(torch.abs(residuals["f_v"])).item() < 1e-9
    assert torch.max(torch.abs(residuals["f_c"])).item() < 1e-9


def test_v2_smoke_has_exact_step_budget_and_controller_accounting(tmp_path):
    config = load_config("configs/vara_v2/lid_driven_cavity.yaml")
    config = deep_update(
        config,
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "training": {"n_collocation": 24, "n_boundary": 16, "log_every": 1},
            "validation": {"nx": 6, "ny": 6},
            "test": {"nx": 6, "ny": 6},
            "patches": {"nx_patches": 2, "ny_patches": 2},
            "controller_v2": {
                "total_steps": 4,
                "warmup_steps": 1,
                "control_blocks": 1,
                "block_steps": 3,
                "probe_steps": 1,
                "gradient_probe_interior": 12,
                "gradient_probe_boundary": 8,
            },
            "experiments": {"root": str(tmp_path)},
        },
    )
    metrics = VARAV2Trainer(config).run()
    assert metrics["applied_optimizer_steps"] == 4
    assert metrics["optimizer_steps"] == 8
    assert metrics["probe_optimizer_steps"] == 5
    assert metrics["controller_gradient_evaluations"] > 0
    assert metrics["controller_gradient_point_evaluations"] > 0


def test_v2_neutral_path_matches_vanilla_exactly(tmp_path):
    config = load_config("configs/vara_v2/lid_driven_cavity.yaml")
    config = deep_update(
        config,
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "training": {
                "adaptive_cycles": 2,
                "epochs_per_cycle": 2,
                "n_collocation": 24,
                "n_boundary": 8,
                "n_data": 0,
                "log_every": 8,
            },
            "validation": {"nx": 6, "ny": 6},
            "test": {"nx": 6, "ny": 6},
            "patches": {"nx_patches": 2, "ny_patches": 2},
            "controller_v2": {
                "total_steps": 4,
                "warmup_steps": 2,
                "control_blocks": 1,
                "block_steps": 2,
                "probe_steps": 1,
                "max_candidates": 0,
                "gradient_probe_interior": 8,
                "gradient_probe_boundary": 4,
            },
            "checkpoint": {"restore_best_before_final": False},
        },
    )
    vanilla_config = deepcopy(config)
    vanilla_config["experiments"] = {
        "root": str(tmp_path / "vanilla"),
        "flat_layout": True,
    }
    v2_config = deepcopy(config)
    v2_config["experiments"] = {
        "root": str(tmp_path / "vara_v2"),
        "flat_layout": True,
    }
    vanilla = VARATrainer(vanilla_config, mode="vanilla_pinn")
    vara_v2 = VARAV2Trainer(v2_config)
    vanilla.run()
    vara_v2.run()
    for vanilla_parameter, v2_parameter in zip(
        vanilla.model.parameters(),
        vara_v2.model.parameters(),
    ):
        assert torch.equal(vanilla_parameter, v2_parameter)


def test_publication_runner_preserves_v2_schedule_and_ablation_overrides(tmp_path):
    from scripts.run_publication_suite_v2 import _case_config, _make_trainer

    config = load_config("configs/vara_v2/lid_driven_cavity.yaml")
    config = deep_update(config, load_config("configs/vara_v2/controller.yaml"))
    config = deep_update(
        config,
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "controller_v2": {
                "total_steps": 4,
                "warmup_steps": 1,
                "control_blocks": 1,
                "block_steps": 3,
                "probe_steps": 1,
                "rollback_enabled": False,
            },
            "experiments": {"root": str(tmp_path)},
        },
    )
    trainer, method = _make_trainer("v2_no_rollback", config)
    assert method == "v2_no_rollback"
    assert trainer.config["controller_v2"]["total_steps"] == 4
    assert trainer.config["controller_v2"]["control_blocks"] == 1
    assert not trainer.rollback_enabled

    re1600 = _case_config("lid_cavity_re1600", quick=True)
    assert re1600["benchmark_params"]["reference"] == "none"
    assert re1600["benchmark_params"]["full_field_reference_path"]


def test_continuation_replay_is_deterministic_and_decays(tmp_path):
    base = load_config("configs/vara_v2/lid_driven_cavity.yaml")
    base = deep_update(
        base,
        {
            "device": "cpu",
            "model": {"hidden_layers": [8, 8]},
            "training": {"n_collocation": 16, "n_boundary": 8, "log_every": 1},
            "validation": {"nx": 4, "ny": 4},
            "test": {"nx": 4, "ny": 4},
            "patches": {"nx_patches": 2, "ny_patches": 2},
            "controller_v2": {
                "total_steps": 4,
                "warmup_steps": 1,
                "control_blocks": 1,
                "block_steps": 3,
                "probe_steps": 1,
                "gradient_probe_interior": 8,
                "gradient_probe_boundary": 4,
            },
            "continuation_anchor": {"enabled": False},
            "continuation_replay": {
                "enabled": True,
                "n_points": 12,
                "active_fraction": 0.5,
                "initial_weight": 0.5,
                "seed_offset": 123,
            },
        },
    )
    source = build_mlp_from_config(base, (0.0, 1.0, 0.0, 1.0))
    optimizer = torch.optim.Adam(source.parameters(), lr=1e-3)
    checkpoint = tmp_path / "previous_re.pt"
    save_checkpoint(checkpoint, source, optimizer, base, {}, 4, -1)
    base["warm_start_checkpoint"] = str(checkpoint)
    base["warm_start"] = {"load_optimizer": False}

    first_config = deepcopy(base)
    first_config["experiments"] = {"root": str(tmp_path / "first")}
    second_config = deepcopy(base)
    second_config["experiments"] = {"root": str(tmp_path / "second")}
    first = VARAV2Trainer(first_config)
    second = VARAV2Trainer(second_config)
    assert torch.equal(first.continuation_replay_points, second.continuation_replay_points)
    assert torch.equal(first.continuation_replay_targets, second.continuation_replay_targets)
    loss, weight = first.continuation_replay_loss()
    assert weight == pytest.approx(0.5)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    first.global_step = 2
    expired_loss, expired_weight = first.continuation_replay_loss()
    assert expired_weight == 0.0
    assert expired_loss.item() == 0.0


def test_statistical_helpers_report_paired_improvement():
    baseline = np.array([2.0, 2.1, 1.9, 2.2, 2.0])
    method = np.array([1.8, 1.9, 1.7, 2.0, 1.8])
    bootstrap = paired_bootstrap_improvement(baseline, method, samples=500, seed=3)
    wilcoxon = wilcoxon_signed_rank(baseline, method)
    adjusted = holm_adjust([wilcoxon["p_value"], 0.04])
    assert bootstrap["mean_improvement_percent"] > 0.0
    assert bootstrap["ci_low"] > 0.0
    assert 0.0 <= wilcoxon["p_value"] <= 1.0
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_csv_logger_rewrites_header_when_schema_expands(tmp_path):
    path = tmp_path / "dynamic.csv"
    logger = CSVLogger(path)
    logger.log({"step": 0, "accepted": False})
    logger.log({"step": 1, "accepted": True, "reward_ratio": 0.75})
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["step", "accepted", "reward_ratio"]
    assert np.isnan(frame.loc[0, "reward_ratio"])
    assert frame.loc[1, "reward_ratio"] == pytest.approx(0.75)
