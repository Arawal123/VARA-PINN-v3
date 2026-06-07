from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from src.controllers import V2ControllerConfig, VARAV2Controller
from src.evaluation.metrics import evaluate_on_grid
from src.evaluation.statistical_tests import (
    holm_adjust,
    paired_bootstrap_improvement,
    wilcoxon_signed_rank,
)
from src.losses.base_losses import compute_global_losses
from src.losses.local_losses import compute_budgeted_patch_losses
from src.models import (
    CavityHardBoundaryWrapper,
    StreamfunctionPressureWrapper,
    build_mlp_from_config,
    parameter_matched_width,
)
from src.physics.navier_stokes import navier_stokes_residuals
from src.physics.rectangular_benchmarks import LidDrivenCavityQualitative
from src.physics.taylor_green import TaylorGreenVortex
from src.training.checkpointing import save_checkpoint
from src.training.vara_v2_trainer import VARAV2Trainer
from src.utils.config import deep_update, load_config
from src.utils.logging import CSVLogger
from src.visualization.streamlines import reconstruct_streamfunction


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
    for name in ["velocity_full_rel_l2", "ghia_profile_score", "cfd_reference_error", "test_rmse"]:
        with pytest.raises(ValueError):
            controller.assert_reference_free([name])


def test_v2_publication_config_hides_reference_metrics_from_legacy_controller(tmp_path):
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
    assert metrics["optimizer_steps"] == 4
    assert metrics["probe_optimizer_steps"] == 1
    assert metrics["controller_gradient_evaluations"] > 0
    assert metrics["controller_gradient_point_evaluations"] > 0


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
