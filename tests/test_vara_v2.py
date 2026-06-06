from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from src.controllers import V2ControllerConfig, VARAV2Controller
from src.evaluation.statistical_tests import (
    holm_adjust,
    paired_bootstrap_improvement,
    wilcoxon_signed_rank,
)
from src.models import (
    CavityHardBoundaryWrapper,
    StreamfunctionPressureWrapper,
    build_mlp_from_config,
    parameter_matched_width,
)
from src.physics.navier_stokes import navier_stokes_residuals
from src.physics.taylor_green import TaylorGreenVortex
from src.training.checkpointing import save_checkpoint
from src.training.vara_v2_trainer import VARAV2Trainer
from src.utils.config import deep_update, load_config
from src.utils.logging import CSVLogger


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


def test_v2_rejects_reference_or_test_signal_names():
    controller = VARAV2Controller(V2ControllerConfig(num_patches=4))
    for name in ["velocity_full_rel_l2", "ghia_profile_score", "cfd_reference_error", "test_rmse"]:
        with pytest.raises(ValueError):
            controller.assert_reference_free([name])


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
