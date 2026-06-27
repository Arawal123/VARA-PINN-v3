from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.mlp import build_mlp_from_config
from src.models.taylor_green_initial import TaylorGreenHardInitialCondition
from src.physics.navier_stokes import navier_stokes_residuals
from src.physics.taylor_green import TaylorGreenVortex
from src.training.taylor_green_vara_v2_trainer import TaylorGreenVARAV2Trainer
from src.utils.config import deep_update, load_config


def test_hard_initial_wrapper_matches_complete_initial_state() -> None:
    benchmark = TaylorGreenVortex(reynolds=100.0)
    config = load_config("configs/taylor_green_repaired.yaml")
    base = build_mlp_from_config(config, benchmark.bounds)
    model = TaylorGreenHardInitialCondition(base, benchmark)
    coords = torch.tensor(
        [
            [0.3, 1.2, 0.0],
            [2.1, 4.0, 0.0],
            [5.8, 0.7, 0.0],
        ],
        dtype=torch.float64,
    )
    model = model.to(dtype=torch.float64)
    prediction = model(coords)
    exact = benchmark.exact_torch(coords)
    expected = torch.cat([exact["u"], exact["v"], exact["p"]], dim=1)
    assert torch.allclose(prediction, expected, atol=1e-12, rtol=0.0)


def test_hard_initial_wrapper_has_finite_transient_residuals() -> None:
    benchmark = TaylorGreenVortex(reynolds=100.0)
    config = load_config("configs/taylor_green_repaired.yaml")
    model = TaylorGreenHardInitialCondition(
        build_mlp_from_config(config, benchmark.bounds),
        benchmark,
    )
    coords = torch.tensor(
        [[0.4, 0.7, 0.2], [2.0, 4.0, 0.8]],
        dtype=torch.float32,
    )
    residuals = navier_stokes_residuals(model, coords, benchmark.nu, steady=False)
    for name in ("f_u", "f_v", "f_c"):
        assert torch.isfinite(residuals[name]).all()


def test_repaired_trainer_covers_every_temporal_patch_and_smokes(tmp_path) -> None:
    config = deep_update(
        load_config("configs/taylor_green_repaired.yaml"),
        {
            "device": "cpu",
            "training": {
                "n_collocation": 16,
                "n_boundary": 8,
                "n_data": 0,
                "log_every": 1,
            },
            "validation": {"nx": 4, "ny": 4},
            "test": {"nx": 6, "ny": 6},
            "patches": {"nx_patches": 2, "ny_patches": 2, "nt_patches": 2},
            "controller_v2": {
                "total_steps": 4,
                "warmup_steps": 1,
                "control_blocks": 1,
                "block_steps": 3,
                "probe_steps": 1,
                "gradient_probe_interior": 8,
                "gradient_probe_boundary": 4,
            },
            "taylor_green": {
                "initial_condition_metric_resolution": 4,
                "temporal_metric_resolution": 8,
                "evaluation_times": [0.0, 1.0],
            },
            "experiments": {"root": str(tmp_path)},
        },
    )
    trainer = TaylorGreenVARAV2Trainer(config)
    assert trainer.initial_condition_mismatch() == pytest.approx(0.0, abs=1e-14)
    coords = trainer._space_time_diagnostic_coords()
    patch_ids = trainer.patch_grid.assign_numpy(coords)
    temporal_ids = np.unique(patch_ids // 4)
    assert temporal_ids.tolist() == [0, 1]

    metrics = trainer.run()
    assert metrics["applied_optimizer_steps"] == 4
    assert metrics["taylor_green_initial_condition_mismatch"] == pytest.approx(
        0.0,
        abs=1e-14,
    )
    assert metrics["taylor_green_temporal_slices_evaluated"] == 2
    assert (trainer.run_dir / "taylor_green_temporal_metrics.csv").exists()
