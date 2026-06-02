from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.vara_trainer import VARATrainer


def _tiny_cavity_config(tmp_path: Path, enabled: bool) -> dict:
    return {
        "benchmark": "lid_driven_cavity",
        "seed": 0,
        "deterministic": True,
        "device": "cpu",
        "benchmark_params": {
            "reynolds": 100.0,
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "lid_velocity": 1.0,
            "reference": "none",
            "profile_only": True,
        },
        "pde": {"steady": True},
        "model": {"input_dim": 2, "output_dim": 3, "hidden_layers": [8], "activation": "tanh"},
        "optimizer": {
            "name": "adam",
            "lr": 0.001,
            "final_repair": {
                "enabled": enabled,
                "score_metric": "unweighted_validation_loss",
                "acceptance_tolerance": 10.0,
                "epochs": 1,
                "lr": 0.1,
                "max_iter": 1,
                "max_eval": 1,
                "history_size": 5,
                "line_search_fn": None,
                "batch_multiplier": 1.0,
                "residual_fraction": 0.25,
            },
        },
        "training": {
            "adaptive_cycles": 1,
            "epochs_per_cycle": 1,
            "log_every": 1,
            "n_collocation": 8,
            "n_boundary": 8,
            "n_data": 0,
            "weights": {
                "pde": 1.0,
                "momentum_u": 1.0,
                "momentum_v": 1.0,
                "continuity": 1.0,
                "bc": 5.0,
                "u": 0.0,
                "v": 0.0,
                "p": 0.0,
                "omega": 0.0,
                "pressure_gradient": 0.0,
            },
        },
        "validation": {"nx": 6, "ny": 6},
        "test": {"nx": 6, "ny": 6},
        "patches": {"nx_patches": 2, "ny_patches": 2, "nt_patches": 1},
        "diagnostics": {"mode": "residual_only"},
        "weak_regions": {"max_active_patches": 1},
        "experiments": {"root": str(tmp_path)},
    }


def test_final_physics_repair_disabled_by_default(tmp_path):
    trainer = VARATrainer(_tiny_cavity_config(tmp_path, enabled=False), mode="vanilla_pinn")
    status = trainer.run_final_physics_repair()
    assert status == {"enabled": False, "accepted": False}
    assert trainer.optimizer_stage == "adam"


def test_final_physics_repair_runs_global_guarded_stage(tmp_path):
    trainer = VARATrainer(_tiny_cavity_config(tmp_path, enabled=True), mode="vanilla_pinn")
    batch = trainer.initial_batch()
    trainer.train_epochs(batch, trainer.controller.state, cycle=0, epochs_override=1)
    status = trainer.run_final_physics_repair(cycle=1)

    assert status["enabled"] is True
    assert status["global_only"] is True
    assert status["score_name"] == "unweighted_validation_loss"
    assert status["batch_n_collocation"] == 8
    assert status["batch_n_boundary"] == 8
    assert np.isfinite(status["pre_repair_score"])
    assert np.isfinite(status["post_repair_score"])


def test_repair_collateral_guard_accepts_bounded_damage(tmp_path):
    trainer = VARATrainer(_tiny_cavity_config(tmp_path, enabled=False), mode="vanilla_pinn")
    ok, report = trainer._repair_collateral_ok(
        {"pde_residual_mean": 1.0, "momentum_residual_mean": 2.0},
        {"pde_residual_mean": 1.05, "momentum_residual_mean": 2.20},
        {"collateral_tolerances": {"pde_residual_mean": 0.10, "momentum_residual_mean": 0.18}},
    )

    assert ok is True
    assert report["collateral_metric_status"] == "ok"
    assert report["collateral_pde_residual_mean_damage"] == pytest.approx(0.05)
    assert report["collateral_momentum_residual_mean_damage"] == pytest.approx(0.10)


def test_repair_collateral_guard_rejects_excessive_damage(tmp_path):
    trainer = VARATrainer(_tiny_cavity_config(tmp_path, enabled=False), mode="vanilla_pinn")
    ok, report = trainer._repair_collateral_ok(
        {"pde_residual_mean": 1.0, "momentum_residual_mean": 2.0},
        {"pde_residual_mean": 1.12, "momentum_residual_mean": 2.10},
        {"collateral_tolerances": {"pde_residual_mean": 0.10, "momentum_residual_mean": 0.18}},
    )

    assert ok is False
    assert report["collateral_metric_status"] == "failed:pde_residual_mean"
    assert report["collateral_max_damage"] == pytest.approx(0.12)
