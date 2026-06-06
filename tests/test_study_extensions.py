from copy import deepcopy
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.vara_trainer import VARATrainer


def _tiny_config(root: Path, run_id: str) -> dict:
    return {
        "benchmark": "lid_driven_cavity",
        "seed": 7,
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
        "optimizer": {"name": "adam", "lr": 0.001, "final_repair": {"enabled": False}},
        "training": {
            "adaptive_cycles": 1,
            "epochs_per_cycle": 2,
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
                "pde_gradient": 0.1,
            },
        },
        "validation": {"nx": 5, "ny": 5},
        "test": {"nx": 5, "ny": 5},
        "patches": {"nx_patches": 2, "ny_patches": 2, "nt_patches": 1},
        "diagnostics": {"mode": "residual_only"},
        "weak_regions": {"max_active_patches": 1},
        "local_controller": {
            "trial_epochs": 1,
            "warmup_cycles": 0,
            "max_actions_per_cycle": 1,
            "rejection_recovery_epochs": 0,
        },
        "experiments": {"root": str(root), "run_id": run_id},
    }


def test_disabled_compute_budget_preserves_legacy_parameter_update(tmp_path):
    legacy_cfg = _tiny_config(tmp_path / "legacy", "legacy")
    disabled_cfg = deepcopy(_tiny_config(tmp_path / "disabled", "disabled"))
    disabled_cfg["compute_budget"] = {
        "enabled": False,
        "type": "optimizer_steps",
        "value": 1,
    }
    legacy = VARATrainer(legacy_cfg, mode="vanilla_pinn")
    disabled = VARATrainer(disabled_cfg, mode="vanilla_pinn")
    legacy.train_epochs(legacy.initial_batch(), cycle=0)
    disabled.train_epochs(disabled.initial_batch(), cycle=0)
    for legacy_value, disabled_value in zip(legacy.model.state_dict().values(), disabled.model.state_dict().values()):
        assert torch.equal(legacy_value, disabled_value)


def test_optimizer_step_budget_stops_exactly(tmp_path):
    config = _tiny_config(tmp_path, "budgeted")
    config["compute_budget"] = {
        "enabled": True,
        "type": "optimizer_steps",
        "value": 1,
    }
    trainer = VARATrainer(config, mode="vanilla_pinn")
    trainer.train_epochs(trainer.initial_batch(), cycle=0, epochs_override=5)
    assert trainer.compute_tracker.optimizer_steps == 1
    assert trainer.compute_tracker.objective_evaluations == 1
    assert trainer.compute_tracker.collocation_evaluations == 8


def test_modern_baseline_modes_run(tmp_path):
    for mode in [
        "self_adaptive_attention_pinn",
        "gradient_balanced_pinn",
        "gradient_enhanced_pinn",
    ]:
        config = _tiny_config(tmp_path / mode, mode)
        config["gradient_balancing"] = {"update_every": 1}
        metrics = VARATrainer(config, mode=mode).run()
        assert metrics["optimizer_steps"] == 2
        assert metrics["accepted_interventions"] == 0
        assert metrics["rollback_count"] == 0
