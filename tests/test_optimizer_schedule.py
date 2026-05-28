from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.trainer import ExperimentTrainer
from src.training.vara_trainer import VARATrainer


def test_second_stage_optimizer_is_disabled_by_default():
    trainer = object.__new__(ExperimentTrainer)
    trainer.config = {"optimizer": {"name": "adam"}}
    assert trainer._optimizer_second_stage_enabled() is False


def test_adam_to_lbfgs_second_stage_runs_on_tiny_cavity(tmp_path):
    config = {
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
            "name": "adam_to_lbfgs",
            "lr": 0.001,
            "lbfgs": {
                "enabled": True,
                "epochs": 1,
                "lr": 1.0,
                "max_iter": 1,
                "max_eval": 1,
                "history_size": 5,
                "line_search_fn": None,
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
    trainer = VARATrainer(config, mode="vanilla_pinn")
    batch = trainer.initial_batch()
    trainer.train_epochs(batch, trainer.controller.state, cycle=0, epochs_override=1)
    result = trainer.run_optimizer_second_stage(batch, trainer.controller.state, cycle=1)
    assert trainer.optimizer_stage == "lbfgs"
    assert trainer.lbfgs_steps_completed == 1
    assert isinstance(trainer.optimizer, torch.optim.LBFGS)
    assert np.isfinite(result["total"])
