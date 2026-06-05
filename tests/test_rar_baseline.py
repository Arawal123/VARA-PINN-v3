from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.vara_trainer import VARATrainer


def _tiny_cavity_config(tmp_path: Path) -> dict:
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
        "optimizer": {"name": "adam", "lr": 0.001, "final_repair": {"enabled": False}},
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
        "rar": {"residual_fraction": 0.5},
        "experiments": {"root": str(tmp_path)},
    }


def test_rar_baseline_runs_without_controller_interventions(tmp_path):
    trainer = VARATrainer(_tiny_cavity_config(tmp_path), mode="rar_pinn")
    metrics = trainer.run()

    assert metrics["accepted_interventions"] == 0
    assert metrics["rejected_interventions"] == 0
    assert metrics["rollback_count"] == 0
    assert metrics["number_of_active_patches"] == 0
    assert (trainer.run_dir / "summary.json").exists()
    assert (trainer.run_dir / "action_log.json").exists()
