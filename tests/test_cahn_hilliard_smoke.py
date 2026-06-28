"""Four-step CPU smoke tests for vanilla and VARA Cahn--Hilliard."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.pde_cahn_hilliard.trainer import CahnHilliardTrainer


@pytest.mark.parametrize("method", ["vanilla", "vara_v2"])
def test_cahn_hilliard_four_step_run_writes_artifacts(tmp_path, method: str) -> None:
    run_dir = tmp_path / method
    trainer = CahnHilliardTrainer(tiny_cahn_hilliard_config(), method, run_dir)
    metrics = trainer.run()
    assert metrics["applied_optimizer_steps"] == 4
    assert "cahn_hilliard_interface_band_rel_l2" in metrics
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "losses.csv").is_file()
    assert (run_dir / "checkpoints" / "final.pt").is_file()
    assert len(pd.read_csv(run_dir / "losses.csv")) == 4
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["controller_reference_metrics_enabled"] is False
    if method == "vara_v2":
        assert (run_dir / "vara_v2_decisions.csv").is_file()
        assert (run_dir / "vara_v2_allocation_history.json").is_file()


def tiny_cahn_hilliard_config() -> dict:
    return {
        "benchmark": {
            "name": "cahn_hilliard",
            "bounds": [0.0, 1.0, 0.0, 1.0],
            "t_bounds": [0.0, 1.0],
            "epsilon": 0.04,
            "mobility": 1.0,
            "delta": 1e-6,
            "sparse_fraction": 0.02,
            "sparse_seed": 0,
        },
        "seed": 5,
        "device": "cpu",
        "dtype": "float32",
        "model": {"input_dim": 3, "output_dim": 2, "hidden_layers": [8, 8], "activation": "tanh"},
        "training": {
            "n_collocation": 12,
            "n_boundary": 8,
            "n_initial": 8,
            "n_sparse_data": 8,
            "lr": 1e-3,
            "weights": {
                "ch_residual": 1.0,
                "chemical_potential_residual": 1.0,
                "bc_u": 1.0,
                "bc_mu": 1.0,
                "ic_u": 1.0,
                "ic_mu": 1.0,
                "sparse_u_mse": 1.0,
                "sparse_mu_mse": 0.0,
                "interface_proxy_regularization": 0.0,
            },
        },
        "patches": {"nx_patches": 2, "ny_patches": 2, "nt_patches": 2},
        "diagnostics": {
            "n_interior": 12,
            "n_boundary": 8,
            "n_initial": 8,
            "aggregation_percentile": 90,
            "interface_tau": 0.25,
            "interface_focus_strength": 1.0,
        },
        "controller_v2": {
            "total_steps": 4,
            "warmup_steps": 1,
            "control_blocks": 1,
            "block_steps": 3,
            "probe_steps": 1,
            "max_patch_mass": 0.5,
            "counterfactual_probe_enabled": True,
            "gradient_prefilter_enabled": False,
            "rollback_enabled": True,
            "guard_metrics": [
                "pde_residual_mean",
                "boundary_condition_error",
                "unweighted_validation_loss",
                "unweighted_physics_validation_loss",
            ],
        },
        "evaluation": {
            "nx": 5,
            "ny": 5,
            "nt": 3,
            "chunk_size": 32,
            "controller_reference_metrics_enabled": False,
        },
        "plots": {"enabled": False},
    }
