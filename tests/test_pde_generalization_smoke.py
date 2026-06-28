"""Four-step CPU smoke runs for every PDE and primary method."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.pde_generalization.trainer import PDEGeneralizationTrainer


@pytest.mark.parametrize("benchmark", ["burgers2d", "allen_cahn", "advection_diffusion"])
@pytest.mark.parametrize("method", ["vanilla", "vara_v2"])
def test_four_step_smoke_writes_research_artifacts(tmp_path, benchmark, method) -> None:
    run_dir = tmp_path / benchmark / method
    trainer = PDEGeneralizationTrainer(_tiny_config(benchmark), method, run_dir)
    metrics = trainer.run()
    assert metrics["applied_optimizer_steps"] == 4
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


def _tiny_config(benchmark: str) -> dict:
    output_dim = 2 if benchmark == "burgers2d" else 1
    benchmark_params = {
        "bounds": [0.0, 1.0, 0.0, 1.0],
        "t_bounds": [0.0, 1.0],
        "nu": 0.01,
        "eps": 0.04,
        "kappa": 0.01,
        "advection_velocity": [1.0, 0.5],
        "sigma": 0.1,
    }
    return {
        "benchmark": benchmark,
        "benchmark_params": benchmark_params,
        "seed": 3,
        "device": "cpu",
        "dtype": "float32",
        "model": {"input_dim": 3, "output_dim": output_dim, "hidden_layers": [8, 8], "activation": "tanh"},
        "training": {
            "n_collocation": 16,
            "n_boundary": 8,
            "n_initial": 8,
            "n_sparse_data": 8,
            "lr": 1e-3,
            "weights": {"pde": 1.0, "bc": 1.0, "ic": 1.0, "sparse_data": 1.0},
        },
        "patches": {"nx_patches": 2, "ny_patches": 2, "nt_patches": 2},
        "diagnostics": {
            "n_interior": 16,
            "n_boundary": 8,
            "n_initial": 8,
            "aggregation_percentile": 90,
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
            "residual_chunk_size": 64,
            "controller_reference_metrics_enabled": False,
        },
        "plots": {"enabled": False},
    }
