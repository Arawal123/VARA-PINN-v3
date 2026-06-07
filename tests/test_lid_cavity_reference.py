from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_on_grid, vector_relative_l2
from src.physics.cavity_reference import load_full_field_reference, load_lid_cavity_profile_reference, validate_full_field_against_ghia
from src.physics.rectangular_benchmarks import LidDrivenCavityQualitative


def test_builtin_ghia_profiles_load_for_required_reynolds():
    for reynolds in [100, 400, 1000]:
        ref = load_lid_cavity_profile_reference("ghia", reynolds)
        assert ref.has_u
        assert ref.has_v
        assert len(ref.u_profile) == 17
        assert len(ref.v_profile) == 17


def test_missing_ghia_reynolds_raises_clear_error():
    with pytest.raises(ValueError, match="Available Re"):
        load_lid_cavity_profile_reference("ghia", 250)


def test_cavity_profile_points_are_centerlines():
    bench = LidDrivenCavityQualitative(reynolds=100, reference="ghia")
    profile = bench.profile_reference_np()
    assert np.allclose(profile["u_xy"][:, 0], 0.5)
    assert np.allclose(profile["v_xy"][:, 1], 0.5)
    assert profile["u_ref"].shape == (17, 1)
    assert profile["v_ref"].shape == (17, 1)


def test_external_centerline_csv_loader(tmp_path):
    path = tmp_path / "cavity_profiles.csv"
    pd.DataFrame(
        [
            {"re": 100, "x": 0.5, "y": 0.0, "u_ref": 0.0, "v_ref": np.nan},
            {"re": 100, "x": 0.5, "y": 0.5, "u_ref": -0.2, "v_ref": np.nan},
            {"re": 100, "x": 0.0, "y": 0.5, "u_ref": np.nan, "v_ref": 0.0},
            {"re": 100, "x": 0.5, "y": 0.5, "u_ref": np.nan, "v_ref": 0.05},
        ]
    ).to_csv(path, index=False)
    ref = load_lid_cavity_profile_reference("external", 100, path)
    assert len(ref.u_profile) == 2
    assert len(ref.v_profile) == 2


def test_cavity_profile_metrics_are_finite():
    model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    bench = LidDrivenCavityQualitative(reynolds=100, reference="ghia")
    _, _, coords = bench.grid(8, 8)
    metrics = evaluate_on_grid(model, bench, coords, torch.device("cpu"), steady=True)
    assert np.isfinite(metrics["u_centerline_rmse"])
    assert np.isfinite(metrics["v_centerline_rmse"])
    assert np.isfinite(metrics["centerline_profile_score"])
    assert np.isfinite(metrics["cavity_benchmark_score"])
    assert np.isnan(metrics["u_rel_l2"])


def test_full_field_reference_missing_pressure_preserves_nan_and_computes_omega(tmp_path):
    path = tmp_path / "full_field_no_pressure.npz"
    x = np.linspace(0.0, 1.0, 6)
    y = np.linspace(0.0, 1.0, 6)
    X, Y = np.meshgrid(x, y)
    np.savez(path, x=X.reshape(-1), y=Y.reshape(-1), u=Y.reshape(-1), v=(-X).reshape(-1))

    ref = load_full_field_reference(path)

    assert not ref["has_p_reference"]
    assert np.isnan(ref["p"]).all()
    assert ref["has_omega_reference"]
    assert ref["omega_reference_source"] == "computed_from_velocity"
    assert np.isfinite(ref["omega"]).all()


def test_full_field_cfd_metrics_are_evaluation_only_and_named(tmp_path):
    path = tmp_path / "full_field_velocity_only.npz"
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 1.0, 5)
    X, Y = np.meshgrid(x, y)
    np.savez(path, x=X.reshape(-1), y=Y.reshape(-1), u=Y.reshape(-1), v=(-X).reshape(-1))
    model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    bench = LidDrivenCavityQualitative(
        reynolds=100,
        reference="none",
        full_field_reference_path=str(path),
        profile_only=False,
        has_reference=True,
        reference_kind="full_field_cfd",
    )
    _, _, coords = bench.grid(5, 5)

    metrics = evaluate_on_grid(model, bench, coords, torch.device("cpu"), steady=True)

    assert np.isfinite(metrics["u_full_rel_l2"])
    assert np.isfinite(metrics["v_full_rel_l2"])
    assert np.isfinite(metrics["velocity_full_rel_l2"])
    assert np.isfinite(metrics["velocity_mag_rmse"])
    assert np.isfinite(metrics["velocity_mag_mae"])
    assert np.isnan(metrics["p_full_rel_l2_centered"])
    assert np.isfinite(metrics["omega_full_rel_l2"])
    assert not metrics["has_p_full_field_reference"]
    assert metrics["has_omega_full_field_reference"]


def test_full_field_reference_can_be_validated_against_ghia(tmp_path):
    path = tmp_path / "flat_full_field.npz"
    x = np.linspace(0.0, 1.0, 8)
    y = np.linspace(0.0, 1.0, 8)
    X, Y = np.meshgrid(x, y)
    np.savez(path, x=X.reshape(-1), y=Y.reshape(-1), u=np.zeros(X.size), v=np.zeros(X.size))

    metrics = validate_full_field_against_ghia(path, 100)

    assert metrics["reynolds"] == 100.0
    assert np.isfinite(metrics["cfd_vs_ghia_u_centerline_rmse"])
    assert np.isfinite(metrics["cfd_vs_ghia_v_centerline_rmse"])


def test_vector_velocity_error_does_not_collapse_to_speed_magnitude_error():
    u_ref = np.array([[1.0], [0.0]])
    v_ref = np.array([[0.0], [1.0]])
    u_pred = -u_ref
    v_pred = -v_ref
    assert vector_relative_l2((u_pred, v_pred), (u_ref, v_ref)) == pytest.approx(2.0)
    speed_ref = np.sqrt(u_ref * u_ref + v_ref * v_ref)
    speed_pred = np.sqrt(u_pred * u_pred + v_pred * v_pred)
    assert np.linalg.norm(speed_pred - speed_ref) == pytest.approx(0.0)
