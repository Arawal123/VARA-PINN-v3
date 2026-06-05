from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_lid_cavity_re_continuation import run_continuation


def test_lid_cavity_re_continuation_writes_comparison_outputs(tmp_path):
    out = tmp_path / "continuation"
    args = Namespace(
        config="configs/lid_driven_cavity.yaml",
        output_dir=str(out),
        reynolds=[100.0, 150.0],
        seeds=[0],
        method="both",
        reference="ghia",
        reference_path=None,
        full_field_reference_map=None,
        device="cpu",
        quick=True,
        overwrite=False,
    )

    run_continuation(args)

    for re_label in ["re_0100", "re_0150"]:
        for method in ["vanilla", "vara"]:
            method_dir = out / "seed_0" / re_label / method
            assert (method_dir / "logs" / "summary.json").exists()
            assert (method_dir / "checkpoints" / "final.pt").exists()
            assert (method_dir / "figures" / "streamlines.png").exists()
            assert (method_dir / "figures" / "predicted_fields.png").exists()
            assert (method_dir / "figures" / "pde_residual.png").exists()
            assert (method_dir / "figures" / "continuity_residual.png").exists()
            assert (method_dir / "figures" / "momentum_residual.png").exists()

    comparison_dir = out / "seed_0" / "re_0150" / "comparison"
    assert (comparison_dir / "streamlines_side_by_side.png").exists()
    assert (comparison_dir / "residuals_side_by_side.png").exists()
    assert (comparison_dir / "metrics_comparison.csv").exists()
    assert (comparison_dir / "metric_comparison_bar.png").exists()

    summary = out / "summary"
    assert (summary / "continuation_results_long.csv").exists()
    assert (summary / "vara_vs_vanilla_by_re.csv").exists()
    assert (summary / "improvement_percent_by_re.csv").exists()
    assert (summary / "available_reference_metrics_by_re.csv").exists()
    assert (summary / "metric_improvement_by_re_bar.png").exists()
    assert (summary / "streamline_montage_vara.png").exists()
    assert (summary / "streamline_montage_vanilla.png").exists()
    assert (summary / "streamline_montage_side_by_side.png").exists()

    long_df = pd.read_csv(summary / "continuation_results_long.csv")
    re_150 = long_df[long_df["reynolds"] == 150.0]
    assert set(re_150["method"]) == {"vanilla", "vara"}
    assert re_150["warm_start_loaded"].astype(bool).all()
    assert re_150["warm_start_checkpoint"].notna().all()

    comparison = pd.read_csv(summary / "vara_vs_vanilla_by_re.csv")
    assert {"vanilla", "vara", "improvement_percent"}.issubset(comparison.columns)


def test_lid_cavity_re_continuation_uses_full_field_reference_map(tmp_path):
    out = tmp_path / "continuation_full_field"
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    ref_path = ref_dir / "re_0100.npz"
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 1.0, 5)
    X, Y = np.meshgrid(x, y)
    np.savez(
        ref_path,
        x=X.reshape(-1),
        y=Y.reshape(-1),
        u=Y.reshape(-1),
        v=(-X).reshape(-1),
        p=(X + 0.25 * Y).reshape(-1),
        omega=(1.0 + X - Y).reshape(-1),
    )
    reference_map = tmp_path / "reference_map.csv"
    pd.DataFrame([{"re": 100.0, "full_field_reference_path": str(ref_path)}]).to_csv(reference_map, index=False)
    args = Namespace(
        config="configs/lid_driven_cavity.yaml",
        output_dir=str(out),
        reynolds=[100.0, 150.0],
        seeds=[0],
        method="both",
        reference="ghia",
        reference_path=None,
        full_field_reference_map=str(reference_map),
        device="cpu",
        quick=True,
        overwrite=False,
    )

    run_continuation(args)

    for method in ["vanilla", "vara"]:
        method_figures = out / "seed_0" / "re_0100" / method / "figures"
        assert (method_figures / "reference_fields.png").exists()
        assert (method_figures / "error_fields.png").exists()
        assert (method_figures / "prediction_reference_error.png").exists()

        unmapped_figures = out / "seed_0" / "re_0150" / method / "figures"
        assert not (unmapped_figures / "reference_fields.png").exists()
        assert not (unmapped_figures / "error_fields.png").exists()

    mapped_comparison = out / "seed_0" / "re_0100" / "comparison"
    assert (mapped_comparison / "full_field_error_side_by_side.png").exists()
    assert (mapped_comparison / "full_field_metric_comparison_bar.png").exists()

    summary = out / "summary"
    reference_df = pd.read_csv(summary / "available_reference_metrics_by_re.csv")
    re_100 = reference_df[reference_df["reynolds"] == 100.0].iloc[0]
    re_150 = reference_df[reference_df["reynolds"] == 150.0].iloc[0]
    assert bool(re_100["full_field_reference_available"])
    assert re_100["quantitative_reference_level"] == "profile+full_field"
    assert not bool(re_150["full_field_reference_available"])
    assert re_150["quantitative_reference_level"] == "residual_only"

    comparison = pd.read_csv(summary / "vara_vs_vanilla_by_re.csv")
    full_field_rows = comparison[(comparison["reynolds"] == 100.0) & (comparison["metric"] == "u_rmse")]
    assert not full_field_rows.empty
    assert np.isfinite(full_field_rows["vanilla"]).all()
    assert np.isfinite(full_field_rows["vara"]).all()
    assert (summary / "full_field_metric_improvement_by_re_bar.png").exists()
