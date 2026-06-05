from argparse import Namespace
from pathlib import Path
import sys

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

    summary = out / "summary"
    assert (summary / "continuation_results_long.csv").exists()
    assert (summary / "vara_vs_vanilla_by_re.csv").exists()
    assert (summary / "improvement_percent_by_re.csv").exists()
    assert (summary / "available_reference_metrics_by_re.csv").exists()
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
