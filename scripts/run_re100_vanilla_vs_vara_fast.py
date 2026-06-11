"""Matched Re=100 Vanilla versus VARA V2 comparison.

Vanilla always runs first. VARA V2 is skipped unless the Vanilla stage passes
the configured reference-free validity gates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_lid_cavity_re100_sanity import build_combined_report
from scripts.run_vara_v2_continuation import run as run_continuation
from src.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--preset",
        choices=["fast_screen", "diagnostic", "reliable", "final"],
        default="reliable",
    )
    parser.add_argument(
        "--config",
        default="configs/vara_v2/lid_cavity_continuation_reliable.yaml",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/vara_v2/re100_vanilla_vs_vara_fast",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_args = argparse.Namespace(
        config=args.config,
        methods=["vanilla", "vara_v2"],
        reynolds=[100.0],
        seeds=[int(args.seed)],
        full_field_reference_map=None,
        device=args.device,
        output_dir=args.output_dir,
        enhanced_backbone=False,
        reliable=True,
        preset=args.preset,
        gate_vara_on_vanilla=True,
        continue_on_invalid=True,
        quick=False,
        overwrite=bool(args.overwrite),
    )
    run_continuation(run_args)
    output = Path(args.output_dir)
    method = "both" if (output / f"seed_{args.seed}" / "re_0100" / "vara_v2").exists() else "vanilla"
    report = build_combined_report(output, method, [int(args.seed)])
    save_json(report, output / "summary" / "re100_vanilla_vs_vara_report.json")
    print(f"Saved matched comparison: {output}")
    print(report.get("vara_dominance", "VARA not evaluated"))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
