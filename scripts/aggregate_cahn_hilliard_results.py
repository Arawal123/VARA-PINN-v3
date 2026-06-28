"""Aggregate isolated Cahn--Hilliard result folders."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pde_cahn_hilliard.summary import collect_summaries, write_aggregate_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    raw = collect_summaries(args.input_dirs)
    if raw.empty:
        raise FileNotFoundError("No valid Cahn--Hilliard summary.json files were found.")
    output = Path(args.output_dir)
    write_aggregate_outputs(raw, output)
    print(f"Collected {len(raw)} Cahn--Hilliard runs into {output}")


if __name__ == "__main__":
    main()
