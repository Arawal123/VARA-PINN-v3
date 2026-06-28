"""Run paired vanilla/VARA experiments for isolated manufactured PDEs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pde_generalization.summary import collect_run_summaries, write_standard_outputs
from src.pde_generalization.trainer import PDEGeneralizationTrainer, SUPPORTED_MODES
from src.pde_generalization.metrics import primary_metric_name
from src.utils.config import deep_update, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["burgers2d", "allen_cahn", "advection_diffusion"], required=True)
    parser.add_argument("--methods", nargs="+", choices=sorted(SUPPORTED_MODES), default=["vanilla", "vara_v2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--preset", choices=["fast_screen", "reliable", "final"], default="fast_screen")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sparse_fraction", type=float, default=None)
    parser.add_argument("--no_sparse_data", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Four-step CPU integration run.")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else ROOT / "configs" / "pde_generalization" / f"{args.benchmark}.yaml"
    config = load_config(config_path)
    config = deep_update(
        config,
        load_config(ROOT / "configs" / "pde_generalization" / "presets" / f"{args.preset}.yaml"),
    )
    if args.ablation:
        ablations = load_config(ROOT / "configs" / "pde_generalization" / "ablations.yaml").get("ablations", {})
        if args.ablation not in ablations:
            raise ValueError(f"Unknown ablation {args.ablation!r}; choose from {sorted(ablations)}.")
        config = deep_update(config, ablations[args.ablation])
    if args.device:
        config["device"] = args.device
    if args.quick:
        config = deep_update(config, _quick_overrides())
        config["device"] = args.device or "cpu"
    if args.sparse_fraction is not None and not 0.0 <= args.sparse_fraction <= 1.0:
        raise ValueError("--sparse_fraction must be between zero and one.")
    configured_fraction = float(
        config.get("benchmark_params", {}).get("sparse_sample_fraction", 0.0)
    )
    effective_fraction = (
        float(args.sparse_fraction)
        if args.sparse_fraction is not None
        else configured_fraction
    )
    if args.sparse_fraction is not None or not args.quick:
        evaluation = config.get("evaluation", {})
        grid_size = (
            int(evaluation.get("nx", 48))
            * int(evaluation.get("ny", 48))
            * int(evaluation.get("nt", 11))
        )
        config.setdefault("benchmark_params", {})[
            "sparse_sample_fraction"
        ] = effective_fraction
        config.setdefault("training", {})["n_sparse_data"] = (
            max(1, int(round(grid_size * effective_fraction)))
            if effective_fraction > 0.0
            else 0
        )
    if args.no_sparse_data:
        config.setdefault("training", {})["n_sparse_data"] = 0
        config.setdefault("training", {}).setdefault("weights", {})[
            "sparse_data"
        ] = 0.0

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and args.overwrite:
        _safe_remove_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for method in args.methods:
            run_config = deepcopy(config)
            run_config["seed"] = int(seed)
            run_dir = output_dir / f"seed_{seed}" / method
            if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
                raise FileExistsError(f"Run directory already exists: {run_dir}. Use --overwrite to replace it.")
            trainer = PDEGeneralizationTrainer(run_config, method, run_dir)
            metrics = trainer.run()
            primary = primary_metric_name(args.benchmark)
            print(f"{args.benchmark} seed={seed} method={method} {primary}={metrics.get(primary, float('nan')):.6g}")

    raw = collect_run_summaries([output_dir])
    write_standard_outputs(raw, output_dir / "summary")
    print(f"Wrote experiment suite to {output_dir}")


def _quick_overrides() -> dict:
    return {
        "model": {"hidden_layers": [8, 8]},
        "training": {
            "n_collocation": 16,
            "n_boundary": 8,
            "n_initial": 8,
            "n_sparse_data": 8,
        },
        "diagnostics": {"n_interior": 16, "n_boundary": 8, "n_initial": 8},
        "controller_v2": {
            "total_steps": 4,
            "warmup_steps": 1,
            "control_blocks": 1,
            "block_steps": 3,
            "probe_steps": 1,
            "gradient_prefilter_enabled": False,
        },
        "evaluation": {"nx": 6, "ny": 6, "nt": 3, "residual_chunk_size": 64},
        "plots": {"enabled": True},
    }


def _safe_remove_output(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 3:
        raise ValueError(f"Refusing to recursively remove unsafe output path: {resolved}")
    shutil.rmtree(resolved)


if __name__ == "__main__":
    main()
