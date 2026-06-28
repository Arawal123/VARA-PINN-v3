"""Run isolated vanilla/VARA split-form Cahn--Hilliard experiments."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pde_cahn_hilliard.summary import collect_summaries, write_runner_outputs
from src.pde_cahn_hilliard.trainer import CahnHilliardTrainer, SUPPORTED_METHODS
from src.utils.config import deep_update, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=sorted(SUPPORTED_METHODS), default=["vanilla", "vara_v2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--preset", choices=["fast_screen", "reliable", "final"], default="fast_screen")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--mobility", type=float, default=None)
    parser.add_argument("--sparse_fraction", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run a four-step integration check.")
    parser.add_argument("--no_sparse_data", action="store_true")
    parser.add_argument("--stress", action="store_true", help="Apply the epsilon=0.03 stress overlay.")
    parser.add_argument("--ablation", default=None)
    args = parser.parse_args()

    config_root = ROOT / "configs" / "cahn_hilliard"
    config = load_config(config_root / "base.yaml")
    config = deep_update(config, load_config(config_root / f"{args.preset}.yaml"))
    if args.stress:
        config = deep_update(config, load_config(config_root / "stress_eps_003.yaml"))
    if args.ablation:
        ablations = load_config(config_root / "ablations.yaml").get("ablations", {})
        if args.ablation not in ablations:
            raise ValueError(f"Unknown ablation {args.ablation!r}; choose from {sorted(ablations)}.")
        config = deep_update(config, ablations[args.ablation])
    if args.quick:
        config = deep_update(config, _quick_overrides())
        config["device"] = args.device or "cpu"
    if args.device:
        config["device"] = args.device
    if args.epsilon is not None:
        if args.epsilon <= 0.0:
            raise ValueError("--epsilon must be positive.")
        config.setdefault("benchmark", {})["epsilon"] = float(args.epsilon)
    if args.mobility is not None:
        if args.mobility <= 0.0:
            raise ValueError("--mobility must be positive.")
        config.setdefault("benchmark", {})["mobility"] = float(args.mobility)
    if args.sparse_fraction is not None and not 0.0 <= args.sparse_fraction <= 1.0:
        raise ValueError("--sparse_fraction must lie between zero and one.")
    configured_fraction = float(config.get("benchmark", {}).get("sparse_fraction", 0.02))
    fraction = float(args.sparse_fraction) if args.sparse_fraction is not None else configured_fraction
    if args.sparse_fraction is not None or not args.quick:
        evaluation = config.get("evaluation", {})
        grid_size = int(evaluation.get("nx", 48)) * int(evaluation.get("ny", 48)) * int(evaluation.get("nt", 11))
        config.setdefault("training", {})["n_sparse_data"] = max(1, int(round(grid_size * fraction))) if fraction > 0.0 else 0
    config.setdefault("benchmark", {})["sparse_fraction"] = fraction
    if args.no_sparse_data:
        config["training"]["n_sparse_data"] = 0
        config["training"].setdefault("weights", {})["sparse_u_mse"] = 0.0
        config["training"]["weights"]["sparse_mu_mse"] = 0.0
        config["benchmark"]["sparse_fraction"] = 0.0

    output = Path(args.output_dir).resolve()
    if output.exists() and args.overwrite:
        _safe_remove_output(output)
    output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for method in args.methods:
            run_config = deepcopy(config)
            run_config["seed"] = int(seed)
            run_dir = output / f"seed_{seed}" / method
            if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
                raise FileExistsError(f"Run directory exists: {run_dir}; use --overwrite.")
            trainer = CahnHilliardTrainer(run_config, method, run_dir)
            metrics = trainer.run()
            print(
                f"seed={seed} method={method} "
                f"u_rel_l2={metrics['cahn_hilliard_u_rel_l2']:.6g}"
            )
    raw = collect_summaries([output])
    write_runner_outputs(raw, output / "summary")
    print(f"Wrote Cahn--Hilliard experiment to {output}")


def _quick_overrides() -> dict:
    return {
        "model": {"hidden_layers": [8, 8]},
        "training": {
            "n_collocation": 12,
            "n_boundary": 8,
            "n_initial": 8,
            "n_sparse_data": 8,
        },
        "patches": {"nx_patches": 2, "ny_patches": 2, "nt_patches": 2},
        "diagnostics": {"n_interior": 12, "n_boundary": 8, "n_initial": 8},
        "controller_v2": {
            "total_steps": 4,
            "warmup_steps": 1,
            "control_blocks": 1,
            "block_steps": 3,
            "probe_steps": 1,
            "max_patch_mass": 0.5,
            "gradient_prefilter_enabled": False,
        },
        "evaluation": {"nx": 5, "ny": 5, "nt": 3, "chunk_size": 32},
        "plots": {"enabled": True},
    }


def _safe_remove_output(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 3:
        raise ValueError(f"Refusing to recursively remove unsafe path: {resolved}")
    shutil.rmtree(resolved)


if __name__ == "__main__":
    main()
