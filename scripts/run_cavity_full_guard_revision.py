"""Frozen Re=100 non-polish primary pair and explicit optional compute control."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.training.cavity_full_guard_revision import CONFIG, BASE_SHA, RevisionTrainer, validate_protocol
from src.utils.config import load_config
from src.utils.io import save_json


def provenance():
    return {"base_branch": "codex/vara-controller-v2", "base_sha": BASE_SHA,
        "revision_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "revision_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "command": sys.argv, "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "pandas", "matplotlib", "pyyaml", "scipy", "tqdm", "pytest")},
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--control_plan", action="store_true")
    parser.add_argument("--compute_matched_control", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    config = load_config(CONFIG)
    config["device"] = args.device
    validate_protocol(config)
    if args.control_plan or args.compute_matched_control:
        from src.training.cavity_full_guard_audit import control_plan, run_control
        plan = control_plan(output)
        print(json.dumps(plan, indent=2))
        if args.compute_matched_control:
            run_control(output, config, plan)
        return
    if args.preflight:
        # Constructor/sampling only: no optimizer steps or training.
        import tempfile
        from src.training.cavity_full_guard_audit import compare_initial
        with tempfile.TemporaryDirectory() as temporary:
            trainers = []
            for method in ("vanilla", "vara_v2_full_guard"):
                c = deepcopy(config)
                c["experiments"] = {"root": str(Path(temporary) / method), "flat_layout": True}
                trainers.append(RevisionTrainer(c, method))
            evidence = compare_initial(*trainers)
            if not all(evidence.values()):
                raise RuntimeError(f"Preflight FAIL: {evidence}")
            print(json.dumps({"resolved_protocol": config, "preflight": evidence,
                "parameter_count": sum(p.numel() for p in trainers[0].model.parameters()),
                "sparse": trainers[0].sparse_manifest, "environment": provenance()}, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    environment = provenance()
    if environment["git_status"]:
        raise RuntimeError("Refusing scientific run from a dirty source checkout")
    output.mkdir(parents=True)
    save_json(environment, output / "provenance.json")
    for method in ("vanilla", "vara_v2_full_guard"):
        c = deepcopy(config)
        c["experiments"] = {"root": str(output / method), "flat_layout": True}
        print(f"Starting {method}: Re=100 seed=0 fixed 4000 committed Adam steps", flush=True)
        RevisionTrainer(c, method).run()
    from src.training.cavity_full_guard_audit import audit
    result = audit(output)
    if not result["passed"]:
        raise RuntimeError("Post-run fairness FAIL; inspect fairness_audit.json")


if __name__ == "__main__":
    main()
