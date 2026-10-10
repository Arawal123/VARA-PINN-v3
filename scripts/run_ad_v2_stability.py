"""Paired, explicitly versioned AD stability study; never changes historical runs."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from src.pde_generalization.ad_stability_trainer import ADStabilityTrainer
from src.pde_generalization.trainer import _tensor_hash

SWITCHES = ("armijo_step_safeguard", "neutral_sampler_parity", "independent_component_guards",
            "frozen_diagnostic_scales", "all_channel_candidate_availability",
            "calibrated_fraction_prediction", "continuation_guard_rechecks")

def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

def smoke_config(cfg):
    cfg = deepcopy(cfg)
    cfg["study_kind"] = "CPU_SMOKE_NOT_PRIMARY"
    cfg["model"]["hidden_layers"] = [8, 8]
    cfg["training"].update(total_steps=12, n_collocation=32, n_boundary=16, n_initial=16, n_sparse_data=8)
    cfg["diagnostics"].update(n_interior=32, n_boundary=16, n_initial=16)
    cfg["controller_v2"].update(total_steps=12, warmup_steps=2, control_blocks=2, block_steps=5, probe_steps=1)
    cfg["ad_stability_revision"]["guard_recheck_steps"] = 2
    cfg["evaluation"].update(nx=6, ny=6, nt=3, residual_chunk_size=64)
    cfg["plots"]["enabled"] = False
    return cfg

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")

def verify_pair(root, seed, write=True):
    summaries = [json.loads((root/f"seed_{seed}"/method/"summary.json").read_text()) for method in ("vanilla", "vara_v2")]
    configs = [yaml.safe_load((root/f"seed_{seed}"/method/"resolved_config.yaml").read_text()) for method in ("vanilla", "vara_v2")]
    manifests = [json.loads((root/f"seed_{seed}"/method/"fairness_manifest.json").read_text()) for method in ("vanilla", "vara_v2")]
    keys = ("initial_model_parameter_hash", "sparse_sample_hash")
    for key in keys:
        if summaries[0][key] != summaries[1][key]:raise RuntimeError(f"seed {seed}: {key} mismatch")
    if configs[0] != configs[1]:raise RuntimeError(f"seed {seed}: resolved protocol mismatch")
    for key in ("boundary_pool_sha256", "initial_pool_sha256", "diagnostic_pool_sha256", "final_state_rule"):
        if manifests[0][key] != manifests[1][key]:raise RuntimeError(f"seed {seed}: {key} mismatch")
    expected = configs[0]["controller_v2"]["total_steps"]
    for method, summary in zip(("vanilla", "vara_v2"), summaries):
        if summary["metrics"]["applied_optimizer_steps"] != expected:raise RuntimeError("Committed-step budget mismatch")
        if summary["protocol_revision"] != "ad_v2_stable_v1":raise RuntimeError("Historical/revision result mixing")
        run = root/f"seed_{seed}"/method
        if not (run/"checkpoints"/"final.pt").is_file():raise RuntimeError("Missing final checkpoint")
    result = {"seed": seed, "matched_initialization": True, "matched_sparse_pool": True,
              "matched_conditions_and_diagnostics": True, "same_resolved_config": True,
              "committed_steps": expected, "final_state_rule": "last_committed_state_no_LBFGS_no_best_selection",
              "extra_VARAV2_compute_accounted_separately": True}
    if write:write_json(root/f"seed_{seed}"/"paired_fairness.json", result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT/"configs/pde_generalization/advection_diffusion_v2_stable.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--disable", choices=SWITCHES, nargs="*", default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume latest complete block; preserve interrupted attempts.")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):raise ValueError("Duplicate seeds")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.smoke:cfg = smoke_config(cfg)
    cfg["device"] = args.device
    for switch in args.disable:cfg["ad_stability_revision"][switch] = False
    if "independent_component_guards" in args.disable:
        cfg["controller_v2"]["guard_metrics"] = ["pde_residual_mean", "boundary_condition_error", "unweighted_validation_loss", "unweighted_physics_validation_loss"]
    if not args.smoke and not args.device.startswith("cuda"):raise RuntimeError("Primary study requires explicit CUDA; use --smoke for CPU checks")
    if args.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    root = Path(args.output_dir).resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:raise FileExistsError(f"Use a fresh output directory or --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    source_files=list((ROOT/"src").rglob("*.py"))+[ROOT/"scripts"/name for name in ("run_ad_v2_stability.py","package_ad_v2_stability.py","verify_ad_stability_zip.py")]
    source_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest() for p in source_files}
    if not args.smoke and git("status","--porcelain"):raise RuntimeError("Primary study requires a clean source checkout")
    identity = {"protocol": cfg, "seeds": args.seeds, "source_commit": git("rev-parse", "HEAD"), "disabled": args.disable,"source_hashes_lf":source_hashes}
    identity_path = root/"study_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:raise ValueError("Resume suite identity mismatch")
    else:write_json(identity_path, identity)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "scipy", "matplotlib", "PyYAML")}
    write_json(root/f"invocation_{timestamp}.json", {"argv": sys.argv, "command": " ".join(sys.argv), "source_commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "git_status": git("status", "--porcelain"), "python": platform.python_version(),
        "platform": platform.platform(), "versions": versions, "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "timestamp_utc": timestamp})
    started = time.perf_counter()
    pairs = []
    for index, seed in enumerate(args.seeds):
        for method in ("vanilla", "vara_v2"):
            run = root/f"seed_{seed}"/method
            if (run/"summary.json").exists():
                saved = yaml.safe_load((run/"resolved_config.yaml").read_text())
                expected = {**cfg, "seed": seed}
                if saved != expected:raise ValueError("Completed run protocol mismatch")
                continue
            resume = None
            if run.exists() and any(run.iterdir()):
                if not args.resume:raise FileExistsError(str(run))
                # Preserve every failed/interrupted artifact before resuming.
                archive = root/"interrupted_attempts"/f"seed_{seed}"/method/timestamp
                shutil.copytree(run, archive)
                checkpoints = list((run/"checkpoints").glob("revision_block_*.pt"))
                if checkpoints:resume = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
            local_cfg = deepcopy(cfg);local_cfg["seed"] = seed
            trainer = ADStabilityTrainer(local_cfg, method, run)
            if not (run/"checkpoints"/"initial.pt").exists():
                torch.save({"model_state_dict": trainer._model_snapshot(),
                    "initial_model_parameter_hash":trainer.initial_model_parameter_hash},run/"checkpoints"/"initial.pt")
            if resume:trainer.resume_from(resume)
            np.savez_compressed(run/"permitted_training_pools.npz",
                boundary=trainer.boundary_coordinates.cpu().numpy(), boundary_target=trainer.boundary_targets.cpu().numpy(),
                initial=trainer.initial_coordinates.cpu().numpy(), initial_target=trainer.initial_targets.cpu().numpy(),
                sparse=trainer.sparse_coordinates.cpu().numpy(), sparse_target=trainer.sparse_targets.cpu().numpy(),
                diagnostic_interior=trainer.diagnostic_batch["interior"].cpu().numpy())
            write_json(run/"fairness_manifest.json", {"seed": seed, "method": method, "protocol_revision": "ad_v2_stable_v1",
                "source_commit": identity["source_commit"], "initial_model_parameter_hash": trainer.initial_model_parameter_hash,
                "sparse_sample_hash": trainer.sparse_sample_hash,
                "boundary_pool_sha256": _tensor_hash(trainer.boundary_coordinates, trainer.boundary_targets),
                "initial_pool_sha256": _tensor_hash(trainer.initial_coordinates, trainer.initial_targets),
                "diagnostic_pool_sha256": _tensor_hash(*trainer.diagnostic_batch.values()),
                "final_state_rule": "last_committed_state_no_LBFGS_no_best_selection", "controller_reference_metrics_enabled": False,
                "sparse_construction": f"{cfg['training']['n_sparse_data']} continuous uniform manufactured-solution labels; primary nominal 2%, not a grid subset or CFD",
                "actual_sparse_count":cfg["training"]["n_sparse_data"],
                "shared_optimizer_safeguard": cfg["ad_stability_revision"]["armijo_step_safeguard"],
                "committed_budget": cfg["controller_v2"]["total_steps"], "resolved_config_sha256": hashlib.sha256((run/"resolved_config.yaml").read_bytes()).hexdigest()})
            metrics = trainer.run()
            print(json.dumps({"seed": seed, "method": method, "metrics": metrics}), flush=True)
        pairs.append(verify_pair(root, seed))
        elapsed = time.perf_counter()-started
        print(f"Completed pairs {index+1}/{len(args.seeds)} ({100*(index+1)/len(args.seeds):.1f}%), estimated remaining {elapsed/(index+1)*(len(args.seeds)-index-1):.1f}s", flush=True)
    write_json(root/"verified_pairs.json", pairs)
    print(root, flush=True)

if __name__ == "__main__":main()
