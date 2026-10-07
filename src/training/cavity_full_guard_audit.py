"""Fail-closed pair validation and measured secondary work matching."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.controllers.v2_controller import V2Candidate
from src.training.cavity_full_guard_revision import RevisionTrainer, state_hash, file_hash, validate_protocol, GUARDS
from src.utils.io import save_json


def compare_initial(left, right):
    return {"initial_model": left.initial_hash == right.initial_hash,
            "sparse_dataset": left.sparse_manifest == right.sparse_manifest,
            "parameter_count": sum(p.numel() for p in left.model.parameters()) ==
                               sum(p.numel() for p in right.model.parameters()),
            "initial_batch": state_hash(left.initial_batch()) == state_hash(right.initial_batch())}


def audit(output):
    output = Path(output)
    checks = []
    def check(name, condition, details=""):
        checks.append({"check": name, "passed": bool(condition), "details": details})
    manifests = []
    for method in ("vanilla", "vara_v2_full_guard"):
        path = output / method / "revision_manifest.json"
        check(method + "_completed", path.exists())
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        manifests.append(manifest)
        try:
            validate_protocol(manifest["config"])
            check(method + "_resolved_protocol", True)
        except (ValueError, KeyError) as exc:
            check(method + "_resolved_protocol", False, str(exc))
        check(method + "_success", manifest["completed"])
        check(method + "_budget", manifest["compute"]["committed_steps"] == 4000)
        for key in ("sparse_polish", "repair", "restore_best", "early_stopping", "controller_reference_access"):
            check(method + "_" + key + "_off", manifest[key] is False)
        checkpoint = output / method / "checkpoints" / "final.pt"
        check(method + "_checkpoint_bytes", file_hash(checkpoint) == manifest["final_checkpoint_sha256"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        check(method + "_final_model", state_hash(payload["model_state"]) == manifest["final_model_sha256"])
        check(method + "_checkpoint_step", payload["epoch"] == 4000)
        arrays = np.load(output / method / "sparse_dataset.npz")
        for name in ("selected_indices", "coordinates", "u_targets", "v_targets"):
            check(method + "_" + name + "_bytes", state_hash(arrays[name]) == manifest["sparse"][name + "_sha256"])
        check(method + "_source_bytes", file_hash(manifest["sparse"]["source_path"]) == manifest["sparse"]["source_cfd_sha256"])
    if len(manifests) == 2:
        a, b = manifests
        for key in ("initial_model_sha256", "parameter_count", "sparse", "boundary_schedule_hashes", "trajectory_batch_schedule", "final_state_rule"):
            check("paired_" + key, a[key] == b[key])
        for section in ("model", "optimizer", "training", "benchmark_params", "validation", "test", "sampling", "evaluation"):
            check("paired_" + section, a["config"][section] == b["config"][section])
        decisions = json.loads((output / "vara_v2_full_guard" / "revision_decisions.json").read_text())
        proposals = json.loads((output / "vara_v2_full_guard" / "revision_proposals.json").read_text())
        vanilla_decisions = json.loads((output / "vanilla" / "revision_decisions.json").read_text())
        check("vanilla_no_adaptation", not vanilla_decisions and not a["guard_active"])
        check("vara_guard_config_active", b["guard_active"])
        check("vara_diagnostics_executed", b["compute"]["diagnostic_evaluations"] > 0)
        check("vara_screen_executed", b["compute"]["gradient_screen_evaluations"] > 0)
        # All-prefilter outcomes are valid negative evidence, not fabricated probes.
        check("proposal_coverage", len(proposals) >= len(decisions))
        pairs = json.loads((output / "vara_v2_full_guard" / "probe_pairs.json").read_text())
        check("counterfactual_start_identity", all(p["neutral_start_sha256"] == p["action_start_sha256"]
              and p["rng_allocation_state_sha256"] == p["restored_rng_allocation_sha256"] for p in pairs))
        check("counterfactual_accounting", len(pairs) * b["config"]["controller_v2"]["probe_steps"] ==
              b["compute"]["action_probe_steps"])
        check("guard_input_allowlist", all(set(d["guard_neutral"]) == set(GUARDS) and
              set(d["guard_candidate"]) == set(GUARDS) for d in decisions if not d.get("prefiltered")))
        check("accepted_guard_conditions", all(d["observed_target_improvement"] > .005 and
              all(change <= .02 for change in d["guard_changes"].values())
              for d in decisions if d.get("accepted")))
        check("rollback_identity", all(d["rollback_model_sha256"] == d["neutral_model_sha256"]
              for d in decisions if d.get("rollback_executed")))
        check("compute_identity", all(m["compute"]["total_optimizer_calls"] ==
              m["compute"]["committed_steps"] + m["compute"]["discarded_branch_steps"] for m in manifests))
        secondary_path = output / "vanilla_compute_matched" / "revision_manifest.json"
        if (output / "vanilla_compute_matched").exists():
            check("secondary_completed", secondary_path.exists())
            if secondary_path.exists():
                secondary = json.loads(secondary_path.read_text())
                plan = control_plan(output)
                check("secondary_initialization", secondary["initial_model_sha256"] == a["initial_model_sha256"])
                check("secondary_sparse_dataset", secondary["sparse"] == a["sparse"])
                for key in ("gradient_screen_evaluations", "gradient_backpropagations"):
                    check("secondary_" + key, secondary["compute"][key] == plan[key])
                check("secondary_optimizer_calls", secondary["compute"]["total_optimizer_calls"] == plan["optimizer_steps"])
                check("secondary_objective_evaluations", secondary["compute"]["objective_evaluations"] == b["compute"]["objective_evaluations"])
                checkpoint = output / "vanilla_compute_matched" / "checkpoints" / "final.pt"
                check("secondary_checkpoint_bytes", file_hash(checkpoint) == secondary["final_checkpoint_sha256"])
    result = {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks}
    save_json(result, output / "fairness_audit.json")
    pd.DataFrame(checks).to_csv(output / "fairness_audit.csv", index=False)
    for item in checks:
        print(("PASS" if item["passed"] else "FAIL") + " " + item["check"])
    return result


def control_plan(output):
    output = Path(output)
    manifest = json.loads((output / "vara_v2_full_guard" / "revision_manifest.json").read_text())
    work = manifest["compute"]
    return {"matching_quantity": "optimizer calls plus identical gradient-screen objective/backprop call counts",
        "optimizer_steps": work["total_optimizer_calls"],
        "gradient_screen_evaluations": work["gradient_screen_evaluations"],
        "gradient_backpropagations": work["gradient_backpropagations"],
        "source_manifest_sha256": file_hash(output / "vara_v2_full_guard" / "revision_manifest.json"),
        "lr_policy": "same 4000-step schedule; floor LR held for additional steps",
        "limitations": "Operation-count match, not FLOP or wall-clock equality; diagnostic work remains separately reported.",
        "command": ["python", "scripts/run_cavity_full_guard_revision.py", "--device", manifest["config"]["device"],
                    "--output_dir", str(output), "--compute_matched_control"]}


def run_control(output, config, plan):
    output = Path(output)
    from scripts.run_cavity_full_guard_revision import provenance
    environment = provenance()
    original_provenance = json.loads((output / "provenance.json").read_text())
    if environment["git_status"] or environment["revision_sha"] != original_provenance["revision_sha"]:
        raise RuntimeError("Secondary control requires the clean original revision checkout")
    root = output / "vanilla_compute_matched"
    if root.exists():
        raise FileExistsError(root)
    c = deepcopy(config)
    total = plan["optimizer_steps"]
    c["revision_full_guard"].update(primary_steps=total, secondary_control=True)
    c["controller_v2"].update(total_steps=total, control_blocks=math.ceil((total - 800) / 200))
    c["compute_budget"]["value"] = total
    c["experiments"] = {"root": str(root), "flat_layout": True}
    trainer = RevisionTrainer(c, "vanilla_compute_matched")
    primary = json.loads((output / "vanilla" / "revision_manifest.json").read_text())
    assert trainer.initial_hash == primary["initial_model_sha256"]
    assert trainer.sparse_manifest == primary["sparse"]
    save_json({"source_vara_manifest_sha256": plan["source_manifest_sha256"],
               "command": plan["command"], "environment": environment}, root / "control_provenance.json")
    # Replay the fixed-size screening work without applying allocations or updating parameters.
    proposals = json.loads((output / "vara_v2_full_guard" / "revision_proposals.json").read_text())
    for block in sorted({p["block"] for p in proposals}):
        candidates = [V2Candidate(**{key: p[key] for key in V2Candidate.__dataclass_fields__})
                      for p in proposals if p["block"] == block]
        trainer._candidate_influence(candidates)
    assert trainer.work["gradient_screen_evaluations"] == plan["gradient_screen_evaluations"]
    assert trainer.work["gradient_backpropagations"] == plan["gradient_backpropagations"]
    manifest = trainer.run()
    assert manifest["compute"]["total_optimizer_calls"] == total
    save_json(plan, root / "compute_matched_control_manifest.json")
