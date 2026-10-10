"""Independent standard-library-only ZIP integrity and paired-budget verifier."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import PurePosixPath
import zipfile

def verify(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError("Duplicate archive members")
        for name in names:
            p=PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name:raise ValueError("Unsafe archive member")
        checks=json.loads(archive.read("checksums/SHA256.json"))
        expected=set(names)-{"checksums/SHA256.json"}
        if set(checks)!=expected:raise ValueError("Checksum coverage mismatch")
        for name,digest in checks.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError(f"Checksum mismatch: {name}")
        identity=json.loads(archive.read("raw/study_identity.json"))
        for seed in identity["seeds"]:
            summaries=[];manifests=[]
            for method in ("vanilla","vara_v2"):
                prefix=f"raw/seed_{seed}/{method}/"
                summary=json.loads(archive.read(prefix+"summary.json"));manifest=json.loads(archive.read(prefix+"fairness_manifest.json"))
                if summary["protocol_revision"]!="ad_v2_stable_v1":raise ValueError("Protocol mixing")
                if summary["git_commit"]!=identity["source_commit"] or manifest["source_commit"]!=identity["source_commit"]:raise ValueError("Source SHA mismatch")
                if manifest["controller_reference_metrics_enabled"]:raise ValueError("Reference gating enabled")
                if hashlib.sha256(archive.read(prefix+"resolved_config.yaml")).hexdigest()!=manifest["resolved_config_sha256"]:raise ValueError("Resolved config hash mismatch")
                metrics=summary["metrics"];steps=identity["protocol"]["controller_v2"]["total_steps"]
                if metrics["applied_optimizer_steps"]!=steps:raise ValueError("Committed-step budget mismatch")
                rows=list(csv.DictReader(io.StringIO(archive.read(prefix+"losses.csv").decode())))
                if [int(r["step"]) for r in rows]!=list(range(1,steps+1)):raise ValueError("Retained loss-log steps mismatch")
                audit=json.loads(archive.read(prefix+"step_safeguard_audit.json"))
                if identity["protocol"]["ad_stability_revision"]["armijo_step_safeguard"]:
                    if len(audit)!=metrics["optimizer_step_calls"]:raise ValueError("Physical Adam calls/audit mismatch")
                    if any(r["loss_after"]>r["loss_before"] for r in audit):raise ValueError("Step safeguard violated")
                    if sum(r["parameter_noop"] for r in audit)!=metrics["noop_physical_parameter_steps"]:raise ValueError("No-op count mismatch")
                for name,value in metrics.items():
                    if name.startswith("advdiff_") and not math.isfinite(value):raise ValueError("Nonfinite primary/secondary metric retained")
                for file in ("checkpoints/initial.pt","checkpoints/final.pt","permitted_training_pools.npz"):
                    if prefix+file not in names:raise ValueError(f"Missing {prefix+file}")
                summaries.append(summary);manifests.append(manifest)
            for key in ("initial_model_parameter_hash","sparse_sample_hash"):
                if summaries[0][key]!=summaries[1][key]:raise ValueError(f"Paired {key} mismatch")
            for key in ("boundary_pool_sha256","initial_pool_sha256","diagnostic_pool_sha256","final_state_rule","resolved_config_sha256"):
                if manifests[0][key]!=manifests[1][key]:raise ValueError(f"Paired {key} mismatch")
        return {"verified_files":len(checks),"verified_pairs":len(identity["seeds"]),"source_commit":identity["source_commit"],
                "scope":"Byte integrity, recorded fairness, log/budget/safeguard consistency; no claim about scientific superiority"}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("zip")
    print(json.dumps(verify(parser.parse_args().zip),indent=2))
