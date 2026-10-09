"""Independent reviewer verifier: no trainer/controller imports or retraining.

Supports a run directory, an extracted supplement, or a ZIP. Tensor checks use
PyTorch's restricted weights_only loader; all ZIP payloads are hashed before use.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import zipfile

GUARDS={"pde_residual_mean","continuity_residual_mean","momentum_residual_mean",
        "boundary_condition_error","unweighted_validation_loss","unweighted_physics_validation_loss"}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
def state_hash(values):
    # Same published canonical state schema, implemented independently here.
    import torch
    def enc(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous().numpy()
            return {"shape":list(a.shape),"dtype":str(a.dtype),"bytes":hashlib.sha256(a.tobytes()).hexdigest()}
        if isinstance(v,dict):return {str(k):enc(v[k]) for k in sorted(v,key=str)}
        if isinstance(v,(tuple,list)):return [enc(x) for x in v]
        return v
    return hashlib.sha256(json.dumps(enc(values),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def verify_run(path,strict=True):
    path=Path(path)
    needed=["summary.json","COMPLETE.json","resolved_config.json","resolved_config.yaml","permitted_observations.npz",
            "evaluation_fields.npz","events.jsonl","losses.csv","diagnostics.json","decisions.jsonl","allocation_history.json",
            "collocation_history.npz","checkpoints/final.pt","checkpoints/complete.pt"]
    for n in needed:require((path/n).is_file(),f"Missing {path/n}")
    s=json.loads((path/"summary.json").read_text());cfg=json.loads((path/"resolved_config.json").read_text())
    marker=json.loads((path/"COMPLETE.json").read_text())
    require(marker["summary_sha256"]==sha(path/"summary.json"),"Summary checksum mismatch")
    require(marker["final_checkpoint_sha256"]==sha(path/"checkpoints/final.pt"),"Checkpoint checksum mismatch")
    total=cfg["controller_v2"]["total_steps"]
    core={k:v for k,v in cfg.items() if k not in {"seed","device","experiments"}}
    require(hashlib.sha256(json.dumps(core,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()==s["protocol_hash"],"Resolved protocol fingerprint mismatch")
    require(s["committed_steps"]==total,"Incomplete committed schedule")
    require(s["seed"]==cfg["seed"] and s["mode"] in {"vanilla","vara_v2"},"Run identity mismatch")
    require(s["controller_reference_metrics_enabled"] is False and s["reference_gate_active"] is True,"Reference isolation declaration absent")
    require(s["final_state_rule"]=="final_committed_adam_state","Asymmetric final selection")
    require(set(cfg["controller_v2"]["guard_metrics"])==GUARDS,"Guard whitelist changed")
    loss=list(csv.DictReader((path/"losses.csv").open(newline="",encoding="utf-8")))
    require([int(r["step"]) for r in loss]==list(range(1,total+1)),"Retained loss trajectory has gaps/duplicates")
    for r in loss:require(all(math.isfinite(float(v)) for k,v in r.items() if v and k not in {"phase"}),"Nonfinite retained loss/gradient")
    events=[];truncated=0
    for journal in sorted(path.glob("events*.jsonl")):
        lines=journal.read_text(encoding="utf-8").splitlines()
        for i,line in enumerate(lines):
            try:events.append(json.loads(line))
            except json.JSONDecodeError:
                require(i==len(lines)-1,"Corrupt non-tail event record");truncated+=1
    steps=[r for r in events if r["kind"]=="optimizer_step"]
    gradients=[r for r in events if r["kind"]=="gradient_probe"]
    require(s["optimizer_calls"]==len(steps),"Physical optimizer call accounting mismatch")
    require(s["objective_evaluations"]==len(steps)+len(gradients),"Objective accounting mismatch")
    require(s["controller_gradient_evaluations"]==2*len(gradients),"Gradient accounting mismatch")
    require(s["training_points_evaluated"]==sum(r["points"] for r in steps),"Training point accounting mismatch")
    require(abs(s["optimization_seconds"]-sum(r["duration_seconds"] for r in steps))<1e-7,"Optimization timing accounting mismatch")
    intent={r["token"] for r in events if r["kind"]=="optimizer_intent"};done={r["token"] for r in steps}
    require(len(done)==len(steps),"Repeated physical-step token")
    require(s["uncertain_optimizer_calls_upper"]==len(intent-done)+truncated,"Uncertain-call accounting mismatch")
    if strict:require(not intent-done and not truncated,"An interrupted optimizer intent has uncertain work; exact-compute badge refused")
    diagnostic=[r for r in events if r["kind"]=="diagnostic"]
    require(s["diagnostic_evaluations"]==len(diagnostic) and s["diagnostic_points_evaluated"]==sum(r["points"] for r in diagnostic),"Diagnostic accounting mismatch")
    allocations=json.loads((path/"allocation_history.json").read_text())
    require(len(allocations)==cfg["controller_v2"]["control_blocks"]+1,"Missing allocation history")
    for a in allocations:
        m=a["sampling_mass"]
        require(abs(sum(m)-1)<1e-8 and min(m)>=0 and max(m)<=.25+1e-9,"Sampling mass/bounds violated")
        for w in a["loss_multipliers"].values():require(abs(sum(w)/len(w)-1)<1e-8 and min(w)>=.5-1e-9 and max(w)<=2+1e-9,"Multiplier constraints violated")
    decisions=rows(path/"decisions.jsonl");tested=[d for d in decisions if not d.get("prefiltered",False)]
    require(len(tested)<=cfg["controller_v2"]["control_blocks"],"More than one tested candidate per block")
    require(len({d["block"] for d in tested})==len(tested),"Repeated tested block")
    for d in tested:
        require(d["comparison_mode"]=="counterfactual" and d["probe_steps"]==cfg["controller_v2"]["probe_steps"],"Not a matched probe")
        require(set(d["neutral_metrics"])==GUARDS and set(d["candidate_metrics"])==GUARDS,"Unexpected/held-out guard quantity")
        target=(d["neutral_target"]-d["candidate_target"])/(abs(d["neutral_target"])+1e-12)
        require(abs(target-d["observed_target_improvement"])<1e-9,"Target improvement mismatch")
        for key in GUARDS:
            change=(d["candidate_metrics"][key]-d["neutral_metrics"][key])/(abs(d["neutral_metrics"][key])+1e-12)
            require(abs(change-d["guard_changes"][key])<1e-9,"Guard computation mismatch")
        should=target>cfg["controller_v2"]["counterfactual_target_margin"] and all(v<=cfg["controller_v2"]["counterfactual_guard_margin"] for v in d["guard_changes"].values())
        require(d["accepted"]==should,"Acceptance mismatch")
        expected=d["candidate_state_hash"] if d["accepted"] else d["neutral_state_hash"]
        require(d["restored_state_hash"]==expected and d["restoration_verified"],"Incorrect branch restoration")
    if s["mode"]=="vara_v2" and not cfg["publication"]["disabled_candidates"]:
        require(len([r for r in events if r["kind"]=="ranked_candidates"])>=cfg["controller_v2"]["control_blocks"],"Controller never diagnosed/ranked all blocks")
    require(s["accepted"]==sum(d["accepted"] for d in tested),"Accepted count mismatch")
    require(s["rejected"]==sum(not d["accepted"] for d in tested),"Rejected count mismatch")
    if not any(r["kind"]=="resume" for r in events):
        require(s["optimizer_calls"]==total+len(tested)*cfg["controller_v2"]["probe_steps"],"Probe-call count mismatch")
    for key in ["u_rel_l2","v_rel_l2","p_rel_l2_centered","omega_rel_l2","velocity_rel_l2","pde_residual_mean","boundary_training_mse"]:
        require(isinstance(s["metrics"].get(key),(int,float)) and math.isfinite(s["metrics"][key]),f"Missing/invalid metric {key}")
    if strict:
        import torch
        final=torch.load(path/"checkpoints/final.pt",map_location="cpu",weights_only=True)
        require(final["global_step"]==total,"Checkpoint final step mismatch")
        require(all(float(v["step"])==total for v in final["optimizer"]["state"].values()),"Adam state not at committed budget")
        complete=torch.load(path/"checkpoints/complete.pt",map_location="cpu",weights_only=True)
        require(complete["global_step"]==total,"Incomplete resume checkpoint")
        require(state_hash(final["model"])==state_hash(complete["runtime"]["model"]),"Final/resume model mismatch")
        require(state_hash(complete["pools"])==s["permitted_pool_hash"],"Permitted-pool hash mismatch")
        initial=torch.load(path/"checkpoints/complete_a0_step00000.pt",map_location="cpu",weights_only=True)
        require(state_hash(initial["runtime"]["model"])==s["initial_model_hash"],"Recorded initialization differs from saved initial state")
        import numpy as np
        with np.load(path/"permitted_observations.npz",allow_pickle=False) as observations:
            require(set(observations.files)==set(complete["pools"]),"Pool member names changed")
            for name in observations.files:
                require(np.array_equal(observations[name],complete["pools"][name].numpy()),"Observation file differs from checkpoint")
        with np.load(path/"evaluation_fields.npz",allow_pickle=False) as fields:
            require(state_hash(torch.tensor(fields["coordinates"]))==s["metrics"]["evaluation_grid_hash"],"Grid fingerprint mismatch")
            pred=fields["prediction"];ref=fields["reference"]
            independent=float(np.linalg.norm(pred[:,:2]-ref[:,:2])/np.linalg.norm(ref[:,:2]))
            require(abs(independent-s["metrics"]["velocity_rel_l2"])<1e-7,"Primary raw-field metric mismatch")
            state=final["model"]
            count=sum(v.numel() for k,v in state.items() if k.endswith(("weight","bias")))
            require(count==s["parameter_count"],"Model parameter-count mismatch")
            x=torch.tensor(fields["coordinates"],dtype=torch.float32)
            x=2*(x-state["input_lower"])/(state["input_upper"]-state["input_lower"]).clamp_min(1e-12)-1
            layer_ids=sorted(int(k.split(".")[1]) for k in state if k.startswith("layers.") and k.endswith(".weight"))
            require(len(layer_ids)==len(cfg["model"]["hidden_layers"])+1,"Unexpected checkpoint architecture")
            with torch.no_grad():
                for i in layer_ids:
                    x=torch.nn.functional.linear(x,state[f"layers.{i}.weight"],state[f"layers.{i}.bias"])
                    if i!=layer_ids[-1]:x=torch.tanh(x)
            require(np.allclose(x.numpy(),pred,atol=2e-5,rtol=2e-5),"Stored field differs from independent final-checkpoint inference")
    return {"valid":True,"seed":s["seed"],"mode":s["mode"],"smoke":s["smoke"],"committed_steps":total,"optimizer_calls":len(steps)}

def verify_zip(path,strict=False):
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None,"ZIP CRC failed")
        require(len(z.namelist())==len(set(z.namelist())),"Duplicate ZIP entries")
        listed=set()
        for line in z.read("checksums/SHA256SUMS").decode().splitlines():
            expected,name=line.split("  ",1);require(name not in listed,"Duplicate checksum entry");listed.add(name)
            require(hashlib.sha256(z.read(name)).hexdigest()==expected,f"Payload checksum failed: {name}")
        require(listed==set(z.namelist())-{"checksums/SHA256SUMS"},"Checksum manifest does not cover all payloads")
        manifest=json.loads(z.read("manifest.json"))
        for f in manifest["files"]:
            data=z.read(f["path"]);require(len(data)==f["size"] and hashlib.sha256(data).hexdigest()==f["sha256"],"Manifest mismatch")
        if strict:
            for name in z.namelist():
                parts=Path(name).parts
                require(not Path(name).is_absolute() and ".." not in parts and ":" not in name and "\\" not in name,"Unsafe ZIP member path")
            with tempfile.TemporaryDirectory(prefix="kova_v2_peer_verify_") as temp:
                z.extractall(temp)
                root=Path(temp);summaries=[]
                for folder in sorted((root/"raw").glob("seed_*/*")):
                    if folder.is_dir():verify_run(folder,True);summaries.append(json.loads((folder/"summary.json").read_text()))
                require(summaries,"Missing raw runs")
                expected=[999] if manifest["study_type"].startswith("SMOKE") else [0,1,2,3,4]
                require(sorted({s["seed"] for s in summaries})==expected and len(summaries)==2*len(expected),"Incomplete ZIP seed/method study")
                for seed in expected:
                    pair=[s for s in summaries if s["seed"]==seed]
                    require({s["mode"] for s in pair}=={"vanilla","vara_v2"},"Missing paired arm")
                    for key in ["initial_model_hash","permitted_pool_hash","protocol_hash","committed_steps","runtime"]:
                        require(pair[0][key]==pair[1][key],"ZIP pair mismatch")
                for s in summaries:
                    for name,value in s["source"]["scientific_files"].items():require(sha(root/"source"/name)==value,"Archived scientific source mismatch")
                tests=root/"provenance/tests/junit.xml"
                if not manifest["study_type"].startswith("SMOKE"):
                    require(tests.exists(),"Missing scientific test evidence")
                    tree=ET.parse(tests);require(not tree.findall(".//failure") and not tree.findall(".//error"),"Archived tests failed")
        return {"valid":True,"zip_crc":True,"payloads_hashed":len(listed),"study_type":manifest["study_type"],"scientific_run_checks":strict}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-dir");p.add_argument("--zip");p.add_argument("--strict",action="store_true")
    a=p.parse_args()
    if a.zip:report=verify_zip(a.zip,a.strict)
    elif a.run_dir:report=verify_run(a.run_dir,a.strict)
    else:p.error("--run-dir or --zip required")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
