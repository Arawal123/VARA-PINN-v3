"""Seed8 causal diagnostic on unchanged historical PDE trainer/controller.

Only retention is forced neutral. Original sampler, eligibility, probes and
proposed-decision trust/memory updates remain. Dense metrics run after training.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
from src.pde_generalization.trainer import PDEGeneralizationTrainer,_tensor_hash
from src.pde_generalization.models import model_parameter_hash
from src.pde_generalization.losses import compute_training_loss

HISTORICAL="cdb27c8be0e681654d8c6c3005d9658c0ae6ee73"
EXPECTED_ARCHIVE_SHA="07986bed095e25f4f9190e9009461af447ac9e08b60293c629b9e09a3c2d5701"
FILES=["src/pde_generalization/trainer.py","src/controllers/v2_controller.py","src/pde_generalization/diagnostics.py",
       "src/pde_generalization/benchmarks.py","src/pde_generalization/residuals.py","src/pde_generalization/losses.py",
       "src/pde_generalization/metrics.py","src/models/mlp.py","src/pde_generalization/models.py"]

def clean(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    return v

def state_hash(v):
    def enc(a):
        if isinstance(a,torch.Tensor):
            n=a.detach().cpu().contiguous().numpy();return [str(n.dtype),list(n.shape),hashlib.sha256(n.tobytes()).hexdigest()]
        if isinstance(a,np.ndarray):return [str(a.dtype),list(a.shape),hashlib.sha256(a.tobytes()).hexdigest()]
        if isinstance(a,dict):return {str(k):enc(a[k]) for k in sorted(a,key=str)}
        if isinstance(a,(list,tuple)):return [enc(x) for x in a]
        return clean(a)
    return hashlib.sha256(json.dumps(enc(v),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def dump(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")

def load_reference(path):
    p=Path(path);sha=hashlib.sha256(p.read_bytes()).hexdigest()
    if sha!=EXPECTED_ARCHIVE_SHA:raise ValueError("Expected verified historical15-seed archive, not the five-seed ZIP or a newer experiment")
    with zipfile.ZipFile(p) as z:
        source=zipfile.ZipFile(io.BytesIO(z.read("source_snapshot/source.zip")))
        assert source.comment.decode()==HISTORICAL
        for name in FILES:
            assert (ROOT/name).read_bytes().replace(b"\r\n",b"\n")==source.read(name).replace(b"\r\n",b"\n"),f"Historical scientific source changed:{name}"
        prefix="raw/seed_8/vara_v2/"
        cfg=yaml.safe_load(z.read(prefix+"resolved_config.yaml"))
        summary=json.loads(z.read(prefix+"summary.json"))
        expected={key:summary[key] for key in ["git_commit","initial_model_parameter_hash","sparse_sample_hash"]}
        assert expected["git_commit"]==HISTORICAL and cfg["seed"]==8
        losses=pd.read_csv(io.BytesIO(z.read(prefix+"losses.csv")),float_precision="round_trip")
        with np.load(io.BytesIO(z.read("fairness/frozen_data_seed_8.npz"))) as data:
            pools={k:data[k].copy() for k in ["coordinates","targets"]}
        provenance=json.loads(z.read("raw/run_provenance.json"))
    return cfg,expected,losses,pools,{"archive_sha256":sha,"path":str(p),"historical_runtime":{k:provenance.get(k) for k in ["torch","cuda","gpu","python"]}}

class ForcedNeutralTrainer(PDEGeneralizationTrainer):
    def __init__(self,cfg,path,expected=None,reference_losses=None,frozen_pool=None,parity_sampler=False):
        super().__init__(cfg,"vara_v2",path)
        self.expected=expected;self.reference_losses=reference_losses;self.parity_sampler=parity_sampler
        self.probe_records=[];self.ranked_records=[];self.batch_records=[];self.neutral_runtime_hash=None
        self.stage="initial";self.warmup_verified=False;self.counterfactual_initial_hash=None
        if frozen_pool is not None:
            self.sparse_coordinates=torch.tensor(frozen_pool["coordinates"],device=self.device,dtype=self.dtype)
            self.sparse_targets=torch.tensor(frozen_pool["targets"],device=self.device,dtype=self.dtype)
            self.sparse_sample_hash=_tensor_hash(self.sparse_coordinates,self.sparse_targets)
            self.diagnostic_batch["sparse"]=self.sparse_coordinates;self.diagnostic_batch["sparse_target"]=self.sparse_targets
        if expected:
            assert self.initial_model_parameter_hash==expected["initial_model_parameter_hash"],"Historical initialization mismatch"
            assert self.sparse_sample_hash==expected["sparse_sample_hash"],"Historical observation hash mismatch"
        original_evaluate=self.controller.evaluate
        def forced_evaluate(*args,**kwargs):
            proposed,decision=original_evaluate(*args,**kwargs)
            record=deepcopy(decision)
            record.update(proposed_accepted=bool(proposed),retained_accepted=False,retained_branch="neutral",
                          policy_update_rule="original_proposed_decision_shadow_updates")
            self.probe_records.append(record)
            decision={**decision,"proposed_accepted":bool(proposed),"accepted":False,"retained_branch":"neutral",
                      "forced_diagnostic":True,"policy_update_rule":"original_proposed_decision_shadow_updates"}
            return False,decision
        self.controller.evaluate=forced_evaluate
        original_rank=self.controller.rank
        def ranked(candidates,influence):
            result=original_rank(candidates,influence)
            self.ranked_records.append({"step":self.applied_optimizer_steps,"candidates":[x.to_record() for x in result],"influence":influence})
            return result
        self.controller.rank=ranked
        self._save_state("initial_step0000",0)

    def runtime_hash(self):
        return state_hash({"model":self.model.state_dict(),"adam":self.optimizer.state_dict(),
            "allocation":self.controller.state.snapshot(),"sampling_rng":self.sampling_rng.bit_generator.state})

    def _sample_adaptive(self,count):
        if self.parity_sampler:return self._sample_uniform(count,self.sampling_rng)
        return super()._sample_adaptive(count)

    def _training_batch(self,*,adaptive):
        b=super()._training_batch(adaptive=adaptive)
        self.batch_records.append({"step":self.applied_optimizer_steps,"stage":self.stage,"adaptive":adaptive,
            "interior_hash":state_hash(b["interior"]),"permitted_pool_hash":state_hash({k:v for k,v in b.items() if k!="interior"}),
            "rng_after":state_hash(self.sampling_rng.bit_generator.state)})
        return b

    def _save_state(self,name,step,batch=None):
        payload={"step":step,"stage":self.stage,"model":self._model_snapshot(),"adam":deepcopy(self.optimizer.state_dict()),
            "allocation":clean(self.controller.state.snapshot()),"sampling_rng":deepcopy(self.sampling_rng.bit_generator.state),
            "trust_radius":self.controller.trust_radius,"effectiveness":deepcopy(self.controller.effectiveness),
            "code_base":HISTORICAL,"runtime_hash":self.runtime_hash(),"batch":None if batch is None else {k:v.detach().cpu() for k,v in batch.items()}}
        torch.save(payload,self.run_dir/"checkpoints"/(name+".pt"))

    def _train_steps(self,batch,steps,phase):
        # Same optimizer/loss sequence as historical trainer: no repair/tuning.
        self.stage=phase;rows=[];probe="probe" in phase
        if probe:
            if "neutral" in phase:self.counterfactual_initial_hash=self.runtime_hash()
            if "candidate" in phase:
                # Candidate allocation differs deliberately; model/Adam/RNG
                # equality is recorded via complete state snapshots.
                pass
            self._save_state(phase+"_before",self.applied_optimizer_steps,batch)
        for i in range(steps):
            self.model.train();self.optimizer.zero_grad(set_to_none=True)
            result=compute_training_loss(self.model,self.benchmark,batch,self.weights,self.patch_grid,self.controller.state)
            self.objective_evaluation_count+=1
            if not torch.isfinite(result.total):raise FloatingPointError("Nonfinite historical objective")
            result.total.backward()
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),float(self.config["training"]["gradient_clip"]))
            old=[p.detach().clone() for p in self.model.parameters()]
            before_step=self.applied_optimizer_steps+i
            if not probe and before_step in {3998,3999}:self._save_state(f"retained_step{before_step:04d}",before_step,batch)
            self.optimizer.step();self.optimizer_step_calls+=1
            delta=float(torch.sqrt(sum((p.detach()-b).square().sum() for p,b in zip(self.model.parameters(),old))))
            rows.append({"local_step":i+1,"phase":phase,"loss_total":float(result.total.detach().cpu()),
                **{f"loss_{k}":float(v.detach().cpu()) for k,v in result.components.items()},
                "raw_gradient_norm":float(norm),"clip_active":bool(norm>self.config["training"]["gradient_clip"]),
                "parameter_update_l2":delta})
            if not probe and ((before_step+1)%250==0 or before_step+1==self.config["controller_v2"]["total_steps"]):
                self._save_state(f"retained_step{before_step+1:04d}",before_step+1,batch)
                print("@@PROGRESS "+json.dumps({"step":before_step+1,"total_steps":self.config["controller_v2"]["total_steps"],"phase":phase,"optimizer_calls":self.optimizer_step_calls}),flush=True)
        if probe:
            self._save_state(phase+"_after",self.applied_optimizer_steps+steps,batch)
            if "neutral" in phase:self.neutral_runtime_hash=self.runtime_hash()
        return rows

    def _record_decision(self,block,candidate,decision):
        if not decision.get("prefiltered",False):
            restored=self.runtime_hash()
            assert restored==self.neutral_runtime_hash,"Forced retention failed exact neutral model/Adam/allocation/RNG equality"
            decision={**decision,"neutral_runtime_hash":self.neutral_runtime_hash,"restored_runtime_hash":restored,"restoration_verified":True}
            self.probe_records[-1].update(block=block,candidate=candidate.to_record(),neutral_runtime_hash=self.neutral_runtime_hash,restored_runtime_hash=restored,restoration_verified=True)
        super()._record_decision(block,candidate,decision)

    def _diagnose(self):
        s=super()._diagnose()
        record={"step":self.applied_optimizer_steps,"stage":self.stage,"names":s.names,"raw_scores":s.raw_scores,"normalized_scores":s.normalized_scores,"guard_metrics":self._controller_metrics(s)}
        with (self.run_dir/"diagnostic_snapshots.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(clean(record),allow_nan=False)+"\n")
        return s

    def _commit_rows(self,rows):
        super()._commit_rows(rows)
        if rows and rows[0]["phase"]=="warmup":
            columns=["loss_total","loss_pde","loss_bc","loss_ic","loss_sparse_data"]
            if self.reference_losses is not None:
                actual=np.asarray([[r[k] for k in columns] for r in self.loss_rows])
                expected=self.reference_losses.iloc[:len(actual)][columns].to_numpy()
                error=float(np.max(abs(actual-expected)))
                self.warmup_verified=bool(np.array_equal(actual,expected))
                dump(self.run_dir/"historical_warmup_check.json",{"bit_identical":self.warmup_verified,"max_absolute_difference":error,"steps":len(actual)})
                if not self.warmup_verified:raise RuntimeError("Archived warmup did not reproduce exactly; historical causal comparison is not clean. Stopped before remaining3500 steps.")
            else:self.warmup_verified=True

    def _log_allocation(self,block):
        super()._log_allocation(block)
        state=self.controller.state
        assert np.allclose(state.sampling_mass,1/state.num_patches,atol=0,rtol=0)
        assert all(np.all(x==1) for x in state.loss_multipliers.values()),"A retained intervention escaped forced-neutral control"

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--reference-zip",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--preflight",action="store_true")
    a=p.parse_args();cfg,expected,losses,pool,provenance=load_reference(a.reference_zip)
    report={"historical_source":HISTORICAL,"seed":8,"local_cuda_available":torch.cuda.is_available(),"reference":provenance,
            "estimated_original_t4_optimization_seconds":112.11,"diagnostic_gpu_outcome":"PENDING"}
    if a.preflight:print(json.dumps(report,indent=2));return
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable: use the supplied Colab launcher. No incomparable CPU primary run is permitted.")
    out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError("Choose a fresh output directory; diagnostics never overwrite historical or existing outputs.")
    cfg["device"]="cuda"
    t=ForcedNeutralTrainer(cfg,out,expected,losses,pool)
    dump(out/"diagnostic_identity.json",{**report,"arm":"forced_neutral_original_vara_sampler","policy":"shadow normal trust/memory updates; forced neutral optimizer/model/allocation/RNG retention",
        "actual_runtime":{"torch":str(torch.__version__),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)},
        "diagnostic_source_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()})
    try:metrics=t.run()
    finally:
        dump(out/"proposed_and_retained_decisions.json",t.probe_records);dump(out/"all_ranked_candidates.json",t.ranked_records);dump(out/"batch_hash_history.json",t.batch_records)
    assert t.accepted_interventions==0 and t.applied_optimizer_steps==4000
    assert t.optimizer_step_calls==4000+25*len(t.probe_records)
    with zipfile.ZipFile(a.reference_zip) as z:
        archived={method:json.loads(z.read(f"raw/seed_8/{method}/summary.json"))["metrics"]["advdiff_u_rel_l2"] for method in ["vanilla","vara_v2"]}
    dump(out/"seed8_comparison.json",{"status":"COMPLETE_DIAGNOSTIC","warmup_bit_identical":t.warmup_verified,
        "original_vanilla_l2":archived["vanilla"],"original_full_vara_l2":archived["vara_v2"],"forced_neutral_l2":metrics["advdiff_u_rel_l2"],
        "proposed_acceptances":sum(d["proposed_accepted"] for d in t.probe_records),"retained_adaptive_actions":0,
        "committed_steps":4000,"actual_optimizer_calls":t.optimizer_step_calls,
        "interpretation":"One-seed necessity test only; not population causality or a replacement of original outcomes."})
    print(json.dumps(json.loads((out/"seed8_comparison.json").read_text()),indent=2))
if __name__=="__main__":main()
