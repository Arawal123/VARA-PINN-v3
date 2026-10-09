"""Small deterministic fixtures only: no primary study or GPU optimization."""
from copy import deepcopy
import ast
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.run_kovasznay_v2_publication import effective_config
from scripts.verify_kovasznay_v2_publication import verify_run,verify_zip
from scripts.package_kovasznay_v2_publication import build
from src.controllers.v2_controller import VARAV2Controller,V2ControllerConfig,V2Candidate
from src.physics.kovasznay import KovasznayFlow
from src.physics.navier_stokes import navier_stokes_residuals
from src.training.kovasznay_v2_publication import KovasznayV2PublicationTrainer,digest,ROOT

torch.set_num_threads(2)

def config():
    return effective_config(ROOT/"configs/kovasznay_v2_publication.yaml","cpu",999,True)

def trainer(tmp_path,mode="vara_v2",cfg=None,name=None,resume=False):
    return KovasznayV2PublicationTrainer(cfg or config(),mode,tmp_path/(name or mode),resume)

def test_reference_isolation(tmp_path,monkeypatch):
    t=trainer(tmp_path)
    def forbidden(*a,**kw):raise AssertionError("Oracle accessed after observations were frozen")
    monkeypatch.setattr(KovasznayFlow,"exact_np",forbidden)
    monkeypatch.setattr(KovasznayFlow,"exact_torch",forbidden)
    with pytest.raises(RuntimeError,match="inaccessible"):t.benchmark.exact_np(np.zeros((2,2)))
    t.train_protocol()
    assert t.global_step==12 and t.diagnostics
    assert all(set(x["metrics"])==set(config()["controller_v2"]["guard_metrics"]) for x in t.diagnostics)

def test_paired_initialization_and_observations(tmp_path):
    a=trainer(tmp_path,"vanilla");b=trainer(tmp_path,"vara_v2")
    assert a.initial_hash==b.initial_hash and a.pool_hash==b.pool_hash
    assert digest(a.initial_batch())==digest(b.initial_batch())
    assert a.fixed_data.shape==(8,2) and torch.equal(a.fixed_uvp,b.fixed_uvp)
    assert isinstance(b, __import__("src.training.vara_v2_trainer",fromlist=["VARAV2Trainer"]).VARAV2Trainer)

def independent_fields(xy):
    x,y=xy[:,0:1],xy[:,1:2];k=2*np.pi;lam=20-np.sqrt(400+k*k);e=np.exp(lam*x)
    return np.column_stack([(1-e*np.cos(k*y)).ravel(),(lam/k*e*np.sin(k*y)).ravel(),(.5*(1-e*e)).ravel()])

def test_independent_pde_and_derivatives():
    rng=np.random.default_rng(117);points=np.column_stack([rng.uniform(-.49,.99,40),rng.uniform(-.49,1.49,40)])
    class IndependentExact(torch.nn.Module):
        def forward(self,p):
            x,y=p[:,0:1],p[:,1:2];lam=20-np.sqrt(400+4*np.pi**2);e=torch.exp(lam*x)
            return torch.cat([1-e*torch.cos(2*np.pi*y),lam/(2*np.pi)*e*torch.sin(2*np.pi*y),.5*(1-e*e)],1)
    p=torch.tensor(points,dtype=torch.float64)
    r=navier_stokes_residuals(IndependentExact(),p,nu=1/40,steady=True)
    assert max(float(r[k].detach().abs().max()) for k in ["f_u","f_v","f_c"])<1e-10
    b=KovasznayFlow();actual=b.exact_np(points);expected=independent_fields(points)
    assert np.max(abs(np.column_stack([actual[k].ravel() for k in ["u","v","p"]])-expected))<1e-14
    h=1e-5;dx=points.copy();dx[:,0]+=h;dm=points.copy();dm[:,0]-=h
    px=(independent_fields(dx)[:,2]-independent_fields(dm)[:,2])/(2*h)
    assert np.max(abs(px-actual["p_x"].ravel()))<1e-8
    dy=points.copy();dy[:,1]+=h;dym=points.copy();dym[:,1]-=h
    omega=(independent_fields(dx)[:,1]-independent_fields(dm)[:,1])/(2*h)-(independent_fields(dy)[:,0]-independent_fields(dym)[:,0])/(2*h)
    assert np.max(abs(omega-actual["omega"].ravel()))<1e-7
    boundaries=np.array([[-.5,-.5],[1,1.5],[0,-.5],[-.5,0]])
    assert np.max(abs(np.column_stack([b.exact_np(boundaries)[k].ravel() for k in ["u","v","p"]])-independent_fields(boundaries)))<1e-14

def test_neutral_matches_vanilla(tmp_path):
    cfg=config();cfg["publication"]["disabled_candidates"]=True
    a=trainer(tmp_path,"vanilla",cfg);b=trainer(tmp_path,"vara_v2",cfg)
    a.train_protocol();b.train_protocol()
    assert digest(a.model.state_dict())==digest(b.model.state_dict())
    assert digest(a.optimizer.state_dict())==digest(b.optimizer.state_dict())
    assert [r["loss_total"] for r in a.trajectory]==[r["loss_total"] for r in b.trajectory]
    assert a._calls==b._calls==12

def test_counterfactual_is_neutral_comparison(tmp_path,monkeypatch):
    t=trainer(tmp_path);original=t.v2_controller.evaluate;captured=[]
    def observe(candidate,before_target,after_target,before_metrics,after_metrics,**kw):
        captured.append((dict(before_metrics),dict(after_metrics),kw))
        return original(candidate,before_target,after_target,before_metrics,after_metrics,**kw)
    monkeypatch.setattr(t.v2_controller,"evaluate",observe)
    t.train_protocol()
    assert captured
    neutral=[r["metrics"] for r in t.diagnostics if r["phase"]=="neutral"]
    initial=[r["metrics"] for r in t.diagnostics if r["phase"]=="before"]
    assert [r[0] for r in captured]==neutral
    assert all(r[2]["comparison_mode"]=="counterfactual" for r in captured)
    assert neutral[0]!=initial[0]

def test_rejected_probe_restores_neutral(tmp_path):
    t=trainer(tmp_path);t.train_protocol(force_reject=True)
    tested=[d for d in t.decisions if not d.get("prefiltered",False)]
    assert tested and all(not d["accepted"] for d in tested)
    assert all(d["restored_state_hash"]==d["neutral_state_hash"] for d in tested)
    assert all(d["initial_state_hash"]!=d["neutral_state_hash"] for d in tested)
    assert t._calls==12+len(tested)
    for d in tested:assert d["expected_state_hash"]==d["neutral_state_hash"]
    s=t._snapshot();saved=digest(s);t._restore(s)
    batch=t.initial_batch();t._train(batch,1,"test_mutation",False)
    assert digest(s)==saved,"Optimizer restore mutated reusable snapshot"

def test_sampling_conservation():
    ctrl=VARAV2Controller(V2ControllerConfig(num_patches=16))
    for i in range(80):
        candidate=V2Candidate("continuity_residual",i%16,"joint",["continuity"],1,1,0)
        ctrl.apply(candidate);ctrl.validate_state()
        assert abs(ctrl.state.sampling_mass.sum()-1)<1e-12
        assert abs(ctrl.state.multiplier("continuity").mean()-1)<1e-12

def test_optimizer_counts(tmp_path):
    t=trainer(tmp_path);s=t.run();v=verify_run(t.run_dir,True)
    assert v["valid"] and s["committed_steps"]==12
    tested=[d for d in t.decisions if not d.get("prefiltered",False)]
    assert s["optimizer_calls"]==12+len(tested)
    assert all(float(x["step"])==12 for x in t.optimizer.state_dict()["state"].values())

def test_resume_matches_uninterrupted(tmp_path):
    full=trainer(tmp_path,name="uninterrupted");full.train_protocol()
    interrupted=trainer(tmp_path,name="interrupted");interrupted.train_protocol(stop_after_block=0)
    continued=trainer(tmp_path,name="interrupted",resume=True);continued.train_protocol()
    assert digest(full.model.state_dict())==digest(continued.model.state_dict())
    assert digest(full.optimizer.state_dict())==digest(continued.optimizer.state_dict())
    assert [r["loss_total"] for r in full.trajectory]==[r["loss_total"] for r in continued.trajectory]
    assert continued._physical_counts()["optimizer_calls"]==full._physical_counts()["optimizer_calls"]

def test_pressure_gauge_and_centered_evaluation(tmp_path):
    t=trainer(tmp_path);batch=t.initial_batch();a=t.pointwise(batch)
    gauge_before=float(t.pressure_gauge_loss().detach())
    linears=[m for m in t.model.modules() if isinstance(m,torch.nn.Linear)]
    with torch.no_grad():linears[-1].bias[2]+=7.
    b=t.pointwise(batch)
    for key in ["momentum_u","momentum_v","continuity","p"]:assert torch.allclose(a[key],b[key],atol=2e-5,rtol=2e-5)
    assert float(t.pressure_gauge_loss().detach())>gauge_before
    with pytest.raises(AssertionError,match="Evaluation forbidden"):t.evaluate_final()

def test_mean_reduction():
    from src.losses.base_losses import compute_global_losses
    values={"pde":torch.tensor([[1.],[4.]])}
    assert compute_global_losses(values,reduction="mean")["pde"].item()==2.5
    assert compute_global_losses(values,reduction="legacy_mse")["pde"].item()==8.5

def test_verifier_rejects_tampering(tmp_path):
    t=trainer(tmp_path);t.run();path=t.run_dir/"summary.json"
    data=json.loads(path.read_text());data["optimizer_calls"]+=1;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="checksum"):verify_run(t.run_dir,True)

def test_frozen_config_and_cli_contract():
    cfg=effective_config(ROOT/"configs/kovasznay_v2_publication.yaml")
    assert cfg["controller_v2"]["total_steps"]==4000 and cfg["training"]["n_data"]==1000
    result=subprocess.run([sys.executable,"-B",str(ROOT/"scripts/run_kovasznay_v2_publication.py"),"--help"],capture_output=True,text=True)
    assert result.returncode==0 and all(k in result.stdout for k in ["--resume","--preflight","--smoke","--output-dir"])

def test_primary_packager_rejects_incomplete(tmp_path):
    with pytest.raises(ValueError,match="No runs"):build(tmp_path,tmp_path.parent/(tmp_path.name+"_package"),True)

def test_tiny_package_roundtrip(tmp_path):
    root=tmp_path/"input";root.mkdir()
    for mode in ["vanilla","vara_v2"]:
        t=KovasznayV2PublicationTrainer(config(),mode,root/"seed_999"/mode);t.run()
    result=build(root,tmp_path/"analysis",strict=False,allow_smoke=True)
    assert verify_zip(result["zip"])["valid"]
    assert result["status"]=="SMOKE_ONLY_PRIMARY_PENDING"
    import zipfile
    with zipfile.ZipFile(result["zip"]) as z:
        assert "raw/seed_999/vanilla/checkpoints/final.pt" in z.namelist()
        assert "raw/seed_999/vara_v2/events.jsonl" in z.namelist()
        assert "tables/negative_results.csv" in z.namelist()

def test_truncated_tail_resume_preserves_raw_and_refuses_exact_compute(tmp_path):
    first=trainer(tmp_path,name="tail");first.train_protocol(stop_after_block=0)
    with first._journal.open("a",encoding="utf-8") as f:f.write("{truncated")
    before=first._journal.read_bytes()
    resumed=trainer(tmp_path,name="tail",resume=True);result=resumed.run()
    assert first._journal.read_bytes()==before
    assert result["uncertain_optimizer_calls_upper"]==1
    with pytest.raises(ValueError,match="uncertain work"):verify_run(resumed.run_dir,True)

def test_trust_calibration_risk_is_retained():
    ctrl=VARAV2Controller(V2ControllerConfig(num_patches=16))
    candidate=V2Candidate("continuity_residual",0,"sampling",["continuity"],4,1,0,predicted_target_improvement=2.)
    metrics={k:1. for k in config()["controller_v2"]["guard_metrics"]}
    before=ctrl.trust_radius
    accepted,result=ctrl.evaluate(candidate,1.,0.,metrics,metrics,target_threshold=.005,guard_threshold=.02)
    assert accepted and result["reward_ratio"]<.75 and ctrl.trust_radius==before

def test_pure_pinn_sensitivity_has_no_interior_labels(tmp_path):
    cfg=effective_config(ROOT/"configs/kovasznay_v2_publication.yaml","cpu",999,True,True)
    t=trainer(tmp_path,cfg=cfg)
    assert len(t.fixed_data)==0 and len(t.fixed_uvp)==0
    t.train_protocol()
    assert all(not any(name.startswith("sparse_") for name in d["names"]) for d in t.diagnostics)

def test_independent_zip_verifier_rejects_modified_payload(tmp_path):
    import zipfile
    p=tmp_path/"bad.zip"
    with zipfile.ZipFile(p,"w") as z:
        z.writestr("x.txt",b"altered");z.writestr("checksums/SHA256SUMS","0"*64+"  x.txt\n")
    with pytest.raises(ValueError,match="checksum failed"):verify_zip(p)
