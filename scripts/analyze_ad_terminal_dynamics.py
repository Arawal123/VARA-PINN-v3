"""New terminal-gradient/null-mode measurements from existing checkpoints only.

Reuses the earlier inverse-Adam and deterministic last-batch construction.
Never trains, selects historical states, or saves corrected model checkpoints.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import types
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
from src.pde_generalization.models import build_pde_model
from src.pde_generalization.benchmarks import build_benchmark
from src.pde_generalization.metrics import make_evaluation_grid
from src.pde_generalization.trainer import PDEGeneralizationTrainer,_tensor_hash
from src.pde_generalization.diagnostics import PDEPatchGrid
from src.pde_generalization.losses import compute_training_loss
from src.pde_generalization.residuals import compute_residuals

def inverse_adam(model,payload):
    # Existing validated recovery formula: no invented gradients/optimizer call.
    before=deepcopy(model);parameters=list(before.parameters());offset=0
    with torch.no_grad():
        for group in payload["optimizer_state_dict"]["param_groups"]:
            assert group.get("weight_decay",0)==0 and not group.get("amsgrad",False)
            b1,b2=group["betas"]
            for key in group["params"]:
                state=payload["optimizer_state_dict"]["state"][key];step=float(state["step"])
                delta=group["lr"]/(1-b1**step)*state["exp_avg"]/(state["exp_avg_sq"].sqrt()/np.sqrt(1-b2**step)+group["eps"])
                parameters[offset].add_(delta);offset+=1
    return before

def final_batch(archive,cfg,seed,method,prefix):
    equation=build_benchmark(cfg)
    ctx=types.SimpleNamespace(benchmark=equation,device=torch.device("cpu"),dtype=torch.float32,
        patch_grid=PDEPatchGrid.from_config(cfg,equation),sampling_rng=np.random.default_rng(seed+10003))
    for name in ["_sample_uniform_numpy","_sample_uniform","_sample_adaptive"]:
        setattr(ctx,name,types.MethodType(getattr(PDEGeneralizationTrainer,name),ctx))
    n=cfg["training"]["n_collocation"]
    interior=ctx._sample_uniform(n,ctx.sampling_rng);allocation=None
    if method=="vara_v2":
        ctx.controller=types.SimpleNamespace(config=types.SimpleNamespace(min_uniform_mass=cfg["controller_v2"]["min_uniform_mass"]),state=None)
        for entry in json.loads(archive.read(prefix+"vara_v2_allocation_history.json")):
            if entry["block"]<0:continue
            ctx.controller.state=types.SimpleNamespace(sampling_mass=np.asarray(entry["sampling_mass"]),
                loss_multipliers={k:np.asarray(v) for k,v in entry["loss_multipliers"].items()},global_multipliers=entry.get("global_multipliers",{}))
            interior=ctx._sample_adaptive(n)
        allocation=ctx.controller.state
    else:
        for _ in range(cfg["controller_v2"]["control_blocks"]):interior=ctx._sample_uniform(n,ctx.sampling_rng)
    rng=np.random.default_rng(seed+20003)
    bc=PDEGeneralizationTrainer._sample_boundary(ctx,cfg["training"]["n_boundary"],rng)
    ic=PDEGeneralizationTrainer._sample_initial(ctx,cfg["training"]["n_initial"],rng)
    with np.load(io.BytesIO(archive.read(f"fairness/frozen_data_seed_{seed}.npz"))) as pool:
        sparse=torch.from_numpy(pool["coordinates"].copy());targets=torch.from_numpy(pool["targets"].copy())
    with torch.no_grad():bt=equation.boundary_values(bc);it=equation.initial_values(ic)
    batch={"interior":interior,"boundary":bc,"boundary_target":bt,"initial":ic,"initial_target":it,"sparse":sparse,"sparse_target":targets}
    return equation,ctx.patch_grid,batch,allocation

def flat_grad(loss,params):
    values=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p,g in zip(params,values)])

def measure(archive,seed,method):
    prefix=f"raw/seed_{seed}/{method}/"
    cfg=yaml.safe_load(archive.read(prefix+"resolved_config.yaml"));summary=json.loads(archive.read(prefix+"summary.json"))
    assert summary["git_commit"]=="cdb27c8be0e681654d8c6c3005d9658c0ae6ee73"
    payload=torch.load(io.BytesIO(archive.read(prefix+"checkpoints/final.pt")),map_location="cpu",weights_only=True)
    model=build_pde_model(cfg).eval();model.load_state_dict(payload["model_state_dict"]);pre=inverse_adam(model,payload)
    equation,patches,batch,allocation=final_batch(archive,cfg,seed,method,prefix)
    assert _tensor_hash(batch["sparse"],batch["sparse_target"])==summary["sparse_sample_hash"]
    before=compute_training_loss(pre,equation,batch,cfg["training"]["weights"],patches,allocation)
    after=compute_training_loss(model,equation,batch,cfg["training"]["weights"],patches,allocation)
    logged=pd.read_csv(io.BytesIO(archive.read(prefix+"losses.csv")),float_precision="round_trip").iloc[-1]
    reconstruction=max(abs(float(value.detach())-logged["loss_"+key]) for key,value in before.components.items())
    assert reconstruction<1e-6,"New quantity not evaluated on recovered historical batch/state"
    params=list(pre.parameters());g=flat_grad(before.total,params)
    delta=torch.cat([(a.detach()-b.detach()).reshape(-1) for a,b in zip(model.parameters(),pre.parameters())])
    gradient_norm=float(g.norm());update_norm=float(delta.norm())
    gd=float(torch.dot(g,delta));cos=float(torch.dot(g,-delta)/(g.norm()*delta.norm()+1e-30))
    label=cfg["benchmark"];row={"benchmark":label,"seed":seed,"method":method,
        "gradient_norm":gradient_norm,"clip_active":gradient_norm>cfg["training"]["gradient_clip"],
        "clip_factor":min(1.,cfg["training"]["gradient_clip"]/(gradient_norm+1e-30)),
        "update_norm":update_norm,"update_relative_parameter_norm":update_norm/float(torch.cat([p.detach().reshape(-1) for p in params]).norm()),
        "gradient_dot_update":gd,"descent_alignment_cosine":cos,
        "fixed_batch_loss_before":float(before.total.detach()),"fixed_batch_loss_after":float(after.total.detach()),
        "logged_loss_reconstruction_error":reconstruction,"source_member":prefix+"checkpoints/final.pt"}
    components=[]
    for key,value in before.components.items():
        cg=flat_grad(cfg["training"]["weights"][key]*value,params)
        components.append({"benchmark":label,"seed":seed,"method":method,"component":key,
            "weighted_gradient_norm":float(cg.norm()),"weighted_gradient_dot_update":float(torch.dot(cg,delta)),
            "loss_before":float(value.detach()),"loss_after":float(after.components[key].detach())})
    # Prespecified shift: minimize ORIGINAL allowed-data quadratic, not held-out error.
    numerator=denominator=0.
    with torch.no_grad():
        for coord,target,name in [("boundary","boundary_target","bc"),("initial","initial_target","ic"),("sparse","sparse_target","sparse_data")]:
            e=(model(batch[coord])-batch[target]).reshape(-1)
            mult=np.ones(len(e)) if allocation is None or name not in allocation.loss_multipliers else allocation.loss_multipliers[name][patches.assign_numpy(batch[coord].numpy())]
            m=torch.tensor(mult,dtype=torch.float32);w=cfg["training"]["weights"][name]
            numerator+=w*float((m*e).mean());denominator+=w*float(m.mean())
        shift=-numerator/denominator
        grid=make_evaluation_grid(equation,48,48,11,device=torch.device("cpu"),dtype=torch.float32)
        truth=equation.exact(grid);pred=model(grid);previous=pre(grid);error=pred-truth;change=pred-previous
        total_sq=float(error.square().sum());constant_sq=len(grid)*float(error.mean())**2
        corrected=error+shift
        row.update(reference_rms=float(truth.square().mean().sqrt()),field_error_mean=float(error.mean()),
            constant_error_fraction=constant_sq/max(total_sq,1e-30),last_field_shift_mean=float(change.mean()),
            last_field_shift_std=float(change.std()),last_update_constant_fraction=len(grid)*float(change.mean())**2/max(float(change.square().sum()),1e-30),
            training_only_constant_shift=shift,original_l2=summary["metrics"]["advdiff_u_rel_l2" if label=="advection_diffusion" else "allen_cahn_u_rel_l2"],
            diagnostic_shifted_l2=float(corrected.norm()/truth.norm()),
            diagnostic_l2_percent_change=100*(float(corrected.norm()/error.norm())-1),
            allowed_data_objective_decrease=denominator*shift**2,
            final_output_bias_update=float(list(model.parameters())[-1].detach()-list(pre.parameters())[-1].detach()))
    # Fixed reference-free interior diagnostic pool, not dense evaluation data.
    ctx=types.SimpleNamespace(benchmark=equation)
    coords=PDEGeneralizationTrainer._sample_uniform_numpy(ctx,1024,np.random.default_rng(seed+40003))
    coordinates=torch.tensor(coords,dtype=torch.float32)
    residual=compute_residuals(model,coordinates,equation)
    signed=residual["f_advdiff" if label=="advection_diffusion" else "f_ac"].detach()
    with torch.no_grad():u=model(coordinates)
    change_r=torch.zeros_like(signed) if label=="advection_diffusion" else shift*(3*u.square()-1)+3*u*shift**2+shift**3
    row.update(fixed_diagnostic_pde_before=float(signed.abs().mean()),fixed_diagnostic_pde_after_shift=float((signed+change_r).abs().mean()),
               pde_operator_shift_max=float(change_r.abs().max()))
    ray=[]
    if label=="advection_diffusion" and ((method=="vara_v2" and seed in [1,3,5,8,18,2]) or (method=="vanilla" and seed==4)):
        # Fixed diagnostic fractions, declared before measurement. No choice of
        # alpha enters training, repair, checkpoint selection or published data.
        for alpha in [0.,.25,.5,.75,1.]:
            probe=deepcopy(pre)
            with torch.no_grad():
                for p,b,a in zip(probe.parameters(),pre.parameters(),model.parameters()):p.copy_(b+alpha*(a-b))
            loss=compute_training_loss(probe,equation,batch,cfg["training"]["weights"],patches,allocation)
            with torch.no_grad():field=probe(grid);field_l2=float((field-truth).norm()/truth.norm())
            ray.append({"benchmark":label,"seed":seed,"method":method,"fraction_of_archived_last_update":alpha,
                "fixed_training_objective":float(loss.total.detach()),"offline_full_field_l2":field_l2,
                **{"loss_"+k:float(v.detach()) for k,v in loss.components.items()}})
    return row,components,ray

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--ad-five",required=True);p.add_argument("--ad-fifteen",required=True);p.add_argument("--allen-cahn",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    paths={"ad5":Path(a.ad_five),"ad15":Path(a.ad_fifteen),"ac":Path(a.allen_cahn)}
    archives={k:zipfile.ZipFile(v) for k,v in paths.items()};results=[];components=[];rays=[]
    requests=[("ad5" if s<5 else "ad15",s,"vara_v2") for s in [1,3,5,8,18,2,4]]
    requests += [("ad5" if s<5 else "ad15",s,"vanilla") for s in [1,4,8,18]]
    requests += [("ac",s,"vara_v2") for s in [0,1,2,4]]
    for archive,seed,method in requests:
        row,parts,ray=measure(archives[archive],seed,method);results.append(row);components.extend(parts);rays.extend(ray)
        print(row["benchmark"],seed,method,"alignment",round(row["descent_alignment_cosine"],4),"constant error fraction",round(row["constant_error_fraction"],4),"diagnostic L2 change %",round(row["diagnostic_l2_percent_change"],2),flush=True)
    pd.DataFrame(results).to_csv(out/"terminal_dynamics_and_constant_mode.csv",index=False,float_format="%.17g")
    pd.DataFrame(components).to_csv(out/"terminal_component_gradients.csv",index=False,float_format="%.17g")
    pd.DataFrame(rays).to_csv(out/"fixed_fraction_terminal_step_probe.csv",index=False,float_format="%.17g")
    (out/"provenance.json").write_text(json.dumps({"input_archives":{k:{"path":str(v),"sha256":hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in paths.items()},
        "research_training_executed":False,"optimizer_step_calls":0,"new_quantities_not_prior_audit_revalidation":True,"torch":str(torch.__version__),
        "shift_rule":"original weighted BC/IC/sparse quadratic minimizer; no dense target used to choose shift","outputs":"diagnostic only; original checkpoints and metrics unchanged"},indent=2))
if __name__=="__main__":main()
