"""Package every completed revision pair without training or changing input files."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
from scripts.run_ad_v2_stability import verify_pair
from scripts.verify_ad_stability_zip import verify
from src.pde_generalization.benchmarks import build_benchmark
from src.pde_generalization.models import build_pde_model,model_parameter_hash

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot(root):
    paths=list(root.rglob("*"))
    if any(p.is_symlink() for p in paths):raise ValueError("Symlink in inputs")
    return {p.relative_to(root).as_posix():digest(p) for p in paths if p.is_file()}

def statistics(v,a):
    v=np.asarray(v,dtype=float);a=np.asarray(a,dtype=float)
    d=v-a
    percent=np.divide(100*d,np.abs(v),out=np.full_like(d,np.nan),where=v!=0)
    indices=np.random.default_rng(20261010).integers(0,len(d),(20000,len(d)))
    ci=np.quantile(d[indices].mean(1),[.025,.975])
    pci=np.quantile(percent[indices].mean(1),[.025,.975]) if np.isfinite(percent).all() else [np.nan,np.nan]
    nz=d[d!=0];ranks=pd.Series(abs(nz)).rank(method="average").to_numpy()
    signed=np.array([0.])
    for rank in ranks:signed=np.concatenate([signed+rank,signed-rank])
    plus=ranks[nz>0].sum();minus=ranks[nz<0].sum()
    p=float(np.mean(abs(signed)>=abs(plus-minus)-1e-12))
    sd=d.std(ddof=1) if len(d)>1 else np.nan
    return percent,{"n_pairs":len(d),"vanilla_mean":v.mean(),"vanilla_std":v.std(ddof=1) if len(v)>1 else np.nan,
        "vanilla_median":np.median(v),"vara_mean":a.mean(),"vara_std":a.std(ddof=1) if len(a)>1 else np.nan,"vara_median":np.median(a),
        "mean_paired_difference":d.mean(),"mean_paired_improvement_percent":percent.mean(),"wins":int((d>0).sum()),"losses":int((d<0).sum()),
        "ties":int((d==0).sum()),"cohen_dz":d.mean()/sd if sd>0 else np.nan,"difference_ci95_low":ci[0],"difference_ci95_high":ci[1],
        "improvement_ci95_low":pci[0],"improvement_ci95_high":pci[1],"exact_wilcoxon_T":min(plus,minus),"exact_wilcoxon_p":p,
        "statistics_rule":"20,000 paired percentile bootstrap resamples, RNG 20261010; two-sided exact conditional signed-rank enumeration, zero differences dropped, tied average ranks"}

def figures(package,raw,paired,aggregate,seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42,"savefig.dpi":350})
    folder=package/"figures";data=folder/"data";data.mkdir()
    def save(fig,name):
        for ext in ("pdf","svg","png"):fig.savefig(folder/f"{name}.{ext}",bbox_inches="tight")
        plt.close(fig)
    primary="advdiff_u_rel_l2"
    selection=paired[paired.metric==primary].sort_values("seed").copy()
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True);ax.bar(selection.seed,selection.paired_improvement_percent,color=np.where(selection.paired_improvement_percent>=0,"#286b87","#bf493d"))
    ax.axhline(0,color="black",lw=.8);ax.set(xlabel="Seed",ylabel="Paired full-field L2 improvement (%)",title="AD stability revision: all paired outcomes")
    save(fig,"paired_seed_improvement")
    fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    for i,row in enumerate(aggregate.itertuples()):
        ax.plot([row.improvement_ci95_low,row.improvement_ci95_high],[i,i],color="0.3");ax.scatter(row.mean_paired_improvement_percent,i,color="#286b87")
    ax.axvline(0,color="black",lw=.8);ax.set_yticks(range(len(aggregate)),aggregate.metric);ax.set_xlabel("Mean paired improvement (%), bootstrap 95% CI")
    save(fig,"aggregate_ci")
    selection["distance_to_median"]=(selection.paired_improvement_percent-selection.paired_improvement_percent.median()).abs()
    seed=int(selection.sort_values(["distance_to_median","seed"]).iloc[0].seed)
    (package/"provenance/representative_seed.json").write_text(json.dumps({"seed":seed,"rule":"Closest to median paired primary improvement, tie smallest seed","candidates":selection.to_dict("records")},indent=2))
    cfg=yaml.safe_load((raw/f"seed_{seed}/vanilla/resolved_config.yaml").read_text());equation=build_benchmark(cfg)
    models={}
    for method in ("vanilla","vara_v2"):
        model=build_pde_model(cfg).cpu().eval()
        checkpoint=torch.load(raw/f"seed_{seed}/{method}/checkpoints/final.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"]);models[method]=model
    nx,ny=cfg["evaluation"]["nx"],cfg["evaluation"]["ny"]
    x=torch.linspace(0,1,nx);y=torch.linspace(0,1,ny);xx,yy=torch.meshgrid(x,y,indexing="ij")
    times=[0.,.5,1.];panels=[];fields=[];profiles=[]
    for t in times:
        coords=torch.stack([xx,yy,torch.full_like(xx,t)],-1).reshape(-1,3)
        with torch.no_grad():
            reference=equation.exact(coords).numpy().ravel();predictions={m:model(coords).numpy().ravel() for m,model in models.items()}
        fields.append(pd.DataFrame({"x":coords[:,0],"y":coords[:,1],"t":t,"reference":reference,**predictions}))
        panels.append((reference,predictions))
        line=torch.stack([torch.linspace(0,1,200),torch.full((200,),.5),torch.full((200,),t)],1)
        with torch.no_grad():profiles.append(pd.DataFrame({"x":line[:,0],"y":.5,"t":t,"reference":equation.exact(line).numpy().ravel(),**{m:model(line).numpy().ravel() for m,model in models.items()}}))
    pd.concat(fields).to_csv(data/"representative_fields.csv",index=False,float_format="%.17g")
    profiles=pd.concat(profiles);profiles.to_csv(data/"centerline_profiles.csv",index=False,float_format="%.17g")
    (data/"traceability.json").write_text(json.dumps({"seed":seed,"checkpoint_sources":[f"raw/seed_{seed}/{m}/checkpoints/final.pt" for m in models],
        "reference_source":"provenance/source_snapshot.zip: src/pde_generalization/benchmarks.py","inference_only":True,"stored_metrics_replaced":False},indent=2))
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for ax,t in zip(axes,times):
        subset=profiles[profiles.t==t]
        for name in ("reference","vanilla","vara_v2"):ax.plot(subset.x,subset[name],label=name)
        ax.set(xlabel="x",ylabel="u(x, 0.5, t)",title=f"t={t:g}")
    axes[0].legend();save(fig,"centerline_comparison")
    fig,axes=plt.subplots(5,3,figsize=(12,16),constrained_layout=True)
    # Shared scales include every prediction value, including negatives and
    # overshoots; never clamp the color scale to the analytical range.
    all_fields=[values for reference,pred in panels for values in [reference,*pred.values()]]
    vmin=min(float(np.min(values)) for values in all_fields)
    vmax=max(float(np.max(values)) for values in all_fields)
    emin=max(float(np.max(abs(values-reference))) for reference,pred in panels for values in pred.values())
    for col,(reference,pred) in enumerate(panels):
        values=[reference,pred["vanilla"],pred["vara_v2"],abs(pred["vanilla"]-reference),abs(pred["vara_v2"]-reference)]
        for row,(label,field) in enumerate(zip(["Reference","Vanilla","VARA","Vanilla absolute error","VARA absolute error"],values)):
            im=axes[row,col].pcolormesh(xx,yy,field.reshape(nx,ny),shading="auto",cmap="magma" if row>=3 else "viridis",vmin=0 if row>=3 else vmin,vmax=emin if row>=3 else vmax,rasterized=True)
            axes[row,col].set_title(f"{label}, t={times[col]:g}");fig.colorbar(im,ax=axes[row,col],shrink=.7)
    save(fig,"representative_field_errors")
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
    for seed in seeds:
        v=json.loads((raw/f"seed_{seed}/vanilla/summary.json").read_text())["metrics"]
        a=json.loads((raw/f"seed_{seed}/vara_v2/summary.json").read_text())["metrics"]
        ax.plot([v[primary],a[primary]],[v["advdiff_pde_residual_mean"],a["advdiff_pde_residual_mean"]],color="0.6")
        ax.scatter([v[primary],a[primary]],[v["advdiff_pde_residual_mean"],a["advdiff_pde_residual_mean"]],c=["#286b87","#bf493d"])
        ax.annotate(str(seed),(a[primary],a["advdiff_pde_residual_mean"]))
    ax.set(xlabel="Full-field relative L2",ylabel="Mean absolute PDE residual",title="Blue safeguarded Vanilla; red stability VARA")
    save(fig,"reconstruction_physics_tradeoff")

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--results",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--console-logs",nargs="*",default=[])
    args=parser.parse_args();raw=Path(args.results).resolve();package=Path(args.output).resolve();archive=package.with_suffix(".zip")
    if package==raw or raw in package.parents or package in raw.parents:raise ValueError("Package must be separate from input results")
    if package.exists() or archive.exists():raise FileExistsError("Fresh package destination required")
    before=snapshot(raw);identity=json.loads((raw/"study_identity.json").read_text());seeds=identity["seeds"]
    if not 1<=len(seeds)<=20:raise ValueError("Supported exact signed-rank cohort size is 1-20")
    fairness=[verify_pair(raw,s,write=False) for s in seeds]
    for name in ("configs","provenance","fairness","statistics","tables","figures","controller_logs","checkpoints","checksums"): (package/name).mkdir(parents=True,exist_ok=True)
    shutil.copytree(raw,package/"raw")
    for log in args.console_logs:
        log=Path(log);target=package/"provenance/console_logs"/log.name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise ValueError("Duplicate console log filename")
        shutil.copy2(log,target)
    values=[];compute=[];metrics={}
    for seed in seeds:
        for method in ("vanilla","vara_v2"):
            run=raw/f"seed_{seed}"/method;summary=json.loads((run/"summary.json").read_text());metrics[seed,method]=summary["metrics"]
            initial=torch.load(run/"checkpoints/initial.pt",map_location="cpu",weights_only=True)
            config=yaml.safe_load((run/"resolved_config.yaml").read_text());model=build_pde_model(config)
            model.load_state_dict(initial["model_state_dict"])
            if model_parameter_hash(model)!=summary["initial_model_parameter_hash"]:raise ValueError("Saved initial checkpoint hash mismatch")
            manifest=json.loads((run/"fairness_manifest.json").read_text())
            shutil.copy2(run/"resolved_config.yaml",package/"configs"/f"seed_{seed}_{method}.yaml")
            shutil.copy2(run/"fairness_manifest.json",package/"fairness"/f"seed_{seed}_{method}.json")
            for name in ("step_safeguard_audit.json","continuation_guard_audit.json","all_channel_proposal_audit.json","vara_v2_decisions.csv","vara_v2_allocation_history.json"):
                if (run/name).exists():shutil.copy2(run/name,package/"controller_logs"/f"seed_{seed}_{method}_{name}")
            for metric,value in summary["metrics"].items():values.append({"seed":seed,"method":method,"metric":metric,"value":value,"source":f"raw/seed_{seed}/{method}/summary.json","json_key":f"metrics.{metric}"})
            compute.append({"seed":seed,"method":method,**{k:summary["metrics"][k] for k in ("applied_optimizer_steps","optimizer_step_calls","objective_evaluation_count","diagnostic_evaluation_count","optimization_wall_clock_sec","safeguard_objective_evaluations","continuation_replay_calls","noop_physical_parameter_steps","continuation_rollback_count")},"source":f"raw/seed_{seed}/{method}/summary.json"})
    pd.DataFrame(values).to_csv(package/"tables/all_seed_raw_values.csv",index=False)
    pd.DataFrame(compute).to_csv(package/"tables/runtime_compute.csv",index=False)
    pd.DataFrame(fairness).to_csv(package/"fairness/paired_fairness.csv",index=False)
    paired=[];aggregate=[]
    for metric in [k for k in metrics[seeds[0],"vanilla"] if k.startswith("advdiff_")]+["optimization_wall_clock_sec"]:
        v=np.array([metrics[s,"vanilla"][metric] for s in seeds]);a=np.array([metrics[s,"vara_v2"][metric] for s in seeds])
        if not np.isfinite(v).all() or not np.isfinite(a).all():raise ValueError("Cannot discard nonfinite outcomes")
        percent,stats=statistics(v,a);aggregate.append({"metric":metric,**stats})
        for s,vi,ai,pi in zip(seeds,v,a,percent):paired.append({"seed":s,"metric":metric,"vanilla":vi,"vara_v2":ai,"paired_difference":vi-ai,"paired_improvement_percent":pi,
            "vanilla_source":f"raw/seed_{s}/vanilla/summary.json","vara_source":f"raw/seed_{s}/vara_v2/summary.json","json_key":f"metrics.{metric}"})
    paired=pd.DataFrame(paired);aggregate=pd.DataFrame(aggregate)
    paired.to_csv(package/"tables/per_seed_paired_results.csv",index=False,float_format="%.17g")
    aggregate.to_csv(package/"statistics/paired_statistics.csv",index=False,float_format="%.17g")
    aggregate.to_csv(package/"tables/main_aggregate_results.csv",index=False,float_format="%.17g")
    paired[paired.paired_difference<0].to_csv(package/"tables/negative_seed_results.csv",index=False,float_format="%.17g")
    aggregate[aggregate.losses>0].to_csv(package/"tables/negative_tradeoff_metrics.csv",index=False,float_format="%.17g")
    figures(package,raw,paired,aggregate,seeds)
    source=package/"provenance/source_snapshot.zip"
    subprocess.run(["git","-C",str(ROOT),"archive","--format=zip",f"--output={source}",identity["source_commit"]],check=True)
    for name,sha in identity["source_hashes_lf"].items():
        src=ROOT/name
        if hashlib.sha256(src.read_bytes().replace(b"\r\n",b"\n")).hexdigest()!=sha:raise ValueError("Scientific source changed since training")
        target=package/"provenance/effective_source"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    (package/"provenance/source_hashes_lf.json").write_text(json.dumps(identity["source_hashes_lf"],indent=2))
    shutil.copy2(ROOT/"scripts/verify_ad_stability_zip.py",package/"verify_ad_stability_zip.py")
    (package/"checkpoints/README.md").write_text("All initial, final and complete-block checkpoints, including interrupted attempts, are retained under raw/. No duplicate binary files are needed here.\n")
    (package/"README.md").write_text(f"""# AD stability revision evidence

Protocol: ad_v2_stable_v1. Cohort: {seeds}. Study kind: {identity['protocol'].get('study_kind','PRIMARY_GPU')}.
Scientific question: does reference-free allocation improve AD beyond the shared safeguarded optimizer?
Both arms use identical architecture, conditions, sparse manufactured labels, initialization and committed-step budget. Final state only; no L-BFGS, reference gating or best selection.
Raw inputs are copied byte-for-byte to raw/. Every raw number is visible in tables/all_seed_raw_values.csv. Negative outcomes remain in tables/negative_seed_results.csv and negative_tradeoff_metrics.csv.
Primary: full-field relative L2; secondary: layer, sparse, PDE, BC and IC metrics; runtime is descriptive. Statistics are exploratory low-N paired analyses, unadjusted p-values; percentages with zero Vanilla denominators are undefined, never imputed.
Figures use saved final checkpoints and evaluation-only analytic references; figures/data retains plotted values. Representative seed is chosen by a declared median-distance rule. Inference creates no training updates or replacement scores.
Guard rechecks compare against the same-block neutral 25-step anchor; they are conservative safety tests, not duration-matched causal probes. Checks/replays add compute; tables/runtime_compute.csv reports it. Checkpoint resume restores the latest completed block. Interrupted attempts are retained under raw/interrupted_attempts; their abandoned wall time/calls are not included in completed-trajectory counters.
Sparse data are 507 continuous uniform analytical labels in the primary protocol, with RNG seed +30003; nominal 2% is relative to 48x48x11 evaluation-grid cardinality, not an evaluation-grid subset and not CFD.
Known limits: minibatch descent does not ensure generalization, Adam moments advance for damped/no-op parameter steps, conservative guards can reject useful proposals, changed scoring/calibration needs fresh paired validation. Do not attribute a shared optimizer improvement solely to VARA.
accepted_interventions counts initial 25-step probe passes; continuation_rollback_count counts subsequent reversals and block_end_retained_interventions subtracts these. Accepted step tests can still produce an exactly zero parameter update; zero_physical_parameter_updates records actual zero updates separately.
Tables map directly to per-seed summary.json paths and JSON keys. Configs map to copied resolved_config.yaml; fairness to per-run manifests; mechanism evidence to controller_logs and raw audits; source and invocations to provenance and raw/invocation_*.json. All checkpoints remain in raw/.
Run python verify_ad_stability_zip.py PACKAGE.zip to independently verify byte integrity and recorded paired-budget consistency.
""",encoding="utf-8")
    checks=snapshot(package);(package/"checksums/SHA256.json").write_text(json.dumps(checks,indent=2))
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(package.rglob("*")):
            if path.is_file():z.write(path,path.relative_to(package).as_posix())
    print(json.dumps(verify(archive),indent=2))
    if snapshot(raw)!=before:raise RuntimeError("Input results changed during packaging")
    print(f"ZIP: {archive}\nSHA256: {digest(archive)}")

if __name__=="__main__":main()
