"""Build a traceable supplement from complete real paired runs; never train."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.verify_kovasznay_v2_publication import verify_run,verify_zip,require

REQUIRED_TESTS={"test_reference_isolation","test_paired_initialization_and_observations","test_independent_pde_and_derivatives",
                "test_neutral_matches_vanilla","test_counterfactual_is_neutral_comparison","test_rejected_probe_restores_neutral",
                "test_sampling_conservation","test_optimizer_counts","test_resume_matches_uninterrupted",
                "test_pressure_gauge_and_centered_evaluation","test_mean_reduction","test_verifier_rejects_tampering"}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def savefig(fig,out,name):
    metadata={"pdf":{"Creator":"Kovasznay V2 post-training analysis","CreationDate":None,"ModDate":None},
              "svg":{"Creator":"Kovasznay V2 post-training analysis","Date":None},"png":{"Software":"Kovasznay V2 post-training analysis"}}
    for ext in ("pdf","svg","png"):fig.savefig(out/f"{name}.{ext}",bbox_inches="tight",dpi=320,metadata=metadata[ext])
    plt.close(fig)

def validate_tests(input_root):
    xml=input_root/"tests/junit.xml";evidence=input_root/"tests/evidence.json"
    require(xml.exists() and evidence.exists(),"Missing real test output/JUnit/source fingerprints")
    tree=ET.parse(xml);cases=tree.findall(".//testcase")
    require(REQUIRED_TESTS<={c.attrib["name"] for c in cases},"Mandatory scientific regression tests missing")
    require(not tree.findall(".//failure") and not tree.findall(".//error") and not tree.findall(".//skipped"),"Tests failed or were skipped")
    metadata=json.loads(evidence.read_text())
    require(metadata["exit_code"]==0 and metadata["junit_sha256"]==sha(xml),"Test evidence mismatch")
    for name,value in metadata["scientific_files"].items():require(sha(ROOT/name)==value,"Tests used different scientific source")
    return {"passed":len(cases),"required_tests":sorted(REQUIRED_TESTS),"junit_sha256":sha(xml)}

def build(input_root,output_dir,strict=True,allow_smoke=False,profile="prescribed_sparse_primary"):
    input_root=Path(input_root).resolve();output_dir=Path(output_dir).resolve()
    require(strict or allow_smoke,"Primary supplements require --strict and passing scientific tests")
    require(not output_dir.exists(),"Output directory exists; choose a fresh analysis directory")
    require(output_dir!=input_root and input_root not in output_dir.parents,"Analysis must be separate from immutable raw run root")
    seed_dirs=sorted(input_root.glob("seed_*"))
    require(seed_dirs,"No runs")
    seeds=[int(p.name[5:]) for p in seed_dirs]
    expected=[999] if allow_smoke else [0,1,2,3,4]
    require(sorted(seeds)==expected,"Incomplete/unexpected seed set")
    tests=validate_tests(input_root) if strict else {"validation":"not_strict"}
    runs=[];summaries=[]
    for seed in expected:
        paired=[]
        for mode in ("vanilla","vara_v2"):
            folder=input_root/f"seed_{seed}"/mode
            require(set(p.name for p in (input_root/f"seed_{seed}").iterdir())=={"vanilla","vara_v2"},"Unexpected arm or hidden duplicate run")
            report=verify_run(folder,strict)
            s=json.loads((folder/"summary.json").read_text());paired.append(s)
            require(s["seed"]==seed and s["mode"]==mode,"Folder/summary identity mismatch")
            require(s["smoke"]==allow_smoke,"Smoke/primary results cannot be pooled")
            require(s["profile"]==profile,"Sensitivity and primary profiles cannot be pooled")
            if not allow_smoke:
                require(s["committed_steps"]==4000 and s["runtime"]["device"].startswith("cuda"),"Primary protocol requires4000 committed CUDA steps")
                require(not s["source"]["dirty"],"Primary source checkout was dirty")
            for name,value in s["source"]["scientific_files"].items():require(sha(ROOT/name)==value,"Current packaging source differs from run")
            runs.append((seed,mode,folder,s,report));summaries.append({"seed":seed,"mode":mode,**s["metrics"],
                "committed_steps":s["committed_steps"],"optimizer_calls":s["optimizer_calls"],"wall_seconds":s["wall_seconds"],
                "optimization_seconds":s["optimization_seconds"],"diagnostic_evaluations":s["diagnostic_evaluations"],"diagnostic_points_evaluated":s["diagnostic_points_evaluated"],
                "objective_evaluations":s["objective_evaluations"],"controller_gradient_evaluations":s["controller_gradient_evaluations"],
                "training_points_evaluated":s["training_points_evaluated"],"controller_points_evaluated":s["controller_points_evaluated"],
                "peak_cuda_memory_bytes":s["peak_cuda_memory_bytes"],"accepted":s["accepted"],"rejected":s["rejected"],"prefiltered":s["prefiltered"]})
        for key in ("initial_model_hash","permitted_pool_hash","protocol_hash","committed_steps","final_state_rule"):
            require(paired[0][key]==paired[1][key],f"Pair mismatch:seed{seed}/{key}")
        require(paired[0]["runtime"]==paired[1]["runtime"],"Paired hardware/runtime environments differ")
        require(paired[0]["metrics"]["evaluation_grid_hash"]==paired[1]["metrics"]["evaluation_grid_hash"],"Evaluation grid differs")
    require(len({s["source"]["git_commit"] for _,_,_,s,_ in runs})==1,"Runs use different commits")
    output_dir.mkdir(parents=True)
    for folder in ("raw","configs","provenance","fairness","statistics","tables","figures","controller_logs","checkpoints","checksums","source","generated_analysis"):
        (output_dir/folder).mkdir()
    for seed,mode,folder,s,report in runs:
        # Copy everything, including failed attempts, partial logs and all checkpoints.
        shutil.copytree(folder,output_dir/"raw"/f"seed_{seed}"/mode)
        for filename in ("resolved_config.yaml","resolved_config.json"):
            shutil.copy2(folder/filename,output_dir/"configs"/f"seed_{seed}_{mode}_{filename}")
        dump(output_dir/"fairness"/f"seed_{seed}_{mode}.json",{"verification":report,"initial_model_hash":s["initial_model_hash"],
            "permitted_pool_hash":s["permitted_pool_hash"],"protocol_hash":s["protocol_hash"],"source":s["source"],
            "grid_hash":s["metrics"]["evaluation_grid_hash"],"final_state_rule":s["final_state_rule"]})
        dump(output_dir/"provenance"/f"seed_{seed}_{mode}.json",s)
        (output_dir/"checkpoints"/f"seed_{seed}_{mode}.txt").write_text(f"Canonical checkpoint: raw/seed_{seed}/{mode}/checkpoints/final.pt\n",encoding="utf-8")
        if mode=="vara_v2":
            shutil.copy2(folder/"decisions.jsonl",output_dir/"controller_logs"/f"seed_{seed}_decisions.jsonl")
    if (input_root/"tests").exists():shutil.copytree(input_root/"tests",output_dir/"provenance/tests")
    for name in ("preflight.json","environment.json","pip_freeze.txt","launcher_execution_log.json"):
        if (input_root/name).exists():shutil.copy2(input_root/name,output_dir/"provenance"/name)
    for name in ("process_logs","launcher_attempts"):
        if (input_root/name).exists():shutil.copytree(input_root/name,output_dir/"provenance"/name)
    # Archive scientific source AND executable analysis/verifier/notebook sources.
    tracked=subprocess.check_output(["git","ls-files"],cwd=ROOT,text=True).splitlines()
    added=subprocess.check_output(["git","ls-files","--others","--exclude-standard"],cwd=ROOT,text=True).splitlines()
    for name in sorted(set(tracked+added)):
        p=ROOT/name
        if p.is_file() and (name.startswith(("src/","scripts/","configs/","tests/","notebooks/")) or p.name in {"IMPLEMENTATION_PLAN.md","V2_PARITY_MATRIX.md","PROTOCOL_FROZEN.json","KOVASZNAY_V2_EXECUTION_GUIDE.md","requirements.txt","pyproject.toml"}):
            dest=output_dir/"source"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name in ("IMPLEMENTATION_PLAN.md","V2_PARITY_MATRIX.md","PROTOCOL_FROZEN.json"):
        shutil.copy2(ROOT/name,output_dir/name)
    shutil.copy2(ROOT/"PROTOCOL_FROZEN.md",output_dir/"PROTOCOL_FROZEN.md")
    from scripts.make_kovasznay_v2_colab import notebook
    nb_path=output_dir/"source/notebooks/kovasznay_v2_5seed_publication_colab.ipynb"
    nb_path.parent.mkdir(exist_ok=True,parents=True)
    dump(nb_path,notebook(runs[0][3]["source"]["git_commit"]))
    shutil.copy2(ROOT/"scripts/verify_kovasznay_v2_publication.py",output_dir/"verify_package.py")
    frozen=json.loads((ROOT/"PROTOCOL_FROZEN.json").read_text());bootstrap=frozen["profiles"][profile]["config"]["publication"]
    raw=pd.DataFrame(summaries).sort_values(["seed","mode"])
    raw.to_csv(output_dir/"tables/per_seed_metrics.csv",index=False,float_format="%.17g")
    raw.to_csv(output_dir/"raw/all_run_metrics.csv",index=False,float_format="%.17g")
    raw[["seed","mode","committed_steps","optimizer_calls","wall_seconds","optimization_seconds","objective_evaluations","controller_gradient_evaluations","training_points_evaluated","controller_points_evaluated","diagnostic_evaluations","diagnostic_points_evaluated","peak_cuda_memory_bytes"]].to_csv(output_dir/"tables/compute_runtime.csv",index=False)
    reported=["u_rel_l2","v_rel_l2","p_rel_l2_centered","omega_rel_l2","velocity_rel_l2","pde_residual_mean",
              "continuity_residual_mean","momentum_residual_mean","pressure_gradient_error","boundary_training_mse",
              "prescribed_data_mse","worst_patch_velocity_rel_l2"]
    scientific=[k for k in reported if all(isinstance(s["metrics"].get(k),(float,int)) for _,_,_,s,_ in runs)]
    stats=[];deltas=[]
    for metric in scientific:
        v=raw[raw["mode"]=="vanilla"].sort_values("seed")[metric].to_numpy(dtype=float)
        a=raw[raw["mode"]=="vara_v2"].sort_values("seed")[metric].to_numpy(dtype=float)
        diff=v-a;percent=100*diff/np.maximum(abs(v),1e-30)
        rng=np.random.default_rng(bootstrap["bootstrap_seed"]);indices=rng.integers(0,len(v),size=(bootstrap["bootstrap_resamples"],len(v)))
        ci=np.quantile(diff[indices].mean(1),[.025,.975]);pci=np.quantile(percent[indices].mean(1),[.025,.975])
        p=float(wilcoxon(diff,method="exact").pvalue) if np.any(diff!=0) else 1.
        sd=diff.std(ddof=1) if len(v)>1 else np.nan
        stats.append({"metric":metric,"n":len(v),"vanilla_mean":v.mean(),"vara_mean":a.mean(),
            "vanilla_std":v.std(ddof=1) if len(v)>1 else np.nan,"vara_std":a.std(ddof=1) if len(v)>1 else np.nan,
            "vanilla_median":np.median(v),"vara_median":np.median(a),"mean_absolute_improvement":diff.mean(),
            "mean_paired_percent_improvement":percent.mean(),"median_paired_percent_improvement":np.median(percent),
            "ratio_of_means_improvement_percent":100*(v.mean()-a.mean())/max(abs(v.mean()),1e-30),
            "wins":int((diff>0).sum()),"losses":int((diff<0).sum()),"ties":int((diff==0).sum()),
            "absolute_ci95_low":ci[0],"absolute_ci95_high":ci[1],"percent_ci95_low":pci[0],"percent_ci95_high":pci[1],
            "paired_effect_size_dz":diff.mean()/sd if sd>0 else np.nan,"wilcoxon_exact_two_sided_p":p})
        for seed,x,y,d,pc in zip(expected,v,a,diff,percent):deltas.append({"seed":seed,"metric":metric,"vanilla":x,"vara":y,"absolute_improvement":d,"percent_improvement":pc})
    statistical=pd.DataFrame(stats)
    statistical.to_csv(output_dir/"statistics/paired_statistics.csv",index=False,float_format="%.17g")
    delta=pd.DataFrame(deltas);delta.to_csv(output_dir/"tables/paired_seed_deltas.csv",index=False,float_format="%.17g")
    delta[delta.absolute_improvement<0].to_csv(output_dir/"tables/negative_results.csv",index=False,float_format="%.17g")
    # A header-only negative table explicitly means no losses observed; never omit it.
    statistical[["metric","vanilla_mean","vara_mean","wins","mean_paired_percent_improvement","wilcoxon_exact_two_sided_p"]].to_latex(output_dir/"tables/main_results.tex",index=False,float_format="%.6g")
    pd.DataFrame([{ "seed":seed,"mode":mode,"initial_hash":s["initial_model_hash"],"pool_hash":s["permitted_pool_hash"],
                   "protocol_hash":s["protocol_hash"],"grid_hash":s["metrics"]["evaluation_grid_hash"],"source_sha":s["source"]["git_commit"],"final_rule":s["final_state_rule"]} for seed,mode,_,s,_ in runs]).to_csv(output_dir/"tables/fairness_provenance.csv",index=False)
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"svg.hashsalt":"kovasznay-v2-frozen"})
    figdir=output_dir/"figures"
    d=delta[delta.metric=="velocity_rel_l2"].sort_values("seed")
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True);ax.scatter(d.vanilla,d.vara);limit=max(d.vanilla.max(),d.vara.max())*1.1;ax.plot([0,limit],[0,limit],"k--")
    for r in d.itertuples():ax.annotate(str(r.seed),(r.vanilla,r.vara))
    ax.set(xlabel="Vanilla velocity relative L2",ylabel="VARA V2 velocity relative L2",title="All paired seeds; lower is better");savefig(fig,figdir,"paired_scatter")
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True);ax.bar(d.seed,d.percent_improvement,color=np.where(d.percent_improvement>=0,"#356597","#bb4d32"));ax.axhline(0,color="black",lw=.6);ax.set(xlabel="Seed",ylabel="Paired velocity-error improvement (%)");savefig(fig,figdir,"seed_improvements")
    focus=statistical[statistical.metric.isin(["velocity_rel_l2","pde_residual_mean","boundary_training_mse","prescribed_data_mse"])]
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True);x=np.arange(len(focus));means=focus.mean_paired_percent_improvement.to_numpy();err=np.vstack([means-focus.percent_ci95_low.to_numpy(),focus.percent_ci95_high.to_numpy()-means]);ax.errorbar(x,means,yerr=err,fmt="o");ax.axhline(0,color="black",lw=.6);ax.set_xticks(x,focus.metric,rotation=15,ha="right");ax.set(ylabel="Mean paired improvement with bootstrap95%CI (%)");savefig(fig,figdir,"aggregate_confidence_intervals")
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for seed,mode,folder,_,_ in runs:
        loss=pd.read_csv(folder/"losses.csv");ax.semilogy(loss.step,loss.loss_total,label=f"{mode},seed{seed}",lw=.8)
    ax.set(xlabel="Committed Adam step",ylabel="Training objective");ax.legend(fontsize=6,ncol=2);savefig(fig,figdir,"training_curves")
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for mode,color in [("vanilla","#356597"),("vara_v2","#bb4d32")]:
        f=raw[raw["mode"]==mode];axes[0].scatter(f.wall_seconds,f.velocity_rel_l2,label=mode,c=color);axes[1].scatter(f.optimizer_calls,f.velocity_rel_l2,label=mode,c=color)
    axes[0].set(xlabel="Observed wall time (s)",ylabel="Velocity relative L2");axes[1].set(xlabel="Physical Adam calls",ylabel="Velocity relative L2");axes[0].legend();savefig(fig,figdir,"accuracy_compute")
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
    for mode in ("vanilla","vara_v2"):
        f=raw[raw["mode"]==mode];ax.scatter(f.pde_residual_mean,f.velocity_rel_l2,label=mode)
    ax.set(xlabel="Mean PDE residual magnitude",ylabel="Velocity relative L2");ax.legend();savefig(fig,figdir,"reconstruction_physics_tradeoff")
    a=raw[raw["mode"]=="vara_v2"]
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True);bottom=np.zeros(len(a))
    for field in ("accepted","rejected","prefiltered"):ax.bar(a.seed,a[field],bottom=bottom,label=field);bottom+=a[field].to_numpy()
    ax.set(xlabel="Seed",ylabel="Candidate records (prefilter is not an optimizer probe)");ax.legend();savefig(fig,figdir,"guard_decision_counts")
    rep=999 if allow_smoke else 0
    selected=[run for run in runs if run[0]==rep]
    fig,axes=plt.subplots(2,3,figsize=(10,7),constrained_layout=True)
    error_limits=[]
    for j in range(3):
        arrays=[]
        for _,_,folder,_,_ in selected:
            f=np.load(folder/"evaluation_fields.npz");a=f["prediction"].copy();b=f["reference"].copy();a[:,2]-=a[:,2].mean();b[:,2]-=b[:,2].mean();arrays.append(abs(a[:,j]-b[:,j]).max())
        error_limits.append(max(max(arrays),1e-12))
    for i,(_,mode,folder,s,_) in enumerate(selected):
        field=np.load(folder/"evaluation_fields.npz");coords=field["coordinates"];pred=field["prediction"].copy();ref=field["reference"].copy();pred[:,2]-=pred[:,2].mean();ref[:,2]-=ref[:,2].mean()
        for j,name in enumerate(("u","v","p(centered)")):
            im=axes[i,j].scatter(coords[:,0],coords[:,1],c=pred[:,j]-ref[:,j],s=5,cmap="coolwarm",vmin=-error_limits[j],vmax=error_limits[j]);axes[i,j].set(title=f"{mode} {name} error;seed{rep}",xlabel="x",ylabel="y");fig.colorbar(im,ax=axes[i,j])
    savefig(fig,figdir,"predeclared_seed_field_errors")
    fig,axes=plt.subplots(3,3,figsize=(10,10),constrained_layout=True)
    first=np.load(selected[0][2]/"evaluation_fields.npz");coords=first["coordinates"];reference=first["reference"].copy();reference[:,2]-=reference[:,2].mean()
    plotted=[reference]
    for _,_,folder,_,_ in selected:
        value=np.load(folder/"evaluation_fields.npz")["prediction"].copy();value[:,2]-=value[:,2].mean();plotted.append(value)
    for i,(title,value) in enumerate(zip(["Analytical evaluation reference","Vanilla","VARA V2"],plotted)):
        for j,name in enumerate(("u","v","p(centered)")):
            im=axes[i,j].scatter(coords[:,0],coords[:,1],c=value[:,j],s=5,cmap="viridis",vmin=min(a[:,j].min() for a in plotted),vmax=max(a[:,j].max() for a in plotted));axes[i,j].set(title=f"{title}:{name};seed{rep}",xlabel="x",ylabel="y");fig.colorbar(im,ax=axes[i,j])
    savefig(fig,figdir,"predeclared_seed_fields")
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for seed,mode,folder,_,_ in runs:
        if mode!="vara_v2":continue
        states=json.loads((folder/"allocation_history.json").read_text());steps=[x["step"] for x in states]
        mass=np.asarray([x["sampling_mass"] for x in states]);entropy=-(mass*np.log(np.maximum(mass,1e-30))).sum(1)
        axes[0].plot(steps,entropy,label=str(seed));axes[1].plot(steps,mass.max(1),label=str(seed))
    axes[0].set(xlabel="Committed step",ylabel="Sampling entropy (nats)");axes[1].set(xlabel="Committed step",ylabel="Maximum patch probability");axes[0].legend(title="Seed");savefig(fig,figdir,"allocation_histories")
    diag=json.loads((input_root/f"seed_{rep}/vara_v2/diagnostics.json").read_text())
    if diag:
        record=diag[0];fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True);im=ax.imshow(record["normalized"],aspect="auto");ax.set_yticks(range(len(record["names"])),record["names"]);ax.set(xlabel="Spatial patch id",title=f"Predeclared seed{rep}:first pre-action diagnosis");fig.colorbar(im,ax=ax,label="Normalized severity");savefig(fig,figdir,"severity_patch_heatmap")
    limits="""# Limitations and claim-evidence boundary

This is a NEW V2 experiment, not reproduction/substitution of historical V1 results. Primary supervision is1000 fixed permitted u/v/p observations. Pure-PINN sensitivity is a distinct profile. Held-out analytic fields enter post-training evaluation only. Scalar/region geometry, NS equation guards, fixed BC pools, pressure gauge and neutral-sampler identity are declared differences from the Allen-Cahn runner. Controller mathematics, including uncalibrated predicted-improvement/reward scaling, remains unchanged. A25-step probe cannot guarantee475-step continuation or final accuracy. Small-n exact two-sided signed-rank Wilcoxon with five nonzero pairs cannot reach p<.05. Exploratory multiple secondary metrics are not multiplicity-controlled confirmatory claims. Win counts and negative rows are preserved. Accepted actions demonstrate monitored short-horizon improvement only, not causal final held-out gains. Smoke fixtures are infrastructure tests and never publication results. Interrupted intents produce explicit physical-call bounds and prevent an exact-compute badge. Historical V1 data are never substituted.
"""
    (output_dir/"LIMITATIONS.md").write_text(limits,encoding="utf-8")
    status="SMOKE_ONLY_PRIMARY_PENDING" if allow_smoke else "COMPLETE_VERIFIED_PRIMARY" if profile=="prescribed_sparse_primary" else "COMPLETE_VERIFIED_SENSITIVITY"
    (output_dir/"README.md").write_text(f"""# Kovasznay Re40 V2 supplementary evidence

Status: {status}. Seeds: {expected}. Methods: Vanilla / full VARA V2. Primary metric: held-out velocity aggregate relative L2. All other error/residual metrics are secondary and lower-is-better. See PROTOCOL_FROZEN.json, V2_PARITY_MATRIX.md and LIMITATIONS.md. Source commit: {runs[0][3]['source']['git_commit']}; actual scientific-file SHA256 values accompany each raw run.

raw/ contains immutable copied run outputs, including every attempt, discarded probe, complete-state checkpoint, final model/Adam and frozen pools. generated_analysis/, tables/, statistics/ and figures/ contain derived products; no raw output was modified. configs/ stores every resolved configuration; fairness/ contains paired manifests; provenance/ records runtimes, invocation and real test evidence. controller_logs/ points to decision evidence also retained under raw/. The representative primary seed0 was preregistered, not selected for success. undefined statistics are empty CSV cells, never zeros. Negative metrics remain in tables/negative_results.csv (a header-only file means no losses observed).

Reproduce: checkout the recorded source commit, install dependencies, run scripts/run_kovasznay_v2_publication.py with the frozen YAML separately for each listed seed/mode, then verify each run with --strict and build with scripts/package_kovasznay_v2_publication.py. Exact invocations are preserved under each raw arm/invocation.json and in the launcher provenance. Full source and the executable notebook are under source/.

Verify ZIP without extracting: python verify_package.py --zip supplement_kovasznay_v2_{'smoke' if allow_smoke else '5seed'}.zip --strict. Verify an extracted run: python verify_package.py --run-dir raw/seed_{expected[0]}/vara_v2 --strict. The latter uses restricted PyTorch checkpoint loading. Rebuild analysis from the same raw root with the source packager into a fresh output directory; no training is performed. Statistical bootstrap seed/resamples are preregistered. All payloads are included in checksums/SHA256SUMS; manifest.json records sizes/types/source/completeness; ZIP CRC is checked independently.

Mapping: tables/per_seed_metrics.csv -> all seed results; tables/main_results.tex -> main aggregate table; statistics/paired_statistics.csv -> uncertainty and signed-rank table; tables/compute_runtime.csv -> compute; tables/fairness_provenance.csv -> protocol/source/pool evidence; figures/ -> named plots; raw/*/diagnostics.json,decisions.jsonl,allocation_history.json,events.jsonl -> full mechanism evidence. Test output is provenance/tests/. Numerical tables trace directly to raw summaries and checkpoints. This archive is not a guarantee of favorable results.
""",encoding="utf-8")
    dump(output_dir/"generated_analysis/build_parameters.json",{"bootstrap_seed":bootstrap["bootstrap_seed"],"bootstrap_resamples":bootstrap["bootstrap_resamples"],"representative_seed":rep,"input_raw_root":str(input_root),"status":status})
    dump(output_dir/"provenance/verification_report.json",{"valid":True,"study_type":status,"runs":[r[4] for r in runs],"tests":tests,"all_paired":True})
    files=[]
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():files.append({"path":p.relative_to(output_dir).as_posix(),"size":p.stat().st_size,"sha256":sha(p),"artifact_type":p.relative_to(output_dir).parts[0]})
    dump(output_dir/"manifest.json",{"study_type":status,"git_commit":runs[0][3]["source"]["git_commit"],"all_required_runs_complete":True,"files":files})
    payloads=sorted(p for p in output_dir.rglob("*") if p.is_file())
    checksum=output_dir/"checksums/SHA256SUMS"
    checksum.write_text("\n".join(sha(p)+"  "+p.relative_to(output_dir).as_posix() for p in payloads)+"\n",encoding="utf-8")
    label="smoke" if allow_smoke else "5seed" if profile=="prescribed_sparse_primary" else "pure_pinn_sensitivity_5seed"
    archive=output_dir.parent/f"supplement_kovasznay_v2_{label}.zip"
    require(not archive.exists(),"ZIP exists; choose a fresh parent output location")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(output_dir.rglob("*")):
            if p.is_file():
                # Stable ZIP timestamps make payload archive generation reproducible.
                info=zipfile.ZipInfo(p.relative_to(output_dir).as_posix(),date_time=(2026,10,9,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
    report=verify_zip(archive,strict)
    archive.with_suffix(".zip.sha256").write_text(sha(archive)+"  "+archive.name+"\n",encoding="utf-8")
    return {"zip":str(archive),"sha256":sha(archive),"verification":report,"status":status}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-root",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--strict",action="store_true");p.add_argument("--allow-smoke",action="store_true",help="Fixture packaging only; never publication-ready");p.add_argument("--profile",choices=["prescribed_sparse_primary","pure_pinn_sensitivity"],default="prescribed_sparse_primary")
    a=p.parse_args();print(json.dumps(build(a.input_root,a.output_dir,a.strict,a.allow_smoke,a.profile),indent=2))
if __name__=="__main__":main()
