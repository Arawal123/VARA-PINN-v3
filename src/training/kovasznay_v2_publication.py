"""Isolated Kovasznay publication protocol; shared V2 controller/NS scaffold.

Analytical labels are frozen at construction. No analytical function is called
by training, diagnosis, gradient screening, probes, checkpointing or acceptance.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch
import yaml

from src.controllers.v2_controller import VARAV2Controller
from src.losses.base_losses import weighted_sum
from src.losses.local_losses import compute_budgeted_patch_losses
from src.physics.navier_stokes import navier_stokes_residuals
from src.training.vara_v2_trainer import VARAV2Trainer

ROOT = Path(__file__).resolve().parents[2]
GUARDS = {"pde_residual_mean", "continuity_residual_mean", "momentum_residual_mean",
          "boundary_condition_error", "unweighted_validation_loss", "unweighted_physics_validation_loss"}
SCIENCE_FILES = ["src/training/kovasznay_v2_publication.py", "src/training/vara_v2_trainer.py",
    "src/controllers/v2_controller.py", "src/physics/kovasznay.py", "src/physics/navier_stokes.py",
    "src/models/mlp.py", "src/losses/local_losses.py", "configs/kovasznay_v2_publication.yaml",
    "src/training/trainer.py", "src/losses/base_losses.py", "src/evaluation/metrics.py",
    "scripts/run_kovasznay_v2_publication.py", "PROTOCOL_FROZEN.json"]

def utc():
    return datetime.now(timezone.utc).isoformat()

def clean(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def digest(value):
    """Canonical tensor/state fingerprint, independent of serialization storage IDs."""
    def encode(v):
        if isinstance(v, (torch.Tensor, np.ndarray)):
            a = v.detach().cpu().contiguous().numpy() if isinstance(v, torch.Tensor) else np.ascontiguousarray(v)
            return {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": hashlib.sha256(a.tobytes()).hexdigest()}
        if isinstance(v, dict):
            return {str(k): encode(v[k]) for k in sorted(v, key=str)}
        if isinstance(v, (tuple, list)):
            return [encode(x) for x in v]
        return clean(v)
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(clean(value), f, sort_keys=True, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def tensor_payload(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: tensor_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tensor_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(tensor_payload(v) for v in value)
    return value

def protocol_config(cfg):
    cfg = deepcopy(cfg)
    for name in ("seed", "device", "experiments"):
        cfg.pop(name, None)
    return cfg

def source_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "--no-optional-locks", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError("A Git checkout is required for source provenance.")
    return {"git_commit": sha, "branch": branch, "dirty": dirty,
            "scientific_files": {name: file_sha(ROOT / name) for name in SCIENCE_FILES}}

def runtime_info(device):
    return {"python":__import__("platform").python_version(),"numpy":str(np.__version__),
            "torch":str(torch.__version__),"cuda":torch.version.cuda,"device":str(device),
            "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
            "cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled()}

def validate_config(cfg):
    c = cfg["controller_v2"]
    assert cfg["benchmark"] == "kovasznay" and cfg["benchmark_params"]["reynolds"] == 40
    assert cfg["benchmark_params"] == dict(reynolds=40.0, x_min=-.5, x_max=1., y_min=-.5, y_max=1.5)
    assert cfg["training"]["pointwise_reduction"] == "mean"
    assert c["warmup_steps"] + c["control_blocks"] * c["block_steps"] == c["total_steps"]
    assert 0 < c["probe_steps"] < c["block_steps"]
    assert set(c["guard_metrics"]) == GUARDS
    for key in ("counterfactual_probe_enabled", "gradient_prefilter_enabled", "trust_region_enabled", "action_memory_enabled", "rollback_enabled"):
        assert c[key] is True, f"Full V2 requires {key}"
    assert cfg["evaluation"]["controller_reference_metrics_enabled"] is False
    assert cfg["evaluation"]["checkpoint_reference_metrics_enabled"] is False
    assert not cfg["checkpoint"]["restore_best_before_final"]
    assert not cfg["optimizer"]["final_repair"]["enabled"]
    assert not cfg["optimizer"]["scheduler"]
    assert cfg["publication"]["final_state_rule"] == "final_committed_adam_state"
    if not cfg["publication"]["smoke"]:
        assert (c["total_steps"], c["warmup_steps"], c["control_blocks"], c["block_steps"], c["probe_steps"]) == (4000, 500, 7, 500, 25)
        assert cfg["model"] == dict(input_dim=2, output_dim=3, hidden_layers=[96]*5, activation="tanh")
        assert not cfg["publication"]["disabled_candidates"]
        assert cfg["training"]["n_data"] == (0 if cfg["publication"]["profile"] == "pure_pinn_sensitivity" else 1000)
    return True

class TrainingReferenceGate:
    """Fail closed on analytical target access after allowed-label construction."""
    def __init__(self,benchmark): self._benchmark=benchmark
    def __getattr__(self,name):
        if name in {"exact_np","exact_torch"}:
            raise RuntimeError("Analytical fields are inaccessible during training/controller execution")
        return getattr(self._benchmark,name)

class KovasznayV2PublicationTrainer(VARAV2Trainer):
    """Reuse NS model, physics, samplers, gradient helpers and V2 mathematics.

    Override NS orchestration to probe ONE candidate per block as Allen-Cahn.
    Vanilla uses this same loss/sampler implementation with actions disabled.
    """
    def _make_probe_batch(self):
        return {}  # Scaffold hook: create isolated, labeled diagnostic pools below.

    def __init__(self, cfg, mode, output_dir, resume=False):
        validate_config(cfg)
        if mode not in {"vanilla", "vara_v2"}:
            raise ValueError(mode)
        output = Path(output_dir).resolve()
        if output.exists() and any(output.iterdir()) and not resume:
            raise FileExistsError(f"Refusing to overwrite {output}; use --resume.")
        if resume and not (output / "checkpoints/latest_complete.json").exists():
            raise FileNotFoundError("Resume requires an atomic complete-state checkpoint.")
        if cfg["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; primary runs never silently fall back to CPU.")
        effective = deepcopy(cfg)
        effective["experiments"] = {"root": str(output), "flat_layout": True}
        super().__init__(effective, mode="vara_v2")
        self.mode = mode
        self.run_dir = output
        self.cfg_hash = digest(protocol_config(cfg))
        torch.use_deterministic_algorithms(True, warn_only=False)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self._source = source_info()
        self.runtime=runtime_info(self.device)
        self._attempt = 0
        self._journal = output / "events.jsonl"
        self.trajectory, self.decisions, self.diagnostics, self.allocations = [], [], [], []
        self.collocations = {}
        self.next_block = -1
        self.started = utc()
        self.elapsed_previous = 0.
        self._start_clock = time.perf_counter()
        self._freeze_observations()
        self.reporting_benchmark=self.benchmark
        self.benchmark=TrainingReferenceGate(self.benchmark)
        self._probe_batch = self.diag_batch
        self.initial_hash = digest(self.model.state_dict())
        self.pool_hash = digest(self.pools)
        self._calls = self._physical_counts()["optimizer_calls"]
        atomic_json(output / "resolved_config.json", cfg) if not resume else None
        if not resume:
            (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            np.savez_compressed(output / "permitted_observations.npz", **{k: v.numpy() for k, v in self.pools.items()})
        if resume:
            self._resume()
        self._event("attempt_start", source=self._source, device=str(self.device), resumed=resume)
        if not resume:self._checkpoint()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _coordinates(self, rng, count, boundary=False):
        x0,x1,y0,y1 = self.benchmark.bounds
        a = np.column_stack([rng.uniform(x0,x1,count), rng.uniform(y0,y1,count)])
        if boundary:
            side = np.arange(count) % 4
            a[side==0,0]=x0; a[side==1,0]=x1; a[side==2,1]=y0; a[side==3,1]=y1
            rng.shuffle(a)
        return torch.tensor(a, dtype=torch.float32)

    def _freeze_observations(self):
        t, d = self.config["training"], self.config["diagnostics"]
        rng = np.random.default_rng(self.seed+20003)
        bc = self._coordinates(rng,t["n_boundary"],True)
        sparse = self._coordinates(np.random.default_rng(self.seed+30003),t["n_data"])
        diag_rng = np.random.default_rng(self.seed+40003)
        interior = self._coordinates(diag_rng,d["n_interior"])
        diag_bc = self._coordinates(diag_rng,d["n_boundary"],True)
        # Only predeclared boundary observations and training observations receive labels.
        with torch.no_grad():
            bc_ref = self.benchmark.exact_torch(bc)
            diag_ref = self.benchmark.exact_torch(diag_bc)
            sparse_ref = self.benchmark.exact_torch(sparse)
        self.pools = {"boundary":bc, "boundary_uv":torch.cat([bc_ref["u"],bc_ref["v"]],1),
                      "sparse":sparse, "sparse_uvp":torch.cat([sparse_ref[k] for k in ("u","v","p")],1),
                      "diagnostic_interior":interior, "diagnostic_boundary":diag_bc,
                      "diagnostic_boundary_uv":torch.cat([diag_ref["u"],diag_ref["v"]],1)}
        self._set_pools()

    def _set_pools(self):
        p = {k:v.to(self.device) for k,v in self.pools.items()}
        self.fixed_bc,self.fixed_data = p["boundary"],p["sparse"]
        self.fixed_bc_uv,self.fixed_uvp = p["boundary_uv"],p["sparse_uvp"]
        self.diag_batch = self.make_batch(p["diagnostic_interior"],p["diagnostic_boundary"],p["sparse"])
        self.diag_batch["boundary_uv"] = p["diagnostic_boundary_uv"]

    def _sample_boundary(self, count):
        return self.fixed_bc

    def _sample_data(self, count):
        return self.fixed_data

    def make_batch(self, xy_f, xy_bc, xy_data):
        return {"xy_f":xy_f, "xy_bc":xy_bc, "xy_data":xy_data,
                "boundary_uv":self.fixed_bc_uv, "targets_uvp":self.fixed_uvp}

    def initial_batch(self):
        return self._resample_v2_batch({}, np.empty((0,2)))

    def pointwise(self, batch):
        r = navier_stokes_residuals(self.model,batch["xy_f"],nu=self.benchmark.nu,steady=True)
        terms = {"momentum_u":r["f_u"].square(), "momentum_v":r["f_v"].square(), "continuity":r["f_c"].square()}
        terms["pde"] = sum(terms.values())
        terms["bc"] = (self.model(batch["xy_bc"])[:,:2]-batch["boundary_uv"]).square().sum(1,keepdim=True)
        if len(batch["xy_data"]):
            pred = self.model(batch["xy_data"])
            target = batch["targets_uvp"]
            terms.update(u=(pred[:,0:1]-target[:,0:1]).square(),v=(pred[:,1:2]-target[:,1:2]).square(),
                         p=((pred[:,2:3]-pred[:,2:3].mean())-(target[:,2:3]-target[:,2:3].mean())).square())
        return terms

    def loss(self, batch):
        pointwise = self.pointwise(batch)
        allocation = self.v2_controller.state.loss_multipliers if self.mode=="vara_v2" else {}
        terms = compute_budgeted_patch_losses(pointwise,batch,self.patch_grid,allocation,reduction="mean")
        gauge = self.pressure_gauge_loss()
        return weighted_sum(terms,self.config["training"]["weights"])+gauge, terms, gauge

    def _event(self, kind, **fields):
        row = clean({"kind":kind,"attempt":self._attempt,"utc":utc(),**fields})
        with self._journal.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n"); f.flush()
        return row

    def _events(self):
        rows=[]
        journals=sorted(self.run_dir.glob("events*.jsonl"),key=lambda p:0 if p.name=="events.jsonl" else int(p.stem.rsplit("_",1)[1]))
        for path in journals:
            lines=path.read_text(encoding="utf-8").splitlines()
            for i,line in enumerate(lines):
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if i!=len(lines)-1:raise RuntimeError("Malformed non-tail journal record; strict resume refused")
                    rows.append({"kind":"truncated_record","attempt":max([r["attempt"] for r in rows] or [0]),
                                 "raw_file":path.name,"raw_line_sha256":hashlib.sha256(line.encode()).hexdigest()})
        return rows

    def _physical_counts(self):
        rows = self._events()
        steps=[r for r in rows if r["kind"]=="optimizer_step"]
        intents={r["token"] for r in rows if r["kind"]=="optimizer_intent"}
        done={r["token"] for r in steps}
        gradients=[r for r in rows if r["kind"]=="gradient_probe"]
        diagnostics=[r for r in rows if r["kind"]=="diagnostic"]
        return {"optimizer_calls":len(steps),"uncertain_optimizer_calls_upper":len(intents-done)+sum(r["kind"]=="truncated_record" for r in rows),
                "objective_evaluations":len(steps)+len(gradients),"controller_gradient_evaluations":2*len(gradients),
                "training_points_evaluated":sum(r["points"] for r in steps),
                "controller_points_evaluated":sum(r["points"] for r in gradients),
                "diagnostic_evaluations":len(diagnostics),"diagnostic_points_evaluated":sum(r["points"] for r in diagnostics),
                "optimization_seconds":sum(r["duration_seconds"] for r in steps),
                "probe_optimizer_calls":sum(r["phase"] in {"neutral_probe","candidate_probe"} for r in steps)}

    def _progress(self, phase):
        total=self.config["controller_v2"]["total_steps"]
        elapsed=self.elapsed_previous+time.perf_counter()-self._start_clock
        row={"step":self.global_step,"total_steps":total,"phase":phase,"seed":self.seed,"mode":self.mode,
             "optimizer_calls":self._calls,"elapsed_seconds":elapsed,
             "eta_seconds":elapsed*(total-self.global_step)/max(self.global_step,1)}
        print("@@PROGRESS "+json.dumps(row),flush=True)

    def _train(self,batch,count,phase,retained=True):
        self.model.train()
        rows=[]
        points=1+sum(len(batch[k]) for k in ("xy_f","xy_bc","xy_data")) # plus reference-free gauge point
        for _ in range(count):
            step_clock=time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            total,terms,gauge=self.loss(batch)
            if not torch.isfinite(total): raise FloatingPointError("Nonfinite loss")
            total.backward()
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config["optimizer"]["max_grad_norm"])
            if not torch.isfinite(norm): raise FloatingPointError("Nonfinite gradient")
            token=f"{self._attempt}:{self._calls}:{phase}:{len(rows)}"
            self._event("optimizer_intent",token=token,phase=phase)
            self.optimizer.step(); self._calls+=1
            if self.device.type=="cuda":torch.cuda.synchronize(self.device)
            row=self._event("optimizer_step",token=token,phase=phase,points=points,
                duration_seconds=time.perf_counter()-step_clock,
                loss_total=float(total.detach()),pressure_gauge=float(gauge.detach()),gradient_norm=float(norm.detach()),
                losses={k:float(v.detach()) for k,v in terms.items()})
            rows.append(row)
            if retained:
                self.global_step+=1; row={**row,"step":self.global_step}; self.trajectory.append(row)
            if self.global_step%self.config["publication"]["progress_every"]==0 or len(rows)==count:
                self._progress(phase)
        return rows

    def _diagnose(self,phase,block):
        self.model.eval()
        b=self.diag_batch
        r=navier_stokes_residuals(self.model,b["xy_f"],nu=self.benchmark.nu,steady=True)
        channels=[("continuity_residual",r["f_c"].detach().abs(),b["xy_f"]),
                  ("momentum_u_residual",r["f_u"].detach().abs(),b["xy_f"]),
                  ("momentum_v_residual",r["f_v"].detach().abs(),b["xy_f"])]
        with torch.no_grad():
            channels.append(("boundary_mismatch",(self.model(b["xy_bc"])[:,:2]-b["boundary_uv"]).abs().mean(1),b["xy_bc"]))
            if len(b["xy_data"]):
                pred=self.model(b["xy_data"]); target=b["targets_uvp"]
                pred=pred.clone();target=target.clone();pred[:,2]-=pred[:,2].mean();target[:,2]-=target[:,2].mean()
                for i,name in enumerate(("u","v","p")):
                    channels.append((f"sparse_{name}_mismatch",(pred[:,i]-target[:,i]).abs(),b["xy_data"]))
        raw=[]
        for name,values,coords in channels:
            ids=self.patch_grid.assign_torch(coords).cpu().numpy();v=values.detach().cpu().numpy().reshape(-1)
            raw.append([float(np.percentile(v[ids==p],90)) if np.any(ids==p) else 0. for p in range(self.patch_grid.num_patches)])
        raw=np.asarray(raw); normalized=[]
        for row in raw:
            positive=row[np.isfinite(row)&(row>0)]
            normalized.append(row/max(float(np.median(positive)),1e-12) if len(positive) else row*0)
        names=[x[0] for x in channels];normalized=np.asarray(normalized)
        self.v2_controller.assert_reference_free(names)
        means=dict(zip(names,raw.mean(1)));p=float(np.mean(raw[:3]));bc=means["boundary_mismatch"]
        sparse=float(np.mean([v for k,v in means.items() if k.startswith("sparse_")])) if len(b["xy_data"]) else 0.
        metrics={"pde_residual_mean":p,"continuity_residual_mean":means["continuity_residual"],
                 "momentum_residual_mean":float(np.mean(raw[1:3])),"boundary_condition_error":bc,
                 "unweighted_physics_validation_loss":p+bc,"unweighted_validation_loss":p+bc+sparse}
        if not all(np.isfinite(list(metrics.values()))): raise FloatingPointError("Nonfinite diagnostic")
        record={"phase":phase,"block":block,"step":self.global_step,"names":names,"raw":raw.tolist(),"normalized":normalized.tolist(),"metrics":metrics,
                "points":sum(len(b[k]) for k in ("xy_f","xy_bc","xy_data"))}
        self.diagnostics.append(record);self._event("diagnostic",**record)
        return names,raw,normalized,metrics

    def _influence(self,candidates):
        results={}
        for c in candidates:
            pointwise=self.pointwise(self.diag_batch)
            target=self._candidate_probe_loss(c,pointwise)
            # Omit duplicate aggregate PDE and configured global scalar weights.
            guard=sum(v.mean() for k,v in pointwise.items() if k!="pde")
            parameters=list(self.model.parameters())
            gt=self._flat_gradient(target,parameters);gg=self._flat_gradient(guard,parameters)
            if not torch.isfinite(gt).all() or not torch.isfinite(gg).all():raise FloatingPointError("Nonfinite controller gradient")
            cosine=float(torch.dot(gt,gg)/(gt.norm()*gg.norm()+1e-12))
            results[c.key()]={"gradient_compatibility":max(0.,cosine),"gradient_conflict":max(0.,-cosine)}
            self._event("gradient_probe",candidate=c.to_record(),cosine=cosine,
                        target_gradient_hash=digest(gt),guard_gradient_hash=digest(gg),
                        points=sum(len(self.diag_batch[k]) for k in ("xy_f","xy_bc","xy_data")))
        self.model.zero_grad(set_to_none=True)
        return results

    def _snapshot(self):
        cpu=np.random.get_state()
        return {"model":{k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},
                "optimizer":deepcopy(self.optimizer.state_dict()),"allocation":self.v2_controller.state.snapshot(),
                "controller":{k:deepcopy(getattr(self.v2_controller,k)) for k in
                              ("trust_radius","effectiveness","score_history","metric_history","decisions")},
                "sampling":deepcopy(self.sampling_state_snapshot()),"torch_rng":torch.get_rng_state().clone(),
                "cuda_rng":torch.cuda.get_rng_state_all() if self.device.type=="cuda" else [],
                "python_rng":random.getstate(),"numpy_rng":(cpu[0],cpu[1].copy(),cpu[2],cpu[3],cpu[4])}

    def _restore(self,s):
        self.model.load_state_dict(s["model"])
        self.optimizer.load_state_dict(deepcopy(s["optimizer"]))
        self.v2_controller.state.restore(s["allocation"])
        for k,v in s["controller"].items(): setattr(self.v2_controller,k,deepcopy(v))
        self.restore_sampling_state(deepcopy(s["sampling"]))
        torch.set_rng_state(s["torch_rng"].cpu())
        if s["cuda_rng"]: torch.cuda.set_rng_state_all(s["cuda_rng"])
        random.setstate(s["python_rng"])
        n=list(s["numpy_rng"]);n[1]=np.asarray(n[1],dtype=np.uint32);np.random.set_state(tuple(n))

    def _checkpoint(self):
        path=self.checkpoint_dir/f"complete_a{self._attempt}_step{self.global_step:05d}.pt"
        if path.exists():return
        tmp=path.with_name(path.name+".tmp")
        payload={"runtime":self._snapshot(),"config_hash":self.cfg_hash,"seed":self.seed,"mode":self.mode,
                 "source":self._source,"global_step":self.global_step,"next_block":self.next_block,
                 "trajectory":self.trajectory,"decisions":self.decisions,"diagnostics":self.diagnostics,
                 "allocations":self.allocations,"pools":self.pools,"initial_hash":self.initial_hash,
                 "pool_hash":self.pool_hash,"collocations":self.collocations,"started":self.started,
                 "elapsed":self.elapsed_previous+time.perf_counter()-self._start_clock,
                 "journal_records":len(self._events()),"device":str(self.device),"runtime_environment":self.runtime}
        with tmp.open("wb") as f:
            torch.save(tensor_payload(payload),f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        compatibility=self.checkpoint_dir/"complete.pt"
        alias_tmp=compatibility.with_name("complete.pt.tmp")
        shutil.copyfile(path,alias_tmp);os.replace(alias_tmp,compatibility)
        atomic_json(self.checkpoint_dir/"latest_complete.json",{"file":path.name,"sha256":file_sha(path),"step":self.global_step})

    def _resume(self):
        pointer=json.loads((self.checkpoint_dir/"latest_complete.json").read_text())
        path=self.checkpoint_dir/pointer["file"]
        if file_sha(path)!=pointer["sha256"]:
            raise RuntimeError("Checkpoint checksum mismatch")
        p=torch.load(path,map_location="cpu",weights_only=True)
        assert (p["config_hash"],p["seed"],p["mode"],p["device"])==(self.cfg_hash,self.seed,self.mode,str(self.device))
        assert p["source"]["scientific_files"]==self._source["scientific_files"],"Source changed; resume refused"
        assert p["runtime_environment"]==self.runtime,"Runtime/backend changed; exact trajectory resume refused"
        self._restore(p["runtime"])
        self.pools=p["pools"];self._set_pools()
        for name in ("trajectory","decisions","diagnostics","allocations","collocations","next_block","started","initial_hash","pool_hash"):
            setattr(self,name,p[name])
        self.global_step=p["global_step"];self.elapsed_previous=p["elapsed"]
        events=self._events();self._attempt=1+max([r["attempt"] for r in events] or [-1])
        self._journal=self.run_dir/f"events_attempt_{self._attempt}.jsonl"
        self._event("resume",discarded_partial_records=max(0,len(events)-p["journal_records"]),from_step=self.global_step)

    def _allocation(self,block):
        self.v2_controller.validate_state()
        self.allocations.append({"block":block,"step":self.global_step,"trust_radius":self.v2_controller.trust_radius,
            "effectiveness":clean(self.v2_controller.effectiveness),**self.v2_controller.state.to_record()})

    def train_protocol(self,stop_after_block=None,force_reject=False):
        if force_reject and not self.config["publication"]["smoke"]:
            raise ValueError("Forced rejection is restricted to labeled CPU regression fixtures")
        c=self.config["controller_v2"]
        if self.next_block==-1:
            batch=self.initial_batch();self.collocations["warmup"]=batch["xy_f"].detach().cpu()
            self._train(batch,c["warmup_steps"],"warmup")
            self.next_block=0;self._allocation(-1);self._checkpoint()
        for block in range(self.next_block,c["control_blocks"]):
            active=[]
            if self.mode=="vara_v2" and not self.config["publication"]["disabled_candidates"]:
                names,raw,norm,metrics=self._diagnose("before",block)
                self.v2_controller.update_history(names,norm,metrics)
                regions=self.weak_detector.detect(norm,names,self.patch_grid)
                candidates=self.v2_controller.candidates(regions)
                mapping={"continuity_residual":["continuity"],"momentum_u_residual":["momentum_u"],
                         "momentum_v_residual":["momentum_v"],"boundary_mismatch":["bc"],
                         "sparse_u_mismatch":["u"],"sparse_v_mismatch":["v"],"sparse_p_mismatch":["p"]}
                for candidate in candidates: candidate.loss_names=mapping[candidate.variable]
                ranked=self.v2_controller.rank(candidates,self._influence(candidates))
                self._event("ranked_candidates",block=block,weak_regions=[vars(x) for x in regions],candidates=[x.to_record() for x in ranked])
                for candidate in ranked:
                    if candidate.prefiltered:
                        d=self.v2_controller.record_prefilter(candidate,update_trust=False)
                        self.decisions.append({"block":block,**candidate.to_record(),**d})
                active=[x for x in ranked if not x.prefiltered]
            if not active:
                batch=self.initial_batch();self.collocations[f"block_{block}_retained"]=batch["xy_f"].detach().cpu()
                self._train(batch,c["block_steps"],"no_action")
            else:
                candidate=active[0];before=self._snapshot()
                neutral_batch=self.initial_batch()
                neutral_rows=self._train(neutral_batch,c["probe_steps"],"neutral_probe",False)
                nn,nr,_,nm=self._diagnose("neutral",block);neutral=self._snapshot()
                self._restore(before)
                assert digest(self._snapshot())==digest(before),"Initial counterfactual state did not restore"
                self.v2_controller.apply(candidate);action_batch=self.initial_batch()
                action_rows=self._train(action_batch,c["probe_steps"],"candidate_probe",False)
                an,ar,_,am=self._diagnose("candidate",block);action=self._snapshot()
                bt=self._candidate_score(candidate,nr,nn);at=self._candidate_score(candidate,ar,an)
                accepted,d=self.v2_controller.evaluate(candidate,bt,at,nm,am,
                    target_threshold=c["counterfactual_target_margin"],guard_threshold=c["counterfactual_guard_margin"],
                    comparison_mode="counterfactual",update_state=False)
                if force_reject: accepted=False;d["rollback_reason"]="test_forced_rejection"
                kept=action if accepted else neutral;self._restore(kept)
                restored_hash=digest(self._snapshot());assert restored_hash==digest(kept),"Branch restoration failed"
                d=self.v2_controller.commit_evaluation(candidate,accepted,d)
                self.decisions.append({"block":block,**candidate.to_record(),**d,"prefiltered":False,
                    "accepted":accepted,"neutral_target":bt,"candidate_target":at,"neutral_metrics":nm,"candidate_metrics":am,
                    "initial_state_hash":digest(before),"neutral_state_hash":digest(neutral),"candidate_state_hash":digest(action),
                    "restored_state_hash":restored_hash,"expected_state_hash":digest(kept),"restoration_verified":True,
                    "probe_steps":c["probe_steps"],"neutral_pool_hash":digest(neutral_batch),"candidate_pool_hash":digest(action_batch)})
                kept_rows=action_rows if accepted else neutral_rows
                for row in kept_rows:
                    self.global_step+=1;self.trajectory.append({**row,"step":self.global_step})
                batch=action_batch if accepted else neutral_batch
                self.collocations[f"block_{block}_neutral"]=neutral_batch["xy_f"].detach().cpu()
                self.collocations[f"block_{block}_candidate"]=action_batch["xy_f"].detach().cpu()
                self._train(batch,c["block_steps"]-c["probe_steps"],"continuation")
            self.next_block=block+1;self._allocation(block);self._checkpoint()
            if stop_after_block==block:return
        assert self.global_step==c["total_steps"]

    def evaluate_final(self):
        assert self.global_step==self.config["controller_v2"]["total_steps"],"Evaluation forbidden before final state"
        from src.evaluation.metrics import evaluate_on_grid
        nx,ny=self.config["test"]["nx"],self.config["test"]["ny"]
        _,_,coords=self.reporting_benchmark.grid(nx,ny)
        metrics=evaluate_on_grid(self.model,self.reporting_benchmark,coords,self.device,True)
        with torch.no_grad():
            pred=self.model(torch.tensor(coords,dtype=torch.float32,device=self.device)).cpu().numpy()
            data_pred=self.model(self.fixed_data) if len(self.fixed_data) else None
            bc_error=float((self.model(self.fixed_bc)[:,:2]-self.fixed_bc_uv).square().mean())
        reference=self.reporting_benchmark.exact_np(coords)
        ref=np.column_stack([reference[k].reshape(-1) for k in ("u","v","p")])
        metrics["velocity_rel_l2"]=float(np.linalg.norm(pred[:,:2]-ref[:,:2])/np.linalg.norm(ref[:,:2]))
        metrics["boundary_training_mse"]=bc_error
        metrics["prescribed_data_mse"]=None
        if data_pred is not None:
            p=data_pred.detach().clone();t=self.fixed_uvp.clone();p[:,2]-=p[:,2].mean();t[:,2]-=t[:,2].mean()
            metrics["prescribed_data_mse"]=float((p-t).square().mean())
        ids=self.patch_grid.assign_numpy(coords)
        metrics["worst_patch_velocity_rel_l2"]=max(float(np.linalg.norm(pred[ids==i,:2]-ref[ids==i,:2])/max(np.linalg.norm(ref[ids==i,:2]),1e-12)) for i in range(self.patch_grid.num_patches))
        residual_fields={key:[] for key in ("f_u","f_v","f_c","omega","p_x","p_y")}
        for part in np.array_split(coords,max(1,int(np.ceil(len(coords)/1024)))):
            residual=navier_stokes_residuals(self.model,torch.tensor(part,dtype=torch.float32,device=self.device),nu=self.benchmark.nu,steady=True)
            for key in residual_fields:residual_fields[key].append(residual[key].detach().cpu().numpy())
        residual_fields={key:np.concatenate(value) for key,value in residual_fields.items()}
        np.savez_compressed(self.run_dir/"evaluation_fields.npz",coordinates=coords,prediction=pred,reference=ref,
                            omega_reference=reference["omega"],pressure_gradient_reference=np.column_stack([reference["p_x"].ravel(),reference["p_y"].ravel()]),
                            **residual_fields)
        metrics["evaluation_grid_hash"]=digest(coords)
        return clean(metrics)

    def run(self):
        self.train_protocol()
        metrics=self.evaluate_final()
        counts=self._physical_counts()
        summary={"benchmark":"kovasznay","mode":self.mode,"seed":self.seed,"profile":self.config["publication"]["profile"],
            "smoke":self.config["publication"]["smoke"],"source":self._source,"protocol_hash":self.cfg_hash,
            "initial_model_hash":self.initial_hash,"permitted_pool_hash":self.pool_hash,
            "committed_steps":self.global_step,**counts,"metrics":metrics,"started_utc":self.started,"finished_utc":utc(),
            "parameter_count":sum(p.numel() for p in self.model.parameters()),
            "wall_seconds":sum(r["duration_seconds"] for r in self._events() if r["kind"]=="attempt_end" and r["attempt"]!=self._attempt)+time.perf_counter()-self._start_clock,
            "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(self.device) if self.device.type=="cuda" else None,
            "controller_reference_metrics_enabled":False,"final_state_rule":"final_committed_adam_state",
            "runtime":self.runtime,
            "accepted":sum(d.get("accepted",False) for d in self.decisions),
            "rejected":sum(not d.get("accepted",False) and not d.get("prefiltered",False) for d in self.decisions),
            "prefiltered":sum(d.get("prefiltered",False) for d in self.decisions),
            "reference_gate_active":True,
            "reference_isolation":"training uses frozen labels; analytical fields only in post-training evaluate_final"}
        self._checkpoint()
        final=self.checkpoint_dir/"final.pt";tmp=final.with_name("final.pt.tmp")
        torch.save(tensor_payload({"model":self._snapshot()["model"],"optimizer":deepcopy(self.optimizer.state_dict()),
                                  "summary":summary,"global_step":self.global_step}),tmp);os.replace(tmp,final)
        atomic_json(self.run_dir/"summary.json",summary)
        atomic_json(self.run_dir/"diagnostics.json",self.diagnostics)
        atomic_json(self.run_dir/"allocation_history.json",self.allocations)
        with (self.run_dir/"decisions.jsonl").open("w",encoding="utf-8") as f:
            for d in self.decisions:f.write(json.dumps(clean(d),sort_keys=True,allow_nan=False)+"\n")
        np.savez_compressed(self.run_dir/"collocation_history.npz",**{k:v.numpy() for k,v in self.collocations.items()})
        with (self.run_dir/"losses.csv").open("w",newline="",encoding="utf-8") as f:
            rows=[{"step":r["step"],"phase":r["phase"],"loss_total":r["loss_total"],"gradient_norm":r["gradient_norm"],"pressure_gauge":r["pressure_gauge"],**r["losses"]} for r in self.trajectory]
            writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
        atomic_json(self.run_dir/"COMPLETE.json",{"summary_sha256":file_sha(self.run_dir/"summary.json"),"final_checkpoint_sha256":file_sha(final),"status":"SMOKE_COMPLETE" if summary["smoke"] else "COMPLETE"})
        self._progress("complete")
        return summary
