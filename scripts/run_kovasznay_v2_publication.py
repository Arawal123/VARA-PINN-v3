"""Run one protected Kovasznay V2/Vanilla arm; primary GPU study is opt-in."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import yaml
from src.training.kovasznay_v2_publication import (
    KovasznayV2PublicationTrainer,atomic_json,digest,protocol_config,source_info,validate_config)

def effective_config(path,device=None,seed=0,smoke=False,pure=False):
    cfg=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["seed"]=seed
    if device:cfg["device"]=device
    if pure:
        cfg["publication"]["profile"]="pure_pinn_sensitivity"
        cfg["publication"]["protocol_id"]="kovasznay_v2_pure_pinn_sensitivity_v1"
        cfg["training"]["n_data"]=0
        for name in ("u","v","p"):cfg["training"]["weights"][name]=0.
    if smoke:
        cfg["publication"]["smoke"]=True
        cfg["publication"]["progress_every"]=1
        cfg["model"]["hidden_layers"]=[8,8]
        cfg["training"].update(total_steps=12,n_collocation=16,n_boundary=8,n_data=0 if pure else 8)
        cfg["diagnostics"].update(n_interior=32,n_boundary=16)
        cfg["controller_v2"].update(total_steps=12,warmup_steps=2,control_blocks=2,block_steps=5,probe_steps=1)
        cfg["test"]={"nx":8,"ny":8}
    validate_config(cfg)
    if not smoke:
        frozen=json.loads((ROOT/"PROTOCOL_FROZEN.json").read_text(encoding="utf-8"))
        assert digest(protocol_config(cfg))==frozen["profiles"][cfg["publication"]["profile"]]["config_hash"],"Configuration deviates from frozen protocol"
    return cfg

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default=str(ROOT/"configs/kovasznay_v2_publication.yaml"))
    p.add_argument("--mode",choices=["vanilla","vara_v2"],default="vara_v2")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",choices=["cpu","cuda"])
    p.add_argument("--output-dir")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--pure-pinn",action="store_true")
    args=p.parse_args()
    cfg=effective_config(args.config,args.device,args.seed,args.smoke,args.pure_pinn)
    info={"valid":True,"protocol_hash":digest(protocol_config(cfg)),"config":cfg,"source":source_info(),
          "primary_study_status":"PENDING", "gpu_available":torch.cuda.is_available()}
    if args.preflight:
        print(json.dumps(info,indent=2));return
    if not args.output_dir:p.error("--output-dir is required for training")
    if args.resume and (Path(args.output_dir)/"COMPLETE.json").exists():
        from scripts.verify_kovasznay_v2_publication import verify_run
        verify_run(Path(args.output_dir),strict=True)
        print("Verified completed run; skipped without overwriting.",flush=True);return
    trainer=None
    try:
        torch.set_num_threads(2 if cfg["device"]=="cpu" else torch.get_num_threads())
        trainer=KovasznayV2PublicationTrainer(cfg,args.mode,args.output_dir,args.resume)
        invocation={"argv":sys.argv,"preflight":info,"runtime":trainer.runtime}
        if not (trainer.run_dir/"invocation.json").exists():atomic_json(trainer.run_dir/"invocation.json",invocation)
        (trainer.run_dir/"invocations").mkdir(exist_ok=True)
        atomic_json(trainer.run_dir/"invocations"/f"attempt_{trainer._attempt}.json",invocation)
        trainer.run()
    except BaseException as exc:
        if trainer is not None:
            trainer._event("attempt_end",duration_seconds=time.perf_counter()-trainer._start_clock,status="interrupted_or_failed",error=repr(exc))
            failure=trainer.run_dir/"failures"/f"attempt_{trainer._attempt}.json"
            failure.parent.mkdir(exist_ok=True)
            atomic_json(failure,{"error":repr(exc),"committed_step_displayed":trainer.global_step,"resume_from":trainer.next_block})
        raise
    else:
        trainer._event("attempt_end",duration_seconds=time.perf_counter()-trainer._start_clock,status="complete")

if __name__=="__main__":main()
