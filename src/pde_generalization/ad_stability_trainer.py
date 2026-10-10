"""Opt-in AD stability revision. Historical V2 source/outputs are untouched.

Reference-free Armijo step checks, independent guards and continuation rollback
address measured failure mechanisms. Neither convergence nor superiority over
the identically safeguarded Vanilla comparator is guaranteed.
"""
from copy import deepcopy
import json
import random
import time

import numpy as np
import torch

from src.controllers.v2_controller import VARAV2Controller,V2Candidate
from src.diagnostics.weak_region_detector import WeakRegionDetector
from src.pde_generalization.trainer import PDEGeneralizationTrainer
from src.pde_generalization.losses import compute_training_loss

GUARDS=("pde_mse","bc_mse","ic_mse","sparse_mse","normalized_physics_loss","normalized_validation_loss")

def serial(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    if isinstance(v,np.generic):return v.item()
    return v

class CalibratedADController(VARAV2Controller):
    """Same V2 screening/rank architecture; reward predictions are fractions."""
    def __init__(self,cfg):
        super().__init__(cfg);self.fraction_ema={}

    def rank(self,candidates,influence):
        # Preserve score-domain gradient prefilter. Do not compare fractional
        # improvement to a severity-domain predicted-damage score.
        ranked=super().rank(candidates,influence)
        for c in ranked:
            c.screen_target_score=c.predicted_target_improvement
            c.screen_damage_score=c.predicted_guard_damage
            c.predicted_target_improvement=float(np.clip(self.fraction_ema.get(c.key(),2*self.config.counterfactual_target_margin),
                                                        self.config.counterfactual_target_margin,1.))
        return ranked

    def _update_after_decision(self,candidate,accepted,reward_ratio,decision):
        super()._update_after_decision(candidate,accepted,reward_ratio,decision)
        observed=float(np.clip(decision.get("observed_target_improvement",0.),0.,1.))
        old=self.fraction_ema.get(candidate.key(),2*self.config.counterfactual_target_margin)
        self.fraction_ema[candidate.key()]=self.config.effectiveness_ema*old+(1-self.config.effectiveness_ema)*observed
        decision["predicted_fraction_ema_after"]=self.fraction_ema[candidate.key()]

class ADStabilityTrainer(PDEGeneralizationTrainer):
    def __init__(self,cfg,mode,run_dir):
        if cfg["benchmark"]!="advection_diffusion":raise ValueError("AD stability revision is benchmark-specific")
        if mode not in {"vanilla","vara_v2"}:raise ValueError("Use the explicitly matched revision modes")
        revision=cfg["ad_stability_revision"]
        fractions=revision["step_fractions"]
        if not fractions or fractions[0]!=1. or any(not 0<a<=1 for a in fractions) or any(a<=b for a,b in zip(fractions,fractions[1:])):
            raise ValueError("Step fractions must strictly decrease from 1 within (0,1]")
        if not 0<revision["armijo_coefficient"]<1 or revision["guard_recheck_steps"]<1 or revision["validation_scale_floor"]<=0:
            raise ValueError("Invalid safeguard constants")
        if not cfg["controller_v2"]["counterfactual_probe_enabled"] or not cfg["controller_v2"]["rollback_enabled"]:
            raise ValueError("The stability revision requires matched probes and rollback")
        if not cfg["controller_v2"]["variable_awareness_enabled"] or cfg["evaluation"]["controller_reference_metrics_enabled"]:
            raise ValueError("The stability revision requires channel awareness and reference isolation")
        effective=deepcopy(cfg)
        if cfg["ad_stability_revision"]["independent_component_guards"]:
            effective["controller_v2"]["guard_metrics"]=list(GUARDS)
        super().__init__(effective,mode,run_dir)
        self.revision=dict(cfg["ad_stability_revision"])
        self.validation_scales=None
        self.step_audit=[];self.long_guard_audit=[];self.proposal_audit=[]
        self.safeguard_objective_evaluations=0;self.continuation_replay_calls=0
        self.accepted_parameter_steps=0;self.noop_parameter_steps=0
        self.continuation_rollback_count=0
        self.start_block=0;self.resumed=False
        self.completed_optimization_seconds=0.;self.segment_started=None
        if self.controller is not None and self.revision["calibrated_fraction_prediction"]:
            self.controller=CalibratedADController(self.controller.config)

    def _sample_adaptive(self,count):
        if self.revision["neutral_sampler_parity"] and np.allclose(self.controller.state.sampling_mass,1/self.patch_grid.num_patches,rtol=0,atol=1e-12):
            return self._sample_uniform(count,self.sampling_rng)
        return super()._sample_adaptive(count)

    def _validation(self):
        result=compute_training_loss(self.model,self.benchmark,self.diagnostic_batch,self.weights,self.patch_grid,None)
        self.safeguard_objective_evaluations+=1
        self.objective_evaluation_count+=1
        raw={key:float(value.detach()) for key,value in result.components.items()}
        if not all(np.isfinite(v) for v in raw.values()):
            raise FloatingPointError("Nonfinite reference-free validation component")
        if self.validation_scales is None:
            self.validation_scales={k:max(v,self.revision["validation_scale_floor"]) for k,v in raw.items()}
        normalized={k:raw[k]/self.validation_scales[k] for k in raw}
        return {"pde_mse":raw["pde"],"bc_mse":raw["bc"],"ic_mse":raw["ic"],"sparse_mse":raw["sparse_data"],
                "normalized_physics_loss":normalized["pde"]+normalized["bc"]+normalized["ic"],
                "normalized_validation_loss":sum(normalized.values())}

    def _diagnose(self):
        snapshot=super()._diagnose()
        snapshot.revision_metrics=self._validation()
        if self.revision["frozen_diagnostic_scales"]:
            channel_component={"pde_residual":"pde","boundary_mismatch":"bc","initial_condition_mismatch":"ic","sparse_u_mismatch":"sparse_data"}
            snapshot.normalized_scores=np.vstack([row/max(np.sqrt(self.validation_scales[channel_component[name]]),1e-12)
                for name,row in zip(snapshot.names,snapshot.raw_scores)])
        return snapshot

    def _controller_metrics(self,snapshot):
        if self.revision["independent_component_guards"]:return snapshot.revision_metrics
        return super()._controller_metrics(snapshot)

    def _train_steps(self,batch,steps,phase):
        if not self.revision["armijo_step_safeguard"]:return super()._train_steps(batch,steps,phase=phase)
        rows=[];allocation=self.controller.state if self.controller is not None else None
        for local_step in range(steps):
            self.model.train();self.optimizer.zero_grad(set_to_none=True)
            result=compute_training_loss(self.model,self.benchmark,batch,self.weights,self.patch_grid,allocation)
            self.objective_evaluation_count+=1
            if not torch.isfinite(result.total):raise FloatingPointError("Nonfinite revision training loss")
            result.total.backward()
            params=list(self.model.parameters())
            gradients=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config["training"]["gradient_clip"])
            if not torch.isfinite(norm):raise FloatingPointError("Nonfinite revision gradient")
            before=[p.detach().clone() for p in params]
            self.optimizer.step();self.optimizer_step_calls+=1
            delta=[p.detach()-b for p,b in zip(params,before)]
            slope=float(sum((g*d).sum() for g,d in zip(gradients,delta)))
            direction="adam"
            if slope>=0:
                # An uphill momentum direction cannot satisfy a descent test
                # near alpha0. The explicit fallback is also shared by Vanilla.
                delta=[torch.zeros_like(p) if p.grad is None else -self.config["training"]["lr"]*p.grad.detach() for p in params]
                slope=float(sum((g*d).sum() for g,d in zip(gradients,delta)));direction="gradient_fallback"
            base_loss=float(result.total.detach());chosen=0.;after_loss=base_loss;trials=[]
            for alpha in self.revision["step_fractions"]:
                with torch.no_grad():
                    for p,b,d in zip(params,before,delta):p.copy_(b+alpha*d)
                checked=compute_training_loss(self.model,self.benchmark,batch,self.weights,self.patch_grid,allocation)
                self.objective_evaluation_count+=1;self.safeguard_objective_evaluations+=1
                value=float(checked.total.detach());trials.append([alpha,value])
                if np.isfinite(value) and value<=base_loss+self.revision["armijo_coefficient"]*alpha*slope:
                    chosen=alpha;after_loss=value;break
            if chosen==0:
                with torch.no_grad():
                    for p,b in zip(params,before):p.copy_(b)
                self.noop_parameter_steps+=1
            else:self.accepted_parameter_steps+=1
            update_norm=float(torch.sqrt(sum((p.detach()-b).square().sum() for p,b in zip(params,before))))
            audit={"phase":phase,"step_before_segment":self.applied_optimizer_steps,"local_step":local_step+1,
                   "direction":direction,"raw_gradient_norm":float(norm),"gradient_dot_direction":slope,"accepted_fraction":chosen,
                   "loss_before":base_loss,"loss_after":after_loss,"parameter_update_l2":update_norm,"trials":trials,
                   "adam_moments_advanced":True,"parameter_noop":chosen==0,"zero_parameter_update":update_norm==0}
            self.step_audit.append(audit)
            rows.append({"local_step":local_step+1,"phase":phase,"loss_total":base_loss,
                         **{f"loss_{k}":float(v.detach()) for k,v in result.components.items()},
                         "loss_after_step":after_loss,"step_fraction":chosen,"raw_gradient_norm":float(norm),"parameter_update_l2":update_norm})
            if (local_step+1)%50==0 and "probe" not in phase:
                print("@@PROGRESS "+json.dumps({"step":self.applied_optimizer_steps+local_step+1,
                    "total_steps":self.config["controller_v2"]["total_steps"],"optimizer_calls":self.optimizer_step_calls,
                    "mode":self.mode,"seed":self.seed,"phase":phase,"segment_pending_commit":True}),flush=True)
        return rows

    def _all_channel_candidates(self,snapshot):
        regions=[]
        # Availability, not forced acceptance/balancing: one best supported
        # region per channel may compete. IC/BC/sparse cannot acquire new data.
        for i,name in enumerate(snapshot.names):
            detector=WeakRegionDetector(percentile_threshold=self.config["diagnostics"]["weak_percentile"],top_k_per_variable=1,max_active_patches=1,persistence_cycles=1)
            found=detector.detect(snapshot.normalized_scores[i:i+1],[name],self.patch_grid)
            regions.extend(r for r in found if r.severity>0)
        candidates=[]
        for region in regions:
            actions=("joint","sampling") if region.variable=="pde_residual" else ("local_loss","joint")
            for action in actions:candidates.append(self.controller._candidate(region,action))
        self._route_candidate_losses(candidates)
        return candidates

    def _state(self):
        return {"model":self._model_snapshot(),"adam":deepcopy(self.optimizer.state_dict()),
                "allocation":None if self.controller is None else self.controller.state.snapshot(),
                "sampling_rng":deepcopy(self.sampling_rng.bit_generator.state),
                "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
                "torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all() if self.device.type=="cuda" else None}

    def _restore(self,s):
        self._restore_model(s["model"]);self.optimizer.load_state_dict(deepcopy(s["adam"]))
        if self.controller is not None:self.controller.state.restore(s["allocation"])
        self.sampling_rng.bit_generator.state=deepcopy(s["sampling_rng"])
        random.setstate(s["python_rng"]);np.random.set_state(s["numpy_rng"]);torch.set_rng_state(s["torch_rng"].cpu())
        if s["cuda_rng"] is not None:torch.cuda.set_rng_state_all([v.cpu() for v in s["cuda_rng"]])

    def _commit(self,rows):self._commit_rows(rows)

    def _guard_safe(self,current,anchor):
        keys=("pde_mse","bc_mse","ic_mse","sparse_mse")
        return all(current[k]<=anchor[k]*(1+self.config["controller_v2"]["counterfactual_guard_margin"])+self.revision["guard_absolute_floor"] for k in keys)

    def _run_vanilla(self,schedule):
        if not self.resumed:
            batch=self._training_batch(adaptive=False)
            self._commit(self._train_steps(batch,schedule["warmup_steps"],"warmup"))
            self._validation();self._checkpoint(-1)
        for block in range(self.start_block,schedule["control_blocks"]):
            self._commit(self._train_steps(self._training_batch(adaptive=False),schedule["block_steps"],f"stable_vanilla_block_{block}"))
            self._checkpoint(block)

    def _checkpoint(self,block):
        controller_fields={}
        if self.controller is not None:
            for key in ("trust_radius","effectiveness","score_history","metric_history","decisions","fraction_ema"):
                if hasattr(self.controller,key):controller_fields[key]=deepcopy(getattr(self.controller,key))
        checkpoint={"state":self._state(),"controller_fields":controller_fields,"detector_history":deepcopy(self.detector._history),
                    "block":block,"config":self.config,"protocol_revision":"ad_v2_stable_v1",
                    "validation_scales":self.validation_scales,
                    "completed_optimization_seconds":self.completed_optimization_seconds+(time.perf_counter()-self.segment_started if self.segment_started else 0.)}
        for key in ("loss_rows","decision_rows","allocation_history","step_audit","long_guard_audit","proposal_audit",
                    "applied_optimizer_steps","optimizer_step_calls","objective_evaluation_count","diagnostic_evaluation_count",
                    "accepted_interventions","rejected_interventions","prefiltered_interventions","rollback_count",
                    "safeguard_objective_evaluations","continuation_replay_calls","accepted_parameter_steps","noop_parameter_steps",
                    "continuation_rollback_count"):
            checkpoint[key]=deepcopy(getattr(self,key))
        path=self.run_dir/"checkpoints"/f"revision_block_{block:02d}.pt"
        temporary=path.with_suffix(".pt.tmp");torch.save(checkpoint,temporary);temporary.replace(path)
        print("@@PROGRESS "+json.dumps({"step":self.applied_optimizer_steps,"total_steps":self.config["controller_v2"]["total_steps"],
            "optimizer_calls":self.optimizer_step_calls,"mode":self.mode,"seed":self.seed}),flush=True)

    def resume_from(self,path):
        # Only load trusted locally generated checkpoints. No dense-reference
        # metric is consulted when choosing the latest completed block.
        checkpoint=torch.load(path,map_location=self.device,weights_only=False)
        if checkpoint["config"]!=self.config:raise ValueError("Resume configuration differs from checkpoint")
        for key,value in checkpoint["controller_fields"].items():setattr(self.controller,key,value)
        self.detector._history=checkpoint["detector_history"]
        self._restore(checkpoint["state"])
        for key,value in checkpoint.items():
            if key not in {"state","controller_fields","detector_history","block","config","protocol_revision"}:setattr(self,key,value)
        self.start_block=checkpoint["block"]+1;self.resumed=True

    def _run_vara(self,schedule):
        if not self.resumed:
            self._commit(self._train_steps(self._training_batch(adaptive=False),schedule["warmup_steps"],"warmup"))
            self._validation();self._log_allocation(-1);self._checkpoint(-1)
        for block in range(self.start_block,schedule["control_blocks"]):
            snapshot=self._diagnose();metrics=self._controller_metrics(snapshot)
            self.controller.update_history(snapshot.names,snapshot.normalized_scores,metrics)
            if self.revision["all_channel_candidate_availability"]:candidates=self._all_channel_candidates(snapshot)
            else:
                regions=self.detector.detect(snapshot.normalized_scores,snapshot.names,self.patch_grid)
                candidates=self.controller.candidates(regions);self._route_candidate_losses(candidates)
            candidates=self._filter_ablation_candidates(candidates)
            ranked=self.controller.rank(candidates,self._candidate_influence(candidates))
            self.proposal_audit.append({"block":block,"names":snapshot.names,"raw":snapshot.raw_scores,"normalized":snapshot.normalized_scores,
                "ranked":[{**c.to_record(),"screen_target_score":getattr(c,"screen_target_score",None),"screen_damage_score":getattr(c,"screen_damage_score",None)} for c in ranked]})
            for c in ranked:
                if c.prefiltered:
                    self.prefiltered_interventions+=1;self._record_decision(block,c,self.controller.record_prefilter(c,update_trust=False))
            active=[c for c in ranked if not c.prefiltered]
            if not active:
                self._commit(self._train_steps(self._training_batch(adaptive=True),schedule["block_steps"],f"stable_no_action_{block}"))
                self._log_allocation(block);self._checkpoint(block);continue
            c=active[0];initial=self._state();start=self.applied_optimizer_steps
            neutral_batch=self._training_batch(adaptive=True)
            neutral_rows=self._train_steps(neutral_batch,schedule["probe_steps"],f"stable_neutral_probe_{block}")
            ns=self._diagnose();nm=self._controller_metrics(ns);neutral=self._state()
            self._restore(initial);self.controller.apply(c);candidate_batch=self._training_batch(adaptive=True)
            candidate_rows=self._train_steps(candidate_batch,schedule["probe_steps"],f"stable_candidate_probe_{block}")
            cs=self._diagnose();cm=self._controller_metrics(cs)
            accepted,d=self.controller.evaluate(c,self._candidate_score(c,ns),self._candidate_score(c,cs),nm,cm,
                target_threshold=self.config["controller_v2"]["counterfactual_target_margin"],guard_threshold=self.config["controller_v2"]["counterfactual_guard_margin"],comparison_mode="counterfactual")
            if accepted:
                self.accepted_interventions+=1;batch=candidate_batch;self._commit(candidate_rows)
            else:
                self.rejected_interventions+=1;self.rollback_count=getattr(self,"rollback_count",0)+1
                self._restore(neutral);batch=neutral_batch;self._commit(neutral_rows)
            self._record_decision(block,c,d)
            # Retain shadow normal trust/memory; neutral snapshots include the
            # model/Adam/allocation/sampler needed for faithful replay.
            consumed=0;remaining=schedule["block_steps"]-schedule["probe_steps"]
            while consumed<remaining:
                chunk=min(self.revision["guard_recheck_steps"],remaining-consumed)
                rows=self._train_steps(batch,chunk,f"stable_continuation_{block}");self._commit(rows);consumed+=chunk
                if accepted and self.revision["continuation_guard_rechecks"]:
                    current=self._validation();safe=self._guard_safe(current,ns.revision_metrics)
                    self.long_guard_audit.append({"block":block,"step":self.applied_optimizer_steps,"anchor":nm,"current":current,"safe":safe,"anchor_rule":"same_block_neutral_probe"})
                    if not safe:
                        # Remove all retained candidate progress for this block,
                        # replay from the saved trained neutral probe, preserve
                        # every physical call/audit, and disable action until
                        # the next control cycle. No final/test metric is used.
                        replay=consumed;self._restore(neutral)
                        self.loss_rows=self.loss_rows[:start]
                        self.applied_optimizer_steps=start
                        self._commit(neutral_rows)
                        self._commit(self._train_steps(neutral_batch,replay,f"stable_neutral_replay_{block}"))
                        self.continuation_replay_calls+=replay;self.rollback_count=getattr(self,"rollback_count",0)+1
                        self.continuation_rollback_count+=1
                        reversal={**d,"accepted":False,"rollback_reason":"continuation_component_guard_violation",
                                  "comparison_mode":"continuation_safety_recheck"}
                        reversal=self.controller.commit_evaluation(c,False,reversal)
                        self._record_decision(block,c,reversal)
                        accepted=False;batch=neutral_batch
            self._log_allocation(block);self._checkpoint(block)

    def run(self):
        self.segment_started=time.perf_counter()
        try:metrics=super().run()
        finally:
            for filename,value in [("step_safeguard_audit.json",self.step_audit),("continuation_guard_audit.json",self.long_guard_audit),("all_channel_proposal_audit.json",self.proposal_audit)]:
                (self.run_dir/filename).write_text(json.dumps(serial(value),indent=2,allow_nan=False),encoding="utf-8")
        metrics.update(protocol_revision="ad_v2_stable_v1",safeguard_objective_evaluations=self.safeguard_objective_evaluations,
                       continuation_replay_calls=self.continuation_replay_calls,accepted_physical_parameter_steps=self.accepted_parameter_steps,
                       noop_physical_parameter_steps=self.noop_parameter_steps,continuation_rollback_count=self.continuation_rollback_count,
                       resumed_from_completed_block=self.resumed,
                       block_end_retained_interventions=self.accepted_interventions-self.continuation_rollback_count,
                       zero_physical_parameter_updates=sum(r["zero_parameter_update"] for r in self.step_audit))
        metrics["optimization_wall_clock_sec"]+=self.completed_optimization_seconds
        summary=json.loads((self.run_dir/"summary.json").read_text());summary["metrics"]=metrics;summary["protocol_revision"]="ad_v2_stable_v1"
        (self.run_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        import pandas as pd
        pd.DataFrame([metrics]).to_csv(self.run_dir/"metrics.csv",index=False)
        return metrics
