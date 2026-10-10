"""Cheap causal micro-tests; never run a historical4000-step CPU substitute."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch

from scripts.run_ad_v2_forced_neutral import ForcedNeutralTrainer,state_hash
from src.pde_generalization.trainer import PDEGeneralizationTrainer
from src.pde_generalization.models import model_parameter_hash
from src.pde_generalization.diagnostics import PDEPatchGrid,_normalize_scores
from src.pde_generalization.benchmarks import AdvectionDiffusionBenchmark,AllenCahnBenchmark
from src.pde_generalization.residuals import compute_residuals
from src.controllers.v2_controller import VARAV2Controller,V2ControllerConfig
from src.diagnostics.weak_region_detector import WeakRegionDetector

torch.set_num_threads(2)

def cfg():
    return {"benchmark":"advection_diffusion","seed":8,"device":"cpu","dtype":"float32",
        "benchmark_params":{"bounds":[0.,1.,0.,1.],"t_bounds":[0.,1.],"kappa":.01,"advection_velocity":[1.,.5],"sigma":.09},
        "model":{"input_dim":3,"output_dim":1,"hidden_layers":[8,8],"activation":"tanh"},
        "training":{"n_collocation":32,"n_boundary":8,"n_initial":8,"n_sparse_data":8,"lr":.001,"gradient_clip":10.,
                    "weights":{"pde":1.,"bc":10.,"ic":10.,"sparse_data":2.}},
        "patches":{"nx_patches":5,"ny_patches":5,"nt_patches":3},
        "diagnostics":{"n_interior":32,"n_boundary":8,"n_initial":8},
        "controller_v2":{"total_steps":4,"warmup_steps":1,"control_blocks":1,"block_steps":3,"probe_steps":1,
                         "gradient_prefilter_enabled":False,"counterfactual_probe_enabled":True,"rollback_enabled":True},
        "evaluation":{"nx":4,"ny":4,"nt":3,"controller_reference_metrics_enabled":False},"plots":{"enabled":False}}

def test_forced_neutral_matches_advanced_original_sampler_without_probes(tmp_path):
    a=ForcedNeutralTrainer(cfg(),tmp_path/"forced")
    b=PDEGeneralizationTrainer(cfg(),"vara_v2",tmp_path/"neutral_direct")
    for t in [a,b]:t._commit_rows(t._train_steps(t._training_batch(adaptive=False),1,phase="warmup"))
    b._commit_rows(b._train_steps(b._training_batch(adaptive=True),3,phase="neutral_direct"))
    before=a._diagnose();regions=a.detector.detect(before.normalized_scores,before.names,a.patch_grid)
    candidate=a.controller.candidates(regions)[0]
    a._counterfactual_block(0,candidate,before,a._controller_metrics(before),a._schedule())
    assert model_parameter_hash(a.model)==model_parameter_hash(b.model)
    assert state_hash(a.optimizer.state_dict())==state_hash(b.optimizer.state_dict())
    assert state_hash(a.sampling_rng.bit_generator.state)==state_hash(b.sampling_rng.bit_generator.state)
    assert a.applied_optimizer_steps==b.applied_optimizer_steps==4
    assert a.optimizer_step_calls==5 and b.optimizer_step_calls==4
    assert a.probe_records[0]["restoration_verified"]

def test_forces_neutral_even_when_proposed_policy_accepts(tmp_path,monkeypatch):
    def proposed_accept(self,candidate,*args,**kwargs):
        return True,{"accepted":True,"observed_target_improvement":.1,"guard_changes":{},"guard_noise":{},"rollback_reason":""}
    monkeypatch.setattr(VARAV2Controller,"evaluate",proposed_accept)
    a=ForcedNeutralTrainer(cfg(),tmp_path/"force_accept_shadow")
    a._run_vara(a._schedule())
    assert a.accepted_interventions==0 and a.rejected_interventions==1
    assert a.probe_records[0]["proposed_accepted"] is True
    assert a.probe_records[0]["restoration_verified"] and a.probe_records[0]["retained_accepted"] is False

def test_parity_sampler_forced_neutral_is_bitwise_vanilla(tmp_path):
    a=ForcedNeutralTrainer(cfg(),tmp_path/"forced_direct_uniform",parity_sampler=True)
    b=PDEGeneralizationTrainer(cfg(),"vanilla",tmp_path/"vanilla")
    a._run_vara(a._schedule());b._run_vanilla(b._schedule())
    assert model_parameter_hash(a.model)==model_parameter_hash(b.model)
    assert state_hash(a.optimizer.state_dict())==state_hash(b.optimizer.state_dict())
    assert state_hash(a.sampling_rng.bit_generator.state)==state_hash(b.sampling_rng.bit_generator.state)

def test_original_uniform_regional_trajectory_differs_from_vanilla(tmp_path):
    a=ForcedNeutralTrainer(cfg(),tmp_path/"forced_original")
    b=PDEGeneralizationTrainer(cfg(),"vanilla",tmp_path/"vanilla")
    a._run_vara(a._schedule());b._run_vanilla(b._schedule())
    assert a.initial_model_parameter_hash==b.initial_model_parameter_hash and a.sparse_sample_hash==b.sparse_sample_hash
    assert model_parameter_hash(a.model)!=model_parameter_hash(b.model)
    assert state_hash(a.sampling_rng.bit_generator.state)!=state_hash(b.sampling_rng.bit_generator.state)

def test_scale_invariance_and_top_two_candidate_availability_bottleneck():
    raw=np.ones((3,75));raw[0,0]=10;raw[0,1]=9;raw[1]*=1000;raw[1,2]=2000;raw[2]*=1e5
    normalized=np.vstack([_normalize_scores(r) for r in raw])
    changed=raw.copy();changed[1]*=17
    assert np.allclose(normalized,np.vstack([_normalize_scores(r) for r in changed]))
    grid=PDEPatchGrid((0,1,0,1),(0,1),5,5,3)
    detector=WeakRegionDetector(percentile_threshold=80,top_k_per_variable=2,max_active_patches=8)
    regions=detector.detect(normalized,["pde_residual","boundary_mismatch","sparse_u_mismatch"],grid)
    assert [r.variable for r in regions[:2]]==["pde_residual","pde_residual"]
    controller=VARAV2Controller(V2ControllerConfig(num_patches=75))
    candidates=controller.candidates(regions)
    assert all(c.variable=="pde_residual" for c in candidates)

def test_ad_constant_shift_nullspace_is_not_ac_nullspace():
    coords=torch.tensor([[.2,.3,.4],[.7,.5,.8]],dtype=torch.float64)
    for equation,name in [(AdvectionDiffusionBenchmark({}),"f_advdiff"),(AllenCahnBenchmark({}),"f_ac")]:
        original=compute_residuals(equation.exact,coords,equation)[name]
        shifted=compute_residuals(lambda p:equation.exact(p)+.17,coords,equation)[name]
        if name=="f_advdiff":assert torch.equal(original,shifted)
        else:
            u=equation.exact(coords).detach();expected=.17*(3*u.square()-1)+3*u*.17**2+.17**3
            assert torch.allclose(shifted-original,expected,atol=1e-10) and shifted.abs().max()>1e-4
