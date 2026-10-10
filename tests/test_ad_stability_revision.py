"""Mechanism tests for the opt-in revision, using tiny CPU models only."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from scripts.run_ad_v2_stability import smoke_config
from scripts.run_ad_v2_forced_neutral import state_hash
from src.controllers.v2_controller import V2Candidate, V2ControllerConfig
from src.pde_generalization.ad_stability_trainer import ADStabilityTrainer, CalibratedADController, GUARDS
from src.pde_generalization.diagnostics import DiagnosticSnapshot
from src.pde_generalization.losses import LossResult
from src.pde_generalization.models import model_parameter_hash

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]

def cfg():
    config = smoke_config(yaml.safe_load((ROOT/"configs/pde_generalization/advection_diffusion_v2_stable.yaml").read_text()))
    config.update(seed=8, device="cpu")
    return config

def trainer(path, mode="vara_v2"):
    return ADStabilityTrainer(cfg(), mode, path)

def test_armijo_backtracks_real_adam_overshoot_and_keeps_moments(tmp_path, monkeypatch):
    import src.pde_generalization.ad_stability_trainer as module
    t = trainer(tmp_path, "vanilla")
    t.model = nn.Linear(1, 1, bias=False)
    t.model.weight.data.fill_(.01)
    t.optimizer = torch.optim.Adam(t.model.parameters(), lr=.1)
    def quadratic(model, *args):
        loss = model.weight.square().sum()
        return LossResult(loss, {"pde": loss, "bc": loss*0, "ic": loss*0, "sparse_data": loss*0}, {})
    monkeypatch.setattr(module, "compute_training_loss", quadratic)
    rows = t._train_steps({}, 1, "overshoot_fixture")
    audit = t.step_audit[0]
    assert audit["trials"][0][1] > audit["loss_before"]
    assert audit["accepted_fraction"] == .125
    assert audit["loss_after"] < audit["loss_before"]
    assert t.optimizer_step_calls == 1 and t.objective_evaluation_count == 5
    assert int(t.optimizer.state[t.model.weight]["step"]) == 1
    assert rows[0]["step_fraction"] == .125

def test_all_failed_fractions_explicit_noop(tmp_path, monkeypatch):
    import src.pde_generalization.ad_stability_trainer as module
    t = trainer(tmp_path, "vanilla")
    t.model = nn.Linear(1, 1, bias=False);t.model.weight.data.fill_(.01)
    t.optimizer = torch.optim.Adam(t.model.parameters(), lr=.1)
    t.revision["step_fractions"] = [1.]
    monkeypatch.setattr(module, "compute_training_loss", lambda model,*args: LossResult(model.weight.square().sum(), {}, {}))
    before = model_parameter_hash(t.model)
    t._train_steps({}, 1, "noop_fixture")
    assert model_parameter_hash(t.model) == before
    assert t.noop_parameter_steps == 1 and t.optimizer_step_calls == 1
    assert int(t.optimizer.state[t.model.weight]["step"]) == 1

def test_neutral_path_is_bitwise_safeguarded_vanilla(tmp_path):
    a = trainer(tmp_path/"vara")
    b = trainer(tmp_path/"vanilla", "vanilla")
    a._all_channel_candidates = lambda snapshot: []
    a._run_vara(a._schedule());b._run_vanilla(b._schedule())
    assert a.initial_model_parameter_hash == b.initial_model_parameter_hash
    assert a.sparse_sample_hash == b.sparse_sample_hash
    assert model_parameter_hash(a.model) == model_parameter_hash(b.model)
    assert state_hash(a.optimizer.state_dict()) == state_hash(b.optimizer.state_dict())
    assert state_hash(a.sampling_rng.bit_generator.state) == state_hash(b.sampling_rng.bit_generator.state)
    assert a.optimizer_step_calls == b.optimizer_step_calls == 12
    assert a.applied_optimizer_steps == b.applied_optimizer_steps == 12

def test_reference_isolation_and_frozen_scales(tmp_path, monkeypatch):
    t = trainer(tmp_path)
    # Manufactured forcing is part of the given PDE and legitimately derives
    # from exact(). Prohibit every OTHER exact() access after fixed labels are
    # created, including dense error computation and controller gating.
    original_exact=t.benchmark.exact;original_forcing=t.benchmark.forcing
    allowed=[False]
    def prescribed_forcing(coords):
        allowed[0]=True
        try:return original_forcing(coords)
        finally:allowed[0]=False
    def isolated_exact(coords):
        if not allowed[0]:raise AssertionError("held-out reference access")
        return original_exact(coords)
    monkeypatch.setattr(t.benchmark,"forcing",prescribed_forcing)
    monkeypatch.setattr(t.benchmark,"exact",isolated_exact)
    t._run_vara(t._schedule())
    assert t.applied_optimizer_steps == 12
    assert set(t.validation_scales) == {"pde", "bc", "ic", "sparse_data"}
    scales = dict(t.validation_scales)
    t._diagnose();assert t.validation_scales == scales
    for row in t.step_audit:
        assert row["loss_after"] <= row["loss_before"]

def test_all_four_channels_have_candidate_access(tmp_path):
    t = trainer(tmp_path)
    names = ["pde_residual", "boundary_mismatch", "initial_condition_mismatch", "sparse_u_mismatch"]
    raw = np.ones((4,75));raw[:,0] = [10.,2.,3.,4.]
    snapshot = DiagnosticSnapshot(names, raw, raw)
    candidates = t._all_channel_candidates(snapshot)
    assert len(candidates) == 8
    assert {c.variable for c in candidates} == set(names)
    for c in candidates:
        t.controller.apply(c);t.controller.validate_state()
    assert np.isclose(t.controller.state.sampling_mass.sum(), 1.)

@pytest.mark.parametrize("harm", ["ic_mse", "sparse_mse"])
def test_component_guard_rejects_harm_hidden_by_aggregate(harm):
    controller = CalibratedADController(V2ControllerConfig(num_patches=75, guard_metrics=list(GUARDS)))
    candidate = V2Candidate("pde_residual", 0, "joint", ["pde"], 1., 1, 0.)
    controller.rank([candidate], {})
    before = {k:1. for k in GUARDS};after = {k:.9 for k in GUARDS};after[harm] = 1.03
    accepted, record = controller.evaluate(candidate, 1., .9, before, after, target_threshold=.005, guard_threshold=.02)
    assert not accepted and record["rollback_reason"] == "pareto_guard_violation"
    assert record["guard_changes"][harm] > .02

def test_fraction_prediction_can_expand_trust_without_changing_prefilter():
    controller = CalibratedADController(V2ControllerConfig(num_patches=75, guard_metrics=list(GUARDS)))
    candidate = V2Candidate("pde_residual", 0, "joint", ["pde"], 100., 1, 0.)
    controller.rank([candidate], {candidate.key(): {"gradient_compatibility":.5,"gradient_conflict":0.}})
    assert candidate.screen_target_score == 75.
    assert 0 < candidate.predicted_target_improvement <= 1
    metrics = {k:1. for k in GUARDS}
    accepted, record = controller.evaluate(candidate, 1., .9, metrics, metrics, target_threshold=.005, guard_threshold=.02)
    assert accepted and record["trust_radius_after"] > record["trust_radius_before"]

def test_probe_rejection_restores_neutral_model_adam_and_rng(tmp_path, monkeypatch):
    a = trainer(tmp_path/"rejected");b = trainer(tmp_path/"direct")
    b._all_channel_candidates = lambda snapshot: []
    monkeypatch.setattr(a.controller, "evaluate", lambda *args, **kwargs: (False, {"accepted":False,"rollback_reason":"test_reject"}))
    a._run_vara(a._schedule());b._run_vara(b._schedule())
    assert model_parameter_hash(a.model) == model_parameter_hash(b.model)
    assert state_hash(a.optimizer.state_dict()) == state_hash(b.optimizer.state_dict())
    assert state_hash(a.sampling_rng.bit_generator.state) == state_hash(b.sampling_rng.bit_generator.state)
    assert a.applied_optimizer_steps == 12 and a.optimizer_step_calls == 12+2
    assert a.rejected_interventions == 2 and a.rollback_count == 2

def test_long_guard_replay_restores_true_neutral_trajectory(tmp_path, monkeypatch):
    a = trainer(tmp_path/"rolled_back");b = trainer(tmp_path/"neutral")
    b._all_channel_candidates = lambda snapshot: []
    monkeypatch.setattr(a.controller, "evaluate", lambda *args, **kwargs: (True, {"accepted":True,"reward_ratio":1.,"observed_target_improvement":.1}))
    monkeypatch.setattr(a, "_guard_safe", lambda *args: False)
    a._run_vara(a._schedule());b._run_vara(b._schedule())
    assert model_parameter_hash(a.model) == model_parameter_hash(b.model)
    assert state_hash(a.optimizer.state_dict()) == state_hash(b.optimizer.state_dict())
    assert state_hash(a.sampling_rng.bit_generator.state) == state_hash(b.sampling_rng.bit_generator.state)
    assert a.applied_optimizer_steps == 12
    assert a.optimizer_step_calls == 12+2+4  # 2 discarded probes, 2*2 replayed steps
    assert a.continuation_replay_calls == 4 and a.continuation_rollback_count == 2
    assert len(a.loss_rows) == 12

def test_complete_block_resume_is_exact(tmp_path):
    a = trainer(tmp_path/"uninterrupted");a._run_vara(a._schedule())
    b = trainer(tmp_path/"resumed")
    b.resume_from(a.run_dir/"checkpoints"/"revision_block_00.pt")
    b._run_vara(b._schedule())
    assert model_parameter_hash(a.model) == model_parameter_hash(b.model)
    assert state_hash(a.optimizer.state_dict()) == state_hash(b.optimizer.state_dict())
    assert a.optimizer_step_calls == b.optimizer_step_calls
    assert a.loss_rows == b.loss_rows
    assert a.decision_rows == b.decision_rows

def test_frozen_primary_protocol():
    config = yaml.safe_load((ROOT/"configs/pde_generalization/advection_diffusion_v2_stable.yaml").read_text())
    assert config["training"]["n_sparse_data"] == 507
    assert config["model"]["hidden_layers"] == [96]*5
    assert config["controller_v2"]["total_steps"] == 4000
    assert config["controller_v2"]["warmup_steps"]+7*500 == 4000
    assert config["controller_v2"]["probe_steps"] == 25
    assert config["evaluation"]["controller_reference_metrics_enabled"] is False
