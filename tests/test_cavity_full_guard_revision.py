"""Small regression/preflight checks; never run the 4000-step experiment."""
from copy import deepcopy
import subprocess

import numpy as np
import pytest
import torch

from src.controllers.v2_controller import VARAV2Controller, V2ControllerConfig, V2Candidate
from src.training.cavity_full_guard_revision import CONFIG, BASE_SHA, RevisionTrainer, validate_protocol, state_hash
from src.training.cavity_full_guard_audit import compare_initial
from src.utils.config import load_config


def config():
    result = load_config(CONFIG)
    result["device"] = "cpu"
    return result


def test_frozen_protocol_and_historical_config():
    c = config()
    validate_protocol(c)
    assert c["controller_v2"]["total_steps"] == 4000
    historical = "configs/vara_v2/lid_cavity_continuation_reliable.yaml"
    assert subprocess.check_output(["git", "show", f"{BASE_SHA}:{historical}"], text=True) == open(historical).read()


@pytest.mark.parametrize("mutation", [
    lambda c: c["controller_v2"].update(counterfactual_probe_enabled=False),
    lambda c: c["data_supervision"].update(mode="sparse_cfd_polish"),
    lambda c: c["training"]["weights"].update(cfd_velocity_mse=0),
    lambda c: c["optimizer"]["final_repair"].update(enabled=True),
    lambda c: c["checkpoint"].update(restore_best_before_final=True),
    lambda c: c["evaluation"].update(controller_reference_metrics_enabled=True),
])
def test_fail_closed(mutation):
    c = config()
    mutation(c)
    with pytest.raises(ValueError):
        validate_protocol(c)


def test_hash_sensitive_to_values_and_shape():
    assert state_hash(np.array([1., 2.], dtype="<f4")) == state_hash(np.array([1., 2.], dtype=">f4"))
    assert state_hash(np.array([1., 2.])) != state_hash(np.array([1., 3.]))
    assert state_hash(np.array([1., 2.])) != state_hash(np.array([[1., 2.]]))


def test_preflight_pair_and_snapshot(tmp_path):
    trainers = []
    for method in ("vanilla", "vara_v2_full_guard"):
        c = config()
        c["experiments"] = {"root": str(tmp_path / method), "flat_layout": True}
        trainers.append(RevisionTrainer(c, method))
    assert all(compare_initial(*trainers).values())
    a, b = trainers
    assert a.sparse_manifest["selected_count"] == 1279
    assert a.sparse_manifest["eligible_count"] == 63940
    assert a.sparse_manifest["requested_fraction"] == .02
    assert a.sparse_manifest["realized_fraction"] != .02
    pre = a.snapshot()
    a.optimizer.zero_grad()
    sum(p.square().sum() for p in a.model.parameters()).backward()
    a.optimizer.step()
    a.restore(pre)
    assert state_hash(a.snapshot()) == state_hash(pre)
    assert not a.benchmark.has_reference and not a.benchmark.has_profile_reference


def test_guard_rejects_harmful_target_gain():
    controller = VARAV2Controller(V2ControllerConfig(num_patches=16))
    candidate = V2Candidate("continuity_residual", 0, "sampling", ["continuity"], 1., 1, 0.)
    accepted, decision = controller.evaluate(candidate, 1., .8,
        {"pde_residual_mean": 1.}, {"pde_residual_mean": 1.1},
        target_threshold=.005, guard_threshold=.02, comparison_mode="counterfactual")
    assert not accepted and decision["rollback_reason"] == "pareto_guard_violation"
    with pytest.raises(ValueError):
        controller.evaluate(candidate, 1., .8, {"velocity_full_rel_l2": 1.}, {"velocity_full_rel_l2": .9})


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_repeated_probe_restores_do_not_mutate_adam_snapshot(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU covers same-device Adam tensor aliasing")
    c = config()
    c["device"] = device
    c["experiments"] = {"root": str(tmp_path), "flat_layout": True}
    trainer = RevisionTrainer(c, "vara_v2_full_guard")
    def synthetic_adam_step():
        # Initialize and mutate real Adam moments without PDE training.
        trainer.optimizer.zero_grad(set_to_none=True)
        sum(p.square().sum() for p in trainer.model.parameters()).backward()
        trainer.optimizer.step()
    synthetic_adam_step()
    pre = trainer.snapshot()
    saved_optimizer_hash = state_hash(pre["optimizer"])
    expected_start = state_hash({"model": pre["model"], "optimizer": pre["optimizer"]})
    for _ in range(3):
        trainer.restore(pre)
        assert state_hash({"model": trainer.model.state_dict(), "optimizer": trainer.optimizer.state_dict()}) == expected_start
        synthetic_adam_step()
        assert state_hash(pre["optimizer"]) == saved_optimizer_hash
    trainer.restore(pre)
    assert state_hash(trainer.snapshot()) == state_hash(pre)


def test_sparse_objective_has_nonzero_gradient(tmp_path):
    from src.losses.base_losses import compute_pointwise_losses
    c = config()
    c["experiments"] = {"root": str(tmp_path), "flat_layout": True}
    trainer = RevisionTrainer(c, "vanilla")
    batch = trainer.initial_batch()
    losses = compute_pointwise_losses(trainer.model, batch, trainer.benchmark, True,
                                     regularization_config=trainer._active_loss_config())
    loss = losses["cfd_velocity_mse"].mean() * c["training"]["weights"]["cfd_velocity_mse"]
    grads = torch.autograd.grad(loss, list(trainer.model.parameters()), allow_unused=True)
    assert loss.item() > 0 and any(g is not None and g.abs().sum() > 0 for g in grads)


def test_fixed_loop_rollback_evidence_without_training(tmp_path, monkeypatch):
    import src.training.cavity_full_guard_revision as revision
    monkeypatch.setattr(revision, "validate_protocol", lambda c: None)
    c = config()
    c["controller_v2"].update(total_steps=4, warmup_steps=2, control_blocks=1, block_steps=2, probe_steps=1)
    c["compute_budget"]["value"] = 4
    c["experiments"] = {"root": str(tmp_path), "flat_layout": True}
    trainer = RevisionTrainer(c, "vara_v2_full_guard")
    candidate = V2Candidate("continuity_residual", 0, "sampling", ["continuity"], 1., 1, 0.)
    monkeypatch.setattr(trainer.v2_controller, "candidates", lambda weak: [candidate])
    monkeypatch.setattr(trainer, "_candidate_influence", lambda candidates: {})
    raw_call = [0]
    def diagnose(*args):
        raw_call[0] += 1
        value = 1. if raw_call[0] <= 2 else .8
        return {}, np.full((1, 16), value), ["continuity_residual"], [], np.zeros((16, 2))
    monkeypatch.setattr(trainer, "_diagnose_reference_free", diagnose)
    guard_call = [0]
    def guard(coords):
        guard_call[0] += 1
        return {key: 1. if guard_call[0] <= 2 else 1.1 for key in revision.GUARDS}
    monkeypatch.setattr(trainer, "_guard_metrics", guard)
    def fake_steps(batch, steps, cycle, phase, probe=False, applied=True):
        # Counter/state simulation only; no optimization or forward/backward training.
        for _ in range(steps):
            trainer.compute_tracker.record_objective(batch)
            trainer.compute_tracker.record_optimizer_step(applied=applied)
            trainer.global_step += 1
        with torch.no_grad():
            next(trainer.model.parameters()).add_(steps * .001)
        if phase == "neutral_probe":
            trainer.work["neutral_probe_steps"] += steps
        if phase == "action_probe":
            trainer.work["action_probe_steps"] += steps
    monkeypatch.setattr(trainer, "_train_v2_steps", fake_steps)
    monkeypatch.setattr(trainer, "evaluate_and_save_final", lambda: {"pde_residual_mean": 1.})
    manifest = trainer.run()
    assert manifest["compute"]["committed_steps"] == 4
    assert manifest["compute"]["total_optimizer_calls"] == 5
    assert manifest["compute"]["discarded_branch_steps"] == 1
    decision = trainer.decisions[0]
    assert not decision["accepted"] and decision["rollback_executed"]
    assert decision["rollback_model_sha256"] == decision["neutral_model_sha256"]
    assert decision["effectiveness_after"] < decision["effectiveness_before"]
