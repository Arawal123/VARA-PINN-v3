"""Small CPU protocol tests; never launch a 4000-step research experiment."""

from argparse import Namespace
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from scripts.run_vara_v2_continuation import (
    _apply_data_supervision_settings, _apply_re_aware_cavity_settings,
    _continuation_validity, _fair_revision_plan, _load_base_config,
)
from src.training import cavity_fair_revision as fair
from src.training.vara_trainer import VARATrainer
from src.training.vara_v2_trainer import VARAV2Trainer, SPARSE_CURRICULUM_ALLOWED_METRICS
from src.utils.config import deep_update, load_config


def _resolved():
    base = _load_base_config("configs/vara_v2/lid_driven_cavity.yaml")
    for path in ("controller.yaml", "continuation.yaml", "lid_cavity_continuation_reliable.yaml", "presets/reliable.yaml"):
        base = deep_update(base, load_config("configs/vara_v2/" + path))
    base = _apply_data_supervision_settings(base, mode="sparse_cfd_polish")
    return _apply_re_aware_cavity_settings(base, 100.0)


def _args(**updates):
    values = dict(
        reynolds=[100.0], methods=["vanilla", "vara_v2"], seeds=[0, 1, 2, 3, 4],
        quick=False, enhanced_backbone=False, disable_stabilizers=False,
        preset=None, data_supervision="sparse_cfd_polish", cavity_base_formulation=None,
        cfd_sample_fraction=0.01, cfd_sample_count=None, cfd_include_pressure=False,
        cfd_include_vorticity=False, cfd_seed=None, output_dir="unused",
    )
    return Namespace(**{**values, **updates})


def _tiny_config(tmp_path, *, enabled=True):
    path = tmp_path / "reference.npz"
    axis = np.linspace(0.0, 1.0, 11)
    x, y = np.meshgrid(axis, axis)
    np.savez(path, x=x.ravel(), y=y.ravel(), u=(x+y).ravel(), v=(x-y).ravel())
    cfg = _resolved()
    if enabled:
        cfg = fair.apply_fair_revision(cfg)
    return deep_update(cfg, {
        "device": "cpu", "model": {"hidden_layers": [4, 4]},
        "benchmark_params": {"full_field_reference_path": str(path), "profile_only": False},
        "data_supervision": {"reference_path": str(path)},
        "training": {
            "adaptive_cycles": 2, "epochs_per_cycle": 2, "log_every": 100,
            "n_collocation": 8, "n_boundary": 8,
            "collocation_curriculum": {"stages": [{"until_step": 4, "n_collocation": 8, "n_boundary": 8}]},
        },
        "controller_v2": {
            "total_steps": 4, "warmup_steps": 2, "control_blocks": 1,
            "block_steps": 2, "probe_steps": 1,
            "gradient_probe_interior": 4, "gradient_probe_boundary": 4,
            "sparse_polish_curriculum": {"update_every_steps": 2},
        },
        "validation": {"nx": 4, "ny": 4}, "test": {"nx": 4, "ny": 4},
        "patches": {"nx_patches": 2, "ny_patches": 2},
        "continuation_replay": {"enabled": False},
        "optimizer": {"final_repair": {"enabled": False}},
        "experiments": {"root": str(tmp_path / "run"), "flat_layout": True},
    })


def test_fair_overlay_applied_after_materialization_preserves_science():
    old = _resolved()
    old_copy = deepcopy(old)
    cfg = fair.apply_fair_revision(old)
    assert old == old_copy
    assert old["optimizer"]["final_repair"]["enabled"]
    assert not cfg["optimizer"]["final_repair"]["enabled"]
    assert cfg["optimizer"]["final_repair"]["epochs"] == 0
    assert cfg["data_supervision"]["polish"]["final_repair_steps"] == 0
    assert not cfg["convergence_early_stopping"]["enabled"]
    assert not cfg["checkpoint"]["restore_best_before_final"]
    assert cfg["compute_budget"] == {"enabled": True, "type": "applied_optimizer_steps", "value": 4000}
    assert cfg["training"] == old["training"]
    assert cfg["sampling"] == old["sampling"]
    assert cfg["model"] == old["model"]
    assert cfg["controller_v2"]["sparse_polish_curriculum"]["enabled"]
    assert cfg["controller_v2"]["sparse_polish_curriculum"]["disable_generic_interventions"]


def test_fair_topology_cannot_gate_and_unresolved_continuation_is_rejected():
    bad = {"lid_cavity_topology_aligned": 0, "lid_cavity_topology_score": float("nan")}
    old = _resolved()
    assert not _continuation_validity(bad, old)["continuation_stage_valid"]
    assert _continuation_validity(bad, fair.apply_fair_revision(old))["continuation_stage_valid"]
    with pytest.raises(ValueError, match="disabled pending manuscript"):
        fair.prepare_fair_revision_args(_args(reynolds=[100, 150, 200, 300, 400, 600, 800, 1000]))
    with pytest.raises(ValueError, match="disabled pending manuscript"):
        fair.prepare_fair_revision_args(_args(reynolds=[100, 400, 1000]))


@pytest.mark.parametrize("enabled", [False, True])
def test_curriculum_final_restore_switch_preserves_historical_default(tmp_path, enabled):
    trainer = VARAV2Trainer(_tiny_config(tmp_path, enabled=enabled))
    trainer._initialize_sparse_polish_curriculum()
    metrics = {name: 0.05 for name in SPARSE_CURRICULUM_ALLOWED_METRICS}
    trainer._update_sparse_curriculum_best(metrics)
    with torch.no_grad():
        next(trainer.model.parameters()).add_(1.0)
    final = fair.tensor_mapping_sha256(trainer.model.state_dict())
    restored = trainer._restore_best_sparse_curriculum_if_needed({k: 1.0 for k in metrics})
    assert restored is (not enabled)
    assert (fair.tensor_mapping_sha256(trainer.model.state_dict()) == final) is enabled
    assert trainer.sparse_polish_curriculum_enabled


def test_fair_short_cpu_pair_evaluates_final_adam_state_and_writes_evidence(tmp_path, monkeypatch):
    # Scale only the test budget; the public runner still accepts exactly 4000.
    monkeypatch.setattr(fair, "PRIMARY_STEPS", 4)
    cfg = _tiny_config(tmp_path)
    vanilla = VARATrainer(deep_update(cfg, {"experiments": {"root": str(tmp_path / "vanilla")}}), mode="vanilla_pinn")
    vara = VARAV2Trainer(deep_update(cfg, {"experiments": {"root": str(tmp_path / "vara")}}))
    assert vanilla.fair_initial_model_sha256 == vara.fair_initial_model_sha256
    assert torch.equal(vanilla.cfd_supervision.coords, vara.cfd_supervision.coords)
    for name in vanilla.cfd_supervision.targets:
        assert torch.equal(vanilla.cfd_supervision.targets[name], vara.cfd_supervision.targets[name])
    assert not vara.v2_config.counterfactual_probe_enabled

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference selection or L-BFGS cannot run in fair mode")

    monkeypatch.setattr(torch.optim, "LBFGS", forbidden)
    manifests = []
    for method, trainer in (("vanilla", vanilla), ("vara_v2", vara)):
        monkeypatch.setattr(trainer, "save_plots", lambda *a: None)
        monkeypatch.setattr(trainer, "_checkpoint_score", forbidden)
        before_eval = []
        evaluate = trainer.evaluate_metrics

        def final_evaluate(coords, trainer=trainer, evaluate=evaluate):
            before_eval.append(fair.tensor_mapping_sha256(trainer.model.state_dict()))
            return evaluate(coords)

        monkeypatch.setattr(trainer, "evaluate_metrics", final_evaluate)
        metrics = trainer.run()
        assert metrics["applied_optimizer_steps"] == 4
        assert metrics["optimizer_steps"] == 4
        assert metrics["probe_optimizer_steps"] == 0
        assert metrics["final_repair_executed"] is False
        assert metrics["final_model_source"] == "final_primary_adam_step"
        assert not trainer.early_stopped
        assert before_eval == [fair.tensor_mapping_sha256(trainer.model.state_dict())]
        assert not (trainer.checkpoint_dir / "best.pt").exists()
        saved = torch.load(trainer.checkpoint_dir / "final.pt", weights_only=False)
        assert saved["epoch"] == 4
        manifests.append(fair.fairness_manifest(trainer, method))

    report = fair.paired_fairness_report(*manifests)
    assert report["matched"], report
    changed = deepcopy(manifests[1])
    changed["sparse_pool_content_sha256"] = "different labels"
    assert not fair.paired_fairness_report(manifests[0], changed)["matched"]
    json.dumps(manifests)  # Manifest is machine-readable without tensor objects.


def test_fair_rejects_incomplete_primary_budget(tmp_path):
    trainer = VARATrainer(_tiny_config(tmp_path), mode="vanilla_pinn")
    with pytest.raises(RuntimeError, match="exactly the final"):
        fair.assert_fair_final_state(trainer)


def test_preflight_resolves_real_pools_without_output_writes(tmp_path):
    args = _args(output_dir=str(tmp_path / "must_not_exist"))
    fair.prepare_fair_revision_args(args)
    path = "data/references/lid_driven_cavity/full_field/re_0100_paddlescience.npz"
    plan = _fair_revision_plan(_resolved(), args, {100.0: path})
    assert not (tmp_path / "must_not_exist").exists()
    assert [p["sparse_pool_seed"] for p in plan["pools"]] == [0, 1, 2, 3, 4]
    assert {p["sparse_cfd_sample_count"] for p in plan["pools"]} == {639}
    assert len({p["source_cfd_sha256"] for p in plan["pools"]}) == 1


def test_both_real_run_loops_dispatch_exactly_4000_primary_steps(tmp_path, monkeypatch):
    """Exercise orchestration at the real budget, with optimization/diagnostics mocked."""
    import src.training.vara_trainer as vanilla_module

    monkeypatch.setattr(vanilla_module, "save_patch_score_map", lambda *a: None)
    cfg = fair.apply_fair_revision(_resolved())
    cfg["device"] = "cpu"
    reference = "data/references/lid_driven_cavity/full_field/re_0100_paddlescience.npz"
    cfg["benchmark_params"].update({"full_field_reference_path": reference, "profile_only": False})
    cfg["data_supervision"]["reference_path"] = reference
    coords = np.array([[0.25, 0.25], [0.75, 0.75]])
    raw = np.ones((1, 16))
    for method in ("vanilla", "vara_v2"):
        config = deep_update(cfg, {"experiments": {"root": str(tmp_path / method)}})
        trainer = VARATrainer(config, mode="vanilla_pinn") if method == "vanilla" else VARAV2Trainer(config)
        chunks = []

        def fake_steps(batch, steps, **kwargs):
            chunks.append(steps)
            trainer.global_step += steps
            trainer.compute_tracker.optimizer_steps += steps
            trainer.compute_tracker.applied_optimizer_steps += steps

        def fake_final():
            fair.assert_fair_final_state(trainer)
            return trainer.compute_tracker.summary()

        monkeypatch.setattr(trainer, "initial_batch", lambda: {})
        monkeypatch.setattr(trainer, "validation_grid", lambda: (None, None, coords))
        monkeypatch.setattr(trainer, "controller_metrics", lambda xy: {})
        monkeypatch.setattr(trainer, "evaluate_and_save_final", fake_final)
        if method == "vanilla":
            monkeypatch.setattr(trainer, "train_epochs", lambda batch, state, **kw: fake_steps(batch, 200, **kw))
            monkeypatch.setattr(trainer, "diagnose", lambda: ({}, raw, ["continuity_residual"], [], None, None, coords))
            monkeypatch.setattr(trainer, "resample_batch", lambda *a, **kw: {})
        else:
            monkeypatch.setattr(trainer, "_train_v2_steps", fake_steps)
            monkeypatch.setattr(trainer, "_diagnose_reference_free", lambda **kw: ({}, raw, ["continuity_residual"], [], coords))
            monkeypatch.setattr(trainer, "_guard_metrics", lambda xy: {})
            monkeypatch.setattr(trainer, "_resample_v2_batch", lambda *a: {})
        metrics = trainer.run()
        assert sum(chunks) == 4000
        assert metrics["applied_optimizer_steps"] == 4000
        assert metrics["optimizer_steps"] == 4000
