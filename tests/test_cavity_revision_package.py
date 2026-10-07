"""Archive integrity and negative mechanism evidence using synthetic fixtures."""
import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts.package_cavity_full_guard_revision import flatten, verify


def test_prevention_selection_retains_negative_events():
    frame = flatten([
        {"block": 0, "accepted": False, "prefiltered": False,
         "observed_target_improvement": .1, "guard_changes": {"pde": .12}, "guard_noise": {"pde": .02}},
        {"block": 1, "accepted": False, "prefiltered": True},
        {"block": 2, "accepted": True, "observed_target_improvement": .2,
         "guard_changes": {"pde": -.1}, "guard_noise": {"pde": .02}},
    ])
    assert frame.prevented_harm.tolist() == [True, False, False]
    assert frame.iloc[0].triggered_guards == "pde"
    assert flatten([]).empty


def test_secondary_budget_comes_from_actual_vara_manifest(tmp_path):
    from src.training.cavity_full_guard_audit import control_plan
    from src.utils.io import save_json
    path = tmp_path / "vara_v2_full_guard" / "revision_manifest.json"
    save_json({"config": {"device": "cpu"}, "compute": {"total_optimizer_calls": 4160,
        "gradient_screen_evaluations": 16, "gradient_backpropagations": 80}}, path)
    plan = control_plan(tmp_path)
    assert plan["optimizer_steps"] == 4160
    assert plan["gradient_backpropagations"] == 80
    assert plan["command"][-1] == "--compute_matched_control"


def test_secondary_rejects_dirty_source_before_training(tmp_path, monkeypatch):
    from src.training.cavity_full_guard_audit import run_control
    from src.utils.io import save_json
    import scripts.run_cavity_full_guard_revision as runner
    save_json({"revision_sha": "frozen"}, tmp_path / "provenance.json")
    monkeypatch.setattr(runner, "provenance", lambda: {"git_status": " M source.py", "revision_sha": "frozen"})
    with pytest.raises(RuntimeError, match="clean original"):
        run_control(tmp_path, {}, {})


def test_zip_member_hash_verification(tmp_path):
    content = b"unfavorable raw result preserved\n"
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("raw/result.csv", content)
        zipped.writestr("checksums/SHA256SUMS.txt", hashlib.sha256(content).hexdigest() + "  raw/result.csv\n")
    assert verify(archive)["all_member_checksums"] == "PASS"
    with zipfile.ZipFile(archive, "a") as zipped:
        zipped.writestr("unlisted.txt", b"unexpected")
    with pytest.raises(AssertionError):
        verify(archive)


def test_complete_synthetic_report_and_archive(tmp_path, monkeypatch):
    """Exercise all 36 figure exports and package paths without training."""
    import json
    import numpy as np
    import torch
    import scripts.package_cavity_full_guard_revision as packaging
    from src.training.cavity_full_guard_revision import CONFIG, RevisionTrainer, file_hash, state_hash
    from src.training.checkpointing import save_checkpoint
    from src.utils.config import load_config
    from src.utils.io import save_json
    output = tmp_path / "synthetic"
    output.mkdir()
    current_sha = packaging.subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    save_json({"revision_sha": current_sha, "fixture": "synthetic; no training"}, output / "provenance.json")
    for method in ("vanilla", "vara_v2_full_guard"):
        c = load_config(CONFIG)
        c["device"] = "cpu"
        c["experiments"] = {"root": str(output / method), "flat_layout": True}
        trainer = RevisionTrainer(c, method)
        checkpoint = trainer.checkpoint_dir / "final.pt"
        metrics = {"velocity_full_rel_l2": 1., "pde_residual_mean": .1}
        save_checkpoint(checkpoint, trainer.model, trainer.optimizer, c, metrics, 4000, -1)
        work = {"committed_steps": 4000, "total_optimizer_calls": 4000, "discarded_branch_steps": 0,
            "action_probe_steps": 0, "neutral_probe_steps": 0, "retained_probe_steps": 0,
            "gradient_screen_evaluations": 1 if method != "vanilla" else 0,
            "gradient_backpropagations": 2 if method != "vanilla" else 0,
            "diagnostic_evaluations": 1 if method != "vanilla" else 0,
            "objective_evaluations": 4001 if method != "vanilla" else 4000,
            "optimization_wall_clock_sec": 0., "controller_seconds": 0., "total_wall_clock_seconds": 0.}
        manifest = {"method": method, "config": c, "metrics": metrics, "compute": work,
            "initial_model_sha256": trainer.initial_hash, "final_model_sha256": state_hash(trainer.model.state_dict()),
            "final_checkpoint_sha256": file_hash(checkpoint), "parameter_count": 8707,
            "sparse": trainer.sparse_manifest, "boundary_schedule_hashes": [], "trajectory_batch_schedule": [], "final_state_rule": "final_committed_adam",
            "sparse_polish": False, "repair": False, "restore_best": False, "early_stopping": False,
            "controller_reference_access": False, "guard_active": method != "vanilla", "completed": True,
            "guard_execution": {"accepted": 0, "rejected": 0, "prefiltered": 0}}
        save_json(manifest, output / method / "revision_manifest.json")
        trainer.flush()
        save_json([], output / method / "vara_v2_allocation_history.json")
    real_check_output = packaging.subprocess.check_output
    def clean_fixture(args, **kwargs):
        if args[:3] == ["git", "status", "--porcelain"]:
            return ""
        return real_check_output(args, **kwargs)
    monkeypatch.setattr(packaging.subprocess, "check_output", clean_fixture)
    destination = tmp_path / "supplement"
    packaging.report(output, destination)
    assert len(list((destination / "figures").glob("*"))) == 36
    packaging.package(output, destination)
    result = packaging.verify(tmp_path / packaging.ZIP_NAME)
    assert result["crc"] == "PASS" and result["files"] > 100
    assert (destination / "raw" / "vanilla" / "checkpoints" / "final.pt").read_bytes() == (output / "vanilla" / "checkpoints" / "final.pt").read_bytes()
