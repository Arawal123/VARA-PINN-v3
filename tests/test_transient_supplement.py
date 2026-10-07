"""Postprocessing checks only; research-length training is never launched."""
from copy import deepcopy
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.package_transient_fullguard import (
    GUARD_FLAGS, audit_run, audit_pairs, exact_statistics, frozen_data,
    resolve_reliable, protocol, guard_evidence,
)
from src.pde_generalization.trainer import PDEGeneralizationTrainer
from tests.test_pde_generalization_smoke import _tiny_config


@pytest.mark.parametrize("benchmark", ["allen_cahn", "advection_diffusion"])
def test_resolved_reliable_protocol_and_frozen_data(benchmark):
    cfg = resolve_reliable(benchmark)
    assert cfg["training"]["n_sparse_data"] == 507
    assert cfg["training"]["weights"]["sparse_data"] == 2
    assert cfg["controller_v2"]["total_steps"] == 4000
    assert all(cfg["controller_v2"][f] for f in GUARD_FLAGS)
    assert cfg["evaluation"]["controller_reference_metrics_enabled"] is False
    cfg["seed"] = 4
    a, b = frozen_data(cfg, "cpu"), frozen_data(deepcopy(cfg), "cpu")
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    plan = protocol(benchmark, "cpu", [0, 1, 2, 3, 4])
    assert plan["trainable_parameters"] == 37729
    assert len({p["combined_sparse_hash"] for p in plan["paired_initialization_and_pools"]}) == 5


def test_exact_statistics_retains_small_n_and_negative_results():
    percent, values = exact_statistics(np.array([1., 2., 3., 4., 5.]), np.array([.5, 1., 1.5, 2., 2.5]))
    assert np.array_equal(percent, np.full(5, 50.))
    assert values["exact_wilcoxon_p"] == 0.0625
    assert values["wins_of_5"] == 5
    _, worse = exact_statistics(np.ones(5), np.array([2., 2., 2., 2., 2.]))
    assert worse["mean_paired_improvement_percent"] == -100
    assert worse["losses_of_5"] == 5
    percent, zero = exact_statistics(np.array([0., 1., 2., 3., 4.]), np.ones(5))
    assert np.isnan(percent[0]) and np.isnan(zero["mean_paired_improvement_percent"])
    _, ties = exact_statistics(np.ones(5), np.ones(5))
    assert ties["exact_wilcoxon_p"] == 1 and np.isnan(ties["cohen_dz"])


@pytest.mark.parametrize("benchmark", ["allen_cahn", "advection_diffusion"])
def test_small_cpu_pair_verifies_final_state_and_counter_accounting(tmp_path, benchmark):
    cfg = _tiny_config(benchmark)
    cfg["seed"] = 0
    audited = {}
    for method in ("vanilla", "vara_v2"):
        folder = tmp_path / method
        trainer = PDEGeneralizationTrainer(cfg, method, folder)
        trainer.run()
        audited[method] = audit_run(folder, benchmark, method, 0, strict=False)
    left, right = audited["vanilla"]["manifest"], audited["vara_v2"]["manifest"]
    assert left["initial_model_hash"] == right["initial_model_hash"]
    assert left["sparse_coordinates_hash"] == right["sparse_coordinates_hash"]
    assert left["sparse_targets_hash"] == right["sparse_targets_hash"]
    assert right["total_optimizer_calls"] - right["committed_steps"] == right["extra_discarded_probe_calls"]
    assert right["final_state_rule"] == "final committed Adam state"
    summary = json.loads((tmp_path / "vara_v2/summary.json").read_text())
    summary["metrics"]["applied_optimizer_steps"] = 3
    (tmp_path / "vara_v2/summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Checkpoint/summary metrics differ"):
        audit_run(tmp_path / "vara_v2", benchmark, "vara_v2", 0, strict=False)


def test_prevented_harm_is_exactly_logged_not_fabricated(tmp_path):
    package = tmp_path / "package"
    (package / "guard_evidence").mkdir(parents=True)
    runs = {}
    for seed in range(5):
        rows = [{"variable": "pde_residual", "patch_id": 0, "action_type": "joint",
                 "accepted": False, "prefiltered": False, "block": 0,
                 "observed_target_improvement": .1, "effectiveness_after": .8,
                 "guard_changes_pde_residual_mean": .03 + seed * .01,
                 "guard_noise_pde_residual_mean": .02}]
        runs[(seed, "vara_v2")] = {"decisions": pd.DataFrame(rows)}
    frame = guard_evidence(runs, package)
    assert len(frame) == 5
    highlight = json.loads((package / "guard_evidence/highlighted_harm.json").read_text())
    assert highlight["event"]["seed"] == 4
    assert highlight["event"]["largest_guard_violation"] == pytest.approx(.05)
    for seed in range(5):
        runs[(seed, "vara_v2")]["decisions"]["guard_changes_pde_residual_mean"] = .01
    guard_evidence(runs, package)
    assert json.loads((package / "guard_evidence/highlighted_harm.json").read_text())["event"] is None


def test_package_end_to_end_preserves_raw_files_and_crc(tmp_path, monkeypatch):
    import scripts.package_transient_fullguard as package_module
    results = tmp_path / "completed_test_fixture"
    for seed in range(5):
        cfg = _tiny_config("advection_diffusion")
        cfg["seed"] = seed
        for method in ("vanilla", "vara_v2"):
            PDEGeneralizationTrainer(cfg, method, results / f"seed_{seed}" / method).run()
    before = package_module.snapshot(results)
    native_audit = package_module.audit_pairs
    monkeypatch.setattr(package_module, "audit_pairs", lambda path, benchmark: native_audit(path, benchmark, strict=False))
    destination = tmp_path / "analysis" / "advection_diffusion"
    path = package_module.build_package(results, "advection_diffusion", destination)
    assert package_module.snapshot(results) == before
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for seed in range(5):
            for method in ("vanilla", "vara_v2"):
                assert archive.read(f"raw/seed_{seed}/{method}/checkpoints/final.pt") == (results / f"seed_{seed}/{method}/checkpoints/final.pt").read_bytes()
        assert "tables/per_seed_paired_results.csv" in archive.namelist()
        assert "fairness/frozen_data_seed_0.npz" in archive.namelist()
        assert "checksums/SHA256SUMS" in archive.namelist()
        for line in archive.read("checksums/SHA256SUMS").decode().splitlines():
            digest, name = line.split("  ", 1)
            import hashlib
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
