"""Isolated reviewer experiment; historical trainers/configs remain untouched."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from src.data.cfd_supervision import _eligible_mask
from src.physics.cavity_reference import load_full_field_reference
from src.training.vara_v2_trainer import VARAV2Trainer
from src.training.checkpointing import save_checkpoint
from src.utils.io import save_json
from src.utils.config import save_config
from src.utils.logging import JSONListLogger

CONFIG = "configs/vara_v2/lid_cavity_full_guard_2pct_fair.yaml"
BASE_SHA = "f14e1d21039e2bc3938830e255eb291f4fea7401"
GUARDS = ("pde_residual_mean", "continuity_residual_mean", "momentum_residual_mean",
          "boundary_condition_error", "unweighted_physics_validation_loss")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(value):
    """Canonical typed, shaped, little-endian arrays; sorted mapping keys."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            item = np.ascontiguousarray(item.astype(item.dtype.newbyteorder("<")))
            digest.update(json.dumps([str(item.dtype), list(item.shape)]).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                digest.update((type(key).__name__ + ":" + str(key)).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            digest.update((type(item).__name__ + ":" + repr(item)).encode())
    visit(value)
    return digest.hexdigest()


def validate_protocol(config):
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    data = config["data_supervision"]
    ctrl = config["controller_v2"]
    require(config["benchmark"] == "lid_driven_cavity", "benchmark")
    require(config["benchmark_params"]["reynolds"] == 100, "Re must be 100")
    require(config["seed"] == 0, "seed must be 0")
    require(config.get("dtype") == "float32", "float32 precision policy")
    require(data["mode"] == "sparse_cfd", "ordinary sparse_cfd required")
    require(data["sample_fraction"] == .02 and data.get("sample_count") is None, "requested fraction .02")
    require(not data.get("include_pressure") and not data.get("include_vorticity"), "velocity supervision only")
    require(not data.get("sparse_cfd_polish_enabled"), "sparse polish must be off")
    require(config["training"]["weights"]["cfd_velocity_mse"] > 0, "sparse velocity objective inactive")
    for key in ("counterfactual_probe_enabled", "gradient_prefilter_enabled", "rollback_enabled",
                "trust_region_enabled", "action_memory_enabled", "variable_awareness_enabled"):
        require(ctrl.get(key) is True, key)
    for key in ("sparse_polish_rescue", "sparse_polish_curriculum"):
        require(not ctrl.get(key, {}).get("enabled"), key)
    require(tuple(ctrl["guard_metrics"]) == GUARDS, "guard allowlist")
    for key in ("final_repair", "lbfgs"):
        require(not config["optimizer"].get(key, {}).get("enabled"), key)
    require(not config["checkpoint"].get("restore_best_before_final"), "best restoration")
    require(not config["convergence_early_stopping"]["enabled"], "early stopping")
    require(not config["continuation_validity"]["enabled"], "continuation gating")
    for key in ("controller_reference_metrics_enabled", "checkpoint_reference_metrics_enabled",
                "controller_streamfunction_metrics"):
        require(config["evaluation"].get(key) is False, key)
    require(ctrl["total_steps"] == config["revision_full_guard"]["primary_steps"], "primary budget")
    require(config["revision_full_guard"].get("secondary_control", False) or
            ctrl["total_steps"] == ctrl["warmup_steps"] + ctrl["control_blocks"] * ctrl["block_steps"], "schedule")
    require(config["compute_budget"] == {"enabled": True, "type": "applied_optimizer_steps",
                                         "value": ctrl["total_steps"]}, "compute stopping policy")
    if errors:
        raise ValueError("Invalid revision protocol: " + "; ".join(errors))


def sparse_evidence(trainer):
    pool = trainer.cfd_supervision
    cfg = trainer.config["data_supervision"]
    reference = load_full_field_reference(Path(pool.source_path))
    coords = np.column_stack((reference["x"], reference["y"]))
    eligible = np.flatnonzero(_eligible_mask(coords, trainer.benchmark.bounds, cfg))
    count = max(1, int(round(len(eligible) * cfg["sample_fraction"])))
    selected = np.sort(np.random.default_rng(cfg["seed"]).choice(eligible, count, replace=False))
    expected_xy = np.asarray(coords[selected], dtype="<f4")
    expected_u = np.asarray(reference["u"][selected], dtype="<f4").reshape(-1, 1)
    expected_v = np.asarray(reference["v"][selected], dtype="<f4").reshape(-1, 1)
    assert np.array_equal(expected_xy, pool.coords.cpu().numpy())
    assert np.array_equal(expected_u, pool.targets["u"].cpu().numpy())
    assert np.array_equal(expected_v, pool.targets["v"].cpu().numpy())
    arrays = {"selected_indices": selected.astype("<i8"), "coordinates": expected_xy,
              "u_targets": expected_u, "v_targets": expected_v}
    evidence = {name + "_sha256": state_hash(value) for name, value in arrays.items()}
    evidence.update(source_cfd_sha256=file_hash(pool.source_path), eligible_count=len(eligible),
                    requested_fraction=cfg["sample_fraction"], selected_count=count,
                    realized_fraction=count / len(eligible), pool_seed=cfg["seed"],
                    source_path=pool.source_path)
    evidence["combined_sparse_dataset_sha256"] = state_hash({**arrays,
        "source_cfd_sha256": evidence["source_cfd_sha256"]})
    return evidence, arrays


class RevisionTrainer(VARAV2Trainer):
    """Reuse V2 objectives and allocation methods, with a transparent fixed loop."""
    def __init__(self, config, method):
        validate_protocol(config)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.revision_method = method
        self.work = {name: 0 for name in ("neutral_probe_steps", "action_probe_steps",
            "retained_probe_steps", "gradient_screen_evaluations", "gradient_backpropagations",
            "diagnostic_evaluations", "sparse_data_evaluations", "diagnostic_grid_point_visits",
            "gradient_screen_collocation_points", "gradient_screen_boundary_points")}
        self.decisions = []
        self.proposals = []
        self.probe_pairs = []
        self.boundary_schedule = []
        self.trajectory_batch_schedule = []
        self.controller_seconds = 0.
        super().__init__(deepcopy(config))
        torch.use_deterministic_algorithms(True)
        # Historical flat layout places native losses in logs/. Revision manifests
        # and evidence use the method root; native subdirectories stay intact.
        self.run_dir = Path(config["experiments"]["root"])
        self.v2_state_logger = JSONListLogger(self.run_dir / "vara_v2_allocation_history.json")
        save_config(self.config, self.run_dir / "resolved_config.yaml")
        assert not self.sparse_cfd_polish_v2 and not self.sparse_polish_curriculum_enabled
        self.initial_hash = state_hash(self.model.state_dict())
        self.sparse_manifest, arrays = sparse_evidence(self)
        np.savez(self.run_dir / "sparse_dataset.npz", **arrays)
        save_json(self.sparse_manifest, self.run_dir / "sparse_manifest.json")
        save_json({"initial_model_sha256": self.initial_hash}, self.run_dir / "initialization.json")

    def maybe_checkpoint(self, *args, **kwargs):
        # No checkpoint ranking or trajectory selection in this experiment.
        return None

    def should_stop_early(self, *args, **kwargs):
        return False

    def _diagnose_reference_free(self, *args, **kwargs):
        started = time.perf_counter()
        self.work["diagnostic_evaluations"] += 1
        self.work["diagnostic_grid_point_visits"] += self.config["validation"]["nx"] * self.config["validation"]["ny"]
        result = super()._diagnose_reference_free(*args, **kwargs)
        self.controller_seconds += time.perf_counter() - started
        return result

    def _guard_metrics(self, coords):
        started = time.perf_counter()
        self.work["diagnostic_evaluations"] += 1
        self.work["diagnostic_grid_point_visits"] += len(coords)
        metrics = super()._guard_metrics(coords)
        # Fail closed: every declared guard must actually be observed.
        selected = {key: metrics[key] for key in GUARDS}
        if not all(np.isfinite(value) for value in selected.values()):
            raise FloatingPointError("Non-finite/missing guard")
        self.controller_seconds += time.perf_counter() - started
        return selected

    def _candidate_influence(self, candidates):
        started = time.perf_counter()
        if candidates:
            self.work["gradient_screen_evaluations"] += 1
            self.work["gradient_backpropagations"] += 1 + len(candidates)
            self.work["sparse_data_evaluations"] += self.cfd_supervision.sample_count
            self.work["gradient_screen_collocation_points"] += len(self._probe_batch["xy_f"])
            self.work["gradient_screen_boundary_points"] += len(self._probe_batch["xy_bc"])
        result = super()._candidate_influence(candidates)
        self.controller_seconds += time.perf_counter() - started
        return result

    def _train_v2_steps(self, batch, steps, cycle, phase, probe=False, applied=True):
        previous = self.compute_tracker.optimizer_steps
        for offset in range(0, steps, 50):
            count = min(50, steps - offset)
            super()._train_v2_steps(batch, count, cycle, phase, probe, applied)
            committed = self.compute_tracker.applied_optimizer_steps
            total = self.config["controller_v2"]["total_steps"]
            print(f"{self.revision_method} | {100 * committed / total:6.2f}% | "
                  f"{committed}/{total} committed | {self.compute_tracker.optimizer_steps} optimizer calls | {phase}",
                  flush=True)
        actual = self.compute_tracker.optimizer_steps - previous
        if actual != steps:
            raise RuntimeError(f"Truncated optimizer phase {phase}: {actual}/{steps}")
        if phase == "neutral_probe":
            self.work["neutral_probe_steps"] += actual
        if phase == "action_probe":
            self.work["action_probe_steps"] += actual
        self.work["sparse_data_evaluations"] += actual * self.cfd_supervision.sample_count
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError("Non-finite model")

    def snapshot(self):
        return {"model": self._model_snapshot(), "optimizer": deepcopy(self.optimizer.state_dict()),
            "allocation": self.v2_controller.state.snapshot(), "sampling": self.sampling_state_snapshot(),
            "normalization": deepcopy(self.loss_normalization_state), "step": self.global_step,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

    def restore(self, state):
        self._restore_model_snapshot(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.v2_controller.state.restore(state["allocation"])
        self.restore_sampling_state(state["sampling"])
        self.loss_normalization_state = deepcopy(state["normalization"])
        self.global_step = state["step"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def flush(self):
        import pandas as pd
        save_json(self.decisions, self.run_dir / "revision_decisions.json")
        save_json(self.proposals, self.run_dir / "revision_proposals.json")
        save_json(self.probe_pairs, self.run_dir / "probe_pairs.json")
        rows = []
        for decision in self.decisions:
            row = {}
            for key, value in decision.items():
                if isinstance(value, dict) and key in ("guard_changes", "guard_noise", "guard_neutral", "guard_candidate"):
                    row.update({key + "_" + name: number for name, number in value.items()})
                elif not isinstance(value, (dict, list, np.ndarray)):
                    row[key] = value
            rows.append(row)
        pd.DataFrame(rows, columns=None if rows else ["seed", "block", "accepted", "prefiltered"]).to_csv(
            self.run_dir / "vara_v2_decisions.csv", index=False)

    def record_batch(self, batch):
        self.boundary_schedule.append(state_hash(batch["xy_bc"]))
        self.trajectory_batch_schedule.append({"start_step": self.global_step,
            "collocation": len(batch["xy_f"]), "boundary": len(batch["xy_bc"]),
            "sparse": len(batch["xy_data"])})

    def run(self):
        started = time.perf_counter()
        self.compute_tracker.start()
        cfg = self.config["controller_v2"]
        _, _, coords = self.validation_grid()
        try:
            # A shared 200-step refresh schedule for all primary trajectory batches.
            for cycle in range(cfg["warmup_steps"] // cfg["block_steps"]):
                batch = self.initial_batch() if cycle == 0 else self._resample_v2_batch({}, coords)
                self.record_batch(batch)
                self._train_v2_steps(batch, cfg["block_steps"], cycle, "warmup")
            for block in range(cfg["control_blocks"]):
                if self.revision_method != "vara_v2_full_guard":
                    batch = self._resample_v2_batch({}, coords)
                    self.record_batch(batch)
                    steps = min(cfg["block_steps"], cfg["total_steps"] - self.global_step)
                    self._train_v2_steps(batch, steps, block, "vanilla_primary")
                    continue
                maps, raw, names, weak, coords = self._diagnose_reference_free()
                before_metrics = self._guard_metrics(coords)
                ctrl = self.v2_controller
                ctrl.update_history(names, raw, before_metrics)
                candidates = ctrl.candidates(weak)
                ranked = ctrl.rank(candidates, self._candidate_influence(candidates))
                pre = self.snapshot()
                active = [candidate for candidate in ranked if not candidate.prefiltered]
                proposal_ids = {}
                for candidate in ranked:
                    proposal_ids[candidate.key()] = len(self.proposals)
                    self.proposals.append({"seed": self.seed, "block": block,
                        **candidate.to_record(), "status": "not_probed_after_acceptance"})
                    if candidate.prefiltered:
                        decision = ctrl.record_prefilter(candidate, update_trust=False)
                        self.prefiltered_interventions += 1
                        self.proposals[-1]["status"] = "prefiltered"
                        self.decisions.append({"seed": self.seed, "block": block,
                            **candidate.to_record(), **decision, "rollback_executed": False,
                            "trust_radius_before": ctrl.trust_radius,
                            "effectiveness_before": ctrl.effectiveness.get(candidate.key(), 1.),
                            "effectiveness_after": ctrl.effectiveness.get(candidate.key(), 1.)})
                neutral_batch = self._resample_v2_batch(maps, coords)
                self.record_batch(neutral_batch)
                if not active:
                    self._train_v2_steps(neutral_batch, cfg["block_steps"], block, "no_action")
                    self._log_state(block)
                    self.flush()
                    continue
                probe = cfg["probe_steps"]
                neutral_start = state_hash({"model": pre["model"], "optimizer": pre["optimizer"]})
                self._train_v2_steps(neutral_batch, probe, block, "neutral_probe", True, False)
                neutral = self.snapshot()
                _, neutral_raw, neutral_names, _, _ = self._diagnose_reference_free(False)
                neutral_metrics = self._guard_metrics(coords)
                kept, kept_batch = neutral, neutral_batch
                rejected = []
                for candidate in active:
                    self.restore(pre)
                    action_start = state_hash({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()})
                    assert neutral_start == action_start
                    pair = {"block": block, "candidate_key": candidate.key(),
                        "pre_probe_model_sha256": state_hash(pre["model"]),
                        "neutral_start_sha256": neutral_start, "action_start_sha256": action_start,
                        "rng_allocation_state_sha256": state_hash({key: pre[key] for key in
                            ("sampling", "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "allocation")}),
                        "restored_rng_allocation_sha256": state_hash({key: self.snapshot()[key] for key in
                            ("sampling", "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "allocation")})}
                    assert pair["rng_allocation_state_sha256"] == pair["restored_rng_allocation_sha256"]
                    ctrl.apply(candidate)
                    action_batch = self._resample_v2_batch(maps, coords)
                    assert state_hash(action_batch["xy_bc"]) == state_hash(neutral_batch["xy_bc"])
                    self._train_v2_steps(action_batch, probe, block, "action_probe", True, False)
                    _, action_raw, action_names, _, _ = self._diagnose_reference_free(False)
                    action_metrics = self._guard_metrics(coords)
                    target_before = self._candidate_score(candidate, raw, names)
                    target_neutral = self._candidate_score(candidate, neutral_raw, neutral_names)
                    target_action = self._candidate_score(candidate, action_raw, action_names)
                    memory_before = ctrl.effectiveness.get(candidate.key(), 1.)
                    accepted, decision = ctrl.evaluate(candidate, target_neutral, target_action,
                        neutral_metrics, action_metrics, target_threshold=cfg["counterfactual_target_margin"],
                        guard_threshold=cfg["counterfactual_guard_margin"], comparison_mode="counterfactual",
                        update_state=False)
                    pair.update(neutral_target_after=target_neutral, candidate_target_after=target_action)
                    self.probe_pairs.append(pair)
                    self.proposals[proposal_ids[candidate.key()]]["status"] = "accepted" if accepted else "rejected"
                    row = {"seed": self.seed, "block": block, **candidate.to_record(), **decision,
                        "target_before": target_before, "neutral_target_after": target_neutral,
                        "candidate_target_after": target_action, "guard_neutral": neutral_metrics,
                        "guard_candidate": action_metrics, "effectiveness_before": memory_before,
                        "allocation_before": pre["allocation"], "allocation_candidate": ctrl.state.to_record(),
                        "rollback_executed": not accepted, "probe_steps": probe}
                    if accepted:
                        row = {**row, **ctrl.commit_evaluation(candidate, True, decision)}
                        self.accepted_interventions += 1
                        kept, kept_batch = self.snapshot(), action_batch
                    else:
                        self.rejected_interventions += 1
                        rejected.append((candidate, decision, row))
                        self.restore(neutral)
                        row["rollback_model_sha256"] = state_hash(self.model.state_dict())
                        row["neutral_model_sha256"] = state_hash(neutral["model"])
                        assert row["rollback_model_sha256"] == row["neutral_model_sha256"]
                    row["allocation_after"] = ctrl.state.to_record()
                    self.decisions.append(row)
                    self.flush()
                    if accepted:
                        break
                self.restore(kept)
                if kept is neutral and rejected:
                    candidate, decision, row = rejected[0]
                    row.update(ctrl.commit_evaluation(candidate, False, decision))
                for _, _, row in rejected:
                    row.setdefault("effectiveness_after", ctrl.effectiveness.get(
                        f"{row['variable']}|{row['patch_id']}|{row['action_type']}", 1.))
                    row["trust_radius_after"] = ctrl.trust_radius
                self.work["retained_probe_steps"] += probe
                self.compute_tracker.record_applied_optimizer_steps(probe)
                self._train_v2_steps(kept_batch, cfg["block_steps"] - probe, block, "continuation")
                self._log_state(block)
                self.flush()
                print(f"{self.revision_method}: block {block + 1}/{cfg['control_blocks']}; "
                      f"committed {self.global_step}/{cfg['total_steps']}", flush=True)
            if self.global_step != cfg["total_steps"] or self.compute_tracker.applied_optimizer_steps != cfg["total_steps"]:
                raise RuntimeError("Committed budget mismatch")
            final_hash = state_hash(self.model.state_dict())
            # Dense CFD/Ghia are attached only now, after optimization ends.
            self.benchmark = replace(self.benchmark, reference="ghia",
                full_field_reference_path=self.cfd_supervision.source_path,
                profile_only=False, has_reference=True, reference_kind="full_field_cfd")
            metrics = self.evaluate_and_save_final()
            assert final_hash == state_hash(self.model.state_dict())
            calls = self.compute_tracker.optimizer_steps
            work = {**self.compute_tracker.summary(), **self.work,
                "committed_steps": self.global_step, "total_optimizer_calls": calls,
                "discarded_branch_steps": calls - self.global_step,
                "objective_evaluations": self.compute_tracker.objective_evaluations + self.work["gradient_screen_evaluations"],
                "objective_backprop_equivalents": calls + self.work["gradient_backpropagations"],
                "collocation_point_evaluations_train_and_screen": self.compute_tracker.collocation_evaluations + self.work["gradient_screen_collocation_points"],
                "boundary_point_evaluations_train_and_screen": self.compute_tracker.boundary_evaluations + self.work["gradient_screen_boundary_points"],
                "counter_convention": "Objective forwards, optimizer steps, gradient queries and grid point visits; not FLOPs or internal autograd primitive counts. Final evaluation separate.",
                "controller_seconds": self.controller_seconds,
                "controller_time_scope": "Diagnostic, guard and gradient-screen calls; snapshot/logging overhead is included only in total wall time.",
                "total_wall_clock_seconds": time.perf_counter() - started}
            manifest = {"method": self.revision_method, "seed": self.seed, "config": self.config,
                "initial_model_sha256": self.initial_hash, "final_model_sha256": final_hash,
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "sparse": self.sparse_manifest, "boundary_schedule_hashes": self.boundary_schedule,
                "trajectory_batch_schedule": self.trajectory_batch_schedule,
                "final_state_rule": "final_committed_adam", "sparse_polish": False,
                "repair": False, "restore_best": False, "early_stopping": False,
                "controller_reference_access": False, "guard_active": self.revision_method == "vara_v2_full_guard",
                "guard_execution": {"proposals": len(self.proposals), "decisions": len(self.decisions),
                    "accepted": self.accepted_interventions, "rejected": self.rejected_interventions,
                    "prefiltered": self.prefiltered_interventions, "probe_pairs": len(self.probe_pairs)},
                "compute": work, "metrics": metrics, "completed": True}
            save_checkpoint(self.checkpoint_dir / "final.pt", self.model, self.optimizer,
                self.config, metrics, self.global_step, -1)
            manifest["final_checkpoint_sha256"] = file_hash(self.checkpoint_dir / "final.pt")
            save_json(manifest, self.run_dir / "revision_manifest.json")
            self.flush()
            return manifest
        except Exception as exc:
            self.flush()
            save_json({"completed": False, "error": repr(exc), "work": self.work,
                       "compute": self.compute_tracker.summary()}, self.run_dir / "failure.json")
            raise
