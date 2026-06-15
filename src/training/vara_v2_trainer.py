"""Opt-in matched-compute VARA Controller V2 trainer."""

from __future__ import annotations

from copy import deepcopy
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.controllers import (
    V2Candidate,
    V2ControllerConfig,
    VARAV2Controller,
)
from src.diagnostics import DiagnosticMapBuilder
from src.evaluation.metrics import evaluate_on_grid
from src.losses.base_losses import compute_pointwise_losses, weighted_sum
from src.losses.local_losses import LOSS_COORD_SOURCE, compute_budgeted_patch_losses
from src.training.trainer import ExperimentTrainer
from src.training.checkpointing import save_checkpoint
from src.utils.io import save_json
from src.utils.logging import CSVLogger, JSONListLogger


SPARSE_POLISH_GUARD_METRICS = (
    "pde_residual_mean",
    "momentum_residual_mean",
    "continuity_residual_mean",
    "boundary_condition_error",
    "cfd_velocity_mse_sparse",
    "cfd_u_mse_sparse",
    "cfd_v_mse_sparse",
    "top_band_continuity_residual_mean",
    "speed_pred_max",
)

SPARSE_POLISH_SCORE_WEIGHTS = {
    "pde_residual_mean": 1.4,
    "momentum_residual_mean": 1.4,
    "continuity_residual_mean": 2.2,
    "boundary_condition_error": 2.0,
    "cfd_velocity_mse_sparse": 1.8,
    "cfd_u_mse_sparse": 0.3,
    "cfd_v_mse_sparse": 0.5,
    "top_band_continuity_residual_mean": 0.8,
    "top_band_pde_residual_mean": 0.3,
    "upper_core_pde_residual_mean": 0.2,
    "near_wall_pde_residual_mean": 0.5,
}

SPARSE_POLISH_RESCUE_CANDIDATES = (
    (
        "continuity_boundary_rescue",
        {
            "continuity": 1.45,
            "top_band_continuity": 1.45,
            "bc_uvp_balanced": 1.30,
            "momentum_u": 1.0,
            "momentum_v": 1.0,
            "cfd_u_mse": 1.0,
            "cfd_v_mse": 1.05,
        },
    ),
    (
        "top_lid_boundary_rescue",
        {
            "bc_uvp_balanced": 1.35,
            "bc_top_u": 1.15,
            "bc_top_v": 1.15,
            "bc_left_v": 1.10,
            "bc_right_v": 1.10,
            "continuity": 1.15,
            "cfd_u_mse": 1.0,
            "cfd_v_mse": 1.05,
        },
    ),
    (
        "near_wall_physics_rescue",
        {
            "momentum_u": 1.10,
            "momentum_v": 1.10,
            "top_band_pde": 1.35,
            "upper_core_pde": 1.15,
            "top_band_continuity": 1.30,
            "bc_uvp_balanced": 1.10,
            "cfd_u_mse": 1.0,
            "cfd_v_mse": 1.0,
        },
    ),
    (
        "data_preserving_rescue",
        {
            "cfd_u_mse": 1.10,
            "cfd_v_mse": 1.20,
            "continuity": 1.20,
            "bc_uvp_balanced": 1.15,
            "momentum_u": 1.0,
            "momentum_v": 1.0,
        },
    ),
    (
        "balanced_gate_rescue",
        {
            "momentum_u": 1.05,
            "momentum_v": 1.05,
            "continuity": 1.25,
            "top_band_continuity": 1.20,
            "bc_uvp_balanced": 1.20,
            "cfd_u_mse": 1.05,
            "cfd_v_mse": 1.10,
        },
    ),
)


class VARAV2Trainer(ExperimentTrainer):
    """Fixed-step, trust-region VARA trainer.

    The controller sees residual, boundary, and optional configured training
    data diagnostics only. Evaluation references never enter proposal,
    ranking, acceptance, or rollback.
    """

    def __init__(self, config: dict[str, Any], mode: str = "vara_v2") -> None:
        if mode != "vara_v2":
            raise ValueError(f"VARAV2Trainer only supports mode='vara_v2', got {mode!r}.")
        super().__init__(config, mode)
        cfg = dict(config.get("controller_v2", {}))
        self.sparse_cfd_polish_v2 = (
            str(config.get("data_supervision", {}).get("mode", "pure_pinn"))
            == "sparse_cfd_polish"
        )
        if self.sparse_cfd_polish_v2:
            cfg["guard_metrics"] = list(
                dict.fromkeys(
                    (*SPARSE_POLISH_GUARD_METRICS, *SPARSE_POLISH_SCORE_WEIGHTS)
                )
            )
            cfg["trust_radius_initial"] = min(
                float(cfg.get("trust_radius_initial", 0.10)), 0.05
            )
            cfg["trust_radius_max"] = min(
                float(cfg.get("trust_radius_max", 0.20)), 0.10
            )
            cfg["trust_shrink"] = min(float(cfg.get("trust_shrink", 0.5)), 0.4)
            cfg["prefilter_damage_ratio"] = min(
                float(cfg.get("prefilter_damage_ratio", 0.25)), 0.15
            )
            # Rejected polish probes must resume the exact neutral trajectory.
            cfg["counterfactual_probe_enabled"] = True
        self.v2_config = V2ControllerConfig.from_dict(cfg, self.patch_grid.num_patches)
        self.v2_controller = VARAV2Controller(self.v2_config)
        self.v2_decision_logger = CSVLogger(self.run_dir / "vara_v2_decisions.csv")
        self.v2_state_logger = JSONListLogger(self.run_dir / "vara_v2_allocation_history.json")
        self.accepted_interventions = 0
        self.rejected_interventions = 0
        self.prefiltered_interventions = 0
        self.rollback_enabled = bool(cfg.get("rollback_enabled", True))
        self._sparse_polish_best: dict[str, Any] | None = None
        self._sparse_polish_score_before = float("nan")
        self._sparse_polish_score_after = float("nan")
        self._sparse_polish_restored_best = False
        self._sparse_polish_score_scales: dict[str, float] = {}
        self._sparse_polish_initial_score = float("nan")
        self._sparse_polish_final_score = float("nan")
        self._sparse_polish_noop_fallback = False
        self._sparse_polish_rescue_status = self._empty_sparse_polish_rescue_status()
        sampling_snapshot = self.sampling_state_snapshot()
        self._probe_batch = self._make_probe_batch()
        # The fixed controller probe must not advance the optimization
        # samplers. Otherwise a neutral V2 run starts from different points
        # than Vanilla even when both use the same seed.
        self.restore_sampling_state(sampling_snapshot)

    def run(self) -> dict[str, float]:
        self.compute_tracker.start()
        cfg = dict(self.config.get("controller_v2", {}))
        warmup_steps = int(cfg.get("warmup_steps", 100))
        control_blocks = int(cfg.get("control_blocks", 6))
        block_steps = int(cfg.get("block_steps", 50))
        probe_steps = int(cfg.get("probe_steps", 10))
        if warmup_steps + control_blocks * block_steps != int(cfg.get("total_steps", 400)):
            raise ValueError("VARA V2 schedule must satisfy warmup + blocks * block_steps == total_steps.")
        if probe_steps <= 0 or probe_steps >= block_steps:
            raise ValueError("controller_v2.probe_steps must be between zero and block_steps.")

        batch = self.initial_batch()
        # Use the same neutral resampling cadence as the comparison trainer.
        # A long V2 warm-up on one fixed batch is not equivalent to Vanilla
        # split into several ordinary cycles, even at equal optimizer steps.
        neutral_cycle_steps = max(
            1,
            int(self.config.get("training", {}).get("epochs_per_cycle", warmup_steps)),
        )
        warmup_remaining = warmup_steps
        warmup_cycle = 0
        while warmup_remaining > 0:
            chunk = min(neutral_cycle_steps, warmup_remaining)
            self._train_v2_steps(
                batch,
                chunk,
                cycle=warmup_cycle,
                phase="v2_warmup",
            )
            warmup_remaining -= chunk
            _, _, warmup_coords = self.validation_grid()
            warmup_metrics = self.controller_metrics(warmup_coords)
            self.maybe_checkpoint(warmup_cycle, warmup_metrics)
            warmup_cycle += 1
            if warmup_remaining > 0:
                batch = self._resample_v2_batch({}, warmup_coords)

        if self.sparse_cfd_polish_v2:
            self._update_sparse_polish_best(self._guard_metrics(warmup_coords))

        for block in range(control_blocks):
            maps_before, raw_before, names, weak_regions, coords = self._diagnose_reference_free()
            metrics_before = self._guard_metrics(coords)
            self.v2_controller.update_history(names, raw_before, metrics_before)
            candidates = self.v2_controller.candidates(weak_regions)
            if self.sparse_cfd_polish_v2:
                rescue = self._sparse_polish_rescue_candidate(
                    weak_regions, metrics_before
                )
                if rescue is not None:
                    candidates.append(rescue)
            influence = self._candidate_influence(candidates)
            ranked = self.v2_controller.rank(candidates, influence)
            active_candidates = [item for item in ranked if not item.prefiltered]
            if self.sparse_cfd_polish_v2:
                active_candidates = self._sparse_polish_candidates(active_candidates)
            # Match the baseline cycle order exactly: diagnose the completed
            # cycle, draw the next cycle's batch, and then optimize it. Keep
            # the pre-draw sampler state so neutral and proposed allocations
            # are counterfactual draws from the same sequence position.
            sampling_snapshot = self.sampling_state_snapshot()
            neutral_batch = self._resample_v2_batch(maps_before, coords)
            neutral_sampling_snapshot = self.sampling_state_snapshot()
            prefiltered = [item for item in ranked if item.prefiltered]
            self.prefiltered_interventions += len(prefiltered)
            for index, rejected in enumerate(prefiltered):
                decision = self.v2_controller.record_prefilter(
                    rejected,
                    update_trust=not active_candidates and index == 0,
                )
                if self.sparse_cfd_polish_v2:
                    decision["rollback_reason"] = "gradient_conflict"
                self._log_decision(block, rejected, decision)

            if not active_candidates:
                self._train_v2_steps(
                    neutral_batch,
                    block_steps,
                    cycle=warmup_cycle + block,
                    phase="v2_no_action",
                )
                maps_kept, _raw_kept, _names_kept, _weak_kept, coords_kept = (
                    self._diagnose_reference_free(update_history=False)
                )
                self._log_state(block)
                block_metrics = self.controller_metrics(coords_kept)
                self.maybe_checkpoint(warmup_cycle + block, block_metrics)
                kept_guard_metrics = self._guard_metrics(coords_kept)
                self._update_sparse_polish_best(kept_guard_metrics)
                if self._maybe_run_sparse_polish_gate_rescue(
                    kept_guard_metrics,
                    coords_kept,
                ):
                    break
                if self.should_stop_early(block_metrics):
                    break
                continue

            model_snapshot = self._model_snapshot()
            optimizer_snapshot = deepcopy(self.optimizer.state_dict())
            loss_normalization_snapshot = deepcopy(self.loss_normalization_state)
            controller_snapshot = self.v2_controller.state.snapshot()
            trust_before = self.v2_controller.trust_radius
            block_start_step = self.global_step
            counterfactual = self.v2_config.counterfactual_probe_enabled
            neutral_model_snapshot: dict[str, torch.Tensor] | None = None
            neutral_optimizer_snapshot: dict[str, Any] | None = None
            neutral_loss_normalization_snapshot: dict[str, float] | None = None
            if counterfactual:
                # A candidate must outperform the training trajectory that
                # would have occurred without intervention. Comparing only
                # against the pre-probe state confounds ordinary Adam progress
                # with controller benefit and can accept a harmful action.
                self._train_v2_steps(
                    neutral_batch,
                    probe_steps,
                    cycle=warmup_cycle + block,
                    phase="v2_neutral_probe",
                    probe=True,
                    applied=False,
                )
                neutral_model_snapshot = self._model_snapshot()
                neutral_optimizer_snapshot = deepcopy(self.optimizer.state_dict())
                neutral_loss_normalization_snapshot = deepcopy(
                    self.loss_normalization_state
                )
                (
                    _neutral_maps,
                    neutral_raw,
                    neutral_names,
                    _neutral_weak,
                    neutral_coords,
                ) = self._diagnose_reference_free(update_history=False)
                neutral_metrics = self._guard_metrics(neutral_coords)
                self._restore_model_snapshot(model_snapshot)
                self.optimizer.load_state_dict(optimizer_snapshot)
                self.loss_normalization_state = deepcopy(loss_normalization_snapshot)
                self.v2_controller.state.restore(controller_snapshot)
                self.restore_sampling_state(sampling_snapshot)
                self.global_step = block_start_step
            else:
                neutral_raw = raw_before
                neutral_names = names
                neutral_metrics = metrics_before

            rejected_trials: list[tuple[V2Candidate, dict[str, Any]]] = []
            selected: dict[str, Any] | None = None
            candidates_to_probe = (
                active_candidates
                if counterfactual and self.rollback_enabled
                else active_candidates[:1]
            )
            if self.sparse_cfd_polish_v2:
                candidates_to_probe = candidates_to_probe[:1]
            for candidate in candidates_to_probe:
                self._restore_model_snapshot(model_snapshot)
                self.optimizer.load_state_dict(optimizer_snapshot)
                self.loss_normalization_state = deepcopy(loss_normalization_snapshot)
                self.v2_controller.state.restore(controller_snapshot)
                self.restore_sampling_state(sampling_snapshot)
                self.global_step = block_start_step
                self.v2_controller.apply(candidate)
                # Sampling actions must be evaluated on the allocation they
                # propose. Every alternative starts from the same sampler
                # state and model checkpoint.
                proposal_batch = self._resample_v2_batch(maps_before, coords)
                proposal_sampling_snapshot = self.sampling_state_snapshot()
                self._train_v2_steps(
                    proposal_batch,
                    probe_steps,
                    cycle=warmup_cycle + block,
                    phase=(
                        "v2_counterfactual_probe"
                        if counterfactual
                        else "v2_trust_probe"
                    ),
                    probe=True,
                    applied=False,
                )
                _maps_after, raw_after, names_after, _weak_after, coords_after = (
                    self._diagnose_reference_free(update_history=False)
                )
                metrics_after = self._guard_metrics(coords_after)
                before_target = self._candidate_score(
                    candidate,
                    neutral_raw,
                    neutral_names,
                )
                after_target = self._candidate_score(
                    candidate,
                    raw_after,
                    names_after,
                )
                if self.sparse_cfd_polish_v2:
                    accepted, decision = self._evaluate_sparse_polish_candidate(
                        candidate,
                        neutral_metrics,
                        metrics_after,
                    )
                else:
                    accepted, decision = self.v2_controller.evaluate(
                        candidate,
                        before_target,
                        after_target,
                        neutral_metrics,
                        metrics_after,
                        target_threshold=(
                            self.v2_config.counterfactual_target_margin
                            if counterfactual
                            else None
                        ),
                        guard_threshold=(
                            self.v2_config.counterfactual_guard_margin
                            if counterfactual
                            else None
                        ),
                        comparison_mode=(
                            "counterfactual" if counterfactual else "temporal"
                        ),
                        update_state=False,
                    )
                decision.update(
                    {
                        "block": block,
                        "prefiltered": False,
                        "gradient_compatibility": candidate.gradient_compatibility,
                        "rank_score": candidate.rank_score,
                        "trust_radius_before": trust_before,
                    }
                )
                if accepted or not self.rollback_enabled:
                    selected = {
                        "candidate": candidate,
                        "accepted": accepted,
                        "decision": decision,
                        "model": self._model_snapshot(),
                        "optimizer": deepcopy(self.optimizer.state_dict()),
                        "loss_normalization": deepcopy(self.loss_normalization_state),
                        "controller": self.v2_controller.state.snapshot(),
                        "sampling": proposal_sampling_snapshot,
                        "batch": proposal_batch,
                    }
                    break
                decision["committed"] = False
                rejected_trials.append((candidate, decision))
                self.compute_tracker.record_rollback_steps(probe_steps)

            committed = selected is not None
            self.rejected_interventions += len(rejected_trials)
            if committed:
                candidate = selected["candidate"]
                decision = self.v2_controller.commit_evaluation(
                    candidate,
                    bool(selected["accepted"]),
                    selected["decision"],
                )
                decision["committed"] = True
                self._restore_model_snapshot(selected["model"])
                self.optimizer.load_state_dict(selected["optimizer"])
                self.loss_normalization_state = deepcopy(selected["loss_normalization"])
                self.v2_controller.state.restore(selected["controller"])
                self.restore_sampling_state(selected["sampling"])
                committed_batch = selected["batch"]
                if bool(selected["accepted"]):
                    self.accepted_interventions += 1
                else:
                    self.rejected_interventions += 1
                self._log_decision(block, candidate, decision)
                if counterfactual:
                    self.compute_tracker.record_rollback_steps(probe_steps)
            else:
                if counterfactual:
                    assert neutral_model_snapshot is not None
                    assert neutral_optimizer_snapshot is not None
                    assert neutral_loss_normalization_snapshot is not None
                    self._restore_model_snapshot(neutral_model_snapshot)
                    self.optimizer.load_state_dict(neutral_optimizer_snapshot)
                    self.loss_normalization_state = deepcopy(
                        neutral_loss_normalization_snapshot
                    )
                else:
                    self._restore_model_snapshot(model_snapshot)
                    self.optimizer.load_state_dict(optimizer_snapshot)
                    self.loss_normalization_state = deepcopy(loss_normalization_snapshot)
                self.v2_controller.state.restore(controller_snapshot)
                self.restore_sampling_state(neutral_sampling_snapshot)
                committed_batch = neutral_batch
                if rejected_trials:
                    first_candidate, first_decision = rejected_trials[0]
                    rejected_trials[0] = (
                        first_candidate,
                        self.v2_controller.commit_evaluation(
                            first_candidate,
                            False,
                            first_decision,
                        ),
                    )

            for rejected_candidate, rejected_decision in rejected_trials:
                self._log_decision(block, rejected_candidate, rejected_decision)
            if counterfactual:
                # Only one probe branch belongs to the committed trajectory;
                # all alternatives remain in actual-compute accounting.
                self.global_step = block_start_step + probe_steps
                self.compute_tracker.record_applied_optimizer_steps(probe_steps)
            elif committed:
                self.compute_tracker.record_applied_optimizer_steps(probe_steps)

            if not counterfactual and not committed:
                # The rejected probe consumed part of the fixed compute
                # budget. Restore the state, but do not hide extra optimizer
                # work by replaying those steps.
                self.global_step = block_start_step + probe_steps
                remaining = block_steps - probe_steps
            else:
                remaining = block_steps - probe_steps
            self._train_v2_steps(
                committed_batch,
                remaining,
                cycle=warmup_cycle + block,
                phase="v2_commit" if committed else "v2_rollback_continue",
            )
            maps_kept, _raw_kept, _names_kept, _weak_kept, coords_kept = self._diagnose_reference_free(
                update_history=False
            )
            self._log_state(block)
            block_metrics = self.controller_metrics(coords_kept)
            self.maybe_checkpoint(warmup_cycle + block, block_metrics)
            kept_guard_metrics = self._guard_metrics(coords_kept)
            self._update_sparse_polish_best(kept_guard_metrics)
            if self._maybe_run_sparse_polish_gate_rescue(
                kept_guard_metrics,
                coords_kept,
            ):
                break
            if self.should_stop_early(block_metrics):
                break

        if self.sparse_cfd_polish_v2:
            _, _, final_guard_coords = self.validation_grid()
            pre_restore_metrics = self._guard_metrics(final_guard_coords)
            if not self._sparse_polish_rescue_status["triggered"]:
                self._sparse_polish_rescue_status = (
                    self._run_sparse_polish_gate_rescue(
                        pre_restore_metrics,
                        final_guard_coords,
                    )
                )
            if self._sparse_polish_rescue_status["triggered"]:
                pre_restore_metrics = self._guard_metrics(final_guard_coords)
            self._sparse_polish_final_score = self._sparse_polish_score(
                pre_restore_metrics
            )
            minimum_gain = float(
                self.config.get("controller_v2", {}).get(
                    "sparse_polish_noop_min_score_gain", 0.01
                )
            )
            best_score = (
                float(self._sparse_polish_best["score"])
                if self._sparse_polish_best is not None
                else math.inf
            )
            score_gain = (
                (self._sparse_polish_initial_score - best_score)
                / max(abs(self._sparse_polish_initial_score), 1e-12)
                if math.isfinite(self._sparse_polish_initial_score)
                and math.isfinite(best_score)
                else 0.0
            )
            self._sparse_polish_noop_fallback = self._sparse_polish_should_noop(
                score_gain,
                minimum_gain=minimum_gain,
            )
            self._restore_sparse_polish_best_if_needed(
                pre_restore_metrics,
                force=self._sparse_polish_noop_fallback,
            )
            if self._sparse_polish_restored_best:
                self._sparse_polish_final_score = self._sparse_polish_score(
                    self._guard_metrics(final_guard_coords)
                )
            checkpoint_cfg = self.config.setdefault("checkpoint", {})
            restore_best = checkpoint_cfg.get("restore_best_before_final", False)
            checkpoint_cfg["restore_best_before_final"] = False
            try:
                metrics = self.evaluate_and_save_final()
            finally:
                checkpoint_cfg["restore_best_before_final"] = restore_best
        else:
            metrics = self.evaluate_and_save_final()
        accepted_improvement = self._accepted_improvement_per_compute()
        metrics.update(
            {
                "accepted_interventions": self.accepted_interventions,
                "rejected_interventions": self.rejected_interventions,
                "prefiltered_interventions": self.prefiltered_interventions,
                "rollback_count": self.rejected_interventions,
                "v2_final_trust_radius": self.v2_controller.trust_radius,
                "v2_accepted_improvement_per_compute": accepted_improvement,
                "accepted_improvement_per_compute": accepted_improvement,
                "vara_sparse_polish_score_before": self._sparse_polish_score_before,
                "vara_sparse_polish_score_after": self._sparse_polish_score_after,
                "vara_sparse_polish_score_initial": self._sparse_polish_initial_score,
                "vara_sparse_polish_score_final": self._sparse_polish_final_score,
                "vara_sparse_polish_best_score": (
                    float(self._sparse_polish_best["score"])
                    if self._sparse_polish_best is not None
                    else float("nan")
                ),
                "vara_sparse_polish_restored_best_checkpoint": (
                    self._sparse_polish_restored_best
                ),
                "vara_sparse_polish_score_improvement": (
                    self._sparse_polish_initial_score
                    - (
                        float(self._sparse_polish_best["score"])
                        if self._sparse_polish_best is not None
                        else self._sparse_polish_final_score
                    )
                ),
                "vara_sparse_polish_acceptance_margin": float(
                    self.config.get("controller_v2", {}).get(
                        "sparse_polish_score_margin", 0.0075
                    )
                ),
                "vara_sparse_polish_noop_fallback": (
                    self._sparse_polish_noop_fallback
                ),
                **self._sparse_polish_rescue_summary(),
            }
        )
        metrics["rollback_count"] = self.rejected_interventions if self.rollback_enabled else 0
        save_json(metrics, self.run_dir / "summary.json")
        if self.sparse_cfd_polish_v2:
            save_json(
                self._sparse_polish_rescue_status["candidate_outcomes"],
                self.run_dir / "vara_sparse_polish_rescue_candidates.json",
            )
        pd.DataFrame([metrics]).to_csv(self.run_dir / "summary_table.csv", index=False)
        pd.DataFrame([metrics]).to_csv(self.table_dir / "summary.csv", index=False)
        save_checkpoint(
            self.checkpoint_dir / "final.pt",
            self.model,
            self.optimizer,
            self.config,
            metrics,
            self.global_step,
            -1,
        )
        return metrics

    def _train_v2_steps(
        self,
        batch: dict[str, Any],
        steps: int,
        cycle: int,
        phase: str,
        probe: bool = False,
        applied: bool = True,
        weight_multipliers: dict[str, float] | None = None,
        lr_scale: float = 1.0,
        ignore_compute_budget: bool = False,
    ) -> int:
        train_cfg = self.config.get("training", {})
        scalar_weights = dict(train_cfg.get("weights", {}))
        if self.sparse_cfd_polish_v2:
            for name, multiplier in (
                self.v2_controller.state.global_multipliers.items()
            ):
                scalar_weights[name] = float(
                    scalar_weights.get(name, 0.0)
                ) * float(multiplier)
        for name, multiplier in dict(weight_multipliers or {}).items():
            current = float(scalar_weights.get(name, 0.0))
            if current == 0.0 and name.startswith("bc_") and name != "bc_uvp_balanced":
                balanced = float(scalar_weights.get("bc_uvp_balanced", 0.0))
                scalar_weights[name] = balanced * max(float(multiplier) - 1.0, 0.0)
            else:
                scalar_weights[name] = current * float(multiplier)
        log_every = max(1, int(train_cfg.get("log_every", 25)))
        self.model.train()
        started = time.perf_counter()
        completed_steps = 0
        for local_step in range(int(steps)):
            if (
                not ignore_compute_budget
                and not self.compute_tracker.can_start_objective(
                    int(batch["xy_f"].shape[0])
                )
            ):
                break
            self._apply_cavity_curriculum()
            self.optimizer.zero_grad(set_to_none=True)
            self.compute_tracker.record_objective(batch)
            profile = self._step_runtime_profile()
            objective_start = time.perf_counter()
            residual_mode, residual_delta = self._residual_loss_settings()
            pointwise = compute_pointwise_losses(
                self.model,
                batch,
                self.benchmark,
                self.steady,
                residual_loss_mode=residual_mode,
                pseudo_huber_delta=residual_delta,
                regularization_config=self._active_loss_config(),
                compute_boundary_loss=self._compute_boundary_training_loss(
                    scalar_weights
                ),
                runtime_profile=profile,
            )
            losses = compute_budgeted_patch_losses(
                pointwise,
                batch,
                self.patch_grid,
                self.v2_controller.state.loss_multipliers,
                reduction=str(train_cfg.get("pointwise_reduction", "legacy_mse")),
            )
            losses, normalization_logs = self.normalize_training_losses(losses)
            total = weighted_sum(losses, scalar_weights)
            anchor = self.pressure_gauge_loss()
            continuation_anchor, continuation_weight = self.continuation_anchor_loss(batch)
            continuation_replay, replay_weight = self.continuation_replay_loss()
            total = total + anchor + continuation_anchor + continuation_replay
            self._record_runtime(
                "loss_objective_total",
                time.perf_counter() - objective_start,
                cycle=cycle,
                phase=phase,
                profile=profile,
            )
            optimizer_start = time.perf_counter()
            total.backward()
            grad_norm = self._grad_norm()
            learning_rate = self.prepare_optimizer_step() * float(lr_scale)
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            self.optimizer.step()
            self.compute_tracker.record_optimizer_step(applied=applied)
            self._record_runtime(
                "backward_optimizer",
                time.perf_counter() - optimizer_start,
                cycle=cycle,
                phase=phase,
            )
            if probe:
                self.compute_tracker.record_probe_step()
            logs = {name: float(value.detach().cpu()) for name, value in losses.items()}
            logs["pressure_gauge"] = float(anchor.detach().cpu())
            logs["continuation_anchor"] = float(continuation_anchor.detach().cpu())
            logs["continuation_anchor_weight"] = float(continuation_weight)
            logs["continuation_replay"] = float(continuation_replay.detach().cpu())
            logs["continuation_replay_weight"] = float(replay_weight)
            logs["total"] = float(total.detach().cpu())
            logs["grad_norm"] = grad_norm
            logs["learning_rate"] = learning_rate
            logs["residual_loss_mode"] = residual_mode
            logs["pseudo_huber_delta"] = float(residual_delta)
            logs.update(self.current_cavity_curriculum)
            logs.update(self._model_auxiliary_logs())
            logs.update(normalization_logs)
            logs.update(self.last_boundary_sampling_summary)
            self.last_losses = logs
            if local_step % log_every == 0 or local_step == steps - 1:
                self.loss_logger.log(
                    {
                        "cycle": cycle,
                        "phase": phase,
                        "epoch": self.global_step,
                        **logs,
                    }
                )
            self.global_step += 1
            completed_steps += 1
        self.compute_tracker.add_phase_time("optimization", time.perf_counter() - started)
        return completed_steps

    def _diagnose_reference_free(
        self,
        update_history: bool = True,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], list[Any], np.ndarray]:
        started = time.perf_counter()
        _x, _y, coords = self.validation_grid()
        builder = self.diagnostic_builder()
        maps = builder.build(coords, mode="residual_only")
        configured = list(self.config.get("diagnostics", {}).get("variables", []))
        names = [
            name for name in configured
            if name in maps and not any(token in name.lower() for token in ("error", "reference", "ghia", "cfd"))
        ]
        if not names:
            names = [
                "continuity_residual",
                "momentum_u_residual",
                "momentum_v_residual",
                "aggregate_pde_residual",
                "boundary_violation",
            ]
        self.v2_controller.assert_reference_free(names)
        original = self.patch_scorer.diagnostics
        self.patch_scorer.diagnostics = names
        normalized, scored_names = self.patch_scorer.compute(maps, coords, update_ema=update_history)
        raw = np.asarray(self.patch_scorer.last_raw_scores, dtype=float)
        self.patch_scorer.diagnostics = original
        weak_regions = self.weak_detector.detect(normalized, scored_names, self.patch_grid)
        self.compute_tracker.add_phase_time("diagnostics", time.perf_counter() - started)
        return maps, raw, scored_names, weak_regions, coords

    def _guard_metrics(self, coords: np.ndarray) -> dict[str, float]:
        metrics = evaluate_on_grid(
            self.model,
            self.benchmark,
            coords,
            self.device,
            self.steady,
            residual_interior_only=self.residual_interior_only(),
            include_reference_metrics=False,
            include_streamfunction_metrics=False,
        )
        selected = {
            name: float(metrics[name])
            for name in self.v2_config.guard_metrics
            if name != "unweighted_validation_loss"
            and name in metrics
            and math.isfinite(float(metrics[name]))
        }
        # The ordinary validation metric may include full-field reference data.
        # V2 instead uses the reference-free PDE + boundary validation objective.
        selected["unweighted_validation_loss"] = float(
            metrics["unweighted_pde_loss"] + metrics["unweighted_bc_loss"]
        )
        if self.sparse_cfd_polish_v2 and self.cfd_supervision is not None:
            selected.update(
                {
                    name: float(value)
                    for name, value in self._cfd_sparse_metrics().items()
                    if name
                    in {
                        "cfd_velocity_mse_sparse",
                        "cfd_u_mse_sparse",
                        "cfd_v_mse_sparse",
                    }
                }
            )
        self.v2_controller.assert_reference_free(selected)
        return selected

    def _sparse_polish_score(self, metrics: dict[str, float]) -> float:
        if not self.sparse_cfd_polish_v2:
            return float("nan")
        score = 0.0
        for name, weight in SPARSE_POLISH_SCORE_WEIGHTS.items():
            value = float(metrics.get(name, float("nan")))
            if not math.isfinite(value):
                return math.inf
            scale = max(
                float(self._sparse_polish_score_scales.get(name, value)),
                1e-12,
            )
            score += weight * max(value, 0.0) / scale
        speed = float(metrics.get("speed_pred_max", float("nan")))
        speed_gate = float(
            self.config.get("continuation_validity", {}).get(
                "max_speed_pred", math.inf
            )
        )
        if math.isfinite(speed) and math.isfinite(speed_gate):
            score += 0.4 * (
                max(speed - speed_gate, 0.0) / max(speed_gate, 1e-12)
            ) ** 2
        return float(score)

    def _evaluate_sparse_polish_candidate(
        self,
        candidate: V2Candidate,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
    ) -> tuple[bool, dict[str, Any]]:
        self.v2_controller.assert_reference_free(
            set(before_metrics) | set(after_metrics)
        )
        if not self._sparse_polish_score_scales:
            self._sparse_polish_score_scales = {
                name: max(float(before_metrics[name]), 1e-12)
                for name in SPARSE_POLISH_SCORE_WEIGHTS
                if name in before_metrics
                and math.isfinite(float(before_metrics[name]))
            }
            self._sparse_polish_initial_score = self._sparse_polish_score(
                before_metrics
            )
        before_score = self._sparse_polish_score(before_metrics)
        after_score = self._sparse_polish_score(after_metrics)
        eps = 1e-12
        observed = (before_score - after_score) / (abs(before_score) + eps)
        score_margin = max(
            self.v2_config.noise_floor,
            float(
                self.config.get("controller_v2", {}).get(
                    "sparse_polish_score_margin", 0.0075
                )
            ),
        )
        tolerances = self._sparse_polish_guard_tolerances()
        probe_steps = max(
            1,
            int(self.config.get("controller_v2", {}).get("probe_steps", 1)),
        )
        improvement_per_compute = observed / probe_steps
        minimum_improvement_per_compute = float(
            self.config.get("controller_v2", {}).get(
                "sparse_polish_min_probe_improvement_per_compute", 5e-4
            )
        )
        guard_changes: dict[str, float] = {}
        speed_cap_violation = False
        for name in SPARSE_POLISH_GUARD_METRICS:
            if name not in before_metrics or name not in after_metrics:
                continue
            before = float(before_metrics[name])
            after = float(after_metrics[name])
            if name == "speed_pred_max":
                speed_gate = float(
                    self.config.get("continuation_validity", {}).get(
                        "max_speed_pred", math.inf
                    )
                )
                speed_cap_violation = math.isfinite(speed_gate) and after > speed_gate
                before = max(before - speed_gate, 0.0)
                after = max(after - speed_gate, 0.0)
            guard_changes[name] = (after - before) / (abs(before) + eps)
        guard_ok = all(
            change <= tolerances.get(name, 0.005)
            for name, change in guard_changes.items()
        ) and not speed_cap_violation
        accepted = (
            math.isfinite(before_score)
            and math.isfinite(after_score)
            and observed > score_margin
            and improvement_per_compute > minimum_improvement_per_compute
            and guard_ok
        )
        predicted = max(
            candidate.predicted_target_improvement, self.v2_config.noise_floor
        )
        reward_ratio = observed / predicted
        self._sparse_polish_score_before = before_score
        self._sparse_polish_score_after = after_score
        reject_reason = self._sparse_polish_reject_reason(
            observed,
            score_margin,
            improvement_per_compute,
            minimum_improvement_per_compute,
            guard_changes,
            tolerances,
            speed_cap_violation,
            candidate,
        )
        return accepted, {
            "accepted": bool(accepted),
            "target_noise": score_margin,
            "observed_target_improvement": observed,
            "accepted_improvement_per_compute": improvement_per_compute,
            "predicted_target_improvement": candidate.predicted_target_improvement,
            "predicted_guard_damage": candidate.predicted_guard_damage,
            "reward_ratio": reward_ratio,
            "comparison_mode": "sparse_polish_counterfactual",
            "guard_changes": guard_changes,
            "guard_noise": {
                name: tolerances.get(name, 0.005) for name in guard_changes
            },
            "trust_radius_before": self.v2_controller.trust_radius,
            "rollback_reason": "" if accepted else reject_reason,
            "score_before": before_score,
            "score_after": after_score,
            "vara_sparse_polish_score_before": before_score,
            "vara_sparse_polish_score_after": after_score,
        }

    def _sparse_polish_candidates(
        self, candidates: list[V2Candidate]
    ) -> list[V2Candidate]:
        conservative: list[V2Candidate] = []
        for candidate in candidates:
            if candidate.action_type == "sampling":
                conservative.append(candidate)
                continue
            if candidate.action_type == "boundary_data_guard":
                conservative.append(candidate)
                continue
            history = self.v2_controller.score_history.get(
                (candidate.variable, candidate.patch_id), []
            )
            if (
                candidate.persistence >= 3
                and len(history) >= 3
                and candidate.gradient_compatibility >= 0.80
            ):
                conservative.append(candidate)
        repeated_failure = any(
            candidate.persistence >= 3
            for candidate in conservative
            if candidate.action_type == "boundary_data_guard"
        )
        priority = {
            "boundary_data_guard": 0 if repeated_failure else 1,
            "sampling": 1 if repeated_failure else 0,
            "local_loss": 2,
            "joint": 3,
        }
        return sorted(
            conservative,
            key=lambda item: (
                priority.get(item.action_type, 4),
                -item.rank_score,
            ),
        )

    def _update_sparse_polish_best(self, metrics: dict[str, float]) -> None:
        if not self.sparse_cfd_polish_v2:
            return
        score = self._sparse_polish_score(metrics)
        if not math.isfinite(score):
            return
        if (
            self._sparse_polish_best is None
            or score < float(self._sparse_polish_best["score"])
        ):
            if not self._sparse_polish_score_scales:
                self._sparse_polish_score_scales = {
                    name: max(float(metrics[name]), 1e-12)
                    for name in SPARSE_POLISH_SCORE_WEIGHTS
                    if name in metrics and math.isfinite(float(metrics[name]))
                }
                score = self._sparse_polish_score(metrics)
                self._sparse_polish_initial_score = score
            self._sparse_polish_best = {
                "score": score,
                "model": self._model_snapshot(),
                "optimizer": deepcopy(self.optimizer.state_dict()),
                "loss_normalization": deepcopy(self.loss_normalization_state),
                "controller": self.v2_controller.state.snapshot(),
                "sampling": self.sampling_state_snapshot(),
            }

    def _restore_sparse_polish_best_if_needed(
        self,
        current_metrics: dict[str, float],
        *,
        force: bool = False,
    ) -> bool:
        if not self.sparse_cfd_polish_v2 or self._sparse_polish_best is None:
            return False
        current_score = self._sparse_polish_score(current_metrics)
        best_score = float(self._sparse_polish_best["score"])
        tolerance = float(
            self.config.get("controller_v2", {}).get(
                "sparse_polish_final_restore_tolerance", 0.002
            )
        )
        if not force and math.isfinite(current_score) and current_score <= best_score * (
            1.0 + tolerance
        ):
            return False
        self._restore_model_snapshot(self._sparse_polish_best["model"])
        self.optimizer.load_state_dict(self._sparse_polish_best["optimizer"])
        self.loss_normalization_state = deepcopy(
            self._sparse_polish_best["loss_normalization"]
        )
        self.v2_controller.state.restore(self._sparse_polish_best["controller"])
        self.restore_sampling_state(self._sparse_polish_best["sampling"])
        self._sync_benchmark_corner_to_model()
        self._sparse_polish_restored_best = True
        return True

    def _sparse_polish_guard_tolerances(self) -> dict[str, float]:
        cfg = dict(self.config.get("controller_v2", {}))
        protected = float(cfg.get("sparse_polish_protected_tolerance", 0.005))
        physics = float(cfg.get("sparse_polish_physics_tolerance", 0.01))
        return {
            "pde_residual_mean": physics,
            "momentum_residual_mean": physics,
            "continuity_residual_mean": protected,
            "boundary_condition_error": protected,
            "cfd_velocity_mse_sparse": protected,
            "cfd_u_mse_sparse": protected,
            "cfd_v_mse_sparse": protected,
            "top_band_continuity_residual_mean": protected,
            "speed_pred_max": 0.0,
        }

    def _sparse_polish_should_noop(
        self,
        score_gain: float,
        *,
        minimum_gain: float | None = None,
    ) -> bool:
        threshold = (
            float(minimum_gain)
            if minimum_gain is not None
            else float(
                self.config.get("controller_v2", {}).get(
                    "sparse_polish_noop_min_score_gain", 0.01
                )
            )
        )
        return self.accepted_interventions == 0 or float(score_gain) < threshold

    def _sparse_polish_reject_reason(
        self,
        observed: float,
        score_margin: float,
        improvement_per_compute: float,
        minimum_improvement_per_compute: float,
        guard_changes: dict[str, float],
        tolerances: dict[str, float],
        speed_cap_violation: bool,
        candidate: V2Candidate,
    ) -> str:
        if speed_cap_violation:
            return "speed_cap_violation"
        for name, reason in (
            ("continuity_residual_mean", "worsened_continuity"),
            ("top_band_continuity_residual_mean", "worsened_continuity"),
            ("boundary_condition_error", "worsened_boundary"),
            ("cfd_velocity_mse_sparse", "worsened_sparse_cfd_mse"),
            ("cfd_u_mse_sparse", "worsened_sparse_cfd_mse"),
            ("cfd_v_mse_sparse", "worsened_sparse_cfd_mse"),
            ("pde_residual_mean", "worsened_pde"),
            ("momentum_residual_mean", "worsened_pde"),
        ):
            if guard_changes.get(name, -math.inf) > tolerances.get(name, 0.005):
                return reason
        if candidate.gradient_compatibility < 0.0:
            return "gradient_conflict"
        if (
            observed <= score_margin
            or improvement_per_compute <= minimum_improvement_per_compute
        ):
            return "insufficient_score_gain"
        return "pareto_guard_violation"

    def _sparse_polish_rescue_candidate(
        self,
        weak_regions: list[Any],
        metrics: dict[str, float],
    ) -> V2Candidate | None:
        if not self.sparse_cfd_polish_v2 or not weak_regions:
            return None
        primary = weak_regions[0]
        variable = str(primary.variable)
        if not any(
            token in variable.lower()
            for token in ("continuity", "boundary", "near_wall", "top_band")
        ):
            return None
        return V2Candidate(
            variable=variable,
            patch_id=int(primary.patch_id),
            action_type="boundary_data_guard",
            loss_names=[
                "bc_uvp_balanced",
                "top_band_continuity",
                "cfd_v_mse",
            ],
            severity=float(primary.severity),
            persistence=int(getattr(primary, "persistence", 1)),
            trend=0.0,
        )

    @staticmethod
    def _empty_sparse_polish_rescue_status() -> dict[str, Any]:
        return {
            "triggered": False,
            "reason": "",
            "candidate_count": 0,
            "candidate_steps": 0,
            "best_candidate": "",
            "accepted": False,
            "score_before": float("nan"),
            "score_after": float("nan"),
            "continuity_before": float("nan"),
            "continuity_after": float("nan"),
            "boundary_before": float("nan"),
            "boundary_after": float("nan"),
            "cfd_mse_before": float("nan"),
            "cfd_mse_after": float("nan"),
            "pde_before": float("nan"),
            "pde_after": float("nan"),
            "extra_optimizer_steps": 0,
            "reverted_final_polish": False,
            "candidate_outcomes": [],
            "gate_gaps": {},
            "thresholds": {},
            "reynolds": float("nan"),
        }

    def _sparse_polish_gate_thresholds(self) -> dict[str, float]:
        validity = dict(self.config.get("continuation_validity", {}))
        return {
            "continuity_residual_mean": float(
                validity.get("max_continuity_residual_mean", math.inf)
            ),
            "boundary_condition_error": float(
                validity.get("max_boundary_condition_error", math.inf)
            ),
            "pde_residual_mean": float(
                validity.get("max_pde_residual_mean", math.inf)
            ),
            "momentum_residual_mean": float(
                validity.get("max_momentum_residual_mean", math.inf)
            ),
            "speed_pred_max": float(
                validity.get("max_speed_pred", math.inf)
            ),
        }

    def _sparse_polish_gate_gaps(
        self,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        thresholds = self._sparse_polish_gate_thresholds()
        gaps: dict[str, float] = {}
        for metric, label in (
            ("continuity_residual_mean", "continuity"),
            ("boundary_condition_error", "boundary"),
            ("pde_residual_mean", "pde"),
            ("momentum_residual_mean", "momentum"),
        ):
            value = float(metrics.get(metric, float("nan")))
            target = float(thresholds[metric])
            gaps[label] = (
                max(value / target - 1.0, 0.0)
                if math.isfinite(value)
                and math.isfinite(target)
                and target > 0.0
                else 0.0
            )
        gaps["total"] = sum(gaps.values())
        return gaps

    def _sparse_polish_rescue_score(
        self,
        metrics: dict[str, float],
    ) -> float:
        gaps = self._sparse_polish_gate_gaps(metrics)
        normalized = 0.0
        for name, weight in (
            ("pde_residual_mean", 1.2),
            ("momentum_residual_mean", 1.2),
            ("cfd_velocity_mse_sparse", 1.4),
            ("top_band_continuity_residual_mean", 0.8),
            ("near_wall_pde_residual_mean", 0.5),
        ):
            value = float(metrics.get(name, float("nan")))
            scale = float(self._sparse_polish_score_scales.get(name, value))
            if not math.isfinite(value) or not math.isfinite(scale):
                return math.inf
            normalized += weight * max(value, 0.0) / max(scale, 1e-12)
        speed = float(metrics.get("speed_pred_max", float("nan")))
        speed_cap = self._sparse_polish_gate_thresholds()["speed_pred_max"]
        speed_penalty = (
            (max(speed - speed_cap, 0.0) / max(speed_cap, 1e-12)) ** 2
            if math.isfinite(speed) and math.isfinite(speed_cap)
            else 0.0
        )
        return float(
            3.0 * gaps["continuity"]
            + 2.8 * gaps["boundary"]
            + normalized
            + speed_penalty
        )

    def _should_run_sparse_polish_rescue(
        self,
        metrics: dict[str, float],
    ) -> tuple[bool, str]:
        if not self.sparse_cfd_polish_v2:
            return False, "not_sparse_cfd_polish_vara"
        cfg = dict(
            self.config.get("controller_v2", {}).get(
                "sparse_polish_rescue", {}
            )
        )
        if not bool(cfg.get("enabled", True)):
            return False, "disabled"
        gaps = self._sparse_polish_gate_gaps(metrics)
        if gaps["continuity"] <= 0.0 and gaps["boundary"] <= 0.0:
            return False, "continuity_and_boundary_gates_pass"
        minimum_rollbacks = int(cfg.get("minimum_rollbacks", 4))
        low_gain = self._accepted_improvement_per_compute() < float(
            cfg.get("accepted_gain_threshold", 5e-4)
        )
        weak_controller = self.accepted_interventions == 0 or low_gain
        if self.rejected_interventions < minimum_rollbacks or not weak_controller:
            return False, "controller_has_not_stalled"
        thresholds = self._sparse_polish_gate_thresholds()
        pde = float(metrics.get("pde_residual_mean", math.inf))
        momentum = float(metrics.get("momentum_residual_mean", math.inf))
        healthy_factor = float(cfg.get("healthy_physics_factor", 2.0))
        if (
            math.isfinite(thresholds["pde_residual_mean"])
            and pde > healthy_factor * thresholds["pde_residual_mean"]
        ) or (
            math.isfinite(thresholds["momentum_residual_mean"])
            and momentum
            > healthy_factor * thresholds["momentum_residual_mean"]
        ):
            return False, "physics_not_close_enough"
        return True, "gate_deficit_after_repeated_rollbacks"

    def _restore_sparse_polish_snapshot(
        self,
        snapshot: dict[str, Any],
    ) -> None:
        self._restore_model_snapshot(snapshot["model"])
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.loss_normalization_state = deepcopy(
            snapshot["loss_normalization"]
        )
        self.v2_controller.state.restore(snapshot["controller"])
        self.restore_sampling_state(snapshot["sampling"])
        self._sync_benchmark_corner_to_model()

    def _sparse_polish_rescue_acceptance(
        self,
        before: dict[str, float],
        after: dict[str, float],
        *,
        require_gate_gain: bool = True,
    ) -> tuple[bool, str]:
        before_score = self._sparse_polish_rescue_score(before)
        after_score = self._sparse_polish_rescue_score(after)
        if not math.isfinite(after_score) or after_score >= before_score * 0.995:
            return False, "insufficient_score_gain"
        tolerances = {
            "pde_residual_mean": 0.01,
            "momentum_residual_mean": 0.01,
            "continuity_residual_mean": 0.0,
            "boundary_condition_error": 0.0,
            "cfd_velocity_mse_sparse": 0.003,
            "cfd_v_mse_sparse": 0.003,
        }
        for name, tolerance in tolerances.items():
            if name not in before or name not in after:
                continue
            change = (float(after[name]) - float(before[name])) / max(
                abs(float(before[name])), 1e-12
            )
            if change > tolerance:
                return False, {
                    "continuity_residual_mean": "worsened_continuity",
                    "boundary_condition_error": "worsened_boundary",
                    "cfd_velocity_mse_sparse": "worsened_sparse_cfd_mse",
                    "cfd_v_mse_sparse": "worsened_sparse_cfd_mse",
                    "pde_residual_mean": "worsened_pde",
                    "momentum_residual_mean": "worsened_pde",
                }[name]
        speed = float(after.get("speed_pred_max", float("nan")))
        speed_cap = self._sparse_polish_gate_thresholds()["speed_pred_max"]
        if math.isfinite(speed_cap) and (
            not math.isfinite(speed) or speed > speed_cap
        ):
            return False, "speed_cap_violation"
        if require_gate_gain:
            continuity_gain = (
                float(before["continuity_residual_mean"])
                - float(after["continuity_residual_mean"])
            ) / max(abs(float(before["continuity_residual_mean"])), 1e-12)
            boundary_gain = (
                float(before["boundary_condition_error"])
                - float(after["boundary_condition_error"])
            ) / max(abs(float(before["boundary_condition_error"])), 1e-12)
            if max(continuity_gain, boundary_gain) < 0.03:
                return False, "insufficient_gate_improvement"
        return True, ""

    def _run_sparse_polish_gate_rescue(
        self,
        current_metrics: dict[str, float],
        coords: np.ndarray,
    ) -> dict[str, Any]:
        status = self._empty_sparse_polish_rescue_status()
        status["reynolds"] = float(
            self.config.get("benchmark_params", {}).get(
                "reynolds", float("nan")
            )
        )
        status["thresholds"] = self._sparse_polish_gate_thresholds()
        status["gate_gaps"] = self._sparse_polish_gate_gaps(current_metrics)
        triggered, reason = self._should_run_sparse_polish_rescue(
            current_metrics
        )
        status["reason"] = reason
        if not triggered or self._sparse_polish_best is None:
            return status
        status["triggered"] = True
        cfg = dict(
            self.config.get("controller_v2", {}).get(
                "sparse_polish_rescue", {}
            )
        )
        max_candidates = min(
            max(int(cfg.get("max_candidates", 5)), 1),
            len(SPARSE_POLISH_RESCUE_CANDIDATES),
        )
        max_extra = max(int(cfg.get("max_extra_optimizer_steps", 1200)), 0)
        requested_steps = max(int(cfg.get("candidate_steps", 200)), 1)
        candidate_steps = min(
            requested_steps,
            max_extra // max(max_candidates, 1),
        )
        if candidate_steps <= 0:
            status["reason"] = "extra_optimizer_budget_zero"
            return status
        status["candidate_count"] = max_candidates
        status["candidate_steps"] = candidate_steps
        lr_scale = float(cfg.get("lr_scale", 0.35))
        base = deepcopy(self._sparse_polish_best)
        self._restore_sparse_polish_snapshot(base)
        base_step = self.global_step
        base_metrics = self._guard_metrics(coords)
        base_score = self._sparse_polish_rescue_score(base_metrics)
        status.update(
            {
                "score_before": base_score,
                "continuity_before": base_metrics["continuity_residual_mean"],
                "boundary_before": base_metrics["boundary_condition_error"],
                "cfd_mse_before": base_metrics["cfd_velocity_mse_sparse"],
                "pde_before": base_metrics["pde_residual_mean"],
            }
        )
        sampling_state = self.sampling_state_snapshot()
        rescue_batch = self.initial_batch()
        self.restore_sampling_state(sampling_state)
        candidates: list[dict[str, Any]] = []
        total_extra = 0
        for name, multipliers in SPARSE_POLISH_RESCUE_CANDIDATES[:max_candidates]:
            self._restore_sparse_polish_snapshot(base)
            self.global_step = base_step
            completed = self._train_v2_steps(
                rescue_batch,
                candidate_steps,
                cycle="sparse_polish_rescue",
                phase=name,
                probe=True,
                applied=False,
                weight_multipliers=multipliers,
                lr_scale=lr_scale,
                ignore_compute_budget=True,
            )
            total_extra += completed
            after = self._guard_metrics(coords)
            accepted, rejection = self._sparse_polish_rescue_acceptance(
                base_metrics,
                after,
            )
            outcome = {
                "candidate_name": name,
                "accepted": accepted,
                "score_before": base_score,
                "score_after": self._sparse_polish_rescue_score(after),
                "continuity_before": base_metrics["continuity_residual_mean"],
                "continuity_after": after["continuity_residual_mean"],
                "boundary_before": base_metrics["boundary_condition_error"],
                "boundary_after": after["boundary_condition_error"],
                "pde_before": base_metrics["pde_residual_mean"],
                "pde_after": after["pde_residual_mean"],
                "cfd_mse_before": base_metrics["cfd_velocity_mse_sparse"],
                "cfd_mse_after": after["cfd_velocity_mse_sparse"],
                "rejection_reason": rejection,
                "optimizer_steps": completed,
                "snapshot": {
                    "model": self._model_snapshot(),
                    "optimizer": deepcopy(self.optimizer.state_dict()),
                    "loss_normalization": deepcopy(
                        self.loss_normalization_state
                    ),
                    "controller": self.v2_controller.state.snapshot(),
                    "sampling": self.sampling_state_snapshot(),
                },
                "metrics": after,
            }
            candidates.append(outcome)
        accepted_candidates = [
            item for item in candidates if bool(item["accepted"])
        ]
        selected = (
            min(accepted_candidates, key=lambda item: item["score_after"])
            if accepted_candidates
            else None
        )
        for item in candidates:
            if item is not selected:
                self.compute_tracker.record_rollback_steps(
                    int(item["optimizer_steps"])
                )
        if selected is None:
            self._restore_sparse_polish_snapshot(base)
            self.global_step = base_step
        else:
            self._restore_sparse_polish_snapshot(selected["snapshot"])
            selected_steps = int(selected["optimizer_steps"])
            self.global_step = base_step + selected_steps
            self.compute_tracker.record_applied_optimizer_steps(selected_steps)
            self.accepted_interventions += 1
            status["accepted"] = True
            status["best_candidate"] = str(selected["candidate_name"])
            selected_metrics = dict(selected["metrics"])
            final_polish_steps = (
                min(50, max(30, int(cfg.get("final_lbfgs_steps", 5)) * 6))
                if int(cfg.get("final_lbfgs_steps", 5)) > 0
                else 0
            )
            remaining_extra = max(max_extra - total_extra, 0)
            final_polish_steps = min(final_polish_steps, remaining_extra)
            if final_polish_steps > 0:
                pre_polish = deepcopy(selected["snapshot"])
                completed = self._train_v2_steps(
                    rescue_batch,
                    final_polish_steps,
                    cycle="sparse_polish_rescue",
                    phase="best_candidate_final_polish",
                    probe=False,
                    applied=False,
                    weight_multipliers=SPARSE_POLISH_RESCUE_CANDIDATES[-1][1],
                    lr_scale=min(lr_scale, 0.25),
                    ignore_compute_budget=True,
                )
                total_extra += completed
                polished_metrics = self._guard_metrics(coords)
                polish_ok, _polish_reason = (
                    self._sparse_polish_rescue_acceptance(
                        selected_metrics,
                        polished_metrics,
                        require_gate_gain=False,
                    )
                )
                if polish_ok:
                    self.compute_tracker.record_applied_optimizer_steps(completed)
                    selected_metrics = polished_metrics
                else:
                    self.compute_tracker.record_rollback_steps(completed)
                    self._restore_sparse_polish_snapshot(pre_polish)
                    self.global_step -= completed
                    status["reverted_final_polish"] = True
            self._update_sparse_polish_best(selected_metrics)
            status.update(
                {
                    "score_after": self._sparse_polish_rescue_score(
                        selected_metrics
                    ),
                    "continuity_after": selected_metrics[
                        "continuity_residual_mean"
                    ],
                    "boundary_after": selected_metrics[
                        "boundary_condition_error"
                    ],
                    "cfd_mse_after": selected_metrics[
                        "cfd_velocity_mse_sparse"
                    ],
                    "pde_after": selected_metrics["pde_residual_mean"],
                }
            )
        status["extra_optimizer_steps"] = total_extra
        status["candidate_outcomes"] = [
            {
                key: value
                for key, value in item.items()
                if key not in {"snapshot", "metrics"}
            }
            for item in candidates
        ]
        return status

    def _maybe_run_sparse_polish_gate_rescue(
        self,
        metrics: dict[str, float],
        coords: np.ndarray,
    ) -> bool:
        if not self.sparse_cfd_polish_v2:
            return False
        triggered, _reason = self._should_run_sparse_polish_rescue(metrics)
        if not triggered:
            return False
        self._sparse_polish_rescue_status = self._run_sparse_polish_gate_rescue(
            metrics,
            coords,
        )
        return bool(self._sparse_polish_rescue_status["triggered"])

    def _sparse_polish_rescue_summary(self) -> dict[str, Any]:
        status = self._sparse_polish_rescue_status
        gaps = dict(status.get("gate_gaps", {}))
        return {
            "vara_gate_continuity_gap": float(gaps.get("continuity", 0.0)),
            "vara_gate_boundary_gap": float(gaps.get("boundary", 0.0)),
            "vara_gate_pde_gap": float(gaps.get("pde", 0.0)),
            "vara_gate_momentum_gap": float(gaps.get("momentum", 0.0)),
            "vara_gate_total_gap": float(gaps.get("total", 0.0)),
            "vara_sparse_polish_rescue_triggered": bool(status["triggered"]),
            "vara_sparse_polish_rescue_reason": str(status["reason"]),
            "vara_sparse_polish_rescue_candidate_count": int(
                status["candidate_count"]
            ),
            "vara_sparse_polish_rescue_candidate_steps": int(
                status["candidate_steps"]
            ),
            "vara_sparse_polish_rescue_best_candidate": str(
                status["best_candidate"]
            ),
            "vara_sparse_polish_rescue_accepted": bool(status["accepted"]),
            "vara_sparse_polish_rescue_score_before": float(
                status["score_before"]
            ),
            "vara_sparse_polish_rescue_score_after": float(
                status["score_after"]
            ),
            "vara_sparse_polish_rescue_continuity_before": float(
                status["continuity_before"]
            ),
            "vara_sparse_polish_rescue_continuity_after": float(
                status["continuity_after"]
            ),
            "vara_sparse_polish_rescue_boundary_before": float(
                status["boundary_before"]
            ),
            "vara_sparse_polish_rescue_boundary_after": float(
                status["boundary_after"]
            ),
            "vara_sparse_polish_rescue_cfd_mse_before": float(
                status["cfd_mse_before"]
            ),
            "vara_sparse_polish_rescue_cfd_mse_after": float(
                status["cfd_mse_after"]
            ),
            "vara_sparse_polish_rescue_pde_before": float(status["pde_before"]),
            "vara_sparse_polish_rescue_pde_after": float(status["pde_after"]),
            "vara_sparse_polish_rescue_extra_optimizer_steps": int(
                status["extra_optimizer_steps"]
            ),
            "vara_sparse_polish_rescue_reverted_final_polish": bool(
                status["reverted_final_polish"]
            ),
            "vara_rescue_reynolds": float(status["reynolds"]),
            "vara_rescue_used_thresholds": dict(status["thresholds"]),
            "vara_rescue_normalized_score": float(status["score_after"]),
        }

    def _make_probe_batch(self) -> dict[str, Any]:
        cfg = dict(self.config.get("controller_v2", {}))
        n_f = int(cfg.get("gradient_probe_interior", 256))
        n_bc = int(cfg.get("gradient_probe_boundary", 128))
        return self.make_batch(
            self.uniform_sampler.sample(n_f),
            self._sample_boundary(n_bc),
            self.uniform_sampler.sample(0),
        )

    def _candidate_influence(self, candidates: list[V2Candidate]) -> dict[str, dict[str, float]]:
        if not candidates:
            return {}
        pointwise = compute_pointwise_losses(
            self.model,
            self._probe_batch,
            self.benchmark,
            self.steady,
            residual_loss_mode=self._residual_loss_settings()[0],
            pseudo_huber_delta=self._residual_loss_settings()[1],
            regularization_config=self._active_loss_config(),
            compute_boundary_loss=self._compute_boundary_training_loss(
                dict(self.config.get("training", {}).get("weights", {}))
            ),
        )
        parameters = self._influence_parameters()
        # The aggregate PDE entry is the sum of the three equation entries.
        # Do not count it a second time when estimating guard compatibility.
        guard_components = [
            pointwise[name].mean()
            for name in ("continuity", "momentum_u", "momentum_v", "bc")
            if name in pointwise
        ]
        guard = sum(guard_components)
        guard_gradient = self._flat_gradient(guard, parameters)
        result: dict[str, dict[str, float]] = {}
        for candidate in candidates:
            target = self._candidate_probe_loss(candidate, pointwise)
            target_gradient = self._flat_gradient(target, parameters)
            cosine = _cosine(target_gradient, guard_gradient)
            result[candidate.key()] = {
                "gradient_compatibility": max(0.0, cosine),
                "gradient_conflict": max(0.0, -cosine),
            }
            points = int(self._probe_batch["xy_f"].shape[0] + self._probe_batch["xy_bc"].shape[0])
            self.compute_tracker.record_controller_gradient_evaluation(points)
        return result

    def _influence_parameters(self) -> list[torch.nn.Parameter]:
        modules = [module for module in self.model.modules() if isinstance(module, torch.nn.Linear)]
        selected = modules[-2:] if len(modules) >= 2 else modules
        parameters: list[torch.nn.Parameter] = []
        for module in selected:
            parameters.extend([module.weight, module.bias])
        return parameters

    def _candidate_probe_loss(
        self,
        candidate: V2Candidate,
        pointwise: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if candidate.action_type == "boundary_data_guard":
            rescue_terms = [
                pointwise[name].mean()
                for name in candidate.loss_names
                if name in pointwise
            ]
            if rescue_terms:
                return sum(rescue_terms)
        terms: list[torch.Tensor] = []
        for name in candidate.loss_names:
            if name not in pointwise:
                continue
            coord_name = LOSS_COORD_SOURCE.get(name)
            if coord_name not in self._probe_batch:
                continue
            patch_ids = self.patch_grid.assign_torch(self._probe_batch[coord_name])
            mask = patch_ids == candidate.patch_id
            if torch.any(mask):
                terms.append(pointwise[name][mask].mean())
        if terms:
            return sum(terms)
        return pointwise["pde"].mean()

    @staticmethod
    def _flat_gradient(
        loss: torch.Tensor,
        parameters: list[torch.nn.Parameter],
    ) -> torch.Tensor:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        pieces = [
            grad.detach().reshape(-1) if grad is not None else torch.zeros_like(parameter).reshape(-1)
            for parameter, grad in zip(parameters, gradients)
        ]
        if not pieces:
            return loss.detach().new_zeros(1)
        return torch.cat(pieces)

    def _resample_v2_batch(
        self,
        maps: dict[str, np.ndarray],
        coords: np.ndarray,
    ) -> dict[str, Any]:
        n_f, n_bc, n_data = self._training_sample_counts()
        started = time.perf_counter()
        neutral_mass = np.full(
            self.patch_grid.num_patches,
            1.0 / self.patch_grid.num_patches,
            dtype=float,
        )
        sampling_is_neutral = np.allclose(
            self.v2_controller.state.sampling_mass,
            neutral_mass,
            rtol=0.0,
            atol=1e-12,
        )
        if sampling_is_neutral:
            # Exact baseline equivalence before a sampling action is
            # committed: same sampler, same seed, and same point count.
            xy_f_np = self._sample_interior_numpy(n_f)
        else:
            circulation = self._circulation_band_counts(n_f)
            if circulation is not None:
                n_available = circulation["uniform"]
                expected_label = "uniform"
            else:
                n_available, _n_wall, _n_lid = (
                    self._interior_component_counts(n_f)
                )
                expected_label = "core"
            uniform_mass = float(self.v2_config.min_uniform_mass)
            n_uniform = int(round(n_available * uniform_mass))
            n_adaptive = n_available - n_uniform
            pieces = [self.uniform_sampler.sample_numpy(n_uniform)]
            if n_adaptive > 0:
                patch_ids = list(range(self.patch_grid.num_patches))
                pieces.append(
                    self.adaptive_sampler.region_sampler.sample_numpy(
                        patch_ids,
                        n_adaptive,
                        self.v2_controller.state.sampling_mass,
                    )
                )
            core_points = np.vstack(pieces)
            if int(core_points.shape[0]) != n_available:
                raise ValueError(
                    "VARA V2 adaptive resampling produced "
                    f"{core_points.shape[0]} {expected_label} points; "
                    f"expected {n_available} for n_collocation={n_f}."
                )
            xy_f_np = self._sample_interior_numpy(n_f, core_points)
            self.adaptive_sampler.rng.shuffle(xy_f_np)
        xy_f = torch.tensor(xy_f_np, dtype=torch.float32, device=self.device)
        xy_bc = self._sample_boundary(n_bc)
        xy_data = self._sample_data(n_data)
        result = self.make_batch(xy_f, xy_bc, xy_data)
        self._record_runtime("sampling", time.perf_counter() - started, phase="v2_resample")
        return result

    @staticmethod
    def _candidate_score(candidate: V2Candidate, raw: np.ndarray, names: list[str]) -> float:
        if candidate.variable not in names:
            return float("inf")
        return float(raw[names.index(candidate.variable), candidate.patch_id])

    def _log_decision(
        self,
        block: int,
        candidate: V2Candidate,
        decision: dict[str, Any],
    ) -> None:
        row = {
            "block": block,
            **candidate.to_record(),
            **decision,
        }
        for nested in ("guard_changes", "guard_noise"):
            values = row.pop(nested, {})
            for name, value in dict(values).items():
                row[f"{nested}_{name}"] = value
        self.v2_decision_logger.log(row)

    def _log_state(self, block: int) -> None:
        self.v2_state_logger.log(
            {
                "block": block,
                "trust_radius": self.v2_controller.trust_radius,
                **self.v2_controller.state.to_record(),
            }
        )

    def _accepted_improvement_per_compute(self) -> float:
        accepted = [
            float(row.get("observed_target_improvement", 0.0))
            for row in self.v2_controller.decisions
            if row.get("accepted")
        ]
        return float(sum(accepted) / max(1, self.compute_tracker.optimizer_steps))


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denom.detach().cpu()) <= 1e-20:
        return 0.0
    return float(torch.dot(left, right).detach().cpu() / denom.detach().cpu())
