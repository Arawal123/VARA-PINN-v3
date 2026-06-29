"""Independent matched-compute trainer for split-form Cahn--Hilliard."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.controllers.v2_controller import V2Candidate, V2ControllerConfig, VARAV2Controller
from src.diagnostics.weak_region_detector import WeakRegionDetector
from src.utils.config import save_config
from src.utils.io import save_json
from src.utils.seed import set_seed

from .benchmark import CahnHilliardBenchmark
from .diagnostics import (
    CahnHilliardPatchGrid,
    DiagnosticSnapshot,
    build_diagnostic_snapshot,
)
from .losses import LossResult, compute_training_loss
from .metrics import evaluate_cahn_hilliard, make_evaluation_grid
from .models import build_cahn_hilliard_model, model_parameter_hash


SUPPORTED_METHODS = {
    "vanilla",
    "vara_v2",
    "vara_v2_no_variable_awareness",
    "vara_sampling_only",
    "vara_local_loss_only",
}


class CahnHilliardTrainer:
    """Train one isolated Cahn--Hilliard run.

    Controller observations contain only residuals, prescribed-condition
    violations, shared sparse-training mismatch, and prediction-derived
    interface indicators. Full-field exact errors are computed only after the
    controller has finished every decision.
    """

    def __init__(
        self,
        config: dict[str, Any],
        method: str,
        run_dir: str | Path,
    ) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported Cahn--Hilliard method {method!r}.")
        self.config = deepcopy(config)
        self.method = method
        self.seed = int(self.config.get("seed", 0))
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for directory in ("logs", "checkpoints", "figures", "tables"):
            (self.run_dir / directory).mkdir(exist_ok=True)

        set_seed(self.seed)
        self.device = _resolve_device(str(self.config.get("device", "cpu")))
        self.dtype = _resolve_dtype(str(self.config.get("dtype", "float32")))
        self.benchmark = CahnHilliardBenchmark(dict(self.config.get("benchmark", {})))
        self.patch_grid = CahnHilliardPatchGrid.from_config(self.config, self.benchmark)
        self.model = build_cahn_hilliard_model(self.config).to(
            device=self.device, dtype=self.dtype
        )
        self.initial_model_parameter_hash = model_parameter_hash(self.model)

        training_cfg = dict(self.config.get("training", {}))
        self.weights = {
            name: float(value)
            for name, value in dict(training_cfg.get("weights", {})).items()
        }
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(training_cfg.get("lr", 1e-3))
        )

        self.sampling_rng = np.random.default_rng(self.seed + 10_007)
        condition_rng = np.random.default_rng(self.seed + 20_007)
        sparse_seed_offset = int(
            self.config.get("benchmark", {}).get("sparse_seed", 0)
        )
        sparse_rng = np.random.default_rng(self.seed + sparse_seed_offset + 30_007)
        diagnostic_rng = np.random.default_rng(self.seed + 40_007)
        n_boundary = int(training_cfg.get("n_boundary", 512))
        n_initial = int(training_cfg.get("n_initial", 512))
        n_sparse = int(training_cfg.get("n_sparse_data", 256))
        self.boundary_coordinates = self._sample_boundary(n_boundary, condition_rng)
        self.initial_coordinates = self._sample_initial(n_initial, condition_rng)
        self.sparse_coordinates = self._sample_uniform(n_sparse, sparse_rng)
        self.boundary_targets = self.benchmark.boundary_values(
            self.boundary_coordinates
        ).detach()
        self.initial_targets = self.benchmark.initial_values(
            self.initial_coordinates
        ).detach()
        self.sparse_targets = self.benchmark.exact(self.sparse_coordinates).detach()
        self.sparse_hash = _tensor_hash(self.sparse_coordinates, self.sparse_targets)
        configured_controller = dict(self.config.get("controller_v2", {}))
        configured_mass_guard = dict(
            configured_controller.get("cahn_hilliard_mass_guard", {})
        )
        mass_source = str(
            configured_mass_guard.get("source", "initial_condition")
        )
        if mass_source == "initial_condition":
            mass_target_values = self.initial_targets[:, 0]
        elif mass_source == "sparse_supervision" and self.sparse_targets.numel():
            mass_target_values = self.sparse_targets[:, 0]
        else:
            mass_target_values = self.initial_targets[:, 0]
        self.mass_proxy_baseline = float(
            mass_target_values.mean().cpu()
        )

        diagnostics_cfg = dict(self.config.get("diagnostics", {}))
        self.diagnostic_batch = self._make_diagnostic_batch(
            diagnostic_rng,
            int(diagnostics_cfg.get("n_interior", min(1024, int(training_cfg.get("n_collocation", 2048))))),
            int(diagnostics_cfg.get("n_boundary", min(256, n_boundary))),
            int(diagnostics_cfg.get("n_initial", min(256, n_initial))),
        )
        self.diagnostic_percentile = float(
            diagnostics_cfg.get("aggregation_percentile", 90.0)
        )
        self.interface_tau = float(diagnostics_cfg.get("interface_tau", 0.25))
        self.interface_focus_strength = float(
            diagnostics_cfg.get(
                "interface_beta",
                diagnostics_cfg.get("interface_focus_strength", 2.0),
            )
        )
        self.interface_threshold = float(
            diagnostics_cfg.get("interface_threshold", 0.8)
        )
        self.diagnostic_priority_weights = {
            name: float(value)
            for name, value in dict(
                diagnostics_cfg.get("priority_weights", {})
            ).items()
        }
        losses_cfg = dict(self.config.get("losses", {}))
        self.mu_support_only = bool(losses_cfg.get("mu_support_only", True))
        self.mu_multiplier_max = float(
            losses_cfg.get("mu_local_multiplier_max", 1.25)
        )
        self.mu_priority_max = float(losses_cfg.get("mu_priority_max", 0.5))
        configured_awareness = bool(
            self.config.get("controller_v2", {}).get(
                "variable_awareness_enabled", True
            )
        )
        self.variable_awareness = (
            method != "vara_v2_no_variable_awareness" and configured_awareness
        )
        self.detector = WeakRegionDetector(
            percentile_threshold=float(diagnostics_cfg.get("weak_percentile", 80.0)),
            top_k_per_variable=int(diagnostics_cfg.get("top_k_per_variable", 2)),
            min_active_patches=1,
            max_active_patches=int(diagnostics_cfg.get("max_active_patches", 8)),
            persistence_cycles=int(diagnostics_cfg.get("persistence_cycles", 1)),
        )

        self.controller: VARAV2Controller | None = None
        self.rollback_enabled = False
        controller_cfg = deepcopy(dict(self.config.get("controller_v2", {})))
        self.u_first_policy_enabled = bool(
            controller_cfg.get("cahn_hilliard_u_first_policy", True)
        )
        self.max_mu_candidates_per_block = int(
            controller_cfg.get("max_mu_candidates_per_block", 1)
        )
        self.u_guard_config = dict(
            controller_cfg.get("cahn_hilliard_guards", {})
        )
        self.pareto_policy_config = dict(
            controller_cfg.get("cahn_hilliard_pareto_policy", {})
        )
        self.pareto_score_config = dict(
            controller_cfg.get("cahn_hilliard_pareto_score", {})
        )
        self.mass_guard_config = dict(
            controller_cfg.get("cahn_hilliard_mass_guard", {})
        )
        self.phase_guard_config = dict(
            controller_cfg.get("cahn_hilliard_phase_guard", {})
        )
        self.post_block_guard_config = dict(
            controller_cfg.get("cahn_hilliard_post_block_guard", {})
        )
        self.best_safe_candidate_config = dict(
            controller_cfg.get("cahn_hilliard_best_safe_candidate", {})
        )
        self.inactivity_recovery_config = dict(
            controller_cfg.get("cahn_hilliard_inactivity_recovery", {})
        )
        if method != "vanilla":
            controller_cfg["variable_awareness_enabled"] = self.variable_awareness
            controller_cfg.setdefault(
                "guard_metrics",
                [
                    "pde_residual_mean",
                    "boundary_condition_error",
                    "unweighted_validation_loss",
                    "unweighted_physics_validation_loss",
                ],
            )
            self.controller = VARAV2Controller(
                V2ControllerConfig.from_dict(
                    controller_cfg, self.patch_grid.num_patches
                )
            )
            self.rollback_enabled = bool(controller_cfg.get("rollback_enabled", True))

        self.loss_rows: list[dict[str, Any]] = []
        self.decision_rows: list[dict[str, Any]] = []
        self.allocation_history: list[dict[str, Any]] = []
        self.applied_optimizer_steps = 0
        self.optimizer_step_calls = 0
        self.objective_evaluations = 0
        self.diagnostic_evaluations = 0
        self.accepted_interventions = 0
        self.rejected_interventions = 0
        self.prefiltered_interventions = 0
        self.rollback_count = 0
        self.accepted_u_interface_interventions = 0
        self.accepted_mu_interventions = 0
        self.rejected_due_to_sparse_u_guard = 0
        self.rejected_due_to_ic_u_guard = 0
        self.rejected_due_to_bc_u_guard = 0
        self.rejected_due_to_phase_range_guard = 0
        self.rejected_due_to_mass_guard = 0
        self.rejected_due_to_interface_proxy_guard = 0
        self.accepted_pareto_safe_interventions = 0
        self.rejected_hard_guard_pde = 0
        self.rejected_hard_guard_ch = 0
        self.rejected_hard_guard_mass = 0
        self.rejected_hard_guard_phase = 0
        self.rejected_hard_guard_sparse_u = 0
        self.rejected_mu_only = 0
        self.post_block_rollbacks = 0
        self.accepted_interface_targets = 0
        self.accepted_sparse_u_targets = 0
        self.all_rejected_blocks = 0
        self.consecutive_all_rejected_blocks = 0
        self.consecutive_all_rejected_blocks_max = 0
        self.best_safe_candidate_activations = 0
        self.accepted_by_standard_pareto = 0
        self.accepted_by_best_safe_fallback = 0
        self.rejected_due_to_primary_reward = 0
        self.rejected_due_to_hard_guard = 0
        self.inactivity_recovery_activations = 0
        self._temporary_reward_multiplier = 1.0
        self._accepted_primary_rewards: list[float] = []
        self._rejected_primary_rewards: list[float] = []
        self._accepted_guard_penalties: list[float] = []
        self._rejected_guard_penalties: list[float] = []
        self.pareto_metric_history: dict[str, list[float]] = {}
        self.candidate_guard_memory: dict[str, float] = {}
        self.optimization_wall_clock_sec = 0.0
        self.wall_clock_eval_sec = 0.0
        save_config(self.config, self.run_dir / "resolved_config.yaml")

    def run(self) -> dict[str, Any]:
        """Train, evaluate, and persist one complete run."""
        schedule = self._schedule()
        started = time.perf_counter()
        if self.method == "vanilla":
            self._run_vanilla(schedule)
        else:
            self._run_vara(schedule)
        self.optimization_wall_clock_sec = time.perf_counter() - started
        if self.applied_optimizer_steps != schedule["total_steps"]:
            raise RuntimeError(
                "Cahn--Hilliard step budget drifted: "
                f"expected {schedule['total_steps']}, got {self.applied_optimizer_steps}."
            )
        evaluation_started = time.perf_counter()
        metrics = self._final_evaluation()
        self.wall_clock_eval_sec = time.perf_counter() - evaluation_started
        metrics.update(
            {
                "optimization_wall_clock_sec": self.optimization_wall_clock_sec,
                "wall_clock_eval_sec": self.wall_clock_eval_sec,
                "applied_optimizer_steps": self.applied_optimizer_steps,
                "optimizer_step_calls": self.optimizer_step_calls,
                "objective_evaluations": self.objective_evaluations,
                "diagnostic_evaluations": self.diagnostic_evaluations,
                "accepted_interventions": self.accepted_interventions,
                "rejected_interventions": self.rejected_interventions,
                "prefiltered_interventions": self.prefiltered_interventions,
                "rollback_count": self.rollback_count,
                "accepted_u_interface_interventions": self.accepted_u_interface_interventions,
                "accepted_mu_interventions": self.accepted_mu_interventions,
                "rejected_legacy_sparse_u_guard": self.rejected_due_to_sparse_u_guard,
                "rejected_due_to_ic_u_guard": self.rejected_due_to_ic_u_guard,
                "rejected_due_to_bc_u_guard": self.rejected_due_to_bc_u_guard,
                "rejected_due_to_phase_range_guard": self.rejected_due_to_phase_range_guard,
                "rejected_legacy_mass_guard": self.rejected_due_to_mass_guard,
                "rejected_due_to_interface_proxy_guard": self.rejected_due_to_interface_proxy_guard,
                "accepted_pareto_safe_interventions": self.accepted_pareto_safe_interventions,
                "rejected_hard_guard_pde": self.rejected_hard_guard_pde,
                "rejected_hard_guard_ch": self.rejected_hard_guard_ch,
                "rejected_hard_guard_mass": self.rejected_hard_guard_mass,
                "rejected_hard_guard_phase": self.rejected_hard_guard_phase,
                "rejected_hard_guard_sparse_u": self.rejected_hard_guard_sparse_u,
                "rejected_mu_only": self.rejected_mu_only,
                "post_block_rollbacks": self.post_block_rollbacks,
                "accepted_interface_targets": self.accepted_interface_targets,
                "accepted_sparse_u_targets": self.accepted_sparse_u_targets,
                "all_rejected_blocks": self.all_rejected_blocks,
                "consecutive_all_rejected_blocks_max": self.consecutive_all_rejected_blocks_max,
                "best_safe_candidate_activations": self.best_safe_candidate_activations,
                "accepted_by_standard_pareto": self.accepted_by_standard_pareto,
                "accepted_by_best_safe_fallback": self.accepted_by_best_safe_fallback,
                "rejected_due_to_primary_reward": self.rejected_due_to_primary_reward,
                "rejected_due_to_hard_guard": self.rejected_due_to_hard_guard,
                "rejected_due_to_mu_only": self.rejected_mu_only,
                "rejected_due_to_pde_guard": self.rejected_hard_guard_pde,
                "rejected_due_to_ch_guard": self.rejected_hard_guard_ch,
                "rejected_due_to_mass_guard": self.rejected_hard_guard_mass,
                "rejected_due_to_phase_guard": self.rejected_hard_guard_phase,
                "rejected_due_to_sparse_u_guard": self.rejected_hard_guard_sparse_u,
                "mean_primary_reward_accepted": _safe_mean(
                    self._accepted_primary_rewards
                ),
                "mean_primary_reward_rejected": _safe_mean(
                    self._rejected_primary_rewards
                ),
                "mean_guard_penalty_accepted": _safe_mean(
                    self._accepted_guard_penalties
                ),
                "mean_guard_penalty_rejected": _safe_mean(
                    self._rejected_guard_penalties
                ),
                "inactivity_recovery_activations": self.inactivity_recovery_activations,
                "cahn_hilliard_pareto_policy_enabled": bool(
                    self.pareto_policy_config.get("enabled", False)
                ),
                "cahn_hilliard_u_first_policy_enabled": self.u_first_policy_enabled,
                "cahn_hilliard_mu_support_only": self.mu_support_only,
                "final_trust_radius": (
                    float(self.controller.trust_radius)
                    if self.controller is not None
                    else float("nan")
                ),
            }
        )
        self._save_artifacts(metrics)
        return metrics

    def _schedule(self) -> dict[str, int]:
        controller_cfg = dict(self.config.get("controller_v2", {}))
        training_cfg = dict(self.config.get("training", {}))
        total = int(controller_cfg.get("total_steps", training_cfg.get("total_steps", 1000)))
        warmup = int(controller_cfg.get("warmup_steps", max(1, total // 4)))
        blocks = int(controller_cfg.get("control_blocks", 3))
        block_steps = int(controller_cfg.get("block_steps", (total - warmup) // max(blocks, 1)))
        probe_steps = int(controller_cfg.get("probe_steps", min(25, max(1, block_steps // 10))))
        if warmup + blocks * block_steps != total:
            raise ValueError(
                "Cahn--Hilliard schedule must satisfy warmup + blocks * block_steps = total_steps."
            )
        if not 0 < probe_steps < block_steps:
            raise ValueError("probe_steps must be positive and smaller than block_steps.")
        return {
            "total_steps": total,
            "warmup_steps": warmup,
            "control_blocks": blocks,
            "block_steps": block_steps,
            "probe_steps": probe_steps,
        }

    def _run_vanilla(self, schedule: dict[str, int]) -> None:
        self._commit_rows(
            self._train_steps(
                self._training_batch(adaptive=False),
                schedule["warmup_steps"],
                phase="warmup",
            )
        )
        for block in range(schedule["control_blocks"]):
            self._commit_rows(
                self._train_steps(
                    self._training_batch(adaptive=False),
                    schedule["block_steps"],
                    phase=f"block_{block}",
                )
            )

    def _run_vara(self, schedule: dict[str, int]) -> None:
        assert self.controller is not None
        self._commit_rows(
            self._train_steps(
                self._training_batch(adaptive=False),
                schedule["warmup_steps"],
                phase="warmup",
            )
        )
        self._log_allocation(-1)
        for block in range(schedule["control_blocks"]):
            before = self._diagnose()
            _adaptation_raw, adaptation_normalized = before.adaptation_scores()
            before_metrics = self._controller_metrics(before)
            self._record_pareto_metric_history(before_metrics)
            self.controller.update_history(
                before.adaptation_names, adaptation_normalized, before_metrics
            )
            weak_regions = self.detector.detect(
                adaptation_normalized,
                before.adaptation_names,
                self.patch_grid,
            )
            candidates = self.controller.candidates(weak_regions)
            self._route_candidate_losses(candidates)
            candidates = self._filter_ablation_candidates(candidates)
            influence = self._candidate_influence(candidates)
            ranked = self.controller.rank(candidates, influence)
            ranked = self._rank_u_first_candidates(ranked)
            prefiltered = [candidate for candidate in ranked if candidate.prefiltered]
            active = [candidate for candidate in ranked if not candidate.prefiltered]
            self.prefiltered_interventions += len(prefiltered)
            for candidate in prefiltered:
                decision = self.controller.record_prefilter(candidate, update_trust=False)
                self._record_decision(block, candidate, decision)
            if not active:
                self._commit_rows(
                    self._train_steps(
                        self._training_batch(adaptive=True),
                        schedule["block_steps"],
                        phase=f"block_{block}_no_action",
                    )
                )
                self._register_block_outcome(False)
                self._log_allocation(block)
                continue
            candidate = active[0]
            if self.controller.config.counterfactual_probe_enabled:
                block_accepted = self._counterfactual_block(
                    block, candidate, before_metrics, schedule
                )
            else:
                block_accepted = self._unguarded_block(
                    block, candidate, before, before_metrics, schedule
                )
            self._register_block_outcome(block_accepted)
            self._log_allocation(block)

    def _counterfactual_block(
        self,
        block: int,
        candidate: V2Candidate,
        pre_block_metrics: dict[str, float],
        schedule: dict[str, int],
    ) -> bool:
        assert self.controller is not None
        model_before = self._model_snapshot()
        optimizer_before = deepcopy(self.optimizer.state_dict())
        allocation_before = self.controller.state.snapshot()
        rng_before = deepcopy(self.sampling_rng.bit_generator.state)
        trust_before = float(self.controller.trust_radius)
        effectiveness_before = deepcopy(self.controller.effectiveness)
        probe_steps = schedule["probe_steps"]

        neutral_batch = self._training_batch(adaptive=True)
        rng_after_neutral = deepcopy(self.sampling_rng.bit_generator.state)
        neutral_rows = self._train_steps(
            neutral_batch,
            probe_steps,
            phase=f"block_{block}_neutral_probe",
        )
        neutral_model = self._model_snapshot()
        neutral_optimizer = deepcopy(self.optimizer.state_dict())
        neutral_snapshot = self._diagnose()
        neutral_metrics = self._controller_metrics(neutral_snapshot)

        self._restore_model(model_before)
        self.optimizer.load_state_dict(optimizer_before)
        self.controller.state.restore(allocation_before)
        self.sampling_rng.bit_generator.state = deepcopy(rng_before)
        self.controller.apply(candidate)
        self._enforce_mu_support_caps()
        candidate_batch = self._training_batch(adaptive=True)
        candidate_rows = self._train_steps(
            candidate_batch,
            probe_steps,
            phase=f"block_{block}_candidate_probe",
        )
        candidate_snapshot = self._diagnose()
        candidate_metrics = self._controller_metrics(candidate_snapshot)
        controller_cfg = dict(self.config.get("controller_v2", {}))
        accepted, decision = self._evaluate_with_u_guards(
            candidate,
            self._candidate_score(candidate, neutral_snapshot),
            self._candidate_score(candidate, candidate_snapshot),
            neutral_metrics,
            candidate_metrics,
            target_threshold=float(controller_cfg.get("counterfactual_target_margin", 0.005)),
            guard_threshold=float(controller_cfg.get("counterfactual_guard_margin", 0.02)),
            comparison_mode="counterfactual",
        )
        if accepted:
            kept_batch, kept_rows = candidate_batch, candidate_rows
        else:
            self.rejected_interventions += 1
            if self.rollback_enabled:
                self.rollback_count += 1
                self._restore_model(neutral_model)
                self.optimizer.load_state_dict(neutral_optimizer)
                self.controller.state.restore(allocation_before)
                self.sampling_rng.bit_generator.state = deepcopy(rng_after_neutral)
                kept_batch, kept_rows = neutral_batch, neutral_rows
            else:
                kept_batch, kept_rows = candidate_batch, candidate_rows
        self._commit_rows(kept_rows)
        continuation_start = self.applied_optimizer_steps + 1
        self._commit_rows(
            self._train_steps(
                kept_batch,
                schedule["block_steps"] - probe_steps,
                phase=f"block_{block}_continuation",
            )
        )
        if accepted and bool(
            self.post_block_guard_config.get("enabled", False)
        ):
            post_metrics = self._controller_metrics(self._diagnose())
            post_ok, post_reason, post_changes = self._post_block_guard_decision(
                pre_block_metrics, post_metrics
            )
            decision["post_block_guard_changes"] = post_changes
            decision["post_block_rollback"] = not post_ok
            if not post_ok:
                self._restore_model(model_before)
                self.optimizer.load_state_dict(optimizer_before)
                self.controller.state.restore(allocation_before)
                self.sampling_rng.bit_generator.state = deepcopy(rng_before)
                self.controller.trust_radius = max(
                    self.controller.config.trust_radius_min,
                    trust_before * self.controller.config.trust_shrink,
                )
                self.controller.effectiveness = effectiveness_before
                self.post_block_rollbacks += 1
                self.rollback_count += 1
                self.rejected_interventions += 1
                decision["probe_accepted"] = True
                decision["accepted"] = False
                decision["rollback_reason"] = post_reason
                decision["rejection_reason"] = post_reason
                for row in self.loss_rows:
                    if int(row.get("step", 0)) >= continuation_start:
                        row["post_block_rolled_back"] = True
                if self.controller.decisions:
                    self.controller.decisions[-1].update(decision)
                accepted = False
        if accepted:
            self.accepted_interventions += 1
            self._count_accepted_channel(candidate)
        self._record_decision(block, candidate, decision)
        self._record_acceptance_health(accepted, decision)
        return accepted

    def _unguarded_block(
        self,
        block: int,
        candidate: V2Candidate,
        before: DiagnosticSnapshot,
        before_metrics: dict[str, float],
        schedule: dict[str, int],
    ) -> bool:
        assert self.controller is not None
        self.controller.apply(candidate)
        self._enforce_mu_support_caps()
        self._commit_rows(
            self._train_steps(
                self._training_batch(adaptive=True),
                schedule["block_steps"],
                phase=f"block_{block}_unguarded",
            )
        )
        after = self._diagnose()
        accepted, decision = self._evaluate_with_u_guards(
            candidate,
            self._candidate_score(candidate, before),
            self._candidate_score(candidate, after),
            before_metrics,
            self._controller_metrics(after),
            comparison_mode="temporal_no_rollback",
        )
        if accepted:
            self.accepted_interventions += 1
            self._count_accepted_channel(candidate)
        else:
            self.rejected_interventions += 1
        self._record_decision(block, candidate, decision)
        self._record_acceptance_health(accepted, decision)
        return accepted

    def _train_steps(
        self,
        batch: dict[str, torch.Tensor],
        steps: int,
        *,
        phase: str,
    ) -> list[dict[str, Any]]:
        rows = []
        allocation = self.controller.state if self.controller is not None else None
        for local_step in range(int(steps)):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            result = compute_training_loss(
                self.model,
                self.benchmark,
                batch,
                self.weights,
                self.patch_grid,
                allocation,
            )
            self.objective_evaluations += 1
            if not torch.isfinite(result.total):
                raise FloatingPointError("Non-finite Cahn--Hilliard training loss.")
            result.total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(self.config.get("training", {}).get("gradient_clip", 10.0)),
            )
            self.optimizer.step()
            self.optimizer_step_calls += 1
            rows.append(
                {
                    "local_step": local_step + 1,
                    "phase": phase,
                    "loss_total": float(result.total.detach().cpu()),
                    **{
                        f"loss_{name}": float(value.detach().cpu())
                        for name, value in result.components.items()
                    },
                }
            )
        return rows

    def _commit_rows(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.applied_optimizer_steps += 1
            self.loss_rows.append({**row, "step": self.applied_optimizer_steps})

    def _candidate_influence(
        self, candidates: list[V2Candidate]
    ) -> dict[str, dict[str, float]]:
        if self.controller is None or not self.controller.config.gradient_prefilter_enabled:
            return {
                candidate.key(): {
                    "gradient_compatibility": 0.0,
                    "gradient_conflict": 0.0,
                }
                for candidate in candidates
            }
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        influence = {}
        for candidate in candidates:
            self.model.zero_grad(set_to_none=True)
            result = compute_training_loss(
                self.model,
                self.benchmark,
                self.diagnostic_batch,
                self.weights,
                self.patch_grid,
                self.controller.state,
            )
            self.objective_evaluations += 1
            target = self._candidate_target_tensor(candidate, result)
            guard = sum(result.components.values())
            target_gradient = _flat_gradient(target, parameters, retain_graph=True)
            guard_gradient = _flat_gradient(guard, parameters, retain_graph=False)
            cosine = _cosine(target_gradient, guard_gradient)
            influence[candidate.key()] = {
                "gradient_compatibility": max(0.0, cosine),
                "gradient_conflict": max(0.0, -cosine),
            }
        self.model.zero_grad(set_to_none=True)
        return influence

    def _candidate_target_tensor(
        self, candidate: V2Candidate, result: LossResult
    ) -> torch.Tensor:
        target_name = self._candidate_target_name(candidate.variable)
        channel = result.channels.get(
            target_name, result.channels.get("pde_residual")
        )
        if channel is None:
            return result.total
        values, coordinates = channel
        patch_ids = self.patch_grid.assign_torch(coordinates.detach())
        selected = values[patch_ids == candidate.patch_id]
        return selected.mean() if selected.numel() else values.mean()

    def _diagnose(self) -> DiagnosticSnapshot:
        self.diagnostic_evaluations += 1
        snapshot = build_diagnostic_snapshot(
            self.model,
            self.benchmark,
            self.patch_grid,
            self.diagnostic_batch,
            percentile=self.diagnostic_percentile,
            variable_awareness=self.variable_awareness,
            interface_tau=self.interface_tau,
            interface_focus_strength=self.interface_focus_strength,
            interface_threshold=self.interface_threshold,
            priority_weights=self.diagnostic_priority_weights,
            mass_baseline=self.mass_proxy_baseline,
        )
        if self.mu_support_only:
            for name in snapshot.names:
                if self._is_mu_channel(name):
                    index = snapshot.names.index(name)
                    configured = min(
                        self.mu_priority_max,
                        float(self.diagnostic_priority_weights.get(name, 1.0)),
                    )
                    snapshot.priority_scores[index] = (
                        snapshot.normalized_scores[index] * configured
                    )
        if float(self.weights.get("sparse_mu_mse", 0.0)) <= 0.0:
            snapshot.adaptation_names = [
                name
                for name in snapshot.adaptation_names
                if name != "sparse_mu_mismatch"
            ]
        return snapshot

    @staticmethod
    def _controller_metrics(snapshot: DiagnosticSnapshot) -> dict[str, float]:
        values = dict(snapshot.scalar_metrics)
        pde = values.get("pde_residual_mean", 0.0)
        boundary = values.get("bc_u_violation", 0.0)
        initial = values.get("ic_u_violation", 0.0)
        sparse = values.get("sparse_u_mse", 0.0)
        phase = values.get("phase_range_violation", 0.0)
        return {
            "pde_residual_mean": pde,
            "boundary_condition_error": boundary,
            "unweighted_physics_validation_loss": pde + boundary + initial + phase,
            "unweighted_validation_loss": pde + boundary + initial + sparse + phase,
            **values,
        }

    @classmethod
    def _candidate_score(
        cls, candidate: V2Candidate, snapshot: DiagnosticSnapshot
    ) -> float:
        name = cls._candidate_target_name(candidate.variable)
        if name not in snapshot.names:
            name = "pde_residual"
        return float(snapshot.raw_scores[snapshot.names.index(name), candidate.patch_id])

    def _route_candidate_losses(self, candidates: list[V2Candidate]) -> None:
        routes = {
            "ch_residual": ["ch_residual"],
            "chemical_potential_residual": ["chemical_potential_residual"],
            "pde_residual": ["ch_residual", "chemical_potential_residual"],
            "boundary_violation": ["bc_u", "bc_mu"],
            "bc_u_violation": ["bc_u"],
            "bc_mu_violation": ["bc_mu"],
            "initial_condition_violation": ["ic_u", "ic_mu"],
            "ic_u_violation": ["ic_u"],
            "ic_mu_violation": ["ic_mu"],
            "sparse_u_mismatch": ["sparse_u_mse"],
            "sparse_mu_mismatch": ["sparse_mu_mse"],
            "sparse_data_mismatch": ["sparse_u_mse", "sparse_mu_mse"],
            "predicted_interface_proxy": ["ch_residual", "sparse_u_mse"],
            "predicted_interface_mask": ["ch_residual", "sparse_u_mse"],
            "predicted_gradient_norm": ["ch_residual", "sparse_u_mse"],
            "phase_range_violation": ["phase_range_penalty"],
            "mass_proxy_violation": ["ch_residual", "sparse_u_mse"],
        }
        for candidate in candidates:
            candidate.loss_names = routes.get(
                candidate.variable, ["ch_residual", "chemical_potential_residual"]
            )

    @staticmethod
    def _candidate_target_name(variable: str) -> str:
        if variable in {
            "predicted_interface_proxy",
            "predicted_interface_mask",
            "predicted_gradient_norm",
        }:
            return "ch_residual"
        return variable

    @staticmethod
    def _is_mu_channel(name: str) -> bool:
        return name in {
            "chemical_potential_residual",
            "sparse_mu_mismatch",
            "bc_mu_violation",
            "ic_mu_violation",
            "bc_mu",
            "ic_mu",
            "sparse_mu_mse",
        }

    def _is_mu_only_candidate(self, candidate: V2Candidate) -> bool:
        return self._is_mu_channel(candidate.variable) or (
            bool(candidate.loss_names)
            and all(self._is_mu_channel(name) for name in candidate.loss_names)
        )

    def _rank_u_first_candidates(
        self, candidates: list[V2Candidate]
    ) -> list[V2Candidate]:
        """Place u/interface candidates before auxiliary mu-only actions."""
        if not self.u_first_policy_enabled:
            return candidates
        interface_channels = {
            "predicted_interface_proxy",
            "predicted_interface_mask",
            "predicted_gradient_norm",
        }

        def order(
            candidate: V2Candidate,
        ) -> tuple[float, float, float, float, float, float]:
            mu_only = 1.0 if self._is_mu_only_candidate(candidate) else 0.0
            priority = float(
                self.diagnostic_priority_weights.get(candidate.variable, 1.0)
            )
            prior_damage = self.candidate_guard_memory.get(candidate.key(), 0.0)
            predicted_safety = 1.0 / (1.0 + max(0.0, prior_damage))
            effectiveness = (
                float(self.controller.effectiveness.get(candidate.key(), 1.0))
                if self.controller is not None
                else 1.0
            )
            sampling_preference = (
                1.0
                if candidate.variable in interface_channels
                and candidate.action_type == "sampling"
                else 0.0
            )
            return (
                mu_only,
                -priority,
                -predicted_safety,
                -effectiveness,
                -sampling_preference,
                -float(candidate.rank_score),
            )

        ranked = sorted(candidates, key=order)
        mu_seen = 0
        limited = []
        for candidate in ranked:
            if self._is_mu_only_candidate(candidate):
                if mu_seen >= max(0, self.max_mu_candidates_per_block):
                    continue
                mu_seen += 1
            limited.append(candidate)
        return limited

    def _record_pareto_metric_history(self, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            if not np.isfinite(float(value)):
                continue
            history = self.pareto_metric_history.setdefault(name, [])
            history.append(float(value))
            del history[:-8]

    def _observed_metric_noise(self, metric: str) -> float:
        history = self.pareto_metric_history.get(metric, [])
        if len(history) < 3:
            return 0.0
        values = np.asarray(history, dtype=float)
        denominator_floor = float(
            self.pareto_policy_config.get(
                "small_metric_denominator_floor", 1e-6
            )
        )
        changes = np.diff(values) / np.maximum(
            np.abs(values[:-1]), denominator_floor
        )
        median = float(np.median(changes))
        mad = float(np.median(np.abs(changes - median)))
        return max(0.0, 1.4826 * mad)

    def _effective_guard_tolerance(
        self, metric: str, configured_tolerance: float
    ) -> float:
        if not bool(
            self.pareto_policy_config.get("adaptive_guard_tolerance", True)
        ):
            return float(configured_tolerance)
        noise_multiplier = float(
            self.pareto_policy_config.get("noise_floor_multiplier", 1.5)
        )
        return max(
            float(configured_tolerance),
            noise_multiplier * self._observed_metric_noise(metric),
            float(self.pareto_policy_config.get("absolute_guard_floor", 1e-5)),
        )

    def _guard_worsening_ratio(
        self,
        metric: str,
        before: float | None,
        after: float | None,
    ) -> float:
        if before is None or after is None:
            return float("nan")
        delta = float(after) - float(before)
        absolute_floor = float(
            self.pareto_policy_config.get("absolute_guard_floor", 1e-5)
        )
        if delta <= absolute_floor:
            return 0.0
        denominator_floor = float(
            self.pareto_policy_config.get(
                "small_metric_denominator_floor", 1e-6
            )
        )
        return delta / max(abs(float(before)), denominator_floor)

    def _register_block_outcome(self, accepted: bool) -> None:
        if accepted:
            self.consecutive_all_rejected_blocks = 0
            self._temporary_reward_multiplier = 1.0
            return
        self.all_rejected_blocks += 1
        self.consecutive_all_rejected_blocks += 1
        self.consecutive_all_rejected_blocks_max = max(
            self.consecutive_all_rejected_blocks_max,
            self.consecutive_all_rejected_blocks,
        )
        if not bool(self.inactivity_recovery_config.get("enabled", True)):
            return
        trigger = int(
            self.inactivity_recovery_config.get(
                "consecutive_rejected_blocks_trigger", 2
            )
        )
        if self.consecutive_all_rejected_blocks < trigger:
            return
        if self.controller is not None:
            factor = float(
                self.inactivity_recovery_config.get(
                    "trust_radius_recovery_factor", 1.10
                )
            )
            maximum = float(
                self.inactivity_recovery_config.get(
                    "max_recovery_trust_radius", 0.12
                )
            )
            self.controller.trust_radius = min(
                maximum,
                max(
                    self.controller.config.trust_radius_min,
                    self.controller.trust_radius * factor,
                ),
            )
        if bool(
            self.inactivity_recovery_config.get(
                "lower_primary_reward_temporarily", True
            )
        ):
            self._temporary_reward_multiplier = float(
                self.inactivity_recovery_config.get(
                    "temporary_reward_multiplier", 0.5
                )
            )
        self.inactivity_recovery_activations += 1

    def _record_acceptance_health(
        self, accepted: bool, decision: dict[str, Any]
    ) -> None:
        primary_reward = float(decision.get("primary_reward", 0.0))
        guard_penalty = float(decision.get("guard_penalty", 0.0))
        key = str(decision.get("candidate_id", ""))
        previous_damage = self.candidate_guard_memory.get(key, 0.0)
        self.candidate_guard_memory[key] = (
            0.8 * previous_damage + 0.2 * max(0.0, guard_penalty)
        )
        if accepted:
            self._accepted_primary_rewards.append(primary_reward)
            self._accepted_guard_penalties.append(guard_penalty)
            if decision.get("acceptance_mode") == "best_safe_fallback":
                self.accepted_by_best_safe_fallback += 1
            elif bool(self.pareto_policy_config.get("enabled", False)):
                self.accepted_by_standard_pareto += 1
        else:
            self._rejected_primary_rewards.append(primary_reward)
            self._rejected_guard_penalties.append(guard_penalty)
            reason = str(decision.get("rejection_reason", ""))
            if reason.startswith("hard_guard_"):
                self.rejected_due_to_hard_guard += 1
            if reason in {"insufficient_primary_reward", "pareto_margin"}:
                self.rejected_due_to_primary_reward += 1

    def _enforce_mu_support_caps(self) -> None:
        """Preserve mean-one multiplier mass while capping auxiliary mu losses."""
        if not self.mu_support_only or self.controller is None:
            return
        for loss_name, values in list(
            self.controller.state.loss_multipliers.items()
        ):
            if self._is_mu_channel(loss_name):
                self.controller.state.loss_multipliers[loss_name] = (
                    _cap_mean_one_multiplier(values, self.mu_multiplier_max)
                )
        self.controller.validate_state()

    def _evaluate_with_u_guards(
        self,
        candidate: V2Candidate,
        before_target: float,
        after_target: float,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
        *,
        target_threshold: float | None = None,
        guard_threshold: float | None = None,
        comparison_mode: str,
    ) -> tuple[bool, dict[str, Any]]:
        assert self.controller is not None
        base_guard_names = set(self.controller.config.guard_metrics)
        base_before_metrics = {
            name: value
            for name, value in before_metrics.items()
            if name in base_guard_names
        }
        base_after_metrics = {
            name: value
            for name, value in after_metrics.items()
            if name in base_guard_names
        }
        base_accepted, decision = self.controller.evaluate(
            candidate,
            before_target,
            after_target,
            base_before_metrics,
            base_after_metrics,
            target_threshold=target_threshold,
            guard_threshold=guard_threshold,
            comparison_mode=comparison_mode,
            update_state=False,
        )
        pareto_enabled = bool(self.pareto_policy_config.get("enabled", False))
        if pareto_enabled:
            accepted, guard_reason, pareto_details = self._pareto_decision(
                candidate, before_metrics, after_metrics
            )
            decision.update(pareto_details)
            if accepted:
                decision["acceptance_mode"] = "standard_pareto"
            else:
                fallback_accepted, fallback_reason = (
                    self._best_safe_candidate_decision(candidate, decision)
                )
                if fallback_accepted:
                    accepted = True
                    guard_reason = ""
                    decision["acceptance_mode"] = "best_safe_fallback"
                    decision["standard_pareto_rejection_reason"] = fallback_reason
                    decision["rejection_reason"] = ""
                    self.best_safe_candidate_activations += 1
            if accepted:
                pareto_score = float(decision.get("pareto_score", 0.0))
                predicted = max(
                    float(candidate.predicted_target_improvement),
                    self.controller.config.noise_floor,
                )
                decision["observed_target_improvement"] = pareto_score
                decision["reward_ratio"] = pareto_score / predicted
        else:
            u_guard_ok, guard_reason, u_changes = self._u_guard_decision(
                before_metrics, after_metrics
            )
            allow_mu_override = bool(
                self.u_guard_config.get(
                    "allow_mu_improvement_to_override_u_damage", False
                )
            )
            if (
                not u_guard_ok
                and allow_mu_override
                and self._is_mu_only_candidate(candidate)
            ):
                u_guard_ok = True
                guard_reason = ""
            accepted = bool(base_accepted and u_guard_ok)
            decision["u_guard_changes"] = u_changes
            decision["guard_metric_changes"] = u_changes
        decision["accepted"] = accepted
        decision["candidate_id"] = candidate.key()
        decision["channel"] = candidate.variable
        decision["target_metric"] = self._candidate_target_name(candidate.variable)
        decision["selected_primary_channel"] = candidate.variable
        decision["selected_patch_id"] = candidate.patch_id
        decision["selected_action_type"] = candidate.action_type
        for metric_name in (
            "sparse_u_mse",
            "sparse_mu_mse",
            "interface_proxy_error",
            "interface_proxy_mean",
            "pde_residual_mean",
            "ch_residual_mean",
            "mass_proxy_error",
            "phase_overshoot",
            "ic_u_error",
            "bc_u_error",
            "mu_residual_mean",
        ):
            decision[f"{metric_name}_change"] = self._guard_worsening_ratio(
                metric_name,
                before_metrics.get(metric_name), after_metrics.get(metric_name)
            )
        decision["sparse_u_change"] = decision["sparse_u_mse_change"]
        decision["sparse_mu_change"] = decision["sparse_mu_mse_change"]
        decision["interface_proxy_change"] = decision[
            "interface_proxy_error_change"
        ]
        decision["pde_residual_change"] = decision["pde_residual_mean_change"]
        decision["ch_residual_change"] = decision["ch_residual_mean_change"]
        decision["mass_proxy_change"] = decision["mass_proxy_error_change"]
        decision["ic_u_change"] = decision["ic_u_error_change"]
        decision["bc_u_change"] = decision["bc_u_error_change"]
        decision["mu_residual_change"] = decision["mu_residual_mean_change"]
        log_metrics = {
            "sparse_u": "sparse_u_mse",
            "pde": "pde_residual_mean",
            "ch": "ch_residual_mean",
            "mass_proxy": "mass_proxy_error",
            "phase_overshoot": "phase_overshoot",
            "mu_residual": "mu_residual_mean",
        }
        for label, metric in log_metrics.items():
            decision[f"{label}_before"] = before_metrics.get(metric, float("nan"))
            decision[f"{label}_after"] = after_metrics.get(metric, float("nan"))
        if not accepted and guard_reason:
            decision["rollback_reason"] = guard_reason
            decision["rejection_reason"] = guard_reason
            if pareto_enabled:
                self._count_pareto_rejection(guard_reason)
            elif base_accepted:
                decision["u_guard_rejection"] = guard_reason
                self._count_guard_rejection(guard_reason)
        committed = self.controller.commit_evaluation(
            candidate, accepted, decision
        )
        return accepted, committed

    def _pareto_decision(
        self,
        candidate: V2Candidate,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
    ) -> tuple[bool, str, dict[str, Any]]:
        """Score primary improvements subject to hard physics/conservation guards."""
        primary_targets = list(
            self.pareto_policy_config.get(
                "primary_targets",
                [
                    "sparse_u_mse",
                    "interface_proxy_error",
                    "ic_u_error",
                    "bc_u_error",
                    "ch_residual_mean",
                ],
            )
        )
        score_weights = dict(self.pareto_score_config.get("weights", {}))
        primary_changes = {
            metric: _improvement_ratio(
                before_metrics.get(metric), after_metrics.get(metric)
            )
            for metric in primary_targets
        }
        primary_reward = float(
            sum(
                float(score_weights.get(metric, 1.0))
                * max(0.0, change)
                for metric, change in primary_changes.items()
                if np.isfinite(change)
            )
        )
        hard_guards = dict(self.pareto_policy_config.get("hard_guards", {}))
        guard_changes: dict[str, float] = {}
        effective_tolerances: dict[str, float] = {}
        violated_guard_names: list[str] = []
        reason_names = {
            "pde_residual_mean": "hard_guard_pde",
            "ch_residual_mean": "hard_guard_ch",
            "mass_proxy_error": "hard_guard_mass",
            "phase_overshoot": "hard_guard_phase",
            "sparse_u_mse": "hard_guard_sparse_u",
            "ic_u_error": "hard_guard_ic_u",
            "bc_u_error": "hard_guard_bc_u",
        }
        for metric, configured_threshold in hard_guards.items():
            change = self._guard_worsening_ratio(
                metric,
                before_metrics.get(metric),
                after_metrics.get(metric),
            )
            guard_changes[metric] = change
            threshold = float(configured_threshold)
            if metric == "mass_proxy_error" and bool(
                self.mass_guard_config.get("enabled", True)
            ):
                threshold = min(
                    threshold,
                    float(
                        self.mass_guard_config.get(
                            "max_relative_worsening", threshold
                        )
                    ),
                )
            if metric == "phase_overshoot" and bool(
                self.phase_guard_config.get("enabled", True)
            ):
                threshold = min(
                    threshold,
                    float(
                        self.phase_guard_config.get(
                            "max_relative_worsening", threshold
                        )
                    ),
                )
                absolute_ceiling = float(
                    self.phase_guard_config.get("absolute_ceiling", float("inf"))
                )
                if float(after_metrics.get(metric, 0.0)) > absolute_ceiling:
                    violated_guard_names.append(metric)
            threshold = self._effective_guard_tolerance(metric, threshold)
            effective_tolerances[metric] = threshold
            if np.isfinite(change) and change > threshold:
                violated_guard_names.append(metric)

        hard_violation = ""
        if violated_guard_names:
            first_violation = violated_guard_names[0]
            hard_violation = reason_names.get(
                first_violation, f"hard_guard_{first_violation}"
            )

        guard_weight_names = {
            "pde_residual_mean": "pde_residual_mean_guard",
            "mass_proxy_error": "mass_proxy_guard",
            "phase_overshoot": "phase_overshoot_guard",
        }
        safe_band = dict(
            self.pareto_policy_config.get("safe_worsening_band", {})
        )
        mild_fraction = float(safe_band.get("mild_guard_fraction", 0.5))
        mild_metrics = [
            metric
            for metric, change in guard_changes.items()
            if np.isfinite(change)
            and change > 0.0
            and change
            <= mild_fraction * effective_tolerances.get(metric, 0.0)
        ]
        exclude_mild = bool(safe_band.get("enabled", True)) and len(
            mild_metrics
        ) <= int(safe_band.get("max_mild_guard_violations", 2))
        guard_penalty = float(
            sum(
                float(
                    score_weights.get(
                        guard_weight_names.get(metric, f"{metric}_guard"), 1.0
                    )
                )
                * (
                    0.0
                    if exclude_mild and metric in mild_metrics
                    else max(0.0, change)
                )
                for metric, change in guard_changes.items()
                if np.isfinite(change)
            )
        )
        pareto_score = primary_reward - guard_penalty
        soft_tolerances = dict(
            self.pareto_policy_config.get("soft_guards", {})
        )
        soft_changes = {
            metric: self._guard_worsening_ratio(
                metric,
                before_metrics.get(metric),
                after_metrics.get(metric),
            )
            for metric in soft_tolerances
        }
        minimum_reward = self._temporary_reward_multiplier * float(
            self.pareto_score_config.get("min_primary_reward", 0.005)
        )
        acceptance_margin = float(
            self.pareto_score_config.get("acceptance_margin", 0.0)
        )
        require_primary = bool(
            self.pareto_policy_config.get("require_primary_improvement", True)
        )
        mu_improvement = _improvement_ratio(
            before_metrics.get("mu_residual_mean"),
            after_metrics.get("mu_residual_mean"),
        )
        reason = hard_violation
        if not reason and require_primary and primary_reward <= minimum_reward:
            if (
                bool(
                    self.pareto_policy_config.get(
                        "reject_if_mu_only_improves", True
                    )
                )
                and np.isfinite(mu_improvement)
                and mu_improvement > 0.0
            ):
                reason = "mu_only"
            else:
                reason = "insufficient_primary_reward"
        if not reason and pareto_score <= acceptance_margin:
            reason = "pareto_margin"
        details: dict[str, Any] = {
            "primary_reward": primary_reward,
            "guard_penalty": guard_penalty,
            "pareto_score": pareto_score,
            "primary_metric_changes": primary_changes,
            "hard_guard_changes": guard_changes,
            "soft_guard_changes": soft_changes,
            "effective_guard_tolerances": effective_tolerances,
            "violated_guard_names": sorted(set(violated_guard_names)),
            "mild_guard_names": mild_metrics,
            "mild_guard_penalty_waived": exclude_mild,
            "min_required_primary_reward": minimum_reward,
            "rejection_reason": reason,
            "pareto_policy_enabled": True,
        }
        return not bool(reason), reason, details

    def _best_safe_candidate_decision(
        self,
        candidate: V2Candidate,
        decision: dict[str, Any],
    ) -> tuple[bool, str]:
        """Allow a low-reward but demonstrably safe candidate after inactivity."""
        reason = str(decision.get("rejection_reason", ""))
        config = self.best_safe_candidate_config
        if not bool(config.get("enabled", True)):
            return False, reason
        trigger = int(
            config.get("activate_after_consecutive_all_rejected_blocks", 2)
        )
        if self.consecutive_all_rejected_blocks < trigger:
            return False, reason
        if bool(config.get("reject_mu_only", True)) and self._is_mu_only_candidate(
            candidate
        ):
            return False, reason
        if float(decision.get("primary_reward", 0.0)) < float(
            config.get("min_primary_reward", 0.0005)
        ):
            return False, reason
        guard_changes = dict(decision.get("hard_guard_changes", {}))
        for metric, threshold in dict(
            config.get("max_guard_worsening", {})
        ).items():
            change = float(guard_changes.get(metric, float("nan")))
            if not np.isfinite(change) or change > float(threshold):
                return False, reason
        return True, reason

    def _u_guard_decision(
        self,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
    ) -> tuple[bool, str, dict[str, float]]:
        """Apply Cahn--Hilliard primary-field Pareto guards."""
        checks = [
            (
                "sparse_u_mse",
                "max_sparse_u_damage_ratio",
                0.05,
                "sparse_u_guard",
                False,
            ),
            (
                "ic_u_violation",
                "max_ic_u_damage_ratio",
                0.05,
                "ic_u_guard",
                False,
            ),
            (
                "bc_u_violation",
                "max_bc_u_damage_ratio",
                0.05,
                "bc_u_guard",
                False,
            ),
            (
                "phase_range_violation",
                "max_phase_range_damage_ratio",
                0.10,
                "phase_range_guard",
                False,
            ),
            (
                "mass_proxy_violation",
                "max_mass_proxy_damage_ratio",
                0.10,
                "mass_guard",
                False,
            ),
            (
                "interface_proxy_mean",
                "max_interface_proxy_change_ratio",
                0.10,
                "interface_proxy_guard",
                True,
            ),
        ]
        changes: dict[str, float] = {}
        for metric, config_name, default, reason, absolute in checks:
            before = before_metrics.get(metric)
            after = after_metrics.get(metric)
            if before is None or after is None:
                continue
            change = _relative_change(before, after, absolute=absolute)
            changes[metric] = change
            threshold = float(self.u_guard_config.get(config_name, default))
            if np.isfinite(change) and change > threshold:
                return False, reason, changes
        return True, "", changes

    def _post_block_guard_decision(
        self,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
    ) -> tuple[bool, str, dict[str, float]]:
        """Detect delayed drift after a probe-safe intervention."""
        checks = [
            (
                "pde_residual_mean",
                "pde_residual_max_worsening",
                0.075,
                "post_block_guard_pde",
            ),
            (
                "ch_residual_mean",
                "ch_residual_max_worsening",
                0.075,
                "post_block_guard_ch",
            ),
            (
                "mass_proxy_error",
                "mass_proxy_max_worsening",
                0.10,
                "post_block_guard_mass",
            ),
            (
                "phase_overshoot",
                "phase_overshoot_max_worsening",
                0.075,
                "post_block_guard_phase",
            ),
            (
                "sparse_u_mse",
                "sparse_u_max_worsening",
                0.075,
                "post_block_guard_sparse_u",
            ),
        ]
        changes: dict[str, float] = {}
        for metric, config_name, default, reason in checks:
            change = self._guard_worsening_ratio(
                metric,
                before_metrics.get(metric),
                after_metrics.get(metric),
            )
            changes[metric] = change
            threshold = float(
                self.post_block_guard_config.get(config_name, default)
            )
            threshold = self._effective_guard_tolerance(metric, threshold)
            if np.isfinite(change) and change > threshold:
                return False, reason, changes
        return True, "", changes

    def _count_accepted_channel(self, candidate: V2Candidate) -> None:
        if bool(self.pareto_policy_config.get("enabled", False)):
            self.accepted_pareto_safe_interventions += 1
            if candidate.variable in {
                "predicted_interface_proxy",
                "predicted_interface_mask",
                "predicted_gradient_norm",
                "ch_residual",
            }:
                self.accepted_interface_targets += 1
            if candidate.variable == "sparse_u_mismatch":
                self.accepted_sparse_u_targets += 1
        if self._is_mu_only_candidate(candidate):
            self.accepted_mu_interventions += 1
        else:
            self.accepted_u_interface_interventions += 1

    def _count_pareto_rejection(self, reason: str) -> None:
        attribute = {
            "hard_guard_pde": "rejected_hard_guard_pde",
            "hard_guard_ch": "rejected_hard_guard_ch",
            "hard_guard_mass": "rejected_hard_guard_mass",
            "hard_guard_phase": "rejected_hard_guard_phase",
            "hard_guard_sparse_u": "rejected_hard_guard_sparse_u",
            "hard_guard_ic_u": "rejected_due_to_ic_u_guard",
            "hard_guard_bc_u": "rejected_due_to_bc_u_guard",
            "mu_only": "rejected_mu_only",
        }.get(reason)
        if attribute:
            setattr(self, attribute, int(getattr(self, attribute)) + 1)

    def _count_guard_rejection(self, reason: str) -> None:
        attribute = {
            "sparse_u_guard": "rejected_due_to_sparse_u_guard",
            "ic_u_guard": "rejected_due_to_ic_u_guard",
            "bc_u_guard": "rejected_due_to_bc_u_guard",
            "phase_range_guard": "rejected_due_to_phase_range_guard",
            "mass_guard": "rejected_due_to_mass_guard",
            "interface_proxy_guard": "rejected_due_to_interface_proxy_guard",
        }.get(reason)
        if attribute:
            setattr(self, attribute, int(getattr(self, attribute)) + 1)

    def _filter_ablation_candidates(
        self, candidates: list[V2Candidate]
    ) -> list[V2Candidate]:
        controller_cfg = dict(self.config.get("controller_v2", {}))
        sampling = bool(controller_cfg.get("sampling_redistribution_enabled", True))
        local_loss = bool(controller_cfg.get("local_loss_multipliers_enabled", True))
        if self.method == "vara_sampling_only":
            sampling, local_loss = True, False
        elif self.method == "vara_local_loss_only":
            sampling, local_loss = False, True
        if sampling and local_loss:
            return candidates
        desired = "sampling" if sampling else "local_loss" if local_loss else ""
        return [candidate for candidate in candidates if candidate.action_type == desired]

    def _record_decision(
        self,
        block: int,
        candidate: V2Candidate,
        decision: dict[str, Any],
    ) -> None:
        row: dict[str, Any] = {
            "block": block,
            "candidate_id": candidate.key(),
            "channel": candidate.variable,
            "candidate_rank_score": candidate.rank_score,
            "predicted_pareto_safety": 1.0
            / (
                1.0
                + max(
                    0.0,
                    self.candidate_guard_memory.get(candidate.key(), 0.0),
                )
            ),
            "previous_candidate_effectiveness": (
                float(self.controller.effectiveness.get(candidate.key(), 1.0))
                if self.controller is not None
                else 1.0
            ),
            "is_mu_only": self._is_mu_only_candidate(candidate),
            "is_u_interface_candidate": not self._is_mu_only_candidate(candidate),
            "policy": "u_first" if self.u_first_policy_enabled else "legacy_priority",
            "selected_primary_channel": candidate.variable,
            "selected_patch_id": candidate.patch_id,
            "selected_action_type": candidate.action_type,
            "target_metric": self._candidate_target_name(candidate.variable),
            **candidate.to_record(),
        }
        for key, value in decision.items():
            if isinstance(value, dict):
                row.update({f"{key}_{nested}": item for nested, item in value.items()})
            else:
                row[key] = value
        row.setdefault("rejection_reason", row.get("rollback_reason", ""))
        self.decision_rows.append(row)

    def _log_allocation(self, block: int) -> None:
        assert self.controller is not None
        self.allocation_history.append(
            {
                "block": block,
                "applied_optimizer_steps": self.applied_optimizer_steps,
                "trust_radius": self.controller.trust_radius,
                "variable_awareness_enabled": self.variable_awareness,
                "cahn_hilliard_u_first_policy": self.u_first_policy_enabled,
                "mu_support_only": self.mu_support_only,
                "patch_grid_shape": [
                    self.patch_grid.nt,
                    self.patch_grid.ny,
                    self.patch_grid.nx,
                ],
                **self.controller.state.to_record(),
            }
        )

    def _training_batch(self, *, adaptive: bool) -> dict[str, torch.Tensor]:
        count = int(self.config.get("training", {}).get("n_collocation", 2048))
        interior = (
            self._sample_adaptive(count)
            if adaptive and self.controller is not None
            else self._sample_uniform(count, self.sampling_rng)
        )
        forcing = self.benchmark.forcing(interior)
        return {
            "interior": interior,
            "forcing": forcing,
            "boundary": self.boundary_coordinates,
            "boundary_target": self.boundary_targets,
            "initial": self.initial_coordinates,
            "initial_target": self.initial_targets,
            "sparse": self.sparse_coordinates,
            "sparse_target": self.sparse_targets,
            "mass_target": self.mass_proxy_baseline,
        }

    def _sample_adaptive(self, count: int) -> torch.Tensor:
        assert self.controller is not None
        n_uniform = min(
            count, int(round(count * float(self.controller.config.min_uniform_mass)))
        )
        pieces = []
        if n_uniform:
            pieces.append(self._sample_uniform_numpy(n_uniform, self.sampling_rng))
        n_regional = count - n_uniform
        if n_regional:
            patch_ids = self.sampling_rng.choice(
                self.patch_grid.num_patches,
                size=n_regional,
                p=self.controller.state.sampling_mass,
            )
            regional = np.empty((n_regional, 3), dtype=np.float64)
            for patch_id in np.unique(patch_ids):
                indexes = np.flatnonzero(patch_ids == patch_id)
                x0, x1, y0, y1, t0, t1 = self.patch_grid.get_patch(
                    int(patch_id)
                ).bounds
                regional[indexes, 0] = self.sampling_rng.uniform(x0, x1, len(indexes))
                regional[indexes, 1] = self.sampling_rng.uniform(y0, y1, len(indexes))
                regional[indexes, 2] = self.sampling_rng.uniform(t0, t1, len(indexes))
            pieces.append(regional)
        values = np.vstack(pieces) if pieces else np.empty((0, 3), dtype=np.float64)
        self.sampling_rng.shuffle(values)
        return torch.as_tensor(values, device=self.device, dtype=self.dtype)

    def _sample_uniform(self, count: int, rng: np.random.Generator) -> torch.Tensor:
        return torch.as_tensor(
            self._sample_uniform_numpy(count, rng),
            device=self.device,
            dtype=self.dtype,
        )

    def _sample_uniform_numpy(
        self, count: int, rng: np.random.Generator
    ) -> np.ndarray:
        count = max(0, int(count))
        x0, x1, y0, y1 = self.benchmark.bounds
        t0, t1 = self.benchmark.t_bounds
        values = np.empty((count, 3), dtype=np.float64)
        if count:
            values[:, 0] = rng.uniform(x0, x1, count)
            values[:, 1] = rng.uniform(y0, y1, count)
            values[:, 2] = rng.uniform(t0, t1, count)
        return values

    def _sample_boundary(self, count: int, rng: np.random.Generator) -> torch.Tensor:
        count = max(1, int(count))
        x0, x1, y0, y1 = self.benchmark.bounds
        t0, t1 = self.benchmark.t_bounds
        wall = rng.integers(0, 4, count)
        s = rng.random(count)
        values = np.empty((count, 3), dtype=np.float64)
        values[:, 0] = x0 + s * (x1 - x0)
        values[:, 1] = y0 + s * (y1 - y0)
        values[:, 2] = rng.uniform(t0, t1, count)
        values[wall == 0, 0] = x0
        values[wall == 1, 0] = x1
        values[wall == 2, 1] = y0
        values[wall == 3, 1] = y1
        return torch.as_tensor(values, device=self.device, dtype=self.dtype)

    def _sample_initial(self, count: int, rng: np.random.Generator) -> torch.Tensor:
        values = self._sample_uniform_numpy(max(1, int(count)), rng)
        values[:, 2] = self.benchmark.t_bounds[0]
        return torch.as_tensor(values, device=self.device, dtype=self.dtype)

    def _make_diagnostic_batch(
        self,
        rng: np.random.Generator,
        n_interior: int,
        n_boundary: int,
        n_initial: int,
    ) -> dict[str, torch.Tensor]:
        interior = self._sample_uniform(max(8, n_interior), rng)
        boundary = self._sample_boundary(max(8, n_boundary), rng)
        initial = self._sample_initial(max(8, n_initial), rng)
        return {
            "interior": interior,
            "forcing": self.benchmark.forcing(interior),
            "boundary": boundary,
            "boundary_target": self.benchmark.boundary_values(boundary).detach(),
            "initial": initial,
            "initial_target": self.benchmark.initial_values(initial).detach(),
            "sparse": self.sparse_coordinates,
            "sparse_target": self.sparse_targets,
            "mass_target": self.mass_proxy_baseline,
        }

    def _model_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().clone() for name, value in self.model.state_dict().items()
        }

    def _restore_model(self, snapshot: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(snapshot)

    def _final_evaluation(self) -> dict[str, Any]:
        evaluation_cfg = dict(self.config.get("evaluation", {}))
        if bool(evaluation_cfg.get("controller_reference_metrics_enabled", False)):
            raise ValueError(
                "Cahn--Hilliard forbids evaluation references in controller decisions."
            )
        coordinates = make_evaluation_grid(
            self.benchmark,
            int(evaluation_cfg.get("nx", 48)),
            int(evaluation_cfg.get("ny", 48)),
            int(evaluation_cfg.get("nt", 11)),
            device=self.device,
            dtype=self.dtype,
        )
        benchmark_cfg = dict(self.config.get("benchmark", {}))
        return evaluate_cahn_hilliard(
            self.model,
            self.benchmark,
            coordinates,
            self.boundary_coordinates,
            self.boundary_targets,
            self.initial_coordinates,
            self.initial_targets,
            self.sparse_coordinates,
            self.sparse_targets,
            sparse_fraction=float(benchmark_cfg.get("sparse_fraction", 0.0)),
            sparse_seed=int(benchmark_cfg.get("sparse_seed", 0)) + self.seed,
            sparse_hash=self.sparse_hash,
            chunk_size=int(evaluation_cfg.get("chunk_size", 1024)),
        )

    def _save_artifacts(self, metrics: dict[str, Any]) -> None:
        pd.DataFrame(self.loss_rows).to_csv(self.run_dir / "losses.csv", index=False)
        pd.DataFrame([metrics]).to_csv(self.run_dir / "metrics.csv", index=False)
        if self.method != "vanilla":
            decisions = pd.DataFrame(self.decision_rows)
            if decisions.empty:
                decisions = pd.DataFrame(
                    columns=[
                        "block",
                        "variable",
                        "patch_id",
                        "action_type",
                        "accepted",
                        "prefiltered",
                        "rollback_reason",
                    ]
                )
            decisions.to_csv(self.run_dir / "vara_v2_decisions.csv", index=False)
            save_json(
                self.allocation_history,
                self.run_dir / "vara_v2_allocation_history.json",
            )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "method": self.method,
                "seed": self.seed,
                "metrics": metrics,
                "initial_model_parameter_hash": self.initial_model_parameter_hash,
                "sparse_hash": self.sparse_hash,
            },
            self.run_dir / "checkpoints" / "final.pt",
        )
        summary = {
            "benchmark": "cahn_hilliard",
            "method": self.method,
            "seed": self.seed,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "git_commit": _git_commit(),
            "run_dir": str(self.run_dir),
            "initial_model_parameter_hash": self.initial_model_parameter_hash,
            "sparse_hash": self.sparse_hash,
            "controller_reference_metrics_enabled": False,
            "metrics": metrics,
        }
        save_json(summary, self.run_dir / "summary.json")
        if bool(self.config.get("plots", {}).get("enabled", True)):
            from .plots import save_run_plots

            save_run_plots(
                self.model,
                self.benchmark,
                self.run_dir,
                self.loss_rows,
                self.decision_rows,
                self.allocation_history,
                device=self.device,
                dtype=self.dtype,
            )


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(normalized)


def _resolve_dtype(requested: str) -> torch.dtype:
    if requested.lower() in {"float32", "fp32", "single"}:
        return torch.float32
    if requested.lower() in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError("Cahn--Hilliard dtype must be float32 or float64.")


def _tensor_hash(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _flat_gradient(
    loss: torch.Tensor,
    parameters: list[nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    pieces = [
        gradient.detach().reshape(-1)
        if gradient is not None
        else torch.zeros_like(parameter).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(pieces) if pieces else loss.detach().new_zeros(1)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.cpu()) <= 1e-20:
        return 0.0
    return float((torch.dot(left, right) / denominator).cpu())


def _relative_change(
    before: float | None,
    after: float | None,
    *,
    absolute: bool = False,
) -> float:
    if before is None or after is None:
        return float("nan")
    denominator = max(abs(float(before)), 1e-10)
    change = (float(after) - float(before)) / denominator
    return abs(change) if absolute else change


def _improvement_ratio(before: float | None, after: float | None) -> float:
    change = _relative_change(before, after)
    return -change if np.isfinite(change) else float("nan")


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _cap_mean_one_multiplier(values: np.ndarray, maximum: float) -> np.ndarray:
    """Cap a multiplier vector without changing its mean-one loss mass."""
    if maximum < 1.0:
        raise ValueError("A mean-one local multiplier cap must be at least 1.0.")
    out = np.minimum(np.asarray(values, dtype=float).copy(), float(maximum))
    deficit = float(out.size) - float(np.sum(out))
    if deficit > 1e-12:
        room = np.maximum(float(maximum) - out, 0.0)
        room_total = float(np.sum(room))
        if room_total <= 1e-12:
            raise RuntimeError("Unable to conserve mu multiplier mass under cap.")
        out += deficit * room / room_total
    out += (float(out.size) - float(np.sum(out))) / float(out.size)
    return np.minimum(out, float(maximum))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
