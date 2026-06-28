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
            diagnostics_cfg.get("interface_focus_strength", 1.0)
        )
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
        if method != "vanilla":
            controller_cfg = deepcopy(dict(self.config.get("controller_v2", {})))
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
                self._log_allocation(block)
                continue
            candidate = active[0]
            if self.controller.config.counterfactual_probe_enabled:
                self._counterfactual_block(block, candidate, schedule)
            else:
                self._unguarded_block(
                    block, candidate, before, before_metrics, schedule
                )
            self._log_allocation(block)

    def _counterfactual_block(
        self,
        block: int,
        candidate: V2Candidate,
        schedule: dict[str, int],
    ) -> None:
        assert self.controller is not None
        model_before = self._model_snapshot()
        optimizer_before = deepcopy(self.optimizer.state_dict())
        allocation_before = self.controller.state.snapshot()
        rng_before = deepcopy(self.sampling_rng.bit_generator.state)
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
        candidate_batch = self._training_batch(adaptive=True)
        candidate_rows = self._train_steps(
            candidate_batch,
            probe_steps,
            phase=f"block_{block}_candidate_probe",
        )
        candidate_snapshot = self._diagnose()
        candidate_metrics = self._controller_metrics(candidate_snapshot)
        controller_cfg = dict(self.config.get("controller_v2", {}))
        accepted, decision = self.controller.evaluate(
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
            self.accepted_interventions += 1
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
        self._record_decision(block, candidate, decision)
        self._commit_rows(
            self._train_steps(
                kept_batch,
                schedule["block_steps"] - probe_steps,
                phase=f"block_{block}_continuation",
            )
        )

    def _unguarded_block(
        self,
        block: int,
        candidate: V2Candidate,
        before: DiagnosticSnapshot,
        before_metrics: dict[str, float],
        schedule: dict[str, int],
    ) -> None:
        assert self.controller is not None
        self.controller.apply(candidate)
        self._commit_rows(
            self._train_steps(
                self._training_batch(adaptive=True),
                schedule["block_steps"],
                phase=f"block_{block}_unguarded",
            )
        )
        after = self._diagnose()
        accepted, decision = self.controller.evaluate(
            candidate,
            self._candidate_score(candidate, before),
            self._candidate_score(candidate, after),
            before_metrics,
            self._controller_metrics(after),
            comparison_mode="temporal_no_rollback",
        )
        if accepted:
            self.accepted_interventions += 1
        else:
            self.rejected_interventions += 1
        self._record_decision(block, candidate, decision)

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
        channel = result.channels.get(
            candidate.variable, result.channels.get("pde_residual")
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
        means = {
            name: float(np.mean(snapshot.raw_scores[index]))
            for index, name in enumerate(snapshot.names)
        }
        pde = means.get(
            "pde_residual",
            float(
                np.mean(
                    [
                        means.get("ch_residual", 0.0),
                        means.get("chemical_potential_residual", 0.0),
                    ]
                )
            ),
        )
        boundary = means.get("boundary_violation", 0.0)
        initial = means.get("initial_condition_violation", 0.0)
        sparse = float(
            np.mean([value for name, value in means.items() if name.startswith("sparse_")])
        ) if any(name.startswith("sparse_") for name in means) else 0.0
        return {
            "pde_residual_mean": pde,
            "boundary_condition_error": boundary,
            "unweighted_physics_validation_loss": pde + boundary + initial,
            "unweighted_validation_loss": pde + boundary + initial + sparse,
        }

    @staticmethod
    def _candidate_score(
        candidate: V2Candidate, snapshot: DiagnosticSnapshot
    ) -> float:
        name = candidate.variable if candidate.variable in snapshot.names else "pde_residual"
        return float(snapshot.raw_scores[snapshot.names.index(name), candidate.patch_id])

    def _route_candidate_losses(self, candidates: list[V2Candidate]) -> None:
        routes = {
            "ch_residual": ["ch_residual"],
            "chemical_potential_residual": ["chemical_potential_residual"],
            "pde_residual": ["ch_residual", "chemical_potential_residual"],
            "boundary_violation": ["bc_u", "bc_mu"],
            "initial_condition_violation": ["ic_u", "ic_mu"],
            "sparse_u_mismatch": ["sparse_u_mse"],
            "sparse_mu_mismatch": ["sparse_mu_mse"],
            "sparse_data_mismatch": ["sparse_u_mse", "sparse_mu_mse"],
        }
        for candidate in candidates:
            candidate.loss_names = routes.get(
                candidate.variable, ["ch_residual", "chemical_potential_residual"]
            )

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
        row: dict[str, Any] = {"block": block, **candidate.to_record()}
        for key, value in decision.items():
            if isinstance(value, dict):
                row.update({f"{key}_{nested}": item for nested, item in value.items()})
            else:
                row[key] = value
        self.decision_rows.append(row)

    def _log_allocation(self, block: int) -> None:
        assert self.controller is not None
        self.allocation_history.append(
            {
                "block": block,
                "applied_optimizer_steps": self.applied_optimizer_steps,
                "trust_radius": self.controller.trust_radius,
                "variable_awareness_enabled": self.variable_awareness,
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


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
