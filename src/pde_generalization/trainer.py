"""Matched-seed trainer for vanilla PINNs and isolated VARA V2 PDE runs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
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

from .benchmarks import ManufacturedBenchmark, build_benchmark
from .diagnostics import (
    DiagnosticSnapshot,
    PDEPatchGrid,
    build_diagnostic_snapshot,
)
from .losses import LossResult, compute_training_loss
from .metrics import evaluate_model, make_evaluation_grid
from .models import build_pde_model, model_parameter_hash


SUPPORTED_MODES = {
    "vanilla",
    "vara_v2",
    "vara_v2_no_variable_awareness",
    "vara_v2_sampling_only",
    "vara_v2_local_loss_only",
    "vara_v2_no_guard",
}


class PDEGeneralizationTrainer:
    """Train one reproducible manufactured-PDE run without legacy trainer coupling.

    Full-field references are only accessed by :meth:`_final_evaluation`, after
    all controller decisions have completed. During adaptation the controller
    receives PDE residuals, prescribed BC/IC mismatch, and configured sparse
    training-data mismatch only.
    """

    def __init__(
        self,
        config: dict[str, Any],
        mode: str,
        run_dir: str | Path,
    ) -> None:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported PDE generalization mode {mode!r}.")
        self.config = deepcopy(config)
        self.mode = mode
        self.seed = int(self.config.get("seed", 0))
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("logs", "checkpoints", "figures", "tables"):
            (self.run_dir / name).mkdir(exist_ok=True)

        set_seed(self.seed)
        self.device = _resolve_device(str(self.config.get("device", "cpu")))
        self.dtype = _resolve_dtype(str(self.config.get("dtype", "float32")))
        self.benchmark = build_benchmark(self.config)
        self.patch_grid = PDEPatchGrid.from_config(self.config, self.benchmark)
        self.model = build_pde_model(self.config).to(device=self.device, dtype=self.dtype)
        self.initial_model_parameter_hash = model_parameter_hash(self.model)

        training_cfg = dict(self.config.get("training", {}))
        self.weights = {
            name: float(value)
            for name, value in dict(
                training_cfg.get(
                    "weights",
                    {"pde": 1.0, "bc": 10.0, "ic": 10.0, "sparse_data": 1.0},
                )
            ).items()
        }
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(training_cfg.get("lr", 1e-3)),
        )

        self.sampling_rng = np.random.default_rng(self.seed + 10_003)
        condition_rng = np.random.default_rng(self.seed + 20_003)
        sparse_rng = np.random.default_rng(
            self.seed
            + 30_003
            + int(self.config.get("benchmark_params", {}).get("sparse_seed", 0))
        )
        diagnostic_rng = np.random.default_rng(self.seed + 40_003)
        n_boundary = int(training_cfg.get("n_boundary", 512))
        n_initial = int(training_cfg.get("n_initial", 512))
        n_sparse = int(training_cfg.get("n_sparse_data", 128))
        self.boundary_coordinates = self._sample_boundary(n_boundary, condition_rng)
        self.initial_coordinates = self._sample_initial(n_initial, condition_rng)
        self.sparse_coordinates = self._sample_uniform(n_sparse, sparse_rng)
        with torch.no_grad():
            self.boundary_targets = self.benchmark.boundary_values(self.boundary_coordinates)
            self.initial_targets = self.benchmark.initial_values(self.initial_coordinates)
            self.sparse_targets = self.benchmark.exact(self.sparse_coordinates)
        self.sparse_sample_hash = _tensor_hash(self.sparse_coordinates, self.sparse_targets)

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
        configured_variable_awareness = bool(
            self.config.get("controller_v2", {}).get(
                "variable_awareness_enabled", True
            )
        )
        self.variable_awareness = (
            mode != "vara_v2_no_variable_awareness"
            and configured_variable_awareness
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
        if self.mode != "vanilla":
            controller_cfg = deepcopy(dict(self.config.get("controller_v2", {})))
            controller_cfg["variable_awareness_enabled"] = self.variable_awareness
            if mode == "vara_v2_no_guard":
                controller_cfg["counterfactual_probe_enabled"] = False
                controller_cfg["rollback_enabled"] = False
                controller_cfg["gradient_prefilter_enabled"] = False
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
                V2ControllerConfig.from_dict(controller_cfg, self.patch_grid.num_patches)
            )
            self.rollback_enabled = bool(controller_cfg.get("rollback_enabled", True))

        self.loss_rows: list[dict[str, Any]] = []
        self.decision_rows: list[dict[str, Any]] = []
        self.allocation_history: list[dict[str, Any]] = []
        self.applied_optimizer_steps = 0
        self.optimizer_step_calls = 0
        self.objective_evaluation_count = 0
        self.diagnostic_evaluation_count = 0
        self.accepted_interventions = 0
        self.rejected_interventions = 0
        self.prefiltered_interventions = 0
        self.rollback_count = 0
        self.wall_clock_seconds = 0.0
        save_config(self.config, self.run_dir / "resolved_config.yaml")

    def run(self) -> dict[str, Any]:
        """Execute training, save all artifacts, and return final metrics."""
        schedule = self._schedule()
        started = time.perf_counter()
        if self.mode == "vanilla":
            self._run_vanilla(schedule)
        else:
            self._run_vara(schedule)
        if self.applied_optimizer_steps != schedule["total_steps"]:
            raise RuntimeError(
                "Applied optimizer-step budget drifted: "
                f"expected {schedule['total_steps']}, got {self.applied_optimizer_steps}."
            )
        self.wall_clock_seconds = time.perf_counter() - started
        metrics = self._final_evaluation()
        metrics.update(
            {
                "optimization_wall_clock_sec": self.wall_clock_seconds,
                "applied_optimizer_steps": self.applied_optimizer_steps,
                "optimizer_step_calls": self.optimizer_step_calls,
                "objective_evaluation_count": self.objective_evaluation_count,
                "diagnostic_evaluation_count": self.diagnostic_evaluation_count,
                "accepted_interventions": self.accepted_interventions,
                "rejected_interventions": self.rejected_interventions,
                "prefiltered_interventions": self.prefiltered_interventions,
                "rollback_count": self.rollback_count,
            }
        )
        self._save_artifacts(metrics)
        return metrics

    def _schedule(self) -> dict[str, int]:
        cfg = dict(self.config.get("controller_v2", {}))
        training_cfg = dict(self.config.get("training", {}))
        total = int(cfg.get("total_steps", training_cfg.get("total_steps", 1000)))
        warmup = int(cfg.get("warmup_steps", max(1, total // 4)))
        blocks = int(cfg.get("control_blocks", 3))
        block_steps = int(cfg.get("block_steps", (total - warmup) // max(1, blocks)))
        probe = int(cfg.get("probe_steps", min(25, max(1, block_steps // 10))))
        if warmup + blocks * block_steps != total:
            raise ValueError(
                "PDE controller schedule must satisfy "
                "warmup_steps + control_blocks * block_steps == total_steps."
            )
        if not 0 < probe < block_steps:
            raise ValueError("probe_steps must be positive and smaller than block_steps.")
        return {
            "total_steps": total,
            "warmup_steps": warmup,
            "control_blocks": blocks,
            "block_steps": block_steps,
            "probe_steps": probe,
        }

    def _run_vanilla(self, schedule: dict[str, int]) -> None:
        batch = self._training_batch(adaptive=False)
        rows = self._train_steps(batch, schedule["warmup_steps"], phase="warmup")
        self._commit_rows(rows)
        for block in range(schedule["control_blocks"]):
            batch = self._training_batch(adaptive=False)
            rows = self._train_steps(
                batch,
                schedule["block_steps"],
                phase=f"block_{block}",
            )
            self._commit_rows(rows)

    def _run_vara(self, schedule: dict[str, int]) -> None:
        assert self.controller is not None
        batch = self._training_batch(adaptive=False)
        rows = self._train_steps(batch, schedule["warmup_steps"], phase="warmup")
        self._commit_rows(rows)
        self._log_allocation(-1)

        for block in range(schedule["control_blocks"]):
            before = self._diagnose()
            before_metrics = self._controller_metrics(before)
            self.controller.update_history(
                before.names,
                before.normalized_scores,
                before_metrics,
            )
            weak_regions = self.detector.detect(
                before.normalized_scores,
                before.names,
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
                rows = self._train_steps(
                    self._training_batch(adaptive=True),
                    schedule["block_steps"],
                    phase=f"block_{block}_no_action",
                )
                self._commit_rows(rows)
                self._log_allocation(block)
                continue

            candidate = active[0]
            if self.controller.config.counterfactual_probe_enabled:
                self._counterfactual_block(block, candidate, before, before_metrics, schedule)
            else:
                self._unguarded_block(block, candidate, before, before_metrics, schedule)
            self._log_allocation(block)

    def _counterfactual_block(
        self,
        block: int,
        candidate: V2Candidate,
        before: DiagnosticSnapshot,
        before_metrics: dict[str, float],
        schedule: dict[str, int],
    ) -> None:
        assert self.controller is not None
        del before_metrics
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

        accepted, decision = self.controller.evaluate(
            candidate,
            self._candidate_score(candidate, neutral_snapshot),
            self._candidate_score(candidate, candidate_snapshot),
            neutral_metrics,
            candidate_metrics,
            target_threshold=float(
                self.config.get("controller_v2", {}).get(
                    "counterfactual_target_margin", 0.005
                )
            ),
            guard_threshold=float(
                self.config.get("controller_v2", {}).get(
                    "counterfactual_guard_margin", 0.02
                )
            ),
            comparison_mode="counterfactual",
        )
        if accepted:
            self.accepted_interventions += 1
            kept_batch = candidate_batch
            kept_rows = candidate_rows
        else:
            self.rejected_interventions += 1
            if self.rollback_enabled:
                self.rollback_count += 1
                self._restore_model(neutral_model)
                self.optimizer.load_state_dict(neutral_optimizer)
                self.controller.state.restore(allocation_before)
                self.sampling_rng.bit_generator.state = deepcopy(rng_after_neutral)
                kept_batch = neutral_batch
                kept_rows = neutral_rows
            else:
                kept_batch = candidate_batch
                kept_rows = candidate_rows
        self._commit_rows(kept_rows)
        self._record_decision(block, candidate, decision)
        remaining = schedule["block_steps"] - probe_steps
        rows = self._train_steps(
            kept_batch,
            remaining,
            phase=f"block_{block}_continuation",
        )
        self._commit_rows(rows)

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
        rows = self._train_steps(
            self._training_batch(adaptive=True),
            schedule["block_steps"],
            phase=f"block_{block}_unguarded",
        )
        self._commit_rows(rows)
        after = self._diagnose()
        after_metrics = self._controller_metrics(after)
        accepted, decision = self.controller.evaluate(
            candidate,
            self._candidate_score(candidate, before),
            self._candidate_score(candidate, after),
            before_metrics,
            after_metrics,
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
        rows: list[dict[str, Any]] = []
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
            self.objective_evaluation_count += 1
            if not torch.isfinite(result.total):
                raise FloatingPointError(
                    f"Non-finite training loss in {self.benchmark.name}/{self.mode}."
                )
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
            committed = dict(row)
            committed["step"] = self.applied_optimizer_steps
            self.loss_rows.append(committed)

    def _candidate_influence(
        self,
        candidates: list[V2Candidate],
    ) -> dict[str, dict[str, float]]:
        if self.controller is None or not self.controller.config.gradient_prefilter_enabled:
            return {candidate.key(): {"gradient_compatibility": 0.0, "gradient_conflict": 0.0} for candidate in candidates}
        influence: dict[str, dict[str, float]] = {}
        probe_batch = self.diagnostic_batch
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        for candidate in candidates:
            self.model.zero_grad(set_to_none=True)
            result = compute_training_loss(
                self.model,
                self.benchmark,
                probe_batch,
                self.weights,
                self.patch_grid,
                self.controller.state,
            )
            self.objective_evaluation_count += 1
            target = self._candidate_target_tensor(candidate, result)
            guard = sum(result.components.values())
            target_grad = _flat_gradient(target, parameters, retain_graph=True)
            guard_grad = _flat_gradient(guard, parameters, retain_graph=False)
            cosine = _cosine(target_grad, guard_grad)
            influence[candidate.key()] = {
                "gradient_compatibility": max(0.0, cosine),
                "gradient_conflict": max(0.0, -cosine),
            }
        self.model.zero_grad(set_to_none=True)
        return influence

    def _candidate_target_tensor(
        self,
        candidate: V2Candidate,
        result: LossResult,
    ) -> torch.Tensor:
        if candidate.variable not in result.channels:
            channel = result.channels.get("pde_residual")
        else:
            channel = result.channels[candidate.variable]
        if channel is None:
            return result.total
        values, coordinates = channel
        patch_ids = self.patch_grid.assign_torch(coordinates.detach())
        selected = values[patch_ids == candidate.patch_id]
        return selected.mean() if selected.numel() else values.mean()

    def _diagnose(self) -> DiagnosticSnapshot:
        self.diagnostic_evaluation_count += 1
        return build_diagnostic_snapshot(
            self.model,
            self.benchmark,
            self.patch_grid,
            self.diagnostic_batch,
            percentile=self.diagnostic_percentile,
            variable_awareness=self.variable_awareness,
        )

    @staticmethod
    def _controller_metrics(snapshot: DiagnosticSnapshot) -> dict[str, float]:
        channel_means = {
            name: float(np.mean(snapshot.raw_scores[index]))
            for index, name in enumerate(snapshot.names)
        }
        pde_channels = [
            value
            for name, value in channel_means.items()
            if "residual" in name and "boundary" not in name
        ]
        pde = float(np.mean(pde_channels)) if pde_channels else 0.0
        boundary = float(channel_means.get("boundary_mismatch", 0.0))
        initial = float(channel_means.get("initial_condition_mismatch", 0.0))
        sparse_channels = [
            value for name, value in channel_means.items() if name.startswith("sparse_")
        ]
        sparse = float(np.mean(sparse_channels)) if sparse_channels else 0.0
        return {
            "pde_residual_mean": pde,
            "boundary_condition_error": boundary,
            "unweighted_physics_validation_loss": pde + boundary + initial,
            "unweighted_validation_loss": pde + boundary + initial + sparse,
        }

    @staticmethod
    def _candidate_score(candidate: V2Candidate, snapshot: DiagnosticSnapshot) -> float:
        name = candidate.variable if candidate.variable in snapshot.names else snapshot.names[0]
        # Raw values retain one physical scale across the neutral/candidate
        # counterfactual. Per-snapshot normalization is only for cross-channel
        # weak-region ranking and would distort a temporal comparison.
        return float(snapshot.raw_scores[snapshot.names.index(name), candidate.patch_id])

    def _route_candidate_losses(self, candidates: list[V2Candidate]) -> None:
        for candidate in candidates:
            name = candidate.variable
            if name == "momentum_u_residual":
                candidate.loss_names = ["momentum_u"]
            elif name == "momentum_v_residual":
                candidate.loss_names = ["momentum_v"]
            elif name.startswith("sparse_"):
                candidate.loss_names = ["sparse_data"]
            elif name.startswith("boundary_"):
                candidate.loss_names = ["bc"]
            elif name.startswith("initial_"):
                candidate.loss_names = ["ic"]
            else:
                candidate.loss_names = ["pde"]

    def _filter_ablation_candidates(
        self,
        candidates: list[V2Candidate],
    ) -> list[V2Candidate]:
        controller_cfg = dict(self.config.get("controller_v2", {}))
        sampling_enabled = bool(controller_cfg.get("sampling_redistribution_enabled", True))
        local_enabled = bool(controller_cfg.get("local_loss_multipliers_enabled", True))
        if self.mode == "vara_v2_sampling_only":
            sampling_enabled, local_enabled = True, False
        elif self.mode == "vara_v2_local_loss_only":
            sampling_enabled, local_enabled = False, True
        if sampling_enabled and local_enabled:
            return candidates
        desired = "sampling" if sampling_enabled else "local_loss" if local_enabled else ""
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
                for nested_key, nested_value in value.items():
                    row[f"{key}_{nested_key}"] = nested_value
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
        n = int(self.config.get("training", {}).get("n_collocation", 2048))
        if adaptive and self.controller is not None:
            interior = self._sample_adaptive(n)
        else:
            interior = self._sample_uniform(n, self.sampling_rng)
        return {
            "interior": interior,
            "boundary": self.boundary_coordinates,
            "boundary_target": self.boundary_targets,
            "initial": self.initial_coordinates,
            "initial_target": self.initial_targets,
            "sparse": self.sparse_coordinates,
            "sparse_target": self.sparse_targets,
        }

    def _sample_adaptive(self, count: int) -> torch.Tensor:
        assert self.controller is not None
        minimum_uniform = float(self.controller.config.min_uniform_mass)
        n_uniform = min(count, int(round(count * minimum_uniform)))
        pieces: list[np.ndarray] = []
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
                patch = self.patch_grid.get_patch(int(patch_id))
                x0, x1, y0, y1, t0, t1 = patch.bounds
                regional[indexes, 0] = self.sampling_rng.uniform(x0, x1, indexes.size)
                regional[indexes, 1] = self.sampling_rng.uniform(y0, y1, indexes.size)
                regional[indexes, 2] = self.sampling_rng.uniform(t0, t1, indexes.size)
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

    def _sample_uniform_numpy(self, count: int, rng: np.random.Generator) -> np.ndarray:
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
        wall = rng.integers(0, 4, size=count)
        s = rng.random(count)
        values = np.empty((count, 3), dtype=np.float64)
        values[:, 2] = rng.uniform(t0, t1, count)
        values[:, 0] = x0 + s * (x1 - x0)
        values[:, 1] = y0 + s * (y1 - y0)
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
        with torch.no_grad():
            boundary_target = self.benchmark.boundary_values(boundary)
            initial_target = self.benchmark.initial_values(initial)
        return {
            "interior": interior,
            "boundary": boundary,
            "boundary_target": boundary_target,
            "initial": initial,
            "initial_target": initial_target,
            "sparse": self.sparse_coordinates,
            "sparse_target": self.sparse_targets,
        }

    def _model_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.model.state_dict().items()}

    def _restore_model(self, snapshot: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(snapshot)

    def _final_evaluation(self) -> dict[str, float]:
        evaluation_cfg = dict(self.config.get("evaluation", {}))
        if bool(evaluation_cfg.get("controller_reference_metrics_enabled", False)):
            raise ValueError(
                "PDE generalization forbids evaluation references in controller decisions."
            )
        coordinates = make_evaluation_grid(
            self.benchmark,
            int(evaluation_cfg.get("nx", 48)),
            int(evaluation_cfg.get("ny", 48)),
            int(evaluation_cfg.get("nt", 11)),
            device=self.device,
            dtype=self.dtype,
        )
        return evaluate_model(
            self.model,
            self.benchmark,
            coordinates,
            self.boundary_coordinates,
            self.initial_coordinates,
            self.sparse_coordinates,
            self.sparse_targets,
            residual_chunk_size=int(evaluation_cfg.get("residual_chunk_size", 2048)),
        )

    def _save_artifacts(self, metrics: dict[str, Any]) -> None:
        pd.DataFrame(self.loss_rows).to_csv(self.run_dir / "losses.csv", index=False)
        pd.DataFrame([metrics]).to_csv(self.run_dir / "metrics.csv", index=False)
        if self.mode != "vanilla":
            decision_columns = [
                "block",
                "variable",
                "patch_id",
                "action_type",
                "accepted",
                "prefiltered",
                "rollback_reason",
            ]
            frame = pd.DataFrame(self.decision_rows)
            if frame.empty:
                frame = pd.DataFrame(columns=decision_columns)
            frame.to_csv(self.run_dir / "vara_v2_decisions.csv", index=False)
            save_json(
                self.allocation_history,
                self.run_dir / "vara_v2_allocation_history.json",
            )
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "benchmark": self.benchmark.name,
            "method": self.mode,
            "seed": self.seed,
            "metrics": metrics,
            "initial_model_parameter_hash": self.initial_model_parameter_hash,
            "sparse_sample_hash": self.sparse_sample_hash,
        }
        torch.save(checkpoint, self.run_dir / "checkpoints" / "final.pt")
        summary = {
            "benchmark": self.benchmark.name,
            "method": self.mode,
            "seed": self.seed,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "git_commit": _git_commit(),
            "run_dir": str(self.run_dir),
            "initial_model_parameter_hash": self.initial_model_parameter_hash,
            "sparse_sample_hash": self.sparse_sample_hash,
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
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return torch.device(normalized)


def _resolve_dtype(requested: str) -> torch.dtype:
    if requested.lower() in {"float32", "fp32", "single"}:
        return torch.float32
    if requested.lower() in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype {requested!r}; use float32 or float64.")


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
