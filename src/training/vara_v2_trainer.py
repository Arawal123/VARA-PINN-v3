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
    ALLOWED_GUARD_METRICS,
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
        self.v2_config = V2ControllerConfig.from_dict(cfg, self.patch_grid.num_patches)
        self.v2_controller = VARAV2Controller(self.v2_config)
        self.v2_decision_logger = CSVLogger(self.run_dir / "vara_v2_decisions.csv")
        self.v2_state_logger = JSONListLogger(self.run_dir / "vara_v2_allocation_history.json")
        self.accepted_interventions = 0
        self.rejected_interventions = 0
        self.prefiltered_interventions = 0
        self.rollback_enabled = bool(cfg.get("rollback_enabled", True))
        self._probe_batch = self._make_probe_batch()

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
        self._train_v2_steps(batch, warmup_steps, cycle=0, phase="v2_warmup")

        for block in range(control_blocks):
            maps_before, raw_before, names, weak_regions, coords = self._diagnose_reference_free()
            metrics_before = self._guard_metrics(coords)
            self.v2_controller.update_history(names, raw_before, metrics_before)
            candidates = self.v2_controller.candidates(weak_regions)
            influence = self._candidate_influence(candidates)
            ranked = self.v2_controller.rank(candidates, influence)
            candidate = next((item for item in ranked if not item.prefiltered), None)
            prefiltered = [item for item in ranked if item.prefiltered]
            self.prefiltered_interventions += len(prefiltered)
            for index, rejected in enumerate(prefiltered):
                decision = self.v2_controller.record_prefilter(
                    rejected,
                    update_trust=candidate is None and index == 0,
                )
                self._log_decision(block, rejected, decision)

            if candidate is None:
                self._train_v2_steps(batch, block_steps, cycle=block + 1, phase="v2_no_action")
                maps_kept, _raw_kept, _names_kept, _weak_kept, coords_kept = (
                    self._diagnose_reference_free(update_history=False)
                )
                batch = self._resample_v2_batch(maps_kept, coords_kept)
                self._log_state(block)
                continue

            model_snapshot = self._model_snapshot()
            optimizer_snapshot = deepcopy(self.optimizer.state_dict())
            controller_snapshot = self.v2_controller.state.snapshot()
            trust_before = self.v2_controller.trust_radius
            self.v2_controller.apply(candidate)
            # Sampling actions must be evaluated on the allocation they
            # propose. Probing the old batch makes sampling-only candidates
            # observationally identical to doing nothing.
            proposal_batch = self._resample_v2_batch(maps_before, coords)
            self._train_v2_steps(
                proposal_batch,
                probe_steps,
                cycle=block + 1,
                phase="v2_trust_probe",
                probe=True,
            )
            _maps_after, raw_after, names_after, _weak_after, coords_after = self._diagnose_reference_free(
                update_history=False
            )
            metrics_after = self._guard_metrics(coords_after)
            before_target = self._candidate_score(candidate, raw_before, names)
            after_target = self._candidate_score(candidate, raw_after, names_after)
            accepted, decision = self.v2_controller.evaluate(
                candidate,
                before_target,
                after_target,
                metrics_before,
                metrics_after,
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
            committed = bool(accepted or not self.rollback_enabled)
            decision["committed"] = committed
            if accepted:
                self.accepted_interventions += 1
            elif self.rollback_enabled:
                self.rejected_interventions += 1
                self._restore_model_snapshot(model_snapshot)
                self.optimizer.load_state_dict(optimizer_snapshot)
                self.v2_controller.state.restore(controller_snapshot)
                self.compute_tracker.record_rollback_steps(probe_steps)
            else:
                self.rejected_interventions += 1
            self._log_decision(block, candidate, decision)

            remaining = block_steps - probe_steps
            committed_batch = proposal_batch if committed else batch
            self._train_v2_steps(
                committed_batch,
                remaining,
                cycle=block + 1,
                phase="v2_commit" if committed else "v2_rollback_continue",
            )
            maps_kept, _raw_kept, _names_kept, _weak_kept, coords_kept = self._diagnose_reference_free(
                update_history=False
            )
            batch = self._resample_v2_batch(maps_kept, coords_kept)
            self._log_state(block)

        metrics = self.evaluate_and_save_final()
        metrics.update(
            {
                "accepted_interventions": self.accepted_interventions,
                "rejected_interventions": self.rejected_interventions,
                "prefiltered_interventions": self.prefiltered_interventions,
                "rollback_count": self.rejected_interventions,
                "v2_final_trust_radius": self.v2_controller.trust_radius,
                "v2_accepted_improvement_per_compute": self._accepted_improvement_per_compute(),
            }
        )
        metrics["rollback_count"] = self.rejected_interventions if self.rollback_enabled else 0
        save_json(metrics, self.run_dir / "summary.json")
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
    ) -> None:
        train_cfg = self.config.get("training", {})
        scalar_weights = dict(train_cfg.get("weights", {}))
        log_every = max(1, int(train_cfg.get("log_every", 25)))
        self.model.train()
        started = time.perf_counter()
        for local_step in range(int(steps)):
            if not self.compute_tracker.can_start_objective(int(batch["xy_f"].shape[0])):
                break
            self.optimizer.zero_grad(set_to_none=True)
            self.compute_tracker.record_objective(batch)
            pointwise = compute_pointwise_losses(self.model, batch, self.benchmark, self.steady)
            losses = compute_budgeted_patch_losses(
                pointwise,
                batch,
                self.patch_grid,
                self.v2_controller.state.loss_multipliers,
                reduction=str(train_cfg.get("pointwise_reduction", "legacy_mse")),
            )
            total = weighted_sum(losses, scalar_weights)
            anchor = self.pressure_gauge_loss()
            continuation_anchor, continuation_weight = self.continuation_anchor_loss(batch)
            continuation_replay, replay_weight = self.continuation_replay_loss()
            total = total + anchor + continuation_anchor + continuation_replay
            total.backward()
            grad_norm = self._grad_norm()
            self.optimizer.step()
            self.compute_tracker.record_optimizer_step()
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
        self.compute_tracker.add_phase_time("optimization", time.perf_counter() - started)

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
        metrics = self.evaluate_metrics(coords)
        selected = {
            name: float(metrics[name])
            for name in ALLOWED_GUARD_METRICS
            if name != "unweighted_validation_loss"
            and name in metrics
            and math.isfinite(float(metrics[name]))
        }
        # The ordinary validation metric may include full-field reference data.
        # V2 instead uses the reference-free PDE + boundary validation objective.
        selected["unweighted_validation_loss"] = float(
            metrics["unweighted_pde_loss"] + metrics["unweighted_bc_loss"]
        )
        self.v2_controller.assert_reference_free(selected)
        return selected

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
        train_cfg = self.config.get("training", {})
        n_f = int(train_cfg.get("n_collocation", 2048))
        n_bc = int(train_cfg.get("n_boundary", 768))
        n_data = int(train_cfg.get("n_data", 0))
        uniform_mass = float(self.v2_config.min_uniform_mass)
        n_uniform = int(round(n_f * uniform_mass))
        n_adaptive = n_f - n_uniform
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
        xy_f_np = np.vstack(pieces)
        self.adaptive_sampler.rng.shuffle(xy_f_np)
        xy_f = torch.tensor(xy_f_np, dtype=torch.float32, device=self.device)
        xy_bc = self._sample_boundary(n_bc)
        xy_data = self._sample_data(n_data)
        return self.make_batch(xy_f, xy_bc, xy_data)

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
