"""VARA-PINN trainer."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.controllers import (
    ConstrainedVARAPolicy,
    LocalControllerConfig,
    LocalVARAController,
    RuleBasedVARAPolicy,
    VARAController,
)
from src.diagnostics import DiagnosticMapBuilder
from src.losses.base_losses import compute_global_losses, compute_pointwise_losses, weighted_sum
from src.losses.modern_baselines import (
    ReLoBRaLoWeights,
    ResidualAttentionState,
    causal_temporal_objective,
    loss_gradient_norm,
    self_adaptive_objective,
)
from src.sampling.residual_sampler import sample_from_score_grid
from src.training.trainer import ExperimentTrainer
from src.utils.io import ensure_dir, save_json
from src.utils.logging import CSVLogger, JSONListLogger
from src.visualization.controller_plots import (
    save_accept_reject_counts,
    save_active_intervention_map,
    save_collateral_damage_timeline,
    save_local_weight_evolution,
    save_patch_grid_overlay,
    save_patch_scalar_heatmap,
    save_patch_score_map,
    save_pressure_collateral_timeline,
    save_targeted_patch_improvement,
)


class VARATrainer(ExperimentTrainer):
    """Train a variable-aware regional adaptive PINN."""

    def __init__(self, config: dict, mode: str = "full_vara") -> None:
        super().__init__(config, mode)
        controller_cfg = config.get("controller", {})
        base_policy = RuleBasedVARAPolicy(
            strength=float(controller_cfg.get("strength", 1.0)),
            max_actions_per_cycle=int(controller_cfg.get("max_actions_per_cycle", 8)),
        )
        constrained = bool(controller_cfg.get("constrained", False)) or mode == "full_vara_constrained"
        policy = (
            ConstrainedVARAPolicy(
                base_policy=base_policy,
                min_improvement=float(controller_cfg.get("min_improvement", 0.005)),
                collateral_tolerance=float(controller_cfg.get("collateral_tolerance", 0.10)),
                pressure_collateral_tolerance=float(controller_cfg.get("pressure_collateral_tolerance", 0.05)),
                objective_weights=controller_cfg.get("objective_weights"),
            )
            if constrained
            else base_policy
        )
        self.controller = VARAController(
            initial_weights=config.get("training", {}).get("weights", {}),
            policy=policy,
            constrained=constrained,
        )
        self.actions_logger = JSONListLogger(self.run_dir / "actions.json")
        self.decision_logger = CSVLogger(self.run_dir / "controller_decisions.csv")
        self.accepted_interventions = 0
        self.rejected_interventions = 0
        self.rollback_count = 0
        self._last_decision: dict[str, Any] | None = None
        self.local_controller = LocalVARAController(
            initial_weights=config.get("training", {}).get("weights", {}),
            config=LocalControllerConfig.from_dict(config.get("local_controller", {})),
        )
        self.local_patch_score_logger = CSVLogger(self.run_dir / "patch_scores.csv")
        self.local_action_logger = JSONListLogger(self.run_dir / "local_actions.json")
        self.local_decision_logger = CSVLogger(self.run_dir / "local_controller_decisions.csv")
        self.local_weights_logger = JSONListLogger(self.run_dir / "local_weights_history.json")
        self.local_figure_dir = ensure_dir(self.run_dir / "figures")
        self.local_decisions: list[dict[str, Any]] = []
        self.local_weight_records: list[dict[str, Any]] = []
        self._relobralo_state: ReLoBRaLoWeights | None = None
        self._residual_attention_state: ResidualAttentionState | None = None

    def run(self) -> dict[str, float]:
        self.compute_tracker.start()
        if self.mode == "rar_pinn":
            return self._run_rar()
        if self.mode == "self_adaptive_attention_pinn":
            return self._run_uniform_baseline(self._train_self_adaptive_epochs)
        if self.mode == "gradient_balanced_pinn":
            return self._run_uniform_baseline(self._train_gradient_balanced_epochs)
        if self.mode == "gradient_enhanced_pinn":
            return self._run_uniform_baseline(self.train_epochs)
        if self.mode == "relobralo_pinn":
            return self._run_uniform_baseline(self._train_relobralo_epochs)
        if self.mode == "residual_attention_pinn":
            return self._run_uniform_baseline(self._train_residual_attention_epochs)
        if self.mode == "causal_pinn":
            return self._run_uniform_baseline(self._train_causal_epochs)
        if self.mode in {"local_vara", "local_constrained_vara"}:
            return self._run_local(constrained=self.mode == "local_constrained_vara")
        if self.mode == "full_vara_constrained":
            return self._run_constrained()

        train_cfg = self.config.get("training", {})
        cycles = int(train_cfg.get("adaptive_cycles", 2))
        batch = self.initial_batch()
        adaptive_sampling = self.mode != "full_vara_no_sampling"
        local_weights_enabled = self.mode != "full_vara_no_local_weights"

        for cycle in range(cycles):
            self.train_epochs(batch, self.controller.state if local_weights_enabled else None, cycle=cycle)
            if self.compute_tracker.enabled and self.compute_tracker.exhausted():
                break
            maps, scores, names, weak_regions, X, Y, coords = self.diagnose()
            acceptance = self.controller.evaluate_pending(scores)
            if acceptance is not None:
                self.accept_logger.log(acceptance)
            metrics = self._validation_metrics(coords)
            self.metrics_logger.log({"cycle": cycle, **metrics})
            self.score_logger.log({"cycle": cycle, "diagnostics": names, "scores": scores})
            self.weak_logger.log({"cycle": cycle, "weak_regions": weak_regions})
            save_patch_score_map(scores, names, self.figure_dir / f"patch_scores_cycle_{cycle:03d}.png")

            if self.mode == "vanilla_pinn":
                record = {"cycle": cycle, "stage": "disabled", "weak_regions": [], "actions": []}
            else:
                record = self.controller.step(cycle, weak_regions, scores, names)
            self.action_logger.log(record)
            self.actions_logger.log(record)
            self.action_records.append(record)
            self.maybe_checkpoint(cycle, metrics)
            batch = self.resample_batch(
                batch,
                maps,
                coords,
                weak_regions,
                self.controller.state,
                adaptive=adaptive_sampling and self.mode != "vanilla_pinn",
            )
        self.run_final_physics_repair(cycle=cycles, log_prefix="final_physics_repair")
        metrics = self.evaluate_and_save_final()
        metrics["J_score"] = self._j_score(metrics)
        metrics["accepted_interventions"] = self.accepted_interventions
        metrics["rejected_interventions"] = self.rejected_interventions
        metrics["rollback_count"] = self.rollback_count
        return metrics

    def _run_uniform_baseline(self, epoch_runner: Any) -> dict[str, float]:
        """Run an independent non-controller baseline on uniform samples."""
        train_cfg = self.config.get("training", {})
        cycles = int(train_cfg.get("adaptive_cycles", 2))
        batch = self.initial_batch()
        for cycle in range(cycles):
            epoch_runner(batch, None, cycle=cycle, log_prefix=self.mode)
            if self.compute_tracker.enabled and self.compute_tracker.exhausted():
                break
            maps, scores, names, _weak_regions, _x, _y, coords = self.diagnose()
            metrics = self._validation_metrics(coords)
            self.metrics_logger.log({"cycle": cycle, "phase": self.mode, **metrics})
            self.score_logger.log({"cycle": cycle, "diagnostics": names, "scores": scores})
            self.weak_logger.log({"cycle": cycle, "weak_regions": []})
            record = {"cycle": cycle, "stage": self.mode, "weak_regions": [], "actions": []}
            self.action_logger.log(record)
            self.actions_logger.log(record)
            self.action_records.append(record)
            self.maybe_checkpoint(cycle, metrics)
            batch = self.resample_batch(batch, maps, coords, [], None, adaptive=False)

        self.run_final_physics_repair(cycle=cycles, log_prefix="final_physics_repair")
        metrics = self.evaluate_and_save_final()
        metrics["J_score"] = self._j_score(metrics)
        metrics["accepted_interventions"] = 0
        metrics["rejected_interventions"] = 0
        metrics["rollback_count"] = 0
        metrics["number_of_active_patches"] = 0
        metrics["most_frequently_targeted_variable"] = None
        metrics["most_frequently_targeted_patch"] = None
        pd.DataFrame([metrics]).to_csv(self.run_dir / "summary_table.csv", index=False)
        pd.DataFrame([metrics]).to_csv(self.table_dir / "summary.csv", index=False)
        save_json(metrics, self.run_dir / "summary.json")
        return metrics

    def _train_self_adaptive_epochs(
        self,
        batch: dict[str, Any],
        _control_state: Any | None = None,
        cycle: int = 0,
        epochs_override: int | None = None,
        log_prefix: str = "",
    ) -> dict[str, float]:
        """Train the soft-attention self-adaptive PINN baseline."""
        train_cfg = self.config.get("training", {})
        cfg = self.config.get("self_adaptive_attention", {})
        epochs = int(epochs_override if epochs_override is not None else train_cfg.get("epochs_per_cycle", 100))
        log_every = max(1, int(train_cfg.get("log_every", 25)))
        scalar_weights = dict(train_cfg.get("weights", {}))
        interior_logits = torch.zeros(
            int(batch["xy_f"].shape[0]),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        boundary_logits = torch.zeros(
            int(batch["xy_bc"].shape[0]),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        attention_optimizer = torch.optim.Adam(
            [interior_logits, boundary_logits],
            lr=float(cfg.get("attention_lr", 0.01)),
        )
        last_losses: dict[str, float] = {}
        self.model.train()
        phase_start = time.perf_counter()
        for local_epoch in range(epochs):
            if not self.compute_tracker.can_start_objective(int(batch["xy_f"].shape[0])):
                break
            self.optimizer.zero_grad(set_to_none=True)
            attention_optimizer.zero_grad(set_to_none=True)
            self.compute_tracker.record_objective(batch)
            total, losses, attention_logs = self_adaptive_objective(
                self.model,
                batch,
                self.benchmark,
                self.steady,
                scalar_weights,
                interior_logits,
                boundary_logits,
                maximum_attention=float(cfg.get("maximum_attention", 20.0)),
            )
            anchor_loss, anchor_weight = self.continuation_anchor_loss(batch)
            replay_loss, replay_weight = self.continuation_replay_loss()
            gauge_loss = self.pressure_gauge_loss()
            total = total + anchor_loss + replay_loss + gauge_loss
            attention_logs["continuation_anchor"] = float(anchor_loss.detach().cpu())
            attention_logs["continuation_anchor_weight"] = float(anchor_weight)
            attention_logs["continuation_replay"] = float(replay_loss.detach().cpu())
            attention_logs["continuation_replay_weight"] = float(replay_weight)
            attention_logs["pressure_gauge"] = float(gauge_loss.detach().cpu())
            total.backward()
            grad_norm = self._grad_norm()
            self.optimizer.step()
            for logits in (interior_logits, boundary_logits):
                if logits.grad is not None:
                    logits.grad.mul_(-1.0)
            attention_optimizer.step()
            with torch.no_grad():
                limit = float(cfg.get("logit_limit", 8.0))
                interior_logits.clamp_(-limit, limit)
                boundary_logits.clamp_(-limit, limit)
            self.compute_tracker.record_optimizer_step()
            self.compute_tracker.record_auxiliary_optimizer_step()
            last_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}
            last_losses.update(attention_logs)
            last_losses["total"] = float(total.detach().cpu())
            last_losses["grad_norm"] = grad_norm
            self.last_losses = dict(last_losses)
            if local_epoch % log_every == 0 or local_epoch == epochs - 1:
                self.loss_logger.log(
                    {"cycle": cycle, "phase": log_prefix or self.mode, "epoch": self.global_step, **last_losses}
                )
            self.global_step += 1
        self.compute_tracker.add_phase_time("optimization", time.perf_counter() - phase_start)
        return last_losses

    def _train_gradient_balanced_epochs(
        self,
        batch: dict[str, Any],
        _control_state: Any | None = None,
        cycle: int = 0,
        epochs_override: int | None = None,
        log_prefix: str = "",
    ) -> dict[str, float]:
        """Train with gradient-statistics loss balancing."""
        train_cfg = self.config.get("training", {})
        cfg = self.config.get("gradient_balancing", {})
        epochs = int(epochs_override if epochs_override is not None else train_cfg.get("epochs_per_cycle", 100))
        log_every = max(1, int(train_cfg.get("log_every", 25)))
        update_every = max(1, int(cfg.get("update_every", 10)))
        ema = float(cfg.get("ema", 0.9))
        minimum = float(cfg.get("minimum_weight", 0.05))
        maximum = float(cfg.get("maximum_weight", 20.0))
        base_weights = {
            name: float(value)
            for name, value in dict(train_cfg.get("weights", {})).items()
            if float(value) > 0.0
        }
        adaptive_weights = dict(base_weights)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        last_losses: dict[str, float] = {}
        self.model.train()
        phase_start = time.perf_counter()
        for local_epoch in range(epochs):
            if not self.compute_tracker.can_start_objective(int(batch["xy_f"].shape[0])):
                break
            self.optimizer.zero_grad(set_to_none=True)
            self.compute_tracker.record_objective(batch)
            pointwise = compute_pointwise_losses(self.model, batch, self.benchmark, self.steady)
            losses = compute_global_losses(
                pointwise,
                reduction=str(train_cfg.get("pointwise_reduction", "legacy_mse")),
            )
            active = [name for name in adaptive_weights if name in losses]
            if active and (local_epoch % update_every == 0):
                norms = {name: loss_gradient_norm(losses[name], parameters) for name in active}
                target = float(np.mean([value for value in norms.values() if np.isfinite(value)]))
                proposed = {
                    name: float(np.clip(target / max(norms[name], 1e-12), minimum, maximum))
                    for name in active
                }
                scale = sum(base_weights.get(name, 1.0) for name in active) / max(sum(proposed.values()), 1e-12)
                for name in active:
                    proposal = proposed[name] * scale
                    adaptive_weights[name] = ema * adaptive_weights[name] + (1.0 - ema) * proposal
            total = weighted_sum(losses, adaptive_weights)
            anchor_loss, anchor_weight = self.continuation_anchor_loss(batch)
            replay_loss, replay_weight = self.continuation_replay_loss()
            gauge_loss = self.pressure_gauge_loss()
            total = total + anchor_loss + replay_loss + gauge_loss
            total.backward()
            grad_norm = self._grad_norm()
            self.optimizer.step()
            self.compute_tracker.record_optimizer_step()
            last_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}
            last_losses.update({f"adaptive_weight_{name}": value for name, value in adaptive_weights.items()})
            last_losses["continuation_anchor"] = float(anchor_loss.detach().cpu())
            last_losses["continuation_anchor_weight"] = float(anchor_weight)
            last_losses["continuation_replay"] = float(replay_loss.detach().cpu())
            last_losses["continuation_replay_weight"] = float(replay_weight)
            last_losses["pressure_gauge"] = float(gauge_loss.detach().cpu())
            last_losses["total"] = float(total.detach().cpu())
            last_losses["grad_norm"] = grad_norm
            self.last_losses = dict(last_losses)
            if local_epoch % log_every == 0 or local_epoch == epochs - 1:
                self.loss_logger.log(
                    {"cycle": cycle, "phase": log_prefix or self.mode, "epoch": self.global_step, **last_losses}
                )
            self.global_step += 1
        self.compute_tracker.add_phase_time("optimization", time.perf_counter() - phase_start)
        return last_losses

    def _train_relobralo_epochs(
        self,
        batch: dict[str, Any],
        _control_state: Any | None = None,
        cycle: int = 0,
        epochs_override: int | None = None,
        log_prefix: str = "",
    ) -> dict[str, float]:
        """Train with published relative-loss random-lookback balancing."""
        train_cfg = self.config.get("training", {})
        cfg = self.config.get("relobralo", {})
        epochs = int(epochs_override if epochs_override is not None else train_cfg.get("epochs_per_cycle", 100))
        log_every = max(1, int(train_cfg.get("log_every", 25)))
        active_names = [
            name
            for name, value in dict(train_cfg.get("weights", {})).items()
            if float(value) > 0.0
        ]
        if self._relobralo_state is None:
            self._relobralo_state = ReLoBRaLoWeights(
                active_names,
                temperature=float(cfg.get("temperature", 0.1)),
                alpha=float(cfg.get("alpha", 0.999)),
                random_lookback_probability=float(cfg.get("random_lookback_probability", 0.999)),
                seed=self.seed + 1709,
            )
        last_losses: dict[str, float] = {}
        started = time.perf_counter()
        self.model.train()
        for local_epoch in range(epochs):
            if not self.compute_tracker.can_start_objective(int(batch["xy_f"].shape[0])):
                break
            self.optimizer.zero_grad(set_to_none=True)
            self.compute_tracker.record_objective(batch)
            pointwise = compute_pointwise_losses(self.model, batch, self.benchmark, self.steady)
            losses = compute_global_losses(
                pointwise,
                reduction=str(train_cfg.get("pointwise_reduction", "legacy_mse")),
            )
            available = [name for name in self._relobralo_state.names if name in losses]
            if available != self._relobralo_state.names:
                self._relobralo_state = ReLoBRaLoWeights(
                    available,
                    temperature=float(cfg.get("temperature", 0.1)),
                    alpha=float(cfg.get("alpha", 0.999)),
                    random_lookback_probability=float(cfg.get("random_lookback_probability", 0.999)),
                    seed=self.seed + 1709,
                )
            adaptive = self._relobralo_state.update(losses)
            total = weighted_sum(losses, adaptive)
            anchor_loss, anchor_weight = self.continuation_anchor_loss(batch)
            replay_loss, replay_weight = self.continuation_replay_loss()
            gauge_loss = self.pressure_gauge_loss()
            total = total + anchor_loss + replay_loss + gauge_loss
            total.backward()
            grad_norm = self._grad_norm()
            self.optimizer.step()
            self.compute_tracker.record_optimizer_step()
            last_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}
            last_losses.update({f"relobralo_weight_{name}": value for name, value in adaptive.items()})
            last_losses["continuation_anchor"] = float(anchor_loss.detach().cpu())
            last_losses["continuation_anchor_weight"] = float(anchor_weight)
            last_losses["continuation_replay"] = float(replay_loss.detach().cpu())
            last_losses["continuation_replay_weight"] = float(replay_weight)
            last_losses["pressure_gauge"] = float(gauge_loss.detach().cpu())
            last_losses["total"] = float(total.detach().cpu())
            last_losses["grad_norm"] = grad_norm
            self.last_losses = dict(last_losses)
            if local_epoch % log_every == 0 or local_epoch == epochs - 1:
                self.loss_logger.log(
                    {"cycle": cycle, "phase": log_prefix or self.mode, "epoch": self.global_step, **last_losses}
                )
            self.global_step += 1
        self.compute_tracker.add_phase_time("optimization", time.perf_counter() - started)
        return last_losses

    def _train_residual_attention_epochs(
        self,
        batch: dict[str, Any],
        _control_state: Any | None = None,
        cycle: int = 0,
        epochs_override: int | None = None,
        log_prefix: str = "",
    ) -> dict[str, float]:
        """Train with residual-based point attention on the interior objective."""
        train_cfg = self.config.get("training", {})
        cfg = self.config.get("residual_attention", {})
        epochs = int(epochs_override if epochs_override is not None else train_cfg.get("epochs_per_cycle", 100))
        log_every = max(1, int(train_cfg.get("log_every", 25)))
        weights = dict(train_cfg.get("weights", {}))
        size = int(batch["xy_f"].shape[0])
        if self._residual_attention_state is None or self._residual_attention_state.values.numel() != size:
            self._residual_attention_state = ResidualAttentionState(
                size,
                decay=float(cfg.get("decay", 0.999)),
                eta=float(cfg.get("eta", 0.01)),
                maximum=float(cfg.get("maximum_attention", 10.0)),
            )
        last_losses: dict[str, float] = {}
        started = time.perf_counter()
        self.model.train()
        for local_epoch in range(epochs):
            if not self.compute_tracker.can_start_objective(size):
                break
            self.optimizer.zero_grad(set_to_none=True)
            self.compute_tracker.record_objective(batch)
            pointwise = compute_pointwise_losses(self.model, batch, self.benchmark, self.steady)
            attention = self._residual_attention_state.update(torch.sqrt(pointwise["pde"] + 1e-18))
            losses: dict[str, torch.Tensor] = {}
            for name, values in pointwise.items():
                if name in {"pde", "momentum_u", "momentum_v", "continuity"}:
                    losses[name] = torch.mean(attention.reshape(values.shape) * values)
                else:
                    losses[name] = torch.mean(values)
            total = weighted_sum(losses, weights)
            anchor_loss, anchor_weight = self.continuation_anchor_loss(batch)
            replay_loss, replay_weight = self.continuation_replay_loss()
            gauge_loss = self.pressure_gauge_loss()
            total = total + anchor_loss + replay_loss + gauge_loss
            total.backward()
            grad_norm = self._grad_norm()
            self.optimizer.step()
            self.compute_tracker.record_optimizer_step()
            last_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}
            last_losses["residual_attention_mean"] = float(attention.detach().mean().cpu())
            last_losses["residual_attention_max"] = float(attention.detach().max().cpu())
            last_losses["continuation_anchor"] = float(anchor_loss.detach().cpu())
            last_losses["continuation_anchor_weight"] = float(anchor_weight)
            last_losses["continuation_replay"] = float(replay_loss.detach().cpu())
            last_losses["continuation_replay_weight"] = float(replay_weight)
            last_losses["pressure_gauge"] = float(gauge_loss.detach().cpu())
            last_losses["total"] = float(total.detach().cpu())
            last_losses["grad_norm"] = grad_norm
            self.last_losses = dict(last_losses)
            if local_epoch % log_every == 0 or local_epoch == epochs - 1:
                self.loss_logger.log(
                    {"cycle": cycle, "phase": log_prefix or self.mode, "epoch": self.global_step, **last_losses}
                )
            self.global_step += 1
        self.compute_tracker.add_phase_time("optimization", time.perf_counter() - started)
        return last_losses

    def _train_causal_epochs(
        self,
        batch: dict[str, Any],
        _control_state: Any | None = None,
        cycle: int = 0,
        epochs_override: int | None = None,
        log_prefix: str = "",
    ) -> dict[str, float]:
        """Train a time-dependent PINN with causal temporal residual weights."""
        if batch["xy_f"].shape[1] < 3:
            raise ValueError("causal_pinn requires a time coordinate.")
        train_cfg = self.config.get("training", {})
        cfg = self.config.get("causal_training", {})
        epochs = int(epochs_override if epochs_override is not None else train_cfg.get("epochs_per_cycle", 100))
        log_every = max(1, int(train_cfg.get("log_every", 25)))
        scalar_weights = dict(train_cfg.get("weights", {}))
        last_losses: dict[str, float] = {}
        started = time.perf_counter()
        self.model.train()
        for local_epoch in range(epochs):
            if not self.compute_tracker.can_start_objective(int(batch["xy_f"].shape[0])):
                break
            self.optimizer.zero_grad(set_to_none=True)
            self.compute_tracker.record_objective(batch)
            pointwise = compute_pointwise_losses(self.model, batch, self.benchmark, self.steady)
            total, losses, causal_logs = causal_temporal_objective(
                pointwise,
                batch["xy_f"][:, 2],
                scalar_weights,
                chunks=int(cfg.get("chunks", 16)),
                epsilon=float(cfg.get("epsilon", 1.0)),
            )
            anchor_loss, anchor_weight = self.continuation_anchor_loss(batch)
            replay_loss, replay_weight = self.continuation_replay_loss()
            gauge_loss = self.pressure_gauge_loss()
            total = total + anchor_loss + replay_loss + gauge_loss
            total.backward()
            grad_norm = self._grad_norm()
            self.optimizer.step()
            self.compute_tracker.record_optimizer_step()
            last_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}
            last_losses.update(causal_logs)
            last_losses["continuation_anchor"] = float(anchor_loss.detach().cpu())
            last_losses["continuation_anchor_weight"] = float(anchor_weight)
            last_losses["continuation_replay"] = float(replay_loss.detach().cpu())
            last_losses["continuation_replay_weight"] = float(replay_weight)
            last_losses["pressure_gauge"] = float(gauge_loss.detach().cpu())
            last_losses["total"] = float(total.detach().cpu())
            last_losses["grad_norm"] = grad_norm
            self.last_losses = dict(last_losses)
            if local_epoch % log_every == 0 or local_epoch == epochs - 1:
                self.loss_logger.log(
                    {"cycle": cycle, "phase": log_prefix or self.mode, "epoch": self.global_step, **last_losses}
                )
            self.global_step += 1
        self.compute_tracker.add_phase_time("optimization", time.perf_counter() - started)
        return last_losses

    def _run_rar(self) -> dict[str, float]:
        """Residual adaptive refinement baseline without VARA controller logic."""
        train_cfg = self.config.get("training", {})
        cycles = int(train_cfg.get("adaptive_cycles", 2))
        batch = self.initial_batch()
        rng = np.random.default_rng(self.seed + 424242)

        for cycle in range(cycles):
            self.train_epochs(batch, None, cycle=cycle, log_prefix="rar_main")
            if self.compute_tracker.enabled and self.compute_tracker.exhausted():
                break
            maps, scores, names, _weak_regions, X, Y, coords = self.diagnose()
            metrics = self._validation_metrics(coords)
            self.metrics_logger.log({"cycle": cycle, "phase": "rar_after", **metrics})
            self.score_logger.log({"cycle": cycle, "diagnostics": names, "scores": scores})
            self.weak_logger.log({"cycle": cycle, "weak_regions": []})
            save_patch_score_map(scores, names, self.figure_dir / f"patch_scores_cycle_{cycle:03d}.png")
            record = {"cycle": cycle, "stage": "rar_residual_refinement", "weak_regions": [], "actions": []}
            self.action_logger.log(record)
            self.actions_logger.log(record)
            self.action_records.append(record)
            self.maybe_checkpoint(cycle, metrics)
            batch = self._resample_rar_batch(maps, coords, rng)

        self.run_final_physics_repair(cycle=cycles, log_prefix="final_physics_repair")
        metrics = self.evaluate_and_save_final()
        metrics["J_score"] = self._j_score(metrics)
        metrics["accepted_interventions"] = 0
        metrics["rejected_interventions"] = 0
        metrics["rollback_count"] = 0
        metrics["number_of_active_patches"] = 0
        metrics["most_frequently_targeted_variable"] = None
        metrics["most_frequently_targeted_patch"] = None
        pd.DataFrame([metrics]).to_csv(self.run_dir / "summary_table.csv", index=False)
        pd.DataFrame([metrics]).to_csv(self.table_dir / "summary.csv", index=False)
        save_json(metrics, self.run_dir / "summary.json")
        return metrics

    def _resample_rar_batch(
        self,
        maps: dict[str, np.ndarray],
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Resample collocation points from PDE residuals only, with no controller state."""
        train_cfg = self.config.get("training", {})
        rar_cfg = self.config.get("rar", {})
        n_f = int(train_cfg.get("n_collocation", 1024))
        n_bc = int(train_cfg.get("n_boundary", 256))
        n_data = int(train_cfg.get("n_data", 256))
        residual_fraction = float(np.clip(float(rar_cfg.get("residual_fraction", 0.5)), 0.0, 1.0))
        n_residual = int(round(n_f * residual_fraction))
        n_uniform = n_f - n_residual
        pieces = []
        if n_uniform > 0:
            pieces.append(self.uniform_sampler.sample(n_uniform).detach().cpu().numpy())
        score = maps.get("aggregate_pde_residual", maps.get("pde_residual"))
        if n_residual > 0 and score is not None:
            pieces.append(sample_from_score_grid(coords, score, n_residual, rng))
        elif n_residual > 0:
            pieces.append(self.uniform_sampler.sample(n_residual).detach().cpu().numpy())
        xy_f_np = np.vstack([piece for piece in pieces if piece.size])
        rng.shuffle(xy_f_np)
        xy_f = torch.tensor(xy_f_np, dtype=torch.float32, device=self.device)
        xy_bc = self._sample_boundary(n_bc)
        xy_data = self.uniform_sampler.sample(n_data)
        return self.make_batch(xy_f, xy_bc, xy_data)

    def _run_constrained(self) -> dict[str, float]:
        """Constrained VARA: try, evaluate, accept/rollback, then continue."""
        train_cfg = self.config.get("training", {})
        controller_cfg = self.config.get("controller", {})
        cycles = int(train_cfg.get("adaptive_cycles", 2))
        trial_epochs = int(controller_cfg.get("trial_epochs", max(10, int(train_cfg.get("epochs_per_cycle", 100)) // 10)))
        main_epochs = max(1, int(train_cfg.get("epochs_per_cycle", 100)) - trial_epochs)
        batch = self.initial_batch()

        for cycle in range(cycles):
            self.train_epochs(batch, self.controller.state, cycle=cycle, epochs_override=main_epochs, log_prefix="main")
            if self.compute_tracker.enabled and self.compute_tracker.exhausted():
                break
            maps_before, scores_before, names, weak_regions, X, Y, coords = self.diagnose()
            metrics_before = self._validation_metrics(coords)
            self.metrics_logger.log({"cycle": cycle, "phase": "before_trial", **metrics_before, "J_score": self._j_score(metrics_before)})
            self.score_logger.log({"cycle": cycle, "diagnostics": names, "scores": scores_before})
            self.weak_logger.log({"cycle": cycle, "weak_regions": weak_regions})
            save_patch_score_map(scores_before, names, self.figure_dir / f"patch_scores_cycle_{cycle:03d}.png")

            model_snapshot = self._model_snapshot()
            optimizer_snapshot = deepcopy(self.optimizer.state_dict())
            trial = self.controller.propose_trial(
                cycle=cycle,
                weak_regions=weak_regions,
                diagnostic_names=names,
                previous_decision=self._last_decision,
            )

            self.train_epochs(batch, self.controller.state, cycle=cycle, epochs_override=trial_epochs, log_prefix="trial")
            _, _, coords_after = self.validation_grid()
            metrics_after = self._validation_metrics(coords_after)
            accepted, decision_metrics = self.controller.policy.accept_metrics(
                metrics_before,
                metrics_after,
                trial["targeted_metric"],
            )
            pressure_worsened = metrics_after["p_rel_l2_centered"] > metrics_before["p_rel_l2_centered"] * 1.05
            rollback = not accepted
            if rollback:
                self._restore_model_snapshot(model_snapshot)
                self.optimizer.load_state_dict(optimizer_snapshot)
                self.controller.rollback_trial(trial)
                self.rollback_count += 1
                self.rejected_interventions += 1
            else:
                self.accepted_interventions += 1
                if pressure_worsened:
                    trial["new_weights"] = self.controller.apply_pressure_protection()
                    trial["new_local_weights"] = {k: dict(v) for k, v in self.controller.state.local_weights.items()}

            repeat_count = self._repeat_count(trial["targeted_variable"])
            decision = {
                "cycle": cycle,
                "accepted": bool(accepted),
                "rejected": bool(not accepted),
                "rollback_triggered": bool(rollback),
                "targeted_variable": trial["targeted_variable"],
                "targeted_patch": trial["targeted_patch"],
                "targeted_metric": trial["targeted_metric"],
                "intervention_strength": self._max_action_strength(trial),
                "strength_factor": trial["strength_factor"],
                "repeat_count": repeat_count,
                "pressure_worsened": bool(pressure_worsened),
                "old_weights": trial["old_weights"],
                "new_weights": dict(self.controller.state.global_weights),
                "old_local_weights": trial.get("old_local_weights", {}),
                "new_local_weights": {k: dict(v) for k, v in self.controller.state.local_weights.items()},
                "old_sampling_priorities": trial.get("old_sampling_priorities", {}),
                "new_sampling_priorities": dict(self.controller.state.sampling_priorities),
                **decision_metrics,
            }
            self.controller.record_decision(decision)
            self._last_decision = decision
            self.action_logger.log({**trial, "decision": decision})
            self.actions_logger.log({**trial, "decision": decision})
            self.action_records.append({**trial, "decision": decision})
            self.decision_logger.log(self._flat_decision(decision))

            metrics_kept = metrics_after if accepted else metrics_before
            self.metrics_logger.log({"cycle": cycle, "phase": "after_trial", **metrics_kept, "J_score": self._j_score(metrics_kept)})
            self.maybe_checkpoint(cycle, metrics_kept)
            batch = self.resample_batch(
                batch,
                maps_before,
                coords,
                weak_regions if accepted else [],
                self.controller.state,
                adaptive=accepted,
            )

        self.run_final_physics_repair(cycle=cycles, log_prefix="final_physics_repair")
        metrics = self.evaluate_and_save_final()
        metrics["J_score"] = self._j_score(metrics)
        metrics["accepted_interventions"] = self.accepted_interventions
        metrics["rejected_interventions"] = self.rejected_interventions
        metrics["rollback_count"] = self.rollback_count
        pd.DataFrame([metrics]).to_csv(self.run_dir / "summary_table.csv", index=False)
        pd.DataFrame([metrics]).to_csv(self.table_dir / "summary.csv", index=False)
        save_json(metrics, self.run_dir / "summary.json")
        return metrics

    def _run_local(self, constrained: bool) -> dict[str, float]:
        """Run local variable-region control without changing vanilla/full modes."""
        train_cfg = self.config.get("training", {})
        cycles = int(train_cfg.get("adaptive_cycles", 2))
        epochs_per_cycle = int(train_cfg.get("epochs_per_cycle", 100))
        trial_epochs = self.local_controller.config.trial_epochs if constrained else 0
        warmup_cycles = max(0, int(self.local_controller.config.warmup_cycles))
        main_epochs = max(1, epochs_per_cycle - trial_epochs)
        rejection_recovery_epochs = max(0, int(self.local_controller.config.rejection_recovery_epochs))
        batch = self.initial_batch()
        save_patch_grid_overlay(self.patch_grid, self.local_figure_dir / "patch_grid.png")

        for cycle in range(cycles):
            cycle_main_epochs = epochs_per_cycle if cycle < warmup_cycles else main_epochs
            self.train_epochs(
                batch,
                self.local_controller.state,
                cycle=cycle,
                epochs_override=cycle_main_epochs,
                log_prefix="local_main",
            )
            if self.compute_tracker.enabled and self.compute_tracker.exhausted():
                break
            maps_before, norm_before, raw_before, names, weak_regions, X, Y, coords = self._diagnose_local(update_ema=True)
            metrics_before = self._validation_metrics(coords)
            self._log_patch_scores(cycle, names, raw_before, norm_before)
            self._save_local_patch_figures(cycle, names, raw_before)

            if cycle < warmup_cycles:
                controller_snapshot = self.local_controller.snapshot()
                decision_metrics = self.local_controller.evaluate_acceptance(
                    [],
                    raw_before,
                    raw_before,
                    names,
                    metrics_before,
                    metrics_before,
                    constrained=False,
                )[1]
                decision = self._build_local_decision(
                    cycle=cycle,
                    constrained=constrained,
                    accepted=True,
                    interventions=[],
                    old_local_weights=deepcopy(controller_snapshot["local_weights"]),
                    old_sampling=deepcopy(controller_snapshot["sampling_priorities"]),
                    metrics_before=metrics_before,
                    metrics_after=metrics_before,
                    raw_before=raw_before,
                    raw_after=raw_before,
                    diagnostic_names=names,
                    decision_metrics=decision_metrics,
                    rejection_reason="warmup_no_intervention",
                )
                decision["warmup_skip"] = True
                self.local_decisions.append(decision)
                self.local_action_logger.log({"cycle": cycle, "mode": self.mode, "actions": [], "decision": decision})
                self.local_decision_logger.log(self._flat_local_decision(decision))
                self._log_local_weights(cycle)
                save_active_intervention_map(
                    [],
                    self.patch_grid,
                    self.local_figure_dir / f"local_action_map_cycle_{cycle:03d}.png",
                    f"Local actions cycle {cycle} warmup",
                )
                self.metrics_logger.log(
                    {"cycle": cycle, "phase": "local_warmup", **metrics_before, "J_score": self.local_controller.objective(metrics_before)}
                )
                self.maybe_checkpoint(cycle, metrics_before)
                batch = self._resample_local_batch(maps_before, coords, [], adaptive=False)
                continue

            interventions = self.local_controller.propose(weak_regions)
            cycle_action_records: list[dict[str, Any]] = []
            accepted_interventions: list[Any] = []
            kept_metrics = metrics_before

            if constrained:
                for candidate_id, intervention in enumerate(interventions):
                    if self.compute_tracker.enabled and self.compute_tracker.exhausted():
                        break
                    (
                        _maps_candidate_before,
                        _norm_candidate_before,
                        raw_candidate_before,
                        _names_candidate_before,
                        _weak_candidate_before,
                        _x_candidate_before,
                        _y_candidate_before,
                        coords_candidate_before,
                    ) = self._diagnose_local(update_ema=False, detect=False)
                    metrics_candidate_before = self._validation_metrics(coords_candidate_before)
                    controller_snapshot = self.local_controller.snapshot()
                    model_snapshot = self._model_snapshot()
                    optimizer_snapshot = deepcopy(self.optimizer.state_dict())
                    old_local_weights = deepcopy(controller_snapshot["local_weights"])
                    old_sampling = deepcopy(controller_snapshot["sampling_priorities"])
                    strength_before = self._pair_strength(intervention.variable, intervention.patch_id)

                    self.local_controller.apply([intervention])
                    self.train_epochs(
                        batch,
                        self.local_controller.state,
                        cycle=cycle,
                        epochs_override=trial_epochs,
                        log_prefix=f"local_trial_{candidate_id}",
                    )
                    _, _, raw_candidate_after, _, _, _, _, coords_candidate_after = self._diagnose_local(
                        update_ema=False,
                        detect=False,
                    )
                    metrics_candidate_after = self._validation_metrics(coords_candidate_after)
                    accepted, decision_metrics = self.local_controller.evaluate_acceptance(
                        [intervention],
                        raw_candidate_before,
                        raw_candidate_after,
                        names,
                        metrics_candidate_before,
                        metrics_candidate_after,
                        constrained=True,
                    )
                    rejection_reason = self._local_rejection_reason(accepted, decision_metrics)

                    if accepted:
                        self.local_controller.mark_accepted(intervention, decision_metrics)
                        self.accepted_interventions += 1
                        kept_metrics = metrics_candidate_after
                        accepted_interventions.append(intervention)
                    else:
                        self._restore_model_snapshot(model_snapshot)
                        self.optimizer.load_state_dict(optimizer_snapshot)
                        self.local_controller.rollback(controller_snapshot)
                        self.local_controller.mark_rejected(intervention, decision_metrics)
                        if rejection_recovery_epochs > 0:
                            self.train_epochs(
                                batch,
                                self.local_controller.state,
                                cycle=cycle,
                                epochs_override=rejection_recovery_epochs,
                                log_prefix=f"local_recovery_{candidate_id}",
                            )
                            _, _, _, _, _, _, _, coords_recovered = self._diagnose_local(update_ema=False, detect=False)
                            recovered_metrics = self._validation_metrics(coords_recovered)
                            kept_metrics = recovered_metrics
                        self.rejected_interventions += 1
                        self.rollback_count += 1
                        if rejection_recovery_epochs == 0:
                            kept_metrics = metrics_candidate_before

                    strength_after = self._pair_strength(intervention.variable, intervention.patch_id)
                    decision = self._build_local_decision(
                        cycle=cycle,
                        constrained=constrained,
                        accepted=accepted,
                        interventions=[intervention],
                        old_local_weights=old_local_weights,
                        old_sampling=old_sampling,
                        metrics_before=metrics_candidate_before,
                        metrics_after=metrics_candidate_after,
                        raw_before=raw_candidate_before,
                        raw_after=raw_candidate_after,
                        diagnostic_names=names,
                        decision_metrics=decision_metrics,
                        candidate_id=candidate_id,
                        rejection_reason=rejection_reason,
                        strength_before=strength_before,
                        strength_after=strength_after,
                    )
                    self.local_controller.record_decision([intervention], decision)
                    self.local_decisions.append(decision)
                    action_record = intervention.to_record()
                    action_record["candidate_id"] = candidate_id
                    action_record["accepted"] = bool(accepted)
                    cycle_action_records.append(action_record)
                    self.local_action_logger.log(
                        {"cycle": cycle, "mode": self.mode, "candidate_id": candidate_id, "actions": [action_record], "decision": decision}
                    )
                    self.local_decision_logger.log(self._flat_local_decision(decision))
                if not interventions and trial_epochs > 0:
                    self.train_epochs(
                        batch,
                        self.local_controller.state,
                        cycle=cycle,
                        epochs_override=trial_epochs,
                        log_prefix="local_no_action_remainder",
                    )
                    _, _, _, _, _, _, _, coords_remainder = self._diagnose_local(update_ema=False, detect=False)
                    kept_metrics = self._validation_metrics(coords_remainder)
            else:
                controller_snapshot = self.local_controller.snapshot()
                old_local_weights = deepcopy(controller_snapshot["local_weights"])
                old_sampling = deepcopy(controller_snapshot["sampling_priorities"])
                self.local_controller.apply(interventions)
                raw_after = raw_before
                metrics_after = metrics_before
                accepted = True
                kept_metrics = metrics_before
                decision_metrics = self.local_controller.evaluate_acceptance(
                    interventions,
                    raw_before,
                    raw_after,
                    names,
                    metrics_before,
                    metrics_after,
                    constrained=False,
                )[1]
                self.accepted_interventions += len(interventions)
                accepted_interventions = list(interventions)
                action_records = [intervention.to_record() for intervention in interventions]
                cycle_action_records.extend(action_records)
                decision = self._build_local_decision(
                    cycle=cycle,
                    constrained=constrained,
                    accepted=accepted,
                    interventions=interventions,
                    old_local_weights=old_local_weights,
                    old_sampling=old_sampling,
                    metrics_before=metrics_before,
                    metrics_after=metrics_after,
                    raw_before=raw_before,
                    raw_after=raw_after,
                    diagnostic_names=names,
                    decision_metrics=decision_metrics,
                )
                self.local_controller.record_decision(interventions, decision)
                self.local_decisions.append(decision)
                self.local_action_logger.log({"cycle": cycle, "mode": self.mode, "actions": action_records, "decision": decision})
                self.local_decision_logger.log(self._flat_local_decision(decision))

            self._log_local_weights(cycle)
            save_active_intervention_map(
                cycle_action_records,
                self.patch_grid,
                self.local_figure_dir / f"local_action_map_cycle_{cycle:03d}.png",
                f"Local actions cycle {cycle}",
            )

            self.metrics_logger.log({"cycle": cycle, "phase": "local_after", **kept_metrics, "J_score": self.local_controller.objective(kept_metrics)})
            self.maybe_checkpoint(cycle, kept_metrics)
            batch = self._resample_local_batch(
                maps_before,
                coords,
                accepted_interventions,
                adaptive=bool(accepted_interventions),
            )

        save_local_weight_evolution(self.local_weight_records, self.local_figure_dir / "local_weight_evolution.png")
        save_accept_reject_counts(self.local_decisions, self.local_figure_dir / "accepted_vs_rejected.png")
        save_targeted_patch_improvement(self.local_decisions, self.local_figure_dir / "targeted_patch_improvement.png")
        save_collateral_damage_timeline(self.local_decisions, self.local_figure_dir / "collateral_damage_timeline.png")
        save_pressure_collateral_timeline(self.local_decisions, self.local_figure_dir / "pressure_collateral_timeline.png")
        self.run_final_physics_repair(cycle=cycles, log_prefix="final_physics_repair")
        metrics = self.evaluate_and_save_final()
        metrics["J_score"] = self.local_controller.objective(metrics)
        metrics["accepted_interventions"] = self.accepted_interventions
        metrics["rejected_interventions"] = self.rejected_interventions
        metrics["rollback_count"] = self.rollback_count
        metrics["number_of_active_patches"] = len(self.local_controller.active_patches())
        metrics["most_frequently_targeted_variable"] = self._most_frequent("targeted_variables")
        metrics["most_frequently_targeted_patch"] = self._most_frequent("targeted_patches")
        pd.DataFrame([metrics]).to_csv(self.run_dir / "summary_table.csv", index=False)
        pd.DataFrame([metrics]).to_csv(self.table_dir / "summary.csv", index=False)
        save_json(metrics, self.run_dir / "summary.json")
        return metrics

    def _validation_metrics(self, coords: np.ndarray) -> dict[str, float]:
        phase_start = time.perf_counter()
        metrics = self.controller_metrics(coords)
        self.compute_tracker.add_phase_time("evaluation", time.perf_counter() - phase_start)
        return metrics

    def _diagnose_local(
        self,
        update_ema: bool,
        detect: bool = True,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[str], list[Any], np.ndarray, np.ndarray, np.ndarray]:
        phase_start = time.perf_counter()
        X, Y, coords = self.validation_grid()
        builder = self.diagnostic_builder()
        maps = builder.build(coords, mode=self.controller_diagnostic_mode())
        norm_scores, names = self.patch_scorer.compute(maps, coords, update_ema=update_ema)
        raw_scores = self.patch_scorer.last_raw_scores
        if raw_scores is None:
            raw_scores = norm_scores
        weak_regions = self.weak_detector.detect(norm_scores, names, self.patch_grid) if detect else []
        self.compute_tracker.add_phase_time("diagnostics", time.perf_counter() - phase_start)
        return maps, norm_scores, raw_scores, names, weak_regions, X, Y, coords

    def _log_patch_scores(self, cycle: int, names: list[str], raw_scores: np.ndarray, norm_scores: np.ndarray) -> None:
        for j, name in enumerate(names):
            for patch_id in range(self.patch_grid.num_patches):
                self.local_patch_score_logger.log(
                    {
                        "cycle": cycle,
                        "mode": self.mode,
                        "variable": name,
                        "patch_id": patch_id,
                        "raw_score": float(raw_scores[j, patch_id]),
                        "normalized_score": float(norm_scores[j, patch_id]),
                    }
                )

    def _save_local_patch_figures(self, cycle: int, names: list[str], raw_scores: np.ndarray) -> None:
        requested = {
            "u_error": "u_error_patch_heatmap",
            "v_error": "v_error_patch_heatmap",
            "p_error_mean_centered": "pressure_error_patch_heatmap",
            "omega_error": "omega_error_patch_heatmap",
            "aggregate_pde_residual": "aggregate_pde_patch_heatmap",
        }
        name_to_idx = {name: i for i, name in enumerate(names)}
        for name, stem in requested.items():
            if name in name_to_idx:
                save_patch_scalar_heatmap(
                    raw_scores[name_to_idx[name]],
                    self.patch_grid,
                    self.local_figure_dir / f"{stem}_cycle_{cycle:03d}.png",
                    f"{name} by patch cycle {cycle}",
                )

    def _build_local_decision(
        self,
        cycle: int,
        constrained: bool,
        accepted: bool,
        interventions: list[Any],
        old_local_weights: dict[str, dict[int, float]],
        old_sampling: dict[int, float],
        metrics_before: dict[str, float],
        metrics_after: dict[str, float],
        raw_before: np.ndarray,
        raw_after: np.ndarray,
        diagnostic_names: list[str],
        decision_metrics: dict[str, Any],
        candidate_id: int | None = None,
        rejection_reason: str = "",
        strength_before: float | None = None,
        strength_after: float | None = None,
    ) -> dict[str, Any]:
        target_before, target_after = self._target_patch_scores(interventions, raw_before, raw_after, diagnostic_names)
        first = interventions[0] if interventions else None
        return {
            "cycle": cycle,
            "mode": self.mode,
            "candidate_id": candidate_id,
            "constrained": constrained,
            "accepted": bool(accepted),
            "rejected": bool(not accepted),
            "rollback_triggered": bool(constrained and not accepted),
            "rejection_reason": rejection_reason,
            "variable": first.variable if first is not None else "",
            "patch_id": first.patch_id if first is not None else None,
            "selected_pairs": [intervention.to_record() for intervention in interventions],
            "targeted_variables": [intervention.variable for intervention in interventions],
            "targeted_patches": [intervention.patch_id for intervention in interventions],
            "active_patches": sorted(self.local_controller.active_patches()),
            "local_weights_before": old_local_weights,
            "local_weights_after": deepcopy(self.local_controller.state.local_weights),
            "sampling_before": old_sampling,
            "sampling_after": deepcopy(self.local_controller.state.sampling_priorities),
            "action_type": ",".join(intervention.action for intervention in interventions),
            "intervention_strength": max([intervention.strength for intervention in interventions], default=0.0),
            "strength_before": strength_before,
            "strength_after": strength_after,
            "target_local_error_before": target_before,
            "target_local_error_after": target_after,
            "u_rel_l2_before": metrics_before.get("u_rel_l2"),
            "u_rel_l2_after": metrics_after.get("u_rel_l2"),
            "v_rel_l2_before": metrics_before.get("v_rel_l2"),
            "v_rel_l2_after": metrics_after.get("v_rel_l2"),
            "p_rel_l2_centered_before": metrics_before.get("p_rel_l2_centered"),
            "p_rel_l2_centered_after": metrics_after.get("p_rel_l2_centered"),
            "omega_rel_l2_before": metrics_before.get("omega_rel_l2"),
            "omega_rel_l2_after": metrics_after.get("omega_rel_l2"),
            "pde_residual_mean_before": metrics_before.get("pde_residual_mean"),
            "pde_residual_mean_after": metrics_after.get("pde_residual_mean"),
            "continuity_residual_mean_before": metrics_before.get("continuity_residual_mean"),
            "continuity_residual_mean_after": metrics_after.get("continuity_residual_mean"),
            "momentum_residual_mean_before": metrics_before.get("momentum_residual_mean"),
            "momentum_residual_mean_after": metrics_after.get("momentum_residual_mean"),
            "boundary_condition_error_before": metrics_before.get("boundary_condition_error"),
            "boundary_condition_error_after": metrics_after.get("boundary_condition_error"),
            "u_boundary_rmse_before": metrics_before.get("u_boundary_rmse"),
            "u_boundary_rmse_after": metrics_after.get("u_boundary_rmse"),
            "v_boundary_rmse_before": metrics_before.get("v_boundary_rmse"),
            "v_boundary_rmse_after": metrics_after.get("v_boundary_rmse"),
            "centerline_pde_residual_mean_before": metrics_before.get("centerline_pde_residual_mean"),
            "centerline_pde_residual_mean_after": metrics_after.get("centerline_pde_residual_mean"),
            "centerline_continuity_residual_mean_before": metrics_before.get("centerline_continuity_residual_mean"),
            "centerline_continuity_residual_mean_after": metrics_after.get("centerline_continuity_residual_mean"),
            "corner_pde_residual_mean_before": metrics_before.get("corner_pde_residual_mean"),
            "corner_pde_residual_mean_after": metrics_after.get("corner_pde_residual_mean"),
            "corner_boundary_error_before": metrics_before.get("corner_boundary_error"),
            "corner_boundary_error_after": metrics_after.get("corner_boundary_error"),
            "unweighted_validation_loss_before": metrics_before.get("unweighted_validation_loss"),
            "unweighted_validation_loss_after": metrics_after.get("unweighted_validation_loss"),
            **decision_metrics,
        }

    def _target_patch_scores(
        self,
        interventions: list[Any],
        raw_before: np.ndarray,
        raw_after: np.ndarray,
        diagnostic_names: list[str],
    ) -> tuple[float, float]:
        name_to_idx = {name: i for i, name in enumerate(diagnostic_names)}
        before_values = []
        after_values = []
        for intervention in interventions:
            if intervention.variable not in name_to_idx:
                continue
            j = name_to_idx[intervention.variable]
            before_values.append(float(raw_before[j, intervention.patch_id]))
            after_values.append(float(raw_after[j, intervention.patch_id]))
        before = float(np.mean(before_values)) if before_values else 0.0
        after = float(np.mean(after_values)) if after_values else before
        return before, after

    def _flat_local_decision(self, decision: dict[str, Any]) -> dict[str, Any]:
        return {
            "cycle": decision["cycle"],
            "mode": decision["mode"],
            "candidate_id": decision.get("candidate_id"),
            "variable": decision.get("variable"),
            "patch_id": decision.get("patch_id"),
            "accepted": decision["accepted"],
            "rejected": decision["rejected"],
            "rollback_triggered": decision["rollback_triggered"],
            "warmup_skip": decision.get("warmup_skip", False),
            "rejection_reason": decision.get("rejection_reason", ""),
            "targeted_variables": ",".join(decision["targeted_variables"]),
            "targeted_patches": ",".join(str(pid) for pid in decision["targeted_patches"]),
            "active_patches": ",".join(str(pid) for pid in decision["active_patches"]),
            "action_type": decision["action_type"],
            "intervention_strength": decision["intervention_strength"],
            "strength_before": decision.get("strength_before"),
            "strength_after": decision.get("strength_after"),
            "target_local_error_before": decision["target_local_error_before"],
            "target_local_error_after": decision["target_local_error_after"],
            "target_local_improvement": decision.get("target_local_improvement"),
            "u_rel_l2_before": decision["u_rel_l2_before"],
            "u_rel_l2_after": decision["u_rel_l2_after"],
            "v_rel_l2_before": decision["v_rel_l2_before"],
            "v_rel_l2_after": decision["v_rel_l2_after"],
            "p_rel_l2_centered_before": decision["p_rel_l2_centered_before"],
            "p_rel_l2_centered_after": decision["p_rel_l2_centered_after"],
            "omega_rel_l2_before": decision["omega_rel_l2_before"],
            "omega_rel_l2_after": decision["omega_rel_l2_after"],
            "pde_residual_mean_before": decision["pde_residual_mean_before"],
            "pde_residual_mean_after": decision["pde_residual_mean_after"],
            "continuity_residual_mean_before": decision.get("continuity_residual_mean_before"),
            "continuity_residual_mean_after": decision.get("continuity_residual_mean_after"),
            "momentum_residual_mean_before": decision.get("momentum_residual_mean_before"),
            "momentum_residual_mean_after": decision.get("momentum_residual_mean_after"),
            "boundary_condition_error_before": decision.get("boundary_condition_error_before"),
            "boundary_condition_error_after": decision.get("boundary_condition_error_after"),
            "u_boundary_rmse_before": decision.get("u_boundary_rmse_before"),
            "u_boundary_rmse_after": decision.get("u_boundary_rmse_after"),
            "v_boundary_rmse_before": decision.get("v_boundary_rmse_before"),
            "v_boundary_rmse_after": decision.get("v_boundary_rmse_after"),
            "centerline_pde_residual_mean_before": decision.get("centerline_pde_residual_mean_before"),
            "centerline_pde_residual_mean_after": decision.get("centerline_pde_residual_mean_after"),
            "centerline_continuity_residual_mean_before": decision.get("centerline_continuity_residual_mean_before"),
            "centerline_continuity_residual_mean_after": decision.get("centerline_continuity_residual_mean_after"),
            "corner_pde_residual_mean_before": decision.get("corner_pde_residual_mean_before"),
            "corner_pde_residual_mean_after": decision.get("corner_pde_residual_mean_after"),
            "corner_boundary_error_before": decision.get("corner_boundary_error_before"),
            "corner_boundary_error_after": decision.get("corner_boundary_error_after"),
            "unweighted_validation_loss_before": decision.get("unweighted_validation_loss_before"),
            "unweighted_validation_loss_after": decision.get("unweighted_validation_loss_after"),
            "J_before": decision["J_before"],
            "J_after": decision["J_after"],
            "max_collateral_damage": decision["max_collateral_damage"],
            "pressure_collateral_damage": decision["pressure_collateral_damage"],
            "continuity_collateral_damage": decision.get("continuity_collateral_damage"),
            "boundary_collateral_damage": decision.get("boundary_collateral_damage"),
            "validation_loss_damage": decision.get("validation_loss_damage"),
            "pde_hard_damage": decision.get("pde_hard_damage"),
            "boundary_hard_damage": decision.get("boundary_hard_damage"),
            "validation_loss_hard_damage": decision.get("validation_loss_hard_damage"),
            "centerline_pde_hard_damage": decision.get("centerline_pde_hard_damage"),
            "corner_boundary_hard_damage": decision.get("corner_boundary_hard_damage"),
            "u_boundary_hard_damage": decision.get("u_boundary_hard_damage"),
            "v_boundary_hard_damage": decision.get("v_boundary_hard_damage"),
        }

    def _log_local_weights(self, cycle: int) -> None:
        record = {
            "cycle": cycle,
            "mode": self.mode,
            "local_weights": deepcopy(self.local_controller.state.local_weights),
            "sampling_priorities": deepcopy(self.local_controller.state.sampling_priorities),
            "pair_state": self.local_controller.pair_state_record(),
        }
        self.local_weight_records.append(record)
        self.local_weights_logger.log(record)

    def _resample_local_batch(
        self,
        maps: dict[str, np.ndarray],
        coords: np.ndarray,
        weak_regions: list[Any],
        adaptive: bool,
    ) -> dict[str, Any]:
        train_cfg = self.config.get("training", {})
        n_f = int(train_cfg.get("n_collocation", 1024))
        n_bc = int(train_cfg.get("n_boundary", 256))
        n_data = int(train_cfg.get("n_data", 256))
        if adaptive:
            xy_f = self.adaptive_sampler.sample_interior(
                n_f,
                maps,
                coords,
                weak_regions,
                self.local_controller.state.sampling_priorities,
            )
            xy_data = self.adaptive_sampler.sample_interior(
                n_data,
                maps,
                coords,
                weak_regions,
                self.local_controller.state.sampling_priorities,
            )
            boundary_frac = float(self.config.get("sampling", {}).get("mixture", {}).get("boundary", 0.05))
            n_focus = int(n_bc * boundary_frac)
            patch_ids = sorted(self.local_controller.active_patches())
            if n_focus > 0 and patch_ids:
                xy_bc_np = np.vstack(
                    [
                        self._sample_boundary_numpy(n_bc - n_focus),
                        self.boundary_sampler.sample_patch_numpy(self.patch_grid, patch_ids, n_focus),
                    ]
                )
                xy_bc = torch.tensor(xy_bc_np, dtype=torch.float32, device=self.device)
            else:
                xy_bc = self._sample_boundary(n_bc)
        else:
            xy_f = self.uniform_sampler.sample(n_f)
            xy_data = self.uniform_sampler.sample(n_data)
            xy_bc = self._sample_boundary(n_bc)
        return self.make_batch(xy_f, xy_bc, xy_data)

    def _most_frequent(self, key: str) -> str | int | None:
        counts: dict[Any, int] = {}
        for decision in self.local_decisions:
            for item in decision.get(key, []):
                counts[item] = counts.get(item, 0) + 1
        if not counts:
            return None
        return max(counts, key=counts.get)

    def _pair_strength(self, variable: str, patch_id: int) -> float:
        key = (variable, int(patch_id))
        state = self.local_controller.pair_state.get(key)
        return float(state.strength) if state is not None else float(self.local_controller.config.initial_strength)

    def _local_rejection_reason(self, accepted: bool, decision_metrics: dict[str, Any]) -> str:
        if accepted:
            return ""
        reasons = []
        cfg = self.local_controller.config
        target = float(decision_metrics.get("target_local_improvement", 0.0))
        j_before = float(decision_metrics.get("J_before", 0.0))
        j_after = float(decision_metrics.get("J_after", 0.0))
        max_collateral = float(decision_metrics.get("max_collateral_damage", 0.0))
        pressure_collateral = float(decision_metrics.get("pressure_collateral_damage", 0.0))
        strong_target = target >= cfg.min_improvement * cfg.strong_target_factor
        tiny_j_ok = j_after <= j_before * (1.0 + cfg.tiny_j_tolerance) and strong_target
        if target < cfg.min_improvement:
            reasons.append("target_improvement_below_min")
        if not (j_after <= j_before or tiny_j_ok):
            reasons.append("objective_not_improved")
        if max_collateral > cfg.collateral_tolerance:
            reasons.append("collateral_too_high")
        if pressure_collateral > cfg.pressure_collateral_tolerance:
            reasons.append("pressure_collateral_too_high")
        if float(decision_metrics.get("continuity_collateral_damage", 0.0)) > cfg.continuity_collateral_tolerance:
            reasons.append("continuity_collateral_too_high")
        if float(decision_metrics.get("boundary_collateral_damage", 0.0)) > cfg.boundary_collateral_tolerance:
            reasons.append("boundary_collateral_too_high")
        if float(decision_metrics.get("validation_loss_damage", 0.0)) > cfg.validation_loss_tolerance:
            reasons.append("validation_loss_too_high")
        if float(decision_metrics.get("pde_hard_damage", 0.0)) > cfg.pde_hard_tolerance:
            reasons.append("pde_hard_damage_too_high")
        if float(decision_metrics.get("boundary_hard_damage", 0.0)) > cfg.boundary_hard_tolerance:
            reasons.append("boundary_hard_damage_too_high")
        if float(decision_metrics.get("validation_loss_hard_damage", 0.0)) > cfg.validation_loss_hard_tolerance:
            reasons.append("validation_loss_hard_damage_too_high")
        if float(decision_metrics.get("centerline_pde_hard_damage", 0.0)) > cfg.centerline_pde_hard_tolerance:
            reasons.append("centerline_pde_hard_damage_too_high")
        if float(decision_metrics.get("corner_boundary_hard_damage", 0.0)) > cfg.corner_boundary_hard_tolerance:
            reasons.append("corner_boundary_hard_damage_too_high")
        if float(decision_metrics.get("u_boundary_hard_damage", 0.0)) > cfg.u_boundary_hard_tolerance:
            reasons.append("u_boundary_hard_damage_too_high")
        if float(decision_metrics.get("v_boundary_hard_damage", 0.0)) > cfg.v_boundary_hard_tolerance:
            reasons.append("v_boundary_hard_damage_too_high")
        return ",".join(reasons) if reasons else "constraint_failed"

    def _j_score(self, metrics: dict[str, float]) -> float:
        if hasattr(self.controller.policy, "score_metrics"):
            return float(self.controller.policy.score_metrics(metrics))
        def metric(name: str) -> float:
            value = float(metrics.get(name, 0.0))
            return value if np.isfinite(value) else 0.0

        return float(
            metric("u_rel_l2")
            + metric("v_rel_l2")
            + 1.5 * metric("p_rel_l2_centered")
            + metric("omega_rel_l2")
            + 0.25 * metric("pde_residual_mean")
        )

    def _model_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}

    def _restore_model_snapshot(self, snapshot: dict[str, torch.Tensor]) -> None:
        device_snapshot = {name: value.to(self.device) for name, value in snapshot.items()}
        self.model.load_state_dict(device_snapshot)

    def _repeat_count(self, variable: str) -> int:
        count = 1
        for decision in reversed(self.controller.variable_history):
            if decision.get("targeted_variable") == variable:
                count += 1
            else:
                break
        return count

    def _flat_decision(self, decision: dict[str, Any]) -> dict[str, Any]:
        old_weights = decision.get("old_weights", {})
        new_weights = decision.get("new_weights", {})
        flat = {
            k: v
            for k, v in decision.items()
            if k
            not in {
                "old_weights",
                "new_weights",
                "old_local_weights",
                "new_local_weights",
                "old_sampling_priorities",
                "new_sampling_priorities",
            }
        }
        for name in ("u", "v", "p", "omega", "bc", "pressure_gradient"):
            flat[f"old_weight_{name}"] = old_weights.get(name)
            flat[f"new_weight_{name}"] = new_weights.get(name)
        return flat

    def _max_action_strength(self, trial: dict[str, Any]) -> float:
        strengths = [float(action.get("strength", 0.0)) for action in trial.get("actions", [])]
        return max(strengths) if strengths else 0.0
