"""VARA-PINN trainer."""

from __future__ import annotations

from copy import deepcopy
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
        local_controller_cfg = dict(config.get("local_controller", {}))
        local_controller_cfg.setdefault("benchmark", config.get("benchmark", ""))
        local_controller_cfg.setdefault("domain_bounds", list(self.benchmark.bounds))
        cavity_focus_cfg = config.get("sampling", {}).get("cavity_focus", {})
        if cavity_focus_cfg:
            local_controller_cfg.setdefault("near_wall_width", cavity_focus_cfg.get("strip_width", 0.10))
            local_controller_cfg.setdefault("centerline_band_width", cavity_focus_cfg.get("centerline_width", 0.06))
        self.local_controller = LocalVARAController(
            initial_weights=config.get("training", {}).get("weights", {}),
            config=LocalControllerConfig.from_dict(local_controller_cfg),
        )
        self.local_patch_score_logger = CSVLogger(self.run_dir / "patch_scores.csv")
        self.local_action_logger = JSONListLogger(self.run_dir / "local_actions.json")
        self.local_decision_logger = CSVLogger(self.run_dir / "local_controller_decisions.csv")
        self.local_weights_logger = JSONListLogger(self.run_dir / "local_weights_history.json")
        self.local_figure_dir = ensure_dir(self.run_dir / "figures")
        self.local_decisions: list[dict[str, Any]] = []
        self.local_weight_records: list[dict[str, Any]] = []

    def run(self) -> dict[str, float]:
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
        metrics = self.evaluate_and_save_final()
        metrics["J_score"] = self._j_score(metrics)
        metrics["accepted_interventions"] = self.accepted_interventions
        metrics["rejected_interventions"] = self.rejected_interventions
        metrics["rollback_count"] = self.rollback_count
        return metrics

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
                    candidate_trial_epochs = self.local_controller.trial_epochs_for(intervention)

                    self.local_controller.apply([intervention])
                    proposed_local_weights = deepcopy(self.local_controller.state.local_weights)
                    proposed_sampling = deepcopy(self.local_controller.state.sampling_priorities)
                    self.train_epochs(
                        batch,
                        self.local_controller.state,
                        cycle=cycle,
                        epochs_override=candidate_trial_epochs,
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
                        trial_epochs=candidate_trial_epochs,
                    )
                    self.local_controller.record_decision([intervention], decision)
                    self.local_decisions.append(decision)
                    action_record = intervention.to_record()
                    action_record["candidate_id"] = candidate_id
                    action_record["accepted"] = bool(accepted)
                    action_record["local_weights_before"] = old_local_weights
                    action_record["local_weights_after"] = proposed_local_weights
                    action_record["kept_local_weights_after"] = deepcopy(self.local_controller.state.local_weights)
                    action_record["sampling_priorities_before"] = old_sampling
                    action_record["sampling_priorities_after"] = proposed_sampling
                    action_record["kept_sampling_priorities_after"] = deepcopy(self.local_controller.state.sampling_priorities)
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
                action_records = []
                for intervention in interventions:
                    action_record = intervention.to_record()
                    action_record["local_weights_before"] = old_local_weights
                    action_record["local_weights_after"] = deepcopy(self.local_controller.state.local_weights)
                    action_record["sampling_priorities_before"] = old_sampling
                    action_record["sampling_priorities_after"] = deepcopy(self.local_controller.state.sampling_priorities)
                    action_records.append(action_record)
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
        from src.evaluation.metrics import evaluate_on_grid

        return evaluate_on_grid(self.model, self.benchmark, coords, self.device, self.steady, include_profile_reference=False)

    def _diagnose_local(
        self,
        update_ema: bool,
        detect: bool = True,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[str], list[Any], np.ndarray, np.ndarray, np.ndarray]:
        X, Y, coords = self.validation_grid()
        builder = DiagnosticMapBuilder(self.model, self.benchmark, self.device, self.steady)
        maps = builder.build(coords, mode=self.config.get("diagnostics", {}).get("mode", "full_reference"))
        norm_scores, names = self.patch_scorer.compute(maps, coords, update_ema=update_ema)
        raw_scores = self.patch_scorer.last_raw_scores
        if raw_scores is None:
            raw_scores = norm_scores
        weak_regions = self.weak_detector.detect(norm_scores, names, self.patch_grid) if detect else []
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
        trial_epochs: int | None = None,
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
            "patch_type": first.patch_type if first is not None else "",
            "action_family": first.action_family if first is not None else "",
            "sampling_only": bool(first.sampling_only) if first is not None else False,
            "near_wall": bool(first.near_wall) if first is not None else False,
            "centerline_band": bool(first.centerline_band) if first is not None else False,
            "trial_epochs": trial_epochs,
            "selected_pairs": [intervention.to_record() for intervention in interventions],
            "targeted_variables": [intervention.variable for intervention in interventions],
            "targeted_patches": [intervention.patch_id for intervention in interventions],
            "targeted_patch_types": [intervention.patch_type for intervention in interventions],
            "action_families": [intervention.action_family for intervention in interventions],
            "sampling_only_actions": [bool(intervention.sampling_only) for intervention in interventions],
            "active_patches": sorted(self.local_controller.active_patches()),
            "local_weights_before": old_local_weights,
            "local_weights_after": deepcopy(self.local_controller.state.local_weights),
            "sampling_before": old_sampling,
            "sampling_after": deepcopy(self.local_controller.state.sampling_priorities),
            "sampling_priorities_before": old_sampling,
            "sampling_priorities_after": deepcopy(self.local_controller.state.sampling_priorities),
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
            "patch_type": decision.get("patch_type"),
            "action_family": decision.get("action_family"),
            "sampling_only": decision.get("sampling_only"),
            "trial_epochs": decision.get("trial_epochs"),
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
            "strict_wall_boundary_damage": decision.get("strict_wall_boundary_damage"),
            "strict_lid_u_boundary_damage": decision.get("strict_lid_u_boundary_damage"),
            "strict_corner_boundary_damage": decision.get("strict_corner_boundary_damage"),
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
            xy_f = self._sample_cavity_focused_interior(
                n_f,
                lambda count: self.adaptive_sampler.sample_interior(
                    count,
                    maps,
                    coords,
                    weak_regions,
                    self.local_controller.state.sampling_priorities,
                ),
            )
            xy_data = self._sample_cavity_focused_interior(
                n_data,
                lambda count: self.adaptive_sampler.sample_interior(
                    count,
                    maps,
                    coords,
                    weak_regions,
                    self.local_controller.state.sampling_priorities,
                ),
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
            xy_f = self._sample_interior(n_f)
            xy_data = self._sample_interior(n_data)
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
        if float(decision_metrics.get("strict_wall_boundary_damage", 0.0)) > cfg.wall_boundary_worsen_tolerance:
            reasons.append("wall_boundary_worsened")
        if float(decision_metrics.get("strict_lid_u_boundary_damage", 0.0)) > cfg.wall_boundary_worsen_tolerance:
            reasons.append("lid_u_boundary_worsened")
        if float(decision_metrics.get("strict_corner_boundary_damage", 0.0)) > cfg.wall_boundary_worsen_tolerance:
            reasons.append("corner_boundary_worsened")
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
