"""Local variable-region controller for VARA-PINN."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .actions import TrainingControlState
from .patch_geometry import PatchClassification, classify_patch


METRIC_BY_DIAGNOSTIC = {
    "u_error": "u_rel_l2",
    "v_error": "v_rel_l2",
    "p_error_mean_centered": "p_rel_l2_centered",
    "pressure_gradient_error": "p_rel_l2_centered",
    "omega_error": "omega_rel_l2",
    "continuity_residual": "continuity_residual_mean",
    "momentum_u_residual": "momentum_residual_mean",
    "momentum_v_residual": "momentum_residual_mean",
    "aggregate_pde_residual": "pde_residual_mean",
    "pde_residual": "pde_residual_mean",
    "centerline_pde_residual": "centerline_pde_residual_mean",
    "centerline_continuity_residual": "centerline_continuity_residual_mean",
    "corner_pde_residual": "corner_pde_residual_mean",
    "corner_boundary_error": "corner_boundary_error",
    "boundary_violation": "boundary_condition_error",
}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _domain_bounds(value: Any) -> tuple[float, float, float, float]:
    seq = list(value) if isinstance(value, (list, tuple)) else [0.0, 1.0, 0.0, 1.0]
    if len(seq) < 4:
        seq = [0.0, 1.0, 0.0, 1.0]
    return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))


@dataclass
class LocalControllerConfig:
    """Configuration for local variable-region control."""

    min_improvement: float = 0.005
    collateral_tolerance: float = 0.10
    pressure_collateral_tolerance: float = 0.05
    trial_epochs: int = 100
    max_actions_per_cycle: int = 6
    initial_strength: float = 1.0
    damping_factor: float = 0.5
    local_velocity_weight_max: float = 2.0
    local_pressure_weight_max: float = 3.0
    local_omega_weight_max: float = 2.0
    local_pde_weight_max: float = 2.0
    local_bc_weight_max: float = 7.0
    objective_weights: dict[str, float] | None = None
    tiny_j_tolerance: float = 0.002
    strong_target_factor: float = 2.0
    continuity_collateral_tolerance: float = 0.05
    boundary_collateral_tolerance: float = 0.05
    validation_loss_tolerance: float = 0.02
    pde_hard_tolerance: float = 1.0
    boundary_hard_tolerance: float = 1.0
    validation_loss_hard_tolerance: float = 1.0
    centerline_pde_hard_tolerance: float = 1.0
    corner_boundary_hard_tolerance: float = 1.0
    u_boundary_hard_tolerance: float = 1.0
    v_boundary_hard_tolerance: float = 1.0
    wall_patch_strength_factor: float = 1.0
    corner_patch_strength_factor: float = 1.0
    boundary_patch_margin: float = 1e-9
    rejection_recovery_epochs: int = 0
    warmup_cycles: int = 0
    benchmark: str = ""
    patch_type_aware: bool = False
    domain_bounds: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    near_wall_width: float = 0.10
    centerline_band_width: float = 0.06
    interior_pde_strength_factor: float = 1.15
    centerline_patch_strength_factor: float = 1.0
    sampling_only_confidence_threshold: float = 0.35
    interior_trial_epochs: int | None = None
    wall_trial_epochs: int | None = None
    corner_trial_epochs: int | None = None
    sampling_only_trial_epochs: int | None = None
    strict_wall_boundary_acceptance: bool = True
    wall_boundary_worsen_tolerance: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LocalControllerConfig":
        data = dict(data or {})
        return cls(
            min_improvement=float(data.get("min_improvement", 0.005)),
            collateral_tolerance=float(data.get("collateral_tolerance", 0.10)),
            pressure_collateral_tolerance=float(data.get("pressure_collateral_tolerance", 0.05)),
            trial_epochs=int(data.get("trial_epochs", 100)),
            max_actions_per_cycle=int(data.get("max_actions_per_cycle", 6)),
            initial_strength=float(data.get("initial_strength", 1.0)),
            damping_factor=float(data.get("damping_factor", 0.5)),
            local_velocity_weight_max=float(data.get("local_velocity_weight_max", 2.0)),
            local_pressure_weight_max=float(data.get("local_pressure_weight_max", 3.0)),
            local_omega_weight_max=float(data.get("local_omega_weight_max", 2.0)),
            local_pde_weight_max=float(data.get("local_pde_weight_max", 2.0)),
            local_bc_weight_max=float(data.get("local_bc_weight_max", 7.0)),
            objective_weights=dict(data.get("objective_weights", {"u": 1.0, "v": 1.0, "p": 1.5, "omega": 1.0, "residual": 0.25})),
            tiny_j_tolerance=float(data.get("tiny_j_tolerance", 0.002)),
            strong_target_factor=float(data.get("strong_target_factor", 2.0)),
            continuity_collateral_tolerance=float(data.get("continuity_collateral_tolerance", 0.05)),
            boundary_collateral_tolerance=float(data.get("boundary_collateral_tolerance", 0.05)),
            validation_loss_tolerance=float(data.get("validation_loss_tolerance", 0.02)),
            pde_hard_tolerance=float(data.get("pde_hard_tolerance", 1.0)),
            boundary_hard_tolerance=float(data.get("boundary_hard_tolerance", 1.0)),
            validation_loss_hard_tolerance=float(data.get("validation_loss_hard_tolerance", 1.0)),
            centerline_pde_hard_tolerance=float(data.get("centerline_pde_hard_tolerance", 1.0)),
            corner_boundary_hard_tolerance=float(data.get("corner_boundary_hard_tolerance", 1.0)),
            u_boundary_hard_tolerance=float(data.get("u_boundary_hard_tolerance", 1.0)),
            v_boundary_hard_tolerance=float(data.get("v_boundary_hard_tolerance", 1.0)),
            wall_patch_strength_factor=float(data.get("wall_patch_strength_factor", 1.0)),
            corner_patch_strength_factor=float(data.get("corner_patch_strength_factor", 1.0)),
            boundary_patch_margin=float(data.get("boundary_patch_margin", 1e-9)),
            rejection_recovery_epochs=int(data.get("rejection_recovery_epochs", 0)),
            warmup_cycles=int(data.get("warmup_cycles", 0)),
            benchmark=str(data.get("benchmark", "")),
            patch_type_aware=bool(data.get("patch_type_aware", False)),
            domain_bounds=_domain_bounds(data.get("domain_bounds", (0.0, 1.0, 0.0, 1.0))),
            near_wall_width=float(data.get("near_wall_width", data.get("strip_width", 0.10))),
            centerline_band_width=float(data.get("centerline_band_width", data.get("centerline_width", 0.06))),
            interior_pde_strength_factor=float(data.get("interior_pde_strength_factor", 1.15)),
            centerline_patch_strength_factor=float(data.get("centerline_patch_strength_factor", 1.0)),
            sampling_only_confidence_threshold=float(data.get("sampling_only_confidence_threshold", 0.35)),
            interior_trial_epochs=_optional_int(data.get("interior_trial_epochs")),
            wall_trial_epochs=_optional_int(data.get("wall_trial_epochs")),
            corner_trial_epochs=_optional_int(data.get("corner_trial_epochs")),
            sampling_only_trial_epochs=_optional_int(data.get("sampling_only_trial_epochs")),
            strict_wall_boundary_acceptance=bool(data.get("strict_wall_boundary_acceptance", True)),
            wall_boundary_worsen_tolerance=float(data.get("wall_boundary_worsen_tolerance", 0.0)),
        )

    @property
    def caps(self) -> dict[str, float]:
        return {
            "u": self.local_velocity_weight_max,
            "v": self.local_velocity_weight_max,
            "p": self.local_pressure_weight_max,
            "pressure_gradient": self.local_pressure_weight_max,
            "omega": self.local_omega_weight_max,
            "pde": self.local_pde_weight_max,
            "momentum_u": self.local_pde_weight_max,
            "momentum_v": self.local_pde_weight_max,
            "continuity": self.local_pde_weight_max,
            "bc": self.local_bc_weight_max,
        }


@dataclass
class LocalIntervention:
    """One variable-region local intervention."""

    variable: str
    patch_id: int
    action: str
    loss_variables: list[str]
    strength: float
    severity: float
    confidence: float
    bounds: tuple[float, float, float, float, float | None, float | None]
    patch_type: str = "generic"
    action_family: str = "local_weight"
    sampling_only: bool = False
    near_wall: bool = False
    centerline_band: bool = False
    patch_tags: tuple[str, ...] = ()
    trial_epochs: int | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LocalPairState:
    """Persistent controller memory for one variable-patch pair."""

    strength: float
    accepted_count: int = 0
    rejected_count: int = 0
    last_target_improvement: float = 0.0
    last_collateral: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class LocalVARAController:
    """Controller that only modifies variable-region local weights and priorities."""

    def __init__(self, initial_weights: dict[str, float], config: LocalControllerConfig) -> None:
        self.state = TrainingControlState(global_weights=dict(initial_weights))
        self.config = config
        self.pair_state: dict[tuple[str, int], LocalPairState] = {}
        self.history: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        return self.state.snapshot()

    def rollback(self, snapshot: dict[str, Any]) -> None:
        self.state.restore(snapshot)

    def propose(self, weak_regions: list[object]) -> list[LocalIntervention]:
        interventions: list[LocalIntervention] = []
        seen: set[tuple[str, int, str]] = set()
        for region in weak_regions:
            action, loss_variables, action_family, sampling_only, patch_class = self._action_for_region(region)
            key = (region.variable, int(region.patch_id), action)
            if key in seen:
                continue
            seen.add(key)
            strength = self._strength(region.variable, int(region.patch_id), float(region.confidence))
            strength *= self._geometry_strength_factor(region.bounds, patch_class)
            interventions.append(
                LocalIntervention(
                    variable=region.variable,
                    patch_id=int(region.patch_id),
                    action=action,
                    loss_variables=loss_variables,
                    strength=strength,
                    severity=float(region.severity),
                    confidence=float(region.confidence),
                    bounds=region.bounds,
                    patch_type=patch_class.patch_type,
                    action_family=action_family,
                    sampling_only=sampling_only,
                    near_wall=patch_class.near_wall,
                    centerline_band=patch_class.centerline_band,
                    patch_tags=patch_class.tags,
                )
            )
            if len(interventions) >= self.config.max_actions_per_cycle:
                break
        for intervention in interventions:
            intervention.trial_epochs = self.trial_epochs_for(intervention)
        return interventions

    def apply(self, interventions: list[LocalIntervention]) -> None:
        for intervention in interventions:
            if not intervention.sampling_only:
                for loss_variable in intervention.loss_variables:
                    self._bump_local(loss_variable, intervention.patch_id, intervention.strength)
            self.state.sampling_priorities[intervention.patch_id] = (
                self.state.sampling_priorities.get(intervention.patch_id, 0.0) + intervention.strength
            )

    def objective(self, metrics: dict[str, float]) -> float:
        weights = self.config.objective_weights or {}
        return float(
            float(weights.get("u", 1.0)) * self._metric_with_fallback(metrics, "u_rel_l2", "u_rmse")
            + float(weights.get("v", 1.0)) * self._metric_with_fallback(metrics, "v_rel_l2", "v_rmse")
            + float(weights.get("p", 1.5)) * self._metric_with_fallback(metrics, "p_rel_l2_centered", "p_rmse_centered")
            + float(weights.get("omega", 1.0)) * self._metric_with_fallback(metrics, "omega_rel_l2", "omega_rmse")
            + float(weights.get("residual", 0.25)) * self._metric(metrics, "pde_residual_mean")
            + float(weights.get("continuity", 0.0)) * self._metric(metrics, "continuity_residual_mean")
            + float(weights.get("momentum", 0.0)) * self._metric(metrics, "momentum_residual_mean")
            + float(weights.get("boundary", 0.0)) * self._metric(metrics, "boundary_condition_error")
            + float(weights.get("unweighted_validation", 0.0)) * self._metric(metrics, "unweighted_validation_loss")
            + float(weights.get("centerline_pde", 0.0)) * self._metric(metrics, "centerline_pde_residual_mean")
            + float(weights.get("centerline_continuity", 0.0)) * self._metric(metrics, "centerline_continuity_residual_mean")
            + float(weights.get("corner_pde", 0.0)) * self._metric(metrics, "corner_pde_residual_mean")
            + float(weights.get("corner_boundary", 0.0)) * self._metric(metrics, "corner_boundary_error")
            + float(weights.get("u_boundary", 0.0)) * self._metric(metrics, "u_boundary_rmse")
            + float(weights.get("v_boundary", 0.0)) * self._metric(metrics, "v_boundary_rmse")
        )

    def evaluate_acceptance(
        self,
        interventions: list[LocalIntervention],
        before_raw_scores: np.ndarray,
        after_raw_scores: np.ndarray,
        diagnostic_names: list[str],
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
        constrained: bool,
    ) -> tuple[bool, dict[str, Any]]:
        if not constrained:
            target_improvement = self._target_improvement(interventions, before_raw_scores, after_raw_scores, diagnostic_names)
            return True, self._decision_metrics(target_improvement, before_metrics, after_metrics, interventions)

        target_improvement = self._target_improvement(interventions, before_raw_scores, after_raw_scores, diagnostic_names)
        metrics = self._decision_metrics(target_improvement, before_metrics, after_metrics, interventions)
        j_ok = metrics["J_after"] <= metrics["J_before"]
        tiny_j_ok = (
            metrics["J_after"] <= metrics["J_before"] * (1.0 + self.config.tiny_j_tolerance)
            and target_improvement >= self.config.min_improvement * self.config.strong_target_factor
        )
        strict_patch_ok = self._strict_patch_acceptance_ok(interventions, metrics)
        accepted = (
            target_improvement >= self.config.min_improvement
            and (j_ok or tiny_j_ok)
            and strict_patch_ok
            and metrics["max_collateral_damage"] <= self.config.collateral_tolerance
            and metrics["pressure_collateral_damage"] <= self.config.pressure_collateral_tolerance
            and metrics["continuity_collateral_damage"] <= self.config.continuity_collateral_tolerance
            and metrics["boundary_collateral_damage"] <= self.config.boundary_collateral_tolerance
            and metrics["validation_loss_damage"] <= self.config.validation_loss_tolerance
            and metrics["pde_hard_damage"] <= self.config.pde_hard_tolerance
            and metrics["boundary_hard_damage"] <= self.config.boundary_hard_tolerance
            and metrics["validation_loss_hard_damage"] <= self.config.validation_loss_hard_tolerance
            and metrics["centerline_pde_hard_damage"] <= self.config.centerline_pde_hard_tolerance
            and metrics["corner_boundary_hard_damage"] <= self.config.corner_boundary_hard_tolerance
            and metrics["u_boundary_hard_damage"] <= self.config.u_boundary_hard_tolerance
            and metrics["v_boundary_hard_damage"] <= self.config.v_boundary_hard_tolerance
        )
        return accepted, metrics

    def record_decision(self, interventions: list[LocalIntervention], decision: dict[str, Any]) -> None:
        self.history.append(decision)

    def mark_accepted(self, intervention: LocalIntervention, decision: dict[str, Any]) -> None:
        """Update persistent pair memory after an accepted action."""
        state = self._pair_state(intervention.variable, intervention.patch_id)
        state.accepted_count += 1
        state.last_target_improvement = float(decision.get("target_local_improvement", 0.0))
        state.last_collateral = float(decision.get("max_collateral_damage", 0.0))
        if state.last_target_improvement < self.config.min_improvement:
            state.strength = max(1e-4, state.strength * self.config.damping_factor)

    def mark_rejected(self, intervention: LocalIntervention, decision: dict[str, Any]) -> None:
        """Update persistent pair memory after a rejected action."""
        state = self._pair_state(intervention.variable, intervention.patch_id)
        state.rejected_count += 1
        state.last_target_improvement = float(decision.get("target_local_improvement", 0.0))
        state.last_collateral = float(decision.get("max_collateral_damage", 0.0))
        state.strength = max(1e-4, state.strength * self.config.damping_factor)

    def pair_state_record(self) -> dict[str, Any]:
        """Serialize persistent pair controller state for logs."""
        return {f"{variable}|{patch_id}": state.to_record() for (variable, patch_id), state in self.pair_state.items()}

    def active_patches(self) -> set[int]:
        patches = set(int(pid) for pid in self.state.sampling_priorities)
        for weights in self.state.local_weights.values():
            patches.update(int(pid) for pid in weights)
        return patches

    def trial_epochs_for(self, intervention: LocalIntervention | None) -> int:
        """Return patch/action-specific trial length while preserving generic defaults."""
        if intervention is None or not self._patch_type_policy_enabled():
            return int(self.config.trial_epochs)
        if intervention.sampling_only:
            return int(self.config.sampling_only_trial_epochs or self.config.trial_epochs)
        if intervention.patch_type == "corner":
            return int(self.config.corner_trial_epochs or self.config.trial_epochs)
        if intervention.patch_type in {"lid", "sidewall", "bottom_wall"} or intervention.near_wall:
            return int(self.config.wall_trial_epochs or self.config.trial_epochs)
        return int(self.config.interior_trial_epochs or self.config.trial_epochs)

    def classify_patch(
        self,
        bounds: tuple[float, float, float, float, float | None, float | None],
    ) -> PatchClassification:
        if not self._patch_type_policy_enabled():
            return PatchClassification("generic", False, False, ("generic",))
        return classify_patch(
            bounds,
            domain_bounds=self.config.domain_bounds,
            near_wall_width=self.config.near_wall_width,
            centerline_width=self.config.centerline_band_width,
            boundary_margin=self.config.boundary_patch_margin,
        )

    def _bump_local(self, variable: str, patch_id: int, amount: float) -> None:
        cap = self.config.caps.get(variable, self.config.local_pde_weight_max)
        self.state.local_weights.setdefault(variable, {})
        old = float(self.state.local_weights[variable].get(int(patch_id), 0.0))
        self.state.local_weights[variable][int(patch_id)] = min(old + float(amount), cap)

    def _strength(self, variable: str, patch_id: int, confidence: float) -> float:
        state = self._pair_state(variable, patch_id)
        strength = state.strength * max(0.5, confidence)
        attempted = state.accepted_count + state.rejected_count
        if attempted > 0 and state.last_target_improvement < self.config.min_improvement:
            strength *= self.config.damping_factor
        return max(1e-4, float(strength))

    def _pair_state(self, variable: str, patch_id: int) -> LocalPairState:
        key = (variable, int(patch_id))
        if key not in self.pair_state:
            self.pair_state[key] = LocalPairState(strength=float(self.config.initial_strength))
        return self.pair_state[key]

    def _patch_type_policy_enabled(self) -> bool:
        return bool(self.config.patch_type_aware) and self.config.benchmark.lower() in {"lid_driven_cavity", "cavity"}

    def _action_for_region(self, region: object) -> tuple[str, list[str], str, bool, PatchClassification]:
        patch_class = self.classify_patch(region.bounds)
        if not self._patch_type_policy_enabled():
            action, loss_variables = self._action_for_variable(region.variable)
            return action, loss_variables, self._action_family(action), False, patch_class
        return self._cavity_action_for_region(region.variable, float(region.confidence), patch_class)

    def _cavity_action_for_region(
        self,
        variable: str,
        confidence: float,
        patch_class: PatchClassification,
    ) -> tuple[str, list[str], str, bool, PatchClassification]:
        patch_type = patch_class.patch_type
        if patch_type == "corner":
            return "increase_local_corner_sampling", [], "sampling", True, patch_class
        if patch_type == "lid":
            if "boundary" in variable:
                return "increase_local_boundary", ["bc"], "boundary", False, patch_class
            return "increase_local_boundary_sampling", [], "sampling", True, patch_class
        if patch_type in {"sidewall", "bottom_wall"}:
            if "boundary" in variable:
                return "increase_local_boundary", ["bc"], "boundary", False, patch_class
            return "increase_local_boundary_sampling", [], "sampling", True, patch_class
        if patch_class.centerline_band and "centerline" in variable:
            return "increase_local_centerline_sampling", [], "sampling", True, patch_class
        if confidence < self.config.sampling_only_confidence_threshold and (
            "pde" in variable or "momentum" in variable or "continuity" in variable
        ):
            return "increase_local_sampling", [], "sampling", True, patch_class
        action, loss_variables = self._action_for_variable(variable)
        return action, loss_variables, self._action_family(action), False, patch_class

    def _action_for_variable(self, variable: str) -> tuple[str, list[str]]:
        if "u_error" in variable:
            return "increase_local_velocity", ["u", "momentum_u"]
        if "v_error" in variable:
            return "increase_local_velocity", ["v", "momentum_v"]
        if "p_error" in variable or "pressure" in variable:
            return "increase_local_pressure", ["p", "pressure_gradient"]
        if "omega" in variable or "vorticity" in variable:
            return "increase_local_vorticity", ["omega", "pde"]
        if "centerline_continuity" in variable:
            return "increase_local_divergence", ["continuity", "pde"]
        if "centerline_pde" in variable:
            return "increase_local_pde", ["pde"]
        if "corner_pde" in variable:
            return "increase_local_pde", ["pde"]
        if "corner_boundary" in variable:
            return "increase_local_boundary", ["bc"]
        if "continuity" in variable:
            return "increase_local_divergence", ["continuity", "pde"]
        if "momentum_u" in variable:
            return "increase_local_momentum", ["momentum_u", "pde"]
        if "momentum_v" in variable:
            return "increase_local_momentum", ["momentum_v", "pde"]
        if "boundary" in variable:
            return "increase_local_boundary", ["bc"]
        return "increase_local_pde", ["pde"]

    def _action_family(self, action: str) -> str:
        if "sampling" in action:
            return "sampling"
        if "boundary" in action:
            return "boundary"
        if "momentum" in action:
            return "momentum"
        if "divergence" in action:
            return "continuity"
        if "pressure" in action:
            return "pressure"
        if "velocity" in action:
            return "velocity"
        if "vorticity" in action:
            return "vorticity"
        return "pde"

    def _target_improvement(
        self,
        interventions: list[LocalIntervention],
        before_raw_scores: np.ndarray,
        after_raw_scores: np.ndarray,
        diagnostic_names: list[str],
    ) -> float:
        eps = 1e-12
        name_to_idx = {name: i for i, name in enumerate(diagnostic_names)}
        values = []
        for intervention in interventions:
            if intervention.variable not in name_to_idx:
                continue
            j = name_to_idx[intervention.variable]
            r = intervention.patch_id
            before = float(before_raw_scores[j, r])
            after = float(after_raw_scores[j, r])
            values.append((before - after) / (abs(before) + eps))
        return float(np.mean(values)) if values else 0.0

    def _decision_metrics(
        self,
        target_improvement: float,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
        interventions: list[LocalIntervention],
    ) -> dict[str, Any]:
        metric_targets = {METRIC_BY_DIAGNOSTIC.get(intervention.variable, "pde_residual_mean") for intervention in interventions}
        collateral = {}
        for metric_name in [
            "u_rel_l2",
            "v_rel_l2",
            "p_rel_l2_centered",
            "omega_rel_l2",
            "pde_residual_mean",
            "continuity_residual_mean",
            "momentum_residual_mean",
            "boundary_condition_error",
            "u_boundary_rmse",
            "v_boundary_rmse",
            "boundary_speed_rmse",
            "centerline_pde_residual_mean",
            "centerline_continuity_residual_mean",
            "corner_pde_residual_mean",
            "corner_boundary_error",
            "unweighted_validation_loss",
        ]:
            before = self._metric(before_metrics, metric_name)
            after = self._metric(after_metrics, metric_name)
            if metric_name in metric_targets:
                continue
            collateral[metric_name] = max(0.0, (after - before) / (abs(before) + 1e-12))
        pressure_before = self._metric(before_metrics, "p_rel_l2_centered")
        pressure_after = self._metric(after_metrics, "p_rel_l2_centered")
        hard_damages = {
            "pde": self._relative_damage(before_metrics, after_metrics, "pde_residual_mean"),
            "boundary": self._relative_damage(before_metrics, after_metrics, "boundary_condition_error"),
            "validation_loss": self._relative_damage(before_metrics, after_metrics, "unweighted_validation_loss"),
            "centerline_pde": self._relative_damage(before_metrics, after_metrics, "centerline_pde_residual_mean"),
            "corner_boundary": self._relative_damage(before_metrics, after_metrics, "corner_boundary_error"),
            "u_boundary": self._relative_damage(before_metrics, after_metrics, "u_boundary_rmse"),
            "v_boundary": self._relative_damage(before_metrics, after_metrics, "v_boundary_rmse"),
        }
        wall_boundary_damage = 0.0
        lid_u_boundary_damage = 0.0
        corner_wall_damage = 0.0
        if self._patch_type_policy_enabled():
            wall_types = {"lid", "sidewall", "bottom_wall", "corner"}
            if any(intervention.patch_type in wall_types for intervention in interventions):
                wall_boundary_damage = max(hard_damages["boundary"], hard_damages["u_boundary"], hard_damages["v_boundary"])
            if any(intervention.patch_type == "lid" for intervention in interventions):
                lid_u_boundary_damage = max(hard_damages["boundary"], hard_damages["u_boundary"])
            if any(intervention.patch_type == "corner" for intervention in interventions):
                corner_wall_damage = max(hard_damages["boundary"], hard_damages["corner_boundary"])
        return {
            "target_local_improvement": float(target_improvement),
            "J_before": self.objective(before_metrics),
            "J_after": self.objective(after_metrics),
            "max_collateral_damage": float(max(collateral.values()) if collateral else 0.0),
            "pressure_collateral_damage": float(max(0.0, (pressure_after - pressure_before) / (abs(pressure_before) + 1e-12))),
            "continuity_collateral_damage": collateral.get("continuity_residual_mean", 0.0),
            "boundary_collateral_damage": collateral.get("boundary_condition_error", 0.0),
            "validation_loss_damage": collateral.get("unweighted_validation_loss", 0.0),
            "pde_hard_damage": hard_damages["pde"],
            "boundary_hard_damage": hard_damages["boundary"],
            "validation_loss_hard_damage": hard_damages["validation_loss"],
            "centerline_pde_hard_damage": hard_damages["centerline_pde"],
            "corner_boundary_hard_damage": hard_damages["corner_boundary"],
            "u_boundary_hard_damage": hard_damages["u_boundary"],
            "v_boundary_hard_damage": hard_damages["v_boundary"],
            "strict_wall_boundary_damage": float(wall_boundary_damage),
            "strict_lid_u_boundary_damage": float(lid_u_boundary_damage),
            "strict_corner_boundary_damage": float(corner_wall_damage),
        }

    def _strict_patch_acceptance_ok(self, interventions: list[LocalIntervention], metrics: dict[str, Any]) -> bool:
        if not (self._patch_type_policy_enabled() and self.config.strict_wall_boundary_acceptance):
            return True
        tolerance = float(self.config.wall_boundary_worsen_tolerance)
        wall_types = {"lid", "sidewall", "bottom_wall", "corner"}
        if any(intervention.patch_type in wall_types for intervention in interventions):
            if float(metrics.get("strict_wall_boundary_damage", 0.0)) > tolerance:
                return False
        if any(intervention.patch_type == "lid" for intervention in interventions):
            if float(metrics.get("strict_lid_u_boundary_damage", 0.0)) > tolerance:
                return False
        if any(intervention.patch_type == "corner" for intervention in interventions):
            if float(metrics.get("strict_corner_boundary_damage", 0.0)) > tolerance:
                return False
        return True

    def _relative_damage(self, before_metrics: dict[str, float], after_metrics: dict[str, float], name: str) -> float:
        before = self._metric(before_metrics, name)
        after = self._metric(after_metrics, name)
        if before == 0.0 and after == 0.0:
            return 0.0
        return float(max(0.0, (after - before) / (abs(before) + 1e-12)))

    def _geometry_strength_factor(
        self,
        bounds: tuple[float, float, float, float, float | None, float | None],
        patch_class: PatchClassification | None = None,
    ) -> float:
        if self._patch_type_policy_enabled() and patch_class is not None:
            factor = 1.0
            if patch_class.patch_type == "centerline_band":
                factor *= float(self.config.centerline_patch_strength_factor)
            elif patch_class.patch_type == "interior":
                factor *= float(self.config.interior_pde_strength_factor)
            if patch_class.patch_type == "corner":
                factor *= float(self.config.wall_patch_strength_factor)
                factor *= float(self.config.corner_patch_strength_factor)
            elif patch_class.patch_type in {"lid", "sidewall", "bottom_wall"} or patch_class.near_wall:
                factor *= float(self.config.wall_patch_strength_factor)
            return max(1e-4, float(factor))
        if not bounds:
            return 1.0
        x0, x1, y0, y1 = (float(bounds[i]) for i in range(4))
        margin = float(self.config.boundary_patch_margin)
        touches_left = x0 <= margin
        touches_right = x1 >= 1.0 - margin
        touches_bottom = y0 <= margin
        touches_top = y1 >= 1.0 - margin
        touches_wall = touches_left or touches_right or touches_bottom or touches_top
        touches_corner = (touches_left or touches_right) and (touches_bottom or touches_top)
        factor = 1.0
        if touches_wall:
            factor *= float(self.config.wall_patch_strength_factor)
        if touches_corner:
            factor *= float(self.config.corner_patch_strength_factor)
        return max(1e-4, float(factor))

    def _metric(self, metrics: dict[str, float], name: str) -> float:
        try:
            value = float(metrics.get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    def _metric_with_fallback(self, metrics: dict[str, float], primary: str, fallback: str) -> float:
        primary_value = self._metric(metrics, primary)
        if primary_value != 0.0:
            return primary_value
        return self._metric(metrics, fallback)
