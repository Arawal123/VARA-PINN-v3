"""Matched-compute, reference-free VARA V2 controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


FORBIDDEN_CONTROLLER_TOKENS = {
    "reference",
    "ghia",
    "cfd",
    "test",
    "rel_l2",
    "rmse",
    "mae",
    "profile_score",
    "benchmark_score",
}

ALLOWED_GUARD_METRICS = (
    "pde_residual_mean",
    "continuity_residual_mean",
    "momentum_residual_mean",
    "boundary_condition_error",
    "unweighted_validation_loss",
)


@dataclass
class V2ControllerConfig:
    num_patches: int
    min_uniform_mass: float = 0.35
    max_patch_mass: float = 0.25
    multiplier_min: float = 0.5
    multiplier_max: float = 2.0
    max_candidates: int = 4
    prefilter_damage_ratio: float = 0.25
    trust_radius_initial: float = 0.10
    trust_radius_min: float = 0.025
    trust_radius_max: float = 0.20
    trust_expand: float = 1.25
    trust_shrink: float = 0.5
    effectiveness_ema: float = 0.8
    noise_floor: float = 1e-4
    reliable_reward_ratio: float = 0.75
    gradient_prefilter_enabled: bool = True
    trust_region_enabled: bool = True
    action_memory_enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any], num_patches: int) -> "V2ControllerConfig":
        return cls(
            num_patches=int(num_patches),
            min_uniform_mass=float(data.get("min_uniform_mass", 0.35)),
            max_patch_mass=float(data.get("max_patch_mass", 0.25)),
            multiplier_min=float(data.get("multiplier_min", 0.5)),
            multiplier_max=float(data.get("multiplier_max", 2.0)),
            max_candidates=int(data.get("max_candidates", 4)),
            prefilter_damage_ratio=float(data.get("prefilter_damage_ratio", 0.25)),
            trust_radius_initial=float(data.get("trust_radius_initial", 0.10)),
            trust_radius_min=float(data.get("trust_radius_min", 0.025)),
            trust_radius_max=float(data.get("trust_radius_max", 0.20)),
            trust_expand=float(data.get("trust_expand", 1.25)),
            trust_shrink=float(data.get("trust_shrink", 0.5)),
            effectiveness_ema=float(data.get("effectiveness_ema", 0.8)),
            noise_floor=float(data.get("noise_floor", 1e-4)),
            reliable_reward_ratio=float(data.get("reliable_reward_ratio", 0.75)),
            gradient_prefilter_enabled=bool(data.get("gradient_prefilter_enabled", True)),
            trust_region_enabled=bool(data.get("trust_region_enabled", True)),
            action_memory_enabled=bool(data.get("action_memory_enabled", True)),
        )


@dataclass
class V2AllocationState:
    """Controller allocation with conserved sampling and loss mass."""

    num_patches: int
    sampling_mass: np.ndarray = field(init=False)
    loss_multipliers: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sampling_mass = np.full(self.num_patches, 1.0 / self.num_patches, dtype=float)

    def snapshot(self) -> dict[str, Any]:
        return {
            "sampling_mass": self.sampling_mass.copy(),
            "loss_multipliers": {name: values.copy() for name, values in self.loss_multipliers.items()},
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.sampling_mass = np.asarray(snapshot["sampling_mass"], dtype=float).copy()
        self.loss_multipliers = {
            str(name): np.asarray(values, dtype=float).copy()
            for name, values in dict(snapshot["loss_multipliers"]).items()
        }

    def multiplier(self, loss_name: str) -> np.ndarray:
        if loss_name not in self.loss_multipliers:
            self.loss_multipliers[loss_name] = np.ones(self.num_patches, dtype=float)
        return self.loss_multipliers[loss_name]

    def to_record(self) -> dict[str, Any]:
        return {
            "sampling_mass": self.sampling_mass.tolist(),
            "loss_multipliers": {
                name: values.tolist() for name, values in self.loss_multipliers.items()
            },
        }


@dataclass
class V2Candidate:
    variable: str
    patch_id: int
    action_type: str
    loss_names: list[str]
    severity: float
    persistence: int
    trend: float
    gradient_compatibility: float = 0.0
    predicted_target_improvement: float = 0.0
    predicted_guard_damage: float = 0.0
    rank_score: float = 0.0
    prefiltered: bool = False

    def key(self) -> str:
        return f"{self.variable}|{self.patch_id}|{self.action_type}"

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class VARAV2Controller:
    """Trust-region allocation controller with no reference-metric access."""

    def __init__(self, config: V2ControllerConfig) -> None:
        self.config = config
        self.state = V2AllocationState(config.num_patches)
        self.trust_radius = float(config.trust_radius_initial)
        self.effectiveness: dict[str, float] = {}
        self.score_history: dict[tuple[str, int], list[float]] = {}
        self.metric_history: dict[str, list[float]] = {name: [] for name in ALLOWED_GUARD_METRICS}
        self.decisions: list[dict[str, Any]] = []

    @staticmethod
    def assert_reference_free(names: list[str] | tuple[str, ...] | set[str]) -> None:
        bad = sorted(
            name for name in names
            if any(token in str(name).lower() for token in FORBIDDEN_CONTROLLER_TOKENS)
        )
        if bad:
            raise ValueError(f"VARA V2 controller received evaluation-only signals: {bad}")

    def reference_free_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        self.assert_reference_free(set(metrics))
        return {
            name: float(metrics[name])
            for name in ALLOWED_GUARD_METRICS
            if name in metrics and np.isfinite(float(metrics[name]))
        }

    def update_history(
        self,
        diagnostic_names: list[str],
        raw_scores: np.ndarray,
        metrics: dict[str, float],
    ) -> None:
        self.assert_reference_free(diagnostic_names)
        for j, name in enumerate(diagnostic_names):
            for patch_id, value in enumerate(raw_scores[j]):
                history = self.score_history.setdefault((name, int(patch_id)), [])
                history.append(float(value))
                del history[:-8]
        for name in ALLOWED_GUARD_METRICS:
            if name in metrics and np.isfinite(float(metrics[name])):
                history = self.metric_history[name]
                history.append(float(metrics[name]))
                del history[:-8]

    def candidates(self, weak_regions: list[Any]) -> list[V2Candidate]:
        candidates: list[V2Candidate] = []
        if not weak_regions:
            return candidates
        primary = weak_regions[0]
        for action_type in ("sampling", "local_loss", "joint"):
            candidates.append(self._candidate(primary, action_type))
        if len(weak_regions) > 1:
            candidates.append(self._candidate(weak_regions[1], "joint"))
        return candidates[: self.config.max_candidates]

    def _candidate(self, region: Any, action_type: str) -> V2Candidate:
        variable = str(region.variable)
        patch_id = int(region.patch_id)
        history = self.score_history.get((variable, patch_id), [])
        trend = 0.0
        if len(history) >= 2:
            scale = abs(history[-2]) + 1e-12
            trend = float((history[-1] - history[-2]) / scale)
        return V2Candidate(
            variable=variable,
            patch_id=patch_id,
            action_type=action_type,
            loss_names=_losses_for_diagnostic(variable),
            severity=float(region.severity),
            persistence=int(getattr(region, "persistence", 1)),
            trend=trend,
        )

    def rank(
        self,
        candidates: list[V2Candidate],
        influence: dict[str, dict[str, float]],
    ) -> list[V2Candidate]:
        ranked: list[V2Candidate] = []
        for candidate in candidates:
            values = influence.get(candidate.key(), {})
            compatibility = float(values.get("gradient_compatibility", 0.0))
            conflict = max(0.0, float(values.get("gradient_conflict", 0.0)))
            effectiveness = (
                float(self.effectiveness.get(candidate.key(), 1.0))
                if self.config.action_memory_enabled
                else 1.0
            )
            persistence_score = min(1.0, candidate.persistence / 3.0)
            trend_score = float(np.clip(candidate.trend, -1.0, 1.0))
            candidate.gradient_compatibility = compatibility
            candidate.predicted_target_improvement = max(
                self.config.noise_floor,
                candidate.severity
                * (0.5 + 0.5 * max(0.0, compatibility))
                * max(0.25, effectiveness),
            )
            candidate.predicted_guard_damage = candidate.severity * conflict
            candidate.prefiltered = self.config.gradient_prefilter_enabled and (
                candidate.predicted_guard_damage
                > self.config.prefilter_damage_ratio * candidate.predicted_target_improvement
            )
            candidate.rank_score = (
                0.40 * candidate.severity
                + 0.15 * persistence_score
                + 0.10 * trend_score
                + 0.20 * compatibility
                + 0.15 * effectiveness
                - candidate.predicted_guard_damage
            )
            ranked.append(candidate)
        return sorted(ranked, key=lambda item: item.rank_score, reverse=True)

    def apply(self, candidate: V2Candidate) -> None:
        radius = float(self.trust_radius)
        if candidate.action_type in {"sampling", "joint"}:
            sampling_radius = radius if candidate.action_type == "sampling" else 0.5 * radius
            self.state.sampling_mass = _redistribute_probability_mass(
                self.state.sampling_mass,
                candidate.patch_id,
                sampling_radius,
                self.config.max_patch_mass,
            )
        if candidate.action_type in {"local_loss", "joint"}:
            loss_radius = radius if candidate.action_type == "local_loss" else 0.5 * radius
            for loss_name in candidate.loss_names:
                current = self.state.multiplier(loss_name)
                self.state.loss_multipliers[loss_name] = _redistribute_multiplier_mass(
                    current,
                    candidate.patch_id,
                    loss_radius,
                    self.config.multiplier_min,
                    self.config.multiplier_max,
                )
        self.validate_state()

    def evaluate(
        self,
        candidate: V2Candidate,
        before_target: float,
        after_target: float,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
    ) -> tuple[bool, dict[str, Any]]:
        self.assert_reference_free(set(before_metrics) | set(after_metrics))
        eps = 1e-12
        observed = (float(before_target) - float(after_target)) / (abs(float(before_target)) + eps)
        noise = self.target_noise(candidate.variable, candidate.patch_id)
        guard_changes: dict[str, float] = {}
        guard_noise: dict[str, float] = {}
        for name in ALLOWED_GUARD_METRICS:
            if name not in before_metrics or name not in after_metrics:
                continue
            before = float(before_metrics[name])
            after = float(after_metrics[name])
            guard_changes[name] = (after - before) / (abs(before) + eps)
            guard_noise[name] = self.metric_noise(name)
        guard_ok = all(
            change <= max(self.config.noise_floor, guard_noise[name])
            for name, change in guard_changes.items()
        )
        accepted = observed > max(self.config.noise_floor, noise) and guard_ok
        predicted = max(candidate.predicted_target_improvement, self.config.noise_floor)
        reward_ratio = observed / predicted
        decision = {
            "accepted": bool(accepted),
            "target_noise": float(noise),
            "observed_target_improvement": float(observed),
            "predicted_target_improvement": float(candidate.predicted_target_improvement),
            "predicted_guard_damage": float(candidate.predicted_guard_damage),
            "reward_ratio": float(reward_ratio),
            "guard_changes": guard_changes,
            "guard_noise": guard_noise,
            "trust_radius_before": float(self.trust_radius),
            "rollback_reason": "" if accepted else (
                "target_below_noise" if observed <= max(self.config.noise_floor, noise)
                else "pareto_guard_violation"
            ),
        }
        self._update_after_decision(candidate, accepted, reward_ratio, decision)
        return accepted, decision

    def record_prefilter(self, candidate: V2Candidate, update_trust: bool = True) -> dict[str, Any]:
        if self.config.trust_region_enabled and update_trust:
            self.trust_radius = max(
                self.config.trust_radius_min,
                self.trust_radius * self.config.trust_shrink,
            )
        decision = {
            "accepted": False,
            "prefiltered": True,
            "rollback_reason": "predicted_guard_damage",
            "predicted_target_improvement": candidate.predicted_target_improvement,
            "predicted_guard_damage": candidate.predicted_guard_damage,
            "trust_radius_after": self.trust_radius,
        }
        self.decisions.append({**candidate.to_record(), **decision})
        return decision

    def _update_after_decision(
        self,
        candidate: V2Candidate,
        accepted: bool,
        reward_ratio: float,
        decision: dict[str, Any],
    ) -> None:
        old = self.effectiveness.get(candidate.key(), 1.0)
        if self.config.action_memory_enabled:
            observed_effect = max(0.0, min(2.0, float(reward_ratio))) if accepted else 0.0
            rho = self.config.effectiveness_ema
            self.effectiveness[candidate.key()] = rho * old + (1.0 - rho) * observed_effect
        else:
            self.effectiveness[candidate.key()] = 1.0
        if self.config.trust_region_enabled:
            if accepted and reward_ratio >= self.config.reliable_reward_ratio:
                self.trust_radius = min(
                    self.config.trust_radius_max,
                    self.trust_radius * self.config.trust_expand,
                )
            elif not accepted:
                self.trust_radius = max(
                    self.config.trust_radius_min,
                    self.trust_radius * self.config.trust_shrink,
                )
        decision["trust_radius_after"] = float(self.trust_radius)
        decision["effectiveness_after"] = float(self.effectiveness[candidate.key()])
        self.decisions.append({**candidate.to_record(), **decision})

    def target_noise(self, variable: str, patch_id: int) -> float:
        return _robust_relative_noise(self.score_history.get((variable, int(patch_id)), []), self.config.noise_floor)

    def metric_noise(self, name: str) -> float:
        return _robust_relative_noise(self.metric_history.get(name, []), self.config.noise_floor)

    def validate_state(self) -> None:
        mass = self.state.sampling_mass
        if not np.isclose(float(np.sum(mass)), 1.0, atol=1e-8):
            raise RuntimeError("VARA V2 sampling mass is not conserved.")
        if np.any(mass < 0.0) or np.any(mass > self.config.max_patch_mass + 1e-10):
            raise RuntimeError("VARA V2 sampling allocation exceeds configured bounds.")
        for values in self.state.loss_multipliers.values():
            if not np.isclose(float(np.mean(values)), 1.0, atol=1e-8):
                raise RuntimeError("VARA V2 local multiplier mass is not conserved.")
            if np.any(values < self.config.multiplier_min - 1e-10):
                raise RuntimeError("VARA V2 local multiplier is below its lower bound.")
            if np.any(values > self.config.multiplier_max + 1e-10):
                raise RuntimeError("VARA V2 local multiplier exceeds its upper bound.")


def _losses_for_diagnostic(name: str) -> list[str]:
    lower = name.lower()
    if "continuity" in lower:
        return ["continuity"]
    if "momentum_u" in lower:
        return ["momentum_u"]
    if "momentum_v" in lower:
        return ["momentum_v"]
    if "boundary" in lower:
        return ["bc"]
    if "u_error" in lower:
        return ["u", "momentum_u"]
    if "v_error" in lower:
        return ["v", "momentum_v"]
    if "pressure" in lower or "p_error" in lower:
        return ["p", "pressure_gradient"]
    if "omega" in lower or "vorticity" in lower:
        return ["omega", "pde"]
    return ["pde", "momentum_u", "momentum_v", "continuity"]


def _redistribute_probability_mass(
    values: np.ndarray,
    target: int,
    radius: float,
    maximum: float,
) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    available = max(0.0, float(maximum) - out[target])
    delta = min(float(radius) / 2.0, available, float(np.sum(out)) - out[target])
    if delta <= 0.0:
        return out
    donor_mask = np.ones(out.size, dtype=bool)
    donor_mask[target] = False
    donor_total = float(np.sum(out[donor_mask]))
    out[donor_mask] *= max(0.0, donor_total - delta) / max(donor_total, 1e-12)
    out[target] += delta
    out /= np.sum(out)
    return out


def _redistribute_multiplier_mass(
    values: np.ndarray,
    target: int,
    radius: float,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    n = out.size
    available_up = max(0.0, float(maximum) - out[target])
    donor_mask = np.ones(n, dtype=bool)
    donor_mask[target] = False
    donor_room = float(np.sum(out[donor_mask] - float(minimum)))
    delta = min(float(radius) * n / 2.0, available_up, donor_room)
    if delta <= 0.0:
        return out
    room = np.maximum(out[donor_mask] - float(minimum), 0.0)
    out[donor_mask] -= delta * room / max(float(np.sum(room)), 1e-12)
    out[target] += delta
    # Correct only floating-point drift while retaining exact mean-one mass.
    out += (float(n) - float(np.sum(out))) / float(n)
    return np.clip(out, minimum, maximum)


def _robust_relative_noise(history: list[float], floor: float) -> float:
    if len(history) < 3:
        return float(floor)
    values = np.asarray(history, dtype=float)
    changes = np.diff(values) / (np.abs(values[:-1]) + 1e-12)
    median = float(np.median(changes))
    mad = float(np.median(np.abs(changes - median)))
    return max(float(floor), 1.4826 * mad)
