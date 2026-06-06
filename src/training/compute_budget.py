"""Opt-in compute accounting and stopping for fair method comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class ComputeTracker:
    """Track optimization effort without changing legacy training schedules."""

    config: dict[str, Any]
    started_at: float | None = None
    optimizer_steps: int = 0
    auxiliary_optimizer_steps: int = 0
    objective_evaluations: int = 0
    collocation_evaluations: int = 0
    boundary_evaluations: int = 0
    data_evaluations: int = 0
    phase_seconds: dict[str, float] = field(default_factory=dict)
    stop_reason: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    @property
    def budget_type(self) -> str:
        value = str(self.config.get("type", "optimizer_steps")).lower()
        aliases = {
            "epochs": "optimizer_steps",
            "gradient_steps": "optimizer_steps",
            "steps": "optimizer_steps",
            "wall_clock": "wall_clock_sec",
            "time": "wall_clock_sec",
            "points": "collocation_evaluations",
        }
        return aliases.get(value, value)

    @property
    def budget_value(self) -> float:
        return float(self.config.get("value", float("inf")))

    def start(self) -> None:
        if self.started_at is None:
            self.started_at = time.perf_counter()

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, time.perf_counter() - self.started_at)

    def can_start_objective(self, n_collocation: int = 0) -> bool:
        """Return whether another objective evaluation fits the selected budget."""
        if not self.enabled:
            return True
        kind = self.budget_type
        value = self.budget_value
        if kind == "optimizer_steps":
            allowed = self.optimizer_steps < int(value)
        elif kind == "objective_evaluations":
            allowed = self.objective_evaluations < int(value)
        elif kind == "collocation_evaluations":
            allowed = self.collocation_evaluations + int(n_collocation) <= int(value)
        elif kind == "wall_clock_sec":
            allowed = self.elapsed() < value
        else:
            raise ValueError(
                "Unknown compute budget type "
                f"'{kind}'. Use optimizer_steps, objective_evaluations, "
                "collocation_evaluations, or wall_clock_sec."
            )
        if not allowed and not self.stop_reason:
            self.stop_reason = f"{kind}_budget_reached"
        return allowed

    def record_objective(self, batch: dict[str, Any]) -> None:
        self.objective_evaluations += 1
        self.collocation_evaluations += _batch_size(batch.get("xy_f"))
        self.boundary_evaluations += _batch_size(batch.get("xy_bc"))
        self.data_evaluations += _batch_size(batch.get("xy_data"))

    def record_optimizer_step(self) -> None:
        self.optimizer_steps += 1

    def record_auxiliary_optimizer_step(self) -> None:
        self.auxiliary_optimizer_steps += 1

    def add_phase_time(self, name: str, seconds: float) -> None:
        self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + max(0.0, float(seconds))

    def exhausted(self) -> bool:
        return not self.can_start_objective(0)

    def summary(self) -> dict[str, Any]:
        total = self.elapsed()
        optimization = self.phase_seconds.get("optimization", 0.0)
        diagnostics = self.phase_seconds.get("diagnostics", 0.0)
        evaluation = self.phase_seconds.get("evaluation", 0.0)
        accounted = optimization + diagnostics + evaluation
        overhead = max(0.0, total - accounted)
        return {
            "compute_budget_enabled": self.enabled,
            "compute_budget_type": self.budget_type if self.enabled else "disabled",
            "compute_budget_value": self.budget_value if self.enabled else None,
            "compute_budget_stop_reason": self.stop_reason,
            "training_wall_clock_sec": total,
            "optimization_wall_clock_sec": optimization,
            "diagnostic_wall_clock_sec": diagnostics,
            "evaluation_wall_clock_sec": evaluation,
            "controller_and_io_overhead_sec": overhead,
            "controller_and_io_overhead_percent": 100.0 * overhead / max(total, 1e-12),
            "optimizer_steps": self.optimizer_steps,
            "auxiliary_optimizer_steps": self.auxiliary_optimizer_steps,
            "objective_evaluations": self.objective_evaluations,
            "collocation_evaluations": self.collocation_evaluations,
            "boundary_evaluations": self.boundary_evaluations,
            "data_evaluations": self.data_evaluations,
        }


def _batch_size(value: Any) -> int:
    if value is None:
        return 0
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return 0
    return int(shape[0])
