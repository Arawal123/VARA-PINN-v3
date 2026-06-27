"""Isolated, corrected VARA V2 wiring for the Taylor--Green vortex."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import evaluate_on_grid
from src.models.taylor_green_initial import TaylorGreenHardInitialCondition
from src.training.vara_v2_trainer import VARAV2Trainer
from src.utils.io import save_json


class TaylorGreenVARAV2Trainer(VARAV2Trainer):
    """VARA V2 with Taylor--Green-specific initial and temporal wiring.

    This subclass exists to keep the repair entirely outside the shared
    trainer.  Other benchmarks continue to instantiate ``VARAV2Trainer`` and
    therefore cannot observe any behavior change from this module.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        name = str(config.get("benchmark", "")).lower()
        if name not in {"taylor_green", "taylor-green", "tgv"}:
            raise ValueError(
                "TaylorGreenVARAV2Trainer only accepts the Taylor--Green benchmark."
            )
        if config.get("warm_start_checkpoint"):
            raise ValueError(
                "The isolated Taylor--Green hard-initial trainer does not accept "
                "legacy warm-start checkpoints."
            )
        super().__init__(deepcopy(config))
        self.model = TaylorGreenHardInitialCondition(
            self.model,
            self.benchmark,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.base_learning_rate,
        )
        # Rebuild the fixed probe after installing the hard-initial model.  The
        # sampler snapshot keeps ordinary optimization draws deterministic.
        sampling_snapshot = self.sampling_state_snapshot()
        self._probe_batch = self._make_probe_batch()
        self.restore_sampling_state(sampling_snapshot)

    def _space_time_diagnostic_coords(self) -> np.ndarray:
        cfg = dict(self.config.get("validation", {}))
        nx = int(cfg.get("nx", 48))
        ny = int(cfg.get("ny", 48))
        x0, x1, y0, y1 = self.benchmark.bounds
        x = np.linspace(x0, x1, nx)
        y = np.linspace(y0, y1, ny)
        x_grid, y_grid = np.meshgrid(x, y)
        nt = max(1, int(self.config.get("patches", {}).get("nt_patches", 1)))
        edges = np.linspace(self.benchmark.t_min, self.benchmark.t_max, nt + 1)
        times = 0.5 * (edges[:-1] + edges[1:])
        slices = [
            np.column_stack(
                [
                    x_grid.reshape(-1),
                    y_grid.reshape(-1),
                    np.full(x_grid.size, float(value)),
                ]
            )
            for value in times
        ]
        return np.vstack(slices)

    def _diagnose_reference_free(
        self,
        update_history: bool = True,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], list[Any], np.ndarray]:
        """Diagnose every temporal patch instead of only the final time."""
        started = time.perf_counter()
        coords = self._space_time_diagnostic_coords()
        maps = self.diagnostic_builder().build(coords, mode="residual_only")
        configured = list(self.config.get("diagnostics", {}).get("variables", []))
        names = [
            name
            for name in configured
            if name in maps
            and not any(
                token in name.lower()
                for token in ("error", "reference", "ghia", "cfd")
            )
        ]
        if not names:
            names = [
                "continuity_residual",
                "momentum_u_residual",
                "momentum_v_residual",
                "aggregate_pde_residual",
                "boundary_violation",
            ]
        if not self.variable_awareness_enabled:
            maps = {
                **maps,
                "regional_severity": np.nanmean(
                    np.vstack(
                        [
                            np.asarray(maps[name], dtype=float).reshape(1, -1)
                            for name in names
                        ]
                    ),
                    axis=0,
                ),
            }
            names = ["regional_severity"]
        self.v2_controller.assert_reference_free(names)
        original = self.patch_scorer.diagnostics
        self.patch_scorer.diagnostics = names
        normalized, scored_names = self.patch_scorer.compute(
            maps,
            coords,
            update_ema=update_history,
        )
        raw = np.asarray(self.patch_scorer.last_raw_scores, dtype=float)
        self.patch_scorer.diagnostics = original
        weak_regions = self.weak_detector.detect(
            normalized,
            scored_names,
            self.patch_grid,
        )
        self.compute_tracker.add_phase_time(
            "diagnostics",
            time.perf_counter() - started,
        )
        return maps, raw, scored_names, weak_regions, coords

    def initial_condition_mismatch(self) -> float:
        """Return the configured training-condition MSE at ``t=t_min``."""
        cfg = dict(self.config.get("taylor_green", {}))
        resolution = max(4, int(cfg.get("initial_condition_metric_resolution", 32)))
        x0, x1, y0, y1 = self.benchmark.bounds
        x_grid, y_grid = np.meshgrid(
            np.linspace(x0, x1, resolution),
            np.linspace(y0, y1, resolution),
        )
        coords = torch.tensor(
            np.column_stack(
                [
                    x_grid.reshape(-1),
                    y_grid.reshape(-1),
                    np.full(x_grid.size, float(self.benchmark.t_min)),
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            prediction = self.model(coords)
            reference = self.benchmark.exact_torch(coords)
            target = torch.cat(
                [reference["u"], reference["v"], reference["p"]],
                dim=1,
            )
            mismatch = torch.mean((prediction - target).pow(2))
        return float(mismatch.detach().cpu())

    def _guard_metrics(self, coords: np.ndarray) -> dict[str, float]:
        metrics = super()._guard_metrics(coords)
        metrics["initial_condition_mismatch"] = self.initial_condition_mismatch()
        return metrics

    def run(self) -> dict[str, float]:
        metrics = super().run()
        temporal, rows = self._temporal_accuracy_metrics()
        metrics.update(temporal)
        pd.DataFrame(rows).to_csv(
            self.run_dir / "taylor_green_temporal_metrics.csv",
            index=False,
        )
        save_json(metrics, self.run_dir / "summary.json")
        pd.DataFrame([metrics]).to_csv(
            self.table_dir / "summary.csv",
            index=False,
        )
        pd.DataFrame([metrics]).to_csv(
            self.run_dir / "summary_table.csv",
            index=False,
        )
        return metrics

    def _temporal_accuracy_metrics(
        self,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        cfg = dict(self.config.get("taylor_green", {}))
        times = list(
            cfg.get(
                "evaluation_times",
                [
                    self.benchmark.t_min,
                    self.benchmark.t_min
                    + 0.25 * (self.benchmark.t_max - self.benchmark.t_min),
                    self.benchmark.t_min
                    + 0.50 * (self.benchmark.t_max - self.benchmark.t_min),
                    self.benchmark.t_min
                    + 0.75 * (self.benchmark.t_max - self.benchmark.t_min),
                    self.benchmark.t_max,
                ],
            )
        )
        resolution = max(8, int(cfg.get("temporal_metric_resolution", 48)))
        x0, x1, y0, y1 = self.benchmark.bounds
        x_grid, y_grid = np.meshgrid(
            np.linspace(x0, x1, resolution),
            np.linspace(y0, y1, resolution),
        )
        rows: list[dict[str, float]] = []
        for value in times:
            coords = np.column_stack(
                [
                    x_grid.reshape(-1),
                    y_grid.reshape(-1),
                    np.full(x_grid.size, float(value)),
                ]
            )
            evaluated = evaluate_on_grid(
                self.model,
                self.benchmark,
                coords,
                self.device,
                steady=False,
                include_reference_metrics=True,
                include_streamfunction_metrics=False,
            )
            rows.append(
                {
                    "time": float(value),
                    "velocity_rel_l2": float(evaluated["velocity_full_rel_l2"]),
                    "u_rel_l2": float(evaluated["u_rel_l2"]),
                    "v_rel_l2": float(evaluated["v_rel_l2"]),
                    "p_rel_l2_centered": float(evaluated["p_rel_l2_centered"]),
                    "omega_rel_l2": float(evaluated["omega_rel_l2"]),
                    "pde_residual_mean": float(evaluated["pde_residual_mean"]),
                    "continuity_residual_mean": float(
                        evaluated["continuity_residual_mean"]
                    ),
                    "boundary_condition_error": float(
                        evaluated["boundary_condition_error"]
                    ),
                }
            )
        velocity = [row["velocity_rel_l2"] for row in rows]
        pressure = [row["p_rel_l2_centered"] for row in rows]
        omega = [row["omega_rel_l2"] for row in rows]
        pde = [row["pde_residual_mean"] for row in rows]
        thresholds = dict(cfg.get("accuracy_thresholds", {}))
        velocity_limit = float(thresholds.get("velocity_rel_l2", 0.02))
        pressure_limit = float(thresholds.get("pressure_rel_l2", 0.05))
        omega_limit = float(thresholds.get("omega_rel_l2", 0.05))
        pde_limit = float(thresholds.get("pde_residual_mean", 0.01))
        summary = {
            "taylor_green_initial_condition_mismatch": self.initial_condition_mismatch(),
            "taylor_green_worst_time_velocity_rel_l2": float(max(velocity)),
            "taylor_green_worst_time_pressure_rel_l2_centered": float(max(pressure)),
            "taylor_green_worst_time_omega_rel_l2": float(max(omega)),
            "taylor_green_worst_time_pde_residual_mean": float(max(pde)),
            "taylor_green_temporal_slices_evaluated": float(len(rows)),
        }
        summary["taylor_green_accuracy_pass"] = float(
            summary["taylor_green_worst_time_velocity_rel_l2"] <= velocity_limit
            and summary["taylor_green_worst_time_pressure_rel_l2_centered"]
            <= pressure_limit
            and summary["taylor_green_worst_time_omega_rel_l2"] <= omega_limit
            and summary["taylor_green_worst_time_pde_residual_mean"] <= pde_limit
        )
        return summary, rows
