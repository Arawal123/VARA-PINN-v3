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
from src.physics.navier_stokes import navier_stokes_residuals
from src.physics.taylor_green_repaired import RepairedTaylorGreenVortex
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
        self.controller_diagnostic_calls = 0
        self.controller_diagnostic_seconds = 0.0
        self.full_evaluation_calls = 0
        self.final_full_evaluation_seconds = 0.0
        self._controller_coords_cache: np.ndarray | None = None
        self._controller_metrics_cache: dict[str, float] | None = None
        self._full_metrics_by_time: dict[float, dict[str, float]] = {}
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

    def _train_v2_steps(self, *args: Any, **kwargs: Any) -> None:
        # Any optimizer step invalidates the cheap fixed-grid guard cache.
        self._controller_metrics_cache = None
        super()._train_v2_steps(*args, **kwargs)

    def _build_benchmark(self, config: dict[str, Any]) -> RepairedTaylorGreenVortex:
        cfg = dict(config.get("benchmark_params", {}))
        return RepairedTaylorGreenVortex(
            reynolds=float(cfg.get("reynolds", 100.0)),
            x_min=float(cfg.get("x_min", 0.0)),
            x_max=float(cfg.get("x_max", 2.0 * np.pi)),
            y_min=float(cfg.get("y_min", 0.0)),
            y_max=float(cfg.get("y_max", 2.0 * np.pi)),
            t_min=float(cfg.get("t_min", 0.0)),
            t_max=float(cfg.get("t_max", 1.0)),
            evaluation_time=float(cfg.get("evaluation_time", cfg.get("t_max", 1.0))),
            amplitude=float(cfg.get("amplitude", 1.0)),
        )

    def _controller_diagnostic_coords(self) -> np.ndarray:
        if self._controller_coords_cache is not None:
            return self._controller_coords_cache
        cfg = dict(self.config.get("taylor_green", {}))
        resolution = max(8, int(cfg.get("controller_diagnostic_resolution", 32)))
        times = list(cfg.get("controller_diagnostic_times", [0.5]))
        x0, x1, y0, y1 = self.benchmark.bounds
        x = np.linspace(x0, x1, resolution)
        y = np.linspace(y0, y1, resolution)
        x_grid, y_grid = np.meshgrid(x, y)
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
        self._controller_coords_cache = np.vstack(slices)
        return self._controller_coords_cache

    def _diagnose_reference_free(
        self,
        update_history: bool = True,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], list[Any], np.ndarray]:
        """Build only the residual and boundary maps needed by the controller."""
        started = time.perf_counter()
        coords = self._controller_diagnostic_coords()
        maps, metrics = self._cheap_controller_evaluation(coords)
        self._controller_metrics_cache = dict(metrics)
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

    def _cheap_controller_evaluation(
        self,
        coords_np: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        started = time.perf_counter()
        self.controller_diagnostic_calls += 1
        coords = torch.tensor(coords_np, dtype=torch.float32, device=self.device)
        residuals = navier_stokes_residuals(
            self.model,
            coords,
            nu=self.benchmark.nu,
            steady=False,
        )
        continuity = residuals["f_c"].detach().abs().cpu().numpy()
        momentum_u = residuals["f_u"].detach().abs().cpu().numpy()
        momentum_v = residuals["f_v"].detach().abs().cpu().numpy()
        aggregate = residuals["pde_residual"].detach().cpu().numpy()
        boundary_violation, boundary_mse = self._boundary_violation_map(coords_np)
        pde_mse = float(
            torch.mean(
                residuals["f_u"].pow(2)
                + residuals["f_v"].pow(2)
                + residuals["f_c"].pow(2)
            )
            .detach()
            .cpu()
        )
        bc_weight = float(
            self.config.get("training", {}).get("weights", {}).get("bc", 1.0)
        )
        objective = pde_mse + bc_weight * boundary_mse
        maps = {
            "continuity_residual": continuity,
            "momentum_u_residual": momentum_u,
            "momentum_v_residual": momentum_v,
            "aggregate_pde_residual": aggregate,
            "boundary_violation": boundary_violation,
        }
        metrics = {
            "pde_residual_mean": float(np.mean(aggregate)),
            "continuity_residual_mean": float(np.mean(continuity)),
            "momentum_residual_mean": float(
                np.mean(np.sqrt(momentum_u * momentum_u + momentum_v * momentum_v))
            ),
            "boundary_condition_error": float(np.mean(boundary_violation[boundary_violation > 0.0]))
            if np.any(boundary_violation > 0.0)
            else 0.0,
            "unweighted_validation_loss": float(objective),
            "controller_objective_j": float(objective),
        }
        self.controller_diagnostic_seconds += time.perf_counter() - started
        return maps, metrics

    def _boundary_violation_map(
        self,
        coords_np: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        x0, x1, y0, y1 = self.benchmark.bounds
        mask = (
            np.isclose(coords_np[:, 0], x0)
            | np.isclose(coords_np[:, 0], x1)
            | np.isclose(coords_np[:, 1], y0)
            | np.isclose(coords_np[:, 1], y1)
        )
        output = np.zeros((coords_np.shape[0], 1), dtype=float)
        if not np.any(mask):
            return output, 0.0
        selected = torch.tensor(
            coords_np[mask],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            prediction = self.model(selected)[:, 0:2]
            reference = self.benchmark.exact_torch(selected)
            target = torch.cat([reference["u"], reference["v"]], dim=1)
            squared = torch.sum((prediction - target).pow(2), dim=1, keepdim=True)
        output[mask] = torch.sqrt(squared).cpu().numpy()
        return output, float(torch.mean(squared).cpu())

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
        if self._controller_metrics_cache is None:
            _maps, metrics = self._cheap_controller_evaluation(
                self._controller_diagnostic_coords()
            )
            self._controller_metrics_cache = dict(metrics)
        return dict(self._controller_metrics_cache)

    def controller_metrics(self, coords: np.ndarray) -> dict[str, float]:
        """Keep checkpoint/early-stop checks on the same cheap fixed grid."""
        return self._guard_metrics(self._controller_diagnostic_coords())

    def evaluate_metrics(self, coords: np.ndarray) -> dict[str, float]:
        """Run the full paper metric set without periodic-invalid streamfunction scores."""
        started = time.perf_counter()
        self.full_evaluation_calls += 1
        metrics = evaluate_on_grid(
            self.model,
            self.benchmark,
            coords,
            self.device,
            steady=False,
            residual_interior_only=self.residual_interior_only(),
            include_reference_metrics=True,
            include_streamfunction_metrics=False,
        )
        elapsed = time.perf_counter() - started
        self.final_full_evaluation_seconds += elapsed
        unique_times = np.unique(np.asarray(coords)[:, 2])
        if unique_times.size == 1:
            self._full_metrics_by_time[float(unique_times[0])] = dict(metrics)
        return metrics

    def run(self) -> dict[str, float]:
        run_started = time.perf_counter()
        metrics = super().run()
        return self._finalize_taylor_green_metrics(metrics, run_started)

    def _finalize_taylor_green_metrics(
        self,
        metrics: dict[str, Any],
        run_started: float,
    ) -> dict[str, Any]:
        """Attach the common repaired temporal report to either training method."""
        final_phase_started = time.perf_counter()
        measured_evaluation_before = self.final_full_evaluation_seconds
        temporal, rows = self._temporal_accuracy_metrics()
        temporal_elapsed = time.perf_counter() - final_phase_started
        measured_evaluation_delta = (
            self.final_full_evaluation_seconds - measured_evaluation_before
        )
        self.final_full_evaluation_seconds += max(
            0.0,
            temporal_elapsed - measured_evaluation_delta,
        )
        metrics.update(temporal)
        metrics.update(
            {
                "taylor_green_total_runtime_sec": float(
                    time.perf_counter() - run_started
                ),
                "taylor_green_training_time_sec": float(
                    metrics.get("optimization_wall_clock_sec", float("nan"))
                ),
                "taylor_green_controller_diagnostic_time_sec": float(
                    self.controller_diagnostic_seconds
                ),
                "taylor_green_final_evaluation_time_sec": float(
                    self.final_full_evaluation_seconds
                ),
                "taylor_green_full_evaluation_calls": int(self.full_evaluation_calls),
                "taylor_green_controller_diagnostic_calls": int(
                    self.controller_diagnostic_calls
                ),
                "taylor_green_streamfunction_diagnostics_status": (
                    "quarantined_optional_qualitative"
                ),
                "streamfunction_consistency_rmse": float("nan"),
            }
        )
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
        print(
            "Taylor-Green timing: "
            f"total={metrics['taylor_green_total_runtime_sec']:.2f}s "
            f"training={metrics['taylor_green_training_time_sec']:.2f}s "
            f"controller_diagnostics={self.controller_diagnostic_seconds:.2f}s "
            f"final_evaluation={self.final_full_evaluation_seconds:.2f}s "
            f"controller_calls={self.controller_diagnostic_calls} "
            f"full_evaluation_calls={self.full_evaluation_calls}"
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
        resolution = max(8, int(cfg.get("final_evaluation_resolution", 48)))
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
            cache_key = float(value)
            evaluated = self._full_metrics_by_time.get(cache_key)
            if evaluated is None or int(evaluated.get("num_eval_points", -1)) != int(
                coords.shape[0]
            ):
                evaluated = self.evaluate_metrics(coords)
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
                    "momentum_residual_mean": float(
                        evaluated["momentum_residual_mean"]
                    ),
                    "boundary_condition_error": float(
                        evaluated["boundary_condition_error"]
                    ),
                    "periodic_velocity_mismatch_rmse": self._periodic_velocity_mismatch(
                        float(value),
                        resolution,
                    ),
                }
            )
        velocity = [row["velocity_rel_l2"] for row in rows]
        pressure = [row["p_rel_l2_centered"] for row in rows]
        omega = [row["omega_rel_l2"] for row in rows]
        pde = [row["pde_residual_mean"] for row in rows]
        continuity = [row["continuity_residual_mean"] for row in rows]
        momentum = [row["momentum_residual_mean"] for row in rows]
        boundary = [row["boundary_condition_error"] for row in rows]
        periodic = [row["periodic_velocity_mismatch_rmse"] for row in rows]
        u_values = [row["u_rel_l2"] for row in rows]
        v_values = [row["v_rel_l2"] for row in rows]
        thresholds = dict(cfg.get("accuracy_thresholds", {}))
        velocity_limit = float(thresholds.get("velocity_rel_l2", 0.02))
        pressure_limit = float(thresholds.get("pressure_rel_l2", 0.05))
        omega_limit = float(thresholds.get("omega_rel_l2", 0.05))
        pde_limit = float(thresholds.get("pde_residual_mean", 0.01))
        continuity_limit = float(thresholds.get("continuity_residual_mean", 0.005))
        vorticity_sanity = self.benchmark.vorticity_reference_sanity(
            resolution=resolution,
            time=float(times[-1]),
        )
        if vorticity_sanity >= 1e-3:
            raise RuntimeError(
                "Taylor-Green analytical vorticity convention failed its sanity check: "
                f"relative difference={vorticity_sanity:.3e}."
            )
        summary = {
            "taylor_green_initial_condition_mismatch": self.initial_condition_mismatch(),
            "taylor_green_mean_time_u_rel_l2": float(np.mean(u_values)),
            "taylor_green_mean_time_v_rel_l2": float(np.mean(v_values)),
            "taylor_green_mean_time_velocity_rel_l2": float(np.mean(velocity)),
            "taylor_green_mean_time_pressure_rel_l2_centered": float(np.mean(pressure)),
            "taylor_green_mean_time_omega_rel_l2": float(np.mean(omega)),
            "taylor_green_mean_time_pde_residual_mean": float(np.mean(pde)),
            "taylor_green_mean_time_continuity_residual_mean": float(
                np.mean(continuity)
            ),
            "taylor_green_mean_time_momentum_residual_mean": float(
                np.mean(momentum)
            ),
            "taylor_green_mean_time_boundary_condition_error": float(
                np.mean(boundary)
            ),
            "taylor_green_mean_time_periodic_velocity_mismatch_rmse": float(
                np.mean(periodic)
            ),
            "taylor_green_worst_time_velocity_rel_l2": float(max(velocity)),
            "taylor_green_worst_time_pressure_rel_l2_centered": float(max(pressure)),
            "taylor_green_worst_time_omega_rel_l2": float(max(omega)),
            "taylor_green_worst_time_pde_residual_mean": float(max(pde)),
            "taylor_green_temporal_slices_evaluated": float(len(rows)),
            "taylor_green_vorticity_reference_sanity_rel_l2": float(
                vorticity_sanity
            ),
        }
        summary["taylor_green_accuracy_pass"] = float(
            summary["taylor_green_worst_time_velocity_rel_l2"] <= velocity_limit
            and summary["taylor_green_worst_time_pressure_rel_l2_centered"]
            <= pressure_limit
            and summary["taylor_green_worst_time_omega_rel_l2"] <= omega_limit
            and summary["taylor_green_worst_time_pde_residual_mean"] <= pde_limit
            and max(continuity) <= continuity_limit
        )
        return summary, rows

    def _periodic_velocity_mismatch(self, time_value: float, resolution: int) -> float:
        x0, x1, y0, y1 = self.benchmark.bounds
        x = np.linspace(x0, x1, resolution)
        y = np.linspace(y0, y1, resolution)
        left = np.column_stack([np.full_like(y, x0), y, np.full_like(y, time_value)])
        right = np.column_stack([np.full_like(y, x1), y, np.full_like(y, time_value)])
        bottom = np.column_stack([x, np.full_like(x, y0), np.full_like(x, time_value)])
        top = np.column_stack([x, np.full_like(x, y1), np.full_like(x, time_value)])
        coords = torch.tensor(
            np.vstack([left, right, bottom, top]),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            velocity = self.model(coords)[:, 0:2]
        n = resolution
        left_v, right_v, bottom_v, top_v = torch.split(velocity, n, dim=0)
        mismatch = torch.cat([left_v - right_v, bottom_v - top_v], dim=0)
        return float(torch.sqrt(torch.mean(mismatch.pow(2))).cpu())
