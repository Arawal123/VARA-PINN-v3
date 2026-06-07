"""Shared trainer utilities."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.diagnostics import DiagnosticMapBuilder, PatchGrid, PatchScorer, WeakRegionDetector
from src.evaluation.metrics import evaluate_on_grid
from src.losses.base_losses import compute_global_losses, compute_pointwise_losses, weighted_sum
from src.losses.local_losses import compute_local_weighted_loss
from src.losses.modern_baselines import gradient_enhanced_pointwise_losses
from src.losses.pressure_losses import pressure_anchor_loss
from src.losses.vorticity_losses import vorticity_transport_residual
from src.models import build_mlp_from_config
from src.physics.kovasznay import KovasznayFlow
from src.physics.pressure_poisson import pressure_poisson_residual
from src.physics.rectangular_benchmarks import (
    BoundaryStressBoxFlow,
    DoubleVortexBoxFlow,
    LidDrivenCavityQualitative,
    PoiseuilleChannelFlow,
)
from src.physics.taylor_green import TaylorGreenVortex
from src.sampling import BoundarySampler, MixedAdaptiveSampler, UniformSampler
from src.sampling.residual_sampler import sample_from_score_grid
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.compute_budget import ComputeTracker
from src.training.lbfgs_utils import make_lbfgs_closure
from src.utils.config import save_config
from src.utils.device import get_device
from src.utils.io import ensure_dir, save_json
from src.utils.logging import CSVLogger, JSONListLogger, make_run_id
from src.utils.seed import set_seed
from src.visualization.controller_plots import save_intervention_timeline, save_patch_score_map
from src.visualization.fields import save_field_panel, save_prediction_reference_error_panel
from src.visualization.heatmaps import save_heatmap
from src.visualization.streamlines import save_streamfunction_contours, save_streamlines


class ExperimentTrainer:
    """Reusable training scaffold for Kovasznay-first experiments."""

    def __init__(self, config: dict[str, Any], mode: str) -> None:
        self.config = config
        self.mode = mode
        self.seed = int(config.get("seed", 0))
        set_seed(self.seed, deterministic=bool(config.get("deterministic", True)))
        self.device = get_device(config.get("device", "auto"))
        self.benchmark = self._build_benchmark(config)
        self.steady = bool(config.get("pde", {}).get("steady", True))
        self.model = build_mlp_from_config(config, self.benchmark.bounds).to(self.device)
        optim_cfg = config.get("optimizer", {})
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(optim_cfg.get("lr", 1e-3)))
        self.optimizer_stage = "adam"
        self.final_repair_status: dict[str, Any] = {}
        self.continuation_anchor_model: torch.nn.Module | None = None
        self.continuation_replay_points: torch.Tensor | None = None
        self.continuation_replay_targets: torch.Tensor | None = None
        self.warm_start_status = self._maybe_load_warm_start(config)
        continuation_enabled = bool(config.get("continuation_anchor", {}).get("enabled", False))
        replay_enabled = bool(config.get("continuation_replay", {}).get("enabled", False))
        if self.warm_start_status.get("loaded") and (continuation_enabled or replay_enabled):
            self.continuation_anchor_model = deepcopy(self.model).to(self.device).eval()
            for parameter in self.continuation_anchor_model.parameters():
                parameter.requires_grad_(False)
        self.repair_rng = np.random.default_rng(self.seed + 7919)
        self.global_step = 0
        self.best_score = math.inf
        self.compute_tracker = ComputeTracker(dict(config.get("compute_budget", {})))

        patch_cfg = config.get("patches", {})
        self.patch_grid = PatchGrid(
            self.benchmark.bounds,
            nx_patches=int(patch_cfg.get("nx_patches", 4)),
            ny_patches=int(patch_cfg.get("ny_patches", 4)),
            nt_patches=int(patch_cfg.get("nt_patches", 1)),
            t_bounds=getattr(self.benchmark, "t_bounds", None),
        )
        diag_cfg = config.get("diagnostics", {})
        self.patch_scorer = PatchScorer(
            self.patch_grid,
            diagnostics=diag_cfg.get("variables"),
            aggregation=diag_cfg.get("aggregation", "mean"),
            normalization=diag_cfg.get("normalization", "percentile"),
            percentile=float(diag_cfg.get("aggregation_percentile", 90.0)),
            ema_rho=float(diag_cfg.get("ema_rho", 0.8)),
        )
        weak_cfg = config.get("weak_regions", {})
        self.weak_detector = WeakRegionDetector(
            percentile_threshold=float(weak_cfg.get("percentile_threshold", 80.0)),
            top_k_per_variable=int(weak_cfg.get("top_k_per_variable", 2)),
            min_active_patches=int(weak_cfg.get("min_active_patches", 1)),
            max_active_patches=int(weak_cfg.get("max_active_patches", 8)),
            persistence_cycles=int(weak_cfg.get("persistence_cycles", 1)),
        )

        exp_cfg = config.get("experiments", {})
        root = Path(exp_cfg.get("root", "experiments"))
        self.run_id = exp_cfg.get("run_id") or make_run_id(config.get("benchmark", "kovasznay"), mode, self.seed)
        if bool(exp_cfg.get("flat_layout", False)):
            self.run_dir = ensure_dir(root / "logs")
            self.checkpoint_dir = ensure_dir(root / "checkpoints")
            self.figure_dir = ensure_dir(root / "figures")
            self.table_dir = ensure_dir(root / "tables")
            self.metrics_dir = ensure_dir(root / "metrics")
        else:
            self.run_dir = ensure_dir(root / "logs" / self.run_id)
            self.checkpoint_dir = ensure_dir(root / "checkpoints" / self.run_id)
            self.figure_dir = ensure_dir(root / "figures" / self.run_id)
            self.table_dir = ensure_dir(root / "tables" / self.run_id)
            self.metrics_dir = ensure_dir(root / "metrics" / self.run_id)
        save_config(config, self.run_dir / "config_snapshot.yaml")

        self.metrics_logger = CSVLogger(self.run_dir / "metrics.csv")
        self.loss_logger = CSVLogger(self.run_dir / "losses.csv")
        self.action_logger = JSONListLogger(self.run_dir / "action_log.json")
        self.weak_logger = JSONListLogger(self.run_dir / "weak_region_log.json")
        self.score_logger = JSONListLogger(self.run_dir / "patch_scores.json")
        self.accept_logger = JSONListLogger(self.run_dir / "acceptance_log.json")
        self.action_records: list[dict[str, Any]] = []
        self.last_losses: dict[str, float] = {}

        t_bounds = getattr(self.benchmark, "t_bounds", None)
        self.uniform_sampler = UniformSampler(self.benchmark.bounds, self.device, self.seed, t_bounds=t_bounds)
        self.boundary_sampler = BoundarySampler(self.benchmark.bounds, self.device, self.seed + 1, t_bounds=t_bounds)
        sampler_cfg = config.get("sampling", {})
        self.adaptive_sampler = MixedAdaptiveSampler(
            self.benchmark.bounds,
            self.patch_grid,
            self.device,
            self.seed + 2,
            mixture=sampler_cfg.get("mixture"),
        )
        self._initialize_continuation_replay()

    def _maybe_load_warm_start(self, config: dict[str, Any]) -> dict[str, Any]:
        checkpoint = config.get("warm_start_checkpoint")
        if not checkpoint:
            return {"enabled": False, "loaded": False}
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {path}")
        warm_cfg = config.get("warm_start", {})
        load_optimizer = bool(warm_cfg.get("load_optimizer", False))
        try:
            payload = load_checkpoint(path, self.model, self.optimizer if load_optimizer else None)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not warm-start from {path}. Check that the model architecture matches the checkpoint."
            ) from exc
        return {
            "enabled": True,
            "loaded": True,
            "checkpoint": str(path),
            "load_optimizer": load_optimizer,
            "source_epoch": payload.get("epoch"),
            "source_cycle": payload.get("cycle"),
        }

    def _build_benchmark(self, config: dict[str, Any]) -> Any:
        name = config.get("benchmark", "kovasznay").lower()
        cfg = config.get("benchmark_params", {})
        common = {
            "reynolds": float(cfg.get("reynolds", 40.0)),
            "x_min": float(cfg.get("x_min", -0.5 if name == "kovasznay" else 0.0)),
            "x_max": float(cfg.get("x_max", 1.0)),
            "y_min": float(cfg.get("y_min", -0.5 if name == "kovasznay" else 0.0)),
            "y_max": float(cfg.get("y_max", 1.5 if name == "kovasznay" else 1.0)),
        }
        if name == "kovasznay":
            return KovasznayFlow(**common)
        if name in {"taylor_green", "taylor-green", "tgv"}:
            return TaylorGreenVortex(
                **common,
                t_min=float(cfg.get("t_min", 0.0)),
                t_max=float(cfg.get("t_max", 1.0)),
                evaluation_time=float(cfg.get("evaluation_time", cfg.get("t_max", 1.0))),
                amplitude=float(cfg.get("amplitude", 1.0)),
            )
        rectangular = {**common, "amplitude": float(cfg.get("amplitude", 1.0))}
        if name in {"channel_inflow_outflow", "channel", "poiseuille"}:
            return PoiseuilleChannelFlow(**rectangular)
        if name in {"double_vortex_box", "double_vortex", "recirculating_vortex"}:
            return DoubleVortexBoxFlow(**rectangular)
        if name in {"boundary_condition_stress_test", "bc_stress"}:
            return BoundaryStressBoxFlow(**rectangular)
        if name in {"lid_driven_cavity", "cavity"}:
            full_field_reference_path = cfg.get("full_field_reference_path")
            return LidDrivenCavityQualitative(
                **rectangular,
                lid_velocity=float(cfg.get("lid_velocity", 1.0)),
                lid_corner_regularization_width=float(
                    cfg.get("lid_corner_regularization_width", 0.0)
                ),
                reference=str(cfg.get("reference", "none")),
                reference_path=cfg.get("reference_path"),
                full_field_reference_path=full_field_reference_path,
                profile_only=bool(cfg.get("profile_only", full_field_reference_path is None)),
                has_reference=bool(full_field_reference_path) and not bool(cfg.get("profile_only", False)),
                reference_kind="full_field_cfd" if full_field_reference_path else str(cfg.get("reference", "residual_only")),
            )
        if name in {"rectangular_aspect_ratio", "rectangular_aspect_ratio_sweep"}:
            return PoiseuilleChannelFlow(**rectangular)
        raise NotImplementedError(f"Unknown benchmark: {name}.")

    def initial_batch(self) -> dict[str, Any]:
        train_cfg = self.config.get("training", {})
        n_f = int(train_cfg.get("n_collocation", 1024))
        n_bc = int(train_cfg.get("n_boundary", 256))
        n_data = int(train_cfg.get("n_data", 256))
        xy_f = self.uniform_sampler.sample(n_f)
        xy_bc = self._sample_boundary(n_bc)
        xy_data = self._sample_data(n_data)
        return self.make_batch(xy_f, xy_bc, xy_data)

    def _sample_data(self, n: int) -> torch.Tensor:
        """Sample reference data; time-dependent benchmarks use initial data only."""
        points = self.uniform_sampler.sample(n)
        t_bounds = getattr(self.benchmark, "t_bounds", None)
        if t_bounds is not None and n > 0 and points.shape[1] >= 3:
            points[:, 2] = float(t_bounds[0])
        return points

    def _sample_boundary_numpy(self, n: int) -> np.ndarray:
        boundary_cfg = self.config.get("sampling", {}).get("cavity_boundary", {})
        if self._is_lid_driven_cavity() and bool(boundary_cfg.get("enabled", True)):
            return self.boundary_sampler.sample_lid_cavity_numpy(
                n,
                lid_fraction=float(boundary_cfg.get("lid_fraction", 0.45)),
                corner_fraction=float(boundary_cfg.get("corner_fraction", 0.25)),
                corner_width=float(boundary_cfg.get("corner_width", 0.12)),
            )
        return self.boundary_sampler.sample_numpy(n)

    def _sample_boundary(self, n: int) -> torch.Tensor:
        return torch.tensor(self._sample_boundary_numpy(n), dtype=torch.float32, device=self.device)

    def _is_lid_driven_cavity(self) -> bool:
        return str(self.config.get("benchmark", "")).lower() in {"lid_driven_cavity", "cavity"}

    def make_batch(self, xy_f: torch.Tensor, xy_bc: torch.Tensor, xy_data: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            targets = self.benchmark.exact_torch(xy_data)
        return {"xy_f": xy_f, "xy_bc": xy_bc, "xy_data": xy_data, "targets_data": targets}

    def validation_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config.get("validation", {})
        return self.benchmark.grid(int(cfg.get("nx", 50)), int(cfg.get("ny", 50)))

    def test_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config.get("test", {})
        return self.benchmark.grid(int(cfg.get("nx", 64)), int(cfg.get("ny", 64)))

    def residual_interior_only(self) -> bool:
        return bool(
            self.config.get("evaluation", {}).get("residual_interior_only", False)
        )

    def evaluate_metrics(self, coords: np.ndarray) -> dict[str, float]:
        return evaluate_on_grid(
            self.model,
            self.benchmark,
            coords,
            self.device,
            self.steady,
            residual_interior_only=self.residual_interior_only(),
        )

    def controller_metrics(self, coords: np.ndarray) -> dict[str, float]:
        metrics = self.evaluate_metrics(coords)
        enabled = bool(
            self.config.get("evaluation", {}).get(
                "controller_reference_metrics_enabled",
                True,
            )
        )
        if enabled:
            return metrics
        reference_names = (
            "u_rel_l2",
            "v_rel_l2",
            "p_rel_l2_centered",
            "speed_rel_l2",
            "omega_rel_l2",
            "u_full_rel_l2",
            "v_full_rel_l2",
            "velocity_full_rel_l2",
            "p_full_rel_l2_centered",
            "omega_full_rel_l2",
            "u_rmse",
            "v_rmse",
            "p_rmse_centered",
            "omega_rmse",
            "velocity_mag_rmse",
            "velocity_mag_mae",
            "unweighted_data_loss",
            "u_centerline_rmse",
            "v_centerline_rmse",
            "u_centerline_rel_l2",
            "v_centerline_rel_l2",
            "centerline_profile_score",
        )
        for name in reference_names:
            if name in metrics:
                metrics[name] = float("nan")
        metrics["unweighted_validation_loss"] = float(
            metrics["unweighted_pde_loss"] + metrics["unweighted_bc_loss"]
        )
        metrics["cavity_benchmark_score"] = float(
            metrics["pde_residual_mean"]
            + metrics["continuity_residual_mean"]
            + metrics["momentum_residual_mean"]
            + metrics["boundary_condition_error"]
        )
        metrics["controller_reference_metrics_enabled"] = False
        return metrics

    def diagnostic_builder(self) -> DiagnosticMapBuilder:
        return DiagnosticMapBuilder(
            self.model,
            self.benchmark,
            self.device,
            self.steady,
            residual_interior_only=self.residual_interior_only(),
        )

    def controller_diagnostic_mode(self) -> str:
        reference_enabled = bool(
            self.config.get("evaluation", {}).get(
                "controller_reference_metrics_enabled",
                True,
            )
        )
        if not reference_enabled:
            return "residual_only"
        return str(
            self.config.get("diagnostics", {}).get("mode", "full_reference")
        )

    def train_epochs(
        self,
        batch: dict[str, Any],
        control_state: Any | None = None,
        cycle: int = 0,
        epochs_override: int | None = None,
        log_prefix: str = "",
    ) -> dict[str, float]:
        train_cfg = self.config.get("training", {})
        epochs = int(epochs_override if epochs_override is not None else train_cfg.get("epochs_per_cycle", 100))
        log_every = max(1, int(train_cfg.get("log_every", 25)))
        weights = dict(train_cfg.get("weights", {}))
        local_weights = {}
        active_aux_losses: set[str] = set()
        pressure_anchor_patches: dict[int, float] = {}
        if control_state is not None:
            weights = control_state.global_weights
            local_weights = control_state.local_weights
            active_aux_losses = control_state.active_aux_losses
            pressure_anchor_patches = control_state.pressure_anchor_patches

        last_losses: dict[str, float] = {}
        self.model.train()
        self.compute_tracker.start()
        phase_start = time.perf_counter()
        for local_epoch in range(epochs):
            if not self.compute_tracker.can_start_objective(int(batch["xy_f"].shape[0])):
                break
            self.optimizer.zero_grad(set_to_none=True)
            total, losses, local_logs = self._training_objective(
                batch,
                weights,
                local_weights,
                active_aux_losses,
                pressure_anchor_patches,
            )
            total.backward()
            grad_norm = self._grad_norm()
            self.optimizer.step()
            self.compute_tracker.record_optimizer_step()

            last_losses = {k: float(v.detach().cpu()) for k, v in losses.items()}
            last_losses.update(local_logs)
            last_losses["total"] = float(total.detach().cpu())
            last_losses["grad_norm"] = grad_norm
            self.last_losses = dict(last_losses)
            if local_epoch % log_every == 0 or local_epoch == epochs - 1:
                self.loss_logger.log({"cycle": cycle, "phase": log_prefix or "main", "epoch": self.global_step, **last_losses})
            self.global_step += 1
        self.compute_tracker.add_phase_time("optimization", time.perf_counter() - phase_start)
        return last_losses

    def _training_objective(
        self,
        batch: dict[str, Any],
        weights: dict[str, float],
        local_weights: dict[str, dict[int, float]],
        active_aux_losses: set[str],
        pressure_anchor_patches: dict[int, float],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        """Build the train objective used by Adam and guarded repair stages."""
        self.compute_tracker.record_objective(batch)
        if self.mode == "gradient_enhanced_pinn":
            pointwise = gradient_enhanced_pointwise_losses(self.model, batch, self.benchmark, self.steady)
        else:
            pointwise = compute_pointwise_losses(self.model, batch, self.benchmark, self.steady)
        if "pressure_poisson" in active_aux_losses:
            pointwise["pressure_poisson"] = pressure_poisson_residual(
                self.model, batch["xy_f"], self.benchmark.nu
            ).pow(2)
        if "vorticity_transport" in active_aux_losses and not self.steady:
            pointwise["vorticity_transport"] = vorticity_transport_residual(
                self.model, batch["xy_f"], self.benchmark.nu, steady=False
            ).pow(2)
        reduction = str(self.config.get("training", {}).get("pointwise_reduction", "legacy_mse"))
        losses = compute_global_losses(pointwise, reduction=reduction)
        total = weighted_sum(losses, weights)
        local_loss, local_logs = compute_local_weighted_loss(
            pointwise,
            batch,
            self.patch_grid,
            local_weights,
            entropy_weight=float(self.config.get("controller", {}).get("entropy_weight", 0.0)),
            reduction=reduction,
        )
        total = total + local_loss
        if pressure_anchor_patches:
            patch_ids = self.patch_grid.assign_torch(batch["xy_f"])
            pred_f = self.model(batch["xy_f"])
            for pid, strength in pressure_anchor_patches.items():
                mask = patch_ids == int(pid)
                if torch.any(mask):
                    total = total + float(strength) * pressure_anchor_loss(pred_f[mask, 2:3], 0.0)
        anchor_loss, anchor_weight = self.continuation_anchor_loss(batch)
        replay_loss, replay_weight = self.continuation_replay_loss()
        gauge_loss = self.pressure_gauge_loss()
        total = total + anchor_loss + replay_loss + gauge_loss
        if anchor_weight > 0.0:
            local_logs["continuation_anchor"] = float(anchor_loss.detach().cpu())
            local_logs["continuation_anchor_weight"] = float(anchor_weight)
        if replay_weight > 0.0:
            local_logs["continuation_replay"] = float(replay_loss.detach().cpu())
            local_logs["continuation_replay_weight"] = float(replay_weight)
        if float(gauge_loss.detach().cpu()) > 0.0:
            local_logs["pressure_gauge"] = float(gauge_loss.detach().cpu())
        return total, losses, local_logs

    def continuation_anchor_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, float]:
        """Shared decaying warm-start anchor used identically by every method."""
        cfg = dict(self.config.get("continuation_anchor", {}))
        if self.continuation_anchor_model is None or not bool(cfg.get("enabled", False)):
            return next(self.model.parameters()).new_tensor(0.0), 0.0
        train_cfg = self.config.get("training", {})
        total_steps = int(
            cfg.get(
                "stage_steps",
                int(train_cfg.get("adaptive_cycles", 1)) * int(train_cfg.get("epochs_per_cycle", 1)),
            )
        )
        active_steps = max(1, int(round(float(cfg.get("active_fraction", 0.20)) * total_steps)))
        if self.global_step >= active_steps:
            return next(self.model.parameters()).new_tensor(0.0), 0.0
        initial_weight = float(cfg.get("initial_weight", 1.0))
        weight = initial_weight * max(0.0, 1.0 - self.global_step / active_steps)
        coords = batch["xy_f"]
        with torch.no_grad():
            target = self.continuation_anchor_model(coords)
        prediction = self.model(coords)
        fields = str(cfg.get("fields", "velocity")).lower()
        if fields == "all":
            difference = prediction - target
        else:
            difference = prediction[:, :2] - target[:, :2]
        return float(weight) * torch.mean(difference * difference), float(weight)

    def _initialize_continuation_replay(self) -> None:
        """Build a fixed, reference-free replay set from the previous-Re model."""
        cfg = dict(self.config.get("continuation_replay", {}))
        if self.continuation_anchor_model is None or not bool(cfg.get("enabled", False)):
            return
        n_points = max(0, int(cfg.get("n_points", 512)))
        if n_points == 0:
            return
        rng = np.random.default_rng(self.seed + int(cfg.get("seed_offset", 104729)))
        x0, x1, y0, y1 = self.benchmark.bounds
        columns = [
            rng.uniform(x0, x1, size=n_points),
            rng.uniform(y0, y1, size=n_points),
        ]
        t_bounds = getattr(self.benchmark, "t_bounds", None)
        if t_bounds is not None:
            columns.append(rng.uniform(float(t_bounds[0]), float(t_bounds[1]), size=n_points))
        points = np.column_stack(columns)
        self.continuation_replay_points = torch.tensor(
            points,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            targets = self.continuation_anchor_model(self.continuation_replay_points)
        self.continuation_replay_targets = targets.detach()

    def continuation_replay_loss(self) -> tuple[torch.Tensor, float]:
        """Decaying previous-model distillation shared identically by all methods."""
        cfg = dict(self.config.get("continuation_replay", {}))
        zero = next(self.model.parameters()).new_tensor(0.0)
        if (
            self.continuation_replay_points is None
            or self.continuation_replay_targets is None
            or not bool(cfg.get("enabled", False))
        ):
            return zero, 0.0
        total_steps = self._continuation_stage_steps(cfg)
        active_steps = max(1, int(round(float(cfg.get("active_fraction", 0.20)) * total_steps)))
        if self.global_step >= active_steps:
            return zero, 0.0
        initial_weight = float(cfg.get("initial_weight", 0.5))
        weight = initial_weight * max(0.0, 1.0 - self.global_step / active_steps)
        prediction = self.model(self.continuation_replay_points)
        fields = str(cfg.get("fields", "velocity")).lower()
        if fields == "all":
            difference = prediction - self.continuation_replay_targets
        else:
            difference = prediction[:, :2] - self.continuation_replay_targets[:, :2]
        return float(weight) * torch.mean(difference * difference), float(weight)

    def _continuation_stage_steps(self, cfg: dict[str, Any]) -> int:
        train_cfg = self.config.get("training", {})
        controller_cfg = self.config.get("controller_v2", {})
        default_steps = int(
            controller_cfg.get(
                "total_steps",
                int(train_cfg.get("adaptive_cycles", 1)) * int(train_cfg.get("epochs_per_cycle", 1)),
            )
        )
        return max(1, int(cfg.get("stage_steps", default_steps)))

    def pressure_gauge_loss(self) -> torch.Tensor:
        """Reference-free center pressure anchor shared by all compared methods."""
        cfg = dict(self.config.get("pressure_gauge", {}))
        weight = float(cfg.get("weight", 0.0))
        if weight <= 0.0:
            return next(self.model.parameters()).new_tensor(0.0)
        x0, x1, y0, y1 = self.benchmark.bounds
        coords = [0.5 * (x0 + x1), 0.5 * (y0 + y1)]
        if int(self.config.get("model", {}).get("input_dim", 2)) == 3:
            coords.append(float(cfg.get("time", self.config.get("model", {}).get("t_min", 0.0))))
        point = torch.tensor([coords], dtype=torch.float32, device=self.device)
        pressure = self.model(point)[:, 2:3]
        return weight * torch.mean(pressure * pressure)

    def run_final_physics_repair(self, cycle: int = -1, log_prefix: str = "final_repair") -> dict[str, Any]:
        """Run a guarded global-physics LBFGS repair stage.

        VARA can leave useful local interventions behind, but LBFGS is too sharp
        to safely optimize those local weights directly. This stage therefore
        builds a fresh global batch, ignores local controller weights, and keeps
        the result only when validation score improves.
        """
        cfg = self._final_repair_config()
        if self.compute_tracker.enabled and self.compute_tracker.exhausted():
            self.final_repair_status = {
                "enabled": bool(cfg.get("enabled", False)),
                "accepted": False,
                "reason": "compute_budget_exhausted",
            }
            return self.final_repair_status
        if not bool(cfg.get("enabled", False)):
            self.final_repair_status = {"enabled": False, "accepted": False}
            return self.final_repair_status

        steps = max(0, int(cfg.get("epochs", cfg.get("steps", 0))))
        if steps <= 0:
            self.final_repair_status = {"enabled": True, "accepted": False, "reason": "zero_steps"}
            return self.final_repair_status

        _, _, validation_coords = self.validation_grid()
        before_metrics = self.evaluate_metrics(validation_coords)
        score_name, before_score = self._repair_score(before_metrics)
        model_snapshot = self._model_snapshot()
        optimizer_snapshot = deepcopy(self.optimizer.state_dict())
        previous_optimizer = self.optimizer
        previous_stage = self.optimizer_stage

        repair_batch = self._make_final_repair_batch(cfg)
        repair_weights = self._repair_weights(cfg)
        lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=float(cfg.get("lr", 0.5)),
            max_iter=int(cfg.get("max_iter", 5)),
            max_eval=int(cfg.get("max_eval", int(cfg.get("max_iter", 5)) * 2)),
            tolerance_grad=float(cfg.get("tolerance_grad", 1e-7)),
            tolerance_change=float(cfg.get("tolerance_change", 1e-9)),
            history_size=int(cfg.get("history_size", 20)),
            line_search_fn=cfg.get("line_search_fn", "strong_wolfe"),
        )
        self.optimizer = lbfgs
        self.optimizer_stage = "final_repair_lbfgs"
        last_logs: dict[str, float] = {}
        log_every = max(1, int(cfg.get("log_every", self.config.get("training", {}).get("log_every", 25))))

        def loss_fn() -> torch.Tensor:
            nonlocal last_logs
            total, losses, local_logs = self._training_objective(
                repair_batch,
                repair_weights,
                {},
                set(),
                {},
            )
            last_logs = {k: float(v.detach().cpu()) for k, v in losses.items()}
            last_logs.update(local_logs)
            last_logs["total"] = float(total.detach().cpu())
            return total

        self.model.train()
        phase_start = time.perf_counter()
        for step in range(steps):
            if not self.compute_tracker.can_start_objective(int(repair_batch["xy_f"].shape[0])):
                break
            closure = make_lbfgs_closure(lbfgs, loss_fn)
            loss = lbfgs.step(closure)
            self.compute_tracker.record_optimizer_step()
            if "total" not in last_logs:
                last_logs["total"] = float(loss.detach().cpu()) if hasattr(loss, "detach") else float(loss)
            last_logs["grad_norm"] = self._grad_norm()
            last_logs["optimizer_stage"] = self.optimizer_stage
            self.last_losses = dict(last_logs)
            if step % log_every == 0 or step == steps - 1:
                self.loss_logger.log(
                    {
                        "cycle": cycle,
                        "phase": log_prefix,
                        "epoch": self.global_step,
                        "repair_step": step,
                        **last_logs,
                    }
                )
            self.global_step += 1
        self.compute_tracker.add_phase_time("optimization", time.perf_counter() - phase_start)

        after_metrics = self.evaluate_metrics(validation_coords)
        _, after_score = self._repair_score(after_metrics)
        tolerance = float(cfg.get("acceptance_tolerance", 0.0))
        accepted = bool(after_score <= before_score * (1.0 + tolerance))
        reason = "accepted" if accepted else "validation_score_worsened"
        if not accepted:
            self._restore_model_snapshot(model_snapshot)
            self.optimizer = previous_optimizer
            self.optimizer.load_state_dict(optimizer_snapshot)
            self.optimizer_stage = previous_stage

        self.final_repair_status = {
            "enabled": True,
            "accepted": accepted,
            "reason": reason,
            "score_name": score_name,
            "pre_repair_score": float(before_score),
            "post_repair_score": float(after_score),
            "epochs": steps,
            "batch_n_collocation": int(repair_batch["xy_f"].shape[0]),
            "batch_n_boundary": int(repair_batch["xy_bc"].shape[0]),
            "global_only": True,
        }
        self.metrics_logger.log({"cycle": cycle, "phase": log_prefix, **self.final_repair_status})
        return self.final_repair_status

    def _final_repair_config(self) -> dict[str, Any]:
        optim_cfg = self.config.get("optimizer", {})
        cfg = dict(optim_cfg.get("final_repair", {}))
        cfg.setdefault("enabled", False)
        cfg.setdefault("epochs", 0)
        cfg.setdefault("lr", 0.5)
        cfg.setdefault("max_iter", 5)
        cfg.setdefault("history_size", 20)
        cfg.setdefault("line_search_fn", "strong_wolfe")
        cfg.setdefault("batch_multiplier", 2.0)
        cfg.setdefault("residual_fraction", 0.25)
        return cfg

    def _repair_weights(self, cfg: dict[str, Any]) -> dict[str, float]:
        weights = dict(self.config.get("training", {}).get("weights", {}))
        weights.update({str(k): float(v) for k, v in dict(cfg.get("weights", {})).items()})
        return weights

    def _make_final_repair_batch(self, cfg: dict[str, Any]) -> dict[str, Any]:
        train_cfg = self.config.get("training", {})
        multiplier = max(1.0, float(cfg.get("batch_multiplier", 2.0)))
        n_f = max(1, int(round(int(train_cfg.get("n_collocation", 1024)) * multiplier)))
        n_bc = max(1, int(round(int(train_cfg.get("n_boundary", 256)) * multiplier)))
        n_data = max(0, int(round(int(train_cfg.get("n_data", 0)) * multiplier)))
        residual_fraction = float(np.clip(float(cfg.get("residual_fraction", 0.25)), 0.0, 0.9))
        n_residual = int(round(n_f * residual_fraction))
        n_uniform = n_f - n_residual
        pieces = [self.uniform_sampler.sample(n_uniform).detach().cpu().numpy()]
        if n_residual > 0:
            _, _, coords = self.validation_grid()
            builder = self.diagnostic_builder()
            maps = builder.build(coords, mode=self.controller_diagnostic_mode())
            score = maps.get("aggregate_pde_residual", maps.get("pde_residual"))
            if score is not None:
                pieces.append(sample_from_score_grid(coords, score, n_residual, self.repair_rng))
            else:
                pieces.append(self.uniform_sampler.sample(n_residual).detach().cpu().numpy())
        xy_f_np = np.vstack([p for p in pieces if p.size])
        self.repair_rng.shuffle(xy_f_np)
        xy_f = torch.tensor(xy_f_np, dtype=torch.float32, device=self.device)
        xy_bc = self._sample_boundary(n_bc)
        xy_data = self._sample_data(n_data)
        return self.make_batch(xy_f, xy_bc, xy_data)

    def _repair_score(self, metrics: dict[str, Any]) -> tuple[str, float]:
        preferred = self._final_repair_config().get("score_metric")
        candidates = [
            str(preferred) if preferred else "",
            "cavity_benchmark_score",
            "unweighted_validation_loss",
            "pde_residual_mean",
        ]
        for name in candidates:
            if not name:
                continue
            value = metrics.get(name)
            if value is None:
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                return name, numeric
        fallback = 0.0
        used = []
        for name in ["pde_residual_mean", "continuity_residual_mean", "momentum_residual_mean", "boundary_condition_error"]:
            value = metrics.get(name)
            if value is not None and math.isfinite(float(value)):
                fallback += float(value)
                used.append(name)
        return "+".join(used) if used else "zero", float(fallback)

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += float(torch.sum(p.grad.detach() ** 2).cpu())
        return float(math.sqrt(total))

    def _model_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}

    def _restore_model_snapshot(self, snapshot: dict[str, torch.Tensor]) -> None:
        device_snapshot = {name: value.to(self.device) for name, value in snapshot.items()}
        self.model.load_state_dict(device_snapshot)

    def diagnose(self) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], list[Any], np.ndarray, np.ndarray, np.ndarray]:
        phase_start = time.perf_counter()
        X, Y, coords = self.validation_grid()
        builder = self.diagnostic_builder()
        maps = builder.build(coords, mode=self.controller_diagnostic_mode())
        scores, names = self.patch_scorer.compute(maps, coords, update_ema=True)
        weak_regions = self.weak_detector.detect(scores, names, self.patch_grid)
        self.compute_tracker.add_phase_time("diagnostics", time.perf_counter() - phase_start)
        return maps, scores, names, weak_regions, X, Y, coords

    def resample_batch(
        self,
        batch: dict[str, Any],
        maps: dict[str, np.ndarray],
        coords: np.ndarray,
        weak_regions: list[Any],
        control_state: Any | None = None,
        adaptive: bool = True,
    ) -> dict[str, Any]:
        train_cfg = self.config.get("training", {})
        n_f = int(train_cfg.get("n_collocation", batch["xy_f"].shape[0]))
        n_bc = int(train_cfg.get("n_boundary", batch["xy_bc"].shape[0]))
        n_data = int(train_cfg.get("n_data", batch["xy_data"].shape[0]))
        if adaptive:
            priorities = control_state.sampling_priorities if control_state is not None else {}
            xy_f = self.adaptive_sampler.sample_interior(n_f, maps, coords, weak_regions, priorities)
            if getattr(self.benchmark, "t_bounds", None) is not None:
                xy_data = self._sample_data(n_data)
            else:
                xy_data = self.adaptive_sampler.sample_interior(n_data, maps, coords, weak_regions, priorities)
        else:
            xy_f = self.uniform_sampler.sample(n_f)
            xy_data = self._sample_data(n_data)
        xy_bc = self._sample_boundary(n_bc)
        return self.make_batch(xy_f, xy_bc, xy_data)

    def evaluate_and_save_final(self) -> dict[str, float]:
        X, Y, coords = self.test_grid()
        phase_start = time.perf_counter()
        metrics = self.evaluate_metrics(coords)
        self.compute_tracker.add_phase_time("evaluation", time.perf_counter() - phase_start)
        metrics["final_total_loss"] = float(self.last_losses.get("total", float("nan")))
        metrics["optimizer_stage"] = self.optimizer_stage
        for key, value in self.final_repair_status.items():
            metrics[f"final_repair_{key}"] = value
        for key, value in self.warm_start_status.items():
            metrics[f"warm_start_{key}"] = value
        metrics["reference_kind"] = getattr(self.benchmark, "reference_kind", "analytical")
        metrics["has_reference"] = bool(getattr(self.benchmark, "has_reference", True))
        metrics["run_type"] = str(self.config.get("run_type", "full"))
        model_cfg = self.config.get("model", {})
        metrics["model_architecture"] = str(model_cfg.get("architecture", "mlp"))
        metrics["physics_formulation"] = str(model_cfg.get("physics_formulation", "direct"))
        metrics["hard_boundary_corner_width"] = (
            float(model_cfg.get("hard_boundary_corner_width", 0.02))
            if metrics["physics_formulation"] == "cavity_hard_boundary"
            else float("nan")
        )
        metrics["continuation_replay_enabled"] = bool(
            self.config.get("continuation_replay", {}).get("enabled", False)
        )
        metrics["controller_reference_metrics_enabled"] = bool(
            self.config.get("evaluation", {}).get(
                "controller_reference_metrics_enabled",
                True,
            )
        )
        metrics["continuation_replay_points"] = int(
            0 if self.continuation_replay_points is None else self.continuation_replay_points.shape[0]
        )
        metrics["reportable"] = metrics["run_type"] != "smoke"
        metrics["collapse_evaluated"] = bool(metrics["reportable"])
        metrics["collapsed"] = self._collapsed(metrics)
        metrics.update(self.compute_tracker.summary())
        self.metrics_logger.log({"cycle": "final_test", **metrics})
        save_json(metrics, self.run_dir / "summary.json")
        pd.DataFrame([metrics]).to_csv(self.table_dir / "summary.csv", index=False)
        pd.DataFrame([metrics]).to_csv(self.run_dir / "summary_table.csv", index=False)
        self.save_plots(X, Y, coords)
        save_checkpoint(
            self.checkpoint_dir / "final.pt",
            self.model,
            self.optimizer,
            self.config,
            metrics,
            self.global_step,
            -1,
        )
        save_intervention_timeline(self.action_records, self.figure_dir / "intervention_timeline.png")
        return metrics

    def save_plots(self, X: np.ndarray, Y: np.ndarray, coords: np.ndarray) -> None:
        builder = self.diagnostic_builder()
        diag_mode = "full_reference" if getattr(self.benchmark, "has_reference", True) else "residual_only"
        maps = builder.build(coords, mode=diag_mode)
        shape = X.shape
        save_field_panel(
            X,
            Y,
            {
                "u pred": maps["u_pred"].reshape(shape),
                "v pred": maps["v_pred"].reshape(shape),
                "p pred centered": maps["p_pred"].reshape(shape),
                "omega pred": maps["omega_pred"].reshape(shape),
            },
            self.figure_dir / "predicted_fields.png",
        )
        if getattr(self.benchmark, "has_reference", True):
            reference_fields = {
                "u ref": maps["u_ref"].reshape(shape),
                "v ref": maps["v_ref"].reshape(shape),
                "p ref centered": maps["p_ref"].reshape(shape),
                "omega ref": maps["omega_ref"].reshape(shape),
            }
            save_field_panel(
                X,
                Y,
                reference_fields,
                self.figure_dir / "reference_fields.png",
            )
            save_field_panel(X, Y, reference_fields, self.figure_dir / "cfd_reference_fields.png")
            error_fields = {
                "u error": maps["u_error"].reshape(shape),
                "v error": maps["v_error"].reshape(shape),
                "p error centered": maps["p_error_mean_centered"].reshape(shape),
                "omega error": maps["omega_error"].reshape(shape),
            }
            save_field_panel(X, Y, error_fields, self.figure_dir / "error_fields.png", cmap="magma")
            save_field_panel(X, Y, error_fields, self.figure_dir / "cfd_prediction_error_fields.png", cmap="magma")
            prediction_reference_error_fields = {
                "u": (
                    maps["u_pred"].reshape(shape),
                    maps["u_ref"].reshape(shape),
                    maps["u_error"].reshape(shape),
                ),
                "v": (
                    maps["v_pred"].reshape(shape),
                    maps["v_ref"].reshape(shape),
                    maps["v_error"].reshape(shape),
                ),
                "p": (
                    maps["p_pred"].reshape(shape),
                    maps["p_ref"].reshape(shape),
                    maps["p_error_mean_centered"].reshape(shape),
                ),
                "omega": (
                    maps["omega_pred"].reshape(shape),
                    maps["omega_ref"].reshape(shape),
                    maps["omega_error"].reshape(shape),
                ),
            }
            save_prediction_reference_error_panel(
                X,
                Y,
                prediction_reference_error_fields,
                self.figure_dir / "prediction_reference_error.png",
            )
            save_prediction_reference_error_panel(
                X,
                Y,
                prediction_reference_error_fields,
                self.figure_dir / "cfd_prediction_reference_error.png",
            )
        heatmap_names = [
            "u_error",
            "v_error",
            "p_error_mean_centered",
            "omega_error",
            "pde_residual",
            "continuity_residual",
            "momentum_u_residual",
            "momentum_v_residual",
        ]
        for name in heatmap_names:
            if name in maps:
                save_heatmap(maps[name].reshape(shape), X, Y, self.figure_dir / f"{name}.png", name)
        if "momentum_u_residual" in maps and "momentum_v_residual" in maps:
            momentum = np.sqrt(maps["momentum_u_residual"] ** 2 + maps["momentum_v_residual"] ** 2)
            save_heatmap(momentum.reshape(shape), X, Y, self.figure_dir / "momentum_residual.png", "momentum_residual")
        save_streamlines(
            X,
            Y,
            maps["u_pred"].reshape(shape),
            maps["v_pred"].reshape(shape),
            self.figure_dir / "streamlines.png",
        )
        save_streamfunction_contours(
            X,
            Y,
            maps["u_pred"].reshape(shape),
            maps["v_pred"].reshape(shape),
            self.figure_dir / "streamfunction_contours.png",
        )

    def maybe_checkpoint(self, cycle: int, metrics: dict[str, float]) -> None:
        score = self._checkpoint_score(metrics)
        save_checkpoint(self.checkpoint_dir / "latest.pt", self.model, self.optimizer, self.config, metrics, self.global_step, cycle)
        if score < self.best_score:
            self.best_score = score
            save_checkpoint(self.checkpoint_dir / "best.pt", self.model, self.optimizer, self.config, metrics, self.global_step, cycle)

    def _checkpoint_score(self, metrics: dict[str, Any]) -> float:
        evaluation_cfg = self.config.get("evaluation", {})
        reference_enabled = bool(
            evaluation_cfg.get(
                "checkpoint_reference_metrics_enabled",
                evaluation_cfg.get("controller_reference_metrics_enabled", True),
            )
        )
        if reference_enabled:
            # Preserve the historical checkpoint selector unless an opt-in
            # reference-free study explicitly disables it.
            return float(
                metrics.get("u_rel_l2", 0.0)
                + metrics.get("v_rel_l2", 0.0)
                + metrics.get("p_rel_l2_centered", 0.0)
                + metrics.get("omega_rel_l2", 0.0)
            )
        pde = float(metrics.get("unweighted_pde_loss", float("nan")))
        boundary = float(metrics.get("unweighted_bc_loss", float("nan")))
        if math.isfinite(pde) and math.isfinite(boundary):
            return pde + boundary
        fallback = float(metrics.get("pde_residual_mean", float("inf")))
        return fallback if math.isfinite(fallback) else math.inf

    def _collapsed(self, metrics: dict[str, Any]) -> bool:
        if not bool(metrics.get("collapse_evaluated", True)):
            return False
        thresholds = self.config.get("collapse_thresholds", {})
        has_reference = bool(metrics.get("has_reference", True))
        relative_names = ["u_rel_l2", "v_rel_l2", "p_rel_l2_centered", "omega_rel_l2"]
        for name in relative_names:
            value = metrics.get(name)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(numeric):
                continue
            if not math.isfinite(numeric):
                return True
            if numeric > float(thresholds.get(name, 5.0)):
                return True
        if has_reference:
            for name in ["u_rmse", "v_rmse", "p_rmse_centered", "omega_rmse"]:
                value = metrics.get(name)
                if value is None:
                    continue
                numeric = float(value)
                if not math.isfinite(numeric):
                    return True
                if numeric > float(thresholds.get(name, 5.0)):
                    return True
        for name in [
            "pde_residual_mean",
            "continuity_residual_mean",
            "momentum_residual_mean",
            "unweighted_validation_loss",
        ]:
            value = metrics.get(name)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                return True
            if numeric > float(thresholds.get(name, 10.0)):
                return True
        bc = metrics.get("boundary_condition_error")
        if bc is not None and math.isfinite(float(bc)) and float(bc) > float(thresholds.get("boundary_condition_error", 1.0)):
            return True
        return False
