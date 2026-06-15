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
from src.data import build_cavity_cfd_supervision
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
from src.sampling.boundary_sampler import boundary_side_fractions
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
from src.visualization.streamlines import (
    save_streamfunction_contours,
    save_streamlines,
)


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
        self.base_learning_rate = float(optim_cfg.get("lr", 1e-3))
        self.max_gradient_norm = float(optim_cfg.get("max_grad_norm", 0.0))
        self.learning_rate_schedule = dict(optim_cfg.get("scheduler", {}))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.base_learning_rate,
        )
        self.optimizer_stage = "adam"
        self.final_repair_status: dict[str, Any] = {}
        self.current_cavity_curriculum: dict[str, float] = {}
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
        self.early_stop_history: list[float] = []
        self.early_stopped = False
        self.early_stop_step: int | None = None
        self.compute_tracker = ComputeTracker(dict(config.get("compute_budget", {})))
        self.loss_normalization_cfg = dict(config.get("loss_normalization", {}))
        self.loss_normalization_state: dict[str, float] = {}

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
            self.metrics_dir = root / "metrics"
        else:
            self.run_dir = ensure_dir(root / "logs" / self.run_id)
            self.checkpoint_dir = ensure_dir(root / "checkpoints" / self.run_id)
            self.figure_dir = ensure_dir(root / "figures" / self.run_id)
            self.table_dir = ensure_dir(root / "tables" / self.run_id)
            self.metrics_dir = root / "metrics" / self.run_id
        save_config(config, self.run_dir / "config_snapshot.yaml")

        self.metrics_logger = CSVLogger(self.run_dir / "metrics.csv")
        self.loss_logger = CSVLogger(self.run_dir / "losses.csv")
        self.runtime_logger = CSVLogger(self.run_dir / "runtime_profile.csv")
        self.action_logger = JSONListLogger(self.run_dir / "action_log.json")
        self.weak_logger = JSONListLogger(self.run_dir / "weak_region_log.json")
        self.score_logger = JSONListLogger(self.run_dir / "patch_scores.json")
        self.accept_logger = JSONListLogger(self.run_dir / "acceptance_log.json")
        self.action_records: list[dict[str, Any]] = []
        self.last_losses: dict[str, float] = {}
        self.effective_loss_keys: list[str] = []
        self.effective_loss_weights: dict[str, float] = {}
        self.last_boundary_sampling_summary: dict[str, float] = {}
        self.last_interior_sampling_summary: dict[str, float] = {}
        self._print_runtime_environment()

        t_bounds = getattr(self.benchmark, "t_bounds", None)
        sampler_cfg = config.get("sampling", {})
        uniform_engine = str(
            sampler_cfg.get("uniform", {}).get("engine", "random")
        )
        self.uniform_sampler = UniformSampler(
            self.benchmark.bounds,
            self.device,
            self.seed,
            t_bounds=t_bounds,
            engine=uniform_engine,
        )
        self.boundary_sampler = BoundarySampler(self.benchmark.bounds, self.device, self.seed + 1, t_bounds=t_bounds)
        self.adaptive_sampler = MixedAdaptiveSampler(
            self.benchmark.bounds,
            self.patch_grid,
            self.device,
            self.seed + 2,
            mixture=sampler_cfg.get("mixture"),
            uniform_engine=uniform_engine,
        )
        self.cfd_supervision = build_cavity_cfd_supervision(
            config,
            self.benchmark.bounds,
            self.device,
        )
        self._initialize_continuation_replay()

    def sampling_state_snapshot(self) -> dict[str, Any]:
        return {
            "uniform": self.uniform_sampler.snapshot(),
            "boundary_rng": deepcopy(self.boundary_sampler.rng.bit_generator.state),
            "adaptive_rng": deepcopy(self.adaptive_sampler.rng.bit_generator.state),
            "adaptive_uniform": self.adaptive_sampler.uniform.snapshot(),
            "region_rng": deepcopy(
                self.adaptive_sampler.region_sampler.rng.bit_generator.state
            ),
        }

    def restore_sampling_state(self, snapshot: dict[str, Any]) -> None:
        self.uniform_sampler.restore(snapshot["uniform"])
        self.boundary_sampler.rng.bit_generator.state = deepcopy(
            snapshot["boundary_rng"]
        )
        self.adaptive_sampler.rng.bit_generator.state = deepcopy(
            snapshot["adaptive_rng"]
        )
        self.adaptive_sampler.uniform.restore(snapshot["adaptive_uniform"])
        self.adaptive_sampler.region_sampler.rng.bit_generator.state = deepcopy(
            snapshot["region_rng"]
        )

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
            self._sync_benchmark_corner_to_model()
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
        if name in {"lid_driven_cavity", "lid_cavity", "lid-driven-cavity", "lid-cavity", "cavity"}:
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
        n_f, n_bc, n_data = self._training_sample_counts()
        started = time.perf_counter()
        xy_f = self._sample_interior(n_f)
        xy_bc = self._sample_boundary(n_bc)
        xy_data = self._sample_data(n_data)
        self._record_runtime("sampling", time.perf_counter() - started, phase="initial")
        return self.make_batch(xy_f, xy_bc, xy_data)

    def _training_sample_counts(self) -> tuple[int, int, int]:
        train_cfg = self.config.get("training", {})
        n_f = int(train_cfg.get("n_collocation", 1024))
        n_bc = int(train_cfg.get("n_boundary", 256))
        n_data = int(train_cfg.get("n_data", 256))
        curriculum = dict(train_cfg.get("collocation_curriculum", {}))
        if bool(curriculum.get("enabled", False)):
            stages = list(curriculum.get("stages", []))
            selected = stages[-1] if stages else {}
            for stage in stages:
                if self.global_step < int(stage.get("until_step", 0)):
                    selected = stage
                    break
            n_f = int(selected.get("n_collocation", n_f))
            n_bc = int(selected.get("n_boundary", n_bc))
        return n_f, n_bc, n_data

    def _near_wall_sample_count(self, n: int) -> int:
        cfg = dict(self.config.get("sampling", {}).get("cavity_near_wall", {}))
        if not self._is_lid_driven_cavity() or not bool(cfg.get("enabled", False)):
            return 0
        fraction = min(max(float(cfg.get("fraction", 0.0)), 0.0), 1.0)
        return min(max(int(round(int(n) * fraction)), 0), int(n))

    def _lid_interior_sample_count(self, n: int) -> int:
        cfg = dict(
            self.config.get("sampling", {}).get("cavity_lid_interior_band", {})
        )
        formulation = str(
            self.config.get("model", {}).get("physics_formulation", "")
        )
        if (
            formulation != "cavity_uvp_soft_bc"
            or not self._is_lid_driven_cavity()
            or not bool(cfg.get("enabled", False))
        ):
            return 0
        fraction = min(max(float(cfg.get("fraction", 0.0)), 0.0), 1.0)
        return min(max(int(round(int(n) * fraction)), 0), int(n))

    def _interior_component_counts(self, n: int) -> tuple[int, int, int]:
        n = max(int(n), 0)
        n_wall = self._near_wall_sample_count(n)
        n_lid = self._lid_interior_sample_count(n)
        cfg = dict(
            self.config.get("sampling", {}).get("cavity_lid_interior_band", {})
        )
        minimum_core = int(
            math.ceil(
                n
                * min(
                    max(float(cfg.get("minimum_core_fraction", 0.35)), 0.0),
                    1.0,
                )
            )
        )
        specialized_limit = max(n - minimum_core, 0)
        if n_wall + n_lid > specialized_limit:
            n_lid = max(specialized_limit - n_wall, 0)
        n_core = n - n_wall - n_lid
        return n_core, n_wall, n_lid

    def _circulation_band_counts(self, n: int) -> dict[str, int] | None:
        cfg = dict(
            self.config.get("sampling", {}).get(
                "cavity_circulation_bands", {}
            )
        )
        formulation = str(
            self.config.get("model", {}).get("physics_formulation", "")
        )
        if (
            formulation != "cavity_uvp_velocity_lift"
            or not self._is_lid_driven_cavity()
            or not bool(cfg.get("enabled", False))
        ):
            return None
        n = max(int(n), 0)
        counts = {
            name: int(round(n * float(cfg.get(f"{name}_fraction", default))))
            for name, default in (
                ("top_band", 0.20),
                ("right_band", 0.15),
                ("lower_band", 0.10),
                ("left_band", 0.10),
            )
        }
        specialized = sum(counts.values())
        if specialized > n:
            overflow = specialized - n
            for name in ("left_band", "lower_band", "right_band", "top_band"):
                reduction = min(counts[name], overflow)
                counts[name] -= reduction
                overflow -= reduction
                if overflow == 0:
                    break
        counts["uniform"] = n - sum(counts.values())
        return counts

    def _sample_normalized_box_numpy(
        self,
        n: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> np.ndarray:
        points = self.uniform_sampler.sample_numpy(n)
        if n <= 0:
            return points
        x0, x1, y0, y1 = self.benchmark.bounds
        xi = (points[:, 0] - x0) / max(x1 - x0, 1e-12)
        eta = (points[:, 1] - y0) / max(y1 - y0, 1e-12)
        points[:, 0] = x0 + (
            x_range[0] + (x_range[1] - x_range[0]) * xi
        ) * (x1 - x0)
        points[:, 1] = y0 + (
            y_range[0] + (y_range[1] - y_range[0]) * eta
        ) * (y1 - y0)
        return points

    def _sample_near_wall_numpy(self, n: int) -> np.ndarray:
        if n <= 0:
            return self.uniform_sampler.sample_numpy(0)
        cfg = dict(self.config.get("sampling", {}).get("cavity_near_wall", {}))
        band = min(max(float(cfg.get("band_width", 0.08)), 1e-6), 0.49)
        unit = self.uniform_sampler.sample_numpy(n)
        x0, x1, y0, y1 = self.benchmark.bounds
        u = (unit[:, 0] - x0) / max(x1 - x0, 1e-12)
        v = (unit[:, 1] - y0) / max(y1 - y0, 1e-12)
        normal_floor = min(max(0.02 * band, 1e-6), 0.5 * band)
        normal = normal_floor + u * (band - normal_floor)
        tangent = band + v * (1.0 - 2.0 * band)
        side = np.arange(n) % 4
        xi = np.where(side == 0, normal, np.where(side == 1, 1.0 - normal, tangent))
        eta = np.where(side == 2, normal, np.where(side == 3, 1.0 - normal, tangent))
        points = unit.copy()
        points[:, 0] = x0 + xi * (x1 - x0)
        points[:, 1] = y0 + eta * (y1 - y0)
        return points

    def _sample_lid_interior_numpy(self, n: int) -> np.ndarray:
        if n <= 0:
            return self.uniform_sampler.sample_numpy(0)
        cfg = dict(
            self.config.get("sampling", {}).get("cavity_lid_interior_band", {})
        )
        x0, x1, y0, y1 = self.benchmark.bounds
        corner_width = min(
            max(float(cfg.get("corner_width", 0.05)), 0.0),
            0.49,
        )
        y_min = min(max(float(cfg.get("y_min", 0.72)), 0.0), 1.0)
        y_max = min(max(float(cfg.get("y_max", 0.98)), y_min), 1.0)
        unit = self.uniform_sampler.sample_numpy(n)
        xi = (unit[:, 0] - x0) / max(x1 - x0, 1e-12)
        eta = (unit[:, 1] - y0) / max(y1 - y0, 1e-12)
        points = unit.copy()
        points[:, 0] = x0 + (
            corner_width + (1.0 - 2.0 * corner_width) * xi
        ) * (x1 - x0)
        points[:, 1] = y0 + (y_min + (y_max - y_min) * eta) * (y1 - y0)
        return points

    def _sample_interior_numpy(
        self,
        n: int,
        core_points: np.ndarray | None = None,
    ) -> np.ndarray:
        circulation = self._circulation_band_counts(n)
        if circulation is not None:
            n_uniform = circulation["uniform"]
            if core_points is None:
                core_points = self.uniform_sampler.sample_numpy(n_uniform)
            if int(core_points.shape[0]) != n_uniform:
                raise ValueError(
                    f"Expected {n_uniform} uniform collocation points, "
                    f"got {core_points.shape[0]}."
                )
            pieces = [
                core_points,
                self._sample_normalized_box_numpy(
                    circulation["top_band"], (0.05, 0.95), (0.70, 0.98)
                ),
                self._sample_normalized_box_numpy(
                    circulation["right_band"], (0.78, 0.98), (0.15, 0.85)
                ),
                self._sample_normalized_box_numpy(
                    circulation["lower_band"], (0.15, 0.85), (0.05, 0.35)
                ),
                self._sample_normalized_box_numpy(
                    circulation["left_band"], (0.02, 0.22), (0.20, 0.85)
                ),
            ]
            result = np.vstack([points for points in pieces if points.size])
            total = max(int(n), 1)
            self.last_interior_sampling_summary = {
                f"interior_{name}_fraction": float(count / total)
                for name, count in circulation.items()
            }
            return result
        n_core, n_wall, n_lid = self._interior_component_counts(n)
        if core_points is None:
            core_points = self.uniform_sampler.sample_numpy(n_core)
        if int(core_points.shape[0]) != n_core:
            raise ValueError(
                f"Expected {n_core} core collocation points, got {core_points.shape[0]}."
            )
        wall_points = self._sample_near_wall_numpy(n_wall)
        lid_points = self._sample_lid_interior_numpy(n_lid)
        pieces = [points for points in (core_points, wall_points, lid_points) if points.size]
        result = np.vstack(pieces) if pieces else self.uniform_sampler.sample_numpy(0)
        total = max(int(n), 1)
        self.last_interior_sampling_summary = {
            "interior_core_fraction": float(n_core / total),
            "interior_near_wall_fraction": float(n_wall / total),
            "interior_lid_band_fraction": float(n_lid / total),
        }
        return result

    def _sample_interior(
        self,
        n: int,
        core_points: torch.Tensor | None = None,
    ) -> torch.Tensor:
        core_numpy = None if core_points is None else core_points.detach().cpu().numpy()
        points = self._sample_interior_numpy(n, core_numpy)
        return torch.tensor(points, dtype=torch.float32, device=self.device)

    def _sample_data(self, n: int) -> torch.Tensor:
        """Sample reference data; time-dependent benchmarks use initial data only."""
        if self.cfd_supervision is not None:
            return self.cfd_supervision.coords
        points = self.uniform_sampler.sample(n)
        t_bounds = getattr(self.benchmark, "t_bounds", None)
        if t_bounds is not None and n > 0 and points.shape[1] >= 3:
            points[:, 2] = float(t_bounds[0])
        return points

    def _sample_boundary_numpy(self, n: int) -> np.ndarray:
        boundary_cfg = self.config.get("sampling", {}).get("cavity_boundary", {})
        if self._is_lid_driven_cavity() and bool(boundary_cfg.get("enabled", True)):
            points = self.boundary_sampler.sample_lid_cavity_numpy(
                n,
                lid_fraction=float(boundary_cfg.get("lid_fraction", 0.45)),
                corner_fraction=float(boundary_cfg.get("corner_fraction", 0.25)),
                corner_width=float(boundary_cfg.get("corner_width", 0.12)),
                mode=str(boundary_cfg.get("mode", "focused")),
            )
        else:
            points = self.boundary_sampler.sample_numpy(n)
        self.last_boundary_sampling_summary = boundary_side_fractions(
            points,
            self.benchmark.bounds,
        )
        return points

    def _sample_boundary(self, n: int) -> torch.Tensor:
        return torch.tensor(self._sample_boundary_numpy(n), dtype=torch.float32, device=self.device)

    def _is_lid_driven_cavity(self) -> bool:
        return str(self.config.get("benchmark", "")).lower() in {
            "lid_driven_cavity",
            "lid_cavity",
            "lid-driven-cavity",
            "lid-cavity",
            "cavity",
        }

    def make_batch(self, xy_f: torch.Tensor, xy_bc: torch.Tensor, xy_data: torch.Tensor) -> dict[str, Any]:
        if self.cfd_supervision is not None:
            return {
                "xy_f": xy_f,
                "xy_bc": xy_bc,
                "xy_data": self.cfd_supervision.coords,
                "targets_data": self.cfd_supervision.targets,
            }
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
        profile: dict[str, float] = {}
        metrics = evaluate_on_grid(
            self.model,
            self.benchmark,
            coords,
            self.device,
            self.steady,
            residual_interior_only=self.residual_interior_only(),
            runtime_profile=profile,
        )
        if self.cfd_supervision is not None:
            metrics.update(self._cfd_sparse_metrics())
        self._record_runtime(
            "evaluation_detail",
            0.0,
            phase="final",
            profile=profile,
        )
        return metrics

    def controller_metrics(self, coords: np.ndarray) -> dict[str, float]:
        started = time.perf_counter()
        enabled = bool(
            self.config.get("evaluation", {}).get(
                "controller_reference_metrics_enabled",
                True,
            )
        )
        metrics = (
            self.evaluate_metrics(coords)
            if enabled
            else evaluate_on_grid(
                self.model,
                self.benchmark,
                coords,
                self.device,
                self.steady,
                residual_interior_only=self.residual_interior_only(),
                include_reference_metrics=False,
                include_streamfunction_metrics=bool(
                    self.config.get("evaluation", {}).get(
                        "controller_streamfunction_metrics",
                        False,
                    )
                ),
            )
        )
        if enabled:
            self._record_runtime(
                "validation",
                time.perf_counter() - started,
                phase="controller_metrics",
            )
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
            "unweighted_reference_evaluation_loss",
            "u_centerline_rmse",
            "v_centerline_rmse",
            "u_centerline_rel_l2",
            "v_centerline_rel_l2",
            "centerline_profile_score",
            "lid_cavity_expected_primary_x",
            "lid_cavity_expected_primary_y",
            "lid_cavity_primary_center_error",
            "lid_cavity_topology_score",
            "lid_cavity_topology_aligned",
        )
        for name in reference_names:
            if name in metrics:
                metrics[name] = float("nan")
        metrics["unweighted_physics_validation_loss"] = float(
            metrics["unweighted_pde_loss"] + metrics["unweighted_bc_loss"]
        )
        metrics["unweighted_reference_evaluation_loss"] = float("nan")
        metrics["unweighted_validation_loss"] = metrics[
            "unweighted_physics_validation_loss"
        ]
        metrics["cavity_benchmark_score"] = float(
            metrics["pde_residual_mean"]
            + metrics["continuity_residual_mean"]
            + metrics["momentum_residual_mean"]
            + metrics["boundary_condition_error"]
        )
        if self.cfd_supervision is not None:
            metrics.update(self._cfd_sparse_metrics())
        metrics["controller_reference_metrics_enabled"] = False
        self._record_runtime(
            "validation",
            time.perf_counter() - started,
            phase="controller_metrics",
        )
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
            self._apply_cavity_curriculum()
            self.optimizer.zero_grad(set_to_none=True)
            profile = self._step_runtime_profile()
            objective_start = time.perf_counter()
            total, losses, local_logs = self._training_objective(
                batch,
                weights,
                local_weights,
                active_aux_losses,
                pressure_anchor_patches,
                runtime_profile=profile,
            )
            self._record_runtime(
                "loss_objective_total",
                time.perf_counter() - objective_start,
                cycle=cycle,
                phase=log_prefix or "main",
                profile=profile,
            )
            optimizer_start = time.perf_counter()
            total.backward()
            grad_norm = self._grad_norm()
            learning_rate = self.prepare_optimizer_step()
            self.optimizer.step()
            self.compute_tracker.record_optimizer_step()
            self._record_runtime(
                "backward_optimizer",
                time.perf_counter() - optimizer_start,
                cycle=cycle,
                phase=log_prefix or "main",
            )

            last_losses = {k: float(v.detach().cpu()) for k, v in losses.items()}
            last_losses.update(local_logs)
            last_losses["total"] = float(total.detach().cpu())
            last_losses["grad_norm"] = grad_norm
            last_losses["learning_rate"] = learning_rate
            last_losses.update(self.current_cavity_curriculum)
            last_losses.update(self.last_boundary_sampling_summary)
            last_losses.update(self.last_interior_sampling_summary)
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
        runtime_profile: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        """Build the train objective used by Adam and guarded repair stages."""
        self.compute_tracker.record_objective(batch)
        pointwise, construction_logs = self._build_pointwise_losses(
            batch,
            weights,
            runtime_profile=runtime_profile,
        )
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
        losses, normalization_logs = self.normalize_training_losses(losses)
        self._record_effective_loss_diagnostics(losses, weights)
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
        local_logs.update(construction_logs)
        local_logs.update(self._model_auxiliary_logs())
        local_logs.update(normalization_logs)
        return total, losses, local_logs

    def _build_pointwise_losses(
        self,
        batch: dict[str, Any],
        weights: dict[str, float],
        *,
        runtime_profile: dict[str, float] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | str]]:
        """Construct the active pointwise dictionary before method weighting."""
        residual_mode, residual_delta = self._residual_loss_settings()
        loss_config = self._active_loss_config()
        loss_kwargs = {
            "residual_loss_mode": residual_mode,
            "pseudo_huber_delta": residual_delta,
            "regularization_config": loss_config,
            "compute_boundary_loss": self._compute_boundary_training_loss(
                weights
            ),
            "runtime_profile": runtime_profile,
        }
        if self.mode == "gradient_enhanced_pinn":
            pointwise = gradient_enhanced_pointwise_losses(
                self.model,
                batch,
                self.benchmark,
                self.steady,
                **loss_kwargs,
            )
            return pointwise, {}
        pointwise = compute_pointwise_losses(
            self.model,
            batch,
            self.benchmark,
            self.steady,
            **loss_kwargs,
        )
        return pointwise, {
            "residual_loss_mode": residual_mode,
            "pseudo_huber_delta": float(residual_delta),
        }

    def _record_effective_loss_diagnostics(
        self,
        losses: dict[str, torch.Tensor],
        weights: dict[str, float],
    ) -> None:
        self.effective_loss_keys = sorted(str(name) for name in losses)
        self.effective_loss_weights = {
            name: float(weights.get(name, 0.0))
            for name in self.effective_loss_keys
        }

    def _active_loss_config(self) -> dict[str, Any]:
        cfg = deepcopy(dict(self.config.get("losses", {})))
        cfg["domain_bounds"] = tuple(self.benchmark.bounds)
        cfg["cfd_supervision_mode"] = str(
            self.config.get("data_supervision", {}).get(
                "mode", "pure_pinn"
            )
        )
        curriculum = dict(cfg.get("near_wall_momentum", {}))
        stages = list(curriculum.get("stages", []))
        if bool(curriculum.get("enabled", False)) and stages:
            selected = stages[-1]
            for stage in stages:
                if self.global_step < int(stage.get("until_step", 0)):
                    selected = stage
                    break
            curriculum.update(
                {
                    "band_width": float(
                        selected.get("band_width", curriculum.get("band_width", 0.08))
                    ),
                    "weight": float(selected.get("weight", 1.0)),
                }
            )
            cfg["near_wall_momentum"] = curriculum
            self.current_cavity_curriculum.update(
                {
                    "near_wall_momentum_band_width": curriculum["band_width"],
                    "near_wall_momentum_weight": curriculum["weight"],
                }
            )
        return cfg

    def _cfd_sparse_metrics(self) -> dict[str, Any]:
        assert self.cfd_supervision is not None
        self.model.eval()
        coords = self.cfd_supervision.coords
        targets = self.cfd_supervision.targets
        with torch.enable_grad():
            prediction = self.model(coords)
            u_mse = torch.mean(
                (prediction[:, 0:1] - targets["u"]).pow(2)
            )
            v_mse = torch.mean(
                (prediction[:, 1:2] - targets["v"]).pow(2)
            )
        return {
            "cfd_sparse_sample_count": self.cfd_supervision.sample_count,
            "cfd_sparse_sample_fraction": self.cfd_supervision.sample_fraction,
            "cfd_sparse_seed": self.cfd_supervision.seed,
            "cfd_sparse_pool_hash": self.cfd_supervision.pool_hash,
            "cfd_sparse_source_path": self.cfd_supervision.source_path,
            "cfd_velocity_mse_sparse": float(
                (u_mse + v_mse).detach().cpu()
            ),
            "cfd_u_mse_sparse": float(u_mse.detach().cpu()),
            "cfd_v_mse_sparse": float(v_mse.detach().cpu()),
        }

    def _compute_boundary_training_loss(self, weights: dict[str, float]) -> bool:
        train_cfg = dict(self.config.get("training", {}))
        formulation = str(self.config.get("model", {}).get("physics_formulation", "direct"))
        skip = bool(train_cfg.get("skip_boundary_loss_if_hard_enforced", False))
        if skip and formulation == "hard_boundary_streamfunction_pressure":
            return False
        return any(
            float(weight) != 0.0
            for name, weight in weights.items()
            if name == "bc" or name == "bc_uvp_balanced" or name.startswith("bc_")
        )

    def _apply_cavity_curriculum(self) -> None:
        cfg = dict(self.config.get("cavity_curriculum", {}))
        if not bool(cfg.get("enabled", False)):
            return
        stages = list(cfg.get("stages", []))
        if not stages:
            return
        selected = stages[-1]
        for stage in stages:
            until = int(stage.get("until_step", stage.get("steps", 0)))
            if self.global_step < until:
                selected = stage
                break
        model = self.model
        if hasattr(model, "corner_width") and "corner_width" in selected:
            model.corner_width = max(float(selected["corner_width"]), 1e-6)
        if hasattr(model, "lid_vertical_power") and "lid_vertical_power" in selected:
            model.lid_vertical_power = max(2, int(selected["lid_vertical_power"]))
        if hasattr(model, "correction_scale") and "correction_scale" in selected:
            model.correction_scale = float(selected["correction_scale"])
        if hasattr(self.benchmark, "lid_corner_regularization_width") and "corner_width" in selected:
            object.__setattr__(
                self.benchmark,
                "lid_corner_regularization_width",
                float(selected["corner_width"]),
            )
        self.current_cavity_curriculum = {
            "cavity_curriculum_corner_width": float(
                getattr(model, "corner_width", float("nan"))
            ),
            "cavity_curriculum_lid_vertical_power": float(
                getattr(model, "lid_vertical_power", float("nan"))
            ),
            "cavity_curriculum_correction_scale": float(
                getattr(model, "correction_scale", float("nan"))
            ),
        }

    def _sync_benchmark_corner_to_model(self) -> None:
        if hasattr(self.model, "corner_width") and hasattr(
            self.benchmark, "lid_corner_regularization_width"
        ):
            object.__setattr__(
                self.benchmark,
                "lid_corner_regularization_width",
                float(self.model.corner_width),
            )

    def _residual_loss_settings(self) -> tuple[str, float]:
        cfg = self.config.get("training", {}).get("residual_loss_mode", "mse")
        delta = float(self.config.get("training", {}).get("pseudo_huber_delta", 1.0))
        if isinstance(cfg, dict):
            switch_step = int(cfg.get("switch_step", 0))
            mode = str(
                cfg.get("initial", "mse")
                if self.global_step < switch_step
                else cfg.get("final", "mse")
            )
            delta = float(cfg.get("pseudo_huber_delta", delta))
            return mode, delta
        return str(cfg), delta

    def _model_auxiliary_logs(self) -> dict[str, float]:
        logs: dict[str, float] = {}
        diagnostics = getattr(self.model, "latest_streamfunction_diagnostics", None)
        if isinstance(diagnostics, dict):
            for key, value in diagnostics.items():
                try:
                    logs[key] = float(value)
                except (TypeError, ValueError):
                    continue
        return logs

    def _step_runtime_profile(self) -> dict[str, float] | None:
        cfg = dict(self.config.get("runtime_profiling", {}))
        if not bool(cfg.get("enabled", False)):
            return None
        every = max(1, int(cfg.get("detailed_every_steps", 50)))
        return {} if self.global_step % every == 0 else None

    def _record_runtime(
        self,
        name: str,
        seconds: float,
        *,
        cycle: int | str = "",
        phase: str = "",
        profile: dict[str, float] | None = None,
    ) -> None:
        seconds = max(0.0, float(seconds))
        self.compute_tracker.add_phase_time(name, seconds)
        row: dict[str, Any] = {
            "step": int(self.global_step),
            "cycle": cycle,
            "phase": phase,
            "component": name,
            "seconds": seconds,
        }
        if profile:
            for component, value in profile.items():
                self.compute_tracker.add_phase_time(component, float(value))
                row[component] = float(value)
        cfg = dict(self.config.get("runtime_profiling", {}))
        if not bool(cfg.get("enabled", False)):
            return
        every = max(1, int(cfg.get("detailed_every_steps", 50)))
        frequent = name in {"loss_objective_total", "backward_optimizer"}
        if not frequent or self.global_step % every == 0:
            self.runtime_logger.log(row)

    def _print_runtime_environment(self) -> None:
        n_f, n_bc, _ = self._training_sample_counts()
        validation_cfg = self.config.get("validation", {})
        total_steps = int(
            self.config.get("controller_v2", {}).get(
                "total_steps",
                int(self.config.get("training", {}).get("adaptive_cycles", 1))
                * int(self.config.get("training", {}).get("epochs_per_cycle", 1)),
            )
        )
        gpu_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else "none"
        )
        dtype = next(self.model.parameters()).dtype
        print(
            "Runtime environment: "
            f"cuda_available={torch.cuda.is_available()} "
            f"gpu={gpu_name} device={self.device} dtype={dtype} "
            f"planned_steps={total_steps} n_collocation={n_f} n_boundary={n_bc} "
            f"validation={validation_cfg.get('nx', 50)}x{validation_cfg.get('ny', 50)}"
        )

    def normalize_training_losses(
        self,
        losses: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Optionally equalize objective scale with reference-free EMA factors."""
        cfg = self.loss_normalization_cfg
        if not bool(cfg.get("enabled", False)):
            return losses, {}
        names = cfg.get("names", ["momentum_u", "momentum_v", "continuity", "bc"])
        names = [str(name) for name in names]
        ema = float(cfg.get("ema", 0.98))
        min_scale = float(cfg.get("min_scale", 1e-6))
        max_scale = float(cfg.get("max_scale", 1e6))
        warmup_steps = max(0, int(cfg.get("warmup_steps", 0)))
        normalized = dict(losses)
        logs: dict[str, float] = {}
        for name in names:
            if name not in losses:
                continue
            raw_value = float(losses[name].detach().cpu())
            if not math.isfinite(raw_value):
                continue
            previous = self.loss_normalization_state.get(name)
            if previous is None:
                scale = raw_value
            else:
                scale = ema * previous + (1.0 - ema) * raw_value
            scale = min(max(scale, min_scale), max_scale)
            self.loss_normalization_state[name] = scale
            logs[f"raw_{name}"] = raw_value
            logs[f"loss_scale_{name}"] = scale
            if self.global_step >= warmup_steps:
                normalized[name] = losses[name] / scale
        return normalized, logs

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
        previous_effective_loss_keys = list(self.effective_loss_keys)
        previous_effective_loss_weights = dict(
            self.effective_loss_weights
        )

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
        score_improved = bool(after_score <= before_score * (1.0 + tolerance))
        pareto_safe, pareto_reason = self._repair_pareto_safe(
            before_metrics,
            after_metrics,
            cfg,
        )
        accepted = score_improved and pareto_safe
        reason = (
            "accepted"
            if accepted
            else pareto_reason
            if not pareto_safe
            else "validation_score_worsened"
        )
        if not accepted:
            self._restore_model_snapshot(model_snapshot)
            self.optimizer = previous_optimizer
            self.optimizer.load_state_dict(optimizer_snapshot)
            self.optimizer_stage = previous_stage
        self.effective_loss_keys = previous_effective_loss_keys
        self.effective_loss_weights = previous_effective_loss_weights

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

    def _repair_pareto_safe(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        cfg: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        cfg = dict(cfg or self._final_repair_config())
        guard = dict(cfg.get("pareto_guard", {}))
        if (
            str(self.config.get("model", {}).get("physics_formulation", ""))
            not in {"cavity_uvp_soft_bc", "cavity_uvp_velocity_lift"}
            or not bool(guard.get("enabled", False))
        ):
            return True, "pareto_guard_disabled"
        tolerance = max(float(guard.get("relative_tolerance", 0.05)), 0.0)
        metrics = list(
            guard.get(
                "metrics",
                [
                    "pde_residual_mean",
                    "momentum_residual_mean",
                    "core_pde_residual_mean",
                    "continuity_residual_mean",
                    "boundary_condition_error",
                    "u_boundary_rmse",
                    "speed_pred_max",
                ],
            )
        )
        for name in metrics:
            old = float(before.get(name, float("nan")))
            new = float(after.get(name, float("nan")))
            if not math.isfinite(old) or not math.isfinite(new):
                return False, f"pareto_nonfinite_{name}"
            if new > old * (1.0 + tolerance) + 1e-12:
                return False, f"pareto_worsened_{name}"
        validity = dict(self.config.get("continuation_validity", {}))
        for name in guard.get(
            "validity_gate_metrics",
            [
                "pde_residual_mean",
                "momentum_residual_mean",
                "core_pde_residual_mean",
                "continuity_residual_mean",
            ],
        ):
            maximum = float(validity.get(f"max_{name}", math.inf))
            old = float(before.get(name, float("nan")))
            new = float(after.get(name, float("nan")))
            if math.isfinite(maximum) and old <= maximum < new:
                return False, f"pareto_crossed_gate_{name}"
        return True, "pareto_safe"

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
        if (
            str(preferred) == "uvp_reference_free_score"
            and str(
                self.config.get("model", {}).get("physics_formulation", "")
            )
            in {"cavity_uvp_soft_bc", "cavity_uvp_velocity_lift"}
        ):
            return "uvp_reference_free_score", self._checkpoint_score(metrics)
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

    def prepare_optimizer_step(self) -> float:
        """Apply shared clipping and a deterministic applied-step LR schedule."""
        if self.max_gradient_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_gradient_norm,
            )
        learning_rate = self.base_learning_rate
        cfg = self.learning_rate_schedule
        if bool(cfg.get("enabled", False)):
            total_steps = int(
                cfg.get(
                    "total_steps",
                    self.config.get("controller_v2", {}).get(
                        "total_steps",
                        int(self.config.get("training", {}).get("adaptive_cycles", 1))
                        * int(self.config.get("training", {}).get("epochs_per_cycle", 1)),
                    ),
                )
            )
            warmup_steps = max(0, int(cfg.get("warmup_steps", 0)))
            minimum_ratio = float(cfg.get("min_lr_ratio", 0.1))
            step = max(0, int(self.global_step))
            if warmup_steps > 0 and step < warmup_steps:
                start_ratio = float(cfg.get("warmup_start_ratio", 0.2))
                fraction = step / max(warmup_steps, 1)
                ratio = start_ratio + (1.0 - start_ratio) * fraction
            else:
                progress = (step - warmup_steps) / max(
                    total_steps - warmup_steps - 1,
                    1,
                )
                progress = min(max(progress, 0.0), 1.0)
                ratio = minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (
                    1.0 + math.cos(math.pi * progress)
                )
            learning_rate = self.base_learning_rate * ratio
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return float(learning_rate)

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
        n_f, n_bc, n_data = self._training_sample_counts()
        started = time.perf_counter()
        if adaptive:
            priorities = control_state.sampling_priorities if control_state is not None else {}
            circulation = self._circulation_band_counts(n_f)
            n_core = (
                circulation["uniform"]
                if circulation is not None
                else self._interior_component_counts(n_f)[0]
            )
            core = self.adaptive_sampler.sample_interior(
                n_core, maps, coords, weak_regions, priorities
            )
            xy_f = self._sample_interior(n_f, core)
            if getattr(self.benchmark, "t_bounds", None) is not None:
                xy_data = self._sample_data(n_data)
            else:
                xy_data = self.adaptive_sampler.sample_interior(n_data, maps, coords, weak_regions, priorities)
        else:
            xy_f = self._sample_interior(n_f)
            xy_data = self._sample_data(n_data)
        xy_bc = self._sample_boundary(n_bc)
        result = self.make_batch(xy_f, xy_bc, xy_data)
        self._record_runtime("sampling", time.perf_counter() - started, phase="resample")
        return result

    def evaluate_and_save_final(self) -> dict[str, float]:
        self._restore_best_checkpoint_if_enabled()
        self._validate_final_cavity_state()
        X, Y, coords = self.test_grid()
        phase_start = time.perf_counter()
        metrics = self.evaluate_metrics(coords)
        self.compute_tracker.add_phase_time("evaluation", time.perf_counter() - phase_start)
        metrics["final_total_loss"] = float(self.last_losses.get("total", float("nan")))
        for name in (
            "speed_cap",
            "raw_psi_l2",
            "raw_psi_mean_l2",
            "scaled_correction_mean_l2",
            "scaled_correction_abs_max_hinge",
            "top_reverse_u",
            "bottom_positive_u",
            "raw_pde_tail",
            "pressure_gradient_l2",
            "vorticity_smoothness",
            "near_wall_vorticity_l2",
            "near_wall_momentum_weight_mean",
        ):
            metrics[f"final_training_{name}"] = float(
                self.last_losses.get(name, float("nan"))
            )
        metrics["stabilizers_enabled"] = bool(
            any(
                bool(dict(value).get("enabled", False))
                for value in self.config.get("losses", {}).values()
                if isinstance(value, dict)
            )
            or isinstance(
                self.config.get("training", {}).get("residual_loss_mode"),
                dict,
            )
            or bool(self.config.get("cavity_curriculum", {}).get("enabled", False))
        )
        metrics["optimizer_stage"] = self.optimizer_stage
        for key, value in self.final_repair_status.items():
            metrics[f"final_repair_{key}"] = value
        for key, value in self.warm_start_status.items():
            metrics[f"warm_start_{key}"] = value
        metrics["reference_kind"] = getattr(self.benchmark, "reference_kind", "analytical")
        metrics["has_reference"] = bool(getattr(self.benchmark, "has_reference", True))
        metrics["run_type"] = str(self.config.get("run_type", "full"))
        model_cfg = self.config.get("model", {})
        supervision_cfg = dict(self.config.get("data_supervision", {}))
        training_weights = dict(
            self.config.get("training", {}).get("weights", {})
        )
        metrics["cfd_supervision_mode"] = str(
            supervision_cfg.get("mode", "pure_pinn")
        )
        metrics["sparse_cfd_polish_enabled"] = bool(
            metrics["cfd_supervision_mode"] == "sparse_cfd_polish"
        )
        metrics["cfd_sampling_mode"] = str(
            supervision_cfg.get("cfd", {})
            .get("sampling", {})
            .get("mode", "uniform")
        )
        metrics["cfd_u_weight"] = float(
            training_weights.get(
                "cfd_u_mse",
                training_weights.get("cfd_velocity_mse", 0.0),
            )
        )
        metrics["cfd_v_weight"] = float(
            training_weights.get(
                "cfd_v_mse",
                training_weights.get("cfd_velocity_mse", 0.0),
            )
        )
        metrics["cfd_supervision_is_oracle"] = bool(
            metrics["cfd_supervision_mode"] == "full_cfd_oracle"
        )
        metrics["effective_loss_keys"] = list(self.effective_loss_keys)
        metrics["effective_loss_weights"] = dict(
            self.effective_loss_weights
        )
        save_json(
            {
                "method_mode": self.mode,
                "loss_keys": metrics["effective_loss_keys"],
                "effective_weights": metrics["effective_loss_weights"],
            },
            self.run_dir / "effective_loss_diagnostics.json",
        )
        if self.cfd_supervision is None:
            metrics["cfd_sparse_sample_count"] = 0
            metrics["cfd_sparse_sample_fraction"] = 0.0
            metrics["cfd_sparse_seed"] = int(
                supervision_cfg.get("seed", self.seed)
            )
            metrics["cfd_sparse_pool_hash"] = ""
            metrics["sparse_cfd_pool_hash"] = ""
            metrics["cfd_velocity_mse_sparse"] = float("nan")
            metrics["cfd_u_mse_sparse"] = float("nan")
            metrics["cfd_v_mse_sparse"] = float("nan")
        else:
            metrics.update(self._cfd_sparse_metrics())
            metrics["sparse_cfd_pool_hash"] = (
                self.cfd_supervision.pool_hash
            )
        metrics["cfd_velocity_full_rel_l2_eval_only"] = metrics.get(
            "velocity_full_rel_l2", float("nan")
        )
        metrics["cfd_u_full_rel_l2_eval_only"] = metrics.get(
            "u_full_rel_l2", float("nan")
        )
        metrics["cfd_v_full_rel_l2_eval_only"] = metrics.get(
            "v_full_rel_l2", float("nan")
        )
        metrics["cfd_omega_full_rel_l2_eval_only"] = metrics.get(
            "omega_full_rel_l2", float("nan")
        )
        metrics["model_architecture"] = str(model_cfg.get("architecture", "mlp"))
        metrics["physics_formulation"] = str(model_cfg.get("physics_formulation", "direct"))
        if metrics["physics_formulation"] == "cavity_uvp_velocity_lift":
            core_speed = float(metrics.get("core_speed_mean", float("nan")))
            upper_speed = float(
                metrics.get("upper_core_speed_mean", float("nan"))
            )
            metrics["lifted_uvp_diagnostics"] = {
                "formulation": metrics["physics_formulation"],
                "lift_enabled": True,
                "lid_u_boundary_rmse": metrics.get(
                    "lid_u_boundary_rmse", float("nan")
                ),
                "no_slip_wall_u_rmse": metrics.get(
                    "no_slip_wall_u_rmse", float("nan")
                ),
                "no_slip_wall_v_rmse": metrics.get(
                    "no_slip_wall_v_rmse", float("nan")
                ),
                "top_band_pde_residual_mean": metrics.get(
                    "top_band_pde_residual_mean", float("nan")
                ),
                "upper_core_pde_residual_mean": metrics.get(
                    "upper_core_pde_residual_mean", float("nan")
                ),
                "core_speed_mean": core_speed,
                "upper_core_speed_mean": upper_speed,
                "speed_core_ratio": core_speed / max(upper_speed, 1e-12),
            }
        metrics["hard_boundary_corner_width"] = (
            float(
                getattr(
                    self.model,
                    "corner_width",
                    model_cfg.get("hard_boundary_corner_width", 0.02),
                )
            )
            if metrics["physics_formulation"]
            in {"cavity_hard_boundary", "hard_boundary_streamfunction_pressure"}
            else float("nan")
        )
        metrics["hard_boundary_lid_vertical_power"] = (
            float(
                getattr(
                    self.model,
                    "lid_vertical_power",
                    model_cfg.get("hard_boundary_lid_vertical_power", float("nan")),
                )
            )
            if metrics["physics_formulation"]
            in {"cavity_hard_boundary", "hard_boundary_streamfunction_pressure"}
            else float("nan")
        )
        metrics["hard_boundary_correction_scale"] = (
            float(
                getattr(
                    self.model,
                    "correction_scale",
                    model_cfg.get("hard_boundary_correction_scale", float("nan")),
                )
            )
            if metrics["physics_formulation"] == "hard_boundary_streamfunction_pressure"
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
        metrics["early_stopped"] = bool(self.early_stopped)
        metrics["early_stop_step"] = self.early_stop_step
        metrics["collapse_evaluated"] = bool(metrics["reportable"])
        metrics["collapsed"] = self._collapsed(metrics)
        metrics.update(self.compute_tracker.summary())
        self.metrics_logger.log({"cycle": "final_test", **metrics})
        save_json(metrics, self.run_dir / "summary.json")
        pd.DataFrame([metrics]).to_csv(self.table_dir / "summary.csv", index=False)
        pd.DataFrame([metrics]).to_csv(self.run_dir / "summary_table.csv", index=False)
        plotting_start = time.perf_counter()
        self.save_plots(X, Y, coords)
        self._record_runtime("plotting", time.perf_counter() - plotting_start, phase="final")
        checkpoint_start = time.perf_counter()
        save_checkpoint(
            self.checkpoint_dir / "final.pt",
            self.model,
            self.optimizer,
            self.config,
            metrics,
            self.global_step,
            -1,
        )
        self._record_runtime(
            "checkpoint_save",
            time.perf_counter() - checkpoint_start,
            phase="final",
        )
        save_intervention_timeline(self.action_records, self.figure_dir / "intervention_timeline.png")
        # Re-save after plotting/checkpoint timings are available.
        metrics.update(self.compute_tracker.summary())
        save_json(metrics, self.run_dir / "summary.json")
        pd.DataFrame([metrics]).to_csv(self.table_dir / "summary.csv", index=False)
        pd.DataFrame([metrics]).to_csv(self.run_dir / "summary_table.csv", index=False)
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
            save_streamlines(
                X,
                Y,
                maps["u_ref"].reshape(shape),
                maps["v_ref"].reshape(shape),
                self.figure_dir / "reference_streamlines.png",
                closed_boundary=self._is_lid_driven_cavity(),
                title="CFD reference velocity streamlines",
            )
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
        streamline_x = X
        streamline_y = Y
        streamline_u = maps["u_pred"].reshape(shape)
        streamline_v = maps["v_pred"].reshape(shape)
        visualization_cfg = dict(self.config.get("visualization", {}))
        visualization_nx = int(visualization_cfg.get("nx", shape[1]))
        visualization_ny = int(visualization_cfg.get("ny", shape[0]))
        if visualization_nx != shape[1] or visualization_ny != shape[0]:
            streamline_x, streamline_y, streamline_coords_np = self.benchmark.grid(
                visualization_nx,
                visualization_ny,
            )
            streamline_coords = torch.tensor(
                streamline_coords_np,
                dtype=torch.float32,
                device=self.device,
            )
            self.model.eval()
            with torch.no_grad():
                streamline_prediction = self.model(streamline_coords)
            streamline_u = (
                streamline_prediction[:, 0:1]
                .detach()
                .cpu()
                .numpy()
                .reshape(streamline_x.shape)
            )
            streamline_v = (
                streamline_prediction[:, 1:2]
                .detach()
                .cpu()
                .numpy()
                .reshape(streamline_x.shape)
            )
        save_streamlines(
            streamline_x,
            streamline_y,
            streamline_u,
            streamline_v,
            self.figure_dir / "streamlines.png",
            closed_boundary=self._is_lid_driven_cavity(),
        )
        save_streamfunction_contours(
            streamline_x,
            streamline_y,
            streamline_u,
            streamline_v,
            self.figure_dir / "streamfunction_contours.png",
            closed_boundary=self._is_lid_driven_cavity(),
        )

    def maybe_checkpoint(self, cycle: int, metrics: dict[str, float]) -> None:
        score = self._checkpoint_score(metrics)
        started = time.perf_counter()
        checkpoint_cfg = dict(self.config.get("checkpoint", {}))
        eligible = self._checkpoint_is_final_restore_eligible()
        improved = eligible and score < self.best_score
        if bool(checkpoint_cfg.get("save_latest_every_cycle", True)):
            save_checkpoint(
                self.checkpoint_dir / "latest.pt",
                self.model,
                self.optimizer,
                self.config,
                metrics,
                self.global_step,
                cycle,
            )
        if improved:
            self.best_score = score
            save_checkpoint(self.checkpoint_dir / "best.pt", self.model, self.optimizer, self.config, metrics, self.global_step, cycle)
        self._record_runtime(
            "checkpoint_save",
            time.perf_counter() - started,
            cycle=cycle,
            phase="best" if improved else "latest",
        )

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
        checkpoint_cfg = dict(self.config.get("checkpoint", {}))
        metric_weights = dict(checkpoint_cfg.get("metric_weights", {}))
        metric_names = list(
            checkpoint_cfg.get(
                "reference_free_metrics",
                [
                    "momentum_residual_mean",
                    "continuity_residual_mean",
                    "boundary_condition_error",
                    "streamfunction_consistency_rmse",
                ],
            )
        )
        floor = float(checkpoint_cfg.get("metric_floor", 1e-8))
        values = []
        for name in metric_names:
            value = metrics.get(name, self.last_losses.get(name))
            if value is None or not math.isfinite(float(value)):
                continue
            values.append(
                (
                    max(float(value), floor),
                    float(metric_weights.get(name, 1.0)),
                )
            )
        if values:
            speed = float(metrics.get("speed_pred_max", float("nan")))
            speed_gate = float(
                self.config.get("continuation_validity", {}).get(
                    "max_speed_pred",
                    math.inf,
                )
            )
            speed_penalty = (
                max(speed - speed_gate, 0.0) ** 2
                if math.isfinite(speed) and math.isfinite(speed_gate)
                else 0.0
            )
            speed_penalty *= float(checkpoint_cfg.get("speed_hinge_weight", 1.0))
            topology_penalty = 0.0
            topology_cfg = dict(
                checkpoint_cfg.get("low_re_vortex_tiebreaker", {})
            )
            reynolds = float(
                self.config.get("benchmark_params", {}).get(
                    "reynolds",
                    math.inf,
                )
            )
            if bool(topology_cfg.get("enabled", False)) and reynolds <= float(
                topology_cfg.get("max_re", 200.0)
            ):
                detected = metrics.get("detected_vortex_count")
                secondary = metrics.get("secondary_vortex_count")
                if detected is not None and math.isfinite(float(detected)):
                    topology_penalty += float(
                        topology_cfg.get("detected_vortex_weight", 0.05)
                    ) * max(
                        float(detected)
                        - float(topology_cfg.get("detected_vortex_limit", 2)),
                        0.0,
                    )
                if secondary is not None and math.isfinite(float(secondary)):
                    topology_penalty += float(
                        topology_cfg.get("secondary_vortex_weight", 0.05)
                    ) * max(
                        float(secondary)
                        - float(topology_cfg.get("secondary_vortex_limit", 1)),
                        0.0,
                    )
            if str(checkpoint_cfg.get("score_mode", "geometric_mean")).lower() in {
                "sum",
                "additive",
            }:
                return float(
                    sum(value * weight for value, weight in values)
                    + speed_penalty
                    + topology_penalty
                )
            total_weight = sum(weight for _value, weight in values)
            geometric = math.exp(
                sum(weight * math.log(value) for value, weight in values)
                / max(total_weight, 1e-12)
            )
            return float(geometric + speed_penalty + topology_penalty)
        pde = float(metrics.get("unweighted_pde_loss", float("nan")))
        boundary = float(metrics.get("unweighted_bc_loss", float("nan")))
        if math.isfinite(pde) and math.isfinite(boundary):
            return pde + boundary
        fallback = float(metrics.get("pde_residual_mean", float("inf")))
        return fallback if math.isfinite(fallback) else math.inf

    def should_stop_early(self, metrics: dict[str, Any]) -> bool:
        """Stop only after a valid, stable reference-free convergence window."""
        cfg = dict(self.config.get("convergence_early_stopping", {}))
        if not bool(cfg.get("enabled", False)):
            return False
        minimum = int(
            cfg.get(
                "min_steps_warm_start"
                if self.warm_start_status.get("loaded")
                else "min_steps_initial",
                0,
            )
        )
        score = self._checkpoint_score(metrics)
        if math.isfinite(score):
            self.early_stop_history.append(score)
        if self.global_step < minimum or not self._passes_reference_free_validity(metrics):
            return False
        patience = max(2, int(cfg.get("patience", 3)))
        if len(self.early_stop_history) < patience:
            return False
        window = self.early_stop_history[-patience:]
        scale = max(min(window), float(cfg.get("score_floor", 1e-12)))
        relative_spread = (max(window) - min(window)) / scale
        latest_is_stable = window[-1] <= min(window) * (
            1.0 + float(cfg.get("latest_degradation_tolerance", 0.01))
        )
        if relative_spread <= float(cfg.get("relative_tolerance", 0.03)) and latest_is_stable:
            self.early_stopped = True
            self.early_stop_step = int(self.global_step)
            self.compute_tracker.stop_reason = "reference_free_convergence"
            return True
        return False

    def _passes_reference_free_validity(self, metrics: dict[str, Any]) -> bool:
        cfg = dict(self.config.get("continuation_validity", {}))
        if not bool(cfg.get("enabled", False)):
            return True
        maximums = {
            "pde_residual_mean": cfg.get("max_pde_residual_mean", math.inf),
            "continuity_residual_mean": cfg.get(
                "max_continuity_residual_mean", math.inf
            ),
            "momentum_residual_mean": cfg.get(
                "max_momentum_residual_mean", math.inf
            ),
            "boundary_condition_error": cfg.get(
                "max_boundary_condition_error", math.inf
            ),
            "speed_pred_max": cfg.get("max_speed_pred", math.inf),
            "streamfunction_consistency_rmse": cfg.get(
                "max_streamfunction_consistency_rmse", math.inf
            ),
        }
        for name, maximum in maximums.items():
            value = float(metrics.get(name, float("nan")))
            if not math.isfinite(value) or value > float(maximum):
                return False
        psi = float(metrics.get("primary_streamfunction_abs", float("nan")))
        return math.isfinite(psi) and psi >= float(
            cfg.get("min_primary_streamfunction_abs", 0.0)
        )

    def _restore_best_checkpoint_if_enabled(self) -> None:
        cfg = dict(self.config.get("checkpoint", {}))
        path = self.checkpoint_dir / "best.pt"
        if bool(cfg.get("restore_best_before_final", False)) and path.exists():
            payload = load_checkpoint(path, self.model, optimizer=None)
            if not self._checkpoint_payload_is_final_restore_eligible(payload):
                raise ValueError(
                    "Reliable best checkpoint is not eligible for final restore."
                )
            self._sync_benchmark_corner_to_model()
            self.final_repair_status["restored_best_checkpoint"] = True
            self.final_repair_status["best_checkpoint_epoch"] = payload.get("epoch")

    def _final_cavity_stage(self) -> dict[str, Any]:
        stages = list(self.config.get("cavity_curriculum", {}).get("stages", []))
        return dict(stages[-1]) if stages else {}

    def _checkpoint_min_restore_step(self) -> int:
        cfg = dict(self.config.get("checkpoint", {}))
        total_steps = int(
            self.config.get("controller_v2", {}).get(
                "total_steps",
                self.config.get("optimizer", {})
                .get("scheduler", {})
                .get("total_steps", 0),
            )
        )
        fraction = min(
            max(float(cfg.get("eligible_final_fraction", 0.0)), 0.0),
            1.0,
        )
        return max(0, int(math.ceil(total_steps * (1.0 - fraction))))

    def _runtime_matches_final_cavity_stage(
        self,
        runtime_state: dict[str, Any] | None = None,
    ) -> bool:
        cfg = dict(self.config.get("checkpoint", {}))
        if not bool(cfg.get("require_final_cavity_stage", False)):
            return True
        final_stage = self._final_cavity_stage()
        if not final_stage:
            return True
        state = runtime_state or {
            "corner_width": getattr(self.model, "corner_width", float("nan")),
            "lid_vertical_power": getattr(
                self.model, "lid_vertical_power", float("nan")
            ),
            "correction_scale": getattr(
                self.model, "correction_scale", float("nan")
            ),
        }
        return (
            math.isclose(
                float(state.get("corner_width", float("nan"))),
                float(final_stage["corner_width"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and int(state.get("lid_vertical_power", -1))
            == int(final_stage["lid_vertical_power"])
            and math.isclose(
                float(state.get("correction_scale", float("nan"))),
                float(final_stage["correction_scale"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    def _checkpoint_is_final_restore_eligible(self) -> bool:
        return (
            self.global_step >= self._checkpoint_min_restore_step()
            and self._runtime_matches_final_cavity_stage()
        )

    def _checkpoint_payload_is_final_restore_eligible(
        self,
        payload: dict[str, Any],
    ) -> bool:
        return (
            int(payload.get("epoch", -1)) >= self._checkpoint_min_restore_step()
            and self._runtime_matches_final_cavity_stage(
                dict(payload.get("model_runtime_state", {}))
            )
        )

    def _validate_final_cavity_state(self) -> None:
        cfg = dict(self.config.get("checkpoint", {}))
        if not bool(cfg.get("require_final_cavity_stage", False)):
            return
        formulation = str(
            self.config.get("model", {}).get("physics_formulation", "")
        )
        if formulation != "hard_boundary_streamfunction_pressure":
            raise ValueError(
                "Reliable final evaluation requires "
                "hard_boundary_streamfunction_pressure."
            )
        if not self._runtime_matches_final_cavity_stage():
            raise ValueError(
                "Reliable final evaluation model does not match the final "
                "cavity curriculum stage."
            )
        controller_steps = int(
            self.config.get("controller_v2", {}).get("total_steps", -1)
        )
        scheduler_steps = int(
            self.config.get("optimizer", {})
            .get("scheduler", {})
            .get("total_steps", -1)
        )
        if controller_steps != scheduler_steps:
            raise ValueError(
                "Reliable final evaluation requires matching controller and "
                "optimizer total steps."
            )

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
