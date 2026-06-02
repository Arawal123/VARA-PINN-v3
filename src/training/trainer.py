"""Shared trainer utilities."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from src.diagnostics import DiagnosticMapBuilder, PatchGrid, PatchScorer, WeakRegionDetector
from src.evaluation.metrics import evaluate_on_grid
from src.losses.base_losses import compute_global_losses, compute_pointwise_losses, weighted_sum
from src.losses.local_losses import compute_local_weighted_loss
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
from src.sampling import BoundarySampler, MixedAdaptiveSampler, UniformSampler
from src.sampling.residual_sampler import sample_from_score_grid
from src.training.checkpointing import save_checkpoint
from src.training.lbfgs_utils import make_lbfgs_closure
from src.utils.config import save_config
from src.utils.device import get_device
from src.utils.io import ensure_dir, save_json
from src.utils.logging import CSVLogger, JSONListLogger, make_run_id
from src.utils.seed import set_seed
from src.visualization.controller_plots import save_intervention_timeline, save_patch_score_map
from src.visualization.fields import save_field_panel
from src.visualization.heatmaps import save_heatmap
from src.visualization.streamlines import save_streamlines


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
        self.repair_rng = np.random.default_rng(self.seed + 7919)
        self.global_step = 0
        self.best_score = math.inf

        patch_cfg = config.get("patches", {})
        self.patch_grid = PatchGrid(
            self.benchmark.bounds,
            nx_patches=int(patch_cfg.get("nx_patches", 4)),
            ny_patches=int(patch_cfg.get("ny_patches", 4)),
            nt_patches=int(patch_cfg.get("nt_patches", 1)),
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

        self.uniform_sampler = UniformSampler(self.benchmark.bounds, self.device, self.seed)
        self.boundary_sampler = BoundarySampler(self.benchmark.bounds, self.device, self.seed + 1)
        sampler_cfg = config.get("sampling", {})
        self.adaptive_sampler = MixedAdaptiveSampler(
            self.benchmark.bounds,
            self.patch_grid,
            self.device,
            self.seed + 2,
            mixture=sampler_cfg.get("mixture"),
        )

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
        xy_f = self._sample_interior(n_f)
        xy_bc = self._sample_boundary(n_bc)
        xy_data = self._sample_interior(n_data)
        return self.make_batch(xy_f, xy_bc, xy_data)

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

    def _sample_interior(self, n: int) -> torch.Tensor:
        return self._sample_cavity_focused_interior(n, self.uniform_sampler.sample)

    def _sample_cavity_focused_interior(self, n: int, base_sampler: Callable[[int], torch.Tensor]) -> torch.Tensor:
        if n <= 0:
            return torch.zeros((0, 2), dtype=torch.float32, device=self.device)
        focus_cfg = self.config.get("sampling", {}).get("cavity_focus", {})
        if not (self._is_lid_driven_cavity() and bool(focus_cfg.get("enabled", False))):
            return base_sampler(n)

        near_wall_fraction = float(focus_cfg.get("near_wall_fraction", 0.12))
        centerline_fraction = float(focus_cfg.get("centerline_fraction", 0.10))
        lid_fraction = float(focus_cfg.get("lid_fraction", 0.06))
        corner_fraction = float(focus_cfg.get("corner_fraction", 0.04))
        fractions = np.clip(
            np.array([near_wall_fraction, centerline_fraction, lid_fraction, corner_fraction], dtype=float),
            0.0,
            1.0,
        )
        total_focus = float(np.sum(fractions))
        if total_focus > 0.70:
            fractions *= 0.70 / total_focus
        counts = np.floor(fractions * n).astype(int)
        n_base = max(0, n - int(np.sum(counts)))
        pieces = [base_sampler(n_base).detach().cpu().numpy()]
        strip_width = float(focus_cfg.get("strip_width", 0.08))
        centerline_width = float(focus_cfg.get("centerline_width", 0.04))
        pieces.append(self._sample_near_wall_strip_numpy(int(counts[0]), strip_width))
        pieces.append(self._sample_centerline_band_numpy(int(counts[1]), centerline_width))
        pieces.append(self._sample_lid_strip_numpy(int(counts[2]), strip_width))
        pieces.append(self._sample_corner_strip_numpy(int(counts[3]), strip_width))
        out = np.vstack([piece for piece in pieces if piece.size])
        if out.shape[0] < n:
            out = np.vstack([out, base_sampler(n - out.shape[0]).detach().cpu().numpy()])
        elif out.shape[0] > n:
            out = out[:n]
        self.uniform_sampler.rng.shuffle(out)
        return torch.tensor(out, dtype=torch.float32, device=self.device)

    def _sample_near_wall_strip_numpy(self, n: int, strip_width: float) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 2), dtype=float)
        x0, x1, y0, y1 = self.benchmark.bounds
        span = min(max(x1 - x0, 1e-12), max(y1 - y0, 1e-12))
        w = min(max(float(strip_width) * span, 1e-9), 0.5 * span)
        sides = self.uniform_sampler.rng.integers(0, 4, n)
        pts = np.zeros((n, 2), dtype=float)
        for i, side in enumerate(sides):
            if side == 0:
                pts[i] = [self.uniform_sampler.rng.uniform(x0, x0 + w), self.uniform_sampler.rng.uniform(y0, y1)]
            elif side == 1:
                pts[i] = [self.uniform_sampler.rng.uniform(x1 - w, x1), self.uniform_sampler.rng.uniform(y0, y1)]
            elif side == 2:
                pts[i] = [self.uniform_sampler.rng.uniform(x0, x1), self.uniform_sampler.rng.uniform(y0, y0 + w)]
            else:
                pts[i] = [self.uniform_sampler.rng.uniform(x0, x1), self.uniform_sampler.rng.uniform(y1 - w, y1)]
        return pts

    def _sample_centerline_band_numpy(self, n: int, centerline_width: float) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 2), dtype=float)
        x0, x1, y0, y1 = self.benchmark.bounds
        span = min(max(x1 - x0, 1e-12), max(y1 - y0, 1e-12))
        w = min(max(float(centerline_width) * span, 1e-9), 0.5 * span)
        x_mid = 0.5 * (x0 + x1)
        y_mid = 0.5 * (y0 + y1)
        vertical = self.uniform_sampler.rng.random(n) < 0.5
        pts = np.zeros((n, 2), dtype=float)
        v_count = int(np.sum(vertical))
        h_count = n - v_count
        pts[vertical, 0] = self.uniform_sampler.rng.uniform(max(x0, x_mid - w), min(x1, x_mid + w), v_count)
        pts[vertical, 1] = self.uniform_sampler.rng.uniform(y0, y1, v_count)
        pts[~vertical, 0] = self.uniform_sampler.rng.uniform(x0, x1, h_count)
        pts[~vertical, 1] = self.uniform_sampler.rng.uniform(max(y0, y_mid - w), min(y1, y_mid + w), h_count)
        return pts

    def _sample_lid_strip_numpy(self, n: int, strip_width: float) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 2), dtype=float)
        x0, x1, y0, y1 = self.benchmark.bounds
        span = min(max(x1 - x0, 1e-12), max(y1 - y0, 1e-12))
        w = min(max(float(strip_width) * span, 1e-9), y1 - y0)
        return np.column_stack(
            [
                self.uniform_sampler.rng.uniform(x0, x1, n),
                self.uniform_sampler.rng.uniform(max(y0, y1 - w), y1, n),
            ]
        )

    def _sample_corner_strip_numpy(self, n: int, strip_width: float) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 2), dtype=float)
        x0, x1, y0, y1 = self.benchmark.bounds
        span = min(max(x1 - x0, 1e-12), max(y1 - y0, 1e-12))
        w = min(max(float(strip_width) * span, 1e-9), 0.5 * span)
        corners = self.uniform_sampler.rng.integers(0, 4, n)
        pts = np.zeros((n, 2), dtype=float)
        for i, corner in enumerate(corners):
            if corner == 0:
                pts[i] = [self.uniform_sampler.rng.uniform(x0, x0 + w), self.uniform_sampler.rng.uniform(y0, y0 + w)]
            elif corner == 1:
                pts[i] = [self.uniform_sampler.rng.uniform(x1 - w, x1), self.uniform_sampler.rng.uniform(y0, y0 + w)]
            elif corner == 2:
                pts[i] = [self.uniform_sampler.rng.uniform(x0, x0 + w), self.uniform_sampler.rng.uniform(y1 - w, y1)]
            else:
                pts[i] = [self.uniform_sampler.rng.uniform(x1 - w, x1), self.uniform_sampler.rng.uniform(y1 - w, y1)]
        return pts

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
        for local_epoch in range(epochs):
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

            last_losses = {k: float(v.detach().cpu()) for k, v in losses.items()}
            last_losses.update(local_logs)
            last_losses["total"] = float(total.detach().cpu())
            last_losses["grad_norm"] = grad_norm
            self.last_losses = dict(last_losses)
            if local_epoch % log_every == 0 or local_epoch == epochs - 1:
                self.loss_logger.log({"cycle": cycle, "phase": log_prefix or "main", "epoch": self.global_step, **last_losses})
            self.global_step += 1
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
        pointwise = compute_pointwise_losses(self.model, batch, self.benchmark, self.steady)
        if "pressure_poisson" in active_aux_losses:
            pointwise["pressure_poisson"] = pressure_poisson_residual(
                self.model, batch["xy_f"], self.benchmark.nu
            ).pow(2)
        if "vorticity_transport" in active_aux_losses and not self.steady:
            pointwise["vorticity_transport"] = vorticity_transport_residual(
                self.model, batch["xy_f"], self.benchmark.nu, steady=False
            ).pow(2)
        losses = compute_global_losses(pointwise)
        total = weighted_sum(losses, weights)
        local_loss, local_logs = compute_local_weighted_loss(
            pointwise,
            batch,
            self.patch_grid,
            local_weights,
            entropy_weight=float(self.config.get("controller", {}).get("entropy_weight", 0.0)),
        )
        total = total + local_loss
        if pressure_anchor_patches:
            patch_ids = self.patch_grid.assign_torch(batch["xy_f"])
            pred_f = self.model(batch["xy_f"])
            for pid, strength in pressure_anchor_patches.items():
                mask = patch_ids == int(pid)
                if torch.any(mask):
                    total = total + float(strength) * pressure_anchor_loss(pred_f[mask, 2:3], 0.0)
        return total, losses, local_logs

    def run_final_physics_repair(self, cycle: int = -1, log_prefix: str = "final_repair") -> dict[str, Any]:
        """Run a guarded global-physics LBFGS repair stage.

        VARA can leave useful local interventions behind, but LBFGS is too sharp
        to safely optimize those local weights directly. This stage therefore
        builds a fresh global batch, ignores local controller weights, and keeps
        the result only when validation score improves.
        """
        cfg = self._final_repair_config()
        if not bool(cfg.get("enabled", False)):
            self.final_repair_status = {"enabled": False, "accepted": False}
            return self.final_repair_status

        steps = max(0, int(cfg.get("epochs", cfg.get("steps", 0))))
        if steps <= 0:
            self.final_repair_status = {"enabled": True, "accepted": False, "reason": "zero_steps"}
            return self.final_repair_status

        _, _, validation_coords = self.validation_grid()
        before_metrics = evaluate_on_grid(self.model, self.benchmark, validation_coords, self.device, self.steady)
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
        for step in range(steps):
            closure = make_lbfgs_closure(lbfgs, loss_fn)
            loss = lbfgs.step(closure)
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

        after_metrics = evaluate_on_grid(self.model, self.benchmark, validation_coords, self.device, self.steady)
        _, after_score = self._repair_score(after_metrics)
        tolerance = float(cfg.get("acceptance_tolerance", 0.0))
        score_ok = bool(after_score <= before_score * (1.0 + tolerance))
        collateral_ok, collateral_report = self._repair_collateral_ok(before_metrics, after_metrics, cfg)
        accepted = bool(score_ok and collateral_ok)
        reason = "accepted"
        if not score_ok:
            reason = "validation_score_worsened"
        elif not collateral_ok:
            reason = "collateral_damage_exceeded"
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
            "score_ok": score_ok,
            "collateral_ok": collateral_ok,
            "epochs": steps,
            "batch_n_collocation": int(repair_batch["xy_f"].shape[0]),
            "batch_n_boundary": int(repair_batch["xy_bc"].shape[0]),
            "global_only": True,
            **collateral_report,
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
        cfg.setdefault("collateral_tolerances", {})
        return cfg

    def _repair_collateral_ok(
        self,
        before_metrics: dict[str, Any],
        after_metrics: dict[str, Any],
        cfg: dict[str, Any],
    ) -> tuple[bool, dict[str, float | str]]:
        tolerances = dict(cfg.get("collateral_tolerances", {}))
        if not tolerances:
            return True, {
                "collateral_metric_status": "disabled",
                "collateral_max_damage": 0.0,
            }
        ok = True
        max_damage = 0.0
        worst_metric = ""
        report: dict[str, float | str] = {"collateral_metric_status": "ok"}
        for metric_name, tolerance in tolerances.items():
            before = self._finite_metric(before_metrics, str(metric_name))
            after = self._finite_metric(after_metrics, str(metric_name))
            if before is None or after is None:
                report[f"collateral_{metric_name}_damage"] = float("nan")
                continue
            damage = max(0.0, (after - before) / (abs(before) + 1e-12))
            report[f"collateral_{metric_name}_damage"] = float(damage)
            report[f"collateral_{metric_name}_tolerance"] = float(tolerance)
            if damage > max_damage:
                max_damage = float(damage)
                worst_metric = str(metric_name)
            if damage > float(tolerance):
                ok = False
        if not ok:
            report["collateral_metric_status"] = f"failed:{worst_metric}"
        report["collateral_max_damage"] = float(max_damage)
        report["collateral_worst_metric"] = worst_metric
        return ok, report

    def _finite_metric(self, metrics: dict[str, Any], name: str) -> float | None:
        try:
            value = float(metrics.get(name))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

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
            builder = DiagnosticMapBuilder(self.model, self.benchmark, self.device, self.steady)
            maps = builder.build(coords, mode=self.config.get("diagnostics", {}).get("mode", "full_reference"))
            score = maps.get("aggregate_pde_residual", maps.get("pde_residual"))
            if score is not None:
                pieces.append(sample_from_score_grid(coords, score, n_residual, self.repair_rng))
            else:
                pieces.append(self.uniform_sampler.sample(n_residual).detach().cpu().numpy())
        xy_f_np = np.vstack([p for p in pieces if p.size])
        self.repair_rng.shuffle(xy_f_np)
        xy_f = torch.tensor(xy_f_np, dtype=torch.float32, device=self.device)
        xy_bc = self._sample_boundary(n_bc)
        xy_data = self.uniform_sampler.sample(n_data)
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
        X, Y, coords = self.validation_grid()
        builder = DiagnosticMapBuilder(self.model, self.benchmark, self.device, self.steady)
        maps = builder.build(coords, mode=self.config.get("diagnostics", {}).get("mode", "full_reference"))
        scores, names = self.patch_scorer.compute(maps, coords, update_ema=True)
        weak_regions = self.weak_detector.detect(scores, names, self.patch_grid)
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
            xy_f = self._sample_cavity_focused_interior(
                n_f,
                lambda count: self.adaptive_sampler.sample_interior(count, maps, coords, weak_regions, priorities),
            )
            xy_data = self._sample_cavity_focused_interior(
                n_data,
                lambda count: self.adaptive_sampler.sample_interior(count, maps, coords, weak_regions, priorities),
            )
        else:
            xy_f = self._sample_interior(n_f)
            xy_data = self._sample_interior(n_data)
        xy_bc = self._sample_boundary(n_bc)
        return self.make_batch(xy_f, xy_bc, xy_data)

    def evaluate_and_save_final(self) -> dict[str, float]:
        X, Y, coords = self.test_grid()
        metrics = evaluate_on_grid(self.model, self.benchmark, coords, self.device, self.steady)
        metrics["final_total_loss"] = float(self.last_losses.get("total", float("nan")))
        metrics["optimizer_stage"] = self.optimizer_stage
        for key, value in self.final_repair_status.items():
            metrics[f"final_repair_{key}"] = value
        metrics["reference_kind"] = getattr(self.benchmark, "reference_kind", "analytical")
        metrics["has_reference"] = bool(getattr(self.benchmark, "has_reference", True))
        metrics["run_type"] = str(self.config.get("run_type", "full"))
        metrics["reportable"] = metrics["run_type"] != "smoke"
        metrics["collapse_evaluated"] = bool(metrics["reportable"])
        metrics["collapsed"] = self._collapsed(metrics)
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
        builder = DiagnosticMapBuilder(self.model, self.benchmark, self.device, self.steady)
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
            save_field_panel(
                X,
                Y,
                {
                    "u ref": maps["u_ref"].reshape(shape),
                    "v ref": maps["v_ref"].reshape(shape),
                    "p ref centered": maps["p_ref"].reshape(shape),
                    "omega ref": maps["omega_ref"].reshape(shape),
                },
                self.figure_dir / "reference_fields.png",
            )
        for name in ["u_error", "v_error", "p_error_mean_centered", "omega_error", "pde_residual"]:
            save_heatmap(maps[name].reshape(shape), X, Y, self.figure_dir / f"{name}.png", name)
        save_streamlines(
            X,
            Y,
            maps["u_pred"].reshape(shape),
            maps["v_pred"].reshape(shape),
            self.figure_dir / "streamlines.png",
        )

    def maybe_checkpoint(self, cycle: int, metrics: dict[str, float]) -> None:
        score = (
            metrics.get("u_rel_l2", 0.0)
            + metrics.get("v_rel_l2", 0.0)
            + metrics.get("p_rel_l2_centered", 0.0)
            + metrics.get("omega_rel_l2", 0.0)
        )
        save_checkpoint(self.checkpoint_dir / "latest.pt", self.model, self.optimizer, self.config, metrics, self.global_step, cycle)
        if score < self.best_score:
            self.best_score = score
            save_checkpoint(self.checkpoint_dir / "best.pt", self.model, self.optimizer, self.config, metrics, self.global_step, cycle)

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
