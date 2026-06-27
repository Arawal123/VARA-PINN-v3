"""Controller-free baseline for the isolated repaired Taylor--Green setup."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any

import pandas as pd

from src.training.taylor_green_vara_v2_trainer import TaylorGreenVARAV2Trainer
from src.utils.io import save_json
from src.utils.logging import make_run_id


class TaylorGreenVanillaTrainer(TaylorGreenVARAV2Trainer):
    """Neutral 10k baseline matched to the repaired VARA V2 experiment.

    The model, hard initial condition, losses, optimizer, sample counts,
    neutral resampling cadence, and final metrics are inherited from the
    repaired Taylor--Green trainer.  This class intentionally performs no
    controller proposal, probe, adaptive allocation, or rollback.  The cheap
    fixed-grid monitoring calls are retained so checkpoints and reporting use
    the same numerical path as the repaired VARA run.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        resolved = deepcopy(config)
        total_steps = int(
            resolved.get("controller_v2", {}).get("total_steps", 10000)
        )
        controller = resolved.setdefault("controller_v2", {})
        resolved.setdefault(
            "taylor_green_comparison",
            {
                "warmup_steps": int(controller.get("warmup_steps", 1000)),
                "block_steps": int(controller.get("block_steps", 500)),
            },
        )
        controller["warmup_steps"] = total_steps
        controller["control_blocks"] = 0
        controller["total_steps"] = total_steps
        resolved["method"] = "vanilla"
        experiments = resolved.setdefault("experiments", {})
        experiments.setdefault(
            "run_id",
            make_run_id(
                str(resolved.get("benchmark", "taylor_green")),
                "repaired_vanilla",
                int(resolved.get("seed", 0)),
            ),
        )
        super().__init__(resolved)

    def run(self) -> dict[str, Any]:
        run_started = time.perf_counter()
        metrics = self._run_neutral_schedule()
        return self._finalize_taylor_green_metrics(metrics, run_started)

    def _run_neutral_schedule(self) -> dict[str, Any]:
        """Apply exactly the comparison schedule with every allocation neutral."""
        self.compute_tracker.start()
        cfg = dict(self.config.get("controller_v2", {}))
        total_steps = int(cfg.get("total_steps", 10000))
        original_cfg = dict(self.config.get("taylor_green_comparison", {}))
        original_warmup = int(original_cfg.get("warmup_steps", 1000))
        block_steps = int(original_cfg.get("block_steps", 500))
        if original_warmup < 0 or original_warmup > total_steps:
            raise ValueError("Taylor-Green vanilla warmup must be within total_steps.")
        if block_steps <= 0:
            raise ValueError("Taylor-Green vanilla block_steps must be positive.")

        batch = self.initial_batch()
        neutral_cycle_steps = max(
            1,
            int(self.config.get("training", {}).get("epochs_per_cycle", 100)),
        )
        remaining_warmup = original_warmup
        cycle = 0
        while remaining_warmup > 0:
            chunk = min(neutral_cycle_steps, remaining_warmup)
            self._train_v2_steps(
                batch,
                chunk,
                cycle=cycle,
                phase="taylor_green_vanilla_warmup",
            )
            remaining_warmup -= chunk
            _, _, coords = self.validation_grid()
            self.maybe_checkpoint(cycle, self.controller_metrics(coords))
            cycle += 1
            if remaining_warmup > 0:
                batch = self._resample_v2_batch({}, coords)

        remaining = total_steps - original_warmup
        while remaining > 0:
            _, _, coords = self.validation_grid()
            batch = self._resample_v2_batch({}, coords)
            chunk = min(block_steps, remaining)
            self._train_v2_steps(
                batch,
                chunk,
                cycle=cycle,
                phase="taylor_green_vanilla_neutral_block",
            )
            remaining -= chunk
            _, _, coords = self.validation_grid()
            self.maybe_checkpoint(cycle, self.controller_metrics(coords))
            cycle += 1

        metrics = self.evaluate_and_save_final()
        metrics.update(
            {
                "method": "vanilla",
                "training_mode": "taylor_green_repaired_vanilla",
                "controller_enabled": False,
                "accepted_interventions": 0,
                "rejected_interventions": 0,
                "prefiltered_interventions": 0,
                "rollback_count": 0,
                "taylor_green_neutral_sampling_only": True,
            }
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
