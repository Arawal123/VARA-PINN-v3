"""Opt-in protocol enforcement and evidence for the cavity revision.

This module changes selection/budget policy, never the sparse curriculum's
losses, update rules, or sampling coefficients.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
from typing import Any

import torch

from src.utils.config import deep_update


PRIMARY_STEPS = 4000
HISTORICAL_COMMIT = "f14e1d21039e2bc3938830e255eb291f4fea7401"


def fair_revision_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("fair_revision", {}).get("enabled", False))


def apply_fair_revision(config: dict[str, Any]) -> dict[str, Any]:
    """Apply *after* Re/formulation materialization so repair stays disabled."""
    return deep_update(deepcopy(config), {
        "fair_revision": {
            "enabled": True,
            "primary_steps": PRIMARY_STEPS,
            "final_model_source": "final_primary_adam_step",
            "historical_commit": HISTORICAL_COMMIT,
            "continuation_protocol": "disabled_pending_manuscript_confirmation",
        },
        "compute_budget": {
            "enabled": True, "type": "applied_optimizer_steps", "value": PRIMARY_STEPS,
        },
        "convergence_early_stopping": {"enabled": False},
        "optimizer": {"final_repair": {"enabled": False, "epochs": 0}},
        "data_supervision": {"polish": {"final_repair_steps": 0}},
        "checkpoint": {"restore_best_before_final": False},
        "controller_v2": {
            "counterfactual_probe_enabled": False,
            "sparse_polish_rescue": {"enabled": False},
            "sparse_polish_curriculum": {
                "enabled": True,
                "disable_generic_interventions": True,
                "restore_best_before_final": False,
            },
        },
        "evaluation": {
            "controller_reference_metrics_enabled": False,
            "checkpoint_reference_metrics_enabled": False,
            "controller_streamfunction_metrics": False,
        },
        "diagnostics": {"mode": "residual_only"},
        "continuation_validity": {"enabled": False},
    })


def prepare_fair_revision_args(args: Any) -> None:
    """Reject unresolved continuation and identity-changing CLI options."""
    if list(args.reynolds) != [100.0]:
        raise ValueError(
            "Fair revision currently supports --reynolds 100 only. Continuation "
            "is disabled pending manuscript protocol confirmation; intermediate "
            "Re stages have no CFD references in the supplied map."
        )
    if len(args.methods) != 2 or set(args.methods) != {"vanilla", "vara_v2"}:
        raise ValueError("Fair revision requires --methods vanilla vara_v2.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Fair revision requires distinct matched training seeds.")
    if args.quick or args.enhanced_backbone or args.disable_stabilizers:
        raise ValueError("Fair revision preserves the historical reliable sparse-polish model.")
    if args.preset not in {None, "reliable"}:
        raise ValueError("Fair revision requires the 4000-step reliable preset.")
    if args.data_supervision != "sparse_cfd_polish":
        raise ValueError("Fair revision requires --data_supervision sparse_cfd_polish.")
    if args.cavity_base_formulation not in {None, "uvp_soft_bc"}:
        raise ValueError("Fair revision requires the historical sparse UVP soft-BC formulation.")
    if args.cfd_sample_fraction != 0.01 or args.cfd_sample_count is not None:
        raise ValueError("Fair revision preserves 1% eligible sparse CFD supervision.")
    if args.cfd_include_pressure or args.cfd_include_vorticity:
        raise ValueError("Fair revision preserves velocity-only sparse labels.")
    args.reliable = True
    args.preset = "reliable"
    args.methods = ["vanilla", "vara_v2"]
    args.gate_vara_on_vanilla = False


def tensor_mapping_sha256(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def pool_evidence(pool: Any) -> dict[str, Any]:
    if pool is None:
        raise ValueError("Fair revision requires a frozen sparse CFD pool.")
    source = Path(pool.source_path).resolve()
    return {
        "sparse_cfd_sample_count": pool.sample_count,
        "sparse_cfd_sample_fraction": pool.sample_fraction,
        "sparse_pool_seed": pool.seed,
        "sparse_pool_hash": pool.pool_hash,
        "source_reference_path": str(source),
        "source_cfd_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        # The historical hash includes indices/path, but not label bytes.
        "sparse_pool_content_sha256": tensor_mapping_sha256({
            "coords": pool.coords, **{f"target_{k}": v for k, v in pool.targets.items()},
        }),
    }


def assert_fair_final_state(trainer: Any) -> None:
    steps = PRIMARY_STEPS
    if int(trainer.config["fair_revision"]["primary_steps"]) != steps:
        raise RuntimeError("Fair revision primary step target cannot be changed.")
    tracker = trainer.compute_tracker
    if (tracker.applied_optimizer_steps != steps or tracker.optimizer_steps != steps
            or trainer.global_step != steps):
        raise RuntimeError("Fair revision must evaluate exactly the final 4000-step Adam state.")
    if tracker.probe_optimizer_steps or tracker.rollback_optimizer_steps:
        raise RuntimeError("Fair sparse curriculum must not perform hidden optimizer probes.")
    if not isinstance(trainer.optimizer, torch.optim.Adam) or trainer.optimizer_stage != "adam":
        raise RuntimeError("Fair revision permits only the primary Adam optimizer.")
    if trainer.early_stopped or trainer.final_repair_status.get("accepted", False):
        raise RuntimeError("Fair revision cannot stop early or accept a repair.")
    if getattr(trainer, "_sparse_curriculum_restored_best", False):
        raise RuntimeError("Fair revision cannot restore a curriculum checkpoint.")
    cfg = trainer.config
    if (cfg["convergence_early_stopping"]["enabled"]
            or cfg["optimizer"]["final_repair"]["enabled"]
            or cfg["checkpoint"]["restore_best_before_final"]
            or cfg["continuation_validity"]["enabled"]
            or cfg["evaluation"]["controller_reference_metrics_enabled"]
            or cfg["evaluation"]["checkpoint_reference_metrics_enabled"]):
        raise RuntimeError("Fair revision policy was changed after resolution.")


def fairness_manifest(trainer: Any, method: str) -> dict[str, Any]:
    assert_fair_final_state(trainer)
    cfg = trainer.config
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    schedule = cfg["training"]["collocation_curriculum"]["stages"]
    return {
        "protocol": "rie_cavity_fair_re100_v1",
        "git_commit": commit,
        "historical_commit": HISTORICAL_COMMIT,
        "method": method,
        "seed": trainer.seed,
        "reynolds": float(cfg["benchmark_params"]["reynolds"]),
        "model_architecture": deepcopy(cfg["model"]),
        "trainable_parameter_count": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        "initial_model_sha256": trainer.fair_initial_model_sha256,
        "final_model_sha256": tensor_mapping_sha256(trainer.model.state_dict()),
        "optimizer": "Adam",
        "learning_rate": trainer.base_learning_rate,
        "final_learning_rate": trainer.optimizer.param_groups[0]["lr"],
        "learning_rate_schedule": deepcopy(cfg["optimizer"].get("scheduler", {})),
        **trainer.compute_tracker.summary(),
        "collocation_schedule": [{"until_step": s["until_step"], "count": s["n_collocation"]} for s in schedule],
        "boundary_schedule": [{"until_step": s["until_step"], "count": s["n_boundary"]} for s in schedule],
        "validation_grid": deepcopy(cfg["validation"]),
        "evaluation_grid": deepcopy(cfg["test"]),
        "sparse_sampling_mode": cfg["data_supervision"]["cfd"]["sampling"]["mode"],
        "evaluation_reference": cfg["benchmark_params"].get("reference", "none"),
        **pool_evidence(trainer.cfd_supervision),
        "early_stopping_enabled": False,
        "final_repair_enabled": False,
        "final_repair_executed": bool(trainer.final_repair_status.get("executed", False)),
        "checkpoint_restoration_enabled": False,
        "checkpoint_restoration_executed": False,
        "topology_continuation_gate_enabled": False,
        "final_model_source": "final_primary_adam_step",
    }


def paired_fairness_report(vanilla: dict[str, Any], vara: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "seed", "reynolds", "git_commit", "model_architecture", "trainable_parameter_count",
        "initial_model_sha256", "optimizer", "learning_rate", "learning_rate_schedule",
        "applied_optimizer_steps", "optimizer_steps", "probe_optimizer_steps",
        "rollback_optimizer_steps", "collocation_schedule", "boundary_schedule",
        "collocation_evaluations", "boundary_evaluations", "data_evaluations",
        "validation_grid", "evaluation_grid", "evaluation_reference", "sparse_sampling_mode",
        "sparse_cfd_sample_count", "sparse_cfd_sample_fraction",
        "sparse_pool_seed", "sparse_pool_hash", "source_reference_path", "source_cfd_sha256",
        "sparse_pool_content_sha256", "early_stopping_enabled", "final_repair_enabled",
        "final_repair_executed", "checkpoint_restoration_enabled",
        "checkpoint_restoration_executed", "topology_continuation_gate_enabled", "final_model_source",
    )
    mismatches = [name for name in fields if vanilla[name] != vara[name]]
    return {
        "seed": vanilla["seed"], "reynolds": vanilla["reynolds"],
        "matched": not mismatches, "mismatches": mismatches,
        "pool_hash_vanilla": vanilla["sparse_pool_hash"],
        "pool_hash_vara": vara["sparse_pool_hash"],
        "source_cfd_sha256": vanilla["source_cfd_sha256"],
    }
