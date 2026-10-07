"""Read-only audit/analysis of completed Allen--Cahn and advection--diffusion runs.

No trainer is constructed and no optimizer is created. All outputs are written
to a fresh supplement directory outside the input results tree.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pde_generalization.benchmarks import build_benchmark
from src.pde_generalization.models import build_pde_model, model_parameter_hash
from src.pde_generalization.metrics import primary_metric_name
from src.pde_generalization.trainer import _tensor_hash
from src.utils.config import load_config, deep_update
from src.utils.seed import set_seed

BENCHMARKS = ("allen_cahn", "advection_diffusion")
METHODS = ("vanilla", "vara_v2")
SEEDS = tuple(range(5))
GUARD_FLAGS = (
    "counterfactual_probe_enabled", "gradient_prefilter_enabled",
    "trust_region_enabled", "action_memory_enabled", "rollback_enabled",
    "variable_awareness_enabled", "sampling_redistribution_enabled",
    "local_loss_multipliers_enabled",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(root):
    paths = sorted(Path(root).rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Input results contain a symlink; audit its target first.")
    return {p.relative_to(root).as_posix(): sha256(p) for p in paths if p.is_file()}


def resolve_reliable(benchmark, device="cpu"):
    """Mirror the existing CLI's reliable merge and runtime count resolution."""
    cfg = deep_update(
        load_config(ROOT / f"configs/pde_generalization/{benchmark}.yaml"),
        load_config(ROOT / "configs/pde_generalization/presets/reliable.yaml"),
    )
    grid = cfg["evaluation"]
    fraction = float(cfg["benchmark_params"]["sparse_sample_fraction"])
    cfg["training"]["n_sparse_data"] = max(1, round(grid["nx"] * grid["ny"] * grid["nt"] * fraction))
    cfg["device"] = device
    return cfg


def frozen_data(cfg, device):
    """Reconstruct the existing sampler exactly, not a new data-selection rule."""
    benchmark = build_benchmark(cfg)
    seed = int(cfg.get("seed", 0)) + 30003 + int(cfg["benchmark_params"].get("sparse_seed", 0))
    rng = np.random.default_rng(seed)
    n = int(cfg["training"]["n_sparse_data"])
    values = np.empty((n, 3), dtype=np.float64)
    x0, x1, y0, y1 = benchmark.bounds
    t0, t1 = benchmark.t_bounds
    values[:, 0] = rng.uniform(x0, x1, n)
    values[:, 1] = rng.uniform(y0, y1, n)
    values[:, 2] = rng.uniform(t0, t1, n)
    dtype = {"float32": torch.float32, "float64": torch.float64}[cfg.get("dtype", "float32")]
    coords = torch.as_tensor(values, dtype=dtype, device=device)
    with torch.no_grad():
        targets = benchmark.exact(coords)
    return coords, targets


def protocol(benchmark, device, seeds):
    cfg = resolve_reliable(benchmark, device)
    set_seed(0)
    model = build_pde_model(cfg)
    pools = []
    for seed in seeds:
        run_cfg = deepcopy(cfg)
        run_cfg["seed"] = seed
        coords, targets = frozen_data(run_cfg, device)
        set_seed(seed)
        initial = build_pde_model(run_cfg).to(device=device)
        pools.append({
            "seed": seed, "sparse_rng_seed": seed + 30003 + cfg["benchmark_params"].get("sparse_seed", 0),
            "coordinates_hash": _tensor_hash(coords), "targets_hash": _tensor_hash(targets),
            "combined_sparse_hash": _tensor_hash(coords, targets),
            "initial_model_hash": model_parameter_hash(initial),
        })
    return {
        "benchmark": benchmark, "configuration": cfg, "paired_seeds": list(seeds),
        "methods": list(METHODS), "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "sparse_count": cfg["training"]["n_sparse_data"],
        "fraction_interpretation": "uniform continuous samples; count is 2% of the 48x48x11 grid size, not a grid subset",
        "final_state": "final committed Adam state; no repair or restoration",
        "fairness_description": "matched committed training budget with explicit controller-overhead accounting",
        "paired_initialization_and_pools": pools,
        "expected_optimizer_calls": "Vanilla 4000; VARA 4000 + 25 per active counterfactual block, at most 4175",
        "scheduler": "none; constant Adam learning rate",
    }


def load_decisions(path):
    frame = pd.read_csv(path)
    for col in ("accepted", "prefiltered"):
        if col not in frame:
            frame[col] = False
        frame[col] = frame[col].fillna(False).astype(str).str.lower().isin(["true", "1", "1.0"])
    return frame


def audit_run(path, benchmark, method, seed, strict=True):
    path = Path(path)
    required = ["summary.json", "resolved_config.yaml", "metrics.csv", "losses.csv", "checkpoints/final.pt"]
    if method == "vara_v2":
        required += ["vara_v2_decisions.csv", "vara_v2_allocation_history.json"]
    for name in required:
        if not (path / name).is_file():
            raise ValueError(f"Missing required artifact: {path / name}")
    summary = read_json(path / "summary.json")
    cfg = load_config(path / "resolved_config.yaml")
    metrics = summary["metrics"]
    checkpoint = torch.load(path / "checkpoints/final.pt", map_location="cpu", weights_only=True)
    if (summary["benchmark"], summary["method"], summary["seed"]) != (benchmark, method, seed):
        raise ValueError(f"Run identity mismatch: {path}")
    if checkpoint["metrics"] != metrics:
        raise ValueError(f"Checkpoint/summary metrics differ: {path}")
    steps = cfg["controller_v2"]["total_steps"]
    if metrics["applied_optimizer_steps"] != steps:
        raise ValueError(f"Committed-step mismatch: {path}")
    loss = pd.read_csv(path / "losses.csv")
    if loss["step"].tolist() != list(range(1, steps + 1)):
        raise ValueError(f"Incomplete/duplicate committed loss rows: {path}")
    adam_steps = {int(state["step"]) for state in checkpoint["optimizer_state_dict"]["state"].values()}
    if adam_steps != {steps}:
        raise ValueError(f"Final checkpoint is not the final committed Adam state: {path}")
    if summary["controller_reference_metrics_enabled"] is not False:
        raise ValueError(f"Reference-access flag invalid: {path}")
    if strict:
        expected = resolve_reliable(benchmark, cfg["device"])
        expected["seed"] = seed
        if cfg != expected:
            raise ValueError(f"Resolved configuration differs from unchanged reliable protocol: {path}")
        if steps != 4000 or cfg["training"]["n_sparse_data"] != 507:
            raise ValueError(f"Reliable step/data count mismatch: {path}")
        if not all(cfg["controller_v2"][flag] for flag in GUARD_FLAGS):
            raise ValueError(f"A full-guard component is disabled: {path}")
        if cfg["training"]["weights"]["sparse_data"] != 2.0:
            raise ValueError(f"Sparse supervision is not the reliable objective: {path}")
    device = torch.device(summary["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Reconstruct frozen targets on the original CUDA runtime; exact target hashes must match.")
    coords, targets = frozen_data(cfg, device)
    if _tensor_hash(coords, targets) != summary["sparse_sample_hash"]:
        raise ValueError(f"Reconstructed frozen-data hash differs: {path}")
    set_seed(seed)
    model = build_pde_model(cfg).to(device=device, dtype=coords.dtype)
    initial_hash = model_parameter_hash(model)
    if initial_hash != summary["initial_model_parameter_hash"]:
        raise ValueError(f"Initial model cannot be reproduced: {path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    final_hash = model_parameter_hash(model)
    if checkpoint["sparse_sample_hash"] != summary["sparse_sample_hash"]:
        raise ValueError(f"Checkpoint data identity differs: {path}")
    decisions = load_decisions(path / "vara_v2_decisions.csv") if method == "vara_v2" else pd.DataFrame()
    active = decisions[~decisions.prefiltered] if len(decisions) else pd.DataFrame()
    blocks = len(active)
    probe = cfg["controller_v2"]["probe_steps"]
    calls = metrics["optimizer_step_calls"]
    if calls != steps + blocks * probe:
        raise ValueError(f"Auxiliary/committed optimizer accounting mismatch: {path}")
    if len(active) and not active["comparison_mode"].eq("counterfactual").all():
        raise ValueError(f"Unmatched probe comparison found: {path}")
    if method == "vara_v2":
        if metrics["accepted_interventions"] != int(decisions.accepted.sum()):
            raise ValueError(f"Acceptance count differs from decisions: {path}")
        if metrics["prefiltered_interventions"] != int(decisions.prefiltered.sum()):
            raise ValueError(f"Prefilter count differs from decisions: {path}")
        if metrics["rejected_interventions"] != int((~active.accepted).sum()):
            raise ValueError(f"Rejection count differs from decisions: {path}")
        if metrics["rollback_count"] != metrics["rejected_interventions"]:
            raise ValueError(f"Rollback/rejection accounting differs: {path}")
    gradient_objectives = metrics["objective_evaluation_count"] - calls
    if gradient_objectives < 0:
        raise ValueError(f"Objective accounting invalid: {path}")
    train, diag = cfg["training"], cfg["diagnostics"]
    manifest = {
        "benchmark": benchmark, "method": method, "seed": seed, "git_commit": summary["git_commit"],
        "initial_model_hash": initial_hash, "final_model_hash": final_hash,
        "sparse_coordinates_hash": _tensor_hash(coords), "sparse_targets_hash": _tensor_hash(targets),
        "sparse_sample_hash": summary["sparse_sample_hash"], "sparse_count": len(coords),
        "sparse_rng_seed": seed + 30003 + cfg["benchmark_params"].get("sparse_seed", 0),
        "architecture": cfg["model"], "parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer": "Adam", "learning_rate": train["lr"], "scheduler": "none",
        "dtype": summary["dtype"], "device": summary["device"],
        "evaluation_grid": cfg["evaluation"], "committed_steps": steps,
        "total_optimizer_calls": calls, "neutral_probe_calls": blocks * probe,
        "action_probe_calls": blocks * probe, "total_probe_calls": 2 * blocks * probe,
        "committed_probe_steps": blocks * probe, "extra_discarded_probe_calls": calls - steps,
        "controller_gradient_objectives": gradient_objectives,
        "controller_gradient_evaluations": 2 * gradient_objectives,
        "objective_evaluations": metrics["objective_evaluation_count"],
        "diagnostic_evaluations": metrics["diagnostic_evaluation_count"],
        "loss_collocation_point_evaluations": calls * train["n_collocation"] + gradient_objectives * diag["n_interior"],
        "loss_boundary_point_evaluations": calls * train["n_boundary"] + gradient_objectives * diag["n_boundary"],
        "loss_initial_point_evaluations": calls * train["n_initial"] + gradient_objectives * diag["n_initial"],
        "loss_sparse_point_evaluations": metrics["objective_evaluation_count"] * len(coords),
        "training_and_controller_wall_clock_sec": metrics["optimization_wall_clock_sec"],
        "final_state_rule": "final committed Adam state", "checkpoint_restoration": False,
        "final_repair": False, "convergence_early_stopping": False,
        "reference_error_controller_access": False,
        "evidence_source": f"raw/seed_{seed}/{method}/summary.json",
        "compute_counter_status": "probe/point counts derived from audited native counters and decision phases",
    }
    return {"path": path, "summary": summary, "cfg": cfg, "manifest": manifest,
            "decisions": decisions, "coordinates": coords.detach().cpu().numpy(),
            "targets": targets.detach().cpu().numpy(), "model": model.cpu()}


def audit_pairs(results, benchmark, strict=True):
    runs, rows = {}, []
    for seed in SEEDS:
        for method in METHODS:
            runs[(seed, method)] = audit_run(Path(results) / f"seed_{seed}" / method, benchmark, method, seed, strict)
        left, right = runs[(seed, "vanilla")], runs[(seed, "vara_v2")]
        shared = ("git_commit", "initial_model_hash", "sparse_coordinates_hash", "sparse_targets_hash",
                  "sparse_sample_hash", "sparse_count", "architecture", "learning_rate", "dtype",
                  "device", "evaluation_grid", "committed_steps", "final_state_rule")
        mismatch = [field for field in shared if left["manifest"][field] != right["manifest"][field]]
        if left["cfg"] != right["cfg"]:
            mismatch.append("resolved_config")
        if mismatch:
            raise ValueError(f"Paired seed {seed} failed: {mismatch}")
        rows.append({"seed": seed, "matched": True, "mismatches": [],
                     "initial_model_hash": left["manifest"]["initial_model_hash"],
                     "sparse_coordinates_hash": left["manifest"]["sparse_coordinates_hash"],
                     "sparse_targets_hash": left["manifest"]["sparse_targets_hash"]})
    return runs, pd.DataFrame(rows)


def table(frame, folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / f"{name}.csv", index=False, float_format="%.17g")
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return format(value, ".17g") if np.isfinite(value) else "NA"
        return str(value).replace("\n", " ")
    header = [str(c) for c in frame.columns]
    rows = [[fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    markdown = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    markdown += ["| " + " | ".join(v.replace("|", "\\|") for v in row) + " |" for row in rows]
    (folder / f"{name}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    def tex(value):
        escapes = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
        return "".join(escapes.get(c, c) for c in value)
    latex = [r"\begin{longtable}{" + "l" * len(header) + "}", r"\hline",
             " & ".join(tex(v) for v in header) + r" \\", r"\hline", r"\endhead"]
    latex += [" & ".join(tex(v) for v in row) + r" \\" for row in rows]
    latex += [r"\hline", r"\end{longtable}"]
    (folder / f"{name}.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def exact_statistics(vanilla, vara):
    difference = np.asarray(vanilla) - np.asarray(vara)
    percent = np.full(len(difference), np.nan)
    np.divide(100 * difference, np.abs(vanilla), out=percent, where=np.asarray(vanilla) != 0)
    indices = np.asarray(list(itertools.product(range(len(difference)), repeat=len(difference))))
    absolute_ci = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
    percent_ci = np.quantile(percent[indices].mean(axis=1), [0.025, 0.975]) if np.isfinite(percent).all() else [np.nan, np.nan]
    nonzero = difference[difference != 0]
    if len(nonzero):
        ranks = pd.Series(abs(nonzero)).rank(method="average").to_numpy()
        plus, minus = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
        signed = np.asarray(list(itertools.product([-1, 1], repeat=len(nonzero)))) @ ranks
        p = float(np.mean(abs(signed) >= abs(plus - minus) - 1e-12))
        statistic, rank_biserial = min(plus, minus), (plus - minus) / ranks.sum()
    else:
        p, statistic, rank_biserial = 1.0, 0.0, 0.0
    std = difference.std(ddof=1)
    return percent, {
        "mean_paired_difference": difference.mean(), "paired_difference_sd": std,
        "mean_paired_improvement_percent": percent.mean() if np.isfinite(percent).all() else np.nan,
        "difference_ci95_low": absolute_ci[0], "difference_ci95_high": absolute_ci[1],
        "improvement_ci95_low": percent_ci[0], "improvement_ci95_high": percent_ci[1],
        "wins_of_5": int((difference > 0).sum()), "losses_of_5": int((difference < 0).sum()),
        "ties_of_5": int((difference == 0).sum()), "cohen_dz": difference.mean() / std if std > 0 else np.nan,
        "rank_biserial": rank_biserial, "exact_wilcoxon_T": statistic,
        "exact_wilcoxon_p": p, "nonzero_pairs": len(nonzero),
        "inference": "low-n descriptive; exact conditional sign enumeration with average tied ranks",
    }


def result_tables(runs, package):
    metrics = [name for name in runs[(0, "vanilla")]["summary"]["metrics"]
               if name.startswith(("allen_cahn_", "advdiff_"))]
    metrics.append("optimization_wall_clock_sec")
    paired, aggregate, traceability, all_raw = [], [], [], []
    for (seed, method), item in runs.items():
        for name, value in item["summary"]["metrics"].items():
            all_raw.append({"seed": seed, "method": method, "metric": name, "value": value,
                            "raw_source": f"raw/seed_{seed}/{method}/summary.json", "json_key": f"metrics.{name}"})
    for metric in metrics:
        v = np.array([runs[(s, "vanilla")]["summary"]["metrics"][metric] for s in SEEDS], dtype=float)
        w = np.array([runs[(s, "vara_v2")]["summary"]["metrics"][metric] for s in SEEDS], dtype=float)
        if not np.isfinite(v).all() or not np.isfinite(w).all():
            raise ValueError(f"Cannot silently discard nonfinite manuscript metric: {metric}")
        percent, statistics = exact_statistics(v, w)
        for i, seed in enumerate(SEEDS):
            paired.append({"metric": metric, "seed": seed, "vanilla": v[i], "vara_v2": w[i],
                           "difference_vanilla_minus_vara": v[i] - w[i], "paired_improvement_percent": percent[i]})
            traceability.append({"metric": metric, "seed": seed,
                                 "vanilla_source": f"raw/seed_{seed}/vanilla/summary.json",
                                 "vara_source": f"raw/seed_{seed}/vara_v2/summary.json", "json_key": f"metrics.{metric}"})
        aggregate.append({"metric": metric, "n_pairs": 5,
                          "vanilla_mean": v.mean(), "vanilla_sd": v.std(ddof=1), "vanilla_median": np.median(v),
                          "vara_mean": w.mean(), "vara_sd": w.std(ddof=1), "vara_median": np.median(w), **statistics})
    paired, aggregate = pd.DataFrame(paired), pd.DataFrame(aggregate)
    pvalues = aggregate.exact_wilcoxon_p.to_numpy()
    order, adjusted, running = np.argsort(pvalues), np.empty(len(pvalues)), 0.0
    for i, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - i) * pvalues[index]))
        adjusted[index] = running
    aggregate["holm_adjusted_p"] = adjusted
    table(pd.DataFrame(all_raw), package / "tables", "all_seed_raw_values")
    table(paired, package / "tables", "per_seed_paired_results")
    table(aggregate, package / "tables", "main_aggregate_results")
    table(aggregate, package / "statistics", "paired_statistics")
    table(paired[paired.difference_vanilla_minus_vara < 0], package / "tables", "negative_seed_results")
    table(aggregate[aggregate.losses_of_5 > 0], package / "tables", "negative_tradeoff_metrics")
    table(pd.DataFrame(traceability), package / "provenance", "metric_traceability")
    return paired, aggregate


def guard_evidence(runs, package):
    frames = []
    for seed in SEEDS:
        item = runs[(seed, "vara_v2")]
        frame = item["decisions"].copy()
        if frame.empty:
            continue
        memory = {}
        for index, row in frame.iterrows():
            key = f"{row.variable}|{row.patch_id}|{row.action_type}"
            frame.loc[index, "effectiveness_before_derived"] = memory.get(key, 1.0)
            after = row.get("effectiveness_after", np.nan)
            if np.isfinite(after):
                memory[key] = float(after)
        frame["seed"] = seed
        frame["raw_source"] = f"raw/seed_{seed}/vara_v2/vara_v2_decisions.csv"
        changes = [name for name in frame if name.startswith("guard_changes_")]
        violation = pd.DataFrame({name: frame[name] - frame.get(name.replace("guard_changes_", "guard_noise_"), 0.02)
                                  for name in changes})
        frame["worst_guard_change"] = frame[changes].max(axis=1) if changes else np.nan
        frame["largest_guard_violation"] = violation.max(axis=1) if changes else np.nan
        observed = frame.get("observed_target_improvement", pd.Series(np.nan, index=frame.index))
        frame["target_threshold_passed"] = observed > frame.get("target_noise", 0.005)
        frames.append(frame)
    decisions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    table(decisions, package / "guard_evidence", "all_decisions")
    if not decisions.empty:
        harmful = decisions[(~decisions.accepted) & (~decisions.prefiltered)
                            & (decisions.observed_target_improvement > 0)
                            & (decisions.largest_guard_violation > 0)].copy()
        harmful = harmful.sort_values(["largest_guard_violation", "seed", "block"], ascending=[False, True, True])
    else:
        harmful = pd.DataFrame()
    table(harmful, package / "guard_evidence", "prevented_harm_candidates")
    decisive = harmful[harmful.target_threshold_passed] if len(harmful) else pd.DataFrame()
    table(decisive, package / "guard_evidence", "guard_decisive_rejections")
    write_json(package / "guard_evidence/highlighted_harm.json", {
        "rule": "largest guard violation among rejected tested candidates with positive target improvement; ties seed/block",
        "event": harmful.iloc[0].to_dict() if len(harmful) else None,
        "status": "recorded event" if len(harmful) else "none recorded; no event fabricated",
        "guard_decisive_event_count": len(decisive),
        "interpretation": "positive improvement alone may be below the target margin; guard_decisive_rejections also pass that margin",
    })
    write_json(package / "guard_evidence/logging_limits.json", {
        "retained": "all native decision/allocation files; accepted, rejected, and prefiltered rows",
        "not_recorded_by_historical_trainer": [
            "all ranked but untested candidate rows", "absolute neutral/candidate target values",
            "discarded auxiliary per-step losses", "per-proposal full allocation snapshots",
        ],
        "memory_before": "derived from initial effectiveness=1 and prior logged effectiveness_after for the same action key",
        "no_imputed_evidence": True,
    })
    return decisions


def publication_figures(runs, paired, aggregate, decisions, package, benchmark):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 350})
    folder = package / "figures"
    data_dir = folder / "data"
    data_dir.mkdir()
    def save(fig, name):
        for extension in ("pdf", "svg", "png"):
            fig.savefig(folder / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)
    names = aggregate.metric.tolist()
    fig, axes = plt.subplots((len(names) + 1) // 2, 2, figsize=(10, 2.7 * ((len(names) + 1) // 2)), constrained_layout=True)
    for ax, name in zip(np.asarray(axes).flat, names):
        rows = paired[paired.metric == name].sort_values("seed")
        for row in rows.itertuples():
            ax.plot([0, 1], [row.vanilla, row.vara_v2], color="0.65", linewidth=1)
            ax.scatter([0, 1], [row.vanilla, row.vara_v2], c=["#37649e", "#c85d35"])
            ax.annotate(str(row.seed), (1, row.vara_v2), xytext=(4, 0), textcoords="offset points")
        ax.set_xticks([0, 1], ["Vanilla", "Full guarded VARA"])
        ax.set_title(name.replace("_", " "))
    for ax in list(np.asarray(axes).flat)[len(names):]:
        ax.axis("off")
    save(fig, "all_seed_pairs")
    fig, ax = plt.subplots(figsize=(10, 0.65 * len(names) + 2), constrained_layout=True)
    for i, row in enumerate(aggregate.itertuples()):
        if np.isfinite([row.mean_paired_improvement_percent, row.improvement_ci95_low, row.improvement_ci95_high]).all():
            ax.hlines(i, row.improvement_ci95_low, row.improvement_ci95_high, color="0.25")
            ax.scatter(row.mean_paired_improvement_percent, i, color="#c85d35")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired improvement (%); percentile bootstrap 95% CI")
    save(fig, "aggregate_ci")
    table(paired, data_dir, "seed_pair_inputs")
    table(aggregate, data_dir, "aggregate_ci_inputs")

    primary = primary_metric_name(benchmark)
    selected = paired[paired.metric == primary].copy()
    median = selected.paired_improvement_percent.median()
    selected["distance_to_median"] = abs(selected.paired_improvement_percent - median)
    representative = int(selected.sort_values(["distance_to_median", "seed"]).iloc[0].seed)
    write_json(package / "provenance/representative_seed.json", {
        "seed": representative, "rule": "closest to median paired full-field improvement; tie smallest seed",
        "all_five_candidates": selected.to_dict("records"),
    })
    cfg = runs[(representative, "vanilla")]["cfg"]
    equation = build_benchmark(cfg)
    nx, ny = cfg["evaluation"]["nx"], cfg["evaluation"]["ny"]
    x = torch.linspace(equation.bounds[0], equation.bounds[1], nx)
    y = torch.linspace(equation.bounds[2], equation.bounds[3], ny)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    models = {m: runs[(representative, m)]["model"].eval() for m in METHODS}
    times = np.linspace(*equation.t_bounds, 3)
    field_rows, panels = [], []
    for time in times:
        coords = torch.stack([xx, yy, torch.full_like(xx, float(time))], -1).reshape(-1, 3)
        with torch.no_grad():
            reference = equation.exact(coords)
            predictions = {m: models[m](coords) for m in METHODS}
            mask = equation.hard_region_mask(coords, reference)
        values = {"reference": reference[:, 0].numpy(), **{m: p[:, 0].numpy() for m, p in predictions.items()}}
        values.update({f"error_{m}": abs(values[m] - values["reference"]) for m in METHODS})
        frame = pd.DataFrame({"x": coords[:, 0].numpy(), "y": coords[:, 1].numpy(), "t": time,
                              "hard_region": mask.numpy(), **values})
        field_rows.append(frame)
        panels.append((time, values, mask.numpy().reshape(nx, ny)))
    fields = pd.concat(field_rows, ignore_index=True)
    fields.to_csv(data_dir / "representative_fields.csv", index=False, float_format="%.17g")
    write_json(data_dir / "field_traceability.json", {
        "seed": representative, "checkpoints": [f"raw/seed_{representative}/{m}/checkpoints/final.pt" for m in METHODS],
        "reference_definition": "source_snapshot/source.zip: src/pde_generalization/benchmarks.py",
        "forward_inference_only": True, "stored_manuscript_metrics_not_replaced": True,
    })
    vmin = min(values[n].min() for _, values, _ in panels for n in ("reference", *METHODS))
    vmax = max(values[n].max() for _, values, _ in panels for n in ("reference", *METHODS))
    error_max = max(values[f"error_{m}"].max() for _, values, _ in panels for m in METHODS)
    fig, axes = plt.subplots(3, 5, figsize=(15, 8), constrained_layout=True)
    for row, (time, values, _) in enumerate(panels):
        for col, name in enumerate(("reference", "vanilla", "vara_v2", "error_vanilla", "error_vara_v2")):
            error = name.startswith("error_")
            image = axes[row, col].pcolormesh(xx.numpy(), yy.numpy(), values[name].reshape(nx, ny),
                                             shading="auto", cmap="magma" if error else "viridis",
                                             vmin=0 if error else vmin, vmax=error_max if error else vmax, rasterized=True)
            axes[row, col].set(title=f"{name}, t={time:.2f}", xlabel="x", ylabel="y", aspect="equal")
            fig.colorbar(image, ax=axes[row, col], shrink=0.75)
    save(fig, "representative_field_reference_error")
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), constrained_layout=True)
    for col, (time, values, mask) in enumerate(panels):
        axes[0, col].pcolormesh(xx.numpy(), yy.numpy(), mask, shading="auto", cmap="Greys", rasterized=True)
        axes[0, col].set_title(f"Interface/layer mask, t={time:.2f}")
        for row, method in enumerate(METHODS, start=1):
            image = axes[row, col].pcolormesh(xx.numpy(), yy.numpy(),
                np.ma.masked_where(~mask, values[f"error_{method}"].reshape(nx, ny)),
                shading="auto", cmap="magma", vmin=0, vmax=error_max, rasterized=True)
            axes[row, col].set_title(f"{method}: localized error")
            fig.colorbar(image, ax=axes[row, col], shrink=0.75)
    save(fig, "interface_layer_region")

    manifest = pd.DataFrame([item["manifest"] for item in runs.values()])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method, color, offset in (("vanilla", "#37649e", -0.16), ("vara_v2", "#c85d35", 0.16)):
        data = manifest[manifest.method == method].sort_values("seed")
        axes[0].bar(data.seed + offset, data.total_optimizer_calls, width=0.30, color=color, label=method)
        axes[1].bar(data.seed + offset, data.training_and_controller_wall_clock_sec, width=0.30, color=color, label=method)
    axes[0].axhline(4000, color="black", linestyle=":", label="committed budget")
    axes[0].set(xlabel="Seed", ylabel="Total optimizer calls")
    axes[1].set(xlabel="Seed", ylabel="Training + controller seconds")
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, "compute_overhead")
    if len(decisions):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for seed in SEEDS:
            data = decisions[decisions.seed == seed]
            tested = data[~data.prefiltered]
            axes[0].scatter(data.block, data.seed, c=np.where(data.prefiltered, "#999999", np.where(data.accepted, "#37649e", "#c85d35")), s=30)
            if len(tested):
                axes[1].scatter(100 * tested.observed_target_improvement, 100 * tested.worst_guard_change, label=f"seed {seed}")
        axes[0].set(xlabel="Control block", ylabel="Seed", title="Gray prefiltered; blue accepted; orange rejected")
        axes[1].axhline(2, color="black", linestyle=":")
        axes[1].axvline(0.5, color="black", linestyle=":")
        axes[1].set(xlabel="Target improvement (%)", ylabel="Worst guard degradation (%)")
        axes[1].legend(fontsize=8)
        save(fig, "controller_decisions_and_guard_tradeoff")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for seed in SEEDS:
        history = read_json(runs[(seed, "vara_v2")]["path"] / "vara_v2_allocation_history.json")
        axes[0].plot([r["applied_optimizer_steps"] for r in history], [r["trust_radius"] for r in history], label=f"seed {seed}")
        if len(decisions) and "effectiveness_after" in decisions:
            data = decisions[(decisions.seed == seed) & (~decisions.prefiltered)]
            axes[1].plot(data.block, data.effectiveness_after, marker="o", label=f"seed {seed}")
    axes[0].set(xlabel="Committed steps", ylabel="Trust radius")
    axes[1].set(xlabel="Control block", ylabel="Selected action effectiveness after decision")
    axes[0].legend(fontsize=8)
    save(fig, "trust_radius_and_action_memory")
    return representative


def build_package(results, benchmark, destination):
    results, destination = Path(results).resolve(), Path(destination).resolve()
    if destination == results or results in destination.parents:
        raise ValueError("Supplement destination must be outside the original results tree.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite supplement: {destination}")
    zip_path = destination.parent / f"supplement_{benchmark}_fullguard_5seed.zip"
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite ZIP: {zip_path}")
    original = snapshot(results)
    runs, pairs = audit_pairs(results, benchmark)
    commits = {item["summary"]["git_commit"] for item in runs.values()}
    if len(commits) != 1:
        raise ValueError("Runs use different source commits.")
    commit = next(iter(commits))
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current != commit:
        raise ValueError("Use the recorded training checkout for source-consistent postprocessing.")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "configs", "scripts"], cwd=ROOT, check=True)
    for name in ("configs", "provenance", "fairness", "statistics", "guard_evidence", "compute", "tables", "figures", "checksums", "source_snapshot"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(results, destination / "raw")
    manifests = []
    for (seed, method), item in runs.items():
        manifest = item["manifest"]
        write_json(destination / f"fairness/seed_{seed}_{method}.json", manifest)
        shutil.copy2(item["path"] / "resolved_config.yaml", destination / f"configs/seed_{seed}_{method}.yaml")
        manifests.append(manifest)
        if method == "vanilla":
            np.savez(destination / f"fairness/frozen_data_seed_{seed}.npz", coordinates=item["coordinates"], targets=item["targets"])
    table(pairs, destination / "fairness", "paired_audit")
    table(pd.DataFrame(manifests), destination / "fairness", "all_run_manifests")
    compute_columns = ["seed", "method", "committed_steps", "total_optimizer_calls", "neutral_probe_calls", "action_probe_calls", "total_probe_calls", "extra_discarded_probe_calls", "objective_evaluations", "diagnostic_evaluations", "controller_gradient_objectives", "controller_gradient_evaluations", "loss_collocation_point_evaluations", "loss_boundary_point_evaluations", "loss_initial_point_evaluations", "loss_sparse_point_evaluations", "training_and_controller_wall_clock_sec"]
    table(pd.DataFrame(manifests)[compute_columns], destination / "compute", "compute_accounting")
    paired, aggregate = result_tables(runs, destination)
    decisions = guard_evidence(runs, destination)
    representative = publication_figures(runs, paired, aggregate, decisions, destination, benchmark)
    with (destination / "source_snapshot/source.zip").open("wb") as stream:
        subprocess.run([
            "git", "archive", "--format=zip", commit, "src",
            "configs/pde_generalization", "scripts/run_vara_v2_pde_generalization.py",
            "scripts/package_transient_fullguard.py", "requirements.txt", "pyproject.toml",
            "docs/TRANSIENT_FULLGUARD_REPRODUCTION.md",
        ], cwd=ROOT, stdout=stream, check=True)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    provenance = {
        "training_git_commit": commit, "checkout_branch_observed_after_run": branch,
        "package_time_utc": datetime.now(timezone.utc).isoformat(),
        "python_at_packaging": sys.version, "platform_at_packaging": platform.platform(),
        "torch_at_packaging": torch.__version__, "cuda_at_packaging": torch.version.cuda,
        "gpu_at_packaging": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "run_provenance": read_json(results / "run_provenance.json") if (results / "run_provenance.json").exists() else None,
        "missing_metadata": ["native trainer does not record isolated optimizer-only time or per-run end-to-end time"],
    }
    write_json(destination / "provenance/provenance.json", provenance)
    (destination / "provenance/library_versions_at_packaging.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    write_json(destination / "statistics/methods.json", {
        "bootstrap": "exhaustive paired five-seed bootstrap, 3125 ordered resamples; percentile 95% intervals",
        "wilcoxon": "exact conditional sign enumeration, average ranks for ties, exact zeros removed",
        "minimum_two_sided_p_with_five_nonzero_pairs": 0.0625, "multiple_testing": "Holm across reported metrics",
        "zero_baseline_percent": "undefined; raw zero retained; full-five percentage aggregate not formed from fewer seeds",
    })
    negatives = aggregate[aggregate.losses_of_5 > 0][["metric", "losses_of_5", "mean_paired_improvement_percent"]].to_dict("records")
    limits = read_json(destination / "guard_evidence/logging_limits.json")
    readme = f"""# {benchmark}: five-seed full-guard supplementary package

Scientific question: does the full guarded controller improve a manufactured
transient PDE under a matched committed training budget with explicit controller-overhead accounting?

Seeds: 0,1,2,3,4. Methods: vanilla and vara_v2. Exactly 4000 committed Adam steps.
Warmup 500; 7 blocks of 500; matched neutral/action probes 25 steps each.
Model: tanh 3-96-96-96-96-96-1, 37729 trainable parameters.
Adam LR 0.001, no scheduler; PDE/BC/IC/sparse weights 1/10/10/2.
Sparse manufactured data: 507 fixed continuous uniform samples, a 2% count
relative to the 48x48x11 evaluation-grid size. This is not a grid-node subset.
No convergence stopping, best-state restoration, or final optimizer repair exists
in this trainer. Final checkpoints are the final committed states.

Information boundary: prescribed BC/IC and manufactured forcing define the PDE.
The frozen sparse subset is allowed training/diagnostic information. Full-grid
reference errors and interface/layer evaluation masks are used only after training.
No full-grid metric is fed into ranking, acceptance, rollback, memory or selection.

All five raw values are visible in tables/per_seed_paired_results and
tables/all_seed_raw_values. Every table row points to raw summaries or decisions.
Negative/trade-off outcomes are retained: {json.dumps(negatives)}.
All source figures/logs/checkpoints are retained in raw/, including unfavorable events.

Statistics: mean, sample SD, median, paired difference/percent, wins, Cohen dz,
rank-biserial effect, paired bootstrap CI, exact signed-rank and Holm results.
Five-pair inference is descriptive; the minimum two-sided exact p is 0.0625.
Undefined zero-baseline percentages/zero-variance dz are recorded as NA.

Representative seed {representative}: closest to median primary full-field
improvement, ties smallest seed. Rule and all candidates are in provenance/.
Figures are PDF/SVG/PNG, with numeric inputs in figures/data/.
Checkpoint-derived figures use forward inference only and do not replace stored metrics.

Compute: compute/compute_accounting distinguishes retained steps, neutral/action
probes, discarded auxiliary calls, total calls, loss objectives, gradient objectives,
diagnostic evaluations, and point counts. Point counts are derived from native
counters/batch sizes; they are not FLOP counts. Native optimization_wall_clock_sec
includes controller/diagnostic time, excluding final evaluation/plots. Run provenance,
when captured by the notebook, additionally records complete suite runtime.

Guard evidence: all native decisions and allocations are copied. Positive target
improvement with a beyond-tolerance guard change is listed in
guard_evidence/prevented_harm_candidates. If absent, the highlighted-event JSON
explicitly records none. Logging limitations: {json.dumps(limits)}.

Package mapping: main manuscript -> tables/main_aggregate_results and
figures/aggregate_ci; supplement -> per-seed/negative tables, fairness/,
statistics/, compute/, guard_evidence/, interface/layer and decision figures.
CSV/Markdown/LaTeX are generated for tables; LaTeX uses longtable.

Reproducibility: source_snapshot/source.zip contains exact commit {commit};
configs/ contains all ten runtime-resolved configs. fairness/ contains independently
verified initial, coordinate, target, combined-data and final-model hashes plus
frozen coordinate/target NPZs. provenance/ records available command/environment
metadata; missing records are labeled rather than invented.

Integrity: raw copies are read-only, byte hashes are verified against the original
input fingerprint. checksums/SHA256SUMS covers every payload except itself.
The ZIP has an external SHA256 sidecar. Original outputs are never overwritten.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    write_json(destination / "checksums/original_results_sha256.json", original)
    if snapshot(results) != original or snapshot(destination / "raw") != original:
        raise ValueError("Original or raw-copy files changed; abort packaging.")
    for path in (destination / "raw").rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    hashes = snapshot(destination)
    manifest_path = destination / "checksums/SHA256SUMS"
    manifest_path.write_text("".join(f"{value}  {name}\n" for name, value in hashes.items()), encoding="utf-8")
    for name, value in hashes.items():
        if sha256(destination / name) != value:
            raise ValueError(f"Package checksum mismatch: {name}")
    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(destination).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed.")
        files = len(archive.namelist())
    if snapshot(results) != original:
        raise ValueError("Original inputs changed during ZIP creation.")
    digest = sha256(zip_path)
    Path(str(zip_path) + ".sha256").write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    print(json.dumps({"zip": str(zip_path), "sha256": digest, "files": files,
                      "bytes": zip_path.stat().st_size, "paired_fairness": "PASS",
                      "checksums": "PASS", "zip_crc": "PASS", "originals_unchanged": True}, indent=2))
    return zip_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--results")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(protocol(args.benchmark, args.device, args.seeds), indent=2))
        return
    if not args.results or not args.output:
        parser.error("Packaging requires --results and a fresh --output directory.")
    build_package(args.results, args.benchmark, args.output)


if __name__ == "__main__":
    main()
