"""Audit, report, render and checksum a completed revision pair; never train."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models import build_mlp_from_config
from src.physics.cavity_reference import load_full_field_reference, interpolate_full_field
from src.training.cavity_full_guard_revision import CONFIG, file_hash
from src.training.cavity_full_guard_audit import audit, control_plan
from src.utils.io import save_json

METHODS = {"vanilla": "vanilla", "vara_v2_full_guard": "full_guard_vara",
           "vanilla_compute_matched": "compute_matched_vanilla"}
ZIP_NAME = "supplement_full_guard_cavity_re100_2pct_seed0.zip"


def manifests(output):
    return {name: json.loads((output / name / "revision_manifest.json").read_text())
            for name in METHODS if (output / name / "revision_manifest.json").exists()}


def flatten(decisions):
    rows = []
    for decision in decisions:
        row = {}
        for key, value in decision.items():
            if isinstance(value, dict) and key.startswith("guard_"):
                row.update({key + "_" + name: number for name, number in value.items()})
            elif not isinstance(value, (dict, list)):
                row[key] = value
        changes = decision.get("guard_changes", {})
        thresholds = decision.get("guard_noise", {})
        violations = {name: change - thresholds[name] for name, change in changes.items()
                      if change > thresholds[name]}
        row["worst_guard_degradation"] = max(changes.values()) if changes else np.nan
        row["worst_guard_excess"] = max(violations.values()) if violations else 0.
        row["triggered_guards"] = ";".join(sorted(violations))
        row["prevented_harm"] = bool(not decision.get("accepted") and
            decision.get("observed_target_improvement", 0) > 0 and violations)
        rows.append(row)
    return pd.DataFrame(rows, columns=None if rows else ["block", "accepted", "prefiltered",
        "rollback_executed", "prevented_harm", "worst_guard_excess", "worst_guard_degradation"])


def tables(frame, root, name):
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / (name + ".csv"), index=False)
    markdown = ["| " + " | ".join(map(str, frame.columns)) + " |",
                "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    markdown.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
                    for row in frame.itertuples(index=False, name=None))
    (root / (name + ".md")).write_text("\n".join(markdown), encoding="utf-8")
    def tex(value):
        return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")
    latex = ["\\begin{tabular}{" + "l" * len(frame.columns) + "}", "\\hline",
             " & ".join(tex(column) for column in frame.columns) + " \\\\"]
    latex.extend(" & ".join(tex(value) for value in row) + " \\\\" for row in frame.itertuples(index=False, name=None))
    latex.extend(["\\hline", "\\end{tabular}"])
    (root / (name + ".tex")).write_text("\n".join(latex), encoding="utf-8")


def report(output, destination):
    result = audit(output)
    if not result["passed"]:
        raise RuntimeError("Fairness failed; raw outputs retained; no manuscript package certified")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite report {destination}")
    destination.mkdir(parents=True)
    data = manifests(output)
    for directory in ("configs", "provenance", "fairness", "results", "guard_evidence", "compute",
                      "tables", "figures", "reference", "source_snapshot", "checksums"):
        (destination / directory).mkdir()
    shutil.copy2(output / "fairness_audit.json", destination / "fairness" / "fairness_audit.json")
    shutil.copy2(output / "fairness_audit.csv", destination / "fairness" / "fairness_audit.csv")
    shutil.copy2(CONFIG, destination / "configs" / Path(CONFIG).name)
    shutil.copy2(output / "provenance.json", destination / "provenance" / "run_provenance.json")
    save_json({name: m["config"] for name, m in data.items()}, destination / "configs" / "resolved_configs.json")
    save_json({name: {key: m[key] for key in ("initial_model_sha256", "final_model_sha256",
        "sparse", "parameter_count", "final_state_rule")} for name, m in data.items()},
        destination / "fairness" / "hashes_and_final_state.json")
    raw_rows = [{"method": name, **m["metrics"]} for name, m in data.items()]
    pd.DataFrame(raw_rows).to_csv(destination / "results" / "seed0_metrics.csv", index=False)
    a, b = data["vanilla"]["metrics"], data["vara_v2_full_guard"]["metrics"]
    changes = []
    for metric in sorted(set(a) & set(b)):
        left, right = a[metric], b[metric]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
            changes.append({"metric": metric, "vanilla": left, "full_guard_vara": right,
                "vara_minus_vanilla": right - left,
                "percentage_improvement": 100 * (left - right) / abs(left) if left else np.nan,
                "source_vanilla": "raw/vanilla/revision_manifest.json:metrics." + metric,
                "source_vara": "raw/full_guard_vara/revision_manifest.json:metrics." + metric,
                "interpretation": "descriptive delta; lower-is-better applies to error/residual metrics only"})
    paired = pd.DataFrame(changes)
    paired.to_csv(destination / "results" / "paired_changes.csv", index=False)
    # Include every negative delta; no statistical inference from this seed.
    paired[paired.percentage_improvement < 0].to_csv(destination / "results" / "negative_outcomes.csv", index=False)
    tables(paired, destination / "tables", "paired_changes")
    decisions = json.loads((output / "vara_v2_full_guard" / "revision_decisions.json").read_text())
    events = flatten(decisions)
    events.to_csv(destination / "guard_evidence" / "decisions.csv", index=False)
    save_json(decisions, destination / "guard_evidence" / "decisions.json")
    for name, mask in (("accepted_actions", events.accepted.fillna(False).astype(bool)),
                       ("prefiltered_actions", events.prefiltered.fillna(False).astype(bool)),
                       ("rollback_events", events.rollback_executed.fillna(False).astype(bool)),
                       ("harmful_candidate_prevention", events.prevented_harm.fillna(False).astype(bool))):
        events[mask].to_csv(destination / "guard_evidence" / (name + ".csv"), index=False)
    events[~events.accepted.fillna(False).astype(bool) & ~events.prefiltered.fillna(False).astype(bool)].to_csv(
        destination / "guard_evidence" / "rejected_actions.csv", index=False)
    harmful = events[events.prevented_harm.fillna(False).astype(bool)].sort_values(
        ["worst_guard_excess", "block"], ascending=[False, True], kind="stable")
    save_json({"count": len(harmful), "selection_rule": "largest guard-threshold excess; stable block/decision-order ties",
        "highlight": None if harmful.empty else harmful.iloc[0].to_dict()},
        destination / "guard_evidence" / "highlight_rule.json")
    for name in ("revision_proposals.json", "probe_pairs.json", "vara_v2_allocation_history.json"):
        shutil.copy2(output / "vara_v2_full_guard" / name, destination / "guard_evidence" / name)
    events.reindex(columns=["block", "trust_radius_before", "trust_radius_after"]).to_csv(
        destination / "guard_evidence" / "trust_radius_history.csv", index=False)
    events.reindex(columns=["block", "variable", "patch_id", "action_type", "effectiveness_before", "effectiveness_after"]).to_csv(
        destination / "guard_evidence" / "action_memory_history.csv", index=False)
    compute = pd.DataFrame([{"method": name, **m["compute"]} for name, m in data.items()])
    tables(compute, destination / "tables", "compute")
    compute.to_csv(destination / "compute" / "primary_compute.csv", index=False)
    for name, columns in {
        "optimizer_call_accounting": ["committed_steps", "neutral_probe_steps", "action_probe_steps", "retained_probe_steps", "discarded_branch_steps", "total_optimizer_calls"],
        "objective_evaluation_accounting": ["objective_evaluations", "gradient_screen_evaluations", "gradient_backpropagations", "collocation_point_evaluations_train_and_screen", "boundary_point_evaluations_train_and_screen", "sparse_data_evaluations"],
        "diagnostic_accounting": ["diagnostic_evaluations", "diagnostic_grid_point_visits"],
        "timing": ["optimization_wall_clock_sec", "controller_seconds", "total_wall_clock_seconds"]}.items():
        compute.reindex(columns=["method", *columns]).to_csv(destination / "compute" / (name + ".csv"), index=False)
    save_json(control_plan(output), destination / "compute" / "compute_matched_control_manifest.json")
    save_json({"source": data["vanilla"]["sparse"], "training": "frozen sparse u/v observations only",
        "evaluation": "dense CFD and Ghia attached only after final committed step",
        "rights": "Original repository attribution/provenance retained; no new redistribution license inferred."},
        destination / "reference" / "reference_manifest.json")
    reference_root = ROOT / "data/references/lid_driven_cavity"
    for name in ("metadata.yaml", "ghia_1982_u_centerline.csv", "ghia_1982_v_centerline.csv"):
        shutil.copy2(reference_root / name, destination / "reference" / name)
    shutil.copy2(data["vanilla"]["sparse"]["source_path"], destination / "reference" / "source_cfd.npz")
    assert file_hash(destination / "reference" / "source_cfd.npz") == data["vanilla"]["sparse"]["source_cfd_sha256"]
    for name in ("reference_provenance.csv", "reference_map.csv", "reference_vs_ghia_validation.csv"):
        source_file = reference_root / "full_field" / name
        if source_file.exists():
            shutil.copy2(source_file, destination / "reference" / name)
    figures(output, destination / "figures", data, paired, events)
    save_json({"raw_source_hashes": {name: file_hash(output / name / "revision_manifest.json") for name in data},
        "lineage": {"results/paired_changes.csv": "raw/*/revision_manifest.json:metrics",
            "guard_evidence/*": "raw/full_guard_vara/revision_decisions.json, revision_proposals.json, probe_pairs.json",
            "compute/*": "raw/*/revision_manifest.json:compute",
            "figures/01-05": "raw/*/checkpoints/final.pt, raw/*/sparse_dataset.npz, reference source",
            "figures/06-10": "raw/full_guard_vara/revision_decisions.json, vara_v2_allocation_history.json",
            "figures/11-12": "results/paired_changes.csv, compute/primary_compute.csv"}},
        destination / "provenance" / "lineage.json")
    (destination / "README.md").write_text(README, encoding="utf-8")
    print(f"Report complete: {destination}; prevented-harm cases={len(harmful)}")


def figures(output, directory, data, paired, events):
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none"})
    def save(fig, name):
        fig.suptitle("Re=100 · seed 0 · non-polish full guard", fontsize=11)
        fig.tight_layout()
        for ext in ("pdf", "svg", "png"):
            fig.savefig(directory / (name + "." + ext), bbox_inches="tight")
        plt.close(fig)
    xy = np.stack(np.meshgrid(np.linspace(0, 1, 192), np.linspace(0, 1, 192)), axis=-1).reshape(-1, 2)
    source = load_full_field_reference(data["vanilla"]["sparse"]["source_path"])
    reference = interpolate_full_field(source, xy)
    fields, models = {}, {}
    for name in data:
        cfg = data[name]["config"]
        model = build_mlp_from_config(cfg, (0., 1., 0., 1.))
        checkpoint = torch.load(output / name / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models[name] = model
        with torch.no_grad():
            fields[name] = model(torch.tensor(xy, dtype=torch.float32)).numpy()[:, :2]
    truth = np.column_stack((reference["u"].reshape(-1), reference["v"].reshape(-1)))
    def maps_plot(arrays, titles, name, signed=False):
        fig, axes = plt.subplots(len(arrays), 2, figsize=(8, 2.6 * len(arrays)), squeeze=False)
        vmax = max(np.max(np.abs(array)) for array in arrays)
        for row, (array, title) in enumerate(zip(arrays, titles)):
            for col in range(2):
                image = axes[row, col].imshow(array[:, col].reshape(192, 192), origin="lower", extent=[0, 1, 0, 1],
                    cmap="RdBu_r" if signed else "magma", vmin=-vmax if signed else 0, vmax=vmax)
                axes[row, col].set_title(title + (" u" if col == 0 else " v"))
                fig.colorbar(image, ax=axes[row, col])
        save(fig, name)
    maps_plot([truth, fields["vanilla"], fields["vara_v2_full_guard"]], ["CFD reference", "Vanilla", "Full guarded VARA"], "01_fields", True)
    maps_plot([abs(fields[name] - truth) for name in ("vanilla", "vara_v2_full_guard")], ["Vanilla error", "VARA error"], "02_errors")
    for component, name in ((0, "03_u_centerline"), (1, "04_v_centerline")):
        fig, ax = plt.subplots(figsize=(6, 4))
        coordinate = np.linspace(0, 1, 257)
        profile_xy = np.column_stack((np.full(257, .5), coordinate)) if component == 0 else np.column_stack((coordinate, np.full(257, .5)))
        ref = interpolate_full_field(source, profile_xy)
        ax.plot(coordinate, ref["u" if component == 0 else "v"], color="black", label="CFD")
        ghia = pd.read_csv(ROOT / f"data/references/lid_driven_cavity/ghia_1982_{'u' if component == 0 else 'v'}_centerline.csv")
        ghia = ghia[np.isclose(ghia.re, 100)]
        ax.scatter(ghia["y" if component == 0 else "x"], ghia["u_ref" if component == 0 else "v_ref"], color="black", marker="x", label="Ghia")
        for method, model in models.items():
            with torch.no_grad():
                prediction = model(torch.tensor(profile_xy, dtype=torch.float32)).numpy()
            ax.plot(coordinate, prediction[:, component], label=method)
        ax.legend(fontsize=7); ax.set_xlabel("y at x=0.5" if component == 0 else "x at y=0.5")
        save(fig, name)
    fig, ax = plt.subplots(figsize=(5, 5))
    sparse = np.load(output / "vanilla" / "sparse_dataset.npz")["coordinates"]
    ax.scatter(sparse[:, 0], sparse[:, 1], s=3, alpha=.5)
    ax.set(xlim=(0, 1), ylim=(0, 1), title=f"Frozen supervision: {len(sparse)} points")
    save(fig, "05_sparse_points")
    fig, ax = plt.subplots(figsize=(7, 3))
    if not events.empty:
        colors = np.where(events.prefiltered.fillna(False), "gray", np.where(events.accepted.fillna(False), "green", "red"))
        ax.scatter(events.block, np.arange(len(events)), c=colors)
    else:
        ax.text(.5, .5, "No controller decisions", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Control block", ylabel="Decision index")
    save(fig, "06_decision_timeline")
    fig, ax = plt.subplots(figsize=(5, 3))
    counts = data["vara_v2_full_guard"]["guard_execution"]
    ax.bar(["accepted", "rejected", "prefiltered"], [counts[k] for k in ("accepted", "rejected", "prefiltered")])
    save(fig, "07_action_counts")
    fig, ax = plt.subplots(figsize=(5, 4))
    if "observed_target_improvement" in events:
        ax.scatter(events.observed_target_improvement, events.worst_guard_degradation,
                   c=np.where(events.accepted.fillna(False), "green", "red"))
    else:
        ax.text(.5, .5, "No probed actions", ha="center", transform=ax.transAxes)
    ax.axhline(.02, color="black", linestyle="--"); ax.axvline(.005, color="black", linestyle=":")
    ax.set(xlabel="Target relative improvement", ylabel="Worst guard degradation")
    save(fig, "08_target_vs_guard")
    allocations = json.loads((output / "vara_v2_full_guard" / "vara_v2_allocation_history.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot([a["block"] for a in allocations], [a["trust_radius"] for a in allocations], marker="o")
    ax.set(xlabel="Block", ylabel="Trust radius"); save(fig, "09_trust_radius")
    fig, ax = plt.subplots(figsize=(7, 4))
    if allocations:
        image = ax.imshow(np.array([a["sampling_mass"] for a in allocations]).T, aspect="auto", origin="lower")
        fig.colorbar(image, ax=ax, label="Sampling probability")
    else:
        ax.text(.5, .5, "No allocations", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Block", ylabel="Patch"); save(fig, "10_allocations")
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, manifest in data.items():
        metrics = manifest["metrics"]
        ax.scatter(metrics["pde_residual_mean"], metrics["velocity_full_rel_l2"], label=name)
    ax.set(xlabel="Mean PDE residual", ylabel="Full-field velocity relative L2"); ax.legend(fontsize=7)
    save(fig, "11_reconstruction_physics")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, key in zip(axes, ("total_optimizer_calls", "objective_evaluations")):
        ax.bar(list(data), [m["compute"][key] for m in data.values()]); ax.set_title(key)
        ax.tick_params(axis="x", labelrotation=20)
    save(fig, "12_compute")


README = """# Re=100 single-seed exploratory mechanism experiment

Reviewer-response lid-driven cavity experiment comparing Vanilla PINN against the full V2 guarded VARA controller at Re=100 using identical 2% sparse-CFD supervision, with sparse-polish disabled, symmetric final-state evaluation, explicit controller-overhead accounting, and supplementary-grade provenance.

This is seed 0 only. No population-level statistical inference or significance is claimed. Requested fraction is .02 of eligible interior CFD points; actual integer count and realized fraction are recorded separately. Both primary methods use the same sparse u/v objective, initialization, deterministic LR and sampling schedules, float32 policy, and 4000 committed Adam steps. Vanilla receives no allocation adaptation. Sparse-polish, L-BFGS, final repair, early stopping and best-state restoration are disabled. The final model is the final committed Adam trajectory; retained neutral probes after rejection are part of that trajectory.

Full VARA uses the existing V2 channel/patch diagnostics, weak-region selection, gradient screen, bounded sampling/local weights, neutral/action probes, target/guard acceptance, rollback, trust update and memory. The primary comparison is **matched committed training budget with explicit controller-overhead accounting**, not equal total compute. Optional secondary Vanilla matches measured optimizer calls and replays the same fixed-size screening forward/gradient-query counts without allocation changes. This is an operation-count control, not FLOP or wall-clock equality. It holds the original LR floor after step 4000. Replay screening queries do not optimize Vanilla.

Dense CFD/Ghia reconstruction is final-evaluation only. Training reads only frozen sparse CFD observations and prescribed PDE/BC quantities. The training benchmark has no dense/profile reference attached. Raw proposal logs include untested proposals after acceptance, prefiltered proposals without observed outcomes, and all rejected trials. Missing observed fields are unavailable, not zero. A zero prevented-harm count is reported honestly. Highlight rule: greatest guard-threshold excess among rejected actions with positive target improvement, stable block/decision-order ties.

## Claim and artifact mapping

* Reconstruction/physics deltas: tables/paired_changes.* -> results/paired_changes.csv -> raw/*/revision_manifest.json:metrics.
* Final fields, errors, centerlines and sparse pool: figures/01-05 -> raw/*/checkpoints/final.pt, sparse_dataset.npz and the hashed reference source.
* Guard efficacy: figures/06-10 and guard_evidence/* -> raw/full_guard_vara/revision_decisions.json, revision_proposals.json, probe_pairs.json and vara_v2_allocation_history.json.
* Compute comparison: figures/12 and compute/* -> raw/*/revision_manifest.json:compute.
* Trade-offs: figures/11 and results/negative_outcomes.csv -> the same raw metrics; unfavorable outcomes are retained.

Full precision CSV/JSON is authoritative; rendered tables may round for readability. provenance/lineage.json records source hashes. configs records the frozen and actual run configuration. fairness records the automatic final-state, pool-byte and paired-config checks. source_snapshot contains exact relevant code. reference contains original attribution and source hashes; no additional license is inferred. checksums/SHA256SUMS.txt covers package files except itself. ZIP CRC and every member hash are verified. Raw copies are verified byte-identical. A later change in source CFD or runtime may change results; seed-0 thresholds were frozen before evaluation.
"""


def package(output, destination):
    source_provenance = json.loads((output / "provenance.json").read_text())
    current_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if dirty or source_provenance["revision_sha"] != current_sha:
        raise RuntimeError("Package source must be clean and match the run's immutable revision SHA")
    if not destination.exists():
        report(output, destination)
    lineage = json.loads((destination / "provenance" / "lineage.json").read_text())
    if lineage["raw_source_hashes"] != {name: file_hash(output / name / "revision_manifest.json") for name in manifests(output)}:
        raise RuntimeError("Raw runs changed since report; create a new supplement directory")
    if not audit(output)["passed"]:
        raise RuntimeError("Fairness failed")
    archive = destination.parent / ZIP_NAME
    if archive.exists() or (destination / "raw").exists():
        raise FileExistsError("Refusing to overwrite an existing package")
    for name, label in METHODS.items():
        if not (output / name).exists():
            continue
        shutil.copytree(output / name, destination / "raw" / label)
        for source in (output / name).rglob("*"):
            if source.is_file():
                copied = destination / "raw" / label / source.relative_to(output / name)
                assert file_hash(source) == file_hash(copied), f"Raw copy mismatch: {source}"
    for directory in ("src", "configs"):
        shutil.copytree(ROOT / directory, destination / "source_snapshot" / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("run_cavity_full_guard_revision.py", "package_cavity_full_guard_revision.py"):
        target = destination / "source_snapshot" / "scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)
    for name in ("requirements.txt", "requirements-cavity-revision.txt"):
        shutil.copy2(ROOT / name, destination / "source_snapshot" / name)
    save_json({"sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "missing_metadata": "No historical metadata invented; prefiltered/unprobed action observations unavailable."},
              destination / "source_snapshot" / "manifest.json")
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = "\n".join(file_hash(path) + "  " + path.relative_to(destination).as_posix() for path in files) + "\n"
    (destination / "checksums" / "SHA256SUMS.txt").write_text(manifest, encoding="utf-8")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(destination).as_posix())
    print(json.dumps(verify(archive), indent=2))


def verify(archive):
    archive = Path(archive)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise RuntimeError("ZIP CRC failed")
        manifest = zipped.read("checksums/SHA256SUMS.txt").decode()
        listed = set()
        for line in manifest.splitlines():
            expected, name = line.split("  ", 1)
            assert hashlib.sha256(zipped.read(name)).hexdigest() == expected, name
            listed.add(name)
        assert set(zipped.namelist()) == listed | {"checksums/SHA256SUMS.txt"}
        count = len(zipped.namelist())
    return {"zip": str(archive.resolve()), "bytes": archive.stat().st_size, "files": count,
            "sha256": file_hash(archive), "crc": "PASS", "all_member_checksums": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "metrics", "report", "package", "verify"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--supplement_dir", default=None)
    args = parser.parse_args()
    output = Path(args.output_dir)
    destination = Path(args.supplement_dir) if args.supplement_dir else output.parent / ("supplement_" + output.name)
    if args.action == "audit":
        if not audit(output)["passed"]:
            raise SystemExit(1)
    elif args.action == "metrics":
        for name, manifest in manifests(output).items():
            print(name, json.dumps({"metrics": manifest["metrics"], "compute": manifest["compute"],
                                   "guard_execution": manifest["guard_execution"]}, indent=2))
    elif args.action == "report":
        report(output, destination)
    elif args.action == "package":
        package(output, destination)
    else:
        print(json.dumps(verify(destination.parent / ZIP_NAME), indent=2))


if __name__ == "__main__":
    main()
