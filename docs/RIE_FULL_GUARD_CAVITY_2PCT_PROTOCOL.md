# Reviewer-response full-guard cavity protocol

Reviewer-response lid-driven cavity experiment comparing Vanilla PINN against the full V2 guarded VARA controller at Re=100 using identical 2% sparse-CFD supervision, with sparse-polish disabled, symmetric final-state evaluation, explicit controller-overhead accounting, and supplementary-grade provenance.

Base: `codex/vara-controller-v2`, `f14e1d21039e2bc3938830e255eb291f4fea7401`.
Revision: `rie-fullguard-cavity-2pct-compute-fair`.

The scientific question is whether the full guarded allocation mechanism helps reconstruction while preserving physics/BC fit, and whether it rejects target-improving but guard-damaging actions. This is a seed-0 exploratory mechanism experiment, not evidence of population-level significance. No thresholds were selected from new seed-0 evaluation results; no full experiment was run during implementation.

## Frozen protocol and lineage

The standalone YAML freezes the existing reliable continuation protocol after its Re=100 ordinary `sparse_cfd`/UVP-soft-BC materialization and the reliable preset. The 4000-step scale, 800-step warmup, 16 blocks of 200 steps and 10-step probes come from the established reliable cavity configuration. It intentionally restores the full V2 counterfactual mechanism and disables repair, checkpoint selection, early stopping and continuation gating. Historical YAMLs, trainers and controllers are not edited. The dedicated wrapper reuses V2 loss, sampler, diagnostics, gradient screen and controller functions; it never enters sparse-polish branches.

| Quantity | Frozen value |
|---|---|
| PDE/domain | Steady cavity Re=100, unit square, regularized unit lid, stationary remaining walls |
| Seed / dtype | 0 / float32 |
| Model | UVP soft BC, tanh MLP 2–64–64–64–3, 8707 trainable parameters |
| Adam base LR | .001 |
| Deterministic LR | 300-step warmup from .2 base-LR ratio; cosine decay to .08 ratio at step 4000 |
| Primary committed steps | 4000 each |
| Batch refresh | Every 200 committed steps; retained probe batch reused for block continuation |
| Collocation curriculum | 1024 through step 1000, 1536 through 3000, 2048 thereafter; inherited `< until_step` selection, frozen when a trajectory batch is drawn |
| Boundary batch | 512, focused sampler; same paired boundary draws |
| Sparse objective | Weight 3 × mean((u−u_target)^2 + (v−v_target)^2); component weights zero to avoid double counting |
| Sparse pool | Uniform without replacement, seed 0; same filter, indices, coordinates and targets |
| Requested fraction | .02 of eligible interior CFD points |
| Actual pool in checked source | 63940 eligible; 1279 selected; realized fraction .020003127932436658 |
| Source CFD SHA256 | eb1bf49a1ab0e4dd09d0a32253da366040ce4ba045d352b22f151cb40030550c |
| Patch grid | 4×4×1 |
| Diagnostic grid / screen batch | 64×64 / 256 interior + 32 boundary, plus frozen sparse observations |
| Final test / plot grid | 96×96 / 192×192 |
| Target / guard margin | Improvement > .005; each degradation ≤ .02 |
| Trust initial / bounds | .10 / [.025, .20] |
| Prefilter damage ratio | .25 |
| Uniform floor / patch cap | .35 / .25 |
| Local multipliers | [.5, 2], conserved mean |
| Final-state rule | Final committed Adam trajectory |
| Sparse-polish / repair / L-BFGS / best restore / early stop | All OFF |

The full resolved YAML is authoritative for PDE/BC/stabilizer weights and sampling details. The preflight prints every effective value. Point-count curriculum follows the original trainer boundary convention; batch construction fixes its counts for all steps in that batch, including neutral/action branches.

## Information and paired fairness

The training benchmark is constructed without dense/profile references. Only prescribed PDE/BC and frozen sparse velocity observations enter training. Runtime protocol validation fails if forbidden switches are enabled. Controller input metrics are restricted to the five declared physics/BC guards; dense/Ghia references are attached only after the final optimizer state is frozen. No controller ranking, stopping, checkpoint/model selection or repair uses dense data. Source-file bytes, selected indices, coordinates, u/v targets and the combined dataset have separate canonical SHA256 hashes. Canonical arrays include dtype/shape, little-endian contiguous bytes; model/state mappings use deterministic key order.

Both primary methods use the V2 objective kernel, identity allocations for Vanilla, identical initialization and exact paired boundary draws. Vanilla never generates/ranks/applies interventions. Neutral/action snapshots restore model, Adam state, loss-normalization state, samplers, Python/NumPy/Torch/CUDA RNG state and committed step. Block-level evidence checks restored identities. Sampling interventions intentionally change collocation coordinates; sample counts/schedules remain shared. Rejections immediately restore neutral model state and log its equality. The existing V2 memory/trust commit policy is retained: accepted action updated, or first rejected trial updated when no action is accepted. Later untested proposals are logged explicitly, never represented as accepted/rejected probes.

## Compute comparisons

Primary: **matched committed training budget with explicit controller-overhead accounting**, never equal total compute. All neutral and action optimizer calls are counted. Exactly one probe trajectory is retained per probed block. Discarded branch steps = total optimizer calls − committed steps. Objective counters include optimizer objectives and screen forwards; screen gradient-query counts include shared guard query plus candidate queries. Point visits and diagnostic calls are separate counters, not FLOP estimates. Total elapsed time includes controller/snapshot/logging/evaluation work; explicit controller time times diagnostics, guards and gradient screening. Native optimization timing includes probe optimization.

Secondary: after VARA finishes, `--control_plan` reads its immutable manifest and derives the optimizer-call budget and gradient-screen call/query counts. `--compute_matched_control` explicitly runs Vanilla with that many Adam updates, replays the same fixed-size gradient-screen work without applying allocations, starts from the same initial state and pool, and holds the original LR floor beyond step 4000. It answers whether additional optimization work could explain a gain. The matching vector is optimizer calls plus screen objective/backpropagation counts. This is not equal FLOPs: collocation curricula, autograd support and diagnostic overhead can differ, and all point/timing counters remain visible. The optional control is never automatically launched by the primary runner or plan command.

## Execution and progress

Install `requirements-cavity-revision.txt`. Use a GPU runtime for the actual experiment. Constructor-only preflight is safe on CPU:

```bash
python scripts/run_cavity_full_guard_revision.py --device cpu --output_dir experiments/revision_full_guard/preflight --preflight
```

The primary runner requires a clean checkout and a nonexistent output directory. Its progress prints committed percentage every 50 ordinary steps, plus actual optimizer calls and phase. Probe progress does not count discarded work as primary completion. CUDA memory exhaustion and other failures preserve buffered decisions/proposals/probe evidence in `failure.json` and evidence files; failed results are not silently packaged as successful comparisons.

```bash
python -u scripts/run_cavity_full_guard_revision.py --device cuda --output_dir experiments/revision_full_guard/cavity_re100_2pct_seed0_TIMESTAMP
python scripts/package_cavity_full_guard_revision.py audit --output_dir experiments/revision_full_guard/cavity_re100_2pct_seed0_TIMESTAMP
python scripts/package_cavity_full_guard_revision.py metrics --output_dir experiments/revision_full_guard/cavity_re100_2pct_seed0_TIMESTAMP
python scripts/package_cavity_full_guard_revision.py report --output_dir experiments/revision_full_guard/cavity_re100_2pct_seed0_TIMESTAMP
python scripts/package_cavity_full_guard_revision.py package --output_dir experiments/revision_full_guard/cavity_re100_2pct_seed0_TIMESTAMP
python scripts/package_cavity_full_guard_revision.py verify --output_dir experiments/revision_full_guard/cavity_re100_2pct_seed0_TIMESTAMP
python scripts/run_cavity_full_guard_revision.py --output_dir experiments/revision_full_guard/cavity_re100_2pct_seed0_TIMESTAMP --control_plan
```

Use a new `--supplement_dir` and archive parent for rebuilding a package after adding the optional control. Builders refuse overwrites. Package checks source SHA/cleanliness, copies complete raw methods separately, verifies byte identity of all copies including final checkpoints, emits full-precision paired CSV/JSON, preserves negative deltas and all guard events, and exports twelve deterministic figures in PDF/SVG/PNG. No representative seed selection is needed. Zero prevented-harm events is an honest outcome. README and lineage registry map tables/figures to raw metrics, checkpoints and controller files. ZIP CRC, every member checksum and manifest coverage are verified. Packages and raw experiment outputs are ignored by Git.

## Local verification scope

Regression tests cover forbidden config changes, historical YAML preservation, sparse-objective gradients, canonical hashes, paired constructor preflight, synthetic rollback/state/step accounting, and a synthetic report/archive with all 36 exports and byte/checksum checks. These fixtures are labeled synthetic and never scientific results. No 4000-step training is executed by the tests. CUDA execution, memory requirements and scientific outcomes remain to be validated in Colab.
