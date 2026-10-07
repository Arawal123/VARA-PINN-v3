# Allen--Cahn and advection--diffusion: reliable full-guard reproduction

The scientific workflow passes the transient-specific source audit. No trainer,
controller, loss, diagnostic, evaluation, runner, or scientific YAML is changed
by this revision. The additions are a read-only protocol preflight and a
post-run audit/analysis/packaging tool with tests. The source base is
`f14e1d21039e2bc3938830e255eb291f4fea7401`.

## Source evidence

`scripts/run_vara_v2_pde_generalization.py` merges the benchmark YAML and
`configs/pde_generalization/presets/reliable.yaml`, then resolves sparse count
as round(48*48*11*0.02)=507. The YAML's nominal 256 is not the runtime count.
Coordinates are continuous uniform x/y/t draws, not a grid-node subset.

`src/pde_generalization/trainer.py` uses the same initialization function and
seed for both methods, separate method-independent RNG streams for prescribed
conditions, frozen sparse data, and diagnostics, and positive sparse-data loss
weight 2. The sparse RNG seed is training_seed+30003+sparse_seed (offset 0).
The combined coordinate/target hash is stored natively. The post-run tool
reconstructs this exact pool and verifies its native combined hash before
exporting separate coordinate and target hashes and NPZs.

`_run_vara`, `_candidate_influence`, and `_counterfactual_block` perform weak
patch detection, candidate generation, gradient screening, matched neutral/action
probes, guard acceptance, neutral-state rollback, trust updates, and effectiveness
memory. Prescribed BC/IC and frozen sparse-data mismatch are permitted signals.
Full-grid reference errors and localized evaluation masks are used only after
training. Manufactured forcing is part of the specified PDE, not an error signal.

`run` checks exactly 4000 applied steps before evaluation. `_save_artifacts`
saves only final.pt from that state. There is no convergence stopping, final
optimizer repair, or best-state restoration path in this trainer.

## Effective protocol for both benchmarks

- Paired methods `vanilla` and `vara_v2`; seeds 0,1,2,3,4.
- Model 3-96-96-96-96-96-1 tanh; 37729 trainable parameters; float32.
- Adam LR 0.001, no scheduler, gradient clipping 10.
- Exactly 4000 committed steps: warmup 500, seven blocks of 500.
- Counterfactual probes 25 neutral + 25 action steps per active block;
  only one 25-step trajectory is retained. At most one action is probed per block.
- Actual VARA optimizer calls = 4000 + 25*active_blocks, at most 4175.
  The primary is a matched committed training budget with explicit controller-overhead accounting,
  not equal total compute. Candidate-gradient objectives are additionally counted.
- Per training objective: collocation 2048, BC 512, IC 512, frozen sparse 507.
- Loss weights PDE/BC/IC/sparse = 1/10/10/2.
- Patch grid 5x5x3 (75 patches); diagnostics 1024 interior, 256 BC, 256 IC,
  plus the frozen sparse pool. Patch aggregation uses the 90th percentile.
- Weak threshold 80th percentile; top two per channel; maximum eight active
  patches; persistence one. Target margin 0.005; guard degradation margin 0.02.
- Gradient prefilter, guard, rollback, trust updates, memory, variable awareness,
  sampling redistribution, and local-loss multipliers remain enabled.
- Trust radius initial/min/max 0.10/0.025/0.20; expand 1.25, shrink 0.5.
  Sampling minimum uniform mass 0.35; maximum patch mass 0.25;
  local-loss multiplier bounds 0.5 to 2.0; effectiveness EMA 0.8.
- Final evaluation grid 48x48x11. Allen--Cahn eps=0.04;
  advection--diffusion kappa=0.01, velocity=(1,0.5), sigma=0.09.
- Domain [0,1]^2 and time [0,1]; manufactured forcing and prescribed conditions.

## Commands (repository root)

Read-only effective settings/pool preflight:

```bash
python -B scripts/package_transient_fullguard.py --benchmark allen_cahn --plan_only --device cuda
python -B scripts/package_transient_fullguard.py --benchmark advection_diffusion --plan_only --device cuda
```

Run separately, later; use fresh timestamped output directories:

```bash
python -u -B scripts/run_vara_v2_pde_generalization.py --benchmark allen_cahn --methods vanilla vara_v2 --seeds 0 1 2 3 4 --preset reliable --sparse_fraction 0.02 --device cuda --output_dir experiments/transient_fullguard/allen_cahn_RUNID
python -u -B scripts/run_vara_v2_pde_generalization.py --benchmark advection_diffusion --methods vanilla vara_v2 --seeds 0 1 2 3 4 --preset reliable --sparse_fraction 0.02 --device cuda --output_dir experiments/transient_fullguard/advection_diffusion_RUNID
```

Post-process only, retaining the original CUDA runtime for byte-exact sparse
target reconstruction. Choose fresh supplement folders outside the input tree:

```bash
python -B scripts/package_transient_fullguard.py --benchmark allen_cahn --results experiments/transient_fullguard/allen_cahn_RUNID --output /content/transient_supplements_RUNID/allen_cahn
python -B scripts/package_transient_fullguard.py --benchmark advection_diffusion --results experiments/transient_fullguard/advection_diffusion_RUNID --output /content/transient_supplements_RUNID/advection_diffusion
```

The ZIPs are named `supplement_allen_cahn_fullguard_5seed.zip` and
`supplement_advection_diffusion_fullguard_5seed.zip` in the supplement parent.

## Evidence and limitations

Every package copies all raw outputs and configs, verified final checkpoints,
initial/data/final hashes, frozen pools, source snapshot, compute accounting,
five-seed tables, descriptive effect sizes, paired bootstrap confidence intervals,
exact signed-rank/Holm results, negative outcomes, and PDF/SVG/PNG figures.
Representative seed selection is deterministic. No training is invoked by packaging.
SHA256s, ZIP CRC, checkpoint/raw-copy integrity and unchanged original inputs are checked.

Raw decision logs retain prefiltered, accepted and rejected proposals, rollback
reasons, relative target improvements, guard changes/thresholds, trust radii, and
effectiveness after decisions. They do not record all untested ranked candidates,
absolute neutral/action targets, per-proposal full allocation snapshots or discarded
auxiliary per-step losses. These gaps are explicitly recorded, not invented.
Effectiveness-before values are derived from initial memory=1 and logged prior after-values.
If no positive-target/beyond-tolerance rejected event exists, the report states none.

The native optimization_wall_clock_sec includes controller/diagnostic work and
excludes final evaluation/plotting. It is not isolated Adam time or complete suite time.
Capture the exact command, environment and complete suite wall time in the notebook's
run_provenance.json; the packager preserves that file when supplied.

With five nonzero pairs, the minimum two-sided exact Wilcoxon p-value is 0.0625.
Do not claim unsupported population significance or select seeds by favorable outcomes.
