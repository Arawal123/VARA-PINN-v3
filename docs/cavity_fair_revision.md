# Re=100 cavity fair revision

This opt-in protocol starts from `codex/vara-controller-v2` commit
`f14e1d21039e2bc3938830e255eb291f4fea7401`. Historical defaults are unchanged.
It retains the sparse-polish curriculum, its coefficients, diagnostic updates,
channel allocation, and mixed sparse-CFD sampling. Generic guarded candidates
remain disabled. No hyperparameters are tuned by the revision mode.

## Primary protocol

Use `scripts/run_vara_v2_continuation.py --fair_revision` with methods `vanilla`
and `vara_v2`, supervision `sparse_cfd_polish`, and Re=100 only. The runner
selects the reliable overlay/preset and applies the fairness overlay **after**
formulation/Re materialization, which historically re-enabled final repair.

- Both methods execute exactly 4000 applied Adam steps, with no early stopping,
  optimizer probes, rollback optimization, or final L-BFGS/physics repair.
- Evaluation and `final.pt` use the model after the final primary Adam step.
  Shared best-checkpoint selection/restoration and V2 curriculum-best final
  restoration are disabled. Curriculum adaptation during training is retained.
- Model: tanh MLP `2-64-64-64-3`, UVP soft BC, 8707 trainable parameters.
- Base Adam LR: 0.001; historical 4000-step warmup/cosine schedule retained.
- Collocation: 1024 for steps 1-1000, 1536 for 1001-3000, 2048 for 3001-4000.
- Boundary: 512 per step. Validation: 64x64. Final evaluation: 96x96.
- CFD: 1% of eligible Re=100 PaddleScience points, rounded to 639 fixed velocity
  labels per run. Pressure/vorticity labels are not used. Mixed sampling and
  boundary/corner exclusions remain historical.
- Unless `--cfd_seed` explicitly fixes one common pool seed, the pool seed is the
  training seed. Each Vanilla/V2 pair has identical pool construction. This
  prevents the historical runner's pre-loop base seed from fixing every pool
  inadvertently to seed zero.
- Full-field CFD/Ghia and literature topology may be evaluated after training;
  they cannot select the final model, stop training, or gate subsequent work.

The collocation/boundary budgets count training-objective point evaluations,
not all diagnostic forward/autograd calls. Report diagnostic cost and wall time
separately. Historical sparse-polish weight adaptation has a different overhead
from Vanilla; equal primary steps do not imply equal wall-clock time.

## Lightweight preflight and run

From the repository root, resolve all five pools/configurations without creating
output directories or constructing a trainer:

```bash
python -B scripts/run_vara_v2_continuation.py \
  --fair_revision --plan_only \
  --methods vanilla vara_v2 --reynolds 100 --seeds 0 1 2 3 4 \
  --data_supervision sparse_cfd_polish --device cuda \
  --output_dir experiments/revision_fair/cavity_re100
```

The expensive experiment is **not** run as part of implementation/testing. To
launch it later, use the same command without `--plan_only`:

```bash
python scripts/run_vara_v2_continuation.py \
  --fair_revision \
  --methods vanilla vara_v2 --reynolds 100 --seeds 0 1 2 3 4 \
  --data_supervision sparse_cfd_polish --device cuda \
  --output_dir experiments/revision_fair/cavity_re100
```

Nonempty output directories are rejected by default; use a fresh output name
instead of overwriting evidence.

## Evidence and assertions

Each `<output>/seed_<seed>/re_0100/<method>/logs/fairness_manifest.json` records
commit, initialization/final model fingerprints, architecture, parameter count,
optimizer/schedule, applied and total optimizer/probe work, point budgets,
evaluation grids, frozen pool identity, source CFD path/SHA256, and all disabled
stopping/repair/restoration/gating flags. The frozen coordinates/labels also have
a content SHA256 because the historical pool hash covers indices and absolute
path, **not** source bytes or label values. Absolute-path-dependent pool hashes
need only match within each pair; source/content hashes permit cross-host checks.

`summary/fairness_pairs.json` and `.csv` report each matched pair; any mismatch
raises an error after saving evidence. `summary/fair_revision_plan.json` retains
the preflight configuration/pools. Ordinary raw results, all-stage comparisons,
figures, and final checkpoints remain available. Negative outcomes are retained;
no reference-validity filtering removes fair pairs from the comparison table.

## Continuation remains disabled

The submitted manuscript and a manuscript-specific stage/run registry are not
present in the inspected repository. The historical docs describe stages
100,150,200,300,400,600,800,1000, while the frozen protocol names held-out reference
cases 100,400,1000,1600,3200. These lists do not establish which generated the
manuscript continuation result. The shipped CFD map supports only the latter.

Option A (100->400->1000) changes the documented intermediate continuation.
Option B (pure-PINN intermediate stages) changes the sparse-information schedule
and is not implemented by the historical single-mode runner. Neither is selected
merely because it can run. Fair mode rejects any sequence other than `[100]`
before output creation/training, pending manuscript attribution confirmation.
Historical continuation remains untouched. If later enabled, method-specific
previous checkpoints give equal warm-start rules, not identical learned states;
requiring identical stage-start tensors would define a common-anchor experiment.

## Colab provenance

The revision commit is local until explicitly published. Before cloning the new
branch in Colab, publish it from the local checkout with:

```bash
git push -u origin rie-cavity-fair-run
```

Then clone `--single-branch --branch rie-cavity-fair-run`, print branch/HEAD/status,
install `requirements.txt`, verify CUDA, run the preflight above, and launch only
when ready. Pin the tested revision commit in the notebook's provenance check.
