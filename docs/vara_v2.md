# VARA Controller V2

V2 is an opt-in matched-compute controller. Existing V1 modes, configs,
checkpoints, and experiment directories are not reused.

## Scientific invariants

- The controller receives residual, boundary, and configured training-data
  signals only.
- Ghia profiles, CFD fields, analytical test fields, and final test metrics are
  evaluation-only.
- Every V2 run uses 100 neutral steps and six 50-step control blocks by
  default. Ten probe steps are part of each block's budget.
- Sampling mass and patch-loss multiplier mass are conserved.
- Continuation chains are method-specific and optimizer moments are reset.
- Previous-Re replay is fixed, seeded, reference-free model distillation and
  is applied identically to every method.
- `final_total_loss` is not used for cross-method ranking.

## Deliberate module separation

- **Core VARA:** normalized diagnosis, budget-neutral allocation, gradient
  conflict screening, trust-region acceptance, action memory, and rollback.
- **Shared physics modules:** optional corner-regularized cavity hard
  boundaries, streamfunction-pressure output, and pressure gauge. These are
  architecture/formulation studies, not controller components.
- **Continuation module:** Reynolds scheduling, method-isolated warm starts,
  optimizer reset, decaying anchor, and fixed previous-model replay.

This separation permits controller gains to be reported independently from
physics-formulation and continuation gains. Hard-boundary and streamfunction
overlays must be applied to every compared method. Streamfunction runs require
third-order derivatives and therefore need separate compute reporting.

## Main commands

Controller smoke or full run:

```bash
python scripts/run_vara_v2.py \
  --config configs/vara_v2/lid_driven_cavity.yaml \
  --seeds 0 \
  --device cuda
```

Core ten-seed comparison on the legacy and parameter-matched enhanced
backbones:

```bash
python scripts/run_publication_suite_v2.py \
  --study core \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --device cuda \
  --output_dir experiments/vara_v2/publication_core
```

Controller ablations:

```bash
python scripts/run_publication_suite_v2.py \
  --study ablation \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --device cuda \
  --output_dir experiments/vara_v2/publication_ablation
```

Held-out benchmark and Reynolds-number study:

```bash
python scripts/run_publication_suite_v2.py \
  --study generalization \
  --heldout_seeds 0 1 2 3 4 \
  --device cuda \
  --output_dir experiments/vara_v2/publication_generalization
```

Binding wall-clock study. The budget is selected before comparison as 75% of
the minimum Vanilla pilot runtime:

```bash
python scripts/run_publication_suite_v2.py \
  --study wall_clock \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --device cuda \
  --output_dir experiments/vara_v2/publication_wall_clock
```

Method-isolated Reynolds continuation:

```bash
python scripts/run_vara_v2_continuation.py \
  --methods vanilla rar relobralo residual_attention gradient_enhanced vara_v1 vara_v2 \
  --reynolds 100 150 200 300 400 600 800 1000 1200 1600 2000 2400 3200 \
  --seeds 0 1 2 3 4 \
  --device cuda \
  --output_dir experiments/vara_v2/re_continuation
```

Shared physics-module study:

```bash
python scripts/run_publication_suite_v2.py \
  --study physics_modules \
  --methods vanilla rar relobralo residual_attention vara_v1 vara_v2 \
  --seeds 0 1 2 3 4 \
  --device cuda \
  --output_dir experiments/vara_v2/physics_modules
```

The `all` suite intentionally excludes `physics_modules` because the
streamfunction formulation has materially different derivative cost. Run and
report it separately.

Exploratory Re `4000-10000` runs must be reported as residual, boundary,
stability, and qualitative evidence unless a trustworthy full-field reference
is supplied.

## Outputs

The publication suite writes raw seed results, mean/std tables, paired
bootstrap intervals, Wilcoxon-Holm tests, effect sizes, seedwise wins, method
ranks, mechanism summaries, success gates, and plots under its `summary/`
folder. Every run also retains its resolved config, checkpoint, field figures,
and controller decision history.
