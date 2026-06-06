# Publication Study Extensions

These studies are opt-in. Existing configs, modes, commands, and experiment
directories retain their previous behavior.

## Equal-Compute Fairness

Run same-schedule, same-model-step, same-collocation-evaluation, and
same-wall-clock comparisons:

```bash
python scripts/run_equal_compute_study.py \
  --config configs/lid_driven_cavity.yaml \
  --methods vanilla rar self_adaptive_attention gradient_balanced gradient_enhanced vara \
  --seeds 0 1 2 3 4 \
  --scenarios same_schedule same_steps same_collocation same_wall_clock \
  --steps 400 \
  --collocation_evaluations 819200 \
  --wall_clock_sec 600 \
  --output_dir experiments/cavity_equal_compute
```

Equal-compute runs disable the guarded LBFGS final repair for every method.
LBFGS line searches use a variable number of closure evaluations and therefore
cannot provide an exact fixed-step comparison.

Main outputs:

- `summary/equal_compute_raw.csv`
- `summary/equal_compute_mean_std.csv`
- `summary/equal_compute_fairness_check.csv`

## Sensitivity Study

The default study is one-factor-at-a-time around
`configs/lid_driven_cavity.yaml`:

```bash
python scripts/run_sensitivity_study.py \
  --config configs/lid_driven_cavity.yaml \
  --study configs/studies/lid_cavity_sensitivity.yaml \
  --variants all \
  --seeds 0 1 2 \
  --output_dir experiments/cavity_sensitivity
```

Use `--plan_only` to create resolved configs without training.

## Modern Baselines

```bash
python scripts/run_modern_baselines.py \
  --config configs/lid_driven_cavity.yaml \
  --methods vanilla rar self_adaptive_attention gradient_balanced gradient_enhanced vara \
  --seeds 0 1 2 3 4 \
  --output_dir experiments/cavity_modern_baselines
```

Implemented independent baselines:

- `self_adaptive_attention_pinn`: adversarial soft attention over interior and
  boundary points.
- `gradient_balanced_pinn`: gradient-statistics adaptive global loss weights.
- `gradient_enhanced_pinn`: spatial gradients of momentum and continuity
  residuals.

The modern-baseline runner disables final repair by default for consistency.
Pass `--include_final_repair` only for a separate optimizer-stage experiment.

## Why VARA Wins

Generate post-hoc mechanism evidence from completed VARA and RAR experiment
folders:

```bash
python scripts/generate_vara_mechanism_evidence.py \
  --results_dir experiments/cavity_modern_baselines \
  --output_dir experiments/cavity_modern_baselines/summary/vara_mechanism_evidence
```

The generator searches existing logs and produces:

- accepted/rejected intervention tables
- rollback-prevented damage table
- rejection-reason counts
- accepted-versus-rejected metric-effect plot
- variable-patch targeting heatmap
- first-to-last patch-score improvement heatmap
- RAR-versus-VARA trade-off table
- a short interpretation guide

## Legacy-Result Protection

- No existing configuration file is modified.
- Compute stopping is inactive unless `compute_budget.enabled` is true.
- New methods use new mode names.
- New runners use separate output directories.
- Tests verify that a disabled compute budget produces bitwise-identical model
  parameter updates to the legacy path.
