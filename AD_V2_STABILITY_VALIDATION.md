# Local validation, 2026-10-10

- 20 targeted tests passed: `tests/test_ad_causal_diagnostic.py`,
  `tests/test_ad_stability_revision.py`, `tests/test_ad_stability_package.py`.
- Final runner integration: seeds 0–4, both methods, **12 committed steps**,
  two 5-step blocks, one-step probes, 8x8 tiny MLP, CPU. All ten runs completed
  and all five paired manifests passed. These smoke results are not scientific
  comparative evidence. Every VARA smoke probe was rejected, and retained
  predictions matched the safeguarded Vanilla path.
- Supplement packaging smoke completed: 352 files covered by SHA256, five
  recorded matched pairs, initial-checkpoint hashes checked, raw inputs
  unchanged. Paired tables, exact signed-rank results, bootstrap summaries,
  negative/runtime tables, PDF/SVG/PNG figures and independent ZIP verification
  exercised successfully. The smoke package is labeled `CPU_SMOKE_NOT_PRIMARY`
  and records the dirty pre-commit source plus effective source-file hashes.
- Colab notebook: every code cell compiled and `nbformat.validate` passed.
- No full 4000-step GPU study or seed-8 historical causal run was executed.
  This machine has no usable CUDA GPU. Primary efficacy and H1/H2 attribution
  remain pending user execution.

Frozen primary defaults are 4000 steps and the original 96x5 architecture;
`--smoke` explicitly labels and overrides these for lightweight validation.
