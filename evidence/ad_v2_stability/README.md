# Evidence origin and interpretation

`twenty_seed_primary_results.csv`, `priority_seed_decision_chronology.csv` and
`root_cause_ranking.csv` are unchanged copies from the earlier 2026-10-08
forensic audit. The ranking file is historical, not the updated attribution.
All twenty original seed-level outcomes, including losses, remain visible.

`terminal_dynamics_and_constant_mode.csv`, `terminal_component_gradients.csv`,
`fixed_fraction_terminal_step_probe.csv` and `provenance.json` are the new
2026-10-10 offline measurements reproduced by
`scripts/analyze_ad_terminal_dynamics.py`. Their sources are the original
archived checkpoints/configs/logs/pools, each under historical source revision
`cdb27c8be0e681654d8c6c3005d9658c0ae6ee73`. Input archive SHA256 hashes are
retained in provenance. No optimizer step or new training was used.

The scalar shift and terminal parameter-ray fractions are diagnostics. They
are not repaired checkpoints, primary training outcomes, a new validation
cohort, or evidence that the stability revision is superior. Original failed
metrics remain unchanged. Updated causal conclusions and limitations are in
`AD_V2_ROOT_CAUSE_INVESTIGATION.md`.
