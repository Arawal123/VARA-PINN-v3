# Local AD stability revision execution

This revision is committed locally on `codex/ad-v2-stability-revision`. No push
is needed to run it in Colab. Historical source/configurations/results are
unchanged; use the new runner and frozen configuration explicitly.

1. Upload `notebooks/Advection_Diffusion_V2_Stability_5Seed.ipynb` to Colab.
2. Select a GPU runtime. Run cells in order and upload the provided
   `ad_v2_stability_revision.bundle` when prompted. The notebook verifies the
   bundle and checks out its exact unpublished source commit over the historical
   GitHub base. It prints the commit used and checks the frozen config hash.
3. Mount Drive. The default experiment is seeds 0–4, both methods, under
   `MyDrive/VARA_PINN_AD_stability/runs/ad_v2_stable_v1_5seed_trial_01`.
   Keep `RUN_LABEL` for resume; change it for a distinct fresh study. Completed
   runs are skipped only after protocol checks, and partial runs resume the
   latest completed block. Interrupted artifacts are retained.
4. The run cell streams loss-step progress, physical Adam calls and an ETA
   estimate. Probes/replay make total work variable; progress measures the
   committed-step budget. Every run is verified before the pair completes.
5. The package cell preserves raw files, all outcomes, configs, pools,
   checkpoints, guards, proposals, allocations, physical step audits, source,
   statistics, figures, and checksums. It independently verifies the ZIP and
   copies it to Drive. The final cell downloads the ZIP.

Equivalent primary command from the new source checkout:

```bash
python -u -B scripts/run_ad_v2_stability.py --device cuda --seeds 0 1 2 3 4 --output-dir /content/drive/MyDrive/VARA_PINN_AD_stability/runs/ad_v2_stable_v1_5seed_trial_01 --resume
```

For a one-seed screening run use `--seeds 8` and a distinct fresh output path.
It is screening evidence, not a substitute for the five-seed primary study.
For ablations append `--disable FEATURE` and use a distinct path. The allowed
feature names are in the runner's `--help`; do not merge ablations into the
primary aggregate.

```bash
python -B scripts/package_ad_v2_stability.py --results RESULTS_DIRECTORY --output /content/supplement_advection_diffusion_v2_stability_5seed
python -B scripts/verify_ad_stability_zip.py /content/supplement_advection_diffusion_v2_stability_5seed.zip
```

Optional historical-source causal diagnostic, separate from this new method:

```bash
python -B scripts/run_ad_v2_forced_neutral.py --reference-zip /content/supplement_advection_diffusion_fullguard_15seed_5to19.zip --output-dir /content/ad_seed8_original_forced_neutral --preflight
python -u -B scripts/run_ad_v2_forced_neutral.py --reference-zip /content/supplement_advection_diffusion_fullguard_15seed_5to19.zip --output-dir /content/ad_seed8_original_forced_neutral
```

Its required archive is the original 5–19 package, not the five-seed package.
It aborts on source/hash/warmup mismatch and must not be combined with the new
revision outcomes. No full historical or new GPU experiment was run locally.

The shared step safeguard changes Vanilla as well as VARA. Name the comparator
“safeguarded Vanilla” in new comparisons and avoid crediting shared optimizer
improvements to VARA. Independent component guards and continuation rechecks
add compute and may reject useful actions. All seeds and negative metrics stay
visible. The raw folder includes interrupted attempts; abandoned attempt cost
is separate from the completed trajectory's counters.
