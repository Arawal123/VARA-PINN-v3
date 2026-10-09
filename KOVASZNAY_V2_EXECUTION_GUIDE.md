# Kovasznay V2 five-seed execution guide

This is a new Re40 study using the shared VARAV2Controller. Historical V1 source/config/results are unchanged. Primary GPU results remain PENDING until the executable Colab notebook completes ten real runs. CPU seed999 fixtures verify infrastructure only.

## Five Colab steps

1. Download notebooks/kovasznay_v2_5seed_publication_colab.ipynb, upload/open it in Google Colab, select a GPU runtime. The first cell clones the exact implementation SHA embedded in the notebook; it refuses to reuse another checkout revision. No floating main branch is used.
2. Run setup, storage and preflight cells. Google Drive is enabled by default. Preserve the source/protocol-specific STUDY folder. The runtime, versions, exact source, protocol and launcher attempts are recorded. Existing metadata/raw results are not silently replaced.
3. Run the scientific regression suite and the separate seed999 CPU smoke cells. A failed or skipped required regression prevents primary packaging. The helper provides nested tqdm progress, phase, real committed steps, physical calls and ETA.
4. Run the explicitly labeled full-study cell. It runs seeds0–4 x Vanilla/V2, verifies each arm, skips verified complete results and resumes interrupted arms from atomic complete-state checkpoints. Do not mix smoke or optional sensitivity outputs with this root.
5. Run packaging. It rejects incomplete/mismatched studies, checks test fingerprints, creates all tables/figures and all-payload checksums, verifies raw tensors and ZIP integrity independently, preserves the ZIP on Drive and offers files.download. Negative rows remain visible.

## Frozen scientific protocol

Re40, x[-.5,1], y[-.5,1.5], steady incompressible Navier-Stokes u/v/p, normalized2-input MLP with96x5 tanh hidden units (37,827 parameters). Paired initialization bytes;2048 collocation points;512 fixed velocity boundary observations;1000 fixed permitted u/v/p interior observations. Known interior pressures are centered within that fixed pool. No additional analytical labels are acquired. Diagnostic pools are1024 interior plus256 prescribed BC points and the permitted training-data pool.4x4 spatial patches. Loss weights: momentum_u1, momentum_v1,continuity1,BC10 and u/v/p each2/3; conventional means of squared errors. Reference-free pressure gauge p(domain midpoint)^2 x.001. Adam LR.001, clip10, no schedule/repair/early stop/best selection.500 warmup+7x500 committed steps;25-step neutral/action probes; at most one ranked candidate tested per block. Counts distinguish committed updates from discarded/replayed calls. Same uniform-state resampling and frozen examples in both arms.

Guard: target fractional reduction>.005; each protected fractional increase<=.02 relative to trained neutral. Protected quantities are patch-percentile PDE,continuity,momentum,BC proxies and physics/physics-plus-training-data sums. No dense analytical-error guard. All-parameter gradient screening, mean-one multipliers[.5,2], conserved sampling probabilities and35% uniform floor use the shared controller. Historic trust/reward calibration mathematics remains unchanged; no promise of trust expansion or final accuracy follows from a short probe. Full parity/deviation detail: V2_PARITY_MATRIX.md.

## Resume and integrity

Atomic generation checkpoints preserve initial state, completed warmup and every completed block, including Adam, allocations, trust/memory, all samplers and Python/NumPy/Torch/CUDA RNG, pools, schedule and retained trajectories. The atomic latest_complete.json pointer selects the authoritative generation. Partial-block work is replayed and remains append-only raw evidence. Source/runtime changes refuse exact resume. A hard interruption between optimizer intent/completion leaves an explicit uncertain-call upper bound; the verifier refuses an exact-compute claim instead of inventing missing metadata. Final metrics still require a complete retained schedule. Checkpoint inference verification permits2e-5 float32 CPU/GPU tolerance, while CPU neutral/Vanilla and resume tests require bit identity.

Per-step point counts include collocation,BC,data and one pressure-gauge input. Gradient-probe and diagnostic points are reported separately. These are logical evaluated samples, not FLOPs. Per-call optimization timing synchronizes CUDA; total wall time additionally includes diagnosis,logging,checkpointing and post-training evaluation. Both arms have4000 retained updates, but V2 has additional probes; this is not a claim of identical total calls or wall time.

## Exact CLI contract

```bash
python -B scripts/run_kovasznay_v2_publication.py --preflight --config configs/kovasznay_v2_publication.yaml
python -B scripts/run_kovasznay_v2_publication.py --config configs/kovasznay_v2_publication.yaml --mode vanilla --seed 0 --device cuda --output-dir STUDY/seed_0/vanilla
python -B scripts/run_kovasznay_v2_publication.py --config configs/kovasznay_v2_publication.yaml --mode vara_v2 --seed 0 --device cuda --output-dir STUDY/seed_0/vara_v2 --resume
python -B scripts/run_kovasznay_v2_publication.py --smoke --mode vanilla --seed 999 --device cpu --config configs/kovasznay_v2_publication.yaml --output-dir SMOKE/seed_999/vanilla
python -B scripts/run_kovasznay_v2_publication.py --smoke --mode vara_v2 --seed 999 --device cpu --config configs/kovasznay_v2_publication.yaml --output-dir SMOKE/seed_999/vara_v2
python -B scripts/verify_kovasznay_v2_publication.py --run-dir STUDY/seed_0/vara_v2 --strict
python -B scripts/package_kovasznay_v2_publication.py --input-root STUDY --output-dir FRESH_ANALYSIS/supplement --strict
python -B scripts/verify_kovasznay_v2_publication.py --zip FRESH_ANALYSIS/supplement_kovasznay_v2_5seed.zip --strict
```

Use --resume only for an existing initialized/interrupted run; omit it on first launch. The first initialization itself is checkpointed. Existing completed runs are verified and skipped. The Colab launcher automatically records scientific test evidence in STUDY/tests; manual execution must supply passing JUnit/stdout/source-fingerprint evidence with the same source. The preflight prints the exact protocol hash and source fingerprint.

Optional genuinely pure-PINN sensitivity: add --pure-pinn and write to a separate PURE_SENSITIVITY/seed_s/mode tree. This disables all interior supervision; only prescribed BCs and an arbitrary pressure gauge remain. Its frozen profile/hash are recorded independently. Package separately with --profile pure_pinn_sensitivity --strict, yielding an explicitly named sensitivity ZIP. Never merge these results into the1000-observation primary study.

## Evidence interpretation

Primary metric: held-out velocity aggregate relative L2. Secondary: u/v/centered-p/omega errors, pressure gradient, physics, BC, training-data fit and worst spatial patch. Analytical fields are used only after final training, beyond predeclared BC/training labels. Seed0 is the preregistered representative. Statistics retain all5 pairs, both improvement estimands and unfavorable values; exact two-sided Wilcoxon with five nonzero pairs has minimum p=.0625. Source, raw tables, model/Adam, arrays, all decisions, severity tensors, allocations, restoration hashes and independent checks accompany the ZIP.25-step acceptance does not prove475-step stability or causal held-out improvement. No V1 values substitute for V2 outcomes.
