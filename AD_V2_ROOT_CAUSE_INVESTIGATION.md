# AD V2 causal investigation and opt-in stability revision

## A. Executive findings

The historical twenty-pair result remains negative on mean full-field error:
Vanilla mean 0.04380236435681577, VARA mean 0.05429096026346083,
10 wins and 10 losses. VARA's ratio of means is 23.95% worse. The prior
two-sided paired signed-rank p-value was 0.95633. None of these outcomes has
been repaired, replaced, filtered, or reclassified as a new success.

New offline measurements identify a finite-step overshoot at the last Adam
update in failed VARA seeds 1, 3, 5 and 8. Their gradient dotted with the actual
update is negative, yet the original fixed-batch weighted training objective
increases after the full update. Gradient clipping was inactive in all four.
Thus an uphill momentum direction and clipping alone do not explain these
particular endpoint deteriorations. Seed 18 improves from its previous state
at the full update but overshoots a substantially better interior point along
the same parameter ray. A Vanilla seed exhibits the same type of overshoot.

This evidence supports a reference-free objective safeguard shared by both
comparators. It does **not** prove that retaining VARA interventions caused the
preceding trajectory or that the new controller will outperform safeguarded
Vanilla. The historical-source seed-8 forced-neutral causal experiment remains
pending CUDA execution. This machine has Torch 2.12.0+cpu and no CUDA device.

The implementation is a new, frozen, opt-in `ad_v2_stable_v1` protocol. The
historical trainer/controller/configurations remain unchanged. Its effects on
held-out accuracy require a new paired experiment; no positive GPU outcome is
claimed here.

## B. Evidence matrix

| Question | Evidence | Conclusion / limit |
|---|---|---|
| Last-step descent direction? | `evidence/ad_v2_stability/terminal_dynamics_and_constant_mode.csv`, `gradient_dot_update`, `descent_alignment_cosine` | All 15 measured terminal states have positive descent alignment; not a generic uphill-direction failure. |
| Finite-step overshoot? | `fixed_fraction_terminal_step_probe.csv` and recovered pre-update objectives | Directly demonstrated on a fixed original training batch; interior fractions reduce objective in failure cases. |
| Gradient clipping responsible? | `clip_active` and gradient norms | Clipping inactive in VARA 1/3/5/8; active in 18. Changing clipping alone cannot address all measured cases. |
| Constant-offset null mode? | Exact PDE invariance test plus training-label-only scalar-offset diagnostic | AD has this null mode, but it explains only 0.9–5.9% of squared field error in the measured failed states; scalar shifts barely help or worsen error. Dominant explanation contradicted. |
| Neutral sampler is equivalent to Vanilla? | `tests/test_ad_causal_diagnostic.py`; original `_sample_adaptive` | Original neutral distribution is uniform but draws 717 global +1331 regional samples and shuffles, instead of direct 2048 uniform draws. Bitwise neutral parity fails historically and passes under the new direct-uniform branch. |
| Genuine historical short probes? | Archived decision chronology and original `_counterfactual_block` | Matched 25-step neutral/intervention probes exist; accepted decisions obey their recorded guards. There is then a 475-step unchecked continuation. |
| All channels can compete? | Historical candidate generator, normalization micro-test, chronology | Original top-two-region bottleneck and per-channel scale cancellation can concentrate availability on PDE. Of 139 AD tested targets, 134 were PDE, 4 IC, 1 sparse. |
| Trust reward units compatible? | Original `rank` versus `evaluate`, earlier scale audit | Severity-valued prediction is divided into fractional observed improvement. Historical predictions exceed 1.333, preventing reliable-reward expansion at threshold 0.75. |
| Sampling collapse? | Existing allocation audit | Not supported as the dominant explanation; failed seed 8 retains approximately 74.9 effective regions out of 75. |
| Cause without retained actions? | Seed 4 has zero acceptances yet 79.99% improvement; micro-tests; pending seed-8 diagnostic | H1 is plausible; seed 8's H1/H2 attribution cannot yet be resolved. A no-action outcome is not evidence of beneficial retained control. |

New measurements were made with zero optimizer calls. Inverse Adam reconstructs
the previous parameter state from the archived final Adam state under the
verified original optimizer settings. Original loss reconstruction error is
checked below 1e-6. The interpolation fractions were predeclared as
0, 0.25, 0.5, 0.75 and 1. These are parameter-space counterfactuals, **not** saved
intermediate training checkpoints or replacement published results.

## C. Failure-seed chronology and terminal evidence

Blocks are zero-indexed. Probe decisions occur after 500, 1000, ..., 3500
committed steps; retained probe completion is 25 steps later. All recorded
historical acceptances satisfy their contemporaneous target and guard rules.

| Seed | Vanilla L2 | Original VARA L2 | Accepted blocks: action / patch | Tested / prefiltered |
|---:|---:|---:|---|---:|
| 1 | 0.025167498737573624 | 0.049375057220458984 | 0 sampling/37; 1 joint/5; 6 sampling/58 | 7 / 0 |
| 3 | 0.038795072585344315 | 0.19134284555912018 | 1 sampling/36; 4 joint/11 | 7 / 1 |
| 5 | 0.014014686457812786 | 0.097240269184112549 | 3 local_loss/63; 4 joint/58; 6 joint/63 | 6 / 8 |
| 8 | 0.014231191948056221 | 0.24011783301830292 | 4 local_loss/11; 6 joint/6 | 7 / 0 |
| 18 | 0.016098691150546074 | 0.17762057483196259 | 3 joint/37; 4 sampling/38; 6 sampling/58 | 7 / 4 |
| 2, successful control | 0.099092267453670502 | 0.013262036256492138 | 0 sampling/63; 1 sampling/37; 3 joint/6 | 7 / 6 |
| 4, no-action control | 0.096714019775390625 | 0.01934894360601902 | none | 7 / 1 |

All accepted targets in this table are `pde_residual`. Complete recorded
proposal chronology, guard changes, rewards, trust values, patch locations and
prefilter evidence are in `priority_seed_decision_chronology.csv`. The complete
twenty-seed raw primary table remains visible in `twenty_seed_primary_results.csv`.

| Original terminal update | Raw gradient norm | Clipped? | Objective before | Objective after full step | Objective at half-step | Offline L2 at half-step |
|---|---:|---|---:|---:|---:|---:|
| VARA 1 | 0.819354 | no | 0.000843 | 0.002000 | 0.000681 | 0.022344 |
| VARA 3 | 6.959694 | no | 0.018038 | 0.024267 | 0.000494 | 0.019759 |
| VARA 5 | 3.705921 | no | 0.005318 | 0.007116 | 0.000597 | 0.015968 |
| VARA 8 | 9.141904 | no | 0.032449 | 0.044519 | 0.000665 | 0.021957 |
| VARA 18 | 10.965668 | yes | 0.044648 | 0.022569 | 0.001363 | 0.040257 |
| VARA 2 | see exact CSV | see exact CSV | 0.000439 | 0.000440 | 0.000435 | 0.013012 |
| Vanilla 4 | 1.010634 | no | 0.000970 | 0.007127 | 0.001412 | 0.039349 |

For seed 1 the quarter-step objective is lower than the half-step objective.
For Vanilla 4 a quarter-step objective is 0.000530 and its offline L2 is
0.018891. The revision does not select these measured fractions or select a
final state by dense error: it tries a fixed conventional halving sequence at
**every** update and uses only the original allowed weighted training loss.

The allowed-data-only constant shifts change failed-seed L2 by approximately
+2.21%, -0.61%, -2.98%, -0.15% and +3.38% for 1/3/5/8/18 respectively.
Positive changes are worse. These negative diagnostic results are retained.
Allen–Cahn comparisons include seeds 0/1/2/4 and show why its reaction term
does not share AD's constant-offset invariance. They do not establish that a
reaction-term modification should be added to AD's governing equation.

## D. Causal attribution

H1: failure can arise without retained adaptive actions through neutral
sampler/optimizer dynamics. H2: retained actions induce harmful trajectories.
H3: locally beneficial actions and short proxy guards fail to protect later
field accuracy. These hypotheses are not mutually exclusive.

The new terminal measurements directly establish a local optimizer overshoot;
they do not distinguish the upstream H1 versus H2 trajectory cause. The
original short horizon and masking channels provide concrete vulnerabilities
consistent with H3, but a failed final metric alone does not prove a particular
accepted action caused it. Removing the sparse mismatch or adding analytical
field errors to decisions would change the information boundary and is not an
acceptable causal repair.

`scripts/run_ad_v2_forced_neutral.py` retains the original source, pool, sampler,
eligible probes and proposed-policy trust/memory updates, while forcing the
actual retained path to neutral. It requires the original 15-seed ZIP SHA,
matched initialization/data hashes, and bit-identical 500-step warmup before
continuing. Historical-source files are verified against the archive. If the
environment fails these checks, it aborts rather than relabeling an unmatched
run as a causal experiment. Its full seed-8 CUDA result is pending.

## E. Ranked implementation repairs and risks

1. **Shared Armijo safeguard (strongest new direct evidence).** One Adam state
   update per physical call, then test parameter fractions 1, 1/2, 1/4, 1/8,
   1/16 against `L(theta+alpha*d) <= L(theta)+1e-4*alpha*grad(L).d` on the same
   allowed batch and allocation. Use the true pre-clip gradient for the slope.
   If momentum is uphill, use an explicit clipped-gradient direction. If all
   tests fail, keep parameters unchanged and record a no-op; Adam moments still
   advance. Both arms use this identical rule. Additional objective evaluations
   and no-ops are reported; committed calls are not renamed as successful
   parameter changes. Cost: up to five extra objective evaluations per call.
   Risk: minibatch monotonicity is not a generalization guarantee; moment/step
   mismatch and no-ops can slow learning.
2. **Neutral sampler parity.** If sampling mass is uniform, call the exact
   direct-uniform Vanilla sampler with the same RNG. Nonuniform allocation uses
   the historical bounded mixture. Tests establish bitwise no-action equality.
   This removes an incidental baseline trajectory difference, not intended
   adaptive sampling. Risk: historical reproduction requires the old sampler;
   the causal launcher therefore deliberately retains it.
3. **Independent component guards.** Protect fixed-diagnostic unweighted PDE,
   BC, IC and sparse MSE individually, plus frozen-scale physics/all-component
   sums, using the original 2% matched-probe margin. Diagnostic scales are each
   channel's unweighted MSE at warmup completion, floored at 1e-12. Target patch
   scores are divided by frozen channel RMS. The scales never use held-out
   errors. Risks: low floors, differing sample coverage and conservative rejection.
4. **Continuation rechecks and faithful replay.** Every 50 continuation steps,
   compare each component to the same-block trained neutral 25-step anchor,
   allowing 2% plus absolute 1e-12. On violation restore neutral model, Adam,
   allocation and Python/NumPy/Torch/CUDA/sampler RNG states; replay consumed
   continuation on the saved neutral batch; discard candidate retained rows,
   preserve every physical audit, and disable the action until the next block.
   Trust/action memory receive a logged rejection update. These are conservative
   safety rechecks against a fixed anchor, **not** duration-matched long probes.
   Cost: diagnostic checks and replay; risks: rejecting useful actions due to
   neutral-anchor staleness, increased runtime, trajectories between checks.
5. **Candidate availability and calibrated trust.** One positive supported
   region per channel may propose two actions, up to eight candidates; only the
   top eligible candidate is probed. BC/IC/sparse weighting cannot acquire new
   labels. Original score-domain gradient screening and rank rules remain;
   the reward denominator alone becomes a per-action fractional EMA, initialized
   at twice the 0.005 margin, updated with rho=0.8 and observed fractional
   improvement clipped to [0,1], with prediction bounded [0.005,1]. This makes
   observed/predicted units compatible and expansion possible. Risks: sparse
   per-key observations, cold-start uncertainty, more gradient evaluations and
   changed rankings under frozen diagnostic scales.

Each feature has a separate `--disable` switch for an explicitly labeled
ablation. The frozen primary uses all features. These are method changes and
must be reported as the new revision, not described as unchanged historical V2.
No thresholds were selected from new primary GPU outcomes; none exist yet.

## F. Reproducibility and validation

Original source revision: `cdb27c8be0e681654d8c6c3005d9658c0ae6ee73`.
Authoritative 0–4 archive:
`D:\VARA-PINN original submission\supplement_advection_diffusion_fullguard_5seed.zip`,
SHA256 `4d86aef428651e152ed2dadb1d5d303957a72868de161a8a9328e372f064f6b6`.
It is byte-identical to the Downloads copy used in the prior audit; it is not
substituted by a newer experiment. The 5–19 archive SHA is
`07986bed095e25f4f9190e9009461af447ac9e08b60293c629b9e09a3c2d5701`;
the Allen–Cahn archive SHA is
`d53c023817278aac6fa9cc98d4294bf4018e5587d5f59e5acf50674fbb6d824a`.

`AD_DIAGNOSTIC_PLAN.md` records the offline measurements before they were made.
`scripts/analyze_ad_terminal_dynamics.py` reproduces them from existing archive
contents, never calling an optimizer. Exact quantities and source members are
in the evidence CSVs; provenance declares zero training calls.

The frozen primary retains the original equation/forcing, domain, Gaussian
parameters, five 96-wide tanh layers, float32, Adam LR 0.001, clip norm 10,
2048 collocation, 512 BC, 512 IC and 507 sparse points. The schedule remains
4000 committed calls =500+7*500, with 25-step probes and a 5x5x3 patch grid.
Sparse labels are continuous uniformly sampled manufactured-solution values,
not CFD. The nominal 2% is 507 relative to the 48x48x11 evaluation-grid
cardinality; it is **not** a subset of that grid. Forcing/BC/IC are prescribed
problem data; analytical held-out **errors** never enter decisions.

Lightweight tests: 20 targeted mechanism/statistics/integrity tests passed on
CPU. The final five-seed 12-step integration smoke validated all ten runs;
it is not primary evidence.
Tests cover overshoot backtracking, explicit no-ops/Adam advancement, reference
isolation while permitting manufactured forcing, initialization/data matching,
independent guard rejection, channel access, sampling conservation, trust
expansion, exact rollback/replay, physical calls and complete-block resume.
The packaging smoke checks initial-checkpoint hashes, input immutability,
raw/table traceability, inference-only figures, all-file checksums and the
standalone verifier. Scientific superiority remains untested.
Primary GPU outcomes are pending. See `AD_V2_STABILITY_EXECUTION.md` and the
completed Colab notebook for the unpublished-local-commit handoff.

## G. Publication-safe interpretation

“The original AD V2 cohort exhibited mixed seed-level outcomes and worse mean
full-field error. Offline reconstruction identified finite-step optimizer
overshoot and several controller vulnerabilities. A reference-free stability
revision with an identically safeguarded Vanilla comparator was frozen before
new primary measurements. Its comparative efficacy is pending.”

Do not claim guaranteed elimination of volatile held-out error, a VARA speedup,
statistical superiority, or a successful seed-8 causal diagnosis from these
offline measurements or small CPU tests. Report all future seeds and trade-offs,
including ties, no-ops, harmful actions, rollbacks and added compute.
