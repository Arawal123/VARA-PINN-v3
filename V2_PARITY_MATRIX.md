# V2 mechanism parity and intentional deviations

Allen-Cahn reference source cdb27c8be0e681654d8c6c3005d9658c0ae6ee73; NS scaffold base ee80c76c6e64a83eb7fa5365e743b2d9789a3c85.

| Mechanism | Allen-Cahn source | New Kovasznay implementation / declared difference |
|---|---|---|
| Controller | src/controllers/v2_controller.py:189 | Same VARAV2Controller, no mathematical calibration edits |
| Diagnosis | src/pde_generalization/diagnostics.py:122 | Same percentile90 / median-positive scheme; spatial16 patches vs spacetime75; continuity/momentum/BC/sparse channels |
| Selection | src/pde_generalization/trainer.py:244 | Same weak-region detector and candidate generation; one ranked eligible candidate per block |
| Gradient screen | src/pde_generalization/trainer.py:464 | Same cosine sign/prefilter; all parameters; NS sums distinct equation, BC and data components |
| Probes | src/pde_generalization/trainer.py:293 | Same matched25-step neutral/action probes; original NS loop can test multiple candidates and is deliberately not used |
| Guard | src/pde_generalization/trainer.py:524 | Same relative target >0.005, protected increases <=0.02; NS adds momentum/continuity to four common guard names; no IC for steady PDE |
| Restoration | src/pde_generalization/trainer.py:321,359 | Deep-copy model/Adam/allocation and Python/NumPy/Torch/CUDA RNG; rejection retains advanced neutral branch |
| Trust/memory | src/controllers/v2_controller.py:446 | Same bounds, reward ratio, shrink/expand and EMA; known expansion calibration weakness retained |
| Schedule | src/pde_generalization/trainer.py:208 | Same500+7x500, probe25 and475 continuation; no final repair or best-state selection |
| Loss | src/pde_generalization/losses.py:24,127 | Same mean of squared values, mean-one local multipliers; NS equation-specific terms and reference-free pressure gauge |
| Neutral sampling | src/pde_generalization/trainer.py:636; NS trainer:1722 | Uniform-state V2 exactly matches Vanilla sampling. This corrects a control-attribution confound and is explicitly a new-protocol deviation |
| Supervision | Allen-Cahn resolved configs:507 fixed scalar observations |1000 fixed predeclared u/v/p observations, equal across arms; separate pure-PINN sensitivity supported |
| Information boundary | src/pde_generalization/trainer.py:731 | Analytical BC/data labels frozen before training; no analytic calls during controller/training; dense fields evaluated only after final committed state |
| Compute | src/pde_generalization/trainer.py:412,457 | Exact retained steps plus separate physical calls, gradient/objective/point counts; raw discarded/replayed work preserved |

These establish mechanism comparability, not identical runners, PDE losses, diagnostic geometry, data budgets or guaranteed improvements. The independent PDE tests, reference traps, forced-neutral identity, rollback, resume and packaging tests provide executable evidence. A25-step probe does not guarantee475-step continuation or final held-out accuracy.
