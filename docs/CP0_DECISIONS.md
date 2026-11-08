# CP0 decision sheet (issue #21)

For a 30-45 minute call with Vishnu, Revanth and Siddharth. Each decision has the proposed
value, the options, the evidence and a recommendation. **Nobody should look at a new result
before these are fixed** (HANDBOOK §3 rule: gates are pre-registered). Status of every number
below is as of 2026-11-08, `main` at `cae9b56`.

Fill in the **Decided** line during the call. Afterwards Vishnu turns the outcome into one PR:
HANDBOOK §13 (D4, §13.3), `configs/x3.yaml` (`confirmed: true`), the TinyStories grid config,
`STRETCH_GOALS.md` wording, and a comment on #21. Then the X3 verdict is one command.

Order matters: **1 and 2 must be settled before the TinyStories grid runs**, or its rules are
no longer pre-registered. 3 must be settled before anyone runs `python -m src.evaluation.alarm`.

---

## 1. Confirmatory GATE 3 rule (HANDBOOK §13.3, #35)  | decide first

**Problem.** On WikiText, GATE 3 failed for a structural reason, not noise. 9 of 12
contaminated trajectories already collapse at the *first* contaminated ratio (0.25), so there is
no stage before onset to warn from. Every AUROC is 0.03-0.25. The same ratios on TinyStories fail
the same way.

**Proposed rule in §13.3 (needs numbers):** two-sided signals, direction fixed from WikiText
(anisotropy -1), negatives = `human_control` trajectories, pass = a signal beats the permutation
baseline at Bonferroni alpha 0.05 and has median lead > 0 stages at false-alarm rate <= 5%.

| Option | What it means | Cost |
|---|---|---|
| **A. Finer low ratios** `[0, 0.05, 0.1, 0.25, 0.5, 1]` | onsets land after stage 1, so there are stages to warn from | +1 ratio: ~15 more TinyStories cells, ~20 min |
| B. Training steps as stages | lead time is measured inside one run, from the X3 logs | no extra grid; but it is then X3's question, not GATE 3's |
| **C. A + two-sided signals** | A, and a signal may move either way | same as A |

Evidence: anisotropy separates the arms *at* onset (flat for control, falls for contaminated
types) but opposite to the pre-registered direction; effective rank and dead neurons do not
separate them. Source: PR #73.

**Recommendation: C.** Keep GATE 3 on the ratio axis (comparable with WikiText) and let X3 own
the training-step question. Numbers to fix: the ratios, alpha, the 5% false-alarm rate.

Decided: ______________________________________________

## 2. Label D4: what counts as "degraded" (§13 D4)

**Proposed:** a cell is degraded if its holdout perplexity exceeds its own seed's ratio-0 run by
more than the baseline spread (max - min of the ratio-0 baselines). Same floor as GATE 2/3, no
new threshold. It is the label the risk model and its conformal sets are trained on.

Evidence: the spread is estimated from **3 seeds** (0.48 ppl), which is fragile. The label is
already in use (`reports/risk/`), so changing it changes the risk results.

| Option | Note |
|---|---|
| **A. Keep the default** | consistent with GATE 2/3; fragile spread |
| B. A relative cut, e.g. +5% ppl | simple, but a new threshold to defend |
| C. Regress the continuous excess, classify afterwards | more work for Revanth; avoids a hard label |

**Recommendation: A** for the paper's first version; state the 3-seed caveat. Revisit if
TinyStories adds seeds.

Decided: ______________________________________________

## 3. Gate X3 (`configs/x3.yaml`, #37)

All values are PROPOSED (`confirmed: false`). The 64 runs are logged; **no verdict has been
computed**, and the command refuses until `confirmed: true`.

| Parameter | Proposed |
|---|---|
| Target false-alarm rate alpha | 0.05 (needs >= 19 healthy calibration runs; there are 20) |
| Pass: early-alarm rate | alarm strictly before the breach on >= 50% of breaching runs |
| Pass: observed false alarms | <= 10% on 16 held-out healthy runs |
| "Breach" | holdout-loss change above the healthy band by its own conformal threshold, one-sided |

Why breach is defined as a gap: on WikiText the holdout loss **never rises within a run**
(PR #98); every run improves, contaminated ones less. "Loss goes up" would never fire.

Prior art: the guarantee itself is not a novelty claim (trajectory-level split conformal exists
for diffusion models, 2610.34617). "When Rank Rises" calibrates on 2 seeds and false-alarms on
3 of 3 held-out healthy runs; we calibrate on 20.

Options: keep (**recommended**); tighten observed false alarms to <= 5%; or widen the collapsing
set. Whatever is chosen must be fixed now.

Decided: ______________________________________________

## 4. Gate X1 and what X1 claims (#62-#65, PR #122)

**Gate as written:** harm-guided recovery beats random by >= 10 points at ratio 0.5 on >= 2 of 3
collapse types, every seed.

**Pilot (1 seed, synthetic + recursive):** harm beats random by +36 and +39 points, so "go". But
**cleaning by origin** (`risk_reflm`) matches or beats harm (recovery 1.15 vs 1.03 on synthetic,
1.06 vs 0.97 on recursive) and sits near the oracle. On `benchmark_near` only the oracle undoes the
memorisation, and harm is the worst learned strategy (its AUROC for near-copies is 0.59;
`risk_knn` reaches 0.96 but is not a strategy). Holdout "recovery" is meaningless there (0.21 ppl
denominator): the metric must be benchmark-item perplexity.

The gate is passable, yet the claim X1 was adopted for, "clean by harm, not by origin", is not
supported so far, and Drayson et al. (EMNLP 2025) already show origin-based cleaning prevents
collapse.

| Option | Note |
|---|---|
| A. Keep gate and claim, run B9 | honest if harm loses; paper reports a negative |
| **B. Keep the gate; add a pre-registered secondary: harm vs origin cleaning, reported whatever the result; fix the benchmark_near metric; add `risk_knn` as a strategy** | claim becomes "cleaning works, perplexity filtering does not, and here is how harm compares" |
| C. Drop the harm-vs-origin framing now | smallest paper claim |

**Recommendation: B.** Decide before B9 runs (cut-off 2026-12-15).

Decided: ______________________________________________

## 5. Gate X2 and the GPU (#38, #39, #66, B2)

**Proposed:** Spearman rho >= 0.7 between proxy-predicted and target-measured damage across the
subset cells, bootstrap CI excluding 0. Target model Pythia-410M or Qwen2.5-0.5B on Siddharth's
GPU. Cut-off 2026-12-15, else drop X2 and put scale in the limitations.

To decide: **which target model**, **which GPU slots** (the booking calendar, B2, is not set up),
and whether the reduced-learning-rate proxy (ICLR 2026) is included. Nothing on X2 can start
without this.

Decided: ______________________________________________

## 6. Novelty claims changed by the second literature search (#112)

Adopt, amend or reject the revised wording in `docs/STRETCH_GOALS.md`:

- **Core:** "predict damage before training" is not new (TuneAhead, ICML 2026). Claim =
  contamination-caused degradation + conformal sets + a ground-truth testbed.
- **X1:** origin-based cleaning is a required baseline; claim = harm vs origin head-to-head on
  collapse damage.
- **X2:** forgetting already extrapolates across scale; claim = contamination damage with a
  calibrated interval.
- **X3:** the guarantee is an application, not a novelty; claim = LM training, enough healthy
  runs, recursive-synthetic collapse.

Decided: ______________________________________________

## 7. Risk model status (Revanth, #50, #51)

Facts for the agenda, no rule to fix: leave-one-type-out AUROC 0.84 [0.72, 0.94]; conformal
coverage **0.72 vs the 0.90 target**; the full model only **ties** the single reference-LM loss
feature (difference +0.001, CI [0, 0.008]), so H2 is not shown. Question: what is the plan, more
calibration data or a different conformal scheme, and does the paper claim H2?

Decided: ______________________________________________

## 8. Critical path and housekeeping

- **TinyStories prep (#58, Siddharth)** is not started and blocks the second grid, GATE 1, the
  confirmatory GATE 3 and the TinyStories features. Needs a date. Work after it is about 2 h of
  grid, 1 h of features, white-box on the checkpoints. **CP1 is Fri 2026-11-20.**
- GPU booking calendar (B2, #57).
- Stacked PRs: retarget to `main` after the base merges, before merging (it lost work twice).
- Weekly HOD update: each student fills hours and a learning line in `docs/hod/`.
- Who merges what, and review turnaround.

Decided: ______________________________________________

---

### After the call (Vishnu, same day)

1. One PR with the decisions: HANDBOOK §13/§11, `configs/x3.yaml`, TinyStories grid ratios,
   STRETCH_GOALS wording. Remove the PROPOSED markers.
2. Comment the outcome on #21 and close it when merged.
3. Run `python -m src.evaluation.alarm` (Gate X3 verdict).
4. Tell Siddharth the TinyStories date, then start #29.
