# Track C: the collapse grid, white-box signals and the live alarm (draft for T3, #23)

*Vishnu. Every number cites the committed file or PR it comes from. Draft of 2026-11-08,
WikiText-2 only. Sections marked PENDING wait on the CP0 decisions (`docs/CP0_DECISIONS.md`),
the TinyStories grid (#58) or the X2 target model; no verdict in them has been computed.*

## 1. Experimental setup
Every cell fine-tunes DistilGPT-2 (82M) with LoRA (r = 8, lr 2e-4, 1 epoch, batch 8, max length
256) on 1,000 samples, and is scored on 200 hidden-holdout texts that no pipeline stage may
train on (`reports/grid/summary.json`, config fingerprint `d497f2268060`). A cell is one
(contamination type, ratio, seed). The grid is 5 types × 5 ratios {0, 0.25, 0.5, 0.75, 1} × 3
seeds, with the ratio-0 run shared across types, so the 75 cells are 63 distinct trainings. A
changed config key changes the fingerprint, and cached pools are versioned
(`grid.POOL_VERSIONS`), so a curve can never mix two generators. How each type is made is in
Track B §1.

**GATE 0 (spike).** Before building the grid, we checked that the phenomenon exists at this
scale: recursive training on the model's own output (generations 0-3) raises hidden-holdout
perplexity on every seed, 56.1 / 56.6 / 56.3 → 67.6 / 67.7 / 67.4
(`reports/collapse_spike/metrics.json`, #8).

## 2. Dose-response (GATE 2)
**Pre-registered rule** (TRD §11): for each collapse type (synthetic, recursive, paraphrased),
mean holdout perplexity is non-decreasing in the ratio (tolerance 0.5%), and at ratio 0.25 it is
already above the noise floor, the spread of the three ratio-0 runs (56.39-56.87, 0.48 ppl).

**Result: passed on all three** (`reports/grid/summary.json`, `gate.status = passed`; #11, #12).

| type | ratio 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| synthetic | 56.60 | 57.91 | 59.59 | 62.58 | 67.58 |
| recursive | 56.60 | 58.07 | 59.93 | 63.25 | 68.41 |
| paraphrased | 56.60 | 57.58 | 59.09 | 61.62 | 65.38 |
| benchmark_near | 56.60 | 56.84 | 56.95 | 57.37 | 57.96 |
| human_control | 56.60 | 56.55 | 56.42 | 56.47 | 56.39 |

(Mean hidden-holdout perplexity over 3 seeds; `reports/figures/collapse_curves.png`.)

- Recursive is the worst at every ratio, and damage is convex in dose: the first 25% costs
  about 1.0-1.5 points, the last 25% about 4-5.
- **human_control is flat** (56.6 → 56.4). The healthy arm swaps in the same amount of *fresh
  human* text, so the damage comes from the contamination, not from changing the training set.
- **benchmark_near barely moves the holdout (+2.4% at ratio 1) but is visible on the items it
  leaks**: benchmark-item perplexity falls 61.31 → 59.24 (−3.4%) with dose. It is the one type
  where the damage is memorisation, not degradation, so it needs a metric on the leaked items
  (Track B §4).
- Limit: three seeds per cell. The noise floor is a 3-run spread and is itself fragile (CP0
  decision 2).

## 3. White-box signals on the checkpoints
For each of the 66 grid checkpoints we measure, on the hidden-holdout probe texts
(`src/evaluation/whitebox.py`, `reports/grid/whitebox_summary.json`): effective rank and
participation ratio of the last-layer representations, anisotropy, the fraction of dead MLP
units, the stable rank of the attention weights, and the relative norm of the update
(`delta_rel_norm`).

At ratio 1, mean of 3 seeds:

| signal | clean | synthetic | recursive | paraphrased | human_control |
|---|---|---|---|---|---|
| anisotropy_last | 0.935 | 0.903 | 0.902 | 0.917 | 0.936 |
| delta_rel_norm (×10⁻³) | 4.9 | 4.1 | 4.1 | 4.5 | 5.0 |
| eff_rank_last | 371.8 | 371.5 | 372.1 | 372.7 | 369.8 |

- **Anisotropy and update norm separate the arms; effective rank does not.** Anisotropy is flat
  for the control and falls with contamination, ordered by dose (0.935 → 0.926 → 0.902 for
  recursive at ratios 0 / 0.5 / 1). The popular "rank collapse" signal, effective rank, stays
  near 370 for every type, so at this scale it is *not* a collapse signature.
- The direction is the opposite of the one pre-registered for GATE 3 (a *rise* in anisotropy,
  from the collapse literature). It is therefore fixed from this data for the confirmatory run,
  and the finding is exploratory (D7).

## 4. Early warning on the ratio axis (GATE 3), a negative result
**Pre-registered rule:** some signal, thresholded on healthy runs, warns at a ratio *before* the
ratio where holdout perplexity first exceeds the noise floor, with AUROC above a permutation
baseline at Bonferroni-corrected α = 0.05/6.

**Result: failed** (`reports/grid/earlywarning.json`, `status = failed`; #73). All six signals
have AUROC 0.03-0.25 (anisotropy 0.22, 95% CI [0, 0.62]) and permutation p between 0.95 and 1.0.

**Why it is structural, not noise.** 9 of the 12 contaminated trajectories already collapse at
the first contaminated ratio (0.25), so on the ratio axis there is no stage *before* onset to
warn from; the three never-collapsing trajectories are the control arm. This does not say the
signals carry no information: they separate the arms *at* onset (§3), in the wrong direction for
the rule. We report the failure as written and do not re-score it with the new direction.

**Confirmatory run: failed** (#35, `reports/grid_tinystories/earlywarning.json`). The rule
(HANDBOOK §13.3, merged before the verdict) uses ratios {0, 0.05, 0.1, 0.25, 0.5, 0.75, 1},
two-sided signals, `human_control` negatives, a median lead and Bonferroni α = 0.05/6, on
TinyStories. Of 15 trajectories, 3 never collapse. No signal passes: AUROC 0.08-0.56 and
permutation p 0.42-0.99 (best: `delta_rel_norm`, AUROC 0.56, p 0.42). Anisotropy, the signal
that separated the arms on WikiText, has AUROC 0.17. We report it as written. The TinyStories
grid itself ran before the rule merged (HANDBOOK §13.3 discloses this), but no GATE 3 output
was read before then.

## 5. X3: a live alarm with a calibrated false-alarm rate
**Question.** Can a monitor read signals *during* training and alarm before the damage is
visible, at a known false-alarm rate?

**Logging.** `TrainLogger` (`src/evaluation/live.py`) writes the six signals plus holdout loss
every 10 steps on a fixed probe (the hidden-holdout texts), as `trainlog/1` rows. It is
bit-for-bit neutral: the 24 logged runs that are grid cells reproduce the grid's holdout
perplexity exactly (#98), so monitoring does not perturb what it monitors. 67 runs are logged
(`reports/trainlogs/`).

**What the logs show.** Within a run the holdout loss **never rises**: every run improves,
contaminated ones less than healthy ones (#98). Damage is a *gap from healthy runs*, not a turn
in the curve, so "alarm when loss goes up" would never fire. The signals move earlier:
anisotropy separates the arms by step 20 of 125.

**Method** (`src/evaluation/alarm.py`; HANDBOOK §10.4).
1. A healthy band per signal and step (median and spread of the change since step 0) from 10
   clean runs.
2. A run's score is its largest two-sided deviation over all signals and steps.
3. **Split conformal on 20 separate healthy runs** sets the threshold, so a healthy run ever
   alarms with probability ≤ α = 0.05 (this needs ≥ 19 calibration runs).
4. A **breach** is the same machinery on the holdout loss, one-sided: the first step at which a
   run's loss change is above the healthy band by the loss's own threshold.
5. The alarm latches and runs live through `TrainLogger(monitor=...)`.

**Gate X3** (`configs/x3.yaml`, confirmed 2026-11-08): alarm strictly before the breach on ≥ 50% of
breaching runs, and an observed false-alarm rate ≤ 10% on 16 held-out healthy runs. The 18
collapsing runs are synthetic, recursive and paraphrased at ratios 0.5 and 1.

**Result: failed** (`reports/x3/gate_x3.json`, WikiText). The alarm fired before the breach on
8 of 18 collapsing runs (44%, bar 50%), with a median lead of 10 steps; 1 of 16 held-out healthy
runs false-alarmed (6%, bar 10%, passed). Every miss is a ratio-1 run: its breach is at step 10,
the first logged step, so no lead was possible at a 10-step log interval. The ratio-0.5 runs are
warned early. The verdict stands as pre-registered. A finer log interval would be a new
experiment with its own registered rule. TinyStories logs exist but have no registered X3 rule.

**Relation to prior work.** A split-conformal guarantee on trajectories exists for diffusion models [jing26], so the guarantee is an application, not a novelty claim. What is
specific here is language-model fine-tuning, a calibration set of 20 healthy runs (a comparable
method [wang26rank] calibrates on 2 and false-alarms on 3 of 3 held-out healthy runs) and recursive-synthetic
collapse (`docs/STRETCH_GOALS.md`, #112).

## 6. X2: does a small model's damage predict a bigger one's? (passed)
Proxy: the DistilGPT-2 WikiText grid. Target: Pythia-410M, plain LoRA on the same pools
(`pools_from`), 4 types × ratios {0, 0.5, 1} × 3 seeds (`reports/x2/`). Damage is holdout
perplexity over the same seed's ratio-0 perplexity, minus 1. **Result: Spearman ρ = 0.78,
95% bootstrap CI [0.51, 0.91] on 24 matched pairs** (`reports/x2/gate_x2.json`; bar ρ ≥ 0.7
with the CI above 0): passed. Caveats: the pairing rule was ours (the handbook did not fix it),
dose drives much of the rank, and the reduced-learning-rate proxy was not run.

## 7. Infrastructure that makes the numbers traceable
- **One config fingerprint** covers every key that changes a result, so results from two
  configs cannot be mixed (`reports/grid/summary.json`).
- **Frozen interfaces** (HANDBOOK §7) connect grid, features, scores, risk and repair. The API
  (v1.4.0, additive only) serves every committed report and the live stream to the console.
- **Reproducibility:** dependencies are pinned (`requirements.lock.txt`), CI checks that the
  OpenAPI document is byte-identical on Python 3.11 and 3.13, and since sampling is not
  reproducible across GPUs, generated pools are cached and the bench refuses to regenerate a
  missing one.

## 8. Limitations (Track C)
- **Scale.** DistilGPT-2 and 1,000-sample sets; no claim about large models until X2 reports.
- **Three seeds per grid cell**, so the noise floor comes from three runs. X3 compensates with
  20 calibration runs, but only for one dataset and one training recipe.
- **One dataset so far.** Every Track C result is WikiText-2 until the TinyStories grid (#58,
  #29) runs; GATE 1 is *not evaluable* before then.
- **GATE 3 failed twice:** on its WikiText rule and on the confirmatory TinyStories rule. The
  signals separate the arms at onset but give no early warning on the ratio axis.
- **One epoch, LoRA.** Full fine-tuning or more epochs could change when the signals move.
- **The loss never rises within a run**, so "breach" means "gap from healthy runs". That is a
  property of this recipe, and the alarm needs healthy reference runs to exist.

## 9. Figures and tables for the merged draft
- [x] Collapse curves (`reports/figures/collapse_curves.png`), white-box signals
  (`whitebox_signals.png`), GATE 0 spike (`collapse_spike.png`).
- [ ] X3 (verdict is in): signal trajectories with alarm and breach marked, and a lead-time histogram (after CP0
  and `python -m src.evaluation.alarm`).
- [ ] TinyStories curves and the confirmatory GATE 3 table (after #29).
- [ ] X2: proxy-vs-target scatter with Spearman ρ (after #38-#39).
