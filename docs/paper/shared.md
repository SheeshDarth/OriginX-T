# Shared sections: introduction, related work, limitations (draft for T3, #23)

*Vishnu. Draft of 2026-11-08. Citations are keys into `docs/paper/bibliography.md` ("[shumailov24]"),
which records each entry's verification status.
Every result cites the track section it comes from (`track_a.md`, `track_b.md`, `track_c.md`).
Claims follow the *revised* novelty wording of #112, not the first survey's. Entries marked
TODO wait on results that do not exist yet; nothing here is a placeholder number.*

## Abstract
Teams learn whether a dataset will degrade their model only by fine-tuning on it. We build ORIGIN-T,
a testbed with ground truth for this question: contaminated datasets (synthetic, recursive,
paraphrased, benchmark-near and duplicate text mixed into human text at known ratios), fine-tuning
runs over a ratio grid on two datasets (WikiText-2, TinyStories; DistilGPT-2 with LoRA), and
pre-registered go/no-go gates fixed before their results existed. Damage is real and dose-dependent
on both datasets, and caused by the contamination: on WikiText-2, hidden-holdout perplexity rises
from 56.6 to 67.6 (synthetic) and 68.4 (recursive) at full contamination while a human-text control
stays flat. A dataset-level risk model reaches a leave-one-type-out AUROC of 0.83 on both datasets together
(GATE 1 passed), but its conformal sets under-cover (0.81 against 0.90) and, within a single
corpus, it is no better than one reference-LM loss feature, so we do not claim that a
multi-feature model beats a simple detector. White-box training signals separate
contaminated from clean runs, yet the early-warning gate failed on both datasets, and a live,
conformally calibrated alarm met its false-alarm bar (6%) but missed its early-alarm bar (44%
against 50%). A small model's damage does predict a larger model's: Spearman rho 0.78, 95% CI
[0.51, 0.91], against Pythia-410M. In a full repair grid replicated on independent pools, harm-guided
repair beats random (Gate X1) but does not beat cleaning by origin, and on leaked benchmark items it
is the worst strategy.
We report the failed gates as written.

## 1. Introduction
Fine-tuning a language model on a dataset is cheap to start and expensive to undo. Today a team
learns whether a candidate dataset will degrade its model only by training on it, and a large
body of work shows the damage can be silent: recursive training on model output makes the tails
of the distribution disappear first [shumailov24], and small amounts of synthetic data can already do
lasting harm [dohmatob24strong]. Measuring it by brute force costs more than most teams can spend: the clearest
mixture study needed more than 1,000 models and 100k GPU-hours [kang25].

Existing tools answer neighbouring questions. Detectors ask whether *a text* is machine-written
[drayson25; dugan24; zhou25mgt; shen26] and benchmark-contamination methods ask whether *a benchmark* has leaked [dekoninck24; yang23; oren23].
Neither is calibrated against the thing a practitioner cares about: how much *this dataset* will
degrade *this model*.

**ORIGIN-T** treats this as a prediction problem on a testbed where the answer is known. We
build contaminated datasets with ground-truth origin labels (synthetic, recursive, paraphrased,
benchmark-near, duplicate), fine-tune a model on each over a grid of contamination ratios, and
measure the damage. On top of that testbed we study four questions, each with a pre-registered
go/no-go gate fixed before its result existed:

1. **Does contamination cause measurable, dose-dependent damage?** (GATE 0, GATE 2)
2. **Can the damage be predicted from the dataset, before training, with a calibrated risk and a
   conformal guarantee?** (Track A; GATE 1)
3. **Can it be seen early in the training signals of the model itself?** (GATE 3, X3)
4. **Can the dataset be repaired, and does cleaning by harm beat cleaning by origin?** (X1)

**What we found so far** (WikiText-2, DistilGPT-2 + LoRA, 3 seeds; one dataset, so every claim
is provisional until the TinyStories grid):
- Damage is real, convex in dose and caused by the contamination: hidden-holdout perplexity
  rises from 56.6 to 67.6 (synthetic), 68.4 (recursive) and 65.4 (paraphrased) at ratio 1, while
  the human-control arm stays at 56.4 (Track C §2; GATE 2 passed).
- A reference language model finds generated text per sample (AUROC 0.98 for recursive), and a
  nearest-neighbour score finds leaked near-copies (0.97) (Track A §1; #92).
- A dataset-level risk model reaches leave-one-type-out AUROC 0.83 [0.76, 0.90] across
  WikiText-2 and TinyStories (GATE 1 passed), but its conformal sets under-cover (0.81 against a
  0.90 target), it misses `benchmark_near`, and within each corpus it is within 0.03 AUROC of
  the single best feature (Track A §4b; #140). We report H2 ("a multi-feature model beats a simple detector") as
  *not shown*.
- White-box signals separate contaminated from clean runs (anisotropy falls, 0.935 → 0.902) but
  the early-warning gate failed on WikiText and again on the confirmatory TinyStories run
  (AUROC 0.08-0.56; Track C §3-4).
- The live alarm **failed its gate**: early on 8 of 18 collapsing runs (44%, bar 50%) with 6%
  false alarms; the misses are runs already breached at the first logged step (Track C §5).
- A small-model canary predicts a larger one's damage: Pythia-410M against the DistilGPT-2
  proxy, Spearman ρ 0.78, CI [0.51, 0.91] (Track C §6; X2 passed).
- In the repair pilot, harm-guided replacement beats random by 36-39 points, but **cleaning by
  origin ties or beats it** (Track B §5; #122). The full grid, replicated on independent pools,
  agrees: Gate X1 passes (harm − random 0.25-0.55 on all 9 ratio-0.5 cells), harm − origin is
  −0.03, CI [−0.07, +0.01], and on benchmark near-copies harm-guided repair recovers nothing
  (−0.02) while the near-duplicate score matches the oracle (Track B §5; #136).

**Contributions.** We make these claims, narrowed by a second literature search to what we
could not find elsewhere:
- **A ground-truth testbed and benchmark (ORIGIN-T-Bench)** with a contamination gradient,
  hidden-holdout safeguards and released data (Track B §1-3).
- **Contamination-caused degradation, predicted with calibrated sets.** Predicting a fine-tune's
  outcome before training is published (TuneAhead [luo26tuneahead]); our claim is limited to
  damage caused by contamination, with conformal sets, measured against ground truth.
- **A live alarm with a stated false-alarm rate**, calibrated on 20 healthy runs. It met the
  false-alarm bar (6%) and missed the early-alarm bar (44%), reported as a failed gate (X3).
- **A pre-registered, reported-as-written record**, including negative results: GATE 3's failure
  and the X1 pilot's finding that origin cleaning is a strong baseline.

## 2. Related work
**Model collapse.** Recursive training on generated text degrades models and loses distribution
tails first [shumailov24; alemohammad23]; the damage grows with the synthetic fraction and the generation depth
[dohmatob24strong; seddik24], and mixing in real data can bound it [gerstgrasser24; kazdan24]. Shi et al. recast collapse as a drift
from generalising to memorising the parent model [shi25]. Not all iteration collapses: under stated
conditions it stays consistent [barzilai25]. This work describes collapse; it does not predict it for a
given dataset. We reproduce it (GATE 0) and measure its dose-response.

**Synthetic-data mixing and mitigation.** Kang et al. map natural, rephrased and generated
mixtures at scale [kang25]; verification can rescue scaling [feng24; yi25]. Du et al. find calibration
error rises before accuracy falls [du25], the closest prior work to our early-warning question,
but with one hand-picked metric and no guarantee. Cleaning by origin already works: a
machine-generated-text detector with importance resampling prevents collapse on GPT-2 and
SmolLM2 and can beat purely human data [drayson25]. This is why origin cleaning is a required
baseline for our harm-based repair, and in our pilot it is hard to beat. An epidemiological model of contamination likewise finds detection-based
filtering among the most effective interventions [wang26sir].

**Benchmark contamination.** Surveys find no detector reliable across contamination tiers
[nourbakhsh26; xu24bdc; fu25]. Rephrased copies evade n-gram decontamination [yang23], black-box tests exist for
verbatim leakage [oren23], and behavioural detection compares scores with a reference benchmark
[dekoninck24]. We reproduce the rephrasing failure with `benchmark_near`, and find that its harm shows
up in perplexity on the leaked items rather than in the holdout.

**Provenance detection.** Detectors are brittle under decoding changes and attacks [dugan24] and a
simple fine-tuned baseline is competitive while distribution shift stays unsolved [shen26].
SynthSentry scores an unlabeled corpus for synthetic contamination before training from
diversity and perplexity-spread measures, but does not tie the score to damage [myakala26].
Our feature suite does not try to out-detect them. It asks whether dataset-level signals
predict *damage*, and finds that a single reference-LM loss feature carries most of the signal.

**Data attribution and repair.** Influence functions are fragile for LLMs and a representation
baseline can beat them ([li24influence; luo26reweight]); LoRA-gradient selection is cheap [xia24less], and attribution-guided
removal followed by retraining exists for *safety* degradation [reddy26; li25rept; unverified-A9]; attribution also removes the effect only unreliably in a subliminal-learning
test [weckbecker26]. Our X1
applies the recipe to collapse and contamination damage with ground-truth labels and compares it
head to head with origin cleaning.

**Predicting fine-tuning outcomes and scale transfer.** TuneAhead forecasts a fine-tune's score
from dataset descriptors and a short probe run, but not harm, and without calibration [luo26tuneahead].
Forgetting during fine-tuning extrapolates across scale to within 1-2% [bethune25]; proxy models can
rank datasets differently from their target unless the learning rate is reduced (ICLR 2026) [wang26proxy]. X2 asks the contamination version of this question, with an interval.

**Calibration and early warning.** Split conformal prediction gives distribution-free coverage
[angelopoulos21] and trajectory-level guarantees exist [jing26] (for
diffusion models), so ours is an application, not a novelty claim. Sequential conformal change detectors with
optimal delay [bhattacharyya26] are the stronger baseline for the alarm and a natural extension. Discrete-time survival models
[wiegrebe24] and time-aware state-space models [zhou26mamba] are the planned extensions and are gated on the
risk model being sound (README feature 6).

**Gap.** No system we found takes a dataset as input and returns a calibrated, contamination-
specific degradation risk, nor a live alarm with a stated false-alarm rate for recursive
collapse (`docs/LITERATURE_SURVEY.md` §5; `STRETCH_GOALS.md`). We re-ran the search on
2026-11-02 and again on 2026-11-09 (`docs/LITERATURE_REFRESH.md`): nothing dated after the first
search was found and the claims stand, though a pre-training contamination score exists
(SynthSentry) and our claim is the calibrated *damage* prediction. It is not a systematic review;
repeat it on Semantic Scholar shortly before submission.

## 3. Limitations (shared)
These are the cross-cutting limits. Each track section lists its own.

- **One model family, one scale, one recipe.** DistilGPT-2 (82M), LoRA, one epoch, 1,000-sample
  sets. We make no claim about large models until X2 reports (pending), and full fine-tuning or
  longer training could change when the signals move.
- **One dataset so far.** Every result is on WikiText-2. GATE 1 requires two datasets and is
  *not evaluable* before the TinyStories grid (#29, blocked on #58).
- **Three seeds per cell**, so the noise floor, the degradation label (D4) and every interval
  rest on small samples. X3 uses 20 calibration runs, for one dataset and one recipe.
- **Synthetic generators are weak.** The generating model is DistilGPT-2, and "paraphrased" is
  closer to prompted continuation because the base model cannot follow the rewrite instruction
  (Track B §6). Results for real paraphrase or frontier-model text are untested.
- **Contamination is injected at known ratios into a clean base.** Real datasets mix sources
  unevenly and the ratio is unknown, so deployment would need an estimate, as repair budget
  *k* does too.
- **The risk model is not yet sound.** Conformal coverage is 0.81 against 0.90, the ranking
  transfers across datasets but the level does not (leave-one-dataset-out coverage 0.50), and
  within a corpus it matches a single feature. We do not claim a calibrated guarantee for the
  dataset-level risk (Track A §4b).
- **Early warning:** GATE 3 failed on its pre-registered rule and on the confirmatory run, and
  the anisotropy finding runs opposite to the registered direction and is exploratory.
  Within a run the loss never rises, so "breach" means a gap from healthy runs and needs
  healthy reference runs.
- **Repair:** harm-guided repair does not beat origin cleaning, and each score cleans only the
  contamination it detects. The full-grid result is a replicate on regenerated pools; the run on
  the grid's own training sets would confirm it (#64). At ratio 1 every strategy replaces the
  whole set, so only ratio 0.5 compares strategies.
- **Environment:** pools are not reproducible across GPUs (sampling), so the bench ships the
  data the grid trained on rather than a regeneration recipe.

## 4. Conclusion
ORIGIN-T shows, on a testbed where the answer is known, that contamination causes dose-dependent
damage that a small proxy model ranks well (X2 passed) and that a reference language model detects
per sample. It also shows what does not yet work: the dataset-level risk model is not calibrated
(0.81 coverage against 0.90) and, within a corpus, does not beat its best single feature, white-box signals give no
early warning on the contamination-ratio axis (GATE 3 failed twice), and the live alarm misses
too many collapsing runs at the 10-step logging resolution (X3 failed), although its false-alarm
rate held. In the full repair grid, harm-guided repair beats random but not cleaning by origin, and
protecting rare clusters changes nothing measurable. We kept the rules fixed before the results and report the failures with the successes;
that record, and the released data and code, are the contribution. The limits are scale (one
82M model, one 410M target), one training recipe and weak synthetic generators. Next: the repair
grid on the grid's own training sets, more calibration data for the risk model, and a
finer-grained alarm registered as a new experiment.

## 5. Open items for the merged draft
- [x] Abstract and conclusion (2026-11-09, from the CP0 verdicts; X1 full grid added 2026-11-10).
- [x] One merged draft: `python docs/paper/build.py` writes `docs/paper/draft.md` from the
  section files (2026-11-10).
- [ ] Convert survey numbers and stretch-goal keys to one bibliography; verify every entry
  marked preprint (the survey's own honesty note).
- [x] Repeat the literature search for work after 2026-11-02 (2026-11-09, `docs/LITERATURE_REFRESH.md`; repeat once more before submission).
- [ ] Update §1 "what we found" after each pending gate; every number must trace to a committed
  `reports/` file.
