# Shared sections: introduction, related work, limitations (draft for T3, #23)

*Vishnu. Draft of 2026-11-08. Citations are by survey number (`docs/LITERATURE_SURVEY.md`, "#n")
or stretch-goal key (`docs/STRETCH_GOALS.md`, "[A6]"); they become a bibliography at merge.
Every result cites the track section it comes from (`track_a.md`, `track_b.md`, `track_c.md`).
Claims follow the *revised* novelty wording of #112, not the first survey's. Entries marked
TODO wait on results that do not exist yet; nothing here is a placeholder number.*

## 1. Introduction
Fine-tuning a language model on a dataset is cheap to start and expensive to undo. Today a team
learns whether a candidate dataset will degrade its model only by training on it, and a large
body of work shows the damage can be silent: recursive training on model output makes the tails
of the distribution disappear first (#1), and small amounts of synthetic data can already do
lasting harm (#4). Measuring it by brute force costs more than most teams can spend: the clearest
mixture study needed more than 1,000 models and 100k GPU-hours (#12).

Existing tools answer neighbouring questions. Detectors ask whether *a text* is machine-written
(#25-28) and benchmark-contamination methods ask whether *a benchmark* has leaked (#20, #23, #24).
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
- A dataset-level risk model reaches leave-one-type-out AUROC 0.84 [0.72, 0.94], but its
  conformal sets under-cover (0.72 against a 0.90 target) and it only ties the single best
  feature (Track A; #105). We report H2 ("a multi-feature model beats a simple detector") as
  *not shown*.
- White-box signals separate contaminated from clean runs (anisotropy falls, 0.935 → 0.902) but
  the early-warning gate on the ratio axis failed, for a structural reason (Track C §3-4).
- In the repair pilot, harm-guided replacement beats random by 36-39 points, but **cleaning by
  origin ties or beats it** (Track B §5; #122).

**Contributions.** We make these claims, narrowed by a second literature search to what we
could not find elsewhere:
- **A ground-truth testbed and benchmark (ORIGIN-T-Bench)** with a contamination gradient,
  hidden-holdout safeguards and released data (Track B §1-3).
- **Contamination-caused degradation, predicted with calibrated sets.** Predicting a fine-tune's
  outcome before training is published (TuneAhead, ICML 2026 [C1]); our claim is limited to
  damage caused by contamination, with conformal sets, measured against ground truth.
- **A live alarm with a stated false-alarm rate**, calibrated on 20 healthy runs (X3, pending).
- **A pre-registered, reported-as-written record**, including negative results: GATE 3's failure
  and the X1 pilot's finding that origin cleaning is a strong baseline.

## 2. Related work
**Model collapse.** Recursive training on generated text degrades models and loses distribution
tails first (#1, #2); the damage grows with the synthetic fraction and the generation depth
(#4, #6), and mixing in real data can bound it (#5, #16). Shi et al. recast collapse as a drift
from generalising to memorising the parent model (#7). Not all iteration collapses: under stated
conditions it stays consistent (#9). This work describes collapse; it does not predict it for a
given dataset. We reproduce it (GATE 0) and measure its dose-response.

**Synthetic-data mixing and mitigation.** Kang et al. map natural, rephrased and generated
mixtures at scale (#12); verification can rescue scaling (#14, #15). Du et al. find calibration
error rises before accuracy falls (#13), the closest prior work to our early-warning question,
but with one hand-picked metric and no guarantee. Cleaning by origin already works: a
machine-generated-text detector with importance resampling prevents collapse on GPT-2 and
SmolLM2 and can beat purely human data (#25, [A6]). This is why origin cleaning is a required
baseline for our harm-based repair, and in our pilot it is hard to beat.

**Benchmark contamination.** Surveys find no detector reliable across contamination tiers
(#18, #19, #21). Rephrased copies evade n-gram decontamination (#23), black-box tests exist for
verbatim leakage (#24), and behavioural detection compares scores with a reference benchmark
(#20). We reproduce the rephrasing failure with `benchmark_near`, and find that its harm shows
up in perplexity on the leaked items rather than in the holdout.

**Provenance detection.** Detectors are brittle under decoding changes and attacks (#26) and a
simple fine-tuned baseline is competitive while distribution shift stays unsolved (#27, #28).
Our feature suite does not try to out-detect them. It asks whether dataset-level signals
predict *damage*, and finds that a single reference-LM loss feature carries most of the signal.

**Data attribution and repair.** Influence functions are fragile for LLMs and a representation
baseline can beat them ([A1], [A2]); LoRA-gradient selection is cheap (LESS), and attribution-guided
removal followed by retraining exists for *safety* degradation ([A7], [A8], [A9]). Our X1
applies the recipe to collapse and contamination damage with ground-truth labels and compares it
head to head with origin cleaning.

**Predicting fine-tuning outcomes and scale transfer.** TuneAhead forecasts a fine-tune's score
from dataset descriptors and a short probe run, but not harm, and without calibration [C1].
Forgetting during fine-tuning extrapolates across scale to within 1-2% [P5]; proxy models can
rank datasets differently from their target unless the learning rate is reduced (ICLR 2026,
[P1]). X2 asks the contamination version of this question, with an interval.

**Calibration and early warning.** Split conformal prediction gives distribution-free coverage
(Angelopoulos and Bates, 2021) and trajectory-level guarantees exist (arXiv 2610.34617, for
diffusion models), so ours is an application, not a novelty claim. Discrete-time survival models
(#29) and time-aware state-space models (#30) are the planned extensions and are gated on the
risk model being sound (README feature 6).

**Gap.** No system we found takes a dataset as input and returns a calibrated, contamination-
specific degradation risk, nor a live alarm with a stated false-alarm rate for recursive
collapse (`docs/LITERATURE_SURVEY.md` §5; `STRETCH_GOALS.md`). We re-ran the search on
2026-11-02 and narrowed the claims accordingly; it is not a systematic review, and it should
be repeated on Semantic Scholar for anything dated after that day before submission. **TODO.**

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
- **The risk model is not yet sound.** Conformal coverage is 0.72 against 0.90, and it ties a
  single feature. We do not claim a calibrated guarantee for the dataset-level risk (Track A).
- **Early warning:** GATE 3 failed on its pre-registered rule, the anisotropy finding runs
  opposite to the registered direction and is exploratory, and the confirmatory run is pending.
  Within a run the loss never rises, so "breach" means a gap from healthy runs and needs
  healthy reference runs.
- **Repair:** harm-guided repair has not been shown to beat origin cleaning, and benchmark_near
  needs a metric on leaked items. Gate X1 is not evaluable until the full grid (B9).
- **Environment:** pools are not reproducible across GPUs (sampling), so the bench ships the
  data the grid trained on rather than a regeneration recipe.

## 4. Open items for the merged draft
- [ ] Abstract and conclusion (after CP0 verdicts, X3 and the TinyStories grid).
- [ ] Convert survey numbers and stretch-goal keys to one bibliography; verify every entry
  marked preprint (the survey's own honesty note).
- [ ] Repeat the literature search for work after 2026-11-02.
- [ ] Update §1 "what we found" after each pending gate; every number must trace to a committed
  `reports/` file.
