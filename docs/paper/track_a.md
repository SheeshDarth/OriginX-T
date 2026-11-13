# Track A: predicting damage from the data (draft for T3, #23)

*Revanth. Every number cites the committed file it comes from. Draft of 2026-11-01 (WikiText-2); §4b added 2026-11-11 (both datasets, GATE 1).*

## 1. Features
Each grid cell's exact training set (1000 samples, rebuilt from the cached pools by
`grid.training_set`) is described by 22 features in four families, through one interface,
`family(texts, ctx) -> dict` (`src/features/suite.py`):
- **lex** (hand-rolled, standard library): unigram entropy, type-token ratio, distinct-1/2/3,
  within-text repeated 3-grams, burstiness of word repeats, and length mean/variance.
- **leak:** each text's best `difflib` similarity to the 200 benchmark items (the max, the mean,
  and the share above 0.9). It's case- and whitespace-blind, the same measure the generator's
  ground truth uses.
- **model** (Track C): loss and token-rank statistics under a fixed GPT-2.
- **sem** (Track C): embedding crowding, cluster entropy, tail retention and MAUVE against the
  seed's clean human draw.

All 75 cells have every feature (`reports/features/wikitext2.jsonl`, `lex@1+leak@1+model@1+sem@2`).

## 2. Label and evaluation design
**Label (D4, default).** A cell is *degraded* if its hidden-holdout perplexity exceeds its own
seed's ratio-0 run by more than the spread of the ratio-0 runs across seeds: GATE 2's noise
floor, with no new threshold (`src/risk/classifier.py: degraded_labels`). On WikiText, 43 of
the 60 contaminated cells are degraded.

**Why held-out types.** Damage rises with the contamination ratio by construction, so a model
that only measures "how contaminated" looks good in-distribution for free. Every reported
prediction is therefore **leave-one-type-out**: the model never trained on the contamination
type it scores. A held-out type's ratio-0 rows duplicate training rows (every type shares
them), so they're excluded from its test fold.

## 3. Models
- **Logistic regression, hand-rolled** (`LogReg`): features standardised on the training fold,
  missing values filled with the training mean, full-batch gradient descent on mean log loss
  + (λ/2)‖w‖², λ = 0.01, run until every gradient component is below 10⁻⁶. It matches
  scikit-learn's solver to every reported digit (`reports/risk/crosscheck.json`).
- **Split conformal, hand-rolled** (`src/risk/conformal.py`):
  - nonconformity s = 1 − p(true class);
  - q is the ⌈(n+1)(1−α)⌉-th smallest calibration score;
  - the set is every label with 1 − p(y) ≤ q.

  α = 0.1. In each fold a third of the training side calibrates (20 cells) and is never fitted on.
- **Baseline (§8.2 b):** mean GPT-2 loss alone, with its direction learned per fold. It's
  compared with a paired bootstrap of the AUROC difference (`src/risk/baselines.py`).

## 4. Results (WikiText-2, leave-one-type-out, n = 60, 95% bootstrap CIs)
Source: `python -m src.risk.classifier --datasets wikitext2` (the committed `reports/risk/` held these
until 2026-11-11, `main` at `c3f409a`; it now holds the two-dataset run of §4b).

| Metric | Value |
|---|---|
| AUROC | **0.84** [0.71, 0.94] |
| AUPRC | 0.95 [0.90, 0.99] |
| ECE | 0.22 |
| Conformal coverage (target 0.90) | **0.72** [0.60, 0.83], mean set size 0.87 |
| vs mean-GPT-2-loss baseline (AUROC 0.84) | difference +0.001 [0.000, 0.008]: **ties, does not beat** |

| Held-out type | Degraded | Mean p(degrade) | Coverage |
|---|---|---|---|
| recursive | 12/12 | 1.00 | 1.00 |
| synthetic | 12/12 | 0.78 | 0.50 |
| paraphrased | 12/12 | 0.65 | 0.67 |
| human_control | 0/12 | 0.05 | 1.00 |
| benchmark_near | 7/12 | **0.00** | 0.42 |

**Reading:**
1. **The model transfers between model-generated contamination types.** Trained without recursive
   text, it still scores recursive cells at p = 1.00, and it never flags the healthy control.
2. **It cannot transfer to benchmark leakage.** That damages a model through a different
   mechanism: leaked texts are copies of human text, so the lexical, LM and semantic features
   read them as human. When benchmark_near is held out, no training type has a nonzero leak
   feature, so the model can't learn that leak features matter. This is why the audit (§5)
   flags leaks with a rule rather than relying on the score.
3. **Coverage of 0.72 is below the 0.90 target.** That's expected rather than a failure of the
   method: the guarantee needs calibration and test cells to be exchangeable, and holding out a
   type breaks that on purpose.
4. **H2 is not supported on WikiText.** The multi-feature model ties mean GPT-2 loss (+0.001, CI
   [0.000, 0.008]), so it ranks the cells almost exactly as that one feature does.
   Single features (`f_model_loss_std` 0.995, `f_lex_entropy` 0.977) score higher, but they were
   chosen after looking at the results, so they're descriptive only.
5. **XGBoost** on the same folds, descriptive only: AUROC 0.86 [0.75, 0.95], coverage 0.83, ECE
   0.14 (`reports/risk/crosscheck.json`). It's a candidate to **pre-register** for TinyStories,
   not a replacement chosen after looking (HANDBOOK §3.4).

## 4b. Both datasets: GATE 1 and leave-one-dataset-out (2026-11-11)
`python -m src.risk.classifier --datasets wikitext2 tinystories` → `reports/risk/evaluation.json`.
Same code, label and folds as §4; nothing was changed after the TinyStories results were seen.
TinyStories features: `reports/features/tinystories.jsonl` (105 cells, Track C, #126).

**GATE 1: passed.** The feature suite and the static baseline ran cleanly on both datasets.

| Leave-one-type-out, both datasets pooled (n = 150, 98 degraded) | Value |
|---|---|
| AUROC | **0.83** [0.76, 0.90] |
| ECE | 0.17 |
| Conformal coverage (target 0.90) | **0.81** [0.75, 0.87], mean set size 1.05 |
| vs mean-GPT-2-loss, pooled | +0.18 [0.10, 0.27]: beaten by the rule |
| vs mean-GPT-2-loss, within WikiText (n = 60) | +0.02 [0.004, 0.05]: beaten, narrowly |
| vs mean-GPT-2-loss, within TinyStories (n = 90) | −0.03 [−0.07, 0.00]: **not beaten** |

**Reading:**
1. **H2 is not shown.** Pooled, the model clears the pre-registered bar, but that margin comes
   from the baseline: a raw GPT-2 loss sits at a different level on each corpus, so pooled it
   ranks the corpora, not the risk. Within each corpus the model and the one feature are within
   0.03 of each other. We report the within-corpus numbers as the honest reading
   (`baselines.ref_lm_loss.per_dataset`).
2. **benchmark_near is still missed on both datasets** (mean p 0.00, 16 of 30 cells degraded).
   The audit's rule-based leak flag stays necessary.
3. **Leave-one-dataset-out** (train on one corpus, test every cell of the other, n = 180):
   AUROC 0.73 [0.65, 0.80], ECE 0.50, coverage 0.50 [0.43, 0.57]. Split by the test corpus,
   the *ranking* transfers almost perfectly (AUROC 0.99 on each), but the *level* does not:
   trained on TinyStories it calls WikiText safe (mean p 0.04); trained on WikiText it calls
   TinyStories degraded (mean p 1.00). The pooled AUROC is low because two shifted scales are
   mixed. A new corpus therefore needs recalibration on a few labelled cells before its
   p(degrade) or conformal sets can be trusted. That is a limitation of the deployed model, not
   a tuning step (§3.4).
4. **The deployed model** (`reports/risk/model.json`) is now fitted on both grids
   (120 fitting cells, 60 calibration cells).

## 5. System: the audit and the console
- `python -m src.risk.audit data.jsonl --grid configs/grid.yaml` and `POST /v1/audits` apply the
  deployed model (`reports/risk/model.json`; 50 fitting cells, 25 calibration cells, q = 0.36) to
  any dataset. The report gives:
  - p(degrade) and its conformal set;
  - each feature's contribution (w × standardised value, which sums exactly to the logit);
  - flagged samples with reasons (benchmark near-copies, self-repeating text).
- **Checked:** a real grid cell exported and re-audited reproduces its committed features to
  ≤ 2 × 10⁻⁶ (except the `sem` clustering columns, which vary between machines).
- **The console** (React + TypeScript, `dashboard/web/`) shows the gates, collapse curves, a
  per-cell page (outcome, white-box signals, features, the held-out risk prediction) and the
  Audit screen. `pnpm e2e` drives these flows in a real browser, in light and dark mode,
  against WCAG 2 AA (axe).

## 6. Limitations
- **Small n:** 60 test cells from one corpus and one 82M-parameter model. The CIs are wide.
- **Size dependence:** the label and features come from 1000-sample sets; type-token ratio and
  distinct-n depend on size, so an audit is only in distribution near that size.
- **Paraphrase caveat:** base DistilGPT-2 paraphrases are closer to prompted continuations
  (HANDBOOK §14.10).
- **Cross-corpus calibration:** probabilities do not transfer between corpora (§4b.3).
- **XGBoost** was never pre-registered for TinyStories, so it stays descriptive.
