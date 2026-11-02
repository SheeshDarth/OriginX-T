# Track A: predicting damage from the data (draft for T3, #23)

*Revanth. Every number cites the committed file it comes from. Draft of 2026-11-01, WikiText-2 only.*

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
Source: `reports/risk/evaluation.json`, `reports/risk/predictions.jsonl`.

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
- **Next:** GATE 1 needs the TinyStories grid (CP1). There, the model, its baselines and XGBoost
  are compared under a rule fixed before the results exist.
