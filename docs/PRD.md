# ORIGIN-T — Product Requirements Document (PRD)

## 1. Problem statement

Given a dataset intended for LLM fine-tuning or evaluation, can we estimate
whether it contains synthetic, recursively generated, paraphrased,
low-diversity, or benchmark-leaked samples; predict the risk of downstream
model degradation; and recommend a minimum intervention that improves
generalization — **before** the expensive fine-tuning run is complete?

## 2. Target users

| Persona | Need |
|---------|------|
| AI/ML research team | Reproducible experiments + a defensible research contribution |
| Startup fine-tuning engineer | Know whether a synthetic dataset will help or hurt before spending compute |
| Dataset curator | Certify a dataset is diverse, minimally leaked, and safe to train on |
| Evaluation researcher | Detect whether a benchmark score is inflated by contamination |

## 3. Product vision

ORIGIN-T is a **data-quality firewall** for foundation-model teams: before a
team fine-tunes or evaluates a model, ORIGIN-T audits the dataset, produces a
contamination report, estimates collapse risk with a lead-time warning, and
suggests mitigation.

## 4. Research hypotheses

- **H1 —** Synthetic / recursive text leaves measurable distributional
  signals (entropy, perplexity, embedding density, cluster coverage).
- **H2 —** A multi-feature risk model beats a simple human-vs-AI detector at
  predicting downstream degradation.
- **H3 —** Risk is trajectory-dependent — recursive Gen-3 data is not
  equivalent to first-generation synthetic data, even when both look fluent.
- **H4 —** A time-aware risk model can forecast degradation *before* full
  fine-tuning completes (genuine early warning, not post-hoc detection).
- **H5 —** Preserving rare semantic clusters + human-anchor sampling recovers
  more utility than random data removal.

## 5. Core features (MoSCoW)

**Must**
1. Dataset ingestion (JSONL/CSV/TXT; prompt/response/source/split schema).
2. Contamination generator — synthetic, recursive (Gen-1->3), paraphrased,
   benchmark-near, at controlled ratios (0/25/50/75/100%).
3. Feature suite — lexical, entropy, perplexity, semantic, leakage.
4. Fine-tuning + evaluation harness producing collapse curves.
5. Calibrated risk classifier with an explicit success threshold.
6. **ORIGIN-T-Bench** release (variants + features + cached activations).

**Should**
7. Split-conformal risk guarantee (distribution-free false-alarm control).
8. Discrete-time survival model with a lead-time-before-collapse metric.
9. One white-box representational-collapse signal (effective rank).

**Could (stretch)**
10. Competing-risks survival (per failure-mode hazard).
11. Causal/Shapley attribution + a fitted collapse scaling law.
12. Safe-mix optimizer (utility-vs-risk Pareto front).
13. Evidence dashboard (Streamlit).

**Won't (this cycle)**
- Training a foundation model from scratch.
- Adversarial / detector-evasion contamination (red-teaming).
- A dedicated cross-model-scale sweep.
- Any hardware/IoT component.

## 6. Non-goals

- Not a chatbot, not an agent-wrapper product.
- Does not claim universal AI-text detection — reports generator-specific and
  cross-generator limitations honestly.
- Does not use private or copyrighted scraped data — open datasets and
  team-generated synthetic data only.

## 7. Success metrics

| Area | Metric |
|------|--------|
| Contamination detection | AUROC >= 0.80 in-distribution; honest cross-generator report |
| Collapse prediction | Correlation >= 0.60 between predicted risk and measured degradation |
| Early warning | Lead-time > 0 steps at a fixed false-alarm rate |
| Mitigation | >= 10-20% relative recovery vs. contaminated baseline |
| Benchmark leakage | >= 1 demonstrated case of inflated vs. rephrased/hidden score |
| Reproducibility | Every experiment runs from a committed config + fixed seed |

## 8. Competitive landscape

| System | Overlap | ORIGIN-T differentiation |
|--------|---------|---------------------------|
| Generic AI-text detectors | Human-vs-AI classification | ORIGIN-T predicts *downstream degradation*, not just origin |
| Benchmark-contamination surveys | Token-overlap leakage detection | ORIGIN-T uses behavioral (performance-gap) leakage detection |
| Model-collapse papers (Shumailov et al.) | Establish the phenomenon | ORIGIN-T builds a *predictive, calibrated* early-warning system on top of it |

## 9. Product risks

- Probes may capture "deception-related" surface text rather than genuine
  synthetic origin — mitigated with control sets and causal ablation.
- White-box signals require open model weights — scope excludes API-only
  models.
- Detection accuracy degrades under distribution shift / larger scale —
  reported honestly, not overclaimed.
