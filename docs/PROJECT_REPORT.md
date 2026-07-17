# ORIGIN-T — Detailed Project Report

*Combined literature review, methodology, roadmap, and risk assessment.
For scope see [PRD.md](PRD.md); for architecture see [TRD.md](TRD.md).*

## Executive summary

ORIGIN-T is a final-year AIML capstone that treats **dataset quality as a
trajectory, not a static label**. It generates controlled contamination
(synthetic, recursive, paraphrased, benchmark-near), fine-tunes small LLMs
across that trajectory, measures the resulting collapse, and trains an
early-warning risk model that predicts damage before fine-tuning completes.
It blends three source ideas — **ORIGIN** (dataset forensics), **SENTINEL-T**
(survival-based early warning), and **IrregMamba** (time-aware modeling) —
into one coherent, gate-driven research system sized for a 3-student,
8-month, <=7 hr/week/student capstone.

## 1. Research landscape and evidence

**Data scarcity.** Publicly available human-written training text may be
exhausted within the decade (Epoch AI; AP reporting, ~2026-2032 window),
pushing teams toward synthetic data — which is cheaper and more scalable,
but riskier.

**Model collapse.** Recursively training models on model-generated data
reduces diversity and erases distribution tails (Shumailov et al., *Nature*
2024). Accumulating real data alongside synthetic data can mitigate this in
some settings (Gerstgrasser et al., arXiv:2404.01413), and the severity
depends on generation depth and mixing ratio (Seddik et al., arXiv:2404.05090)
— the correct research target is measurement and risk-scoring, not blanket
synthetic-data rejection.

**Benchmark contamination.** Leaked or paraphrased evaluation items inflate
benchmark scores; current detection methods are inconsistent under semantic
(non-token-overlap) contamination (Xu et al., arXiv:2406.04244; Dekoninck et
al.'s ConStat, arXiv:2405.16281).

## 2. The gap / novelty

No existing tool combines: (a) dataset-level contamination forensics, (b) a
*forward-looking, calibrated* risk model (not just post-hoc detection), and
(c) white-box, model-internal collapse signals. ORIGIN-T's contribution is
precisely this combination — predicting degradation before it happens, with
optional survival-analysis rigor and conformal guarantees, evaluated on a
released, reproducible benchmark.

## 3. Experiment design

- **E1 — Provenance & detection.** Human / synthetic / recursive /
  paraphrased / benchmark-near classification; baselines vs. multi-feature
  classifiers; honest cross-generator generalization results.
- **E2 — Collapse curves.** Fine-tune on 0-100% synthetic mixtures and a
  recursive Gen-1->3 loop; plot degradation in perplexity, diversity, and
  hidden-holdout accuracy.
- **E3 — Early-warning risk.** Static vs. sequence vs. (gated) survival
  model; report lead-stage at a fixed false-alarm rate.
- **E4 — Mitigation & rescue.** Random vs. high-risk removal vs. cluster
  balancing vs. (gated) safe-mix / human-anchor sampling; document failure
  cases honestly — a negative result here is still valid research output.

## 4. Roadmap {#roadmap}

3 students x 7 hr/week ~= 21 team-hr/week ~= 672 person-hours over 32 weeks,
run as **16 fortnightly sprints (14 delivery + 2 buffer)**.

| Sprint | Weeks | Goal | Gate |
|--------|-------|------|------|
| S0 | 1-2 | Inception + collapse spike | **GATE 0**: collapse reproducible? |
| S1 | 3-4 | Data foundations + literature review | M1 mini-viva |
| S2 | 5-6 | Contamination generators | - |
| S3 | 7-8 | ORIGIN-T-Bench v0.1 | Bench shipped |
| S4 | 9-10 | Feature suite I | - |
| S5 | 11-12 | Feature suite II + static baseline | **GATE 1** + M2 mini-viva |
| S6 | 13-14 | Fine-tuning runner (compute-heavy) | - |
| S7 | 15-16 | Collapse curves | **GATE 2** |
| S8 | 17-18 | Buffer + risk classifier | M3 mini-viva |
| S9 | 19-20 | Survival model (gated) | **GATE 3** |
| S10 | 21-22 | Split conformal + 1 representational signal | Coverage plot |
| S11 | 23-24 | Time-aware head + calibration | M4 mini-viva |
| S12 | 25-26 | Mitigation + attribution (gated) | Mitigation result |
| S13 | 27-28 | Consolidation + paper start | Results freeze |
| S14 | 29-30 | Bench release + dashboard | ORIGIN-T-Bench public |
| S15 | 31-32 | Buffer + demo + viva | Final mini-viva rehearsal |

**Committed core:** ORIGIN-T-Bench, feature suite, collapse curves, a
calibrated risk classifier, split conformal, one representational signal,
mitigation, and this report. **Gated (add only if ahead):** discrete-time
survival, time-aware head, causal attribution, scaling law, safe-mix
optimizer.

## 5. Team, RACI & Agile process

| Role | Owns | Theory owner | Backup |
|------|------|--------------|--------|
| S1 — Data & Bench Lead | Contamination pipeline, ORIGIN-T-Bench, hidden holdout | Distribution shift, information theory, MAUVE | S3 |
| S2 — Modeling Lead | Feature suite, risk models, calibration/conformal | Survival analysis, conformal prediction | S1 |
| S3 — Experiments/Infra Lead | Fine-tuning runner, eval harness, reproducibility | LoRA/PEFT, embedding geometry, mech-interp | S2 |

**Ceremonies (per 2-week sprint):** Sprint Planning (45 min) - async written
standup 2x/week (Mon/Thu) - Backlog Refinement (30 min) - Sprint Review/Demo
(30 min, feeds the HOD update) - Retro (30 min). Full **Definition of Done**
in [CONTRIBUTING.md](../CONTRIBUTING.md).

## 6. Risk register

| Risk | Mitigation |
|------|------------|
| Collapse not visible on tiny LoRA models | Sprint-0 spike + pre-registered smallest known-collapse setting; pivot to static contamination-detection if it fails (still a valid FYP) |
| Velocity below plan (3 people @ 7 hr) | Measure real velocity from Sprint 2; 14 delivery + 2 buffer sprints; cut gated depth by default |
| Bus-factor-1 on survival/conformal/interp | Pairing on hard sprints + teach-backs + named backup owner per component |
| Kaggle/Colab quota & OOM | QLoRA 4-bit + tiny models from Sprint 0; checkpoint to HF Hub; never run the grid at sprint-end |
| Depth becomes a library-wrapper (vibe-coded) | From-scratch-before-library graded checkpoints; examiner-style monthly mini-vivas |
| Scope creep on stretch items | Hard go/no-go gates; committed core shippable by end of Sprint 11 |

## 7. HOD reporting {#hod-reporting}

**Weekly update (1 page):** planned vs. done per student, key result/plot,
metrics vs. target, blockers, next week, hours logged (<=7/student), gate
status, and a per-student learning line (concept + evidence link + one open
question).

**Monthly mini-viva (a defence, not a demo):** live deliverable + each
student explains their component and its theory (~10 min, 2-3 probing
questions) + metric-vs-gate with confidence intervals + updated risk/cut
list + guide's per-student rubric note.

## 8. Publication strategy

**Venue:** workshop, not main track — data-centric / trustworthy-ML or
*ACL data-contamination workshops; release **ORIGIN-T-Bench on HuggingFace**
+ an arXiv preprint. Stretch: NeurIPS Datasets & Benchmarks. **Result
needed:** a reproducible contamination-gradient benchmark plus one
significant finding (positive lead-time or calibrated-FPR risk prediction
beating a static baseline on >= 2 datasets, with confidence intervals). If
GATE 2 fails, there is no positive paper — ship the benchmark plus an honest
negative-results write-up, which remains a defensible final-year
contribution.

## 9. References

1. Shumailov et al., *AI models collapse when trained on recursively
   generated data*, Nature 2024.
2. Gerstgrasser et al., *Is Model Collapse Inevitable?*, arXiv:2404.01413.
3. Seddik et al., *How Bad is Training on Synthetic Data?*, arXiv:2404.05090.
4. Xu et al., *Benchmark Data Contamination of LLMs: A Survey*, arXiv:2406.04244.
5. Dekoninck et al., *ConStat*, arXiv:2405.16281.
6. Pillutla et al., *MAUVE*, NeurIPS 2021 (arXiv:2102.01454).
7. Ethayarajh, *How Contextual are Contextualized Word Representations?*
   (anisotropy), EMNLP 2019 (arXiv:1909.00512).
8. Roy & Vetterli, *The effective rank*, EUSIPCO 2007.
9. Hu et al., *LoRA*, arXiv:2106.09685; HF PEFT docs.
10. Gensheimer & Narasimhan, *A scalable discrete-time survival model for
    neural networks*, PeerJ 2019.
11. Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*,
    arXiv:2107.07511.
12. Gibbs & Candes, *Adaptive Conformal Inference Under Distribution Shift*,
    arXiv:2106.00170.
13. Cho et al., *GRU / Encoder-Decoder*, arXiv:1406.1078.
14. Lundberg & Lee, *SHAP*, arXiv:1705.07874.
15. *Dynamic Survival Analysis for Early Event Prediction*, arXiv:2403.12818.

*Preprint IDs move; re-verify each link before formal citation.*
