# ORIGIN-T — Detailed Execution Plan & Reporting Cadence

The operational contract for the team: exactly who builds what, the week-by-week
targets for all 32 weeks, the *from-scratch-before-library* rules that keep this
research (not vibe coding), and the weekly/monthly updates each of us must file.

**Source-of-truth cross-references** — scope: [PRD.md](PRD.md) · design:
[TRD.md](TRD.md) · roadmap & reporting: [PROJECT_REPORT.md](PROJECT_REPORT.md) ·
sprint overview: [TEAM_WORKPLAN.md](TEAM_WORKPLAN.md) · Definition of Done:
[../CONTRIBUTING.md](../CONTRIBUTING.md) · ownership: [../CODEOWNERS](../CODEOWNERS).

Budget reality (do not exceed): **3 students × ≤7 hr/week ≈ 21 team-hr/week ≈
672 person-hours over 32 weeks**, run as **16 fortnightly sprints (14 delivery +
2 buffer)**. Binding compute limit is **Kaggle's free ~30 GPU-hr/week**, not the
6 GB local GPU. Weeks are relative (Wk 1 = start of Sprint 0); fill calendar
dates in during Sprint-0 planning.

---

## 1. Division of labour (substantive, module-owned)

| | S1 — Data & Bench Lead | S2 — Modeling Lead | S3 — Experiments/Infra Lead |
|---|---|---|---|
| **Person / handle** | Siddharth · `@SheeshDarth` | Revanth · `@Revanthm2027` | Vishnu · `@vishnu-k-dev` |
| **Code owned** | `src/ingestion`, `src/generation` | `src/features`, `src/risk`, `src/mitigation` | `src/finetune`, `src/evaluation`, `dashboard` |
| **Primary artifacts** | Schema+loaders, contamination generators, **ORIGIN-T-Bench**, hidden holdout, datasheet, HF release | Feature suite, calibrated risk classifier, survival head, conformal wrapper, mitigation optimizer | Fine-tune runner, eval harness, collapse curves, representational signals, reproducibility infra, dashboard |
| **Theory to master** | Distribution shift; information theory (entropy, KL/JS); MAUVE; n-gram diversity | Survival analysis (hazard, censoring); conformal prediction; calibration/ECE | LoRA/PEFT math; embedding geometry; effective rank; mech-interp basics |
| **Compute footprint** | CPU + local GPU (embeddings) | CPU + light GPU (classifiers) | **Owns the Kaggle GPU grid** |
| **Backup owner** | S3 | S1 | S2 |

**The one shared interface:** the `Sample` schema (`src/ingestion/schema.py`,
TRD §7). It is the contract between every module. Any change is a PR reviewed by
**all three**. Nobody hand-edits data outside the schema.

---

## 2. The anti-vibe-coding contract (graded at every mini-viva)

The single biggest project risk is *"depth becomes a library wrapper"*. Defence:
for each non-trivial component the owner ships a **from-scratch implementation +
a short `reports/` notebook** that reproduces a library's result on a toy input,
**before** the library is allowed in the pipeline. At the mini-viva the owner
must answer the listed teach-back question live.

| Component | Owner | From-scratch deliverable (do first) | Library allowed after | Teach-back question |
|-----------|-------|--------------------------------------|-----------------------|---------------------|
| Perplexity / reference-LM loss | S2 | Token NLL from model logits by hand; matches HF to 1e-4 | `evaluate` | Why is perplexity = exp(mean NLL)? What does a +10% shift mean? |
| Diversity (TTR, distinct-n, self-BLEU) | S3 | Hand-count on a toy corpus | — (keep hand code) | How does distinct-n collapse under recursion? |
| Entropy / KL / JS | S1 | Shannon entropy + KL from count tables | `scipy` (checks) | Why is JS symmetric and bounded; KL not? |
| MAUVE | S1 | Explain divergence-frontier + quantised KL | `mauve-text` | What does MAUVE measure that BLEU cannot? |
| Calibration / ECE | S2 | Binned ECE + reliability diagram by hand | `sklearn` (checks) | What does a classifier being "calibrated" mean? |
| Kaplan–Meier + discrete-time hazard | S2 | KM estimator + discrete-time likelihood from scratch | `lifelines`/`pycox` | What is right-censoring; why does it break naive accuracy? |
| Split-conformal | S2 | Quantile of calibration nonconformity scores | `MAPIE` | Why does split-conformal give distribution-free coverage? |
| Effective rank / participation ratio | S3 | Entropy of normalised singular-value spectrum | `numpy.linalg` (checks) | How does effective rank drop signal representational collapse? |
| LoRA / QLoRA | S3 | Derive ΔW = BA, parameter count, 4-bit NF4 idea | `peft`/`bitsandbytes` | Why does rank-r adaptation cut trainable params ~100×? |
| Shapley attribution | S2 | Exact Shapley on ≤4 features by enumeration | `shap` | What axioms make Shapley the unique fair attribution? |

**Rule of thumb:** if a component cannot be explained on a whiteboard by its
owner, it is not done — regardless of whether the code runs.

---

## 3. Pre-registered definitions (freeze in Sprint 0, never move mid-project)

- **Degradation event τ** (any one, vs. clean baseline): perplexity **+10%**,
  cluster coverage **−15%**, or hidden-benchmark accuracy **−5 pp**.
- **Risk target** `R_t = P(degradation within next k stages | z_0..z_t)`; fix
  `k` (default 1) in Sprint 0.
- **Success metrics** (from PRD §7): detection AUROC **≥ 0.80** in-distribution;
  collapse-prediction correlation **≥ 0.60**; early-warning **lead-time > 0** at
  a fixed false-alarm rate; mitigation **10–20%** relative recovery; **≥ 1**
  demonstrated benchmark-leakage inflation case; every experiment reproducible
  from a committed config + fixed seed.
- **Core grid** (S3): 1 model (DistilGPT-2, QLoRA 4-bit) × 5 ratios
  (0/25/50/75/100%) × 4 conditions (clean/synthetic/recursive/benchmark-near) ×
  3 seeds ≈ **60 LoRA fine-tunes** (~15–30 min each). Cache every checkpoint +
  activation to HF Hub — never re-extract.
- **Statistical rigor:** ≥ 3 seeds with bootstrap CIs on every headline number.

---

## 4. Week-by-week targets (all 32 weeks)

Legend: **[P]** primary build · **[S]** support/review. Each sprint lists an
**Exit** (what "done" means) and any **Gate / Milestone**. FS = from-scratch
deliverable from §2.

### Sprint 0 · Wk 1–2 · Inception + collapse spike · **GATE 0**
| Wk | S1 (Siddharth) | S2 (Revanth) | S3 (Vishnu) |
|----|----------------|--------------|-------------|
| 1 | ✅ ingestion schema+loaders (done); normalise WikiText-2 + TinyStories → JSONL | Draft τ + success-metric definitions with team; read Shumailov + ConStat | Kaggle env + QLoRA DistilGPT-2 trains end-to-end; log 1 perplexity number |
| 2 | Build **hidden holdout** v0 + written split policy; sanity-check no leakage | Freeze τ/k pre-registration doc in `docs/`; FS perplexity notebook started | Fine-tune on model-generated text; produce **collapse-or-not plot** |
**Exit:** collapse plot exists; τ frozen; two datasets normalised.
**GATE 0 (S3-led, all decide):** is collapse reproducible on a tiny model? If **no** → pivot to static contamination-detection framing (still a valid FYP). Decision recorded in Sprint Review.

### Sprint 1 · Wk 3–4 · Data foundations + literature review · **M1 mini-viva**
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 3 | Add 3rd base dataset (small permissive code/QA); finalise schema edge cases | Literature-review draft (collapse + benchmark contamination) | Turn spike into **config-driven fine-tune runner** (seed, config, logging) |
| 4 | Holdout access-control + datasheet stub; FS entropy/KL notebook | Signed-off **feature-list spec** (lexical/entropy/perplexity/semantic/leakage); FS perplexity notebook done | Checkpoints → HF Hub; MLflow/W&B offline logging wired |
**Exit / M1:** each student explains their component's theory (~10 min + Q&A).

### Sprint 2 · Wk 5–6 · Contamination generators
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 5 | `src/generation`: synthetic sampler (DistilGPT-2/Qwen) at ratios 0/25/50/75/100% | Prototype lexical + entropy features on Sprint-0 data | Wire generator→runner: "variant → fine-tune" is one command |
| 6 | Recursive Gen-1→3 loop + paraphrase + benchmark-near injection; each output schema-valid | Perplexity + reference-LM-loss feature; FS diversity checks | Reproducibility: variant build from clean clone + config |
**Exit:** all four contamination types generate, validate, and record `source`/`generation`/`contamination_ratio`.

### Sprint 3 · Wk 7–8 · ORIGIN-T-Bench v0.1 · **Bench shipped**
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 7 | Assemble variants + metadata → SQLite(meta) + JSONL/Parquet; datasheet | Confirm bench exposes every field the feature suite needs | Bench build is one reproducible command; CI smoke on a bench slice |
| 8 | Tag **`bench-v0.1`**; provenance labels complete | Semantic features prototype (embeddings, NN distance) | Cache embeddings/activations to Hub |
**Exit:** versioned ORIGIN-T-Bench v0.1 with datasheet, builds from config.

### Sprint 4 · Wk 9–10 · Feature suite I
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 9 | MAUVE + diversity reference distributions for the bench; FS MAUVE notebook | `src/features`: lexical (TTR, repetition, burstiness) + entropy, tested | Feature extraction runs on Kaggle within quota; cache outputs |
| 10 | Provide labelled provenance splits for training the detector | Semantic (embedding density, cluster entropy, tail retention) + leakage features | Batch feature-extraction harness + timing/quota report |
**Exit:** feature suite computes on ≥ 2 datasets, unit-tested.

### Sprint 5 · Wk 11–12 · Feature suite II + static baseline · **GATE 1 · M2**
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 11 | Cross-generator eval split (train on gen A, test on gen B) | Calibrated static classifier: logistic → XGBoost; FS ECE + reliability diagram | Reproducible train/eval of the classifier; seed sweep |
| 12 | Honest cross-generator report table | AUROC/AUPRC/F1 + calibration on ≥ 2 datasets | Figures to `reports/`; CI green |
**GATE 1:** feature suite + static baseline run cleanly on ≥ 2 datasets. **M2:** live defence.

### Sprint 6 · Wk 13–14 · Fine-tuning runner (compute-heavy)
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 13 | Supply exact grid variants; verify holdout never enters training | Define per-run degradation label τ for the risk model | Launch **core grid** batch 1 (ratios × clean/synthetic); checkpoint everything |
| 14 | Leakage audit across all variants | Trajectory-structured feature tables (stage z_0..z_t) | Core grid batch 2 (recursive/benchmark-near); quota-managed |
**Exit:** ≥ ~60 LoRA fine-tunes cached with checkpoints + activations.

### Sprint 7 · Wk 15–16 · Collapse curves · **GATE 2**
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 15 | Verify stage alignment (ratio/recursion depth) per curve | Compute τ events per run; assemble risk-model dataset | Collapse curves: perplexity, distinct-n, cluster coverage, holdout acc |
| 16 | Diversity/MAUVE deltas per stage | Correlation(risk features, measured degradation) first pass | Bootstrap CIs on curves; figures to `reports/` |
**GATE 2:** ≥ 1 contamination setting causes **measurable degradation** (τ crossed). If **no** → negative-results paper track (ship bench + honest write-up).

### Sprint 8 · Wk 17–18 · Buffer + risk classifier · **M3**
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 17 | Buffer: fix any bench/leakage debt surfaced by the grid | Calibrated **static risk classifier** `R_t`; reliability + ECE | Scoring harness (AUROC + calibration) reproducible |
| 18 | Bench v0.2 with corrections | Risk-vs-degradation correlation vs **≥ 0.60** target | Seed-averaged results + CIs |
**M3:** live defence; risk model beats random baseline (pre-GATE-3 check).

### Sprint 9 · Wk 19–20 · Survival model (gated) · **GATE 3**
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 19 | Provide right-censored trajectories (datasets that never degrade) | FS **Kaplan–Meier + discrete-time hazard**; lead-time metric | C-index + integrated Brier scoring |
| 20 | Stage-metadata QA for survival inputs | Switch to `lifelines`/`pycox` only after FS matches | Lead-stage @ fixed false-alarm-rate plot |
**GATE 3:** risk model beats random **and lead-time > 0**.

### Sprint 10 · Wk 21–22 · Split-conformal + 1 representational signal
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 21 | Embedding-FID / reference distributions for signals | FS **split-conformal**; coverage-vs-nominal plot | FS **effective rank / participation ratio** from existing checkpoints (no new training) |
| 22 | — | `MAPIE` cross-check after FS | Anisotropy / neuron-death rate as secondary signals |
**Exit:** conformal coverage plot + one white-box collapse signal, both reproducible.

### Sprint 11 · Wk 23–24 · Time-aware head + calibration · **M4**
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 23 | Support sequence-structured inputs (z_0..z_t per dataset) | Time-aware risk head (sequence vs static) + calibration | Time-dependent C-index harness |
| 24 | — | Reliability diagrams post-calibration | Figures + CIs to `reports/` |
**M4:** live defence of calibration + early-warning.

### Sprint 12 · Wk 25–26 · Mitigation + attribution (gated)
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 25 | Cluster-balancing + human-anchor candidate pools | `src/mitigation`: remove/downweight high-risk clusters; preserve rare clusters | Re-run fine-tune on mitigated data → measure recovery |
| 26 | Rare-cluster retention metric | (Gated) safe-mix optimiser + FS Shapley on ≤4 features | Recovery vs **10–20%** target with CIs |
**Exit:** ≥ one mitigation result showing relative recovery vs contaminated baseline.

### Sprint 13 · Wk 27–28 · Consolidation + paper start · **Results freeze**
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 27 | Bench final QA + datasheet completion | Paper: methods + results tables (with CIs) | Reproducibility freeze: every figure regenerates from config |
| 28 | ≥ 1 benchmark-leakage inflation case documented | Paper: calibration/survival/conformal sections | Archive all run configs + seeds |
**Exit:** results frozen; paper skeleton complete.

### Sprint 14 · Wk 29–30 · Bench release + dashboard · **Public release**
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 29 | Publish **ORIGIN-T-Bench** on HuggingFace + datasheet + license check | Paper draft → internal review | Streamlit `dashboard` over audit reports/plots |
| 30 | Release notes + versioning | arXiv preprint prep | Demo path end-to-end from clean clone |
**Milestone:** ORIGIN-T-Bench public; dashboard demoable.

### Sprint 15 · Wk 31–32 · Buffer + demo + viva · **Final rehearsal**
| Wk | S1 | S2 | S3 |
|----|----|----|----|
| 31 | Buffer: address examiner-style gaps in data/bench | Buffer: tighten weakest metric; finalise CIs | Buffer: reproducibility + demo polish |
| 32 | Final datasheet/report sign-off | Paper final; viva slides | Live demo rehearsal; freeze tag `v1.0` |
**Exit:** final mini-viva rehearsal passed; submission-ready.

---

## 5. Reporting cadence — what each of us files, and when

### 5.1 Async standup — **twice weekly (Mon & Thu)**, written, ≤ 5 lines each
Yesterday/since-last → today/next → blockers → hours logged so far this week
(cap 7). Posted to the team channel; no meeting.

### 5.2 Weekly guide/HOD update — **1 page, every week** (S-lead compiles)
Copy-paste template:

```
ORIGIN-T — Weekly Update — Week N  (dd Mon – dd Mon)
Sprint: S_  |  Gate status: G0[ ] G1[ ] G2[ ] G3[ ]

Per student (planned vs done; hours <=7):
  S1 Siddharth — planned: ...   done: ...   hours: _/7
  S2 Revanth   — planned: ...   done: ...   hours: _/7
  S3 Vishnu    — planned: ...   done: ...   hours: _/7

Key result this week (attach 1 plot/table + link): ...
Metric vs target: ... (e.g. detection AUROC 0.74 vs 0.80 target)
Blockers / risks: ...
Plan next week: ...

Per-student learning line (concept + evidence link + one open question):
  S1: ...
  S2: ...
  S3: ...
```
Rules: state metric **vs. target** (not just "did work"); every result links to a
committed figure in `reports/`; hours are honest and capped at 7/student.

### 5.3 Sprint ceremonies — **every 2 weeks**
Planning (45 min) → Backlog refinement (30 min) → Review/Demo (30 min, feeds the
guide update) → Retro (30 min). Sprint Review must show a **runnable artifact**,
not slides.

### 5.4 Monthly mini-viva — **a defence, not a demo** (~10 min/student)
Live deliverable + each student explains their component **and its theory** with
2–3 probing questions + metric-vs-gate **with confidence intervals** + updated
risk/cut list + guide's per-student rubric note. Mini-vivas land at **M1 (Wk 4),
M2 (Wk 12), M3 (Wk 18), M4 (Wk 24)**; final rehearsal Wk 31–32.

---

## 6. Gates & decision points (one-line kill criteria)

| Gate | Wk | Owner | Pass condition | If it fails |
|------|----|-------|----------------|-------------|
| GATE 0 | 2 | S3 | Collapse reproducible on a tiny model | Pivot to static contamination-detection FYP |
| GATE 1 | 12 | S2 | Feature suite + static baseline clean on ≥ 2 datasets | Cut semantic/leakage features; ship lexical baseline |
| GATE 2 | 16 | S3 | ≥ 1 setting crosses τ (measurable degradation) | Negative-results paper: ship bench + honest write-up |
| GATE 3 | 20 | S2 | Risk model beats random **and** lead-time > 0 | Drop survival/early-warning claim; keep calibrated static risk |

**Default posture:** ship the committed core (bench, feature suite, collapse
curves, calibrated risk classifier, split-conformal, one representational signal,
mitigation, report). Gated depth (survival, time-aware head, causal/Shapley,
scaling law, safe-mix) is added **only if ahead of plan** — never at the cost of
the core.

---

## 7. Definition of Done (every story, every person)

Config-driven + fixed seed (no magic numbers) · reproducible from a clean clone
on Kaggle/Colab · `pytest -q` green · run logged to MLflow/W&B (offline OK) ·
figure/table committed to `reports/` · PR approved by one teammate · logbook
line added. For survival + conformal: **the from-scratch notebook is part of
Done** — no exceptions.

---

## 8. Risk register mapped to owners

| Risk | Owner | Mitigation |
|------|-------|------------|
| Collapse invisible on tiny LoRA | S3 | Sprint-0 spike + smallest known-collapse setting; GATE-0 pivot ready |
| Velocity < plan (3×7 hr) | All | Measure real velocity from S2; 2 buffer sprints; cut gated depth first |
| Bus-factor-1 on survival/conformal/interp | S2/S3 | Pair on hard sprints; teach-backs; named backup owner |
| Kaggle quota / OOM | S3 | QLoRA 4-bit + tiny models; checkpoint to Hub; never run grid at sprint-end |
| Library-wrapper / vibe-coded depth | All | §2 from-scratch-before-library + mini-viva teach-backs |
| Scope creep on stretch items | All | Hard gates; core shippable by end of Sprint 11 |
| Hidden-holdout leakage | S1 | Access control + per-sprint leakage audit before every grid run |
