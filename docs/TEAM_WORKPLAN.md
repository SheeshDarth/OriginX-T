# ORIGIN-T — Team Work Plan

Who builds what, in what order, and how the three streams hand off to each
other. Roles are defined in [PROJECT_REPORT.md](PROJECT_REPORT.md#5-team-raci--agile-process);
scope in [PRD.md](PRD.md); design in [TRD.md](TRD.md). Ownership is enforced by
[CODEOWNERS](../CODEOWNERS).

## 1. Owners at a glance

| | S1 — Data & Bench Lead | S2 — Modeling Lead | S3 — Experiments/Infra Lead |
|---|---|---|---|
| **Person** | Siddharth (`@SheeshDarth`) | Revanth (`@Revanthm2027`) | Vishnu (`@vishnu-k-dev`) |
| **Owns** | `src/ingestion`, `src/generation` | `src/features`, `src/risk`, `src/mitigation` | `src/finetune`, `src/evaluation`, `dashboard` |
| **Delivers** | Data schema, loaders, contamination generators, **ORIGIN-T-Bench**, hidden holdout | Feature suite, calibrated risk classifier, calibration/conformal, survival (gated), mitigation | Fine-tuning runner, evaluation harness, collapse curves, reproducibility, dashboard |
| **Theory to own** | Distribution shift, information theory, MAUVE | Survival analysis, conformal prediction, calibration | LoRA/PEFT, embedding geometry, mech-interp |
| **Compute** | CPU (RTX 4050 fine for embeddings) | CPU + light GPU | **Kaggle 30 GPU-hr/wk grid owner** |

Everything the whole team consumes flows through **one shared contract**: the
`Sample` schema in `src/ingestion/schema.py` (TRD §7). Change that schema only by
PR reviewed by all three — it is the interface between every module.

## 2. Dependency spine (why order matters)

```
S1 schema+loaders ─► S1 contamination generator ─► ORIGIN-T-Bench (variants)
                                                          │
                        ┌─────────────────────────────────┼───────────────────┐
                        ▼                                   ▼                   ▼
                 S2 feature suite                    S3 fine-tune grid   S3 eval harness
                        │                                   │                   │
                        └──────────► S2 risk model ◄────────┴── collapse labels ┘
                                          │
                                 S2 calibration/conformal ─► S3 dashboard ─► report
```

S2 and S3 are both **blocked on S1's schema + a first contaminated variant**.
That is why S1's Week-1 job (the loader) is the critical path for the whole team.

## 3. Per-sprint plan (maps to the 16-sprint roadmap)

Legend: **[P]** primary owner does the build, **[S]** support/review, **[T]** theory teach-back owed at the monthly mini-viva.

### Sprint 0 (Wk 1–2) — Inception + collapse spike — **GATE 0: is collapse reproducible?**
- **S3 [P]:** minimal QLoRA fine-tune of DistilGPT-2 on WikiText-2 on Kaggle; measure perplexity on a clean holdout before/after training on model-generated text. This is the gate — if collapse is not reproducible, the whole team pivots to static contamination detection.
- **S1 [P]:** `src/ingestion` — `Sample` schema + JSONL/CSV/TXT loaders + normalizer (the deliverable this session starts). Pull WikiText-2 and TinyStories into normalized JSONL.
- **S2 [S]:** read the survival/conformal papers; draft the degradation-threshold definitions (τ) with S1 so they are pre-registered before any experiment.
- **Gate artifact:** one plot showing (or failing to show) degradation. Decision recorded in the sprint review.

### Sprint 1 (Wk 3–4) — Data foundations + literature review — M1 mini-viva
- **S1 [P]:** finalize the schema; add a third base dataset (small permissive code/QA set); build the **hidden holdout** (kept out of every training set, access-controlled) and document the split policy.
- **S2 [P]:** literature-review write-up on model collapse + benchmark contamination; specify the feature list (lexical/entropy/perplexity/semantic/leakage) as a signed-off spec.
- **S3 [P]:** turn the Sprint-0 spike into a **config-driven fine-tune runner** (fixed seed, checkpoints to HF Hub, MLflow/W&B offline logging).
- **[T]** Each student explains their component's theory for M1.

### Sprint 2 (Wk 5–6) — Contamination generators
- **S1 [P]:** `src/generation` — synthetic (DistilGPT-2/Qwen sampling), recursive Gen-1→3 loop, paraphrase, benchmark-near injection; controlled ratios 0/25/50/75/100%. Each output validates against the schema and records `source`, `generation`, `contamination_ratio`.
- **S3 [S]:** wire the generator into the runner so a variant → fine-tune is one command.
- **S2 [S]:** review that generated variants preserve label/split integrity.

### Sprint 3 (Wk 7–8) — ORIGIN-T-Bench v0.1 — **Bench shipped**
- **S1 [P]:** assemble variants + metadata + a datasheet into a versioned bench (Parquet/JSONL + SQLite metadata); tag `v0.1`.
- **S3 [S]:** reproducibility check — bench builds from a clean clone + config.
- **S2 [S]:** confirm the bench exposes everything the feature suite needs.

### Sprint 4–5 (Wk 9–12) — Feature suite I & II + static baseline — **GATE 1** + M2 mini-viva
- **S2 [P]:** `src/features` — lexical (TTR, repetition, burstiness), entropy, reference-LM perplexity, semantic (embedding density, NN distance, cluster entropy), leakage; then a **calibrated static classifier** (logistic → XGBoost) with reliability diagram/ECE.
- **S1 [S]:** provide labelled provenance data + MAUVE/diversity metric support.
- **S3 [S]:** feature extraction runs reproducibly on Kaggle; cache activations.
- **Gate 1:** feature suite + static baseline run cleanly on ≥2 datasets.

### Sprint 6–7 (Wk 13–16) — Fine-tuning runner (compute-heavy) + collapse curves — **GATE 2**
- **S3 [P]:** run the **core grid** — 1 model × 5 ratios × 4 conditions × 3 seeds ≈ 60 LoRA fine-tunes; cache every checkpoint + activation. Produce collapse curves (perplexity, distinct-n, cluster coverage, hidden-holdout accuracy).
- **S1 [S]:** supply the exact variants; verify hidden holdout never leaks into training.
- **S2 [S]:** define the degradation label τ per run for the risk model.
- **Gate 2:** ≥1 contamination setting causes measurable degradation. (If not → negative-result paper track.)

### Sprint 8–9 (Wk 17–20) — Risk classifier + survival model (gated) — M3 + **GATE 3**
- **S2 [P]:** calibrated risk classifier `R_t = P(degrade within k stages | z_0..z_t)`; then (gated) hand-rolled discrete-time survival head with lead-time metric.
- **S3 [S]:** feed features + collapse labels; C-index / Brier scoring harness.
- **S1 [S]:** provide trajectory-structured data (stages per dataset).
- **Gate 3:** risk model beats a random baseline; lead-time > 0.

### Sprint 10–11 (Wk 21–24) — Conformal + representational signal + calibration — M4
- **S2 [P]:** hand-rolled split-conformal → `MAPIE`; coverage-vs-nominal plot; time-aware head + calibration.
- **S3 [P]:** one white-box signal (effective rank / participation ratio) from existing checkpoints — no new training.
- **S1 [S]:** MAUVE / embedding-FID reference distributions.

### Sprint 12 (Wk 25–26) — Mitigation + attribution (gated) — Mitigation result
- **S2 [P]:** `src/mitigation` — remove/downweight high-risk clusters, preserve rare clusters, re-evaluate; (gated) safe-mix optimizer + Shapley attribution.
- **S1 [S]:** cluster-balancing + human-anchor sampling support.
- **S3 [S]:** re-run the fine-tune on mitigated data to measure recovery.

### Sprint 13–15 (Wk 27–32) — Consolidation, bench release, dashboard, viva
- **S1 [P]:** public **ORIGIN-T-Bench** release on HuggingFace + datasheet.
- **S3 [P]:** `dashboard` (Streamlit) + reproducibility freeze + demo.
- **S2 [P]:** paper draft (results, calibration, CIs); arXiv preprint.
- **All:** final mini-viva rehearsal; results freeze.

## 4. Standing per-person Definition of Done

Every story, every person (from [CONTRIBUTING.md](../CONTRIBUTING.md)):
config-driven + fixed seed · reproducible from a clean clone on Kaggle ·
`pytest -q` green · run logged to MLflow/W&B · figure/table in `reports/` ·
one teammate PR approval · logbook line. **From-scratch-before-library** for
survival + conformal (keep the hand-rolled notebook as viva evidence).

## 5. This week — concrete next actions

- **S1 (Siddharth):** land `src/ingestion` (schema + loaders + tests) — *in progress this session* — then normalize WikiText-2 + TinyStories to JSONL.
- **S3 (Vishnu):** stand up the Kaggle QLoRA spike for GATE 0; get DistilGPT-2 training end-to-end and log one perplexity number.
- **S2 (Revanth):** draft the degradation-threshold (τ) definitions + the feature-list spec so they are pre-registered before experiments; start the literature-review doc.

Branch per story: `<initials>/<short-name>` → PR to `main` → one teammate approves.
