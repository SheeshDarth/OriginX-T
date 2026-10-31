# ORIGIN-T

**Time-Aware Dataset Forensics & Early-Warning Risk Prediction for LLM Fine-Tuning**

> "Predict whether a dataset will damage an LLM — before the fine-tuning damage becomes visible."

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-GATE%200%20passed-green.svg)](docs/PROJECT_REPORT.md)
[![CI](https://github.com/SheeshDarth/OriginX-T/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

ORIGIN-T is a final-year AIML research project that audits datasets for synthetic,
recursive, paraphrased, and benchmark-leaked contamination, measures the resulting
model-collapse when a small LLM is fine-tuned on that data, and learns an
**early-warning risk model** that flags danger **before** fine-tuning damage is
fully visible. It blends three ideas into one coherent system: **ORIGIN**
(dataset forensics), **SENTINEL-T** (survival-based early warning), and
**IrregMamba** (time-aware sequence modeling).

## Why this project

- **Not a wrapper.** No LangChain glue, no prompt-chaining product — a measurable
  ML research system: contamination generators, a fine-tuning grid, a feature
  suite, and a risk model with a defined go/no-go gate at every stage.
- **Frontier-relevant.** Model collapse (Shumailov, *Nature* 2024) and benchmark
  contamination are active, unsolved problems in how foundation models are
  trained and evaluated.
- **Depth over breadth.** A calibrated risk classifier is the committed core;
  competing-risks survival analysis, conformal guarantees, white-box
  representational signals, causal attribution, and a safe-mix optimizer are
  **gated** additions layered on the *same* experiments — not scope creep.

## Core features

| # | Feature | Status |
|---|---------|--------|
| 1 | Controlled contamination generator (synthetic / recursive / paraphrased / benchmark-near) | Committed |
| 2 | Feature suite (lexical, entropy, perplexity, semantic, MAUVE, leakage) | Committed |
| 3 | LoRA fine-tuning grid + collapse-curve measurement | Committed |
| 4 | Calibrated risk classifier (degrade / no-degrade) | Committed |
| 5 | Split-conformal risk guarantee | Gated |
| 6 | Discrete-time survival early-warning model | Gated |
| 7 | White-box representational-collapse signals (effective rank, anisotropy) | Gated |
| 8 | Causal attribution (Shapley) + collapse scaling law | Gated |
| 9 | Safe-mix mitigation optimizer | Gated |
| 10 | **ORIGIN-T-Bench** — released benchmark + leaderboard | Committed (M8) |

See [docs/PRD.md](docs/PRD.md) for full scope and [docs/TRD.md](docs/TRD.md) for
the technical design.

## Architecture (high level)

```
Raw dataset -> Normalizer -> Contamination generator -> ORIGIN-T Bench store
                                                              |
                              +-------------------------------+
                              v                                v
                      Feature extractor                 Fine-tuning runner
                              |                                 |
                        Risk model  <---- risk-model training --+-- Evaluation harness -> Degradation curves
                              |
                Calibration (conformal) -> Alert / Report -> Mitigation (safe-mix)
```

## Tech stack

Python 3.11 &middot; PyTorch &middot; HF `transformers` / `datasets` / `peft` /
`accelerate` &middot; `bitsandbytes` (4-bit) &middot; scikit-learn, XGBoost,
`sentence-transformers`, FAISS, `mauve-text` &middot; `lifelines` / `pycox`
(survival) &middot; `MAPIE` (conformal) &middot; `shap` &middot;
`TransformerLens` / `nnsight` (interpretability) &middot; MLflow / W&B
(offline) &middot; Streamlit (dashboard).

## Repository structure

```
OriginX-T/
  README.md
  docs/
    PRD.md                 Product Requirements Document
    TRD.md                 Technical Requirements Document
    PROJECT_REPORT.md       Full research report (literature, roadmap, risks)
  configs/                 Experiment & model configs
  data/                    Raw / processed datasets (large data git-ignored)
  src/
    ingestion/              Dataset loaders, schema inference
    generation/              Synthetic + recursive contamination generation
    features/                Lexical / entropy / semantic / representational features
    finetune/                Small-model LoRA fine-tuning
    evaluation/               Degradation metrics, benchmark evaluation
    risk/                    Static, survival, conformal, causal, scaling-law models
    mitigation/               Data cleaning, safe-mix optimization
  dashboard/                Streamlit evidence dashboard (Sprint 14)
  reports/                  Figures, experiment logs
  tests/                    Unit / smoke tests
  .github/workflows/ci.yml  Lint + test CI
```

## Getting started

```bash
git clone https://github.com/SheeshDarth/OriginX-T.git
cd OriginX-T
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                            # smoke test should pass
```

To reproduce results on another machine (pools, grids, X1/X2 runs), install the exact
versions that produced them instead:

```bash
pip install -r requirements.lock.txt   # Python 3.13, CUDA 12.4 torch wheel
```

The lock pins `torch==2.6.0+cu124` from the PyTorch index. On a machine without CUDA 12.4,
install a matching `torch==2.6.0` build first, then the lock file. Regenerate the lock when
a dependency is added.

Experiments are config-driven and logged to a local MLflow database
(`mlruns/`, git-ignored; view with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`):

```bash
python -m src.finetune.collapse --config configs/collapse_spike.yaml   # GATE 0 spike
python -m src.finetune.grid --config configs/grid_smoke.yaml           # minutes: dry run of the grid
python -m src.finetune.grid --config configs/grid.yaml                 # the full grid (resumable)
python -m src.finetune.grid --config configs/grid.yaml --summarize-only   # re-apply GATE 2, redraw the figure; trains nothing
python -m src.finetune.grid --config configs/grid.yaml --rerun-types benchmark_near   # redo one type's cells after its generator changed
python -m src.evaluation.whitebox --config configs/whitebox.yaml   # white-box signals (effective rank, anisotropy, ...) from the kept checkpoints
python -m src.features.run_table                                    # one CSV row per grid run: outcomes + white-box signals
```

GPU work (fine-tuning) runs on free Google Colab / Kaggle notebooks — see
[docs/TRD.md](docs/TRD.md#compute-budget) for the compute plan. A local RTX
4050 6GB GPU is sufficient for feature extraction and small-model inference.

## Team

| Member | Role | Owns | Theory owner |
|--------|------|------|--------------|
| Siddharth (`@SheeshDarth`) | S1 — Data & Bench Lead | Contamination pipeline, ORIGIN-T-Bench, hidden holdout | Distribution shift, information theory, MAUVE |
| Revanth (`@Revanthm2027`) | S2 — Modeling Lead | Feature suite, risk models, calibration/conformal | Survival analysis, conformal prediction |
| Vishnu (`@vishnu-k-dev`) | S3 — Experiments/Infra Lead | Fine-tuning runner, eval harness, reproducibility | LoRA/PEFT, embedding geometry, mech-interp |

## Development workflow

2-week sprints, async standups Mon/Thu, Definition of Done in
[CONTRIBUTING.md](CONTRIBUTING.md). Full 8-month roadmap and current sprint
status: [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md#roadmap).

## Status

**GATE 0 passed** ([#8](https://github.com/SheeshDarth/OriginX-T/pull/8)): recursive
training on a model's own output raises hidden-holdout perplexity on every one of
3 seeds (mean 56.3 → 67.6 over Gen-0..3).

**GATE 2 passed** ([#11](https://github.com/SheeshDarth/OriginX-T/pull/11),
[#12](https://github.com/SheeshDarth/OriginX-T/pull/12)) on the full 60-cell grid
(4 contamination types × 5 ratios × 3 seeds, DistilGPT-2 + LoRA on WikiText-2), under
a dose-response rule fixed before the results were seen
([TRD §11](docs/TRD.md#11-gono-go-gates)). Hidden-holdout perplexity rises steadily
with the contamination ratio, from 56.6 at ratio 0 to 67.6 (synthetic), 68.4
(recursive) and 65.4 (paraphrased) at ratio 1, and is already above seed noise at
ratio 0.25. Recursive is worst at every ratio.

**benchmark_near re-run** (2026-10-31, after the recasing fix in
[#13](https://github.com/SheeshDarth/OriginX-T/pull/13)): training on leaked items now
lowers benchmark-item perplexity steadily with dose, 61.3 → 59.2 (−3.4%) at ratio 1, so
memorisation is visible. Its hidden-holdout damage dropped from +8% to +2.4%. GATE 2 is
unchanged.

**GATE 3 failed (exploratory)** (2026-10-31). The healthy `human_control` arm
([#72](https://github.com/SheeshDarth/OriginX-T/pull/72)) now gives GATE 3 three
never-collapsing trajectories: hidden-holdout perplexity stays flat (56.6 → 56.4) while the
contaminated types rise. White-box signals were measured on all 66 checkpoints.

No signal passes, and that is structural, not noise. 9 of the 12 contaminated trajectories
already collapse at the first contaminated ratio (0.25), so on this ratio axis there is no
stage before onset to warn from. Anisotropy still separates the arms *at* onset: it is flat
for the control and falls for the contaminated types, the opposite of the pre-registered
direction. It is reported as exploratory (D7). The confirmatory GATE 3 on TinyStories
needs stages before onset (finer low ratios, or training steps as in X3), fixed in its rule
beforehand ([HANDBOOK §13](docs/HANDBOOK.md)).

**Working on this repo? Read [docs/HANDBOOK.md](docs/HANDBOOK.md) first.** It is the
operating manual: tracks, owners, interfaces, checkpoints and rules.

Done so far: the contamination generators, the hidden-holdout safeguards, run logging
(MLflow), the fine-tuning grid with its results, the white-box signals
(`src/evaluation/whitebox.py`, [#15](https://github.com/SheeshDarth/OriginX-T/pull/15);
measured on all 66 checkpoints), the scoring metrics
(`src/evaluation/scoring.py`, [#16](https://github.com/SheeshDarth/OriginX-T/pull/16):
AUROC/AUPRC/F1, bootstrap CIs, ECE, conformal coverage, lead stage) and a per-run table
joining the two ([#17](https://github.com/SheeshDarth/OriginX-T/pull/17)). Not in the repo
yet: the feature suite proper (`src/features/` holds only that table), and with it the
risk classifier, conformal prediction and survival models that need it (GATE 1 and
GATE 3). What counts as "collapsed" (the label) is still to be fixed by the team. See the roadmap for the full plan.
Weekly updates go to the faculty guide; see the template in
[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md#hod-reporting).

## License

[MIT](LICENSE) — permissive by default so ORIGIN-T-Bench can be released
freely on HuggingFace; change before submission if your institution requires
otherwise.
