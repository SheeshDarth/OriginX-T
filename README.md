# ORIGIN-T

**Time-Aware Dataset Forensics & Early-Warning Risk Prediction for LLM Fine-Tuning**

> "Predict whether a dataset will damage an LLM — before the fine-tuning damage becomes visible."

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Sprint%200-orange.svg)](docs/PROJECT_REPORT.md)
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

GPU work (fine-tuning) runs on free Google Colab / Kaggle notebooks — see
[docs/TRD.md](docs/TRD.md#compute-budget) for the compute plan. A local RTX
4050 6GB GPU is sufficient for feature extraction and small-model inference.

## Team

| Role | Owns | Theory owner |
|------|------|--------------|
| S1 — Data & Bench Lead | Contamination pipeline, ORIGIN-T-Bench, hidden holdout | Distribution shift, information theory, MAUVE |
| S2 — Modeling Lead | Feature suite, risk models, calibration/conformal | Survival analysis, conformal prediction |
| S3 — Experiments/Infra Lead | Fine-tuning runner, eval harness, reproducibility | LoRA/PEFT, embedding geometry, mech-interp |

## Development workflow

2-week sprints, async standups Mon/Thu, Definition of Done in
[CONTRIBUTING.md](CONTRIBUTING.md). Full 8-month roadmap and current sprint
status: [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md#roadmap).

## Status

**Sprint 0 — Inception + collapse spike.** See the roadmap for the current
gate. Weekly updates go to the faculty guide; see the template in
[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md#hod-reporting).

## License

[MIT](LICENSE) — permissive by default so ORIGIN-T-Bench can be released
freely on HuggingFace; change before submission if your institution requires
otherwise.
