# ORIGIN-T

**Time-Aware Dataset Forensics & Early-Warning Risk Prediction for LLM Fine-Tuning**

> "Predict whether a dataset will damage an LLM — before the fine-tuning damage becomes visible."

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-CP0%3A%20GATE%200%20%2B%202%20passed-green.svg)](#status)
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

| # | Feature | Scope | Status (2026-11-02) |
|---|---------|-------|---------------------|
| 1 | Controlled contamination generator (synthetic / recursive / paraphrased / benchmark-near / duplicate) + a healthy `human_control` arm | Committed | Done |
| 2 | Feature suite (lexical, leakage, reference-LM, semantic incl. MAUVE) | Committed | Done on WikiText; TinyStories waits for its grid |
| 3 | LoRA fine-tuning grid + collapse-curve measurement | Committed | Done on WikiText (75 cells); TinyStories next |
| 4 | Calibrated risk classifier (degrade / no-degrade) | Committed | Done on WikiText; GATE 1 needs a second dataset |
| 5 | Split-conformal risk guarantee | Gated | Built; under-covers on WikiText (0.72 vs 0.90) |
| 6 | Discrete-time survival early-warning model | Gated | Not started (only if CP2 is green) |
| 7 | White-box representational-collapse signals (effective rank, anisotropy) | Gated | Done on 66 checkpoints, and live during training (X3) |
| 8 | Causal attribution (Shapley) + collapse scaling law | Gated | Not started |
| 9 | Mitigation: harm-based repair (X1) and rare-cluster preservation | Gated | Pilot go; full grid replicated on independent pools (#136): X1 passes, harm ties origin cleaning, rare-cluster preservation has no measurable effect |
| 10 | **ORIGIN-T-Bench** — released benchmark + leaderboard | Committed (M8) | Not started |
| 11 | **ORIGIN-T Console**: web UI over a FastAPI backend | Could | Overview, curves, cell detail, audit, repair, live monitor |

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
(offline) &middot; console: FastAPI + Pydantic v2 API, React + TypeScript (Vite, TanStack,
Tailwind tokens, Observable Plot), Playwright + axe. No Streamlit (HANDBOOK D6). Conformal
prediction is hand-rolled; survival libraries are not used yet.

## Repository structure

```
OriginX-T/
  README.md
  docs/
    HANDBOOK.md            The operating manual: tracks, owners, interfaces, gates (read first)
    PRD.md                 Product Requirements Document
    TRD.md                 Technical Requirements Document
    STRETCH_GOALS.md       X1-X3 research, prior art and novelty claims
    PROJECT_REPORT.md       Full research report (literature, roadmap, risks)
    paper/                 Paper draft, one section per track
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
  dashboard/
    api/                    FastAPI: read-only reports, audit jobs, SSE run stream (openapi.json = the contract)
    web/                    ORIGIN-T Console (React + TypeScript; see its README)
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
python -m src.evaluation.earlywarning --config configs/earlywarning.yaml   # GATE 3 on the grid
python -m src.features.run_table                                    # one CSV row per grid run: outcomes + white-box signals
```

Features, risk and audit (they need the grid's `data/processed/` files and cached pools;
copy them from the machine that ran the grid, HANDBOOK §12.3):

```bash
python -m src.features.suite --config configs/features.yaml --dataset wikitext2   # features/1 per cell
python -m src.features.scores --dataset wikitext2     # per-sample risk scores (risk_reflm, risk_knn)
python -m src.risk.classifier                         # risk model, conformal sets, baselines, GATE 1
python -m src.risk.audit my_data.jsonl --grid configs/grid.yaml   # audit a new dataset
python -m src.evaluation.live --preset x3             # X3: log white-box signals during training
python -m src.evaluation.alarm --config configs/x3.yaml          # Gate X3 (after CP0 confirms the rule)
```

The console: `python -m dashboard.api` (http://127.0.0.1:8000, docs at `/docs`), then
`dashboard/web/README.md` for the UI.

Every grid in this repo ran on one RTX 2050 (4 GB): DistilGPT-2 + LoRA cells take about
75 s each. Bigger targets (X2) run on Siddharth's GPU or Colab/Kaggle, see
[docs/TRD.md](docs/TRD.md#compute-budget) and HANDBOOK §12.

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

**Snapshot of 2026-11-09 (CP0 decisions merged 2026-11-08; CP1 is 2026-11-20).** Every gate is a rule
fixed before its results existed ([TRD §11](docs/TRD.md#11-gono-go-gates), HANDBOOK §11, §13).

| Gate | Rule (short) | Verdict | Evidence |
|---|---|---|---|
| GATE 0 | Recursive self-training raises holdout ppl on every seed | **Passed** | 3/3 seeds, 56.3 → 67.6 ([#8](https://github.com/SheeshDarth/OriginX-T/pull/8)) |
| GATE 1 | Feature suite + static baseline run cleanly on ≥ 2 datasets | **Not evaluable** | WikiText evaluated; the TinyStories grid, features and scores now exist (#126), the risk evaluation on both datasets is pending (Track A) ([#94](https://github.com/SheeshDarth/OriginX-T/pull/94), [#105](https://github.com/SheeshDarth/OriginX-T/pull/105)) |
| GATE 2 | Dose-response, above seed noise at ratio 0.25 | **Passed** on WikiText and TinyStories | 75-cell WikiText grid and 105-cell TinyStories grid (#126; [#11](https://github.com/SheeshDarth/OriginX-T/pull/11), [#12](https://github.com/SheeshDarth/OriginX-T/pull/12), [#73](https://github.com/SheeshDarth/OriginX-T/pull/73)) |
| GATE 3 | A white-box signal warns before collapse | **Failed** (WikiText, then the confirmatory run on TinyStories 2026-11-08) | No signal beats the permutation bar (AUROC 0.08-0.56); `reports/grid_tinystories/earlywarning.json` |
| X1 | Harm-guided repair beats random by ≥ 10 pts | **Passed** on the independent-pool replicate | Full grid (240 retrains, #136): harm − random 0.25-0.55 on all 9 ratio-0.5 cells; harm − origin −0.03 [−0.07, +0.01], no difference; on benchmark near-copies harm recovers nothing. Pilot on the grid's pools: [#122](https://github.com/SheeshDarth/OriginX-T/pull/122) |
| X2 | Small-model canary predicts a bigger model's damage (ρ ≥ 0.7) | **Passed 2026-11-08** | Spearman ρ 0.78, CI [0.51, 0.91], 24 pairs, Pythia-410M target; `reports/x2/gate_x2.json` |
| X3 | Live alarm before the breach at a calibrated 5% FA | **Failed 2026-11-08** | Early alarm 44% (bar 50%), false alarms 6% (bar 10%); `reports/x3/gate_x3.json`;  Monitor, data, pre-registered plan ([#98](https://github.com/SheeshDarth/OriginX-T/pull/98), [#110](https://github.com/SheeshDarth/OriginX-T/pull/110)) |

**Risk model (WikiText, 2026-11-01, [#105](https://github.com/SheeshDarth/OriginX-T/pull/105)).**
Leave-one-type-out AUROC **0.84** [0.72, 0.94], ECE 0.22. The conformal sets **under-cover**:
0.72 against the 0.90 target. And the full model only **ties** the single reference-LM loss
feature (difference +0.001, CI [0, 0.008]): H2 ("a multi-feature model beats a simple
detector") is not shown yet. Prior art also narrowed the novelty claim: predicting a
fine-tuning outcome before training is published (TuneAhead, ICML 2026), so ORIGIN-T's
claim is contamination-caused degradation with conformal sets on a ground-truth testbed
([docs/STRETCH_GOALS.md](docs/STRETCH_GOALS.md), [#112](https://github.com/SheeshDarth/OriginX-T/pull/112)).

**Features (WikiText, [#91](https://github.com/SheeshDarth/OriginX-T/pull/91),
[#103](https://github.com/SheeshDarth/OriginX-T/pull/103)).** Generated text is easy for a
fixed reference LM (gpt2 loss 3.99 clean → 2.13 for recursive data), and benchmark
near-copies crowd together (MAUVE 0.08). Per sample, the reference-LM score finds generated
text (AUROC 0.98 recursive) and the kNN score finds near-copies (0.97)
([#92](https://github.com/SheeshDarth/OriginX-T/pull/92)). Rare-cluster loss, the classic
collapse signature, does *not* separate generated text from healthy text at this scale.

**X3 (live, during training, [#98](https://github.com/SheeshDarth/OriginX-T/pull/98)).** Logging
the white-box signals every 10 steps is bit-for-bit neutral: of the 67 logged runs, the 24
that are grid cells reproduce the grid's holdout perplexity exactly (the rest use new seeds). Within a run the holdout loss never rises: contaminated runs just improve less than
healthy ones, so damage is a gap, not a turn. Anisotropy separates by step 20 of 125. The
Gate X3 rule is confirmed in `configs/x3.yaml` (2026-11-08); the verdict (2026-11-08) is a fail: the alarm fired early on 8 of 18 collapsing runs (bar 50%), with 1 of 16 false alarms. Every miss is a ratio-1 run whose breach is at the first logged step, so no lead was possible at a 10-step log interval.

**Next: CP1 (Fri 2026-11-20).** GATE 1 needs the risk model evaluated on both datasets (Track A);
X1's full repair grid ran as an independent-pool replicate (#136, Gate X1 passed, harm ties
origin cleaning) with the B10 rare-cluster rule (#135); the bench release (B12) follows. The three Track C
verdicts are in: GATE 3 failed, X3 failed, X2 passed. What remains for Track C is the paper (#23).

<details><summary>Earlier results (GATE 0, GATE 2, benchmark_near, GATE 3 in detail)</summary>

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

</details>

**Working on this repo? Read [docs/HANDBOOK.md](docs/HANDBOOK.md) first.** It is the
operating manual: tracks, owners, interfaces, checkpoints and rules. Work is tracked as
[GitHub Issues](https://github.com/SheeshDarth/OriginX-T/issues) by checkpoint milestone.
Weekly updates for the faculty guide are in [docs/hod/](docs/hod/) (template:
[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md#hod-reporting)).

## License

[MIT](LICENSE) — permissive by default so ORIGIN-T-Bench can be released
freely on HuggingFace; change before submission if your institution requires
otherwise.
