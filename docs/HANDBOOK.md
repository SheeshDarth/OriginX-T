# ORIGIN-T Handbook — the single source of truth

**Read this whole file before doing anything.** It's written for both the team and their
coding agents. An agent that has read it should be able to start its owner's work
without asking questions that are answered here.

- **Repo:** https://github.com/SheeshDarth/OriginX-T (default branch `main`)
- **Last full update:** 2026-10-31 (`main` at `ffe97d5`, PRs #1–#19 merged)
- **If this file and another doc disagree, this file wins.** Fix the other doc in the
  same PR. The PRD (`docs/PRD.md`) and TRD (`docs/TRD.md`) remain the
  *research spec*. This handbook is the *operating manual*: who does what, when, how,
  and with which interfaces.
- **Updating this file:** only at a checkpoint (§6), or when a frozen interface (§7)
  must change. Every change to §6, §7 or §13 needs all three members' approval in the PR.

---

## 0. Agent bootstrap: start here

When your human says who they are (for example "I'm Revanth"), do this, in order, with
no further prompting:

1. **Identify the track.** Siddharth = Track B (§9). Revanth = Track A (§8). Vishnu =
   Track C (§10). If the name doesn't match, stop and ask.
2. **Sync.** `git fetch origin && git checkout main && git pull --ff-only`. Read
   §1, §3, §4, §6, §7 and your track section in full. Skim the rest.
3. **Set the commit identity to your human, never someone else's.**
   - `git config user.name "<their GitHub username>"`
   - `git config user.email "<their GitHub noreply email>"`. Find it at
     github.com → Settings → Emails, format `<id>+<username>@users.noreply.github.com`.
     Ask once if you can't find it.
   - Vishnu's is already known: `vishnu-k-dev <190108551+vishnu-k-dev@users.noreply.github.com>`.
4. **Check the environment** (§12.1): Python ≥ 3.11, `pip install -r requirements.txt`,
   `python -m pytest -q` passes. Record the GPU (`nvidia-smi`).
5. **Find your next task in GitHub Issues**, the tracker for this project:
   `gh issue list --repo SheeshDarth/OriginX-T --assignee <their-username> --milestone "<current checkpoint>" --state open`.
   Pick the lowest-numbered open issue whose "Depends on" issues are closed. §6 tells you
   which checkpoint is current by date. Issue titles carry the handbook task id (`[C4]`,
   `[A3]`, `[B7]`), and §17 maps ids to issue numbers. If a task depends on an interface
   from another track that isn't merged yet, build against the **stub fixtures** described
   in §7. Never wait for another track. Put `Closes #<n>` in the PR so the issue closes on
   merge, and tick the issue's checklist as you go.
6. **Tell your human in 5 lines or fewer:** the current checkpoint, your next task, what
   you'll produce, and roughly how long it takes. Then start. Don't ask permission for
   routine steps (§4).

---

## 1. The project in one page

**ORIGIN-T** is a data-quality firewall for LLM fine-tuning. Given a dataset, it should
(1) detect contamination: synthetic, recursive (model-trained-on-model), paraphrased,
benchmark-leaked; (2) predict *downstream degradation* before training, with a
**calibrated** risk score and a **conformal** false-alarm guarantee; (3) **repair** the
dataset and prove the repair worked by retraining. It's a final-year AIML research
project for three students over 32 weeks (Jul 2026 – Mar 2027), with monthly mini-vivas.

**Hypotheses** (PRD §4): H1 synthetic/recursive text leaves measurable signals. H2 a
multi-feature risk model beats a plain AI-text detector at predicting damage. H3
recursive ≠ first-generation synthetic. H4 risk can be forecast before training
finishes. H5 rare-cluster-preserving repair beats random removal.

**Status at 2026-10-31**

| Item | State |
|---|---|
| Contamination generators: synthetic, recursive, paraphrased, benchmark_near | Done (`duplicate` not yet) |
| Hidden-holdout safeguards (split by text, leak gate, mixer guard) | Done |
| Collapse spike, **GATE 0** | **Passed**, 3 seeds: holdout ppl 56.3 → 67.6 over Gen-0..3 |
| Fine-tuning grid (4 types × 5 ratios × 3 seeds, DistilGPT-2 + LoRA, WikiText-2) | Done, 60 cells, 57 checkpoints kept (16.7 GB, on Vishnu's machine) |
| **GATE 2** (dose-response, pre-registered) | **Passed**: all 3 collapse types degrade, above seed noise at ratio 0.25 |
| benchmark_near re-run after the recasing fix (#13) | Done locally 2026-10-31: benchmark ppl falls −3.4% at ratio 1 (memorisation visible). **Not yet committed** (CP0) |
| White-box signals on all 57 checkpoints | Measured 2026-10-31. **Not yet committed** (CP0) |
| **GATE 3** (early warning) | **Not evaluable**: all 12 trajectories collapse, so there are no negatives. Needs a healthy control arm (Track C) |
| Scoring metrics (AUROC/AUPRC/F1, bootstrap CI, ECE, conformal coverage, lead stage) | Done (`src/evaluation/scoring.py`) |
| Feature suite, static baseline (**GATE 1**) | **Not started.** Critical path, Track A |
| Risk classifier, calibration, split conformal | Not started, Track A |
| Mitigation / repair, ORIGIN-T-Bench packaging + release | Not started, Track B |
| UI (product console) | Not started, Track A (§8.6) |
| Stretch X1 / X2 / X3 (§11) | Researched (`docs/STRETCH_GOALS.md`), not started |

**Headline numbers so far** (seed mean, hidden-holdout perplexity, ratio 0 → 1):
synthetic 56.6 → 67.6 · recursive 56.6 → 68.4 · paraphrased 56.6 → 65.4 ·
benchmark_near 56.6 → 58.0 (and benchmark-item ppl 61.3 → 59.2).
White-box: **anisotropy falls** steadily with contamination (0.935 → 0.90). Effective rank
and dead-neuron fraction stay flat.

---

## 2. Repository map

```
configs/         every experiment is a YAML config here; no magic numbers in code
  corpora.yaml          base corpora (WikiText-2, TinyStories) → data/processed/
  collapse_spike.yaml   GATE 0 spike
  grid.yaml             the Sprint-6 grid (+ GATE 2 rule in `gate:`)
  grid_smoke.yaml       2-minute dry run of the grid
  whitebox.yaml         white-box signal measurement
  earlywarning.yaml     GATE 3 rule
src/
  ingestion/     schema.py (Sample, the shared contract), loaders, normalize_corpora,
                 holdout (split by text hash), check_leakage (leak gate)
  generation/    mixer (refuses hidden_holdout), benchmark_near, model_based
                 (make_synthetic / make_paraphrased / hf_generator, batched: list→list)
  finetune/      lora.py (train_lora), collapse.py (GATE 0), grid.py (grid, pools,
                 --summarize-only, --rerun-types, GATE 2)
  evaluation/    metrics.py (perplexity, distinct_n), scoring.py, tracking.py (MLflow),
                 whitebox.py, earlywarning.py (GATE 3)
  features/      run_table.py (joins grid results + white-box); the feature suite goes here
  risk/          EMPTY: risk classifier, calibration, conformal (Track A)
  mitigation/    EMPTY: repair (Track B)
dashboard/       EMPTY: api/ (Track C) and web/ (Track A) go here (§8.6, §10)
reports/         committed results: grid/, collapse_spike/, figures/
data/            git-ignored: raw/, processed/ (normalized corpora, pools), models/
docs/            PRD, TRD, PROJECT_REPORT, LITERATURE_SURVEY, STRETCH_GOALS, this file
tests/           pytest, stub-based, no torch needed (CI installs only ruff, pytest, pyyaml)
```

**Commands that exist today**

```bash
python -m src.ingestion.normalize_corpora --config configs/corpora.yaml
python -m src.ingestion.holdout --input data/processed/wikitext2_train.jsonl
python -m src.ingestion.check_leakage --holdout <holdout.jsonl> --train <train.jsonl ...>
python -m src.finetune.collapse --config configs/collapse_spike.yaml
python -m src.finetune.grid --config configs/grid_smoke.yaml
python -m src.finetune.grid --config configs/grid.yaml [--summarize-only] [--rerun-types TYPE ...]
python -m src.evaluation.whitebox --config configs/whitebox.yaml [--summarize-only]
python -m src.evaluation.earlywarning --config configs/earlywarning.yaml
python -m src.features.run_table [--grid ...] [--whitebox ...] [--out reports/grid/run_table.csv]
python -m pytest -q
ruff check --isolated --select F src tests
```

---

## 3. Ground rules (non-negotiable; CI or review will reject violations)

1. **The hidden holdout never enters training.** Use `mix()` (it raises on
   `split="hidden_holdout"`) and `check_leakage` before any new training data. The holdout
   is carved by **text hash**, so duplicated lines stay on one side.
2. **Never change a fingerprinted key** of an existing config. Those are
   `_FINGERPRINT_KEYS` in `src/finetune/grid.py`: base_model, generator_model,
   recursive_depth, data, benchmark_near, train, generate, share_baseline. Doing so
   invalidates every committed result. New experiment → **new config file + new
   output_dir**.
3. **A generator code change must bump its pool version** (`POOL_VERSIONS` in grid.py).
   Pools are cached by fingerprint, and a code change doesn't change the fingerprint, so
   stale pools would be silently reused. Then `--rerun-types <type>`.
4. **Gates and rules are pre-registered.** Write the rule into a config/doc and merge it
   **before** looking at the results. Never change a rule after seeing data. If the data
   shows the rule was wrong, write a *new* rule and test it on *new* data (§13).
5. **Heavy imports stay lazy** (torch, transformers, peft, datasets, sentence-transformers,
   mlflow, matplotlib): import inside functions. Tests use stubs and must pass with only
   `ruff pytest pyyaml` installed. That's exactly what CI installs.
6. **Every PR:** `python -m pytest -q` and `ruff check --isolated --select F src tests`
   pass. New logic gets a test that **fails without the fix**: show it in the PR.
7. **Results are committed, data and models are not.** Commit `reports/**` JSON/JSONL/PNG.
   Never commit `data/`, checkpoints, `mlruns/`, `*.bak*`, or anything with secrets.
8. **Stay in your lane** (CODEOWNERS, §5). Touching another track's directory needs that
   owner's review. Prefer asking them, or opening an issue.
9. **Commit as your own human** (§0 step 3). Branch names `<initials>/<story>`:
   `sp/…` Siddharth, `rm/…` Revanth, `vk/…` Vishnu. Never push to `main`. PR → one approval →
   merge.
10. **No UTF-8 BOMs, LF line endings.** Windows PowerShell's `Set-Content -Encoding utf8`
    writes a BOM: use your editor or `[IO.File]::WriteAllText(path, text, UTF8Encoding($false))`.

---

## 4. How agents work here (execution guidelines)

The goal is speed *and* correctness. Waste is rework, not typing.

**Do**
- **Act, don't narrate.** Read what you need, then build. Don't ask permission for routine,
  reversible steps: creating a branch, writing code and tests, running tests, running a
  smoke config. Do ask before anything outward-facing or hard to reverse: pushing, opening
  a PR, deleting data or checkpoints, long GPU runs on a shared machine (§12.2).
- **Smallest correct change.** Reuse what exists before writing new code. Check `src/`
  first (e.g. `scoring.py` already has AUROC/ECE/conformal coverage; `run_table.py`
  already joins results). No speculative abstractions, no frameworks for one use.
- **Prove it.** Each PR states what was *run* (commands, numbers) and what wasn't. A test
  that only passes is weak: show it fails on `main`. For logic-heavy code, run a quick
  mutation check: break the logic deliberately and confirm a test catches it.
- **Pilot before you spend compute.** Smoke config first (minutes), then one seed, then the
  full run.
- **Write results honestly.** Negative and "not evaluable" results are reported as such.
  GATE 3 being "not evaluable" was the *correct* output.
- **Keep the PR small and single-purpose.** One story per PR, with a description that follows
  `CONTRIBUTING.md`: summary, why, what changed, testing (run / not run), risks.
- **Leave a note for the next agent.** If you discover a trap, add it to §14 in your PR.

**Don't**
- Don't re-derive what this handbook already states, and don't re-litigate settled decisions
  (§13) unless new data contradicts them. Then raise it at the next checkpoint.
- Don't fabricate numbers, citations or test results. If you didn't run it, say so.
- Don't "improve" another track's code in passing. Open an issue for its owner.
- Don't block on another track: build against the §7 stub fixtures.
- Don't generate the parts a member will be examined on without them understanding it.
  Survival and conformal prediction must be **hand-rolled first** (CONTRIBUTING.md
  "from-scratch-before-library"). The agent explains every non-trivial piece in the PR
  so its human can defend it at the mini-viva.
- No `Streamlit` (decision §13). No AI-generic UI (§8.6).

**Windows notes** (all three members use Windows):
- In PowerShell, `git am 000*.patch` doesn't glob; use Git Bash.
- `huggingface_hub` prints a symlink warning without Developer Mode. It's harmless.
- Clear stale bytecode if tests act strangely after mutation runs:
  `python -B -m pytest -p no:cacheprovider`.

---

## 5. Team, tracks and ownership

| Member | GitHub | Track | Owns (code) | Theory owner (viva) |
|---|---|---|---|---|
| **Siddharth** | `@SheeshDarth` | **B: Data, Repair & Bench** | `src/ingestion/`, `src/generation/`, `src/mitigation/`, bench packaging/release, **X1** | distribution shift, information theory, MAUVE |
| **Revanth** | `@Revanthm2027` | **A: Modeling & Product UI** | `src/features/` (suite CLI, `lex.py`, `leak.py`), `src/risk/`, `dashboard/web/` (5 of 7 screens) | survival analysis, conformal prediction |
| **Vishnu** | `@vishnu-k-dev` | **C: Infra, Experiments & API** | `src/finetune/`, `src/evaluation/`, `src/features/model.py` + `semantic.py`, `dashboard/api/`, the Live monitor + Canary screens, all grid runs, **X2**, **X3** | LoRA/PEFT, embedding geometry, mech-interp |

**CODEOWNERS changes required at CP0** (the current file differs):
`src/mitigation/` → @SheeshDarth (was Revanth; repair moves to Track B with X1).
`dashboard/web/` → @Revanthm2027, `dashboard/api/` → @vishnu-k-dev (was all Vishnu).
`src/features/` stays @SheeshDarth @Revanthm2027, and Revanth leads, except
`src/features/model.py` and `src/features/semantic.py` → @vishnu-k-dev.
`dashboard/web/src/routes/monitor/` and `dashboard/web/src/routes/canary/` → @vishnu-k-dev
(Revanth reviews for design consistency).

**Why this split:** every track starts from artefacts that already exist (grid results,
pools, checkpoints, white-box code), so all three can start on day one. The only coupling
is through the frozen interfaces in §7.

**Load note:** Track A carries the critical path (features → classifier) *and* the UI, so
part of it already moved to Track C: the GPU-heavy feature families (`model.py`,
`semantic.py`, §10.7) and the two UI screens for Track C's own features (Live monitor,
Canary; §10.8). Within Track A, features come first. The UI builds in parallel against
fixtures (§8.6), never blocking the features. If Track A still slips at CP1, Track C takes
the API-side UI wiring and Track B takes `duplicate` + bench docs, and nothing moves *into*
Track A.

---

## 6. Schedule and checkpoints

Tracks run independently between checkpoints and merge at each one. At a checkpoint:
everyone's PRs for that phase are merged, the integration step is run, the gates due are
evaluated, and the next phase is confirmed. One 45-minute meeting.

| Checkpoint | Date | Integration step | Gates / decisions |
|---|---|---|---|
| **CP0 Kickoff** | **Tue 2026-11-03** | Merge CP0 PR (§6.1). Freeze §7 interfaces. Each member's agent bootstraps (§0) | Owners confirmed. Label definition (§13 D4). Gate X1/X2/X3 + new GATE 3 rule pre-registered |
| **CP1** | **Fri 2026-11-20** | Features computed on both grids (WikiText + TinyStories). Harm scores on the WikiText grid. Bench v0.1 builds. API serves results. UI renders real data | **GATE 1**. Go/no-go for X1/X2 pilots |
| **CP2** | **Fri 2026-12-11** | Risk classifier + conformal on run-table features. Full repair grid. Alarm calibrated. X2 subset trained | **GATE 3 (confirmatory, TinyStories)**, **Gate X1**, **Gate X2** (cut-off 2026-12-15) |
| **CP3 Integration** | **Mon 2027-01-04** | End-to-end *audit → repair → verify* from the UI. Bench release candidate. **Results freeze** | Gate X3 (cut-off 2027-01-15). Nothing new after this |
| **CP4 Paper** | **Fri 2027-01-29** | Each track writes its sections + figures. Merge into one draft | February = buffer + mini-viva/viva rehearsal. Final viva ~early March |

Weekly between checkpoints: async standup Mon/Thu in the team channel (template §15).

### 6.1 CP0 PR (Vishnu prepares, all approve), by 2026-11-03
- [x] Commit the 2026-10-31 results: updated `reports/grid/results.jsonl` + `summary.json`
      + `figures/collapse_curves.png` (benchmark_near re-run), `whitebox.jsonl`,
      `whitebox_summary.json`, `figures/whitebox_signals.png`, `earlywarning.json`. Do **not**
      commit `results.jsonl.bak-*`.
- [x] Commit `docs/HANDBOOK.md` (this file), `docs/STRETCH_GOALS.md`,
      `docs/LITERATURE_SURVEY.md`, and `CLAUDE.md` / `AGENTS.md` at the root.
- [x] CODEOWNERS changes from §5.
- [x] PRD §5 "Could" + TRD §6: add X1–X3 with a link to §11. TRD §3: replace Streamlit with
      the §8.6 stack.
- [x] Add `.gitignore`: `reports/**/*.bak-*`, `dashboard/web/node_modules/`, `dashboard/web/dist/`.
- [x] README status: benchmark_near re-run, GATE 3 not evaluable, pointer to this handbook.
- [ ] **At the CP0 meeting, before merging:** confirm or amend the *proposed* GATE 3
      confirmatory rule (§13.3), Gates X1/X2/X3 (§11) and label D4 (§13). Record the agreed
      numbers in this PR. They become binding when it merges, which must happen before the
      TinyStories grid runs.

---

## 7. Frozen interfaces (contracts between tracks)

All are **JSON Lines, UTF-8, one object per line, no BOM**, written atomically (write to a temp
file, then rename). Every row carries `schema` (string, e.g. `"features/1"`). A breaking
change bumps the version and needs §0-level agreement. Each owner commits a **fixture**
(5–20 rows of realistic fake data) at `tests/fixtures/<name>.jsonl` at CP0+3 days, so
consumers can build before the producer finishes.

**Shared keys.** `cell` = grid cell id (`"<type>_r<ratio:g>_s<seed>"`, e.g.
`"synthetic_r0.5_s1"`), `dataset` (`"wikitext2"` | `"tinystories"`), `type`, `ratio`
(float), `seed` (int). A grid's rows are identified by `(dataset, cell)`.

### 7.1 Grid results: exists (Track C produces)
`reports/grid/results.jsonl` (WikiText) and `reports/grid_tinystories/results.jsonl`
(new). Fields today: `cell, type, ratio, seed, n_train, n_contaminated, holdout_ppl,
benchmark_ppl, distinct_1, distinct_2, seconds, config, pool_version, shared_baseline`.
Consumers derive `dataset` from the file's path. Don't add fields to old rows.

### 7.2 Dataset features: Track A produces (`lex`, `leak` families from Track A; `model`, `sem` from Track C)
`reports/features/<dataset>.jsonl`, one row per grid cell (the cell's *training set*):
```json
{"schema":"features/1","dataset":"wikitext2","cell":"synthetic_r0.5_s1","type":"synthetic",
 "ratio":0.5,"seed":1,"n_samples":1000,"f_lex_ttr":0.41,"f_lex_entropy":9.87, "...":"...",
 "feature_version":"lex@1+model@1+sem@1"}
```
Every feature column is prefixed `f_<family>_` (`lex`, `model`, `sem`, `leak`). Values are
floats; `null` if not computable.

### 7.3 Per-sample scores: Track C (`risk_*`, from model/semantic features) and Track B (`harm_*`) produce
`reports/scores/<method>/<dataset>.jsonl`, one row per training sample of a cell:
```json
{"schema":"scores/1","dataset":"wikitext2","cell":"synthetic_r0.5_s1","sample_id":"...",
 "method":"harm_lora_grad","score":0.73,"higher_is":"worse","true_source":"synthetic"}
```
`true_source` is copied from the sample (ground truth, for evaluation only; **never a model
input**).

### 7.4 Risk predictions: Track A produces
`reports/risk/predictions.jsonl`: `{"schema":"risk/1","dataset","cell","p_degrade",
"label","conformal_set":[0,1]|[1]|[0],"alpha","model":"<name@version>","split":"train|cal|test"}`.

### 7.5 Repair results: Track B produces
`reports/repair/<dataset>.jsonl`: `{"schema":"repair/1","dataset","cell","strategy",
"k_replaced","holdout_ppl","benchmark_ppl","distinct_2","recovery","seed"}` where
`recovery = (ppl_contaminated − ppl_repaired) / (ppl_contaminated − ppl_clean)`.

### 7.6 Training-signal logs: Track C produces (X3, consumed by X2 and the UI)
`reports/trainlogs/<run_id>.jsonl`, one row every N steps:
`{"schema":"trainlog/1","run_id","step","holdout_loss","sig_anisotropy_last",
"sig_eff_rank_last",...,"alarm":false,"alarm_signal":null}`.

### 7.7 HTTP API: Track C produces, Track A consumes
FastAPI, read-only over `reports/**` plus job endpoints. The **OpenAPI spec is the contract**:
`dashboard/api/openapi.json`, committed and versioned. Frozen v1 by CP0+5 days:
`GET /v1/gates`, `GET /v1/grids/{dataset}/results`, `GET /v1/grids/{dataset}/curves`,
`GET /v1/features/{dataset}`, `GET /v1/risk`, `GET /v1/repair/{dataset}`,
`GET /v1/runs/{run_id}/trainlog` and `GET /v1/runs/{run_id}/stream` (SSE),
`POST /v1/audits` (upload a JSONL dataset → job id) and `GET /v1/audits/{id}`,
`GET /v1/bench/samples?dataset&type&ratio&limit`. Errors use RFC 7807 problem+json. The UI
generates its TypeScript client from the spec and never hand-writes request types.

---

## 8. Track A: Revanth (Modeling & Product UI)

**Mission:** turn a dataset into features → a calibrated, conformal risk score, and build
the product UI that makes ORIGIN-T usable. **Critical path:** features (§8.1) before
anything else.

### 8.1 Feature suite (`src/features/`), by CP1
Compute per-cell features on each grid cell's **training set**. Rebuild the training set
deterministically, the way `grid.py` does: `draw(seed)` → pool for the type →
`mix(human, pool, ratio, seed, total)`. **Import and reuse grid.py's own helpers**, don't
re-implement them. If one isn't importable, ask Vishnu to expose it (Track C, same day).
- [ ] `lex.py` (stdlib only): Shannon entropy (unigram), type-token ratio, distinct-1/2/3,
      repetition rate (repeated 3-grams), burstiness (variance/mean of token gaps), mean/var
      length. Hand-rolled. These are cheap and defensible.
- [ ] `leak.py`: max fuzzy similarity to the benchmark items (reuse
      `src.generation.benchmark_near.similarity`), fraction of rows above 0.9.
- [ ] `python -m src.features.suite --config configs/features.yaml` → writes §7.2 rows for
      every cell of a grid; resumable; `--dataset`. It calls every feature family through
      one function signature, `family(texts: list[str], ctx) -> dict[str, float]`, so
      Track C's `model.py` and `semantic.py` (§10.7) plug in without changes here. Until
      they land, the suite runs with `lex` + `leak` only and leaves their columns out.
- [ ] Tests on tiny stub corpora with hand-computed expected values (entropy of "a a b" etc.).
- *Moved to Track C (§10.7):* `model.py` (reference-LM features) and `semantic.py`
  (embedding, cluster and MAUVE features), including their per-sample scores.

### 8.2 Static baseline + GATE 1, by CP1
- [ ] Baselines: (a) an off-the-shelf AI-text detector score averaged per dataset;
      (b) a single feature, mean ref-LM loss. H2 must beat these.
- [ ] GATE 1 (TRD §11): "feature suite + static baseline run cleanly on ≥ 2 datasets".
      WikiText grid exists; TinyStories grid arrives from Track C by CP1 (§10.2).

### 8.3 Risk classifier + calibration (`src/risk/`), by CP2
- [ ] Label: §13 D4 (decided at CP0). Default: a cell is "degraded" if its holdout ppl
      exceeds its own seed's ratio-0 run by more than the baseline spread across seeds (the
      GATE 2/3 noise floor).
- [ ] **Evaluation design that isn't trivial:** damage rises with ratio by construction, so a
      model that only detects "how synthetic" looks good for free. Report **leave-one-type-out**
      and **leave-one-dataset-out** (train WikiText → test TinyStories) as the headline.
      In-distribution is secondary.
- [ ] Logistic regression (hand-rolled gradient descent first, then scikit-learn) and XGBoost.
      Reliability diagram + ECE via `scoring.py`.
- [ ] Write §7.4 predictions.

### 8.4 Split conformal, by CP2
- [ ] **Hand-roll** split conformal (nonconformity = 1 − p of the true class, quantile with the
      (n+1) correction) *before* trying MAPIE. Report empirical coverage via `scoring.py`.
      Small n (60–120 cells): report coverage with bootstrap CI and say n.

### 8.5 Survival / time-aware head: gated, only if ahead at CP2
Hand-rolled Kaplan–Meier + discrete-time hazard over ratio stages (TRD §5B). Skip
unless CP2 is green.

### 8.6 Product UI (`dashboard/web/`): built in parallel, against fixtures

**Product:** the "ORIGIN-T Console", a research-grade tool a team would actually use to
audit a dataset before fine-tuning. Think Linear / Grafana / Weights & Biases, not a demo
page.

**Stack (decided, §13 D6; no Streamlit, no Gradio):**

| Concern | Choice |
|---|---|
| Language / build | **TypeScript (strict)**, **React 18**, **Vite** |
| Package manager / runtime | **pnpm**, Node **20 LTS** |
| Routing + data | **TanStack Router** + **TanStack Query** (cache, retries, polling) |
| API client | Generated from `dashboard/api/openapi.json` with **openapi-typescript** + **openapi-fetch** |
| Styling | **Tailwind CSS** with a project token file (`src/styles/tokens.css`): colours, spacing, type scale. No default-palette classes in components |
| Accessible primitives | **Radix UI** (unstyled): dialog, popover, tabs, tooltip, select |
| Tables | **TanStack Table** (sorting, column pinning, virtualised rows) |
| Charts | **Observable Plot** (D3-based) for all charts; raw **D3** only where Plot can't |
| Live data | Server-Sent Events from `/v1/runs/{id}/stream` |
| Tests | **Vitest** + Testing Library (components), **Playwright** (e2e, 5 core flows), **axe-core** (a11y in Playwright) |
| Quality | ESLint (typescript-eslint, react-hooks, jsx-a11y) + Prettier; `pnpm typecheck && pnpm lint && pnpm test` in CI |
| Mock backend | **MSW** (Mock Service Worker) serving `tests/fixtures/*` through the OpenAPI shapes. The UI never waits for the API |

**Screens (in build order):**
1. **Overview:** gate board (0, 1, 2, 3, X1–X3) with status, rule text, numbers, link to PR.
2. **Collapse curves:** grid explorer: dataset / type / seed filters, holdout vs benchmark
   ppl, mean line + min–max band (matching `reports/figures/collapse_curves.png`), click a
   point → cell detail (features, white-box signals, sample browser).
3. **Audit:** upload a JSONL dataset (or pick a bench variant) → job progress →
   **risk report**: calibrated p(degrade) with the conformal set stated plainly ("At 95%
   coverage, the possible outcomes are {degrade}"), feature contributions, flagged samples
   table with per-sample score and reason, export to JSON/CSV.
4. **Repair:** strategies side by side (harm-guided, risk-guided, random, perplexity filter,
   oracle), recovery % with CI, before/after diversity; download the repaired dataset.
5. **Bench browser:** samples by type × ratio; near-duplicate diff view (original vs leaked
   copy, character-level diff).
6. **Live monitor (X3): built by Vishnu (§10.8)** in this app, with these design rules.
7. **Canary (X2): built by Vishnu (§10.8)** in this app, with these design rules.

Revanth owns the app shell, the tokens, the shared components (`src/components/`) and the
chart helpers. Vishnu's two screens use only those, so the app looks like one product.

**Design rules: this is what "not AI-sloppy" means here:**
- No gradients, glassmorphism, glowing borders, emoji, sparkle/robot icons, or "AI-powered"
  copy. No hero sections, no marketing language, no lorem ipsum, no placeholder charts.
- Neutral greys plus **one** accent, and semantic colours only for state (risk: red / amber /
  green with text labels, never colour alone). Contamination types use the **fixed palette**
  already in the figures: synthetic `#2a78d6`, recursive `#eb6834`, paraphrased `#1baf7a`,
  benchmark_near `#eda100`, human/control neutral grey.
- Typography: one sans (Inter or IBM Plex Sans) plus one mono (JetBrains Mono / IBM Plex Mono)
  for ids and numbers. **Tabular numerals** everywhere numbers align.
- Every number has its unit, n and uncertainty where one exists ("ppl 67.6 [67.4–67.7],
  3 seeds"). Every chart has axis titles, units and a one-line caption that says what to look at.
- Dense, scannable layouts. Tables over cards for lists. Keyboard-navigable (⌘K command palette
  for jumping to a cell or run). Designed empty, loading (skeletons) and error states. Light
  and dark mode via tokens.
- Accessibility: WCAG 2.1 AA contrast, focus rings, labels, `prefers-reduced-motion`.
- Copy is plain and specific: "Hidden-holdout perplexity rose 19% at 100% synthetic", not
  "Insights detected!".

**UI tasks**
- [ ] CP0+3d: scaffold (`pnpm create vite`, router, query, tailwind tokens, MSW, CI scripts),
      app shell with nav + ⌘K, Overview screen on fixtures.
- [ ] CP1: Collapse curves + cell detail on **real** data through the API (Track C serves it).
- [ ] CP2: Audit + Repair screens (job flow against `POST /v1/audits` mocks, then real).
- [ ] CP3: Bench browser; Playwright e2e for: load overview, filter curves, run an audit,
      compare repairs, browse bench samples. (Vishnu adds the e2e for his two screens.)
- [ ] `dashboard/web/README.md`: how to run (`pnpm i && pnpm dev`, with `VITE_API_URL`) and
      how to regenerate the client (`pnpm gen:api`).

---

## 9. Track B: Siddharth (Data, Repair & Bench, incl. X1)

**Mission:** make the contaminated data trustworthy, then **repair** it and prove the
repair works. Siddharth has the **strongest GPU**: Track B runs its own retrains and
Track C's X2 jobs (§12.2).

### 9.1 Data completeness, by CP1
- [ ] `duplicate` generator (`source="duplicate"` is in the schema, nothing produces it):
      exact and near-exact duplicates of *human* training rows (not benchmark items), at
      controlled rates. Tests. **Bump a pool version only if it changes an existing type.**
- [ ] WikiText heading lines (`= = History = =` ×129): decide to drop or keep `= … =` lines in
      the normalizer **for new datasets only**. Changing WikiText's normalizer would change
      fingerprinted data (§3.2). Record the decision in §13.
- [ ] Provide the TinyStories normalization + holdout for Track C's grid (§10.2):
      `normalize_corpora` already has the TinyStories entry (`max_rows: 50000`). Run holdout +
      leak check, and confirm sizes.

### 9.2 ORIGIN-T-Bench v0.1, by CP1
- [ ] `python -m src.bench.build --config configs/bench.yaml`: normalize → holdout → generate
      each type → mix at each ratio → leak check → write `bench/<dataset>/<type>/r<ratio>/s<seed>.jsonl`
      + `manifest.json` (counts, hashes, generator versions, config fingerprint, licence of
      each source). Reuse `grid.py`'s pools and draw. Don't regenerate differently.
- [ ] A datasheet (`bench/README.md`): sources, licences, intended use, known artefacts (the
      recasing story), how to cite.

### 9.3 X1: harm-based repair (`src/mitigation/`), pilot by CP1, full by CP2
Research and gate: §11 X1 and `docs/STRETCH_GOALS.md`.
- [ ] Harm scores per training sample, written as §7.3 with `method`:
      - `harm_lora_grad`: TracIn-style: dot product of each sample's LoRA-parameter gradient
        with the gradient of hidden-holdout loss, at the cell's trained checkpoint. LoRA only,
        so it's cheap.
      - `harm_repr_sim`: representation-similarity baseline (the method that beat influence
        functions in the literature). **Mandatory comparison.**
      - the `risk_*` per-sample scores from `model.py` / `semantic.py` (Track C, §10.7)
        when available.
- [ ] Repair at **equal size**: replace the top-k scored samples with held-back human samples
      (from `draw(seed)`'s unused pool, never the holdout). Strategies: harm-guided,
      repr-sim, risk-guided, random, perplexity filter, **oracle** (true labels).
- [ ] Verify: retrain each repaired set with the grid's `train` config → §7.5 rows with
      `recovery`. Also report `distinct_2`: aggressive selection can cut diversity.
- [ ] Pilot: 1 seed, ratio 0.5, synthetic + recursive → decide at CP1. Full: ratios {0.5, 1}
      × 4 types × 3 seeds.

### 9.4 Mitigation (core, PRD Must #5 / H5), by CP2
The core mitigation *is* the repair pipeline above plus **rare-cluster preservation**: when
replacing, never empty a semantic cluster that's rare in the human reference (use the cluster assignments
from `src/features/semantic.py`, Track C, §10.7). Compare with and without. That's H5.

### 9.5 Bench release, by CP3
Public HF dataset + GitHub release, versioned, with manifest hashes; license check.

---

## 10. Track C: Vishnu (Infra, Experiments & API, incl. X2, X3)

**Mission:** run every experiment correctly, serve results to the UI, and build the live
alarm and the canary.

### 10.1 CP0 PR (§6.1). First thing.

### 10.2 Second dataset + GATE 3 fix, by CP1
- [ ] **Healthy control arm:** a new contamination type `human_control` whose "pool" is
      *fresh human* samples (disjoint from the draw and the holdout). Its trajectories should
      never collapse, which gives GATE 3 its negatives. Add to grid.py with tests. New type →
      new config only, so existing fingerprints are unaffected.
- [ ] `configs/grid_tinystories.yaml` → `reports/grid_tinystories/`: 5 types (4 + control) ×
      5 ratios × 3 seeds. Estimated ~1.5–2 h on the RTX 2050 (cells ~75 s each on WikiText;
      verify with the smoke config first). Keep checkpoints, then white-box on them.
- [ ] Also run `human_control` on WikiText (new config `grid_wikitext_control.yaml`, new
      output_dir), ~15 min, so WikiText's GATE 3 becomes evaluable as **exploratory**.
- [ ] **GATE 3 confirmatory rule** (§13.3): written and merged at CP0, evaluated on TinyStories
      at CP2.

### 10.3 API (`dashboard/api/`), v1 frozen by CP0+5 days
- [ ] FastAPI + Pydantic v2 + uvicorn. Read-only routes over `reports/**` (§7.7), with an ETag
      from file mtimes. Job routes for audits run in a background worker. Plain
      `concurrent.futures` is enough; no Celery or Redis unless proven necessary.
- [ ] `python -m dashboard.api` serves on `:8000`. CORS for `localhost:5173`. Commit
      `openapi.json`. Contract test: every route in the spec returns schema-valid fixtures.
- [ ] `POST /v1/audits`: validates the upload against `Sample`, runs Track A's feature suite +
      risk model + per-sample scores, and returns the §7.4/§7.3 shapes. Size limit and a clear
      problem+json error on bad input.

### 10.4 X3: live collapse alarm, by CP2 (cut-off 2027-01-15)
- [ ] `src/evaluation/live.py`: a HF `TrainerCallback` **and** a hook for our `train_lora`
      loop. Every N steps, compute white-box signals on a fixed probe set (reuse
      `whitebox.py`) → §7.6 trainlog, plus SSE events.
- [ ] Monitors are **two-sided** (signals can move either way; anisotropy *falls* in our data).
- [ ] Calibrate the alarm threshold on healthy (`human_control` / ratio-0) runs with split
      conformal, and report the false-alarm rate on held-out healthy seeds. Optionally use the
      dataset risk from Track A as a prior (tighter monitoring for risky data).
- [ ] Re-run a subset of the grid with logging on, so curves over *training steps* exist.

### 10.5 X2: canary, by CP2 (cut-off 2026-12-15)
- [ ] Target model Pythia-410M (or Qwen2.5-0.5B) on **Siddharth's GPU** (§12.2). With ≥ 16 GB,
      plain LoRA (same code). Otherwise add 4-bit QLoRA (bitsandbytes) to `train_lora` behind a
      config flag.
- [ ] Subset: 4 types × ratios {0, 0.5, 1} × 3 seeds. Proxy = existing DistilGPT-2 runs, plus a
      reduced-learning-rate proxy variant (literature: fixes proxy ranking).
- [ ] Report Spearman ρ (proxy damage vs target damage) with bootstrap CI, and a calibrated
      predictor of target damage from proxy signals.

### 10.6 Reproducibility
Every run: config committed, seed fixed, MLflow logged, figure in `reports/`. Keep
`README.md`'s status section current at each checkpoint.

### 10.7 Model-based and semantic features (moved from Track A), by CP1
These follow Track A's feature-family signature, `family(texts, ctx) -> dict[str, float]`
(§8.1), with columns prefixed `f_model_` and `f_sem_` (§7.2). Revanth's suite CLI picks them up.
- [ ] `src/features/model.py`: reference-LM loss stats (mean, p90, std) under a **fixed**
      reference model (`gpt2`, never one of the grid's checkpoints), plus token-rank
      instability. Lazy torch.
- [ ] `src/features/semantic.py`: sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`):
      density (mean kNN distance), cluster entropy (k-means, fixed k and seed), tail-cluster
      retention vs the ratio-0 set, MAUVE vs the human draw (`mauve-text`, already in
      requirements). Expose the cluster assignments: Track B's rare-cluster preservation
      (§9.4) needs them.
- [ ] **Per-sample** versions (ref-LM loss, kNN distance) → §7.3 with `method:"risk_<name>"`.
      Track B's X1 compares against them.
- [ ] Tests with stub models and stub embeddings (no torch in CI), with hand-checked expected
      values.
- [ ] Run both families on the WikiText and TinyStories grids by CP1, so GATE 1 has full features.

### 10.8 UI screens for X3 and X2, by CP3
Built inside Revanth's app (`dashboard/web/`), using only his shell, tokens, shared
components and chart helpers (§8.6). Revanth reviews for consistency.
- [ ] `routes/monitor/`: **Live monitor (X3).** Streaming training signals from
      `/v1/runs/{id}/stream` (SSE), alarm threshold bands, alarm event log. Replays committed
      trainlogs when no run is live.
- [ ] `routes/canary/`: **Canary (X2).** Proxy-vs-target scatter with ρ and CI, and a
      predict-for-this-dataset card.
- [ ] Playwright e2e for both screens (one flow each), plus axe checks.

---

## 11. The three new features (stretch, gated)

Full research, prior art and sources: `docs/STRETCH_GOALS.md`. Summary:

| | **X1 Harm-based repair** | **X2 Canary run** | **X3 Live collapse alarm** |
|---|---|---|---|
| One line | Clean by *harm* (what raises held-out loss), not by "looks AI-written" | A 10-minute small-model fine-tune predicts a big model's damage | Training-time monitor that alarms with a controlled false-alarm rate |
| Owner | Siddharth (Track B) | Vishnu (Track C), runs on Siddharth's GPU | Vishnu (Track C); risk prior from Revanth |
| Closest prior art | Influence functions are fragile on LLMs; repr-similarity beats them (arXiv 2409.19998) | Proxies rank *useful* data for pretraining (2512.24503, 2509.21013) | "When Rank Rises as LLMs Degrade" (2610.09647): CUSUM rank monitors, union-bound FA |
| What's new for us | Ground-truth testbed + measured recovery on dose-response curves | Forecasting *harm*, calibrated, across scale | Conformal false-alarm guarantee; recursive-synthetic collapse; dataset-risk prior |
| **Gate (PROPOSED: confirm at CP0)** | Harm-guided recovers ≥ 10 pp more than random at ratio 0.5 on ≥ 2 of 3 collapse types, every seed | Spearman ρ ≥ 0.7 proxy vs target damage, bootstrap CI excludes 0 | At FA 5%: alarm before the held-out-loss breach on ≥ 50% of collapsing runs; observed FA ≤ 10% on held-out healthy runs |
| Cut-off | Pilot beats random by 2026-12-15, else report as negative | Target trainable by 2026-12-15, else drop | 2027-01-15 (ship callback even without guarantee) |

**The committed core ships first.** If time runs short, a stretch goal is cut, never the
feature suite, classifier, conformal or mitigation.

---

## 12. Environment, data and compute logistics

### 12.1 Setup (every member)
```bash
git clone https://github.com/SheeshDarth/OriginX-T.git && cd OriginX-T
python -m venv .venv && .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                                       # must pass before any work
python -m src.ingestion.normalize_corpora --config configs/corpora.yaml   # WikiText-2 + TinyStories
python -m src.ingestion.holdout --input data/processed/wikitext2_train.jsonl
```
UI (Revanth): Node 20 LTS + pnpm (`corepack enable`).

### 12.2 GPU allocation
| Machine | Runs |
|---|---|
| **Siddharth (strongest GPU)** | X1 repair retrains · X2 target-model runs (code from Vishnu) · his own bench builds |
| **Vishnu (RTX 2050, 4 GB)** | TinyStories grid · control arms · white-box · X3 development · smoke runs |
| Revanth | Feature extraction (small models: gpt2, MiniLM) on his machine or Vishnu's |

A **GPU calendar** (pinned team message, or `docs/TEAM_WORKPLAN.md`, git-ignored) lists booked
slots on Siddharth's machine. Around CP2 both X1 and X2 land there: **X1 pilot first, then the
X2 subset.** Always state the expected runtime when booking.

### 12.3 Moving artefacts between machines (prevents silent mismatch)
- **Pools aren't reproducible across machines.** Sampling depends on GPU and library
  versions. Anyone rebuilding a grid's training sets must **copy** `data/processed/grid_pools/`
  from the machine that ran the grid (Vishnu's, for WikiText and TinyStories). Never regenerate.
- **Checkpoints** (16.7 GB for WikiText) are copied, not retrained, for X1's gradient scoring:
  shared drive or an external disk. Verify with the white-box config's weights-hash check.
- Normalized corpora and holdouts *are* deterministic: regenerate them locally (§12.1), then
  confirm `check_leakage` prints OK.
- Pin library versions at CP0: `pip freeze > requirements.lock.txt` from Vishnu's environment,
  and use it on Siddharth's GPU machine.

---

## 13. Decisions register (settled; don't reopen without new data)

| # | Decision | Where / why |
|---|---|---|
| D1 | Hidden holdout carved by **text hash** | #6; ids let duplicate lines straddle the split |
| D2 | GATE 2 = pre-registered dose-response rule (monotone + above seed noise at 0.25) | #11, `configs/grid.yaml` `gate:` |
| D3 | benchmark_near: no whole-text recasing; pools versioned | #13; recasing hid memorisation |
| D4 | **Label for "degraded" (decide at CP0)**. Default: holdout ppl above own-seed ratio-0 + baseline spread | Same noise floor as GATE 2/3, no new threshold |
| D5 | `keep_models: true`; checkpoints live under `data/models/` on the run machine | White-box, X1 |
| D6 | **UI = React + TypeScript + Vite + TanStack + Tailwind tokens + Radix + Observable Plot; API = FastAPI. No Streamlit/Gradio** | Product-grade UI is a deliverable (§8.6) |
| D7 | WikiText GATE 3 results are **exploratory**. Confirmatory GATE 3 on TinyStories with a rule fixed at CP0 | 2026-10-31: GATE 3 not evaluable (no negatives); anisotropy fell, against the pre-registered +1 direction |
| D8 | Stretch X1–X3 adopted with gates and cut-offs (§11) | 2026-10-31 |

### 13.3 GATE 3 confirmatory rule (PROPOSED: confirm at CP0, merge before TinyStories runs)
Proposed: signals are **two-sided** (direction learned on WikiText, fixed for TinyStories:
anisotropy −1). Negatives = `human_control` trajectories. Pass = at least one signal beats
the permutation baseline at Bonferroni α = 0.05 *and* has median lead > 0 stages at false-alarm
rate ≤ 5%. Fill in exact numbers at CP0, then **don't change them**.

---

## 14. Known pitfalls (lessons already paid for)

1. **Case-folded "near-duplicates" aren't near to the model.** Similarity checks are case-blind;
   GPT-2 tokenization isn't (#12, #13).
2. **Stale pools after generator changes**: bump `POOL_VERSIONS` (§3.3).
3. **Effective rank can rise as LLMs degrade** (duplication / dispersion). Use two-sided monitors.
4. **Grid cells are ~75 s**, not minutes. A full 60-cell grid is ~1.5–2 h on 4 GB. Don't book
   "overnight" for it.
5. **The background job limit in agent tools is 2 h.** For longer runs, launch a detached process
   with a log file (`Start-Process ... > logs/x.log`). Runs are resumable.
6. **Gate validation rejects configs with no collapse type.** Throwaway diagnostic configs must
   set `gate.collapse_types` to types they contain.
7. **PowerShell writes BOMs and doesn't glob for native commands** (§3.10, §4).
8. **Multiple GitHub accounts on one machine:** pushes can go out as the wrong account (403 or
   wrong author). Check `gh auth status` and the commit author before pushing.
9. **Stale `.pyc` after mutation testing** can fake failures: `python -B`, `-p no:cacheprovider`.
10. **Base DistilGPT-2 can't follow a rewrite instruction**, so "paraphrased" is closer to
    prompted continuation. Say so in the paper, or use an instruct model for X2-scale runs.

---

## 15. Communication

**Standup (Mon/Thu, async, 5 lines):**
```
Track: A/B/C — <name>
Done since last: <PRs / results with numbers>
Next: <task id from §8/9/10>
Blocked by: <nothing | interface/fixture/GPU slot + who>
GPU needed: <hours, which machine, when>
```
**Escalation:** a blocker older than 24 h goes to the team channel, tagging the owner. A gate
rule or interface change waits for the next checkpoint unless all three agree sooner.
**Weekly HOD update:** template in `docs/PROJECT_REPORT.md` §7. Vishnu compiles it from the
standups.

---

## 16. Glossary

- **Cell:** one fine-tune in a grid: (type, ratio, seed). **Trajectory:** one (type, seed)
  across ratios.
- **Pool:** generated contamination for a type and seed, mixed into human data at a ratio.
- **Ratio:** share of the training set replaced by contamination (0, 0.25, 0.5, 0.75, 1).
- **Hidden holdout:** human texts never trained on, used to measure damage (holdout ppl).
- **Benchmark items:** 200 WikiText validation texts used as "leaked" eval items for
  benchmark_near.
- **Fingerprint:** hash of a grid config's result-affecting keys. Results with another
  fingerprint are refused.
- **Pre-registered:** a rule merged before the results it judges exist.
- **Recovery:** share of the damage a repair undoes (§7.5).

---

## 17. Issue tracker (GitHub Issues)

All work is tracked in [GitHub Issues](https://github.com/SheeshDarth/OriginX-T/issues),
with one **milestone per checkpoint** (CP0 Kickoff, CP1, CP2, CP3 Integration, CP4 Paper)
and labels `track:A|B|C`, `team`, `X1|X2|X3`, `stretch`, `feature`, `research`, `infra`,
`ui`, `gate`, `docs`, `blocked`. Each issue has the task's acceptance criteria, its owner,
its milestone and its dependencies. A new task gets an issue *before* work starts.

| Track | Task id → issue |
|---|---|
| Team | T1 #21 · T2 #22 · T3 #23 |
| C: Vishnu | C1 #24 · C2 #25 · C3 #26 · C4 #27 · C5 #28 · C6 #29 · C7 #30 · C8 #31 · C9 #32 · C10 #33 · C11 #34 · C12 #35 · C13 #36 · C14 #37 · C15 #38 · C16 #39 · C17 #40 · C18 #41 · C19 #42 |
| A: Revanth | A1 #43 · A2 #44 · A3 #45 · A4 #46 · A5 #47 · A6 #48 · A7 #49 · A8 #50 · A9 #51 · A10 #52 · A11 #53 · A12 #54 · A13 #55 |
| B: Siddharth | B1 #56 · B2 #57 · B3 #58 · B4 #59 · B5 #60 · B6 #61 · B7 #62 · B8 #63 · B9 #64 · B10 #65 · B11 #66 · B12 #67 |

**First issues, in order:** CP0: #21 (all), #24, #25 (Vishnu), #43, #44 (Revanth), #56, #57
(Siddharth). Then the CP1 unblockers: **#26** (C3, unblocks Revanth's suite), **#58** (B3,
unblocks the TinyStories grid), **#27** (C4, healthy control arm).
