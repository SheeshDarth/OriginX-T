# Contributing to ORIGIN-T

## Workflow

1. Pick a story from the current sprint (GitHub Projects board).
2. Branch: `git checkout -b <initials>/<short-story-name>`.
3. Work in small, reviewable commits.
4. Open a PR against `main`; request review from one teammate.
5. Merge once the Definition of Done is met.

## Definition of Done (every story)

- [ ] Config-driven, fixed seed — no magic numbers in code.
- [ ] Reproducible on Colab/Kaggle from a clean clone.
- [ ] Smoke test passes (`pytest -q`).
- [ ] Run logged to MLflow/W&B (offline mode is fine).
- [ ] Figure/table committed to `reports/`.
- [ ] Peer-reviewed by one teammate (PR approval).
- [ ] One-line entry added to your personal logbook.

## From-scratch-before-library rule

Survival analysis and conformal prediction are graded on understanding, not
just usage. Hand-roll Kaplan-Meier and split-conformal **before** switching
to `lifelines`/`pycox` or `MAPIE` — keep the from-scratch notebook in
`reports/` as evidence for the monthly mini-viva.

## Commit style

Short imperative subject line (`Add recursive contamination generator`),
body explains *why* if not obvious. Reference the story/issue number.

## Code review

One teammate must approve every PR. Look for: does the DoD checklist hold,
is the config reproducible, are magic numbers/paths avoided, is the metric
computed correctly.
