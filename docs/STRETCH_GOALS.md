# ORIGIN-T — Stretch goals: research and plan (draft, not yet agreed)

Status: **proposal**, written 2026-10-31 for the team to review. Not part of the committed
core (PRD §5 "Must"). Owners are left blank on purpose: the split is a team decision.
Nothing here changes `docs/PRD.md` or `docs/TRD.md` until the team agrees.

Three extensions are proposed, all building on what already exists: the Sprint-6 grid,
57 kept checkpoints, the white-box signals and the GATE 3 harness. Each one has a prior-art
check, the angle that is new as far as the search found, a design, a go/no-go gate and a
cut-off date. **The committed core ships first.** If time runs short, a stretch goal is
cut, never the feature suite, risk classifier or mitigation.

> Search scope and caveat: one round of web searches on 2026-10-31 (sources at the end).
> It found close neighbours for every idea, but it is not a systematic review. Before
> writing the paper, repeat the search on arXiv/Semantic Scholar with the exact framing,
> and re-check anything dated after this file.

---

## X1 — Clean by harm, not by origin (harm-based data repair)

**Idea.** Score every training sample by how much it *hurts* the model (raises held-out
loss), not by whether it *looks* machine-written. Then remove or replace the most harmful
samples, retrain, and measure recovery against the grid's contaminated and clean runs.

**Why it matters.** AI-text detection is unreliable (survey #18: no detector holds across
contamination tiers), so cleaning by origin is on weak ground. Harm is what users care
about, and it also catches harmful *human* text.

**Prior art found.**
- Influence functions on LLMs are fragile. One study found them *ineffective* at
  flagging harmful fine-tuning data, beaten by a simple **representation-similarity
  baseline** [A1]. Influence estimates can correspond only weakly to the effect of actually
  deleting or rewriting the samples [A2].
- Concept Influence (2026) attributes behaviour to semantic directions (probes, SAE
  features) and reports cleaner separation of harmful from benign examples than classical
  influence functions [A3, A4]. Influence also changes over training [A5].
- Collapse mitigation by filtering or verification works when the verifier is good
  [M1, M2]. A poor or insular selector can itself *cause* diversity collapse [M3].
  Reweighting rather than hard removal is an alternative [M4].
- LoRA-gradient data selection exists (LESS, Xia et al. 2024; cite after checking).

**What looks new.** No source found applies data attribution to *collapse/contamination
damage* on a testbed with **ground-truth labels** (we know which samples are synthetic,
recursive or leaked) and **measured recovery on dose-response curves**. The grid makes
the harm score checkable, which most attribution papers cannot do.

**Design.**
1. Harm scores, compared head-to-head on the same grid cells:
   - (a) LoRA-gradient similarity to the held-out loss gradient (TracIn-style, cheap with
     LoRA);
   - (b) representation similarity (the baseline that beat influence functions in [A1]);
     **this must be included**;
   - (c) the feature suite's per-sample risk, once it exists.
2. Repair at **equal dataset size**: remove-and-replace with held-back human samples.
   Removal alone shrinks the data and confounds the result.
3. Baselines: random replacement, perplexity filter, and an **oracle** using the true
   labels (the upper bound).
4. Verify: retrain on each repaired set and report recovery =
   (ppl_contaminated − ppl_repaired) / (ppl_contaminated − ppl_clean).
5. Also report diversity (distinct-n). Per [M3], aggressive selection can buy loss at
   the cost of diversity.

**Gate X1 (fixed before results).** Harm-guided repair's recovery is ≥ 10 percentage
points higher than random replacement's at ratio 0.5 on ≥ 2 of the 3 collapse types, on every seed. (PRD's mitigation
target is 10–20% relative recovery.)
**Cost.** 2–3 weeks. About 36 retrains (ratios 0.5 and 1 × 3 collapse types + benchmark_near
× 3 seeds, per strategy subset), roughly one overnight GPU run.
**Cut-off.** If the harm score can't beat random on a 1-seed pilot by **2026-12-15**,
fall back to (b) or (c) only and report attribution as a negative result.

---

## X2 — Canary run: a small cheap fine-tune that predicts the big one

**Idea.** Before an expensive fine-tune, run a 10-minute fine-tune of a small model on the
candidate data, read its signals, and predict how much a *larger* model will degrade.

**Prior art found.**
- Small proxies are routinely used to rank datasets for *pretraining* and can mislead.
  Proxies can rank datasets differently from the target model. A reduced proxy learning
  rate restored a ~0.97 correlation [P1]. Smaller proxies are noisier and sometimes point
  the wrong way [P2].
- Proxy-tuning steers a large model with a small tuned one at decoding time. That's a
  different goal [P3].
- Predictability across scale cannot be settled from proxy experiments alone. Large-scale
  checks are needed [P4].

**What looks new.** The search found proxies used to rank *useful* data, not to forecast
*harm* (collapse or contamination damage) in a larger model, and none with a calibrated
prediction.

**Design.**
1. Target model: Pythia-410M or Qwen2.5-0.5B (allowed by TRD §4). It won't train on the
   4 GB RTX 2050 without 4-bit QLoRA, so plan on Kaggle/Colab (T4 16 GB).
2. Run the target on a **subset** of the grid (e.g. 4 types × ratios {0, 0.5, 1} ×
   3 seeds), not the full grid.
3. Proxy = the existing DistilGPT-2 runs. Add the reduced-learning-rate proxy variant from [P1].
4. Report: rank correlation of proxy damage vs target damage, and a calibrated predictor
   of target damage from proxy signals, with conformal intervals.
5. This experiment also answers the main reviewer critique ("does it hold beyond 82M?").

**Gate X2.** Spearman ρ ≥ 0.7 between proxy-predicted and target-measured damage across
the subset cells, with the bootstrap CI excluding 0.
**Cost.** About 2 weeks, plus about 1–2 days of T4 compute.
**Cut-off.** If the target model can't be trained reliably by **2026-12-15**, drop X2.
The scale question then goes in the paper's limitations.

---

## X3 — Live collapse alarm during training

**Idea.** A Hugging Face `Trainer` callback that computes the white-box signals every N
steps on a fixed probe set, raises an alarm with a controlled false-alarm rate, and can
stop the run. It makes PRD H4 (warning *before training finishes*) real. The current
GATE 3 "lead time" runs across contamination ratios, not training time.

**Prior art found. This is the most crowded of the three.**
- **"When Rank Rises as LLMs Degrade"** (Wang, Oct 2026) [R1] is the closest. It uses
  two-sided per-channel CUSUM monitors on RankMe, effective rank, k95 and anisotropy during
  post-training (Qwen3-0.6B, dolly-15k), with a union-bound false-alarm target. Key
  findings:
  - **under data duplication, rank RISES while held-out loss rises ~75%** ("dispersion,
    not collapse");
  - raw intermediate-layer effective rank is pinned near 1 by massive activations;
  - the monitor led the loss breach in only 1 of 9 folds;
  - it still false-alarmed on a held-out healthy seed.
  It is not conformal or anytime-valid, and it only monitors during training (no
  dataset-side prediction).
- Per-checkpoint collapse indices exist for sequential post-training [R2]. Topology-based
  online early-warning statistics exist [R3]. SIGMA (2026) monitors spectral signals of
  LLM model collapse; its full method was not readable in this search [R4].

**What looks new.** Given [R1], a rank monitor alone is **not** new. What's still open:
1. a **conformal / anytime-valid** false-alarm guarantee (none found);
2. monitoring specifically **recursive synthetic-data collapse** ([R1] studies
   duplication, high learning rate and domain shift);
3. **fusing** dataset-side risk (the feature suite) with in-training signals, which
   neither [R1] nor [R2] does.

**Design.**
1. Callback logs the existing `src/evaluation/whitebox.py` signals every N steps. The same
   logs feed X2.
2. Two-sided monitors (not one-sided), following [R1]: a signal may move either way.
3. Calibrate the alarm on healthy (ratio-0) runs with split-conformal, or with an
   anytime-valid e-process if exchangeability fails. Report the empirical false-alarm rate
   on held-out healthy seeds.
4. Prior from the dataset: start the monitor with the feature-suite risk score, so a risky
   dataset is watched more tightly.

**Gate X3.** At a pre-set false-alarm rate of 5%, the alarm fires before the held-out-loss
breach on ≥ 50% of collapsing runs, and the observed false-alarm rate on held-out healthy
runs is ≤ 10%.
**Cost.** 1–2 weeks, light compute (reuses the white-box code).
**Cut-off.** **2027-01-15**. X3 is also the viva/dashboard demo, so a working callback
without the guarantee is still useful to ship.

---

## Cross-cutting finding: check the GATE 3 signal directions NOW

`configs/earlywarning.yaml` pre-registers effective rank and participation ratio as
**falling** with damage (−1). [R1] reports rank **rising** under duplication-style
degradation in LLM post-training. Our `benchmark_near` arm cycles 200 items into 1,000
rows, which is duplication. Two cheap actions, before anyone reads the GATE 3 result:
- note in the GATE 3 write-up that a signal moving the "wrong" way counts as *not warning*
  under the current rule. That's correct, and it's pre-registered;
- for future rules, consider two-sided signals, as [R1] recommends.
Also check that `whitebox.py` measures effective rank on a representation that isn't
dominated by massive activations. Its config already excludes the first token
(attention sink), which is the issue [R1] describes.

---

## Schedule sketch (owners TBD)

| Item | Depends on | Est. | Cut-off |
|---|---|---|---|
| Feature suite (core) | — | 2–3 wk | critical path |
| X3 live alarm | white-box (done) | 1–2 wk | 2027-01-15 |
| X1 harm-based repair | grid pools (done); feature suite for score (c) | 2–3 wk | pilot by 2026-12-15 |
| X2 canary | X3 logging; T4 compute | ~2 wk | target trainable by 2026-12-15 |

All three together add about 6 weeks. Code-complete about early Feb 2027, within the March
finish, with no slack. That only works if the work runs in parallel.

## Before building

1. Team agrees on owners and adds X1–X3 to PRD §5 ("Could") and TRD §6 with the gates
   above. Gates are fixed before results, as for GATE 2/3.
2. Re-run the literature search with the exact framing of each idea, especially X3 after [R1].
3. Pilot each idea on one seed before committing compute.

---

## Sources (searched 2026-10-31)

- [A1] Do Influence Functions Work on Large Language Models? — https://arxiv.org/html/2409.19998v1
- [A2] From Reweighting to Rewriting: Intervention Effects of Influential Samples — https://arxiv.org/pdf/2609.02771
- [A3] Concept Influence (2026) — https://arxiv.org/html/2602.14869
- [A4] FAR.AI blog on Concept Influence — https://www.far.ai/blog/concept-data-attribution-02-2026
- [A5] Timaeus, influence dynamics — https://timaeus.co/research/2025-10-05-influence-dynamics
- [M1] Escaping Model Collapse via Synthetic Data Verification — https://arxiv.org/pdf/2510.16657
- [M2] ICLR 2026 version of [M1] — https://proceedings.iclr.cc/paper_files/paper/2026/file/29806b2e4dcb35e42abed21d43e2b7d8-Paper-Conference.pdf
- [M3] When Sample Selection Bias Precipitates Model Collapse — https://arxiv.org/pdf/2606.13732
- [M4] Recursive Learning Without Collapse: A Weighting-Based Stabilization Framework — https://arxiv.org/pdf/2502.18049
- [P1] Can Small Training Runs Reliably Guide Data Curation? — https://www.alphaxiv.org/abs/2512.24503
- [P2] Predicting LLM Reasoning Performance with Small Proxy Model — https://arxiv.org/pdf/2509.21013
- [P3] Tuning Language Models by Proxy — https://arxiv.org/abs/2401.08565
- [P4] Validity Threats for Foundation Model Research — https://arxiv.org/pdf/2606.05029
- [R1] When Rank Rises as LLMs Degrade (Oct 2026) — https://arxiv.org/html/2610.09647
- [R2] Representation Collapse in Sequential Post-Training of LLMs — https://arxiv.org/pdf/2605.30524
- [R3] Monitoring Neural Training with Topology — https://arxiv.org/pdf/2604.26984
- [R4] SIGMA: Scalable Spectral Insights for LLM Model Collapse — https://arxiv.org/pdf/2601.03385
