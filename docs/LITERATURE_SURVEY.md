# ORIGIN-T — Literature Survey (Phase I Review, 23AML175)

**Project:** ORIGIN-T — Time-Aware Dataset Forensics & Early-Warning Risk
Prediction for LLM Fine-Tuning
**Team:** Siddharth (S1, Data & Bench) · Revanth (S2, Modeling) · Vishnu (S3, Experiments/Infra)
**Review date:** 18 September 2026 · **Papers surveyed:** 30 core + 10 foundational

---

## 1. Survey method (state this on the slide — examiners ask)

| Item | Detail |
|---|---|
| Databases | arXiv (cs.CL / cs.LG / stat.ML), ACL Anthology, IEEE Xplore, Springer, Nature, PeerJ, ACM DL |
| Query strings | `model collapse`, `recursive training synthetic data`, `benchmark data contamination`, `machine-generated text detection`, `data valuation`, `conformal risk control`, `discrete-time survival`, `time-aware state space model`, `pretraining data curation` |
| Window | 2023–2026 for the core set (foundational method papers 2007–2022 listed separately) |
| Inclusion | Peer-reviewed venue **or** arXiv preprint with a named acceptance / journal_ref / heavy citation; must touch dataset quality → model degradation, contamination detection, or the risk-modelling machinery ORIGIN-T needs |
| Exclusion | Vision-only collapse studies without a text analogue; vendor whitepapers; blog posts |
| Verification | Every entry below was resolved against the **arXiv API**, **Crossref DOI API**, or **ACL Anthology** on 16 Sep 2026 — titles, authors and venues are as returned, not as remembered |

**Honesty note for the viva:** entries marked *arXiv preprint* have no confirmed
venue. Say "preprint" on the slide. Do not upgrade a preprint to a conference
name you have not verified — that is the easiest thing for an examiner to catch.

---

## 2. Taxonomy — six pillars

ORIGIN-T sits at the intersection of six literatures. The survey is organised
the same way, so every pillar maps to a module in the repo.

```
P1 Model collapse           -> the phenomenon we must reproduce   (GATE 0)
P2 Synthetic-data mixing    -> the contamination grid we generate (src/generation)
P3 Benchmark contamination  -> the leakage axis + hidden holdout  (src/ingestion/check_leakage.py)
P4 Provenance detection     -> the feature suite + baselines      (src/features)
P5 Risk / survival / UQ     -> the early-warning model            (src/risk)
P6 Time-aware + geometry    -> the trajectory head + white-box    (src/risk, src/evaluation)
```

---

## 3. Master comparison table (30 papers)

> Put **rows 1–24** on the slides (the circular asks for ≥20). Keep rows 25–30
> plus the "additional reading" list in the handout.

### P1 — Model collapse: the phenomenon and its theory

| # | Paper / Venue / Year | Problem addressed | Method | Key result | Limitation / open gap | ORIGIN-T uses it for |
|---|---|---|---|---|---|---|
| 1 | Shumailov et al., **Nature** 631:755–759, **2024** | Does recursive training on model output degrade models? | Iterative retrain of LM / VAE / GMM on their own generations | Distribution **tails vanish first**; collapse is irreversible once the tail is gone | Purely *descriptive* — no prediction, no risk score, needs a full retrain to observe | The phenomenon ORIGIN-T predicts. GATE 0 replication target |
| 2 | Alemohammad et al., *Self-Consuming Generative Models Go MAD*, arXiv:2307.01850, **2023** | Autophagous (self-consuming) training loops | Fresh-data / synthetic-only / fixed-real loop regimes | Without enough **fresh real data** each generation, quality or diversity decays ("MAD") | Mostly image domain; no per-dataset risk score | Justifies the human-anchor ratio in the safe-mix optimizer |
| 3 | Dohmatob et al., *A Tale of Tails: Model Collapse as a Change of Scaling Laws*, **ICML 2024** | Why does collapse break scaling? | Analytic + empirical scaling-law study | Synthetic data **changes the scaling exponent** — more compute stops helping | Idealised settings; no detector | The collapse-scaling-law fit (gated module, TRD §6) |
| 4 | Dohmatob et al., *Strong Model Collapse*, arXiv:2410.04840, **2024** | How little synthetic data is already harmful? | Random-matrix / regression analysis | Even a **tiny** synthetic fraction causes non-vanishing degradation | Linear-model idealisation; LLM transfer unproven | Motivates including *low* ratios (25 %) in the sweep, not just 50–100 % |
| 5 | Gerstgrasser et al., *Is Model Collapse Inevitable?*, arXiv:2404.01413, **2024** | Is collapse avoidable? | Replace-vs-**accumulate** regimes; LMs + diffusion + VAE | **Accumulating** real data alongside synthetic bounds test error — collapse avoided | Answers *whether*, not *how much*, *when*, or *for this dataset* | The "risk is a function of mixture policy" framing; mitigation baseline |
| 6 | Seddik et al., *How Bad is Training on Synthetic Data?*, arXiv:2404.05090, **2024** | Quantify collapse vs. ratio and depth | Statistical analysis of LM collapse | Severity scales with **generation depth** and **synthetic proportion** | No sample-level features, no early warning | Direct support for hypothesis **H3** (trajectory-dependence) |
| 7 | Shi et al., *A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective*, **NeurIPS 2025 (Spotlight)** | What mechanism drives collapse? | Generalization→memorization analysis across generations | Collapse reframed as drift from generalising to **memorising the parent model** | Mechanistic, not operational — no dataset audit tool | Theory backing the white-box signals (effective rank, neuron death) |
| 8 | Xu, He & Cheng, *A Probabilistic Perspective on Model Collapse*, arXiv:2505.13947, **2025** | When is collapse statistically guaranteed? | Probabilistic framework over iterative retraining | Gives sample-size regimes under which collapse does / does not occur | Assumes clean generational structure; no empirical LLM audit | Formal grounding for the τ (first-degradation-stage) definition, TRD §2 |
| 9 | Barzilai & Shamir, *When Models Don't Collapse: On the Consistency of Iterative MLE*, arXiv:2505.19046, **2025** | Counter-case: when is iteration safe? | Consistency analysis of iterative maximum-likelihood estimation | Under stated conditions iterative MLE stays consistent — collapse is **not universal** | Theoretical; narrow estimator class | Stops ORIGIN-T over-claiming; defines the honest "no-degrade" class label |
| 10 | Wang et al., *LLM Web Dynamics: Tracing Model Collapse in a Network of LLMs*, arXiv:2506.15690, **2025** | Collapse at web/ecosystem scale | RAG-coupled network of LLMs simulating the open web | Collapse **propagates through networks** of models, not just single retrain loops | Simulation; no per-dataset decision output | Motivation slide — why dataset-level auditing matters ecosystem-wide |
| 11 | Xie & Hu, *Reviewing Model Collapse and Countermeasures*, **IEEE AAIML 2026** | Consolidated causes + fixes | Systematic review | Organises causes (statistical error, functional approximation error) and countermeasure families | Review only; no new benchmark or predictor | The most recent P1 survey — related-work backbone |

### P2 — Synthetic-data mixing, scaling and mitigation

| # | Paper / Venue / Year | Problem addressed | Method | Key result | Limitation / open gap | ORIGIN-T uses it for |
|---|---|---|---|---|---|---|
| 12 | Kang et al., *Demystifying Synthetic Data in LLM Pre-training*, **EMNLP 2025 (Main)** | Do scaling laws hold for synthetic mixtures? | >1000 LLMs, >100k GPU-hours; natural vs. rephrased vs. textbook-synthetic | **Rephrased** synthetic ≈ no degradation, ~30 % near-optimal; **pure-generated textbook** synthetic reproduces collapse patterns | Pre-training scale only — costs ~100k GPU-hr to learn what ORIGIN-T wants to *predict* cheaply | Strongest single justification for ORIGIN-T: the answer is currently unaffordable |
| 13 | Du et al., *Characterizing Model Behavior Under Synthetic Data Training*, arXiv:2510.05133, **2025** | Can anything warn **before** accuracy drops? | Empirical sweep across scales and mixing ratios, tracking calibration | **ECE is an early-warning signal**: at 30 % synthetic, ECE 0.08→0.14 while accuracy barely moved; a ≥50 % ECE rise predicted next-iteration degradation with high precision/recall | One hand-picked metric; no risk model, no dataset-side features, no false-alarm guarantee | **Closest prior work to H4.** ORIGIN-T generalises it into a calibrated multi-feature risk model — cite as the baseline to beat |
| 14 | Feng et al., *Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification*, arXiv:2406.07515, **2024** | Can verification rescue synthetic data? | Verifier / pruner applied to synthesised data before training | **Verified** synthetic data scales; unverified collapses | Verifier is task-specific; no calibration, no lead-time | Mitigation baseline (E4): "verify" vs. ORIGIN-T's "score-then-safe-mix" |
| 15 | Yi et al., *Escaping Model Collapse via Synthetic Data Verification*, arXiv:2510.16657, **2025** | Near-term vs. long-term effect of verifiers | Theory + experiments across verifier strengths | Stronger verifiers (more real data) give larger gains; weak verifiers plateau or hurt | Generative-quality (FID) framing; no dataset audit report | Sets the expectation curve for our 10–20 % mitigation-recovery target |
| 16 | Kazdan et al., *Collapse or Thrive? Perils and Promises of Synthetic Data*, **NeurIPS 2024 Workshops (M3L, ATTRIB)** | Reconcile contradictory collapse results | Unified replace / accumulate / mix experimental study | Outcome depends on the **data-accumulation regime** — "collapse" and "no collapse" results are compatible | No predictive tooling | Experimental-design template for E2 (collapse curves) |
| 17 | Wright et al., *What and Whose Knowledge? Measuring Epistemic Diversity in LLMs*, **EMNLP 2026** | Is diversity loss measurable in deployed models? | 27 LLMs × 155 topics × 12 countries; claim-level diversity metric | Diversity improved over 3 years but still below search engines; skews English / global-North | Output-side measurement, not tied to a training dataset | Diversity & coverage metric design for the eval harness (cluster coverage, tail retention) |

### P3 — Benchmark contamination and leakage

| # | Paper / Venue / Year | Problem addressed | Method | Key result | Limitation / open gap | ORIGIN-T uses it for |
|---|---|---|---|---|---|---|
| 18 | Nourbakhsh et al., *Are LLM Benchmarks Already Contaminated? A Systematic Review*, **GEM Workshop 2026** | State of contamination detection | Systematic review of 55 studies to late 2025; 4-tier taxonomy (Exact / Syntactic / Semantic / Task-level) | **No detection method is consistently reliable** across tiers, access settings and training stages; inflation ≈6–40 % | Review; proposes a transparency card, not a detector | Newest proof the gap is real; adopt the 4-tier taxonomy for our `benchmark_near` labels |
| 19 | Xu et al., *Benchmark Data Contamination of LLMs: A Survey*, arXiv:2406.04244, **2024** | Taxonomy of BDC and mitigations | Survey, 31 pp. | Organises contamination types + detection families; **semantic** contamination is the weak spot | Pre-2024 coverage; static-evaluation focus | Framing of the contamination axis in ORIGIN-T-Bench |
| 20 | Dekoninck et al., *ConStat: Performance-Based Contamination Detection*, arXiv:2405.16281, **2024** | Detect contamination with no training-data access | Statistical comparison of scores on a benchmark vs. a matched reference benchmark | Contamination is detectable **behaviourally**, as an abnormal performance *gap* — no n-gram overlap needed | Needs a reference benchmark + model family; per-model, not per-dataset | Direct template for our behavioural leakage feature (inflated vs. rephrased / hidden score) |
| 21 | Fu et al., *Does Data Contamination Detection Work (Well) for LLMs?*, **NAACL 2025 (Findings)** | Are detection assumptions sound? | Survey + evaluation of 50 papers; formal contamination definitions | Methods rest on assumptions that often fail; results are **not comparable** across papers | Diagnostic; offers no replacement method | Why ORIGIN-T reports honest cross-generator numbers instead of one headline AUROC |
| 22 | Chen et al., *LLM Benchmarks against Data Contamination: From Static to Dynamic Evaluation*, arXiv:2502.17521, **2025** | Can benchmarks be made contamination-resistant? | Survey of static→dynamic evaluation designs | Static patches see limited adoption; **dynamic / regenerated** benchmarks are the trend | Survey; no detection contribution | Justifies ORIGIN-T-Bench's **hidden holdout** (already implemented in `holdout.py`) |
| 23 | Yang et al., *Rethinking Benchmark and Contamination with Rephrased Samples*, arXiv:2311.04850, **2023** | Does paraphrase defeat decontamination? | Rephrased test samples injected into training | Standard n-gram decontamination **misses** rephrased leakage; scores inflate massively | Their detector is LLM-based and costly | The exact failure mode `src/generation/benchmark_near.py` reproduces |
| 24 | Oren et al., *Proving Test Set Contamination in Black Box Language Models*, arXiv:2310.17623, **2023** | Contamination proof without weights | Exchangeability test over example ordering | Provides a **statistical guarantee** of contamination in black-box models | Needs a canonically-ordered benchmark; binary output, no severity | Statistical-rigour model for our leakage gate; complements `check_leakage.py` |

### P4 — Provenance / machine-generated-text detection

| # | Paper / Venue / Year | Problem addressed | Method | Key result | Limitation / open gap | ORIGIN-T uses it for |
|---|---|---|---|---|---|---|
| 25 | Drayson, Yilmaz & Lampos, *Machine-generated text detection prevents language model collapse*, **EMNLP 2025 (Main)** | Can a detector stop collapse? | MGT detector + **importance resampling** to up-weight likely-human data; GPT-2 & SmolLM2, 124M–1.7B | Prevents collapse **and beats training on purely human data** — curation beats purity | Binary human/synthetic only; no recursion depth, no leakage axis, no risk calibration | Strongest baseline for **H2**. ORIGIN-T extends detection → *degradation prediction* |
| 26 | Dugan et al., *RAID: A Shared Benchmark for Robust Evaluation of MGT Detectors*, **ACL 2024** | Are detectors robust? | 6M+ generation benchmark across models, domains, adversarial attacks | Detectors **collapse under decoding changes and adversarial edits**; published accuracy is optimistic | Detection only; no downstream-damage link | Benchmark-design reference for ORIGIN-T-Bench; source of our cross-generator protocol |
| 27 | Zhou et al., *Kill two birds with one stone: generalized and robust AI-generated text detection via dynamic perturbations*, **NAACL 2025 (Main)** | Generalisation *and* robustness together | Dynamic perturbation-based detection | Improves cross-generator generalisation and attack robustness simultaneously | Still a binary origin classifier | Feature-suite candidate (perturbation sensitivity) for `src/features` |
| 28 | Shen et al., *Rethinking AI-Generated Text Detection: A Strong Baseline and the Distribution-Shift Problem That Remains*, arXiv:2607.03680, **2026** | Is the reported progress real? | Re-evaluates a simple fine-tuned baseline under matched data and protocols | A strong simple baseline is competitive; **distribution shift remains unsolved** | Negative-result framing; no downstream-impact metric | The 2026 reality check — sets our baseline honestly, supports the conformal/drift module |

### P5–P6 — Risk modelling, calibration and time-aware sequence models

| # | Paper / Venue / Year | Problem addressed | Method | Key result | Limitation / open gap | ORIGIN-T uses it for |
|---|---|---|---|---|---|---|
| 29 | Wiegrebe et al., *Deep Learning for Survival Analysis: A Review*, **Artificial Intelligence Review 57, art. 65, 2024** *(journal)* | State of deep survival analysis | Review of DL survival architectures and evaluation practice | Taxonomy of discrete/continuous, competing-risks and time-varying models; evaluation is inconsistent across papers | ML-methods review; no LLM or data-quality application | Reference for S2's survival head — discrete-time hazard, C-index, integrated Brier score |
| 30 | Zhou et al., *Δt-Mamba3D: A Time-Aware Spatio-Temporal State-Space Model for Breast Cancer Risk Prediction*, **AAAI 2026** | Risk prediction from **irregularly spaced** observations | Continuous-time selective scan that puts the real inter-visit Δt into the state transitions; linear complexity | Beats RNN, Transformer and vanilla-Mamba baselines on longitudinal risk prediction | Medical-imaging domain; never applied to dataset/training trajectories | The "IrregMamba" pillar — the concrete architecture for our time-aware trajectory head |

**Additional reading (handout only):** Yèche et al., *Dynamic Survival Analysis
for Early Event Prediction*, arXiv:2403.12818, 2024 · Timans et al., *On
Continuous Monitoring of Risk Violations under Unknown Shift*, **UAI 2025** ·
Xu et al., *Wasserstein-regularized Conformal Prediction under General
Distribution Shift*, arXiv:2501.13430, 2025 · Hultberg et al., *Anytime-Valid
Conformal Risk Control*, arXiv:2602.04364, 2026 · Wang et al., *Data Shapley in
One Training Run*, **ICLR 2025 (Outstanding Paper Runner-Up)** · Loizou &
Tsoumakos, *Chunked Data Shapley*, **CIKM 2025** · Li et al., *CuraWeb: Joint
Optimization of Quality, Redundancy and Diversity*, arXiv:2607.22662, 2026 ·
He et al., *Learning from the Best, Differently: A Diversity-Driven Rethinking
on Data Selection*, arXiv:2510.18909, 2025 · Marion et al., *When Less is More:
Data Pruning for Pretraining LLMs at Scale*, arXiv:2309.04564, 2023 ·
Villalobos et al., *Will we run out of data?*, arXiv:2211.04325, 2022 *(the
motivation statistic)* · Soydan et al., *TIDES: Implicit Time-Awareness in
Selective State Space Models*, arXiv:2605.09742, 2026 · Cai et al., *MambaTS*,
**Pattern Recognition, 2026** *(journal)* · Jung et al., *Unveiling Key Aspects
of Fine-Tuning in Sentence Embeddings: A Representation Rank Analysis*,
arXiv:2405.11297, 2024 · Ojewale et al., *Audit Trails for Accountability in
LLMs*, arXiv:2601.20727, 2026.

---

## 4. Foundational method papers (cited as tools, not as "related work")

| Tool | Paper | Where ORIGIN-T uses it |
|---|---|---|
| LoRA / PEFT | Hu et al., *LoRA: Low-Rank Adaptation of LLMs*, arXiv:2106.09685, 2021 | The 60-run fine-tuning grid (4-bit QLoRA) |
| Text-distribution metric | Pillutla et al., *MAUVE*, **NeurIPS 2021 (Oral)** | Human-vs-synthetic divergence feature |
| Embedding geometry | Ethayarajh, *How Contextual are Contextualized Word Representations?*, **EMNLP 2019** | Anisotropy signal (white-box, gated) |
| Matrix diversity | Roy & Vetterli, *The Effective Rank*, **EUSIPCO 2007** | Effective-rank collapse signal |
| Discrete-time survival | Gensheimer & Narasimhan, **PeerJ 7:e6257, 2019** | Hand-rolled hazard model (from-scratch-before-library rule) |
| Conformal prediction | Angelopoulos & Bates, arXiv:2107.07511, 2021 | Split-conformal false-alarm guarantee |
| Conformal under drift | Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift*, arXiv:2106.00170, 2021 | ACI fallback when trajectories break exchangeability |
| Conformal risk | Angelopoulos et al., *Conformal Risk Control*, arXiv:2208.02814, 2022 | Risk-budget formulation of the alert threshold |
| Calibration | Guo et al., *On Calibration of Modern Neural Networks*, **ICML 2017** | ECE / reliability diagrams — the same metric Du et al. (#13) found predictive |
| Attribution | Lundberg & Lee, *SHAP*, arXiv:1705.07874, 2017 | Shapley attribution over risk features |

---

## 5. Gap matrix — the slide that earns the marks

Capability coverage across the strongest prior systems.
**✓** = does it · **~** = partial · **✗** = does not.

| Capability | Shumailov'24 (#1) | Gerstgrasser'24 (#5) | Kang'25 (#12) | Du'25 (#13) | Drayson'25 (#25) | ConStat (#20) | RAID (#26) | **ORIGIN-T** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Reproduces collapse empirically | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| Controlled contamination **gradient** (0–100 %, Gen-1→3) | ~ | ~ | ~ | ✓ | ~ | ✗ | ~ | ✓ |
| Benchmark-leakage axis in the same corpus | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ |
| Dataset-level **feature suite** (lexical + entropy + semantic + leakage) | ✗ | ✗ | ✗ | ✗ | ~ | ✗ | ~ | ✓ |
| Predicts **downstream degradation** (not just origin) | ✗ | ✗ | ~ | ~ | ✗ | ✗ | ✗ | ✓ |
| **Calibrated** risk output (ECE / reliability) | ✗ | ✗ | ✗ | ~ | ✗ | ✗ | ✗ | ✓ |
| **Early warning with lead-time** before the run finishes | ✗ | ✗ | ✗ | ~ | ✗ | ✗ | ✗ | ✓ |
| Distribution-free guarantee (conformal) | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✗ | ✓ |
| White-box representational signal | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Mitigation evaluated on the same curves | ✗ | ✓ | ~ | ✗ | ✓ | ✗ | ✗ | ✓ |
| Released reproducible benchmark | ✗ | ✗ | ✗ | ✗ | ~ | ~ | ✓ | ✓ |

**The gap in one sentence:** the literature can *establish* collapse (P1),
*measure* it at enormous compute cost (P2 — >100k GPU-hours in #12), *detect
origin* (P4) and *detect leakage* (P3), and separately has mature machinery for
*calibrated time-to-event risk* (P5–P6) — but **no system takes a dataset as
input and returns a calibrated, lead-time risk score for downstream degradation
before the fine-tuning run completes.** The paper that gets closest (#13, Du et
al. 2025) finds a *single* early-warning metric (ECE) by observation; it builds
no risk model, uses no dataset-side features, and offers no false-alarm
guarantee.

---

## 6. Derived problem statement and objectives (straight onto the slides)

**Problem statement.** Teams fine-tuning LLMs cannot tell, before committing
compute, whether a candidate dataset will degrade the model. Evidence of
degradation today is retrospective (#1, #6) and obtainable only through very
large training sweeps (#12); existing detectors answer a *different* question —
"is this text machine-generated?" (#25–28) or "is this benchmark leaked?"
(#20, #23) — neither of which is calibrated against **downstream damage**.

**Objectives.**

1. **O1** Build a controlled contamination gradient (synthetic, recursive
   Gen-1→3, paraphrased, benchmark-near) at 0/25/50/75/100 % — closing the
   "no gradient corpus" gap in #1, #5, #25.
2. **O2** Extract a multi-family dataset feature suite (lexical, entropy,
   perplexity, semantic, MAUVE, leakage) — beyond the binary features of #25.
3. **O3** Measure collapse curves with a LoRA fine-tuning grid on small models
   at roughly four orders of magnitude less compute than #12.
4. **O4** Train a **calibrated** risk classifier predicting degradation, with
   ECE / reliability reporting — generalising #13's single-metric observation.
5. **O5** Add a discrete-time survival head (#29) with a time-aware trajectory
   encoder (#30) and report a **lead-time > 0** at a fixed false-alarm rate.
6. **O6** Wrap the alert in split-conformal calibration for a distribution-free
   false-alarm guarantee under trajectory drift.
7. **O7** Release **ORIGIN-T-Bench** — corpus variants, features and curves —
   answering the reproducibility complaint raised in #21 and #26.

---

## 7. Methodology proposed (already partly implemented — show the repo)

| Stage | Literature basis | Repo status (16 Sep 2026) |
|---|---|---|
| Canonical schema | #19 taxonomy, #18 4-tier tiers | **Done** — `src/ingestion/schema.py` (`Sample`, 9 validated fields) |
| Corpus normalisation | WikiText-2 / TinyStories baselines | **Done** — `src/ingestion/normalize_corpora.py` + `configs/corpora.yaml` |
| Hidden holdout | #22 (dynamic / held-out evaluation) | **Done** — `src/ingestion/holdout.py`, stable blake2b hash split |
| Leakage gate | #23, #24 | **Done** — `src/ingestion/check_leakage.py` (id + normalised-text overlap; exit code 1 fails CI) |
| Benchmark-near generation | #23 (rephrased samples) | **Done** — `src/generation/benchmark_near.py`, difflib-*measured* similarity |
| Ratio mixing | #4, #6, #12, #16 | **Done** — `src/generation/mixer.py` (0–100 %, dataset size held fixed) |
| Synthetic / recursive generation | #1, #2, #6 | *Next* — `src/generation` (DistilGPT-2, Qwen2.5-0.5B) |
| Feature suite | #25, #27, MAUVE | *Sprint 4–5* — `src/features` |
| Fine-tuning grid | LoRA, #12, #16 | *Sprint 6–7* — `src/finetune` |
| Risk model + calibration | #13, Guo'17 | *Sprint 8* — `src/risk` |
| Survival + time-aware head | #29, #30 | *Sprint 9, 11* — `src/risk` |
| Conformal | Angelopoulos; Gibbs & Candès; Timans'25 | *Sprint 10* — `src/risk` |
| Mitigation | #5, #14, #15, #25 | *Sprint 12* — `src/mitigation` |

Point at this table in the viva: **six of thirteen stages are already code on
`main`**, with unit tests in `tests/` and green CI. That is what separates this
from a slide-deck project.

---

## 8. Slide plan mapped to the circular

| Circular requirement | Slide | Content source |
|---|---|---|
| Title slide | 1 | Project title, 3 members + USN, guide name & designation |
| Introduction (max 2) | 2–3 | §2 taxonomy diagram; Villalobos (data exhaustion) + #1 Nature result |
| Problem Statement | 4 | §6 problem statement |
| Objectives | 5 | §6 O1–O7 |
| **Literature Survey (≥20 papers)** | 6–11 | §3 rows 1–24 — four rows per slide; columns: Paper / Venue / Year · Method · Key finding · Limitation |
| Gap analysis | 12 | §5 gap matrix + the one-sentence gap |
| Methodology proposed | 13–14 | §7 table + the architecture block diagram from `README.md` |
| Project Plan (Gantt) | 15 | `docs/PROJECT_REPORT.md` §4 — 16 sprints S0–S15 with GATE 0–3 markers |

Formatting per the circular: **white background, black font**, supplied
template, 10 min + 2 min Q&A, handout verified and signed by the guide before
the session, formal wear, attendance compulsory. Also prepare the **review
paper** (5 marks) — §3 + §5 + §6 of this file is already ~80 % of that draft.

---

## 9. Three questions the panel will ask — and the answers

1. **"Twenty papers say collapse exists. What is left to do?"**
   → Gap matrix (§5). Existence ≠ prediction. Nobody outputs a calibrated
   pre-training risk score with a lead-time and a false-alarm guarantee.
2. **"Du et al. already found ECE as an early-warning signal. Isn't your work done?"**
   → #13 is our **baseline, not our contribution**. It observes one metric
   post-hoc from model outputs. ORIGIN-T predicts from **dataset-side** features
   before the run, over a controlled contamination gradient, with conformal
   false-alarm control — and we will report whether we actually beat ECE-alone.
3. **"Can you reproduce collapse on a 6 GB GPU?"**
   → That is **GATE 0**, pre-registered in `docs/PROJECT_REPORT.md`. If it
   fails we pivot to static contamination forensics — the benchmark and the
   detection contribution survive either way.

---

## 10. References (IEEE style)

[1] I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, and Y. Gal, "AI models collapse when trained on recursively generated data," *Nature*, vol. 631, pp. 755–759, 2024, doi: 10.1038/s41586-024-07566-y.

[2] S. Alemohammad, J. Casco-Rodriguez, L. Luzi, A. I. Humayun *et al.*, "Self-consuming generative models go MAD," arXiv:2307.01850, 2023.

[3] E. Dohmatob, Y. Feng, P. Yang, F. Charton, and J. Kempe, "A tale of tails: Model collapse as a change of scaling laws," in *Proc. ICML*, 2024.

[4] E. Dohmatob, Y. Feng, A. Subramonian, and J. Kempe, "Strong model collapse," arXiv:2410.04840, 2024.

[5] M. Gerstgrasser, R. Schaeffer, A. Dey, R. Rafailov *et al.*, "Is model collapse inevitable? Breaking the curse of recursion by accumulating real and synthetic data," arXiv:2404.01413, 2024.

[6] M. E. A. Seddik, S.-W. Chen, S. Hayou, P. Youssef *et al.*, "How bad is training on synthetic data? A statistical analysis of language model collapse," arXiv:2404.05090, 2024.

[7] L. Shi, M. Wu, H. Zhang, Z. Zhang *et al.*, "A closer look at model collapse: From a generalization-to-memorization perspective," in *Proc. NeurIPS*, 2025 (Spotlight).

[8] S. Xu, H. He, and G. Cheng, "A probabilistic perspective on model collapse," arXiv:2505.13947, 2025.

[9] D. Barzilai and O. Shamir, "When models don't collapse: On the consistency of iterative MLE," arXiv:2505.19046, 2025.

[10] T. Wang, A. Horiguchi, L. Pang, and C. E. Priebe, "LLM web dynamics: Tracing model collapse in a network of LLMs," arXiv:2506.15690, 2025.

[11] X. Xie and B. Hu, "Reviewing model collapse and countermeasures," in *Proc. IEEE Int. Conf. Advances in Artificial Intelligence and Machine Learning (AAIML)*, 2026.

[12] F. Kang, N. Ardalani, M. Kuchnik, Y. Emad *et al.*, "Demystifying synthetic data in LLM pre-training: A systematic study of scaling laws, benefits, and pitfalls," in *Proc. EMNLP*, 2025.

[13] Y. Du, G. Wu, G. Tang, and W. Wang, "Characterizing model behavior under synthetic data training: An empirical study across scales and mixing ratios," arXiv:2510.05133, 2025.

[14] Y. Feng, E. Dohmatob, P. Yang, F. Charton, and J. Kempe, "Beyond model collapse: Scaling up with synthesized data requires verification," arXiv:2406.07515, 2024.

[15] B. Yi, Q. Liu, Y. Cheng, and H. Xu, "Escaping model collapse via synthetic data verification: Near-term improvements and long-term convergence," arXiv:2510.16657, 2025.

[16] J. Kazdan, R. Schaeffer, A. Dey, M. Gerstgrasser *et al.*, "Collapse or thrive? Perils and promises of synthetic data in a self-generating world," in *NeurIPS Workshops (M3L, ATTRIB)*, 2024.

[17] D. Wright, S. Masud, J. Moore, S. Yadav *et al.*, "What and whose knowledge? Measuring epistemic diversity in large language models," in *Proc. EMNLP*, 2026.

[18] E. Nourbakhsh, M. S. Sirjani, A. Mousavi, K. Nguyen, J. Quarles, M. Xie, and R. Slavin, "Are LLM benchmarks already contaminated? A systematic review of contamination detection methods," in *Proc. 5th Workshop on Generation, Evaluation and Metrics (GEM)*, 2026.

[19] C. Xu, S. Guan, D. Greene, and M.-T. Kechadi, "Benchmark data contamination of large language models: A survey," arXiv:2406.04244, 2024.

[20] J. Dekoninck, M. N. Müller, and M. Vechev, "ConStat: Performance-based contamination detection in large language models," arXiv:2405.16281, 2024.

[21] Y. Fu, O. Uzuner, M. Yetisgen, and F. Xia, "Does data contamination detection work (well) for LLMs? A survey and evaluation on detection assumptions," in *Findings of NAACL*, 2025.

[22] S. Chen, Y. Chen, Z. Li, Y. Jiang *et al.*, "Recent advances in large language model benchmarks against data contamination: From static to dynamic evaluation," arXiv:2502.17521, 2025.

[23] S. Yang, W.-L. Chiang, L. Zheng, J. E. Gonzalez *et al.*, "Rethinking benchmark and contamination for language models with rephrased samples," arXiv:2311.04850, 2023.

[24] Y. Oren, N. Meister, N. Chatterji, F. Ladhak *et al.*, "Proving test set contamination in black box language models," arXiv:2310.17623, 2023.

[25] G. Drayson, E. Yilmaz, and V. Lampos, "Machine-generated text detection prevents language model collapse," in *Proc. EMNLP*, 2025.

[26] L. Dugan, A. Hwang, F. Trhlik, J. M. Ludan *et al.*, "RAID: A shared benchmark for robust evaluation of machine-generated text detectors," in *Proc. ACL*, 2024.

[27] Y. Zhou, J. Wen, W. Peng, and Y. Xue, "Kill two birds with one stone: Generalized and robust AI-generated text detection via dynamic perturbations," in *Proc. NAACL*, 2025.

[28] Z. Shen, M. Wang, S. Zou, and Y. Bu, "Rethinking AI-generated text detection: A strong baseline and the distribution-shift problem that remains," arXiv:2607.03680, 2026.

[29] S. Wiegrebe, P. Kopper, R. Sonabend, B. Bischl, and A. Bender, "Deep learning for survival analysis: A review," *Artificial Intelligence Review*, vol. 57, art. 65, 2024, doi: 10.1007/s10462-023-10681-3.

[30] Z. Zhou, D. Arefan, M. Zuley, and S. Wu, "Δt-Mamba3D: A time-aware spatio-temporal state-space model for breast cancer risk prediction," in *Proc. AAAI*, 2026, doi: 10.1609/aaai.v40i16.38395.

**Method references.** [31] E. J. Hu *et al.*, "LoRA: Low-rank adaptation of large language models," arXiv:2106.09685, 2021. [32] K. Pillutla *et al.*, "MAUVE: Measuring the gap between neural text and human text using divergence frontiers," in *Proc. NeurIPS*, 2021. [33] K. Ethayarajh, "How contextual are contextualized word representations?," in *Proc. EMNLP*, 2019. [34] O. Roy and M. Vetterli, "The effective rank: A measure of effective dimensionality," in *Proc. EUSIPCO*, 2007. [35] M. F. Gensheimer and B. Narasimhan, "A scalable discrete-time survival model for neural networks," *PeerJ*, vol. 7, e6257, 2019, doi: 10.7717/peerj.6257. [36] A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," arXiv:2107.07511, 2021. [37] I. Gibbs and E. Candès, "Adaptive conformal inference under distribution shift," arXiv:2106.00170, 2021. [38] A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster, "Conformal risk control," arXiv:2208.02814, 2022. [39] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017. [40] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," arXiv:1705.07874, 2017.

---

*Compiled 16 Sep 2026. Every arXiv ID, DOI and venue above was resolved against
the arXiv API, Crossref, or the ACL Anthology on that date. Re-verify preprint
venues before the camera-ready review paper.*
