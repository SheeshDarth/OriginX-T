# ORIGIN-T — Technical Requirements Document (TRD)

## 1. System architecture

```
Raw Dataset -> Normalizer -> Contamination Generator -> ORIGIN-T Bench Store
ORIGIN-T Bench Store -> Feature Extractor -> Provenance / Risk Classifier
ORIGIN-T Bench Store -> Fine-tuning Runner -> Evaluation Harness
Feature Extractor + Evaluation Harness -> Collapse Risk Predictor
Risk Predictor -> Mitigation Engine -> Cleaned Dataset + Audit Report
Audit Report -> Dashboard / Final Report
```

## 2. Formal problem setup

For a dataset trajectory D_0, D_1, ..., D_T (stages = contamination ratio or
recursive generation), each stage has:

- **z_t** — feature vector of D_t (lexical, semantic, perplexity, leakage,
  diversity, and — gated — representational-collapse signals).
- **M_t** — a small model fine-tuned or evaluated using D_t.
- **u_t** — measured utility of M_t on a clean holdout + hidden benchmark.
- **tau** — first stage where degradation is confirmed (perplexity +10%,
  cluster coverage -15%, or benchmark accuracy -5pp vs. clean baseline —
  thresholds declared before experiments run).
- **R_t** — predicted risk that degradation occurs within the next *k*
  stages, given z_0...z_t: `R_t = P(degradation within next k stages | z_0..z_t)`.

## 3. Tech stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| ML/DL | PyTorch, HF Transformers, Datasets, PEFT/LoRA, `accelerate`, `bitsandbytes` (4-bit) |
| Classical ML | scikit-learn, XGBoost |
| Embeddings | `sentence-transformers`, FAISS |
| Evaluation | `evaluate`, `mauve-text`, custom diversity metrics |
| Survival (gated) | `lifelines`, `pycox` - hand-rolled Kaplan-Meier / discrete-time hazard first |
| Conformal (gated) | `MAPIE` - hand-rolled split-conformal first |
| Attribution (gated) | `shap` |
| Interpretability (gated) | `TransformerLens` or `nnsight` |
| Experiment tracking | MLflow or W&B (offline mode) |
| Backend / dashboard | FastAPI, Streamlit (Sprint 14 only) |
| Storage | SQLite (metadata), Parquet/JSONL (datasets) |

## 4. Model & dataset choices

| Component | Decision |
|-----------|----------|
| Base datasets | WikiText-2, TinyStories, small permissive code/QA set, team hidden holdout |
| Generation models | DistilGPT-2, Qwen2.5-0.5B/1.5B |
| Fine-tuning model (core grid) | DistilGPT-2 (QLoRA 4-bit) |
| Fine-tuning model (gated robustness) | Pythia-410M / Qwen2.5-0.5B |
| Detectors | Logistic regression baseline -> random forest/XGBoost -> MLP -> contrastive embedding classifier |
| Hardware | ASUS G16, RTX 4050 6GB local; free Colab/Kaggle for the fine-tune grid |

## 5. Core algorithms

**A. Synthetic contamination score** — token-level (entropy, type-token
ratio, repetition, n-gram diversity, burstiness), semantic (embedding
density, nearest-neighbor distance, cluster entropy, tail-cluster
retention), and model-based (reference-LM loss, disagreement, rank
instability) features feed a classifier producing P(human), P(synthetic),
P(recursive), P(benchmark-near).

**B. Collapse risk model** (committed core -> gated depth)
1. *Committed:* calibrated static classifier (logistic/XGBoost) + reliability
   diagram / ECE.
2. *Gated:* discrete-time survival head — hazard `lambda_t(k)`, hand-rolled
   Kaplan-Meier and discrete-time likelihood before using `lifelines`/`pycox`;
   logit-bias init + horizon-truncated loss for class imbalance.
3. *Gated:* competing-risks extension — cause-specific hazards for diversity
   collapse vs. factual/holdout degradation vs. benchmark inflation, with
   right-censoring for datasets that never degrade.
4. *Gated:* split-conformal wrapper (hand-rolled, then `MAPIE`) for a
   distribution-free false-alarm-rate guarantee; adaptive/online conformal
   (ACI) if the trajectory violates exchangeability under drift.

**C. Mitigation engine**
- *Committed:* remove/downweight high-risk clusters; preserve rare semantic
  clusters; re-evaluate risk after cleaning.
- *Gated:* **safe-mix optimizer** — `max U(w) s.t. risk(w) <= budget` over
  mixing weights w = (human, synthetic, recursive), producing a
  utility-vs-risk Pareto front; active human-anchor selection via
  submodular/coreset selection.

## 6. Depth-upgrade modules (gated; reuse existing experiment outputs - no new training campaigns)

| Module | What it adds |
|--------|--------------|
| White-box representational signals | Effective rank / participation ratio, weight stable rank, embedding anisotropy, neuron-death rate — computed from checkpoints already produced in the fine-tuning grid |
| Info-theoretic / behavioral-leakage signals | MAUVE, embedding-space Frechet distance, KL/JS human-vs-synthetic; ConStat-style performance-gap leakage detection |
| Causal attribution | Ablate a contamination type -> reuse the matched fine-tune -> counterfactual degradation; Shapley attribution on the risk model |
| Collapse scaling law | Fit degradation ~ f(synthetic ratio, recursion depth, size); report R-squared + held-out extrapolation error |

## 7. Data schema

| Field | Type | Description |
|-------|------|--------------|
| sample_id | string | Unique identifier |
| prompt | text | Input / source context |
| response | text | Target output |
| label | string/int | Optional supervised label |
| source | string | human / synthetic / recursive / paraphrased / benchmark_near / duplicate |
| generation | int | 0 = human, 1 = first-gen synthetic, 2+ = recursive depth |
| contamination_ratio | float | Dataset-level mixture ratio for this variant |
| benchmark_near_score | float | Similarity to a known benchmark item |
| split | string | train / validation / test / hidden_holdout |

## 8. API (Sprint 14 - dashboard release, not immediate)

| Endpoint | Purpose |
|----------|---------|
| `POST /datasets/import` | Upload dataset, infer schema |
| `POST /datasets/{id}/contaminate` | Generate synthetic/recursive/paraphrased/benchmark-near variants |
| `POST /audit/{dataset_id}` | Run feature extraction + provenance scoring |
| `POST /experiments/finetune` | Run a fine-tuning job from a fixed config |
| `GET /reports/{run_id}` | Return contamination + collapse-risk + evaluation plots |
| `POST /mitigate/{dataset_id}` | Create a cleaned/reweighted dataset variant |

## 9. Evaluation metrics

| Group | Metrics |
|-------|---------|
| Detection | AUROC / AUPRC / F1; cross-generator transfer AUROC |
| Collapse (data + model) | Perplexity, self-BLEU, distinct-n, rare-token recall, cluster coverage, MAUVE, embedding-FID, effective rank |
| Survival / early-warning (gated) | Time-dependent C-index, integrated Brier score, cause-specific AUROC, lead-stage @ fixed false-alarm rate |
| Calibration / UQ (gated) | ECE + reliability diagrams, conformal coverage vs. nominal |
| Causal / scaling law (gated) | Shapley attribution, counterfactual recovery, scaling-law fit R-squared |
| Mitigation | Utility at a fixed risk budget, recovery per human-anchor sample |
| Rigor (cross-cutting) | >= 3 seeds with bootstrap CIs, pre-registered degradation thresholds |

## 10. Compute budget

Binding limit is **Kaggle's free ~30 GPU-hr/week quota, not the 6GB GPU**.
Core grid: 1 model x 5 ratios x 4 conditions x 3 seeds ~= 60 LoRA fine-tunes
(~15-30 min each). Cache every checkpoint + activation to the HF Hub —
never re-extract. Extra models are a gated robustness check only.

## 11. Go/No-Go gates

| Gate | Condition |
|------|-----------|
| GATE 0 | Collapse reproducible on a tiny model? If not, pivot to static contamination-detection framing. |
| GATE 1 | Feature suite + static baseline runs cleanly on >= 2 datasets. |
| GATE 2 | >= 1 contamination setting causes measurable degradation. |
| GATE 3 | Risk model beats a random baseline; lead-time > 0. |

## 12. Repository structure

See [README.md](../README.md#repository-structure).
