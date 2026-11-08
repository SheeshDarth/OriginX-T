# Bibliography (draft for T3, #23)

*Vishnu, 2026-11-09. Keys are what `docs/paper/*.md` cite. **Status** says how each entry was
checked, so nobody upgrades a preprint to a venue from memory:*
- ***survey***: resolved against the arXiv API, Crossref or the ACL Anthology on 2026-09-16
  (`docs/LITERATURE_SURVEY.md`; the survey's own honesty note applies).
- ***abstract***: the arXiv abstract page was fetched on 2026-11-09, and title, first author, date
  and venue are as it states.
- ***unverified***: taken from `docs/STRETCH_GOALS.md` without a check. Verify before submission.

Authors are the first author and "et al." until the final formatting pass. A `preprint` has no
confirmed venue: say "preprint" in the paper.

## Model collapse
| Key | Reference | Status |
|---|---|---|
| shumailov24 | Shumailov et al. AI models collapse when trained on recursively generated data. *Nature* 631:755-759, 2024 | survey |
| alemohammad23 | Alemohammad et al. Self-Consuming Generative Models Go MAD. arXiv:2307.01850, 2023 (preprint) | survey |
| dohmatob24strong | Dohmatob et al. Strong Model Collapse. arXiv:2410.04840, 2024 (preprint) | survey |
| gerstgrasser24 | Gerstgrasser et al. Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data. arXiv:2404.01413, 2024 (preprint) | survey |
| seddik24 | Seddik et al. How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse. arXiv:2404.05090, 2024 (preprint) | survey |
| shi25 | Shi et al. A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective. NeurIPS 2025 (Spotlight) | survey |
| barzilai25 | Barzilai and Shamir. When Models Don't Collapse: On the Consistency of Iterative MLE. arXiv:2505.19046, 2025 (preprint) | survey |
| kazdan24 | Kazdan et al. Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World. NeurIPS 2024 Workshops (M3L, ATTRIB) | survey |
| wang26sir | Wang. Epidemiology of Model Collapse: Modeling Synthetic Data Contamination via Bilayer SIR Dynamics. arXiv:2606.05168, submitted 2026-04-14 (preprint) | abstract |

## Synthetic-data mixing and mitigation
| Key | Reference | Status |
|---|---|---|
| kang25 | Kang et al. Demystifying Synthetic Data in LLM Pre-training. EMNLP 2025 (Main) | survey |
| du25 | Du et al. Characterizing Model Behavior Under Synthetic Data Training. arXiv:2510.05133, 2025 (preprint) | survey |
| feng24 | Feng et al. Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification. arXiv:2406.07515, 2024 (preprint) | survey |
| yi25 | Yi et al. Escaping Model Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence. arXiv:2510.16657, 2025. `STRETCH_GOALS.md` lists an ICLR 2026 version | survey (arXiv); ICLR version unverified |
| drayson25 | Drayson, Yilmaz and Lampos. Machine-generated text detection prevents language model collapse. EMNLP 2025 (Main), arXiv:2502.15654 | survey |

## Benchmark contamination
| Key | Reference | Status |
|---|---|---|
| nourbakhsh26 | Nourbakhsh et al. Are LLM Benchmarks Already Contaminated? A Systematic Review. GEM Workshop 2026 | survey |
| xu24bdc | Xu et al. Benchmark Data Contamination of Large Language Models: A Survey. arXiv:2406.04244, 2024 (preprint) | survey |
| dekoninck24 | Dekoninck et al. ConStat: Performance-Based Contamination Detection in Large Language Models. arXiv:2405.16281, 2024 (preprint) | survey |
| fu25 | Fu et al. Does Data Contamination Detection Work (Well) for LLMs? NAACL 2025 (Findings) | survey |
| yang23 | Yang et al. Rethinking Benchmark and Contamination for Language Models with Rephrased Samples. arXiv:2311.04850, 2023 (preprint) | survey |
| oren23 | Oren et al. Proving Test Set Contamination in Black Box Language Models. arXiv:2310.17623, 2023 (preprint) | survey |

## Provenance and synthetic-text detection
| Key | Reference | Status |
|---|---|---|
| dugan24 | Dugan et al. RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. ACL 2024 | survey |
| zhou25mgt | Zhou et al. Kill two birds with one stone: generalized and robust AI-generated text detection via dynamic perturbations. NAACL 2025 (Main) | survey |
| shen26 | Shen et al. Rethinking AI-Generated Text Detection: A Strong Baseline and the Distribution-Shift Problem That Remains. arXiv:2607.03680, 2026 (preprint) | survey |
| myakala26 | Myakala et al. SynthSentry: Detecting Synthetic Data Contamination in Language Model Training Data. arXiv:2609.12353, submitted 2026-09-11 (preprint) | abstract |

## Attribution and repair
| Key | Reference | Status |
|---|---|---|
| li24influence | Li et al. Do Influence Functions Work on Large Language Models? arXiv:2409.19998, 2024 (preprint) | abstract |
| luo26reweight | Luo et al. From Reweighting to Rewriting: Unlocking the Intervention Effects of Influential Samples in Training Data Attribution. arXiv:2609.02771, 2026-09-02 (preprint) | abstract |
| xia24less | Xia et al. LESS: Selecting Influential Data for Targeted Instruction Tuning. ICML 2024 | abstract |
| reddy26 | Reddy et al. Follow the Gradient: Tracing Safety Degradation Back to Training Data. arXiv:2610.37481, 2026-10-27 (preprint) | abstract |
| li25rept | Li et al. Where Did It Go Wrong? Attributing Undesirable LLM Behaviors via Representation Gradient Tracing. arXiv:2510.02334, 2025 (preprint) | abstract |
| unverified-A9 | "Identifying Benign Data that Breaks Safety", OpenReview x3ceCulAsa. Authors, venue and year unchecked (the page would not load) | **unverified**: find it or drop it |
| weckbecker26 | Weckbecker et al. Can Data Attribution Filter Out Subliminal Learning? Not Reliably. arXiv:2609.20027, 2026-09-17 (preprint) | abstract |

## Predicting fine-tuning outcomes and scale transfer
| Key | Reference | Status |
|---|---|---|
| luo26tuneahead | Luo et al. TuneAhead: Predicting Fine-tuning Performance Before Full Training Begins. ICML 2026 (poster), arXiv:2606.17660 | abstract |
| wang26proxy | Wang et al. Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice. ICLR 2026, arXiv:2512.24503 | abstract |
| bethune25 | Bethune et al. Scaling Laws for Forgetting during Finetuning with Pretraining Data Injection. arXiv:2502.06042, 2025 (preprint) | abstract |

## Calibration, early warning and survival models
| Key | Reference | Status |
|---|---|---|
| angelopoulos21 | Angelopoulos and Bates. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511, 2021 (preprint) | survey |
| jing26 | Jing et al. Valid Early Detection of Memorization in Diffusion Models. arXiv:2610.34617, 2026-10-25; extended version of a NeurIPS 2026 E-Values workshop poster | abstract |
| wang26rank | Wang. When Rank Rises as LLMs Degrade. arXiv:2610.09647, 2026-10-07; NeurIPS 2026 CL4FMAgents workshop | abstract |
| bhattacharyya26 | Bhattacharyya and Ramdas. Optimal distribution-free sequential change detection with conformal martingales. arXiv:2609.27179, v1 2026-09-23, v2 2026-11-04 (preprint) | abstract |
| wiegrebe24 | Wiegrebe et al. Deep Learning for Survival Analysis: A Review. *Artificial Intelligence Review* 57, art. 65, 2024 | survey |
| zhou26mamba | Zhou et al. Δt-Mamba3D: A Time-Aware Spatio-Temporal State-Space Model for Breast Cancer Risk Prediction. AAAI 2026 | survey |

## Not cited yet, kept for the background
`scaria25`: Scaria et al. Sensitivity of Small Language Models to Fine-tuning Data Contamination,
arXiv:2511.06763, 2025-11-10 (*abstract*). `xie26`: Xie and Hu. Reviewing Model Collapse and
Countermeasures, arXiv:2608.21366; the survey lists IEEE AAIML 2026, but the abstract page says
submitted 2026-06-17, so the venue needs checking. Cite them where §2 discusses dose-response on
small models and collapse reviews.

## Open items
- [ ] Resolve `unverified-A9`: find the paper or drop the citation.
- [ ] Verify the ICLR 2026 version of `yi25`, and confirm the venue of `xie26`.
- [ ] Full author lists, and the target venue's reference style, at formatting.
- [ ] Add references for LoRA, MAUVE, effective rank and anisotropy (`LITERATURE_SURVEY.md` §4)
  where `track_a.md` and `track_c.md` name those methods.
- [ ] `track_a.md` and `track_b.md` still cite by number or name; convert them when Track A's
  section is merged.
