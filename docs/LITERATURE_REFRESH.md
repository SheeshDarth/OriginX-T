# Literature refresh, 2026-11-09 (#23)

Repeats the second search (2026-11-02, `docs/STRETCH_GOALS.md`) for work dated after it, and
checks the claims in `docs/paper/shared.md` against what turned up. Not a systematic review: it
is eight web searches on 2026-11-09 plus a fetch of each arXiv abstract cited below. Titles,
authors and dates are as the arXiv abstract pages return them.

## Result
**No paper dated after 2026-11-02 was found, and no novelty claim is broken.** The only item
touched after that day is a revision (v2, 2026-11-04) of a September paper. The searches did,
however, find five 2026 papers that the 2026-11-02 search missed. One of them narrows a claim.

## What the search found

| Paper | Date | Relevance | Effect on our claims |
|---|---|---|---|
| **SynthSentry: Detecting Synthetic Data Contamination in Language Model Training Data** (Myakala et al., arXiv 2609.12353) | 2026-09-11 | Scores an unlabeled corpus for synthetic contamination **before training**, from vocabulary diversity, rare n-gram coverage and perplexity spread across reference models, with a bootstrap false-positive threshold. Small English corpora. Its own downstream check found no accuracy loss at that scale. | **Narrows the core claim.** A pre-training synthetic-contamination *score* is published. Ours is a *damage* prediction with conformal sets against ground-truth runs. It is related work, not a rebuttal, and must be cited in §2 ("Provenance detection") and the Gap paragraph. |
| **Optimal distribution-free sequential change detection with conformal martingales** (Bhattacharyya and Ramdas, arXiv 2609.27179) | v1 2026-09-23, v2 2026-11-04 | Sequential change detection with minimax-optimal delay and a controlled false-alarm probability; says the common conformal test martingales are inefficient. | **Relevant to X3.** Our alarm uses split conformal on the maximum deviation, so a sequential e-process detector is the natural stronger baseline and the way to trade false alarms against delay. Cite as a future-work and comparison point. |
| **Can Data Attribution Filter Out Subliminal Learning? Not Reliably** (Weckbecker et al., arXiv 2609.20027) | 2026-09-17 | Gradient-based attribution (EK-FAC, others) removes part of the effect in some settings and varies widely across methods and settings. | **Consistent with our X1 pilot** (harm-guided repair does not beat origin cleaning). Cite in "Data attribution and repair". |
| **Epidemiology of Model Collapse: Modeling Synthetic Data Contamination via Bilayer SIR Dynamics** (Wang, arXiv 2606.05168) | 2026-04-14 | Models contamination spreading between corpora and models; finds detecting synthetic text is the most influential parameter and detection-based filtering is among the most effective interventions. | Supports origin cleaning as a required baseline. Cite in "Synthetic-data mixing and mitigation". |
| **Sensitivity of Small Language Models to Fine-tuning Data Contamination** (Scaria et al., arXiv 2511.06763) | 2025-11-10 | 23 small models (270M-4B), syntactic and semantic corruption of instruction data at 25-100%; larger models can learn semantic corruption more readily. | Dose-response on small models with a *different* contamination (corruption, not generated text). Cite in the dose-response discussion; its "larger is worse" finding is a caution for our X2 reading. |
| **Reviewing Model Collapse and Countermeasures** (Xie and Hu, arXiv 2608.21366) | 2026-06-17 | A review of collapse and countermeasures. | Background citation only. |

## The four claims, rechecked
- **Core ("contamination-caused degradation predicted with conformal sets on a ground-truth testbed"):** still stands. No work found combines a damage label, conformal sets and known origin labels. SynthSentry is the nearest and scores contamination, not damage.
- **X1 (harm vs origin cleaning):** no head-to-head study found. 2609.20027 supports our pilot's direction.
- **X2 (contamination damage across scale, with an interval):** no study of proxy-predicted *contamination* damage found. The proxy-curation paper we already cite (ICLR 2026, [P1]) remains the closest, and it says a proxy trained at a fixed learning rate can rank poorly, which our X2 did not test (reduced-LR proxy not run).
- **X3 (live alarm with a stated false-alarm rate):** no LM-training alarm found. 2609.27179 is the stronger sequential method to compare against.

## Limits of this refresh
- A one-week window, run through a web search engine. The absence of a 2611 paper is weak evidence, since arXiv listings are not searchable by date from here.
- Details above come from arXiv abstracts, not the full texts. Read each paper before putting a number from it in the paper.
- Repeat on Semantic Scholar and arXiv listings shortly before submission.
