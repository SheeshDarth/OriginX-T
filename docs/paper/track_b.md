# Track B: contaminated data, the bench, and repair by harm (draft for T3, #23)

*Siddharth. Every number cites the committed file or issue it comes from. Draft of
2026-11-10, WikiText-2 only. §5 has the X1 pilot (B8) and a full-grid replicate on independent
pools; the official B9 is still to run.*

## 1. Contamination with ground truth
A detector or cleaner can only be scored against a known answer, so every contaminated
sample is made by us and carries its origin in `source` (`src/ingestion/schema.py`). Per
seed, one draw of human WikiText-2 text is split in two disjoint halves (`grid.draw_human`).
The first half is the clean training data. The second is the *source* that contamination is
made from, so a contaminated sample never shares text with a clean one in the same mixture.
- **synthetic:** pretrained DistilGPT-2 continues each source text (generation 1).
- **recursive:** a model fine-tuned on the synthetic pool writes generation 2: the
  collapse-spike recipe, at depth 2.
- **paraphrased:** DistilGPT-2 asked to rewrite each source text. The base model cannot
  follow the instruction, so this is closer to prompted continuation (§6).
- **benchmark_near:** 200 WikiText-2 *validation* items, 30% copied verbatim and 70% with
  token-preserving surface edits (`src/generation/benchmark_near.py`, v2). Each copy's
  similarity to its original is measured with `difflib`, not assumed.
- **duplicate** (bench only): copies of the training set's own human rows, half verbatim
  and half near-exact (`src/generation/duplicate.py`). Here the damage is lost diversity,
  not leaked answers.
- **human_control:** a third, disjoint human draw. This healthy arm adds no contamination.

A cell mixes contamination into the clean half at a fixed ratio and keeps the size fixed
(1,000 samples), so the only difference between cells is the contamination.

**The recasing lesson.** The first benchmark_near generator recased about half its copies
in ALL CAPS or lowercase. To GPT-2's tokenizer, ALL CAPS changes 71–87% of a passage's
pre-tokens, so training on those copies *raised* benchmark perplexity (+4.7% at ratio 1)
instead of showing memorisation. v2 makes only token-preserving edits, and benchmark
perplexity then falls (−3.4% at ratio 1; HANDBOOK §1, D3 #13). Pools are versioned
(`grid.POOL_VERSIONS`), so a generator fix can never silently mix two generators in one curve.

## 2. Data hygiene
- **Hidden holdout carved by text hash** (`src/ingestion/holdout.py`, D1). Splitting by id
  would let duplicate lines straddle the split.
- **Leak checks by id and normalised text** (`src/ingestion/check_leakage.py`): case,
  whitespace and punctuation are folded, so a lightly edited holdout text still counts as a leak.
  Every bench file is checked against the whole holdout before anything is written (§3).
- **TinyStories (B3, #58).** The Hub's validation split repeats 100 training stories, and 5
  of them fell in our hidden holdout. Because benchmark_near copies validation items into
  training, that is a live leak path. The grid's benchmark draw (200 items, 20–150 words)
  contains none of the 5. A different draw must be leak-checked first.

## 3. ORIGIN-T-Bench v0.1
`python -m src.bench.build` writes every grid cell's training set as
`bench/<dataset>/<type>/r<ratio>/s<seed>.jsonl` (75 WikiText cells, plus 12 duplicate cells (no r1: a ratio-1 file would have nothing left to copy))
with a manifest of SHA-256 hashes, generator versions and the grid's config fingerprint.
The cells are rebuilt with the grid's own function (`grid.training_set`) from the grid's
cached pools. Sampling is not reproducible across GPUs, so the build refuses to regenerate
a missing pool rather than publish data the grid never trained on. The datasheet is
`bench/README.md`. The licence is CC BY-SA 3.0, inherited from WikiText-2.

## 4. X1: cleaning by harm, not by origin
**Question.** Does replacing the training samples that *hurt* the model recover more of the
damage than replacing the ones that *look* machine-written?

**Harm scores** (`src/mitigation/harm.py`, §7.3 `scores/1`), computed at each cell's trained
checkpoint. Both are negated cosines, so higher means more harmful:
- **harm_lora_grad** (TracIn-style, one checkpoint): −cos(∇ℓ(sample), ∇ℓ̄(target)). A step
  on a sample whose gradient opposes the held-out gradient raises held-out loss. Gradients
  are taken w.r.t. a fresh rank-8 LoRA adapter on the merged checkpoint. Its zero-initialised
  B matrices receive the full weight gradient projected through a fixed random A, about 110k
  numbers per sample. Cosine is used rather than the dot product so short texts don't dominate
  (LESS, Xia et al. 2024).
- **harm_repr_sim:** −cos between a sample's mean-pooled last hidden state and the held-out
  centroid, after removing the mean of all embeddings. This is the representation baseline
  that beat influence functions at flagging harmful fine-tuning data, so it is mandatory.

**The target is not the test set.** Recovery is measured on the grid's 200 hidden-holdout
evaluation texts. The harm target is a *disjoint* draw of 200 texts from the rest of the
hidden holdout, so selection is never tuned to the texts it is judged on.

**Repair at equal size** (`src/mitigation/repair.py`, §7.5 `repair/1`). Each strategy
replaces the top *k* samples with fresh human text and leaves the rest of the set unchanged.
*k* is the cell's number of contaminated samples. The fresh text is disjoint from the
seed's draw, its human_control pool and the holdout. Each repaired set is then retrained
with the grid's settings and seed. The strategies:
- **harm_guided**: harm_lora_grad.
- **repr_sim**: harm_repr_sim.
- **risk_guided**: Track C's reference-model score, i.e. cleaning *by origin*.
- **knn_guided**: Track C's nearest-neighbour redundancy score (`risk_knn`), added before
  B9 (D10) because it is the score that finds benchmark_near copies.
- **perplexity_filter**: the classic quality filter.
- **random**: the floor.
- **oracle**: the true labels, the upper bound.

recovery = (ppl_contaminated − ppl_repaired) / (ppl_contaminated − ppl_clean), and distinct-2
is reported alongside because aggressive selection can cost diversity.

**Origin cleaning is a strong baseline.** On the grid's contaminated cells, `risk_reflm`
ranks contaminated samples above human ones with AUROC 0.976 (recursive), 0.892
(synthetic) and 0.875 (paraphrased), but 0.381 on benchmark_near (worse than chance:
leaked items are human-written) (`reports/scores/wikitext2_eval.json`). So harm has room
to win where origin is uninformative (benchmark_near, harmful human text), and must at
least match it on the synthetic types.

**Gate X1** (pre-registered, §11). Harm-guided recovery must be ≥ 10 points above random's
at ratio 0.5 on ≥ 2 of the 3 collapse types, on every seed. The pilot (seed 0, synthetic
and recursive at ratio 0.5) decides go/no-go at CP1. If harm cannot beat random by
2026-12-15, it is reported as a negative result.

**Secondary: harm vs origin** (pre-registered 2026-11-08, D10, before B9). The question X1
was adopted for is whether harm beats *origin* cleaning, not random, so the full grid also
reports harm_guided − risk_guided on every cell with a paired bootstrap 95% CI, whatever it
shows. benchmark_near is scored by **benchmark recovery** instead, because its holdout damage
is within noise (0.21 ppl): (bench_repaired − bench_contaminated) / (bench_clean −
bench_contaminated), where 1 means the memorised items are as unfamiliar as before training.

## 5. Results
**X1 pilot** (B8; seed 0, ratio 0.5, k = 500; PR #122, `reports/repair/wikitext2.jsonl`).
Hidden-holdout perplexity is 56.52 clean, 59.41 for synthetic and 59.60 for recursive
(`reports/grid/results.jsonl`), so about 3 points to recover per cell.

| strategy | synthetic recovery | recursive recovery | distinct-2 (syn / rec) |
|---|---|---|---|
| oracle (true labels) | 1.12 | 1.11 | 0.662 / 0.662 |
| risk_guided (origin) | **1.15** | 1.06 | 0.670 / 0.666 |
| harm_guided | 1.02 | 0.97 | 0.643 / 0.622 |
| repr_sim | 0.88 | **1.09** | 0.660 / 0.661 |
| random | 0.66 | 0.58 | 0.618 / 0.563 |
| perplexity_filter | −0.02 | −0.06 | 0.568 / 0.475 |

- **Pilot: go.** harm − random = +0.36 (synthetic) and +0.39 (recursive), above the
  pre-registered 0.10 (`reports/repair/wikitext2_summary.json`). Gate X1 itself needs all three
  seeds and the paraphrased type (B9).
- **Origin cleaning ties or beats harm.** harm − origin is −0.12 and −0.09 (D10 secondary,
  2 cells, so its CI means little until B9). Both sit near the oracle; recovery above 1 means
  the repaired set beat the clean baseline, since the replacements are fresh human text.
- **The classic quality filter makes things worse.** perplexity_filter removes the *human*
  samples a reference model finds hardest, keeps the fluent machine text, and loses diversity
  (distinct-2 0.475 on recursive, against 0.662 for the oracle).
- **Detection explains the ranking** (AUROC against `true_source`, the two pilot cells pooled,
  recomputed from `reports/scores/`): risk_reflm 0.930, harm_repr_sim 0.904, harm_lora_grad
  0.902, risk_knn 0.410. The harm scores find generated text almost as well as the origin
  score, so they clean about as well. The pilot shows no sign yet that harm picks out *which*
  generated samples hurt most; B9's twelve cells per comparison can test that.

**benchmark_near pilot** (extra, not part of Gate X1; `reports/repair/wikitext2_benchmark_near_summary.json`).
Benchmark-item perplexity is 61.24 clean and 59.80 contaminated (memorisation lowers it).
Benchmark recovery: oracle 0.90, perplexity_filter 0.78, repr_sim 0.54, random 0.48,
risk_guided 0.18, harm_guided 0.06. **Harm-guided repair is worst here**, close to doing
nothing. Its AUROC for the near-copies is 0.59 (risk_reflm 0.37), while risk_knn reaches
0.96 (`reports/scores/wikitext2_harm_eval.json`), which is why knn_guided joins B9 (D10). A
near-copy of a benchmark item looks like ordinary human text to the model, so neither its
gradient nor its representation points away from the held-out target.

**B9 independent-pool replicate** (2026-11-10; `reports/repair_replicate/`). The full grid,
24 cells × 7 strategies plus 3 `+rare` arms (240 retrains), on pools *regenerated* on
Siddharth's RTX 4050 with the same config fingerprint and rules. These are not the grid's
training sets, so this is a replicate on independent data, not the official B9 (#64). At ratio 1
every row is contaminated, so k = N and every strategy replaces the whole set: all ratio-1
recoveries are ≈ 1 by construction. **Only ratio 0.5 separates the strategies.**

Mean recovery at ratio 0.5 (3 seeds each):

| strategy | synthetic | recursive | paraphrased |
|---|---|---|---|
| oracle | 1.05 | 1.04 | 1.06 |
| risk_guided (origin) | 1.01 | 1.04 | 1.07 |
| harm_guided | 1.02 | 0.93 | 1.08 |
| repr_sim | 0.89 | 0.99 | 1.03 |
| random | 0.60 | 0.61 | 0.58 |
| knn_guided | 0.45 | 0.43 | 0.52 |
| perplexity_filter | 0.07 | 0.11 | −0.08 |

- **Gate X1 passes on the replicate:** harm − random is 0.25–0.55 on every type and seed, above
  0.10 on all 9 cells, 3 of 3 types. (The official verdict is the grid machine's run.)
- **D10 secondary: harm does not beat origin cleaning.** At ratio 0.5, harm − risk_guided =
  −0.031, 95% CI [−0.073, +0.007] over 9 cells: no difference. Harm vs repr_sim: +0.039
  [−0.023, +0.103], also no difference. The summary's pooled number (−0.015, 18 cells) is diluted
  by the ratio-1 ties.
- **On benchmark_near, harm-guided repair is the worst strategy.** Benchmark recovery at ratio 0.5:
  oracle 0.94, knn_guided 0.93, perplexity_filter 0.88, repr_sim 0.70, random 0.39,
  risk_guided 0.19, harm_guided **−0.02**. The D10 secondary on benchmark recovery reads
  "origin better" (−0.11, CI [−0.23, −0.01], 6 cells). knn_guided, the near-duplicate score, is
  as good as the oracle here and the worst learned strategy on the collapse types (0.43–0.52):
  each score cleans the contamination it detects.
- **H5: rare-cluster preservation changes nothing measurable.** With the binding rule (#135),
  `+rare` minus plain at ratio 0.5 is +0.004 [−0.021, +0.028] for harm_guided, +0.010
  [−0.014, +0.033] for risk_guided and +0.014 [−0.010, +0.036] for random, with distinct-2 moving
  by under 0.001. Protecting rare clusters costs no recovery and buys no measurable diversity at
  this scale.
- **The perplexity filter is harmful again**, as in the pilot, and costs the most diversity
  (distinct-2 0.596 against 0.661 for the oracle, collapse types pooled).

**Still to run:**
- [ ] The official B9 on the grid machine (`configs/repair.yaml`, #64), to confirm the replicate
  on the grid's own training sets.

## 6. Limitations
- **Scale.** DistilGPT-2 (82M) with LoRA on 1,000-sample sets. The bench makes no claim about
  large models.
- **One checkpoint.** TracIn is usually summed over several checkpoints. We score only at the
  end of a one-epoch fine-tune, and influence is known to change during training.
- **The budget k uses the true contamination count.** Every strategy gets the same k, so the
  comparison is fair, but a deployed cleaner would have to estimate it.
- **One seed so far.** The pilot is seed 0 on two cells. The ordering of the strong strategies
  (oracle, risk_guided, harm_guided, repr_sim) changes between the two cells, so only the
  large gaps (all four vs random, random vs perplexity_filter) are worth reading before B9.
- **Paraphrase is weak.** The base model rarely preserves meaning, so "paraphrased" results
  say little about real paraphrase contamination.
