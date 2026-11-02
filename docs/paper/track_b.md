# Track B: contaminated data, the bench, and repair by harm (draft for T3, #23)

*Siddharth. Every number cites the committed file or issue it comes from. Draft of
2026-11-02, WikiText-2 only. The X1 results (§5) are placeholders until the pilot runs.*

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
`bench/<dataset>/<type>/r<ratio>/s<seed>.jsonl` (75 WikiText cells, plus 15 duplicate cells)
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

## 5. Results
*Pending the pilot (B8). Reference points from the grid, seed 0 (`reports/grid/results.jsonl`):
hidden-holdout perplexity 56.52 clean, 59.41 for synthetic at ratio 0.5, and 59.60 for
recursive at ratio 0.5. That leaves about 3 perplexity points to recover per pilot cell.*
- [ ] Harm-score AUROC vs `true_source`, next to `risk_reflm` and `risk_knn`
  (`reports/scores/wikitext2_harm_eval.json`).
- [ ] Recovery and distinct-2 per strategy (`reports/repair/wikitext2.jsonl`), the pilot
  go/no-go and Gate X1 (`reports/repair/wikitext2_summary.json`).
- [ ] Rare-cluster preservation (H5, B10): the same repair with and without the rule never to
  empty a semantic cluster that is rare in the human reference.

## 6. Limitations
- **Scale.** DistilGPT-2 (82M) with LoRA on 1,000-sample sets. The bench makes no claim about
  large models.
- **One checkpoint.** TracIn is usually summed over several checkpoints. We score only at the
  end of a one-epoch fine-tune, and influence is known to change during training.
- **The budget k uses the true contamination count.** Every strategy gets the same k, so the
  comparison is fair, but a deployed cleaner would have to estimate it.
- **Paraphrase is weak.** The base model rarely preserves meaning, so "paraphrased" results
  say little about real paraphrase contamination.
