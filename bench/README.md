# ORIGIN-T-Bench v0.1 (datasheet)

Fine-tuning datasets with a **known, labelled share of contaminated text**: model-written,
recursively model-written, paraphrased, leaked benchmark items, and duplicated rows.
Every file is a training set that ORIGIN-T's grid fine-tuned DistilGPT-2 on, or could
have. Each row says where it came from, so a detector or cleaner can be scored against
ground truth rather than against a guess.

Built by `python -m src.bench.build --config configs/bench.yaml` (HANDBOOK §9.2). The
files are not in git. They are released on the Hugging Face Hub at CP3 (§9.5), and
`manifest.json` holds a SHA-256 for every file.

## Layout

```
bench/
  manifest.json                         counts, hashes, generator versions, config fingerprint, licences
  <dataset>/<type>/r<ratio>/s<seed>.jsonl
```

- **dataset:** `wikitext2`. `tinystories` is added once Track C's TinyStories grid has run.
- **type:** `synthetic`, `recursive`, `paraphrased`, `benchmark_near`, `human_control`,
  `duplicate`.
- **ratio:** the share of the set that is contaminated: 0, 0.25, 0.5, 0.75 or 1.
- **seed:** 0, 1, 2. Each seed is a different draw of human text and its own contamination pool.

Every set has 1,000 rows of 20–150 words. At ratio 0 a set is all human, and it is the
same for every type of that seed.

Each row uses the project's sample schema (`src/ingestion/schema.py`):

| field | meaning |
|---|---|
| `sample_id` | unique within a file; copies get new ids (`<id>-dup<i>`) |
| `prompt`, `response` | the text; WikiText rows have an empty prompt |
| `source` | `human`, `synthetic`, `recursive`, `paraphrased`, `benchmark_near`, `duplicate`. **The ground-truth label.** |
| `generation` | 0 for human or copied text, 1 for text the pretrained model wrote, 2 for text a model fine-tuned on generation 1 wrote |
| `benchmark_near_score` | measured `difflib` similarity to the original benchmark item (1.0 = verbatim) |
| `contamination_ratio`, `label` | carried over from the source rows |
| `split` | `train` for corpus rows; benchmark items keep `validation` |

`human_control` sets are all `source="human"`. Their extra rows are a third, disjoint draw
of human text: the healthy arm that should not degrade.

## How each type is made

| type | made from | by |
|---|---|---|
| `synthetic` | the second half of the seed's human draw (disjoint from the clean half) | pretrained DistilGPT-2 continuing each text (96 new tokens) |
| `recursive` | the `synthetic` pool | DistilGPT-2 fine-tuned (LoRA) on that pool, writing generation 2 |
| `paraphrased` | the second half of the seed's draw | pretrained DistilGPT-2 asked to rewrite each text (see the caveat below) |
| `benchmark_near` | 200 items from the WikiText-2 **validation** split | surface edits on 70% of copies, the other 30% verbatim (`src/generation/benchmark_near.py`, v2) |
| `human_control` | a third human draw, disjoint from the hidden holdout | no edits |
| `duplicate` | the seed's own clean half | 50% verbatim copies, 50% with the same surface edits (`src/generation/duplicate.py`) |

The model-written pools are the exact ones the grid trained on, copied from the machine
that ran it. Sampling isn't reproducible across GPUs and library versions, so they are
never regenerated (§12.3). `duplicate` isn't part of the grid. It is pure stdlib and seeded,
so every machine builds the same files.

## Leakage guarantees

The hidden holdout is carved from each corpus by text hash. Every file is checked
against **the whole holdout** by id and by normalised-text fingerprint before anything is
written, and the build refuses on any hit. The corpus kept for training is checked too.
`benchmark_near` sets *deliberately* contain validation-split items: that leak is the
contamination being measured.

## Intended use

- Training and evaluating contamination detectors, data cleaners and dataset-risk models
  against ground-truth `source` labels.
- Reproducing ORIGIN-T's collapse curves: train on a set with the grid's `train` settings
  (`configs/grid.yaml`), then score the hidden holdout.

**Not for:** training general-purpose models, or claims about large models. These are
small sets made with an 82M-parameter model.

## Known artefacts and caveats

- **The recasing story (why `benchmark_near` is v2).** The first generator rewrote about
  half of the perturbed copies in full ALL CAPS or all lowercase. That is the same content
  to a person but not to GPT-2's tokenizer: ALL CAPS changes 71–87% of a passage's
  pre-tokens. Training on those "near-duplicates" *raised* benchmark-item perplexity by
  4.7% at ratio 1, so the copies hid the memorisation they were meant to cause. v2
  removes whole-text recasing and the doubled-space padding; each edit now leaves most
  tokens unchanged (decision D3, #13). After the fix, benchmark perplexity falls 3.4% at
  ratio 1. `manifest.json` records each pool's generator version.
- **`paraphrased` is closer to prompted continuation.** Base DistilGPT-2 cannot follow a
  rewrite instruction, so these rows rarely preserve the source's meaning (HANDBOOK §14,
  pitfall 10).
- **WikiText heading lines** (`= = History = =`): the 20-word minimum drops almost all of
  them, but 8 heading-like rows in the kept corpus fall inside the window. ` @-@ ` hyphens and
  `<unk>` tokens remain, as in the source.
- **Length window.** Only 20–150-word texts are used, so long passages are under-represented.
- **TinyStories** (when added): the Hub's validation split repeats 100 training stories.
  The benchmark items are checked against the holdout before use (B3, #58).

## Sources and licences

| dataset | source | licence |
|---|---|---|
| WikiText-2 (raw) | Merity et al., 2016, *Pointer Sentinel Mixture Models*; https://huggingface.co/datasets/Salesforce/wikitext | CC BY-SA 3.0 (and GFDL). Derived files are shared under **CC BY-SA 3.0** |
| TinyStories | Eldan & Li, 2023, *TinyStories*; https://huggingface.co/datasets/roneneldan/TinyStories | CDLA-Sharing-1.0 |

The generated text comes from DistilGPT-2 (Apache-2.0). Check the licences again before
the public release (§9.5).

## How to cite

Cite the ORIGIN-T project (BNMIT capstone, 2026–27) and the source corpora above. A BibTeX
entry will be added with the public release.
