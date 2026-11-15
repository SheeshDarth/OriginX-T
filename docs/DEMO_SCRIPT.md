# ORIGIN-T demo script

About 10 minutes, one story: **a dataset can quietly damage a model, so can we know before we train, can we see it while training, and can we fix it?** Every number below is in a committed report; the file is named after each one. Say the failures out loud: they are the credibility.

Before you start: API (`python -m dashboard.api`) and web app (`vite --port 5173`) running, one audit already run so the models are warm.

---

## Cold open: the landing page (`/`), 45 s
**Show:** the hero and the chart drawing in.

**Say:**
"Training a language model on a dataset is cheap to start and expensive to undo. Today the only way to learn whether a dataset will hurt your model is to train on it. And the damage can be silent: when a model learns from text that other models wrote, the rare things in the language disappear first. That is called model collapse.

ORIGIN-T asks: *can we tell, before we train, whether this dataset will do that?*"

**Components:**
- **The chart is real data.** Each line is a type of contamination mixed into human WikiText at 0, 25, 50, 75 and 100 percent. The y axis is perplexity on text the model never saw, so up is worse. At full contamination it goes from 56.6 to 67.6 for generated text and 68.4 for recursive text. The grey line is a control where we swapped in the same amount of *fresh human* text: it stays flat at about 56.4. That is how we know the damage comes from the contamination, not from changing the data. (`reports/grid/summary.json`)
- **The strip below** is the four steps of an audit. **The gate table** at the bottom lists every pre-registered test with its verdict, including the ones that failed. We return to it at the end.

---

## Act 1, the evidence: Collapse curves (`/curves`), 1 min
**Show:** click the human_control checkbox off and on; switch Metric to benchmark; click one dot.

**Say:**
"To train a predictor you first need ground truth, so we built a testbed. We take clean text, mix in a known share of five kinds of contamination, fine-tune a small model, and measure the damage. 75 grid cells on WikiText, 105 on TinyStories, three seeds each."

**Components:**
- **Filters** by contamination type and seed; **mean line with a min-max band** over seeds, so you see the noise.
- **The benchmark metric** matters for one type: `benchmark_near` (leaked test items) barely moves the held-out perplexity but is visible on the leaked items themselves, because that damage is memorisation, not degradation.
- **Clicking a dot** opens the cell: its features, its white-box signals, and the risk model's prediction made *before* training next to what actually happened.
- "Before we looked at any of it, we wrote down the pass or fail rules. Dose-response, the first of those gates, passed on both datasets."

---

## Act 2, can we know in advance? Try it (`/try`), 3 min
This is the centre of the demo. Take your time.

### Step 1, Dataset
**Show:** drag a file onto the box, or click **Half recursively generated**.

**Say:** "A dataset is a JSONL file, one record per line. The page checks it in the browser first: record count, the first records, the first bad line if there is one. Nothing is sent until you press Run."

**Components:** the **drop zone**; the **preview card** (what the audit will read, and a warning if the size is far from 1,000 records, because the risk model learned from 1,000-record datasets); the **leak toggle** (also compares every record against the WikiText benchmark items); the **three samples**, built from the experiment data so you can try it with no file.

### Step 2, Analysis
**Show:** the stages stepping through; point at the scrolling records.

**Say:** "These are the audit's real stages, and the times are what the server actually spent, not a progress bar we made up."

**Components, one line each:**
1. **Read and validate:** parse every record, load the models.
2. **Lexical statistics:** vocabulary diversity, repeated phrases, text length. Generated text repeats itself and has a narrower vocabulary.
3. **Benchmark leak check:** how close each text is to a benchmark item.
4. **Reference language model:** how predictable each text is to a fixed GPT-2. Machine text is unusually predictable. This single feature is the strongest signal we found.
5. **Semantic structure:** sentence embeddings; do the texts cluster tightly, do rare topics survive.
6. **Risk model:** combines 22 features into a probability of degradation.
7. **Flag samples:** near-copies of benchmark items, and texts that repeat themselves.
8. **Per-sample scores:** a score for every record, from the reference model and from embedding density.

"The panel on the right cycles through records from your own file, so you can see what is being read." (Each stage is held on screen for a moment so a fast run stays followable. First run after a restart is slower because the models load.)

### Step 3, Results
**Show:** the verdict, then scroll slowly.

**Say:** "The answer, in words: likely to degrade a model fine-tuned on it. 99 percent."

**Components:**
- **Verdict banner:** the call, p(degrade), and a gauge.
- **The conformal set**, stated plainly: "at 90 percent coverage, the possible outcomes are {degrade}". The model can also say *uncertain*, and we treat that as a correct answer, not a failure.
- **The caveat line**, read live from the evaluation: leave-one-type-out AUROC 0.83 on contamination types the model had not seen, and its sets covered the true outcome 81 percent of the time against a 90 percent target. "So treat it as a screen, not a guarantee." Say this yourself before anyone asks. (`reports/risk/evaluation.json`)
- **What drives the score:** each feature's push towards degrade or safe, in plain names. For the sample, the spread of the reference-model loss and repeated 3-grams dominate.
- **How your samples score:** a histogram. Human text sits on the left; a mixed dataset shows a second hump on the right. That second hump is the contamination, visible with no labels.
- **Flagged samples:** the individual records to remove first, with the reason, filterable, exportable as CSV or JSON.

---

## Act 3, the honest scoreboard: Evidence (`/overview`), 1 min
**Show:** the gate board.

**Say:**
"Every test here had a pass-or-fail rule written *before* its result existed. Here is the scoreboard, including the losses."

**Components:** one row per **gate**: status, the rule in one sentence, the numbers, and the pull request behind it.
- **Passed:** the collapse reproduces (gate 0); damage grows with dose on both datasets (gate 2); the feature suite and risk model run on both datasets (gate 1); a small model's damage predicts a bigger one's (X2).
- **Failed:** early warning on the contamination-ratio axis (gate 3, twice); the live alarm (X3).

"A negative result with a pre-registered rule is a finding. We report it as written."

---

## Act 4, can we see it coming while training? Live monitor (`/monitor`), 1.5 min
**Show:** pick a contaminated run and replay it; point at the band and the alarm.

**Say:** "Next question: during training, can the model's own internals warn us? We log six signals every 10 steps and compare them to a band learned from healthy runs. The alarm is calibrated so a healthy run triggers it at most 5 percent of the time."

**Components:**
- **Run list**, grouped healthy and contaminated.
- **Signal charts** with the healthy band; the **loss chart**; the **alarm event** with the step it fired. The stream is live (server-sent events) while a run trains, and replays the logs otherwise.
- **The finding:** the loss never rises inside a run. Contaminated runs just improve less than healthy ones, so "alarm when loss goes up" would never fire. One signal, anisotropy, separates the arms by step 20 of 125.
- **The verdict, honestly:** the alarm fired before the damage on 8 of 18 collapsing runs, 44 percent against a 50 percent bar, with 1 of 16 healthy runs false-alarming (6 percent, within the bar). Every miss was a run already damaged at the first logged step. We did not move the bar. (`reports/x3/gate_x3.json`)

---

## Act 5, does it scale? Canary (`/canary`), 45 s
**Show:** the scatter.

**Say:** "A skeptic says: 82 million parameters is a toy. So we trained a bigger model, Pythia-410M, on exactly the same data, and asked whether the small model's damage predicts the big one's."

**Components:** one **dot per matched run** (type, ratio, seed); x is the small model's damage, y the big model's; colour is the contamination type, shape is the ratio. **ρ = 0.78**, 95 percent interval [0.51, 0.91] across 24 runs, which cleared the bar of 0.7. "Part of that is simply more contamination hurting both models more; the page says so." (`reports/x2/gate_x2.json`)

---

## Act 6, can we fix it? Repair (`/repair`) and Bench (`/bench`), 1 min
**Show:** the repair table, then the bench.

**Say:** "If a dataset is bad, can we repair it? We replace the 500 most harmful records, retrain, and measure how much of the lost quality comes back."

**Components:**
- **Repair table:** strategies side by side (harm-guided, risk-guided, random, perplexity filter, oracle) with recovery, where 100 percent means back to clean.
- **The finding:** harm-guided repair beats random replacement by 25 to 55 points, so the gate passes. But cleaning by *origin* (removing the records that look machine-written) does just as well. Gradient attribution is a fine tool and not a better one. We report that as it is.
- **Bench:** the datasets themselves, browsable by type and ratio. We are packaging them for release so others can test their own detectors against ground truth.

---

## Close, 30 s
"What we built: a testbed where the answer is known, and a screen that tells you, before training, how likely a dataset is to hurt your model, with the records to blame. What we can claim: contamination causes damage that grows with dose, a small model predicts a bigger one, and one reference-model feature carries most of the signal. What we cannot: the risk sets under-cover, the early warning does not beat its bar, and everything is on small models with one training recipe. Those limits are written into the paper, next to the results."

---

## Questions to expect
- **"Isn't predicting fine-tuning outcomes already published?"** Yes (TuneAhead, ICML 2026). Ours is narrower: damage *caused by contamination*, with conformal prediction sets, against ground truth. A pre-training contamination score also exists (SynthSentry); it does not tie the score to damage. (`docs/LITERATURE_REFRESH.md`)
- **"Why did the gates fail?"** GATE 3: on WikiText nine of twelve contaminated runs already collapse at the first contamination level, so there is no earlier stage to warn from; the confirmatory run on TinyStories, with finer levels, still found no signal that beats the baseline (AUROC 0.08 to 0.56). X3: the damage is visible at the first logged step on the heaviest runs. Both are properties of this setup, reported, not tuned away.
- **"Is the risk model better than just using GPT-2 loss?"** Pooled, yes; within one corpus, no, they are within 0.03 AUROC. We do not claim the multi-feature model beats the simple detector.
- **"Does it work on my model?"** Not shown. Validated on DistilGPT-2 + LoRA and one 410M target.
- **"What does it miss?"** `benchmark_near`: the risk model gives it about 0 probability, which is why the leak flag exists separately.
