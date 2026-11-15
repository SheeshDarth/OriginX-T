# ORIGIN-T demo: the short script (about 6 minutes)

**One idea:** the only way to know if a dataset will hurt your model is to train on it. We built a way to know first.

Revanth drives the laptop. Everyone else ends with a cue ("Revanth, open Repair"). Warm the server first: run one audit on the "Half recursively generated" sample before you start.

| # | Time | Who | Screen | The one thing to land |
|---|---|---|---|---|
| 1 | 0:00 | Revanth | Home `/` | Bad data quietly hurts models |
| 2 | 0:45 | Siddharth | Collapse curves `/curves` | We made an answer key |
| 3 | 1:45 | Revanth + Vishnu | Try it `/try` | Upload, run, get a verdict |
| 4 | 3:45 | Vishnu | Evidence `/overview` | We tell you what failed |
| 5 | 4:45 | Siddharth | Repair `/repair` | Bad data can be fixed |
| 6 | 5:30 | Revanth | Home `/` | Close |

---

## 1. Revanth: the problem (Home)
**Show:** the hero. Let the chart draw in.

> "Training a model on a dataset is slow and hard to undo. And today the only way to learn whether a dataset will hurt your model is to train on it.
>
> This chart is the problem. Each line is bad text mixed into normal text, from none on the left to all of it on the right. **Up means the model got worse.** The grey line is our control. It stays flat, so the damage comes from the bad text and nothing else.
>
> We built ORIGIN-T to tell you this **before** you train."

**Cue:** "Siddharth, where does this come from?"

---

## 2. Siddharth: the answer key (Collapse curves)
**Click:** nav "Collapse curves". Then click one dot.

> "To test a tool you need data where you already know the answer. So we made it. We took clean text, mixed in bad text at amounts we chose, trained a small model, and measured it.
>
> Because we know exactly what is in every dataset, we can check any tool against the truth. That is 180 experiment settings across two datasets.
>
> Every dot is one experiment. Click it and you see what the tool predicted **before** training, next to what actually happened."

**Cue:** "Revanth, show them the tool."

---

## 3. Revanth and Vishnu: try it (Try it)
**Revanth. Click:** nav "Try it", then the sample **Half recursively generated**.

> "Here is the tool. You drop in a dataset, or use a sample. This one is half normal text, half text written by a model trained on its own output. The page checks the file before anything runs."

**Click:** "Run the audit".  **Cue:** "Vishnu, what is it doing?"

**Vishnu. Point at the stages as they light up.**

> "It looks at your data three ways. How varied the words are. How predictable the text is to another language model, because machine-written text is too predictable. And how the texts cluster. A risk model turns those into one answer. The times you see are real."

**Revanth. When the result appears:**

> "The answer in plain words: **likely to damage a model.** Over 99 percent. Below, it tells you why, in normal words. This histogram has two humps: the left one is human text, the right one is the generated half, found without being told. And here are the exact records to remove first."

**Cue:** "Vishnu, how far can we trust that?"

---

## 4. Vishnu: the honest scoreboard (Evidence)
**Click:** nav "Evidence".

> "Every test here had its pass-or-fail rule written **before** we had the result.
>
> What passed: damage grows with how much bad text there is, on both datasets. And a small model predicts a bigger one, a correlation of 0.78.
>
> What failed: early warning, twice. And our live alarm caught 44 percent of failures against a bar of 50.
>
> And this tool's confidence is slightly too optimistic: 81 percent against a target of 90. We report all of it."

*Optional, 20 seconds each if there is time:* **Live monitor** ("the model's insides, watched during training") and **Canary** ("each dot is a small-model run against a bigger-model run, and they line up").

**Cue:** "Siddharth, if the data is bad, can we fix it?"

---

## 5. Siddharth: the fix (Repair)
**Click:** nav "Repair".

> "Yes. We find the 500 most harmful records, swap them for clean ones, retrain, and measure how much quality comes back. A hundred percent means fully repaired.
>
> It works: it beats random replacement by 25 to 55 points. The honest part: simply removing text that looks machine-written works about as well. We report that, because it is true."

**Cue:** "Revanth, to close."

---

## 6. Revanth: close (Home)
**Click:** the ORIGIN-T logo.

> "To sum up. We made a testbed where the answer is known. On top of it, a tool that tells you, **before you train**, how likely a dataset is to hurt your model, and which records to remove.
>
> It is a screen, not a guarantee, and we have shown you where it falls short. Thank you."

---

## If you only remember three things
1. **Up means worse; flat is the control.** (Home)
2. **We know what is in the data, so we can test against the truth.** (Siddharth)
3. **We show what failed, not only what passed.** (Vishnu)

## If something goes wrong
- The audit sits on the first stage: the models are loading. Vishnu keeps talking; it clears in about a minute (this is why you warm it first).
- A page is empty: say "that screen needs the experiment reports on this machine" and move on.
- Running long: drop the optional Live monitor and Canary lines.

## Quick answers (one line each)
- **Is this new?** Predicting fine-tuning results exists; ours predicts damage caused by bad data, with stated confidence, against known ground truth. (Vishnu)
- **Will it work on my model?** Not shown yet: we tested a small model and one larger one. (Vishnu)
- **What do I do with the result?** Remove the flagged records first. (Revanth)
- **Is the data clean and unleaked?** Yes: a hidden test set no step may train on, and leak checks on everything. (Siddharth)
