"""Dataset risk classifier (HANDBOOK §8.3): p(degrade) for a training set from its features.

    python -m src.risk.classifier --datasets wikitext2 [tinystories]

**Label (§13 D4, default until CP0 fixes it).** A cell is degraded (1) if its
hidden-holdout perplexity exceeds its own seed's ratio-0 run by more than the
spread (max - min) of the ratio-0 runs across seeds: GATE 2's noise floor, no new
threshold. Ratio-0 cells are 0 by definition.

**Model.** Logistic regression, hand-rolled: features standardised with the
training fold's mean and std, then full-batch gradient descent on the mean log
loss plus an L2 penalty (bias not penalised). Missing values (``None``) become the
training mean (0 after standardising). Standard library only.

**Evaluation (the headline, §8.3).** Damage rises with ratio by construction, so a
model that only detects "how contaminated" looks good in-distribution for free.
Every prediction here is out-of-group: leave-one-type-out (train on the other
types, predict the held-out one) and, when two datasets exist, leave-one-dataset-out.
Ratio-0 sets are shared by every type of a seed, so a held-out type's ratio-0 rows
duplicate training rows; they are left out of its test fold.

Writes ``reports/risk/predictions.jsonl`` (§7.4, ``split: "test"``) and
``reports/risk/evaluation.json`` (n, AUROC/AUPRC with bootstrap CIs, ECE,
reliability bins, split-conformal coverage with a bootstrap CI). Each fold holds
out a third of its training rows to calibrate the conformal sets (``conformal.py``).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..evaluation.scoring import auprc, auroc, bootstrap_ci, ece, reliability_bins, set_coverage
from . import conformal

MODEL = "logreg@1"
ALPHA = 0.1
KEYS = ("dataset", "cell", "type", "ratio", "seed")


def degraded_labels(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """cell -> 0/1 under the D4 default, from one grid's results rows."""
    base: dict[tuple[str, int], float] = {}
    for r in results:
        if r["ratio"] == 0:
            base[(r["type"], r["seed"])] = r["holdout_ppl"]
    out = {}
    for r in results:
        if r["ratio"] == 0:
            out[r["cell"]] = 0
            continue
        seeds = [v for (t, _), v in base.items() if t == r["type"]]
        if (r["type"], r["seed"]) not in base or len(seeds) < 2:
            continue  # no baseline or no spread: unlabelled, not guessed
        out[r["cell"]] = int(r["holdout_ppl"] - base[(r["type"], r["seed"])] > max(seeds) - min(seeds))
    return out


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


class LogReg:
    def __init__(self, lr: float = 0.1, steps: int = 2000, l2: float = 0.01) -> None:
        self.lr, self.steps, self.l2 = lr, steps, l2

    def _x(self, row: Mapping[str, Any]) -> list[float]:
        return [0.0 if row.get(c) is None else (row[c] - m) / s for c, m, s in zip(self.cols, self.mean, self.std)]

    def fit(self, rows: Sequence[Mapping[str, Any]], y: Sequence[int], cols: Sequence[str]) -> "LogReg":
        self.cols = list(cols)
        self.mean, self.std = [], []
        for c in self.cols:
            vs = [r[c] for r in rows if r.get(c) is not None]
            m = sum(vs) / len(vs) if vs else 0.0
            s = math.sqrt(sum((v - m) ** 2 for v in vs) / len(vs)) if vs else 0.0
            self.mean.append(m)
            self.std.append(s or 1.0)  # a constant column standardises to 0 and carries no weight
        X = [self._x(r) for r in rows]
        n, d = len(X), len(self.cols)
        self.w, self.b = [0.0] * d, 0.0
        for _ in range(self.steps):
            err = [_sigmoid(sum(wj * xj for wj, xj in zip(self.w, x)) + self.b) - yi for x, yi in zip(X, y)]
            # gradient of mean log loss + (l2/2)|w|^2
            for j in range(d):
                self.w[j] -= self.lr * (sum(e * x[j] for e, x in zip(err, X)) / n + self.l2 * self.w[j])
            self.b -= self.lr * sum(err) / n
        return self

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> list[float]:
        return [_sigmoid(sum(wj * xj for wj, xj in zip(self.w, self._x(r))) + self.b) for r in rows]


def feature_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every f_* column with at least one value."""
    return sorted({k for r in rows for k, v in r.items() if k.startswith("f_") and v is not None})


def leave_one_out(
    rows: Sequence[Mapping[str, Any]], group: str, *, alpha: float = ALPHA, cal_fraction: float = 1 / 3, **fit: Any
) -> list[dict[str, Any]]:
    """Out-of-group predictions: for each value of ``group``, train on the rest and predict it.

    The rest is shuffled (seeded per fold) and split: ``cal_fraction`` calibrates the
    split-conformal threshold (``conformal.py``), the remainder fits the model, so
    the calibration scores come from data the model never saw. ``rows`` are feature
    rows carrying a ``label``. A fold whose fitting part lacks a class is skipped.
    """
    cols = feature_columns(rows)
    preds = []
    for g in sorted({r[group] for r in rows}):
        rest = [r for r in rows if r[group] != g]
        random.Random(f"cal-{group}-{g}").shuffle(rest)
        n_cal = round(len(rest) * cal_fraction)
        cal, train = rest[:n_cal], rest[n_cal:]
        test = [r for r in rows if r[group] == g and not (group == "type" and r["ratio"] == 0)]
        if not test or len({r["label"] for r in train}) < 2:
            continue
        model = LogReg(**fit).fit(train, [r["label"] for r in train], cols)
        q = conformal.quantile([conformal.nonconformity(p, r["label"]) for r, p in zip(cal, model.predict(cal))], alpha)
        for r, p in zip(test, model.predict(test)):
            preds.append({**{k: r[k] for k in KEYS}, "label": r["label"], "p_degrade": p, "fold": f"{group}={g}",
                          "conformal_set": conformal.prediction_set(p, q), "alpha": alpha, "n_cal": len(cal)})
    return preds


def evaluate(preds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    y = [p["label"] for p in preds]
    ps = [p["p_degrade"] for p in preds]
    out: dict[str, Any] = {"n": len(y), "n_degraded": sum(y)}
    if 0 < sum(y) < len(y):
        for name, metric in (("auroc", auroc), ("auprc", auprc)):
            point, lo, hi = bootstrap_ci(metric, y, ps)
            out[name] = {"value": point, "ci95": [lo, hi]}
    else:
        out["note"] = "one class only: AUROC/AUPRC undefined"
    if y:
        out["ece"] = ece(y, ps)
        out["reliability"] = reliability_bins(y, ps)
        sets = [p["conformal_set"] for p in preds]
        cov, lo, hi = bootstrap_ci(lambda t, s: set_coverage(t, s)["coverage"], y, sets)
        out["conformal"] = {"alpha": preds[0]["alpha"], "target": 1 - preds[0]["alpha"],
                            "coverage": {"value": cov, "ci95": [lo, hi]},
                            "mean_set_size": set_coverage(y, sets)["mean_size"],
                            "n_cal_per_fold": sorted({p["n_cal"] for p in preds})}
    return out


def risk_rows(preds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """§7.4 rows."""
    return [{"schema": "risk/1", "dataset": p["dataset"], "cell": p["cell"], "p_degrade": p["p_degrade"],
             "label": p["label"], "conformal_set": p["conformal_set"], "alpha": p["alpha"], "model": MODEL,
             "split": "test"}
            for p in preds]


def load_rows(dataset: str, features_dir: Path, grid_results: Path) -> list[dict[str, Any]]:
    """Feature rows of one dataset with their D4 label; unlabelled cells are dropped."""
    def jsonl(p: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    labels = degraded_labels(jsonl(grid_results))
    return [{**r, "label": labels[r["cell"]]} for r in jsonl(features_dir / f"{dataset}.jsonl") if r["cell"] in labels]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    tmp.replace(path)


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    grids = {"wikitext2": "reports/grid/results.jsonl", "tinystories": "reports/grid_tinystories/results.jsonl"}
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", default=["wikitext2"], choices=list(grids))
    p.add_argument("--features-dir", default="reports/features")
    p.add_argument("--out-dir", default="reports/risk")
    args = p.parse_args(argv)
    rows = [r for d in args.datasets for r in load_rows(d, Path(args.features_dir), Path(grids[d]))]
    preds = leave_one_out(rows, "type")
    report: dict[str, Any] = {"model": MODEL, "label": "D4 default (HANDBOOK §13)", "features": feature_columns(rows),
                              "leave_one_type_out": evaluate(preds)}
    if len(args.datasets) > 1:
        lodo = leave_one_out(rows, "dataset")
        report["leave_one_dataset_out"] = evaluate(lodo)
        preds += lodo
    out = Path(args.out_dir)
    _write_jsonl(out / "predictions.jsonl", risk_rows(preds))
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    lt = report["leave_one_type_out"]
    print(f"leave-one-type-out: n={lt['n']} ({lt['n_degraded']} degraded), "
          f"AUROC {lt.get('auroc', {}).get('value', float('nan')):.3f}, ECE {lt.get('ece', float('nan')):.3f} -> {out}")


if __name__ == "__main__":
    main()
