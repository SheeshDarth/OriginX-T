"""Cross-check the hand-rolled risk model against library models (HANDBOOK §8.3, A8).

    python -m src.risk.crosscheck      # -> reports/risk/crosscheck.json

- ``SkLogReg``: the same model as ``LogReg`` (same standardisation, missing-value fill and
  L2 penalty) with scikit-learn's solver instead of our gradient descent. Matching
  predictions show the hand-rolled optimiser is right. Our loss is mean log loss +
  (l2 / 2)|w|^2; sklearn minimises C * sum(log loss) + |w|^2 / 2, the same optimum at
  C = 1 / (n * l2).
- ``XGB``: gradient-boosted trees, a non-linear model on the same features and folds.

Every model is scored with the same leave-one-type-out folds and calibration split as
``classifier.py``. scikit-learn and xgboost are imported only here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .classifier import LogReg, evaluate, leave_one_out, load_rows


class SkLogReg(LogReg):
    def fit(self, rows: Sequence[Mapping[str, Any]], y: Sequence[int], cols: Sequence[str]) -> "SkLogReg":
        from sklearn.linear_model import LogisticRegression

        steps, self.steps = self.steps, 0
        super().fit(rows, y, cols)  # standardisation stats only
        self.steps = steps
        clf = LogisticRegression(C=1 / (len(rows) * self.l2), tol=1e-10, max_iter=10_000)
        clf.fit([self._x(r) for r in rows], list(y))
        self.w, self.b = [float(v) for v in clf.coef_[0]], float(clf.intercept_[0])
        return self


class XGB(LogReg):
    def fit(self, rows: Sequence[Mapping[str, Any]], y: Sequence[int], cols: Sequence[str]) -> "XGB":
        from xgboost import XGBClassifier

        self.steps = 0
        super().fit(rows, y, cols)
        # ponytail: library defaults, small trees for ~50 rows; tune only on new data (§3.4)
        self.clf = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=0)
        self.clf.fit([self._x(r) for r in rows], list(y))
        return self

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> list[float]:
        return [float(p) for p in self.clf.predict_proba([self._x(r) for r in rows])[:, 1]]


def main() -> None:
    rows = load_rows("wikitext2", Path("reports/features"), Path("reports/grid/results.jsonl"))
    out: dict[str, Any] = {}
    for name, cls in (("logreg@1 (hand-rolled)", LogReg), ("sklearn LogisticRegression", SkLogReg), ("xgboost", XGB)):
        preds = leave_one_out(rows, "type", model_cls=cls)
        e = evaluate(preds)
        out[name] = {k: e.get(k) for k in ("n", "auroc", "ece")} | {"coverage": e["conformal"]["coverage"]}
        print(f"{name:28s} AUROC {e['auroc']['value']:.3f} {[round(v, 3) for v in e['auroc']['ci95']]}  "
              f"coverage {e['conformal']['coverage']['value']:.2f}  ECE {e['ece']:.3f}")
    path = Path("reports/risk/crosscheck.json")
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
