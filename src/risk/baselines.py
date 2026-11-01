"""Static baselines the risk model must beat (HANDBOOK §8.2, H2), and the GATE 1 verdict.

**Single-feature baseline.** One feature column used as the risk score itself. Its
direction (does a higher value mean more risk?) is learned on the training fold
only, from the sign of mean(feature | degraded) - mean(feature | clean). Same
out-of-group folds as the model, so the two are scored on the same test cells.

- ``ref_lm_loss`` (§8.2 b): ``f_model_loss_mean``, from Track C's ``model`` family.
- The AI-text detector (§8.2 a) becomes a baseline the same way once a ``detector``
  feature family exists: add its column to ``BASELINES``.

**H2 comparison.** AUROC(model) - AUROC(baseline) on the same cells, with a paired
bootstrap CI (cells resampled once for both). The model beats a baseline when that
interval's lower end is above 0.

**GATE 1 (TRD §11).** The feature suite and a static baseline run cleanly on at
least 2 datasets. ``not_evaluable`` with one dataset, never ``passed``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..evaluation.scoring import auroc, bootstrap_ci
from .classifier import KEYS

BASELINES = {"ref_lm_loss": "f_model_loss_mean"}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def single_feature(rows: Sequence[Mapping[str, Any]], col: str, group: str) -> list[dict[str, Any]]:
    """Out-of-group scores from one column; the direction is fitted on each training fold."""
    preds = []
    for g in sorted({r[group] for r in rows}):
        train = [r for r in rows if r[group] != g and r.get(col) is not None]
        test = [r for r in rows if r[group] == g and not (group == "type" and r["ratio"] == 0)]
        pos = [r[col] for r in train if r["label"] == 1]
        neg = [r[col] for r in train if r["label"] == 0]
        if not test or not pos or not neg:
            continue
        sign = 1.0 if _mean(pos) >= _mean(neg) else -1.0
        fill = _mean([r[col] for r in train])  # a missing value scores as an average training cell
        for r in test:
            v = fill if r.get(col) is None else r[col]
            preds.append({**{k: r[k] for k in KEYS}, "label": r["label"], "score": sign * v, "fold": f"{group}={g}"})
    return preds


def compare(model: Sequence[Mapping[str, Any]], base: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """AUROC(model) - AUROC(baseline) on the cells both scored, with a paired bootstrap 95% CI."""
    scores = {(b["dataset"], b["cell"], b["fold"]): b["score"] for b in base}
    pairs = [(m["label"], m["p_degrade"], scores[k]) for m in model
             if (k := (m["dataset"], m["cell"], m["fold"])) in scores]
    if not pairs or len({y for y, _, _ in pairs}) < 2:
        return {"n": len(pairs), "note": "one class only or no shared cells: not comparable"}
    y, pm, pb = (list(c) for c in zip(*pairs))
    diff, lo, hi = bootstrap_ci(lambda y, a, b: auroc(y, a) - auroc(y, b), y, pm, pb)
    return {"n": len(y), "auroc_model": auroc(y, pm), "auroc_baseline": auroc(y, pb),
            "difference": {"value": diff, "ci95": [lo, hi]}, "model_beats_baseline": lo > 0}


def baseline_report(rows: Sequence[Mapping[str, Any]], model_preds: Sequence[Mapping[str, Any]],
                    group: str = "type") -> dict[str, Any]:
    """Each pre-registered baseline against the model, plus every single feature's AUROC (descriptive)."""
    out: dict[str, Any] = {}
    for name, col in BASELINES.items():
        base = single_feature(rows, col, group)
        out[name] = {"column": col, **compare(model_preds, base)} if base else \
            {"column": col, "note": "column not computed yet"}
    singles = {}
    for col in sorted({k for r in rows for k, v in r.items() if k.startswith("f_") and v is not None}):
        preds = single_feature(rows, col, group)
        y = [p["label"] for p in preds]
        if 0 < sum(y) < len(y):
            singles[col] = auroc(y, [p["score"] for p in preds])
    # chosen after seeing test AUROCs, so optimistic: descriptive only, never the H2 bar
    out["single_feature_auroc"] = singles
    return out


def gate_1(datasets_with_features: Sequence[str], baseline_ran: bool) -> dict[str, Any]:
    n = len(set(datasets_with_features))
    status = "not_evaluable" if n < 2 else ("passed" if baseline_ran else "failed")
    return {"status": status, "rule": "TRD §11: feature suite + static baseline run cleanly on >= 2 datasets",
            "datasets": sorted(set(datasets_with_features))}
