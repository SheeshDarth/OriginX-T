"""Per-sample risk scores (HANDBOOK §7.3, §10.7): ``risk_reflm`` and ``risk_knn``.

    python -m src.features.scores --config configs/features.yaml --dataset wikitext2

For every training sample of every contaminated grid cell (ratio > 0; a ratio-0 cell is all
human, so there is nothing to rank), one ``scores/1`` row per method:

- ``risk_reflm``: minus the sample's mean next-token loss under the fixed reference model
  (``features.model.per_text``). Generated text is easy for a language model, so a low
  loss is suspicious; negated so that higher is worse.
- ``risk_knn``: minus the sample's mean cosine distance to its nearest neighbours in the
  same cell (``features.semantic.per_text``). Near-copies sit close to each other.

Rows go to ``reports/scores/<method>/<dataset>.jsonl``. ``true_source`` is copied from the
sample for evaluation only; no score ever reads it. ``evaluate`` ranks each method against
``true_source != "human"`` (``src.evaluation.scoring``), pooled over all cells and per
contamination type, and the CLI writes that to ``reports/scores/<dataset>_eval.json``.
Track B's harm scores (X1) are compared against these.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from ..evaluation.scoring import auroc, bootstrap_ci
from . import model, semantic

METHODS = ("risk_reflm", "risk_knn")


def _neg(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(-x, 6)


def score_cell(
    dataset: str, cell: str, samples: Sequence[Any], texts: list[str], ctx: Mapping[str, Any],
    *, reflm: Callable[..., list] = model.per_text, knn: Callable[..., list] = semantic.per_text,
) -> dict[str, list[dict[str, Any]]]:
    """``{method: rows}`` for one cell; ``samples[i]`` is the sample behind ``texts[i]``."""
    values = {
        "risk_reflm": [_neg(v and v["loss"]) for v in reflm(texts, ctx)],
        "risk_knn": [_neg(v) for v in knn(texts, ctx)],
    }
    return {
        method: [{"schema": "scores/1", "dataset": dataset, "cell": cell, "sample_id": s.sample_id,
                  "method": method, "score": v, "higher_is": "worse", "true_source": s.source}
                 for s, v in zip(samples, vals)]
        for method, vals in values.items()
    }


def evaluate(rows: Sequence[Mapping[str, Any]], *, n_boot: int = 200, seed: int = 0) -> dict[str, Any]:
    """AUROC of one method's rows against ``true_source != "human"``: pooled, then per type."""
    scored = [r for r in rows if r["score"] is not None]

    def one(subset: Sequence[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
        labels = [int(r["true_source"] != "human") for r in subset]
        if len(set(labels)) < 2:
            return None
        scores = [r["score"] for r in subset]
        point, lo, hi = bootstrap_ci(auroc, labels, scores, n_boot=n_boot, seed=seed)
        return {"auroc": point, "ci": [lo, hi], "n": len(subset), "n_positive": sum(labels)}

    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for r in scored:
        by_type.setdefault(r["cell"].rsplit("_r", 1)[0], []).append(r)
    return {"pooled": one(scored), "by_type": {t: one(rs) for t, rs in sorted(by_type.items())},
            "n_unscored": len(rows) - len(scored)}


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    tmp.replace(path)


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    import yaml

    from ..finetune.collapse import training_text
    from ..finetune.grid import make_cells, training_set
    from ..ingestion.loaders import load_dataset
    from .suite import _device

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/features.yaml")
    p.add_argument("--dataset", default="wikitext2")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    grid = yaml.safe_load(Path(cfg["datasets"][args.dataset]["grid"]).read_text(encoding="utf-8"))
    kept = load_dataset(grid["data"]["kept"])
    ctx = {"device": _device(cfg.get("device", "auto")), "max_len": grid["train"]["max_len"]}
    out = {m: [] for m in METHODS}
    cells = [c for c in make_cells(grid["types"], grid["ratios"], grid["seeds"]) if c.ratio > 0]
    for i, cell in enumerate(cells, 1):
        samples = training_set(grid, cell, kept=kept)
        for method, rows in score_cell(args.dataset, cell.id, samples,
                                       [training_text(s) for s in samples], ctx).items():
            out[method] += rows
        print(f"  [{i}/{len(cells)}] {cell.id}", flush=True)
    root = Path("reports/scores")
    for method, rows in out.items():
        _write(root / method / f"{args.dataset}.jsonl", rows)
    report = {m: evaluate(rows) for m, rows in out.items()}
    (root / f"{args.dataset}_eval.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8",
                                                    newline="\n")
    for m, r in report.items():
        pooled = r["pooled"]
        print(f"{m}: pooled AUROC {pooled['auroc']:.3f} [{pooled['ci'][0]:.3f}, {pooled['ci'][1]:.3f}]; "
              + ", ".join(f"{t} {v['auroc']:.3f}" for t, v in r["by_type"].items() if v))


if __name__ == "__main__":
    main()
