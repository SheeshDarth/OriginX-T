"""Audit a dataset before fine-tuning on it: the product's core question (HANDBOOK §1, §8.6 Audit).

    python -m src.risk.audit my_data.jsonl [--model reports/risk/model.json]
                             [--benchmark items.jsonl | --grid configs/grid.yaml] [--out audit.json]

Steps: load the JSONL (any shape ``src.ingestion.loaders`` accepts), turn each record
into the text the model would train on, compute the same feature families as the
grid (``src.features.suite.FAMILIES``), and apply the deployed risk model
(``reports/risk/model.json``, written by ``python -m src.risk.classifier``).

The report (``audit/1``):
- ``p_degrade`` and its ``conformal_set`` at the model's ``alpha``.
- ``contributions``: each feature's push on the log-odds, ``w * standardised x``,
  largest first. That is what drives the score, for this dataset.
- ``missing_features``: columns the model knows but this run could not compute
  (no benchmark file, or a family not installed). They enter as the training mean.
- ``flagged``: samples worth a look, each with its reasons. A near-copy of a benchmark
  item (similarity > 0.9) is a leak; a text whose 3-grams repeat a lot is a model loop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..features import leak as leak_family
from ..features.lex import lex
from ..features.suite import FAMILIES, features_for
from . import conformal
from .classifier import LogReg

LEAK_THRESHOLD = leak_family.THRESHOLD
REPEAT_THRESHOLD = 0.2  # ponytail: a fixed cut on in-text 3-gram repeats; fit it on the bench once Track B has labels


def flag_samples(texts: Sequence[str], ids: Sequence[str], bench: Sequence[str]) -> list[dict[str, Any]]:
    """Samples with at least one reason, most suspicious first (leaks before loops)."""
    best = leak_family.best_matches(list(texts), list(bench)) if bench else [None] * len(texts)
    out = []
    for i, (t, sid, sim) in enumerate(zip(texts, ids, best)):
        rep = lex([t])["repeat_3gram"]
        reasons = []
        if sim is not None and sim > LEAK_THRESHOLD:
            reasons.append(f"near-copy of a benchmark item (similarity {sim:.2f})")
        if rep is not None and rep > REPEAT_THRESHOLD:
            reasons.append(f"repeats itself ({rep:.0%} of its 3-grams recur)")
        if reasons:
            out.append({"index": i, "sample_id": sid, "text": t[:300], "benchmark_similarity": sim,
                        "repeat_3gram": rep, "reasons": reasons})
    return sorted(out, key=lambda f: (-(f["benchmark_similarity"] or 0), -(f["repeat_3gram"] or 0)))


def audit(texts: Sequence[str], ids: Sequence[str], deployed: Mapping[str, Any],
          bench: Sequence[str] = ()) -> dict[str, Any]:
    model = LogReg.from_dict(deployed)
    feats = features_for(list(texts), FAMILIES, {"benchmark": list(bench)})
    p = model.predict([feats])[0]
    q = math.inf if deployed["q"] is None else deployed["q"]
    contrib = model.contributions(feats)
    flagged = flag_samples(texts, ids, bench)
    return {
        "schema": "audit/1",
        "n_samples": len(texts),
        "model": deployed["model"],
        "p_degrade": p,
        "conformal_set": conformal.prediction_set(p, q),
        "alpha": deployed["alpha"],
        "features": feats,
        "missing_features": [c for c in model.cols if feats.get(c) is None],
        "contributions": [{"feature": c, "value": feats.get(c), "contribution": v}
                          for c, v in sorted(contrib.items(), key=lambda kv: -abs(kv[1]))],
        "n_flagged": len(flagged),
        "flagged": flagged,
    }


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    from ..finetune.collapse import training_text
    from ..ingestion.loaders import load_dataset

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dataset")
    p.add_argument("--model", default="reports/risk/model.json")
    p.add_argument("--benchmark", help="JSONL of benchmark items to check for leaks")
    p.add_argument("--grid", help="or: use the benchmark items this grid config leaked and scored (e.g. configs/grid.yaml)")
    p.add_argument("--out", help="write the report here (default: stdout)")
    args = p.parse_args(argv)
    samples = load_dataset(args.dataset)
    bench = [s.response for s in load_dataset(args.benchmark)] if args.benchmark else []
    if args.grid:
        import random

        import yaml

        from ..finetune.collapse import _pick

        d = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))["data"]
        # the same draw as grid._train_all, so "leak" means a leak of what the grid measured
        bench = [s.response for s in _pick(load_dataset(d["benchmark"]), d["n_benchmark"], d["min_words"],
                                           d["max_words"], random.Random("benchmark"))]
    deployed = json.loads(Path(args.model).read_text(encoding="utf-8"))
    report = audit([training_text(s) for s in samples], [s.sample_id for s in samples], deployed, bench)
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        print(f"p(degrade) {report['p_degrade']:.2f}, set {report['conformal_set']}, "
              f"{report['n_flagged']} of {report['n_samples']} samples flagged -> {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
