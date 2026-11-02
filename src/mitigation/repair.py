"""X1 repair-and-verify (HANDBOOK §9.3, B8/B9): replace the worst samples, retrain, measure recovery.

    python -m src.mitigation.repair --config configs/repair.yaml
    python -m src.mitigation.repair --config configs/repair.yaml --summarize-only

For each cell and strategy: rank the cell's training samples, replace the top ``k`` with
fresh human samples **at equal size**, fine-tune from ``base_model`` with the grid's
``train`` settings and the cell's own training seed (so the only difference from the grid
cell is the data), then score the grid's hidden-holdout and benchmark texts. One §7.5
``repair/1`` row per (cell, strategy), with

    recovery = (ppl_contaminated - ppl_repaired) / (ppl_contaminated - ppl_clean)

from the grid's own results for the cell and its seed's ratio-0 baseline (None when the
cell did not degrade, so there is nothing to recover).

Strategies (what is ranked worst first):

- ``harm_guided``: ``harm_lora_grad`` (``src.mitigation.harm``).
- ``repr_sim``: ``harm_repr_sim``, the mandatory representation-similarity baseline.
- ``risk_guided``: Track C's ``risk_reflm`` (low reference-model loss = looks generated).
  This is cleaning **by origin**, the comparison X1 has to win (STRETCH_GOALS X1).
- ``perplexity_filter``: the classic quality filter, highest reference-model loss first
  (``-risk_reflm``).
- ``random``: a seeded random choice, the floor Gate X1 is measured against.
- ``oracle``: the true contaminated samples first (``true_source``): the upper bound, and the
  only strategy allowed to read the label.

Ties are broken in a seeded random order. A strategy whose scores are missing for a cell
is skipped with a warning, so the pilot can run the risk, random and oracle arms before the
harm scores exist.

``k`` is ``contaminated`` (the cell's number of contaminated samples: the oracle's exact
budget, so every strategy gets the same budget a perfect cleaner would need) or an integer.

Replacements are a fresh draw of human text disjoint from both halves of the seed's draw,
the ``human_control`` pool and the whole hidden holdout, so a repaired set never trains on
the holdout and never repeats a text it already holds.

**Rare-cluster preservation** (§9.4, the core mitigation, PRD H5; B10). For the strategies
listed in ``preserve_rare_clusters``, each is also run as ``<strategy>+rare``: the same
ranking, except that a sample is never replaced while it is the last one left in the set
from a cluster that is rare in the seed's clean human draw (Track C's
``semantic.cluster_assignments``: the reference's k-means tail). Such samples move to the
end of the ranking, so they are replaced only if the budget cannot be met otherwise. The
summary's ``h5`` compares recovery and distinct-2 with and without the rule.

Rows are appended to ``output`` after every retrain, so a run cut short resumes. Repaired
checkpoints are deleted after scoring unless ``keep_models`` is set. The summary
(``<output stem>_summary.json``) holds the pilot go/no-go and Gate X1 (§11, fixed before
results): harm-guided recovery at least 10 points above random's at ratio 0.5 on at least
2 of the 3 collapse types, on every seed.
"""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

STRATEGIES = ("harm_guided", "repr_sim", "risk_guided", "random", "perplexity_filter", "oracle")
# strategy -> (scores/1 method it ranks by, sign); higher signed score = replaced first
SCORED: dict[str, tuple[str, int]] = {
    "harm_guided": ("harm_lora_grad", 1),
    "repr_sim": ("harm_repr_sim", 1),
    "risk_guided": ("risk_reflm", 1),
    "perplexity_filter": ("risk_reflm", -1),
}
GATE_MARGIN = 0.10      # Gate X1: harm-guided recovery - random recovery, in recovery units (10 points)
GATE_MIN_TYPES = 2      # ... on at least this many collapse types, every seed
RARE = "+rare"          # suffix of a strategy run with rare-cluster preservation (B10)


def recovery(ppl_contaminated: float, ppl_clean: float, ppl_repaired: float) -> Optional[float]:
    """§7.5 recovery: 1 = back to clean, 0 = no better, negative = worse. None if nothing degraded."""
    if ppl_contaminated <= ppl_clean:
        return None
    return round((ppl_contaminated - ppl_repaired) / (ppl_contaminated - ppl_clean), 6)


def rank(samples: Sequence[Any], strategy: str, scores: Optional[Mapping[str, Optional[float]]],
         seed: int) -> list[int]:
    """Indices of ``samples``, worst first under ``strategy``. Unscored samples go last."""
    order = list(range(len(samples)))
    random.Random(f"repair-rank-{strategy}-{seed}").shuffle(order)  # the tie-break
    if strategy == "random":
        return order
    if strategy == "oracle":
        return sorted(order, key=lambda i: samples[i].source == "human")
    method, sign = SCORED[strategy]
    if scores is None:
        raise ValueError(f"{strategy} needs {method} scores")
    signed = [None if scores.get(s.sample_id) is None else sign * scores[s.sample_id] for s in samples]
    return sorted(order, key=lambda i: (signed[i] is None, -(signed[i] or 0.0)))


def protect_rare(worst: Sequence[int], labels: Sequence[int], tail: set[int]) -> list[int]:
    """``worst`` with every sample that would be the last of its tail cluster moved to the end.

    Walking the ranking, a sample from a tail cluster is deferred when it is the only one of
    its cluster still in the set; otherwise it is taken and its cluster's count drops. So any
    top-k of the result keeps every tail cluster non-empty, unless k exceeds the samples taken.
    """
    left = {c: sum(1 for x in labels if x == c) for c in tail}
    taken, deferred = [], []
    for i in worst:
        c = labels[i]
        if c in tail and left[c] == 1:
            deferred.append(i)
            continue
        taken.append(i)
        if c in tail:
            left[c] -= 1
    return taken + deferred


def replacements(kept: Sequence[Any], holdout: Sequence[Any], data: Mapping[str, Any], seed: int,
                 k: int) -> list[Any]:
    """``k`` fresh human samples for a seed: disjoint from its draw, its human_control pool and the holdout."""
    from ..finetune.collapse import _pick
    from ..finetune.grid import draw_human, human_control_pool
    from ..ingestion.check_leakage import text_fingerprint

    human, source = draw_human(kept, data, seed)
    control = human_control_pool(kept, holdout, data, seed)
    used = {text_fingerprint(s) for s in (*human, *source, *control, *holdout)}
    fresh = [s for s in kept if text_fingerprint(s) not in used]
    picked = _pick(fresh, k, data["min_words"], data["max_words"], random.Random(f"repair-{seed}"))
    if len(picked) < k:
        raise ValueError(f"only {len(picked)} fresh human samples for seed {seed}, need {k}")
    return picked


def repaired(samples: Sequence[Any], worst: Sequence[int], fresh: Sequence[Any], k: int) -> list[Any]:
    """``samples`` with its ``k`` worst replaced in place by ``fresh[:k]``: same size, same order."""
    out = list(samples)
    for i, new in zip(worst[:k], fresh[:k]):
        out[i] = new
    return out


def load_scores(root: Path, dataset: str, method: str) -> dict[str, dict[str, Optional[float]]]:
    """``{cell: {sample_id: score}}`` from ``reports/scores/<method>/<dataset>.jsonl``; empty if absent."""
    path = root / method / f"{dataset}.jsonl"
    out: dict[str, dict[str, Optional[float]]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["cell"], {})[r["sample_id"]] = r["score"]
    return out


def run_repair(
    cells: Sequence[Any], strategies: Sequence[str], *, dataset: str,
    samples_for: Callable[[Any], list[Any]],
    fresh_for: Callable[[int, int], list[Any]],
    scores_for: Callable[[str, str], Optional[Mapping[str, Optional[float]]]],
    grid_rows: Mapping[str, Mapping[str, Any]],
    train: Callable[[list[Any], Any, str], Any],
    evaluate: Callable[[Any], dict[str, float]],
    release: Callable[[Any], None] = lambda model: None,
    k: Any = "contaminated",
    done: Optional[Mapping[tuple[str, str], Mapping[str, Any]]] = None,
    on_result: Callable[[dict[str, Any]], None] = lambda row: None,
    warn: Callable[[str], None] = print,
    preserve: Sequence[str] = (),
    clusters_for: Optional[Callable[[Any, list[Any]], tuple[list[int], set[int]]]] = None,
) -> list[dict[str, Any]]:
    """Retrain every (cell, strategy) not in ``done`` and return the repair/1 rows, in order.

    ``train(samples, cell, strategy)`` returns a model handle that ``evaluate`` scores and
    ``release`` frees. ``grid_rows`` is the grid's results by cell id (for recovery). Each
    strategy in ``preserve`` also runs as ``<strategy>+rare``, with ``clusters_for(cell,
    samples)`` giving each sample's cluster and the rare (tail) clusters.
    """
    if preserve and clusters_for is None:
        raise ValueError("preserve needs clusters_for")
    from ..evaluation.metrics import distinct_n
    from ..finetune.collapse import training_text

    rows: list[dict[str, Any]] = []
    for cell in cells:
        samples = samples_for(cell)
        n_bad = sum(s.source != "human" for s in samples)
        budget = n_bad if k == "contaminated" else int(k)
        if not 0 < budget <= len(samples):
            raise ValueError(f"{cell.id}: k={budget} must be in 1..{len(samples)}")
        cont = grid_rows[cell.id]
        clean = next(r for r in grid_rows.values() if r["ratio"] == 0 and r["seed"] == cell.seed)
        fresh, clusters = None, None
        runs = [(s, False) for s in strategies] + [(s, True) for s in strategies if s in preserve]
        for strategy, rare in runs:
            name = strategy + RARE if rare else strategy
            if done and (cell.id, name) in done:
                rows.append(dict(done[(cell.id, name)]))
                continue
            scores = None
            if strategy in SCORED:
                scores = scores_for(SCORED[strategy][0], cell.id)
                if not scores:
                    warn(f"  {cell.id} {name}: no {SCORED[strategy][0]} scores, skipped")
                    continue
            fresh = fresh if fresh is not None else fresh_for(cell.seed, budget)
            worst = rank(samples, strategy, scores, cell.seed)
            if rare:
                clusters = clusters if clusters is not None else clusters_for(cell, samples)
                worst = protect_rare(worst, clusters[0], clusters[1])
            data = repaired(samples, worst, fresh, budget)
            model = train(data, cell, name)
            try:
                metrics = evaluate(model)
            finally:
                release(model)
            row = {
                "schema": "repair/1", "dataset": dataset, "cell": cell.id, "strategy": name,
                "k_replaced": budget,
                "holdout_ppl": round(metrics["holdout_ppl"], 4),
                "benchmark_ppl": round(metrics["benchmark_ppl"], 4),
                "distinct_2": round(distinct_n([training_text(s) for s in data], 2), 6),
                "recovery": recovery(cont["holdout_ppl"], clean["holdout_ppl"], metrics["holdout_ppl"]),
                "seed": cell.seed,
            }
            rows.append(row)
            on_result(row)
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], *, collapse_types: Sequence[str],
              seeds: Sequence[int], pilot_cells: Sequence[str]) -> dict[str, Any]:
    """Pilot go/no-go (harm-guided beats random on every pilot cell), Gate X1, and H5.

    ``h5``: per strategy run with rare-cluster preservation, per cell, the change in recovery
    and in distinct-2 that the rule made (``+rare`` minus without).
    """
    rec = {(r["cell"], r["strategy"]): r["recovery"] for r in rows}
    by_key = {(r["cell"], r["strategy"]): r for r in rows}
    h5: dict[str, dict[str, Any]] = {}
    for (cell, name), r in by_key.items():
        base = by_key.get((cell, name[: -len(RARE)])) if name.endswith(RARE) else None
        if base is not None:
            d_rec = (None if r["recovery"] is None or base["recovery"] is None
                     else round(r["recovery"] - base["recovery"], 6))
            h5.setdefault(name[: -len(RARE)], {})[cell] = {
                "recovery_change": d_rec, "distinct_2_change": round(r["distinct_2"] - base["distinct_2"], 6)}

    def margin(cell: str) -> Optional[float]:
        h, r = rec.get((cell, "harm_guided")), rec.get((cell, "random"))
        return None if h is None or r is None else round(h - r, 6)

    pilot = {c: margin(c) for c in pilot_cells}
    pilot_status = ("not evaluable" if any(m is None for m in pilot.values())
                    else "go" if all(m > 0 for m in pilot.values()) else "no-go")
    per_type = {t: {s: margin(f"{t}_r0.5_s{s}") for s in seeds} for t in collapse_types}
    complete = all(m is not None for by_seed in per_type.values() for m in by_seed.values())
    passing = [t for t, by_seed in per_type.items()
               if all(m is not None and m >= GATE_MARGIN for m in by_seed.values())]
    gate = "not evaluable" if not complete else "passed" if len(passing) >= GATE_MIN_TYPES else "failed"
    return {"pilot": {"margins": pilot, "status": pilot_status},
            "gate_x1": {"margins": per_type, "passing_types": passing, "status": gate,
                        "rule": f"harm_guided - random recovery >= {GATE_MARGIN} at ratio 0.5 on >= "
                                f"{GATE_MIN_TYPES} of {list(collapse_types)}, every seed"},
            "h5": h5}


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    import yaml

    from ..finetune.collapse import _pick, load_holdout, training_text
    from ..finetune.grid import make_cells, training_set
    from ..ingestion.loaders import load_dataset

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/repair.yaml")
    p.add_argument("--summarize-only", action="store_true", help="retrain nothing; rebuild the summary")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    grid = yaml.safe_load(Path(cfg["grid_config"]).read_text(encoding="utf-8"))
    preserve = cfg.get("preserve_rare_clusters") or []
    unknown = sorted(set(cfg["strategies"]) - set(STRATEGIES)) + sorted(set(preserve) - set(cfg["strategies"]))
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; use {list(STRATEGIES)} (preserve_rare_clusters "
                         "must also be in strategies)")
    by_id = {c.id: c for c in make_cells(grid["types"], grid["ratios"], grid["seeds"])}
    bad = [c for c in cfg["cells"] if c not in by_id or by_id[c].ratio == 0]
    if bad:
        raise SystemExit(f"{bad} are not contaminated cells of {cfg['grid_config']}")
    cells = [by_id[c] for c in cfg["cells"]]
    out = Path(cfg["output"])
    done = {(r["cell"], r["strategy"]): r for r in _read(out)}

    if not args.summarize_only:
        from ..evaluation.metrics import perplexity
        from ..features.suite import _device
        from ..finetune.lora import train_lora

        d, t = grid["data"], grid["train"]
        device = _device(cfg.get("device", "auto"))
        kept = load_dataset(d["kept"])
        holdout_rows = load_dataset(d["holdout"])
        holdout = load_holdout(d)
        bench = [s.response for s in _pick(load_dataset(d["benchmark"]), d["n_benchmark"],
                                           d["min_words"], d["max_words"], random.Random("benchmark"))]
        grid_rows = {r["cell"]: r for r in _read(Path(grid["output_dir"]) / "results.jsonl")}
        score_root = Path(cfg.get("scores_dir", "reports/scores"))
        cache: dict[str, dict[str, dict[str, Optional[float]]]] = {}

        def scores_for(method: str, cell: str) -> Optional[Mapping[str, Optional[float]]]:
            if method not in cache:
                cache[method] = load_scores(score_root, cfg["dataset"], method)
            return cache[method].get(cell)

        models = Path(cfg["models_dir"])

        def train(data: list[Any], cell: Any, strategy: str) -> Path:
            print(f"{cell.id} {strategy}: training on {len(data)} texts", flush=True)
            return train_lora([training_text(s) for s in data], models / f"{cell.id}_{strategy}",
                              base_model=grid["base_model"], seed=cell.seed * 100, device=device, **t)

        def evaluate(model_dir: Path) -> dict[str, float]:
            scores = {name: perplexity(model_dir, texts, max_len=t["max_len"], batch_size=t["batch_size"],
                                       device=device)
                      for name, texts in (("holdout_ppl", holdout), ("benchmark_ppl", bench))}
            print(f"    holdout {scores['holdout_ppl']:.2f}  benchmark {scores['benchmark_ppl']:.2f}", flush=True)
            return scores

        def release(model_dir: Path) -> None:
            if not cfg.get("keep_models", False):
                shutil.rmtree(model_dir, ignore_errors=True)

        def on_result(row: dict[str, Any]) -> None:
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row) + "\n")

        def clusters_for(cell: Any, samples: list[Any]) -> tuple[list[int], set[int]]:
            from ..features.semantic import cluster_assignments
            from ..finetune.grid import draw_human

            reference = [training_text(s) for s in draw_human(kept, d, cell.seed)[0]]  # the seed's clean draw
            got = cluster_assignments([training_text(s) for s in samples],
                                      {"reference_texts": reference, "device": device})
            return got["labels"], set(got["tail"])

        run_repair(
            cells, cfg["strategies"], dataset=cfg["dataset"],
            samples_for=lambda cell: training_set(grid, cell, kept=kept),
            fresh_for=lambda seed, k: replacements(kept, holdout_rows, d, seed, k),
            scores_for=scores_for, grid_rows=grid_rows, train=train, evaluate=evaluate,
            release=release, k=cfg.get("k", "contaminated"), done=done, on_result=on_result,
            preserve=preserve, clusters_for=clusters_for,
        )

    rows = _read(out)
    summary = summarize(rows, collapse_types=grid["gate"]["collapse_types"], seeds=grid["seeds"],
                        pilot_cells=cfg["pilot_cells"])
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / f"{out.stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                         encoding="utf-8", newline="\n")
    for r in rows:
        print(f"{r['cell']:<22} {r['strategy']:<18} holdout {r['holdout_ppl']:.2f}  recovery {r['recovery']}")
    print(f"Pilot: {summary['pilot']['status']}   Gate X1: {summary['gate_x1']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
