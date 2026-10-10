"""Sprint-6 fine-tuning grid: contamination type x mix ratio x seed.

Every cell fine-tunes the base model on a dataset that is ``ratio`` contaminated
with one contamination type, then scores it on the hidden holdout. The result is
a collapse curve per type (holdout perplexity against contamination ratio).
GATE 2 is a pre-registered rule over those curves (``evaluate_gate``): every
collapse type must show a dose-response and clear the seed-to-seed noise at a
low dose. ``benchmark_near`` is reported, not gated.

    python -m src.finetune.grid --config configs/grid.yaml
    python -m src.finetune.grid --config configs/grid.yaml --summarize-only

``--summarize-only`` (alias ``--report-only``) trains nothing: it recomputes
summary.json, the tables and the figure from the existing results.jsonl, and
refuses if that file was made under different settings.

Changing how a contamination type is generated:

    python -m src.finetune.grid --config configs/grid.yaml --rerun-types benchmark_near

Generated pools are cached under ``pools_dir/<config fingerprint>/``, and a code
change does not change the fingerprint, so a fixed generator would silently keep
using its old pool. Each type has a version in ``POOL_VERSIONS`` (default 1). Bump
the type's version when its generator changes: version 1 keeps the original file
name (``<type>_s<seed>.jsonl``, so existing pools are still reused) and a higher
version is part of it (``<type>_v<version>_s<seed>.jsonl``), so a bump makes a new
pool file and leaves the old one on disk. A pool built from another (``recursive``
from ``synthetic``) goes stale with it. Rows record the version they were made
with (``pool_version``); rows made with an older one are reported as ``stale_cells``
and warned about, since they would mix two generators in one curve.

``--rerun-types TYPE [TYPE ...]`` is how to redo those cells: it drops those types'
contaminated rows (ratio > 0) from results.jsonl, after copying the file to
``results.jsonl.bak-<time>`` next to it, then trains just those cells. The ratio-0
rows are kept: they are all human, whatever the type, so the shared baselines do
not change. It refuses results made under different settings, like resume.

If a re-run is cut short, finish it with the plain command, without the flag: it
trains the cells that are missing and drops nothing. Giving ``--rerun-types``
again drops that type's rows once more, including any already redone, and makes
another backup. A re-run's cells are logged as new MLflow runs next to the old ones
(the ``pool_version`` parameter tells them apart), and replace the cells' checkpoints
if ``keep_models`` is on.

Results are appended to ``{output_dir}/results.jsonl`` after every cell, so a
run cut short by a Kaggle session limit resumes where it stopped. Each cell is
also logged as an MLflow run (see ``src.evaluation.tracking``).

How a cell is built, so the only difference between cells is the contamination:

- Per seed, one draw of human text is split in two disjoint halves. The first
  is the clean data (identical to the spike's Gen-0 data for that seed); the
  second is the *source* the contamination is made from. A contaminated sample
  therefore never shares text with a clean one in the same mixture.
- ``synthetic`` is the pretrained ``generator_model`` continuing a source text
  (Gen-1), ``paraphrased`` is that model rewriting the source, and
  ``benchmark_near`` is surface-perturbed copies of eval-benchmark items.
- ``recursive`` is the collapse spike's recipe: a model is fine-tuned on Gen-1
  (the ``synthetic`` pool) and that fine-tuned model writes Gen-2, then a model
  is fine-tuned on Gen-2 to write Gen-3, and so on up to ``recursive_depth``.
  Each fine-tune starts from ``base_model`` with the grid's ``train`` settings.
  Only the last generation is the contamination; the intermediate checkpoints
  are deleted unless ``keep_models`` is set. This costs ``recursive_depth - 1``
  extra fine-tunes per seed on top of the cells.
- At ratio 0 the training set is all human whatever the type, so one model per
  seed serves every type (``share_baseline``). That is 12 of the 60 cells; set
  it to false to train them all separately.
- Every model is also scored on the benchmark items. Benchmark leakage shows up
  as a *drop* there (the model has memorised them), not as a holdout rise.

``run_grid`` takes the trainer, scorer and data builders as functions, so the
bookkeeping is tested on CPU with stubs. ``main`` plugs in the real ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from ..evaluation.metrics import distinct_n, perplexity
from ..evaluation.tracking import Run, track_run
from ..generation.benchmark_near import make_near_duplicates
from ..generation.mixer import mix
from ..generation.model_based import Generate, hf_generator, make_paraphrased, make_synthetic
from ..ingestion.loaders import load_dataset, write_jsonl
from ..ingestion.schema import Sample
from .collapse import _pick, load_holdout, training_text
from .lora import train_lora

CONTAMINATION_TYPES: tuple[str, ...] = ("synthetic", "recursive", "paraphrased", "benchmark_near")

# Settings that change what a cell's number means. A results file (and a cache of
# generated contamination) is only reusable while these are unchanged; the axes
# (seeds, ratios, types) are left out so a finished grid can be extended.
_FINGERPRINT_KEYS = (
    "base_model", "generator_model", "recursive_depth", "data", "benchmark_near", "train",
    "generate", "share_baseline",
)


# The version of each type's pool generator (default 1). Not part of the config fingerprint
# on purpose: bumping it must not invalidate the results of the other types. Bump a type's
# version when the code that makes its pool changes (see the module docstring).
#   benchmark_near 2: full-text ALL CAPS / lowercase recasing removed (PR #12).
POOL_VERSIONS: dict[str, int] = {"benchmark_near": 2}
# Pools built from other pools go stale with them.
_POOL_INPUTS: dict[str, tuple[str, ...]] = {"recursive": ("synthetic",)}


def pool_tag(kind: str) -> str:
    """The pool's version, e.g. ``"1"``, ``"2"``, or ``"1.2"`` for a pool built from another."""
    return ".".join(str(POOL_VERSIONS.get(k, 1)) for k in (kind, *_POOL_INPUTS.get(kind, ())))


def _first_version_tag(kind: str) -> str:
    return ".".join("1" for _ in pool_tag(kind).split("."))


def pool_path(pools_dir: str | Path, kind: str, seed: int) -> Path:
    """Where a pool is cached. The first version keeps the original file name."""
    tag = pool_tag(kind)
    name = f"{kind}_s{seed}.jsonl" if tag == _first_version_tag(kind) else f"{kind}_v{tag}_s{seed}.jsonl"
    return Path(pools_dir) / name


def stale_cells(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Cells whose contamination came from an older version of its pool generator.

    Rows from before versions existed count as the first version. Ratio-0 rows have
    no pool, so they are never stale.
    """
    return [
        r["cell"] for r in rows
        if r["ratio"] > 0 and r.get("pool_version", _first_version_tag(r["type"])) != pool_tag(r["type"])
    ]


@dataclass(frozen=True)
class Cell:
    type: str
    ratio: float
    seed: int

    @property
    def id(self) -> str:
        return f"{self.type}_r{self.ratio:g}_s{self.seed}"


def make_cells(
    types: Sequence[str], ratios: Sequence[float], seeds: Sequence[int]
) -> list[Cell]:
    """The full grid, one seed at a time.

    Seed-major order means a run cut short still holds complete curves for the
    seeds that finished, rather than a thin slice of every seed.
    """
    unknown = [t for t in types if t not in CONTAMINATION_TYPES]
    if unknown:
        raise ValueError(f"unknown contamination type(s) {unknown}; use {CONTAMINATION_TYPES}")
    bad = [r for r in ratios if not 0.0 <= r <= 1.0]
    if bad:
        raise ValueError(f"ratios must be in [0, 1], got {bad}")
    for name, values in (("types", types), ("ratios", ratios), ("seeds", seeds)):
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate entries in {name}: {list(values)}")
    return [Cell(t, r, s) for s in seeds for r in ratios for t in types]


def resample_empty(generate: Generate, tries: int = 5) -> Generate:
    """Re-ask for outputs that come back empty.

    A sampling model can emit end-of-text first, and an empty paraphrase is not a
    valid sample: one such output out of a thousand would otherwise kill a
    multi-hour run, then recur on resume because generation is seeded. Only the
    empty ones are regenerated; the rest are kept as they are.
    """

    def wrapped(texts: list[str]) -> list[str]:
        outs = list(generate(texts))
        for _ in range(tries):
            bad = [i for i, o in enumerate(outs) if not o.strip()]
            if not bad:
                return outs
            for i, o in zip(bad, generate([texts[i] for i in bad])):
                outs[i] = o
        if any(not o.strip() for o in outs):
            raise ValueError(f"the model still returned empty text after {tries} retries")
        return outs

    return wrapped


def make_recursive(
    gen1: Sequence[Sample],
    *,
    depth: int,
    train: Callable[[list[str], int], Any],
    generator_for: Callable[[Any], Generate],
    release: Callable[[Any], None] = lambda model: None,
) -> list[Sample]:
    """Gen-``depth`` contamination, each generation written by a model trained on the last.

    ``gen1`` is already model-written (the ``synthetic`` pool, from the pretrained
    generator). Each further step fine-tunes ``train(texts, g)`` on the data so far
    and has that model continue it with ``make_synthetic``, so the final samples
    are ``source="recursive"`` with ``generation == depth``. This is the loop of
    ``collapse.run_generations`` without its last step: that one also trains and
    scores a model on the final generation and returns only metrics, which here
    would cost an unused fine-tune per seed and still not return the data.
    """
    if depth < 2:
        raise ValueError(f"depth must be >= 2 (depth 1 is the synthetic pool), got {depth}")
    wrong = {s.generation for s in gen1} - {1}
    if wrong:
        raise ValueError(f"gen1 must be generation-1 samples, found generation(s) {sorted(wrong)}")
    data = list(gen1)
    for g in range(1, depth):
        model = train([training_text(s) for s in data], g)
        try:
            data = make_synthetic(data, generator_for(model))
        finally:
            release(model)
    return data


def config_fingerprint(cfg: Mapping[str, Any]) -> str:
    """Short hash of the settings that define what a cell measures."""
    blob = json.dumps({k: cfg.get(k) for k in _FINGERPRINT_KEYS}, sort_keys=True, default=str)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=6).hexdigest()


def load_results(path: str | Path, fingerprint: str) -> dict[str, dict[str, Any]]:
    """Finished cells from a results file, by cell id. Empty if there is no file.

    Refuses rows made under different settings: mixing them into one curve
    would be wrong, and a silent restart would throw hours of work away.
    """
    path = Path(path)
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("config") != fingerprint:
            raise ValueError(
                f"{path} holds results from different settings (config {row.get('config')}, "
                f"now {fingerprint}). Point output_dir somewhere new, or delete the file."
            )
        done[row["cell"]] = row
    return done


def drop_types(
    path: str | Path, fingerprint: str, types: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], list[str], Optional[Path]]:
    """Remove the contaminated rows (ratio > 0) of ``types`` from a results file.

    Returns ``(kept rows by cell id, dropped cell ids, backup path)``. The ratio-0
    rows are kept even for these types: they are all human, so they are the shared
    baselines and do not depend on the generator. Nothing is touched if the file was
    made under different settings (``load_results`` raises first) or if there is
    nothing to drop. Otherwise the file is first copied to ``<name>.bak-<time>``
    beside it, then replaced by the kept rows.
    """
    path = Path(path)
    rows = load_results(path, fingerprint)
    dropped = [k for k, r in rows.items() if r["type"] in types and r["ratio"] > 0]
    if not dropped:
        return rows, [], None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    n = 1
    while backup.exists():  # two re-runs in the same second must not overwrite a backup
        backup = path.with_name(f"{path.name}.bak-{stamp}-{n}")
        n += 1
    shutil.copy2(path, backup)
    kept = {k: r for k, r in rows.items() if k not in set(dropped)}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in kept.values()), encoding="utf-8")
    tmp.replace(path)
    return kept, dropped, backup


def run_grid(
    cells: Sequence[Cell],
    *,
    human_for: Callable[[int], list[Sample]],
    pool_for: Callable[[str, int], list[Sample]],
    train: Callable[[list[str], Cell], Any],
    evaluate: Callable[[Any], dict[str, float]],
    release: Callable[[Any], None] = lambda model: None,
    track: Callable[[Cell], AbstractContextManager[Run]] = lambda cell: track_run(None, cell.id, {}),
    on_result: Callable[[dict[str, Any]], None] = lambda row: None,
    done: Optional[Mapping[str, dict[str, Any]]] = None,
    share_baseline: bool = True,
    fingerprint: str = "",
) -> list[dict[str, Any]]:
    """Run every cell not already in ``done`` and return one row per cell.

    ``human_for(seed)`` is the clean data and ``pool_for(type, seed)`` the
    contaminated pool a mixture draws from. ``train(texts, cell)`` returns a
    model handle, ``evaluate(model)`` scores it, and ``release(model)`` frees it
    once scored. ``on_result`` sees each new row as soon as it exists.
    """
    finished: dict[str, dict[str, Any]] = dict(done or {})
    humans: dict[int, list[Sample]] = {}
    rows: list[dict[str, Any]] = []

    for cell in cells:
        if cell.id in finished:
            rows.append(finished[cell.id])
            continue
        with track(cell) as run:
            baseline = None
            if share_baseline and cell.ratio == 0:
                baseline = next(
                    (r for r in finished.values() if r["ratio"] == 0 and r["seed"] == cell.seed),
                    None,
                )
            if baseline is not None:
                # Same data, same training seed: nothing to retrain, only to relabel.
                row = {**baseline, "cell": cell.id, "type": cell.type,
                       "shared_baseline": True, "seconds": 0.0}
            else:
                if cell.seed not in humans:
                    humans[cell.seed] = human_for(cell.seed)
                human = humans[cell.seed]
                pool = pool_for(cell.type, cell.seed) if cell.ratio > 0 else []
                mixed = mix(human, pool, ratio=cell.ratio, seed=cell.seed, total=len(human))
                texts = [training_text(s) for s in mixed]
                start = time.monotonic()
                model = train(texts, cell)
                try:
                    metrics = evaluate(model)
                finally:
                    release(model)
                row = {
                    "cell": cell.id, "type": cell.type, "ratio": cell.ratio, "seed": cell.seed,
                    "shared_baseline": False,
                    "n_train": len(texts),
                    "n_contaminated": sum(s.source != "human" for s in mixed),
                    **({"pool_version": pool_tag(cell.type)} if cell.ratio > 0 else {}),
                    **metrics,
                    "distinct_1": distinct_n(texts, 1),
                    "distinct_2": distinct_n(texts, 2),
                    "seconds": time.monotonic() - start,
                }
            row["config"] = fingerprint
            run.log_metrics({
                k: float(v) for k, v in row.items()
                if isinstance(v, (int, float)) and k not in ("ratio", "seed", "n_train")
            })
        finished[cell.id] = row
        rows.append(row)
        on_result(row)
    return rows


# A drop of exactly the tolerance counts as inside it; this absorbs float rounding at that edge.
_DROP_EPS = 1e-9
_GATE_KEYS = ("collapse_types", "monotone_tolerance", "low_dose_ratio")


def validate_gate(gate: Any, *, types: Sequence[str], ratios: Sequence[float]) -> None:
    """Raise ValueError unless the ``gate:`` config section can be evaluated on this grid."""
    if not isinstance(gate, Mapping):
        raise ValueError("the config needs a `gate:` section (it replaces gate_min_ppl_increase)")
    missing = [k for k in _GATE_KEYS if k not in gate]
    if missing:
        raise ValueError(f"gate: is missing {missing}")
    collapse = gate["collapse_types"]
    if not collapse or any(t not in types or t not in CONTAMINATION_TYPES for t in collapse):
        raise ValueError(
            f"gate.collapse_types must be a non-empty subset of the configured types {list(types)}, "
            f"got {collapse}"
        )
    if not 0 <= gate["monotone_tolerance"] < 1:
        raise ValueError(f"gate.monotone_tolerance must be in [0, 1), got {gate['monotone_tolerance']}")
    if gate["low_dose_ratio"] not in ratios or gate["low_dose_ratio"] <= 0:
        raise ValueError(
            f"gate.low_dose_ratio must be one of the configured non-zero ratios {list(ratios)}, "
            f"got {gate['low_dose_ratio']}"
        )


def evaluate_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    ratios: Sequence[float],
    gate: Mapping[str, Any],
    types: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """GATE 2: does every collapse type degrade the model, on a pre-registered rule?

    A collapse type degrades when both hold, computed from the result rows alone:

    - dose-response: mean holdout perplexity across seeds is non-decreasing over
      the configured ratios, allowing a relative drop of at most
      ``monotone_tolerance`` between neighbouring ratios;
    - above noise at low dose: at ``low_dose_ratio``, every seed's perplexity is
      above that seed's own ratio-0 baseline by more than the spread (max - min)
      of the ratio-0 baselines across seeds.

    The gate passes when every type in ``collapse_types`` degrades. It cannot be
    evaluated, and so never passes, while the grid is partial: any cell of the
    configured ``types`` (every ratio and seed; default just the collapse types)
    with no result makes the status ``not_evaluable``, even when the missing cell
    is of a type that is not gated. Each gated type still carries the numbers
    behind its own verdict, so a failure says which condition failed and by how
    much.
    """
    ratios, seeds = sorted(ratios), list(seeds)
    if 0 not in ratios:
        raise ValueError("ratios must include 0.0: every degradation is measured against it")
    low, tolerance = gate["low_dose_ratio"], gate["monotone_tolerance"]
    ppl = {(r["type"], r["ratio"], r["seed"]): r["holdout_ppl"] for r in rows}

    verdicts: dict[str, dict[str, Any]] = {}
    for kind in gate["collapse_types"]:
        missing = [Cell(kind, r, s).id for s in seeds for r in ratios if (kind, r, s) not in ppl]
        if missing:
            verdicts[kind] = {"status": "not_evaluable", "missing_cells": missing}
            continue

        mean = {r: sum(ppl[(kind, r, s)] for s in seeds) / len(seeds) for r in ratios}
        steps = []
        for a, b in zip(ratios, ratios[1:]):
            drop = (mean[a] - mean[b]) / mean[a]  # positive means perplexity fell
            steps.append({
                "from_ratio": a, "to_ratio": b, "mean_from": mean[a], "mean_to": mean[b],
                "relative_drop": drop, "ok": drop <= tolerance + _DROP_EPS,
            })
        baseline = {s: ppl[(kind, 0, s)] for s in seeds}
        spread = max(baseline.values()) - min(baseline.values())
        per_seed = [
            {
                "seed": s, "baseline": baseline[s], "low_dose_ppl": ppl[(kind, low, s)],
                "excess": ppl[(kind, low, s)] - baseline[s],
                "ok": ppl[(kind, low, s)] - baseline[s] > spread,
            }
            for s in seeds
        ]
        dose_ok = all(st["ok"] for st in steps)
        noise_ok = all(p["ok"] for p in per_seed)
        failed = [name for name, ok in (("dose_response", dose_ok), ("above_noise", noise_ok)) if not ok]
        verdicts[kind] = {
            "status": "fails" if failed else "degrades",
            "failed_conditions": failed,
            "missing_cells": [],
            "mean_holdout_ppl": [{"ratio": r, "mean": mean[r]} for r in ratios],
            "dose_response": {"ok": dose_ok, "tolerance": tolerance, "steps": steps},
            "above_noise": {"ok": noise_ok, "ratio": low, "spread": spread, "seeds": per_seed},
        }

    required = list(gate["collapse_types"] if types is None else types)
    missing_cells = [
        Cell(kind, r, s).id for s in seeds for r in ratios for kind in required if (kind, r, s) not in ppl
    ]
    statuses = {t["status"] for t in verdicts.values()}
    status = (
        "not_evaluable" if missing_cells or "not_evaluable" in statuses
        else "passed" if statuses == {"degrades"}
        else "failed"
    )
    return {"rule": dict(gate), "status": status, "missing_cells": missing_cells, "types": verdicts}


def benchmark_report(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Change in mean benchmark-item perplexity versus ratio 0, for benchmark_near. Not gated.

    Leakage shows up as memorisation, so a *negative* change is the signal.
    """
    by_ratio: dict[float, list[float]] = {}
    for r in rows:
        if r["type"] == "benchmark_near":
            by_ratio.setdefault(r["ratio"], []).append(r["benchmark_ppl"])
    means = {ratio: sum(v) / len(v) for ratio, v in by_ratio.items()}
    base = means.get(0)
    return [
        {
            "ratio": ratio, "n_seeds": len(by_ratio[ratio]), "benchmark_ppl_mean": means[ratio],
            "delta_vs_ratio0": None if base is None else means[ratio] - base,
            "relative_delta": None if base is None else means[ratio] / base - 1,
        }
        for ratio in sorted(means)
    ]


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    ratios: Sequence[float],
    gate: Mapping[str, Any],
    types: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Collapse curves per (type, ratio), the GATE 2 verdict, and the benchmark_near report.

    Everything here is computed from the result rows, so it can be re-applied to
    finished results without training. ``seeds`` and ``ratios`` are the configured
    ones, not whatever has rows: a run cut short cannot pass GATE 2 on what it
    got through.
    """
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["type"], r["ratio"]), []).append(r)

    curves = []
    for (kind, ratio), rs in sorted(groups.items()):
        ppl = [r["holdout_ppl"] for r in rs]
        bench = [r["benchmark_ppl"] for r in rs]
        curves.append({
            "type": kind, "ratio": ratio, "n_seeds": len(rs),
            "holdout_ppl_mean": sum(ppl) / len(ppl), "holdout_ppl_min": min(ppl),
            "holdout_ppl_max": max(ppl),
            "benchmark_ppl_mean": sum(bench) / len(bench), "benchmark_ppl_min": min(bench),
            "benchmark_ppl_max": max(bench),
        })
    verdict = evaluate_gate(rows, seeds=seeds, ratios=ratios, gate=gate, types=types)
    return {
        "curves": curves,
        "gate": verdict,
        "benchmark_near": benchmark_report(rows),
        "gate_2_status": verdict["status"],
        "gate_2_passed": verdict["status"] == "passed",
    }


def format_gate(summary: Mapping[str, Any]) -> str:
    """The GATE 2 verdict and, for each type, why, for the run log."""
    gate = summary["gate"]
    label = {"passed": "PASSED", "failed": "NOT PASSED", "not_evaluable": "NOT EVALUABLE (grid is partial)"}
    lines = [f"GATE 2 {label[gate['status']]}"]
    if gate["missing_cells"]:
        lines.append(f"  {len(gate['missing_cells'])} configured cell(s) have no result yet")
    for kind, t in gate["types"].items():
        if t["status"] == "not_evaluable":
            lines.append(f"  {kind}: not evaluable, {len(t['missing_cells'])} cell(s) missing")
        elif t["status"] == "degrades":
            lines.append(f"  {kind}: degrades")
        else:
            lines.append(f"  {kind}: FAILS {', '.join(t['failed_conditions'])}")
            for st in t["dose_response"]["steps"]:
                if not st["ok"]:
                    lines.append(
                        f"    dose_response: mean {st['mean_from']:.2f} at ratio {st['from_ratio']:g} -> "
                        f"{st['mean_to']:.2f} at {st['to_ratio']:g} is a {st['relative_drop']:.2%} drop, "
                        f"allowed {t['dose_response']['tolerance']:.2%}"
                    )
            noise = t["above_noise"]
            for p in noise["seeds"]:
                if not p["ok"]:
                    lines.append(
                        f"    above_noise: seed {p['seed']} is {p['excess']:+.2f} over its baseline at "
                        f"ratio {noise['ratio']:g}, needs more than the baseline spread {noise['spread']:.2f}"
                    )
    bench = summary["benchmark_near"]
    if bench:
        lines.append("benchmark_near (reported, not gated): change in mean benchmark perplexity vs ratio 0")
        for b in bench:
            if b["delta_vs_ratio0"] is not None and b["ratio"] != 0:
                lines.append(f"    ratio {b['ratio']:g}: {b['delta_vs_ratio0']:+.2f} ({b['relative_delta']:+.1%})")
        lines.append("    (negative = the model has memorised the leaked items)")
    return "\n".join(lines)


def format_table(curves: Sequence[Mapping[str, Any]], key: str = "holdout_ppl_mean") -> str:
    """Mean perplexity by ratio (rows) and type (columns), for the run log."""
    kinds = [t for t in CONTAMINATION_TYPES if any(c["type"] == t for c in curves)]
    ratios = sorted({c["ratio"] for c in curves})
    cell = {(c["type"], c["ratio"]): c[key] for c in curves}
    lines = ["ratio  " + "".join(f"{t:>16}" for t in kinds)]
    for r in ratios:
        vals = "".join(
            f"{cell[(t, r)]:>16.2f}" if (t, r) in cell else f"{'-':>16}" for t in kinds
        )
        lines.append(f"{r:<7g}" + vals)
    return "\n".join(lines)


# Slots 1-4 of the validated categorical palette; adjacent pairs clear the CVD and
# normal-vision floors. Aqua and yellow are under 3:1 on white, so every line also
# gets its own marker and a direct label rather than relying on colour alone.
_COLORS = {"synthetic": "#2a78d6", "recursive": "#eb6834",
           "paraphrased": "#1baf7a", "benchmark_near": "#eda100"}
_MARKERS = {"synthetic": "o", "recursive": "s", "paraphrased": "^", "benchmark_near": "D"}
_INK, _INK_MUTED = "#0b0b0b", "#52514e"


def _plot(curves: Sequence[Mapping[str, Any]], seeds: int, path: Path) -> None:
    """Mean across seeds as a line, the min-max range across seeds as a band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panels = (
        ("holdout_ppl", "Hidden-holdout perplexity", "higher = more degraded"),
        ("benchmark_ppl", "Benchmark-item perplexity", "lower = items memorised"),
    )
    for ax, (key, title, note) in zip(axes, panels):
        ends = []
        for kind in CONTAMINATION_TYPES:
            pts = sorted((c for c in curves if c["type"] == kind), key=lambda c: c["ratio"])
            if not pts:
                continue
            x = [c["ratio"] for c in pts]
            mean = [c[f"{key}_mean"] for c in pts]
            ax.plot(x, mean, color=_COLORS[kind], marker=_MARKERS[kind], markersize=6,
                    linewidth=2, label=kind)
            ax.fill_between(x, [c[f"{key}_min"] for c in pts], [c[f"{key}_max"] for c in pts],
                            color=_COLORS[kind], alpha=0.15, linewidth=0)
            ends.append([mean[-1], kind])
        # Direct labels at the right edge, nudged apart so close lines stay legible.
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.06
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        for y, kind in ends:
            ax.annotate(kind, (1.0, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=_INK_MUTED, annotation_clip=False)
        ax.set_title(f"{title}\n({note})", fontsize=10, color=_INK)
        ax.set_xlabel("contamination ratio", color=_INK_MUTED)
        ax.set_ylabel("perplexity", color=_INK_MUTED)
        ax.set_xticks(sorted({c["ratio"] for c in curves}))
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(x=0.02)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Collapse curves, {seeds} seed(s): mean line, min-max band",
                 fontsize=9, color=_INK_MUTED)
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Sprint-6 fine-tuning grid.")
    parser.add_argument("--config", default="configs/grid.yaml")
    parser.add_argument("--summarize-only", "--report-only", dest="summarize_only", action="store_true",
                        help="train nothing; rebuild summary.json, the tables and the figure from "
                             "results.jsonl (refused if it was made under different settings)")
    parser.add_argument("--rerun-types", nargs="+", metavar="TYPE",
                        help="drop these types' contaminated rows from results.jsonl (a backup is kept "
                             "beside it) and train just those cells again, after their generator changed")
    args = parser.parse_args(argv)
    if args.rerun_types and args.summarize_only:
        parser.error("--rerun-types trains, so it can not be combined with --summarize-only")

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cells = make_cells(cfg["types"], cfg["ratios"], cfg["seeds"])
    if 0 not in cfg["ratios"]:
        raise ValueError("ratios must include 0.0: every degradation is measured against it")
    if not isinstance(cfg["recursive_depth"], int) or cfg["recursive_depth"] < 2:
        raise ValueError(
            f"recursive_depth must be an integer >= 2, got {cfg['recursive_depth']!r}"
        )
    if "gate_min_ppl_increase" in cfg:
        raise ValueError("gate_min_ppl_increase is replaced by the `gate:` section; see configs/grid.yaml")
    validate_gate(cfg.get("gate"), types=cfg["types"], ratios=cfg["ratios"])
    fingerprint = config_fingerprint(cfg)
    out = Path(cfg["output_dir"])
    results_path = out / "results.jsonl"
    wanted = {c.id for c in cells}
    # The fingerprint leaves the axes out so a finished grid can be extended, which
    # also means the file can hold cells this config no longer asks for. Ignore them.
    if args.rerun_types:
        unknown = [t for t in args.rerun_types if t not in cfg["types"]]
        if unknown:
            raise ValueError(f"--rerun-types {unknown} are not among this config's types {cfg['types']}")
        kept, dropped, backup = drop_types(results_path, fingerprint, args.rerun_types)
        done = {k: v for k, v in kept.items() if k in wanted}
        if backup is not None:
            print(f"Dropped {len(dropped)} result(s) of {args.rerun_types}; the previous file is "
                  f"saved as {backup}", flush=True)
        else:
            print(f"No contaminated results of {args.rerun_types} to drop.", flush=True)
        cells_to_train = [c for c in cells if c.type in args.rerun_types]
    else:
        done = {k: v for k, v in load_results(results_path, fingerprint).items() if k in wanted}
        cells_to_train = cells

    if not args.summarize_only:
        _train_all(cfg, cells_to_train, fingerprint, results_path, done)

    results = load_results(results_path, fingerprint)
    rows = [r for k, r in results.items() if k in wanted]
    if not rows:
        raise SystemExit(f"no results for this config's cells in {results_path}")
    if len(results) > len(rows):
        print(f"Ignoring {len(results) - len(rows)} result(s) for cells outside this config.")
    missing = [c.id for c in cells if c.id not in results]
    if missing:
        print(f"WARNING: {len(missing)} of {len(cells)} cells have no result yet; curves are partial.")

    summary = summarize(
        rows, seeds=cfg["seeds"], ratios=cfg["ratios"], gate=cfg["gate"], types=cfg["types"]
    )
    summary["complete"] = not missing
    summary["missing_cells"] = missing
    stale = stale_cells(rows)
    summary["stale_cells"] = stale
    if stale:
        kinds = sorted({r["type"] for r in rows if r["cell"] in set(stale)})
        print(f"WARNING: {len(stale)} cell(s) were made with an older version of their contamination "
              f"generator, so one curve mixes two generators. Re-run them with "
              f"--rerun-types {' '.join(kinds)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps({"config": cfg, "fingerprint": fingerprint, **summary}, indent=2),
        encoding="utf-8",
    )
    _plot(summary["curves"], len({r["seed"] for r in rows}), Path(cfg["figure"]))
    print("\nMean hidden-holdout perplexity:\n" + format_table(summary["curves"]))
    print("\nMean benchmark-item perplexity:\n" + format_table(summary["curves"], "benchmark_ppl_mean"))
    print("\n" + format_gate(summary))
    return 0


def _train_all(
    cfg: dict[str, Any],
    cells: list[Cell],
    fingerprint: str,
    results_path: Path,
    done: dict[str, dict[str, Any]],
) -> None:
    """Wire the real models into ``run_grid`` and train every cell not yet done."""
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # per-sample generation warnings drown the progress lines
    d, t, g = cfg["data"], cfg["train"], cfg["generate"]
    lo, hi = d["min_words"], d["max_words"]
    kept = load_dataset(d["kept"])
    holdout = load_holdout(d)
    # Benchmark items are the leaked material; they come from a split that is not
    # the hidden holdout, which mix() would refuse to train on.
    bench_items = _pick(load_dataset(d["benchmark"]), d["n_benchmark"], lo, hi, random.Random("benchmark"))
    bench_texts = [s.response for s in bench_items]
    models_dir = Path(cfg["models_dir"])
    pools_dir = Path(cfg["pools_dir"]) / fingerprint

    @lru_cache(maxsize=None)
    def draw(seed: int) -> tuple[list[Sample], list[Sample]]:
        rng = random.Random(seed)
        human = _pick(kept, d["n_train"], lo, hi, rng)  # the spike's Gen-0 draw for this seed
        used = {s.response for s in human}
        source = _pick([s for s in kept if s.response not in used], d["n_train"], lo, hi, rng)
        return human, source

    def release(model_dir: Path) -> None:
        # ponytail: 60 merged DistilGPT-2 checkpoints are ~20 GB. Keep them
        # (keep_models) only if the white-box signals need them later.
        if not cfg["keep_models"]:
            shutil.rmtree(model_dir, ignore_errors=True)

    def pool_for(kind: str, seed: int) -> list[Sample]:
        path = pool_path(pools_dir, kind, seed)
        if path.exists():
            return load_dataset(path)
        print(f"  generating {kind} contamination for seed {seed}", flush=True)

        def generator(model: str):  # fresh per use, so each pool is reproducible on its own
            return hf_generator(
                model, max_new_tokens=g["max_new_tokens"],
                batch_size=g["batch_size"], seed=seed, device=cfg["device"],
            )

        def fine_tune(texts: list[str], gen: int) -> Path:
            # the spike's recipe and seeds: from base_model, seed * 100 + generation
            print(f"  recursive gen {gen}: fine-tuning on {len(texts)} texts", flush=True)
            return train_lora(
                texts, models_dir / f"pool_recursive_s{seed}_gen{gen}",
                base_model=cfg["base_model"], seed=seed * 100 + gen,
                device=cfg["device"], **t,
            )

        if kind == "benchmark_near":
            pool = make_near_duplicates(
                bench_items, count=d["n_train"],
                exact_fraction=cfg["benchmark_near"]["exact_fraction"], seed=seed,
            )
        elif kind == "recursive":
            pool = make_recursive(
                pool_for("synthetic", seed), depth=cfg["recursive_depth"], train=fine_tune,
                generator_for=lambda ckpt: generator(str(ckpt)), release=release,
            )
        elif kind == "synthetic":
            pool = make_synthetic(draw(seed)[1], generator(cfg["generator_model"]))
        else:
            pool = make_paraphrased(
                draw(seed)[1], resample_empty(generator(cfg["generator_model"]))
            )
        write_jsonl(pool, path)
        return pool

    def train(texts: list[str], cell: Cell) -> Path:
        print(f"{cell.id}: training on {len(texts)} texts", flush=True)
        return train_lora(
            texts, models_dir / cell.id, base_model=cfg["base_model"],
            seed=cell.seed * 100,  # the spike's Gen-0 seed, so the ratio-0 cells reproduce it
            device=cfg["device"], **t,
        )

    def evaluate(model_dir: Path) -> dict[str, float]:
        scores = {
            name: perplexity(model_dir, texts, max_len=t["max_len"], batch_size=t["batch_size"],
                             device=cfg["device"])
            for name, texts in (("holdout_ppl", holdout), ("benchmark_ppl", bench_texts))
        }
        print(f"    holdout {scores['holdout_ppl']:.2f}  benchmark {scores['benchmark_ppl']:.2f}",
              flush=True)
        return scores

    def track(cell: Cell) -> AbstractContextManager[Run]:
        params = {
            "type": cell.type, "ratio": cell.ratio, "seed": cell.seed,
            "base_model": cfg["base_model"], "generator_model": cfg["generator_model"],
            "n_train": d["n_train"], "n_holdout": d["n_holdout"], "n_benchmark": d["n_benchmark"],
            "share_baseline": cfg["share_baseline"], "recursive_depth": cfg["recursive_depth"],
            "pool_version": pool_tag(cell.type) if cell.ratio > 0 else "none", **t,
        }
        return track_run(cfg.get("tracking"), cell.id, params,
                         {"kind": "grid", "type": cell.type, "config": fingerprint})

    results_path.parent.mkdir(parents=True, exist_ok=True)

    def on_result(row: dict[str, Any]) -> None:
        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    todo = [c for c in cells if c.id not in done]
    print(f"{len(cells)} cells, {len(cells) - len(todo)} already done, {len(todo)} to run", flush=True)
    run_grid(
        cells, human_for=lambda seed: draw(seed)[0], pool_for=pool_for, train=train,
        evaluate=evaluate, release=release, track=track, on_result=on_result, done=done,
        share_baseline=cfg["share_baseline"], fingerprint=fingerprint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
