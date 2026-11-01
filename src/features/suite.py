"""Dataset features for every grid cell (HANDBOOK §8.1): one ``features/1`` row per cell (§7.2).

    python -m src.features.suite --config configs/features.yaml --dataset wikitext2

Each cell's training set is rebuilt exactly as the grid trained on it
(``src.finetune.grid.training_set``, from the cached pools; nothing is
regenerated), turned into the strings the model saw (``training_text``), and
passed to every registered family.

**Family signature.** ``family(texts: list[str], ctx) -> dict[str, float | None]``.
``ctx`` is a dict shared by all families: ``benchmark`` (the grid's benchmark item
texts), ``dataset``, ``cell``, ``type``, ``ratio``, ``seed``, ``reference_texts`` (the
seed's ratio-0 training texts, the clean human draw that ``sem`` compares against),
and from the config ``device`` and ``max_len`` (the grid's ``train.max_len``). The
suite prefixes each key as ``f_<family>_<key>``. A family module also has an int
``VERSION``; ``feature_version`` records them (``lex@1+leak@1``). ``model`` and ``sem``
are Track C's (§10.7); they load their models lazily, on first use.

**Resumable.** Rows already in the output with the current ``feature_version`` are
kept and their cells skipped; the file is rewritten atomically after every cell.
Ratio-0 cells of one seed share their training set, so it is measured once.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from . import leak, lex, model, semantic

Family = Callable[[list[str], Mapping[str, Any]], Mapping[str, Optional[float]]]
FAMILIES: dict[str, tuple[Family, int]] = {
    "lex": (lex.lex, lex.VERSION),
    "leak": (leak.leak, leak.VERSION),
    "model": (model.model, model.VERSION),
    "sem": (semantic.semantic, semantic.VERSION),
}


def feature_version(families: Mapping[str, tuple[Family, int]]) -> str:
    return "+".join(f"{name}@{v}" for name, (_, v) in families.items())


def features_for(texts: list[str], families: Mapping[str, tuple[Family, int]], ctx: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name, (family, _) in families.items():
        for k, v in family(texts, ctx).items():
            row[f"f_{name}_{k}"] = None if v is None else float(v)
    return row


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    tmp.replace(path)


def run_suite(
    dataset: str,
    cells: Sequence[Any],
    texts_for: Callable[[Any], list[str]],
    out: str | Path,
    *,
    families: Mapping[str, tuple[Family, int]] = FAMILIES,
    ctx: Optional[Mapping[str, Any]] = None,
    on_row: Callable[[dict[str, Any]], None] = lambda row: None,
) -> list[dict[str, Any]]:
    """Measure every cell not already in ``out`` at this ``feature_version``; return all rows."""
    out = Path(out)
    version = feature_version(families)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()] if out.exists() else []
    rows = [r for r in rows if r.get("feature_version") == version]
    done = {r["cell"] for r in rows}
    baseline: dict[int, tuple[int, dict[str, Any]]] = {}  # seed -> (n_samples, features) at ratio 0
    references: dict[int, list[str]] = {}  # seed -> its ratio-0 texts (the same for every type)

    def reference(cell: Any) -> list[str]:
        if cell.seed not in references:
            references[cell.seed] = texts_for(replace(cell, ratio=0.0))
        return references[cell.seed]

    for cell in cells:
        if cell.id in done:
            continue
        if cell.ratio == 0 and cell.seed in baseline:
            n, feats = baseline[cell.seed]
        else:
            texts = texts_for(cell)
            if cell.ratio == 0:
                references.setdefault(cell.seed, texts)
            cell_ctx = {**(ctx or {}), "dataset": dataset, "cell": cell.id, "type": cell.type,
                        "ratio": cell.ratio, "seed": cell.seed, "reference_texts": reference(cell)}
            n, feats = len(texts), features_for(texts, families, cell_ctx)
            if cell.ratio == 0:
                baseline[cell.seed] = (n, feats)
        row = {"schema": "features/1", "dataset": dataset, "cell": cell.id, "type": cell.type,
               "ratio": cell.ratio, "seed": cell.seed, "n_samples": n, **feats, "feature_version": version}
        rows.append(row)
        _write(out, rows)
        on_row(row)
    return rows


def _device(setting: str) -> str:
    """``auto`` is CUDA when torch can see a GPU, else CPU; anything else is used as given."""
    if setting != "auto":
        return setting
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: list[str] | None = None) -> None:
    import argparse

    import yaml

    from ..finetune.collapse import _pick, training_text
    from ..finetune.grid import make_cells, training_set
    from ..ingestion.loaders import load_dataset

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/features.yaml")
    p.add_argument("--dataset", default="wikitext2")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ds = cfg["datasets"][args.dataset]
    grid = yaml.safe_load(Path(ds["grid"]).read_text(encoding="utf-8"))
    d = grid["data"]
    kept = load_dataset(d["kept"])
    # the same benchmark items the grid leaked and scored (grid._train_all)
    bench = _pick(load_dataset(d["benchmark"]), d["n_benchmark"], d["min_words"], d["max_words"],
                  random.Random("benchmark"))
    families = {name: FAMILIES[name] for name in cfg["families"]}
    cells = make_cells(grid["types"], grid["ratios"], grid["seeds"])
    out = Path(cfg["output_dir"]) / f"{args.dataset}.jsonl"
    rows = run_suite(
        args.dataset, cells, lambda c: [training_text(s) for s in training_set(grid, c, kept=kept)], out,
        families=families,
        ctx={"benchmark": [s.response for s in bench], "device": _device(cfg.get("device", "auto")),
             "max_len": grid["train"]["max_len"]},
        on_row=lambda r: print(f"  {r['cell']}: {r['n_samples']} samples", flush=True),
    )
    print(f"{len(rows)} cells -> {out}")


if __name__ == "__main__":
    main()
