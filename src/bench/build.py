"""Build ORIGIN-T-Bench: every grid cell's training set as a file, with a manifest.

    python -m src.bench.build --config configs/bench.yaml
    python -m src.bench.build --config configs/bench.yaml --datasets wikitext2

For each dataset in the config this checks the normalized corpus and its hidden holdout
exist and do not leak into each other, rebuilds every cell of that dataset's grid config
(type x ratio x seed) exactly as the grid trained on it, checks each against the whole
holdout, and writes

    bench/<dataset>/<type>/r<ratio>/s<seed>.jsonl
    bench/manifest.json

The cells come from ``grid.training_set``, so they are the grid's own draw and its cached
pools. Pools are not reproducible across machines (HANDBOOK §12.3), so nothing is
generated here: a missing pool stops the build with a pointer to copy it from the machine
that ran the grid. The one exception is ``duplicate``, which the grid does not train on:
it is made from the seed's clean half by :func:`~src.generation.duplicate.make_duplicates`,
which is pure stdlib and deterministic, so it is the same on every machine.

The normalized corpora and holdouts are deterministic: when they are missing, the error
gives the commands that regenerate them (HANDBOOK §12.1). Files are written with LF line
endings whatever the OS, so the manifest's hashes match across machines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from ..finetune.grid import (
    Cell, config_fingerprint, draw_human, make_cells, mixture, pool_tag, training_set,
)
from ..generation.duplicate import make_duplicates
from ..ingestion.check_leakage import find_leaks
from ..ingestion.loaders import load_dataset
from ..ingestion.schema import Sample

SCHEMA = "bench-manifest/1"
# Bench-only types: built here (deterministically) rather than read from a grid pool.
EXTRA_TYPES = ("duplicate",)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_lf(samples: Sequence[Sample], path: Path) -> None:
    """``write_jsonl`` with LF endings on every OS (``write_jsonl`` gets CRLF on Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def cell_path(out: Path, dataset: str, cell: Cell) -> Path:
    return out / dataset / cell.type / f"r{cell.ratio:g}" / f"s{cell.seed}.jsonl"


def cell_samples(
    grid: Mapping[str, Any], cell: Cell, kept: Sequence[Sample], extra: Mapping[str, Any]
) -> list[Sample]:
    """The training set of one bench cell: the grid's, or ``duplicate`` built from the clean half."""
    if cell.type != "duplicate":
        return training_set(grid, cell, kept=kept)
    human = draw_human(kept, grid["data"], cell.seed)[0]
    pool = make_duplicates(
        human, count=len(human), exact_fraction=extra["duplicate"]["exact_fraction"], seed=cell.seed,
    ) if cell.ratio > 0 else []
    return mixture(human, pool, cell)


def _require(path: str | Path, how: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Regenerate it (HANDBOOK §12.1):\n  {how}")
    return path


def build_dataset(entry: Mapping[str, Any], cfg: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """Write one dataset's cells and return its manifest entry. Refuses on any holdout leak."""
    import yaml

    name = entry["name"]
    grid = yaml.safe_load(Path(entry["grid_config"]).read_text(encoding="utf-8"))
    d = grid["data"]
    kept_path = _require(d["kept"], "python -m src.ingestion.normalize_corpora --config configs/corpora.yaml")
    holdout_path = _require(
        d["holdout"], f"python -m src.ingestion.holdout --input {str(d['kept']).replace('_kept', '')}"
    )
    kept, holdout = load_dataset(kept_path), load_dataset(holdout_path)
    leak = find_leaks(holdout, kept)
    if leak:
        raise ValueError(f"{name}: the kept corpus leaks the hidden holdout.\n{leak.summary()}")

    extra = cfg.get("extra_types") or {}
    unknown = sorted(set(extra) - set(EXTRA_TYPES))
    if unknown:
        raise ValueError(f"unknown extra_types {unknown}; use {list(EXTRA_TYPES)}")
    cells = make_cells(grid["types"], grid["ratios"], grid["seeds"])
    cells += [Cell(t, r, s) for t in extra for s in grid["seeds"] for r in grid["ratios"]]

    # Every cell is built and checked before any is written, so a missing pool or a leak
    # leaves no half-built dataset behind.
    built = [(cell, cell_samples(grid, cell, kept, extra)) for cell in cells]
    for cell, samples in built:
        leak = find_leaks(holdout, samples)
        if leak:
            raise ValueError(f"{name} {cell.id}: the training set leaks the hidden holdout.\n{leak.summary()}")

    files = []
    for cell, samples in built:
        path = cell_path(out, name, cell)
        write_lf(samples, path)
        files.append({
            "path": path.relative_to(out).as_posix(),
            "type": cell.type, "ratio": cell.ratio, "seed": cell.seed,
            "n": len(samples), "n_contaminated": sum(s.source != "human" for s in samples),
            "sha256": sha256(path),
        })
        print(f"  {name} {cell.id}: {len(samples)} rows", flush=True)

    types = [*grid["types"], *extra]
    return {
        "grid_config": Path(entry["grid_config"]).as_posix(),
        "config_fingerprint": config_fingerprint(grid),
        "source": entry["source"],
        "licence": entry["licence"],
        "generator_model": grid["generator_model"],
        "generator_versions": {t: pool_tag(t) for t in types if t not in EXTRA_TYPES and t != "human_control"},
        "inputs": {
            "kept": {"path": kept_path.as_posix(), "n": len(kept), "sha256": sha256(kept_path)},
            "hidden_holdout": {"path": holdout_path.as_posix(), "n": len(holdout), "sha256": sha256(holdout_path)},
        },
        "files": files,
    }


def build(cfg: Mapping[str, Any], only: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Build every dataset (or only ``only``) and write ``manifest.json``. Returns the manifest.

    A partial build keeps the other datasets' entries of an existing manifest.
    """
    out = Path(cfg["output_dir"])
    names = [e["name"] for e in cfg["datasets"]]
    missing = sorted(set(only or ()) - set(names))
    if missing:
        raise ValueError(f"--datasets {missing} are not in the config ({names})")
    manifest_path = out / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if only and manifest_path.exists() else {}
    datasets = dict(old.get("datasets", {}))
    for entry in cfg["datasets"]:
        if not only or entry["name"] in only:
            datasets[entry["name"]] = build_dataset(entry, cfg, out)
    manifest = {"schema": SCHEMA, "version": str(cfg["version"]), "datasets": datasets}
    out.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build ORIGIN-T-Bench from the grid's pools.")
    parser.add_argument("--config", default="configs/bench.yaml")
    parser.add_argument("--datasets", nargs="+", metavar="NAME", help="build only these datasets")
    args = parser.parse_args(argv)

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = build(cfg, args.datasets)
    n = sum(len(d["files"]) for d in manifest["datasets"].values())
    print(f"ORIGIN-T-Bench v{manifest['version']}: {n} files, manifest at {Path(cfg['output_dir']) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
