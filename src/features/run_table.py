"""One row per grid run: what was done to it, how it scored, and its white-box signals.

This is the table the feature suite and the risk model start from. It joins
``reports/grid/results.jsonl`` (outcomes) with ``reports/grid/whitebox.jsonl``
(representational signals, ``src.evaluation.whitebox``) on the cell id, and
nothing else: it does not decide what counts as "collapsed". ``ppl_excess`` is
the plain relative rise over the same type's and seed's ratio-0 run, left for
whoever defines the label.

A white-box row is used only if it was measured on the grid run now in the file
(its ``grid_holdout_ppl`` matches); one measured on a run that was since redone
is reported as ``stale`` and carries no ``wb_*`` values rather than wrong ones.
Standard library only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GRID_COLUMNS = ("holdout_ppl", "benchmark_ppl", "distinct_1", "distinct_2")


def _read(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_run_table(grid_path: str | Path, whitebox_path: str | Path) -> list[dict[str, Any]]:
    """Grid runs in file order, each with ``wb_status``: fresh, stale or missing."""
    grid = _read(grid_path)
    wb = {r["cell"]: r for r in _read(whitebox_path) if r.get("kind") in ("cell", "shared_baseline")}
    baseline = {(r["type"], r["seed"]): r["holdout_ppl"] for r in grid if r["ratio"] == 0}
    table = []
    for g in grid:
        row: dict[str, Any] = {k: g[k] for k in ("cell", "type", "ratio", "seed")}
        row.update({k: g[k] for k in GRID_COLUMNS})
        base = baseline.get((g["type"], g["seed"]))
        row["baseline_ppl"] = base
        row["ppl_excess"] = g["holdout_ppl"] / base - 1.0 if base else None
        w = wb.get(g["cell"])
        if w is None:
            row["wb_status"] = "missing"
        elif w.get("grid_holdout_ppl") != g["holdout_ppl"]:
            row["wb_status"] = "stale"
        else:
            row["wb_status"] = "fresh"
            row.update({f"wb_{k}": v for k, v in w["headline"].items()})
        table.append(row)
    return table


def write_csv(table: list[dict[str, Any]], path: str | Path) -> None:
    """Write the table as CSV. Columns are the union over rows (stale and missing rows have fewer)."""
    import csv

    columns: list[str] = []
    for row in table:
        columns += [k for k in row if k not in columns]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(table)


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--grid", default="reports/grid/results.jsonl")
    p.add_argument("--whitebox", default="reports/grid/whitebox.jsonl")
    p.add_argument("--out", default="reports/grid/run_table.csv")
    args = p.parse_args(argv)
    table = build_run_table(args.grid, args.whitebox)
    write_csv(table, args.out)
    counts = {s: sum(1 for r in table if r["wb_status"] == s) for s in ("fresh", "stale", "missing")}
    print(f"{len(table)} runs -> {args.out}  (white-box: {counts})")


if __name__ == "__main__":
    main()
