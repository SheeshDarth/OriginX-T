"""Early-warning evaluation on the grid: can a signal see collapse coming? (GATE 3)

A *trajectory* is one contamination type at one seed, followed over its ratios
0, 0.25, ... 1 (the stages). Built from ``src.features.run_table`` rows. Pure
Python, so the tests run everywhere.

**Collapse onset.** The first stage whose hidden-holdout perplexity exceeds that
seed's own ratio-0 baseline by more than the spread (max - min) of the ratio-0
baselines across seeds: the same noise floor as GATE 2's low-dose condition, so
there is no new threshold to choose. Stage 0 is the baseline and never collapses.
A trajectory that never crosses it has onset ``None``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

Row = Mapping[str, Any]
Key = tuple[str, int]  # (contamination type, seed)


def trajectories(table: Sequence[Row]) -> dict[Key, list[Row]]:
    """Rows grouped by (type, seed), each ordered by ratio."""
    out: dict[Key, list[Row]] = defaultdict(list)
    for row in table:
        out[(row["type"], row["seed"])].append(row)
    return {k: sorted(v, key=lambda r: r["ratio"]) for k, v in sorted(out.items())}


def noise_spread(table: Sequence[Row]) -> dict[str, float]:
    """Per type: max - min of the ratio-0 hidden-holdout perplexity across seeds."""
    base: dict[str, list[float]] = defaultdict(list)
    for row in table:
        if row["ratio"] == 0:
            base[row["type"]].append(row["holdout_ppl"])
    return {t: max(v) - min(v) for t, v in base.items()}


def collapse_onsets(table: Sequence[Row]) -> dict[Key, int | None]:
    """Stage index of the first collapse per trajectory, or None if it never collapses."""
    spread = noise_spread(table)
    onsets: dict[Key, int | None] = {}
    for key, rows in trajectories(table).items():
        if rows[0]["ratio"] != 0:
            raise ValueError(f"{key} has no ratio-0 run to measure collapse against")
        base = rows[0]["holdout_ppl"]
        onsets[key] = next(
            (i for i, r in enumerate(rows) if r["holdout_ppl"] - base > spread[key[0]]), None
        )
    return onsets
