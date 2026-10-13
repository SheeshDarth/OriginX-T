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


def risk_trajectories(
    table: Sequence[Row], signal: str, direction: int
) -> tuple[list[Key], list[list[float]], list[int | None], list[Key]]:
    """Per-stage risk of each trajectory from one white-box signal.

    Risk at stage k is ``direction * (x_k - x_0)``: how far the signal has moved,
    in the direction that means "worse" (+1 if it rises as the model degrades, -1
    if it falls), from that trajectory's own ratio-0 value. It uses nothing from
    later stages. Returns (keys, risks, onsets, skipped): a trajectory with any
    stage lacking a fresh value for the signal is skipped, not patched.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    onsets = collapse_onsets(table)
    col = f"wb_{signal}"
    keys, risks, ons, skipped = [], [], [], []
    for key, rows in trajectories(table).items():
        values = [r.get(col) for r in rows]
        if any(v is None for v in values) or any(r.get("wb_status") != "fresh" for r in rows):
            skipped.append(key)
            continue
        keys.append(key)
        risks.append([direction * (v - values[0]) for v in values])
        ons.append(onsets[key])
    return keys, risks, ons, skipped


def evaluate_risk(
    risks: Sequence[Sequence[float]],
    onsets: Sequence[int | None],
    far: float,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict:
    """How well one risk signal warns of collapse, with a random baseline.

    Run-level: each trajectory's score is its highest risk *before* its onset (or
    anywhere, if it never collapses), so a collapsing run cannot score from the
    collapse itself; AUROC of that score against "does it collapse", with a
    bootstrap interval and a permutation p-value (the share of shuffled labelings
    that do at least as well; 1 / (n_perm + 1) at best). Early warning: the lead
    stage at false-alarm rate ``far``. With no collapsing or no never-collapsing
    trajectory there is nothing to separate, and the result says so instead of
    returning a number.
    """
    from random import Random

    from .scoring import auroc, bootstrap_ci, lead_stage_at_far

    labels = [0 if o is None else 1 for o in onsets]
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    if n_pos == 0 or n_neg == 0:
        which = "no collapsing trajectory" if n_pos == 0 else "no trajectory that never collapses"
        return {"status": "not_evaluable", "reason": which, "n_collapsing": n_pos, "n_never": n_neg}
    peaks = [max(r[: max(o, 1)] if o is not None else r) for r, o in zip(risks, onsets)]
    point, lo, hi = bootstrap_ci(auroc, labels, peaks, n_boot=n_boot, seed=seed)
    rng = Random(seed)
    shuffled = list(labels)
    at_least = 0
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        at_least += auroc(shuffled, peaks) >= point
    return {
        "status": "ok",
        "n_collapsing": n_pos,
        "n_never": n_neg,
        "auroc": point,
        "auroc_ci": [lo, hi],
        "permutation_p": (at_least + 1) / (n_perm + 1),
        "early_warning": lead_stage_at_far(risks, onsets, far),
    }
