"""Benchmark-leak features of a training set (HANDBOOK §8.1): the ``f_leak_*`` family.

Each training text is scored by its best match among the benchmark items, with
``src.generation.benchmark_near.similarity`` (case- and whitespace-blind, so the
generator's ground truth and this detector measure the same thing).

- ``max_sim``: the highest such score in the set.
- ``mean_sim``: the mean over texts of each text's best score.
- ``frac_above_0_9``: share of texts whose best score exceeds 0.9 (near or exact copies).

``ctx["benchmark"]`` holds the benchmark item texts. Without them, every value is ``None``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..generation.benchmark_near import similarity

VERSION = 1
THRESHOLD = 0.9


def best_matches(texts: list[str], bench: list[str]) -> list[float]:
    """Each text's highest similarity to any benchmark item."""
    # ponytail: all pairs, ~0.13 s per text for 200 items (~2 min a cell); shard over processes if that hurts
    return [max(similarity(t, b) for b in bench) for t in texts]


def leak(texts: list[str], ctx: Mapping[str, Any] | None = None) -> dict[str, Optional[float]]:
    bench = list((ctx or {}).get("benchmark") or [])
    if not texts or not bench:
        return {"max_sim": None, "mean_sim": None, "frac_above_0_9": None}
    best = best_matches(texts, bench)
    return {
        "max_sim": max(best),
        "mean_sim": sum(best) / len(best),
        "frac_above_0_9": sum(s > THRESHOLD for s in best) / len(best),
    }
