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


_CACHE: dict[tuple[int, str], float] = {}  # (benchmark set, text) -> best similarity, for this process


def _best(text: str, bench: tuple[str, ...]) -> float:
    return max(similarity(text, b) for b in bench)


def best_matches(texts: list[str], bench: list[str]) -> list[float]:
    """Each text's highest similarity to any benchmark item.

    All pairs, about 0.35 s per real WikiText text against 200 items. Two things keep the
    grid affordable: every cell of a seed shares its 1000 human texts, so each distinct
    text is scored once (cached), and new texts are spread over CPU cores.
    """
    items = tuple(bench)
    key = hash(items)
    todo = sorted({t for t in texts if (key, t) not in _CACHE})
    if len(todo) > 100:  # a pool costs ~1 s to start; small calls (tests, one audit row) stay in-process
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        with ProcessPoolExecutor() as pool:
            scores = list(pool.map(partial(_best, bench=items), todo, chunksize=16))
    else:
        scores = [_best(t, items) for t in todo]
    _CACHE.update(((key, t), s) for t, s in zip(todo, scores))
    return [_CACHE[(key, t)] for t in texts]


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
