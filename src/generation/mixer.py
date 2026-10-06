"""Mix human and contaminated samples at a controlled ratio.

Every contamination type (synthetic, recursive, paraphrased, benchmark-near)
ends up here: generating the bad text is type-specific, but blending it into a
dataset at 0/25/50/75/100% is the same operation every time. Keeping it
model-free means the mixing logic is deterministic and testable without a GPU.

The output is the unit of the fine-tuning grid — one mixed dataset per (type,
ratio, seed) cell.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Optional, Sequence

from ..ingestion.schema import Sample


def mix(
    human: Sequence[Sample],
    contaminated: Sequence[Sample],
    *,
    ratio: float,
    seed: int = 0,
    total: Optional[int] = None,
) -> list[Sample]:
    """Build a dataset that is ``ratio`` contaminated by sample count.

    ``ratio`` is the share of the output drawn from ``contaminated`` (0.0 = all
    human, 1.0 = all contaminated). ``total`` defaults to ``len(human)`` so a
    ratio sweep holds dataset size fixed — otherwise a ratio change would
    confound contamination with training-set size.

    Every returned sample carries ``contamination_ratio=ratio`` so the mixture
    is recoverable from the data alone. Deterministic for a given ``seed``.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")

    total = len(human) if total is None else total
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")

    n_contaminated = round(total * ratio)
    n_human = total - n_contaminated

    if n_human > len(human):
        raise ValueError(
            f"need {n_human} human samples for ratio {ratio} at total {total}, have {len(human)}"
        )
    if n_contaminated > len(contaminated):
        raise ValueError(
            f"need {n_contaminated} contaminated samples for ratio {ratio} at total {total}, "
            f"have {len(contaminated)}"
        )

    # Every generator feeds this function, so guarding here keeps the hidden
    # holdout (and anything derived from it) out of every training mixture.
    leaked = [s.sample_id for s in (*human, *contaminated) if s.split == "hidden_holdout"]
    if leaked:
        raise ValueError(
            f"{len(leaked)} hidden_holdout sample(s) passed to mix(), e.g. {leaked[:3]}"
        )

    rng = random.Random(seed)
    # ponytail: sample without replacement, no stratification by source/length.
    # Add stratified selection if a mixture turns out to be skewed by cluster.
    picked = rng.sample(list(human), n_human) + rng.sample(list(contaminated), n_contaminated)
    rng.shuffle(picked)

    return [replace(s, contamination_ratio=ratio).validate() for s in picked]


def ratio_sweep(
    human: Sequence[Sample],
    contaminated: Sequence[Sample],
    *,
    ratios: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    seed: int = 0,
    total: Optional[int] = None,
) -> dict[float, list[Sample]]:
    """Build one mixed dataset per ratio — the contamination gradient.

    Each ratio uses the same ``seed`` so the variants differ by contamination
    level, not by which samples happened to be drawn.
    """
    return {
        r: mix(human, contaminated, ratio=r, seed=seed, total=total) for r in ratios
    }
