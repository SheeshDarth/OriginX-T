"""Generate duplicate contamination: repeated copies of the training set's own human rows.

Real scraped corpora are full of the same document seen twice: mirrored pages,
re-posts, boilerplate that survived dedup with a changed space or full stop. A
fine-tune on such data spends its steps re-learning rows it has already seen, so the
effective dataset shrinks and the model over-fits to the repeated rows. This is the
``duplicate`` source in the schema, and unlike ``benchmark_near`` it copies **human
training rows**, not benchmark items: the damage is lost diversity, not leaked answers.

``exact_fraction`` of the copies are verbatim (what exact-hash dedup catches); the rest
are near-exact, made with the same mild surface edits as ``benchmark_near``
(:func:`~src.generation.benchmark_near.perturb`), which hash dedup misses. The rate in a
dataset is set the usual way, by :func:`~src.generation.mixer.mix`'s ``ratio``.

Pass the rows the mixture's human side is drawn from, so the copies duplicate rows the
model can also see in their original form. The hidden holdout is refused here as well as
in ``mix``: a copy of a holdout row would leak it into training under a new id.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

from ..ingestion.schema import Sample
from .benchmark_near import perturb


def make_duplicates(
    human: Sequence[Sample],
    *,
    count: Optional[int] = None,
    exact_fraction: float = 0.5,
    seed: int = 0,
) -> list[Sample]:
    """Copy ``count`` human rows, ``exact_fraction`` verbatim and the rest surface-perturbed.

    Source rows are picked in a seeded random order without replacement, cycling only
    when ``count`` exceeds ``len(human)``, so every row is copied about equally often.
    Each copy is ``source="duplicate"``, ``generation=0`` (copied, not generated), keeps
    the original's label, and gets a new id (``<id>-dup<i>``) so it is never mistaken for
    the row it copies. Deterministic for a given ``seed``.
    """
    if not 0.0 <= exact_fraction <= 1.0:
        raise ValueError(f"exact_fraction must be in [0, 1], got {exact_fraction}")
    if not human:
        raise ValueError("human must not be empty")
    count = len(human) if count is None else count
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    leaked = [s.sample_id for s in human if s.split == "hidden_holdout"]
    if leaked:
        raise ValueError(f"{len(leaked)} hidden_holdout sample(s) passed to make_duplicates(), e.g. {leaked[:3]}")
    not_human = sorted({s.source for s in human} - {"human"})
    if not_human:
        raise ValueError(f"make_duplicates() copies human rows only, got sources {not_human}")

    rng = random.Random(seed)
    order = list(range(len(human)))
    rng.shuffle(order)
    n_exact = round(count * exact_fraction)

    out: list[Sample] = []
    for i in range(count):
        item = human[order[i % len(order)]]
        verbatim = i < n_exact
        out.append(
            Sample(
                sample_id=f"{item.sample_id}-dup{i}",
                prompt=item.prompt if verbatim or not item.prompt else perturb(item.prompt, rng),
                response=item.response if verbatim or not item.response else perturb(item.response, rng),
                label=item.label,
                source="duplicate",
                generation=0,
                split=item.split,
            ).validate()
        )
    return out
