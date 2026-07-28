"""Carve the trust-critical hidden holdout out of a normalized corpus.

The hidden holdout is the set ORIGIN-T measures degradation against, so it must
never enter any training mixture. Assignment is by a **stable hash of the
sample_id**, not a shuffle: the same sample always lands on the same side of the
split, even if the corpus grows or rows are reordered later. That means a
holdout built today stays valid when new data arrives — a reshuffle would
silently leak previously-held-out samples into training.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Iterable

from .schema import Sample

HOLDOUT_SPLIT = "hidden_holdout"


def _bucket(sample_id: str, salt: str) -> float:
    """Map a sample_id to a stable float in [0, 1).

    Uses blake2b rather than :func:`hash` because Python's built-in hash is
    randomized per process (PYTHONHASHSEED) and would give a different split on
    every run.
    """
    digest = hashlib.blake2b(f"{salt}:{sample_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def is_holdout(sample_id: str, *, fraction: float, salt: str = "origin-t") -> bool:
    """True if this id belongs in the hidden holdout at the given fraction."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return _bucket(sample_id, salt) < fraction


def carve_holdout(
    samples: Iterable[Sample], *, fraction: float = 0.1, salt: str = "origin-t"
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into ``(kept, holdout)``.

    Holdout samples get ``split="hidden_holdout"``; kept samples are returned
    unchanged. Deterministic for a given ``fraction`` and ``salt``.
    """
    kept: list[Sample] = []
    holdout: list[Sample] = []
    for sample in samples:
        if is_holdout(sample.sample_id, fraction=fraction, salt=salt):
            holdout.append(replace(sample, split=HOLDOUT_SPLIT).validate())
        else:
            kept.append(sample)
    return kept, holdout
