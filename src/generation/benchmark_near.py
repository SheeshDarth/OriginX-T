"""Generate benchmark-near contamination — leaked eval items in disguise.

Benchmark leakage rarely looks like an exact copy. An item gets reformatted,
recased, or rewrapped somewhere in a data pipeline and lands in training as a
near-duplicate that exact-match and n-gram overlap checks miss, while the model
still memorises the answer. That is the failure ORIGIN-T has to detect, so the
generator has to produce it.

This is the one contamination type that needs no model: the perturbations are
surface-level and meaning-preserving, so they run on CPU and stay deterministic.
Genuine paraphrase (same meaning, different words) needs a generation model and
belongs with the synthetic generators.

``benchmark_near_score`` is *measured* per sample with ``difflib``, not assumed
— an exact copy scores 1.0, a reformatted one lower, and the detector is then
scored against a real number rather than a label we invented.
"""

from __future__ import annotations

import difflib
import random
import re
from typing import Optional, Sequence

from ..ingestion.schema import Sample

_PUNCT_SWAPS = {".": " .", ",": " ,", "!": ".", "?": " ?", ";": ",", "'": "", '"': ""}


def _recase(text: str, rng: random.Random) -> str:
    return text.lower() if rng.random() < 0.5 else text.upper()


def _respace(text: str, rng: random.Random) -> str:
    """Collapse or pad whitespace — the classic pipeline reformat."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed.replace(" ", "  ") if rng.random() < 0.5 else collapsed


def _repunct(text: str, rng: random.Random) -> str:
    for old, new in _PUNCT_SWAPS.items():
        if old in text and rng.random() < 0.5:
            text = text.replace(old, new)
    return text


_PERTURBATIONS = (_recase, _respace, _repunct)


def perturb(text: str, rng: random.Random) -> str:
    """Apply one or two surface edits. Meaning is preserved; the string is not."""
    for op in rng.sample(_PERTURBATIONS, rng.randint(1, 2)):
        text = op(text, rng)
    return text.strip() or text


def similarity(a: str, b: str) -> float:
    """Content similarity in [0, 1], ignoring case and whitespace (1.0 == same content).

    Deliberately blind to formatting: an ALL-CAPS or re-wrapped copy of a
    benchmark item is a *complete* leak — the model still memorises the answer —
    so it must score 1.0. Comparing raw characters would score it near zero and
    hand the detector wrong ground truth.
    """
    fold = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731
    return difflib.SequenceMatcher(None, fold(a), fold(b)).ratio()


def make_near_duplicates(
    benchmark: Sequence[Sample],
    *,
    count: Optional[int] = None,
    exact_fraction: float = 0.3,
    seed: int = 0,
) -> list[Sample]:
    """Turn benchmark items into leaked-looking training samples.

    ``exact_fraction`` of them are verbatim copies (the easy case a detector
    must not miss); the rest are surface-perturbed (the hard case). Each result
    is marked ``source="benchmark_near"`` and carries its measured similarity to
    the item it came from.

    Ids are suffixed rather than reused so a near-duplicate is never mistaken
    for the original benchmark row.
    """
    if not 0.0 <= exact_fraction <= 1.0:
        raise ValueError(f"exact_fraction must be in [0, 1], got {exact_fraction}")
    if not benchmark:
        raise ValueError("benchmark must not be empty")

    count = len(benchmark) if count is None else count
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")

    rng = random.Random(seed)
    n_exact = round(count * exact_fraction)

    out: list[Sample] = []
    for i in range(count):
        # ponytail: cycle the pool when count exceeds it, rather than sampling
        # with replacement -- keeps coverage even, matters at small benchmark sizes.
        item = benchmark[i % len(benchmark)]
        verbatim = i < n_exact
        response = item.response if verbatim else perturb(item.response, rng)
        prompt = item.prompt if verbatim else (perturb(item.prompt, rng) if item.prompt else "")

        out.append(
            Sample(
                sample_id=f"{item.sample_id}-near{i}",
                prompt=prompt,
                response=response,
                label=item.label,
                source="benchmark_near",
                generation=0,  # copied, not model-generated
                benchmark_near_score=similarity(item.response, response),
                split=item.split,
            ).validate()
        )
    return out
