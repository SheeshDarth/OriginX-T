"""Lexical features of a training set (HANDBOOK §8.1): the ``f_lex_*`` family.

Hand-rolled with the standard library so every number can be derived on paper.
Text is lowercased and split on whitespace; n-grams never cross a text boundary.

- ``entropy``: Shannon entropy (bits) of the pooled unigram distribution.
  Model text reuses fewer words, so it falls.
- ``ttr``: type-token ratio, distinct words / words, pooled. Comparable across
  cells only because every cell has the same number of samples.
- ``distinct_1/2/3``: distinct n-grams / n-grams, pooled.
- ``repeat_3gram``: mean, per text, of the share of 3-gram occurrences that
  repeat an earlier 3-gram of the same text. Catches a model looping inside one
  text, which pooled ``distinct_3`` mixes up with texts repeating each other.
- ``burstiness``: variance / mean of the gaps (in tokens) between repeats of the
  same word within a text, pooled over all repeated words. 1 is Poisson-like;
  larger means words come in clumps.
- ``len_mean``, ``len_var``: words per text (population variance).

A value that can't be computed (no tokens, no repeats) is ``None``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Optional

VERSION = 1


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _mean_var(xs: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    return m, sum((x - m) ** 2 for x in xs) / len(xs)


def lex(texts: list[str], ctx: Mapping[str, Any] | None = None) -> dict[str, Optional[float]]:
    docs = [t.lower().split() for t in texts]
    words = [w for d in docs for w in d]
    out: dict[str, Optional[float]] = {}

    counts = Counter(words)
    n = len(words)
    out["entropy"] = -sum(c / n * math.log2(c / n) for c in counts.values()) if n else None
    out["ttr"] = len(counts) / n if n else None

    for k in (1, 2, 3):
        grams = [g for d in docs for g in _ngrams(d, k)]
        out[f"distinct_{k}"] = len(set(grams)) / len(grams) if grams else None

    rates = []
    for d in docs:
        grams = _ngrams(d, 3)
        if grams:
            rates.append(1 - len(set(grams)) / len(grams))
    out["repeat_3gram"] = sum(rates) / len(rates) if rates else None

    gaps: list[float] = []
    for d in docs:
        last: dict[str, int] = {}
        for i, w in enumerate(d):
            if w in last:
                gaps.append(i - last[w])
            last[w] = i
    m, v = _mean_var(gaps)
    out["burstiness"] = v / m if m else None

    out["len_mean"], out["len_var"] = _mean_var([float(len(d)) for d in docs])
    return out
