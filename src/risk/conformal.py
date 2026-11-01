"""Split conformal prediction for the degrade / no-degrade label (HANDBOOK §8.4), hand-rolled.

1. Hold out a calibration set the model was not fitted on.
2. Score each calibration example by how strange its true label looks to the
   model: nonconformity ``s = 1 - p(true class)``.
3. ``q`` = the ``k``-th smallest score, ``k = ceil((n + 1)(1 - alpha))``. The
   ``+1`` counts the test point itself among the n + 1 exchangeable scores. If
   ``k > n`` there are too few calibration points for that alpha, and ``q`` is
   infinite (every set is {0, 1}).
4. The set for a new example is every label ``y`` with ``1 - p(y) <= q``.

Guarantee: if calibration and test examples are exchangeable, the set holds the
true label with probability at least ``1 - alpha``. Leave-one-type-out breaks
exchangeability on purpose (the test type is new), so the measured coverage is
the honest number to report, not the promise.
"""

from __future__ import annotations

import math
from typing import Sequence


def nonconformity(p_degrade: float, label: int) -> float:
    return 1 - (p_degrade if label == 1 else 1 - p_degrade)


def quantile(scores: Sequence[float], alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    return math.inf if k > len(scores) else sorted(scores)[k - 1]


def prediction_set(p_degrade: float, q: float) -> list[int]:
    return [y for y in (0, 1) if nonconformity(p_degrade, y) <= q]
