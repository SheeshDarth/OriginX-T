"""Scoring metrics for the detectors and the risk model (TRD section 9).

Plain Python, no numpy, so they run anywhere the tests do. Every function takes
``labels`` (0 = negative, 1 = positive) and ``scores`` (higher = more likely
positive) as parallel sequences, and raises ``ValueError`` on input it cannot
score honestly (mismatched lengths, labels outside {0, 1}, NaN scores, or only
one class present) instead of returning a number that looks fine.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Collection, Sequence


def _check(labels: Sequence[int], scores: Sequence[float], both_classes: bool = True) -> None:
    if len(labels) != len(scores):
        raise ValueError(f"labels ({len(labels)}) and scores ({len(scores)}) differ in length")
    if not labels:
        raise ValueError("no examples")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("labels must be 0 or 1")
    if any(math.isnan(s) for s in scores):
        raise ValueError("scores contain NaN")
    if both_classes and (sum(labels) == 0 or sum(labels) == len(labels)):
        raise ValueError("need at least one positive and one negative example")


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve: P(random positive scores above a random negative).

    Rank-sum form with mid-ranks, so tied scores count half (a constant score
    gives exactly 0.5).
    """
    _check(labels, scores)
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # mid-rank, 1-based
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision: sum over thresholds of (recall gain) x (precision there).

    Examples with the same score move the threshold together, so the result
    does not depend on the order ties arrive in.
    """
    _check(labels, scores)
    n_pos = sum(labels)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = seen = 0
    prev_recall = 0.0
    total = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            tp += labels[order[k]]
        seen = j + 1
        recall = tp / n_pos
        total += (recall - prev_recall) * (tp / seen)
        prev_recall = recall
        i = j + 1
    return total


def f1_at(labels: Sequence[int], scores: Sequence[float], threshold: float) -> float:
    """F1 when ``score >= threshold`` is called positive. 0.0 when nothing is right."""
    _check(labels, scores, both_classes=False)
    pred = [s >= threshold for s in scores]
    tp = sum(1 for p, y in zip(pred, labels) if p and y == 1)
    fp = sum(1 for p, y in zip(pred, labels) if p and y == 0)
    fn = sum(1 for p, y in zip(pred, labels) if not p and y == 1)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def bootstrap_ci(
    metric: Callable[..., float],
    *columns: Sequence,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper): a percentile bootstrap interval for ``metric``.

    ``columns`` are parallel sequences (for example labels and scores); examples
    are resampled together with replacement and ``metric(*resampled)`` is
    recomputed. A resample the metric refuses (``ValueError``, such as one that
    lost every positive) is redrawn; if more than half the draws are refused the
    interval is not trustworthy and this raises. Seeded, so it is reproducible.
    """
    n = len(columns[0])
    if any(len(c) != n for c in columns):
        raise ValueError("columns differ in length")
    if n_boot < 1 or not 0 < alpha < 1:
        raise ValueError("need n_boot >= 1 and 0 < alpha < 1")
    point = metric(*columns)
    rng = random.Random(seed)
    stats: list[float] = []
    refused = 0
    while len(stats) < n_boot:
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            stats.append(metric(*[[c[i] for i in idx] for c in columns]))
        except ValueError:
            refused += 1
            if refused > n_boot:
                raise ValueError("over half the bootstrap resamples could not be scored") from None
    stats.sort()
    lo = stats[min(n_boot - 1, int(math.floor(alpha / 2 * n_boot)))]
    hi = stats[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return point, lo, hi


def reliability_bins(
    labels: Sequence[int], probs: Sequence[float], n_bins: int = 10
) -> list[dict]:
    """Equal-width reliability table over [0, 1]; empty bins are left out.

    Each row: ``lo``, ``hi``, ``n``, ``confidence`` (mean predicted probability)
    and ``accuracy`` (observed positive rate). A probability of exactly 1.0 goes
    in the last bin. This is what a reliability diagram plots.
    """
    _check(labels, probs, both_classes=False)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if any(not 0.0 <= p <= 1.0 for p in probs):
        raise ValueError("probabilities must be in [0, 1]")
    members: list[list[int]] = [[] for _ in range(n_bins)]
    for i, p in enumerate(probs):
        members[min(int(p * n_bins), n_bins - 1)].append(i)
    return [
        {
            "lo": b / n_bins,
            "hi": (b + 1) / n_bins,
            "n": len(ids),
            "confidence": sum(probs[i] for i in ids) / len(ids),
            "accuracy": sum(labels[i] for i in ids) / len(ids),
        }
        for b, ids in enumerate(members)
        if ids
    ]


def ece(labels: Sequence[int], probs: Sequence[float], n_bins: int = 10) -> float:
    """Expected calibration error: bin-size-weighted |accuracy - confidence|."""
    bins = reliability_bins(labels, probs, n_bins)
    total = sum(b["n"] for b in bins)
    return sum(b["n"] / total * abs(b["accuracy"] - b["confidence"]) for b in bins)


def interval_coverage(
    truths: Sequence[float], intervals: Sequence[tuple[float, float]]
) -> dict:
    """Share of truths inside their (lo, hi) interval, ends included, and the mean width."""
    if len(truths) != len(intervals):
        raise ValueError(f"truths ({len(truths)}) and intervals ({len(intervals)}) differ in length")
    if not truths:
        raise ValueError("no examples")
    if any(lo > hi or math.isnan(lo) or math.isnan(hi) for lo, hi in intervals):
        raise ValueError("every interval needs lo <= hi")
    hits = sum(1 for y, (lo, hi) in zip(truths, intervals) if lo <= y <= hi)
    return {
        "coverage": hits / len(truths),
        "mean_width": sum(hi - lo for lo, hi in intervals) / len(intervals),
    }


def set_coverage(truths: Sequence, sets: Sequence[Collection]) -> dict:
    """Share of truths inside their prediction set, and the mean set size.

    An empty set never covers, and is counted in the size, so a method cannot
    look efficient by abstaining.
    """
    if len(truths) != len(sets):
        raise ValueError(f"truths ({len(truths)}) and sets ({len(sets)}) differ in length")
    if not truths:
        raise ValueError("no examples")
    hits = sum(1 for y, s in zip(truths, sets) if y in s)
    return {"coverage": hits / len(truths), "mean_size": sum(len(s) for s in sets) / len(sets)}


def coverage_gap(coverage: float, alpha: float) -> float:
    """Observed coverage minus the nominal 1 - alpha. Negative means under-covering."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return coverage - (1.0 - alpha)


def lead_stage_at_far(
    risks: Sequence[Sequence[float]],
    onsets: Sequence[int | None],
    far: float = 0.05,
) -> dict:
    """Early-warning lead time at a fixed false-alarm rate.

    ``risks[i]`` is run *i*'s risk score at each stage; ``onsets[i]`` is the stage
    at which it collapsed, or ``None`` for a run that never does. An alarm is the
    first stage whose risk is strictly above a threshold.

    The threshold is set on the runs that never collapse alone: the lowest value
    at which at most ``far`` of them ever raise an alarm. Then, for each run that
    does collapse, the lead is ``onset - first alarm stage``; an alarm at or after
    the onset is a miss, not a lead of zero. Returns the threshold, the false
    alarm rate actually reached (ties can leave it below ``far``), the detection
    rate, each detected run's lead, and their mean and median.
    """
    if len(risks) != len(onsets):
        raise ValueError(f"risks ({len(risks)}) and onsets ({len(onsets)}) differ in length")
    if not 0.0 <= far < 1.0:
        raise ValueError("far must be in [0, 1)")
    for r, o in zip(risks, onsets):
        if not r:
            raise ValueError("every run needs at least one stage")
        if any(math.isnan(x) for x in r):
            raise ValueError("risk contains NaN")
        if o is not None and not 0 <= o < len(r):
            raise ValueError(f"onset {o} is outside the run's {len(r)} stages")
    neg_peaks = sorted((max(r) for r, o in zip(risks, onsets) if o is None), reverse=True)
    positives = [(r, o) for r, o in zip(risks, onsets) if o is not None]
    if not neg_peaks or not positives:
        raise ValueError("need at least one run that collapses and one that never does")

    allowed = int(math.floor(far * len(neg_peaks) + 1e-9))
    threshold = neg_peaks[allowed] if allowed < len(neg_peaks) else -math.inf
    false_alarms = sum(1 for p in neg_peaks if p > threshold)

    leads = []
    for r, o in positives:
        first = next((k for k, x in enumerate(r) if x > threshold), None)
        if first is not None and first < o:
            leads.append(o - first)
    mid = sorted(leads)
    median = None
    if mid:
        median = mid[len(mid) // 2] if len(mid) % 2 else (mid[len(mid) // 2 - 1] + mid[len(mid) // 2]) / 2
    return {
        "threshold": threshold,
        "false_alarm_rate": false_alarms / len(neg_peaks),
        "n_collapsing": len(positives),
        "detected": len(leads),
        "detection_rate": len(leads) / len(positives),
        "leads": leads,
        "mean_lead": sum(leads) / len(leads) if leads else None,
        "median_lead": median,
    }
