"""Scoring metrics for the detectors and the risk model (TRD section 9).

Plain Python, no numpy, so they run anywhere the tests do. Every function takes
``labels`` (0 = negative, 1 = positive) and ``scores`` (higher = more likely
positive) as parallel sequences, and raises ``ValueError`` on input it cannot
score honestly (mismatched lengths, labels outside {0, 1}, NaN scores, or only
one class present) instead of returning a number that looks fine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


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
