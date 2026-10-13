"""Tests for src/evaluation/scoring.py. Stdlib only, so CI runs all of them."""

import itertools
import math

import pytest

from src.evaluation.scoring import auprc, auroc, f1_at


def test_auroc_perfect_inverted_and_constant():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auroc_equals_the_pairwise_definition_with_ties_counted_half():
    labels = [1, 0, 1, 0, 0, 1, 0, 1]
    scores = [0.9, 0.9, 0.4, 0.4, 0.1, 0.7, 0.7, 0.2]
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    pairs = [(p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg)]
    assert auroc(labels, scores) == pytest.approx(sum(pairs) / len(pairs))


def test_auroc_ignores_the_order_of_the_examples_and_monotone_rescaling():
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.8, 0.3, 0.6, 0.65, 0.2, 0.1]
    base = auroc(labels, scores)
    order = [3, 0, 5, 1, 4, 2]
    assert auroc([labels[i] for i in order], [scores[i] for i in order]) == base
    assert auroc(labels, [math.exp(10 * s) for s in scores]) == pytest.approx(base)


def test_auprc_hand_computed():
    # ranked by score: 1, 0, 1, 0  -> precision 1/1 at recall 1/2, 2/3 at recall 1
    assert auprc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)
    assert auprc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auprc_of_a_constant_score_is_the_positive_rate_whatever_the_order():
    # all tied: one threshold, precision = prevalence, recall jumps 0 -> 1
    assert auprc([1, 0, 0, 0], [0.5] * 4) == pytest.approx(0.25)
    assert auprc([0, 0, 0, 1], [0.5] * 4) == pytest.approx(0.25)


def test_f1_at_threshold_is_inclusive_and_zero_without_a_hit():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.5, 0.5, 0.1]
    assert f1_at(labels, scores, 0.5) == pytest.approx(2 * 2 / (2 * 2 + 1 + 0))  # the 0.5 negative is flagged
    assert f1_at(labels, scores, 0.6) == pytest.approx(2 / (2 + 0 + 1))
    assert f1_at(labels, scores, 2.0) == 0.0
    assert f1_at([0, 0], [0.1, 0.2], 0.0) == 0.0  # no positives at all: defined, not an error


@pytest.mark.parametrize("fn", [auroc, auprc])
def test_ranking_metrics_refuse_input_they_cannot_score(fn):
    with pytest.raises(ValueError, match="differ in length"):
        fn([0, 1], [0.1])
    with pytest.raises(ValueError, match="no examples"):
        fn([], [])
    with pytest.raises(ValueError, match="0 or 1"):
        fn([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match="NaN"):
        fn([0, 1], [0.1, float("nan")])
    with pytest.raises(ValueError, match="positive and one negative"):
        fn([1, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="positive and one negative"):
        fn([0, 0], [0.1, 0.2])
