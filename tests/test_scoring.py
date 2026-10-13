"""Tests for src/evaluation/scoring.py. Stdlib only, so CI runs all of them."""

import itertools
import math

import pytest

from src.evaluation.scoring import auprc, auroc, bootstrap_ci, f1_at


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


def _mean(values):
    return sum(values) / len(values)


def test_bootstrap_is_seeded_and_the_point_estimate_is_the_plain_metric():
    labels = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    scores = [0.9, 0.3, 0.6, 0.65, 0.2, 0.1, 0.4, 0.8, 0.5, 0.7]
    a = bootstrap_ci(auroc, labels, scores, n_boot=200, seed=3)
    assert a == bootstrap_ci(auroc, labels, scores, n_boot=200, seed=3)
    assert a != bootstrap_ci(auroc, labels, scores, n_boot=200, seed=4)
    assert a[0] == auroc(labels, scores)
    assert 0.0 <= a[1] <= a[2] <= 1.0


def test_bootstrap_interval_of_a_mean_contains_it_and_narrows_with_more_data():
    small = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    point, lo, hi = bootstrap_ci(_mean, small, n_boot=500, seed=0)
    assert point == 3.5 and lo < point < hi
    big = small * 20
    _, lo2, hi2 = bootstrap_ci(_mean, big, n_boot=500, seed=0)
    assert (hi2 - lo2) < (hi - lo)


def test_bootstrap_interval_of_a_constant_collapses_to_it():
    assert bootstrap_ci(_mean, [2.0] * 8, n_boot=50) == (2.0, 2.0, 2.0)


def test_bootstrap_alpha_widens_the_interval():
    data = [float(i % 7) for i in range(30)]
    _, lo90, hi90 = bootstrap_ci(_mean, data, n_boot=400, alpha=0.10, seed=1)
    _, lo99, hi99 = bootstrap_ci(_mean, data, n_boot=400, alpha=0.01, seed=1)
    assert lo99 <= lo90 and hi99 >= hi90 and (hi99 - lo99) > (hi90 - lo90)


def test_bootstrap_redraws_resamples_the_metric_refuses_and_gives_up_when_most_fail():
    # one positive in 4: about a third of resamples lose it and are redrawn
    point, lo, hi = bootstrap_ci(auroc, [1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3], n_boot=100, seed=0)
    assert point == lo == hi == 1.0
    # a metric that scores the data once and then refuses every resample
    calls = []

    def flaky(values):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("refused")
        return 0.0

    with pytest.raises(ValueError, match="could not be scored"):
        bootstrap_ci(flaky, [1.0, 2.0, 3.0], n_boot=20)


def test_bootstrap_rejects_bad_arguments():
    with pytest.raises(ValueError, match="differ in length"):
        bootstrap_ci(auroc, [0, 1], [0.1])
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci(_mean, [1.0, 2.0], n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(_mean, [1.0, 2.0], alpha=1.0)
