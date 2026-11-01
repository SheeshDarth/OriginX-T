"""Split conformal by hand: the (n+1) quantile, the sets, and coverage on exchangeable data."""

import math
import random

from src.risk.conformal import nonconformity, prediction_set, quantile


def test_nonconformity_is_one_minus_p_of_the_true_class():
    assert math.isclose(nonconformity(0.8, 1), 0.2)
    assert math.isclose(nonconformity(0.8, 0), 0.8)


def test_quantile_uses_the_n_plus_1_correction():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # n = 9
    assert quantile(scores, 0.2) == 0.8  # k = ceil(10 * 0.8) = 8 -> 8th smallest
    assert quantile(scores, 0.5) == 0.5  # k = ceil(10 * 0.5) = 5
    assert quantile(scores, 0.05) == math.inf  # k = ceil(9.5) = 10 > n: too few points


def test_sets_hold_every_label_that_is_not_too_strange():
    assert prediction_set(0.9, 0.2) == [1]  # 1 - 0.9 = 0.1 <= 0.2; 1 - 0.1 = 0.9 > 0.2
    assert prediction_set(0.1, 0.2) == [0]
    assert prediction_set(0.5, 0.6) == [0, 1]
    assert prediction_set(0.5, 0.4) == []  # the model is unsure and q is tight: abstain
    assert prediction_set(0.5, math.inf) == [0, 1]
    assert prediction_set(0.75, 0.25) == [1]  # a score equal to q is in: "<= q", exact in binary


def test_coverage_is_at_least_1_minus_alpha_on_exchangeable_data():
    rng = random.Random(0)

    def draw():
        y = rng.random() < 0.5
        p = min(max(rng.gauss(0.7 if y else 0.3, 0.2), 0.0), 1.0)  # a mediocre, noisy model
        return p, int(y)

    alpha, hits, trials = 0.1, 0, 2000
    for _ in range(trials):
        cal = [draw() for _ in range(30)]
        q = quantile([nonconformity(p, y) for p, y in cal], alpha)
        p, y = draw()
        hits += y in prediction_set(p, q)
    assert hits / trials >= 1 - alpha - 0.02  # 0.9 promised; 2000 draws leave ~0.007 of noise
