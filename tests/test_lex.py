"""f_lex_* features against values worked out by hand."""

import math

import pytest

from src.features.lex import lex


def test_a_a_b_by_hand():
    f = lex(["a a b"])
    # p(a) = 2/3, p(b) = 1/3: H = -(2/3 log2 2/3 + 1/3 log2 1/3) = log2 3 - 2/3
    assert f["entropy"] == pytest.approx(math.log2(3) - 2 / 3)  # 0.9183
    assert f["ttr"] == pytest.approx(2 / 3)
    assert f["distinct_1"] == pytest.approx(2 / 3)
    assert f["distinct_2"] == 1.0  # (a a), (a b)
    assert f["distinct_3"] == 1.0  # (a a b)
    assert f["repeat_3gram"] == 0.0
    assert f["burstiness"] == 0.0  # one gap (a at 0 and 1): mean 1, variance 0
    assert (f["len_mean"], f["len_var"]) == (3.0, 0.0)


def test_looping_text_and_boundaries():
    f = lex(["x y z x y z", "A b"])  # case folded; n-grams do not cross texts
    # text 1 3-grams: xyz yzx zxy xyz -> one repeat of 4; text 2 has none, so it is skipped
    assert f["repeat_3gram"] == pytest.approx(1 / 4)
    assert f["distinct_2"] == pytest.approx(4 / 6)  # xy yz zx xy yz | ab: 4 distinct of 6
    assert f["len_mean"] == 4.0 and f["len_var"] == 4.0  # lengths 6 and 2


def test_burstiness_is_variance_over_mean_of_gaps():
    # a at 0, 1, 5 -> gaps 1, 4: mean 2.5, var 2.25 -> 0.9
    assert lex(["a a b c d a"])["burstiness"] == pytest.approx(0.9)


def test_nothing_to_measure_is_none():
    f = lex(["", "   "])
    assert all(f[k] is None for k in ("entropy", "ttr", "distinct_1", "repeat_3gram", "burstiness"))
    assert lex(["a b"])["burstiness"] is None  # no repeated word
