"""f_leak_* features: best match of each training text against the benchmark items."""

import pytest

from src.features.leak import leak
from src.generation.benchmark_near import similarity

BENCH = ["the quick brown fox jumps over the lazy dog", "a completely different sentence about trains"]


def test_exact_and_reformatted_copies_count_as_leaks_and_unrelated_text_does_not():
    texts = [
        "the quick brown fox jumps over the lazy dog",  # verbatim: 1.0
        "The  Quick brown fox jumps over the lazy dog",  # recased + respaced: still 1.0
        "nothing alike here at all, zebra xylophone",
        "nothing alike here at all, quartz vortex",
    ]
    f = leak(texts, {"benchmark": BENCH})
    best = [max(similarity(t, b) for b in BENCH) for t in texts]
    assert best[:2] == [1.0, 1.0] and max(best[2:]) < 0.9
    assert f["max_sim"] == 1.0
    assert f["mean_sim"] == pytest.approx(sum(best) / 4)
    assert f["frac_above_0_9"] == 0.5


def test_each_text_takes_its_best_benchmark_match():
    f = leak(["a completely different sentence about trains"], {"benchmark": BENCH})
    assert f["mean_sim"] == 1.0  # matched the second item, not the first


def test_a_near_copy_between_0_9_and_1_counts_as_a_leak():
    near = "the quick brown fox jumps over the lazy cat"
    assert 0.9 < similarity(near, BENCH[0]) < 1.0
    assert leak([near, "zebra"], {"benchmark": BENCH})["frac_above_0_9"] == 0.5


def test_no_benchmark_or_no_texts_is_none():
    assert set(leak(["x"], {}).values()) == {None}
    assert set(leak([], {"benchmark": BENCH}).values()) == {None}
