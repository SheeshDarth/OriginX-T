"""Tests for contamination mixing."""

import pytest

from src.generation import mix, ratio_sweep
from src.ingestion.schema import Sample


def human(n=100):
    return [Sample(sample_id=f"h-{i}", response=f"human {i}") for i in range(n)]


def synthetic(n=100):
    return [
        Sample(sample_id=f"s-{i}", response=f"synth {i}", source="synthetic", generation=1)
        for i in range(n)
    ]


def counts(samples):
    return sum(1 for s in samples if s.source == "synthetic")


def test_ratio_controls_contaminated_share():
    out = mix(human(), synthetic(), ratio=0.25)
    assert len(out) == 100
    assert counts(out) == 25


def test_size_is_held_fixed_across_ratios():
    """A ratio sweep must not confound contamination with dataset size."""
    sizes = {len(v) for v in ratio_sweep(human(), synthetic()).values()}
    assert sizes == {100}


def test_ratio_extremes():
    assert counts(mix(human(), synthetic(), ratio=0.0)) == 0
    assert counts(mix(human(), synthetic(), ratio=1.0)) == 100


def test_every_sample_records_the_mixture():
    out = mix(human(), synthetic(), ratio=0.5)
    assert {s.contamination_ratio for s in out} == {0.5}


def test_deterministic_for_a_given_seed():
    a = [s.sample_id for s in mix(human(), synthetic(), ratio=0.5, seed=7)]
    b = [s.sample_id for s in mix(human(), synthetic(), ratio=0.5, seed=7)]
    assert a == b


def test_seed_changes_the_draw():
    a = [s.sample_id for s in mix(human(), synthetic(), ratio=0.5, seed=1)]
    b = [s.sample_id for s in mix(human(), synthetic(), ratio=0.5, seed=2)]
    assert a != b


def test_sweep_covers_the_gradient():
    sweep = ratio_sweep(human(), synthetic())
    assert sorted(sweep) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert [counts(sweep[r]) for r in sorted(sweep)] == [0, 25, 50, 75, 100]


def test_rejects_impossible_requests():
    with pytest.raises(ValueError):
        mix(human(), synthetic(), ratio=1.5)
    with pytest.raises(ValueError):  # not enough contaminated samples
        mix(human(100), synthetic(10), ratio=0.5)
    with pytest.raises(ValueError):  # not enough human samples
        mix(human(10), synthetic(100), ratio=0.1, total=100)


def test_originals_are_not_mutated():
    pool = human()
    mix(pool, synthetic(), ratio=0.5)
    assert all(s.contamination_ratio == 0.0 for s in pool)


def test_rejects_hidden_holdout_samples():
    held = [Sample(sample_id="h0", response="secret", split="hidden_holdout")]
    with pytest.raises(ValueError, match="hidden_holdout"):
        mix(human(4), held + synthetic(3), ratio=0.25)
    with pytest.raises(ValueError, match="hidden_holdout"):
        mix(held + human(4), synthetic(4), ratio=0.0)
