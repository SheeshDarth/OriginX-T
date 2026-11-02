"""Tests for duplicate contamination generation (HANDBOOK §9.1, B4)."""

from collections import Counter

import pytest

from src.generation import make_duplicates, mix, similarity
from src.ingestion.schema import Sample


def human(n=10, split="train"):
    return [
        Sample(sample_id=f"h-{i}", response=f"Row {i} says the river rose, then it fell. It was quiet.",
               label=str(i), split=split)
        for i in range(n)
    ]


def _origin(copy):
    return copy.sample_id.rsplit("-dup", 1)[0]


def test_marks_provenance_and_new_ids_and_keeps_labels():
    src = human()
    out = make_duplicates(src)
    assert len(out) == 10
    assert all(s.source == "duplicate" and s.generation == 0 for s in out)
    assert not {s.sample_id for s in src} & {s.sample_id for s in out}, "must not reuse ids"
    by_id = {s.sample_id: s for s in src}
    assert all(s.label == by_id[_origin(s)].label for s in out)


def test_exact_fraction_gives_verbatim_copies_and_the_rest_are_near_exact():
    src = human(10)
    by_id = {s.sample_id: s for s in src}
    out = make_duplicates(src, count=10, exact_fraction=0.3, seed=1)
    assert all(s.response == by_id[_origin(s)].response for s in out[:3])
    assert all(similarity(s.response, by_id[_origin(s)].response) > 0.8 for s in out[3:])
    assert any(s.response != by_id[_origin(s)].response for s in out[3:]), "near-exact copies differ"
    assert all(s.response == by_id[_origin(s)].response for s in make_duplicates(src, exact_fraction=1.0))


def test_copies_spread_evenly_and_cycle_only_past_the_pool():
    src = human(4)
    once = make_duplicates(src, count=4, exact_fraction=1.0, seed=2)
    assert sorted(map(_origin, once)) == sorted(s.sample_id for s in src), "no row twice before all once"
    counts = Counter(map(_origin, make_duplicates(src, count=10, seed=2)))
    assert set(counts.values()) <= {2, 3}


def test_deterministic_per_seed():
    a = [s.to_dict() for s in make_duplicates(human(), seed=5)]
    assert a == [s.to_dict() for s in make_duplicates(human(), seed=5)]
    assert a != [s.to_dict() for s in make_duplicates(human(), seed=6)]
    picks = {tuple(map(_origin, make_duplicates(human(), count=3, exact_fraction=1.0, seed=s))) for s in range(5)}
    assert len(picks) > 1, "which rows get copied depends on the seed, not on input order"


def test_refuses_the_hidden_holdout_and_non_human_rows():
    with pytest.raises(ValueError, match="hidden_holdout"):
        make_duplicates(human(3, split="hidden_holdout"))
    synthetic = [Sample(sample_id="s-0", response="generated text", source="synthetic")]
    with pytest.raises(ValueError, match="human rows only"):
        make_duplicates(human(3) + synthetic)


@pytest.mark.parametrize("kwargs", [{"exact_fraction": 1.5}, {"count": -1}])
def test_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        make_duplicates(human(), **kwargs)
    with pytest.raises(ValueError, match="empty"):
        make_duplicates([])


def test_mixes_at_a_controlled_rate():
    src = human(20)
    out = mix(src, make_duplicates(src, seed=0), ratio=0.25, seed=0)
    assert len(out) == 20
    assert Counter(s.source for s in out) == {"human": 15, "duplicate": 5}
