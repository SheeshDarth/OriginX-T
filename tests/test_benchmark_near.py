"""Tests for benchmark-near contamination generation."""

import random

import pytest

from src.generation import make_near_duplicates, mix, perturb, similarity
from src.ingestion.schema import Sample


def benchmark(n=10):
    return [
        Sample(
            sample_id=f"bench-{i}",
            prompt=f"What is {i} plus one?",
            response=f"The answer is {i + 1}, clearly.",
            label=str(i + 1),
            split="test",
        )
        for i in range(n)
    ]


def test_marks_provenance_and_new_ids():
    out = make_near_duplicates(benchmark())
    assert len(out) == 10
    assert all(s.source == "benchmark_near" for s in out)
    assert all(s.generation == 0 for s in out)  # copied, not model-generated
    originals = {s.sample_id for s in benchmark()}
    assert not originals & {s.sample_id for s in out}, "must not reuse benchmark ids"


def test_exact_fraction_produces_verbatim_copies():
    out = make_near_duplicates(benchmark(10), exact_fraction=0.3)
    verbatim = [s for s in out if s.benchmark_near_score == 1.0]
    assert len(verbatim) >= 3


def test_perturbed_items_differ_from_source_but_stay_similar():
    src = benchmark(10)
    out = make_near_duplicates(src, exact_fraction=0.0, seed=1)
    responses = {s.response for s in src}
    changed = [s for s in out if s.response not in responses]
    assert changed, "expected perturbed text to differ from the original"
    # still recognisably the same item -- that is what makes it *near*
    assert all(s.benchmark_near_score > 0.3 for s in changed)


def test_score_is_measured_not_assumed():
    out = make_near_duplicates(benchmark(20), exact_fraction=0.5, seed=3)
    scores = [s.benchmark_near_score for s in out]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert 1.0 in scores  # exact copies
    # measured, not hardcoded: changing content lowers the score...
    assert similarity("The answer is 5.", "The answer is 6.") < 1.0
    # ...while reformatting alone does not, because it is still a full leak
    assert similarity("The Answer Is 5.", "the  answer is 5.") == 1.0


def test_deterministic_for_a_given_seed():
    a = [s.response for s in make_near_duplicates(benchmark(), seed=5)]
    b = [s.response for s in make_near_duplicates(benchmark(), seed=5)]
    assert a == b


def test_count_can_exceed_the_benchmark_pool():
    out = make_near_duplicates(benchmark(4), count=10)
    assert len(out) == 10
    assert len({s.sample_id for s in out}) == 10  # ids stay unique


def test_perturbation_preserves_content_not_formatting():
    rng = random.Random(0)
    text = "The Quick Brown Fox. Jumps!"
    variants = {perturb(text, rng) for _ in range(20)}
    assert any(v != text for v in variants)
    assert all(v.strip() for v in variants)  # never produces empty text


def test_similarity_bounds():
    assert similarity("abc", "abc") == 1.0
    assert similarity("abc", "xyz") < 0.5


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        make_near_duplicates([])
    with pytest.raises(ValueError):
        make_near_duplicates(benchmark(), exact_fraction=1.5)


def test_composes_with_the_mixer():
    """The end-to-end shape: leak benchmark items into clean data at a ratio."""
    clean = [Sample(sample_id=f"c-{i}", response=f"clean text {i}") for i in range(100)]
    leaked = make_near_duplicates(benchmark(10), count=50)
    out = mix(clean, leaked, ratio=0.25)
    assert len(out) == 100
    assert sum(1 for s in out if s.source == "benchmark_near") == 25
