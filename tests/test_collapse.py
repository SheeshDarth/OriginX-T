"""Tests for the collapse spike loop and metrics, using stub models (no torch)."""

from src.evaluation.metrics import distinct_n
from src.finetune.collapse import gate_passes, run_generations
from src.ingestion.schema import Sample


def test_distinct_n():
    assert distinct_n(["a b a b"], 1) == 0.5
    assert distinct_n(["a b c", "a b"], 2) == 2 / 3
    assert distinct_n(["one"], 2) == 0.0  # no bigrams


def test_each_generation_trains_on_the_previous_ones_output():
    human = [Sample(sample_id=f"h{i}", response=f"word{i} " * 6) for i in range(3)]
    seen = []

    def train(texts, g):
        seen.append(texts)
        return g  # the "model" is just its generation number

    rows = run_generations(
        human,
        generations=2,
        train=train,
        generator_for=lambda g: lambda texts: [f"<m{g}>"] * len(texts),
        evaluate=lambda g: {"holdout_ppl": 10.0 + g},
    )

    assert [r["generation"] for r in rows] == [0, 1, 2]
    assert [r["holdout_ppl"] for r in rows] == [10.0, 11.0, 12.0]
    assert not any("<m" in t for t in seen[0])  # Gen-0 is pure human
    assert all("<m0>" in t for t in seen[1])  # Gen-1 learns from Gen-0's model
    assert all("<m1>" in t for t in seen[2])


def test_gate():
    rows = [{"holdout_ppl": 100.0}, {"holdout_ppl": 104.0}]
    assert not gate_passes(rows, 0.05)
    assert gate_passes(rows, 0.03)
