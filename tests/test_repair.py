"""Tests for X1 repair-and-verify (HANDBOOK §9.3, B8/B9)."""

import json

import pytest
import yaml

from src.finetune.grid import Cell, draw_human, human_control_pool
from src.ingestion.check_leakage import text_fingerprint
from src.ingestion.schema import Sample
from src.mitigation.repair import (
    STRATEGIES, protect_rare, rank, recovery, repaired, replacements, run_repair, summarize,
)

REPAIR_KEYS = {"schema", "dataset", "cell", "strategy", "k_replaced", "holdout_ppl", "benchmark_ppl",
               "distinct_2", "recovery", "seed"}


def _s(i, source="human", split="train"):
    return Sample(sample_id=f"{source}-{i}", response=f"{source} text number {i} " + "word " * 20,
                  source=source, split=split)


def test_recovery_follows_the_frozen_formula():
    assert recovery(70.0, 60.0, 65.0) == 0.5
    assert recovery(70.0, 60.0, 60.0) == 1.0 and recovery(70.0, 60.0, 75.0) == -0.5
    assert recovery(60.0, 60.0, 50.0) is None and recovery(55.0, 60.0, 50.0) is None


def test_oracle_ranks_every_contaminated_sample_first():
    samples = [_s(0), _s(1, "synthetic"), _s(2), _s(3, "synthetic"), _s(4)]
    assert set(rank(samples, "oracle", None, 0)[:2]) == {1, 3}


def test_scored_strategies_rank_by_score_with_unscored_samples_last():
    samples = [_s(i) for i in range(5)]
    scores = {"human-0": -3.0, "human-1": -1.0, "human-2": None, "human-3": -2.0, "human-4": -5.0}
    assert rank(samples, "risk_guided", scores, 0) == [1, 3, 0, 4, 2]
    assert rank(samples, "perplexity_filter", scores, 0) == [4, 0, 3, 1, 2]
    tied = {s.sample_id: 0.0 for s in samples}
    assert rank(samples, "harm_guided", tied, 0) == rank(samples, "harm_guided", tied, 0)
    with pytest.raises(ValueError, match="needs harm_lora_grad"):
        rank(samples, "harm_guided", None, 0)


def test_random_is_seeded_and_not_the_identity():
    samples = [_s(i) for i in range(50)]
    assert rank(samples, "random", None, 0) == rank(samples, "random", None, 0)
    assert rank(samples, "random", None, 0) != list(range(50))
    assert rank(samples, "random", None, 0) != rank(samples, "random", None, 1)


def test_repaired_keeps_size_and_order_and_replaces_exactly_the_worst_k():
    samples, fresh = [_s(i, "synthetic") for i in range(4)], [_s(10), _s(11), _s(12)]
    out = repaired(samples, [2, 0, 3, 1], fresh, 2)
    assert [s.sample_id for s in out] == ["human-11", "synthetic-1", "human-10", "synthetic-3"]


def _data():
    return {"n_train": 10, "min_words": 5, "max_words": 150}


def test_replacements_are_fresh_human_text_and_never_the_holdout():
    kept = [_s(i) for i in range(80)]
    holdout = [_s(i, split="hidden_holdout") for i in range(100, 110)]
    fresh = replacements(kept, holdout, _data(), 0, 10)
    human, source = draw_human(kept, _data(), 0)
    control = human_control_pool(kept, holdout, _data(), 0)
    used = {text_fingerprint(s) for s in (*human, *source, *control, *holdout)}
    assert len(fresh) == 10 and not {text_fingerprint(s) for s in fresh} & used
    assert all(s.source == "human" for s in fresh)
    assert replacements(kept, holdout, _data(), 0, 10) == fresh
    with pytest.raises(ValueError, match="fresh human samples"):
        replacements(kept, holdout, _data(), 0, 60)


GRID_ROWS = {
    "synthetic_r0.5_s0": {"cell": "synthetic_r0.5_s0", "ratio": 0.5, "seed": 0, "holdout_ppl": 70.0},
    "synthetic_r0_s0": {"cell": "synthetic_r0_s0", "ratio": 0.0, "seed": 0, "holdout_ppl": 60.0},
}


def test_protect_rare_never_empties_a_tail_cluster_within_the_budget():
    labels = [0, 7, 0, 7, 0, 9]  # 7 and 9 are rare (tail) clusters
    out = protect_rare([5, 1, 3, 0, 2, 4], labels, {7, 9})
    assert out == [1, 0, 2, 4, 5, 3], "5 is 9's only member and 3 becomes 7's last, so both move to the end"
    assert sorted(out) == list(range(6))
    for k in range(1, 5):
        assert {7, 9} <= {labels[i] for i in range(6) if i not in out[:k]}
    assert protect_rare([2, 0, 1], [0, 0, 0], set()) == [2, 0, 1], "no tail clusters: ranking unchanged"


def _run(strategies, *, scores=None, done=None, k="contaminated", **kw):
    samples = [_s(i) if i % 2 else _s(i, "synthetic") for i in range(10)]
    trained, warnings = [], []

    def train(data, cell, strategy):
        trained.append((strategy, data))
        return strategy

    def evaluate(model):  # fewer contaminated samples left = closer to clean
        n_bad = sum(s.source != "human" for s in dict(trained)[model])
        return {"holdout_ppl": 60.0 + 2 * n_bad, "benchmark_ppl": 50.0}

    rows = run_repair(
        [Cell("synthetic", 0.5, 0)], strategies, dataset="wikitext2",
        samples_for=lambda cell: samples, fresh_for=lambda seed, k: [_s(100 + i) for i in range(k)],
        scores_for=lambda method, cell: (scores or {}).get(method), grid_rows=GRID_ROWS,
        train=train, evaluate=evaluate, k=k, done=done, warn=warnings.append, **kw,
    )
    return rows, trained, warnings


def test_preserved_strategies_also_run_with_the_rare_rule():
    # sample 0 (contaminated) is the only member of rare cluster 5: the oracle must keep it
    labels = [5] + [0] * 9
    rows, trained, _ = _run(["oracle", "random"], preserve=["oracle"],
                            clusters_for=lambda cell, samples: (labels, {5}))
    assert [r["strategy"] for r in rows] == ["oracle", "random", "oracle+rare"]
    rare = dict(trained)["oracle+rare"]
    assert len(rare) == 10 and rare[0].sample_id == "synthetic-0"
    assert rows[0]["recovery"] == 1.0 and rows[2]["recovery"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="clusters_for"):
        _run(["oracle"], preserve=["oracle"])


def test_rows_are_repair_1_with_recovery_from_the_grid():
    rows, trained, warnings = _run(["oracle", "random", "harm_guided"])
    assert [r["strategy"] for r in rows] == ["oracle", "random"]
    assert warnings == ["  synthetic_r0.5_s0 harm_guided: no harm_lora_grad scores, skipped"]
    assert all(set(r) == REPAIR_KEYS and r["schema"] == "repair/1" and r["k_replaced"] == 5 for r in rows)
    oracle = rows[0]
    assert oracle["holdout_ppl"] == 60.0 and oracle["recovery"] == 1.0, "oracle removes every contaminated row"
    assert all(len(data) == 10 for _, data in trained), "equal size"
    assert 0.0 < rows[1]["distinct_2"] <= 1.0


def test_scored_strategy_uses_its_scores_and_resume_skips_finished_rows():
    scores = {"harm_lora_grad": {f"synthetic-{i}": 1.0 for i in range(0, 10, 2)}}
    rows, trained, _ = _run(["harm_guided"], scores=scores)
    assert rows[0]["recovery"] == 1.0
    done = {("synthetic_r0.5_s0", "harm_guided"): rows[0]}
    again, trained_again, _ = _run(["harm_guided", "oracle"], scores=scores, done=done)
    assert again[0] == rows[0] and [s for s, _ in trained_again] == ["oracle"]


def test_an_integer_k_overrides_the_contaminated_budget():
    rows, _, _ = _run(["random"], k=3)
    assert rows[0]["k_replaced"] == 3
    with pytest.raises(ValueError, match="k=11"):
        _run(["random"], k=11)


def _row(cell, strategy, rec):
    return {"cell": cell, "strategy": strategy, "recovery": rec}


def test_pilot_and_gate_x1_follow_the_preregistered_rule():
    types, seeds = ["synthetic", "recursive", "paraphrased"], [0, 1]
    pilot = ["synthetic_r0.5_s0", "recursive_r0.5_s0"]
    rows = [_row(c, "random", 0.1) for c in pilot] + [_row(c, "harm_guided", 0.2) for c in pilot]
    s = summarize(rows, collapse_types=types, seeds=seeds, pilot_cells=pilot)
    assert s["pilot"]["status"] == "go" and s["gate_x1"]["status"] == "not evaluable"

    def full(margins):
        out = []
        for t, m in margins.items():
            for seed in seeds:
                cell = f"{t}_r0.5_s{seed}"
                out += [_row(cell, "random", 0.2), _row(cell, "harm_guided", 0.2 + m)]
        return out

    passed = summarize(full({"synthetic": 0.1, "recursive": 0.15, "paraphrased": 0.0}),
                       collapse_types=types, seeds=seeds, pilot_cells=pilot)
    assert passed["gate_x1"]["status"] == "passed"
    assert passed["gate_x1"]["passing_types"] == ["synthetic", "recursive"]
    failed = summarize(full({"synthetic": 0.3, "recursive": 0.09, "paraphrased": 0.0}),
                       collapse_types=types, seeds=seeds, pilot_cells=pilot)
    assert failed["gate_x1"]["status"] == "failed"
    tie = [_row(c, "random", 0.3) for c in pilot] + [_row(c, "harm_guided", 0.3) for c in pilot]
    nogo = summarize(tie, collapse_types=types, seeds=seeds, pilot_cells=pilot)
    assert nogo["pilot"]["status"] == "no-go", "a tie does not beat random"


def test_h5_reports_what_the_rare_rule_changed():
    rows = [{"cell": "c", "strategy": "random", "recovery": 0.5, "distinct_2": 0.6},
            {"cell": "c", "strategy": "random+rare", "recovery": 0.4, "distinct_2": 0.65},
            {"cell": "d", "strategy": "random+rare", "recovery": None, "distinct_2": 0.6}]
    h5 = summarize(rows, collapse_types=["synthetic"], seeds=[0], pilot_cells=[])["h5"]
    assert h5 == {"random": {"c": {"recovery_change": -0.1, "distinct_2_change": 0.05}}}


def test_shipped_config_and_fixture_use_the_same_strategies():
    with open("configs/repair.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert tuple(cfg["strategies"]) == STRATEGIES
    with open("tests/fixtures/repair.jsonl", encoding="utf-8") as fh:
        assert {json.loads(line)["strategy"] for line in fh if line.strip()} == set(STRATEGIES)
