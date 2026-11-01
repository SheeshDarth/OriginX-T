"""Tests for src/features/scores.py with stub per-text functions (no models)."""

import pytest

from src.features import scores
from src.ingestion.schema import Sample

SAMPLES = [Sample(sample_id="h0", response="a"), Sample(sample_id="s0", response="b", source="synthetic"),
           Sample(sample_id="h1", response="c")]
LOSS = {"a": 4.0, "b": 2.0, "c": None}  # c: too short to score
KNN = {"a": 0.6, "b": 0.3, "c": 0.7}


def _reflm(texts, ctx):
    return [None if LOSS[t] is None else {"loss": LOSS[t]} for t in texts]


def _knn(texts, ctx):
    return [KNN[t] for t in texts]


def test_rows_follow_scores_1_and_higher_is_worse():
    out = scores.score_cell("wikitext2", "synthetic_r0.5_s0", SAMPLES, ["a", "b", "c"], {},
                            reflm=_reflm, knn=_knn)
    assert set(out) == set(scores.METHODS)
    r = out["risk_reflm"][1]
    assert r == {"schema": "scores/1", "dataset": "wikitext2", "cell": "synthetic_r0.5_s0", "sample_id": "s0",
                 "method": "risk_reflm", "score": -2.0, "higher_is": "worse", "true_source": "synthetic"}
    assert [x["score"] for x in out["risk_reflm"]] == [-4.0, -2.0, None]  # low loss = riskier
    assert [x["score"] for x in out["risk_knn"]] == [-0.6, -0.3, -0.7]   # crowded = riskier
    assert [x["sample_id"] for x in out["risk_knn"]] == ["h0", "s0", "h1"]


def test_scores_never_see_the_true_source():
    seen = []
    scores.score_cell("d", "c", SAMPLES, ["a", "b", "c"], {"device": "cpu"},
                      reflm=lambda t, ctx: seen.append(ctx) or _reflm(t, ctx), knn=_knn)
    assert seen == [{"device": "cpu"}]  # texts and settings only


def _row(cell, source, score):
    return {"cell": cell, "true_source": source, "score": score}


def test_evaluate_pools_and_splits_by_type_and_skips_unscored():
    rows = [_row("synthetic_r0.5_s0", "synthetic", 0.9), _row("synthetic_r0.5_s0", "human", 0.1),
            _row("synthetic_r0.5_s0", "synthetic", 0.8), _row("synthetic_r0.5_s0", "human", 0.2),
            _row("human_control_r0.5_s0", "human", 0.85), _row("human_control_r0.5_s0", "human", None)]
    out = scores.evaluate(rows, n_boot=50)
    assert out["by_type"]["synthetic"]["auroc"] == 1.0
    assert out["by_type"]["human_control"] is None  # all human: nothing to rank
    # pooled: the control sample at 0.85 outranks one synthetic sample (0.8) of 2 x 3 pairs
    assert out["pooled"]["auroc"] == pytest.approx(5 / 6)
    assert out["pooled"]["n"] == 5 and out["pooled"]["n_positive"] == 2 and out["n_unscored"] == 1
    lo, hi = out["pooled"]["ci"]
    assert 0 <= lo <= hi <= 1
