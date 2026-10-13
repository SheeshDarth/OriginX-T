"""Tests for src/evaluation/earlywarning.py (stdlib only)."""

import pytest

from src.evaluation.earlywarning import collapse_onsets, noise_spread, trajectories

RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _table(type_, seed, ppls):
    return [{"type": type_, "seed": seed, "ratio": r, "holdout_ppl": p} for r, p in zip(RATIOS, ppls)]


def test_trajectories_group_by_type_and_seed_and_sort_by_ratio():
    rows = _table("a", 0, [1, 2, 3, 4, 5])[::-1] + _table("a", 1, [1, 2, 3, 4, 5]) + _table("b", 0, [1] * 5)
    t = trajectories(rows)
    assert list(t) == [("a", 0), ("a", 1), ("b", 0)]
    assert [r["ratio"] for r in t[("a", 0)]] == list(RATIOS)


def test_noise_spread_is_the_range_of_the_ratio_zero_runs_per_type():
    rows = _table("a", 0, [50, 60, 70, 80, 90]) + _table("a", 1, [50.5, 0, 0, 0, 0]) + _table("a", 2, [49.7, 0, 0, 0, 0])
    assert noise_spread(rows)["a"] == pytest.approx(0.8)


def test_onset_is_the_first_stage_clearly_above_the_runs_own_baseline():
    rows = (
        _table("a", 0, [50.0, 50.2, 52.0, 56.0, 60.0])   # baselines 50 and 50.5 -> spread 0.5; 2.0 above at stage 2
        + _table("a", 1, [50.5, 50.9, 51.1, 51.2, 51.3])  # 0.4 above at stage 1, 0.6 at stage 2
    )
    onsets = collapse_onsets(rows)
    assert onsets[("a", 0)] == 2
    assert onsets[("a", 1)] == 2


def test_a_trajectory_that_stays_inside_the_noise_never_collapses():
    rows = _table("a", 0, [50.0, 50.1, 50.2, 50.3, 50.4]) + _table("a", 1, [50.5, 50.5, 50.5, 50.5, 50.5])
    assert collapse_onsets(rows) == {("a", 0): None, ("a", 1): None}


def test_the_comparison_is_strictly_greater_than_the_spread():
    rows = _table("a", 0, [50.0, 50.5, 51.0, 52.0, 53.0]) + _table("a", 1, [50.5, 50.5, 50.5, 50.5, 50.5])
    assert collapse_onsets(rows)[("a", 0)] == 2  # 50.5 - 50.0 == spread 0.5 is not above it


def test_stage_zero_is_never_a_collapse_and_a_missing_baseline_is_refused():
    assert collapse_onsets(_table("a", 0, [50] * 5) + _table("a", 1, [50] * 5))[("a", 0)] is None
    with pytest.raises(ValueError, match="no ratio-0"):
        collapse_onsets(_table("a", 0, [50] * 5) + [{"type": "a", "seed": 1, "ratio": 0.5, "holdout_ppl": 60}])


def _wb(rows, values, status="fresh"):
    return [{**r, "wb_eff_rank_last": v, "wb_status": status} for r, v in zip(rows, values)]


def test_risk_is_the_signed_move_from_the_trajectorys_own_first_stage():
    from src.evaluation.earlywarning import risk_trajectories

    rows = _wb(_table("a", 0, [50, 52, 56, 60, 64]), [10, 9, 7, 6, 5])
    rows += _wb(_table("a", 1, [50.5] * 5), [20, 20, 21, 20, 20])
    keys, risks, onsets, skipped = risk_trajectories(rows, "eff_rank_last", direction=-1)
    assert keys == [("a", 0), ("a", 1)] and skipped == []
    assert risks[0] == [0, 1, 3, 4, 5]  # effective rank falling = rising risk
    assert risks[1] == [0, 0, -1, 0, 0]
    assert onsets == [1, None]
    assert risk_trajectories(rows, "eff_rank_last", direction=1)[1][0] == [0, -1, -3, -4, -5]


def test_a_trajectory_with_a_stale_or_missing_stage_is_skipped_not_patched():
    from src.evaluation.earlywarning import risk_trajectories

    good = _wb(_table("a", 0, [50, 52, 56, 60, 64]), [10, 9, 7, 6, 5])
    stale = _wb(_table("a", 1, [50.5] * 5), [1, 2, 3, 4, 5])
    stale[2]["wb_status"] = "stale"
    gap = _wb(_table("b", 0, [50] * 5), [1, 2, None, 4, 5])
    keys, _, _, skipped = risk_trajectories(good + stale + gap, "eff_rank_last", direction=-1)
    assert keys == [("a", 0)] and skipped == [("a", 1), ("b", 0)]


def test_direction_must_be_plus_or_minus_one():
    from src.evaluation.earlywarning import risk_trajectories

    with pytest.raises(ValueError, match="direction"):
        risk_trajectories([], "eff_rank_last", direction=0)
