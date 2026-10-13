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


def _separable():
    # three collapsers (onset 3) whose risk climbs before it; three never-collapsers that stay flat
    risks = [[0, 2, 5, 9], [0, 3, 6, 9], [0, 4, 7, 9], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
    return risks, [3, 3, 3, None, None, None]


def test_a_signal_that_climbs_before_collapse_beats_the_random_baseline():
    from src.evaluation.earlywarning import evaluate_risk

    risks, onsets = _separable()
    out = evaluate_risk(risks, onsets, far=0.0, n_boot=200, n_perm=500, seed=0)
    assert out["status"] == "ok" and out["n_collapsing"] == 3 and out["n_never"] == 3
    assert out["auroc"] == 1.0
    assert out["permutation_p"] == pytest.approx(1 / 20, abs=0.04)  # 1 of C(6,3)=20 labelings is perfect
    assert out["early_warning"]["detected"] == 3 and out["early_warning"]["mean_lead"] > 0


def test_risk_from_the_collapse_itself_does_not_count():
    from src.evaluation.earlywarning import evaluate_risk

    # the signal only moves at/after the onset: nothing before it can warn
    risks = [[0, 0, 9, 9], [0, 0, 9, 9], [0, 0, 0, 0], [0, 0, 0, 0]]
    out = evaluate_risk(risks, [2, 2, None, None], far=0.0, n_boot=100, n_perm=100, seed=0)
    assert out["auroc"] == 0.5
    assert out["early_warning"]["detected"] == 0


def test_a_signal_unrelated_to_collapse_is_not_significant():
    from src.evaluation.earlywarning import evaluate_risk

    risks = [[0, 3], [0, 1], [0, 2], [0, 2], [0, 3], [0, 1]]
    out = evaluate_risk(risks, [1, 1, 1, None, None, None], far=0.0, n_boot=100, n_perm=400, seed=1)
    assert out["permutation_p"] > 0.2


def test_all_trajectories_collapsing_is_not_evaluable_not_an_error():
    from src.evaluation.earlywarning import evaluate_risk

    out = evaluate_risk([[0, 1], [0, 2]], [1, 1], far=0.05, n_boot=10, n_perm=10, seed=0)
    assert out == {"status": "not_evaluable", "reason": "no trajectory that never collapses",
                   "n_collapsing": 2, "n_never": 0}
    out = evaluate_risk([[0, 1]], [None], far=0.05, n_boot=10, n_perm=10, seed=0)
    assert out["reason"] == "no collapsing trajectory"


def test_evaluation_is_reproducible_for_a_seed():
    from src.evaluation.earlywarning import evaluate_risk

    risks, onsets = _separable()
    a = evaluate_risk(risks, onsets, far=0.0, n_boot=100, n_perm=100, seed=5)
    assert a == evaluate_risk(risks, onsets, far=0.0, n_boot=100, n_perm=100, seed=5)


def _gate_table(signal_moves=True):
    """Two collapsers (onset stage 2) whose rank falls at stage 1, two runs that never collapse."""
    specs = [
        ([50.0, 50.2, 56, 60, 66], [10, 8, 6, 5, 4]),
        ([50.3, 50.4, 57, 61, 67], [10, 9, 6, 5, 3]),
        ([50.1, 50.2, 50.3, 50.2, 50.1], [10, 10, 10, 10, 10]),
        ([50.2, 50.1, 50.2, 50.3, 50.1], [10, 10, 10, 10, 10]),
    ]
    rows = []
    for seed, (ppls, sig) in enumerate(specs):
        rows += _wb(_table("t", seed, ppls), sig if signal_moves else [10] * 5)
    return rows


CFG = {"signals": {"eff_rank_last": -1}, "alpha": 0.5, "far": 0.0, "n_boot": 50, "n_perm": 200, "seed": 0}


def test_gate3_passes_when_a_signal_moves_before_collapse():
    from src.evaluation.earlywarning import evaluate_gate3

    out = evaluate_gate3(_gate_table(), CFG)
    sig = out["signals"]["eff_rank_last"]
    assert out["status"] == "passed" and sig["warns"] and sig["early_warning"]["mean_lead"] > 0
    assert out["alpha_adjusted"] == 0.5


def test_gate3_fails_when_the_signal_is_flat_but_collapse_is_there_to_see():
    from src.evaluation.earlywarning import evaluate_gate3

    out = evaluate_gate3(_gate_table(signal_moves=False), CFG)
    assert out["status"] == "failed" and not out["signals"]["eff_rank_last"]["warns"]


def test_gate3_is_not_evaluable_when_every_trajectory_collapses():
    from src.evaluation.earlywarning import evaluate_gate3

    rows = _wb(_table("t", 0, [50, 60, 70, 80, 90]), [10, 9, 8, 7, 6])
    rows += _wb(_table("t", 1, [50.1, 61, 71, 81, 91]), [10, 9, 8, 7, 6])
    out = evaluate_gate3(rows, CFG)
    assert out["status"] == "not_evaluable"
    assert out["signals"]["eff_rank_last"]["reason"] == "no trajectory that never collapses"


def test_gate3_without_fresh_signal_values_is_not_evaluable_and_names_what_was_skipped():
    from src.evaluation.earlywarning import evaluate_gate3

    rows = [{**r, "wb_status": "missing"} for r in _gate_table()]
    out = evaluate_gate3(rows, CFG)
    sig = out["signals"]["eff_rank_last"]
    assert out["status"] == "not_evaluable" and sig["n_trajectories"] == 0 and len(sig["skipped"]) == 4


def test_gate3_splits_alpha_across_the_signals_tried():
    from src.evaluation.earlywarning import evaluate_gate3

    cfg = {**CFG, "alpha": 0.05, "signals": {"eff_rank_last": -1, "anisotropy_last": 1}}
    rows = [{**r, "wb_anisotropy_last": 0.5} for r in _gate_table()]
    out = evaluate_gate3(rows, cfg)
    assert out["alpha_adjusted"] == 0.025
    assert out["signals"]["eff_rank_last"]["permutation_p"] > 0.025  # 4 trajectories can't reach 0.025
    assert out["status"] == "failed"
