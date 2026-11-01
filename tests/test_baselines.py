"""Single-feature baselines, the paired model-vs-baseline comparison, and the GATE 1 verdict."""

from src.risk.baselines import baseline_report, compare, gate_1, single_feature


def _row(i, type_, label, value):
    return {"dataset": "wikitext2", "cell": f"c{i}", "type": type_, "ratio": 0.5, "seed": 0, "label": label,
            "f_lex_entropy": value}


def test_direction_is_learned_on_the_training_fold():
    # lower entropy means degraded on both sides: the score must flip sign so higher = riskier
    rows = [_row(i, "a" if i < 4 else "b", i % 2, 10.0 - (i % 2) * 3) for i in range(8)]
    preds = single_feature(rows, "f_lex_entropy", "type")
    assert len(preds) == 8
    assert all((p["score"] > -8.5) == bool(p["label"]) for p in preds)  # degraded: -7 > clean: -10


def test_a_missing_value_scores_as_the_training_mean_and_ratio_0_is_never_tested():
    rows = [_row(i, "a" if i < 4 else "b", i % 2, float(i % 2)) for i in range(8)]
    rows[5]["f_lex_entropy"] = None
    rows.append({**_row(9, "a", 0, 0.0), "ratio": 0.0})
    preds = {p["cell"]: p for p in single_feature(rows, "f_lex_entropy", "type")}
    # c5 is type b: its training fold is type a, values 0, 1, 0, 1 plus c9 (0): mean 2/5
    assert preds["c5"]["score"] == 0.4 and "c9" not in preds


def test_compare_pairs_the_same_cells_and_needs_a_ci_above_zero():
    model = [{"dataset": "wikitext2", "cell": f"c{i}", "fold": "type=a", "label": i % 2, "p_degrade": 0.1 + 0.8 * (i % 2)}
             for i in range(20)]
    weak = [{"dataset": "wikitext2", "cell": f"c{i}", "fold": "type=a", "score": float(i % 3)} for i in range(20)]
    out = compare(model, weak)
    assert out["n"] == 20 and out["auroc_model"] == 1.0 and out["model_beats_baseline"]
    same = [{**m, "score": m["p_degrade"]} for m in model]
    assert not compare(model, same)["model_beats_baseline"]  # identical scores: difference 0, CI [0, 0]


def test_a_baseline_column_that_does_not_exist_yet_is_reported_not_faked():
    rows = [_row(i, "a" if i < 4 else "b", i % 2, float(i % 2)) for i in range(8)]
    report = baseline_report(rows, [])
    assert report["ref_lm_loss"]["note"] == "column not computed yet"
    assert report["single_feature_auroc"] == {"f_lex_entropy": 1.0}


def test_gate_1_needs_two_datasets_before_it_can_pass():
    assert gate_1(["wikitext2", "wikitext2"], True)["status"] == "not_evaluable"
    assert gate_1(["wikitext2", "tinystories"], True)["status"] == "passed"
    assert gate_1(["wikitext2", "tinystories"], False)["status"] == "failed"
