"""The D4 label, the hand-rolled logistic regression and the out-of-group evaluation."""

import json
import random

import pytest

from src.risk.classifier import LogReg, degraded_labels, evaluate, leave_one_out, main


def _res(type_, ratio, seed, ppl):
    return {"cell": f"{type_}_r{ratio:g}_s{seed}", "type": type_, "ratio": ratio, "seed": seed, "holdout_ppl": ppl}


def test_label_is_above_own_seed_baseline_by_more_than_the_baseline_spread():
    rows = [_res("synthetic", 0.0, 0, 56.0), _res("synthetic", 0.0, 1, 57.0),  # spread 1.0
            _res("synthetic", 0.5, 0, 57.5),  # +1.5 over its own 56.0: degraded
            _res("synthetic", 0.5, 1, 57.5)]  # +0.5 over its own 57.0: not
    assert degraded_labels(rows) == {"synthetic_r0_s0": 0, "synthetic_r0_s1": 0,
                                     "synthetic_r0.5_s0": 1, "synthetic_r0.5_s1": 0}


def test_no_baseline_means_unlabelled_not_guessed():
    assert degraded_labels([_res("recursive", 1.0, 0, 90.0)]) == {}


def _rows(n=40, seed=0):
    """Two types; f_signal separates the label, f_noise does not, one value missing."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        y = i % 2
        rows.append({"dataset": "wikitext2", "cell": f"c{i}", "type": "synthetic" if i < n // 2 else "recursive",
                     "ratio": 0.5, "seed": 0, "label": y,
                     "f_lex_signal": y * 2.0 + rng.gauss(0, 0.3), "f_lex_noise": rng.gauss(0, 1)})
    rows[3]["f_lex_noise"] = None
    return rows


def test_logreg_learns_a_separable_signal_and_ignores_missing_values():
    rows = _rows()
    m = LogReg().fit(rows, [r["label"] for r in rows], ["f_lex_signal", "f_lex_noise"])
    p = m.predict(rows)
    assert all((pi > 0.5) == bool(r["label"]) for pi, r in zip(p, rows))
    assert abs(m.w[0]) > 5 * abs(m.w[1])  # the signal carries the weight


def test_predictions_do_not_depend_on_a_feature_s_units():
    rows = _rows()
    scaled = [{**r, "f_lex_signal": r["f_lex_signal"] * 1000 + 500} for r in rows]
    cols, y = ["f_lex_signal", "f_lex_noise"], [r["label"] for r in rows]
    assert LogReg().fit(scaled, y, cols).predict(scaled) == pytest.approx(LogReg().fit(rows, y, cols).predict(rows))


def test_leave_one_type_out_never_tests_on_training_rows_or_shared_ratio_0():
    rows = _rows() + [{**_rows()[0], "cell": "synthetic_r0_s0", "ratio": 0.0, "label": 0}]
    preds = leave_one_out(rows, "type", alpha=0.2)  # 7 calibration rows: alpha 0.1 would need k = 8 > 7
    assert {p["fold"] for p in preds} == {"type=recursive", "type=synthetic"}
    assert len(preds) == 40 and "synthetic_r0_s0" not in {p["cell"] for p in preds}
    assert {p["n_cal"] for p in preds} == {7}  # a third of the 21 training-side rows
    assert evaluate(preds)["conformal"]["mean_set_size"] < 1.5  # a good model gives mostly single labels
    assert evaluate(preds)["auroc"]["value"] > 0.95


def test_the_model_is_never_fitted_on_its_calibration_rows(monkeypatch):
    sizes = []
    fit = LogReg.fit
    monkeypatch.setattr(LogReg, "fit", lambda self, rows, y, cols: sizes.append(len(rows)) or fit(self, rows, y, cols))
    preds = leave_one_out(_rows(), "type")  # each fold: 20 rows on the other side, 7 calibrate
    assert sizes == [13, 13] and {p["n_cal"] for p in preds} == {7}


def test_cli_writes_risk_1_rows_and_an_evaluation(tmp_path):
    results, feats = [], []
    for t in ("synthetic", "recursive", "human_control"):
        for s in (0, 1, 2):
            for ratio in (0.0, 0.5, 1.0):
                bad = t != "human_control" and ratio > 0
                results.append(_res(t, ratio, s, 56 + s * 0.1 + (5 * ratio if bad else 0)))
                feats.append({"schema": "features/1", "dataset": "wikitext2", "cell": f"{t}_r{ratio:g}_s{s}",
                              "type": t, "ratio": ratio, "seed": s, "n_samples": 10,
                              "f_lex_entropy": 9.9 - (ratio if bad else 0) + s * 0.01, "feature_version": "lex@1"})
    grid = tmp_path / "reports/grid/results.jsonl"
    grid.parent.mkdir(parents=True)
    grid.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    (tmp_path / "f").mkdir()
    (tmp_path / "f/wikitext2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in feats), encoding="utf-8")
    import os
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        main(["--features-dir", "f", "--out-dir", "out"])
    finally:
        os.chdir(cwd)
    rows = [json.loads(line) for line in (tmp_path / "out/predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 18  # 3 types x 3 seeds x 2 contaminated ratios
    assert all(r["schema"] == "risk/1" and r["split"] == "test" and r["alpha"] == 0.1 for r in rows)
    assert all(r["conformal_set"] in ([], [0], [1], [0, 1]) for r in rows)
    report = json.loads((tmp_path / "out/evaluation.json").read_text())
    assert report["leave_one_type_out"]["n"] == 18
    assert report["leave_one_type_out"]["auroc"]["value"] == pytest.approx(1.0)
    conf = report["leave_one_type_out"]["conformal"]
    # 6 calibration rows can't support alpha 0.1 (k = ceil(7 * 0.9) = 7 > 6): every set is {0, 1}
    assert all(r["conformal_set"] == [0, 1] for r in rows) and conf["coverage"]["value"] == 1.0
    assert conf["target"] == 0.9 and 0 <= conf["coverage"]["value"] <= 1 and conf["n_cal_per_fold"] == [6]


def test_gradient_descent_runs_to_the_optimum_not_a_step_count():
    rows = _rows()
    y = [r["label"] for r in rows]
    m = LogReg(lr=0.005, l2=0.1).fit(rows, y, ["f_lex_signal", "f_lex_noise"])  # small steps: needs >2000
    X = [m._x(r) for r in rows]
    p = m.predict(rows)
    grad = [sum((pi - yi) * x[j] for pi, yi, x in zip(p, y, X)) / len(y) + 0.1 * m.w[j] for j in range(2)]
    assert max(map(abs, grad)) < 1e-6


def test_the_hand_rolled_model_matches_scikit_learn():
    pytest.importorskip("sklearn")
    from src.risk.crosscheck import SkLogReg

    rows = _rows()
    y, cols = [r["label"] for r in rows], ["f_lex_signal", "f_lex_noise"]
    ours, theirs = LogReg(l2=0.1).fit(rows, y, cols), SkLogReg(l2=0.1).fit(rows, y, cols)
    assert ours.predict(rows) == pytest.approx(theirs.predict(rows), abs=1e-4)
