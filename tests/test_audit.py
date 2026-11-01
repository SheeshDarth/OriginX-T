"""The deployed model round-trips through model.json, and an audit reports p, the set, drivers and flags."""

import json
import math

import pytest

from src.risk.audit import audit, flag_samples, main
from src.risk.classifier import LogReg, fit_deployed

BENCH = ["the quick brown fox jumps over the lazy dog"]


def _deployed(q=0.5):
    # one feature: lower unigram entropy = degraded
    m = LogReg.from_dict({"cols": ["f_lex_entropy"], "mean": [3.0], "std": [1.0], "w": [-2.0], "b": 0.0})
    return {"model": "logreg@1", **m.to_dict(), "alpha": 0.1, "q": q}


def test_model_json_round_trip_keeps_predictions():
    rows = [{"f_lex_entropy": float(i % 5), "label": int(i % 5 < 2)} for i in range(30)]
    d = fit_deployed(rows)
    assert d["n_cal"] == 10 and d["n_train"] == 20 and set(d) >= {"cols", "w", "b", "q", "alpha"}
    again = LogReg.from_dict(json.loads(json.dumps(d)))
    m = LogReg.from_dict(d)
    assert again.predict(rows) == m.predict(rows)


def test_too_few_calibration_rows_store_q_as_null_not_infinity():
    rows = [{"f_lex_entropy": float(i), "label": i % 2} for i in range(6)]  # 2 cal rows: alpha 0.1 needs k=3
    assert fit_deployed(rows)["q"] is None


def test_audit_reports_p_set_drivers_and_flags():
    texts = ["a a a a a a a a", "the quick brown fox jumps over the lazy dog", "one two three four"]
    r = audit(texts, ["s0", "s1", "s2"], _deployed(), BENCH)
    assert r["schema"] == "audit/1" and r["n_samples"] == 3
    # contribution = w * (x - mean) / std, so p = sigmoid(contribution + b)
    c = r["contributions"][0]
    assert c["feature"] == "f_lex_entropy" and c["contribution"] == pytest.approx(-2.0 * (c["value"] - 3.0))
    assert r["p_degrade"] == pytest.approx(1 / (1 + math.exp(-c["contribution"])))
    assert r["missing_features"] == []
    assert [f["sample_id"] for f in r["flagged"]] == ["s1", "s0"]  # the leak first, then the loop
    assert "near-copy" in r["flagged"][0]["reasons"][0] and "repeats itself" in r["flagged"][1]["reasons"][0]


def test_no_q_means_both_outcomes_and_no_benchmark_means_no_leak_flags():
    r = audit(["the quick brown fox jumps over the lazy dog"], ["s0"], _deployed(q=None))
    assert r["conformal_set"] == [0, 1] and r["n_flagged"] == 0


def test_unknown_model_columns_are_listed_as_missing():
    d = {**_deployed(), "cols": ["f_model_ref_loss_mean"]}
    assert audit(["x y"], ["s0"], d)["missing_features"] == ["f_model_ref_loss_mean"]


def test_a_clean_text_is_not_flagged():
    assert flag_samples(["an ordinary sentence with no loops"], ["s0"], BENCH) == []


def test_cli(tmp_path, capsys):
    data = tmp_path / "d.jsonl"
    data.write_text("".join(json.dumps({"text": t}) + "\n" for t in ["a a a a a a a a", "b c d e"]), encoding="utf-8")
    model = tmp_path / "m.json"
    model.write_text(json.dumps(_deployed()), encoding="utf-8")
    main([str(data), "--model", str(model), "--out", str(tmp_path / "a.json")])
    report = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert report["n_samples"] == 2 and report["n_flagged"] == 1
    assert "1 of 2 samples flagged" in capsys.readouterr().out


def test_a_reference_reaches_the_families_and_none_leaves_it_out(monkeypatch):
    seen = []
    monkeypatch.setattr("src.risk.audit.FAMILIES", {"spy": (lambda texts, ctx: seen.append(dict(ctx)) or {"x": 1.0}, 1)})
    audit(["a b"], ["s0"], _deployed(), BENCH, ["clean"])
    audit(["a b"], ["s0"], _deployed(), BENCH)
    assert seen[0]["reference_texts"] == ["clean"] and seen[0]["benchmark"] == BENCH
    assert "reference_texts" not in seen[1]  # the sem family then reports its reference columns as missing
