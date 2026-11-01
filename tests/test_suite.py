"""The feature suite with stub families: row shape, prefixes, resume and the shared ratio-0 set."""

import json

from src.finetune.grid import make_cells
from src.features.suite import run_suite


def _stub(calls):
    def count(texts, ctx):
        calls.append(ctx["cell"])
        return {"n": len(texts), "missing": None}
    return {"stub": (count, 1)}


def _texts(cell):
    return ["human"] * 4 + [cell.type] * int(cell.ratio * 4)


def test_rows_follow_features_1_and_ratio_0_is_measured_once_per_seed(tmp_path):
    calls = []
    cells = make_cells(["synthetic", "recursive"], [0.0, 0.5], [0, 1])
    rows = run_suite("wikitext2", cells, _texts, tmp_path / "f.jsonl", families=_stub(calls))
    assert [r["cell"] for r in rows] == [c.id for c in cells]
    r = rows[2]  # synthetic_r0.5_s0
    assert r == {"schema": "features/1", "dataset": "wikitext2", "cell": "synthetic_r0.5_s0",
                 "type": "synthetic", "ratio": 0.5, "seed": 0, "n_samples": 6,
                 "f_stub_n": 6.0, "f_stub_missing": None, "feature_version": "stub@1"}
    assert calls.count("synthetic_r0_s0") == 1 and "recursive_r0_s0" not in calls  # shared baseline
    assert rows[1]["f_stub_n"] == rows[0]["f_stub_n"]
    on_disk = [json.loads(line) for line in (tmp_path / "f.jsonl").read_text(encoding="utf-8").splitlines()]
    assert on_disk == rows


def test_resume_skips_done_cells_and_a_new_version_recomputes(tmp_path):
    out = tmp_path / "f.jsonl"
    cells = make_cells(["synthetic"], [0.0, 1.0], [0])
    run_suite("wikitext2", cells[:1], _texts, out, families=_stub([]))
    calls = []
    rows = run_suite("wikitext2", cells, _texts, out, families=_stub(calls))
    assert calls == ["synthetic_r1_s0"] and len(rows) == 2
    calls = []
    count = _stub(calls)["stub"][0]
    run_suite("wikitext2", cells, _texts, out, families={"stub": (count, 2)})
    assert calls == ["synthetic_r0_s0", "synthetic_r1_s0"]  # stub@1 rows are stale under stub@2


def test_real_families_plug_in(tmp_path):
    rows = run_suite("wikitext2", make_cells(["benchmark_near"], [1.0], [0]),
                     lambda c: ["the quick brown fox"], tmp_path / "f.jsonl",
                     ctx={"benchmark": ["The quick brown fox"]})
    assert rows[0]["feature_version"] == "lex@1+leak@1"
    assert rows[0]["f_leak_frac_above_0_9"] == 1.0 and rows[0]["f_lex_ttr"] == 1.0
