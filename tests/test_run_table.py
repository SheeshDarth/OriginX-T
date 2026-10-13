"""Tests for src/features/run_table.py (stdlib only)."""

import json

import pytest

from src.features.run_table import build_run_table


def _grid_row(type_, ratio, seed, ppl):
    return {"cell": f"{type_}_r{ratio:g}_s{seed}", "type": type_, "ratio": ratio, "seed": seed,
            "holdout_ppl": ppl, "benchmark_ppl": ppl + 1, "distinct_1": 0.1, "distinct_2": 0.6}


def _wb_row(grid_row, **headline):
    return {"cell": grid_row["cell"], "kind": "cell", "grid_holdout_ppl": grid_row["holdout_ppl"],
            "headline": {"eff_rank_last": 12.0, **headline}}


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_each_run_gets_its_outcomes_its_baseline_and_its_white_box_signals(tmp_path):
    r0, r1 = _grid_row("synthetic", 0.0, 0, 50.0), _grid_row("synthetic", 0.5, 0, 60.0)
    grid = _write(tmp_path / "g.jsonl", [r0, r1])
    wb = _write(tmp_path / "w.jsonl", [_wb_row(r0, delta_rel_norm=0.0), _wb_row(r1, delta_rel_norm=0.2)])
    table = build_run_table(grid, wb)
    assert [t["cell"] for t in table] == [r0["cell"], r1["cell"]]
    assert table[1]["baseline_ppl"] == 50.0 and table[1]["ppl_excess"] == pytest.approx(0.2)
    assert table[0]["ppl_excess"] == 0.0
    assert table[1]["wb_status"] == "fresh" and table[1]["wb_delta_rel_norm"] == 0.2
    assert table[1]["wb_eff_rank_last"] == 12.0
    assert table[1]["benchmark_ppl"] == 61.0


def test_baseline_is_per_type_and_per_seed(tmp_path):
    rows = [_grid_row("synthetic", 0.0, 0, 50.0), _grid_row("synthetic", 0.0, 1, 40.0),
            _grid_row("paraphrased", 0.0, 0, 45.0), _grid_row("synthetic", 1.0, 1, 44.0)]
    table = build_run_table(_write(tmp_path / "g.jsonl", rows), tmp_path / "none.jsonl")
    assert table[3]["baseline_ppl"] == 40.0 and table[3]["ppl_excess"] == pytest.approx(0.1)


def test_a_white_box_row_measured_on_a_since_redone_run_is_stale_and_carries_no_values(tmp_path):
    old = _grid_row("benchmark_near", 0.5, 0, 60.0)
    new = {**old, "holdout_ppl": 58.0}
    grid = _write(tmp_path / "g.jsonl", [_grid_row("benchmark_near", 0.0, 0, 50.0), new])
    wb = _write(tmp_path / "w.jsonl", [_wb_row(old)])
    row = build_run_table(grid, wb)[1]
    assert row["wb_status"] == "stale" and not any(k.startswith("wb_") and k != "wb_status" for k in row)


def test_missing_white_box_file_or_cell_is_reported_not_an_error(tmp_path):
    grid = _write(tmp_path / "g.jsonl", [_grid_row("synthetic", 0.0, 0, 50.0)])
    assert build_run_table(grid, tmp_path / "absent.jsonl")[0]["wb_status"] == "missing"


def test_recursion_stage_rows_are_not_joined_to_grid_cells(tmp_path):
    r0 = _grid_row("recursive", 0.0, 0, 50.0)
    stage = {**_wb_row(r0), "kind": "recursion_stage"}
    grid = _write(tmp_path / "g.jsonl", [r0])
    assert build_run_table(grid, _write(tmp_path / "w.jsonl", [stage]))[0]["wb_status"] == "missing"


def test_a_missing_baseline_leaves_the_excess_empty(tmp_path):
    grid = _write(tmp_path / "g.jsonl", [_grid_row("synthetic", 0.5, 0, 60.0)])
    row = build_run_table(grid, tmp_path / "w.jsonl")[0]
    assert row["baseline_ppl"] is None and row["ppl_excess"] is None
