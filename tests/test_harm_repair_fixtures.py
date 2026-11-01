"""The scores/1 (harm) and repair/1 fixtures (HANDBOOK §7.3, §7.5) match their documented shapes.

The API's /v1/repair route and the UI's Repair screen build against these before X1's
producers exist, so the fixtures must hold every rule the producers will.
"""

import json
import re
from pathlib import Path

from src.finetune.grid import CONTAMINATION_TYPES, Cell
from src.ingestion.schema import SOURCE_TYPES

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
SCORE_KEYS = {"schema", "dataset", "cell", "sample_id", "method", "score", "higher_is", "true_source"}
REPAIR_KEYS = {"schema", "dataset", "cell", "strategy", "k_replaced", "holdout_ppl", "benchmark_ppl",
               "distinct_2", "recovery", "seed"}
# §9.3: the repair strategies X1 compares (risk_guided uses Track C's risk_* scores)
STRATEGIES = {"harm_guided", "repr_sim", "risk_guided", "random", "perplexity_filter", "oracle"}
_CELL = re.compile(r"(?P<type>\w+?)_r(?P<ratio>[\d.]+)_s(?P<seed>\d+)")


def _rows(name):
    data = (FIXTURES / name).read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf"), "no BOM"
    assert b"\r" not in data, "LF line endings"
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    assert 5 <= len(rows) <= 20
    return rows


def _num(v):
    return isinstance(v, float | int) and not isinstance(v, bool)


def _cell(cell_id):
    m = _CELL.fullmatch(cell_id)
    assert m and m["type"] in CONTAMINATION_TYPES
    cell = Cell(m["type"], float(m["ratio"]), int(m["seed"]))
    assert cell.id == cell_id
    return cell


def test_harm_score_rows_have_the_shared_keys_and_a_harm_method():
    rows = _rows("scores_harm.jsonl")
    for r in rows:
        assert set(r) == SCORE_KEYS and r["schema"] == "scores/1"
        assert r["dataset"] in ("wikitext2", "tinystories")
        assert _cell(r["cell"]).ratio > 0, "a ratio-0 cell is all human: nothing to rank"
        assert isinstance(r["sample_id"], str) and r["sample_id"]
        assert r["method"].startswith("harm_") and _num(r["score"])
        assert r["higher_is"] == "worse" and r["true_source"] in SOURCE_TYPES
    assert {r["method"] for r in rows} == {"harm_lora_grad", "harm_repr_sim"}, "repr-sim baseline is mandatory"
    assert {r["true_source"] for r in rows} >= {"human", "synthetic"}, "both classes, so AUROC is defined"
    keys = [(r["dataset"], r["cell"], r["sample_id"], r["method"]) for r in rows]
    assert len(set(keys)) == len(keys), "one score per sample and method"


def test_repair_rows_match_the_grid_and_their_recovery_follows_the_formula():
    grid = {r["cell"]: r for r in map(json.loads, (ROOT / "reports/grid/results.jsonl").open())}
    rows = _rows("repair.jsonl")
    for r in rows:
        assert set(r) == REPAIR_KEYS and r["schema"] == "repair/1"
        assert r["dataset"] in ("wikitext2", "tinystories") and r["strategy"] in STRATEGIES
        cell = _cell(r["cell"])
        assert cell.ratio > 0 and r["seed"] == cell.seed
        assert isinstance(r["k_replaced"], int) and 0 < r["k_replaced"] <= grid[r["cell"]]["n_train"]
        assert all(_num(r[k]) and r[k] > 0 for k in ("holdout_ppl", "benchmark_ppl", "distinct_2"))
        assert r["distinct_2"] <= 1
        # §7.5: recovery = (ppl_contaminated - ppl_repaired) / (ppl_contaminated - ppl_clean)
        contaminated = grid[r["cell"]]["holdout_ppl"]
        clean = grid[Cell(cell.type, 0.0, cell.seed).id]["holdout_ppl"]
        if contaminated <= clean:
            assert r["recovery"] is None, "no damage to recover"
        else:
            expected = (contaminated - r["holdout_ppl"]) / (contaminated - clean)
            assert abs(r["recovery"] - expected) < 1e-3
    assert {r["strategy"] for r in rows} == STRATEGIES
    assert any(r["recovery"] is None for r in rows), "covers a cell with nothing to recover"
    keys = [(r["dataset"], r["cell"], r["strategy"]) for r in rows]
    assert len(set(keys)) == len(keys)
