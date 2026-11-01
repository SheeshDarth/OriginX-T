"""The features/1 and risk/1 fixtures (HANDBOOK §7.2, §7.4) match their documented shapes.

The API (C7) and the UI's MSW mocks build against these before the producers exist.
"""

import json
import re
from pathlib import Path

from src.finetune.grid import CONTAMINATION_TYPES, Cell

FIXTURES = Path(__file__).parent / "fixtures"
FEATURE_KEYS = {"schema", "dataset", "cell", "type", "ratio", "seed", "n_samples", "feature_version"}
RISK_KEYS = {"schema", "dataset", "cell", "p_degrade", "label", "conformal_set", "alpha", "model", "split"}


def _rows(name):
    data = (FIXTURES / name).read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf"), "no BOM"
    assert b"\r" not in data, "LF line endings"
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    assert 5 <= len(rows) <= 20
    return rows


def _num(v):
    return isinstance(v, float | int) and not isinstance(v, bool)


def test_feature_rows_have_the_shared_keys_and_prefixed_float_columns():
    rows = _rows("features.jsonl")
    families = set()
    for r in rows:
        assert FEATURE_KEYS <= set(r) and r["schema"] == "features/1"
        assert r["dataset"] in ("wikitext2", "tinystories") and r["type"] in CONTAMINATION_TYPES
        assert r["cell"] == Cell(r["type"], r["ratio"], r["seed"]).id
        cols = set(r) - FEATURE_KEYS
        assert cols and all(re.fullmatch(r"f_(lex|leak|model|sem)_\w+", c) for c in cols)
        assert all(r[c] is None or _num(r[c]) for c in cols)
        families |= {c.split("_")[1] for c in cols}
    assert families == {"lex", "leak", "model", "sem"}
    assert any(v is None for r in rows for v in r.values()), "covers a not-computable feature"
    assert len({(r["dataset"], r["cell"]) for r in rows}) == len(rows)


def test_risk_rows_match_the_feature_cells_and_have_valid_conformal_sets():
    cells = {(r["dataset"], r["cell"]) for r in _rows("features.jsonl")}
    rows = _rows("risk.jsonl")
    for r in rows:
        assert set(r) == RISK_KEYS and r["schema"] == "risk/1"
        assert (r["dataset"], r["cell"]) in cells
        assert 0 <= r["p_degrade"] <= 1 and 0 < r["alpha"] < 1
        assert r["label"] in (0, 1) and r["split"] in ("train", "cal", "test")
        assert r["conformal_set"] in ([0], [1], [0, 1])
        assert re.fullmatch(r"\w+@\w+", r["model"])
    assert {r["split"] for r in rows} == {"train", "cal", "test"}
    assert {tuple(r["conformal_set"]) for r in rows} == {(0,), (1,), (0, 1)}
