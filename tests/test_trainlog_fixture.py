"""The trainlog/1 fixture (HANDBOOK §7.6) matches its documented shape (stdlib only).

The UI's live monitor and X2 build against tests/fixtures/trainlog.jsonl before X3's
producer exists, so the fixture must hold every rule the producer will.
"""

import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "trainlog.jsonl"

# the six GATE 3 signals (configs/earlywarning.yaml), named as in whitebox.jsonl's headline
SIGNALS = {"sig_eff_rank_last", "sig_part_ratio_last", "sig_anisotropy_last",
           "sig_dead_fraction_mean", "sig_stable_rank_attn", "sig_delta_rel_norm"}
FIELDS = {"schema", "run_id", "step", "holdout_loss", "alarm", "alarm_signal"} | SIGNALS


def _runs():
    data = FIXTURE.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf"), "no BOM"
    runs: dict[str, list[dict]] = {}
    for line in data.decode("utf-8").splitlines():
        row = json.loads(line)
        runs.setdefault(row["run_id"], []).append(row)
    return runs


def test_every_row_has_exactly_the_documented_fields_and_types():
    for rows in _runs().values():
        for r in rows:
            assert set(r) == FIELDS
            assert r["schema"] == "trainlog/1"
            assert isinstance(r["run_id"], str) and r["run_id"]
            assert isinstance(r["step"], int) and r["step"] >= 0
            for k in SIGNALS | {"holdout_loss"}:
                assert r[k] is None or isinstance(r[k], float | int) and not isinstance(r[k], bool)
            assert isinstance(r["alarm"], bool)


def test_steps_increase_and_the_alarm_latches_on_one_named_signal():
    for rows in _runs().values():
        steps = [r["step"] for r in rows]
        assert steps == sorted(set(steps))
        fired = None
        for r in rows:
            if fired is not None:
                assert r["alarm"] and r["alarm_signal"] == fired, "once raised, the alarm stays raised"
            if r["alarm"]:
                assert r["alarm_signal"] in SIGNALS
                fired = r["alarm_signal"]
            else:
                assert r["alarm_signal"] is None


def test_fixture_covers_a_quiet_run_and_an_alarm_event():
    runs = _runs()
    assert 5 <= sum(map(len, runs.values())) <= 20
    assert any(not any(r["alarm"] for r in rows) for rows in runs.values())
    assert any(any(r["alarm"] for r in rows) for rows in runs.values())
