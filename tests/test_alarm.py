"""Tests for src/evaluation/alarm.py (plain Python) and the live alarm in TrainLogger."""

import json
import math
import random

import pytest

from src.evaluation import alarm, live

STEPS = list(range(0, 110, 10))


def _run(run_id, values, loss=None):
    """A trainlog: ``values[i]`` is sig_anisotropy_last at STEPS[i]; the other signals stay flat."""
    loss = loss or [4.0 - 0.01 * i for i in range(len(values))]
    return [{"run_id": run_id, "step": t, "holdout_loss": loss[i], "sig_anisotropy_last": values[i],
             "sig_eff_rank_last": 370.0} for i, t in enumerate(STEPS[: len(values)])]


def _healthy(rng, run_id, n=len(STEPS)):
    x, out = 0.93, []
    for _ in range(n):
        out.append(x)
        x += rng.gauss(0, 0.001)
    return _run(run_id, out)


def test_the_band_and_deviations_by_hand():
    runs = [_run("a", [0.9, 0.91]), _run("b", [0.9, 0.92]), _run("c", [0.9, 0.93])]
    m = alarm.Monitor.fit(runs, ["anisotropy_last"])
    assert m.center["anisotropy_last"][10] == pytest.approx(0.02)   # median change at step 10
    assert m.scale["anisotropy_last"][10] == pytest.approx(0.01)    # sd of 0.01, 0.02, 0.03
    # two-sided: a change of 0 and of 0.04 are both 2 sd from the median
    assert m.score(_run("x", [0.9, 0.90])) == pytest.approx(2.0)
    assert m.score(_run("y", [0.9, 0.94])) == pytest.approx(2.0)
    one = alarm.Monitor.fit(runs, ["anisotropy_last"], one_sided=True)
    assert one.score(_run("x", [0.9, 0.90])) == pytest.approx(-2.0)  # a fall is not a breach upward
    with pytest.raises(ValueError):
        alarm.Monitor.fit(runs[:1], ["anisotropy_last"])


def test_calibration_needs_enough_runs_and_says_so():
    rng = random.Random(0)
    fit = [_healthy(rng, f"f{i}") for i in range(10)]
    m = alarm.Monitor.fit(fit, ["anisotropy_last"])
    few = m.calibrate([_healthy(rng, f"c{i}") for i in range(10)], 0.05)
    assert math.isinf(few.threshold) and few.first_alarm(_run("z", [0.93, 0.5])) is None
    assert few.to_dict()["threshold_infinite"] and few.to_dict()["threshold"] is None
    cal = [_healthy(rng, f"c{i}") for i in range(19)]
    m.calibrate(cal, 0.05)
    assert m.threshold == max(m.score(r) for r in cal)  # k = ceil(20 * 0.95) = 19 of 19
    with pytest.raises(ValueError):
        m.calibrate(cal, 0.0)


def test_the_false_alarm_rate_is_controlled_on_new_healthy_runs():
    rng = random.Random(1)
    alarms = total = 0
    for _ in range(60):
        m = alarm.Monitor.fit([_healthy(rng, "f") for _ in range(10)], ["anisotropy_last"])
        m.calibrate([_healthy(rng, "c") for _ in range(19)], 0.05)
        for _ in range(20):
            alarms += m.first_alarm(_healthy(rng, "t")) is not None
            total += 1
    assert alarms / total <= 0.07  # conformal: <= 0.05 in expectation; 1200 runs leave a little slack


def test_first_alarm_is_the_earliest_step_and_names_the_signal():
    rng = random.Random(2)
    m = alarm.Monitor.fit([_healthy(rng, "f") for _ in range(10)], ["anisotropy_last", "eff_rank_last"])
    m.calibrate([_healthy(rng, "c") for _ in range(19)], 0.05)
    drifting = _run("d", [0.93, 0.93, 0.93, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78])
    step, signal = m.first_alarm(drifting)
    assert signal == "anisotropy_last" and 20 <= step <= 40
    copy = alarm.Monitor.from_dict(json.loads(json.dumps(m.to_dict())))
    assert copy.first_alarm(drifting) == (step, signal)


def test_evaluate_counts_early_alarms_and_false_alarms():
    rng = random.Random(3)
    fit = [_healthy(rng, f"f{i}") for i in range(10)]
    cal = [_healthy(rng, f"c{i}") for i in range(19)]
    test = [_healthy(rng, f"t{i}") for i in range(10)]
    healthy_loss = [4.0 - 0.01 * i for i in range(len(STEPS))]
    # anisotropy drifts from step 30; the loss falls behind the healthy curve only from step 70
    lagging_loss = [x + (0.05 * (i - 6) if i >= 7 else 0.0) for i, x in enumerate(healthy_loss)]
    collapsing = [_run(f"x{i}", [0.93, 0.93, 0.93, 0.90, 0.87, 0.84, 0.81, 0.78, 0.75, 0.72, 0.69],
                       loss=lagging_loss) for i in range(4)]
    out = alarm.evaluate(fit, cal, test, collapsing, ["anisotropy_last"], 0.05)
    assert out["n"] == {"healthy_fit": 10, "healthy_cal": 19, "healthy_test": 10, "collapsing": 4, "breached": 4}
    assert out["early_alarm_rate"] == 1.0 and out["median_lead_steps"] >= 30
    assert all(r["alarm_signal"] == "anisotropy_last" for r in out["runs"])
    assert out["observed_false_alarm_rate"] <= 0.2


def test_the_cli_refuses_an_unconfirmed_rule(tmp_path):
    cfg = tmp_path / "x3.yaml"
    cfg.write_text("gate: {confirmed: false}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not confirmed"):
        alarm.main(["--config", str(cfg)])


def test_the_shipped_x3_config_keeps_the_sets_disjoint_and_calibratable():
    import yaml

    cfg = yaml.safe_load(open("configs/x3.yaml", encoding="utf-8"))
    sets = {k: set(v) for k, v in cfg["runs"].items()}
    names = list(sets)
    assert all(not sets[a] & sets[b] for i, a in enumerate(names) for b in names[i + 1:])
    assert len(sets["healthy_cal"]) >= math.ceil(1 / cfg["gate"]["alpha"]) - 1
    assert set(cfg["signals"]) == set(live.SIGNALS)


# ---------------------------------------------------------------- the live alarm in TrainLogger


class FakeModel:
    training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


def test_the_logger_raises_a_latching_alarm_from_a_calibrated_monitor(tmp_path):
    rng = random.Random(4)
    m = alarm.Monitor.fit([_healthy(rng, "f") for _ in range(10)], ["anisotropy_last"])
    m.calibrate([_healthy(rng, "c") for _ in range(19)], 0.05)
    path = iter([0.93, 0.93, 0.93, 0.90, 0.85, 0.80, 0.93, 0.93, 0.93, 0.93, 0.93])

    def measure(model, tok, logger):
        return {"holdout_loss": 4.0, "anisotropy_last": next(path),
                **{k: 1.0 for k in live.SIGNALS if k != "anisotropy_last"}}

    logger = live.TrainLogger(tmp_path / "r.jsonl", "r", ["p"], every=10, measure=measure, monitor=m)
    for step in range(0, 101):
        logger(step, FakeModel(), None, last=step == 100)
    fired = [r["step"] for r in logger.rows if r["alarm"]]
    assert fired and fired == list(range(fired[0], 101, 10))  # latches, even after the signal recovers
    assert {r["alarm_signal"] for r in logger.rows if r["alarm"]} == {"sig_anisotropy_last"}
    assert all(r["alarm_signal"] is None for r in logger.rows if not r["alarm"])
    with pytest.raises(ValueError, match="not calibrated"):
        live.TrainLogger(tmp_path / "s.jsonl", "s", ["p"], monitor=alarm.Monitor.fit(
            [_healthy(rng, "f") for _ in range(3)], ["anisotropy_last"]))
