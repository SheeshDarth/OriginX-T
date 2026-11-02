"""X3 live collapse alarm (HANDBOOK §10.4, §11): two-sided, with a split-conformal false-alarm rate.

    python -m src.evaluation.alarm --config configs/x3.yaml

**The healthy band.** From healthy runs (no contamination) logged on the same step grid,
each signal's *change since step 0* is summarised per step by its median and standard
deviation across runs. Every run starts from the same base model on the same probe texts,
so step 0 is shared and changes are comparable across runs.

**A run's deviation** at step t is, per signal, ``|change_t - median_t| / sd_t``: two-sided,
since signals can move either way (anisotropy *falls* as WikiText collapses). Its *score* is
the largest deviation over every signal and every step so far. One maximum over all signals
handles the multiple comparisons: no Bonferroni split.

**The threshold** is split conformal: score held-out healthy *calibration* runs (not the ones
that built the band) and take the ``ceil((n + 1)(1 - alpha))``-th smallest score. For a new
healthy run exchangeable with them, P(it ever alarms) <= alpha. This needs n >= 1/alpha - 1
calibration runs (19 for 5%); with fewer the threshold is infinite and nothing can alarm,
which ``calibrate`` reports rather than hides.

**The alarm** fires at the first step whose deviation passes the threshold, names the signal
that passed it, and latches (trainlog/1, §7.6).

**A breach** (the damage the alarm should anticipate) is the same machinery on the holdout
loss alone, one-sided: the first step where the run's loss change is above the healthy band
by more than its own conformal threshold.

Plain Python, so it runs and is tested without numpy or torch. ``Monitor.to_dict`` /
``from_dict`` store a calibrated monitor (``reports/x3/monitor.json``) for live use:
``TrainLogger(..., monitor=Monitor.from_dict(...))``.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

Run = Sequence[Mapping[str, Any]]  # trainlog/1 rows of one run, in step order
SD_FLOOR = 1e-9  # a step where healthy runs agree exactly is not a zero-width band


def _value(row: Mapping[str, Any], key: str) -> Optional[float]:
    v = row.get(key if key == "holdout_loss" else f"sig_{key}")
    return None if v is None else float(v)


class Monitor:
    """The healthy band per (signal, step) and, once calibrated, the alarm threshold."""

    def __init__(self, signals: Sequence[str], center: Mapping[str, Mapping[int, float]],
                 scale: Mapping[str, Mapping[int, float]], *, one_sided: bool = False,
                 threshold: Optional[float] = None, alpha: Optional[float] = None,
                 n_calibration: int = 0) -> None:
        self.signals, self.one_sided = list(signals), one_sided
        self.center = {s: {int(k): v for k, v in c.items()} for s, c in center.items()}
        self.scale = {s: {int(k): v for k, v in c.items()} for s, c in scale.items()}
        self.threshold, self.alpha, self.n_calibration = threshold, alpha, n_calibration

    @classmethod
    def fit(cls, healthy: Sequence[Run], signals: Sequence[str], *, one_sided: bool = False) -> "Monitor":
        """The band from healthy runs (at least 2), per signal and step > 0."""
        if len(healthy) < 2:
            raise ValueError("the band needs at least 2 healthy runs")
        center: dict[str, dict[int, float]] = {}
        scale: dict[str, dict[int, float]] = {}
        for sig in signals:
            changes: dict[int, list[float]] = {}
            for run in healthy:
                base = _value(run[0], sig)
                for row in run[1:]:
                    v = _value(row, sig)
                    if base is not None and v is not None:
                        changes.setdefault(row["step"], []).append(v - base)
            center[sig] = {t: statistics.median(c) for t, c in changes.items() if len(c) >= 2}
            scale[sig] = {t: max(statistics.stdev(c), SD_FLOOR) for t, c in changes.items() if len(c) >= 2}
        return cls(signals, center, scale, one_sided=one_sided)

    def deviations(self, run: Run) -> list[tuple[int, str, float]]:
        """``(step, signal, deviation)`` for every step and signal the band covers."""
        out = []
        for sig in self.signals:
            base = _value(run[0], sig)
            for row in run[1:]:
                v, t = _value(row, sig), row["step"]
                if base is None or v is None or t not in self.center[sig]:
                    continue
                d = (v - base - self.center[sig][t]) / self.scale[sig][t]
                out.append((t, sig, d if self.one_sided else abs(d)))
        return out

    def score(self, run: Run) -> float:
        """The run's largest deviation (``-inf`` if the band covers none of its steps)."""
        return max((d for _, _, d in self.deviations(run)), default=-math.inf)

    def calibrate(self, healthy: Sequence[Run], alpha: float) -> "Monitor":
        """Set the split-conformal threshold from healthy runs *not* used by ``fit``."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        scores = sorted(self.score(r) for r in healthy)
        n = len(scores)
        k = math.ceil((n + 1) * (1 - alpha))
        self.threshold = scores[k - 1] if k <= n else math.inf  # too few runs: nothing may alarm
        self.alpha, self.n_calibration = alpha, n
        return self

    def first_alarm(self, run: Run) -> Optional[tuple[int, str]]:
        """``(step, signal)`` of the first deviation above the threshold, or None."""
        if self.threshold is None:
            raise ValueError("calibrate the monitor first")
        hits = sorted((t, -d, sig) for t, sig, d in self.deviations(run) if d > self.threshold)
        return (hits[0][0], hits[0][2]) if hits else None

    def to_dict(self) -> dict[str, Any]:
        return {"signals": self.signals, "one_sided": self.one_sided, "center": self.center,
                "scale": self.scale, "alpha": self.alpha, "n_calibration": self.n_calibration,
                "threshold": None if self.threshold is None or math.isinf(self.threshold) else self.threshold,
                "threshold_infinite": self.threshold is not None and math.isinf(self.threshold)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Monitor":
        threshold = math.inf if d.get("threshold_infinite") else d.get("threshold")
        return cls(d["signals"], d["center"], d["scale"], one_sided=d["one_sided"], threshold=threshold,
                   alpha=d.get("alpha"), n_calibration=d.get("n_calibration", 0))


def evaluate(healthy_fit: Sequence[Run], healthy_cal: Sequence[Run], healthy_test: Sequence[Run],
             collapsing: Sequence[Run], signals: Sequence[str], alpha: float) -> dict[str, Any]:
    """Gate X3's numbers: alarms on collapsing runs vs their loss breach, false alarms on held-out healthy runs."""
    alarm = Monitor.fit(healthy_fit, signals).calibrate(healthy_cal, alpha)
    breach = Monitor.fit(healthy_fit, ["holdout_loss"], one_sided=True).calibrate(healthy_cal, alpha)
    per_run = []
    for run in collapsing:
        a, b = alarm.first_alarm(run), breach.first_alarm(run)
        per_run.append({"run_id": run[0]["run_id"], "alarm_step": a and a[0], "alarm_signal": a and a[1],
                        "breach_step": b and b[0],
                        "lead_steps": None if a is None or b is None else b[0] - a[0]})
    breached = [r for r in per_run if r["breach_step"] is not None]
    early = [r for r in breached if r["lead_steps"] is not None and r["lead_steps"] > 0]
    false_alarms = [r[0]["run_id"] for r in healthy_test if alarm.first_alarm(r) is not None]
    return {
        "alpha": alpha, "signals": list(signals),
        "n": {"healthy_fit": len(healthy_fit), "healthy_cal": len(healthy_cal),
              "healthy_test": len(healthy_test), "collapsing": len(collapsing), "breached": len(breached)},
        "threshold": alarm.to_dict()["threshold"], "breach_threshold": breach.to_dict()["threshold"],
        "early_alarm_rate": len(early) / len(breached) if breached else None,
        "median_lead_steps": statistics.median([r["lead_steps"] for r in early]) if early else None,
        "observed_false_alarm_rate": len(false_alarms) / len(healthy_test) if healthy_test else None,
        "false_alarm_runs": false_alarms, "runs": per_run,
        "monitor": alarm.to_dict(),
    }


def load_runs(trainlog_dir: str | Path, run_ids: Sequence[str]) -> list[list[dict[str, Any]]]:
    runs = []
    for rid in run_ids:
        path = Path(trainlog_dir) / f"{rid}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        runs.append(sorted(rows, key=lambda r: r["step"]))
    return runs


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    import yaml

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/x3.yaml")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rule = cfg["gate"]
    if not rule.get("confirmed"):
        raise SystemExit("the Gate X3 rule in this config is not confirmed yet (gate.confirmed: false). "
                         "Fix it at CP0 (HANDBOOK §11, #21) before computing the verdict.")
    split = {k: [f"{cfg['dataset']}_{c}" for c in v] for k, v in cfg["runs"].items()}
    runs = {k: load_runs(cfg["trainlog_dir"], v) for k, v in split.items()}
    out = evaluate(runs["healthy_fit"], runs["healthy_cal"], runs["healthy_test"], runs["collapsing"],
                   cfg["signals"], rule["alpha"])
    out["rule"] = rule
    out["passed"] = (out["early_alarm_rate"] is not None and out["early_alarm_rate"] >= rule["min_early_alarm_rate"]
                     and out["observed_false_alarm_rate"] is not None
                     and out["observed_false_alarm_rate"] <= rule["max_observed_false_alarm_rate"])
    target = Path(cfg["output_dir"])
    target.mkdir(parents=True, exist_ok=True)
    (target / "gate_x3.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    (target / "monitor.json").write_text(json.dumps(out["monitor"], indent=2) + "\n", encoding="utf-8",
                                         newline="\n")
    print(f"Gate X3 {'PASSED' if out['passed'] else 'FAILED'}: early alarms {out['early_alarm_rate']}, "
          f"observed false alarms {out['observed_false_alarm_rate']} -> {target}")


if __name__ == "__main__":
    main()
