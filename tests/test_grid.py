"""Tests for the fine-tuning grid, using stub models (no torch)."""

import inspect
import json
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

import src.finetune.grid as grid
from src.finetune.lora import train_lora as real_train_lora
from src.generation.model_based import hf_generator as real_hf_generator
from src.evaluation.tracking import Run
from src.finetune.grid import (
    CONTAMINATION_TYPES,
    Cell,
    config_fingerprint,
    format_table,
    load_results,
    make_cells,
    resample_empty,
    run_grid,
)
from src.finetune.collapse import _pick
from src.ingestion.loaders import load_dataset, write_jsonl
from src.ingestion.schema import Sample

N = 8
REPO = Path(__file__).resolve().parent.parent


def _human(seed):
    return [Sample(sample_id=f"h{seed}-{i}", response=f"human {i} " * 5) for i in range(N)]


def _pool(kind, seed):
    return [
        Sample(sample_id=f"{kind}-{seed}-{i}", response=f"{kind} {i} " * 5, source=kind, generation=1)
        for i in range(N)
    ]


class Harness:
    """Stub trainer/scorer that records what the grid asked of it."""

    def __init__(self):
        self.trained: list[tuple[Cell, list[str]]] = []
        self.pools: list[tuple[str, int]] = []
        self.released: list[Cell] = []
        self.tracked: list[str] = []
        self.logged: list[dict] = []
        self.rows: list[dict] = []

    def pool_for(self, kind, seed):
        self.pools.append((kind, seed))
        return _pool(kind, seed)

    def train(self, texts, cell):
        self.trained.append((cell, texts))
        return cell

    def evaluate(self, cell):
        return {"holdout_ppl": 50.0 + 10 * cell.ratio, "benchmark_ppl": 40.0 - 5 * cell.ratio}

    def release(self, model):
        self.released.append(model)

    def track(self, cell):
        harness = self

        class _Run(Run):
            def log_metrics(self, metrics, step=None):
                harness.logged.append(dict(metrics))

        @contextmanager
        def ctx():
            harness.tracked.append(cell.id)
            yield _Run()

        return ctx()

    def run(self, cells, **kw):
        return run_grid(
            cells, human_for=_human, pool_for=self.pool_for, train=self.train,
            evaluate=self.evaluate, release=self.release, track=self.track,
            on_result=self.rows.append, fingerprint="fp", **kw,
        )


def test_make_cells_is_the_full_grid_one_seed_at_a_time():
    cells = make_cells(CONTAMINATION_TYPES, [0, 0.25, 0.5, 0.75, 1], [0, 1, 2])
    assert len(cells) == 60
    assert len(set(c.id for c in cells)) == 60
    assert {c.seed for c in cells[:20]} == {0}  # seed 0 is finished before seed 1 starts
    assert Cell("synthetic", 0.25, 1).id == "synthetic_r0.25_s1"


@pytest.mark.parametrize(
    "types, ratios, seeds",
    [(["nope"], [0.5], [0]), (["synthetic"], [1.5], [0]), (["synthetic"], [0.5, 0.5], [0])],
)
def test_make_cells_rejects_bad_axes(types, ratios, seeds):
    with pytest.raises(ValueError):
        make_cells(types, ratios, seeds)


def test_each_cell_trains_on_the_requested_mixture():
    h = Harness()
    h.run(make_cells(["synthetic", "benchmark_near"], [0.25, 1.0], [0]))

    for cell, texts in h.trained:
        assert len(texts) == N
        marker = cell.type
        assert sum(marker in t for t in texts) == round(N * cell.ratio)


def test_ratio_zero_trains_once_per_seed_and_is_shared_across_types():
    h = Harness()
    rows = h.run(make_cells(CONTAMINATION_TYPES, [0.0, 0.5], [0, 1]))

    assert len(rows) == 4 * 2 * 2
    assert len(h.trained) == 2 * (1 + 4)  # per seed: one baseline + one cell per type at 0.5
    baselines = [r for r in rows if r["ratio"] == 0]
    assert len(baselines) == 8
    assert sum(not r["shared_baseline"] for r in baselines) == 2
    for seed in (0, 1):
        assert len({r["holdout_ppl"] for r in baselines if r["seed"] == seed}) == 1
    assert {r["type"] for r in baselines} == set(CONTAMINATION_TYPES)
    assert all(r["seconds"] == 0.0 for r in baselines if r["shared_baseline"])
    # nothing is generated for a ratio-0 cell: one pool per (type, seed) at ratio 0.5
    assert len(h.pools) == 4 * 2


def test_the_mixture_depends_on_the_seed_but_not_on_the_contamination_type():
    h = Harness()
    h.run(make_cells(["synthetic", "recursive"], [0.5], [0, 1]))
    clean = {
        (cell.type, cell.seed): {t for t in texts if t.startswith("human")}
        for cell, texts in h.trained
    }

    # same seed: both types keep the same human half, so only the contamination differs
    assert clean[("synthetic", 0)] == clean[("recursive", 0)]
    assert clean[("synthetic", 1)] == clean[("recursive", 1)]
    assert len(clean[("synthetic", 0)]) == N // 2
    # different seed: a different draw
    assert clean[("synthetic", 0)] != clean[("synthetic", 1)]


def test_share_baseline_off_trains_every_cell():
    h = Harness()
    h.run(make_cells(CONTAMINATION_TYPES, [0.0, 1.0], [0]), share_baseline=False)
    assert len(h.trained) == 8


def test_row_contents_and_logging():
    h = Harness()
    rows = h.run(make_cells(["synthetic"], [0.5], [2]))
    row = rows[0]

    assert row["cell"] == "synthetic_r0.5_s2"
    assert (row["n_train"], row["n_contaminated"]) == (N, N // 2)
    assert row["holdout_ppl"] == 55.0 and row["benchmark_ppl"] == 37.5
    assert 0 < row["distinct_1"] <= 1 and 0 < row["distinct_2"] <= 1
    assert row["config"] == "fp"
    assert h.rows == rows and h.tracked == ["synthetic_r0.5_s2"]
    assert h.logged[0]["holdout_ppl"] == 55.0
    assert "seed" not in h.logged[0] and "ratio" not in h.logged[0]  # those are params, not metrics
    json.dumps(row)  # must be writable to results.jsonl


def test_resume_skips_finished_cells_and_reuses_their_rows():
    cells = make_cells(["synthetic", "recursive"], [0.0, 0.5], [0])
    first = Harness()
    rows = first.run(cells[:3])

    second = Harness()
    again = second.run(cells, done={r["cell"]: r for r in rows})

    assert [r["cell"] for r in again] == [c.id for c in cells]
    assert [c.id for c, _ in second.trained] == [cells[3].id]
    assert second.rows == [again[3]]  # only the new cell is reported as a new result


def test_a_failed_evaluation_still_releases_the_model():
    h = Harness()

    def boom(cell):
        raise RuntimeError("out of memory")

    h.evaluate = boom
    with pytest.raises(RuntimeError):
        h.run(make_cells(["synthetic"], [0.5], [0]))
    assert len(h.released) == 1


def test_resample_empty_regenerates_only_the_empty_outputs():
    calls = []

    def generate(texts):
        calls.append(list(texts))
        # "b" comes back empty the first time and fine the second
        return [("" if t == "b" and len(calls) == 1 else f"out-{t}") for t in texts]

    assert resample_empty(generate)(["a", "b", "c"]) == ["out-a", "out-b", "out-c"]
    assert calls == [["a", "b", "c"], ["b"]]


def test_resample_empty_gives_up_loudly():
    with pytest.raises(ValueError, match="empty"):
        resample_empty(lambda texts: ["  "] * len(texts), tries=2)(["a"])


def test_load_results_refuses_rows_from_other_settings(tmp_path):
    path = tmp_path / "results.jsonl"
    assert load_results(path, "fp") == {}
    path.write_text(json.dumps({"cell": "a", "config": "fp"}) + "\n\n", encoding="utf-8")
    assert list(load_results(path, "fp")) == ["a"]
    with pytest.raises(ValueError, match="different settings"):
        load_results(path, "other")


def test_fingerprint_tracks_result_defining_settings_only():
    cfg = yaml.safe_load((REPO / "configs/grid.yaml").read_text())
    base = config_fingerprint(cfg)

    assert config_fingerprint({**cfg, "seeds": [7], "device": "cpu", "output_dir": "x"}) == base
    # the GATE 2 rule and where checkpoints go do not change what a cell measures
    gate = {**cfg["gate"], "monotone_tolerance": 0.5, "low_dose_ratio": 0.5}
    assert config_fingerprint({**cfg, "gate": gate, "keep_models": not cfg["keep_models"]}) == base
    assert config_fingerprint({**cfg, "tracking": {"backend": "none"}, "figure": "elsewhere.png"}) == base
    assert config_fingerprint({**cfg, "train": {**cfg["train"], "lr": 1e-3}}) != base
    assert config_fingerprint({**cfg, "generator_model": "gpt2"}) != base
    assert config_fingerprint({**cfg, "recursive_depth": cfg["recursive_depth"] + 1}) != base


SEEDS = [0, 1, 2]
RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
GATE = {
    "collapse_types": ["synthetic", "recursive", "paraphrased"],
    "monotone_tolerance": 0.005,
    "low_dose_ratio": 0.25,
}
# a collapse curve that clearly degrades: baselines 100/101/102 (spread 2), +10 at the low dose
GOOD = {0.0: [100, 101, 102], 0.25: [110, 111, 112], 0.5: [120, 121, 122],
        0.75: [130, 131, 132], 1.0: [140, 141, 142]}


def _rows(spec):
    """spec: {(type, ratio): [holdout ppl per seed]}. benchmark_ppl is a constant 30."""
    return [
        {"type": k, "ratio": r, "seed": s, "holdout_ppl": ppl, "benchmark_ppl": 30.0}
        for (k, r), ppls in spec.items() for s, ppl in enumerate(ppls)
    ]


def _grid(curves, **bench):
    """Full-grid rows. curves: {type: {ratio: [ppl per seed]}}; bench: {type: {ratio: [benchmark ppl]}}."""
    rows = _rows({(k, r): v for k, c in curves.items() for r, v in c.items()})
    for row in rows:
        row["benchmark_ppl"] = bench.get(row["type"], {}).get(row["ratio"], [30.0] * 3)[row["seed"]]
    return rows


def _all_good():
    return _grid({k: GOOD for k in GATE["collapse_types"]})


def _gate(rows, gate=GATE, seeds=SEEDS, ratios=RATIOS):
    return grid.evaluate_gate(rows, seeds=seeds, ratios=ratios, gate=gate)


def _with(curve, **changes):
    """``curve`` with some ratios replaced; the ratio is given as a string key: _with(GOOD, **{"0.5": [...]})."""
    return {**curve, **{float(k): v for k, v in changes.items()}}


def test_a_type_that_meets_both_conditions_degrades_and_the_gate_passes():
    out = _gate(_all_good())

    assert out["status"] == "passed"
    t = out["types"]["synthetic"]
    assert t["status"] == "degrades" and t["failed_conditions"] == []
    assert t["dose_response"]["ok"] and t["above_noise"]["ok"]
    assert [m["mean"] for m in t["mean_holdout_ppl"]] == [101.0, 111.0, 121.0, 131.0, 141.0]
    assert t["above_noise"]["spread"] == 2 and t["above_noise"]["ratio"] == 0.25
    assert [p["excess"] for p in t["above_noise"]["seeds"]] == [10, 10, 10]


def test_the_gate_needs_every_collapse_type_to_degrade():
    flat = {r: [100, 101, 102] for r in RATIOS}  # no dose-response at all
    out = _gate(_grid({"synthetic": GOOD, "recursive": GOOD, "paraphrased": flat}))

    assert out["status"] == "failed"
    assert [k for k, t in out["types"].items() if t["status"] == "degrades"] == ["synthetic", "recursive"]
    assert out["types"]["paraphrased"]["status"] == "fails"


def test_dose_response_failing_alone():
    # clearly above noise at the low dose, but perplexity falls 2% between 0.5 and 0.75
    dipped = _with(GOOD, **{"0.75": [118, 119, 120]})
    t = _gate(_grid({k: dipped for k in GATE["collapse_types"]}))["types"]["synthetic"]

    assert t["status"] == "fails" and t["failed_conditions"] == ["dose_response"]
    assert t["above_noise"]["ok"]
    bad = [st for st in t["dose_response"]["steps"] if not st["ok"]]
    assert [(st["from_ratio"], st["to_ratio"]) for st in bad] == [(0.5, 0.75)]
    assert bad[0]["mean_from"] == 121.0 and bad[0]["mean_to"] == 119.0
    assert bad[0]["relative_drop"] == pytest.approx(2 / 121)


def test_above_noise_failing_alone():
    # baselines 100/110/120 (spread 20); seed 1 is only +15 at the low dose. Means still rise steadily.
    noisy = {0.0: [100, 110, 120], 0.25: [130, 125, 145], 0.5: [150, 150, 160],
             0.75: [170, 170, 180], 1.0: [190, 190, 200]}
    t = _gate(_grid({k: noisy for k in GATE["collapse_types"]}))["types"]["synthetic"]

    assert t["status"] == "fails" and t["failed_conditions"] == ["above_noise"]
    assert t["dose_response"]["ok"]
    assert t["above_noise"]["spread"] == 20
    assert [(p["seed"], p["excess"], p["ok"]) for p in t["above_noise"]["seeds"]] == [
        (0, 30, True), (1, 15, False), (2, 25, True)
    ]


@pytest.mark.parametrize(
    "ratio, values, step",
    [
        ("0.25", [95, 96, 97], (0.0, 0.25)),     # first step: below the baseline
        ("0.5", [108, 109, 110], (0.25, 0.5)),
        ("0.75", [118, 119, 120], (0.5, 0.75)),
        ("1.0", [125, 126, 127], (0.75, 1.0)),   # last step
    ],
)
def test_a_dip_at_any_step_of_the_dose_response_is_caught(ratio, values, step):
    dipped = _with(GOOD, **{ratio: values})
    t = _gate(_grid({k: dipped for k in GATE["collapse_types"]}))["types"]["synthetic"]

    bad = [(st["from_ratio"], st["to_ratio"]) for st in t["dose_response"]["steps"] if not st["ok"]]
    assert bad == [step]
    assert "dose_response" in t["failed_conditions"] and t["status"] == "fails"


def test_both_conditions_can_fail_together():
    bad = {0.0: [100, 110, 120], 0.25: [105, 110, 125], 0.5: [104, 108, 120],
           0.75: [103, 107, 119], 1.0: [102, 106, 118]}
    t = _gate(_grid({k: bad for k in GATE["collapse_types"]}))["types"]["synthetic"]
    assert t["failed_conditions"] == ["dose_response", "above_noise"]


def test_monotone_tolerance_boundary_is_inclusive():
    # one seed, so the means are exact: 200 -> 199 is a drop of exactly 0.5%
    def run(dip, tolerance=0.005):
        curve = {0.0: [100], 0.25: [200], 0.5: [dip], 0.75: [250], 1.0: [300]}
        gate = {**GATE, "collapse_types": ["synthetic"], "monotone_tolerance": tolerance}
        return _gate(_grid({"synthetic": curve}, synthetic={}), gate, seeds=[0])["types"]["synthetic"]

    assert run(199)["status"] == "degrades"  # exactly at the tolerance
    just_over = run(198.9)  # a 0.55% drop
    assert just_over["failed_conditions"] == ["dose_response"]
    assert just_over["dose_response"]["steps"][1]["relative_drop"] == pytest.approx(0.0055)
    assert run(198.9, tolerance=0.01)["status"] == "degrades"  # the tolerance is the config's
    assert run(199, tolerance=0.0)["status"] == "fails"  # zero tolerance: any drop fails


def test_the_tolerance_edge_survives_float_rounding():
    # mathematically a drop of exactly 0.5%, but 100.7 -> 100.1965 computes as 0.005000000000000024
    assert (100.7 - 100.1965) / 100.7 > 0.005
    curve = {0.0: [50], 0.25: [100.7], 0.5: [100.1965], 0.75: [150], 1.0: [200]}
    gate = {**GATE, "collapse_types": ["synthetic"]}
    assert _gate(_grid({"synthetic": curve}), gate, seeds=[0])["types"]["synthetic"]["status"] == "degrades"


def test_the_dose_response_uses_the_mean_across_seeds():
    # at 0.5 two seeds are high and one collapsed: the mean (110) is an 8.3% drop from 120, the median
    # (125) is not a drop at all
    curve = {0.0: [100] * 3, 0.25: [120] * 3, 0.5: [125, 125, 80], 0.75: [130] * 3, 1.0: [140] * 3}
    t = _gate(_grid({k: curve for k in GATE["collapse_types"]}))["types"]["synthetic"]

    assert t["failed_conditions"] == ["dose_response"]
    step = next(st for st in t["dose_response"]["steps"] if not st["ok"])
    assert (step["mean_from"], step["mean_to"]) == (120, 110)
    assert step["relative_drop"] == pytest.approx(1 / 12)


@pytest.mark.parametrize(
    "low, verdict, failing_seed",
    [(0.25, "degrades", None), (0.5, "fails", 1), (1.0, "fails", 2)],
)
def test_the_noise_check_is_made_at_the_configured_low_dose(low, verdict, failing_seed):
    # seed 1 is no higher than its baseline at 0.5, and seed 2 at 1.0; everything else is far above it,
    # and the means rise throughout, so only the choice of low dose decides the verdict
    curve = {0.0: [100] * 3, 0.25: [110] * 3, 0.5: [220, 100, 220], 0.75: [250] * 3, 1.0: [400, 400, 100]}
    gate = {**GATE, "collapse_types": ["synthetic"], "low_dose_ratio": low}
    t = _gate(_grid({"synthetic": curve}), gate)["types"]["synthetic"]

    assert t["status"] == verdict and t["dose_response"]["ok"]
    assert t["above_noise"]["ratio"] == low
    assert [p["seed"] for p in t["above_noise"]["seeds"] if not p["ok"]] == (
        [] if failing_seed is None else [failing_seed]
    )


def test_a_flat_curve_is_non_decreasing():
    curve = {0.0: [100], 0.25: [150], 0.5: [150], 0.75: [150], 1.0: [150]}
    gate = {**GATE, "collapse_types": ["synthetic"]}
    assert _gate(_grid({"synthetic": curve}), gate, seeds=[0])["types"]["synthetic"]["status"] == "degrades"


def test_the_low_dose_must_beat_the_baseline_spread_strictly():
    # baselines 100 and 102: spread 2. Seed 0 gains exactly 2, which is not more than the spread.
    def run(gain0):
        curve = {0.0: [100, 102], 0.25: [100 + gain0, 112], 0.5: [130, 130], 0.75: [140, 140], 1.0: [150, 150]}
        gate = {**GATE, "collapse_types": ["synthetic"]}
        return _gate(_grid({"synthetic": curve}), gate, seeds=[0, 1])["types"]["synthetic"]

    assert run(2)["failed_conditions"] == ["above_noise"]
    assert run(3)["status"] == "degrades"


def test_the_spread_is_max_minus_min_of_the_baselines_and_each_seed_uses_its_own():
    # baselines 100/120/110: spread 20. Seed 1 sits at 120, so 139 is only +19 over ITS baseline,
    # even though 139 is well above the mean baseline plus the spread (110 + 20 = 130).
    base = {0.0: [100, 120, 110]}
    ok = {**base, 0.25: [125, 141, 135], 0.5: [150, 150, 150], 0.75: [160, 160, 160], 1.0: [170, 170, 170]}
    bad = {**ok, 0.25: [125, 139, 135]}
    gate = {**GATE, "collapse_types": ["synthetic"]}

    good = _gate(_grid({"synthetic": ok}), gate)["types"]["synthetic"]["above_noise"]
    assert good["spread"] == 20 and good["ok"]
    failing = _gate(_grid({"synthetic": bad}), gate)["types"]["synthetic"]["above_noise"]
    assert failing["spread"] == 20 and not failing["ok"]
    assert [(p["seed"], p["baseline"], p["excess"], p["ok"]) for p in failing["seeds"]] == [
        (0, 100, 25, True), (1, 120, 19, False), (2, 110, 25, True)
    ]


def test_a_single_seed_has_no_spread_so_any_rise_counts():
    gate = {**GATE, "collapse_types": ["synthetic"]}
    curve = {0.0: [100], 0.25: [100.5], 0.5: [101], 0.75: [102], 1.0: [103]}
    t = _gate(_grid({"synthetic": curve}), gate, seeds=[0])["types"]["synthetic"]
    assert t["above_noise"]["spread"] == 0 and t["status"] == "degrades"


def test_a_partial_grid_is_not_evaluable_and_never_passes():
    rows = _all_good()
    one_cell = [r for r in rows if not (r["type"] == "paraphrased" and r["ratio"] == 1.0 and r["seed"] == 2)]
    out = _gate(one_cell)

    assert out["status"] == "not_evaluable"
    assert out["types"]["paraphrased"]["status"] == "not_evaluable"
    assert out["types"]["paraphrased"]["missing_cells"] == ["paraphrased_r1_s2"]
    assert out["types"]["synthetic"]["status"] == "degrades"  # the complete types are still judged
    assert grid.summarize(one_cell, seeds=SEEDS, ratios=RATIOS, gate=GATE)["gate_2_passed"] is False

    # the #9 check: a configured seed with no results can not be ignored
    two_seeds = [r for r in rows if r["seed"] != 2]
    assert _gate(two_seeds)["status"] == "not_evaluable"
    assert _gate(two_seeds, seeds=[0, 1])["status"] == "passed"  # unless it is not configured

    # missing cells win over a failure elsewhere: it is not evaluable, not "failed"
    flat = _grid({"synthetic": {r: [100, 101, 102] for r in RATIOS}, "recursive": GOOD})
    assert _gate(flat)["status"] == "not_evaluable"

    # no baseline row for a seed makes that type not evaluable too
    no_base = [r for r in rows if not (r["type"] == "recursive" and r["ratio"] == 0 and r["seed"] == 1)]
    assert _gate(no_base)["types"]["recursive"]["missing_cells"] == ["recursive_r0_s1"]


def test_nothing_to_evaluate_is_not_a_pass():
    assert _gate([])["status"] == "not_evaluable"


def test_benchmark_near_is_reported_but_never_gated():
    # benchmark_near has a flat holdout curve, which would fail the rule if it were gated
    flat = {r: [100, 101, 102] for r in RATIOS}
    bench = {0.0: [30, 31, 32], 0.25: [24, 25, 26], 0.5: [18, 19, 20], 0.75: [12, 13, 14], 1.0: [6, 7, 8]}
    rows = _grid({**{k: GOOD for k in GATE["collapse_types"]}, "benchmark_near": flat},
                 benchmark_near=bench)
    out = grid.summarize(rows, seeds=SEEDS, ratios=RATIOS, gate=GATE)

    assert out["gate_2_status"] == "passed" and out["gate_2_passed"]
    assert "benchmark_near" not in out["gate"]["types"]
    rep = out["benchmark_near"]
    assert [b["ratio"] for b in rep] == RATIOS
    assert [b["benchmark_ppl_mean"] for b in rep] == [31, 25, 19, 13, 7]
    assert [b["delta_vs_ratio0"] for b in rep] == [0, -6, -12, -18, -24]  # negative: memorisation
    assert rep[4]["relative_delta"] == pytest.approx(7 / 31 - 1) and rep[0]["n_seeds"] == 3
    text = grid.format_gate(out)
    assert "benchmark_near (reported, not gated)" in text and "-24.00" in text and "memorised" in text

    # a complete grid, with benchmark_near counted among the configured types, is evaluable
    every = list(CONTAMINATION_TYPES)
    assert grid.summarize(rows, seeds=SEEDS, ratios=RATIOS, gate=GATE, types=every)["gate_2_status"] == "passed"
    assert grid.summarize(_all_good(), seeds=SEEDS, ratios=RATIOS, gate=GATE)["benchmark_near"] == []

    # but a partial grid is never a pass, even when only the ungated type is missing cells
    fewer = [r for r in rows if not (r["type"] == "benchmark_near" and r["ratio"] == 1.0)]
    out = grid.summarize(fewer, seeds=SEEDS, ratios=RATIOS, gate=GATE, types=every)
    assert out["gate_2_status"] == "not_evaluable" and not out["gate_2_passed"]
    assert out["gate"]["missing_cells"] == ["benchmark_near_r1_s0", "benchmark_near_r1_s1", "benchmark_near_r1_s2"]
    assert {k: t["status"] for k, t in out["gate"]["types"].items()} == {  # the gated types are still judged
        "synthetic": "degrades", "recursive": "degrades", "paraphrased": "degrades"
    }
    assert "3 configured cell(s) have no result yet" in grid.format_gate(out)


def test_a_failed_gate_explains_itself():
    dipped = _with(GOOD, **{"0.75": [118, 119, 120]})
    noisy = {0.0: [100, 110, 120], 0.25: [130, 125, 145], 0.5: [150, 150, 160],
             0.75: [170, 170, 180], 1.0: [190, 190, 200]}
    out = grid.summarize(_grid({"synthetic": dipped, "recursive": noisy, "paraphrased": GOOD}),
                         seeds=SEEDS, ratios=RATIOS, gate=GATE)
    text = grid.format_gate(out)

    assert text.startswith("GATE 2 NOT PASSED")
    assert "synthetic: FAILS dose_response" in text and "recursive: FAILS above_noise" in text
    assert "paraphrased: degrades" in text
    assert "mean 121.00 at ratio 0.5 -> 119.00 at 0.75 is a 1.65% drop, allowed 0.50%" in text
    assert "seed 1 is +15.00 over its baseline at ratio 0.25, needs more than the baseline spread 20.00" in text
    # the same numbers are in the JSON the summary writes
    json.dumps(out)
    assert out["gate"]["types"]["synthetic"]["dose_response"]["steps"][2]["relative_drop"] > 0.005


def test_the_gate_text_for_the_other_verdicts():
    assert grid.format_gate(grid.summarize(_all_good(), seeds=SEEDS, ratios=RATIOS, gate=GATE)).startswith(
        "GATE 2 PASSED")
    partial = [r for r in _all_good() if r["ratio"] != 1.0]
    text = grid.format_gate(grid.summarize(partial, seeds=SEEDS, ratios=RATIOS, gate=GATE))
    assert text.startswith("GATE 2 NOT EVALUABLE") and "3 cell(s) missing" in text


@pytest.mark.parametrize(
    "gate, match",
    [
        (None, "gate:"),
        ({"collapse_types": ["synthetic"], "monotone_tolerance": 0.005}, "low_dose_ratio"),
        ({**GATE, "collapse_types": []}, "collapse_types"),
        ({**GATE, "collapse_types": ["nope"]}, "collapse_types"),
        ({**GATE, "monotone_tolerance": -0.1}, "monotone_tolerance"),
        ({**GATE, "monotone_tolerance": 1}, "monotone_tolerance"),
        ({**GATE, "low_dose_ratio": 0.3}, "low_dose_ratio"),
        ({**GATE, "low_dose_ratio": 0}, "low_dose_ratio"),
    ],
)
def test_validate_gate_rejects_bad_settings(gate, match):
    with pytest.raises(ValueError, match=match):
        grid.validate_gate(gate, types=list(CONTAMINATION_TYPES), ratios=RATIOS)
    grid.validate_gate(GATE, types=list(CONTAMINATION_TYPES), ratios=RATIOS)  # and this one is fine


def test_format_table_lines_up_ratios_and_types():
    out = grid.summarize(
        _rows({("synthetic", 0.0): [100.0], ("synthetic", 1.0): [150.0], ("recursive", 0.0): [100.0]}),
        seeds=[0], ratios=[0.0, 1.0], gate={**GATE, "collapse_types": ["synthetic"], "low_dose_ratio": 1.0},
    )
    lines = format_table(out["curves"]).splitlines()
    assert lines[0].split() == ["ratio", "synthetic", "recursive"]
    assert lines[2].split() == ["1", "150.00", "-"]


def test_plot_writes_a_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from src.finetune.grid import _plot

    rows = _rows({
        (k, r): [100 + 20 * r * (i + 1), 101 + 20 * r * (i + 1)]
        for i, k in enumerate(CONTAMINATION_TYPES) for r in (0.0, 0.5, 1.0)
    })
    path = tmp_path / "sub" / "curves.png"
    gate = {**GATE, "collapse_types": ["synthetic"], "low_dose_ratio": 0.5}
    _plot(grid.summarize(rows, seeds=[0, 1], ratios=[0.0, 0.5, 1.0], gate=gate)["curves"], 2, path)
    assert path.stat().st_size > 1000


@pytest.mark.parametrize("name, cells", [("grid", 60), ("grid_smoke", 12)])
def test_shipped_configs_describe_the_expected_grid(name, cells):
    cfg = yaml.safe_load((REPO / f"configs/{name}.yaml").read_text())
    assert len(make_cells(cfg["types"], cfg["ratios"], cfg["seeds"])) == cells
    # every key main() reads must exist
    for key in ("base_model", "generator_model", "recursive_depth", "device", "share_baseline", "keep_models",
                "pools_dir", "models_dir", "output_dir", "figure", "gate"):
        assert key in cfg, key
    assert "gate_min_ppl_increase" not in cfg  # replaced by the gate: section
    grid.validate_gate(cfg["gate"], types=cfg["types"], ratios=cfg["ratios"])
    assert set(cfg["gate"]) == {"collapse_types", "monotone_tolerance", "low_dose_ratio"}
    assert set(cfg["data"]) >= {"kept", "holdout", "benchmark", "n_train", "n_holdout",
                                "n_benchmark", "min_words", "max_words"}
    assert set(cfg["train"]) == {"epochs", "lr", "batch_size", "max_len", "lora_r"}  # train_lora kwargs
    assert set(cfg["generate"]) == {"max_new_tokens", "batch_size"}
    assert cfg["benchmark_near"]["exact_fraction"] <= 1
    assert isinstance(cfg["recursive_depth"], int) and cfg["recursive_depth"] >= 2
    if name == "grid":  # pre-registered before the full grid's results were seen
        assert cfg["gate"] == {
            "collapse_types": ["synthetic", "recursive", "paraphrased"],
            "monotone_tolerance": 0.005,
            "low_dose_ratio": 0.25,
        }
    # the smoke run must never write over the real grid's results or figure
    if name == "grid_smoke":
        real = yaml.safe_load((REPO / "configs/grid.yaml").read_text())
        for key in ("pools_dir", "models_dir", "output_dir", "figure"):
            assert cfg[key] != real[key], key
        assert cfg["tracking"]["experiment"] != real["tracking"]["experiment"]


def _words(prefix, i):
    return " ".join(f"{prefix}{i}w{j}" for j in range(30))


def _setup(tmp_path, monkeypatch, seeds=(0, 1), **overrides):
    """A tiny corpus, a smoke-sized config, and stubbed models. Returns (config path, cfg, spies).

    ``overrides`` replace top-level config keys (types, ratios, recursive_depth, keep_models...).
    """
    # a fake `transformers` so main() runs without the real one installed
    utils = types.ModuleType("transformers.utils")
    utils.logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)

    data = tmp_path / "data"
    for name, n in (("kept", 60), ("holdout", 20), ("validation", 20)):
        write_jsonl(
            [Sample(sample_id=f"{name}{i}", response=_words(name, i)) for i in range(n)],
            data / f"{name}.jsonl",
        )

    cfg = yaml.safe_load((REPO / "configs/grid_smoke.yaml").read_text())
    cfg.update(
        seeds=list(seeds), ratios=[0.0, 1.0], types=["synthetic", "recursive"], device="cpu",
        pools_dir=str(tmp_path / "pools"), models_dir=str(tmp_path / "models"),
        output_dir=str(tmp_path / "out"), figure=str(tmp_path / "out/curves.png"),
        tracking={"backend": "none"},
    )
    cfg.update(overrides)
    if "gate" not in overrides:  # the smoke gate is for the smoke grid; fit it to this one
        gated = [t for t in cfg["types"] if t != "benchmark_near"] or list(cfg["types"])
        cfg["gate"] = {**cfg["gate"], "collapse_types": gated, "low_dose_ratio": max(cfg["ratios"])}
    cfg["data"].update(
        kept=str(data / "kept.jsonl"), holdout=str(data / "holdout.jsonl"),
        benchmark=str(data / "validation.jsonl"), n_train=10, n_holdout=8, n_benchmark=8,
    )
    config = tmp_path / "grid.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    spies = types.SimpleNamespace(
        trained=[], generated=[], generators=[], generator_kwargs=[], plots=[], kwargs={}
    )

    def fake_train_lora(texts, out_dir, **kw):
        inspect.signature(real_train_lora).bind(texts, out_dir, **kw)  # a real call would raise
        spies.trained.append((Path(out_dir).name, list(texts), kw["seed"]))
        spies.kwargs[Path(out_dir).name] = kw
        Path(out_dir).mkdir(parents=True)
        (Path(out_dir) / "config.json").write_text("{}")  # what loading a checkpoint needs
        return Path(out_dir)

    def fake_perplexity(model_dir, texts, **kw):
        # more contamination, worse model; benchmark items score 7 higher than holdout
        ratio = float(Path(model_dir).name.split("_r")[1].split("_")[0])
        return 50.0 + 20 * ratio + (7.0 if texts[0].startswith("validation") else 0.0)

    def fake_generator(model, **kw):
        inspect.signature(real_hf_generator).bind(model, **kw)  # a real call would raise
        # the output names the model that wrote it, so a test can tell base from fine-tuned
        name = Path(model).name
        fine_tuned = model != cfg["generator_model"]
        spies.generators.append(name)
        spies.generator_kwargs.append(kw)
        if fine_tuned:  # the real pipeline loads the checkpoint, so it must still be there
            assert (Path(model) / "config.json").exists(), f"checkpoint {model} was already deleted"

        def generate(texts):
            if fine_tuned:
                assert (Path(model) / "config.json").exists(), f"checkpoint {model} was deleted"
            spies.generated.append(list(texts))
            return [f"generated by {name}"] * len(texts)

        return generate

    monkeypatch.setattr(grid, "train_lora", fake_train_lora)
    monkeypatch.setattr(grid, "perplexity", fake_perplexity)
    monkeypatch.setattr(grid, "hf_generator", fake_generator)
    # the figure is checked in test_plot_writes_a_figure; stubbing it here keeps this test
    # running in CI, which has no matplotlib
    monkeypatch.setattr(grid, "_plot", lambda curves, n_seeds, path: spies.plots.append((n_seeds, path)))
    return config, cfg, spies


def test_main_end_to_end_with_stubbed_models(tmp_path, monkeypatch):
    config, cfg, spy = _setup(tmp_path, monkeypatch)

    assert grid.main(["--config", str(config)]) == 0

    # per seed: ratio 0 once (shared by both types) + ratio 1 per type
    cells = [(name, ts, seed) for name, ts, seed in spy.trained if not name.startswith("pool_")]
    names = sorted(name for name, _, _ in cells)
    assert names == sorted(
        f"{t}_r{r}_s{s}" for s in (0, 1) for t, r in (("synthetic", 0), ("synthetic", 1), ("recursive", 1))
    )
    assert {name: seed for name, _, seed in cells} == {n: int(n[-1]) * 100 for n in names}
    texts = {name: ts for name, ts, _ in cells}
    assert not any("generated" in t for t in texts["synthetic_r0_s0"])
    assert all("generated" in t for t in texts["synthetic_r1_s1"])
    assert all(len(ts) == 10 for ts in texts.values())

    # the clean data is the spike's Gen-0 draw for the seed, and differs between seeds
    rng = random.Random(1)
    kept = load_dataset(cfg["data"]["kept"])
    expected = [s.response for s in _pick(kept, 10, 20, 150, rng)]
    assert sorted(texts["synthetic_r0_s1"]) == sorted(expected)
    assert set(texts["synthetic_r0_s0"]) != set(texts["synthetic_r0_s1"])

    # synthetic continues the source texts; recursive continues that output
    assert len(spy.generated) == 2 * 2 and all(len(g) == 10 for g in spy.generated)
    # the clean data and the contaminated source are disjoint: no text is in both
    clean = {t.split()[0] for t in texts["synthetic_r0_s0"]}
    source = {t.split()[0] for t in spy.generated[0]}
    assert len(clean) == len(source) == 10 and not clean & source

    assert not list((tmp_path / "models").iterdir())  # checkpoints are released
    rows = [json.loads(line) for line in (tmp_path / "out/results.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    for r in rows:  # holdout and benchmark are scored on their own texts, not swapped
        assert r["holdout_ppl"] == 50.0 + 20 * r["ratio"]
        assert r["benchmark_ppl"] == r["holdout_ppl"] + 7.0
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["gate_2_passed"] and summary["gate_2_status"] == "passed"
    assert summary["complete"] and summary["missing_cells"] == []
    assert {k: t["status"] for k, t in summary["gate"]["types"].items()} == {
        "synthetic": "degrades", "recursive": "degrades"
    }
    assert "degrading_settings" not in summary and "gate_min_ppl_increase" not in summary["config"]
    assert spy.plots == [(2, Path(cfg["figure"]))]

    # contamination is generated once per (type, seed) and cached under the fingerprint
    fp = grid.config_fingerprint(cfg)
    assert sorted(p.name for p in (tmp_path / "pools" / fp).iterdir()) == [
        "recursive_s0.jsonl", "recursive_s1.jsonl", "synthetic_s0.jsonl", "synthetic_s1.jsonl"
    ]
    n_generated = len(spy.generated)
    (tmp_path / "out/results.jsonl").unlink()
    spy.trained.clear()
    assert grid.main(["--config", str(config)]) == 0
    assert len(spy.trained) == 6 and len(spy.generated) == n_generated  # retrained, not regenerated

    # a second run has nothing left to train, and a changed setting is refused
    spy.trained.clear()
    assert grid.main(["--config", str(config)]) == 0
    assert spy.trained == []
    cfg["train"]["lr"] = 1.0
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="different settings"):
        grid.main(["--config", str(config)])


def test_a_cut_short_run_reports_missing_cells_and_cannot_pass(tmp_path, monkeypatch, capsys):
    config, cfg, spy = _setup(tmp_path, monkeypatch, seeds=(0,))
    assert grid.main(["--config", str(config)]) == 0  # seed 0 only: complete and passing
    capsys.readouterr()

    cfg["seeds"] = [0, 1]  # the grid grows a seed that has not run
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert grid.main(["--config", str(config), "--report-only"]) == 0

    out = capsys.readouterr().out
    assert "4 of 8 cells have no result yet" in out
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert not summary["gate_2_passed"] and not summary["complete"]
    assert summary["gate_2_status"] == "not_evaluable"  # not "failed": seed 1 simply has not run
    assert "GATE 2 NOT EVALUABLE" in out
    assert summary["missing_cells"] == [
        "synthetic_r0_s1", "recursive_r0_s1", "synthetic_r1_s1", "recursive_r1_s1"
    ]


def test_results_for_cells_outside_the_config_are_ignored(tmp_path, monkeypatch, capsys):
    config, cfg, spy = _setup(tmp_path, monkeypatch, seeds=(0, 1))
    assert grid.main(["--config", str(config)]) == 0
    capsys.readouterr()

    cfg["seeds"] = [0]  # drop seed 1: its rows stay in results.jsonl
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    spy.trained.clear()
    assert grid.main(["--config", str(config)]) == 0

    assert spy.trained == []
    assert "Ignoring 4 result(s)" in capsys.readouterr().out
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert {c["n_seeds"] for c in summary["curves"]} == {1}
    assert spy.plots[-1][0] == 1


def test_main_requires_a_zero_ratio(tmp_path, monkeypatch):
    config, cfg, _ = _setup(tmp_path, monkeypatch)
    cfg["ratios"] = [0.5, 1.0]
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="0.0"):
        grid.main(["--config", str(config)])


def _gen1(n=3):
    return [
        Sample(sample_id=f"s{i}-g1", response=f"human{i} " * 4 + "gen1", source="synthetic", generation=1)
        for i in range(n)
    ]


def test_make_recursive_fine_tunes_between_generations():
    events, released = [], []

    def train(texts, g):
        events.append(("train", g, list(texts)))
        return f"model{g}"

    def generator_for(model):
        events.append(("generator", model))
        return lambda texts: [f"<{model}>"] * len(texts)

    def release(model):
        events.append(("release", model))
        released.append(model)

    out = grid.make_recursive(
        _gen1(), depth=3, train=train, generator_for=generator_for, release=release
    )

    # train on Gen-1, write Gen-2 with that model, train on Gen-2, write Gen-3 with that model;
    # a checkpoint is only deleted once the model it holds has been used
    assert [e[:2] for e in events] == [
        ("train", 1), ("generator", "model1"), ("release", "model1"),
        ("train", 2), ("generator", "model2"), ("release", "model2"),
    ]
    assert all("gen1" in t for t in events[0][2])
    assert all("<model1>" in t for t in events[3][2])
    assert len(out) == 3
    assert {(s.source, s.generation) for s in out} == {("recursive", 3)}
    assert all("<model2>" in s.response and "<model1>" not in s.response for s in out)
    assert released == ["model1", "model2"]


def test_make_recursive_releases_the_model_even_if_generation_fails():
    released = []

    def broken(model):
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError):
        grid.make_recursive(
            _gen1(), depth=2, train=lambda texts, g: "m", generator_for=broken,
            release=released.append,
        )
    assert released == ["m"]


@pytest.mark.parametrize("depth", [0, 1])
def test_make_recursive_needs_at_least_two_generations(depth):
    with pytest.raises(ValueError, match="depth"):
        grid.make_recursive(_gen1(), depth=depth, train=lambda t, g: g, generator_for=lambda m: None)


def test_make_recursive_wants_generation_one_input():
    human = [Sample(sample_id="h", response="human text " * 5)]  # generation 0
    with pytest.raises(ValueError, match="generation-1"):
        grid.make_recursive(human, depth=2, train=lambda t, g: g, generator_for=lambda m: None)


def _recursive_run(tmp_path, monkeypatch, depth, seeds=(1,), **overrides):
    config, cfg, spy = _setup(
        tmp_path, monkeypatch, seeds=seeds, types=["recursive"], ratios=[0.0, 1.0],
        recursive_depth=depth, device="cuda-sentinel", **overrides,
    )
    assert grid.main(["--config", str(config)]) == 0
    return config, cfg, spy


@pytest.mark.parametrize("seed", [0, 2])
@pytest.mark.parametrize("depth", [2, 3])
def test_the_recursive_pool_is_written_by_models_fine_tuned_on_the_previous_generation(
    tmp_path, monkeypatch, depth, seed
):
    _, cfg, spy = _recursive_run(tmp_path, monkeypatch, depth, seeds=(seed,))
    ckpt = [f"pool_recursive_s{seed}_gen{g}" for g in range(1, depth)]

    # one fine-tune per extra generation, seeded from the grid seed the way the spike seeds generations
    tuned = [(name, texts, sd) for name, texts, sd in spy.trained if name.startswith("pool_")]
    assert [name for name, _, _ in tuned] == ckpt
    assert [sd for _, _, sd in tuned] == [seed * 100 + g for g in range(1, depth)]
    assert len(tuned[0][1]) == 10
    assert all("generated by distilgpt2" in t for t in tuned[0][1])  # Gen-1: the pretrained generator
    for (_, texts, _), previous in zip(tuned[1:], ckpt):
        assert all(f"generated by {previous}" in t for t in texts)  # trained on the last generation

    # Gen-1 comes from the pretrained generator, every later step from the fine-tuned checkpoint
    assert spy.generators == ["distilgpt2", *ckpt]
    # both are built the way every other pool's generator is: grid seed, device and generate settings
    expected = {**cfg["generate"], "seed": seed, "device": "cuda-sentinel"}
    assert all(kw == expected for kw in spy.generator_kwargs)

    pool = load_dataset(tmp_path / "pools" / grid.config_fingerprint(cfg) / f"recursive_s{seed}.jsonl")
    assert len(pool) == 10
    assert {(s.source, s.generation) for s in pool} == {("recursive", depth)}
    assert all(s.response.endswith(f"generated by {ckpt[-1]}") for s in pool)  # the last step
    assert not any("distilgpt2" in s.response for s in pool)  # not the pretrained model

    # the contaminated cell trains on exactly that pool
    cell = next(texts for name, texts, _ in spy.trained if name == f"recursive_r1_s{seed}")
    assert sorted(cell) == sorted(s.response for s in pool)

    assert not list((tmp_path / "models").iterdir())  # intermediate checkpoints are deleted too


def test_the_final_generation_is_written_by_the_fine_tuned_model_not_the_base_one(
    tmp_path, monkeypatch
):
    _, cfg, spy = _recursive_run(tmp_path, monkeypatch, depth=2)

    assert spy.generators[-1] == "pool_recursive_s1_gen1"  # the checkpoint, not cfg["generator_model"]
    assert spy.generators[-1] != cfg["generator_model"]
    pool = load_dataset(tmp_path / "pools" / grid.config_fingerprint(cfg) / "recursive_s1.jsonl")
    assert {s.generation for s in pool} == {cfg["recursive_depth"]}
    assert all(s.response.endswith("generated by pool_recursive_s1_gen1") for s in pool)


def test_the_intermediate_fine_tune_uses_the_grids_base_model_and_train_settings(
    tmp_path, monkeypatch
):
    # distinct names, so base_model and generator_model cannot be confused
    _, cfg, spy = _recursive_run(
        tmp_path, monkeypatch, depth=2, base_model="base-x", generator_model="gen-y"
    )

    kw = spy.kwargs["pool_recursive_s1_gen1"]
    assert kw["base_model"] == "base-x"  # like the spike: every generation starts from the base model
    assert {k: kw[k] for k in cfg["train"]} == cfg["train"]
    assert kw["device"] == "cuda-sentinel"  # the grid's device, not a hardcoded one
    assert spy.generators[0] == "gen-y"  # Gen-1 is written by the pretrained generator model


def test_keep_models_also_keeps_the_intermediate_checkpoints(tmp_path, monkeypatch):
    _recursive_run(tmp_path, monkeypatch, depth=3, keep_models=True)
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == [
        "pool_recursive_s1_gen1", "pool_recursive_s1_gen2", "recursive_r0_s1", "recursive_r1_s1"
    ]


def test_the_recursive_pool_is_built_once_per_seed(tmp_path, monkeypatch):
    config, _, spy = _recursive_run(tmp_path, monkeypatch, depth=2)
    (tmp_path / "out/results.jsonl").unlink()  # retrain the cells, keep the cached pools
    spy.trained.clear()
    generators = len(spy.generators)

    assert grid.main(["--config", str(config)]) == 0

    assert sorted(name for name, _, _ in spy.trained) == ["recursive_r0_s1", "recursive_r1_s1"]
    assert len(spy.generators) == generators  # nothing regenerated, no intermediate fine-tune


def test_the_synthetic_arm_still_uses_only_the_pretrained_generator(tmp_path, monkeypatch):
    config, cfg, spy = _setup(tmp_path, monkeypatch, seeds=(0,), types=["synthetic"], ratios=[0.0, 1.0])
    assert grid.main(["--config", str(config)]) == 0
    assert spy.generators == ["distilgpt2"]
    assert not [name for name, _, _ in spy.trained if name.startswith("pool_")]


@pytest.mark.parametrize(
    "kind, generators", [("paraphrased", ["gen-y"]), ("benchmark_near", [])]
)
def test_the_other_contamination_pools_are_built_and_cached(tmp_path, monkeypatch, kind, generators):
    config, cfg, spy = _setup(
        tmp_path, monkeypatch, seeds=(0,), types=[kind], ratios=[0.0, 1.0],
        base_model="base-x", generator_model="gen-y",
    )
    assert grid.main(["--config", str(config)]) == 0

    # only paraphrasing uses a model, and it is the pretrained generator, not the base model
    assert spy.generators == generators
    pool = load_dataset(tmp_path / "pools" / grid.config_fingerprint(cfg) / f"{kind}_s0.jsonl")
    assert len(pool) == 10 and {s.source for s in pool} == {kind}
    cell = next(texts for name, texts, _ in spy.trained if name == f"{kind}_r1_s0")
    assert sorted(cell) == sorted(s.response for s in pool)
    assert not [name for name, _, _ in spy.trained if name.startswith("pool_")]


@pytest.mark.parametrize("depth", [1, 0, 2.5, None])
def test_main_rejects_a_bad_recursive_depth(tmp_path, monkeypatch, depth):
    config, cfg, _ = _setup(tmp_path, monkeypatch, recursive_depth=depth)
    with pytest.raises(ValueError, match="recursive_depth"):
        grid.main(["--config", str(config)])


# --- the running grid depends on these: its results.jsonl is stamped with this fingerprint ----------

# The settings that make up a fingerprint, and the fingerprint of configs/grid.yaml, as they were
# when the full grid was started. If a change makes either assertion fail, results.jsonl files from
# that run stop loading: change them only on purpose, and never mid-run.
RUNNING_GRID_FINGERPRINT = "d497f2268060"
RUNNING_GRID_KEYS = (
    "base_model", "generator_model", "recursive_depth", "data", "benchmark_near", "train",
    "generate", "share_baseline",
)


def test_the_shipped_grid_fingerprint_is_unchanged_so_a_running_grids_results_still_load(tmp_path):
    cfg = yaml.safe_load((REPO / "configs/grid.yaml").read_text())

    assert grid._FINGERPRINT_KEYS == RUNNING_GRID_KEYS
    assert config_fingerprint(cfg) == RUNNING_GRID_FINGERPRINT

    # a results.jsonl row written under that config still loads, and --summarize-only would accept it
    path = tmp_path / "results.jsonl"
    row = {"cell": "synthetic_r0_s0", "type": "synthetic", "ratio": 0.0, "seed": 0, "holdout_ppl": 56.0,
           "benchmark_ppl": 70.0, "config": RUNNING_GRID_FINGERPRINT}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_results(path, config_fingerprint(cfg)) == {"synthetic_r0_s0": row}


# --- --summarize-only ------------------------------------------------------------------------------

def _finished_grid(tmp_path, monkeypatch, seeds=(0, 1)):
    config, cfg, spy = _setup(tmp_path, monkeypatch, seeds=seeds)
    assert grid.main(["--config", str(config)]) == 0
    return config, cfg, spy


@pytest.mark.parametrize("flag", ["--summarize-only", "--report-only"])
def test_summarize_only_trains_nothing_and_rebuilds_the_report(tmp_path, monkeypatch, capsys, flag):
    config, cfg, spy = _finished_grid(tmp_path, monkeypatch)
    results = (tmp_path / "out/results.jsonl").read_bytes()
    (tmp_path / "out/summary.json").unlink()
    spy.trained.clear()
    spy.plots.clear()
    generators = len(spy.generators)
    capsys.readouterr()

    def no_training(*a, **kw):
        raise AssertionError("--summarize-only must not train")

    monkeypatch.setattr(grid, "_train_all", no_training)
    assert grid.main(["--config", str(config), flag]) == 0

    assert spy.trained == [] and len(spy.generators) == generators  # nothing trained or generated
    assert (tmp_path / "out/results.jsonl").read_bytes() == results  # results are not touched
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["gate_2_status"] == "passed" and summary["complete"]
    assert spy.plots == [(2, Path(cfg["figure"]))]
    out = capsys.readouterr().out
    assert "Mean hidden-holdout perplexity" in out and "GATE 2 PASSED" in out


def test_summarize_only_refuses_results_made_under_different_settings(tmp_path, monkeypatch):
    config, cfg, spy = _finished_grid(tmp_path, monkeypatch)
    (tmp_path / "out/summary.json").unlink()
    spy.trained.clear()
    spy.plots.clear()

    cfg["train"]["lr"] = 1.0  # a setting that is part of the fingerprint
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="different settings"):
        grid.main(["--config", str(config), "--summarize-only"])

    assert spy.trained == [] and spy.plots == []
    assert not (tmp_path / "out/summary.json").exists()  # no report from mismatched results


def test_summarize_only_re_applies_a_changed_gate_to_finished_results(tmp_path, monkeypatch):
    config, cfg, spy = _finished_grid(tmp_path, monkeypatch)
    spy.trained.clear()

    # the gate and keep_models are not part of the fingerprint, so finished results still load
    cfg["gate"]["monotone_tolerance"] = 0.02
    cfg["keep_models"] = True
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert grid.main(["--config", str(config), "--summarize-only"]) == 0

    assert spy.trained == []
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["gate"]["rule"]["monotone_tolerance"] == 0.02
    assert summary["gate"]["types"]["synthetic"]["dose_response"]["tolerance"] == 0.02


def test_summarize_only_with_no_results_says_so(tmp_path, monkeypatch):
    config, _, spy = _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="no results"):
        grid.main(["--config", str(config), "--summarize-only"])
    assert spy.trained == []


def test_main_rejects_the_old_gate_setting_and_a_missing_gate(tmp_path, monkeypatch):
    config, cfg, _ = _setup(tmp_path, monkeypatch)
    config.write_text(yaml.safe_dump({**cfg, "gate_min_ppl_increase": 0.05}), encoding="utf-8")
    with pytest.raises(ValueError, match="gate_min_ppl_increase"):
        grid.main(["--config", str(config)])

    del cfg["gate"]
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="gate:"):
        grid.main(["--config", str(config)])


def test_a_missing_benchmark_near_cell_makes_the_gate_not_evaluable(tmp_path, monkeypatch, capsys):
    config, cfg, spy = _setup(tmp_path, monkeypatch, types=["synthetic", "recursive", "benchmark_near"])
    assert grid.main(["--config", str(config)]) == 0
    assert json.loads((tmp_path / "out/summary.json").read_text())["gate_2_status"] == "passed"

    results = tmp_path / "out/results.jsonl"
    kept = [line for line in results.read_text().splitlines() if '"benchmark_near_r1_s1"' not in line]
    assert len(kept) == 11  # 12 cells, one removed
    results.write_text("\n".join(kept) + "\n")
    capsys.readouterr()
    assert grid.main(["--config", str(config), "--summarize-only"]) == 0

    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["gate_2_status"] == "not_evaluable" and not summary["gate_2_passed"]
    assert summary["gate"]["missing_cells"] == ["benchmark_near_r1_s1"]
    assert "GATE 2 NOT EVALUABLE" in capsys.readouterr().out


def test_the_gate_uses_the_configured_ratios_not_the_ratios_that_have_rows(tmp_path, monkeypatch):
    config, cfg, spy = _finished_grid(tmp_path, monkeypatch)
    assert json.loads((tmp_path / "out/summary.json").read_text())["gate_2_status"] == "passed"

    # the config grows a ratio that has not been run: the old rows do not cover it
    cfg["ratios"] = [0.0, 0.5, 1.0]
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    spy.trained.clear()
    assert grid.main(["--config", str(config), "--summarize-only"]) == 0

    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["gate_2_status"] == "not_evaluable"
    assert summary["gate"]["types"]["synthetic"]["missing_cells"] == ["synthetic_r0.5_s0", "synthetic_r0.5_s1"]
    assert spy.trained == []
