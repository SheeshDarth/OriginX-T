"""Tests for the fine-tuning grid, using stub models (no torch)."""

import json
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

import src.finetune.grid as grid
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
    summarize,
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
    assert config_fingerprint({**cfg, "train": {**cfg["train"], "lr": 1e-3}}) != base
    assert config_fingerprint({**cfg, "generator_model": "gpt2"}) != base


def _rows(spec):
    """spec: {(type, ratio): [holdout ppl per seed]}"""
    return [
        {"type": k, "ratio": r, "seed": s, "holdout_ppl": ppl, "benchmark_ppl": 30.0}
        for (k, r), ppls in spec.items() for s, ppl in enumerate(ppls)
    ]


def test_summarize_flags_a_setting_only_if_every_seed_degrades():
    rows = _rows({
        ("synthetic", 0.0): [100.0, 100.0],
        ("synthetic", 0.5): [110.0, 120.0],        # +10%, +20%: degrades
        ("synthetic", 1.0): [104.0, 130.0],        # +4% on one seed: does not
        ("recursive", 0.0): [100.0, 100.0],
        ("recursive", 0.5): [99.0, 98.0],
    })
    out = summarize(rows, 0.05, seeds=[0, 1])

    assert out["degrading_settings"] == [{"type": "synthetic", "ratio": 0.5}]
    assert out["gate_2_passed"]
    curve = next(c for c in out["curves"] if (c["type"], c["ratio"]) == ("synthetic", 0.5))
    assert curve["holdout_ppl_mean"] == 115.0
    assert (curve["holdout_ppl_min"], curve["holdout_ppl_max"]) == (110.0, 120.0)
    assert curve["rel_increase"] == pytest.approx([0.10, 0.20])
    assert not any(c["degrades"] for c in out["curves"] if c["ratio"] == 0)


def test_summarize_does_not_pass_without_a_baseline_or_without_degradation():
    no_baseline = summarize(_rows({("synthetic", 0.5): [500.0]}), 0.05, seeds=[0])
    assert not no_baseline["gate_2_passed"]

    flat = summarize(
        _rows({("synthetic", 0.0): [100.0], ("synthetic", 1.0): [101.0]}), 0.05, seeds=[0]
    )
    assert not flat["gate_2_passed"] and flat["degrading_settings"] == []


def test_a_setting_cannot_pass_on_seeds_that_have_not_finished():
    # seed 0 degrades clearly; seeds 1 and 2 have not run yet (a cut-short Kaggle session)
    rows = _rows({("synthetic", 0.0): [100.0], ("synthetic", 1.0): [150.0]})

    assert not summarize(rows, 0.05, seeds=[0, 1, 2])["gate_2_passed"]
    assert summarize(rows, 0.05, seeds=[0])["gate_2_passed"]  # but it is fine if only seed 0 is asked for

    # a seed that has its ratio-1 row but no baseline row cannot vouch either
    half = _rows({("synthetic", 0.0): [100.0], ("synthetic", 1.0): [150.0, 160.0]})
    assert not summarize(half, 0.05, seeds=[0, 1])["gate_2_passed"]


def test_format_table_lines_up_ratios_and_types():
    out = summarize(_rows({
        ("synthetic", 0.0): [100.0], ("synthetic", 1.0): [150.0], ("recursive", 0.0): [100.0],
    }), 0.05, seeds=[0])
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
    _plot(summarize(rows, 0.05, seeds=[0, 1])["curves"], 2, path)
    assert path.stat().st_size > 1000


@pytest.mark.parametrize("name, cells", [("grid", 60), ("grid_smoke", 12)])
def test_shipped_configs_describe_the_expected_grid(name, cells):
    cfg = yaml.safe_load((REPO / f"configs/{name}.yaml").read_text())
    assert len(make_cells(cfg["types"], cfg["ratios"], cfg["seeds"])) == cells
    # every key main() reads must exist
    for key in ("base_model", "generator_model", "device", "share_baseline", "keep_models",
                "pools_dir", "models_dir", "output_dir", "figure", "gate_min_ppl_increase"):
        assert key in cfg, key
    assert set(cfg["data"]) >= {"kept", "holdout", "benchmark", "n_train", "n_holdout",
                                "n_benchmark", "min_words", "max_words"}
    assert set(cfg["train"]) == {"epochs", "lr", "batch_size", "max_len", "lora_r"}  # train_lora kwargs
    assert set(cfg["generate"]) == {"max_new_tokens", "batch_size"}
    assert cfg["benchmark_near"]["exact_fraction"] <= 1
    # the smoke run must never write over the real grid's results or figure
    if name == "grid_smoke":
        real = yaml.safe_load((REPO / "configs/grid.yaml").read_text())
        for key in ("pools_dir", "models_dir", "output_dir", "figure"):
            assert cfg[key] != real[key], key
        assert cfg["tracking"]["experiment"] != real["tracking"]["experiment"]


def _words(prefix, i):
    return " ".join(f"{prefix}{i}w{j}" for j in range(30))


def _setup(tmp_path, monkeypatch, seeds=(0, 1)):
    """A tiny corpus, a smoke-sized config, and stubbed models. Returns (config path, cfg, spies)."""
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
    cfg["data"].update(
        kept=str(data / "kept.jsonl"), holdout=str(data / "holdout.jsonl"),
        benchmark=str(data / "validation.jsonl"), n_train=10, n_holdout=8, n_benchmark=8,
    )
    config = tmp_path / "grid.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    spies = types.SimpleNamespace(trained=[], generated=[], plots=[])

    def fake_train_lora(texts, out_dir, **kw):
        spies.trained.append((Path(out_dir).name, list(texts), kw["seed"]))
        Path(out_dir).mkdir(parents=True)
        return Path(out_dir)

    def fake_perplexity(model_dir, texts, **kw):
        # more contamination, worse model; benchmark items score 7 higher than holdout
        ratio = float(Path(model_dir).name.split("_r")[1].split("_")[0])
        return 50.0 + 20 * ratio + (7.0 if texts[0].startswith("validation") else 0.0)

    def fake_generator(model, **kw):
        def generate(texts):
            spies.generated.append(list(texts))
            return ["generated words here"] * len(texts)

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
    names = sorted(name for name, _, _ in spy.trained)
    assert names == sorted(
        f"{t}_r{r}_s{s}" for s in (0, 1) for t, r in (("synthetic", 0), ("synthetic", 1), ("recursive", 1))
    )
    assert {name: seed for name, _, seed in spy.trained} == {n: int(n[-1]) * 100 for n in names}
    texts = {name: ts for name, ts, _ in spy.trained}
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
    assert summary["gate_2_passed"] and summary["complete"] and summary["missing_cells"] == []
    assert {(d["type"], d["ratio"]) for d in summary["degrading_settings"]} == {
        ("synthetic", 1.0), ("recursive", 1.0)
    }
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
