"""Tests for the fine-tuning grid, using stub models (no torch)."""

import json
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
    run_grid,
    summarize,
)
from src.ingestion.loaders import write_jsonl
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
    out = summarize(rows, 0.05)

    assert out["degrading_settings"] == [{"type": "synthetic", "ratio": 0.5}]
    assert out["gate_2_passed"]
    curve = next(c for c in out["curves"] if (c["type"], c["ratio"]) == ("synthetic", 0.5))
    assert curve["holdout_ppl_mean"] == 115.0
    assert (curve["holdout_ppl_min"], curve["holdout_ppl_max"]) == (110.0, 120.0)
    assert curve["rel_increase"] == pytest.approx([0.10, 0.20])
    assert not any(c["degrades"] for c in out["curves"] if c["ratio"] == 0)


def test_summarize_does_not_pass_without_a_baseline_or_without_degradation():
    no_baseline = summarize(_rows({("synthetic", 0.5): [500.0]}), 0.05)
    assert not no_baseline["gate_2_passed"]

    flat = summarize(_rows({("synthetic", 0.0): [100.0], ("synthetic", 1.0): [101.0]}), 0.05)
    assert not flat["gate_2_passed"] and flat["degrading_settings"] == []


def test_format_table_lines_up_ratios_and_types():
    out = summarize(_rows({
        ("synthetic", 0.0): [100.0], ("synthetic", 1.0): [150.0], ("recursive", 0.0): [100.0],
    }), 0.05)
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
    _plot(summarize(rows, 0.05)["curves"], 2, path)
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


def test_main_end_to_end_with_stubbed_models(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    # a fake `transformers` so main() runs without the real one installed
    hf_logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
    utils = types.ModuleType("transformers.utils")
    utils.logging = hf_logging
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
        ratios=[0.0, 1.0], types=["synthetic", "recursive"], device="cpu",
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

    trained, generated = [], []

    def fake_train_lora(texts, out_dir, **kw):
        trained.append((Path(out_dir).name, list(texts), kw["seed"]))
        Path(out_dir).mkdir(parents=True)
        return Path(out_dir)

    def fake_perplexity(model_dir, texts, **kw):
        ratio = float(Path(model_dir).name.split("_r")[1].split("_")[0])
        return 50.0 + 20 * ratio  # more contamination, worse model

    def fake_generator(model, **kw):
        def generate(texts):
            generated.append(list(texts))
            return ["generated words here"] * len(texts)

        return generate

    monkeypatch.setattr(grid, "train_lora", fake_train_lora)
    monkeypatch.setattr(grid, "perplexity", fake_perplexity)
    monkeypatch.setattr(grid, "hf_generator", fake_generator)

    assert grid.main(["--config", str(config)]) == 0

    # ratio 0 once (shared by both types) + ratio 1 per type
    assert sorted(name for name, _, _ in trained) == ["recursive_r1_s0", "synthetic_r0_s0", "synthetic_r1_s0"]
    assert all(seed == 0 for _, _, seed in trained)  # seed * 100: the spike's Gen-0 seed
    texts = {name: ts for name, ts, _ in trained}
    assert not any("generated" in t for t in texts["synthetic_r0_s0"])
    assert all("generated" in t for t in texts["synthetic_r1_s0"])
    assert all(len(ts) == 10 for ts in texts.values())
    # synthetic continues the source texts; recursive continues that output
    assert len(generated) == 2 and all(len(g) == 10 for g in generated)
    # the clean data and the contaminated source are disjoint: no text is in both
    clean = {t.split()[0] for t in texts["synthetic_r0_s0"]}
    source = {t.split()[0] for t in generated[0]}
    assert len(clean) == len(source) == 10 and not clean & source
    assert not list((tmp_path / "models").iterdir())  # checkpoints are released
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["gate_2_passed"]
    assert {(d["type"], d["ratio"]) for d in summary["degrading_settings"]} == {
        ("synthetic", 1.0), ("recursive", 1.0)
    }
    assert (tmp_path / "out/curves.png").exists()
    assert len((tmp_path / "out/results.jsonl").read_text().splitlines()) == 4

    # a second run has nothing left to train, and a changed setting is refused
    trained.clear()
    assert grid.main(["--config", str(config)]) == 0
    assert trained == []
    cfg["train"]["lr"] = 1.0
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="different settings"):
        grid.main(["--config", str(config)])
