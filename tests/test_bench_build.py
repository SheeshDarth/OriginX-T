"""Tests for the ORIGIN-T-Bench build command (HANDBOOK §9.2, B6)."""

import json

import pytest
import yaml

from src.bench.build import build, main
from src.finetune.grid import Cell, config_fingerprint, pool_path, training_set
from src.ingestion.loaders import load_dataset, write_jsonl
from src.ingestion.schema import Sample

TYPES = ["synthetic", "benchmark_near"]


def _rows(prefix, n, source="human", split="train", generation=0):
    return [
        Sample(sample_id=f"{prefix}-{i}", response=f"{prefix} row {i} " + "word " * 25,
               source=source, split=split, generation=generation)
        for i in range(n)
    ]


def _setup(tmp_path, *, leak=False, pools=True):
    """A tiny corpus, holdout and grid config, with the pools the grid would have cached."""
    data = tmp_path / "data"
    kept = _rows("kept", 40)
    holdout = _rows("hold", 6, split="hidden_holdout")
    write_jsonl(kept, data / "kept.jsonl")
    write_jsonl([*holdout, *(kept[:1] if leak else [])], data / "holdout.jsonl")
    grid = {
        "seeds": [0, 1], "ratios": [0.0, 0.5, 1.0], "types": TYPES, "share_baseline": True,
        "base_model": "distilgpt2", "generator_model": "distilgpt2", "recursive_depth": 2,
        "data": {"kept": str(data / "kept.jsonl"), "holdout": str(data / "holdout.jsonl"),
                 "benchmark": str(data / "val.jsonl"), "n_train": 10, "n_holdout": 4,
                 "n_benchmark": 4, "min_words": 5, "max_words": 150},
        "benchmark_near": {"exact_fraction": 0.3}, "train": {"epochs": 1}, "generate": {},
        "pools_dir": str(tmp_path / "pools"),
    }
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.safe_dump(grid), encoding="utf-8")
    if pools:
        for t in TYPES:
            for seed in grid["seeds"]:
                path = pool_path(tmp_path / "pools" / config_fingerprint(grid), t, seed)
                write_jsonl(_rows(f"{t}{seed}", 10, source=t, generation=1 if t == "synthetic" else 0), path)
    cfg = {
        "version": "0.1", "output_dir": str(tmp_path / "bench"),
        "datasets": [{"name": "toy", "grid_config": str(grid_path), "source": "toy", "licence": "CC0"}],
        "extra_types": {"duplicate": {"exact_fraction": 0.5}},
    }
    return cfg, grid


def test_writes_every_cell_as_the_grid_trained_it_and_a_manifest(tmp_path):
    cfg, grid = _setup(tmp_path)
    manifest = build(cfg)
    entry = manifest["datasets"]["toy"]
    assert manifest["schema"] == "bench-manifest/1" and manifest["version"] == "0.1"
    assert len(entry["files"]) == len(TYPES) * 2 * 3 + 2 * 2  # grid types x seeds x ratios + duplicate without r1
    assert entry["config_fingerprint"] == config_fingerprint(grid)
    assert entry["generator_versions"] == {"synthetic": "1", "benchmark_near": "2"}
    kept = load_dataset(grid["data"]["kept"])
    for t in TYPES:
        written = load_dataset(tmp_path / "bench" / "toy" / t / "r0.5" / "s1.jsonl")
        expected = training_set(grid, Cell(t, 0.5, 1), kept=kept)
        assert [s.to_dict() for s in written] == [s.to_dict() for s in expected]
    on_disk = json.loads((tmp_path / "bench" / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_counts_contamination_and_duplicates_copy_the_clean_half(tmp_path):
    cfg, _ = _setup(tmp_path)
    files = {(f["type"], f["ratio"], f["seed"]): f for f in build(cfg)["datasets"]["toy"]["files"]}
    assert files[("synthetic", 0.0, 0)]["n_contaminated"] == 0
    assert files[("synthetic", 0.5, 0)]["n_contaminated"] == 5
    assert files[("duplicate", 0.5, 0)]["n_contaminated"] == 5 and ("duplicate", 1.0, 0) not in files
    clean = {s.response for s in load_dataset(tmp_path / "bench" / "toy" / "duplicate" / "r0" / "s0.jsonl")}
    rows = load_dataset(tmp_path / "bench" / "toy" / "duplicate" / "r0.5" / "s0.jsonl")
    dups = [s for s in rows if s.source == "duplicate"]
    by_id = {s.sample_id: s for s in rows}
    assert len(dups) == 5 and all(s.sample_id.rsplit("-dup", 1)[0] in by_id for s in dups),         "every copy repeats a row that is in the same file"
    assert sum(s.response == by_id[s.sample_id.rsplit("-dup", 1)[0]].response for s in dups) in (2, 3),         "exact_fraction 0.5: about half are verbatim copies"
    assert all(s.response in clean for s in rows if s.source == "human")


def test_files_are_lf_and_hashes_are_reproducible(tmp_path):
    cfg, _ = _setup(tmp_path)
    first = build(cfg)
    raw = (tmp_path / "bench" / "toy" / "synthetic" / "r0.5" / "s0.jsonl").read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    assert build(cfg) == first


def test_refuses_a_missing_pool_instead_of_regenerating(tmp_path):
    cfg, _ = _setup(tmp_path, pools=False)
    with pytest.raises(FileNotFoundError, match="never regenerate"):
        build(cfg)
    assert not (tmp_path / "bench").exists(), "nothing is written until every cell is built"


def test_refuses_a_holdout_leak(tmp_path):
    cfg, _ = _setup(tmp_path, leak=True)
    with pytest.raises(ValueError, match="leaks the hidden holdout"):
        build(cfg)


def test_refuses_a_cell_whose_pool_copies_a_holdout_text(tmp_path):
    cfg, grid = _setup(tmp_path)
    path = pool_path(tmp_path / "pools" / config_fingerprint(grid), "synthetic", 1)
    hold = load_dataset(grid["data"]["holdout"])[0]
    leaked = [Sample(sample_id=f"x-{i}", response=hold.response, source="synthetic", generation=1)
              for i in range(10)]
    write_jsonl(leaked, path)
    with pytest.raises(ValueError, match="synthetic_r0.5_s1: the training set leaks"):
        build(cfg)


def test_missing_corpus_says_how_to_regenerate_it(tmp_path):
    cfg, _ = _setup(tmp_path)
    (tmp_path / "data" / "kept.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="normalize_corpora"):
        build(cfg)


def test_partial_build_keeps_other_datasets_and_rejects_unknown_names(tmp_path):
    cfg, _ = _setup(tmp_path)
    build(cfg)
    manifest_path = tmp_path / "bench" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasets"]["other"] = {"files": []}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert set(build(cfg, ["toy"])["datasets"]) == {"toy", "other"}
    with pytest.raises(ValueError, match="not in the config"):
        build(cfg, ["nope"])


def test_rejects_unknown_extra_types(tmp_path):
    cfg, _ = _setup(tmp_path)
    cfg["extra_types"] = {"made_up": {}}
    with pytest.raises(ValueError, match="unknown extra_types"):
        build(cfg)


def test_cli(tmp_path, capsys):
    cfg, _ = _setup(tmp_path)
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert main(["--config", str(path)]) == 0
    assert "16 files" in capsys.readouterr().out


def test_shipped_config_points_at_real_grid_configs():
    with open("configs/bench.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for entry in cfg["datasets"]:
        with open(entry["grid_config"], encoding="utf-8") as fh:
            assert yaml.safe_load(fh)["types"]
        assert entry["licence"] and entry["source"]
