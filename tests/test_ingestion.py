"""Tests for the ORIGIN-T ingestion layer (schema + loaders)."""

import json

import pytest

from src.ingestion import Sample, load_dataset, write_jsonl


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"prompt": "2+2?", "response": "4", "source": "human", "split": "test"}\n'
        '{"prompt": "cap of France?", "response": "Paris"}\n',
        encoding="utf-8",
    )
    samples = load_dataset(path)
    assert len(samples) == 2
    assert samples[0].prompt == "2+2?" and samples[0].response == "4"
    assert samples[0].split == "test"
    # deterministic id derived from file stem + index when none supplied
    assert samples[0].sample_id == "data-0000000"
    assert samples[1].source == "human"  # schema default


def test_csv_alias_mapping(tmp_path):
    path = tmp_path / "qa.csv"
    path.write_text("input,output\nHello,World\n", encoding="utf-8")
    samples = load_dataset(path)
    assert samples[0].prompt == "Hello"  # `input` -> prompt
    assert samples[0].response == "World"  # `output` -> response


def test_txt_one_record_per_line(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("first line\n\nsecond line\n", encoding="utf-8")
    samples = load_dataset(path, source="human")
    assert len(samples) == 2  # blank line skipped
    assert samples[0].response == "first line"
    assert samples[0].prompt == "" and samples[0].generation == 0


def test_explicit_args_override_record(tmp_path):
    path = tmp_path / "gen.jsonl"
    path.write_text('{"response": "abc", "source": "human"}\n', encoding="utf-8")
    samples = load_dataset(path, source="recursive", generation=3, split="validation")
    assert samples[0].source == "recursive"
    assert samples[0].generation == 3
    assert samples[0].split == "validation"


def test_deterministic_ids_across_loads(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    ids_first = [s.sample_id for s in load_dataset(path)]
    ids_second = [s.sample_id for s in load_dataset(path)]
    assert ids_first == ids_second == ["d-0000000", "d-0000001", "d-0000002"]


def test_write_jsonl_roundtrips(tmp_path):
    src = [
        Sample(sample_id="x-1", response="hi", source="synthetic", generation=1),
        Sample(sample_id="x-2", prompt="q", response="a"),
    ]
    out = write_jsonl(src, tmp_path / "out.jsonl")
    reloaded = load_dataset(out)
    assert [s.sample_id for s in reloaded] == ["x-1", "x-2"]
    assert reloaded[0].source == "synthetic" and reloaded[0].generation == 1
    # every written line is valid JSON with the full schema
    line0 = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert set(line0) == set(Sample.field_names())


def test_validation_rejects_bad_source():
    with pytest.raises(ValueError):
        Sample(sample_id="x", response="t", source="not_a_source").validate()


def test_validation_rejects_empty_content():
    with pytest.raises(ValueError):
        Sample(sample_id="x").validate()  # no prompt and no response


def test_validation_rejects_out_of_range_ratio():
    with pytest.raises(ValueError):
        Sample(sample_id="x", response="t", contamination_ratio=1.5).validate()
