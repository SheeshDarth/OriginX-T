"""Tests for the corpus normalizer.

These exercise the pure row-normalizing core with fake in-memory rows, so the
suite runs offline and needs neither `datasets` nor a network.
"""

import pytest

from src.ingestion.normalize_corpora import _normalize_rows


def test_skips_blank_and_whitespace_rows():
    rows = [{"text": "real line"}, {"text": ""}, {"text": "   \n "}, {"text": "another"}]
    samples = _normalize_rows(rows, name="wikitext2", split="train")
    assert [s.response for s in samples] == ["real line", "another"]


def test_marks_rows_as_clean_human_baseline():
    samples = _normalize_rows([{"text": "hello"}], name="wikitext2", split="train")
    assert samples[0].source == "human"
    assert samples[0].generation == 0
    assert samples[0].contamination_ratio == 0.0
    assert samples[0].prompt == ""  # raw corpora are response-only


def test_split_is_recorded_on_every_sample():
    samples = _normalize_rows(
        [{"text": "a"}, {"text": "b"}], name="tinystories", split="validation"
    )
    assert {s.split for s in samples} == {"validation"}


def test_ids_are_deterministic_and_keyed_on_source_row_index():
    rows = [{"text": "first"}, {"text": ""}, {"text": "third"}]
    first = _normalize_rows(rows, name="wikitext2", split="test")
    second = _normalize_rows(rows, name="wikitext2", split="test")
    ids = [s.sample_id for s in first]
    # index 1 was blank and skipped, so ids jump 0 -> 2 (stable across runs)
    assert ids == ["wikitext2_test-0000000", "wikitext2_test-0000002"]
    assert ids == [s.sample_id for s in second]


def test_max_rows_caps_kept_samples_not_scanned_rows():
    rows = [{"text": ""}] + [{"text": f"line {i}"} for i in range(10)]
    samples = _normalize_rows(rows, name="c", split="train", max_rows=3)
    assert len(samples) == 3  # the leading blank does not count toward the cap


def test_custom_text_field_and_plain_string_rows():
    from_dict = _normalize_rows([{"content": "x"}], name="c", split="train", text_field="content")
    from_str = _normalize_rows(["plain row"], name="c", split="train")
    assert from_dict[0].response == "x"
    assert from_str[0].response == "plain row"


def test_rejects_unknown_split():
    # validation happens per sample via the shared schema contract
    with pytest.raises(ValueError):
        _normalize_rows([{"text": "x"}], name="c", split="not_a_split")
