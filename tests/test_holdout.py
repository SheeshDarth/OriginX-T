"""Tests for hidden-holdout carving and the leakage gate."""

import pytest

from src.ingestion import Sample, write_jsonl
from src.ingestion.check_leakage import check_files, find_leaks, text_fingerprint
from src.ingestion.holdout import carve_holdout, is_holdout


def make(n, prefix="c_train"):
    return [Sample(sample_id=f"{prefix}-{i:07d}", response=f"line {i}") for i in range(n)]


# --- carving ---------------------------------------------------------------


def test_carve_splits_and_labels_holdout():
    kept, holdout = carve_holdout(make(200), fraction=0.1)
    assert len(kept) + len(holdout) == 200
    assert holdout, "expected a non-empty holdout"
    assert all(s.split == "hidden_holdout" for s in holdout)
    assert all(s.split == "train" for s in kept)  # kept samples untouched


def test_carve_is_deterministic_across_runs():
    samples = make(200)
    first = {s.sample_id for s in carve_holdout(samples, fraction=0.1)[1]}
    second = {s.sample_id for s in carve_holdout(samples, fraction=0.1)[1]}
    assert first == second


def test_assignment_is_stable_when_corpus_grows():
    """The key property: adding data must not move existing samples."""
    small = {s.sample_id for s in carve_holdout(make(100), fraction=0.2)[1]}
    grown = {s.sample_id for s in carve_holdout(make(300), fraction=0.2)[1]}
    # every id held out before is still held out after the corpus tripled
    assert small <= grown


def test_carve_order_independent():
    samples = make(100)
    forward = {s.sample_id for s in carve_holdout(samples, fraction=0.15)[1]}
    reverse = {s.sample_id for s in carve_holdout(list(reversed(samples)), fraction=0.15)[1]}
    assert forward == reverse


def test_fraction_extremes_and_validation():
    samples = make(50)
    assert carve_holdout(samples, fraction=0.0)[1] == []
    assert len(carve_holdout(samples, fraction=1.0)[1]) == 50
    with pytest.raises(ValueError):
        is_holdout("x", fraction=1.5)


def test_different_salt_gives_different_split():
    samples = make(200)
    a = {s.sample_id for s in carve_holdout(samples, fraction=0.2, salt="a")[1]}
    b = {s.sample_id for s in carve_holdout(samples, fraction=0.2, salt="b")[1]}
    assert a != b


# --- leakage ---------------------------------------------------------------


def test_clean_split_reports_no_leakage():
    kept, holdout = carve_holdout(make(200), fraction=0.1)
    report = find_leaks(holdout, kept)
    assert report.clean and not report
    assert "OK" in report.summary()


def test_detects_id_overlap():
    shared = Sample(sample_id="dup-1", response="held out text", split="hidden_holdout")
    train = [Sample(sample_id="dup-1", response="completely different text")]
    report = find_leaks([shared], train)
    assert report.id_overlap == ["dup-1"] and report


def test_detects_text_reappearing_under_a_new_id():
    """The realistic failure: a generator copies holdout text with a fresh id."""
    holdout = [Sample(sample_id="h-1", response="The quick brown fox.", split="hidden_holdout")]
    train = [Sample(sample_id="t-999", response="the   QUICK brown fox.")]  # case/space differ
    report = find_leaks(holdout, train)
    assert report.text_overlap == ["h-1"]
    assert report.id_overlap == []  # ids differ, so only the text check catches it


def test_fingerprint_folds_case_and_whitespace():
    a = Sample(sample_id="1", response="Hello   World")
    b = Sample(sample_id="2", response="hello world")
    c = Sample(sample_id="3", response="different")
    assert text_fingerprint(a) == text_fingerprint(b)
    assert text_fingerprint(a) != text_fingerprint(c)


def test_check_files_end_to_end(tmp_path):
    kept, holdout = carve_holdout(make(200), fraction=0.1)
    h = write_jsonl(holdout, tmp_path / "holdout.jsonl")
    t = write_jsonl(kept, tmp_path / "train.jsonl")
    assert check_files(h, [t]).clean

    # now poison the training file with a holdout row
    poisoned = write_jsonl(kept + holdout[:1], tmp_path / "poisoned.jsonl")
    assert not check_files(h, [poisoned]).clean
