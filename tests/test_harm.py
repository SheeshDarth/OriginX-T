"""Tests for the X1 harm scores (HANDBOOK §9.3, B7)."""

import json
import random

import pytest

from src.finetune.collapse import load_holdout
from src.ingestion.loaders import write_jsonl
from src.ingestion.schema import Sample
from src.mitigation.harm import (
    METHODS, harm_report, lora_grad_harm, merge_rows, neg_cosine, repr_sim_harm, score_cell, target_texts,
)

KEYS = {"schema", "dataset", "cell", "sample_id", "method", "score", "higher_is", "true_source"}


def test_neg_cosine_is_higher_for_vectors_pointing_away_from_the_target():
    assert neg_cosine([[1, 0], [0, 1], [-2, 0], [0, 0]], [3, 0]) == [-1.0, 0.0, 1.0, None]
    assert neg_cosine([[1, 1]], [0, 0]) == [None]


def test_score_cell_writes_scores_1_rows_in_sample_order_without_reading_the_label():
    samples = [Sample(sample_id=f"s{i}", response=f"text {i}", source=src)
               for i, src in enumerate(["human", "synthetic", "human"])]
    seen = {}

    def stub(name):
        def f(model_dir, texts, targets, **kw):
            seen[name] = (model_dir, list(texts), list(targets), kw)
            return [0.1, 0.9, None]
        return f

    rows = score_cell("wikitext2", "synthetic_r0.5_s0", samples, ["t0", "t1", "t2"], ["h"], "ckpt",
                      train={"lora_r": 8, "max_len": 64, "batch_size": 4}, device="cpu",
                      lora_grad=stub("grad"), repr_sim=stub("repr"))
    assert set(rows) == set(METHODS)
    for method, rs in rows.items():
        assert [set(r) for r in rs] == [KEYS] * 3
        assert [r["sample_id"] for r in rs] == ["s0", "s1", "s2"]
        assert [r["true_source"] for r in rs] == ["human", "synthetic", "human"]
        assert all(r["schema"] == "scores/1" and r["higher_is"] == "worse" and r["method"] == method for r in rs)
        assert [r["score"] for r in rs] == [0.1, 0.9, None]
    assert seen["grad"] == ("ckpt", ["t0", "t1", "t2"], ["h"], {"lora_r": 8, "max_len": 64, "device": "cpu"})
    assert seen["repr"][3] == {"max_len": 64, "batch_size": 4, "device": "cpu"}


def test_rescoring_a_cell_replaces_only_its_rows():
    old = [{"cell": "a", "v": 1}, {"cell": "b", "v": 1}]
    assert merge_rows(old, [{"cell": "a", "v": 2}]) == [{"cell": "b", "v": 1}, {"cell": "a", "v": 2}]


def test_the_harm_target_never_uses_the_texts_recovery_is_measured_on(tmp_path):
    words = "one two three four five six seven eight nine ten eleven twelve".split()
    rng = random.Random(0)
    rows = [Sample(sample_id=f"h{i}", response=" ".join(rng.choices(words, k=25)) + f" {i}",
                   split="hidden_holdout") for i in range(60)]
    write_jsonl(rows, tmp_path / "holdout.jsonl")
    d = {"holdout": str(tmp_path / "holdout.jsonl"), "n_holdout": 20, "min_words": 5, "max_words": 150}
    targets = target_texts(d)
    assert len(targets) == 20 and not set(targets) & set(load_holdout(d))
    assert target_texts(d) == targets, "deterministic"


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A 2-layer random GPT-2 with GPT-2's tokenizer (skipped when either isn't available offline)."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    except OSError:
        pytest.skip("gpt2 tokenizer not available offline")
    import torch

    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=len(tok), n_positions=64, n_embd=32, n_layer=2, n_head=2)
    path = tmp_path_factory.mktemp("tiny")
    transformers.GPT2LMHeadModel(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


A = "The river rose in the spring and flooded the lower fields of the valley."
B = "Quantum chromodynamics describes the strong interaction between quarks and gluons."


def test_lora_grad_harm_is_minus_one_for_a_sample_identical_to_the_target(tiny_model):
    pytest.importorskip("peft")
    scores = lora_grad_harm(tiny_model, [A, B], [A, A], lora_r=4, max_len=32)
    assert scores[0] == pytest.approx(-1.0, abs=1e-6)
    assert -1.0 <= scores[1] <= 1.0 and scores[1] > scores[0]


def test_lora_grad_harm_does_not_depend_on_scoring_order(tiny_model):
    pytest.importorskip("peft")
    forward = lora_grad_harm(tiny_model, [A, B], [A], lora_r=4, max_len=32)
    backward = lora_grad_harm(tiny_model, [B, A], [A], lora_r=4, max_len=32)
    assert forward == backward[::-1]


def test_repr_sim_harm_separates_texts_like_and_unlike_the_target(tiny_model):
    scores = repr_sim_harm(tiny_model, [A, B], [A], max_len=32, batch_size=2)
    assert scores[0] == pytest.approx(-1.0, abs=1e-6) and scores[1] == pytest.approx(1.0, abs=1e-6)


def test_harm_report_covers_every_scored_cell_not_only_the_latest_run(tmp_path):
    def rows(method, cell, scores):
        return [{"schema": "scores/1", "dataset": "toy", "cell": cell, "sample_id": f"{cell}-{i}", "method": method,
                 "score": s, "higher_is": "worse", "true_source": "synthetic" if i % 2 else "human"}
                for i, s in enumerate(scores)]

    def write(method, data):
        (tmp_path / method).mkdir(exist_ok=True)
        (tmp_path / method / "toy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data), encoding="utf-8")

    for m in METHODS:  # two cells scored in two separate runs: both must be in the report
        write(m, rows(m, "a_r0.5_s0", [0.0, 1.0, 0.1, 0.9]) + rows(m, "b_r0.5_s0", [0.2, 0.8, 0.3, 0.7]))
    write("risk_reflm", rows("risk_reflm", "a_r0.5_s0", [1.0, 0.0, 0.9, 0.1])
          + rows("risk_reflm", "c_r0.5_s0", [0.0, 1.0, 0.0, 1.0]))  # c has no harm scores: left out
    report = harm_report(tmp_path, "toy")
    assert report["cells"] == ["a_r0.5_s0", "b_r0.5_s0"]
    assert report["harm_lora_grad"]["pooled"]["auroc"] == 1.0 and report["harm_lora_grad"]["pooled"]["n"] == 8
    assert report["risk_reflm"]["pooled"]["n"] == 4 and report["risk_reflm"]["pooled"]["auroc"] == 0.0
    assert "risk_knn" not in report
    saved = (tmp_path / "toy_harm_eval.json").read_bytes()
    assert b"\r" not in saved and json.loads(saved) == report
