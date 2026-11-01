"""Tests for src/features/model.py: hand-checked arithmetic on stub token stats, plus one
check of the token scoring against the real reference model (skipped without it)."""

import math

import pytest

from src.features import model as fm

# two texts: losses [1, 3] with ranks [1, 20]; losses [2, 2, 2] with ranks [1, 1, 1]
STATS = [([1.0, 3.0], [1, 20]), ([2.0, 2.0, 2.0], [1, 1, 1])]


def test_aggregate_by_hand():
    out = fm.aggregate(STATS)
    assert set(out) == set(fm.COLUMNS)
    # per-text mean loss: 2 and 2
    assert out["loss_mean"] == pytest.approx(2.0)
    assert out["loss_std"] == pytest.approx(0.0)
    assert out["loss_p90"] == pytest.approx(2.0)
    # per-text mean ln(rank): (0 + ln 20) / 2 and 0
    assert out["rank_log_mean"] == pytest.approx(math.log(20) / 4)
    # share in the top 10: 1/2 and 1
    assert out["rank_top10"] == pytest.approx(0.75)
    # per-text spread of ln(rank): ln(20) / 2 (two values 0 and ln 20) and 0
    assert out["rank_instability"] == pytest.approx(math.log(20) / 4)


def test_p90_interpolates_and_every_text_weighs_the_same():
    stats = [([float(v)], [1]) for v in range(1, 11)]  # per-text losses 1..10
    out = fm.aggregate(stats)
    assert out["loss_p90"] == pytest.approx(9.1)  # inclusive method: 1 + 0.9 * 9
    assert out["loss_mean"] == pytest.approx(5.5)
    # a long text counts once, like a short one
    assert fm.aggregate([([1.0] * 100, [1] * 100), ([3.0], [1])])["loss_mean"] == pytest.approx(2.0)


def test_unscorable_texts_are_left_out_and_nothing_scorable_gives_none():
    assert fm.aggregate([*STATS, ([], [])]) == fm.aggregate(STATS)
    assert fm.aggregate([([], [])]) == dict.fromkeys(fm.COLUMNS)
    assert fm.aggregate([]) == dict.fromkeys(fm.COLUMNS)


def test_family_and_per_text_pass_ctx_settings_to_the_scorer(monkeypatch):
    seen = {}

    def fake(texts, **kw):
        seen.update(kw)
        return [STATS[0] if t == "a" else ([], []) for t in texts]

    monkeypatch.setattr(fm, "token_stats", fake)
    assert fm.model(["a"], {"device": "cuda", "batch_size": 4, "max_len": 64, "cell": "x"}) == fm.aggregate(
        [STATS[0]])
    assert seen == {"device": "cuda", "batch_size": 4, "max_len": 64}
    fm.model(["a"], None)
    assert seen == {"device": "cpu", "batch_size": 8, "max_len": 256}
    rows = fm.per_text(["a", "b"], {})
    assert rows[0] == {"loss": 2.0, "rank_log": math.log(20) / 2, "top10": 0.5,
                       "instability": pytest.approx(math.log(20) / 2)}
    assert rows[1] is None


def test_token_stats_match_the_reference_model_itself():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        tok, lm = fm._load("cpu")
    except OSError:
        pytest.skip("reference model not available offline")
    texts = ["The cat sat on the mat.", "Colourless green ideas sleep furiously in the night."]
    fm._CACHE.clear()
    stats = fm.token_stats(texts, batch_size=2)  # one padded batch: padding must not leak in
    for text, (losses, ranks) in zip(texts, stats):
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            out = lm(ids, labels=ids)
        assert len(losses) == len(ranks) == ids.shape[1] - 1
        assert sum(losses) / len(losses) == pytest.approx(out.loss.item(), rel=1e-4)
        logits = out.logits[0, :-1]
        top = logits.argmax(-1)
        for i, r in enumerate(ranks):
            assert (r == 1) == bool(top[i] == ids[0, i + 1])
            assert r == int((logits[i] > logits[i, ids[0, i + 1]]).sum()) + 1
    # cached: a second call scores nothing new and returns the same values
    assert fm.token_stats(texts, batch_size=2) == stats
