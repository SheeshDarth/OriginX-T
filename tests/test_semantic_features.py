"""Tests for src/features/semantic.py on stub embeddings with hand-checked values, plus one
check of the real embedding model (skipped without it)."""

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from src.features import semantic as sem  # noqa: E402

DIM = sem.N_CLUSTERS


def _stub(text):
    """Text "c<i>-<j>" embeds as basis vector i: one tight, well-separated cluster per i."""
    v = np.zeros(DIM, np.float32)
    v[int(text[1:].split("-")[0])] = 1.0
    return v


@pytest.fixture
def stub(monkeypatch):
    sem._fit.cache_clear()
    monkeypatch.setattr(sem, "embed", lambda texts, **kw: np.stack([_stub(t) for t in texts]))
    monkeypatch.setattr(sem, "_mauve", lambda ref, emb: None)


def _texts(counts):
    return [f"c{i}-{j}" for i, n in counts.items() for j in range(n)]


# clusters 0-9 hold 8 texts each, 10-19 hold 2: the tail is 10-19, 20 of 100 reference texts
REF = _texts({**{i: 8 for i in range(10)}, **{i: 2 for i in range(10, 20)}})
CTX = {"reference_texts": REF}


def test_knn_distances_by_hand():
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    d = sem.knn_distances(np.stack([e1, e1, e2]), k=1)
    assert d.tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert sem.knn_distances(np.stack([e1, e1, e2]), k=5).tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert math.isnan(sem.knn_distances(np.stack([e1]))[0])


def test_the_reference_itself_keeps_its_tail_and_spread(stub):
    out = sem.semantic(REF, CTX)
    assert out["tail_retention"] == pytest.approx(1.0)
    p = np.array([0.08] * 10 + [0.02] * 10)
    assert out["cluster_entropy"] == pytest.approx(-(p * np.log(p)).sum() / math.log(20))
    # every text has 1 (tail) or 7 (head) exact copies; k=10 neighbours, the rest at distance 1
    assert out["knn_dist"] == pytest.approx((80 * 3 / 10 + 20 * 9 / 10) / 100)


def test_collapse_empties_the_tail_and_narrows_the_spread(stub):
    head_only = _texts({i: 10 for i in range(10)})
    out = sem.semantic(head_only, CTX)
    assert out["tail_retention"] == 0.0
    assert out["cluster_entropy"] == pytest.approx(math.log(10) / math.log(20))
    one_cluster = sem.semantic(_texts({0: 50}), CTX)
    assert one_cluster["cluster_entropy"] == 0.0 and one_cluster["knn_dist"] == 0.0
    # over-representing the tail scores above 1
    assert sem.semantic(_texts({10: 5, 0: 5}), CTX)["tail_retention"] == pytest.approx(0.5 / 0.2)


def test_without_a_reference_only_the_reference_free_column_is_computed(stub):
    out = sem.semantic(REF, {})
    assert out["knn_dist"] is not None
    assert out["cluster_entropy"] is out["tail_retention"] is out["mauve"] is None
    assert sem.semantic(["c0-0"], CTX) == dict.fromkeys(sem.COLUMNS)  # one text: nothing to measure


def test_cluster_assignments_and_per_text(stub):
    out = sem.cluster_assignments(["c3-0", "c3-1", "c15-0"], CTX)
    assert out["n_clusters"] == 20 and len(out["tail"]) == 10
    a, b, c = out["labels"]
    assert a == b != c and c in out["tail"] and a not in out["tail"]
    with pytest.raises(ValueError, match="reference_texts"):
        sem.cluster_assignments(["c0-0"], {})
    assert sem.per_text(["c0-0", "c0-1", "c1-0"], {}) == pytest.approx([0.5, 0.5, 1.0])


def test_mauve_is_high_for_the_same_distribution_and_low_for_a_different_one():
    pytest.importorskip("mauve")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 8)), rng.normal(size=(300, 8))
    far = rng.normal(loc=4.0, size=(300, 8))
    same, diff = sem._mauve(a, b), sem._mauve(a, far)
    assert 0 <= diff < same <= 1 and same > 0.9 and diff < 0.2


def test_real_embeddings_are_unit_length_and_meaningful():
    pytest.importorskip("sentence_transformers")
    try:
        sem._load("cpu")
    except OSError:
        pytest.skip("embedding model not available offline")
    texts = ["The cat sat on the mat.", "A cat was sitting on a mat.", "Stock markets fell sharply today."]
    emb = sem.embed(texts)
    assert emb.shape == (3, 384)
    assert np.linalg.norm(emb, axis=1) == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)
    assert emb[0] @ emb[1] > emb[0] @ emb[2] + 0.3


def test_the_order_of_the_texts_never_changes_the_numbers(monkeypatch):
    # random, overlapping embeddings: k-means is order-sensitive here, so only sorting saves it
    sem._fit.cache_clear()
    vecs = {}

    def fake(texts, **kw):
        for t in texts:
            if t not in vecs:
                v = np.random.default_rng(int(t[1:])).normal(size=16)
                vecs[t] = (v / np.linalg.norm(v)).astype(np.float32)
        return np.stack([vecs[t] for t in texts])

    monkeypatch.setattr(sem, "embed", fake)
    monkeypatch.setattr(sem, "_mauve", lambda ref, emb: float(ref[:, 0] @ np.arange(len(ref))
                                                               + emb[:, 0] @ np.arange(len(emb))))
    ref = [f"t{i}" for i in range(300)]
    cell = ref[:150] + [f"t{i}" for i in range(1000, 1150)]
    first = sem.semantic(cell, {"reference_texts": ref})
    for k in range(3):
        rng = __import__("random").Random(k)
        sem._fit.cache_clear()
        again = sem.semantic(rng.sample(cell, len(cell)), {"reference_texts": rng.sample(ref, len(ref))})
        assert again == first  # exactly, every column (the stub MAUVE is order-sensitive too)