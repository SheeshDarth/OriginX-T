"""Semantic features of a dataset (HANDBOOK §10.7): the ``semantic`` feature family.

Where a cell's texts sit in sentence-embedding space (``all-MiniLM-L6-v2``, unit vectors),
compared with the seed's clean human draw (the ratio-0 training set, passed by the suite
as ``ctx["reference_texts"]``). Collapse narrows a distribution: texts crowd together and
the rare regions of the human data empty out first (Shumailov et al., 2024). Columns (the
suite writes them as ``f_sem_<name>``):

- ``knn_dist``: mean cosine distance from each text to its ``K_NN`` nearest neighbours in
  the cell. Lower = more crowded, more redundant. Needs no reference.
- ``cluster_entropy``: entropy of the cell's spread over the reference's ``N_CLUSTERS``
  k-means clusters, divided by ln(N_CLUSTERS): 1 = evenly spread, 0 = all in one cluster.
- ``tail_retention``: the cell's share of the reference's *tail* clusters (those holding
  less than an even share, 1/N_CLUSTERS, of the reference) over the reference's own share
  of them. 1 = the rare regions are kept as often as in human data, 0 = emptied.
- ``mauve``: MAUVE (Pillutla et al., 2021) between the reference and the cell embeddings.
  1 = indistinguishable distributions.

The clusters are fitted once per reference (fixed k and seed), so every cell of a seed is
measured against the same clusters. Without ``reference_texts`` the reference columns are
``None``; ``mauve`` is ``None`` too when ``mauve-text`` is not installed.

``embed`` runs the model (sentence-transformers imported inside it); the rest is numpy and
scikit-learn on embeddings, so tests stub ``embed``. ``cluster_assignments`` gives each
text's cluster and the tail clusters (Track B's rare-cluster preservation, §9.4), and
``per_text`` each text's kNN distance (a per-sample score, §7.3, C10).

``ctx``: ``reference_texts`` (list of str), ``device`` (default ``"cpu"``), ``batch_size`` (64).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, Optional

VERSION = 1  # bump when a column's definition changes (the suite recomputes)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K_NN = 10
N_CLUSTERS = 20
SEED = 0
COLUMNS = ("knn_dist", "cluster_entropy", "tail_retention", "mauve")


@lru_cache(maxsize=2)
def _load(device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device=device)


# ponytail: unbounded, keyed by text; a grid's ~18k distinct texts are ~28 MB of float32.
_CACHE: dict[str, Any] = {}


def embed(texts: Sequence[str], *, device: str = "cpu", batch_size: int = 64) -> Any:
    """Unit-length embeddings of ``texts``, one row each (numpy float32)."""
    import numpy as np

    todo = list(dict.fromkeys(t for t in texts if t not in _CACHE))
    if todo:
        vecs = _load(device).encode(todo, batch_size=batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        _CACHE.update(zip(todo, vecs.astype(np.float32)))
    return np.stack([_CACHE[t] for t in texts]) if texts else np.zeros((0, 0), np.float32)


def knn_distances(emb: Any, k: int = K_NN) -> Any:
    """Each row's mean cosine distance to its ``k`` nearest other rows (fewer if the set is small)."""
    import numpy as np

    n = len(emb)
    if n < 2:
        return np.full(n, np.nan)
    dist = 1.0 - emb @ emb.T
    np.fill_diagonal(dist, np.inf)
    k = min(k, n - 1)
    return np.sort(dist, axis=1)[:, :k].mean(axis=1)


@lru_cache(maxsize=8)
def _fit(ref_key: tuple[str, ...], device: str, batch_size: int) -> tuple[Any, Any]:
    """k-means centroids of a reference set, and the indices of its tail clusters."""
    import numpy as np
    from sklearn.cluster import KMeans

    ref = embed(list(ref_key), device=device, batch_size=batch_size)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10).fit(ref)
    share = np.bincount(km.labels_, minlength=N_CLUSTERS) / len(ref)
    return km.cluster_centers_, np.flatnonzero(share < 1.0 / N_CLUSTERS)


def _assign(emb: Any, centers: Any) -> Any:
    return ((emb[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)


def _entropy(labels: Any) -> float:
    import numpy as np

    p = np.bincount(labels, minlength=N_CLUSTERS) / len(labels)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(N_CLUSTERS))


def _mauve(ref: Any, emb: Any) -> Optional[float]:
    try:
        import mauve
    except ImportError:
        return None
    out = mauve.compute_mauve(p_features=ref, q_features=emb, seed=SEED, verbose=False)
    return float(out.mauve)


def _settings(ctx: Optional[Mapping[str, Any]]) -> tuple[Optional[tuple[str, ...]], dict[str, Any]]:
    ctx = ctx or {}
    ref = ctx.get("reference_texts")
    kw = {"device": ctx.get("device", "cpu"), "batch_size": ctx.get("batch_size", 64)}
    return (tuple(ref) if ref else None), kw


def semantic(texts: list[str], ctx: Optional[Mapping[str, Any]] = None) -> dict[str, Optional[float]]:
    """The ``semantic`` family: embedding density, cluster spread, tail retention, MAUVE."""
    out: dict[str, Optional[float]] = dict.fromkeys(COLUMNS)
    if len(texts) < 2:
        return out
    ref_key, kw = _settings(ctx)
    emb = embed(texts, **kw)
    out["knn_dist"] = float(knn_distances(emb).mean())
    if ref_key is None or len(ref_key) < N_CLUSTERS:
        return out
    centers, tail = _fit(ref_key, **kw)
    labels = _assign(emb, centers)
    ref_labels = _assign(embed(list(ref_key), **kw), centers)
    out["cluster_entropy"] = _entropy(labels)
    if len(tail):
        out["tail_retention"] = float(
            (labels[:, None] == tail).any(1).mean() / (ref_labels[:, None] == tail).any(1).mean())
    out["mauve"] = _mauve(embed(list(ref_key), **kw), emb)
    return out


def cluster_assignments(texts: list[str], ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Each text's cluster among the reference's, and which clusters are the tail (for §9.4).

    Returns ``{"labels": [int per text], "tail": [cluster ids], "n_clusters": N_CLUSTERS}``.
    """
    ref_key, kw = _settings(ctx)
    if ref_key is None or len(ref_key) < N_CLUSTERS:
        raise ValueError(f"ctx['reference_texts'] needs at least {N_CLUSTERS} texts")
    centers, tail = _fit(ref_key, **kw)
    labels = _assign(embed(texts, **kw), centers) if texts else []
    return {"labels": [int(x) for x in labels], "tail": [int(x) for x in tail], "n_clusters": N_CLUSTERS}


def per_text(texts: list[str], ctx: Optional[Mapping[str, Any]] = None) -> list[Optional[float]]:
    """Each text's mean cosine distance to its nearest neighbours in ``texts`` (low = redundant)."""
    _, kw = _settings(ctx)
    d = knn_distances(embed(texts, **kw)) if texts else []
    return [None if math.isnan(x) else float(x) for x in d]
