"""Tests for the white-box signals.

Three tiers, so CI (which has only ruff, pytest and pyyaml) still runs most of it:
the formulas and the orchestration need nothing; the streaming statistics need numpy; the
real model path needs torch and transformers and uses a tiny randomly initialised GPT-2.
"""

import json
import math
import random
from pathlib import Path

import pytest
import yaml

from src.evaluation import whitebox as wb

REPO = Path(__file__).resolve().parent.parent


# --- formulas (standard library only) -------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 7, 64])
def test_effective_rank_of_k_equal_singular_values_is_k(k):
    assert wb.effective_rank([3.0] * k) == pytest.approx(k)


def test_effective_rank_one_direction_and_zero_padding():
    assert wb.effective_rank([5.0]) == pytest.approx(1.0)
    assert wb.effective_rank([5.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert wb.effective_rank([1.0, 1.0, 0.0, 0.0]) == pytest.approx(2.0)  # zeros do not count


def test_effective_rank_is_scale_free_and_between_one_and_the_count():
    s = [4.0, 2.0, 1.0, 0.5]
    assert wb.effective_rank([100 * v for v in s]) == pytest.approx(wb.effective_rank(s))
    assert 1.0 < wb.effective_rank(s) < len(s)


def test_effective_rank_known_value():
    # p = (0.5, 0.25, 0.25): H = 1.5 ln 2, so exp(H) = 2 ** 1.5
    assert wb.effective_rank([2.0, 1.0, 1.0]) == pytest.approx(2**1.5)


def test_effective_rank_shrinks_as_one_direction_takes_over():
    ranks = [wb.effective_rank([w, 1, 1, 1]) for w in (1, 2, 5, 20, 100)]
    assert ranks == sorted(ranks, reverse=True) and ranks[0] == pytest.approx(4.0)


def test_participation_ratio():
    assert wb.participation_ratio([2.0] * 5) == pytest.approx(5.0)
    assert wb.participation_ratio([9.0]) == pytest.approx(1.0)
    assert wb.participation_ratio([4.0, 2.0, 2.0]) == pytest.approx(64 / 24)  # (sum 8)^2 / (16+4+4)
    assert wb.participation_ratio([1000.0, 1.0, 1.0]) < 1.01  # one dominant eigenvalue


def test_stable_rank():
    assert wb.stable_rank([1.0] * 6) == pytest.approx(6.0)  # an identity matrix
    assert wb.stable_rank([7.0, 0.0, 0.0]) == pytest.approx(1.0)  # rank one
    assert wb.stable_rank([3.0, 1.0]) == pytest.approx(10 / 9)


@pytest.mark.parametrize("fn", [wb.effective_rank, wb.participation_ratio, wb.stable_rank])
@pytest.mark.parametrize("bad", [[], [0.0, 0.0], [1.0, -1.0], [1.0, float("nan")]])
def test_formulas_reject_empty_zero_negative_and_nan(fn, bad):
    with pytest.raises(ValueError):
        fn(bad)


# --- streaming statistics and weights (numpy) -----------------------------------------------------

def _np():
    return pytest.importorskip("numpy")


def _chunks(x, sizes):
    start = 0
    for size in sizes:
        yield x[start : start + size]
        start += size
    yield x[start:]


def test_streaming_spectrum_matches_the_batch_covariance():
    np = _np()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 8)) @ rng.normal(size=(8, 8)) + 5.0  # correlated, with a mean
    stats = wb.RepresentationStats(8)
    for chunk in _chunks(x, [37, 100, 1, 200]):
        stats.update(chunk)

    expected = np.linalg.eigvalsh(np.cov(x.T, bias=True))[::-1]
    assert stats.n == 500
    assert np.allclose(stats.eigenvalues(), expected)
    res = stats.result()
    assert res["effective_rank"] == pytest.approx(wb.effective_rank(np.sqrt(expected)))
    assert res["participation_ratio"] == pytest.approx(wb.participation_ratio(expected))


def test_anisotropy_is_the_exact_mean_pairwise_cosine():
    np = _np()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 6)) + 1.5
    stats = wb.RepresentationStats(6)
    stats.update(x[:25])
    stats.update(x[25:])

    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    cos = unit @ unit.T
    brute = (cos.sum() - np.trace(cos)) / (len(x) * (len(x) - 1))
    assert stats.result()["anisotropy"] == pytest.approx(brute)


def test_anisotropy_extremes():
    np = _np()
    same = wb.RepresentationStats(4)
    same.update(np.tile([1.0, 2.0, 3.0, 4.0], (10, 1)) * np.arange(1, 11)[:, None])  # one direction
    assert same.result()["anisotropy"] == pytest.approx(1.0)

    opposite = wb.RepresentationStats(2)  # n points, half x and half -x: the mean cosine is -1/(n-1)
    opposite.update(np.array([[1.0, 0.0], [-1.0, 0.0]] * 5))
    assert opposite.result()["anisotropy"] == pytest.approx(-1 / 9)

    rng = np.random.default_rng(2)
    isotropic = wb.RepresentationStats(32)
    isotropic.update(rng.normal(size=(6000, 32)))
    assert abs(isotropic.result()["anisotropy"]) < 0.01


def test_anisotropy_ignores_zero_vectors():
    np = _np()
    stats = wb.RepresentationStats(2)
    stats.update(np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0], [3.0, 0.0]]))
    assert stats.result()["anisotropy"] == pytest.approx(1.0)  # the zero row has no direction


def test_a_low_rank_cloud_has_low_effective_rank_and_a_full_rank_one_does_not():
    np = _np()
    rng = np.random.default_rng(3)
    low = wb.RepresentationStats(16)
    low.update(rng.normal(size=(2000, 3)) @ rng.normal(size=(3, 16)))
    full = wb.RepresentationStats(16)
    full.update(rng.normal(size=(20000, 16)))

    assert low.result()["effective_rank"] == pytest.approx(3.0, abs=0.35)
    assert low.result()["participation_ratio"] <= 3.0 + 1e-6
    assert full.result()["effective_rank"] > 15.0
    assert full.result()["participation_ratio"] > 15.0


def test_a_representation_with_no_variance_has_rank_zero_but_full_anisotropy():
    np = _np()
    stats = wb.RepresentationStats(3)
    stats.update(np.tile([1.0, 2.0, 3.0], (50, 1)))  # every token the same vector: total collapse
    res = stats.result()
    assert res["effective_rank"] == 0.0 and res["participation_ratio"] == 0.0 and res["degenerate"]
    assert res["anisotropy"] == pytest.approx(1.0) and res["n_tokens"] == 50

    zeros = wb.RepresentationStats(3)  # no direction at all: nothing to compare, flagged
    zeros.update(np.zeros((10, 3)))
    res = zeros.result()
    assert (res["effective_rank"], res["participation_ratio"], res["anisotropy"], res["degenerate"]) == (0.0, 0.0, 0.0, True)


def test_stats_reject_wrong_shapes_and_too_few_tokens():
    np = _np()
    stats = wb.RepresentationStats(3)
    with pytest.raises(ValueError, match="shape"):
        stats.update(np.zeros((4, 2)))
    stats.update(np.ones((1, 3)))
    with pytest.raises(ValueError, match="at least 2"):
        stats.result()


def test_activation_max_and_dead_fraction():
    np = _np()
    acts = wb.ActivationMax(4)
    acts.update(np.array([[0.0, 0.5, -1.0, 0.009], [-0.2, 0.7, -0.3, 0.005]]))
    acts.update(np.zeros((0, 4)))  # an empty batch changes nothing
    acts.update(np.array([[0.0, 0.1, -0.5, 0.01]]))
    # per-unit maxima: 0.0, 0.7, -0.3, 0.01  -> with eps 0.01 the units at <= 0.01 are dead
    assert acts.dead_fraction(0.01) == pytest.approx(3 / 4)
    assert acts.dead_fraction(0.009) == pytest.approx(2 / 4)
    assert acts.dead_fraction(-0.5) == 0.0
    with pytest.raises(ValueError, match="shape"):
        acts.update(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="no tokens"):
        wb.ActivationMax(2).dead_fraction(0.01)


def test_weight_stats():
    np = _np()
    ident = wb.weight_stats(np.eye(6))
    assert ident["stable_rank"] == pytest.approx(6.0) and ident["effective_rank"] == pytest.approx(6.0)
    assert ident["frobenius"] == pytest.approx(math.sqrt(6)) and ident["spectral"] == pytest.approx(1.0)

    rank1 = wb.weight_stats(np.outer([1.0, 2.0, 3.0], [4.0, 5.0]))
    assert rank1["stable_rank"] == pytest.approx(1.0) and rank1["effective_rank"] == pytest.approx(1.0)

    zero = wb.weight_stats(np.zeros((3, 4)))  # a collapsed matrix is rank 0, not an error
    assert (zero["stable_rank"], zero["effective_rank"], zero["frobenius"], zero["spectral"]) == (0.0, 0.0, 0.0, 0.0)

    diag = wb.weight_stats(np.diag([3.0, 1.0]))
    assert diag["stable_rank"] == pytest.approx(10 / 9)
    # effective rank uses the singular values themselves: p = (3/4, 1/4)
    assert diag["effective_rank"] == pytest.approx(math.exp(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25))))
    assert wb.weight_stats(5 * np.diag([3.0, 1.0]))["stable_rank"] == pytest.approx(10 / 9)  # scale-free


def test_delta_norms():
    np = _np()
    w0 = np.eye(4)
    assert wb.delta_norms(w0, w0) == (0.0, 2.0, None)  # no change: no rank to speak of
    fro, fro0, rank = wb.delta_norms(w0 + np.outer([1.0, 0, 0, 0], [0, 1.0, 0, 0]), w0)
    assert (fro, fro0, rank) == (pytest.approx(1.0), pytest.approx(2.0), pytest.approx(1.0))


# --- which checkpoints, resume, summary (standard library, stub measurements) ---------------------

def fake_signals(v: float, delta: bool = True) -> dict:
    """A full model_signals-shaped result whose numbers all derive from ``v``."""
    sig = {
        "n_tokens": 100,
        "layers": [
            {"layer": 0, "n_tokens": 100, "effective_rank": v + 10, "participation_ratio": v + 1, "anisotropy": 0.1},
            {"layer": 1, "n_tokens": 100, "effective_rank": v, "participation_ratio": v / 2, "anisotropy": v / 100},
        ],
        "mlp": [{"block": 0, "dead_fraction": 0.0}, {"block": 1, "dead_fraction": v / 1000}],
        "weights": {k: {"stable_rank": v / 10 + i, "effective_rank": v / 5 + i}
                    for i, k in enumerate(wb.WEIGHT_KINDS)},  # attn_qkv, attn_out, mlp_in, mlp_out
    }
    if delta:
        sig["delta"] = {"rel_norm": v / 1000, "effective_rank": 2.0}
    return sig


def grid_rows(types=("synthetic", "benchmark_near"), seeds=(0, 1), ratios=(0.0, 1.0)):
    """Rows as run_grid writes them: the first type trains the ratio-0 baseline, the others copy it."""
    rows = []
    for s in seeds:
        for r in ratios:
            for t in types:
                shared = r == 0 and t != types[0]
                rows.append({
                    "cell": f"{t}_r{r:g}_s{s}", "type": t, "ratio": r, "seed": s, "shared_baseline": shared,
                    "holdout_ppl": 50.0 + 10 * r + s + (0.5 if t == "benchmark_near" and r else 0),
                })
    return rows


def make_models(tmp_path, rows, recursion=(), weights=False):
    """Checkpoint directories for every trained cell; with ``weights`` each holds a distinct weights file."""
    models = tmp_path / "models"
    names = [r["cell"] for r in rows if not r["shared_baseline"]] + list(recursion)
    for name in names:
        (models / name).mkdir(parents=True)
        if weights:
            (models / name / "model.safetensors").write_bytes(f"weights of {name}".encode() * 50)
    return models


def test_plan_lists_cells_baseline_copies_and_recursion_stages(tmp_path):
    rows = grid_rows()
    models = make_models(tmp_path, rows, recursion=["pool_recursive_s0_gen1", "pool_recursive_s1_gen1",
                                                    "pool_recursive_s9_gen1", "somethingelse"])
    items, missing = wb.plan(rows, models, seeds=[0, 1])

    assert missing == []
    kinds = {i["cell"]: i for i in items}
    assert kinds["synthetic_r1_s0"]["kind"] == "cell" and kinds["synthetic_r1_s0"]["checkpoint"] == models / "synthetic_r1_s0"
    copy = kinds["benchmark_near_r0_s1"]
    assert copy["kind"] == "shared_baseline" and copy["source"] == "synthetic_r0_s1"  # the seed's own baseline
    assert copy["grid_holdout_ppl"] == 51.0
    stages = [i for i in items if i["kind"] == "recursion_stage"]
    assert [(i["cell"], i["generation"], i["seed"]) for i in stages] == [
        ("pool_recursive_s0_gen1", 1, 0), ("pool_recursive_s1_gen1", 1, 1)  # seed 9 is not in the grid
    ]
    assert len(items) == len(rows) + 2


def test_plan_reports_cells_without_a_checkpoint(tmp_path):
    rows = grid_rows()
    models = make_models(tmp_path, rows)
    (models / "synthetic_r1_s1").rmdir()  # a checkpoint was deleted (keep_models was off for it)
    (models / "synthetic_r0_s1").rmdir()  # and so was the baseline the copies rely on

    items, missing = wb.plan(rows, models, seeds=[0, 1])

    assert sorted(missing) == ["benchmark_near_r0_s1", "synthetic_r0_s1", "synthetic_r1_s1"]
    assert "synthetic_r1_s1" not in {i["cell"] for i in items}


def test_plan_with_a_missing_models_dir(tmp_path):
    items, missing = wb.plan(grid_rows(), tmp_path / "nope", seeds=[0, 1])
    assert items == [] and len(missing) == len(grid_rows())


class Spy:
    def __init__(self, value=lambda name: float(sum(map(ord, name)) % 50 + 5)):
        self.measured, self.written, self.value = [], [], value

    def compute(self, checkpoint):
        self.measured.append(Path(checkpoint).name)
        return fake_signals(self.value(Path(checkpoint).name))

    def run(self, items, done=None, settings="s1"):
        return wb.run_whitebox(items, compute=self.compute, done=done or {}, on_result=self.written.append,
                               settings=settings)


def test_every_checkpoint_is_measured_once_and_baselines_are_copied(tmp_path):
    rows = grid_rows()
    items, _ = wb.plan(rows, make_models(tmp_path, rows, ["pool_recursive_s0_gen1"]), seeds=[0, 1])
    spy = Spy()
    out = spy.run(items)

    assert len(out) == len(items) == len(rows) + 1
    assert sorted(spy.measured) == sorted(  # not the two copies per seed: those reuse the baseline
        [r["cell"] for r in rows if not r["shared_baseline"]] + ["pool_recursive_s0_gen1"]
    )
    by = {r["cell"]: r for r in out}
    copy, source = by["benchmark_near_r0_s0"], by["synthetic_r0_s0"]
    assert copy["kind"] == "shared_baseline" and copy["shared_from"] == "synthetic_r0_s0"
    assert copy["type"] == "benchmark_near" and copy["signals"] == source["signals"]
    assert all(r["settings"] == "s1" and "headline" in r for r in out)
    assert by["pool_recursive_s0_gen1"]["generation"] == 1 and by["pool_recursive_s0_gen1"]["kind"] == "recursion_stage"
    assert [r["cell"] for r in out] == [i["cell"] for i in items]  # the order of the plan
    assert len(spy.written) == len(items)
    json.dumps(out)  # the rows are what whitebox.jsonl holds


def test_rows_keep_the_checkpoint_name_but_not_the_path(tmp_path):
    rows = grid_rows(types=("synthetic",), seeds=(0,))
    items, _ = wb.plan(rows, make_models(tmp_path, rows), seeds=[0])
    out = wb.run_whitebox(items, compute=Spy().compute, done={}, on_result=lambda r: None, settings="s")
    assert all(r.get("checkpoint_name") == r["cell"] for r in out if r["kind"] == "cell")
    assert str(tmp_path) not in json.dumps(out)  # no machine-specific paths in a committed file


def test_a_second_run_measures_and_writes_nothing(tmp_path):
    rows = grid_rows()
    items, _ = wb.plan(rows, make_models(tmp_path, rows, ["pool_recursive_s1_gen1"]), seeds=[0, 1])
    first = Spy()
    out = first.run(items)

    again = Spy()
    out2 = again.run(items, done={r["cell"]: r for r in out})

    assert again.measured == [] and again.written == []
    assert out2 == out


def test_a_rerun_grid_cell_is_measured_again_but_nothing_else(tmp_path):
    rows = grid_rows()
    items, _ = wb.plan(rows, make_models(tmp_path, rows), seeds=[0, 1])
    first = Spy()
    done = {r["cell"]: r for r in first.run(items)}

    # benchmark_near_r1_s0 was re-run with --rerun-types: same cell id, new checkpoint, new holdout ppl
    changed = [dict(r, holdout_ppl=r["holdout_ppl"] - 3.0) if r["cell"] == "benchmark_near_r1_s0" else r for r in rows]
    items2, _ = wb.plan(changed, tmp_path / "models", seeds=[0, 1])
    again = Spy(value=lambda name: 99.0)
    out = again.run(items2, done=done)

    assert again.measured == ["benchmark_near_r1_s0"]
    assert [r["cell"] for r in again.written] == ["benchmark_near_r1_s0"]
    new = {r["cell"]: r for r in out}["benchmark_near_r1_s0"]
    assert new["signals"]["layers"][1]["effective_rank"] == 99.0
    assert new["grid_holdout_ppl"] == 57.5  # 60.5 in the grid, minus 3 after the re-run


def test_a_changed_baseline_is_re_measured_and_its_copies_follow_it(tmp_path):
    rows = grid_rows()
    items, _ = wb.plan(rows, make_models(tmp_path, rows), seeds=[0, 1])
    done = {r["cell"]: r for r in Spy().run(items)}

    changed = [dict(r, holdout_ppl=r["holdout_ppl"] + 1) if r["ratio"] == 0 and r["seed"] == 0 else r for r in rows]
    items2, _ = wb.plan(changed, tmp_path / "models", seeds=[0, 1])
    again = Spy(value=lambda name: 77.0)
    out = {r["cell"]: r for r in again.run(items2, done=done)}

    assert again.measured == ["synthetic_r0_s0"]  # only the trained baseline
    assert [r["cell"] for r in again.written] == ["synthetic_r0_s0", "benchmark_near_r0_s0"]  # and its copy follows
    assert out["benchmark_near_r0_s0"]["signals"] == out["synthetic_r0_s0"]["signals"]
    assert out["benchmark_near_r0_s1"] == done["benchmark_near_r0_s1"]  # the other seed is untouched


def test_a_copy_without_its_source_is_an_error():
    item = {"cell": "b_r0_s0", "type": "b", "ratio": 0.0, "seed": 0, "kind": "shared_baseline",
            "source": "a_r0_s0", "grid_holdout_ppl": 50.0}
    with pytest.raises(ValueError, match="a_r0_s0"):
        wb.run_whitebox([item], compute=lambda p: fake_signals(1.0), done={}, on_result=lambda r: None, settings="s")


def test_is_current():
    item = {"grid_holdout_ppl": 50.0}
    assert wb.is_current(item, {"grid_holdout_ppl": 50.0})
    assert not wb.is_current(item, {"grid_holdout_ppl": 50.1})
    assert not wb.is_current(item, None)
    assert wb.is_current({}, {"kind": "recursion_stage"})  # recursion stages have no grid row


def test_load_rows_last_row_wins_and_other_settings_are_refused(tmp_path):
    path = tmp_path / "whitebox.jsonl"
    assert wb.load_rows(path, "s") == {}
    lines = [{"cell": "a", "v": 1, "settings": "s"}, {"cell": "b", "v": 2, "settings": "s"},
             {"cell": "a", "v": 3, "settings": "s"}]
    path.write_text("".join(json.dumps(l) + "\n" for l in lines) + "\n")

    rows = wb.load_rows(path, "s")
    assert {k: r["v"] for k, r in rows.items()} == {"a": 3, "b": 2}
    with pytest.raises(ValueError, match="different settings"):
        wb.load_rows(path, "other")


def test_settings_fingerprint_follows_what_changes_the_numbers():
    base = wb.settings_fingerprint("g", {"eps": 0.01, "exclude_first_token": True})
    assert base == wb.settings_fingerprint("g", {"exclude_first_token": True, "eps": 0.01})  # order-free
    assert base != wb.settings_fingerprint("g", {"eps": 0.02, "exclude_first_token": True})
    assert base != wb.settings_fingerprint("g", {"eps": 0.01, "exclude_first_token": False})
    assert base != wb.settings_fingerprint("other-grid", {"eps": 0.01, "exclude_first_token": True})


def test_headline_scalars():
    h = wb.headline(fake_signals(40.0))
    assert h["eff_rank_last"] == 40.0 and h["eff_rank_mean"] == 45.0
    assert h["part_ratio_last"] == 20.0 and h["anisotropy_last"] == 0.4
    assert h["anisotropy_mean"] == pytest.approx(0.25)
    assert h["dead_fraction_mean"] == pytest.approx(0.02)
    assert h["stable_rank_attn"] == 4.0  # the fused QKV matrix (kind 0)
    assert h["stable_rank_mlp"] == 6.0   # the MLP input matrix (kind 2), not the attention output or MLP output
    assert h["delta_rel_norm"] == 0.04 and h["delta_eff_rank"] == 2.0
    assert "delta_rel_norm" not in wb.headline(fake_signals(40.0, delta=False))


def _summary_rows(tmp_path, value):
    rows = grid_rows(seeds=(0, 1, 2))
    items, _ = wb.plan(rows, make_models(tmp_path, rows, ["pool_recursive_s0_gen1", "pool_recursive_s1_gen1"]),
                       seeds=[0, 1, 2])
    return wb.run_whitebox(items, compute=lambda p: fake_signals(value(p.name)), done={},
                           on_result=lambda r: None, settings="s")


def test_summary_is_per_type_and_ratio_across_seeds_with_recursion_stages_apart(tmp_path):
    def value(name):
        seed = int(name.split("_s")[1][0])
        base = 100.0 if "_r0_" in name else 60.0
        return base + [0.0, 1.0, 5.0][seed]  # asymmetric on purpose: the mean 62 is not the median 61
    rows = _summary_rows(tmp_path, value)
    summary = wb.summarize(rows)

    curves = {(c["type"], c["ratio"]): c for c in summary["curves"]}
    assert set(curves) == {("synthetic", 0.0), ("synthetic", 1.0), ("benchmark_near", 0.0), ("benchmark_near", 1.0)}
    c = curves[("synthetic", 1.0)]
    assert c["n_seeds"] == 3
    assert c["eff_rank_last"] == {"mean": 62.0, "min": 60.0, "max": 65.0}
    # the baseline copy of benchmark_near is the synthetic baseline, so the two agree
    assert curves[("benchmark_near", 0.0)]["eff_rank_last"] == curves[("synthetic", 0.0)]["eff_rank_last"]
    assert [(s["generation"], s["n_seeds"]) for s in summary["recursion_stages"]] == [(1, 2)]
    json.dumps(summary)


def test_summary_skips_signals_that_are_missing(tmp_path):
    rows = grid_rows(types=("synthetic",), seeds=(0,))
    items, _ = wb.plan(rows, make_models(tmp_path, rows), seeds=[0])
    out = wb.run_whitebox(items, compute=lambda p: fake_signals(10.0, delta=False), done={},
                          on_result=lambda r: None, settings="s")
    curve = wb.summarize(out)["curves"][0]
    assert "eff_rank_last" in curve and "delta_rel_norm" not in curve


def test_format_table(tmp_path):
    rows = _summary_rows(tmp_path, lambda name: 100.0 if "_r0_" in name else 60.0)
    table = wb.format_table(wb.summarize(rows), "eff_rank_last").splitlines()
    assert table[0].split() == ["ratio", "benchmark_near", "synthetic"]
    assert table[1].split() == ["0", "100.0000", "100.0000"] and table[2].split() == ["1", "60.0000", "60.0000"]


def test_format_table_shows_the_mean_over_seeds_and_a_dash_for_a_missing_point(tmp_path):
    seed_offset = [0.0, 1.0, 5.0]  # seeds 0, 1, 2: the mean of 60, 61, 65 is 62, the minimum 60, the median 61
    rows = _summary_rows(tmp_path, lambda name: 60.0 + seed_offset[int(name.split("_s")[1][0])])
    rows = [r for r in rows if not (r["type"] == "benchmark_near" and r["ratio"] == 0.0)]  # a point with no rows
    table = wb.format_table(wb.summarize(rows), "eff_rank_last").splitlines()
    assert table[1].split() == ["0", "-", "62.0000"]  # synthetic's baseline is the mean 62; benchmark_near has none
    assert table[2].split() == ["1", "62.0000", "62.0000"]


def test_the_summary_is_ordered_by_type_then_ratio_and_by_generation(tmp_path):
    rows = _summary_rows(tmp_path, lambda name: 50.0)
    rows += [dict(rows[-1], cell="pool_recursive_s0_gen2", kind="recursion_stage", generation=2, ratio=None)]
    summary = wb.summarize(list(reversed(rows)))  # fed in the opposite order on purpose
    assert [(c["type"], c["ratio"]) for c in summary["curves"]] == [
        ("benchmark_near", 0.0), ("benchmark_near", 1.0), ("synthetic", 0.0), ("synthetic", 1.0)]
    assert [g["generation"] for g in summary["recursion_stages"]] == [1, 2]


def test_plot_writes_a_figure(tmp_path):
    pytest.importorskip("matplotlib")
    rows = _summary_rows(tmp_path, lambda name: 100.0 if "_r0_" in name else 60.0)
    path = tmp_path / "sub" / "signals.png"
    wb._plot(wb.summarize(rows), path)
    assert path.stat().st_size > 1000


# --- the command line, with the measurement stubbed -----------------------------------------------

def _cli_setup(tmp_path, monkeypatch, *, with_checkpoints=True):
    """A grid config + results.jsonl + checkpoint dirs in tmp, and a whitebox config pointing at them."""
    from src.finetune import collapse
    from src.finetune.grid import config_fingerprint

    grid = yaml.safe_load((REPO / "configs/grid_smoke.yaml").read_text())
    grid.update(output_dir=str(tmp_path / "grid_out"), models_dir=str(tmp_path / "models"),
                seeds=[0, 1], ratios=[0.0, 1.0], types=["synthetic", "benchmark_near"])
    grid_cfg = tmp_path / "grid.yaml"
    grid_cfg.write_text(yaml.safe_dump(grid))
    fp = config_fingerprint(grid)
    rows = grid_rows()
    (tmp_path / "grid_out").mkdir()
    (tmp_path / "grid_out/results.jsonl").write_text(
        "".join(json.dumps({**r, "config": fp, "benchmark_ppl": 60.0}) + "\n" for r in rows)
    )
    if with_checkpoints:
        make_models(tmp_path, rows, ["pool_recursive_s0_gen1"])

    cfg = {"grid_config": str(grid_cfg), "eps": 0.01, "exclude_first_token": True, "delta_from_base": True,
           "output_dir": str(tmp_path / "wb"), "figure": str(tmp_path / "wb/signals.png")}
    config = tmp_path / "whitebox.yaml"
    config.write_text(yaml.safe_dump(cfg))

    spy = Spy()
    calls = []

    def fake_compute(checkpoint, texts, **kw):
        calls.append((Path(checkpoint).name, list(texts), kw))
        return spy.compute(checkpoint)

    plots = []
    monkeypatch.setattr(wb, "compute_checkpoint", fake_compute)
    monkeypatch.setattr(wb, "_plot", lambda summary, path: plots.append(path))
    monkeypatch.setattr(collapse, "load_holdout", lambda data: ["held out text one", "held out text two"])
    return config, cfg, grid, calls, plots


def test_main_measures_each_checkpoint_and_writes_rows_summary_and_figure(tmp_path, monkeypatch, capsys):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)

    assert wb.main(["--config", str(config)]) == 0

    names = sorted(name for name, _, _ in calls)
    assert names == sorted([r["cell"] for r in grid_rows() if not r["shared_baseline"]] + ["pool_recursive_s0_gen1"])
    _, texts, kw = calls[0]
    assert texts == ["held out text one", "held out text two"]  # the grid's fixed holdout
    assert kw == {"max_len": grid["train"]["max_len"], "batch_size": grid["train"]["batch_size"],
                  "device": grid["device"], "eps": 0.01, "exclude_first_token": True,
                  "base_model": grid["base_model"]}
    rows = [json.loads(l) for l in (tmp_path / "wb/whitebox.jsonl").read_text().splitlines()]
    assert len(rows) == len(grid_rows()) + 1
    summary = json.loads((tmp_path / "wb/whitebox_summary.json").read_text())
    assert summary["missing_cells"] == [] and len(summary["curves"]) == 4
    assert plots == [Path(cfg["figure"])]
    out = capsys.readouterr().out
    assert "eff_rank_last (mean over seeds)" in out and "anisotropy_last" in out and "dead_fraction_mean" in out


def test_main_second_run_measures_nothing_and_the_file_does_not_grow(tmp_path, monkeypatch):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    assert wb.main(["--config", str(config)]) == 0
    before = (tmp_path / "wb/whitebox.jsonl").read_bytes()
    calls.clear()

    assert wb.main(["--config", str(config)]) == 0

    assert calls == [] and (tmp_path / "wb/whitebox.jsonl").read_bytes() == before


def test_main_measures_again_only_the_cell_whose_grid_row_changed(tmp_path, monkeypatch, capsys):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    assert wb.main(["--config", str(config)]) == 0
    calls.clear()
    capsys.readouterr()

    results = tmp_path / "grid_out/results.jsonl"  # benchmark_near_r1_s1 was re-run: a new holdout ppl
    lines = []
    for line in results.read_text().splitlines():
        row = json.loads(line)
        if row["cell"] == "benchmark_near_r1_s1":
            row["holdout_ppl"] = 40.0
        lines.append(json.dumps(row))
    results.write_text("\n".join(lines) + "\n")
    assert wb.main(["--config", str(config)]) == 0

    assert "1 to do" in capsys.readouterr().out  # the progress line counts the stale cell, not zero
    assert [name for name, _, _ in calls] == ["benchmark_near_r1_s1"]
    latest = wb.load_rows(tmp_path / "wb/whitebox.jsonl", json.loads(
        (tmp_path / "wb/whitebox_summary.json").read_text())["settings"])
    assert latest["benchmark_near_r1_s1"]["grid_holdout_ppl"] == 40.0


def test_main_summarize_only_measures_nothing(tmp_path, monkeypatch):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    assert wb.main(["--config", str(config)]) == 0
    (tmp_path / "wb/whitebox_summary.json").unlink()
    calls.clear()
    plots.clear()

    assert wb.main(["--config", str(config), "--summarize-only"]) == 0

    assert calls == [] and plots == [Path(cfg["figure"])]
    assert (tmp_path / "wb/whitebox_summary.json").exists()


def test_main_refuses_rows_made_under_different_settings(tmp_path, monkeypatch):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    assert wb.main(["--config", str(config)]) == 0
    calls.clear()

    config.write_text(yaml.safe_dump({**cfg, "eps": 0.5}))
    with pytest.raises(ValueError, match="different settings"):
        wb.main(["--config", str(config)])
    with pytest.raises(ValueError, match="different settings"):
        wb.main(["--config", str(config), "--summarize-only"])
    assert calls == []


def test_main_refuses_grid_results_from_other_settings(tmp_path, monkeypatch):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    grid["train"]["lr"] = 1.0  # the grid config no longer matches the results.jsonl it wrote
    (tmp_path / "grid.yaml").write_text(yaml.safe_dump(grid))
    with pytest.raises(ValueError, match="different settings"):
        wb.main(["--config", str(config)])
    assert calls == []


def test_main_warns_about_cells_without_a_checkpoint_and_says_so_without_any(tmp_path, monkeypatch, capsys):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    import shutil
    shutil.rmtree(tmp_path / "models" / "synthetic_r1_s1")
    assert wb.main(["--config", str(config)]) == 0
    assert "1 grid cell(s) have no checkpoint" in capsys.readouterr().out
    assert json.loads((tmp_path / "wb/whitebox_summary.json").read_text())["missing_cells"] == ["synthetic_r1_s1"]


def test_main_with_no_checkpoints_at_all_says_so(tmp_path, monkeypatch, capsys):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch, with_checkpoints=False)
    with pytest.raises(SystemExit, match="no white-box rows"):
        wb.main(["--config", str(config)])
    assert "keep_models" in capsys.readouterr().out and calls == []


def test_main_needs_every_setting_and_some_grid_results(tmp_path, monkeypatch):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    config.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k != "eps"}))
    with pytest.raises(ValueError, match="eps"):
        wb.main(["--config", str(config)])

    config.write_text(yaml.safe_dump(cfg))
    (tmp_path / "grid_out/results.jsonl").write_text("")
    with pytest.raises(SystemExit, match="no grid results"):
        wb.main(["--config", str(config)])


def test_the_shipped_config_points_at_the_grid_and_sets_everything_main_reads():
    cfg = yaml.safe_load((REPO / "configs/whitebox.yaml").read_text())
    assert set(cfg) == set(wb._SETTING_KEYS)
    assert (REPO / cfg["grid_config"]).exists()
    assert cfg["eps"] > 0 and isinstance(cfg["exclude_first_token"], bool) and isinstance(cfg["delta_from_base"], bool)
    # it writes next to the grid's results, and does not touch the grid's own files
    assert cfg["output_dir"] == yaml.safe_load((REPO / cfg["grid_config"]).read_text())["output_dir"]
    assert Path(cfg["figure"]).name == "whitebox_signals.png"


def test_random_baseline_for_reference():
    # the formulas agree with each other on a spectrum built from a known covariance
    rng = random.Random(0)
    s = [rng.uniform(0.1, 5) for _ in range(20)]
    lam = [v * v for v in s]
    assert 1 < wb.effective_rank(s) <= 20 and 1 < wb.participation_ratio(lam) <= 20
    assert wb.stable_rank(s) == pytest.approx(sum(lam) / max(lam))


# --- the real model path: a tiny randomly initialised GPT-2 (torch and transformers) -------------------

N_EMBD, N_LAYER, N_HEAD = 16, 2, 2


@pytest.fixture(scope="module")
def torch_stack():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    return torch, transformers


def tiny_model(torch_stack, seed=0, init_range=0.5, n_layer=N_LAYER):
    """A random GPT-2 (no download). A large init range keeps the MLP units from being dead by chance."""
    torch, transformers = torch_stack
    torch.manual_seed(seed)
    config = transformers.GPT2Config(
        vocab_size=50, n_positions=32, n_embd=N_EMBD, n_layer=n_layer, n_head=N_HEAD,
        initializer_range=init_range, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    return transformers.GPT2LMHeadModel(config).eval()


def batch_of(torch_stack, lengths, seed=0, pad_to=None):
    """Right-padded (input_ids, attention_mask) for sequences of the given lengths."""
    torch, _ = torch_stack
    g = torch.Generator().manual_seed(seed)
    width = pad_to or max(lengths)
    ids = torch.zeros((len(lengths), width), dtype=torch.long)
    mask = torch.zeros((len(lengths), width), dtype=torch.long)
    for i, n in enumerate(lengths):
        ids[i, :n] = torch.randint(1, 50, (n,), generator=g)
        mask[i, :n] = 1
    return ids, mask


def test_model_signals_have_the_expected_shape_and_sane_values(torch_stack):
    sig = wb.model_signals(tiny_model(torch_stack), [batch_of(torch_stack, [12, 9, 14, 11], pad_to=16)])

    assert len(sig["layers"]) == N_LAYER + 1 and len(sig["mlp"]) == N_LAYER  # embeddings + one per block
    assert not any(l["degenerate"] for l in sig["layers"])
    assert sig["n_tokens"] == (12 + 9 + 14 + 11) - 4  # the first token of each sequence is excluded
    for layer in sig["layers"]:
        assert 1.0 <= layer["effective_rank"] <= N_EMBD + 1e-6
        assert 1.0 <= layer["participation_ratio"] <= N_EMBD + 1e-6
        assert -1.0 <= layer["anisotropy"] <= 1.0
    assert all(0.0 <= m["dead_fraction"] <= 1.0 for m in sig["mlp"])
    assert set(sig["weights"]) == set(wb.WEIGHT_KINDS)
    assert all(w["stable_rank"] >= 1.0 for w in sig["weights"].values())
    assert "delta" not in sig
    json.dumps(sig)
    assert set(wb.headline(sig)) >= {"eff_rank_last", "anisotropy_last", "dead_fraction_mean"}


def test_padding_and_batching_do_not_change_the_signals(torch_stack):
    # the same four sequences, once in one heavily padded batch and once one by one
    lengths = [12, 9, 14, 11]
    ids, mask = batch_of(torch_stack, lengths, seed=3, pad_to=20)
    together = wb.model_signals(tiny_model(torch_stack), [(ids, mask)])
    separately = wb.model_signals(
        tiny_model(torch_stack), [(ids[i : i + 1, :n], mask[i : i + 1, :n]) for i, n in enumerate(lengths)]
    )

    assert together["n_tokens"] == separately["n_tokens"]
    for a, b in zip(together["layers"], separately["layers"]):
        for key in ("effective_rank", "participation_ratio", "anisotropy"):
            assert a[key] == pytest.approx(b[key], rel=1e-4, abs=1e-6), key
    assert together["mlp"] == separately["mlp"]


def test_what_the_pad_tokens_are_does_not_matter(torch_stack):
    # the same real tokens, padded with token 0 or with token 7: nothing may change, including which MLP
    # units count as dead (a loose threshold makes the dead count sensitive to every token it sees)
    ids, mask = batch_of(torch_stack, [6, 5, 7, 4], seed=9, pad_to=24)
    other = ids.clone()
    other[mask == 0] = 7
    model = tiny_model(torch_stack)
    a = wb.model_signals(model, [(ids, mask)], eps=0.3)
    b = wb.model_signals(model, [(other, mask)], eps=0.3)

    assert a["mlp"] == b["mlp"]
    assert 0.0 < a["mlp"][0]["dead_fraction"] < 1.0 and 0.0 < a["mlp"][1]["dead_fraction"] < 1.0  # a sensitive test
    for x, y in zip(a["layers"], b["layers"]):
        assert x == y


def test_the_first_token_is_excluded_only_when_asked(torch_stack):
    model, batch = tiny_model(torch_stack), batch_of(torch_stack, [10, 8, 12])
    kept = wb.model_signals(model, [batch], exclude_first_token=False)
    dropped = wb.model_signals(model, [batch], exclude_first_token=True)
    assert kept["n_tokens"] == 30 and dropped["n_tokens"] == 27


def test_dead_neurons_are_counted_per_block(torch_stack):
    torch, _ = torch_stack
    model = tiny_model(torch_stack)
    batch = batch_of(torch_stack, [14, 13, 15, 12], seed=5)
    before = wb.model_signals(model, [batch])["mlp"]
    assert before[0]["dead_fraction"] < 1.0

    # silence every unit of block 0: a large negative bias and no input weights give GELU(-1e4) = 0
    mlp = model.transformer.h[0].mlp
    with torch.no_grad():
        mlp.c_fc.weight.zero_()
        mlp.c_fc.bias.fill_(-1e4)
    after = wb.model_signals(model, [batch])["mlp"]
    assert after[0]["dead_fraction"] == 1.0
    assert after[1]["dead_fraction"] < 1.0  # block 1 still sees varied inputs and is not dead


def test_a_looser_threshold_marks_more_units_dead(torch_stack):
    model, batch = tiny_model(torch_stack), batch_of(torch_stack, [14, 13, 15, 12], seed=6)
    strict = wb.model_signals(model, [batch], eps=0.01)["mlp"]
    loose = wb.model_signals(model, [batch], eps=1e9)["mlp"]
    assert all(l["dead_fraction"] == 1.0 for l in loose)
    assert all(l["dead_fraction"] >= s["dead_fraction"] for l, s in zip(loose, strict))


def test_a_collapsed_representation_has_low_rank_and_high_anisotropy(torch_stack):
    torch, _ = torch_stack
    model = tiny_model(torch_stack)
    batch = batch_of(torch_stack, [14, 13, 15, 12], seed=7)
    spread = wb.model_signals(model, [batch])["layers"][0]

    with torch.no_grad():  # every token embeds along one direction u, with a positive size
        u = torch.randn(N_EMBD)
        size = 1.0 + torch.rand(model.transformer.wte.weight.shape[0], 1)
        model.transformer.wte.weight.copy_(size * u + 1e-3 * torch.randn_like(model.transformer.wte.weight))
        model.transformer.wpe.weight.zero_()
    collapsed = wb.model_signals(model, [batch])["layers"][0]

    assert collapsed["effective_rank"] < 2.0 < spread["effective_rank"]  # the variance lives in one direction
    assert collapsed["participation_ratio"] < 1.5 < spread["participation_ratio"]
    assert collapsed["anisotropy"] > 0.99 > spread["anisotropy"]  # and every token points the same way


def test_a_completely_collapsed_model_can_still_be_measured(torch_stack):
    torch, _ = torch_stack
    model = tiny_model(torch_stack)
    with torch.no_grad():  # every parameter zero: no variance anywhere, all-zero weight matrices
        for p in model.parameters():
            p.zero_()
        model.transformer.wte.weight[:, 0] = 1.0  # tokens still have a direction, so anisotropy is defined
    sig = wb.model_signals(model, [batch_of(torch_stack, [10, 11, 9])])

    first = sig["layers"][0]  # every token is the same vector (1, 0, 0, ...): no variance, full anisotropy
    assert first["degenerate"] and first["effective_rank"] == 0.0 and first["anisotropy"] == pytest.approx(1.0)
    assert all(l["degenerate"] and l["effective_rank"] == 0.0 for l in sig["layers"])  # and zeros after that
    assert all(w["stable_rank"] == 0.0 and w["effective_rank"] == 0.0 for w in sig["weights"].values())
    assert all(m["dead_fraction"] == 1.0 for m in sig["mlp"])
    json.dumps(sig)


def test_weight_signals_and_distance_from_the_base(torch_stack):
    torch, _ = torch_stack
    model, batch = tiny_model(torch_stack), batch_of(torch_stack, [10, 11, 12])
    base = wb.block_weights(model)
    assert set(base) == {f"{kind}.{i}" for kind in wb.WEIGHT_KINDS for i in range(N_LAYER)}

    same = wb.model_signals(model, [batch], base_weights=base)["delta"]
    assert same == {"rel_norm": 0.0, "effective_rank": None}

    with torch.no_grad():  # a rank-one change to one matrix, like a LoRA update of rank 1
        w = model.transformer.h[1].attn.c_attn.weight
        w += torch.outer(torch.ones(w.shape[0]), torch.ones(w.shape[1])) * 0.05
    moved = wb.model_signals(model, [batch], base_weights=base)["delta"]
    assert moved["rel_norm"] > 0
    assert moved["effective_rank"] == pytest.approx(1.0, abs=1e-3)  # float32 rounding leaves a trace
    assert wb.headline(wb.model_signals(model, [batch], base_weights=base))["delta_rel_norm"] == pytest.approx(
        moved["rel_norm"])


def test_a_model_without_gpt2_blocks_is_refused(torch_stack):
    torch, _ = torch_stack
    with pytest.raises(ValueError, match="GPT-2"):
        wb.model_signals(torch.nn.Linear(2, 2), [])


def test_compute_checkpoint_end_to_end_on_a_saved_tiny_model(torch_stack, tmp_path):
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    model = tiny_model(torch_stack)
    words = ["<unk>", "<eos>", "the", "cat", "sat", "on", "mat", "dog", "ran", "far", "away", "home"]
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<eos>")
    ckpt = tmp_path / "ckpt"
    model.save_pretrained(ckpt)
    fast.save_pretrained(ckpt)

    texts = ["the cat sat on the mat", "the dog ran far away from home", "cat dog", "the mat the cat sat"]
    kw = dict(max_len=8, batch_size=2, device="cpu", eps=0.01, exclude_first_token=True)
    sig = wb.compute_checkpoint(ckpt, texts, base_model=None, **kw)
    assert len(sig["layers"]) == N_LAYER + 1 and sig["n_tokens"] > 0 and "delta" not in sig

    # batch size must not matter, and measuring against itself as the base moves nothing
    other = wb.compute_checkpoint(ckpt, texts, base_model=str(ckpt), **{**kw, "batch_size": 4})
    assert other["delta"]["rel_norm"] == 0.0
    for a, b in zip(sig["layers"], other["layers"]):
        assert a["effective_rank"] == pytest.approx(b["effective_rank"], rel=1e-4)
        assert a["anisotropy"] == pytest.approx(b["anisotropy"], abs=1e-5)
    assert sig["mlp"] == other["mlp"]
    # "cat dog" is 2 tokens, so truncation to 8 is not what limits it; the 7-word text is cut to 8
    assert sig["n_tokens"] == sum(min(len(t.split()), 8) - 1 for t in texts)


def _save_tiny(torch_stack, directory):
    """A saved tiny GPT-2 plus a word-level tokenizer, loadable by name like a Hugging Face model."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<eos>", "the", "cat", "sat", "on", "mat", "dog", "ran", "far", "away", "home"]
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<eos>")
    tiny_model(torch_stack).save_pretrained(directory)
    fast.save_pretrained(directory)
    return directory


def test_it_reads_what_the_grids_train_lora_actually_writes(torch_stack, tmp_path):
    """The real pipeline: train_lora (merged LoRA checkpoint) -> white-box signals against the base."""
    pytest.importorskip("peft")
    from src.finetune.lora import train_lora

    base = _save_tiny(torch_stack, tmp_path / "base")
    texts = ["the cat sat on the mat", "the dog ran far away from home", "the mat the cat sat",
             "cat dog ran home"] * 4
    r = 2
    ckpt = train_lora(texts, tmp_path / "ckpt", base_model=str(base), epochs=3, lr=1e-2, batch_size=4,
                      max_len=8, lora_r=r, seed=0, device="cpu")

    kw = dict(max_len=8, batch_size=2, device="cpu", eps=0.01, exclude_first_token=True)
    moved = wb.compute_checkpoint(ckpt, texts[:6], base_model=str(base), **kw)
    untouched = wb.compute_checkpoint(base, texts[:6], base_model=str(base), **kw)

    assert untouched["delta"] == {"rel_norm": 0.0, "effective_rank": None}  # the base is not away from itself
    assert moved["delta"]["rel_norm"] > 1e-3  # training moved the weights...
    assert 1.0 <= moved["delta"]["effective_rank"] <= r + 0.05  # ...by an update of rank at most r
    # PEFT's default target for GPT-2 is the attention input projection: only those matrices changed
    base_w, new_w = wb.block_weights(_load(base)), wb.block_weights(_load(ckpt))
    changed = {key.split(".")[0] for key in base_w if not (base_w[key] == new_w[key]).all()}
    assert changed == {"attn_qkv"}
    # and the model's representations differ from the base's, so the signals can see the fine-tune
    assert moved["layers"][-1]["anisotropy"] != pytest.approx(untouched["layers"][-1]["anisotropy"], abs=1e-9)
    json.dumps(moved)


def _load(path):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path).eval()


def test_a_unit_is_dead_by_its_activation_not_by_its_input(torch_stack):
    torch, _ = torch_stack
    model = tiny_model(torch_stack)
    batch = batch_of(torch_stack, [12, 11, 13])
    with torch.no_grad():  # every unit of block 0 gets the constant input 0.015, and GELU(0.015) = 0.0075
        mlp = model.transformer.h[0].mlp
        mlp.c_fc.weight.zero_()
        mlp.c_fc.bias.fill_(0.015)
    sig = wb.model_signals(model, [batch], eps=0.01)
    assert sig["mlp"][0]["dead_fraction"] == 1.0  # the pre-activation 0.015 is above eps; the activation is not
    assert wb.model_signals(model, [batch], eps=0.005)["mlp"][0]["dead_fraction"] == 0.0


def test_weight_signals_average_over_every_block(torch_stack):
    np = _np()
    model = tiny_model(torch_stack)
    weights = wb.block_weights(model)
    sig = wb.model_signals(model, [batch_of(torch_stack, [10, 11])])

    for kind in wb.WEIGHT_KINDS:
        per_block = [wb.weight_stats(weights[f"{kind}.{i}"]) for i in range(N_LAYER)]
        assert per_block[0]["stable_rank"] != pytest.approx(per_block[1]["stable_rank"])  # blocks differ...
        assert sig["weights"][kind]["stable_rank"] == pytest.approx(np.mean([b["stable_rank"] for b in per_block]))
        assert sig["weights"][kind]["effective_rank"] == pytest.approx(np.mean([b["effective_rank"] for b in per_block]))


def test_the_distance_from_the_base_is_over_all_the_block_matrices(torch_stack):
    torch, _ = torch_stack
    model = tiny_model(torch_stack)
    base = wb.block_weights(model)
    with torch.no_grad():
        model.transformer.h[0].mlp.c_proj.weight.add_(0.1)  # one matrix changes everywhere
    norms = [float(((wb.block_weights(model)[k] - base[k]) ** 2).sum()) for k in base]
    total0 = sum(float((base[k] ** 2).sum()) for k in base)
    rel = wb.model_signals(model, [batch_of(torch_stack, [10, 11])], base_weights=base)["delta"]["rel_norm"]
    assert rel == pytest.approx(math.sqrt(sum(norms) / total0), rel=1e-5)


def test_main_can_skip_the_distance_from_the_base(tmp_path, monkeypatch):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    config.write_text(yaml.safe_dump({**cfg, "delta_from_base": False}))
    assert wb.main(["--config", str(config)]) == 0
    assert calls and all(kw["base_model"] is None for _, _, kw in calls)


# ===== review follow-ups: shifted accumulation, checkpoint identity, reconciling against the grid ======

def test_a_layer_where_every_token_is_one_vector_is_exactly_degenerate_even_for_awkward_numbers():
    np = _np()
    v = np.random.default_rng(0).normal(size=768) * 10  # not a dyadic number: E[xx^T] - mean mean^T would leave residue
    stats = wb.RepresentationStats(768)
    for _ in range(10):
        stats.update(np.tile(v, (2048, 1)))
    res = stats.result()
    assert res["degenerate"] and res["effective_rank"] == 0.0 and res["participation_ratio"] == 0.0
    assert res["anisotropy"] == pytest.approx(1.0)


def test_a_huge_common_offset_does_not_blur_the_spectrum():
    np = _np()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5000, 12)) * np.linspace(0.5, 3.0, 12) + 1e8  # mean 1e8, std 0.5 to 3
    stats = wb.RepresentationStats(12)
    for chunk in _chunks(x, [700, 1300, 11]):
        stats.update(chunk)
    exact = np.linalg.eigvalsh(np.cov((x - x.mean(axis=0)).T, bias=True))[::-1]
    assert np.allclose(stats.eigenvalues(), exact, rtol=1e-6)
    assert stats.result()["effective_rank"] == pytest.approx(wb.effective_rank(np.sqrt(exact)), rel=1e-6)


def test_the_first_batch_can_be_empty():
    np = _np()
    stats = wb.RepresentationStats(2)
    stats.update(np.zeros((0, 2)))
    stats.update(np.array([[1.0, 2.0], [3.0, 5.0], [0.0, 1.0]]))
    assert stats.n == 3 and stats.result()["effective_rank"] > 1.0


def test_delta_effective_rank_is_the_entropy_rank_of_the_update():
    np = _np()
    w0 = np.eye(4)
    _, _, rank = wb.delta_norms(w0 + np.diag([3.0, 1.0, 0.0, 0.0]), w0)
    assert rank == pytest.approx(math.exp(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25))))


# --- checkpoint identity ----------------------------------------------------------------------------

def _ckpt(tmp_path, name="ckpt", content=b"weights" * 100):
    d = tmp_path / name
    d.mkdir()
    (d / "model.safetensors").write_bytes(content)
    return d


def test_checkpoint_identity_and_when_it_still_matches(tmp_path):
    import os
    import shutil

    d = _ckpt(tmp_path)
    ident = wb.checkpoint_identity(d)
    assert set(ident) == {"size", "mtime_ns", "hash"} and ident["size"] == 700
    assert wb.same_checkpoint(d, ident)  # untouched

    copy = tmp_path / "copy"  # copied elsewhere: a new mtime, the same content
    shutil.copytree(d, copy)
    os.utime(copy / "model.safetensors", ns=(1, 1))
    assert wb.same_checkpoint(copy, ident)

    changed = _ckpt(tmp_path, "changed", b"WEIGHTS" * 100)  # replaced: same size, other content, other mtime
    os.utime(changed / "model.safetensors", ns=(5, 5))
    assert not wb.same_checkpoint(changed, ident)

    longer = _ckpt(tmp_path, "longer", b"weights" * 101)
    assert not wb.same_checkpoint(longer, ident)


def test_nothing_to_compare_is_not_a_difference(tmp_path):
    d = _ckpt(tmp_path)
    ident = wb.checkpoint_identity(d)
    assert wb.same_checkpoint(d, None) and wb.same_checkpoint(d, {})  # a row without an identity
    assert wb.same_checkpoint(None, ident)  # an item without a checkpoint
    (tmp_path / "bare").mkdir()
    assert wb.same_checkpoint(tmp_path / "bare", ident)  # a checkpoint with no weights file here
    assert wb.checkpoint_identity(tmp_path / "bare") is None
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "pytorch_model.bin").write_bytes(b"x" * 10)
    assert wb.checkpoint_identity(legacy)["size"] == 10  # the older file name is understood


def test_rows_record_the_identity_and_an_overwritten_checkpoint_is_measured_again(tmp_path):
    import os

    rows = grid_rows(types=("synthetic",), seeds=(0,))
    models = make_models(tmp_path, rows, recursion=["pool_recursive_s0_gen1"], weights=True)
    items, _ = wb.plan(rows, models, seeds=[0])
    first = Spy()
    done = {r["cell"]: r for r in first.run(items)}
    assert done["pool_recursive_s0_gen1"]["checkpoint_id"] == wb.checkpoint_identity(models / "pool_recursive_s0_gen1")

    # the recursion intermediate is retrained into the same directory: no grid row can tell
    weights = models / "pool_recursive_s0_gen1" / "model.safetensors"
    weights.write_bytes(b"retrained weights" * 50)
    os.utime(weights, ns=(7, 7))
    again = Spy()
    out = {r["cell"]: r for r in again.run(items, done=done)}

    assert again.measured == ["pool_recursive_s0_gen1"]  # and only it
    assert out["pool_recursive_s0_gen1"]["checkpoint_id"] != done["pool_recursive_s0_gen1"]["checkpoint_id"]


def test_a_cell_retrained_to_the_same_holdout_ppl_is_still_noticed(tmp_path):
    import os

    rows = grid_rows(types=("synthetic",), seeds=(0,))
    models = make_models(tmp_path, rows, weights=True)
    items, _ = wb.plan(rows, models, seeds=[0])
    done = {r["cell"]: r for r in Spy().run(items)}

    weights = models / "synthetic_r1_s0" / "model.safetensors"
    weights.write_bytes(b"another model" * 50)  # same grid row, different checkpoint
    os.utime(weights, ns=(9, 9))
    again = Spy()
    again.run(items, done=done)
    assert again.measured == ["synthetic_r1_s0"]


def test_a_baseline_copy_listed_before_its_source_is_still_copied(tmp_path):
    rows = grid_rows(types=("synthetic", "benchmark_near"), seeds=(0,))
    items, _ = wb.plan(rows, make_models(tmp_path, rows), seeds=[0])
    reordered = sorted(items, key=lambda i: i["kind"] != "shared_baseline")  # copies first
    assert reordered[0]["kind"] == "shared_baseline"
    out = {r["cell"]: r for r in Spy().run(reordered)}
    assert out["benchmark_near_r0_s0"]["signals"] == out["synthetic_r0_s0"]["signals"]


# --- reconciling the rows with the grid as it is now ---------------------------------------------------

def _measured(tmp_path, rows, **kw):
    models = make_models(tmp_path, rows, **kw)
    items, _ = wb.plan(rows, models, seeds=[0, 1])
    return models, {r["cell"]: r for r in Spy().run(items)}


def test_reconcile_trusts_rows_without_any_checkpoint_on_disk(tmp_path):
    rows = grid_rows()
    _, done = _measured(tmp_path, rows)
    fresh, stale, missing = wb.reconcile(done, rows, {r["cell"] for r in rows}, tmp_path / "gone", [0, 1])
    assert len(fresh) == len(rows) and stale == [] and missing == []


def test_reconcile_flags_a_rerun_cell_stale_and_a_cell_without_a_row_missing(tmp_path):
    rows = grid_rows()
    _, done = _measured(tmp_path, rows)
    now = [dict(r, holdout_ppl=r["holdout_ppl"] - 3) if r["cell"] == "benchmark_near_r1_s1" else r for r in rows]
    done.pop("synthetic_r1_s0")

    fresh, stale, missing = wb.reconcile(done, now, {r["cell"] for r in rows}, tmp_path / "models", [0, 1])
    assert stale == ["benchmark_near_r1_s1"] and missing == ["synthetic_r1_s0"]
    assert "benchmark_near_r1_s1" not in {r["cell"] for r in fresh} and len(fresh) == len(rows) - 2


def test_reconcile_flags_a_replaced_checkpoint_where_it_is_on_disk(tmp_path):
    import os

    rows = grid_rows()
    models, done = _measured(tmp_path, rows, recursion=["pool_recursive_s0_gen1"], weights=True)
    assert wb.reconcile(done, rows, {r["cell"] for r in rows}, models, [0, 1])[1] == []
    for name in ("synthetic_r1_s0", "pool_recursive_s0_gen1"):
        f = models / name / "model.safetensors"
        f.write_bytes(b"replaced" * 70)
        os.utime(f, ns=(3, 3))
    _, stale, missing = wb.reconcile(done, rows, {r["cell"] for r in rows}, models, [0, 1])
    assert stale == ["pool_recursive_s0_gen1", "synthetic_r1_s0"] and missing == []


def test_reconcile_ignores_cells_and_stages_outside_the_configured_grid(tmp_path):
    rows = grid_rows()
    _, done = _measured(tmp_path, rows, recursion=["pool_recursive_s0_gen1", "pool_recursive_s1_gen1"])
    seed0 = [r for r in rows if r["seed"] == 0]
    fresh, stale, missing = wb.reconcile(done, seed0, {r["cell"] for r in seed0}, tmp_path / "models", [0])
    assert {r["seed"] for r in fresh} == {0} and stale == [] and missing == []
    assert [r["cell"] for r in fresh if r["kind"] == "recursion_stage"] == ["pool_recursive_s0_gen1"]


def test_partial_points_name_the_thin_means():
    rows = [
        {"cell": "a", "type": "t", "ratio": 1.0, "seed": 0, "kind": "cell", "headline": {"eff_rank_last": 5.0}},
        {"cell": "b", "type": "t", "ratio": 1.0, "seed": 1, "kind": "cell", "headline": {"eff_rank_last": 6.0}},
        {"cell": "c", "type": "t", "ratio": 0.0, "seed": 0, "kind": "cell", "headline": {"eff_rank_last": 9.0}},
        {"cell": "p", "type": "recursive", "ratio": None, "seed": 0, "kind": "recursion_stage", "generation": 1,
         "headline": {"eff_rank_last": 4.0}},
    ]
    assert wb.partial_points(wb.summarize(rows), 2) == [
        "t ratio 0: 1 of 2 seeds", "recursion generation 1: 1 of 2 seeds"
    ]
    assert wb.partial_points(wb.summarize(rows), 1) == []


# --- the command line against the grid as it is now -----------------------------------------------------

def test_a_clone_without_the_checkpoints_can_still_summarize_the_committed_rows(tmp_path, monkeypatch, capsys):
    import shutil
    from src.finetune import collapse

    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    assert wb.main(["--config", str(config)]) == 0
    shutil.rmtree(tmp_path / "models")  # data/models is git-ignored: a fresh clone has none
    (tmp_path / "wb/whitebox_summary.json").unlink()
    monkeypatch.setattr(collapse, "load_holdout", lambda data: pytest.fail("summarising must not load the holdout"))
    calls.clear()
    capsys.readouterr()

    assert wb.main(["--config", str(config), "--summarize-only"]) == 0

    summary = json.loads((tmp_path / "wb/whitebox_summary.json").read_text())
    assert summary["complete"] and summary["missing_cells"] == [] and summary["stale_cells"] == []
    assert len(summary["curves"]) == 4 and calls == []
    assert "keep_models" not in capsys.readouterr().out  # no checkpoints were looked for


def test_a_checkpoint_deleted_after_it_was_measured_does_not_remove_its_row(tmp_path, monkeypatch, capsys):
    import shutil
    from src.finetune import collapse

    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    assert wb.main(["--config", str(config)]) == 0
    full = json.loads((tmp_path / "wb/whitebox_summary.json").read_text())
    shutil.rmtree(tmp_path / "models" / "synthetic_r0_s0")  # freeing space, and the baseline the copies came from
    shutil.rmtree(tmp_path / "models" / "synthetic_r1_s1")
    monkeypatch.setattr(collapse, "load_holdout", lambda data: pytest.fail("nothing is left to measure"))
    calls.clear()
    capsys.readouterr()

    assert wb.main(["--config", str(config)]) == 0

    after = json.loads((tmp_path / "wb/whitebox_summary.json").read_text())
    assert calls == [] and after["missing_cells"] == [] and after["complete"]
    assert "WARNING" not in capsys.readouterr().out  # it was measured: there is nothing to worry about
    assert after["curves"] == full["curves"]  # every point still has all its seeds


def test_summarize_only_leaves_out_a_rerun_cell_and_says_so(tmp_path, monkeypatch, capsys):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    assert wb.main(["--config", str(config)]) == 0
    full = json.loads((tmp_path / "wb/whitebox_summary.json").read_text())
    results = tmp_path / "grid_out/results.jsonl"  # benchmark_near_r1_s1 re-run: a new holdout ppl
    lines = []
    for line in results.read_text().splitlines():
        row = json.loads(line)
        if row["cell"] == "benchmark_near_r1_s1":
            row["holdout_ppl"] = 40.0
        lines.append(json.dumps(row))
    results.write_text("\n".join(lines) + "\n")
    calls.clear()
    capsys.readouterr()

    assert wb.main(["--config", str(config), "--summarize-only"]) == 0

    out = capsys.readouterr().out
    summary = json.loads((tmp_path / "wb/whitebox_summary.json").read_text())
    assert calls == []  # summarising never measures
    assert summary["stale_cells"] == ["benchmark_near_r1_s1"] and not summary["complete"]
    assert "1 row(s) are out of date" in out and "benchmark_near ratio 1: 1 of 2 seeds" in out
    thin = next(c for c in summary["curves"] if (c["type"], c["ratio"]) == ("benchmark_near", 1.0))
    assert thin["n_seeds"] == 1 and thin["eff_rank_last"] != next(
        c for c in full["curves"] if (c["type"], c["ratio"]) == ("benchmark_near", 1.0))["eff_rank_last"]
    assert summary["partial_points"] == [  # (the fixture has the recursion intermediate of seed 0 only)
        "benchmark_near ratio 1: 1 of 2 seeds", "recursion generation 1: 1 of 2 seeds"
    ]

    # a plain run measures it again, and the summary is whole
    assert wb.main(["--config", str(config)]) == 0
    assert [name for name, _, _ in calls] == ["benchmark_near_r1_s1"]
    assert json.loads((tmp_path / "wb/whitebox_summary.json").read_text())["complete"]


def test_cells_outside_the_configured_grid_are_left_out(tmp_path, monkeypatch):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    assert wb.main(["--config", str(config)]) == 0
    grid["seeds"] = [0]  # the grid config is narrowed; seed 1's rows stay in both files
    grid["types"] = ["synthetic"]
    (tmp_path / "grid.yaml").write_text(yaml.safe_dump(grid))
    calls.clear()

    assert wb.main(["--config", str(config), "--summarize-only"]) == 0

    summary = json.loads((tmp_path / "wb/whitebox_summary.json").read_text())
    assert {c["type"] for c in summary["curves"]} == {"synthetic"} and {c["n_seeds"] for c in summary["curves"]} == {1}
    assert summary["partial_points"] == [] and summary["complete"]  # one seed is all this grid has now
    assert [g["n_seeds"] for g in summary["recursion_stages"]] == [1]  # seed 0's stage only


def test_a_configured_cell_with_no_measurement_is_reported_not_hidden(tmp_path, monkeypatch, capsys):
    import shutil

    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    shutil.rmtree(tmp_path / "models" / "benchmark_near_r1_s1")  # never measured, and now cannot be
    capsys.readouterr()
    assert wb.main(["--config", str(config)]) == 0

    out = capsys.readouterr().out
    summary = json.loads((tmp_path / "wb/whitebox_summary.json").read_text())
    assert summary["missing_cells"] == ["benchmark_near_r1_s1"] and not summary["complete"]
    assert "1 configured cell(s) have no measurement" in out
    assert "NOTE: averaged over fewer seeds" in out and "benchmark_near ratio 1: 1 of 2 seeds" in out


def test_a_plain_run_with_nothing_to_measure_does_not_load_the_holdout(tmp_path, monkeypatch, capsys):
    from src.finetune import collapse

    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    assert wb.main(["--config", str(config)]) == 0
    monkeypatch.setattr(collapse, "load_holdout", lambda data: pytest.fail("no holdout needed"))
    capsys.readouterr()
    assert wb.main(["--config", str(config)]) == 0
    assert "0 to do" in capsys.readouterr().out


@pytest.mark.parametrize("change", [{"eps": 0.5}, {"exclude_first_token": False}, {"delta_from_base": False}])
def test_every_setting_that_changes_the_numbers_makes_old_rows_refused(tmp_path, monkeypatch, change):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    assert wb.main(["--config", str(config)]) == 0
    config.write_text(yaml.safe_dump({**cfg, **change}))
    with pytest.raises(ValueError, match="different settings"):
        wb.main(["--config", str(config)])


def test_the_settings_reach_the_measurement(tmp_path, monkeypatch):
    config, cfg, grid, calls, plots = _cli_setup(tmp_path, monkeypatch)
    config.write_text(yaml.safe_dump({**cfg, "eps": 0.3, "exclude_first_token": False}))
    assert wb.main(["--config", str(config)]) == 0
    assert calls and all(kw["eps"] == 0.3 and kw["exclude_first_token"] is False for _, _, kw in calls)


# --- model-level gaps -----------------------------------------------------------------------------------------

def test_the_excluded_position_is_the_first_one(torch_stack):
    """The representations used are exactly the non-pad tokens after position 0, layer by layer."""
    torch, _ = torch_stack
    np = _np()
    model = tiny_model(torch_stack)
    ids, mask = batch_of(torch_stack, [10, 7, 12], seed=11, pad_to=14)
    with torch.no_grad():
        hidden = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
    keep = mask.bool().clone()
    keep[:, 0] = False

    sig = wb.model_signals(model, [(ids, mask)])
    for layer, h in enumerate(hidden):
        stats = wb.RepresentationStats(N_EMBD)
        stats.update(h[keep].double().numpy())
        expected = stats.result()
        for key in ("effective_rank", "participation_ratio", "anisotropy", "n_tokens"):
            assert sig["layers"][layer][key] == pytest.approx(expected[key], rel=1e-6, abs=1e-9), (layer, key)
    assert np is not None


def test_the_mlp_statistic_also_leaves_out_the_first_position(torch_stack):
    torch, _ = torch_stack
    model = tiny_model(torch_stack)
    with torch.no_grad():  # a unit that can only fire at position 0
        for p in model.parameters():
            p.zero_()
        for ln in [b.ln_1 for b in model.transformer.h] + [b.ln_2 for b in model.transformer.h]:
            ln.weight.fill_(1.0)
        model.transformer.ln_f.weight.fill_(1.0)
        model.transformer.wpe.weight[0, 1] = 100.0  # only position 0 has a non-zero embedding
        mlp = model.transformer.h[0].mlp
        mlp.c_fc.weight[1, :] = 1.0  # every unit reads dimension 1
        mlp.c_fc.bias.fill_(-3.0)
    batch = batch_of(torch_stack, [10, 9, 11])

    excluded = wb.model_signals(model, [batch], exclude_first_token=True)["mlp"][0]["dead_fraction"]
    included = wb.model_signals(model, [batch], exclude_first_token=False)["mlp"][0]["dead_fraction"]
    assert excluded == 1.0  # at every other position the input is -3 and GELU(-3) is about -0.004
    assert included == 0.0  # position 0 sees 100/std - 3 and every unit fires


def test_a_deeper_model_averages_its_blocks_by_mean_not_median(torch_stack):
    np = _np()
    model = tiny_model(torch_stack, n_layer=3)
    weights = wb.block_weights(model)
    sig = wb.model_signals(model, [batch_of(torch_stack, [10, 11])])
    per_block = [wb.weight_stats(weights[f"attn_qkv.{i}"])["stable_rank"] for i in range(3)]
    assert len(set(round(v, 6) for v in per_block)) == 3 and np.mean(per_block) != np.median(per_block)
    assert sig["weights"]["attn_qkv"]["stable_rank"] == pytest.approx(np.mean(per_block))
    assert len(sig["layers"]) == 4 and len(sig["mlp"]) == 3


def test_each_weight_kind_is_the_matrix_it_says(torch_stack):
    weights = wb.block_weights(tiny_model(torch_stack))
    shapes = {kind: weights[f"{kind}.0"].shape for kind in wb.WEIGHT_KINDS}
    assert shapes == {"attn_qkv": (16, 48), "attn_out": (16, 16), "mlp_in": (16, 64), "mlp_out": (64, 16)}


def _tiny_checkpoint(torch_stack, directory, padding_side="right"):
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<eos>", "the", "cat", "sat", "on", "mat", "dog", "ran", "far", "away", "home"]
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<eos>", padding_side=padding_side)
    tiny_model(torch_stack).save_pretrained(directory)
    fast.save_pretrained(directory)
    return directory


def test_texts_longer_than_max_len_are_truncated(torch_stack, tmp_path):
    ckpt = _tiny_checkpoint(torch_stack, tmp_path / "c")
    kw = dict(batch_size=2, device="cpu", eps=0.01, exclude_first_token=True, base_model=None)
    texts = ["the cat sat on the mat the dog ran far away home", "cat dog", "the mat"]  # 12, 2, 2 tokens
    sig = wb.compute_checkpoint(ckpt, texts, max_len=6, **kw)
    assert sig["n_tokens"] == (6 - 1) + (2 - 1) + (2 - 1)  # the long text is cut to 6 tokens; position 0 is excluded


def test_a_tokenizer_that_pads_on_the_left_is_measured_as_if_it_padded_on_the_right(torch_stack, tmp_path):
    texts = ["the cat sat on the mat", "cat dog", "the dog ran far away from home", "the mat"]
    kw = dict(max_len=8, batch_size=4, device="cpu", eps=0.01, exclude_first_token=True, base_model=None)
    right = wb.compute_checkpoint(_tiny_checkpoint(torch_stack, tmp_path / "r"), texts, **kw)
    left = wb.compute_checkpoint(_tiny_checkpoint(torch_stack, tmp_path / "l", padding_side="left"), texts, **kw)

    assert left["n_tokens"] == right["n_tokens"]
    for a, b in zip(left["layers"], right["layers"]):
        for key in ("effective_rank", "participation_ratio", "anisotropy"):
            assert a[key] == pytest.approx(b[key], rel=1e-4, abs=1e-6)


def test_the_first_token_and_the_threshold_reach_compute_checkpoint(torch_stack, tmp_path):
    ckpt = _tiny_checkpoint(torch_stack, tmp_path / "c")
    texts = ["the cat sat on the mat", "the dog ran far away home", "cat dog sat"]
    kw = dict(max_len=8, batch_size=2, device="cpu", base_model=None)
    base = wb.compute_checkpoint(ckpt, texts, eps=0.01, exclude_first_token=True, **kw)
    with_first = wb.compute_checkpoint(ckpt, texts, eps=0.01, exclude_first_token=False, **kw)
    loose = wb.compute_checkpoint(ckpt, texts, eps=1e9, exclude_first_token=True, **kw)
    assert with_first["n_tokens"] == base["n_tokens"] + len(texts)
    assert all(m["dead_fraction"] == 1.0 for m in loose["mlp"])
    assert any(m["dead_fraction"] < 1.0 for m in base["mlp"])


def test_the_update_rank_is_the_mean_over_the_matrices_that_changed(torch_stack):
    torch, _ = torch_stack
    np = _np()
    model = tiny_model(torch_stack)
    base = wb.block_weights(model)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():  # a rank-1 update to one block's QKV matrix, a rank-4 update to the other's
        w0, w1 = model.transformer.h[0].attn.c_attn.weight, model.transformer.h[1].attn.c_attn.weight
        w0 += 0.3 * torch.outer(torch.randn(w0.shape[0], generator=g), torch.randn(w0.shape[1], generator=g))
        w1 += 0.3 * torch.randn(w1.shape[0], 4, generator=g) @ torch.randn(4, w1.shape[1], generator=g)
    ranks = [wb.delta_norms(wb.block_weights(model)[k], base[k])[2] for k in ("attn_qkv.0", "attn_qkv.1")]
    assert ranks[0] == pytest.approx(1.0, abs=1e-3) and ranks[1] > 2.0  # different ranks, so mean != max != min

    delta = wb.model_signals(model, [batch_of(torch_stack, [10, 11])], base_weights=base)["delta"]
    assert delta["effective_rank"] == pytest.approx(np.mean(ranks)) and delta["effective_rank"] < max(ranks)
