"""Contract tests for the v1 API (HANDBOOK §7.7): every route in openapi.json, from fixtures."""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

from dashboard.api import app as api  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
SPEC = REPO / "dashboard/api/openapi.json"


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def root(tmp_path):
    """A repo root holding the committed grid reports plus §7 rows for what isn't produced yet."""
    for rel in ("reports/grid", "reports/collapse_spike"):
        shutil.copytree(REPO / rel, tmp_path / rel)
    runs = {}
    for line in (FIXTURES / "trainlog.jsonl").read_text(encoding="utf-8").splitlines():
        runs.setdefault(json.loads(line)["run_id"], []).append(json.loads(line))
    for run_id, rows in runs.items():
        _jsonl(tmp_path / f"reports/trainlogs/{run_id}.jsonl", rows)
    key = {"dataset": "wikitext2", "cell": "synthetic_r0.5_s1"}
    _jsonl(tmp_path / "reports/features/wikitext2.jsonl", [{
        "schema": "features/1", **key, "type": "synthetic", "ratio": 0.5, "seed": 1, "n_samples": 1000,
        "f_lex_ttr": 0.41, "f_lex_entropy": 9.87, "f_sem_mauve": None, "feature_version": "lex@1"}])
    _jsonl(tmp_path / "reports/risk/predictions.jsonl", [{
        "schema": "risk/1", **key, "p_degrade": 0.83, "label": 1, "conformal_set": [1], "alpha": 0.1,
        "model": "logreg@1", "split": "test"}])
    _jsonl(tmp_path / "reports/repair/wikitext2.jsonl", [{
        "schema": "repair/1", **key, "strategy": "harm_guided", "k_replaced": 250, "holdout_ppl": 57.9,
        "benchmark_ppl": 62.0, "distinct_2": 0.66, "recovery": 0.54, "seed": 1}])
    sample = {"prompt": "", "source": "synthetic", "generation": 1, "contamination_ratio": 0.5,
              "benchmark_near_score": 0.0, "split": "train"}
    _jsonl(tmp_path / "bench/wikitext2/synthetic/r0.5/s0.jsonl", [
        *({"sample_id": f"b{i}", "response": f"text {i}", **sample} for i in range(5)),
        {"sample_id": "leak", "response": "holdout text", **sample, "split": "hidden_holdout"}])
    return tmp_path


@pytest.fixture
def client(root):
    return TestClient(api.create_app(root))


# one concrete URL per route in the spec; a new route without an entry fails the coverage test
URLS = {
    "/v1/gates": "/v1/gates",
    "/v1/grids/{dataset}/results": "/v1/grids/wikitext2/results",
    "/v1/grids/{dataset}/curves": "/v1/grids/wikitext2/curves",
    "/v1/grids/{dataset}/whitebox": "/v1/grids/wikitext2/whitebox",
    "/v1/features/{dataset}": "/v1/features/wikitext2",
    "/v1/risk": "/v1/risk",
    "/v1/repair/{dataset}": "/v1/repair/wikitext2",
    "/v1/runs/{run_id}/trainlog": "/v1/runs/wikitext2_synthetic_r1_s0/trainlog",
    "/v1/bench/samples": "/v1/bench/samples?dataset=wikitext2&type=synthetic&ratio=0.5",
}


def test_committed_openapi_json_matches_the_code():
    assert SPEC.read_text(encoding="utf-8") == api.openapi_json(api.create_app()), (
        "dashboard/api/openapi.json is stale: run `python -m dashboard.api --openapi` "
        "(and remember v1 is frozen: additive changes only)")


def test_every_route_in_the_spec_returns_schema_valid_data(client):
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert set(spec["paths"]) == set(URLS)
    route_models = {r.path: r.response_model for r in client.app.routes if hasattr(r, "response_model")}
    for template, url in URLS.items():
        r = client.get(url)
        assert r.status_code == 200, (url, r.text)
        assert r.headers["content-type"] == "application/json"
        rows = TypeAdapter(route_models[template]).validate_python(r.json())
        assert rows, f"{url} returned nothing from the fixtures"


def test_shapes_the_ui_relies_on(client):
    gates = {(g["id"], g["dataset"]): g for g in client.get("/v1/gates").json()}
    assert gates[("gate_2", "wikitext2")]["status"] == "passed"
    assert gates[("gate_3", "wikitext2")]["exploratory"] is True
    assert {g["id"] for g in gates.values()} == {"gate_0", "gate_1", "gate_2", "gate_3", "x1", "x2", "x3"}

    log = client.get("/v1/runs/wikitext2_synthetic_r1_s0/trainlog").json()
    assert set(log[0]["signals"]) == {"eff_rank_last", "part_ratio_last", "anisotropy_last",
                                      "dead_fraction_mean", "stable_rank_attn", "delta_rel_norm"}
    fired = [row for row in log if row["alarm"]]
    assert fired and fired[0]["alarm_signal"] == "anisotropy_last" in fired[0]["signals"]

    feats = client.get("/v1/features/wikitext2").json()[0]["features"]
    assert feats == {"lex_ttr": 0.41, "lex_entropy": 9.87, "sem_mauve": None}


def test_bench_never_serves_holdout_text_and_respects_the_limit(client):
    url = "/v1/bench/samples?dataset=wikitext2&type=synthetic&ratio=0.5"
    ids = [s["sample_id"] for s in client.get(url).json()]
    assert "leak" not in ids and len(ids) == 5
    assert len(client.get(url + "&limit=2").json()) == 2


@pytest.mark.parametrize("url, status", [
    ("/v1/grids/tinystories/results", 404),        # not produced yet
    ("/v1/runs/missing_run/trainlog", 404),
    ("/v1/grids/imagenet/results", 422),            # unknown dataset
    ("/v1/runs/.hidden/trainlog", 422),             # not a plain file name
    ("/v1/runs/..%2F..%2Fsecrets/trainlog", 404),   # never leaves reports/trainlogs
    ("/v1/bench/samples?dataset=wikitext2&type=synthetic&ratio=1.5", 422),
    ("/v1/bench/samples?dataset=wikitext2&type=nope&ratio=0.5", 422),
    ("/v1/nothing-here", 404),
])
def test_errors_are_problem_json(client, url, status):
    r = client.get(url)
    assert r.status_code == status
    assert r.headers["content-type"] == "application/problem+json"
    body = r.json()
    assert body["status"] == status and body["title"] and body["detail"]


def test_etag_gives_304_until_the_report_changes(client, root):
    url = "/v1/grids/wikitext2/results"
    etag = client.get(url).headers["etag"]
    assert etag.startswith('W/"')
    again = client.get(url, headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.headers["etag"] == etag
    path = root / "reports/grid/results.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")  # same rows, new file
    fresh = client.get(url, headers={"If-None-Match": etag})
    assert fresh.status_code == 200 and fresh.headers["etag"] != etag


def test_cors_lets_the_vite_dev_server_read_responses_and_etags(client):
    r = client.get("/v1/gates", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "etag" in r.headers["access-control-expose-headers"].lower()
    assert "access-control-allow-origin" not in client.get(
        "/v1/gates", headers={"Origin": "http://evil.example"}).headers
