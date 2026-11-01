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
    (tmp_path / "reports/risk").mkdir(parents=True)
    shutil.copy(REPO / "reports/risk/evaluation.json", tmp_path / "reports/risk/evaluation.json")
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
    "/v1/risk/evaluation": "/v1/risk/evaluation",
    "/v1/repair/{dataset}": "/v1/repair/wikitext2",
    "/v1/runs/{run_id}/trainlog": "/v1/runs/wikitext2_synthetic_r1_s0/trainlog",
    "/v1/bench/samples": "/v1/bench/samples?dataset=wikitext2&type=synthetic&ratio=0.5",
}


def test_committed_openapi_json_matches_the_code():
    assert SPEC.read_text(encoding="utf-8") == api.openapi_json(api.create_app()), (
        "dashboard/api/openapi.json is stale: run `python -m dashboard.api --openapi` "
        "(and remember v1 is frozen: additive changes only)")


# the audit job routes and the SSE stream have their own tests below
AUDIT_PATHS = {"/v1/audits", "/v1/audits/{audit_id}", "/v1/runs/{run_id}/stream"}  # + the SSE stream


def test_every_route_in_the_spec_returns_schema_valid_data(client):
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert set(spec["paths"]) == set(URLS) | AUDIT_PATHS
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
    g1 = gates[("gate_1", None)]  # from reports/risk/evaluation.json, one dataset so far
    assert g1["status"] == "not_evaluable" and g1["source"] == "reports/risk/evaluation.json"
    assert 0 < g1["details"]["auroc"] < 1 and g1["details"]["coverage_target"] == 0.9
    assert ">= 2 datasets" in g1["rule"]
    ev = client.get("/v1/risk/evaluation").json()
    assert ev["leave_one_type_out"]["auroc"]["value"] == g1["details"]["auroc"]
    assert ev["baselines"]["ref_lm_loss"]["model_beats_baseline"] is g1["details"]["beats_ref_lm_baseline"]

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


# ---------------------------------------------------------------- audit jobs (#34)

DEPLOYED = {"model": "logreg@1", "cols": ["f_lex_entropy"], "mean": [3.0], "std": [1.0], "w": [-2.0],
            "b": 0.0, "alpha": 0.1, "q": 0.5}
JSONL = b'{"id": "a", "text": "the cat sat on the mat"}\n\n{"prompt": "q", "response": "an answer"}\n'


def _report(n):
    return {"schema": "audit/1", "n_samples": n, "model": "logreg@1", "p_degrade": 0.7,
            "conformal_set": [1], "alpha": 0.1, "features": {"f_lex_entropy": 2.5, "f_sem_mauve": None},
            "missing_features": [], "contributions": [{"feature": "f_lex_entropy", "value": 2.5,
                                                       "contribution": 1.0}],
            "n_flagged": 0, "flagged": []}


@pytest.fixture
def audits(root):
    """A client whose audits run a stub; ``calls`` records what each audit was given."""
    (root / "reports/risk").mkdir(parents=True, exist_ok=True)
    (root / "reports/risk/model.json").write_text(json.dumps(DEPLOYED), encoding="utf-8")
    calls = []

    def runner(samples, deployed, bench, reference):
        calls.append((samples, deployed, bench, reference))
        if samples[0].response == "boom":
            raise RuntimeError("the model fell over")
        return _report(len(samples)), [{"sample_id": s.sample_id, "method": "risk_reflm", "score": -3.0}
                                       for s in samples]

    return TestClient(api.create_app(root, runner=runner)), calls


def _finish(client, location, seconds=10.0):
    import time

    deadline = time.monotonic() + seconds
    while True:
        job = client.get(location).json()
        if job["status"] in ("done", "failed") or time.monotonic() > deadline:
            return job
        time.sleep(0.02)


def test_an_upload_becomes_a_job_that_finishes_with_a_typed_report(audits):
    client, calls = audits
    r = client.post("/v1/audits?name=my data.jsonl", content=JSONL,
                    headers={"Content-Type": "application/x-ndjson"})
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "queued" and job["n_samples"] == 2 and job["report"] is None
    assert r.headers["location"] == f"/v1/audits/{job['id']}"
    done = _finish(client, r.headers["location"])
    assert done["status"] == "done" and done["finished_at"] and done["error"] is None
    report = api.AuditReport.model_validate(done["report"])
    assert done["report"]["schema"] == "audit/1" and report.p_degrade == 0.7
    assert [s["sample_id"] for s in done["scores"]] == ["a", "my data-0000001"]
    assert done["scores"][0]["higher_is"] == "worse"
    samples, deployed, bench, reference = calls[0]
    assert [s.response for s in samples] == ["the cat sat on the mat", "an answer"]  # same aliases as files
    assert deployed == DEPLOYED and bench == [] and reference == []


def test_a_benchmark_brings_its_grid_s_reference_for_the_sem_features(audits, monkeypatch):
    from src.risk import audit as audit_mod

    client, calls = audits
    monkeypatch.setattr(audit_mod, "grid_benchmark", lambda cfg: ["leaked item"])
    monkeypatch.setattr(audit_mod, "grid_reference", lambda cfg: ["clean human text"])
    r = client.post("/v1/audits?benchmark=wikitext2", content=JSONL)
    assert _finish(client, r.headers["location"])["status"] == "done"
    assert calls[0][2:] == (["leaked item"], ["clean human text"])


def test_a_failing_audit_is_reported_not_raised(audits):
    client, _ = audits
    r = client.post("/v1/audits", content=b'{"text": "boom"}\n')
    done = _finish(client, r.headers["location"])
    assert done["status"] == "failed" and "the model fell over" in done["error"] and done["report"] is None


@pytest.mark.parametrize("body, query, status, detail", [
    (b'{"text": "ok"}\nnot json\n', "", 422, "line 2"),
    (b"[1, 2]\n", "", 422, "line 1"),
    (b'{"text": ""}\n', "", 422, "line 1"),            # a record without text fails Sample validation
    (b"\n\n", "", 422, "no records"),
    (b"\xff\xfe\x00", "", 422, "UTF-8"),
    (JSONL, "?benchmark=imagenet", 422, "benchmark"),
    (JSONL, "?name=../../etc", 422, "name"),
    (JSONL, "?benchmark=tinystories", 503, "benchmark data"),  # no grid data in this root
])
def test_bad_uploads_are_problem_json_and_start_nothing(audits, body, query, status, detail):
    client, calls = audits
    r = client.post("/v1/audits" + query, content=body)
    assert r.status_code == status, r.text
    assert r.headers["content-type"] == "application/problem+json"
    assert detail in r.json()["detail"]
    assert calls == []


def test_size_limits(audits, monkeypatch):
    client, calls = audits
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    assert client.post("/v1/audits", content=JSONL).status_code == 413
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 2**20)
    monkeypatch.setattr(api, "MAX_SAMPLES", 1)
    r = client.post("/v1/audits", content=JSONL)
    assert r.status_code == 413 and "more than 1 records" in r.json()["detail"] and calls == []


def test_no_deployed_model_is_a_503_that_says_what_to_run(root):
    r = TestClient(api.create_app(root, runner=lambda *a: None)).post("/v1/audits", content=JSONL)
    assert r.status_code == 503 and "src.risk.classifier" in r.json()["detail"]


def test_unknown_and_malformed_audit_ids(audits):
    client, _ = audits
    assert client.get("/v1/audits/" + "0" * 32).status_code == 404
    assert client.get("/v1/audits/not-an-id").status_code == 422


def test_cors_allows_the_ui_to_post_and_read_the_location(audits):
    client, _ = audits
    pre = client.options("/v1/audits", headers={"Origin": "http://localhost:5173",
                                                "Access-Control-Request-Method": "POST",
                                                "Access-Control-Request-Headers": "content-type"})
    assert pre.status_code == 200 and "POST" in pre.headers["access-control-allow-methods"]
    r = client.post("/v1/audits", content=JSONL, headers={"Origin": "http://localhost:5173"})
    assert "location" in r.headers["access-control-expose-headers"].lower()


def test_the_default_runner_wires_the_audit_and_both_scores(monkeypatch):
    from src.features import model as reflm
    from src.features import semantic
    from src.features.suite import FAMILIES
    from src.ingestion.schema import Sample
    from src.risk import audit as audit_mod

    monkeypatch.setattr(audit_mod, "FAMILIES", {k: FAMILIES[k] for k in ("lex", "leak")})  # no models
    monkeypatch.setattr(reflm, "per_text", lambda texts, ctx: [{"loss": 2.0}, None])
    monkeypatch.setattr(semantic, "per_text", lambda texts, ctx: [0.25, 0.5])
    samples = [Sample(sample_id="x", response="the quick brown fox jumps"),
               Sample(sample_id="y", response="the quick brown fox jumps over the lazy dog")]
    report, scores = api.run_audit(samples, DEPLOYED, ["the quick brown fox jumps over the lazy dog"])
    assert api.AuditReport.model_validate(report).n_samples == 2
    assert report["n_flagged"] == 1 and report["flagged"][0]["sample_id"] == "y"  # the benchmark copy
    assert scores == [{"sample_id": "x", "method": "risk_reflm", "score": -2.0},
                      {"sample_id": "y", "method": "risk_reflm", "score": None},
                      {"sample_id": "x", "method": "risk_knn", "score": -0.25},
                      {"sample_id": "y", "method": "risk_knn", "score": -0.5}]

# ---------------------------------------------------------------- the trainlog SSE stream (#36)

RUN = "wikitext2_synthetic_r1_s0"


def _events(response):
    """Parse an SSE body into (event, id, data) triples."""
    out, cur = [], {}
    for line in response.iter_lines():
        if not line:
            if cur:
                out.append((cur.get("event"), cur.get("id"), json.loads(cur.get("data", "null"))))
            cur = {}
        elif not line.startswith(":"):
            key, _, value = line.partition(": ")
            cur[key] = value
    return out


@pytest.fixture
def fast_stream(monkeypatch):
    monkeypatch.setattr(api, "STREAM_POLL_SECONDS", 0.02)
    monkeypatch.setattr(api, "STREAM_IDLE_SECONDS", 0.3)


def _age(path, seconds=10):
    import os
    import time

    os.utime(path, (time.time() - seconds, time.time() - seconds))


def test_a_finished_run_replays_every_row_then_ends(client, root, fast_stream):
    _age(root / f"reports/trainlogs/{RUN}.jsonl")
    with client.stream("GET", f"/v1/runs/{RUN}/stream") as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = _events(r)
    rows = [e for e in events if e[0] == "row"]
    assert [int(i) for _, i, _ in rows] == list(range(0, 250, 25))
    assert rows[5][2]["alarm"] and rows[5][2]["alarm_signal"] == "anisotropy_last"  # same shape as /trainlog
    assert rows[0][2] == client.get(f"/v1/runs/{RUN}/trainlog").json()[0]
    assert events[-1][0] == "end"


def test_last_event_id_resumes_after_that_step(client, root, fast_stream):
    _age(root / f"reports/trainlogs/{RUN}.jsonl")
    with client.stream("GET", f"/v1/runs/{RUN}/stream", headers={"Last-Event-ID": "175"}) as r:
        steps = [int(i) for e, i, _ in _events(r) if e == "row"]
    assert steps == [200, 225]


def test_rows_written_during_the_stream_arrive_live_and_a_restart_resets(client, root, fast_stream,
                                                                         monkeypatch):
    import threading
    import time

    monkeypatch.setattr(api, "STREAM_IDLE_SECONDS", 1.0)  # well above the writer's 0.1 s gaps, even on slow CI
    path = root / "reports/trainlogs/live_run.jsonl"
    first = json.loads((root / f"reports/trainlogs/{RUN}.jsonl").read_text(encoding="utf-8").splitlines()[0])

    def row(step):
        return json.dumps({**first, "run_id": "live_run", "step": step}) + "\n"

    path.write_text(row(0), encoding="utf-8")

    def writer():
        time.sleep(0.1)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(row(10))
        time.sleep(0.1)
        path.write_text(row(0), encoding="utf-8")  # a new run truncates the log
        time.sleep(0.1)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(row(5))

    t = threading.Thread(target=writer)
    t.start()
    with client.stream("GET", "/v1/runs/live_run/stream") as r:
        events = [(e, i) for e, i, _ in _events(r)]
    t.join()
    assert events == [("row", "0"), ("row", "10"), ("reset", None), ("row", "0"), ("row", "5"), ("end", None)]


def test_stream_errors_are_problem_json(client):
    assert client.get("/v1/runs/missing_run/stream").status_code == 404
    r = client.get("/v1/runs/.hidden/stream")
    assert r.status_code == 422 and r.headers["content-type"] == "application/problem+json"