"""The v1 routes, response models and error format. ``create_app(root)`` serves ``root/reports``."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.finetune.grid import CONTAMINATION_TYPES
from src.ingestion.loaders import normalize_record
from src.ingestion.schema import Sample

REPO = Path(__file__).resolve().parents[2]
API_VERSION = "1.3.0"  # 1.1: audits; 1.2: GET /v1/runs/{id}/stream; 1.3: GET /v1/risk/evaluation

Dataset = Literal["wikitext2", "tinystories"]
ContaminationType = Literal[CONTAMINATION_TYPES]  # type: ignore[valid-type]
GRID_DIRS: dict[str, str] = {"wikitext2": "reports/grid", "tinystories": "reports/grid_tinystories"}
GRID_CONFIGS: dict[str, str] = {"wikitext2": "configs/grid.yaml", "tinystories": "configs/grid_tinystories.yaml"}
DEPLOYED_MODEL = "reports/risk/model.json"  # written by python -m src.risk.classifier
MAX_UPLOAD_BYTES = 20 * 2**20
MAX_SAMPLES = 20_000
STREAM_POLL_SECONDS = 0.5   # how often a stream checks its trainlog for new rows
STREAM_IDLE_SECONDS = 60.0  # a log quiet this long is a finished run: the stream ends
AuditId = Annotated[str, PathParam(pattern=r"^[0-9a-f]{32}$")]
# a file name, never a path: no separators, no leading dot
RunId = Annotated[str, PathParam(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]


# ---------------------------------------------------------------- response models (the contract)

class Model(BaseModel):
    # report rows may carry fields the contract does not promise; the response leaves them out
    model_config = ConfigDict(extra="ignore")


class Problem(Model):
    """RFC 7807 problem details, sent as application/problem+json for every error."""

    type: str = "about:blank"
    title: str
    status: int
    detail: Optional[str] = None
    instance: Optional[str] = None


class Gate(Model):
    id: Literal["gate_0", "gate_1", "gate_2", "gate_3", "x1", "x2", "x3"]
    name: str
    dataset: Optional[Dataset] = None
    status: Literal["passed", "failed", "not_evaluable", "pending"]
    exploratory: bool = False
    rule: str
    details: Optional[dict[str, Any]] = None
    source: Optional[str] = None  # report file the status comes from, relative to the repo


class GridResult(Model):
    """One grid cell's outcome (§7.1)."""

    cell: str
    type: str
    ratio: float
    seed: int
    n_train: int
    n_contaminated: int
    holdout_ppl: float
    benchmark_ppl: float
    distinct_1: float
    distinct_2: float
    seconds: float
    config: str
    shared_baseline: bool
    pool_version: Optional[str] = None


class Curve(Model):
    """Mean and min-max over seeds of one (type, ratio) point."""

    type: str
    ratio: float
    n_seeds: int
    holdout_ppl_mean: float
    holdout_ppl_min: float
    holdout_ppl_max: float
    benchmark_ppl_mean: float
    benchmark_ppl_min: float
    benchmark_ppl_max: float


class WhiteboxCell(Model):
    """White-box signals of one checkpoint (the headline numbers of whitebox.jsonl).

    ``kind``: a grid ``cell``; a ratio-0 ``shared_baseline`` (same checkpoint as another type's);
    or a ``recursion_stage``, an intermediate model of the recursive chain, which has no ratio
    and carries its ``generation``.
    """

    cell: str
    type: str
    ratio: Optional[float]
    seed: int
    kind: Literal["cell", "shared_baseline", "recursion_stage"]
    generation: Optional[int] = None
    signals: dict[str, Optional[float]]


class FeatureRow(Model):
    """One grid cell's dataset features (§7.2); ``f_<family>_<name>`` columns go into ``features``."""

    dataset: Dataset
    cell: str
    type: str
    ratio: float
    seed: int
    n_samples: int
    feature_version: str
    features: dict[str, Optional[float]]


class RiskPrediction(Model):
    """§7.4."""

    dataset: Dataset
    cell: str
    p_degrade: float
    label: int
    conformal_set: list[int]
    alpha: float
    model: str
    split: Literal["train", "cal", "test"]


class Estimate(Model):
    value: Optional[float] = None
    ci95: Optional[list[float]] = None


class ReliabilityBin(Model):
    lo: float
    hi: float
    n: int
    confidence: float
    accuracy: float


class ConformalCoverage(Model):
    alpha: float
    target: float
    coverage: Estimate
    mean_set_size: float
    n_cal_per_fold: list[int] = []


class RiskScores(Model):
    """How the risk model scores on held-out groups (``src.risk.classifier.evaluate``)."""

    n: int
    n_degraded: int
    auroc: Optional[Estimate] = None
    auprc: Optional[Estimate] = None
    ece: Optional[float] = None
    reliability: list[ReliabilityBin] = []
    conformal: Optional[ConformalCoverage] = None


class BaselineComparison(Model):
    column: str
    n: int
    auroc_model: float
    auroc_baseline: float
    difference: Estimate
    model_beats_baseline: bool


class RiskBaselines(Model):
    ref_lm_loss: Optional[BaselineComparison] = None   # H2: the model vs one reference-LM feature
    single_feature_auroc: dict[str, Optional[float]] = {}


class Gate1(Model):
    status: Literal["passed", "failed", "not_evaluable"]
    rule: str
    datasets: list[str]


class RiskEvaluation(Model):
    """``reports/risk/evaluation.json`` (``python -m src.risk.classifier``): GATE 1's numbers."""

    model: str
    label: str
    features: list[str]
    leave_one_type_out: RiskScores
    leave_one_dataset_out: Optional[RiskScores] = None
    baselines: RiskBaselines
    gate_1: Gate1


class RepairResult(Model):
    """§7.5. ``recovery`` = (ppl_contaminated - ppl_repaired) / (ppl_contaminated - ppl_clean)."""

    dataset: Dataset
    cell: str
    strategy: str
    k_replaced: int
    holdout_ppl: float
    benchmark_ppl: float
    distinct_2: float
    recovery: Optional[float]
    seed: int


class TrainlogRow(Model):
    """§7.6, with the ``sig_`` columns gathered into ``signals`` (prefix dropped)."""

    run_id: str
    step: int
    holdout_loss: Optional[float]
    signals: dict[str, Optional[float]]
    alarm: bool
    alarm_signal: Optional[str]  # a key of ``signals``


class BenchSample(Model):
    sample_id: str
    prompt: str
    response: str
    source: str
    generation: int
    contamination_ratio: float
    benchmark_near_score: float
    split: str


class Contribution(Model):
    feature: str
    value: Optional[float]
    contribution: float


class FlaggedSample(Model):
    index: int
    sample_id: str
    text: str
    benchmark_similarity: Optional[float]
    repeat_3gram: Optional[float]
    reasons: list[str]


class AuditReport(Model):
    """The ``audit/1`` report of ``src.risk.audit`` (what the Audit screen renders)."""

    schema_: Literal["audit/1"] = Field(alias="schema")
    n_samples: int
    model: str
    p_degrade: float
    conformal_set: list[int]
    alpha: float
    features: dict[str, Optional[float]]
    missing_features: list[str]
    contributions: list[Contribution]
    n_flagged: int
    flagged: list[FlaggedSample]


class SampleScore(Model):
    """A per-sample risk score (§7.3, without ``true_source``, which an upload does not have)."""

    sample_id: str
    method: Literal["risk_reflm", "risk_knn"]
    score: Optional[float]
    higher_is: Literal["worse"] = "worse"


class AuditJob(Model):
    id: str
    status: Literal["queued", "running", "done", "failed"]
    name: Optional[str] = None
    n_samples: int
    benchmark: Optional[Dataset] = None
    created_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None
    report: Optional[AuditReport] = None   # once done
    scores: Optional[list[SampleScore]] = None


# ---------------------------------------------------------------- reading reports

def _problem(status: int, title: str, detail: Optional[str], instance: str) -> JSONResponse:
    body = Problem(title=title, status=status, detail=detail, instance=instance)
    return JSONResponse(body.model_dump(), status_code=status, media_type="application/problem+json")


class Reports:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, rel: str) -> Path:
        return self.root / rel

    def etag(self, *rels: str) -> str:
        """Weak ETag over the files a response is built from (path, mtime, size; missing counts too)."""
        h = hashlib.sha256()
        for rel in rels:
            p = self.path(rel)
            st = p.stat() if p.exists() else None
            h.update(f"{rel}:{st.st_mtime_ns if st else '-'}:{st.st_size if st else '-'};".encode())
        return f'W/"{h.hexdigest()[:20]}"'

    def _require(self, rel: str) -> Path:
        p = self.path(rel)
        if not p.is_file():
            raise HTTPException(404, f"{rel} has not been produced yet")
        return p

    def json(self, rel: str) -> Any:
        return json.loads(self._require(rel).read_text(encoding="utf-8"))

    def jsonl(self, rel: str) -> list[dict[str, Any]]:
        lines = self._require(rel).read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def json_or_none(self, rel: str) -> Any:
        return self.json(rel) if self.path(rel).is_file() else None


def _cached(request: Request, response: Response, etag: str) -> Optional[Response]:
    """A 304 when the client already has this version; otherwise tag the response."""
    if etag in [t.strip() for t in request.headers.get("if-none-match", "").split(",")]:
        return Response(status_code=304, headers={"ETag": etag})
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "no-cache"
    return None


_RULES = {
    "gate_0": "Recursive training on a model's own output raises hidden-holdout perplexity on every seed.",
    "gate_1": "The dataset risk classifier beats the pre-registered baseline (HANDBOOK §8.3).",
    "gate_2": "Every collapse type degrades: mean holdout perplexity rises with ratio (within the "
              "monotone tolerance) and clears the seed spread at the low dose.",
    "gate_3": "A white-box signal separates collapsing from never-collapsing trajectories above the "
              "permutation baseline (Bonferroni) with lead time > 0 at the allowed false-alarm rate.",
    "x1": "Harm-guided repair recovers >= 10 points more than random replacement at ratio 0.5 (§11).",
    "x2": "A small canary run predicts the target model's damage ranking (§11).",
    "x3": "A live alarm fires before collapse at a calibrated false-alarm rate (§11).",
}
_NAMES = {"gate_0": "GATE 0: collapse spike", "gate_1": "GATE 1: risk model",
          "gate_2": "GATE 2: dose-response", "gate_3": "GATE 3: early warning",
          "x1": "X1: harm-based repair", "x2": "X2: canary run", "x3": "X3: live collapse alarm"}
_EXPLORATORY = {("gate_3", "wikitext2")}  # HANDBOOK §13 D7


def _gates(reports: Reports) -> list[Gate]:
    def gate(id_: str, status: str, dataset: Optional[str] = None, details: Any = None,
             source: Optional[str] = None) -> Gate:
        return Gate(id=id_, name=_NAMES[id_], dataset=dataset, status=status, rule=_RULES[id_],
                    exploratory=(id_, dataset) in _EXPLORATORY, details=details, source=source)

    out = []
    spike = reports.json_or_none("reports/collapse_spike/metrics.json")
    out.append(gate("gate_0", "pending") if spike is None else gate(
        "gate_0", "passed" if spike["gate_0_passed"] else "failed",
        details={"per_seed": spike["gate_0_per_seed"]}, source="reports/collapse_spike/metrics.json"))
    ev = reports.json_or_none("reports/risk/evaluation.json")
    if ev is None:
        out.append(gate("gate_1", "pending"))
    else:
        lt, base = ev["leave_one_type_out"], ev["baselines"].get("ref_lm_loss") or {}
        conformal = lt.get("conformal") or {}
        g1 = gate("gate_1", ev["gate_1"]["status"], source="reports/risk/evaluation.json", details={
            "datasets": ev["gate_1"]["datasets"], "auroc": (lt.get("auroc") or {}).get("value"),
            "ece": lt.get("ece"), "coverage": (conformal.get("coverage") or {}).get("value"),
            "coverage_target": conformal.get("target"),
            "beats_ref_lm_baseline": base.get("model_beats_baseline")})
        out.append(g1.model_copy(update={"rule": ev["gate_1"]["rule"]}))
    for dataset, folder in GRID_DIRS.items():
        summary = reports.json_or_none(f"{folder}/summary.json")
        if summary is not None:
            g2 = summary["gate"]
            out.append(gate("gate_2", g2["status"], dataset, source=f"{folder}/summary.json", details={
                "rule": g2["rule"], "types": {t: v["status"] for t, v in g2["types"].items()},
                "complete": summary["complete"]}))
        ew = reports.json_or_none(f"{folder}/earlywarning.json")
        if ew is not None:
            out.append(gate("gate_3", ew["status"], dataset, source=f"{folder}/earlywarning.json", details={
                "trajectories": ew["trajectories"], "never_collapse": ew["never_collapse"],
                "signals": {k: {"auroc": v.get("auroc"), "permutation_p": v.get("permutation_p"),
                                "warns": v.get("warns")} for k, v in ew["signals"].items()}}))
    out += [gate(x, "pending") for x in ("x1", "x2", "x3")]
    return out


def _with_prefix(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {k[len(prefix):]: v for k, v in row.items() if k.startswith(prefix)}


def _trainlog_row(r: dict[str, Any]) -> dict[str, Any]:
    """A trainlog/1 file row as the API serves it (``TrainlogRow``)."""
    return {"run_id": r["run_id"], "step": r["step"], "holdout_loss": r["holdout_loss"],
            "signals": _with_prefix(r, "sig_"), "alarm": r["alarm"],
            "alarm_signal": r["alarm_signal"] and r["alarm_signal"].removeprefix("sig_")}


# ---------------------------------------------------------------- audit jobs

# (samples, deployed model, benchmark texts, reference texts) -> (audit/1 report, per-sample score rows)
Runner = Callable[[list[Sample], Mapping[str, Any], list[str], list[str]],
                  tuple[dict[str, Any], list[dict[str, Any]]]]


def run_audit(samples: list[Sample], deployed: Mapping[str, Any], bench: list[str],
              reference: list[str] = ()) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The audit/1 report (``src.risk.audit``) and the two per-sample risk scores (§7.3)."""
    from src.features import model as reflm
    from src.features import semantic
    from src.features.suite import _device
    from src.finetune.collapse import training_text
    from src.risk.audit import audit

    texts, ids = [training_text(s) for s in samples], [s.sample_id for s in samples]
    report = audit(texts, ids, deployed, bench, reference)
    ctx = {"device": _device("auto")}
    scores: list[dict[str, Any]] = []
    for method, per_text, value in (
        ("risk_reflm", reflm.per_text, lambda v: None if v is None else -v["loss"]),
        ("risk_knn", semantic.per_text, lambda v: None if v is None else -v),
    ):
        try:
            values = [value(v) for v in per_text(texts, ctx)]
        except ImportError:  # the family's libraries are not installed on this server
            values = [None] * len(texts)
        scores += [{"sample_id": i, "method": method, "score": v} for i, v in zip(ids, values)]
    return report, scores


def parse_upload(body: bytes, name: str) -> list[Sample]:
    """JSONL records -> validated Samples, the way ``load_dataset`` reads a .jsonl file."""
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, "the upload is not UTF-8 text") from None
    samples: list[Sample] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("expected a JSON object")
            samples.append(normalize_record(record, len(samples), stem=name))
        except (ValueError, TypeError) as e:
            raise HTTPException(422, f"line {n}: {str(e)[:200]}") from None
        if len(samples) > MAX_SAMPLES:
            raise HTTPException(413, f"more than {MAX_SAMPLES} records")
    if not samples:
        raise HTTPException(422, "no records: send one JSON object per line")
    return samples


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- the app

def create_app(root: Path = REPO, *, runner: Optional[Runner] = None) -> FastAPI:
    """``runner`` does an audit's work (default ``run_audit``); tests pass a stub."""
    reports = Reports(root)
    run = runner or run_audit
    # Descriptions are explicit: the default is Python's status phrase, which 3.13 changed for
    # 422, and the committed spec must not depend on the Python version.
    problem = {"model": Problem, "content": {"application/problem+json": {}}}
    app = FastAPI(
        title="ORIGIN-T API", version=API_VERSION,
        description="Read-only access to ORIGIN-T's reports (HANDBOOK §7.7). Errors are RFC 7807 "
                    "problem+json. Responses carry a weak ETag; send it as If-None-Match for a 304.",
        responses={404: {**problem, "description": "Not found, or not produced yet"},
                   422: {**problem, "description": "Invalid request parameters"}},
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"], allow_headers=["If-None-Match", "Content-Type"],
        expose_headers=["ETag", "Location"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        titles = {404: "Not found", 405: "Method not allowed", 413: "Upload too large",
                  422: "Invalid request", 503: "Not available on this server"}
        return _problem(exc.status_code, titles.get(exc.status_code, "Error"), str(exc.detail),
                        request.url.path)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return _problem(422, "Invalid request", detail, request.url.path)

    def grid_file(dataset: str, name: str) -> str:
        return f"{GRID_DIRS[dataset]}/{name}"

    @app.get("/v1/gates", response_model=list[Gate], tags=["gates"])
    def gates(request: Request, response: Response) -> Any:
        files = ["reports/collapse_spike/metrics.json", "reports/risk/evaluation.json",
                 *(f"{d}/{n}" for d in GRID_DIRS.values() for n in ("summary.json", "earlywarning.json"))]
        return _cached(request, response, reports.etag(*files)) or _gates(reports)

    @app.get("/v1/grids/{dataset}/results", response_model=list[GridResult], tags=["grids"])
    def grid_results(dataset: Dataset, request: Request, response: Response) -> Any:
        rel = grid_file(dataset, "results.jsonl")
        return _cached(request, response, reports.etag(rel)) or reports.jsonl(rel)

    @app.get("/v1/grids/{dataset}/curves", response_model=list[Curve], tags=["grids"])
    def grid_curves(dataset: Dataset, request: Request, response: Response) -> Any:
        rel = grid_file(dataset, "summary.json")
        return _cached(request, response, reports.etag(rel)) or reports.json(rel)["curves"]

    @app.get("/v1/grids/{dataset}/whitebox", response_model=list[WhiteboxCell], tags=["grids"])
    def grid_whitebox(dataset: Dataset, request: Request, response: Response) -> Any:
        rel = grid_file(dataset, "whitebox.jsonl")
        return _cached(request, response, reports.etag(rel)) or [
            {**r, "signals": r["headline"]} for r in reports.jsonl(rel)]

    @app.get("/v1/features/{dataset}", response_model=list[FeatureRow], tags=["features"])
    def features(dataset: Dataset, request: Request, response: Response) -> Any:
        rel = f"reports/features/{dataset}.jsonl"
        return _cached(request, response, reports.etag(rel)) or [
            {**r, "features": _with_prefix(r, "f_")} for r in reports.jsonl(rel)]

    @app.get("/v1/risk", response_model=list[RiskPrediction], tags=["risk"])
    def risk(request: Request, response: Response) -> Any:
        rel = "reports/risk/predictions.jsonl"
        return _cached(request, response, reports.etag(rel)) or reports.jsonl(rel)

    @app.get("/v1/risk/evaluation", response_model=RiskEvaluation, tags=["risk"])
    def risk_evaluation(request: Request, response: Response) -> Any:
        """The risk model's held-out AUROC/AUPRC/ECE, conformal coverage, baselines and GATE 1."""
        rel = "reports/risk/evaluation.json"
        return _cached(request, response, reports.etag(rel)) or reports.json(rel)

    @app.get("/v1/repair/{dataset}", response_model=list[RepairResult], tags=["repair"])
    def repair(dataset: Dataset, request: Request, response: Response) -> Any:
        rel = f"reports/repair/{dataset}.jsonl"
        return _cached(request, response, reports.etag(rel)) or reports.jsonl(rel)

    @app.get("/v1/runs/{run_id}/trainlog", response_model=list[TrainlogRow], tags=["runs"])
    def trainlog(run_id: RunId, request: Request, response: Response) -> Any:
        rel = f"reports/trainlogs/{run_id}.jsonl"
        return _cached(request, response, reports.etag(rel)) or [_trainlog_row(r) for r in reports.jsonl(rel)]

    stream_doc = {200: {"description": "Server-Sent Events: `event: row` (`id` = step, `data` = one "
                        "TrainlogRow as JSON) for every logged row, existing ones first, then live; "
                        "`event: reset` if a new run restarts the log; `event: end` once the log has "
                        "been quiet for a while (a finished run ends right after its replay). "
                        "Send `Last-Event-ID` to resume after a step.",
                        "content": {"text/event-stream": {"schema": {"type": "string"}}}}}

    @app.get("/v1/runs/{run_id}/stream", tags=["runs"], response_class=StreamingResponse,
             response_model=None, responses=stream_doc)
    async def trainlog_stream(run_id: RunId, request: Request) -> StreamingResponse:
        """The run's trainlog as it is written (X3 live monitor)."""
        path = reports.path(f"reports/trainlogs/{run_id}.jsonl")
        if not path.is_file():
            raise HTTPException(404, f"reports/trainlogs/{run_id}.jsonl has not been produced yet")
        last = request.headers.get("last-event-id", "")
        after = int(last) if last.isdigit() else -1

        async def events() -> Any:
            pos, buf, after_step = 0, "", after
            while not await request.is_disconnected():
                size = path.stat().st_size if path.exists() else 0
                if size < pos:  # truncated: a new run started this log, so send it all
                    pos, buf, after_step = 0, "", -1
                    yield "event: reset\ndata: {}\n\n"
                if size > pos:
                    with path.open("rb") as fh:
                        fh.seek(pos)
                        chunk = fh.read(size - pos)
                    pos += len(chunk)
                    *lines, buf = (buf + chunk.decode("utf-8")).split("\n")
                    for line in lines:
                        if line.strip():
                            row = _trainlog_row(json.loads(line))
                            if row["step"] > after_step:
                                yield f"id: {row['step']}\nevent: row\ndata: {json.dumps(row)}\n\n"
                elif not path.exists() or time.time() - path.stat().st_mtime > STREAM_IDLE_SECONDS:
                    yield "event: end\ndata: {}\n\n"
                    return
                await asyncio.sleep(STREAM_POLL_SECONDS)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/v1/bench/samples", response_model=list[BenchSample], tags=["bench"])
    def bench_samples(
        request: Request, response: Response, dataset: Dataset, type: ContaminationType,
        ratio: Annotated[float, Query(ge=0, le=1)], seed: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> Any:
        rel = f"bench/{dataset}/{type}/r{ratio:g}/s{seed}.jsonl"
        cached = _cached(request, response, reports.etag(rel))
        if cached:
            return cached
        # belt and braces: a bench file must never hold holdout text, but never serve it either
        return [s for s in reports.jsonl(rel) if s.get("split") != "hidden_holdout"][:limit]

    # ponytail: jobs live in memory and are lost on restart; persist them (reports/audits/)
    # if audits must outlive the server. One worker: audits share one GPU and its loaded models.
    jobs: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audit")

    def update(job_id: str, **fields: Any) -> None:
        with lock:
            jobs[job_id] = {**jobs[job_id], **fields}

    def work(job_id: str, samples: list[Sample], deployed: Mapping[str, Any], bench: list[str],
             reference: list[str]) -> None:
        update(job_id, status="running")
        try:
            report, scores = run(samples, deployed, bench, reference)
            update(job_id, status="done", report=report, scores=scores, finished_at=_now())
        except Exception as e:  # the client sees it as the job's error
            update(job_id, status="failed", error=f"{type(e).__name__}: {e}"[:500], finished_at=_now())

    upload = {"requestBody": {"required": True, "description": "The dataset: one JSON object per line "
                              "(prompt/response, text, ... as the loaders accept), UTF-8.",
                              "content": {"application/x-ndjson": {"schema": {"type": "string",
                                                                              "format": "binary"}}}}}

    @app.post("/v1/audits", status_code=202, response_model=AuditJob, tags=["audits"],
              openapi_extra=upload, responses={413: {**problem, "description": "Upload too large"},
                                               503: {**problem, "description": "No deployed risk model, "
                                                     "or the benchmark data is not on this server"}})
    async def create_audit(
        request: Request, response: Response, benchmark: Optional[Dataset] = None,
        name: Annotated[str, Query(max_length=120, pattern=r"^[\w .-]+$")] = "upload",
    ) -> Any:
        """Start an audit of the uploaded dataset. Poll ``GET /v1/audits/{id}`` for the report.

        ``benchmark`` also checks every sample for near-copies of that grid's benchmark items.
        """
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"the upload is over {MAX_UPLOAD_BYTES // 2**20} MB")
        body = await request.body()
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"the upload is over {MAX_UPLOAD_BYTES // 2**20} MB")
        samples = parse_upload(body, Path(name).stem)  # ids as load_dataset would give the file
        model_file = reports.path(DEPLOYED_MODEL)
        if not model_file.is_file():
            raise HTTPException(503, f"no deployed risk model ({DEPLOYED_MODEL}); "
                                     "run python -m src.risk.classifier")
        deployed = json.loads(model_file.read_text(encoding="utf-8"))
        bench: list[str] = []
        reference: list[str] = []  # the sem family's reference comes with the grid's data
        if benchmark:
            from src.risk.audit import grid_benchmark, grid_reference

            try:
                bench = grid_benchmark(reports.path(GRID_CONFIGS[benchmark]))
                reference = grid_reference(reports.path(GRID_CONFIGS[benchmark]))
            except (OSError, KeyError) as e:
                raise HTTPException(503, f"the {benchmark} benchmark data is not on this server "
                                         f"({type(e).__name__})") from None
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "status": "queued", "name": name, "n_samples": len(samples),
               "benchmark": benchmark, "created_at": _now()}
        with lock:
            jobs[job_id] = job
        worker.submit(work, job_id, samples, deployed, bench, reference)
        response.headers["Location"] = f"/v1/audits/{job_id}"
        return job

    @app.get("/v1/audits/{audit_id}", response_model=AuditJob, tags=["audits"])
    def get_audit(audit_id: AuditId, response: Response) -> Any:
        """An audit's status; ``report`` and ``scores`` once it is ``done``."""
        with lock:
            job = jobs.get(audit_id)
        if job is None:
            raise HTTPException(404, f"no audit {audit_id} (jobs are kept until the server restarts)")
        response.headers["Cache-Control"] = "no-store"
        return job

    return app


def openapi_json(app: Optional[FastAPI] = None) -> str:
    """The spec as committed: stable key order, trailing newline."""
    return json.dumps((app or create_app()).openapi(), indent=2, sort_keys=True) + "\n"
