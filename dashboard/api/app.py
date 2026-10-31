"""The v1 routes, response models and error format. ``create_app(root)`` serves ``root/reports``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.finetune.grid import CONTAMINATION_TYPES

REPO = Path(__file__).resolve().parents[2]
API_VERSION = "1.0.0"

Dataset = Literal["wikitext2", "tinystories"]
ContaminationType = Literal[CONTAMINATION_TYPES]  # type: ignore[valid-type]
GRID_DIRS: dict[str, str] = {"wikitext2": "reports/grid", "tinystories": "reports/grid_tinystories"}
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
    out.append(gate("gate_1", "pending"))
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


# ---------------------------------------------------------------- the app

def create_app(root: Path = REPO) -> FastAPI:
    reports = Reports(root)
    problem = {"model": Problem, "content": {"application/problem+json": {}}}
    app = FastAPI(
        title="ORIGIN-T API", version=API_VERSION,
        description="Read-only access to ORIGIN-T's reports (HANDBOOK §7.7). Errors are RFC 7807 "
                    "problem+json. Responses carry a weak ETag; send it as If-None-Match for a 304.",
        responses={404: problem, 422: problem},
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET"], allow_headers=["If-None-Match"], expose_headers=["ETag"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        titles = {404: "Not found", 405: "Method not allowed"}
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
        files = ["reports/collapse_spike/metrics.json",
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

    @app.get("/v1/repair/{dataset}", response_model=list[RepairResult], tags=["repair"])
    def repair(dataset: Dataset, request: Request, response: Response) -> Any:
        rel = f"reports/repair/{dataset}.jsonl"
        return _cached(request, response, reports.etag(rel)) or reports.jsonl(rel)

    @app.get("/v1/runs/{run_id}/trainlog", response_model=list[TrainlogRow], tags=["runs"])
    def trainlog(run_id: RunId, request: Request, response: Response) -> Any:
        rel = f"reports/trainlogs/{run_id}.jsonl"
        return _cached(request, response, reports.etag(rel)) or [
            {**r, "signals": _with_prefix(r, "sig_"),
             "alarm_signal": r["alarm_signal"] and r["alarm_signal"].removeprefix("sig_")}
            for r in reports.jsonl(rel)]

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

    return app


def openapi_json(app: Optional[FastAPI] = None) -> str:
    """The spec as committed: stable key order, trailing newline."""
    return json.dumps((app or create_app()).openapi(), indent=2, sort_keys=True) + "\n"
