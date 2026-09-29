"""BRE scoring service (Phase 5): the only source of predictions for the dashboard.

Run with ``make api`` (``uvicorn api.main:app --port $BRE_API_PORT``). Every number a screen
shows comes from ``bre.predict`` through these endpoints; no model code lives in the UI.

Endpoints
---------
* ``GET /health`` — liveness, served model version, demo mode.
* ``GET /model`` — version, type, metrics, calibrated contexts, provenance (training data refs
  and the ``data/CATALOG.md`` entry names), ``is_synthetic_training``, ``demo_mode``, the
  pre-registered decision rule and the evaluation status.
* ``GET /clients``, ``GET /interventions``, ``GET /market`` — the served book, the intervention
  table, the current market state (cached files unless ``BRE_MARKET_NETWORK=1``) with its
  mapping to a scenario.
* ``POST /predict`` — one investor, one scenario: predicted probability of selling with its
  80/95% intervals, interference terms, top driver and calibration counts.
* ``POST /profile`` — decision-state profile (baseline state, context sensitivities,
  consistency score, classical-equivalent risk score with its R² against the state angle).
* ``POST /score_book`` (JSON) and ``POST /score_book/csv`` (CSV upload) — the triage table of a
  book under one market state.
* ``POST /interventions/rank`` — interventions ranked by predicted change in P(sell), labelled
  "predicted effect, not causally validated".

Every prediction shown is written to ``predictions_log`` (under the request's ``client_id``,
created on first use, or the anonymous API client / the batch client for uploads) and
``audit_log``. Input validation errors return 422 with the field names. Every response derived
from a synthetic-training model carries ``synthetic: true``; ``demo_mode`` is true when the
served model was trained on synthetic data or the database lives under ``data/synthetic``.

Environment: ``BRE_DB_URL`` (default ``sqlite:///<bre>/data/synthetic/demo.db``),
``BRE_MODELS_ROOT`` (default ``runs/models``), ``BRE_API_AUTOSEED`` (default 1: seed the demo
book and fit the demo model when the database has no active model), ``BRE_API_WARMUP``
(default 1), ``BRE_MARKET_NETWORK`` (default 0).
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from api import schemas as Sch
from bre import __version__ as bre_version
from bre import predict as P
from bre import schema as S
from bre.market import latest_market_state
from db.models import Client, Intervention, PredictionLog, Response
from db.session import get_engine, init_db, session_scope, sqlite_file_path, write_audit

ANON_CLIENT_ID = "api:anonymous"
BATCH_CLIENT_ID = "api:batch"
"""Reserved client rows: predictions without a client, and batch rows for unknown clients."""

ACTOR = "api"

DECISION_RULE = (
    '"Q-model supported" only if, on real data, the best Q-model beats the best classical baseline '
    "on split (b) by held-out NLL with a bootstrap 95% CI excluding zero, at matched or lower "
    "parameter count, and the interference-term CI excludes zero. Otherwise the conclusion is "
    '"no evidence of a quantum-probability advantage in the available data", and the report names '
    "the data that would resolve it. The dashboard serves whichever model wins on held-out NLL; if "
    "that is a classical model, the screen says so."
)
"""PLAN.md section 6, verbatim (the owner's wording)."""

CALIBRATION_RULE = "a context is calibrated when at least db.demo_seed.CALIBRATION_MIN_RESPONSES training sell rows show it; otherwise its parameter rests on the prior only"

_CATALOG_HEADER = re.compile(r"^### ([A-Z])\. (.+?)(?: — (.+))?$")


def catalog_entries(path: Path | None = None) -> list[dict[str, str]]:
    """Entry letters, names and statuses of ``data/CATALOG.md`` (``### X. Name — STATUS``)."""
    path = path or (S.DATA_DIR / "CATALOG.md")
    out: list[dict[str, str]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _CATALOG_HEADER.match(line.strip())
        if m:
            out.append({"id": m.group(1), "name": m.group(2).strip(), "status": (m.group(3) or "").strip()})
    return out


# ---------------------------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------------------------


def default_db_url() -> str:
    from db.demo_seed import DEMO_DB_PATH

    return os.environ.get("BRE_DB_URL") or f"sqlite:///{DEMO_DB_PATH}"


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


class ServiceState:
    """Engine, artifact and registry row of the served model."""

    def __init__(self) -> None:
        self.engine: Engine | None = None
        self.artifact: Any = None
        self.registry_row: dict[str, Any] | None = None
        self.started_at = time.time()
        self.warmup: dict[str, float] = {}
        self.seed_summary: dict[str, Any] | None = None

    @property
    def demo_mode(self) -> bool:
        if self.artifact is not None and bool(self.artifact.is_synthetic_training):
            return True
        if self.engine is not None:
            p = sqlite_file_path(self.engine)
            return p is not None and S.is_synthetic_path(p)
        return True

    @property
    def db_label(self) -> str:
        if self.engine is None:
            return "not connected"
        p = sqlite_file_path(self.engine)
        return str(p) if p is not None else str(self.engine.url)


def build_state(db_url: str | None = None, *, autoseed: bool | None = None, warm: bool | None = None) -> ServiceState:
    """Connect, make sure an active model exists (seeding the demo when allowed), load it and
    warm the forward passes."""
    from db.demo_seed import seed_demo

    state = ServiceState()
    state.engine = get_engine(db_url or default_db_url())
    init_db(state.engine)
    with session_scope(state.engine) as s:
        for cid, label in ((ANON_CLIENT_ID, "Anonymous API requests"), (BATCH_CLIENT_ID, "Batch scoring rows of unregistered clients")):
            if s.get(Client, cid) is None:
                s.add(Client(client_id=cid, display_label=label, covariates="{}"))
    row = P.active_registry_row(state.engine)
    autoseed = _env_flag("BRE_API_AUTOSEED") if autoseed is None else autoseed
    if row is None and autoseed:
        models_root = os.environ.get("BRE_MODELS_ROOT")
        state.seed_summary = seed_demo(state.engine, models_root=models_root)
        row = P.active_registry_row(state.engine)
    if row is None:
        raise RuntimeError("model_registry has no active model and BRE_API_AUTOSEED is off; run `python -m db.demo_seed`")
    state.registry_row = row
    state.artifact = P.load_active_artifact(state.engine)
    warm = _env_flag("BRE_API_WARMUP") if warm is None else warm
    if warm:
        state.warmup = P.warm_up(state.artifact)
    return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = build_state()
    yield


app = FastAPI(
    title="Behavioral Risk Engine API",
    version=bre_version,
    description=__doc__,
    lifespan=lifespan,
)


def service(request: Request) -> ServiceState:
    st = getattr(request.app.state, "service", None)
    if st is None:
        raise HTTPException(status_code=503, detail="service not initialized")
    return st


@app.exception_handler(ValueError)
async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
    """Facade-level validation (e.g. an unknown context tag for the served vocabulary) as a 422
    with the same shape FastAPI uses."""
    return JSONResponse(status_code=422, content={"detail": [{"loc": ["body"], "msg": str(exc), "type": "value_error"}]})


def _served(st: ServiceState) -> dict[str, Any]:
    a = st.artifact
    return {"model_version": a.version, "model_type": a.model_name, "family": a.family, "synthetic": bool(a.is_synthetic_training),
            "demo_mode": st.demo_mode, "wording": P.PROBABILITY_WORDING}


# ---------------------------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------------------------


def _ensure_client(session, client_id: str | None, display_label: str | None, covariates: dict[str, Any] | None) -> str:
    """The client row a prediction is logged under (created when unknown)."""
    cid = client_id or ANON_CLIENT_ID
    if session.get(Client, cid) is None:
        session.add(Client(client_id=cid, display_label=display_label, covariates=P.covariates_text(covariates or {})))
        session.flush()
        write_audit(session, actor=ACTOR, action="create_client", entity="clients", entity_id=cid, payload={"display_label": display_label})
    return cid


def _log(session, st: ServiceState, client_id: str, kind: str, inputs: dict[str, Any], outputs: dict[str, Any], market_state: dict[str, Any] | None = None) -> int:
    row = PredictionLog(
        client_id=client_id,
        model_version=st.artifact.version,
        market_state=json.dumps(P.jsonable(market_state or {}), allow_nan=False),
        outputs=json.dumps(P.jsonable({"kind": kind, "inputs": inputs, "outputs": outputs, "synthetic": bool(st.artifact.is_synthetic_training)}), allow_nan=False),
    )
    session.add(row)
    session.flush()
    write_audit(session, actor=ACTOR, action=f"predict:{kind}", entity="predictions_log", entity_id=str(row.id),
                payload={"client_id": client_id, "model_version": st.artifact.version, "synthetic": bool(st.artifact.is_synthetic_training)})
    return int(row.id)


def _strip_outputs(d: dict[str, Any]) -> dict[str, Any]:
    """Logged outputs: everything except the long free-text scripts."""
    return {k: v for k, v in d.items() if k not in ("ranked",)}


# ---------------------------------------------------------------------------------------------
# Endpoints: service
# ---------------------------------------------------------------------------------------------


@app.get("/health", response_model=Sch.Health)
def health(request: Request) -> Any:
    st = getattr(request.app.state, "service", None)
    if st is None:
        return Sch.Health(status="starting", model_version=None, model_type=None, demo_mode=True, synthetic=True, db="-", uptime_s=0.0)
    return Sch.Health(status="ok", model_version=st.artifact.version, model_type=st.artifact.model_name, demo_mode=st.demo_mode,
                      synthetic=bool(st.artifact.is_synthetic_training), db=st.db_label, uptime_s=time.time() - st.started_at)


@app.get("/model", response_model=Sch.ModelInfo)
def model_info(request: Request) -> Any:
    st = service(request)
    a = st.artifact
    row = st.registry_row or {}
    entries = catalog_entries()
    refs = list(a.training_data_refs)
    used = [e for e in entries if any(e["name"].split(" ")[0].lower() in r.lower() for r in refs)]
    metrics = P.jsonable(a.metrics)
    return Sch.ModelInfo(
        **_served(st), version=a.version, n_params=int(a.n_params), n_params_population=metrics.get("n_params_population"),
        design_version=a.design_version, created_at=a.created_at, promoted_at=row.get("promoted_at"), metrics=metrics,
        calibrated_contexts=P.jsonable(a.calibrated_contexts), calibration_rule=CALIBRATION_RULE, training_data_refs=refs,
        provenance={
            "training_data_refs": refs,
            "catalog_entries": entries,
            "catalog_entries_used_in_training": used,
            "note": ("the served model was fitted on the synthetic demo book only; no catalogued dataset entered its training"
                     if a.is_synthetic_training else "see training_data_refs for the catalogued datasets behind this fit"),
            "registry_row": {k: row.get(k) for k in ("version", "model_type", "promoted_at", "is_active")},
            "seed_summary": P.jsonable(st.seed_summary) if st.seed_summary else None,
        },
        is_synthetic_training=bool(a.is_synthetic_training), interval_source=P.INTERVAL_SOURCE, decision_rule=DECISION_RULE,
        evaluation_status=("not evaluated on real data: demo fit on synthetic responses (Phase 4 pending); the decision rule has not been applied"
                           if a.is_synthetic_training else str(metrics.get("verdict", "see metrics"))),
        notes=str(a.notes), n_draws=int(a.n_samples),
    )


@app.get("/clients", response_model=list[Sch.ClientOut])
def list_clients(request: Request) -> Any:
    st = service(request)
    known = set(st.artifact.subject_ids or ())
    with session_scope(st.engine) as s:
        counts = dict(s.execute(select(Response.client_id, func.count()).group_by(Response.client_id)).all())
        syn = dict(s.execute(select(Response.client_id, func.max(Response.is_synthetic)).group_by(Response.client_id)).all())
        rows = s.scalars(select(Client).order_by(Client.client_id)).all()
        out = []
        for c in rows:
            if c.client_id in (ANON_CLIENT_ID, BATCH_CLIENT_ID):
                continue
            out.append(Sch.ClientOut(client_id=c.client_id, display_label=c.display_label, covariates=json.loads(c.covariates or "{}"),
                                     created_at=c.created_at.isoformat(), n_responses=int(counts.get(c.client_id, 0)),
                                     known_to_model=c.client_id in known, synthetic=bool(syn.get(c.client_id, st.demo_mode))))
    return out


def _interventions(st: ServiceState) -> list[dict[str, Any]]:
    with session_scope(st.engine) as s:
        rows = s.scalars(select(Intervention).order_by(Intervention.id)).all()
        return [{"id": r.id, "name": r.name, "script": r.script, "mapped_context_transform": json.loads(r.mapped_context_transform), "active": bool(r.active)} for r in rows]


@app.get("/interventions", response_model=list[Sch.InterventionOut])
def list_interventions(request: Request) -> Any:
    st = service(request)
    return [Sch.InterventionOut(**it, label=P.INTERVENTION_LABEL) for it in _interventions(st)]


@app.get("/market", response_model=Sch.MarketResponse)
def market(request: Request) -> Any:
    st = service(request)
    state = latest_market_state(allow_network=_env_flag("BRE_MARKET_NETWORK", "0"))
    scen = P.market_state_to_contexts({k: state.get(k) for k in ("drawdown_pct", "duration_days", "recovery_pct", "vix_bucket")}, st.artifact.calibrated_contexts, st.artifact.n_ctx_positions)
    return Sch.MarketResponse(market_state=P.jsonable(state), scenario=scen, demo_mode=st.demo_mode)


# ---------------------------------------------------------------------------------------------
# Endpoints: predictions
# ---------------------------------------------------------------------------------------------


def _effects(st: ServiceState, subject_id: str, covariates: dict[str, Any], prior: list[Sch.PriorResponse] | None) -> tuple[P.SubjectEffects | None, dict[str, Any] | None]:
    if not prior:
        return None, None
    refined = P.refine_subject(st.artifact, subject_id, covariates, [p.model_dump() for p in prior])
    if "u" not in refined:
        return None, refined
    return {subject_id: {k: refined[k] for k in ("u", "log_gamma") if k in refined}}, refined


@app.post("/predict", response_model=Sch.PredictResponse)
def predict(req: Sch.PredictRequest, request: Request) -> Any:
    st = service(request)
    cov = req.covariates.record()
    subject = req.client_id or ANON_CLIENT_ID
    effects, refined = _effects(st, subject, cov, req.prior_responses)
    out = P.predict_sell(st.artifact, cov, req.loss_pct, req.context_tags, req.question_order_id, req.tol_answer,
                         subject_id=subject, mix_weights=req.mix_weights, subject_effects=effects)
    with session_scope(st.engine) as s:
        cid = _ensure_client(s, req.client_id, req.display_label, cov)
        log_id = _log(s, st, cid, "predict", req.model_dump(), out)
    return Sch.PredictResponse(**{**out, **_served(st)}, client_id=subject, refined=refined, prediction_log_id=log_id)


@app.post("/profile", response_model=Sch.ProfileResponse)
def profile(req: Sch.ProfileRequest, request: Request) -> Any:
    st = service(request)
    cov = req.covariates.record()
    subject = req.client_id or ANON_CLIENT_ID
    book = None
    if req.include_book_r2:
        with session_scope(st.engine) as s:
            rows = s.scalars(select(Client).order_by(Client.client_id)).all()
            book = pd.DataFrame([{"client_id": c.client_id, "covariates": c.covariates} for c in rows if c.client_id not in (ANON_CLIENT_ID, BATCH_CLIENT_ID)])
            if len(book) < 3:
                book = None
    out = P.profile(st.artifact, cov, [p.model_dump() for p in req.prior_responses] if req.prior_responses else None, subject_id=subject, book=book)
    with session_scope(st.engine) as s:
        cid = _ensure_client(s, req.client_id, req.display_label, cov)
        log_id = _log(s, st, cid, "profile", req.model_dump(), out)
    return Sch.ProfileResponse(**{**out, **_served(st)}, client_id=subject, prediction_log_id=log_id)


def _score(st: ServiceState, clients: list[Sch.ClientIn], market_state: Sch.MarketState, target: float, register_unknown: bool) -> Sch.ScoreBookResponse:
    t0 = time.time()
    frame = pd.DataFrame([{"client_id": c.client_id, "display_label": c.display_label, "covariates": c.covariates.record()} for c in clients])
    table = P.score_book(st.artifact, frame, market_state.model_dump(), _interventions(st), target=target)
    seconds = time.time() - t0
    rows = table.to_dict(orient="records")
    ms = market_state.model_dump()
    with session_scope(st.engine) as s:
        existing = set(s.scalars(select(Client.client_id)).all())
        for c, row in zip(clients, rows):
            if c.client_id in existing:
                cid = c.client_id
            elif register_unknown:
                cid = _ensure_client(s, c.client_id, c.display_label, c.covariates.record())
                existing.add(cid)
            else:
                cid = BATCH_CLIENT_ID
            row["prediction_log_id"] = _log(s, st, cid, "score_book", {"client_id": c.client_id, "covariates": c.covariates.record(), "target": target}, row, ms)
    attrs = table.attrs
    return Sch.ScoreBookResponse(
        **_served(st), rows=[Sch.ScoreRow(**r) for r in rows], scenario=attrs["scenario"], n_calibration=attrs["n_calibration"],
        meta={"target": target, "interval": attrs["interval"], "intervention_label": attrs["intervention_label"], "capacity_definition": attrs["capacity_definition"],
              "n_clients": len(rows), "seconds": seconds, "status_thresholds": P.STATUS_THRESHOLDS},
    )


@app.post("/score_book", response_model=Sch.ScoreBookResponse)
def score_book(req: Sch.ScoreBookRequest, request: Request) -> Any:
    st = service(request)
    return _score(st, req.clients, req.market_state, req.target, req.register_unknown)


CSV_COVARIATE_COLUMNS = tuple(S.COVARIATE_KEYS)


@app.post("/score_book/csv", response_model=Sch.ScoreBookResponse)
async def score_book_csv(
    request: Request,
    file: UploadFile = File(..., description="CSV with client_id, optional display_label and the six covariate columns"),
    market_state: str = Form("{}", description="JSON object with the market-state fields"),
    target: float = Form(0.25),
    register_unknown: bool = Form(False),
) -> Any:
    st = service(request)
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=[{"loc": ["file"], "msg": f"unreadable CSV: {exc}", "type": "value_error"}]) from exc
    if "client_id" not in df.columns:
        raise HTTPException(status_code=422, detail=[{"loc": ["file", "client_id"], "msg": "CSV needs a client_id column", "type": "missing"}])
    clients: list[Sch.ClientIn] = []
    errors: list[dict[str, Any]] = []
    for i, rec in enumerate(df.to_dict(orient="records")):
        cov: dict[str, Any] = {}
        for k in CSV_COVARIATE_COLUMNS:
            v = str(rec.get(k, "")).strip()
            if v == "":
                continue
            cov[k] = int(float(v)) if k in ("invest_experience_yrs", "self_reported_risk_tolerance", "financial_literacy_score") and re.fullmatch(r"-?\d+(\.0+)?", v) else v
        try:
            clients.append(Sch.ClientIn(client_id=str(rec["client_id"]).strip(), display_label=(str(rec.get("display_label", "")).strip() or None), covariates=cov))
        except ValidationError as exc:
            for e in exc.errors():
                errors.append({"loc": ["file", f"row {i + 2}", *[str(x) for x in e["loc"]]], "msg": e["msg"], "type": e["type"]})
    try:
        ms = Sch.MarketState(**json.loads(market_state or "{}"))
    except (ValidationError, ValueError) as exc:
        errs = exc.errors() if isinstance(exc, ValidationError) else [{"loc": [], "msg": str(exc), "type": "value_error"}]
        errors.extend({"loc": ["market_state", *[str(x) for x in e["loc"]]], "msg": e["msg"], "type": e["type"]} for e in errs)
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    try:
        req = Sch.ScoreBookRequest(clients=clients, market_state=ms, target=target, register_unknown=register_unknown)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=[{"loc": ["file", *[str(x) for x in e["loc"]]], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]) from exc
    return _score(st, req.clients, req.market_state, req.target, req.register_unknown)


@app.post("/interventions/rank", response_model=Sch.RankResponse)
def rank(req: Sch.RankRequest, request: Request) -> Any:
    st = service(request)
    cov = req.covariates.record()
    subject = req.client_id or ANON_CLIENT_ID
    effects, _ = _effects(st, subject, cov, req.prior_responses)
    out = P.rank_interventions(st.artifact, cov, req.market_state.model_dump(), _interventions(st), subject_id=subject, subject_effects=effects)
    with session_scope(st.engine) as s:
        cid = _ensure_client(s, req.client_id, req.display_label, cov)
        log_id = _log(s, st, cid, "interventions_rank", req.model_dump(), {**_strip_outputs(out), "ranked": [{k: r[k] for k in ("rank", "name", "delta_p", "delta_ci80", "label")} for r in out["ranked"]]}, req.market_state.model_dump())
    return Sch.RankResponse(**{**out, **_served(st)}, client_id=subject, prediction_log_id=log_id)


@app.exception_handler(RequestValidationError)
async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Default 422 shape, kept explicit so the contract is visible in this file."""
    return JSONResponse(status_code=422, content={"detail": P.jsonable(exc.errors())})


__all__ = ["ANON_CLIENT_ID", "BATCH_CLIENT_ID", "DECISION_RULE", "ServiceState", "app", "build_state", "catalog_entries", "default_db_url"]
