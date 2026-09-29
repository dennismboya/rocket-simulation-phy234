"""Phase 6 endpoints of the BRE API (dashboard pages 4-7): intake, editor documents,
settings, registry / activation, transparency, retrain, client export and delete.

Mounted by ``api.main`` (``app.include_router``); the Phase 5 routes and their field names are
untouched. Helpers of ``api.main`` (service state, client bookkeeping, catalog parsing) are
reached through the module object at call time, so this module never imports ``api.main`` at
import time.

Admin flag (``POST /model/activate``): a request is an admin request when the API process runs
with ``BRE_ADMIN=1`` in its environment **or** the request carries the header
``X-BRE-Admin: 1``. The header is a convenience for the local dashboard (which sends it when
its own ``BRE_ADMIN=1``); it is not authentication. Behind anything but a loopback interface,
set neither and activate models with ``python -m db.demo_seed`` / ``bre.retrain`` instead.

Real-vs-synthetic rule (CLAUDE.md rule 3) on the intake endpoints: the database layer refuses
real rows in a database under ``data/synthetic`` and synthetic rows elsewhere, and never mixes
the two; the endpoints return 409 with that reason. In demo mode the dashboard stores the rows
it collects as synthetic and says so on screen and in the audit log.

Transparency: ``GET /transparency`` shows Phase 4 (real-data) numbers only when
``reports/phase4/verdict.json`` exists; until then ``verdict_status`` is exactly
:data:`VERDICT_NOT_RUN` and ``phase4_metrics`` is null. Numbers of a synthetic-training model
are always returned (the dashboard shows them under the SYNTHETIC DATA banner).
``BRE_REPORTS_DIR`` (default ``bre/reports``) is where the verdict, the model card and the
recovery study's ``N_target.json`` are read from.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from api import schemas as Sch
from api import store
from bre import design as D
from bre import predict as P
from bre import schema as S
from bre.loaders import intake_battery as IB
from db.models import AuditLog, Client, Intervention, ModelRegistry, Response
from db.seed import validate_transform
from db.session import SyntheticPolicyError, delete_client, export_client, frame_to_responses, session_scope, sqlite_file_path, write_audit

router = APIRouter()

VERDICT_NOT_RUN = "Decision rule not yet run: no real-data numbers are shown"
"""Exact sentence returned (and shown) while ``reports/phase4/verdict.json`` is absent."""

ADMIN_RULE = "admin when the API process has BRE_ADMIN=1 in its environment or the request carries the header X-BRE-Admin: 1 (local convenience, not authentication)"

SELECTABLE_MODEL_TYPES: tuple[str, ...] = ("Q2", "Q4", "B2")

INTAKE_DATASET = IB.DATASET
INTAKE_ACTOR = "api:intake"
RESERVED_CLIENT_IDS: tuple[str, ...] = ("api:anonymous", "api:batch")

_INSTRUMENT_INDEX = S.PROJECT_ROOT / "instrument" / "static" / "index.html"


def _main():
    """The ``api.main`` module (imported by then; see the module docstring)."""
    return sys.modules["api.main"]


def _state(request: Request):
    return _main().service(request)


def is_admin(request: Request) -> bool:
    if os.environ.get("BRE_ADMIN", "").strip() == "1":
        return True
    return request.headers.get("x-bre-admin", "").strip() == "1"


def _require_admin(request: Request) -> None:
    if not is_admin(request):
        raise HTTPException(status_code=403, detail=[{"loc": ["header", "X-BRE-Admin"], "msg": f"admin only: {ADMIN_RULE}", "type": "forbidden"}])


def _422(loc: list[str], msg: str) -> HTTPException:
    return HTTPException(status_code=422, detail=[{"loc": loc, "msg": msg, "type": "value_error"}])


def _409(msg: str) -> HTTPException:
    return HTTPException(status_code=409, detail=[{"loc": ["body"], "msg": msg, "type": "conflict"}])


def reports_dir() -> Path:
    return Path(os.environ.get("BRE_REPORTS_DIR") or (S.PROJECT_ROOT / "reports"))


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------


def _settings_out(st) -> Sch.SettingsOut:
    s = dict(st.settings)
    version = int(s.pop("_version", 0))
    updated = s.pop("_updated_at", None)
    return Sch.SettingsOut(settings=s, defaults=store.DEFAULT_SETTINGS, rules=store.SETTINGS_RULES, version=version, updated_at=updated,
                           demo_mode=st.demo_mode, demo_mode_forced_by_data=st.demo_mode_from_data)


@router.get("/settings", response_model=Sch.SettingsOut, tags=["settings"])
def get_settings(request: Request) -> Any:
    st = _state(request)
    st.reload_settings()
    return _settings_out(st)


@router.put("/settings", response_model=Sch.SettingsOut, tags=["settings"])
def put_settings(patch: Sch.SettingsPatch, request: Request) -> Any:
    st = _state(request)
    body = {k: v for k, v in patch.model_dump().items() if v is not None}
    if "typical_crisis_contexts" in body:
        unknown = [t for t in body["typical_crisis_contexts"] if t not in st.artifact.ctx_vocab]
        if unknown:
            raise _422(["body", "typical_crisis_contexts"], f"context tag(s) {unknown} are not in the served model's vocabulary {list(st.artifact.ctx_vocab)}")
    try:
        with session_scope(st.engine) as s:
            out = store.put_settings(s, body, actor="api")
            write_audit(s, actor="api", action="settings", entity="bre_documents", entity_id=store.SETTINGS_KEY, payload={"patch": body, "version": out["_version"]})
    except ValueError as exc:
        raise _422(["body"], str(exc)) from exc
    st.reload_settings()
    return _settings_out(st)


# ---------------------------------------------------------------------------------------------
# Contexts, scenarios, interventions (versioned documents)
# ---------------------------------------------------------------------------------------------


def _calibration_entry(st, tag: str) -> dict[str, Any]:
    table = st.artifact.calibrated_contexts or {}
    entry = table.get(tag)
    if entry is None:
        return {"n_responses": 0, "status": P.calibration_of(st.artifact, (tag,))[tag]["status"], "theta": None, "theta_ci95": None,
                "min_responses": next((e.get("min_responses") for e in table.values() if isinstance(e, dict) and e.get("min_responses")), None),
                "note": "not in the served model's context vocabulary: no parameter, prior only"}
    return {"n_responses": int(entry.get("n_responses", 0)), "status": str(entry.get("status")), "theta": entry.get("theta"), "theta_ci95": entry.get("theta_ci95"),
            "min_responses": entry.get("min_responses")}


def _context_out(st, row: dict[str, Any]) -> Sch.ContextOut:
    tag = row["tag"]
    return Sch.ContextOut(tag=tag, kind=str(row.get("kind") or tag.split(":", 1)[0]), display=str(row.get("display") or ""), active=bool(row.get("active", True)),
                          triggers_delay=row.get("triggers_delay"), note=row.get("note"), version=int(row["version"]), updated_at=str(row["updated_at"]), actor=str(row["actor"]),
                          in_design=tag in D.NONNULL_CONTEXTS, in_served_vocabulary=tag in st.artifact.ctx_vocab, calibration=_calibration_entry(st, tag))


@router.get("/contexts", response_model=list[Sch.ContextOut], tags=["editor"])
def get_contexts(request: Request) -> Any:
    st = _state(request)
    with session_scope(st.engine) as s:
        rows = store.context_definitions(s)
    return [_context_out(st, r) for r in rows]


@router.post("/contexts", response_model=Sch.ContextOut, tags=["editor"])
def post_context(body: Sch.ContextIn, request: Request) -> Any:
    st = _state(request)
    if not S.TAG_PATTERN.fullmatch(body.tag):
        raise _422(["body", "tag"], "a context tag is namespace:value (letters, digits, _ . + -)")
    payload = body.model_dump()
    if payload.get("triggers_delay") is None:
        payload["triggers_delay"] = body.kind == "news"
    with session_scope(st.engine) as s:
        prev = store.latest(s, "context", body.tag)
        doc = store.append(s, "context", body.tag, payload, actor="api")
        write_audit(s, actor="api", action="context_version", entity="bre_documents", entity_id=body.tag,
                    payload={"version": doc.version, "active": body.active, "new": prev is None})
        row = next(r for r in store.context_definitions(s) if r["tag"] == body.tag)
    return _context_out(st, row)


def _scenarios_out(s, with_history: bool) -> Sch.ScenariosOut:
    texts = store.scenario_texts(s)
    hist = [d.to_dict() for d in store.history(s, "scenario")] if with_history else None
    return Sch.ScenariosOut(texts={k: Sch.ScenarioOut(key=k, **v) for k, v in texts.items()}, keys=list(store.SCENARIO_KEYS), history=hist,
                            design_version=D.DESIGN_VERSION,
                            note="edits create a new version; the intake battery shows the newest version and records the version numbers shown with every session; stored responses are never rewritten")


@router.get("/scenarios", response_model=Sch.ScenariosOut, tags=["editor"])
def get_scenarios(request: Request, history: bool = False) -> Any:
    st = _state(request)
    with session_scope(st.engine) as s:
        return _scenarios_out(s, history)


@router.post("/scenarios", response_model=Sch.ScenarioOut, tags=["editor"])
def post_scenario(body: Sch.ScenarioIn, request: Request) -> Any:
    st = _state(request)
    if body.key not in store.SCENARIO_KEYS:
        raise _422(["body", "key"], f"key must be one of {list(store.SCENARIO_KEYS)}")
    if not body.text.strip():
        raise _422(["body", "text"], "text must not be empty")
    with session_scope(st.engine) as s:
        doc = store.append(s, "scenario", body.key, {"text": body.text, "note": body.note}, actor="api")
        write_audit(s, actor="api", action="scenario_version", entity="bre_documents", entity_id=body.key, payload={"version": doc.version})
        row = store.scenario_texts(s)[body.key]
    return Sch.ScenarioOut(key=body.key, **row)


def _intervention_version_out(s, it: Intervention, history: bool) -> Sch.InterventionVersionOut:
    doc = store.latest(s, "intervention", it.name)
    hist = [d.to_dict() for d in store.history(s, "intervention", it.name)] if history else None
    return Sch.InterventionVersionOut(id=it.id, name=it.name, script=it.script, mapped_context_transform=json.loads(it.mapped_context_transform), active=bool(it.active),
                                      label=P.INTERVENTION_LABEL, version=0 if doc is None else int(doc.version), updated_at=None if doc is None else doc.created_at.isoformat(), history=hist)


@router.get("/interventions/versions", response_model=list[Sch.InterventionVersionOut], tags=["editor"])
def interventions_versions(request: Request, history: bool = False) -> Any:
    st = _state(request)
    with session_scope(st.engine) as s:
        rows = s.scalars(select(Intervention).order_by(Intervention.id)).all()
        return [_intervention_version_out(s, it, history) for it in rows]


@router.put("/interventions", response_model=Sch.InterventionVersionOut, tags=["editor"])
def put_intervention(body: Sch.InterventionPut, request: Request) -> Any:
    st = _state(request)
    if body.script is None and body.mapped_context_transform is None and body.active is None:
        raise _422(["body"], "nothing to change: give script, mapped_context_transform and/or active")
    if body.mapped_context_transform is not None:
        try:
            validate_transform(body.mapped_context_transform)
        except ValueError as exc:
            raise _422(["body", "mapped_context_transform"], str(exc)) from exc
    if body.script is not None and not body.script.strip():
        raise _422(["body", "script"], "script must not be empty")
    with session_scope(st.engine) as s:
        it = s.scalars(select(Intervention).where(Intervention.name == body.name)).first()
        if it is None:
            raise HTTPException(status_code=404, detail=[{"loc": ["body", "name"], "msg": f"no intervention named {body.name!r}", "type": "not_found"}])
        if body.script is not None:
            it.script = body.script
        if body.mapped_context_transform is not None:
            it.mapped_context_transform = json.dumps(body.mapped_context_transform, sort_keys=True, allow_nan=False)
        if body.active is not None:
            it.active = bool(body.active)
        s.flush()
        doc = store.append(s, "intervention", it.name, {"script": it.script, "mapped_context_transform": json.loads(it.mapped_context_transform), "active": bool(it.active), "note": body.note}, actor="api")
        write_audit(s, actor="api", action="intervention_version", entity="interventions", entity_id=str(it.id), payload={"name": it.name, "version": doc.version, "active": bool(it.active)})
        out = _intervention_version_out(s, it, False)
    return out


# ---------------------------------------------------------------------------------------------
# Intake: in-session rows, uploads, config, response history
# ---------------------------------------------------------------------------------------------


def _rows_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"rows[{i}] is not an object")
        missing = [c for c in S.COLUMNS if c not in r]
        extra = [c for c in r if c not in S.COLUMNS]
        if missing or extra:
            raise ValueError(f"rows[{i}]: missing {missing}, unknown {extra}")
    df = pd.DataFrame({c: pd.Series([r[c] for r in rows], dtype="object") for c in S.COLUMNS})
    return S.conform_frame(df)


def _covariates_of(df: pd.DataFrame) -> dict[str, Any]:
    try:
        rec = S.json_loads_strict(str(df["covariates"].iloc[0]))
    except ValueError:
        return {}
    return {k: v for k, v in rec.items() if k in S.COVARIATE_KEYS}


def _insert_intake(st, client_id: str, df: pd.DataFrame, *, display_label: str | None, assignment: dict[str, Any], session_info: dict[str, Any], source: str) -> Sch.IntakeResponsesOut:
    main = _main()
    try:
        S.validate_frame(df, strict=True)
    except S.SchemaError as exc:
        raise _422(["body", "rows"], str(exc)) from exc
    is_syn = bool(df["is_synthetic"].iloc[0])
    session_ids = sorted(df["session_id"].unique().tolist())
    if len(session_ids) != 1:
        raise _422(["body", "rows"], "rows must belong to one session_id")
    subjects = sorted(df["subject_id"].unique().tolist())
    if len(subjects) != 1:
        raise _422(["body", "rows"], "rows must belong to one subject_id")
    session_id = session_ids[0]
    consent = bool(df["consent_training"].all())
    dataset = str(df["dataset"].iloc[0])
    path = sqlite_file_path(st.engine)
    if path is not None and S.is_synthetic_path(path) and not is_syn:
        raise _409(f"real responses (is_synthetic=false) cannot be stored in the demo database under data/synthetic ({path}); in demo mode the dashboard stores its rows as synthetic and says so")
    if path is not None and not S.is_synthetic_path(path) and is_syn:
        raise _409(f"synthetic responses cannot be stored in a real database ({path})")
    try:
        with session_scope(st.engine) as s:
            cid = main._ensure_client(s, client_id, display_label, _covariates_of(df))
            n = frame_to_responses(s, df, client_id=cid, actor=INTAKE_ACTOR)
            payload = {"client_id": cid, "subject_id": subjects[0], "session_id": session_id, "form": session_info.get("form") or assignment.get("form"),
                       "seed": session_info.get("seed", session_id), "source": source, "assignment": assignment, "session": session_info,
                       "n_rows": int(n), "is_synthetic": is_syn, "consent_training": consent, "dataset": dataset, "stored_at": _now()}
            doc = store.append(s, "intake_session", session_id, payload, actor=INTAKE_ACTOR)
            write_audit(s, actor=INTAKE_ACTOR, action="intake_session", entity="responses", entity_id=cid,
                        payload={"session_id": session_id, "n_rows": int(n), "is_synthetic": is_syn, "consent_training": consent, "source": source, "form": payload["form"]})
            version = int(doc.version)
    except SyntheticPolicyError as exc:
        raise _409(str(exc)) from exc
    except IntegrityError as exc:
        raise _409(f"rows conflict with stored responses (duplicate dataset/subject/session/position): {exc.orig}") from exc
    except ValueError as exc:
        raise _422(["body", "rows"], str(exc)) from exc
    note = ("stored as SYNTHETIC rows in the demo database (demo mode); they never enter a real-data fit" if is_syn else
            ("stored as real rows with training consent" if consent else "stored as real rows WITHOUT training consent (excluded from every fit)"))
    return Sch.IntakeResponsesOut(client_id=cid, session_id=session_id, n_rows=int(n), is_synthetic=is_syn, consent_training=consent, dataset=dataset,
                                  document_version=version, note=note, demo_mode=st.demo_mode)


@router.post("/clients/{client_id}/responses", response_model=Sch.IntakeResponsesOut, tags=["intake"])
def post_client_responses(client_id: str, body: Sch.IntakeResponsesIn, request: Request) -> Any:
    st = _state(request)
    if not re.fullmatch(Sch.CLIENT_ID_PATTERN, client_id) or client_id in RESERVED_CLIENT_IDS:
        raise _422(["path", "client_id"], "invalid or reserved client_id")
    try:
        df = _rows_frame(body.rows)
    except (ValueError, S.SchemaError) as exc:
        raise _422(["body", "rows"], str(exc)) from exc
    if not (df["subject_id"] == client_id).all():
        raise _422(["body", "rows"], "subject_id must equal the client_id on every row (the client's fitted effects are keyed by it)")
    return _insert_intake(st, client_id, df, display_label=body.display_label, assignment=body.assignment, session_info=body.session, source="in-session")


@router.post("/intake/upload", response_model=Sch.IntakeResponsesOut, tags=["intake"])
async def intake_upload(
    request: Request,
    file: UploadFile = File(..., description="intake_<session_id>.json downloaded from the static instrument"),
    client_id: str = Form(..., description="the client the session belongs to"),
    display_label: str | None = Form(None),
    store_as_synthetic: bool = Form(False, description="demo mode only: store the rows as synthetic (the demo database refuses real rows)"),
) -> Any:
    st = _state(request)
    if not re.fullmatch(Sch.CLIENT_ID_PATTERN, client_id) or client_id in RESERVED_CLIENT_IDS:
        raise _422(["client_id"], "invalid or reserved client_id")
    raw = await file.read()
    try:
        rows = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _422(["file"], f"not a JSON file: {exc}") from exc
    problem = IB._contract_problem(rows)
    if problem:
        raise _422(["file"], f"file violates the battery output contract: {problem}")
    form = IB._form_of(rows)
    expected = IB.n_presentations_by_form().get(form)
    if expected is None or len(rows) != expected * IB.ROWS_PER_PRESENTATION:
        raise _422(["file"], f"incomplete session: {len(rows)} rows, form {form or '?'} needs {expected} presentations x {IB.ROWS_PER_PRESENTATION} rows (instrument/README.md, pre-specified handling)")
    original_subject = str(rows[0]["subject_id"])
    fname = file.filename or "upload.json"
    for r in rows:
        r["subject_id"] = client_id
        r["source_row_ref"] = f"{fname}:{r['source_row_ref']}"
        if store_as_synthetic:
            r["is_synthetic"] = True
    try:
        df = _rows_frame(rows)
    except (ValueError, S.SchemaError) as exc:
        raise _422(["file"], str(exc)) from exc
    session_info = {"form": form, "seed": str(rows[0]["session_id"]), "file": fname, "original_subject_id": original_subject,
                    "relabelled_synthetic": bool(store_as_synthetic), "loader": "bre.loaders.intake_battery contract and completeness checks"}
    return _insert_intake(st, client_id, df, display_label=display_label, assignment={"form": form, "source": "static instrument (assignment in the session log file)"},
                          session_info=session_info, source=f"upload:{fname}")


@router.get("/intake/config", response_model=Sch.IntakeConfig, tags=["intake"])
def intake_config(request: Request) -> Any:
    st = _state(request)
    url = os.environ.get("BRE_INSTRUMENT_URL", "").strip()
    source = "BRE_INSTRUMENT_URL"
    if not url:
        url = _INSTRUMENT_INDEX.as_uri() if _INSTRUMENT_INDEX.exists() else "instrument/static/index.html (not found)"
        source = "default: the repository's static instrument (set BRE_INSTRUMENT_URL to the deployed page)"
    intake: dict[str, Any] = {}
    try:
        battery = store.load_battery()
        forms = {k: {kk: v.get(kk) for kk in ("n_items", "n_repeats", "n_presentations", "target_duration_min")} for k, v in battery["forms"].items()}
        url_params = dict(battery["randomization"].get("url_params") or {})
        file_names = dict(battery["output"].get("file_names") or {})
        version = str(battery["battery_version"])
        intake = {k: battery["intake"].get(k) for k in ("consent", "instructions", "covariates", "covariate_null_rule", "financial_literacy_quiz")}
        intake["questions"] = battery["design"].get("questions")
        intake["delay_page"] = battery["design"].get("delay_page")
        intake["repeated_scenario_rule"] = battery["design"].get("repeated_scenario_rule")
    except (OSError, ValueError, KeyError):
        forms, url_params, file_names, version = {}, {}, {}, "unknown"
    return Sch.IntakeConfig(instrument_url=url, instrument_url_source=source, battery_version=version, design_version=D.DESIGN_VERSION, forms=forms, url_params=url_params,
                            file_names=file_names, intake=intake, demo_mode=st.demo_mode,
                            storage_note=("demo mode: the database lives under data/synthetic, so collected rows are stored as SYNTHETIC and never enter a real-data fit" if st.demo_mode
                                          else "rows are stored as real responses; only rows with training consent enter a fit"))


@router.get("/clients/{client_id}/responses", response_model=list[Sch.ResponseRowOut], tags=["intake"])
def client_responses(client_id: str, request: Request) -> Any:
    st = _state(request)
    with session_scope(st.engine) as s:
        if s.get(Client, client_id) is None:
            raise HTTPException(status_code=404, detail=[{"loc": ["path", "client_id"], "msg": f"client {client_id!r} does not exist", "type": "not_found"}])
        rows = s.scalars(select(Response).where(Response.client_id == client_id).order_by(Response.session_id, Response.position_in_session, Response.id)).all()
        out = []
        for r in rows:
            ob = r.outcome_behavior
            out.append(Sch.ResponseRowOut(id=r.id, dataset=r.dataset, session_id=r.session_id, timestamp=r.timestamp, position_in_session=r.position_in_session,
                                          scenario_id=r.scenario_id, loss_pct=r.loss_pct, context_tags=list(r.context_tags), question_order_id=r.question_order_id,
                                          elicitation_type=r.elicitation_type, response=float(r.response), response_time_ms=r.response_time_ms,
                                          outcome_behavior=None if ob is None else json.loads(ob), consent_training=bool(r.consent_training), is_synthetic=bool(r.is_synthetic),
                                          source_row_ref=r.source_row_ref))
    return out


# ---------------------------------------------------------------------------------------------
# Export and delete
# ---------------------------------------------------------------------------------------------


@router.get("/clients/{client_id}/export", response_model=Sch.ExportOut, tags=["clients"])
def export_client_data(client_id: str, request: Request) -> Any:
    st = _state(request)
    with session_scope(st.engine) as s:
        try:
            dump = export_client(s, client_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=[{"loc": ["path", "client_id"], "msg": str(exc), "type": "not_found"}]) from exc
        sessions = [d.to_dict() for d in store.history(s, "intake_session") if json.loads(d.payload).get("client_id") == client_id]
        write_audit(s, actor="api", action="export_client", entity="clients", entity_id=client_id, payload={"n_responses": len(dump["responses"])})
    synthetic = any(bool(r.get("is_synthetic")) for r in dump["responses"]) or bool(st.artifact.is_synthetic_training)
    return Sch.ExportOut(**dump, intake_sessions=sessions, exported_at=_now(), demo_mode=st.demo_mode, synthetic=synthetic)


@router.delete("/clients/{client_id}", response_model=Sch.DeleteOut, tags=["clients"])
def delete_client_data(client_id: str, request: Request) -> Any:
    st = _state(request)
    if client_id in RESERVED_CLIENT_IDS:
        raise _409(f"{client_id!r} is a reserved service client and cannot be deleted")
    with session_scope(st.engine) as s:
        try:
            n_docs = store.delete_intake_sessions(s, client_id)
            counts = delete_client(s, client_id, actor="api")
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=[{"loc": ["path", "client_id"], "msg": str(exc), "type": "not_found"}]) from exc
        counts["bre_documents"] = n_docs
        row = s.scalars(select(AuditLog).where(AuditLog.action == "delete_client", AuditLog.entity_id == client_id).order_by(AuditLog.id.desc())).first()
        audit_id = int(row.id) if row is not None else -1
    st._loss_response = None  # the served book changed
    return Sch.DeleteOut(client_id=client_id, deleted=counts, audit_log_id=audit_id)


# ---------------------------------------------------------------------------------------------
# Registry, activation, retrain
# ---------------------------------------------------------------------------------------------


def _registry_rows(st) -> list[Sch.RegistryRow]:
    with session_scope(st.engine) as s:
        rows = s.scalars(select(ModelRegistry).order_by(ModelRegistry.promoted_at.desc().nullslast(), ModelRegistry.version)).all()
        out = []
        for r in rows:
            metrics = json.loads(r.metrics or "{}")
            refs = json.loads(r.training_data_refs or "[]")
            path = P.artifact_path_of({"training_data_refs": refs, "metrics": metrics})
            full = P.resolve_artifact_dir(path) if path else None
            ho = metrics.get("held_out") if isinstance(metrics.get("held_out"), dict) else None
            train_nll = metrics.get("train_nll_per_response", metrics.get("train_nll"))
            out.append(Sch.RegistryRow(version=r.version, model_type=r.model_type, family=metrics.get("family"), is_active=bool(r.is_active),
                                       promoted_at=None if r.promoted_at is None else r.promoted_at.isoformat(), created_at=metrics.get("created_at"),
                                       is_synthetic_training=metrics.get("is_synthetic_training"), n_params=metrics.get("n_params"),
                                       train_nll=None if train_nll is None else float(train_nll), held_out_nll=None if not ho or ho.get("nll") is None else float(ho["nll"]),
                                       training_data_refs=[str(x) for x in refs], artifact_dir=path, artifact_exists=bool(full and Path(full, "artifact.json").exists())))
    return out


@router.get("/model/registry", response_model=Sch.RegistryOut, tags=["model"])
def model_registry(request: Request) -> Any:
    st = _state(request)
    return Sch.RegistryOut(active_version=st.artifact.version, rows=_registry_rows(st), selectable_model_types=list(SELECTABLE_MODEL_TYPES), admin=is_admin(request), admin_rule=ADMIN_RULE)


@router.post("/model/activate", response_model=Sch.ActivateOut, tags=["model"])
def model_activate(body: Sch.ActivateIn, request: Request) -> Any:
    st = _state(request)
    _require_admin(request)
    with session_scope(st.engine) as s:
        row = s.get(ModelRegistry, body.version)
        if row is None:
            raise HTTPException(status_code=404, detail=[{"loc": ["body", "version"], "msg": f"no registry row {body.version!r}", "type": "not_found"}])
        refs = json.loads(row.training_data_refs or "[]")
        path = P.artifact_path_of({"training_data_refs": refs, "metrics": json.loads(row.metrics or "{}")})
        if not path or not Path(P.resolve_artifact_dir(path), "artifact.json").exists():
            raise _409(f"registry row {body.version!r} names no artifact directory on disk")
        artifact = P.load_artifact(path)
        if artifact.version != body.version:
            raise _409(f"artifact at {path} is version {artifact.version!r}, not {body.version!r}")
        previous = st.artifact.version
        from db.models import utcnow

        s.execute(update(ModelRegistry).values(is_active=False))
        row.is_active = True
        row.promoted_at = utcnow()
        s.flush()
        write_audit(s, actor="api:admin", action="activate_model", entity="model_registry", entity_id=body.version, payload={"previous": previous, "model_type": artifact.model_name})
    st.swap_artifact(artifact)
    return Sch.ActivateOut(activated=artifact.version, previous=previous, model_type=artifact.model_name, demo_mode=st.demo_mode, synthetic=bool(artifact.is_synthetic_training))


@router.post("/retrain", response_model=Sch.RetrainOut, tags=["model"])
def retrain_endpoint(body: Sch.RetrainIn, request: Request) -> Any:
    from bre import retrain as R

    st = _state(request)
    st.reload_settings()
    budget = dict(st.settings.get("retrain") or store.DEFAULT_SETTINGS["retrain"])
    mt = body.model_type or st.artifact.model_name
    if mt not in SELECTABLE_MODEL_TYPES:
        raise _422(["body", "model_type"], f"model_type must be one of {list(SELECTABLE_MODEL_TYPES)}")
    models_root = os.environ.get("BRE_MODELS_ROOT") or None
    try:
        result = R.retrain(
            st.engine, model_type=mt, steps=int(body.steps or budget["steps"]), restarts=int(body.restarts or budget["restarts"]),
            n_samples=int(budget["n_samples"] if body.n_samples is None else body.n_samples), val_frac=float(budget["val_frac"]),
            seed=int(budget["seed"] if body.seed is None else body.seed), prior_strength=float(st.settings.get("prior_strength", 1.0)),
            n_boot=int(R.DEFAULT_N_BOOT if body.n_boot is None else body.n_boot), models_root=models_root, actor="api:retrain",
        )
    except Exception as exc:  # noqa: BLE001 - report the failure as a service error with its message
        raise HTTPException(status_code=500, detail=[{"loc": ["retrain"], "msg": f"{type(exc).__name__}: {exc}", "type": "retrain_error"}]) from exc
    if result.get("promoted"):
        st.swap_artifact(P.load_active_artifact(st.engine))
    ho = result.get("held_out") or {}
    return Sch.RetrainOut(run_id=result["run_id"], status=result["status"], promoted=bool(result.get("promoted")), reason=result.get("reason"), version=result.get("version"),
                          model_type=result["model_type"], nll_new=ho.get("nll_new"), nll_reference=ho.get("nll_reference"), recorded_reference_nll=ho.get("recorded_reference_nll"),
                          active_version=st.artifact.version, demo_mode=st.demo_mode, result=P.jsonable({k: v for k, v in result.items() if k != "calibrated_contexts"}))


# ---------------------------------------------------------------------------------------------
# Transparency
# ---------------------------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _license_of(entry_id: str, catalog_path: Path) -> str:
    """The ``License / terms`` line of a catalog entry, else the first line mentioning a licence."""
    try:
        text = catalog_path.read_text(encoding="utf-8")
    except OSError:
        return "see data/CATALOG.md"
    section: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("### "):
            if inside:
                break
            inside = line.startswith(f"### {entry_id}. ")
            continue
        if inside:
            section.append(line)
    for line in section:
        s = line.strip().lstrip("*").strip()
        if s.lower().startswith("license / terms:"):
            return s.split(":", 1)[1].strip()
    for line in section:
        if re.search(r"licen[cs]e|Apache|ODC-PDDL|CC-BY|MIT", line, re.IGNORECASE):
            return line.strip().lstrip("*").strip()[:300]
    return "see data/CATALOG.md"


def provenance_rows(st) -> list[dict[str, Any]]:
    """One row per processed dataset (``data/processed/README.md`` table joined with the catalog
    entry names and licence lines) plus the synthetic tables behind the served model."""
    main = _main()
    entries = {e["id"]: e for e in main.catalog_entries()}
    catalog_path = S.DATA_DIR / "CATALOG.md"
    rows: list[dict[str, Any]] = []
    readme = S.DATA_DIR / "processed" / "README.md"
    if readme.exists():
        for line in readme.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| ") or line.startswith("| dataset") or line.startswith("|---"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 6:
                continue
            name, cat, fname, n_rows, n_subj = cells[0], cells[1], cells[2], cells[3], cells[4]
            try:
                n_rows_i, n_subj_i = int(n_rows.replace(",", "")), int(n_subj.replace(",", ""))
            except ValueError:
                n_rows_i, n_subj_i = None, None
            entry = entries.get(cat)
            rows.append({"dataset": name, "catalog_entry": cat, "catalog_name": entry["name"] if entry else ("intake battery (instrument/)" if cat == "instrument" else cat),
                         "catalog_status": entry["status"] if entry else None, "file": fname, "n_rows": n_rows_i, "n_subjects": n_subj_i, "real_or_synthetic": "real",
                         "license": _license_of(cat, catalog_path) if entry else ("volunteer sessions; consent per row (consent_training)" if cat == "instrument" else "see data/CATALOG.md"),
                         "used_by_served_model": False})
    used = {e["id"] for e in main.catalog_entries_used(list(entries.values()), list(st.artifact.training_data_refs))}
    for r in rows:
        r["used_by_served_model"] = r["catalog_entry"] in used
    with session_scope(st.engine) as s:
        counts = s.execute(select(Response.dataset, Response.is_synthetic, Response.subject_id).distinct()).all()
    by_ds: dict[str, dict[str, Any]] = {}
    for ds, syn, subj in counts:
        d = by_ds.setdefault(ds, {"synthetic": bool(syn), "subjects": set()})
        d["subjects"].add(subj)
    with session_scope(st.engine) as s:
        n_rows_by_ds = dict(s.execute(select(Response.dataset, __import__("sqlalchemy").func.count()).group_by(Response.dataset)).all())
    for ds, d in sorted(by_ds.items()):
        rows.append({"dataset": ds, "catalog_entry": "db", "catalog_name": f"database {st.db_label}", "catalog_status": "stored responses", "file": st.db_label,
                     "n_rows": int(n_rows_by_ds.get(ds, 0)), "n_subjects": len(d["subjects"]), "real_or_synthetic": "synthetic" if d["synthetic"] else "real",
                     "license": "generated (bre.demo / bre.sim), no licence applies" if d["synthetic"] else "volunteer sessions; consent per row",
                     "used_by_served_model": any(ds in str(r) for r in st.artifact.training_data_refs)})
    return rows


def calibration_plot(st) -> dict[str, Any] | None:
    """Reliability table of the served model on its stored training rows (in-sample), cached per
    artifact version. Synthetic in demo mode; withheld for a real-data model until the verdict exists."""
    from bre.artifact import predict_rows
    from bre.eval import brier_score, ece, reliability_table
    from bre.retrain import consented_responses

    cache = getattr(st, "_calibration_plot", None)
    if cache is not None and cache[0] == st.artifact.version:
        return cache[1]
    df = consented_responses(st.engine)
    if len(df) == 0:
        out = None
    else:
        known = np.asarray([all(t in st.artifact.ctx_vocab for t in tags) for tags in df["context_tags"].tolist()], dtype=bool)
        sell = (df["elicitation_type"] == "binary_sell").to_numpy()
        df2 = df.loc[known].reset_index(drop=True)
        if len(df2) == 0:
            out = None
        else:
            p = predict_rows(st.artifact, df2)
            m = (df2["elicitation_type"] == "binary_sell").to_numpy()
            y = df2["response"].to_numpy(dtype=float)[m]
            pp = p[m]
            out = {"reliability": reliability_table(pp, y).to_dict(orient="records"), "ece": float(ece(pp, y)), "brier": float(brier_score(pp, y)),
                   "n_rows": int(m.sum()), "n_rows_unknown_context": int((~known & sell).sum()), "is_synthetic": bool(df2["is_synthetic"].any()),
                   "scope": "in-sample: the served model on its stored training rows (fitted per-subject effects); not a held-out evaluation",
                   "model_version": st.artifact.version}
    st._calibration_plot = (st.artifact.version, out)
    return out


@router.get("/transparency", response_model=Sch.TransparencyOut, tags=["model"])
def transparency(request: Request) -> Any:
    from bre.retrain import PROMOTION_RULE

    main = _main()
    st = _state(request)
    a = st.artifact
    rdir = reports_dir()
    verdict = _read_json(rdir / "phase4" / "verdict.json")
    if verdict is not None and not isinstance(verdict, dict):
        verdict = {"verdict": str(verdict)}
    phase4 = None
    if verdict is not None:
        phase4 = verdict.get("metrics") if isinstance(verdict.get("metrics"), dict) else _read_json(rdir / "phase4" / "metrics.json")
    verdict_status = VERDICT_NOT_RUN if verdict is None else str(verdict.get("verdict") or verdict.get("status") or "verdict.json holds no verdict sentence")
    real_numbers = verdict is not None and not bool(verdict.get("is_synthetic", False))
    withhold = (not a.is_synthetic_training) and verdict is None
    metrics = P.jsonable(a.metrics or {})
    if withhold:
        training = {"withheld": True, "note": VERDICT_NOT_RUN, "n_responses": metrics.get("n_responses"), "n_subjects": metrics.get("n_subjects")}
        cal = None
    else:
        training = {k: metrics.get(k) for k in ("train_nll_per_response", "train_nll", "train_brier", "val_nll", "n_responses", "n_subjects", "n_responses_no_context", "held_out", "held_out_note", "steps", "restarts") if k in metrics}
        training["is_synthetic"] = bool(a.is_synthetic_training)
        cal = calibration_plot(st)
    n_target = None
    for cand, label in ((rdir / "recovery" / "N_target.json", "reports/recovery/N_target.json (current recovery study)"),
                        (rdir / "recovery_pass1_olddefaults" / "N_target.json", "reports/recovery_pass1_olddefaults/N_target.json (archived first pass, old G_Q defaults; the current grid is still running)")):
        d = _read_json(cand)
        if d is not None:
            n_target = {**d, "source": label}
            break
    card_path = rdir / "MODEL_CARD.md"
    card = card_path.read_text(encoding="utf-8") if card_path.exists() else None
    return Sch.TransparencyOut(
        model_version=a.version, model_type=a.model_name, family=a.family, n_params=int(a.n_params), n_params_population=metrics.get("n_params_population"),
        is_synthetic_training=bool(a.is_synthetic_training), demo_mode=st.demo_mode, decision_rule=main.DECISION_RULE, verdict=verdict, verdict_status=verdict_status,
        phase4_metrics=phase4, real_data_numbers_shown=real_numbers, training_metrics=training, calibration_plot=cal, n_target=n_target,
        provenance=provenance_rows(st), model_card=card, model_card_path=str(card_path) if card else None, retrain_rule=PROMOTION_RULE,
    )


__all__ = ["ADMIN_RULE", "INTAKE_DATASET", "RESERVED_CLIENT_IDS", "SELECTABLE_MODEL_TYPES", "VERDICT_NOT_RUN", "calibration_plot", "is_admin", "provenance_rows", "reports_dir", "router"]
