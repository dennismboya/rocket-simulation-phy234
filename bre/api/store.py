"""Versioned configuration documents of the dashboard (Phase 6, pages 5-7).

One table, ``bre_documents``, holds every editable, versioned document the advisor pages
maintain: scenario texts, context definitions, intervention versions, the settings document
and the record of every in-session intake battery (its randomized assignment). The table is
**append-only**: an edit inserts a new row with ``version = previous + 1`` and never rewrites an
earlier row, so the history of every text is kept and a response row that was collected under
version ``k`` still traces to that wording (the intake-session document names the versions that
were shown). Nothing in the DecisionEvent tables is touched by any edit.

The table lives on its own declarative base (:class:`ConfigBase`) and is created by
:func:`ensure_tables` (``create_all``, migration-free) when the API starts, so the core
schema in ``db.models`` (``Base``) and its table set stay exactly as before.

Document kinds (``kind`` column) and their keys:

* ``settings`` / ``settings``: the dashboard settings (:data:`DEFAULT_SETTINGS` documents the
  defaults; a stored document holds only the keys that were set, merged over the defaults on
  read).
* ``context`` / ``<tag>``: a context definition ``{tag, kind, display, active, note}``; seeded
  from ``instrument/battery.json`` (``design.contexts``) on first start.
* ``scenario`` / ``<key>``: an editable scenario text ``{text, note}``; keys and seed values
  come from ``instrument/battery.json`` (:func:`seed_scenarios`).
* ``intervention`` / ``<name>``: a version of an intervention ``{script, mapped_context_transform,
  active}``; the live ``interventions`` table (what the scoring uses) is updated to the newest
  version by ``PUT /interventions``; the versions here are the history.
* ``intake_session`` / ``<session_id>``: the assignment record and session log of an intake
  battery run in-session or uploaded (``client_id``, ``form``, ``seed``, ``assignment``,
  ``text_versions``, ``n_rows``, ``is_synthetic``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from bre import design as D
from bre import predict as P
from bre import schema as S

BATTERY_JSON = S.PROJECT_ROOT / "instrument" / "battery.json"

KINDS: tuple[str, ...] = ("settings", "context", "scenario", "intervention", "intake_session")

SETTINGS_KEY = "settings"

DEFAULT_SETTINGS: dict[str, Any] = {
    "alert_threshold": float(P.STATUS_THRESHOLDS["high"]),
    "status_thresholds": {"elevated": float(P.STATUS_THRESHOLDS["elevated"]), "high": float(P.STATUS_THRESHOLDS["high"])},
    "capacity_target": float(P.DRAWDOWN_TARGET),
    "prior_strength": 1.0,
    "typical_crisis_contexts": list(P.TYPICAL_CRISIS_CONTEXTS),
    "demo_mode": False,
    "retrain": {"steps": 1500, "restarts": 3, "n_samples": 50, "val_frac": 0.2, "seed": 0},
}
"""Defaults of ``GET /settings`` (documented here, returned as ``defaults`` by the endpoint):

* ``alert_threshold``: the triage alert (``N clients above threshold``) starts at this predicted
  probability of selling (``bre.predict.STATUS_THRESHOLDS['high']``);
* ``status_thresholds``: the ``elevated`` / ``high`` boundaries of the triage status;
* ``capacity_target``: default target of the behavioral drawdown capacity
  (``bre.predict.DRAWDOWN_TARGET``);
* ``prior_strength``: multiplier on the strength of the context-rotation prior used by a refit
  (``theta_prior_scale = 2.0 / prior_strength`` for Q2/Q4, ``bre.retrain``); 1 = the fit default;
* ``typical_crisis_contexts``: the ordered context sequence of the drawdown capacity
  (``bre.predict.TYPICAL_CRISIS_CONTEXTS``);
* ``demo_mode``: force demo mode on (the banner on every screen). It can never switch demo mode
  off while the served model was trained on synthetic data or the database lives under
  ``data/synthetic`` (CLAUDE.md rule 3);
* ``retrain``: the optimizer budget of ``POST /retrain`` / ``make retrain``."""

SETTINGS_RULES: dict[str, str] = {
    "alert_threshold": "float in [0, 1]",
    "status_thresholds": "object {elevated, high} with 0 <= elevated <= high <= 1",
    "capacity_target": "float in (0, 1)",
    "prior_strength": "float > 0",
    "typical_crisis_contexts": "list of at most two distinct context tags (namespace:value)",
    "demo_mode": "bool",
    "retrain": "object {steps (1..5000), restarts (1..10), n_samples (0..500), val_frac [0.05, 0.5], seed (int)}",
}

SCENARIO_KEYS: tuple[str, ...] = (
    "intro", "loss_sentence", "context_lead_in_single", "context_lead_in_pair", "reminder",
    "tolerance_question", "sell_hold_question", "allocation_question", "delay_page_text",
)
"""The editable scenario texts (seeded from ``instrument/battery.json``)."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ConfigBase(DeclarativeBase):
    """Declarative base of the dashboard's configuration documents (separate from ``db.models.Base``)."""


class Document(ConfigBase):
    """One version of one document (module docstring). Rows are never updated or deleted,
    except that ``intake_session`` documents of a deleted client are removed with the client."""

    __tablename__ = "bre_documents"
    __table_args__ = (UniqueConstraint("kind", "key", "version", name="uq_bre_documents_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String, nullable=False, index=True)
    key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    actor: Mapped[str] = mapped_column(String, nullable=False, default="api")
    payload: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "key": self.key, "version": self.version, "created_at": self.created_at.isoformat(),
                "actor": self.actor, "payload": json.loads(self.payload)}


def ensure_tables(engine: Engine) -> None:
    """``create_all`` for the document table (no-op when it exists; no migration)."""
    ConfigBase.metadata.create_all(engine)


# ---------------------------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------------------------


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown document kind {kind!r}; one of {KINDS}")


def latest(session: Session, kind: str, key: str) -> Document | None:
    _check_kind(kind)
    return session.scalars(select(Document).where(Document.kind == kind, Document.key == key).order_by(Document.version.desc()).limit(1)).first()


def latest_all(session: Session, kind: str) -> list[Document]:
    """The newest version of every key of ``kind``, ordered by key."""
    _check_kind(kind)
    sub = select(Document.key, func.max(Document.version).label("v")).where(Document.kind == kind).group_by(Document.key).subquery()
    stmt = select(Document).join(sub, (Document.key == sub.c.key) & (Document.version == sub.c.v)).where(Document.kind == kind).order_by(Document.key)
    return list(session.scalars(stmt).all())


def history(session: Session, kind: str, key: str | None = None) -> list[Document]:
    _check_kind(kind)
    stmt = select(Document).where(Document.kind == kind)
    if key is not None:
        stmt = stmt.where(Document.key == key)
    return list(session.scalars(stmt.order_by(Document.key, Document.version)).all())


def append(session: Session, kind: str, key: str, payload: dict[str, Any], *, actor: str = "api") -> Document:
    """Insert the next version of ``(kind, key)`` (never updates an existing row)."""
    _check_kind(kind)
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    current = session.execute(select(func.max(Document.version)).where(Document.kind == kind, Document.key == key)).scalar()
    doc = Document(kind=kind, key=key, version=int(current or 0) + 1, created_at=utcnow(), actor=actor,
                   payload=json.dumps(P.jsonable(payload), sort_keys=True, allow_nan=False))
    session.add(doc)
    session.flush()
    return doc


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------


def _merge(defaults: dict[str, Any], stored: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(defaults))
    for k, v in stored.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def validate_settings(patch: dict[str, Any]) -> dict[str, Any]:
    """Check a settings patch against :data:`SETTINGS_RULES`; returns the cleaned patch.
    Unknown keys are an error (so a typo never silently persists)."""
    if not isinstance(patch, dict):
        raise ValueError("settings must be a JSON object")
    unknown = sorted(set(patch) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(f"unknown settings key(s) {unknown}; allowed {sorted(DEFAULT_SETTINGS)}")
    out: dict[str, Any] = {}
    for k, v in patch.items():
        if k == "alert_threshold":
            x = float(v)
            if not 0.0 <= x <= 1.0:
                raise ValueError("alert_threshold must lie in [0, 1]")
            out[k] = x
        elif k == "status_thresholds":
            if not isinstance(v, dict) or set(v) - {"elevated", "high"}:
                raise ValueError("status_thresholds must be an object with keys elevated and high")
            cur = {**DEFAULT_SETTINGS["status_thresholds"], **{kk: float(vv) for kk, vv in v.items()}}
            if not 0.0 <= cur["elevated"] <= cur["high"] <= 1.0:
                raise ValueError("status_thresholds must satisfy 0 <= elevated <= high <= 1")
            out[k] = cur
        elif k == "capacity_target":
            x = float(v)
            if not 0.0 < x < 1.0:
                raise ValueError("capacity_target must lie in (0, 1)")
            out[k] = x
        elif k == "prior_strength":
            x = float(v)
            if not x > 0:
                raise ValueError("prior_strength must be > 0")
            out[k] = x
        elif k == "typical_crisis_contexts":
            if not isinstance(v, list) or len(v) > 2 or len(set(v)) != len(v):
                raise ValueError("typical_crisis_contexts must be a list of at most two distinct context tags")
            for t in v:
                if not isinstance(t, str) or not S.TAG_PATTERN.fullmatch(t):
                    raise ValueError(f"context tag {t!r} is not of the form namespace:value")
            out[k] = list(v)
        elif k == "demo_mode":
            if not isinstance(v, bool):
                raise ValueError("demo_mode must be a bool")
            out[k] = v
        elif k == "retrain":
            if not isinstance(v, dict) or set(v) - set(DEFAULT_SETTINGS["retrain"]):
                raise ValueError(f"retrain must be an object with keys among {sorted(DEFAULT_SETTINGS['retrain'])}")
            cur = {**DEFAULT_SETTINGS["retrain"], **v}
            steps, restarts, n_samples, val_frac, seed = int(cur["steps"]), int(cur["restarts"]), int(cur["n_samples"]), float(cur["val_frac"]), int(cur["seed"])
            if not (1 <= steps <= 5000 and 1 <= restarts <= 10 and 0 <= n_samples <= 500 and 0.05 <= val_frac <= 0.5):
                raise ValueError("retrain budget out of range: steps 1..5000, restarts 1..10, n_samples 0..500, val_frac 0.05..0.5")
            out[k] = {"steps": steps, "restarts": restarts, "n_samples": n_samples, "val_frac": val_frac, "seed": seed}
    return out


def get_settings(session: Session) -> dict[str, Any]:
    """The effective settings: the newest stored document merged over :data:`DEFAULT_SETTINGS`."""
    doc = latest(session, "settings", SETTINGS_KEY)
    stored = json.loads(doc.payload) if doc is not None else {}
    out = _merge(DEFAULT_SETTINGS, stored)
    out["_version"] = 0 if doc is None else int(doc.version)
    out["_updated_at"] = None if doc is None else doc.created_at.isoformat()
    return out


def put_settings(session: Session, patch: dict[str, Any], *, actor: str = "api") -> dict[str, Any]:
    """Validate ``patch``, store the merged document as a new version, return the effective settings."""
    clean = validate_settings(patch)
    doc = latest(session, "settings", SETTINGS_KEY)
    stored = json.loads(doc.payload) if doc is not None else {}
    merged = _merge(stored, clean) if stored else clean
    append(session, "settings", SETTINGS_KEY, merged, actor=actor)
    return get_settings(session)


# ---------------------------------------------------------------------------------------------
# Seeds from the instrument definition
# ---------------------------------------------------------------------------------------------


def load_battery(path: Path = BATTERY_JSON) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def battery_scenario_texts(battery: dict[str, Any]) -> dict[str, str]:
    """The seed values of :data:`SCENARIO_KEYS` read from ``battery.json``."""
    d = battery["design"]
    tpl = d["scenario"]["template"]
    q = d["questions"]
    return {
        "intro": str(tpl["intro"]),
        "loss_sentence": str(tpl["loss_sentence"]),
        "context_lead_in_single": str(tpl["context_lead_in"]["single"]),
        "context_lead_in_pair": str(tpl["context_lead_in"]["pair"]),
        "reminder": str(tpl["reminder"]),
        "tolerance_question": str(q["tolerance"]["text"]),
        "sell_hold_question": str(q["sell_hold"]["text"]),
        "allocation_question": str(q["allocation_share"]["text"]),
        "delay_page_text": str(d["delay_page"]["text"]),
    }


def battery_contexts(battery: dict[str, Any]) -> list[dict[str, Any]]:
    """The design's non-null contexts with their instrument wording."""
    out = []
    for tag, c in battery["design"]["contexts"].items():
        if tag == D.CONTEXT_NONE:
            continue
        out.append({"tag": tag, "kind": str(c.get("kind")), "display": str(c.get("display") or ""), "active": True,
                    "triggers_delay": bool(c.get("triggers_delay", False)), "note": "seeded from instrument/battery.json (design 1.0.0)"})
    return out


def seed_documents(session: Session, battery_path: Path = BATTERY_JSON, *, actor: str = "seed") -> dict[str, int]:
    """Seed the scenario texts and context definitions that are missing (idempotent)."""
    try:
        battery = load_battery(battery_path)
    except (OSError, ValueError):
        battery = None
    seeded = {"scenario": 0, "context": 0}
    if battery is None:
        texts = {k: "" for k in SCENARIO_KEYS}
        texts.update({"tolerance_question": D.TOLERANCE_QUESTION, "sell_hold_question": D.SELL_PROMPT})
        contexts = [{"tag": t, "kind": t.split(":", 1)[0], "display": D.CONTEXT_SENTENCES[t], "active": True, "triggers_delay": t.startswith("news:"), "note": "seeded from bre.design (battery.json unreadable)"} for t in D.NONNULL_CONTEXTS]
    else:
        texts = battery_scenario_texts(battery)
        contexts = battery_contexts(battery)
    have_s = {d.key for d in latest_all(session, "scenario")}
    for key in SCENARIO_KEYS:
        if key not in have_s:
            append(session, "scenario", key, {"text": texts.get(key, ""), "note": "seed"}, actor=actor)
            seeded["scenario"] += 1
    have_c = {d.key for d in latest_all(session, "context")}
    for c in contexts:
        if c["tag"] not in have_c:
            append(session, "context", c["tag"], c, actor=actor)
            seeded["context"] += 1
    return seeded


def scenario_texts(session: Session) -> dict[str, dict[str, Any]]:
    """``key -> {text, note, version, updated_at, actor}`` of the newest scenario versions."""
    out: dict[str, dict[str, Any]] = {}
    for d in latest_all(session, "scenario"):
        p = json.loads(d.payload)
        out[d.key] = {"text": p.get("text", ""), "note": p.get("note"), "version": d.version, "updated_at": d.created_at.isoformat(), "actor": d.actor}
    return out


def context_definitions(session: Session) -> list[dict[str, Any]]:
    """The newest version of every context definition, design contexts first."""
    rows = []
    for d in latest_all(session, "context"):
        p = json.loads(d.payload)
        rows.append({**p, "tag": d.key, "version": d.version, "updated_at": d.created_at.isoformat(), "actor": d.actor})
    order = {t: i for i, t in enumerate(D.NONNULL_CONTEXTS)}
    rows.sort(key=lambda r: (order.get(r["tag"], len(order)), r["tag"]))
    return rows


def delete_intake_sessions(session: Session, client_id: str) -> int:
    """Remove the ``intake_session`` documents of a client (the only documents tied to a
    client; called by ``DELETE /clients/{id}`` so the cascade covers this table too)."""
    docs = [d for d in history(session, "intake_session") if json.loads(d.payload).get("client_id") == client_id]
    for d in docs:
        session.delete(d)
    session.flush()
    return len(docs)


__all__ = [
    "BATTERY_JSON", "ConfigBase", "DEFAULT_SETTINGS", "Document", "KINDS", "SCENARIO_KEYS", "SETTINGS_KEY", "SETTINGS_RULES",
    "append", "battery_contexts", "battery_scenario_texts", "context_definitions", "delete_intake_sessions", "ensure_tables",
    "get_settings", "history", "latest", "latest_all", "load_battery", "put_settings", "scenario_texts", "seed_documents",
    "validate_settings",
]
