"""HTTP client of the BRE scoring API for the dashboard.

The FastAPI service (``api/main.py``) is the only source of predictions: nothing in ``app/``
imports ``bre`` model code. Every function returns the service's JSON as plain dicts and lists.
GET results and prediction results are memoised with ``st.cache_data`` keyed on the request
payload and the API base URL, so a page rerun with unchanged inputs does not call the service
again (and does not write another ``predictions_log`` row).

Base URL: ``st.session_state["api_base_url"]``, else ``$BRE_API_URL``, else
``http://127.0.0.1:$BRE_API_PORT`` (port 8000 by default, the ``make dashboard`` default).

Endpoints used: ``GET /health``, ``GET /model``, ``GET /clients``, ``GET /interventions``,
``GET /market``, ``GET /openapi.json`` (enums and route probing), ``POST /score_book``,
``POST /predict``, ``POST /profile``, ``POST /interventions/rank``, and the Phase 6 endpoints
(pages 4-7): ``GET/PUT /settings``, ``GET/POST /contexts``, ``GET/POST /scenarios``,
``PUT /interventions``, ``GET /interventions/versions``, ``GET /intake/config``,
``POST /clients/{client_id}/responses``, ``POST /intake/upload``,
``GET /clients/{client_id}/responses``, ``GET /clients/{client_id}/export``,
``DELETE /clients/{client_id}``, ``GET /model/registry``, ``POST /model/activate``,
``GET /transparency``, ``POST /retrain``.

The per-client response history is probed through ``/openapi.json``: when the route is absent
(an older service) the client falls back to a **read-only** SQLite read of the database the
service reports in ``/health``. Client contact status and the intervention log still live in
the session store (:mod:`components.store`).
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

import httpx
import streamlit as st

DEFAULT_PORT = "8000"
TIMEOUT_S = 180.0
CACHE_TTL_S = 900

RESPONSES_PATH = "/clients/{client_id}/responses"
"""Route probed for a client's response history (not served by the Phase 5 API)."""


class ApiError(RuntimeError):
    """A failed request: unreachable service or a non-2xx answer (``detail`` is the body)."""

    def __init__(self, message: str, status: int | None = None, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


def base_url() -> str:
    url = st.session_state.get("api_base_url") or os.environ.get("BRE_API_URL")
    if not url:
        url = f"http://127.0.0.1:{os.environ.get('BRE_API_PORT', DEFAULT_PORT)}"
    return str(url).rstrip("/")


def dumps(obj: Any) -> str:
    """Canonical JSON for cache keys and request bodies."""
    return json.dumps(obj, sort_keys=True, default=str)


def _format_detail(detail: Any) -> str:
    if isinstance(detail, list):
        parts = []
        for e in detail:
            if isinstance(e, dict):
                loc = ".".join(str(x) for x in e.get("loc", ()))
                parts.append(f"{loc}: {e.get('msg', '')}".strip(": "))
            else:
                parts.append(str(e))
        return "; ".join(parts)
    return str(detail)


def admin_headers() -> dict[str, str]:
    """``X-BRE-Admin: 1`` when this dashboard process runs with ``BRE_ADMIN=1`` (the API accepts
    the header as its admin flag; see api/phase6.py — a local convenience, not authentication)."""
    return {"X-BRE-Admin": "1"} if os.environ.get("BRE_ADMIN", "").strip() == "1" else {}


def is_admin() -> bool:
    return bool(admin_headers())


def _request(method: str, path: str, base: str, json_body: Any = None, *, headers: dict[str, str] | None = None, files: Any = None, data: Any = None, params: Any = None) -> Any:
    try:
        with httpx.Client(base_url=base, timeout=TIMEOUT_S) as c:
            r = c.request(method, path, json=json_body, headers=headers, files=files, data=data, params=params)
    except httpx.HTTPError as exc:
        raise ApiError(f"cannot reach the BRE API at {base} ({type(exc).__name__}: {exc}). Start it with `make api` or `make dashboard`.") from exc
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        raise ApiError(f"{method} {path} failed with HTTP {r.status_code}: {_format_detail(detail)}", r.status_code, detail)
    return r.json()


# ---------------------------------------------------------------------------------------------
# Service (uncached: liveness and the market feed are read fresh)
# ---------------------------------------------------------------------------------------------


def health(base: str) -> dict[str, Any]:
    return _request("GET", "/health", base)


def market(base: str) -> dict[str, Any]:
    """``GET /market``: the service's current market state and its scenario mapping."""
    return _request("GET", "/market", base)


# ---------------------------------------------------------------------------------------------
# Cached reads
# ---------------------------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def model_info(base: str) -> dict[str, Any]:
    return _request("GET", "/model", base)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def clients(base: str) -> list[dict[str, Any]]:
    return _request("GET", "/clients", base)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def interventions(base: str) -> list[dict[str, Any]]:
    return _request("GET", "/interventions", base)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def openapi(base: str) -> dict[str, Any]:
    try:
        return _request("GET", "/openapi.json", base)
    except ApiError:
        return {}


def has_path(base: str, path: str) -> bool:
    return path in (openapi(base).get("paths") or {})


def _enum_in(schema: Any) -> list[str] | None:
    if not isinstance(schema, dict):
        return None
    if "enum" in schema:
        return [str(v) for v in schema["enum"]]
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []) or []:
            found = _enum_in(sub)
            if found:
                return found
    return None


def enum_values(base: str, schema_name: str, prop: str, fallback: list[str]) -> list[str]:
    """The allowed values of a request field, read from the service's OpenAPI document."""
    spec = openapi(base)
    try:
        found = _enum_in(spec["components"]["schemas"][schema_name]["properties"][prop])
    except (KeyError, TypeError):
        found = None
    return found or list(fallback)


# ---------------------------------------------------------------------------------------------
# Cached predictions (keyed on the canonical JSON of the request)
# ---------------------------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def score_book(base: str, clients_json: str, market_json: str, target: float) -> dict[str, Any]:
    body = {"clients": json.loads(clients_json), "market_state": json.loads(market_json), "target": float(target)}
    return _request("POST", "/score_book", base, body)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def predict(base: str, payload_json: str) -> dict[str, Any]:
    return _request("POST", "/predict", base, json.loads(payload_json))


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def profile(base: str, payload_json: str) -> dict[str, Any]:
    return _request("POST", "/profile", base, json.loads(payload_json))


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def rank_interventions(base: str, payload_json: str) -> dict[str, Any]:
    return _request("POST", "/interventions/rank", base, json.loads(payload_json))


def client_payload(book: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``GET /clients`` rows as ``POST /score_book`` client entries (id, label, covariates)."""
    return [{"client_id": c["client_id"], "display_label": c.get("display_label"), "covariates": c.get("covariates") or {}} for c in book]


# ---------------------------------------------------------------------------------------------
# Response history: API route when it exists, else a read-only read of the served database
# ---------------------------------------------------------------------------------------------

RESPONSE_COLUMNS = (
    "id", "dataset", "session_id", "timestamp", "position_in_session", "scenario_id", "loss_pct",
    "context_tags", "question_order_id", "elicitation_type", "response", "response_time_ms",
    "outcome_behavior", "is_synthetic",
)

SOURCE_API = "api"
SOURCE_DB_READONLY = "database (read-only fallback: the API has no response-history endpoint)"


def _loads(text: Any, default: Any) -> Any:
    if text is None:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def client_responses(base: str, client_id: str) -> tuple[list[dict[str, Any]], str]:
    """A client's stored intake responses (oldest first) and where they were read from."""
    if has_path(base, RESPONSES_PATH):
        rows = _request("GET", RESPONSES_PATH.format(client_id=client_id), base)
        return list(rows), SOURCE_API
    db_path = str(health(base).get("db") or "")
    if not db_path or not os.path.exists(db_path):
        return [], SOURCE_DB_READONLY
    cols = ", ".join(RESPONSE_COLUMNS)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        con.row_factory = sqlite3.Row
        cur = con.execute(f"SELECT {cols} FROM responses WHERE client_id = ? ORDER BY session_id, position_in_session, id", (client_id,))
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["context_tags"] = _loads(d.get("context_tags"), [])
            d["outcome_behavior"] = _loads(d.get("outcome_behavior"), None)
            d["is_synthetic"] = bool(d.get("is_synthetic"))
            out.append(d)
    finally:
        con.close()
    return out, SOURCE_DB_READONLY


# ---------------------------------------------------------------------------------------------
# Phase 6 endpoints (pages 4-7): settings, editor documents, intake, registry, transparency
# ---------------------------------------------------------------------------------------------


def clear_caches() -> None:
    """After a write that changes what the cached reads return (settings, activation, retrain,
    contexts, deletes)."""
    st.cache_data.clear()


def get_settings(base: str) -> dict[str, Any]:
    return _request("GET", "/settings", base)


def put_settings(base: str, patch: dict[str, Any]) -> dict[str, Any]:
    out = _request("PUT", "/settings", base, patch)
    clear_caches()
    return out


def contexts(base: str) -> list[dict[str, Any]]:
    return _request("GET", "/contexts", base)


def post_context(base: str, body: dict[str, Any]) -> dict[str, Any]:
    out = _request("POST", "/contexts", base, body)
    clear_caches()
    return out


def scenarios(base: str, history: bool = False) -> dict[str, Any]:
    return _request("GET", "/scenarios", base, params={"history": "true"} if history else None)


def post_scenario(base: str, key: str, text: str, note: str | None = None) -> dict[str, Any]:
    return _request("POST", "/scenarios", base, {"key": key, "text": text, "note": note})


def interventions_versions(base: str, history: bool = False) -> list[dict[str, Any]]:
    return _request("GET", "/interventions/versions", base, params={"history": "true"} if history else None)


def put_intervention(base: str, body: dict[str, Any]) -> dict[str, Any]:
    out = _request("PUT", "/interventions", base, body)
    clear_caches()
    return out


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def intake_config(base: str) -> dict[str, Any]:
    return _request("GET", "/intake/config", base)


def post_responses(base: str, client_id: str, rows: list[dict[str, Any]], assignment: dict[str, Any], session: dict[str, Any], display_label: str | None = None) -> dict[str, Any]:
    out = _request("POST", f"/clients/{client_id}/responses", base, {"rows": rows, "assignment": assignment, "session": session, "display_label": display_label})
    clear_caches()
    return out


def upload_intake(base: str, client_id: str, filename: str, content: bytes, *, store_as_synthetic: bool = False, display_label: str | None = None) -> dict[str, Any]:
    data = {"client_id": client_id, "store_as_synthetic": "true" if store_as_synthetic else "false"}
    if display_label:
        data["display_label"] = display_label
    out = _request("POST", "/intake/upload", base, files={"file": (filename, content, "application/json")}, data=data)
    clear_caches()
    return out


def registry(base: str) -> dict[str, Any]:
    return _request("GET", "/model/registry", base, headers=admin_headers())


def activate(base: str, version: str) -> dict[str, Any]:
    out = _request("POST", "/model/activate", base, {"version": version}, headers=admin_headers())
    clear_caches()
    return out


def transparency(base: str) -> dict[str, Any]:
    return _request("GET", "/transparency", base)


def retrain(base: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    out = _request("POST", "/retrain", base, body or {})
    clear_caches()
    return out


def export_client(base: str, client_id: str) -> dict[str, Any]:
    return _request("GET", f"/clients/{client_id}/export", base)


def delete_client(base: str, client_id: str) -> dict[str, Any]:
    out = _request("DELETE", f"/clients/{client_id}", base)
    clear_caches()
    return out


__all__ = [
    "ApiError", "activate", "admin_headers", "base_url", "clear_caches", "client_payload", "client_responses", "clients", "contexts",
    "delete_client", "dumps", "enum_values", "export_client", "get_settings", "has_path", "health", "intake_config", "interventions",
    "interventions_versions", "is_admin", "market", "model_info", "openapi", "post_context", "post_responses", "post_scenario", "predict",
    "profile", "put_intervention", "put_settings", "rank_interventions", "registry", "retrain", "scenarios", "score_book", "transparency",
    "upload_intake",
]
