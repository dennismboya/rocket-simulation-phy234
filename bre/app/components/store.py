"""Session store for advisor bookkeeping that the Phase 5 API cannot persist.

The API has no endpoint for a client's contact status or for logging an applied intervention
(the database has an ``intervention_log`` table, but the service does not expose it), so both
live in ``st.session_state`` for the browser session and are labelled as such wherever they
are shown. Nothing here is a prediction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

STATUS_OPTIONS: tuple[str, ...] = ("not contacted", "contacted", "intervention logged")
PERSISTENCE_NOTE = "kept in this browser session only — the API has no endpoint for client status or intervention logs, so nothing is written to the database"

_KEY = "client_store"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _store() -> dict[str, dict[str, Any]]:
    return st.session_state.setdefault(_KEY, {})


def entry(client_id: str) -> dict[str, Any]:
    return _store().setdefault(client_id, {"status": STATUS_OPTIONS[0], "updated_at": None, "log": []})


def get_status(client_id: str) -> str:
    return str(entry(client_id)["status"])


def set_status(client_id: str, status: str) -> None:
    if status not in STATUS_OPTIONS:
        raise ValueError(f"status must be one of {STATUS_OPTIONS}")
    e = entry(client_id)
    if e["status"] != status:
        e["status"] = status
        e["updated_at"] = _now()


def last_contact(client_id: str) -> str | None:
    """Timestamp of the last status change or logged intervention in this session, else None."""
    e = entry(client_id)
    stamps = [e["updated_at"]] + [x["logged_at"] for x in e["log"]]
    stamps = [s for s in stamps if s]
    return max(stamps) if stamps else None


def log_intervention(client_id: str, name: str, note: str | None, p_before: float | None, delta_p: float | None, label: str) -> dict[str, Any]:
    e = entry(client_id)
    rec = {"intervention": name, "note": note or "", "logged_at": _now(), "p_before": p_before, "predicted_delta_p": delta_p, "label": label, "persisted": False}
    e["log"].append(rec)
    e["status"] = "intervention logged"
    e["updated_at"] = rec["logged_at"]
    return rec


def intervention_log(client_id: str) -> list[dict[str, Any]]:
    return list(entry(client_id)["log"])


__all__ = ["PERSISTENCE_NOTE", "STATUS_OPTIONS", "entry", "get_status", "intervention_log", "last_contact", "log_intervention", "set_status"]
