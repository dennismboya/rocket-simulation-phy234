"""Engine, session and data-access helpers for the BRE SQLite database.

Rules implemented here:

* ``get_engine``: SQLite via ``sqlite:///...``; the URL comes from the argument, else the
  environment variable ``BRE_DB_URL``, else ``sqlite:///bre.db``. Foreign keys are enforced by
  executing ``PRAGMA foreign_keys=ON`` on every new connection (SQLite defaults to OFF).
* ``delete_client``: removes the client row; ``ON DELETE CASCADE`` removes every dependent row in
  responses, predictions_log and intervention_log; an audit_log entry records the counts.
* Real vs synthetic (CLAUDE.md rule 3): a frame written by ``frame_to_responses`` must be
  homogeneous in ``is_synthetic``; synthetic rows are only written to a database file located
  under ``data/synthetic/``; real rows are never written to a file under ``data/synthetic/``; and
  the responses table never mixes the two flags. In-memory databases are exempt from the location
  rule (they hold no file) but still may not mix.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, create_engine, event, exists, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from db.models import (
    BOOL_COLUMNS,
    LIST_COLUMNS,
    NULLABLE_COLUMNS,
    SCHEMA_COLUMNS,
    AuditLog,
    Base,
    Client,
    InterventionLog,
    PredictionLog,
    Response,
    encode_string_list,
    utcnow,
)

DEFAULT_DB_URL = "sqlite:///bre.db"
ENV_DB_URL = "BRE_DB_URL"
SYNTHETIC_DIR_PARTS = ("data", "synthetic")


# --------------------------------------------------------------------------------------------
# Engine and session
# --------------------------------------------------------------------------------------------


def resolve_db_url(url: str | None = None) -> str:
    """Rule: explicit argument, else ``$BRE_DB_URL``, else ``sqlite:///bre.db``."""
    if url:
        return url
    return os.environ.get(ENV_DB_URL) or DEFAULT_DB_URL


def _is_memory_sqlite(engine_url: Any) -> bool:
    """True for ``sqlite://``, ``sqlite:///:memory:`` and ``mode=memory`` URIs."""
    if engine_url.get_backend_name() != "sqlite":
        return False
    database = engine_url.database
    return database in (None, "", ":memory:") or "mode=memory" in str(database)


def sqlite_file_path(engine: Engine) -> Path | None:
    """Absolute path of a file-backed SQLite database, or None for in-memory / non-SQLite."""
    url = engine.url
    if url.get_backend_name() != "sqlite" or _is_memory_sqlite(url):
        return None
    database = str(url.database)
    if database.startswith("file:"):
        database = database[len("file:") :].split("?", 1)[0]
    return Path(database).expanduser().resolve()


def is_under_synthetic_dir(path: Path) -> bool:
    """True when ``.../data/synthetic/...`` appears in the absolute path."""
    parts = path.resolve().parts
    n = len(SYNTHETIC_DIR_PARTS)
    return any(tuple(parts[i : i + n]) == SYNTHETIC_DIR_PARTS for i in range(len(parts) - n + 1))


def get_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Create an engine; for SQLite, enable foreign keys on connect and share in-memory DBs.

    In-memory SQLite uses a StaticPool so every session sees the same database.
    A file-backed SQLite path has its parent directory created if missing.
    """
    resolved = resolve_db_url(url)
    kwargs: dict[str, Any] = {"echo": echo}
    engine_probe = make_url(resolved)
    if engine_probe.get_backend_name() == "sqlite":
        if _is_memory_sqlite(engine_probe):
            kwargs["poolclass"] = StaticPool
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            db_path = Path(str(engine_probe.database)).expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(resolved, **kwargs)

    if engine.url.get_backend_name() == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, connection_record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


@lru_cache(maxsize=8)
def default_engine(url: str | None = None) -> Engine:
    """Process-wide cached engine for the resolved URL (used when no engine is passed)."""
    return get_engine(resolve_db_url(url))


def init_db(engine: Engine) -> None:
    """Create every table declared on ``Base`` (no-op for tables that already exist)."""
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Yield a Session; commit on success, roll back on exception, always close."""
    session = Session(bind=engine or default_engine(), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


# --------------------------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------------------------


def write_audit(
    session: Session,
    *,
    actor: str,
    action: str,
    entity: str,
    entity_id: str,
    payload: dict[str, Any] | None = None,
) -> AuditLog:
    """Append one audit_log row (payload serialized as JSON) and flush it."""
    entry = AuditLog(
        ts=utcnow(),
        actor=actor,
        action=action,
        entity=entity,
        entity_id=str(entity_id),
        payload=json.dumps(payload or {}, sort_keys=True, default=str),
    )
    session.add(entry)
    session.flush()
    return entry


# --------------------------------------------------------------------------------------------
# Client lifecycle
# --------------------------------------------------------------------------------------------


def _dependent_counts(session: Session, client_id: str) -> dict[str, int]:
    """Row counts of every table that references the client."""
    counts: dict[str, int] = {}
    for name, model in (
        ("responses", Response),
        ("predictions_log", PredictionLog),
        ("intervention_log", InterventionLog),
    ):
        counts[name] = int(
            session.execute(select(func.count()).select_from(model).where(model.client_id == client_id)).scalar_one()
        )
    return counts


def delete_client(session: Session, client_id: str, *, actor: str = "system") -> dict[str, int]:
    """Delete a client and every dependent row; write an audit entry; return the deleted counts.

    Raises LookupError when the client does not exist. The DB-level ON DELETE CASCADE (enabled by
    PRAGMA foreign_keys=ON) removes dependents; the ORM cascade covers rows already loaded.
    """
    client = session.get(Client, client_id)
    if client is None:
        raise LookupError(f"client {client_id!r} does not exist")
    counts = _dependent_counts(session, client_id)
    session.delete(client)
    session.flush()
    write_audit(
        session,
        actor=actor,
        action="delete_client",
        entity="clients",
        entity_id=client_id,
        payload={"deleted": counts},
    )
    return counts


def export_client(session: Session, client_id: str) -> dict[str, Any]:
    """Return every stored row belonging to a client as plain dicts (JSON text left as stored).

    Keys: ``client``, ``responses``, ``predictions_log``, ``intervention_log``.
    Raises LookupError when the client does not exist.
    """
    client = session.get(Client, client_id)
    if client is None:
        raise LookupError(f"client {client_id!r} does not exist")
    responses = session.scalars(
        select(Response)
        .where(Response.client_id == client_id)
        .order_by(Response.dataset, Response.session_id, Response.position_in_session, Response.id)
    ).all()
    predictions = session.scalars(
        select(PredictionLog).where(PredictionLog.client_id == client_id).order_by(PredictionLog.shown_at, PredictionLog.id)
    ).all()
    logs = session.scalars(
        select(InterventionLog)
        .where(InterventionLog.client_id == client_id)
        .order_by(InterventionLog.logged_at, InterventionLog.id)
    ).all()
    return {
        "client": client.to_dict(),
        "responses": [r.to_dict() for r in responses],
        "predictions_log": [p.to_dict() for p in predictions],
        "intervention_log": [entry.to_dict() for entry in logs],
    }


# --------------------------------------------------------------------------------------------
# DataFrame <-> responses
# --------------------------------------------------------------------------------------------


def responses_to_frame(
    session: Session,
    *,
    client_id: str | None = None,
    dataset: str | None = None,
    subject_id: str | None = None,
    is_synthetic: bool | None = None,
) -> pd.DataFrame:
    """Return matching responses as a DataFrame with exactly ``SCHEMA_COLUMNS`` in order.

    List columns hold Python lists; JSON columns hold the stored JSON text; nulls are NaN/NA.
    Rows are ordered by (dataset, subject_id, session_id, position_in_session).
    """
    stmt = select(Response)
    if client_id is not None:
        stmt = stmt.where(Response.client_id == client_id)
    if dataset is not None:
        stmt = stmt.where(Response.dataset == dataset)
    if subject_id is not None:
        stmt = stmt.where(Response.subject_id == subject_id)
    if is_synthetic is not None:
        stmt = stmt.where(Response.is_synthetic == bool(is_synthetic))
    stmt = stmt.order_by(Response.dataset, Response.subject_id, Response.session_id, Response.position_in_session)
    rows = [r.to_event() for r in session.scalars(stmt)]
    frame = pd.DataFrame.from_records(rows, columns=list(SCHEMA_COLUMNS))
    frame = frame.astype(
        {
            "position_in_session": "int64",
            "loss_pct": "float64",
            "horizon_days": "Int64",
            "response": "float64",
            "response_time_ms": "float64",
            "incentivized": "bool",
            "consent_training": "bool",
            "is_synthetic": "bool",
        }
    )
    return frame[list(SCHEMA_COLUMNS)]


def _is_null(value: Any) -> bool:
    """Null test for a scalar cell: None, NaN, pandas NA/NaT. Lists and arrays are never null."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, np.ndarray, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_timestamp_string(value: Any) -> str | None:
    """Coerce a cell to an ISO-8601 string (str passes through, datetimes are formatted) or None."""
    if _is_null(value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    raise ValueError(f"timestamp must be an ISO-8601 string or datetime, got {type(value).__name__}")


def _to_bool(value: Any, field: str) -> bool:
    """Coerce a cell to bool; accepts bool, numpy bool and exact 0/1 integers."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{field} must be boolean, got {value!r}")


def _to_json_text(value: Any, field: str, *, allow_null: bool) -> str | None:
    """Coerce a cell to JSON text: str passes through, dicts are serialized, null if allowed."""
    if _is_null(value):
        if allow_null:
            return None
        raise ValueError(f"{field} must not be null")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    raise ValueError(f"{field} must be a JSON string or an object, got {type(value).__name__}")


def _row_to_kwargs(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one frame row (schema columns) to ``Response`` constructor arguments."""
    kwargs: dict[str, Any] = {}
    for col in SCHEMA_COLUMNS:
        value = row[col]
        if col in LIST_COLUMNS:
            if _is_null(value):
                raise ValueError(f"{col} must be a list (empty allowed), got null")
            kwargs[f"{col}_json"] = encode_string_list(value, col)
        elif col == "timestamp":
            kwargs[col] = _to_timestamp_string(value)
        elif col in BOOL_COLUMNS:
            kwargs[col] = _to_bool(value, col)
        elif col == "covariates":
            kwargs[col] = _to_json_text(value, col, allow_null=False)
        elif col == "outcome_behavior":
            kwargs[col] = _to_json_text(value, col, allow_null=True)
        elif col in ("position_in_session", "horizon_days"):
            if _is_null(value):
                if col in NULLABLE_COLUMNS:
                    kwargs[col] = None
                    continue
                raise ValueError(f"{col} must not be null")
            kwargs[col] = int(value)
        elif col in ("loss_pct", "response", "response_time_ms"):
            if _is_null(value):
                if col in NULLABLE_COLUMNS:
                    kwargs[col] = None
                    continue
                raise ValueError(f"{col} must not be null")
            kwargs[col] = float(value)
        else:  # string columns
            if _is_null(value):
                if col in NULLABLE_COLUMNS:
                    kwargs[col] = None
                    continue
                raise ValueError(f"{col} must not be null")
            kwargs[col] = str(value)
    return kwargs


def _check_synthetic_policy(session: Session, is_synthetic: bool) -> None:
    """Rule: synthetic rows only under data/synthetic/; real rows never there; no mixing."""
    bind = session.get_bind()
    path = sqlite_file_path(bind) if isinstance(bind, Engine) else None
    if path is not None:
        under = is_under_synthetic_dir(path)
        if is_synthetic and not under:
            raise ValueError(
                f"synthetic responses (is_synthetic=True) may only be written to a database under "
                f"data/synthetic/, not {path}"
            )
        if not is_synthetic and under:
            raise ValueError(
                f"real responses (is_synthetic=False) may not be written to a database under "
                f"data/synthetic/ ({path})"
            )
    mixed = session.execute(select(exists().where(Response.is_synthetic != is_synthetic))).scalar()
    if mixed:
        raise ValueError(
            "the responses table already holds rows with is_synthetic="
            f"{not is_synthetic}; a table may not mix real and synthetic rows"
        )


def frame_to_responses(
    session: Session,
    df: pd.DataFrame,
    client_id: str | None = None,
    *,
    actor: str = "system",
) -> int:
    """Insert every row of a schema-conformant frame as a ``Response``; return the row count.

    The frame must contain every column of ``SCHEMA_COLUMNS`` (extra columns are ignored), be
    homogeneous in ``is_synthetic`` and satisfy the real-vs-synthetic location rule. Rows are
    validated by the model's ``@validates`` hooks; uniqueness and foreign keys are enforced by the
    database at flush. An audit_log row records the insert.
    """
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing schema columns: {missing}")
    if len(df) == 0:
        return 0
    flags = {_to_bool(v, "is_synthetic") for v in df["is_synthetic"].tolist()}
    if len(flags) != 1:
        raise ValueError("frame mixes is_synthetic=True and is_synthetic=False rows")
    is_synthetic = flags.pop()
    _check_synthetic_policy(session, is_synthetic)

    records = df[list(SCHEMA_COLUMNS)].to_dict(orient="records")
    objects = [Response(client_id=client_id, **_row_to_kwargs(rec)) for rec in records]
    session.add_all(objects)
    session.flush()
    write_audit(
        session,
        actor=actor,
        action="insert_responses",
        entity="responses",
        entity_id=client_id if client_id is not None else "-",
        payload={
            "rows": len(objects),
            "is_synthetic": is_synthetic,
            "datasets": sorted({o.dataset for o in objects}),
        },
    )
    return len(objects)


__all__ = [
    "DEFAULT_DB_URL",
    "ENV_DB_URL",
    "default_engine",
    "delete_client",
    "export_client",
    "frame_to_responses",
    "get_engine",
    "init_db",
    "is_under_synthetic_dir",
    "resolve_db_url",
    "responses_to_frame",
    "session_scope",
    "sqlite_file_path",
    "write_audit",
]
