"""Engine, session and data-access helpers for the BRE SQLite database.

Rules implemented here:

* ``get_engine``: SQLite via ``sqlite:///...``; the URL comes from the argument, else the
  environment variable ``BRE_DB_URL``, else ``sqlite:///bre.db``. A relative SQLite file path is
  made absolute (``os.path.abspath``, no symlink resolution) when the engine is created, so the
  file the pool opens and the file the location rule judges are the same one whatever the current
  directory does later. Foreign keys are enforced by executing ``PRAGMA foreign_keys=ON`` on every
  new connection (SQLite defaults to OFF); ``session_scope`` verifies the pragma on SQLite binds
  and refuses an engine that was not built this way.
* ``default_engine``: one cached engine per *resolved* URL, so a change of ``BRE_DB_URL`` is
  honoured by the next call.
* ``delete_client``: removes the client row; ``ON DELETE CASCADE`` removes every dependent row in
  responses, predictions_log and intervention_log; the function re-counts the dependents after
  the flush and raises (rolling the transaction back) if any survived, so the audit_log entry only
  records a delete that happened.
* Real vs synthetic (CLAUDE.md rule 3): synthetic rows are only written to a database file located
  under ``data/synthetic/``; real rows are never written to a file under ``data/synthetic/``; the
  responses table never mixes the two flags. The rule is enforced in a ``before_flush`` listener
  on every ``Session`` (registered when this module is imported and re-asserted by ``get_engine``
  and ``session_scope``), so ``frame_to_responses`` and a plain ``session.add(Response(...))`` are
  checked alike, and a Session bound to a Connection is judged by that connection's engine.
  In-memory databases are exempt from the location rule (they hold no file) but still may not mix.
* ``frame_to_responses`` runs ``bre.schema.validate_frame(df, strict=True)`` first: a frame the
  parquet writer refuses is refused here too, cell values are never coerced.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Connection, Engine, create_engine, event, exists, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import get_history
from sqlalchemy.pool import StaticPool

from bre import schema as S
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


class SyntheticPolicyError(ValueError):
    """Raised when a flush would break the real-vs-synthetic rule (CLAUDE.md rule 3)."""


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


def _split_sqlite_database(database: str) -> tuple[str, str, str]:
    """Split a SQLite ``database`` URL part into (``file:`` prefix or "", path, ``?query`` or "")."""
    prefix = ""
    query = ""
    if database.startswith("file:"):
        prefix = "file:"
        database = database[len("file:") :]
    if "?" in database:
        database, query = database.split("?", 1)
        query = "?" + query
    return prefix, database, query


def absolute_sqlite_url(url: str) -> str:
    """Return ``url`` with a relative SQLite file path made absolute (``os.path.abspath``).

    In-memory and non-SQLite URLs are returned unchanged; ``file:`` URIs keep their prefix and
    query string. No symlink is resolved.
    """
    probe = make_url(url)
    if probe.get_backend_name() != "sqlite" or _is_memory_sqlite(probe):
        return url
    prefix, path, query = _split_sqlite_database(str(probe.database))
    absolute = os.path.abspath(os.path.expanduser(path))
    return str(probe.set(database=f"{prefix}{absolute}{query}"))


def sqlite_file_path(bind: Engine | Connection) -> Path | None:
    """Absolute path of a file-backed SQLite database, or None for in-memory / non-SQLite.

    A Connection is judged by its engine. The path is absolute by construction for engines from
    :func:`get_engine`; for any other engine it is made absolute here (no symlink resolution).
    """
    engine = bind.engine if isinstance(bind, Connection) else bind
    url = engine.url
    if url.get_backend_name() != "sqlite" or _is_memory_sqlite(url):
        return None
    _, path, _ = _split_sqlite_database(str(url.database))
    return Path(os.path.abspath(os.path.expanduser(path)))


def is_under_synthetic_dir(path: str | Path) -> bool:
    """True when a consecutive ``data/synthetic`` pair appears in the absolute path
    (``bre.schema.is_synthetic_path``: ``os.path.abspath``, no symlink resolution)."""
    return S.is_synthetic_path(path)


def get_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Create an engine; for SQLite, fix the file path, enable foreign keys, share in-memory DBs.

    A file-backed SQLite URL is rewritten with the absolute path before the engine is created
    (so ``engine.url`` names the file the pool will open) and its parent directory is created if
    missing. In-memory SQLite uses a StaticPool so every session sees the same database.
    """
    resolved = absolute_sqlite_url(resolve_db_url(url))
    kwargs: dict[str, Any] = {"echo": echo}
    engine_probe = make_url(resolved)
    if engine_probe.get_backend_name() == "sqlite":
        if _is_memory_sqlite(engine_probe):
            kwargs["poolclass"] = StaticPool
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            _, path, _ = _split_sqlite_database(str(engine_probe.database))
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(resolved, **kwargs)

    if engine.url.get_backend_name() == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, connection_record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    install_policy_listener()
    return engine


@lru_cache(maxsize=8)
def _engine_for(resolved_url: str) -> Engine:
    return get_engine(resolved_url)


def default_engine(url: str | None = None) -> Engine:
    """Process-wide cached engine per resolved URL (used when no engine is passed).

    The cache key is the resolved, absolute URL, not the argument: after ``BRE_DB_URL`` changes,
    the next call returns the engine of the new database.
    """
    return _engine_for(absolute_sqlite_url(resolve_db_url(url)))


def init_db(engine: Engine) -> None:
    """Create every table declared on ``Base`` (no-op for tables that already exist)."""
    Base.metadata.create_all(engine)


def assert_foreign_keys_enabled(session: Session) -> None:
    """Rule: on a SQLite bind ``PRAGMA foreign_keys`` must be 1 (engines from :func:`get_engine`).

    Raises RuntimeError otherwise, because ON DELETE CASCADE and the FK checks would silently be
    off for an engine built with a bare ``create_engine``.
    """
    bind = session.get_bind()
    if bind.engine.url.get_backend_name() != "sqlite":
        return
    value = session.execute(text("PRAGMA foreign_keys")).scalar()
    if value != 1:
        raise RuntimeError(
            f"PRAGMA foreign_keys is {value!r} on {bind.engine.url}; build the engine with "
            "db.get_engine so foreign keys and ON DELETE CASCADE are enforced"
        )


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Yield a Session; commit on success, roll back on exception, always close.

    The real-vs-synthetic policy listener is in place for the session, and on SQLite the
    ``foreign_keys`` pragma is verified before the session is handed out.
    """
    install_policy_listener()
    session = Session(bind=engine or default_engine(), expire_on_commit=False)
    try:
        assert_foreign_keys_enabled(session)
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


# --------------------------------------------------------------------------------------------
# Real-vs-synthetic policy (CLAUDE.md rule 3), enforced on every flush
# --------------------------------------------------------------------------------------------


def check_location_rule(bind: Engine | Connection, is_synthetic: bool) -> None:
    """Rule: synthetic rows only under data/synthetic/; real rows never there (file DBs only)."""
    path = sqlite_file_path(bind)
    if path is None:
        return
    under = is_under_synthetic_dir(path)
    if is_synthetic and not under:
        raise SyntheticPolicyError(
            f"synthetic responses (is_synthetic=True) may only be written to a database under "
            f"data/synthetic/, not {path}"
        )
    if not is_synthetic and under:
        raise SyntheticPolicyError(
            f"real responses (is_synthetic=False) may not be written to a database under "
            f"data/synthetic/ ({path})"
        )


def _pending_response_flags(session: Session) -> tuple[set[bool], set[int]]:
    """``is_synthetic`` values of the Response rows this flush will write, and the ids of rows
    whose flag is being changed (they are excluded from the 'already stored' comparison)."""
    flags: set[bool] = set()
    changed_ids: set[int] = set()
    for obj in session.new:
        if isinstance(obj, Response):
            flags.add(bool(obj.is_synthetic))
    for obj in session.dirty:
        if isinstance(obj, Response) and get_history(obj, "is_synthetic").has_changes():
            flags.add(bool(obj.is_synthetic))
            if obj.id is not None:
                changed_ids.add(obj.id)
    return flags, changed_ids


def enforce_synthetic_policy(session: Session, flush_context=None, instances=None) -> None:  # noqa: ANN001
    """``before_flush`` listener: the location rule and the no-mixing rule for every insert path."""
    flags, changed_ids = _pending_response_flags(session)
    if not flags:
        return
    if len(flags) > 1:
        raise SyntheticPolicyError("a flush mixes is_synthetic=True and is_synthetic=False responses")
    is_synthetic = flags.pop()
    check_location_rule(session.get_bind(), is_synthetic)
    stmt = select(exists().where(Response.is_synthetic != is_synthetic))
    if changed_ids:
        stmt = select(exists().where(Response.is_synthetic != is_synthetic, Response.id.not_in(changed_ids)))
    if session.execute(stmt).scalar():
        raise SyntheticPolicyError(
            "the responses table already holds rows with is_synthetic="
            f"{not is_synthetic}; a table may not mix real and synthetic rows"
        )


def install_policy_listener() -> None:
    """Register :func:`enforce_synthetic_policy` on the ``Session`` class once (idempotent)."""
    if not event.contains(Session, "before_flush", enforce_synthetic_policy):
        event.listen(Session, "before_flush", enforce_synthetic_policy)


install_policy_listener()


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
    """Append one audit_log row (payload serialized as JSON, no NaN/Infinity) and flush it."""
    entry = AuditLog(
        ts=utcnow(),
        actor=actor,
        action=action,
        entity=entity,
        entity_id=str(entity_id),
        payload=json.dumps(payload or {}, sort_keys=True, default=str, allow_nan=False),
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
    PRAGMA foreign_keys=ON) removes dependents; after the flush the dependents are counted again
    and, if any survived (foreign keys off on a foreign engine), the transaction is rolled back
    and RuntimeError is raised, so no audit entry ever claims a delete that did not happen.
    """
    client = session.get(Client, client_id)
    if client is None:
        raise LookupError(f"client {client_id!r} does not exist")
    counts = _dependent_counts(session, client_id)
    session.delete(client)
    session.flush()
    left = _dependent_counts(session, client_id)
    if any(left.values()):
        session.rollback()
        raise RuntimeError(
            f"deleting client {client_id!r} left dependent rows {left}; ON DELETE CASCADE is not "
            "active on this connection (use db.get_engine), nothing was deleted"
        )
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
    Rows are ordered by (dataset, subject_id, session_id, position_in_session). A corrupt row
    (JSON written around the ORM) raises ValueError naming its id.
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
    if isinstance(value, (list, tuple, np.ndarray, dict)):
        return False
    return S.is_null(value)


def _to_json_text(value: Any, field: str, *, allow_null: bool) -> str | None:
    """A JSON text cell: str passes through, a dict is serialized (no NaN), null if allowed."""
    if _is_null(value):
        if allow_null:
            return None
        raise ValueError(f"{field} must not be null")
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, allow_nan=False)
    raise ValueError(f"{field} must be a JSON string or an object, got {type(value).__name__}")


def _python_scalar(value: Any) -> Any:
    """numpy scalar -> Python scalar; everything else unchanged (no type coercion)."""
    return value.item() if isinstance(value, np.generic) else value


def _row_to_kwargs(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one validated frame row (schema columns) to ``Response`` constructor arguments.

    The frame has passed ``bre.schema.validate_frame``; this only unwraps frame-native values
    (numpy scalars, arrays, NaN nulls, pandas timestamps). Nothing is coerced: the ORM validators
    see the cell's own type and reject anything that is not exactly what the rule asks for.
    """
    kwargs: dict[str, Any] = {}
    for col in SCHEMA_COLUMNS:
        value = row[col]
        if col in LIST_COLUMNS:
            if _is_null(value):
                raise ValueError(f"{col} must be a list (empty allowed), got null")
            kwargs[f"{col}_json"] = encode_string_list(value, col)
        elif col == "timestamp":
            kwargs[col] = S.parse_timestamp(value)
        elif col in BOOL_COLUMNS:
            kwargs[col] = _python_scalar(value)  # the ORM validator accepts bool only
        elif col == "covariates":
            kwargs[col] = _to_json_text(value, col, allow_null=False)
        elif col == "outcome_behavior":
            kwargs[col] = _to_json_text(value, col, allow_null=True)
        else:
            if _is_null(value):
                if col in NULLABLE_COLUMNS:
                    kwargs[col] = None
                    continue
                raise ValueError(f"{col} must not be null")
            kwargs[col] = _python_scalar(value)
    return kwargs


def _check_synthetic_policy(session: Session, is_synthetic: bool) -> None:
    """Rule: synthetic rows only under data/synthetic/; real rows never there; no mixing.

    The same rule runs again in the ``before_flush`` listener; checking it here gives the caller
    the error before any Response object is built.
    """
    check_location_rule(session.get_bind(), is_synthetic)
    mixed = session.execute(select(exists().where(Response.is_synthetic != is_synthetic))).scalar()
    if mixed:
        raise SyntheticPolicyError(
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

    The frame must pass ``bre.schema.validate_frame(df, strict=True)`` (exactly the schema
    columns, no extra columns, every rule, homogeneous ``is_synthetic``); a failing frame raises
    ``bre.schema.SchemaError`` (a ValueError) before anything is built. The real-vs-synthetic
    location rule is then applied to the session's database. Rows are validated again by the
    model's ``@validates`` hooks; uniqueness and foreign keys are enforced by the database at
    flush. An audit_log row records the insert.
    """
    S.validate_frame(df, strict=True)
    if len(df) == 0:
        return 0
    is_synthetic = bool(df["is_synthetic"].iloc[0])
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
    "SyntheticPolicyError",
    "absolute_sqlite_url",
    "assert_foreign_keys_enabled",
    "check_location_rule",
    "default_engine",
    "delete_client",
    "enforce_synthetic_policy",
    "export_client",
    "frame_to_responses",
    "get_engine",
    "init_db",
    "install_policy_listener",
    "is_under_synthetic_dir",
    "resolve_db_url",
    "responses_to_frame",
    "session_scope",
    "sqlite_file_path",
    "write_audit",
]
