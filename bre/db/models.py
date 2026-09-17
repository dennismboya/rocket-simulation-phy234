"""SQLAlchemy 2.x declarative models for the Behavioral Risk Engine database.

The ``responses`` table carries the unified DecisionEvent schema (PLAN.md section 2) column for
column, in the same order as the parquet files, plus an autoincrement ``id`` and an optional
``client_id`` linking a row to a dashboard client. Everything that is a list or an object in the
schema is stored as JSON text so that SQLite holds exactly what the parquet files hold.

Single source of truth: every schema constant here is imported from :mod:`bre.schema` (column
order, nullable/list/JSON/bool columns, elicitation types, covariate keys, uniqueness key) and the
``@validates`` hooks call the same rule helpers the parquet side uses, so a row the ORM accepts is
a row ``bre.schema.validate_frame`` accepts after ``responses_to_frame`` and vice versa:

* non-empty strings for the identifier columns; no ``str()`` / ``int()`` / ``float()`` coercion
  anywhere (a numeric string, a bool for a number, a 0/1 int for a bool, a non-integer float for
  an integer column, a NaN or infinity are all errors);
* position_in_session >= 0; horizon_days >= 0 or null; response_time_ms >= 0 or null;
  loss_pct in [-1, 1] or null; all finite;
* elicitation_type in ``bre.schema.ELICITATION_TYPES`` and the response range by type
  (binary_sell / lottery_choice / binary_yes_no in {0, 1}; allocation_pct / choice_rate in [0, 1];
  likert an integer in 1..7);
* question_order_id in ``bre.design.QUESTION_ORDER_IDS`` when battery_version is the shared
  design's ``DESIGN_VERSION``, free-form otherwise;
* covariates a JSON object with the six product keys, ``weight`` and ``x_*`` keys (responses) or
  the six product keys only (clients); outcome_behavior null or a JSON object with ``action``
  (non-empty string) and ``lag_days`` (number >= 0); context_tags and prior_question_ids JSON
  arrays of non-empty strings, context_tags matching ``namespace:value``; no NaN/Infinity tokens;
* uniqueness of (dataset, subject_id, session_id, position_in_session).

Where SQLite can express a rule it is also a CHECK constraint (ranges, the elicitation enum,
``json_valid`` / ``json_type`` on every JSON column, ``IN (0, 1)`` on every boolean column), so a
row written around the ORM cannot corrupt the table either.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    validates,
)

from bre import schema as S
from bre.design import DESIGN_VERSION, QUESTION_ORDER_IDS

# --------------------------------------------------------------------------------------------
# Schema constants: re-exported from bre.schema (PLAN.md section 2), never redefined here
# --------------------------------------------------------------------------------------------

#: DecisionEvent columns in canonical order. Every DataFrame produced by this package uses it.
SCHEMA_COLUMNS: tuple[str, ...] = tuple(S.COLUMNS)

#: Columns whose stored value is a JSON array of strings.
LIST_COLUMNS: tuple[str, ...] = tuple(S.LIST_COLUMNS)

#: Columns whose stored value is a JSON-encoded object (or null where allowed).
JSON_COLUMNS: tuple[str, ...] = tuple(S.JSON_COLUMNS)

#: Columns that may be null in the schema.
NULLABLE_COLUMNS: frozenset[str] = frozenset(S.NULLABLE_COLUMNS)

#: Boolean schema columns.
BOOL_COLUMNS: tuple[str, ...] = tuple(S.BOOL_COLUMNS)

#: Allowed elicitation types.
ELICITATION_TYPES: tuple[str, ...] = tuple(S.ELICITATION_TYPES)

#: The six product covariate keys (PLAN.md section 3). ``responses.covariates`` additionally takes
#: ``weight`` and ``x_*`` keys (see ``bre.schema.validate_covariates``); ``clients.covariates``
#: takes these six only.
COVARIATE_KEYS: tuple[str, ...] = tuple(S.COVARIATE_KEYS)

#: Keys an outcome_behavior object must carry.
OUTCOME_BEHAVIOR_REQUIRED_KEYS: tuple[str, ...] = tuple(S.OUTCOME_REQUIRED_KEYS)

#: Uniqueness key of a decision event.
UNIQUE_KEY: tuple[str, ...] = tuple(S.KEY_COLUMNS)

_STRING_ID_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "dataset",
    "session_id",
    "scenario_id",
    "source_row_ref",
)


def utcnow() -> datetime:
    """Return the current UTC time as a naive datetime (SQLite stores no zone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _sql_in(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


# --------------------------------------------------------------------------------------------
# Validation helpers shared by the models and by session.frame_to_responses
# --------------------------------------------------------------------------------------------


def _json_loads(value: str, field: str) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a JSON string, got {type(value).__name__}")
    try:
        return S.json_loads_strict(value)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError; so is the NaN-token error
        raise ValueError(f"{field} is not valid JSON: {exc}") from exc


def parse_json_object(value: str | None, field: str, *, allow_null: bool) -> dict[str, Any] | None:
    """Rule: ``field`` holds a JSON-encoded object (``{...}``) or, if allowed, null.

    Returns the decoded object so callers can inspect keys. The non-standard ``NaN`` /
    ``Infinity`` tokens are rejected. Raises ValueError otherwise.
    """
    if value is None:
        if allow_null:
            return None
        raise ValueError(f"{field} must be a JSON object string, got null")
    obj = _json_loads(value, field)
    if obj is None and allow_null:
        return None
    if not isinstance(obj, dict):
        raise ValueError(f"{field} must encode a JSON object, got {type(obj).__name__}")
    return obj


def parse_json_array(value: str, field: str) -> list[Any]:
    """Rule: ``field`` holds a JSON array (any items). Returns the list."""
    items = _json_loads(value, field)
    if not isinstance(items, list):
        raise ValueError(f"{field} must encode a JSON array, got {type(items).__name__}")
    return items


def parse_json_string_list(value: str, field: str, *, tags: bool = False) -> list[str]:
    """Rule: ``field`` holds a JSON array of non-empty strings (empty array allowed); with
    ``tags`` every item is a ``namespace:value`` context tag. Returns the list."""
    items = parse_json_array(value, field)
    return S.validate_string_list(items, field, tags=tags)


def encode_string_list(value: Any, field: str) -> str:
    """Serialize an ordered list of strings (list, tuple or 1-d numpy array) to JSON text.

    Nothing is coerced: a str, bytes, dict, set or generator is rejected (order is data), and so
    is any item that is not a non-empty str. ``context_tags`` items must be ``namespace:value``.
    """
    items = S.validate_string_list(value, field, tags=(field == "context_tags"))
    return json.dumps(items, allow_nan=False)


def validate_iso_timestamp(value: str | None) -> str | None:
    """Rule: timestamp is an ISO-8601 string (``datetime.fromisoformat`` must parse it) or null."""
    if value is None:
        return None
    if isinstance(value, np.str_):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError(f"timestamp must be an ISO-8601 string, got {type(value).__name__}")
    return S.parse_timestamp(value)


def _number(value: Any, field: str) -> float:
    """A finite Python float from an int/float/numpy number; bool, str, None and NaN are errors."""
    if value is None or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a number, got {value!r}")
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a number, not a string ({value!r})")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field} must be a number, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{field} must be finite, got {v}")
    return v


def _integer(value: Any, field: str) -> int:
    """A Python int from an int, numpy integer or integer-valued finite float; nothing else."""
    v = _number(value, field)
    if not v.is_integer():
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return int(v)


def _boolean(value: Any, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{field} must be a bool, got {value!r}")


def _nonempty_str(value: Any, field: str) -> str:
    if isinstance(value, np.str_):
        value = str(value)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string, got {value!r}")
    return value


def validate_response_range(elicitation_type: str, response: Any) -> float:
    """Rule: response range by elicitation type (``bre.schema.response_in_range``).

    binary_sell, lottery_choice, binary_yes_no: response in {0, 1};
    allocation_pct, choice_rate: 0 <= response <= 1;
    likert: response is an integer with 1 <= response <= 7.
    """
    if elicitation_type not in ELICITATION_TYPES:
        raise ValueError(f"elicitation_type must be one of {ELICITATION_TYPES}, got {elicitation_type!r}")
    r = _number(response, "response")
    if not S.response_in_range(elicitation_type, r):
        raise ValueError(f"response {r} is out of range for {elicitation_type!r} ({S.COLUMN_RULES['response']})")
    return r


def validate_question_order(question_order_id: str | None, battery_version: str | None) -> str | None:
    """Rule: a row of the shared design (``battery_version == DESIGN_VERSION``) uses one of
    ``bre.design.QUESTION_ORDER_IDS``; any string is allowed otherwise; null always is."""
    if question_order_id is None:
        return None
    if isinstance(question_order_id, np.str_):
        question_order_id = str(question_order_id)
    if not isinstance(question_order_id, str):
        raise ValueError(f"question_order_id must be a string or null, got {type(question_order_id).__name__}")
    if battery_version is not None and not S.question_order_ok(question_order_id, battery_version):
        raise ValueError(
            f"question_order_id {question_order_id!r} is not one of {QUESTION_ORDER_IDS} although "
            f"battery_version is the shared design {DESIGN_VERSION!r}"
        )
    return question_order_id


# --------------------------------------------------------------------------------------------
# Declarative base and tables
# --------------------------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Declarative base for every BRE table."""

    def to_dict(self) -> dict[str, Any]:
        """Return the stored column values keyed by DB column name.

        Datetimes become ISO strings; JSON text columns (including the list columns of
        ``responses``) are returned exactly as stored, not decoded.
        """
        out: dict[str, Any] = {}
        for prop in sa_inspect(type(self)).column_attrs:
            value = getattr(self, prop.key)
            if isinstance(value, datetime):
                value = value.isoformat()
            out[prop.columns[0].name] = value
        return out


class Client(Base):
    """A dashboard client (investor) whose intake responses and predictions are kept together.

    ``covariates`` takes the six product keys only (no ``weight``, no ``x_*`` keys), so nothing
    identifying can be stored outside ``display_label``. Deleting a client cascades
    (ON DELETE CASCADE) to responses, predictions_log and intervention_log; see
    ``db.session.delete_client`` which verifies the cascade and writes the audit entry.
    """

    __tablename__ = "clients"
    __table_args__ = (
        CheckConstraint("json_valid(covariates) AND json_type(covariates) = 'object'", name="ck_clients_covariates_json"),
    )

    client_id: Mapped[str] = mapped_column(String, primary_key=True)
    display_label: Mapped[str | None] = mapped_column(String, nullable=True)
    covariates: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)

    responses: Mapped[list["Response"]] = relationship(
        back_populates="client", cascade="all, delete-orphan", passive_deletes=True
    )
    predictions: Mapped[list["PredictionLog"]] = relationship(
        back_populates="client", cascade="all, delete-orphan", passive_deletes=True
    )
    intervention_logs: Mapped[list["InterventionLog"]] = relationship(
        back_populates="client", cascade="all, delete-orphan", passive_deletes=True
    )

    @validates("client_id")
    def _validate_client_id(self, key: str, value: Any) -> str:
        return _nonempty_str(value, key)

    @validates("covariates")
    def _validate_covariates(self, key: str, value: Any) -> str:
        obj = parse_json_object(value, key, allow_null=False)
        S.validate_covariates(obj, allow_extras=False)
        return value


class Response(Base):
    """One elicited decision (DecisionEvent). Columns follow ``SCHEMA_COLUMNS`` in order."""

    __tablename__ = "responses"
    __table_args__ = (
        UniqueConstraint(*UNIQUE_KEY, name="uq_responses_event"),
        CheckConstraint("position_in_session >= 0", name="ck_responses_position_nonneg"),
        CheckConstraint("loss_pct IS NULL OR (loss_pct >= -1 AND loss_pct <= 1)", name="ck_responses_loss_pct"),
        CheckConstraint("horizon_days IS NULL OR horizon_days >= 0", name="ck_responses_horizon"),
        CheckConstraint("response_time_ms IS NULL OR response_time_ms >= 0", name="ck_responses_rt"),
        CheckConstraint(f"elicitation_type IN {_sql_in(ELICITATION_TYPES)}", name="ck_responses_elicitation_type"),
        CheckConstraint(
            f"(elicitation_type IN {_sql_in(tuple(sorted(S.BINARY_ELICITATION_TYPES)))} AND response IN (0, 1)) OR "
            f"(elicitation_type IN {_sql_in(tuple(sorted(S.RATE_ELICITATION_TYPES)))} AND response >= 0 AND response <= 1) OR "
            "(elicitation_type = 'likert' AND response >= 1 AND response <= 7 "
            "AND response = CAST(response AS INTEGER))",
            name="ck_responses_response_range",
        ),
        CheckConstraint("json_valid(covariates) AND json_type(covariates) = 'object'", name="ck_responses_covariates_json"),
        CheckConstraint("json_valid(context_tags) AND json_type(context_tags) = 'array'", name="ck_responses_context_tags_json"),
        CheckConstraint(
            "json_valid(prior_question_ids) AND json_type(prior_question_ids) = 'array'",
            name="ck_responses_prior_question_ids_json",
        ),
        CheckConstraint(
            "outcome_behavior IS NULL OR (json_valid(outcome_behavior) AND json_type(outcome_behavior) = 'object')",
            name="ck_responses_outcome_behavior_json",
        ),
        CheckConstraint("incentivized IN (0, 1)", name="ck_responses_incentivized_bool"),
        CheckConstraint("consent_training IN (0, 1)", name="ck_responses_consent_training_bool"),
        CheckConstraint("is_synthetic IN (0, 1)", name="ck_responses_is_synthetic_bool"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("clients.client_id", ondelete="CASCADE"), nullable=True, index=True
    )

    # --- DecisionEvent schema, in SCHEMA_COLUMNS order ---
    subject_id: Mapped[str] = mapped_column(String, nullable=False)
    dataset: Mapped[str] = mapped_column(String, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[str | None] = mapped_column(String, nullable=True)
    position_in_session: Mapped[int] = mapped_column(Integer, nullable=False)
    scenario_id: Mapped[str] = mapped_column(String, nullable=False)
    loss_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    horizon_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context_tags_json: Mapped[str] = mapped_column("context_tags", Text, nullable=False, default="[]")
    question_order_id: Mapped[str | None] = mapped_column(String, nullable=True)
    prior_question_ids_json: Mapped[str] = mapped_column(
        "prior_question_ids", Text, nullable=False, default="[]"
    )
    elicitation_type: Mapped[str] = mapped_column(String, nullable=False)
    response: Mapped[float] = mapped_column(Float, nullable=False)
    response_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    covariates: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    outcome_behavior: Mapped[str | None] = mapped_column(Text, nullable=True)
    incentivized: Mapped[bool] = mapped_column(Boolean, nullable=False)
    consent_training: Mapped[bool] = mapped_column(Boolean, nullable=False)
    battery_version: Mapped[str] = mapped_column(String, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    source_row_ref: Mapped[str] = mapped_column(String, nullable=False)

    client: Mapped[Client | None] = relationship(back_populates="responses")

    # --- list fields: JSON text in the DB, Python lists on the object ---

    @hybrid_property
    def context_tags(self) -> list[str]:
        """Ordered context tags, e.g. ["news:recession", "social:friend_sells"].

        An unassigned column (before INSERT applies the "[]" default) reads as the empty list.
        """
        if self.context_tags_json is None:
            return []
        return parse_json_string_list(self.context_tags_json, "context_tags", tags=True)

    @context_tags.inplace.setter
    def _context_tags_setter(self, value: Any) -> None:
        self.context_tags_json = encode_string_list(value, "context_tags")

    @context_tags.inplace.expression
    @classmethod
    def _context_tags_expression(cls):
        return cls.context_tags_json

    @hybrid_property
    def prior_question_ids(self) -> list[str]:
        """Ordered ids of the questions asked before this one in the session (empty if unassigned)."""
        if self.prior_question_ids_json is None:
            return []
        return parse_json_string_list(self.prior_question_ids_json, "prior_question_ids")

    @prior_question_ids.inplace.setter
    def _prior_question_ids_setter(self, value: Any) -> None:
        self.prior_question_ids_json = encode_string_list(value, "prior_question_ids")

    @prior_question_ids.inplace.expression
    @classmethod
    def _prior_question_ids_expression(cls):
        return cls.prior_question_ids_json

    # --- Python-level validation: the bre.schema rules, with readable errors ---

    @validates(*_STRING_ID_COLUMNS)
    def _validate_strings(self, key: str, value: Any) -> str:
        return _nonempty_str(value, key)

    @validates("battery_version")
    def _validate_battery_version(self, key: str, value: Any) -> str:
        value = _nonempty_str(value, key)
        validate_question_order(self.question_order_id, value)
        return value

    @validates("question_order_id")
    def _validate_question_order_id(self, key: str, value: Any) -> str | None:
        return validate_question_order(value, self.battery_version)

    @validates("context_tags_json", "prior_question_ids_json")
    def _validate_lists(self, key: str, value: Any) -> str:
        field = key.removesuffix("_json")
        parse_json_string_list(value, field, tags=(field == "context_tags"))
        return value

    @validates("covariates")
    def _validate_covariates(self, key: str, value: Any) -> str:
        obj = parse_json_object(value, key, allow_null=False)
        S.validate_covariates(obj, allow_extras=True)
        return value

    @validates("outcome_behavior")
    def _validate_outcome_behavior(self, key: str, value: Any) -> str | None:
        obj = parse_json_object(value, key, allow_null=True)
        if obj is None:
            return None
        S.validate_outcome_behavior(obj)
        return value

    @validates("timestamp")
    def _validate_timestamp(self, key: str, value: Any) -> str | None:
        return validate_iso_timestamp(value)

    @validates("position_in_session")
    def _validate_position(self, key: str, value: Any) -> int:
        v = _integer(value, key)
        if v < 0:
            raise ValueError(f"position_in_session must be an integer >= 0, got {value!r}")
        return v

    @validates("loss_pct")
    def _validate_loss_pct(self, key: str, value: Any) -> float | None:
        if value is None:
            return None
        v = _number(value, key)
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"loss_pct must lie in [-1, 1], got {v}")
        return v

    @validates("horizon_days")
    def _validate_horizon(self, key: str, value: Any) -> int | None:
        if value is None:
            return None
        v = _integer(value, key)
        if v < 0:
            raise ValueError(f"horizon_days must be an integer >= 0 or null, got {value!r}")
        return v

    @validates("response_time_ms")
    def _validate_rt(self, key: str, value: Any) -> float | None:
        if value is None:
            return None
        v = _number(value, key)
        if v < 0:
            raise ValueError(f"response_time_ms must be >= 0 or null, got {v}")
        return v

    @validates("elicitation_type")
    def _validate_elicitation_type(self, key: str, value: Any) -> str:
        if isinstance(value, np.str_):
            value = str(value)
        if not isinstance(value, str) or value not in ELICITATION_TYPES:
            raise ValueError(f"elicitation_type must be one of {ELICITATION_TYPES}, got {value!r}")
        # Re-check the response if it was assigned before the type.
        if self.response is not None:
            validate_response_range(value, self.response)
        return value

    @validates("response")
    def _validate_response(self, key: str, value: Any) -> float:
        if self.elicitation_type is not None:
            return validate_response_range(self.elicitation_type, value)
        return _number(value, key)

    @validates(*BOOL_COLUMNS)
    def _validate_bools(self, key: str, value: Any) -> bool:
        return _boolean(value, key)

    def to_event(self) -> dict[str, Any]:
        """Return the DecisionEvent as a dict in ``SCHEMA_COLUMNS`` order with list fields as lists.

        A row whose stored JSON text is corrupt (written around the ORM) raises ValueError naming
        the row id, so it can be found and repaired.
        """
        out: dict[str, Any] = {}
        try:
            for col in SCHEMA_COLUMNS:
                if col == "context_tags":
                    out[col] = self.context_tags
                elif col == "prior_question_ids":
                    out[col] = self.prior_question_ids
                else:
                    out[col] = getattr(self, col)
        except ValueError as exc:
            raise ValueError(f"responses row id={self.id!r} is corrupt: {exc}") from exc
        return out


class Intervention(Base):
    """An advisor-facing intervention script and the context transform it is assumed to apply.

    ``mapped_context_transform`` is a JSON object using the vocabulary documented in
    ``db.seed``: ``add`` / ``remove`` (lists of context tags) and ``question_order``.
    """

    __tablename__ = "interventions"
    __table_args__ = (
        CheckConstraint(
            "json_valid(mapped_context_transform) AND json_type(mapped_context_transform) = 'object'",
            name="ck_interventions_transform_json",
        ),
        CheckConstraint("active IN (0, 1)", name="ck_interventions_active_bool"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    script: Mapped[str] = mapped_column(Text, nullable=False)
    mapped_context_transform: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    logs: Mapped[list["InterventionLog"]] = relationship(back_populates="intervention")

    @validates("mapped_context_transform")
    def _validate_transform(self, key: str, value: Any) -> str:
        parse_json_object(value, key, allow_null=False)
        return value

    @validates("active")
    def _validate_active(self, key: str, value: Any) -> bool:
        return _boolean(value, key)


class PredictionLog(Base):
    """A prediction shown to an advisor for a client, with the model version and inputs used."""

    __tablename__ = "predictions_log"
    __table_args__ = (
        CheckConstraint("json_valid(market_state) AND json_type(market_state) = 'object'", name="ck_predictions_market_state_json"),
        CheckConstraint("json_valid(outputs) AND json_type(outputs) = 'object'", name="ck_predictions_outputs_json"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[str] = mapped_column(
        String, ForeignKey("clients.client_id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    market_state: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    outputs: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    shown_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)

    client: Mapped[Client] = relationship(back_populates="predictions")

    @validates("market_state", "outputs")
    def _validate_json(self, key: str, value: Any) -> str:
        parse_json_object(value, key, allow_null=False)
        return value


class InterventionLog(Base):
    """An intervention applied to a client, the advisor's note and the later observed behavior."""

    __tablename__ = "intervention_log"
    __table_args__ = (
        CheckConstraint(
            "outcome_behavior IS NULL OR (json_valid(outcome_behavior) AND json_type(outcome_behavior) = 'object')",
            name="ck_intervention_log_outcome_json",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[str] = mapped_column(
        String, ForeignKey("clients.client_id", ondelete="CASCADE"), nullable=False, index=True
    )
    intervention_id: Mapped[int] = mapped_column(Integer, ForeignKey("interventions.id"), nullable=False)
    advisor_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcome_behavior: Mapped[str | None] = mapped_column(Text, nullable=True)
    logged_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)

    client: Mapped[Client] = relationship(back_populates="intervention_logs")
    intervention: Mapped[Intervention] = relationship(back_populates="logs")

    @validates("outcome_behavior")
    def _validate_outcome_behavior(self, key: str, value: Any) -> str | None:
        obj = parse_json_object(value, key, allow_null=True)
        if obj is None:
            return None
        S.validate_outcome_behavior(obj)
        return value


class ModelRegistry(Base):
    """A fitted model version: what it is, what it was trained on, how it scored, where it is calibrated."""

    __tablename__ = "model_registry"
    __table_args__ = (
        CheckConstraint(
            "json_valid(training_data_refs) AND json_type(training_data_refs) = 'array'",
            name="ck_model_registry_refs_json",
        ),
        CheckConstraint("json_valid(metrics) AND json_type(metrics) = 'object'", name="ck_model_registry_metrics_json"),
        CheckConstraint(
            "json_valid(calibrated_contexts) AND json_type(calibrated_contexts) = 'array'",
            name="ck_model_registry_contexts_json",
        ),
        CheckConstraint("is_active IN (0, 1)", name="ck_model_registry_is_active_bool"),
    )

    version: Mapped[str] = mapped_column(String, primary_key=True)
    model_type: Mapped[str] = mapped_column(String, nullable=False)
    training_data_refs: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    metrics: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    calibrated_contexts: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    @validates("training_data_refs", "calibrated_contexts")
    def _validate_json_arrays(self, key: str, value: Any) -> str:
        parse_json_array(value, key)
        return value

    @validates("metrics")
    def _validate_metrics(self, key: str, value: Any) -> str:
        parse_json_object(value, key, allow_null=False)
        return value

    @validates("is_active")
    def _validate_is_active(self, key: str, value: Any) -> bool:
        return _boolean(value, key)


class AuditLog(Base):
    """Append-only record of who did what to which entity, with a JSON payload."""

    __tablename__ = "audit_log"
    __table_args__ = (
        CheckConstraint("json_valid(payload) AND json_type(payload) = 'object'", name="ck_audit_log_payload_json"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    actor: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    entity: Mapped[str] = mapped_column(String, nullable=False)
    entity_id: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    @validates("payload")
    def _validate_payload(self, key: str, value: Any) -> str:
        parse_json_object(value, key, allow_null=False)
        return value


__all__ = [
    "AuditLog",
    "BOOL_COLUMNS",
    "Base",
    "COVARIATE_KEYS",
    "Client",
    "ELICITATION_TYPES",
    "Intervention",
    "InterventionLog",
    "JSON_COLUMNS",
    "LIST_COLUMNS",
    "ModelRegistry",
    "NULLABLE_COLUMNS",
    "OUTCOME_BEHAVIOR_REQUIRED_KEYS",
    "PredictionLog",
    "Response",
    "SCHEMA_COLUMNS",
    "UNIQUE_KEY",
    "encode_string_list",
    "parse_json_array",
    "parse_json_object",
    "parse_json_string_list",
    "utcnow",
    "validate_iso_timestamp",
    "validate_question_order",
    "validate_response_range",
]
