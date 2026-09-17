"""Unified DecisionEvent schema of the Behavioral Risk Engine (PLAN.md section 2).

One row per elicited decision. This module defines the column list and order, the pydantic row
model :class:`DecisionEvent`, the pyarrow storage schema, frame constructors, the
:class:`ValidationReport` produced by :func:`validate_frame`, and the parquet writer/reader that
every loader and generator must go through. The real-vs-synthetic rule (CLAUDE.md rule 3) is
enforced here: synthetic tables live only under ``data/synthetic/`` and no table mixes the two.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from bre.design import COVARIATE_SPEC, DESIGN_VERSION, QUESTION_ORDER_IDS

SCHEMA_VERSION = "bre-schema-1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""The ``bre/`` directory."""
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
"""Only location where tables with ``is_synthetic == True`` may be written."""

# ---------------------------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------------------------

COLUMNS: list[str] = [
    "subject_id",
    "dataset",
    "session_id",
    "timestamp",
    "position_in_session",
    "scenario_id",
    "loss_pct",
    "horizon_days",
    "context_tags",
    "question_order_id",
    "prior_question_ids",
    "elicitation_type",
    "response",
    "response_time_ms",
    "covariates",
    "outcome_behavior",
    "incentivized",
    "consent_training",
    "battery_version",
    "is_synthetic",
    "source_row_ref",
]
"""Column order of every DecisionEvent table (build prompt section 3)."""

KEY_COLUMNS: list[str] = ["dataset", "subject_id", "session_id", "position_in_session"]
"""Uniqueness key of a row."""

NULLABLE_COLUMNS: frozenset[str] = frozenset(
    {"timestamp", "loss_pct", "horizon_days", "question_order_id", "response_time_ms", "outcome_behavior"}
)

LIST_COLUMNS: tuple[str, ...] = ("context_tags", "prior_question_ids")
JSON_COLUMNS: tuple[str, ...] = ("covariates", "outcome_behavior")
BOOL_COLUMNS: tuple[str, ...] = ("incentivized", "consent_training", "is_synthetic")
STRING_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "dataset",
    "session_id",
    "timestamp",
    "scenario_id",
    "question_order_id",
    "elicitation_type",
    "covariates",
    "outcome_behavior",
    "battery_version",
    "source_row_ref",
)

ELICITATION_TYPES: tuple[str, ...] = (
    "binary_sell",
    "allocation_pct",
    "likert",
    "lottery_choice",
    "binary_yes_no",
    "choice_rate",
)
"""PLAN.md section 2 (amendment 2026-09-17): ``binary_yes_no`` is a yes/no survey item such as the
tolerance question (response in {0, 1}); ``choice_rate`` is an aggregate share in [0, 1] choosing
the target option (choices13k bRate, CPC block rates, published joint proportions)."""

ElicitationType = Literal[
    "binary_sell", "allocation_pct", "likert", "lottery_choice", "binary_yes_no", "choice_rate"
]

BINARY_ELICITATION_TYPES: frozenset[str] = frozenset({"binary_sell", "lottery_choice", "binary_yes_no"})
"""Elicitation types whose response is in {0, 1}."""
RATE_ELICITATION_TYPES: frozenset[str] = frozenset({"allocation_pct", "choice_rate"})
"""Elicitation types whose response is a share in [0, 1]."""

TAG_PATTERN = re.compile(r"[A-Za-z0-9_]+:[A-Za-z0-9_.+\-]+\Z")
"""Context tags are namespaced ``namespace:value`` strings, e.g. ``news:recession``.

Always test with :meth:`re.Pattern.fullmatch` (``$`` would accept a trailing newline)."""

COVARIATE_KEYS: tuple[str, ...] = tuple(COVARIATE_SPEC)
"""The six product covariates of PLAN.md section 3 (the only keys ``clients.covariates`` takes)."""
COVARIATE_WEIGHT_KEY = "weight"
"""Extra covariate key of aggregate rows: the number of respondents behind the row (>= 0) or null."""
COVARIATE_EXTRA_PREFIX = "x_"
"""Prefix of dataset-specific covariate keys (any JSON scalar value; loaders keep PII out of them)."""
OUTCOME_REQUIRED_KEYS: tuple[str, ...] = ("action", "lag_days")

COLUMN_RULES: dict[str, str] = {
    "subject_id": "non-empty string",
    "dataset": "non-empty string",
    "session_id": "non-empty string",
    "timestamp": "ISO-8601 string or null",
    "position_in_session": "integer >= 0",
    "scenario_id": "non-empty string",
    "loss_pct": "float in [-1, 1] (signed) or null",
    "horizon_days": "integer >= 0 or null",
    "context_tags": "ordered list of 'namespace:value' strings (empty allowed)",
    "question_order_id": (
        "string or null; one of " + " | ".join(QUESTION_ORDER_IDS) + f" when battery_version == {DESIGN_VERSION!r}"
    ),
    "prior_question_ids": "ordered list of non-empty strings (empty allowed)",
    "elicitation_type": "one of " + " | ".join(ELICITATION_TYPES),
    "response": (
        "binary_sell/lottery_choice/binary_yes_no in {0,1}; allocation_pct/choice_rate in [0,1]; "
        "likert integer 1..7"
    ),
    "response_time_ms": "float >= 0 or null",
    "covariates": (
        "JSON object with keys among " + ", ".join(COVARIATE_KEYS) + " (values per COVARIATE_SPEC), "
        f"{COVARIATE_WEIGHT_KEY} (number >= 0 or null) and {COVARIATE_EXTRA_PREFIX}* keys (JSON scalars)"
    ),
    "outcome_behavior": "JSON object with at least action (string) and lag_days (number >= 0), or null",
    "incentivized": "bool",
    "consent_training": "bool",
    "battery_version": "non-empty string",
    "is_synthetic": "bool, constant within a table, True only under data/synthetic/",
    "source_row_ref": "non-empty string",
}
"""Human-readable rule per column, used in error messages and validation reports."""

PANDAS_DTYPES: dict[str, str] = {
    "subject_id": "str",
    "dataset": "str",
    "session_id": "str",
    "timestamp": "str",
    "position_in_session": "int64",
    "scenario_id": "str",
    "loss_pct": "float64",
    "horizon_days": "Int64",
    "context_tags": "object",
    "question_order_id": "str",
    "prior_question_ids": "object",
    "elicitation_type": "str",
    "response": "float64",
    "response_time_ms": "float64",
    "covariates": "str",
    "outcome_behavior": "str",
    "incentivized": "bool",
    "consent_training": "bool",
    "battery_version": "str",
    "is_synthetic": "bool",
    "source_row_ref": "str",
}
"""Canonical in-memory dtypes (pandas 3: ``str`` uses NaN for null, ``Int64`` uses NA)."""

ARROW_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("subject_id", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.string(), nullable=True),
        pa.field("position_in_session", pa.int64(), nullable=False),
        pa.field("scenario_id", pa.string(), nullable=False),
        pa.field("loss_pct", pa.float64(), nullable=True),
        pa.field("horizon_days", pa.int64(), nullable=True),
        pa.field("context_tags", pa.list_(pa.string()), nullable=False),
        pa.field("question_order_id", pa.string(), nullable=True),
        pa.field("prior_question_ids", pa.list_(pa.string()), nullable=False),
        pa.field("elicitation_type", pa.string(), nullable=False),
        pa.field("response", pa.float64(), nullable=False),
        pa.field("response_time_ms", pa.float64(), nullable=True),
        pa.field("covariates", pa.string(), nullable=False),
        pa.field("outcome_behavior", pa.string(), nullable=True),
        pa.field("incentivized", pa.bool_(), nullable=False),
        pa.field("consent_training", pa.bool_(), nullable=False),
        pa.field("battery_version", pa.string(), nullable=False),
        pa.field("is_synthetic", pa.bool_(), nullable=False),
        pa.field("source_row_ref", pa.string(), nullable=False),
    ],
    metadata={"schema_version": SCHEMA_VERSION},
)
"""Parquet storage schema; ``context_tags`` and ``prior_question_ids`` are ``list<string>``."""

if [f.name for f in ARROW_SCHEMA] != COLUMNS or set(PANDAS_DTYPES) != set(COLUMNS):
    raise RuntimeError("bre.schema: COLUMNS, ARROW_SCHEMA and PANDAS_DTYPES disagree")


class SchemaError(ValueError):
    """Raised on hard validation errors; carries the :class:`ValidationReport` when there is one."""

    def __init__(self, message: str, report: "ValidationReport | None" = None) -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------------------------
# Scalar rule helpers (shared by the row model and the vectorized frame validator)
# ---------------------------------------------------------------------------------------------


def is_null(value: object) -> bool:
    """Scalar null test: None, NaN, pandas NA/NaT. Lists and arrays are never null."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))
    return False


def _is_bool(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not _is_bool(value)


def _is_integer_valued(value: object) -> bool:
    return _is_number(value) and math.isfinite(float(value)) and float(value).is_integer()


def parse_timestamp(value: object) -> str | None:
    """Return the ISO-8601 string for ``value`` (str, datetime, pandas Timestamp) or None for null.

    Rule: strings must parse with ``datetime.fromisoformat`` (Python 3.11 accepts ``Z`` and
    offsets); they are kept verbatim. Datetime inputs are rendered with ``isoformat()``.
    """
    if is_null(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"timestamp {value!r} is not ISO-8601: {exc}") from exc
        return value
    raise ValueError(f"timestamp must be an ISO-8601 string or datetime; got {type(value).__name__}")


def response_in_range(elicitation_type: str, response: object) -> bool:
    """Response-range rule per elicitation type.

    binary_sell, lottery_choice and binary_yes_no: response in {0, 1}; allocation_pct and
    choice_rate: 0 <= response <= 1; likert: integer-valued with 1 <= response <= 7. Non-numeric
    (including bool and numeric strings) or non-finite values fail.
    """
    if not _is_number(response) or not math.isfinite(float(response)):
        return False
    r = float(response)
    if elicitation_type in BINARY_ELICITATION_TYPES:
        return r in (0.0, 1.0)
    if elicitation_type in RATE_ELICITATION_TYPES:
        return 0.0 <= r <= 1.0
    if elicitation_type == "likert":
        return r.is_integer() and 1.0 <= r <= 7.0
    return False


def question_order_ok(question_order_id: object, battery_version: object) -> bool:
    """Cross-column rule: a row of the shared design (``battery_version == DESIGN_VERSION``) must
    use one of :data:`bre.design.QUESTION_ORDER_IDS`; any string is allowed otherwise; null is
    always allowed (the column is nullable)."""
    if is_null(question_order_id):
        return True
    if not isinstance(question_order_id, str):
        return False
    if battery_version == DESIGN_VERSION:
        return question_order_id in QUESTION_ORDER_IDS
    return True


def _reject_nan_token(token: str) -> None:
    raise ValueError(f"non-finite JSON token {token!r} is not allowed")


def json_loads_strict(text: str | bytes) -> Any:
    """``json.loads`` that rejects the non-standard ``NaN`` / ``Infinity`` tokens."""
    return json.loads(text, parse_constant=_reject_nan_token)


def _load_json_object(value: object, column: str) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (str, bytes)):
        try:
            obj = json_loads_strict(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} is not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{column} must be a JSON object; got {type(obj).__name__}")
        return obj
    raise ValueError(f"{column} must be a JSON string or a dict; got {type(value).__name__}")


def _json_dumps(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_scalar(value: object) -> object:
    """Return ``value`` as a plain JSON scalar (str, int, float, bool, None) or raise ValueError."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number {value!r}")
        return value
    raise ValueError(f"not a JSON scalar: {type(value).__name__}")


def validate_covariates(value: object, allow_extras: bool = True) -> dict:
    """Parse and check a covariates object; returns the cleaned dict.

    Rules (PLAN.md section 2 amendment): the six product keys of :data:`COVARIATE_KEYS` must
    satisfy their :class:`bre.design.CovariateField` (band membership or integer range) and are
    dropped when null (unknown covariate); numeric values are stored as ints. With
    ``allow_extras`` (the ``responses`` rule) two more kinds of key are accepted and kept verbatim:
    ``weight`` (finite number >= 0, or null) and any key starting with ``x_`` (a JSON scalar:
    str, int, finite float, bool or null). ``clients.covariates`` uses ``allow_extras=False``,
    i.e. only the six product keys. Every other key is an error.
    """
    obj = _load_json_object(value, "covariates")
    unknown = sorted(
        k
        for k in obj
        if not (
            k in COVARIATE_KEYS
            or (allow_extras and (k == COVARIATE_WEIGHT_KEY or (isinstance(k, str) and k.startswith(COVARIATE_EXTRA_PREFIX))))
        )
    )
    if unknown:
        allowed = list(COVARIATE_KEYS) + ([COVARIATE_WEIGHT_KEY, COVARIATE_EXTRA_PREFIX + "*"] if allow_extras else [])
        raise ValueError(f"covariates has unknown key(s) {unknown}; allowed {allowed}")
    cleaned: dict[str, Any] = {}
    for key, val in obj.items():
        if key in COVARIATE_KEYS:
            if is_null(val):
                continue
            spec = COVARIATE_SPEC[key]
            if not spec.is_valid(val):
                raise ValueError(f"covariates[{key!r}] = {val!r} violates {spec}")
            cleaned[key] = int(val) if spec.kind == "numeric" else str(val)
        elif key == COVARIATE_WEIGHT_KEY:
            if is_null(val):
                cleaned[key] = None
                continue
            if not _is_number(val) or not math.isfinite(float(val)) or float(val) < 0:
                raise ValueError(f"covariates['weight'] must be a finite number >= 0 or null; got {val!r}")
            cleaned[key] = val.item() if isinstance(val, np.generic) else val
        else:  # x_* keys
            try:
                cleaned[key] = _json_scalar(val)
            except ValueError as exc:
                raise ValueError(f"covariates[{key!r}] must be a JSON scalar; {exc}") from exc
    return cleaned


def validate_outcome_behavior(value: object) -> dict | None:
    """Parse and check an outcome_behavior object; returns the dict or None for null.

    Rule: when present it is a JSON object with at least ``action`` (non-empty string) and
    ``lag_days`` (finite number >= 0); further keys are kept as given.
    """
    if is_null(value):
        return None
    obj = _load_json_object(value, "outcome_behavior")
    missing = [k for k in OUTCOME_REQUIRED_KEYS if k not in obj]
    if missing:
        raise ValueError(f"outcome_behavior is missing required key(s) {missing}")
    if not isinstance(obj["action"], str) or not obj["action"]:
        raise ValueError("outcome_behavior['action'] must be a non-empty string")
    lag = obj["lag_days"]
    if not _is_number(lag) or not math.isfinite(float(lag)) or float(lag) < 0:
        raise ValueError(f"outcome_behavior['lag_days'] must be a number >= 0; got {lag!r}")
    return obj


def validate_string_list(value: object, column: str, *, tags: bool = False) -> list[str]:
    """Check an ordered list cell and return it as a Python list of str.

    Rules: the value must be a list, tuple or 1-d numpy array (order is data: a set, dict, string,
    bytes or generator is rejected, nothing is coerced with ``str()``); every item must be a
    non-empty str; with ``tags`` every item must ``fullmatch`` :data:`TAG_PATTERN`.
    """
    if is_null(value):
        raise ValueError(f"{column} must be a list (empty allowed), not null")
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{column} must be a 1-d array of strings; got ndim={value.ndim}")
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{column} must be a list of strings; got {type(value).__name__}")
    out: list[str] = []
    for item in value:
        if isinstance(item, np.str_):
            item = str(item)
        if not isinstance(item, str) or not item:
            raise ValueError(f"{column} entries must be non-empty strings; got {item!r}")
        if tags and not TAG_PATTERN.fullmatch(item):
            raise ValueError(f"{column} entry {item!r} is not a 'namespace:value' tag")
        out.append(item)
    return out


def _check_str_list(value: object, column: str, tags: bool) -> list[str]:
    return validate_string_list(value, column, tags=tags)


# ---------------------------------------------------------------------------------------------
# Row model
# ---------------------------------------------------------------------------------------------


class DecisionEvent(BaseModel):
    """One elicited decision. Field rules are those of :data:`COLUMN_RULES`.

    ``covariates`` and ``outcome_behavior`` are stored as canonical JSON strings (sorted keys,
    compact separators) but dict inputs are accepted and serialized. ``context_tags`` and
    ``prior_question_ids`` accept any sequence of strings and are stored as lists.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    subject_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    timestamp: str | None = None
    position_in_session: int = Field(ge=0)
    scenario_id: str = Field(min_length=1)
    loss_pct: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    horizon_days: int | None = Field(default=None, ge=0)
    context_tags: list[str] = Field(default_factory=list)
    question_order_id: str | None = None
    prior_question_ids: list[str] = Field(default_factory=list)
    elicitation_type: ElicitationType
    response: float = Field(allow_inf_nan=False)
    response_time_ms: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    covariates: str
    outcome_behavior: str | None = None
    incentivized: bool
    consent_training: bool
    battery_version: str = Field(min_length=1)
    is_synthetic: bool
    source_row_ref: str = Field(min_length=1)

    # -- before-validators: normalize frame-native values (NaN, numpy scalars, arrays) ----------

    @field_validator("timestamp", mode="before")
    @classmethod
    def _v_timestamp(cls, v: object) -> str | None:
        return parse_timestamp(v)

    # Null handling: NaN/NA/None become None here; whether None is allowed is then decided by
    # the field annotation (``int`` rejects it, ``int | None`` accepts it).

    @field_validator("position_in_session", "horizon_days", mode="before")
    @classmethod
    def _v_integer(cls, v: object) -> object:
        if is_null(v):
            return None
        if _is_bool(v):
            raise ValueError("must be an integer, not a bool")
        if isinstance(v, (str, bytes)):
            raise ValueError(f"must be an integer, not a string ({v!r})")
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float):
            if not v.is_integer():
                raise ValueError(f"must be integer-valued; got {v}")
            return int(v)
        return v

    @field_validator("loss_pct", "response", "response_time_ms", mode="before")
    @classmethod
    def _v_number(cls, v: object) -> object:
        if is_null(v):
            return None
        if _is_bool(v):
            raise ValueError("must be a number, not a bool")
        if isinstance(v, (str, bytes)):
            raise ValueError(f"must be a number, not a string ({v!r})")
        return v.item() if isinstance(v, np.generic) else v

    @field_validator("incentivized", "consent_training", "is_synthetic", mode="before")
    @classmethod
    def _v_bool(cls, v: object) -> bool:
        if not _is_bool(v):
            raise ValueError(f"must be a bool; got {v!r}")
        return bool(v)

    @field_validator("context_tags", mode="before")
    @classmethod
    def _v_context_tags(cls, v: object) -> list[str]:
        return _check_str_list(v, "context_tags", tags=True)

    @field_validator("prior_question_ids", mode="before")
    @classmethod
    def _v_prior_question_ids(cls, v: object) -> list[str]:
        return _check_str_list(v, "prior_question_ids", tags=False)

    @field_validator("covariates", mode="before")
    @classmethod
    def _v_covariates(cls, v: object) -> str:
        return _json_dumps(validate_covariates(v, allow_extras=True))

    @field_validator("outcome_behavior", mode="before")
    @classmethod
    def _v_outcome_behavior(cls, v: object) -> str | None:
        obj = validate_outcome_behavior(v)
        return None if obj is None else _json_dumps(obj)

    @field_validator(
        "subject_id",
        "dataset",
        "session_id",
        "scenario_id",
        "battery_version",
        "source_row_ref",
        "question_order_id",
        "elicitation_type",
        mode="before",
    )
    @classmethod
    def _v_string(cls, v: object) -> object:
        if is_null(v):
            return None
        if isinstance(v, np.str_):
            return str(v)
        if not isinstance(v, str):
            raise ValueError(f"must be a string; got {type(v).__name__}")
        return v

    # -- cross-field rule ------------------------------------------------------------------------

    @model_validator(mode="after")
    def _v_response_range(self) -> "DecisionEvent":
        if not response_in_range(self.elicitation_type, self.response):
            raise ValueError(
                f"response {self.response!r} is out of range for elicitation_type "
                f"{self.elicitation_type!r} ({COLUMN_RULES['response']})"
            )
        if not question_order_ok(self.question_order_id, self.battery_version):
            raise ValueError(
                f"question_order_id {self.question_order_id!r} is not one of {QUESTION_ORDER_IDS} "
                f"although battery_version is the shared design {DESIGN_VERSION!r}"
            )
        return self

    # -- conveniences ----------------------------------------------------------------------------

    @property
    def key(self) -> tuple[str, str, str, int]:
        """Uniqueness key ``(dataset, subject_id, session_id, position_in_session)``."""
        return (self.dataset, self.subject_id, self.session_id, self.position_in_session)

    @property
    def covariates_dict(self) -> dict:
        return json.loads(self.covariates)

    @property
    def outcome_behavior_dict(self) -> dict | None:
        return None if self.outcome_behavior is None else json.loads(self.outcome_behavior)

    def to_row(self) -> dict[str, Any]:
        """Column-ordered dict for a DataFrame row."""
        data = self.model_dump()
        return {c: data[c] for c in COLUMNS}


# ---------------------------------------------------------------------------------------------
# Frame constructors and normalization
# ---------------------------------------------------------------------------------------------


def empty_frame() -> pd.DataFrame:
    """An empty DataFrame with :data:`COLUMNS` in order and :data:`PANDAS_DTYPES`."""
    return pd.DataFrame({c: pd.Series([], dtype=PANDAS_DTYPES[c]) for c in COLUMNS})


def _to_list_column(series: pd.Series) -> pd.Series:
    def conv(v: object) -> object:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return list(v)
        return v

    return series.map(conv).astype("object")


def _to_json_column(series: pd.Series) -> pd.Series:
    def conv(v: object) -> object:
        if isinstance(v, dict):
            return _json_dumps(v)
        return np.nan if is_null(v) else v

    return series.map(conv).astype("str")


def conform_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with :data:`COLUMNS` in order and canonical dtypes.

    Rules: all columns must be present (extra columns raise); lists/arrays become Python lists;
    dicts in the JSON columns are serialized; nullable integers become ``Int64``; bool columns
    must already hold only bools (they are never coerced, because ``astype(bool)`` would silently
    turn null into False). Validation is not performed here; use :func:`validate_frame`.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in COLUMNS]
    if missing or extra:
        raise SchemaError(f"cannot conform frame: missing columns {missing}, extra columns {extra}")
    out = pd.DataFrame(index=df.index)
    for c in COLUMNS:
        s = df[c]
        if c in LIST_COLUMNS:
            out[c] = _to_list_column(s)
        elif c in JSON_COLUMNS:
            out[c] = _to_json_column(s)
        elif c in BOOL_COLUMNS:
            if not s.map(_is_bool).all():
                raise SchemaError(f"cannot conform frame: {c} holds non-bool values")
            out[c] = s.astype("bool")
        elif PANDAS_DTYPES[c] == "str":
            out[c] = s.map(lambda v: np.nan if is_null(v) else v).astype("str")
        elif PANDAS_DTYPES[c] == "Int64":
            out[c] = pd.to_numeric(s, errors="raise").astype("Int64")
        else:
            out[c] = s.astype(PANDAS_DTYPES[c])
    return out


def events_to_frame(events: list[DecisionEvent]) -> pd.DataFrame:
    """Build a canonical DataFrame from validated events (empty list -> :func:`empty_frame`)."""
    if not events:
        return empty_frame()
    return conform_frame(pd.DataFrame([e.to_row() for e in events], columns=COLUMNS))


def frame_to_events(df: pd.DataFrame) -> list[DecisionEvent]:
    """Validate every row through :class:`DecisionEvent`; raises :class:`SchemaError` on the
    first invalid row with its index in the message.

    Like :func:`validate_frame`, missing and unexpected columns are hard errors, and numeric
    strings are not numbers.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in COLUMNS]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing columns {missing}")
        if extra:
            parts.append(f"unexpected columns {extra}")
        raise SchemaError("; ".join(parts))
    events = []
    for idx, row in zip(df.index, df[COLUMNS].itertuples(index=False, name=None)):
        try:
            events.append(DecisionEvent(**dict(zip(COLUMNS, row))))
        except ValidationError as exc:
            raise SchemaError(f"row {idx!r}: {exc}") from exc
    return events


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Result of :func:`validate_frame`; ``ok`` is True iff ``errors`` is empty."""

    n_rows: int = 0
    nulls_per_column: dict[str, int] = field(default_factory=dict)
    range_violations: dict[str, int] = field(default_factory=dict)
    duplicate_keys: int = 0
    mixed_synthetic: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_markdown(self) -> str:
        lines = [
            "# DecisionEvent validation report",
            "",
            f"- schema: {SCHEMA_VERSION}",
            f"- rows: {self.n_rows}",
            f"- ok: {self.ok}",
            f"- duplicate keys ({', '.join(KEY_COLUMNS)}): {self.duplicate_keys}",
            f"- mixed is_synthetic: {self.mixed_synthetic}",
            "",
            "## Nulls and rule violations per column",
            "",
            "| column | rule | nulls | violations |",
            "|---|---|---:|---:|",
        ]
        for c in COLUMNS:
            lines.append(
                f"| {c} | {COLUMN_RULES[c]} | {self.nulls_per_column.get(c, 0)} | "
                f"{self.range_violations.get(c, 0)} |"
            )
        lines += ["", "## Errors", ""]
        lines += [f"- {e}" for e in self.errors] or ["- none"]
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in self.warnings] or ["- none"]
        return "\n".join(lines) + "\n"


def _violations(series: pd.Series, predicate) -> int:
    """Count non-null values for which ``predicate`` is False."""
    values = series[~series.isna()]
    if values.empty:
        return 0
    return int((~values.map(lambda v: bool(predicate(v)))).sum())


def _nonempty_str(v: object) -> bool:
    return isinstance(v, str) and len(v) > 0


def _safe(fn):
    """Wrap a raising checker into a bool predicate."""

    def pred(v: object) -> bool:
        try:
            fn(v)
            return True
        except ValueError:
            return False

    return pred


def _response_violations(df: pd.DataFrame) -> int:
    et = df["elicitation_type"]
    resp = df["response"]
    bad = 0
    for t, r in zip(et.tolist(), resp.tolist()):
        if is_null(r):
            continue  # counted as a null, not a range violation
        if not isinstance(t, str) or t not in ELICITATION_TYPES:
            continue  # elicitation_type violation is reported on its own column
        if not response_in_range(t, r):
            bad += 1
    return bad


def validate_frame(
    df: pd.DataFrame, expect_synthetic: bool | None = None, strict: bool = True
) -> ValidationReport:
    """Validate a DecisionEvent frame against every rule in :data:`COLUMN_RULES`.

    Hard errors (``report.errors``): missing or extra columns; nulls in non-nullable columns; any
    rule violation (counted per column in ``range_violations``); duplicate
    ``(dataset, subject_id, session_id, position_in_session)`` keys; a table mixing
    ``is_synthetic`` values; ``is_synthetic`` disagreeing with ``expect_synthetic`` when given.
    Warnings: empty frame, non-canonical column order, rows without training consent.
    With ``strict`` a failing report raises :class:`SchemaError` (``exc.report`` holds it).
    The input frame is never modified.
    """
    report = ValidationReport(n_rows=int(len(df)))
    errors, warnings = report.errors, report.warnings

    missing = [c for c in COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in COLUMNS]
    if missing:
        errors.append(f"missing columns {missing}")
    if extra:
        errors.append(f"unexpected columns {extra}")
    if missing:
        if strict:
            raise SchemaError("; ".join(errors), report)
        return report
    if not extra and list(df.columns) != COLUMNS:
        warnings.append("columns are not in canonical order; write_events reorders them")
    if len(df) == 0:
        warnings.append("frame has no rows")

    for c in COLUMNS:
        k = int(df[c].isna().sum())
        report.nulls_per_column[c] = k
        if k and c not in NULLABLE_COLUMNS:
            errors.append(f"{c}: {k} null value(s) in a non-nullable column")

    checks = {
        "subject_id": _nonempty_str,
        "dataset": _nonempty_str,
        "session_id": _nonempty_str,
        "timestamp": _safe(parse_timestamp),
        "position_in_session": lambda v: _is_integer_valued(v) and float(v) >= 0,
        "scenario_id": _nonempty_str,
        "loss_pct": lambda v: _is_number(v) and -1.0 <= float(v) <= 1.0,
        "horizon_days": lambda v: _is_integer_valued(v) and float(v) >= 0,
        "context_tags": _safe(lambda v: _check_str_list(v, "context_tags", tags=True)),
        "question_order_id": lambda v: isinstance(v, str),
        "prior_question_ids": _safe(lambda v: _check_str_list(v, "prior_question_ids", tags=False)),
        "elicitation_type": lambda v: isinstance(v, str) and v in ELICITATION_TYPES,
        "response_time_ms": lambda v: _is_number(v) and math.isfinite(float(v)) and float(v) >= 0,
        "covariates": _safe(lambda v: validate_covariates(v, allow_extras=True)),
        "outcome_behavior": _safe(validate_outcome_behavior),
        "incentivized": _is_bool,
        "consent_training": _is_bool,
        "battery_version": _nonempty_str,
        "is_synthetic": _is_bool,
        "source_row_ref": _nonempty_str,
    }
    for c, pred in checks.items():
        n_bad = _violations(df[c], pred)
        report.range_violations[c] = n_bad
        if n_bad:
            errors.append(f"{c}: {n_bad} value(s) violate rule '{COLUMN_RULES[c]}'")
    n_bad = _response_violations(df)
    report.range_violations["response"] = n_bad
    if n_bad:
        errors.append(f"response: {n_bad} value(s) violate rule '{COLUMN_RULES['response']}'")
    # cross-column rule: design rows use the design's question-order ids (string cells only; a
    # non-string cell is already counted above)
    n_bad = sum(
        1
        for q, b in zip(df["question_order_id"].tolist(), df["battery_version"].tolist())
        if isinstance(q, str) and not question_order_ok(q, b)
    )
    if n_bad:
        report.range_violations["question_order_id"] += n_bad
        errors.append(f"question_order_id: {n_bad} value(s) violate rule '{COLUMN_RULES['question_order_id']}'")
    report.range_violations = {c: report.range_violations[c] for c in COLUMNS}

    key_ok = all(report.nulls_per_column[c] == 0 for c in KEY_COLUMNS)
    if key_ok and len(df):
        report.duplicate_keys = int(df.duplicated(subset=KEY_COLUMNS).sum())
        if report.duplicate_keys:
            errors.append(f"{report.duplicate_keys} duplicate key(s) on {KEY_COLUMNS}")

    syn = df["is_synthetic"]
    if len(df) and report.nulls_per_column["is_synthetic"] == 0 and report.range_violations["is_synthetic"] == 0:
        values = {bool(v) for v in syn.tolist()}
        report.mixed_synthetic = len(values) > 1
        if report.mixed_synthetic:
            errors.append("is_synthetic mixes True and False within one table")
        if expect_synthetic is not None and values != {bool(expect_synthetic)}:
            errors.append(
                f"is_synthetic is {sorted(values)} but the destination requires {bool(expect_synthetic)}"
            )

    if len(df) and report.range_violations["consent_training"] == 0:
        n_noconsent = int((~df["consent_training"].map(bool)).sum())
        if n_noconsent:
            warnings.append(f"{n_noconsent} row(s) have consent_training=False; exclude them from fits")

    if strict and not report.ok:
        raise SchemaError("; ".join(report.errors), report)
    return report


# ---------------------------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------------------------


def is_synthetic_path(path: str | Path) -> bool:
    """True iff the absolute path has a consecutive ``data/synthetic`` directory pair among its parents.

    Rule (CLAUDE.md rule 3): synthetic tables live only under ``data/synthetic/``. The path is
    made absolute with :func:`os.path.abspath` (relative to the current directory, symlinks are
    not resolved, so a symlinked ``data/synthetic`` still counts as synthetic). The rule is
    location-based, not project-based, so temporary directories in tests follow it too.
    """
    parts = Path(os.path.abspath(os.path.expanduser(str(path)))).parts
    return any(parts[i] == "data" and parts[i + 1] == "synthetic" for i in range(len(parts) - 1))


def validation_report_path(path: str | Path) -> Path:
    """``foo.parquet`` -> ``foo.validation.md`` in the same directory."""
    p = Path(path)
    return p.with_name(p.stem + ".validation.md")


def frame_to_table(df: pd.DataFrame) -> pa.Table:
    """Conform ``df`` and convert it to an Arrow table with :data:`ARROW_SCHEMA`."""
    return pa.Table.from_pandas(conform_frame(df), schema=ARROW_SCHEMA, preserve_index=False)


def table_to_frame(table: pa.Table) -> pd.DataFrame:
    """Convert an Arrow table read with :data:`ARROW_SCHEMA` to the canonical frame."""
    return conform_frame(table.to_pandas())


def write_events(df: pd.DataFrame, path: str | Path) -> ValidationReport:
    """Validate ``df`` and write it as parquet with :data:`ARROW_SCHEMA` plus a validation report.

    Rules: the destination decides the expected ``is_synthetic`` value (True iff
    :func:`is_synthetic_path`); validation is strict, so nothing is written for an invalid frame;
    the sibling ``<name>.validation.md`` records the report, the row count, the file SHA-256 and
    the write time. Returns the report.
    """
    path = Path(path)
    if path.suffix != ".parquet":
        raise SchemaError(f"events are written as .parquet files; got {path.name!r}")
    expect = is_synthetic_path(path)
    report = validate_frame(df, expect_synthetic=expect, strict=True)
    table = frame_to_table(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    written = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = (
        f"<!-- generated by bre.schema.write_events -->\n\n"
        f"- file: {path.name}\n- written: {written}\n- sha256: {digest}\n"
        f"- is_synthetic: {expect}\n\n"
    )
    validation_report_path(path).write_text(header + report.to_markdown(), encoding="utf-8")
    return report


def read_events(path: str | Path) -> pd.DataFrame:
    """Read a parquet table written by :func:`write_events` and validate it against its location.

    The file's Arrow schema must match :data:`ARROW_SCHEMA` (field names and types); list columns
    come back as Python lists and dtypes are canonical (:data:`PANDAS_DTYPES`).
    """
    path = Path(path)
    table = pq.read_table(path)
    names = table.schema.names
    if names != COLUMNS:
        raise SchemaError(f"{path.name}: columns {names} do not match the schema {COLUMNS}")
    try:
        table = table.cast(ARROW_SCHEMA)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise SchemaError(f"{path.name}: cannot cast to ARROW_SCHEMA: {exc}") from exc
    df = table_to_frame(table)
    validate_frame(df, expect_synthetic=is_synthetic_path(path), strict=True)
    return df
