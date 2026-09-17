"""Helpers shared by the loaders: default directories, frame assembly, age bands.

Rules common to every external dataset (build prompt / CATALOG.md): ``is_synthetic = False``,
``consent_training = False`` (no training consent was collected by the original studies; the
flag is a property of the BRE intake battery), ``battery_version = "external"``,
``outcome_behavior = null``, ``timestamp = null`` unless the source records one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from bre.schema import COLUMNS, DATA_DIR, conform_frame, empty_frame

RAW_DIR = DATA_DIR / "raw"
"""``bre/data/raw`` (absolute, anchored on the package location so any working directory works)."""
PROCESSED_DIR = DATA_DIR / "processed"
"""``bre/data/processed``; every loader's parquet output goes here."""

EXTERNAL_BATTERY_VERSION = "external"
"""``battery_version`` of every row that did not come from the BRE intake battery."""

_DEFAULTS: dict[str, Any] = {
    "timestamp": None,
    "loss_pct": np.nan,
    "horizon_days": pd.NA,
    "question_order_id": None,
    "response_time_ms": np.nan,
    "outcome_behavior": None,
    "consent_training": False,
    "battery_version": EXTERNAL_BATTERY_VERSION,
    "is_synthetic": False,
}


def assemble(columns: Mapping[str, Any], n_rows: int) -> pd.DataFrame:
    """Build a canonical DecisionEvent frame from per-column sequences.

    ``columns`` maps schema column names to sequences of length ``n_rows`` (or scalars, which are
    broadcast). Columns absent from ``columns`` take the external-dataset defaults above;
    ``context_tags`` and ``prior_question_ids`` default to empty lists. Dict cells in
    ``covariates`` are serialized by :func:`bre.schema.conform_frame`. Nothing is validated here
    (the writer validates); an empty input returns :func:`bre.schema.empty_frame`.
    """
    unknown = sorted(set(columns) - set(COLUMNS))
    if unknown:
        raise KeyError(f"assemble: unknown schema column(s) {unknown}")
    if n_rows == 0:
        return empty_frame()
    data: dict[str, Any] = {}
    for c in COLUMNS:
        if c in columns:
            v = columns[c]
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)) and not isinstance(v, str):
                if len(v) != n_rows:
                    raise ValueError(f"assemble: column {c!r} has {len(v)} values, expected {n_rows}")
                data[c] = list(v) if c in ("context_tags", "prior_question_ids") else v
            else:
                data[c] = [v] * n_rows
        elif c in ("context_tags", "prior_question_ids"):
            data[c] = [[] for _ in range(n_rows)]
        elif c in _DEFAULTS:
            data[c] = [_DEFAULTS[c]] * n_rows
        else:
            raise KeyError(f"assemble: required column {c!r} missing")
    df = pd.DataFrame({c: pd.Series(data[c], dtype="object") for c in COLUMNS})
    return conform_frame(df)


def age_band(age: object) -> str | None:
    """Map an age in years onto the PLAN.md section 3 bands; None for null or < 18."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return None
    a = int(age)
    if a < 18:
        return None
    if a <= 29:
        return "18-29"
    if a <= 44:
        return "30-44"
    if a <= 59:
        return "45-59"
    return "60+"


def json_scalar(v: object) -> object:
    """numpy scalars -> Python scalars so covariate dicts serialize."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def dumps(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Row count, subject count and elicitation types of a frame (for loader meta)."""
    return {
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if len(df) else 0,
        "elicitation_types": sorted(df["elicitation_type"].dropna().unique().tolist()) if len(df) else [],
    }
