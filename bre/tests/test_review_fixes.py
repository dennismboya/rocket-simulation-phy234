"""Regression tests for the Phase 0 adversarial review (one test per finding, numbered as in the
review; finding 19, committing the scaffold, is the main session's job and has no test here).

The contract decisions behind them are PLAN.md section 2 "Amendment 2026-09-17": two new
elicitation types, ``weight`` / ``x_*`` covariate keys on responses only, ``namespace:value`` tags
checked with ``fullmatch``, hyphenated question-order ids validated for design rows, the
real-vs-synthetic location rule on every insert path with ``os.path.abspath`` and absolute SQLite
paths, and the DB layer reusing the ``bre.schema`` validators.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import db
from bre import design as D
from bre import schema as S
from db import Client, Response, SyntheticPolicyError, frame_to_responses, get_engine, init_db, responses_to_frame, session_scope
from db.models import encode_string_list, parse_json_object, parse_json_string_list, validate_response_range
from db.session import absolute_sqlite_url, sqlite_file_path, write_audit

BRE_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def row(**overrides) -> dict:
    """A valid real (is_synthetic=False) design row as a plain dict in schema order."""
    base = dict(
        subject_id="s1",
        dataset="unit",
        session_id="sess1",
        timestamp="2026-09-17T10:00:00+00:00",
        position_in_session=0,
        scenario_id="L10|rec>friend",
        loss_pct=-0.10,
        horizon_days=365,
        context_tags=["news:recession", "social:friend_sells"],
        question_order_id=D.SCENARIO_FIRST,
        prior_question_ids=[],
        elicitation_type="binary_sell",
        response=1.0,
        response_time_ms=1234.5,
        covariates=json.dumps({"age_band": "30-44", "self_reported_risk_tolerance": 4}),
        outcome_behavior=json.dumps({"action": "sold", "lag_days": 3}),
        incentivized=False,
        consent_training=True,
        battery_version=D.DESIGN_VERSION,
        is_synthetic=False,
        source_row_ref="unit:0",
    )
    base.update(overrides)
    return base


def frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=S.COLUMNS)


def frame_with_cell(column: str, value) -> pd.DataFrame:
    """A one-row frame whose ``column`` cell is exactly ``value`` (lists, sets, arrays included)."""
    r = row()
    r[column] = value
    return pd.DataFrame({c: pd.Series([r[c]], dtype="object") for c in S.COLUMNS})


def orm_kwargs(event: dict) -> dict:
    kw = dict(event)
    kw["context_tags_json"] = json.dumps(kw.pop("context_tags"))
    kw["prior_question_ids_json"] = json.dumps(kw.pop("prior_question_ids"))
    return kw


def event_from_row(**overrides) -> S.DecisionEvent:
    return S.DecisionEvent(**row(**overrides))


@pytest.fixture()
def memory_engine():
    eng = get_engine("sqlite://")
    init_db(eng)
    return eng


def _count(engine, table: str = "responses") -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


# ---------------------------------------------------------------------------------------------
# 0 / 18: the catalogued mappings (data/CATALOG.md A-D) and the intake battery fit the contract
# ---------------------------------------------------------------------------------------------

CATALOGUED_ROWS = {
    "choices13k": dict(
        dataset="choices13k",
        subject_id="agg:1:1:1",
        session_id="agg",
        scenario_id="1",
        loss_pct=None,
        horizon_days=None,
        context_tags=["feedback:on", "block:1"],
        question_order_id=None,
        elicitation_type="choice_rate",
        response=0.37,
        covariates={"weight": 20, "x_worst_outcome_rel": -0.4},
        outcome_behavior=None,
        incentivized=True,
        battery_version="choices13k",
        source_row_ref="row:0",
    ),
    "cpc18": dict(
        dataset="cpc18",
        subject_id="10101",
        session_id="set1",
        scenario_id="G17",
        context_tags=["exp:loss", "feedback:on", "block:2"],
        question_order_id=None,
        prior_question_ids=["G12", "G3"],
        elicitation_type="lottery_choice",
        response=1.0,
        covariates={"age_band": "18-29", "x_gender": "F", "x_location": "Technion", "x_chose_b": False},
        battery_version="cpc18",
    ),
    "psych201": dict(
        dataset="psych201:spektor2024lossaversion",
        subject_id="p7",
        session_id="block1",
        scenario_id="trial3",
        context_tags=["domain:gain"],
        question_order_id=None,
        elicitation_type="lottery_choice",
        response=0.0,
        covariates={},
        battery_version="psych201",
    ),
    "order_effects": dict(
        dataset="order_effects:clinton_gore",
        subject_id="agg",
        session_id="AyBn",
        scenario_id="AyBn",
        loss_pct=None,
        horizon_days=None,
        context_tags=["order:A_first"],
        question_order_id="A_first",
        elicitation_type="choice_rate",
        response=0.25,
        covariates={"weight": None},
        battery_version="ozawa_khrennikov_2021",
    ),
    "intake_tolerance": dict(
        dataset="intake_battery",
        scenario_id="tolerance",
        loss_pct=None,
        horizon_days=None,
        context_tags=[],
        question_order_id=D.TOLERANCE_FIRST,
        elicitation_type="binary_yes_no",
        response=1.0,
        covariates={"age_band": "45-59", "education": "graduate"},
        battery_version=D.DESIGN_VERSION,
    ),
}


@pytest.mark.parametrize("name", sorted(CATALOGUED_ROWS))
def test_finding_0_catalogued_mappings_pass_schema_and_db(name, memory_engine):
    """Every CATALOG.md schema mapping builds a DecisionEvent, validates as a frame, inserts into
    the DB and validates again after responses_to_frame (parquet side == SQLite side)."""
    ev = event_from_row(**CATALOGUED_ROWS[name])
    assert ev.elicitation_type in S.ELICITATION_TYPES
    df = S.events_to_frame([ev])
    assert S.validate_frame(df, expect_synthetic=False).ok
    with session_scope(memory_engine) as s:
        assert frame_to_responses(s, df) == 1
        back = responses_to_frame(s)
    assert S.validate_frame(back, expect_synthetic=False).ok
    assert json.loads(back["covariates"].iloc[0]) == ev.covariates_dict
    assert back["elicitation_type"].iloc[0] == ev.elicitation_type


def test_finding_0_new_elicitation_types_have_ranges_on_both_sides():
    assert "binary_yes_no" in S.ELICITATION_TYPES and "choice_rate" in S.ELICITATION_TYPES
    assert tuple(db.ELICITATION_TYPES) == tuple(S.ELICITATION_TYPES)
    assert S.response_in_range("binary_yes_no", 1) and not S.response_in_range("binary_yes_no", 0.5)
    assert S.response_in_range("choice_rate", 0.37) and not S.response_in_range("choice_rate", 1.5)
    assert validate_response_range("choice_rate", 0.37) == 0.37
    with pytest.raises(ValueError):
        validate_response_range("binary_yes_no", 0.5)
    with pytest.raises(ValidationError, match="out of range"):
        event_from_row(elicitation_type="binary_yes_no", response=0.5)
    with pytest.raises(ValueError):
        Response(**orm_kwargs(row(elicitation_type="choice_rate", response=2.0)))


def test_finding_18_covariate_extras_rule_is_shared_by_schema_and_db():
    """weight and x_* keys: accepted by responses on both sides, rejected by clients on both."""
    extras = {"age_band": "18-29", "weight": 42, "x_worst_outcome_rel": -0.4, "x_note": None}
    assert S.validate_covariates(extras, allow_extras=True) == extras
    assert S.validate_frame(frame(row(covariates=json.dumps(extras)))).ok
    Response(**orm_kwargs(row(covariates=json.dumps(extras))))  # accepted
    for bad in ({"weight": -1}, {"weight": "20"}, {"weight": True}, {"x_a": [1]}, {"x_a": {"b": 1}}, {"gender": "f"}):
        with pytest.raises(ValueError):
            S.validate_covariates(bad, allow_extras=True)
        with pytest.raises(ValueError):
            Response(**orm_kwargs(row(covariates=json.dumps(bad))))
    for client_bad in ({"weight": 42}, {"x_gender": "F"}, {"gender": "f"}):
        with pytest.raises(ValueError, match="unknown key"):
            S.validate_covariates(client_bad, allow_extras=False)
        with pytest.raises(ValueError, match="unknown key"):
            Client(client_id="c", covariates=json.dumps(client_bad))
    Client(client_id="c", covariates=json.dumps({"age_band": "18-29", "invest_experience_yrs": 3}))
    assert set(S.COVARIATE_KEYS) == set(db.COVARIATE_KEYS) and len(S.COVARIATE_KEYS) == 6


# ---------------------------------------------------------------------------------------------
# 1: no int() truncation of position_in_session / horizon_days
# ---------------------------------------------------------------------------------------------


def test_finding_1_non_integer_positions_are_rejected_not_truncated(memory_engine):
    df = frame(
        row(position_in_session=0.9, horizon_days=365.9),
        row(position_in_session=1.2, horizon_days=100.5, source_row_ref="unit:1"),
    )
    with session_scope(memory_engine) as s:
        with pytest.raises(S.SchemaError) as excinfo:
            frame_to_responses(s, df)
    assert excinfo.value.report.range_violations["position_in_session"] == 2
    assert excinfo.value.report.range_violations["horizon_days"] == 2
    assert _count(memory_engine) == 0
    for bad in (1.2, True, "3", float("nan")):
        with pytest.raises(ValueError):
            Response(**orm_kwargs(row(position_in_session=bad)))
    assert Response(**orm_kwargs(row(position_in_session=np.int64(3), horizon_days=365.0))).horizon_days == 365


# ---------------------------------------------------------------------------------------------
# 2 / 16: the ORM enforces exactly the schema's rules; nothing weaker, no coercion
# ---------------------------------------------------------------------------------------------

WEAKER_THAN_SCHEMA = [
    {"subject_id": ""},
    {"subject_id": 123},
    {"battery_version": ""},
    {"battery_version": True},
    {"scenario_id": 3.5},
    {"loss_pct": "0.5"},
    {"response": "1"},
    {"context_tags": ["recession", ""]},
    {"context_tags": [""]},
    {"context_tags": ["recession"]},
    {"covariates": json.dumps({"age_band": "child", "gender": "f"})},
    {"covariates": json.dumps({"full_name": "x", "email": "y", "phone": "z"})},
    {"outcome_behavior": json.dumps({"action": "", "lag_days": -5})},
    {"outcome_behavior": json.dumps({"action": 7, "lag_days": 1})},
    {"is_synthetic": 1},
    {"incentivized": 0},
    {"consent_training": "yes"},
    {"position_in_session": True},
    {"horizon_days": True},
    {"timestamp": "yesterday"},
    {"elicitation_type": "slider"},
]


@pytest.mark.parametrize("override", WEAKER_THAN_SCHEMA, ids=[next(iter(o)) + "=" + repr(o[next(iter(o))]) for o in WEAKER_THAN_SCHEMA])
def test_finding_2_and_16_db_rejects_what_the_schema_rejects(override, memory_engine):
    df = frame(row(**override))
    assert not S.validate_frame(df, strict=False).ok
    with session_scope(memory_engine) as s:
        with pytest.raises(ValueError):
            frame_to_responses(s, df)
    assert _count(memory_engine) == 0
    with pytest.raises(ValueError):
        Response(**orm_kwargs(row(**override)))


def test_finding_2_every_row_the_orm_accepts_validates_after_export(memory_engine):
    variants = [
        row(),
        row(source_row_ref="unit:1", position_in_session=1, timestamp=None, loss_pct=None, horizon_days=None,
            question_order_id=None, response_time_ms=None, outcome_behavior=None, context_tags=[], covariates="{}"),
        row(source_row_ref="unit:2", position_in_session=2, elicitation_type="likert", response=4,
            battery_version="ext", question_order_id="whatever"),
        row(source_row_ref="unit:3", position_in_session=3, elicitation_type="choice_rate", response=0.5,
            covariates=json.dumps({"weight": 12, "x_k": "v"})),
    ]
    with session_scope(memory_engine) as s:
        for v in variants:
            s.add(Response(**orm_kwargs(v)))
        s.flush()
        back = responses_to_frame(s)
    assert len(back) == 4
    assert S.validate_frame(back, expect_synthetic=False).ok


# ---------------------------------------------------------------------------------------------
# 3 / 15: list cells are never coerced, unordered inputs are rejected
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [[1, None], {"a": 1, "b": 2}, b"ab", "news:recession", '["news:recession"]', {"b:1", "a:2", "c:3"},
     [1, 2], [1, None, float("nan")], ["news:recession", ""], (x for x in ["news:recession"]), np.array([["a:b"]])],
    ids=lambda v: type(v).__name__ + ":" + repr(v)[:24],
)
def test_finding_3_and_15_encode_string_list_rejects_instead_of_coercing(value, memory_engine):
    with pytest.raises(ValueError):
        encode_string_list(value, "context_tags")
    with pytest.raises(ValueError):
        Response(**{**orm_kwargs(row()), "context_tags_json": None, "context_tags": value})
    if not isinstance(value, (str, bytes)):
        df = frame_with_cell("context_tags", value)
        assert type(df["context_tags"].iloc[0]) is type(value)
        with session_scope(memory_engine) as s:
            with pytest.raises(ValueError):
                frame_to_responses(s, df)
    assert _count(memory_engine) == 0


def test_finding_15_order_is_data_and_tags_are_namespaced():
    tags = ["social:friend_sells", "news:recession"]
    assert json.loads(encode_string_list(tags, "context_tags")) == tags
    assert json.loads(encode_string_list(tuple(tags), "context_tags")) == tags
    assert json.loads(encode_string_list(np.array(tags), "context_tags")) == tags
    assert json.loads(encode_string_list([], "prior_question_ids")) == []
    assert json.loads(encode_string_list(["q1", "L05|none"], "prior_question_ids")) == ["q1", "L05|none"]
    with pytest.raises(ValueError, match="namespace:value"):
        encode_string_list(["recession"], "context_tags")
    with pytest.raises(ValueError, match="namespace:value"):
        parse_json_string_list('["recession"]', "context_tags", tags=True)
    with pytest.raises(ValueError):
        Response(**orm_kwargs(row(context_tags=["recession"])))


# ---------------------------------------------------------------------------------------------
# 4 / 12: the policy is enforced on every insert path and for Connection-bound sessions
# ---------------------------------------------------------------------------------------------


def test_finding_4_connection_bound_session_is_judged_by_its_engine(tmp_path):
    eng = get_engine(f"sqlite:///{tmp_path / 'not_synthetic_dir' / 'leak.db'}")
    init_db(eng)
    synthetic = frame(row(is_synthetic=True))
    with eng.connect() as conn:
        with Session(bind=conn) as s:
            with pytest.raises(SyntheticPolicyError, match="synthetic responses"):
                frame_to_responses(s, synthetic)
            s.rollback()
        with Session(bind=conn) as s:
            s.add(Response(**orm_kwargs(row(is_synthetic=True))))
            with pytest.raises(SyntheticPolicyError, match="synthetic responses"):
                s.flush()
            s.rollback()
    assert sqlite_file_path(eng.connect()) == sqlite_file_path(eng)
    assert _count(eng) == 0


def test_finding_12_plain_session_add_is_checked_by_before_flush(tmp_path, memory_engine):
    # (a) location rule on a plain add
    eng = get_engine(f"sqlite:///{tmp_path / 'x' / 'bre.db'}")
    init_db(eng)
    with pytest.raises(SyntheticPolicyError, match="synthetic responses"):
        with session_scope(eng) as s:
            s.add(Response(**orm_kwargs(row(is_synthetic=True))))
    assert _count(eng) == 0
    syn_eng = get_engine(f"sqlite:///{tmp_path / 'data' / 'synthetic' / 'demo.db'}")
    init_db(syn_eng)
    with pytest.raises(SyntheticPolicyError, match="real responses"):
        with session_scope(syn_eng) as s:
            s.add(Response(**orm_kwargs(row(is_synthetic=False))))
    with session_scope(syn_eng) as s:
        s.add(Response(**orm_kwargs(row(is_synthetic=True))))
    assert _count(syn_eng) == 1
    # (b) no-mixing rule on a plain add (in-memory DB: location rule exempt, mixing rule not)
    with session_scope(memory_engine) as s:
        s.add(Response(**orm_kwargs(row(is_synthetic=True))))
    with pytest.raises(SyntheticPolicyError, match="may not mix"):
        with session_scope(memory_engine) as s:
            s.add(Response(**orm_kwargs(row(is_synthetic=False, source_row_ref="unit:1", position_in_session=1))))
    with pytest.raises(SyntheticPolicyError, match="mixes"):
        with session_scope(get_engine("sqlite://")) as s:
            init_db(s.get_bind())
            s.add(Response(**orm_kwargs(row(is_synthetic=True))))
            s.add(Response(**orm_kwargs(row(is_synthetic=False, position_in_session=1))))
    with session_scope(memory_engine) as s:
        assert {r.is_synthetic for r in s.query(Response)} == {True}


# ---------------------------------------------------------------------------------------------
# 5: numeric strings and extra columns are rejected by the row model and frame_to_events
# ---------------------------------------------------------------------------------------------


def test_finding_5_row_model_and_frame_to_events_reject_numeric_strings_and_extra_columns():
    strings = dict(response="1", loss_pct="-0.1", position_in_session="3", horizon_days="365", response_time_ms="12")
    for k, v in strings.items():
        with pytest.raises(ValidationError, match="string"):
            event_from_row(**{k: v})
    df = frame(row()).astype({k: "object" for k in strings})
    for k, v in strings.items():
        df.loc[df.index[0], k] = v
    report = S.validate_frame(df, strict=False)
    assert sum(report.range_violations[k] for k in strings) == 5
    with pytest.raises(S.SchemaError):
        S.frame_to_events(df)
    with pytest.raises(S.SchemaError, match="unexpected columns"):
        S.frame_to_events(frame(row()).assign(extra_col=1))
    assert len(S.frame_to_events(frame(row()))) == 1


# ---------------------------------------------------------------------------------------------
# 6: non-finite numbers raise ValueError everywhere in the ORM
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), np.inf, np.nan])
def test_finding_6_non_finite_numbers_raise_value_error(value, memory_engine):
    with pytest.raises(ValueError):  # ValueError, never OverflowError
        validate_response_range("likert", value)
    for col in ("position_in_session", "horizon_days", "response_time_ms", "loss_pct", "response"):
        with pytest.raises(ValueError):
            Response(**orm_kwargs(row(**{col: value})))
    # in a frame, NaN is the null marker: accepted in a nullable column, an error in response;
    # an infinity is an error everywhere and never reaches the table
    with pytest.raises(ValueError):
        with session_scope(memory_engine) as s:
            frame_to_responses(s, frame(row(response=value)))
    if math.isnan(value):
        with session_scope(memory_engine) as s:
            assert frame_to_responses(s, frame(row(response_time_ms=value))) == 1
            assert s.query(Response).one().response_time_ms is None
    else:
        with pytest.raises(ValueError):
            with session_scope(memory_engine) as s:
                frame_to_responses(s, frame(row(response_time_ms=value)))
        assert _count(memory_engine) == 0
    with memory_engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM responses WHERE typeof(response_time_ms) = 'real'")).scalar() == 0


# ---------------------------------------------------------------------------------------------
# 7: tags use fullmatch, so a trailing newline is rejected
# ---------------------------------------------------------------------------------------------


def test_finding_7_trailing_newline_tag_is_rejected_everywhere(tmp_path):
    tag = "news:recession\n"
    assert S.TAG_PATTERN.fullmatch("news:recession") and not S.TAG_PATTERN.fullmatch(tag)
    assert S.TAG_PATTERN.pattern.endswith(r"\Z")
    with pytest.raises(ValidationError, match="namespace:value"):
        event_from_row(context_tags=[tag])
    report = S.validate_frame(frame(row(context_tags=[tag])), strict=False)
    assert report.range_violations["context_tags"] == 1
    with pytest.raises(S.SchemaError):
        S.write_events(frame(row(context_tags=[tag])), tmp_path / "data" / "processed" / "x.parquet")
    with pytest.raises(ValueError):
        Response(**orm_kwargs(row(context_tags=[tag])))
    with pytest.raises(ValueError):
        D.condition_id([tag])


# ---------------------------------------------------------------------------------------------
# 8: os.path.abspath, no symlink resolution; relative paths follow the cwd
# ---------------------------------------------------------------------------------------------


def test_finding_8_symlinked_data_synthetic_counts_as_synthetic(tmp_path, monkeypatch):
    store = tmp_path / "elsewhere_store"
    store.mkdir()
    proj = tmp_path / "proj" / "data"
    proj.mkdir(parents=True)
    link = proj / "synthetic"
    try:
        link.symlink_to(store, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platform without symlinks
        pytest.skip(f"symlinks unavailable: {exc}")
    parquet = link / "gq.parquet"
    assert S.is_synthetic_path(parquet) and not S.is_synthetic_path(store / "gq.parquet")
    assert db.session.is_under_synthetic_dir(link / "demo.db")
    report = S.write_events(S.events_to_frame([event_from_row(is_synthetic=True)]), parquet)
    assert report.ok and (store / "gq.parquet").exists()
    eng = get_engine(f"sqlite:///{link / 'demo.db'}")
    init_db(eng)
    with session_scope(eng) as s:
        assert frame_to_responses(s, frame(row(is_synthetic=True))) == 1
    # relative paths are judged against the current directory, without resolving symlinks
    monkeypatch.chdir(tmp_path / "proj")
    assert S.is_synthetic_path("data/synthetic/a.parquet")
    assert not S.is_synthetic_path("data/processed/a.parquet")
    assert not S.is_synthetic_path("synthetic/data/a.parquet")


# ---------------------------------------------------------------------------------------------
# 9: one spelling of the question-order ids, validated for design rows
# ---------------------------------------------------------------------------------------------


def test_finding_9_question_order_ids_are_hyphenated_and_validated_for_design_rows():
    assert D.QUESTION_ORDER_IDS == ("tolerance-first", "scenario-first")
    for spec in db.INTERVENTIONS:
        order = spec["mapped_context_transform"].get("question_order")
        assert order is None or order in D.QUESTION_ORDER_IDS
        db.validate_transform(spec["mapped_context_transform"])
    battery = json.loads((BRE_ROOT / "instrument" / "battery.json").read_text(encoding="utf-8"))
    assert [o["question_order_id"] for o in battery["design"]["question_orders"]] == list(D.QUESTION_ORDER_IDS)
    assert battery["battery_version"] == D.DESIGN_VERSION
    assert "tolerance-first" in (BRE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "tolerance_first" not in (BRE_ROOT / "README.md").read_text(encoding="utf-8")
    assert D.item_id(-0.1, (), "tolerance-first") == "L10|none|tolerance-first"
    with pytest.raises(ValueError):
        D.item_id(-0.1, (), "tolerance_first")
    # schema: design rows must use the ids; other batteries are free-form; null always allowed
    event_from_row(question_order_id="tolerance-first")
    event_from_row(question_order_id=None)
    event_from_row(question_order_id="whatever", battery_version="cpc18")
    with pytest.raises(ValidationError, match="question_order_id"):
        event_from_row(question_order_id="tolerance_first")
    report = S.validate_frame(frame(row(question_order_id="tolerance_first")), strict=False)
    assert report.range_violations["question_order_id"] == 1 and not report.ok
    assert S.validate_frame(frame(row(question_order_id="tolerance_first", battery_version="cpc18"))).ok
    # DB: both assignment orders are checked
    with pytest.raises(ValueError, match="question_order_id"):
        Response(**orm_kwargs(row(question_order_id="tolerance_first")))
    r = Response(**orm_kwargs(row(battery_version="cpc18", question_order_id="tolerance_first")))
    with pytest.raises(ValueError, match="question_order_id"):
        r.battery_version = D.DESIGN_VERSION
    with pytest.raises(ValueError):
        db.validate_transform({"question_order": "tolerance_first"})
    with pytest.raises(ValueError):
        db.validate_transform({"add": ["news:war"]})


# ---------------------------------------------------------------------------------------------
# 10: delete_client verifies the cascade; session_scope asserts the FK pragma
# ---------------------------------------------------------------------------------------------


def test_finding_10_delete_client_fails_loudly_without_cascade(tmp_path):
    bare = create_engine(f"sqlite:///{tmp_path / 'bare.db'}")  # what a foreign framework would build
    init_db(bare)
    with Session(bind=bare) as s:
        assert s.execute(text("PRAGMA foreign_keys")).scalar() == 0
        s.add(Client(client_id="c1"))
        s.flush()
        frame_to_responses(s, frame(row(), row(position_in_session=1, source_row_ref="unit:1")), client_id="c1")
        s.commit()
    with pytest.raises(RuntimeError, match="PRAGMA foreign_keys"):
        with session_scope(bare):
            pass
    with Session(bind=bare) as s:
        with pytest.raises(RuntimeError, match="left dependent rows"):
            db.delete_client(s, "c1")
        s.commit()
    with Session(bind=bare) as s:
        assert s.get(Client, "c1") is not None
        assert s.query(Response).filter_by(client_id="c1").count() == 2
        assert s.query(db.AuditLog).filter_by(action="delete_client").count() == 0
    # the same on an engine from get_engine succeeds and the audit entry is written
    good = get_engine(f"sqlite:///{tmp_path / 'good.db'}")
    init_db(good)
    with session_scope(good) as s:
        s.add(Client(client_id="c1"))
        s.flush()
        frame_to_responses(s, frame(row()), client_id="c1")
    with session_scope(good) as s:
        assert db.delete_client(s, "c1") == {"responses": 1, "predictions_log": 0, "intervention_log": 0}
    with session_scope(good) as s:
        assert s.query(Response).count() == 0
        assert s.query(db.AuditLog).filter_by(action="delete_client").count() == 1


# ---------------------------------------------------------------------------------------------
# 11 / 22: default_engine follows BRE_DB_URL
# ---------------------------------------------------------------------------------------------


def test_finding_11_and_22_default_engine_follows_the_environment(tmp_path, monkeypatch):
    a, b = tmp_path / "A.db", tmp_path / "B.db"
    monkeypatch.setenv("BRE_DB_URL", f"sqlite:///{a}")
    assert db.default_engine().url.database == str(a)
    monkeypatch.setenv("BRE_DB_URL", f"sqlite:///{b}")
    assert db.resolve_db_url() == f"sqlite:///{b}"
    assert db.default_engine().url.database == str(b)
    assert db.default_engine() is db.default_engine()  # cached per resolved URL
    init_db(db.default_engine())
    with session_scope() as s:
        assert s.get_bind().url.database == str(b)
    monkeypatch.setenv("BRE_DB_URL", f"sqlite:///{a}")
    assert db.default_engine().url.database == str(a)
    # a relative default is cached on its absolute form
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BRE_DB_URL", "sqlite:///rel.db")
    assert db.default_engine().url.database == str(tmp_path / "rel.db")


# ---------------------------------------------------------------------------------------------
# 13: relative SQLite URLs are fixed to absolute paths at engine creation
# ---------------------------------------------------------------------------------------------


def test_finding_13_relative_sqlite_urls_are_absolute_at_engine_creation(tmp_path, monkeypatch):
    root = tmp_path
    syn_dir = root / "data" / "synthetic"
    syn_dir.mkdir(parents=True)
    assert absolute_sqlite_url("sqlite://") == "sqlite://"
    assert absolute_sqlite_url("sqlite:///:memory:") == "sqlite:///:memory:"
    monkeypatch.chdir(syn_dir)
    assert absolute_sqlite_url("sqlite:///demo.db") == f"sqlite:///{syn_dir / 'demo.db'}"
    assert absolute_sqlite_url("sqlite:///file:demo.db?uri=true") == f"sqlite:///file:{syn_dir / 'demo.db'}?uri=true"
    # (a) engine created inside data/synthetic, cwd changed afterwards
    eng = get_engine("sqlite:///demo.db")
    assert eng.url.database == str(syn_dir / "demo.db")
    init_db(eng)
    monkeypatch.chdir(root)
    with pytest.raises(SyntheticPolicyError, match="real responses"):
        with session_scope(eng) as s:
            frame_to_responses(s, frame(row(is_synthetic=False)))
    with session_scope(eng) as s:
        assert frame_to_responses(s, frame(row(is_synthetic=True))) == 1
    with sqlite3.connect(syn_dir / "demo.db") as raw:
        assert raw.execute("SELECT COUNT(*) FROM responses WHERE is_synthetic = 1").fetchone()[0] == 1
    assert not (root / "demo.db").exists()
    # (b) engine created at the root, cwd moved into data/synthetic afterwards
    eng2 = get_engine("sqlite:///bre.db")
    assert eng2.url.database == str(root / "bre.db")
    init_db(eng2)
    monkeypatch.chdir(syn_dir)
    with pytest.raises(SyntheticPolicyError, match="synthetic responses"):
        with session_scope(eng2) as s:
            frame_to_responses(s, frame(row(is_synthetic=True)))
    with session_scope(eng2) as s:
        assert frame_to_responses(s, frame(row(is_synthetic=False))) == 1
    assert not (syn_dir / "bre.db").exists()


# ---------------------------------------------------------------------------------------------
# 14: NaN / Infinity never reach the database
# ---------------------------------------------------------------------------------------------


def test_finding_14_nan_and_infinity_are_rejected_on_every_json_path(memory_engine):
    with pytest.raises(ValueError):
        Client(client_id="c", covariates=json.dumps({"invest_experience_yrs": float("nan")}))
    for tok in ('{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_json_object(tok, "x", allow_null=False)
    with pytest.raises(ValueError):
        parse_json_string_list("[NaN]", "x")
    with pytest.raises(ValueError):
        S.validate_covariates('{"weight": NaN}')
    with pytest.raises(ValueError):
        S.validate_outcome_behavior('{"action": "sold", "lag_days": Infinity}')
    with session_scope(memory_engine) as s:
        for bad in (
            frame(row(covariates=json.dumps({"weight": float("inf")}))),
            frame(row(covariates={"x_v": float("nan")})),
            frame(row(response_time_ms=float("inf"))),
            frame(row(outcome_behavior={"action": "sold", "lag_days": float("nan")})),
        ):
            with pytest.raises(ValueError):
                frame_to_responses(s, bad)
        with pytest.raises(ValueError):
            write_audit(s, actor="t", action="a", entity="e", entity_id="1", payload={"nll": float("nan")})
        with pytest.raises(ValueError):
            db.AuditLog(actor="t", action="a", entity="e", entity_id="1", payload='{"nll": NaN}')
        with pytest.raises(ValueError):
            db.ModelRegistry(version="v", model_type="Q2", metrics='{"nll": Infinity}')
    assert _count(memory_engine) == 0
    with session_scope(memory_engine) as s:
        write_audit(s, actor="t", action="a", entity="e", entity_id="1", payload={"nll": 0.61})
    with memory_engine.connect() as conn:
        assert conn.execute(text("SELECT MIN(json_valid(payload)) FROM audit_log")).scalar() == 1


# ---------------------------------------------------------------------------------------------
# 17: DB-level guards on JSON and boolean columns; corrupt rows are identifiable
# ---------------------------------------------------------------------------------------------


def test_finding_17_check_constraints_guard_json_and_bool_columns(memory_engine):
    with session_scope(memory_engine) as s:
        s.add(Client(client_id="c1"))
        s.flush()
        frame_to_responses(
            s, frame(row(), row(position_in_session=1, source_row_ref="unit:1"), row(position_in_session=2, source_row_ref="unit:2")),
            client_id="c1",
        )
    updates = [
        "UPDATE responses SET context_tags = 'not json' WHERE position_in_session = 1",
        "UPDATE responses SET context_tags = '{}' WHERE position_in_session = 1",
        "UPDATE responses SET prior_question_ids = '\"x\"' WHERE position_in_session = 1",
        "UPDATE responses SET covariates = '[1]' WHERE position_in_session = 1",
        "UPDATE responses SET outcome_behavior = '\"x\"' WHERE position_in_session = 1",
        "UPDATE responses SET incentivized = 7 WHERE position_in_session = 1",
        "UPDATE responses SET consent_training = -1 WHERE position_in_session = 1",
        "UPDATE responses SET is_synthetic = 2 WHERE position_in_session = 1",
        "UPDATE clients SET covariates = '[]' WHERE client_id = 'c1'",
        "INSERT INTO audit_log (ts, actor, action, entity, entity_id, payload) VALUES ('2026-01-01', 'a', 'b', 'c', 'd', 'nope')",
        "INSERT INTO interventions (name, script, mapped_context_transform, active) VALUES ('n', 's', '{}', 3)",
        "INSERT INTO model_registry (version, model_type, training_data_refs, metrics, calibrated_contexts, is_active) VALUES ('v', 'Q', '{}', '{}', '[]', 0)",
    ]
    for sql in updates:
        with pytest.raises(IntegrityError):
            with memory_engine.begin() as conn:
                conn.execute(text(sql))
    # a row that passes the CHECKs but breaks a schema rule is reported with its id
    with memory_engine.begin() as conn:
        conn.execute(text("UPDATE responses SET context_tags = '[\"no_namespace\"]' WHERE position_in_session = 2"))
        bad_id = conn.execute(text("SELECT id FROM responses WHERE position_in_session = 2")).scalar()
    with session_scope(memory_engine) as s:
        with pytest.raises(ValueError, match=f"row id={bad_id}"):
            responses_to_frame(s, client_id="c1")
        assert s.get(Response, bad_id).incentivized is False


# ---------------------------------------------------------------------------------------------
# 20 / 21: Makefile hint and instrument targets
# ---------------------------------------------------------------------------------------------


def _make(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [shutil.which("make"), "-C", str(BRE_ROOT), *args],
        capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=60,
    )


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
def test_finding_20_make_prints_a_hint_naming_bre_venv_when_python_is_missing(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "BRE_VENV"}
    result = subprocess.run(
        [shutil.which("make"), "-C", str(BRE_ROOT), "test", f"VENV={tmp_path / 'nope'}"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert result.returncode != 0
    assert "BRE_VENV" in result.stdout and "make setup" in result.stdout
    assert "No such file or directory" not in result.stderr + result.stdout
    # with the real interpreter the check passes and pytest would be invoked
    dry = _make("-n", "test", f"BRE_VENV={Path(sys.executable).parents[1]}")
    assert dry.returncode == 0 and "pytest" in dry.stdout
    readme = (BRE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "BRE_VENV=/path/to/venv make test" in readme


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
def test_finding_21_instrument_targets_run_the_validator_and_the_balance_test():
    venv = str(Path(sys.executable).parents[1])
    full = _make("-n", "instrument", f"BRE_VENV={venv}", "NODE=/opt/node22/bin/node")
    assert full.returncode == 0, full.stdout + full.stderr
    assert "instrument/validate_battery.py --write-static" in full.stdout
    assert "/opt/node22/bin/node instrument/tests/test_balance.js" in full.stdout
    assert "instrument/build.py" not in full.stdout
    read_only = _make("-n", "test-instrument", f"BRE_VENV={venv}", "NODE=mynode")
    assert read_only.returncode == 0, read_only.stdout + read_only.stderr
    assert "instrument/validate_battery.py &&" in read_only.stdout and "--write-static" not in read_only.stdout
    assert "mynode instrument/tests/test_balance.js" in read_only.stdout
    makefile = (BRE_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "NODE ?= node" in makefile
    readme = (BRE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "/opt/node22/bin/node" in readme and "test-instrument" in readme
