"""Tests for the BRE database package (bre/db): tables, cascades, frame round-trip, foreign keys."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

BRE_ROOT = Path(__file__).resolve().parents[1]
if str(BRE_ROOT) not in sys.path:  # allow `pytest tests/test_db.py` without a conftest
    sys.path.insert(0, str(BRE_ROOT))

from db import (  # noqa: E402
    INTERVENTIONS,
    SCHEMA_COLUMNS,
    AuditLog,
    Client,
    Intervention,
    InterventionLog,
    ModelRegistry,
    PredictionLog,
    Response,
    delete_client,
    export_client,
    frame_to_responses,
    get_engine,
    init_db,
    resolve_db_url,
    responses_to_frame,
    seed_interventions,
    session_scope,
)
from db.session import is_under_synthetic_dir  # noqa: E402

EXPECTED_TABLES = {
    "clients",
    "responses",
    "interventions",
    "predictions_log",
    "intervention_log",
    "model_registry",
    "audit_log",
}


# --------------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------------


@pytest.fixture()
def engine():
    """Fresh in-memory SQLite engine with all tables created."""
    eng = get_engine("sqlite://")
    init_db(eng)
    return eng


def make_event(**overrides) -> dict:
    """A valid real (is_synthetic=False) DecisionEvent row, in schema order, with overrides."""
    row = {
        "subject_id": "s1",
        "dataset": "unit",
        "session_id": "sess1",
        "timestamp": "2026-09-17T10:00:00+00:00",
        "position_in_session": 0,
        "scenario_id": "L-0.10",
        "loss_pct": -0.10,
        "horizon_days": 365,
        "context_tags": ["news:recession", "social:friend_sells"],
        "question_order_id": "scenario-first",
        "prior_question_ids": ["q_tol"],
        "elicitation_type": "binary_sell",
        "response": 1.0,
        "response_time_ms": 1234.5,
        "covariates": json.dumps({"age_band": "30-44", "self_reported_risk_tolerance": 4}),
        "outcome_behavior": json.dumps({"action": "sold", "lag_days": 3}),
        "incentivized": False,
        "consent_training": True,
        "battery_version": "v1",
        "is_synthetic": False,
        "source_row_ref": "unit:0",
    }
    row.update(overrides)
    return row


def make_frame(n: int = 3, **overrides) -> pd.DataFrame:
    """A frame of n rows in one session, positions 0..n-1, mixing null and non-null optionals."""
    rows = []
    for i in range(n):
        rows.append(
            make_event(
                position_in_session=i,
                scenario_id=f"L-{i}",
                context_tags=[] if i == 0 else [f"tag:{i}", "news:technical"],
                prior_question_ids=[f"q{j}" for j in range(i)],
                outcome_behavior=None if i % 2 else json.dumps({"action": "held", "lag_days": i}),
                timestamp=None if i == 1 else f"2026-09-17T10:0{i}:00",
                loss_pct=None if i == 2 else -0.05 * (i + 1),
                horizon_days=None if i == 1 else 365,
                response_time_ms=None if i == 0 else 800.0 + i,
                question_order_id=None if i == 2 else "scenario-first",
                source_row_ref=f"unit:{i}",
                **overrides,
            )
        )
    return pd.DataFrame(rows, columns=list(SCHEMA_COLUMNS))


def add_client_with_rows(session, client_id: str = "c1") -> None:
    """Insert a client with two responses, one prediction, one intervention log."""
    session.add(Client(client_id=client_id, display_label="Test", covariates="{}"))
    session.flush()
    frame_to_responses(session, make_frame(2, subject_id=client_id), client_id=client_id)
    session.add(
        PredictionLog(
            client_id=client_id,
            model_version="test",
            market_state=json.dumps({"loss_pct": -0.1}),
            outputs=json.dumps({"p_sell": 0.5}),
        )
    )
    seed_interventions(session)
    intervention = session.query(Intervention).filter_by(name="social_proof_counter").one()
    session.add(
        InterventionLog(
            client_id=client_id,
            intervention_id=intervention.id,
            advisor_note="applied",
            outcome_behavior=json.dumps({"action": "held", "lag_days": 7}),
        )
    )
    session.flush()


# --------------------------------------------------------------------------------------------
# Tables and configuration
# --------------------------------------------------------------------------------------------


def test_init_creates_all_tables(engine) -> None:
    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES


def test_responses_table_has_schema_columns_in_order(engine) -> None:
    names = [c["name"] for c in inspect(engine).get_columns("responses")]
    assert names[:2] == ["id", "client_id"]
    assert tuple(names[2:]) == SCHEMA_COLUMNS


def test_resolve_db_url_precedence(monkeypatch) -> None:
    monkeypatch.delenv("BRE_DB_URL", raising=False)
    assert resolve_db_url() == "sqlite:///bre.db"
    monkeypatch.setenv("BRE_DB_URL", "sqlite:///env.db")
    assert resolve_db_url() == "sqlite:///env.db"
    assert resolve_db_url("sqlite:///explicit.db") == "sqlite:///explicit.db"


def test_file_engine_enables_foreign_keys_on_every_connection(tmp_path) -> None:
    eng = get_engine(f"sqlite:///{tmp_path / 'fk.db'}")
    init_db(eng)
    for _ in range(2):
        with eng.connect() as conn:
            assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


# --------------------------------------------------------------------------------------------
# Foreign keys and cascade delete
# --------------------------------------------------------------------------------------------


def test_foreign_keys_enforced_on_insert(engine) -> None:
    with pytest.raises(IntegrityError):
        with session_scope(engine) as s:
            s.add(Response(client_id="missing", **_kwargs(make_event())))
            s.flush()
    with pytest.raises(IntegrityError):
        with session_scope(engine) as s:
            s.add(Client(client_id="c9"))
            s.flush()
            s.add(InterventionLog(client_id="c9", intervention_id=999, advisor_note=None))
            s.flush()


def test_raw_sql_delete_cascades(engine) -> None:
    with session_scope(engine) as s:
        add_client_with_rows(s, "c2")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM clients WHERE client_id = 'c2'"))
    with session_scope(engine) as s:
        assert s.query(Response).filter_by(client_id="c2").count() == 0
        assert s.query(PredictionLog).filter_by(client_id="c2").count() == 0
        assert s.query(InterventionLog).filter_by(client_id="c2").count() == 0


def test_delete_client_removes_dependents_and_writes_audit(engine) -> None:
    with session_scope(engine) as s:
        add_client_with_rows(s, "c1")
        audit_before = s.query(AuditLog).count()
    with session_scope(engine) as s:
        counts = delete_client(s, "c1", actor="tester")
    assert counts == {"responses": 2, "predictions_log": 1, "intervention_log": 1}
    with session_scope(engine) as s:
        assert s.get(Client, "c1") is None
        assert s.query(Response).count() == 0
        assert s.query(PredictionLog).count() == 0
        assert s.query(InterventionLog).count() == 0
        assert s.query(Intervention).count() == len(INTERVENTIONS)  # interventions survive
        entries = s.query(AuditLog).filter_by(action="delete_client").all()
        assert len(entries) == 1
        assert s.query(AuditLog).count() == audit_before + 1
        entry = entries[0]
        assert (entry.actor, entry.entity, entry.entity_id) == ("tester", "clients", "c1")
        assert json.loads(entry.payload)["deleted"] == counts
    with pytest.raises(LookupError):
        with session_scope(engine) as s:
            delete_client(s, "c1")


def test_export_client_returns_every_row(engine) -> None:
    with session_scope(engine) as s:
        add_client_with_rows(s, "c3")
        dump = export_client(s, "c3")
    assert dump["client"]["client_id"] == "c3"
    assert len(dump["responses"]) == 2
    assert len(dump["predictions_log"]) == 1
    assert len(dump["intervention_log"]) == 1
    assert json.loads(dump["responses"][0]["context_tags"]) == []
    json.dumps(dump)  # everything is JSON-serializable


# --------------------------------------------------------------------------------------------
# Frame round trip
# --------------------------------------------------------------------------------------------


def _kwargs(event: dict) -> dict:
    """Constructor kwargs for Response from a schema row (lists go through the hybrid setters)."""
    kw = dict(event)
    kw["context_tags_json"] = json.dumps(kw.pop("context_tags"))
    kw["prior_question_ids_json"] = json.dumps(kw.pop("prior_question_ids"))
    return kw


def test_frame_round_trip_preserves_lists_json_and_nulls(engine) -> None:
    src = make_frame(3)
    with session_scope(engine) as s:
        assert frame_to_responses(s, src) == 3
    with session_scope(engine) as s:
        out = responses_to_frame(s, dataset="unit")
    assert list(out.columns) == list(SCHEMA_COLUMNS)
    assert len(out) == 3
    for i in range(3):
        a, b = src.iloc[i], out.iloc[i]
        assert list(b["context_tags"]) == list(a["context_tags"])
        assert list(b["prior_question_ids"]) == list(a["prior_question_ids"])
        assert json.loads(b["covariates"]) == json.loads(a["covariates"])
        if pd.isna(a["outcome_behavior"]):
            assert pd.isna(b["outcome_behavior"])
        else:
            assert json.loads(b["outcome_behavior"]) == json.loads(a["outcome_behavior"])
        for col in ("timestamp", "loss_pct", "horizon_days", "response_time_ms", "question_order_id"):
            if pd.isna(a[col]):
                assert pd.isna(b[col]), col
            else:
                assert b[col] == a[col], col
        for col in ("subject_id", "dataset", "session_id", "position_in_session", "scenario_id",
                    "elicitation_type", "response", "incentivized", "consent_training",
                    "battery_version", "is_synthetic", "source_row_ref"):
            assert b[col] == a[col], col
    # a second round trip through the DB is the identity on the frame content
    eng2 = get_engine("sqlite://")
    init_db(eng2)
    with session_scope(eng2) as s:
        frame_to_responses(s, out)
        again = responses_to_frame(s)
    pd.testing.assert_frame_equal(again, out)


def test_frame_to_responses_accepts_client_and_arrays(engine) -> None:
    import numpy as np

    src = make_frame(2)
    src["context_tags"] = [np.array(["a", "b"]), np.array([], dtype=str)]  # parquet-style arrays
    with session_scope(engine) as s:
        s.add(Client(client_id="c4"))
        s.flush()
        frame_to_responses(s, src, client_id="c4")
        rows = s.query(Response).order_by(Response.position_in_session).all()
        assert [r.context_tags for r in rows] == [["a", "b"], []]
        assert all(r.client_id == "c4" for r in rows)
        assert s.query(AuditLog).filter_by(action="insert_responses").count() == 1
        assert responses_to_frame(s, client_id="c4").shape[0] == 2
        assert responses_to_frame(s, client_id="nobody").shape[0] == 0


def test_frame_missing_column_rejected(engine) -> None:
    with session_scope(engine) as s:
        with pytest.raises(ValueError, match="missing schema columns"):
            frame_to_responses(s, make_frame(1).drop(columns=["covariates"]))


# --------------------------------------------------------------------------------------------
# Row-level rules
# --------------------------------------------------------------------------------------------


def test_uniqueness_key_enforced(engine) -> None:
    with pytest.raises(IntegrityError):
        with session_scope(engine) as s:
            frame_to_responses(s, make_frame(1))
            frame_to_responses(s, make_frame(1, battery_version="v2"))


@pytest.mark.parametrize(
    ("elicitation_type", "response"),
    [
        ("binary_sell", 0.5),
        ("lottery_choice", 2),
        ("allocation_pct", 1.5),
        ("allocation_pct", -0.1),
        ("likert", 0),
        ("likert", 8),
        ("likert", 3.5),
    ],
)
def test_response_range_by_type_rejected(elicitation_type, response) -> None:
    with pytest.raises(ValueError):
        Response(**_kwargs(make_event(elicitation_type=elicitation_type, response=response)))


@pytest.mark.parametrize(
    ("elicitation_type", "response"),
    [("binary_sell", 0), ("lottery_choice", 1), ("allocation_pct", 0.35), ("likert", 7)],
)
def test_response_range_by_type_accepted(engine, elicitation_type, response) -> None:
    with session_scope(engine) as s:
        frame_to_responses(s, make_frame(1, elicitation_type=elicitation_type, response=response))
        assert s.query(Response).one().response == float(response)


def test_db_level_check_constraints(engine) -> None:
    """The CHECK constraints hold even for rows written around the ORM validators."""
    cols = ", ".join(SCHEMA_COLUMNS)
    good = make_event(context_tags="[]", prior_question_ids="[]")
    placeholders = ", ".join(f":{c}" for c in SCHEMA_COLUMNS)
    with engine.begin() as conn:
        conn.execute(text(f"INSERT INTO responses ({cols}) VALUES ({placeholders})"), good)
    bad_rows = [
        dict(good, position_in_session=1, response=0.5),  # binary_sell outside {0,1}
        dict(good, position_in_session=2, loss_pct=-1.5),
        dict(good, position_in_session=3, elicitation_type="slider"),
        dict(good, position_in_session=4, elicitation_type="likert", response=2.5),
        dict(good, position_in_session=0),  # duplicate uniqueness key
    ]
    for bad in bad_rows:
        with pytest.raises(IntegrityError):
            with engine.begin() as conn:
                conn.execute(text(f"INSERT INTO responses ({cols}) VALUES ({placeholders})"), bad)


def test_invalid_json_and_lists_rejected() -> None:
    with pytest.raises(ValueError):
        Response(**_kwargs(make_event(covariates="not json")))
    with pytest.raises(ValueError):
        Response(**_kwargs(make_event(covariates="[1, 2]")))
    with pytest.raises(ValueError):
        Response(**_kwargs(make_event(outcome_behavior=json.dumps({"action": "sold"}))))
    with pytest.raises(ValueError):
        Response(**_kwargs(make_event(timestamp="yesterday")))
    with pytest.raises(ValueError):
        Response(**_kwargs(make_event(context_tags=[1, 2])))
    with pytest.raises(ValueError):
        Client(client_id="x", covariates="{")


# --------------------------------------------------------------------------------------------
# Real vs synthetic policy
# --------------------------------------------------------------------------------------------


def test_is_under_synthetic_dir(tmp_path) -> None:
    assert is_under_synthetic_dir(tmp_path / "data" / "synthetic" / "demo.db")
    assert not is_under_synthetic_dir(tmp_path / "data" / "processed" / "x.db")
    assert not is_under_synthetic_dir(tmp_path / "synthetic" / "data" / "x.db")


def test_synthetic_location_rule(tmp_path) -> None:
    synthetic_engine = get_engine(f"sqlite:///{tmp_path / 'data' / 'synthetic' / 'demo.db'}")
    real_engine = get_engine(f"sqlite:///{tmp_path / 'bre.db'}")
    init_db(synthetic_engine)
    init_db(real_engine)
    with session_scope(synthetic_engine) as s:
        assert frame_to_responses(s, make_frame(1, is_synthetic=True)) == 1
        with pytest.raises(ValueError, match="real responses"):
            frame_to_responses(s, make_frame(1, is_synthetic=False, subject_id="s2"))
    with session_scope(real_engine) as s:
        assert frame_to_responses(s, make_frame(1, is_synthetic=False)) == 1
        with pytest.raises(ValueError, match="synthetic responses"):
            frame_to_responses(s, make_frame(1, is_synthetic=True, subject_id="s2"))


def test_tables_never_mix_real_and_synthetic(engine) -> None:
    mixed = pd.concat([make_frame(1), make_frame(1, is_synthetic=True, subject_id="s2")], ignore_index=True)
    with session_scope(engine) as s:
        with pytest.raises(ValueError, match="mixes"):
            frame_to_responses(s, mixed)
        frame_to_responses(s, make_frame(1, is_synthetic=True))
        with pytest.raises(ValueError, match="may not mix"):
            frame_to_responses(s, make_frame(1, is_synthetic=False, subject_id="s2"))


# --------------------------------------------------------------------------------------------
# Seed data and registry
# --------------------------------------------------------------------------------------------


def test_seed_interventions_is_complete_and_idempotent(engine) -> None:
    expected = {
        "reframe_time_horizon",
        "show_historical_recoveries",
        "precommitment_reminder",
        "cash_sleeve_check",
        "social_proof_counter",
    }
    with session_scope(engine) as s:
        assert set(seed_interventions(s)) == expected
        assert seed_interventions(s) == []
        rows = s.query(Intervention).all()
    assert {r.name for r in rows} == expected
    for r in rows:
        transform = json.loads(r.mapped_context_transform)
        assert set(transform) <= {"add", "remove", "question_order"}
        assert transform, r.name
        assert r.active is True
        assert not any(ch.isdigit() for ch in r.script), f"{r.name} script contains a number"
        assert len(r.script.split()) >= 40


def test_model_registry_and_audit_rows(engine) -> None:
    with session_scope(engine) as s:
        s.add(
            ModelRegistry(
                version="q2-2026-09-17",
                model_type="Q2",
                training_data_refs=json.dumps(["data/synthetic/gq_n200_seed0.parquet"]),
                metrics=json.dumps({"nll": 0.61}),
                calibrated_contexts=json.dumps(["news:recession"]),
                is_active=True,
            )
        )
        with pytest.raises(ValueError):
            ModelRegistry(version="bad", model_type="Q2", metrics="{oops")
    with session_scope(engine) as s:
        reg = s.get(ModelRegistry, "q2-2026-09-17")
        assert reg is not None and reg.is_active and reg.promoted_at is None
