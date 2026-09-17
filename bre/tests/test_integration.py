"""Contract tests between ``bre.schema`` (parquet side) and ``db`` (SQLite side).

PLAN.md section 2: "The DB ``responses`` table has the same columns" as the unified schema, and every
loader/generator calls ``validate_frame``. The two modules are deliberately decoupled (``db`` does
not import ``bre.schema``), so these tests pin the shared contract:

1. every column-level constant is defined identically on both sides;
2. a frame that passes ``bre.schema.validate_frame`` inserts into the DB unchanged and comes back
   (``responses_to_frame``) as a frame that passes ``validate_frame`` again and is cell-for-cell
   equal after ``conform_frame`` (lists, JSON text, nulls in every nullable column, Int64 horizon);
3. the real-vs-synthetic location rule (CLAUDE.md rule 3) is applied consistently by the parquet
   writer and the DB writer for both a synthetic table under ``data/synthetic/`` and a real table
   outside it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import db
from bre import design as D
from bre import schema as S

# ---------------------------------------------------------------------------------------------
# 1. Shared constants
# ---------------------------------------------------------------------------------------------


def test_schema_and_db_column_constants_agree():
    assert list(S.COLUMNS) == list(db.SCHEMA_COLUMNS)
    assert tuple(S.ELICITATION_TYPES) == tuple(db.ELICITATION_TYPES)
    assert frozenset(S.NULLABLE_COLUMNS) == frozenset(db.NULLABLE_COLUMNS)
    assert tuple(S.LIST_COLUMNS) == tuple(db.LIST_COLUMNS)
    assert tuple(S.JSON_COLUMNS) == tuple(db.JSON_COLUMNS)
    assert tuple(S.BOOL_COLUMNS) == tuple(db.BOOL_COLUMNS)
    assert tuple(S.KEY_COLUMNS) == tuple(db.UNIQUE_KEY)
    assert tuple(S.COVARIATE_KEYS) == tuple(db.COVARIATE_KEYS)
    assert tuple(S.OUTCOME_REQUIRED_KEYS) == tuple(db.OUTCOME_BEHAVIOR_REQUIRED_KEYS)


def test_response_table_columns_follow_schema_order():
    """``responses`` = (id, client_id) + the 21 schema columns in PLAN order."""
    names = [c.name for c in db.Response.__table__.columns]
    assert names[:2] == ["id", "client_id"]
    assert names[2:] == list(S.COLUMNS)


# ---------------------------------------------------------------------------------------------
# 2./3. Parquet -> DB -> frame round trips
# ---------------------------------------------------------------------------------------------


def _design_events(rng: np.random.Generator, *, is_synthetic: bool, dataset: str) -> list[S.DecisionEvent]:
    """One subject answering the whole 170-item design (PLAN.md section 3)."""
    cov = D.sample_covariates(rng, 1).iloc[0].to_dict()
    events = []
    for pos, row in enumerate(D.full_design().itertuples(index=False)):
        events.append(
            S.DecisionEvent(
                subject_id="s000",
                dataset=dataset,
                session_id="full",
                timestamp=None,
                position_in_session=pos,
                scenario_id=row.scenario_id,
                loss_pct=row.loss_pct,
                horizon_days=row.horizon_days,
                context_tags=row.context_tags,
                question_order_id=row.question_order_id,
                prior_question_ids=row.prior_question_ids,
                elicitation_type=row.elicitation_type,
                response=float(rng.integers(0, 2)),
                response_time_ms=None,
                covariates=cov,
                outcome_behavior=None,
                incentivized=False,
                consent_training=True,
                battery_version=D.DESIGN_VERSION,
                is_synthetic=is_synthetic,
                source_row_ref=row.item_id,
            )
        )
    return events


def _mixed_events(*, is_synthetic: bool, dataset: str) -> list[S.DecisionEvent]:
    """Eight events over two subjects covering all four elicitation types and every nullable column."""
    types = ["binary_sell", "allocation_pct", "likert", "lottery_choice"]
    responses = {"binary_sell": 1.0, "allocation_pct": 0.35, "likert": 5.0, "lottery_choice": 0.0}
    events = []
    for i in range(8):
        et = types[i % 4]
        events.append(
            S.DecisionEvent(
                subject_id=f"s{i % 2:03d}",
                dataset=dataset,
                session_id="sess-1",
                timestamp=None if i % 3 == 0 else "2026-09-17T03:19:00+00:00",
                position_in_session=i // 2,
                scenario_id="L10|rec>friend" if et != "likert" else "tolerance",
                loss_pct=None if et == "likert" else -0.05 * (1 + i % 5),
                horizon_days=None if et == "likert" else 365,
                context_tags=[] if i % 2 else ["news:recession", "social:friend_sells"],
                question_order_id=None if i % 2 else "scenario_first",
                prior_question_ids=[] if i % 2 else ["L05|none"],
                elicitation_type=et,
                response=responses[et],
                response_time_ms=None if i == 1 else 100.0 * (i + 1),
                covariates={"age_band": "30-44", "wealth_band": "<50k", "invest_experience_yrs": 3},
                outcome_behavior=None if i % 2 else {"action": "sold_all", "lag_days": 12, "amount_pct": 1.0},
                incentivized=False,
                consent_training=True,
                battery_version="bre-design-1.0",
                is_synthetic=is_synthetic,
                source_row_ref=f"row:{i}",
            )
        )
    return events


def _comparable(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical dtypes, key order, list cells as tuples so ``assert_frame_equal`` can compare them."""
    out = S.conform_frame(df).sort_values(S.KEY_COLUMNS).reset_index(drop=True)
    for c in S.LIST_COLUMNS:
        out[c] = out[c].map(tuple)
    return out


def _parquet_to_db_and_back(frame: pd.DataFrame, parquet: Path, db_file: Path, *, is_synthetic: bool) -> pd.DataFrame:
    parquet.parent.mkdir(parents=True, exist_ok=True)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    report = S.write_events(frame, parquet)
    assert report.ok
    loaded = S.read_events(parquet)

    engine = db.get_engine(f"sqlite:///{db_file}")
    try:
        db.init_db(engine)
        with db.session_scope(engine) as session:
            assert db.frame_to_responses(session, loaded) == len(frame)
        with db.session_scope(engine) as session:
            back = db.responses_to_frame(session, is_synthetic=is_synthetic)
    finally:
        engine.dispose()

    assert list(back.columns) == list(S.COLUMNS)
    assert S.validate_frame(back, expect_synthetic=is_synthetic).ok
    pd.testing.assert_frame_equal(_comparable(back), _comparable(loaded))
    return back


def test_full_design_round_trips_parquet_to_db_under_data_synthetic(tmp_path: Path, rng: np.random.Generator):
    frame = S.events_to_frame(_design_events(rng, is_synthetic=True, dataset="synthetic_design"))
    back = _parquet_to_db_and_back(
        frame,
        tmp_path / "data" / "synthetic" / "design.parquet",
        tmp_path / "data" / "synthetic" / "design.db",
        is_synthetic=True,
    )
    assert len(back) == 170
    assert back["scenario_id"].nunique() == 85
    assert back["context_tags"].map(len).value_counts().to_dict() == {2: 120, 1: 40, 0: 10}
    assert back["horizon_days"].dtype == "Int64" and (back["horizon_days"] == D.HORIZON_DAYS).all()


def test_mixed_types_and_nulls_round_trip_parquet_to_db_outside_data_synthetic(tmp_path: Path):
    frame = S.events_to_frame(_mixed_events(is_synthetic=False, dataset="real_x"))
    back = _parquet_to_db_and_back(
        frame,
        tmp_path / "data" / "processed" / "real_x.parquet",
        tmp_path / "real_x.db",
        is_synthetic=False,
    )
    assert set(back["elicitation_type"]) == set(S.ELICITATION_TYPES)
    for column in S.NULLABLE_COLUMNS:
        assert back[column].isna().any(), f"{column} should contain a null in this fixture"
    assert back["context_tags"].map(len).tolist().count(0) == 4
    assert back["prior_question_ids"].map(len).tolist().count(0) == 4


def test_location_rule_is_enforced_identically_by_parquet_and_db_writers(tmp_path: Path):
    """A synthetic table outside data/synthetic (and a real one inside) is refused on both sides."""
    synthetic = S.events_to_frame(_mixed_events(is_synthetic=True, dataset="synthetic_gq"))
    real = S.events_to_frame(_mixed_events(is_synthetic=False, dataset="real_x"))
    (tmp_path / "data" / "synthetic").mkdir(parents=True)

    with pytest.raises(S.SchemaError):
        S.write_events(synthetic, tmp_path / "synthetic_gq.parquet")
    with pytest.raises(S.SchemaError):
        S.write_events(real, tmp_path / "data" / "synthetic" / "real_x.parquet")

    for frame, db_file in ((synthetic, tmp_path / "leak.db"), (real, tmp_path / "data" / "synthetic" / "leak.db")):
        engine = db.get_engine(f"sqlite:///{db_file}")
        try:
            db.init_db(engine)
            with pytest.raises(ValueError):
                with db.session_scope(engine) as session:
                    db.frame_to_responses(session, frame)
        finally:
            engine.dispose()
