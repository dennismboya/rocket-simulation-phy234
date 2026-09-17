"""Tests for bre.schema: row model rules, frame validation, the synthetic-path rule and parquet I/O."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from bre import schema as S
from bre.schema import DecisionEvent, SchemaError, ValidationReport

# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------


def base_kwargs(**overrides) -> dict:
    """A valid binary_sell event; ``overrides`` replace fields."""
    kw = dict(
        subject_id="s001",
        dataset="synthetic_gq",
        session_id="sess-1",
        timestamp="2026-09-17T03:19:00Z",
        position_in_session=0,
        scenario_id="L10|rec>friend",
        loss_pct=-0.10,
        horizon_days=365,
        context_tags=["news:recession", "social:friend_sells"],
        question_order_id="tolerance_first",
        prior_question_ids=["tolerance"],
        elicitation_type="binary_sell",
        response=1.0,
        response_time_ms=850.0,
        covariates={"age_band": "30-44", "wealth_band": "<50k", "invest_experience_yrs": 3},
        outcome_behavior=None,
        incentivized=False,
        consent_training=True,
        battery_version="bre-design-1.0",
        is_synthetic=True,
        source_row_ref="gen:0",
    )
    kw.update(overrides)
    return kw


def make_events(n: int = 6, is_synthetic: bool = True, dataset: str = "synthetic_gq") -> list[DecisionEvent]:
    """n events over two subjects covering all four elicitation types and null-able columns."""
    types = ["binary_sell", "allocation_pct", "likert", "lottery_choice"]
    responses = {"binary_sell": 1.0, "allocation_pct": 0.35, "likert": 5.0, "lottery_choice": 0.0}
    events = []
    for i in range(n):
        et = types[i % 4]
        events.append(
            DecisionEvent(
                **base_kwargs(
                    subject_id=f"s{i % 2:03d}",
                    dataset=dataset,
                    position_in_session=i // 2,
                    elicitation_type=et,
                    response=responses[et],
                    timestamp=None if i % 3 == 0 else "2026-09-17T03:19:00+00:00",
                    loss_pct=None if et == "likert" else -0.05 * (1 + i % 5),
                    horizon_days=None if et == "likert" else 365,
                    context_tags=[] if i % 2 else ["news:technical"],
                    question_order_id=None if i % 2 else "scenario_first",
                    prior_question_ids=[] if i % 2 else ["L05|none"],
                    response_time_ms=None if i == 1 else 100.0 * (i + 1),
                    outcome_behavior=None if i % 2 else {"action": "sold_all", "lag_days": 12, "amount_pct": 1.0},
                    is_synthetic=is_synthetic,
                    source_row_ref=f"row:{i}",
                )
            )
        )
    return events


@pytest.fixture
def frame() -> pd.DataFrame:
    return S.events_to_frame(make_events())


# ---------------------------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------------------------


def test_columns_arrow_schema_and_dtypes_agree():
    assert len(S.COLUMNS) == 21
    assert S.COLUMNS[0] == "subject_id" and S.COLUMNS[-1] == "source_row_ref"
    assert [f.name for f in S.ARROW_SCHEMA] == S.COLUMNS
    assert S.ARROW_SCHEMA.field("context_tags").type == pa.list_(pa.string())
    assert S.ARROW_SCHEMA.field("prior_question_ids").type == pa.list_(pa.string())
    assert not S.ARROW_SCHEMA.field("response").nullable
    assert S.ARROW_SCHEMA.field("outcome_behavior").nullable
    assert set(S.PANDAS_DTYPES) == set(S.COLUMNS)
    assert S.ELICITATION_TYPES == ("binary_sell", "allocation_pct", "likert", "lottery_choice")
    assert S.KEY_COLUMNS == ["dataset", "subject_id", "session_id", "position_in_session"]


def test_empty_frame_has_canonical_columns_and_dtypes():
    df = S.empty_frame()
    assert list(df.columns) == S.COLUMNS
    assert len(df) == 0
    assert str(df["horizon_days"].dtype) == "Int64"
    assert df["is_synthetic"].dtype == np.dtype(bool)
    assert isinstance(df["subject_id"].dtype, pd.StringDtype)
    report = S.validate_frame(df, strict=False)
    assert report.ok and "frame has no rows" in report.warnings[0]


# ---------------------------------------------------------------------------------------------
# Row model
# ---------------------------------------------------------------------------------------------


def test_event_serializes_dicts_to_canonical_json_and_exposes_key():
    ev = DecisionEvent(**base_kwargs(outcome_behavior={"lag_days": 3, "action": "sold_part"}))
    assert ev.covariates == '{"age_band":"30-44","invest_experience_yrs":3,"wealth_band":"<50k"}'
    assert ev.outcome_behavior == '{"action":"sold_part","lag_days":3}'
    assert ev.covariates_dict["invest_experience_yrs"] == 3
    assert ev.outcome_behavior_dict == {"action": "sold_part", "lag_days": 3}
    assert ev.key == ("synthetic_gq", "s001", "sess-1", 0)
    assert list(ev.to_row()) == S.COLUMNS


def test_event_accepts_json_strings_numpy_scalars_arrays_and_datetimes():
    ev = DecisionEvent(
        **base_kwargs(
            covariates='{"education": "graduate", "financial_literacy_score": 4}',
            outcome_behavior='{"action": "held", "lag_days": 0.5}',
            context_tags=np.array(["news:recession"], dtype=object),
            prior_question_ids=("tolerance",),
            position_in_session=np.int64(2),
            response=np.float64(0.0),
            loss_pct=np.nan,
            horizon_days=np.nan,
            response_time_ms=None,
            timestamp=pd.Timestamp("2026-09-17T03:19:00Z"),
        )
    )
    assert ev.context_tags == ["news:recession"] and ev.prior_question_ids == ["tolerance"]
    assert ev.position_in_session == 2 and ev.loss_pct is None and ev.horizon_days is None
    assert ev.timestamp == "2026-09-17T03:19:00+00:00"
    assert json.loads(ev.covariates) == {"education": "graduate", "financial_literacy_score": 4}


@pytest.mark.parametrize(
    "override, fragment",
    [
        ({"subject_id": ""}, "subject_id"),
        ({"dataset": ""}, "dataset"),
        ({"session_id": ""}, "session_id"),
        ({"timestamp": "yesterday"}, "ISO-8601"),
        ({"position_in_session": -1}, "position_in_session"),
        ({"position_in_session": True}, "not a bool"),
        ({"position_in_session": 1.5}, "integer-valued"),
        ({"scenario_id": ""}, "scenario_id"),
        ({"loss_pct": 1.5}, "loss_pct"),
        ({"loss_pct": -1.01}, "loss_pct"),
        ({"loss_pct": "big"}, "loss_pct"),
        ({"horizon_days": -1}, "horizon_days"),
        ({"context_tags": ["recession"]}, "namespace:value"),
        ({"context_tags": ["news:recession", ""]}, "non-empty"),
        ({"context_tags": "news:recession"}, "list of strings"),
        ({"context_tags": None}, "not null"),
        ({"prior_question_ids": [""]}, "non-empty"),
        ({"prior_question_ids": None}, "not null"),
        ({"elicitation_type": "slider"}, "elicitation_type"),
        ({"response": float("nan")}, "response"),
        ({"response": True}, "not a bool"),
        ({"response_time_ms": -1.0}, "response_time_ms"),
        ({"covariates": '["a"]'}, "JSON object"),
        ({"covariates": "{not json"}, "not valid JSON"),
        ({"covariates": {"gender": "f"}}, "unknown key"),
        ({"covariates": {"age_band": "35-44"}}, "violates"),
        ({"covariates": {"invest_experience_yrs": 41}}, "violates"),
        ({"covariates": {"self_reported_risk_tolerance": 2.5}}, "violates"),
        ({"covariates": {"financial_literacy_score": True}}, "violates"),
        ({"outcome_behavior": {"action": "sold"}}, "lag_days"),
        ({"outcome_behavior": {"action": "", "lag_days": 1}}, "action"),
        ({"outcome_behavior": {"action": "sold", "lag_days": -2}}, "lag_days"),
        ({"outcome_behavior": "[1, 2]"}, "JSON object"),
        ({"incentivized": 1}, "must be a bool"),
        ({"consent_training": "yes"}, "must be a bool"),
        ({"is_synthetic": None}, "must be a bool"),
        ({"battery_version": ""}, "battery_version"),
        ({"source_row_ref": ""}, "source_row_ref"),
        ({"extra_column": 1}, "extra_column"),
    ],
)
def test_event_rejects_rule_violations(override, fragment):
    with pytest.raises(ValidationError) as excinfo:
        DecisionEvent(**base_kwargs(**override))
    assert fragment in str(excinfo.value)


@pytest.mark.parametrize(
    "elicitation_type, good, bad",
    [
        ("binary_sell", [0, 1, 1.0], [0.5, 2, -1]),
        ("lottery_choice", [0.0, 1], [0.25, 3]),
        ("allocation_pct", [0.0, 0.5, 1.0], [-0.01, 1.01, 7]),
        ("likert", [1, 4, 7.0], [0, 8, 2.5, 0.5]),
    ],
)
def test_response_range_per_elicitation_type(elicitation_type, good, bad):
    for r in good:
        ev = DecisionEvent(**base_kwargs(elicitation_type=elicitation_type, response=r))
        assert ev.response == float(r)
        assert S.response_in_range(elicitation_type, r)
    for r in bad:
        with pytest.raises(ValidationError, match="out of range"):
            DecisionEvent(**base_kwargs(elicitation_type=elicitation_type, response=r))
        assert not S.response_in_range(elicitation_type, r)


def test_missing_required_fields_are_errors():
    kw = base_kwargs()
    for required in ("covariates", "incentivized", "consent_training", "is_synthetic", "response", "elicitation_type"):
        bad = {k: v for k, v in kw.items() if k != required}
        with pytest.raises(ValidationError):
            DecisionEvent(**bad)


# ---------------------------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------------------------


def test_valid_frame_passes_and_is_not_mutated(frame):
    before = frame.copy(deep=True)
    report = S.validate_frame(frame, expect_synthetic=True)
    assert report.ok
    assert report.n_rows == 6
    assert report.duplicate_keys == 0 and not report.mixed_synthetic
    assert report.nulls_per_column["timestamp"] == 2
    assert report.nulls_per_column["outcome_behavior"] == 3
    assert report.nulls_per_column["loss_pct"] == 1
    assert all(v == 0 for v in report.range_violations.values())
    assert report.errors == [] and report.warnings == []
    pd.testing.assert_frame_equal(frame, before)


def test_events_to_frame_dtypes_are_canonical(frame):
    assert list(frame.columns) == S.COLUMNS
    for c, dt in S.PANDAS_DTYPES.items():
        assert str(frame[c].dtype) == str(pd.Series([], dtype=dt).dtype), c
    assert frame["context_tags"].map(type).eq(list).all()
    assert frame["horizon_days"].isna().sum() == 1
    events = S.frame_to_events(frame)
    assert [e.to_row() for e in events] == [e.to_row() for e in make_events()]


def _corrupt(frame: pd.DataFrame, column: str, row: int, value) -> pd.DataFrame:
    df = frame.copy()
    df[column] = df[column].astype("object")
    df.loc[df.index[row], column] = value
    return df


@pytest.mark.parametrize(
    "column, value",
    [
        ("subject_id", ""),
        ("timestamp", "not-a-date"),
        ("position_in_session", -3),
        ("loss_pct", 2.0),
        ("horizon_days", -365),
        ("context_tags", ["no_namespace"]),
        ("context_tags", "news:recession"),
        ("prior_question_ids", [1]),
        ("elicitation_type", "unknown"),
        ("response", 0.5),
        ("response_time_ms", -5.0),
        ("covariates", '{"age_band": "child"}'),
        ("covariates", "oops"),
        ("outcome_behavior", '{"action": "sold"}'),
        ("incentivized", "no"),
        ("is_synthetic", 1),
        ("battery_version", ""),
        ("source_row_ref", 7),
    ],
)
def test_validate_frame_counts_each_rule_violation(frame, column, value):
    df = _corrupt(frame, column, 0, value)
    with pytest.raises(SchemaError) as excinfo:
        S.validate_frame(df)
    report = excinfo.value.report
    assert report.range_violations[column] == 1, report.errors
    assert any(err.startswith(column) for err in report.errors)
    assert S.validate_frame(df, strict=False).ok is False


@pytest.mark.parametrize("column", ["subject_id", "position_in_session", "context_tags", "response", "covariates", "is_synthetic"])
def test_validate_frame_flags_nulls_in_non_nullable_columns(frame, column):
    df = _corrupt(frame, column, 1, None)
    report = S.validate_frame(df, strict=False)
    assert not report.ok
    assert report.nulls_per_column[column] == 1
    assert any("null" in e and e.startswith(column) for e in report.errors)


def test_validate_frame_allows_nulls_in_nullable_columns(frame):
    df = frame.copy()
    for c in S.NULLABLE_COLUMNS:
        df[c] = df[c].astype("object")
        df.loc[df.index[0], c] = None
    report = S.validate_frame(df)
    assert report.ok and all(report.nulls_per_column[c] >= 1 for c in S.NULLABLE_COLUMNS)


def test_validate_frame_catches_duplicate_keys(frame):
    df = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = S.validate_frame(df, strict=False)
    assert report.duplicate_keys == 1
    assert any("duplicate" in e for e in report.errors)
    # a different session makes the key unique again
    df.loc[df.index[-1], "session_id"] = "sess-2"
    assert S.validate_frame(df).ok


def test_validate_frame_catches_mixed_and_unexpected_is_synthetic(frame):
    df = frame.copy()
    df.loc[df.index[0], "is_synthetic"] = False
    report = S.validate_frame(df, strict=False)
    assert report.mixed_synthetic and any("mixes" in e for e in report.errors)
    report = S.validate_frame(frame, expect_synthetic=False, strict=False)
    assert not report.mixed_synthetic
    assert any("requires False" in e for e in report.errors)
    assert S.validate_frame(frame, expect_synthetic=True).ok


def test_validate_frame_column_checks(frame):
    with pytest.raises(SchemaError, match="missing columns"):
        S.validate_frame(frame.drop(columns=["response"]))
    with pytest.raises(SchemaError, match="unexpected columns"):
        S.validate_frame(frame.assign(extra=1))
    shuffled = frame[list(reversed(S.COLUMNS))]
    report = S.validate_frame(shuffled)
    assert report.ok and any("canonical order" in w for w in report.warnings)


def test_validate_frame_warns_on_missing_training_consent(frame):
    df = frame.copy()
    df.loc[df.index[:2], "consent_training"] = False
    report = S.validate_frame(df)
    assert report.ok and any("2 row(s) have consent_training=False" in w for w in report.warnings)


def test_validate_frame_accepts_dicts_and_arrays_before_conforming(frame):
    df = frame.copy()
    df["covariates"] = df["covariates"].map(json.loads)
    df["context_tags"] = df["context_tags"].map(np.array)
    assert S.validate_frame(df, expect_synthetic=True).ok
    conformed = S.conform_frame(df)
    pd.testing.assert_frame_equal(conformed, frame)


def test_validation_report_markdown_and_ok():
    report = ValidationReport(n_rows=3, nulls_per_column={"timestamp": 1}, range_violations={"response": 2}, duplicate_keys=1, errors=["response: 2 value(s) violate"], warnings=["w1"])
    assert not report.ok
    md = report.to_markdown()
    assert "| response |" in md and "| 2 |" in md and "- response: 2 value(s) violate" in md and "- w1" in md
    assert ValidationReport().ok


# ---------------------------------------------------------------------------------------------
# pandas 3 behaviour (copy-on-write, str dtype)
# ---------------------------------------------------------------------------------------------


def test_pandas3_str_dtype_frames_with_nan_nulls_validate(frame):
    # A frame built the pandas-3 way: default str dtype, NaN for missing strings, Int64 for ints.
    df = pd.DataFrame(frame.to_dict("records"))
    assert isinstance(df["subject_id"].dtype, pd.StringDtype)
    df["timestamp"] = df["timestamp"].astype("str")
    assert df["timestamp"].isna().sum() == 2
    report = S.validate_frame(df, expect_synthetic=True)
    assert report.ok
    # copy-on-write: a slice validates on its own and the parent is untouched
    view = df.iloc[:3]
    assert S.validate_frame(view, expect_synthetic=True).ok
    conformed = S.conform_frame(view)
    conformed.loc[conformed.index[0], "subject_id"] = "changed"
    assert df.loc[df.index[0], "subject_id"] == "s000"
    pd.testing.assert_frame_equal(S.conform_frame(df), frame)


# ---------------------------------------------------------------------------------------------
# Synthetic-path rule and parquet round trip
# ---------------------------------------------------------------------------------------------


def test_is_synthetic_path(tmp_path):
    assert S.is_synthetic_path(tmp_path / "data" / "synthetic" / "gq.parquet")
    assert S.is_synthetic_path(tmp_path / "data" / "synthetic" / "sub" / "gq.parquet")
    assert not S.is_synthetic_path(tmp_path / "data" / "processed" / "cpc18.parquet")
    assert not S.is_synthetic_path(tmp_path / "synthetic" / "data" / "x.parquet")
    assert S.SYNTHETIC_DIR == S.PROJECT_ROOT / "data" / "synthetic"
    assert S.PROJECT_ROOT.name == "bre"


def test_synthetic_table_may_only_be_written_under_data_synthetic(tmp_path, frame):
    ok_path = tmp_path / "data" / "synthetic" / "gq.parquet"
    report = S.write_events(frame, ok_path)
    assert report.ok and ok_path.exists()
    md = S.validation_report_path(ok_path)
    assert md == ok_path.with_name("gq.validation.md") and "is_synthetic: True" in md.read_text()

    bad_path = tmp_path / "data" / "processed" / "gq.parquet"
    with pytest.raises(SchemaError, match="requires False"):
        S.write_events(frame, bad_path)
    assert not bad_path.exists() and not S.validation_report_path(bad_path).exists()


def test_real_table_never_under_data_synthetic_and_no_mixing(tmp_path):
    real = S.events_to_frame(make_events(is_synthetic=False, dataset="cpc18"))
    with pytest.raises(SchemaError, match="requires True"):
        S.write_events(real, tmp_path / "data" / "synthetic" / "cpc18.parquet")
    report = S.write_events(real, tmp_path / "data" / "processed" / "cpc18.parquet")
    assert report.ok
    mixed = pd.concat([real, S.events_to_frame(make_events(is_synthetic=True))], ignore_index=True)
    for target in ("data/synthetic/mixed.parquet", "data/processed/mixed.parquet"):
        with pytest.raises(SchemaError, match="mixes True and False"):
            S.write_events(mixed, tmp_path / target)


def test_write_events_rejects_invalid_frames_without_writing(tmp_path, frame):
    path = tmp_path / "data" / "synthetic" / "bad.parquet"
    with pytest.raises(SchemaError):
        S.write_events(_corrupt(frame, "response", 0, 9.0), path)
    assert not path.exists()
    with pytest.raises(SchemaError, match="parquet"):
        S.write_events(frame, tmp_path / "data" / "synthetic" / "bad.csv")


def test_parquet_round_trip_preserves_lists_json_and_nulls(tmp_path, frame):
    path = tmp_path / "data" / "synthetic" / "roundtrip.parquet"
    S.write_events(frame, path)
    stored = pq.read_schema(path)
    assert stored.field("context_tags").type == pa.list_(pa.string())
    assert stored.metadata[b"schema_version"] == S.SCHEMA_VERSION.encode()

    back = S.read_events(path)
    pd.testing.assert_frame_equal(back, frame)
    assert back["context_tags"].tolist() == frame["context_tags"].tolist()
    assert back["context_tags"].map(type).eq(list).all()
    assert back["prior_question_ids"].tolist()[0] == ["L05|none"] and back["prior_question_ids"].tolist()[1] == []
    assert json.loads(back["outcome_behavior"].iloc[0]) == {"action": "sold_all", "amount_pct": 1.0, "lag_days": 12}
    assert pd.isna(back["outcome_behavior"].iloc[1])
    assert back["outcome_behavior"].isna().tolist() == frame["outcome_behavior"].isna().tolist()
    assert str(back["horizon_days"].dtype) == "Int64" and back["horizon_days"].isna().sum() == 1
    assert [e.to_row() for e in S.frame_to_events(back)] == [e.to_row() for e in make_events()]

    md = S.validation_report_path(path).read_text()
    assert "sha256:" in md and "- rows: 6" in md and "- ok: True" in md


def test_read_events_rejects_foreign_or_misplaced_tables(tmp_path, frame):
    foreign = tmp_path / "data" / "processed" / "foreign.parquet"
    foreign.parent.mkdir(parents=True)
    pq.write_table(pa.table({"a": [1, 2]}), foreign)
    with pytest.raises(SchemaError, match="do not match"):
        S.read_events(foreign)
    # a synthetic table copied outside data/synthetic is refused on read
    src = tmp_path / "data" / "synthetic" / "gq.parquet"
    S.write_events(frame, src)
    moved = tmp_path / "data" / "processed" / "gq.parquet"
    moved.write_bytes(src.read_bytes())
    with pytest.raises(SchemaError, match="requires False"):
        S.read_events(moved)


def test_empty_frame_round_trip(tmp_path):
    path = tmp_path / "data" / "synthetic" / "empty.parquet"
    report = S.write_events(S.empty_frame(), path)
    assert report.ok and "frame has no rows" in report.warnings
    back = S.read_events(path)
    assert list(back.columns) == S.COLUMNS and len(back) == 0
