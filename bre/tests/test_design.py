"""Tests for bre.design: PLAN.md section 3 counts, wording, covariate spec and battery subsets."""

from __future__ import annotations

import json
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from bre import design as D
from bre import schema as S

# ---------------------------------------------------------------------------------------------
# Constants of PLAN.md section 3
# ---------------------------------------------------------------------------------------------


def test_plan_constants():
    assert D.LOSS_PCTS == (-0.05, -0.10, -0.15, -0.20, -0.30)
    assert D.HORIZON_DAYS == 365
    assert D.CONTEXTS == ("none", "news:recession", "news:technical", "social:friend_sells", "market:recovered_5pct")
    assert D.CONTEXTS[0] == D.CONTEXT_NONE
    assert D.NONNULL_CONTEXTS == D.CONTEXTS[1:]
    assert set(D.QUESTION_ORDERS) == {"tolerance-first", "scenario-first"}
    assert D.QUESTION_ORDER_IDS == (D.TOLERANCE_FIRST, D.SCENARIO_FIRST) == ("tolerance-first", "scenario-first")
    assert D.QUESTION_ORDERS["tolerance-first"] == ("tolerance", "sell")
    assert D.QUESTION_ORDERS["scenario-first"] == ("sell", "tolerance")
    assert D.DESIGN_VERSION == "1.0.0"
    assert D.TOLERANCE_QUESTION == (
        "Would you describe yourself as someone who avoids investment losses even at the cost of "
        "lower returns?"
    )
    assert D.N_ITEMS == 170 and D.N_CONDITIONS == 17
    assert all(S.TAG_PATTERN.fullmatch(c) for c in D.NONNULL_CONTEXTS)


def test_context_conditions_are_the_17_ordered_tuples():
    conds = D.context_conditions()
    assert len(conds) == 17
    assert len(set(conds)) == 17
    assert conds[0] == ()
    singles = [c for c in conds if len(c) == 1]
    pairs = [c for c in conds if len(c) == 2]
    assert [c[0] for c in singles] == list(D.NONNULL_CONTEXTS)
    assert len(pairs) == 12
    assert set(pairs) == set(permutations(D.NONNULL_CONTEXTS, 2))
    assert all(a != b for a, b in pairs)
    for a, b in pairs:  # both orders of every pair are present: order effects are testable
        assert (b, a) in pairs
    assert all(len(c) <= 2 for c in conds)


def test_condition_ids_round_trip():
    for cond in D.context_conditions():
        cid = D.condition_id(cond)
        assert D.condition_from_id(cid) == cond
    assert D.condition_id(()) == "none"
    assert D.condition_id(("news:recession", "social:friend_sells")) == "rec>friend"
    assert len({D.condition_id(c) for c in D.context_conditions()}) == 17
    with pytest.raises(ValueError):
        D.condition_id(("none",))
    with pytest.raises(ValueError):
        D.condition_from_id("rec>bogus")


# ---------------------------------------------------------------------------------------------
# Wording
# ---------------------------------------------------------------------------------------------


def test_render_sell_question_contains_loss_contexts_in_order_and_recovery():
    text = D.render_sell_question(-0.20, ("social:friend_sells", "market:recovered_5pct"))
    assert "fallen by 20%" in text and "$80,000" in text and "one year" in text
    i_friend = text.index(D.CONTEXT_SENTENCES["social:friend_sells"])
    i_recov = text.index(D.CONTEXT_SENTENCES["market:recovered_5pct"])
    assert i_friend < i_recov
    assert "regained 5%" in text
    assert text.endswith(D.SELL_PROMPT)
    reversed_text = D.render_sell_question(-0.20, ("market:recovered_5pct", "social:friend_sells"))
    assert reversed_text != text and sorted(reversed_text) == sorted(text)


def test_render_sell_question_none_condition_and_cause_frames():
    none_text = D.render_sell_question(-0.05)
    assert not any(s in none_text for s in D.CONTEXT_SENTENCES.values())
    assert "fallen by 5%" in none_text and "$95,000" in none_text
    rec = D.render_sell_question(-0.10, ("news:recession",))
    tech = D.render_sell_question(-0.10, ("news:technical",))
    assert "recession" in rec and "recession" not in tech
    assert "outage" in tech and "outage" not in rec
    assert D.SELL_RESPONSE_OPTIONS == {1: "Sell", 0: "Hold"}
    assert D.TOLERANCE_RESPONSE_OPTIONS == {1: "Yes", 0: "No"}
    assert set(D.CONTEXT_SENTENCES) == set(D.NONNULL_CONTEXTS)
    assert "{loss_sentence}" in D.SELL_QUESTION and "{context_sentences}" in D.SELL_QUESTION


def test_render_sell_question_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        D.render_sell_question(0.10)
    with pytest.raises(ValueError):
        D.render_sell_question(-0.10, ("news:war",))


def test_horizon_text():
    assert D.horizon_text(365) == "one year"
    assert D.horizon_text(730) == "2 years"
    assert D.horizon_text(90) == "3 months"
    assert D.horizon_text(10) == "10 days"
    with pytest.raises(ValueError):
        D.horizon_text(-1)


# ---------------------------------------------------------------------------------------------
# Full design
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def design() -> pd.DataFrame:
    return D.full_design()


def test_full_design_counts(design):
    assert len(design) == 170
    assert list(design.columns) == list(D.DESIGN_COLUMNS)
    assert design["item_id"].is_unique
    assert design["scenario_id"].nunique() == 85
    assert design["context_condition_id"].nunique() == 17
    assert design["loss_pct"].nunique() == 5 and set(design["loss_pct"]) == set(D.LOSS_PCTS)
    assert design["question_order_id"].value_counts().to_dict() == {"tolerance-first": 85, "scenario-first": 85}
    assert design.groupby("loss_pct").size().eq(34).all()
    assert design.groupby("context_condition_id").size().eq(10).all()
    assert design["n_contexts"].value_counts().to_dict() == {2: 120, 1: 40, 0: 10}
    assert (design["horizon_days"] == 365).all()
    assert (design["elicitation_type"] == "binary_sell").all()
    assert design["question_text"].nunique() == 85


def test_full_design_rows_are_internally_consistent(design):
    for row in design.itertuples(index=False):
        assert tuple(row.context_tags) == D.condition_from_id(row.context_condition_id)
        assert row.scenario_id == D.scenario_id(row.loss_pct, row.context_tags)
        assert row.item_id == f"{row.scenario_id}|{row.question_order_id}"
        expected_prior = ["tolerance"] if row.question_order_id == D.TOLERANCE_FIRST else []
        assert row.prior_question_ids == expected_prior
        assert row.question_text == D.render_sell_question(row.loss_pct, row.context_tags)
        assert row.n_contexts == len(row.context_tags)


def test_full_design_items_form_valid_decision_events(design, tmp_path):
    """One synthetic subject answering the whole design passes the schema and round-trips."""
    rng = np.random.default_rng(0)
    cov = D.sample_covariates(rng, 1).iloc[0].to_dict()
    events = []
    for pos, row in enumerate(design.itertuples(index=False)):
        events.append(
            S.DecisionEvent(
                subject_id="s000",
                dataset="synthetic_design",
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
                is_synthetic=True,
                source_row_ref=row.item_id,
            )
        )
    frame = S.events_to_frame(events)
    report = S.write_events(frame, tmp_path / "data" / "synthetic" / "design_check.parquet")
    assert report.ok and report.n_rows == 170
    back = S.read_events(tmp_path / "data" / "synthetic" / "design_check.parquet")
    assert back["scenario_id"].nunique() == 85
    assert back["context_tags"].map(len).value_counts().to_dict() == {2: 120, 1: 40, 0: 10}


# ---------------------------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------------------------


def test_covariate_spec_matches_plan():
    spec = D.COVARIATE_SPEC
    assert tuple(spec) == (
        "age_band",
        "wealth_band",
        "invest_experience_yrs",
        "self_reported_risk_tolerance",
        "financial_literacy_score",
        "education",
    )
    assert spec["age_band"].levels == ("18-29", "30-44", "45-59", "60+")
    assert spec["wealth_band"].levels == ("<50k", "50-250k", "250k-1M", ">1M")
    assert (spec["invest_experience_yrs"].lo, spec["invest_experience_yrs"].hi) == (0, 40)
    assert (spec["self_reported_risk_tolerance"].lo, spec["self_reported_risk_tolerance"].hi) == (1, 7)
    assert (spec["financial_literacy_score"].lo, spec["financial_literacy_score"].hi) == (0, 5)
    assert spec["education"].levels == ("hs", "some_college", "bachelor", "graduate")
    assert spec["age_band"].is_valid("60+") and not spec["age_band"].is_valid("70+")
    assert spec["invest_experience_yrs"].is_valid(40) and not spec["invest_experience_yrs"].is_valid(40.5)
    assert not spec["financial_literacy_score"].is_valid(True)
    assert D.COVARIATE_MARGINALS_SOURCE == "placeholder-uniform; replace with FINRA NFCS Investor Survey fit"
    for name, marginal in D.COVARIATE_MARGINALS.items():
        assert abs(sum(marginal.values()) - 1.0) < 1e-12
        if spec[name].kind == "categorical":
            assert tuple(marginal) == spec[name].levels
        else:
            assert [int(k) for k in marginal] == list(range(int(spec[name].lo), int(spec[name].hi) + 1))


def test_sample_covariates_respects_spec_and_seed():
    a = D.sample_covariates(np.random.default_rng(42), 500)
    b = D.sample_covariates(np.random.default_rng(42), 500)
    c = D.sample_covariates(np.random.default_rng(43), 500)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    assert list(a.columns) == list(D.COVARIATE_KEYS) and len(a) == 500
    for name, spec in D.COVARIATE_SPEC.items():
        assert a[name].map(spec.is_valid).all(), name
        if spec.kind == "categorical":
            assert set(a[name]) == set(spec.levels)  # every band drawn at n=500
    for record in a.head(20).to_dict("records"):
        cleaned = S.validate_covariates(record)
        assert cleaned == record
        json.dumps(cleaned)
    assert D.sample_covariates(7, 0).shape == (0, 6)
    with pytest.raises(ValueError):
        D.sample_covariates(7, -1)


# ---------------------------------------------------------------------------------------------
# Battery subset
# ---------------------------------------------------------------------------------------------


def _assert_balanced(subset: pd.DataFrame, record: D.BatteryAssignment, n: int, form: str, design: pd.DataFrame):
    assert len(subset) == n and record.n_items == n and record.form == form
    assert record.design_version == D.DESIGN_VERSION
    assert subset["position"].tolist() == list(range(n))
    assert subset["item_id"].is_unique
    assert set(subset["item_id"]) <= set(design["item_id"])
    assert record.item_ids == subset["item_id"].tolist()
    assert record.condition_ids == subset["context_condition_id"].tolist()
    # conditions: none once, each single once, remaining items distinct pairs
    assert subset["context_condition_id"].is_unique
    assert record.condition_class_counts == {"none": 1, "single": 4, "pair": n - 5}
    # losses balanced to within one item
    loss_counts = subset["loss_pct"].value_counts()
    assert set(loss_counts.index) == set(D.LOSS_PCTS)
    assert loss_counts.max() - loss_counts.min() <= 1
    assert sum(record.loss_counts.values()) == n and max(record.loss_counts.values()) - min(record.loss_counts.values()) <= 1
    # orders balanced to within one item
    orders = subset["question_order_id"].value_counts().to_dict()
    assert set(orders) == {"tolerance-first", "scenario-first"}
    assert abs(orders["tolerance-first"] - orders["scenario-first"]) <= 1
    assert record.order_counts == orders
    # first-position contexts among the pairs balanced to within one item
    pairs = subset[subset["n_contexts"] == 2]
    if len(pairs):
        firsts = pairs["context_tags"].map(lambda t: t[0]).value_counts().reindex(D.NONNULL_CONTEXTS, fill_value=0)
        assert firsts.max() - firsts.min() <= 1
    # every context appears at least once (as a single) and the counts in the record are right
    counts = {c: 0 for c in D.NONNULL_CONTEXTS}
    for tags in subset["context_tags"]:
        for t in tags:
            counts[t] += 1
    assert record.context_counts == counts and min(counts.values()) >= 1


@pytest.mark.parametrize("n", [12, 13, 14, 15, 16])
def test_battery_subset_full_form_is_balanced(n, design):
    for seed in (0, 1, 2):
        subset, record = D.battery_subset(np.random.default_rng(seed), n, form="full")
        _assert_balanced(subset, record, n, "full", design)


def test_battery_subset_short_form(design):
    subset, record = D.battery_subset(np.random.default_rng(5), 5, form="short")
    _assert_balanced(subset, record, 5, "short", design)
    assert subset["loss_pct"].value_counts().eq(1).all()  # every loss once
    assert (subset["n_contexts"] <= 1).all()  # none + the four singles


def test_battery_subset_is_reproducible_under_fixed_seed():
    s1, r1 = D.battery_subset(np.random.default_rng(2026), 14)
    s2, r2 = D.battery_subset(np.random.default_rng(2026), 14)
    s3, r3 = D.battery_subset(2026, 14)  # int seed is accepted too
    pd.testing.assert_frame_equal(s1, s2)
    pd.testing.assert_frame_equal(s1, s3)
    assert r1.to_dict() == r2.to_dict() == r3.to_dict()
    json.dumps(r1.to_dict())
    s4, r4 = D.battery_subset(np.random.default_rng(2027), 14)
    assert r4.item_ids != r1.item_ids


def test_battery_subset_randomizes_across_seeds():
    ids = [tuple(D.battery_subset(np.random.default_rng(seed), 12)[1].item_ids) for seed in range(6)]
    assert len(set(ids)) == 6


@pytest.mark.parametrize("n, form", [(11, "full"), (17, "full"), (4, "short"), (6, "short"), (5, "full")])
def test_battery_subset_rejects_out_of_range_sizes(n, form):
    with pytest.raises(ValueError):
        D.battery_subset(np.random.default_rng(0), n, form=form)


def test_battery_subset_rejects_unknown_form():
    with pytest.raises(ValueError):
        D.battery_subset(np.random.default_rng(0), 12, form="long")
