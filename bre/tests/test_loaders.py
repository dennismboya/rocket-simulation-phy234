"""Tests for the Phase 1 loaders (src/bre/loaders) and the data CLI.

Real-file tests compute every asserted count from the raw file inside the test (nothing is typed
in); rule tests use hand-built examples. Tests that need the raw copies skip when
``data/raw/<dataset>`` is absent.
"""

from __future__ import annotations

import gzip
import json
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from bre import data_cli
from bre import schema as S
from bre.loaders import RAW_DIR, choices13k, cpc15, cpc18, intake_battery, order_effects, psych201

BATTERY_JSON = S.PROJECT_ROOT / "instrument" / "battery.json"


def _needs(*parts: str):
    p = RAW_DIR.joinpath(*parts)
    return pytest.mark.skipif(not p.exists(), reason=f"raw file {p} not present")


def _ok(df: pd.DataFrame) -> S.ValidationReport:
    report = S.validate_frame(df, expect_synthetic=False, strict=False)
    assert report.ok, report.errors
    return report


def _common_checks(df: pd.DataFrame, dataset_prefix: str, incentivized: bool) -> None:
    assert list(df.columns) == S.COLUMNS
    assert df["dataset"].str.startswith(dataset_prefix).all()
    assert (~df["is_synthetic"]).all()
    assert (df["incentivized"] == incentivized).all()
    assert (~df["consent_training"]).all()
    assert (df["battery_version"] == "external").all()
    assert df["outcome_behavior"].isna().all()
    assert df["source_row_ref"].str.contains(":").all()


# ---------------------------------------------------------------------------------------------
# choices13k
# ---------------------------------------------------------------------------------------------


@_needs("choices13k", "c13k_selections.csv")
def test_choices13k_counts_mapping_and_validation():
    sel = pd.read_csv(RAW_DIR / "choices13k" / choices13k.SELECTIONS_FILE)
    feats = pd.read_csv(RAW_DIR / "choices13k" / choices13k.FEATURES_FILE, index_col=0)
    with open(RAW_DIR / "choices13k" / choices13k.PROBLEMS_FILE, encoding="utf-8") as fh:
        problems = json.load(fh)
    n_dom = int((feats["Dom"].fillna(0) != 0).sum())  # loader default: first-order dominance (CATALOG A)
    n_any = int((feats["StochDom"].fillna(0) != 0).sum())

    df, meta = choices13k.load_choices13k_with_meta()
    assert len(df) == len(sel) - n_dom
    assert sum(meta["dropped"].values()) == n_dom
    assert meta["n_raw_rows"] == len(sel)
    full = choices13k.load_choices13k(drop_dominated=False)
    assert len(full) == len(sel)
    strict = choices13k.load_choices13k(dominance_flag="StochDom")
    assert len(strict) == len(sel) - n_any
    _ok(df)
    _common_checks(df, "choices13k", incentivized=True)
    assert df["subject_id"].is_unique
    assert (df["elicitation_type"] == "choice_rate").all()
    assert df["loss_pct"].isna().all()

    # a sample of rows traces back to the CSV row named in source_row_ref
    for _, row in df.sample(50, random_state=0).iterrows():
        r = int(row["source_row_ref"].split(":")[1])
        cov = json.loads(row["covariates"])
        assert cov["weight"] == int(sel.loc[r, "n"])
        assert row["response"] == pytest.approx(sel.loc[r, "bRate"])
        assert row["scenario_id"] == str(sel.loc[r, "Problem"])
        fb = bool(sel.loc[r, "Feedback"])
        assert row["context_tags"] == ["feedback:on" if fb else "feedback:off", f"block:{int(sel.loc[r, 'Block'])}"]
        assert row["subject_id"] == f"agg:{sel.loc[r, 'Problem']}:{fb}:{int(sel.loc[r, 'Block'])}"
        outs = [o for opt in ("A", "B") for p, o in problems[str(r)][opt] if p > 0]
        assert cov["x_worst_outcome_rel"] == pytest.approx(min(outs) / max(abs(o) for o in outs))
        assert feats.loc[r, "Dom"] == 0


@_needs("choices13k", "c13k_selections.csv")
def test_choices13k_dominance_flag_option_and_bad_flag():
    feats = pd.read_csv(RAW_DIR / "choices13k" / choices13k.FEATURES_FILE, index_col=0)
    df, meta = choices13k.load_choices13k_with_meta(dominance_flag="Dom")
    n_first_order = int((feats["Dom"].fillna(0) != 0).sum())
    assert len(df) == len(feats) - n_first_order
    assert meta["dominance_flag_counts"]["Dom"] == n_first_order
    with pytest.raises(ValueError):
        choices13k.load_choices13k(dominance_flag="nope")


def test_worst_outcome_rel_hand_examples():
    assert choices13k.worst_outcome_rel({"A": [[0.5, 10], [0.5, -4]], "B": [[1.0, 2]]}) == pytest.approx(-0.4)
    # zero-probability outcomes are ignored
    assert choices13k.worst_outcome_rel({"A": [[1.0, 5], [0.0, -100]], "B": [[1.0, 10]]}) == pytest.approx(0.5)
    assert choices13k.worst_outcome_rel({"A": [[1.0, 0]], "B": [[1.0, 0]]}) == 0.0


# ---------------------------------------------------------------------------------------------
# CPC15
# ---------------------------------------------------------------------------------------------


@_needs("cpc15", "cpc15_thomas2024_aggregate.csv")
def test_cpc15_counts_mapping_and_validation():
    raw = pd.read_csv(RAW_DIR / "cpc15" / cpc15.RAW_FILE, index_col=0)
    df, meta = cpc15.load_cpc15_with_meta()
    assert len(df) == len(raw) == meta["n_raw_rows"]
    assert df["scenario_id"].nunique() == raw["GameID"].nunique()
    _ok(df)
    _common_checks(df, "cpc15", incentivized=True)
    assert df["subject_id"].is_unique
    for _, row in df.sample(30, random_state=1).iterrows():
        r = int(row["source_row_ref"].split(":")[1])
        assert row["response"] == pytest.approx(raw.loc[r, "Rate"])
        assert json.loads(row["covariates"]) == {"weight": None}
        assert row["context_tags"] == ["feedback:on" if raw.loc[r, "Feedback"] == 1 else "feedback:off", f"block:{int(raw.loc[r, 'Block'])}"]


# ---------------------------------------------------------------------------------------------
# CPC18
# ---------------------------------------------------------------------------------------------


def test_cpc18_distribution_hand_examples():
    assert cpc18.cpc18_distribution(10, 0.3, -5, "-", 1) == ((-5.0, 10.0), (0.7, 0.3))
    assert cpc18.cpc18_distribution(10, 1.0, -5, "-", 1) == ((10.0,), (1.0,))
    # Symm, 3 outcomes: H-1, H, H+1 with pH * (1/4, 1/2, 1/4), then L with 1 - pH
    outs, probs = cpc18.cpc18_distribution(10, 0.8, 0, "Symm", 3)
    assert outs == (0.0, 9.0, 10.0, 11.0)
    assert probs == pytest.approx((0.2, 0.2, 0.4, 0.2))
    # R-skew, 3 outcomes: H - 4 + (2, 4, 8) with pH * (1/2, 1/4, 1/8 * 2)
    outs, probs = cpc18.cpc18_distribution(10, 0.8, 0, "R-skew", 3)
    assert outs == (0.0, 8.0, 10.0, 14.0)
    assert probs == pytest.approx((0.2, 0.4, 0.2, 0.2))
    # L-skew mirrors R-skew around H
    outs, probs = cpc18.cpc18_distribution(10, 0.8, 0, "L-skew", 3)
    assert outs == (0.0, 6.0, 10.0, 12.0)
    assert probs == pytest.approx((0.2, 0.2, 0.2, 0.4))
    # L coinciding with a lottery outcome merges probabilities
    outs, probs = cpc18.cpc18_distribution(10, 0.5, 10, "Symm", 3)
    assert outs == (9.0, 10.0, 11.0)
    assert probs == pytest.approx((0.125, 0.75, 0.125))
    # probabilities sum to one and the lottery branch is centred on H for every shape and size
    for shape in ("Symm", "R-skew", "L-skew"):
        for n in range(2, 10):
            outs, probs = cpc18.cpc18_distribution(37, 0.6, -3, shape, n)
            assert sum(probs) == pytest.approx(1.0)
            mean = sum(o * p for o, p in zip(outs, probs))
            assert (mean - 0.4 * (-3)) / 0.6 == pytest.approx(37)
    with pytest.raises(ValueError):
        cpc18.cpc18_distribution(1, 0.5, 0, "Weird", 2)


def test_cpc18_safer_option_rule_hand_examples():
    # sure thing vs coin flip: A safer
    assert cpc18.safer_option(10, 1.0, 10, "-", 1, 50, 0.5, -50, "-", 1) == "A"
    # A: 100/0 at 50% (var 2500) vs B: 60/40 at 50% (var 100): B safer
    assert cpc18.safer_option(100, 0.5, 0, "-", 1, 60, 0.5, 40, "-", 1) == "B"
    # identical options: tie -> A
    assert cpc18.safer_option(5, 0.5, 0, "-", 1, 5, 0.5, 0, "-", 1) == "A"
    # same mean, B's high branch is a spread lottery: A safer
    assert cpc18.safer_option(10, 0.5, 0, "-", 1, 10, 0.5, 0, "Symm", 5) == "A"
    # variance of the moments helper on a known lottery
    mean, var, lo, hi = cpc18.option_moments(50, 0.5, -50, "-", 1)
    assert (mean, var, lo, hi) == (0.0, 2500.0, -50.0, 50.0)


@_needs("cpc18", "cpc18_calibration_raw.csv.gz")
def test_cpc18_subset_counts_rules_and_validation():
    with gzip.open(RAW_DIR / "cpc18" / cpc18.RAW_FILE, "rt") as fh:
        raw = pd.read_csv(fh, na_values=["NA"])
    first = raw["SubjID"].drop_duplicates().iloc[:5].tolist()
    expected_rows = int(raw["SubjID"].isin(first).sum())

    df, meta = cpc18.load_cpc18_with_meta(subjects=5, loss_norm="range")  # exercise the original range rule
    assert len(df) == expected_rows
    assert df["subject_id"].nunique() == 5
    _ok(df)
    _common_checks(df, "cpc18", incentivized=True)
    assert (df["elicitation_type"] == "lottery_choice").all()
    for _, g in df.groupby("subject_id"):
        assert g["position_in_session"].tolist() == list(range(len(g)))

    sub = raw[raw["SubjID"].isin(first)].sort_values(["SubjID", "Order", "Trial"], kind="stable")
    ptab = cpc18.problem_table(sub.drop_duplicates("GameID")).set_index("GameID")
    by_ref = df.set_index("source_row_ref")
    for subj, g in sub.groupby("SubjID"):
        games_by_order = g.drop_duplicates("Order").set_index("Order")["GameID"]
        for order, gg in g.groupby("Order"):
            gg = gg.sort_values("Trial")
            rows = [by_ref.loc[f"{cpc18.RAW_FILE}:{i}"] for i in gg.index]
            game = int(gg["GameID"].iloc[0])
            rng = float(ptab.loc[game, "outcome_range"])
            expected_prior = [str(int(games_by_order[o])) for o in range(max(1, order - 5), order)]
            for k, (i, t) in enumerate(zip(gg.index, gg.itertuples())):
                row = rows[k]
                assert row["session_id"] == f"set{t.Set}"
                assert row["scenario_id"] == str(game)
                assert row["prior_question_ids"] == expected_prior
                cov = json.loads(row["covariates"])
                assert cov["x_chose_b"] == int(t.B) and cov["x_gender"] == t.Gender and cov["x_location"] == t.Location
                chosen = "B" if t.B == 1 else "A"
                assert row["response"] == float(chosen == ptab.loc[game, "safer"])
                if k == 0:
                    assert np.isnan(row["loss_pct"])
                else:
                    prev = gg.iloc[k - 1]
                    assert row["loss_pct"] == pytest.approx(max(-1.0, min(1.0, prev["Payoff"] / rng)))
                tags = row["context_tags"]
                assert tags[-2:] == ["feedback:on" if t.Feedback == 1 else "feedback:off", f"block:{int(t.block)}"]
                exp_tags = [x for x in tags if x.startswith("exp:")]
                if k > 0 and gg.iloc[k - 1]["Feedback"] == 1:
                    assert exp_tags == ["exp:loss" if gg.iloc[k - 1]["Payoff"] < 0 else "exp:gain"]
                else:
                    assert exp_tags == []
            # trial 6 (first feedback trial) has no exp tag; trial 7 has one
            assert not [x for x in rows[5]["context_tags"] if x.startswith("exp:")]
            assert len([x for x in rows[6]["context_tags"] if x.startswith("exp:")]) == 1

    # pair variant: from trial 8 on two experienced outcomes in chronological order
    dfp = cpc18.load_cpc18(subjects=1, pairs=True)
    g1 = sub[sub["SubjID"] == first[0]]
    gg = g1[g1["Order"] == 1].sort_values("Trial")
    row8 = dfp.set_index("source_row_ref").loc[f"{cpc18.RAW_FILE}:{gg.index[7]}"]
    exp_tags = [x for x in row8["context_tags"] if x.startswith("exp:")]
    assert exp_tags == ["exp:loss" if p < 0 else "exp:gain" for p in gg["Payoff"].iloc[5:7]]
    row7 = dfp.set_index("source_row_ref").loc[f"{cpc18.RAW_FILE}:{gg.index[6]}"]
    assert len([x for x in row7["context_tags"] if x.startswith("exp:")]) == 1

    # max_abs normalization (the default) never needs clipping
    dfm, metam = cpc18.load_cpc18_with_meta(subjects=5)
    assert dfm["loss_pct"].equals(cpc18.load_cpc18(subjects=5, loss_norm="max_abs")["loss_pct"])
    assert metam["n_loss_pct_clipped"] == 0
    assert dfm["loss_pct"].abs().max() <= 1.0
    with pytest.raises(ValueError):
        cpc18.load_cpc18(subjects=1, loss_norm="nope")


@_needs("cpc18", "cpc18_calibration_raw.csv.gz")
def test_cpc18_full_load_matches_raw_rows_and_estset_block_rates():
    with gzip.open(RAW_DIR / "cpc18" / cpc18.RAW_FILE, "rt") as fh:
        n_raw = sum(1 for _ in fh) - 1
    df = cpc18.load_cpc18()
    assert len(df) == n_raw
    assert not df.duplicated(subset=S.KEY_COLUMNS).any()
    est = pd.read_csv(RAW_DIR / "cpc18" / "CPC18_EstSet210.csv").set_index("GameID")
    chose_b = df["covariates"].map(lambda s: json.loads(s)["x_chose_b"])
    block = df["context_tags"].map(lambda t: int(t[-1].split(":")[1]))
    rates = pd.DataFrame({"game": df["scenario_id"].astype(int), "block": block, "b": chose_b}).groupby(["game", "block"])["b"].mean().unstack("block")
    for k in range(1, 6):
        assert np.allclose(rates[k].reindex(est.index).to_numpy(), est[f"B.{k}"].to_numpy(), atol=1e-4)


# ---------------------------------------------------------------------------------------------
# Psych-201
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("folder", psych201.FOLDERS)
def test_psych201_roundtrip_counts_and_validation(folder):
    path = RAW_DIR / "psych201" / folder / psych201.ZIP_NAME
    if not path.exists():
        pytest.skip(f"{path} not present")
    with zipfile.ZipFile(path) as z, z.open(psych201.JSONL_NAME) as fh:
        records = [json.loads(ln) for ln in fh.read().decode("utf-8").splitlines() if ln.strip()]
    n_decisions = sum(r["text"].count("<<") for r in records)
    df, meta = psych201.load_psych201_with_meta(folder=folder)
    assert meta["roundtrip_fraction"] == 1.0
    assert meta["n_unrecognized_lines"] == 0
    assert meta["n_roundtrip_ok"] == meta["n_lines_checked"] > 0
    assert len(df) == n_decisions
    assert meta["dropped"] == {}
    assert df["subject_id"].nunique() == len(records)
    _ok(df)
    _common_checks(df, f"psych201:{folder}", incentivized=True)
    assert (df["elicitation_type"] == "lottery_choice").all()
    if folder == "spektor2024lossaversion":
        assert df["context_tags"].map(lambda t: sum(x.startswith("domain:") for x in t)).eq(1).all()
        assert df["context_tags"].map(lambda t: any(x.startswith("session:") for x in t)).all()
        assert df["response_time_ms"].notna().all()
        assert df["loss_pct"].notna().all()
        assert meta["n_rt_length_mismatch"] == 0
    else:
        assert df["context_tags"].map(lambda t: t[0] in ("options:2", "options:3")).all()
        assert df["loss_pct"].isna().all()
        assert df["response_time_ms"].isna().all()
        assert df["session_id"].str.len().gt(0).all()


SPEKTOR2024_TEXT = (
    "Dear participant,\nsome instructions.\nThe lotteries' names are S and Q.\n\n"
    "Session 1:\n"
    "The outcomes of lottery S are -9 with 50% chance and 16 with 50% chance. The outcomes of lottery Q are -15 with 50% chance and 22 with 50% chance. You choose <<Q>>.\n"
    "The outcomes of lottery S are 20 with 50% chance and 38 with 50% chance. The outcomes of lottery Q are 32 with 50% chance and 26 with 50% chance. You choose <<S>>.\n"
    "\nSession 2:\n"
    "The outcomes of lottery S are -14 with 50% chance and -12 with 50% chance. The outcomes of lottery Q are -8 with 50% chance and -18 with 50% chance. You choose <<S>>.\n"
    "The outcomes of lottery S are -3 with 50% chance and 3 with 50% chance. The outcomes of lottery Q are 3 with 50% chance and -3 with 50% chance. You choose <<Q>>."
)


def test_psych201_spektor2024_rules_on_hand_built_transcript(tmp_path):
    rec = {"text": SPEKTOR2024_TEXT, "experiment": "spektor2024lossaversion/exp9.csv", "participant": 7, "RTs": [100, 200, 300, 400]}
    folder = "spektor2024lossaversion"
    raw = tmp_path / "psych201" / folder
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / psych201.ZIP_NAME, "w") as z:
        z.writestr(psych201.JSONL_NAME, json.dumps(rec) + "\n")
    df, meta = psych201.load_psych201_with_meta(tmp_path, folder)
    assert meta["roundtrip_fraction"] == 1.0 and len(df) == 4
    _ok(df)
    assert df["subject_id"].tolist() == ["exp9:7"] * 4
    assert df["session_id"].tolist() == ["session1", "session1", "session2", "session2"]
    assert df["position_in_session"].tolist() == [0, 1, 0, 1]
    # variances: |a-b|: S 25 vs Q 37 -> S safer, chose Q -> 0; S 18 vs Q 6 -> Q safer, chose S -> 0;
    # S 2 vs Q 10 -> S safer, chose S -> 1; tie -> first-named S safer, chose Q -> 0
    assert df["response"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert [t[0] for t in df["context_tags"]] == ["domain:mixed", "domain:gain", "domain:loss", "domain:mixed"]
    assert [t[1] for t in df["context_tags"]] == ["session:1", "session:1", "session:2", "session:2"]
    # loss_pct = min / max|.| of the chosen lottery: Q(-15,22) -> -15/22; S(20,38) -> 20/38; S(-14,-12) -> -1; Q(3,-3) -> -1
    assert df["loss_pct"].tolist() == pytest.approx([-15 / 22, 20 / 38, -1.0, -1.0])
    assert df["response_time_ms"].tolist() == [100.0, 200.0, 300.0, 400.0]
    assert df["scenario_id"].iloc[0] == "-9,16|-15,22"
    assert json.loads(df["covariates"].iloc[0]) == {"x_chosen_position": 1, "x_experiment": "exp9"}
    assert df["source_row_ref"].iloc[2] == f"{folder}/prompts.jsonl:0:t2"


EXPERIENCE_TEXT = (
    "Instructions.\nThe options' names are J, G, and F.\n\n"
    "Block 1:\nThere are two options to choose from, J and G.\n"
    "You choose <<G>>. Option J yields 50 points. Option G yields 0 points.\n"
    "You choose <<J>>. Option J yields 10 points. Option G yields 0 points.\n"
    "You choose <<G>>. Option J yields 30 points. Option G yields 0 points.\n"
    "You choose <<J>>. Option J yields 0 points. Option G yields 0 points.\n"
    "\nBlock 2:\nThere are three options to choose from, J, G, and F.\n"
    "You choose <<F>>. Option J yields 1 points. Option G yields 2 points. Option F yields 3 points.\n"
    "You choose <<F>>. Option J yields 1 points. Option G yields 2 points. Option F yields 30 points.\n"
    "You choose <<J>>. Option J yields 1 points. Option G yields 2 points. Option F yields 3 points."
)


def test_psych201_experience_rules_on_hand_built_transcript(tmp_path):
    rec = {"text": EXPERIENCE_TEXT, "experiment": "spektor2019contexteffects/exp1.csv", "participant": 3}
    folder = "spektor2019contexteffects"
    raw = tmp_path / "psych201" / folder
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / psych201.ZIP_NAME, "w") as z:
        z.writestr(psych201.JSONL_NAME, json.dumps(rec) + "\n")
    df, meta = psych201.load_psych201_with_meta(tmp_path, folder)
    assert meta["roundtrip_fraction"] == 1.0 and len(df) == 7
    _ok(df)
    b1 = df[df["session_id"] == "block1"]
    b2 = df[df["session_id"] == "block2"]
    assert b1["position_in_session"].tolist() == [0, 1, 2, 3] and b2["position_in_session"].tolist() == [0, 1, 2]
    # block 1, shown-so-far variances: t0 nothing -> J (first-named) safer, chose G -> 0;
    # t1 J[50] G[0] both var 0 -> J safer, chose J -> 1; t2 J[50,10] var 400, G[0,0] var 0 -> G, chose G -> 1;
    # t3 J[50,10,30] G[0,0,0] -> G safer, chose J -> 0
    assert b1["response"].tolist() == [0.0, 1.0, 1.0, 0.0]
    # exp tags: previous outcome of the chosen option vs running mean of everything shown so far
    # t0 none; t1 chose J: J's last 50 >= mean(50,0)=25 -> gain; t2 chose G: last 0 < mean(50,0,10,0)=15 -> loss;
    # t3 chose J: last 30 >= mean(50,0,10,0,30,0)=15 -> gain
    assert b1["context_tags"].tolist() == [["options:2"], ["options:2", "exp:gain"], ["options:2", "exp:loss"], ["options:2", "exp:gain"]]
    assert [json.loads(c)["x_n_shown"] for c in b1["covariates"]] == [0, 1, 2, 3]
    # block 2: t0 -> J safer (nothing shown), chose F -> 0; t1 all var 0 -> J, chose F -> 0;
    # t2 F[3,30] var big, J and G var 0 -> J (earliest), chose J -> 1
    assert b2["response"].tolist() == [0.0, 0.0, 1.0]
    # t1 chose F: last 3 >= mean(1,2,3)=2 -> gain; t2 chose J: last 1 < mean(1,2,3,1,2,30) -> loss
    assert b2["context_tags"].tolist() == [["options:3"], ["options:3", "exp:gain"], ["options:3", "exp:loss"]]
    assert [json.loads(c)["x_n_options"] for c in b2["covariates"]] == [3, 3, 3]
    assert b2["scenario_id"].iloc[0] == "exp1:block2"


BROKER_TEXT = (
    "Welcome.\nThe stocks' names are P and Y.\n\n"
    "Stock P yields 10 points. Stock Y yields 10 points.\n"
    "Stock P yields 10 points. Stock Y yields 20 points.\n"
    "Stock P yields 10 points. Stock Y yields 0 points.\n"
    "You choose Stock <<Y>>.\n\n"
    "Stock P yields 5 points. Stock Y yields 1 points.\n"
    "Stock P yields 3 points. Stock Y yields 1 points.\n"
    "You choose Stock <<Y>>."
)


def test_psych201_broker_game_without_headers_gets_implicit_blocks(tmp_path):
    rec = {"text": BROKER_TEXT, "experiment": "olschewski2024skewness/study04.csv", "participant": 11}
    folder = "olschewski2024skewness"
    raw = tmp_path / "psych201" / folder
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / psych201.ZIP_NAME, "w") as z:
        z.writestr(psych201.JSONL_NAME, json.dumps(rec) + "\n")
    df, meta = psych201.load_psych201_with_meta(tmp_path, folder)
    assert meta["roundtrip_fraction"] == 1.0 and len(df) == 2
    _ok(df)
    assert df["session_id"].tolist() == ["market1", "market2"]
    assert df["position_in_session"].tolist() == [0, 0]
    # market1: P var 0, Y var of (10,20,0) > 0 -> P safer, chose Y -> 0; Y's last 0 < mean(10,10,10,20,10,0)=10 -> loss
    # market2: P var of (5,3) = 1, Y var 0 -> Y safer, chose Y -> 1; Y's last 1 < mean(5,1,3,1)=2.5 -> loss
    assert df["response"].tolist() == [0.0, 1.0]
    assert df["context_tags"].tolist() == [["options:2", "exp:loss"], ["options:2", "exp:loss"]]
    assert [json.loads(c)["x_n_shown"] for c in df["covariates"]] == [3, 2]


def test_psych201_parser_reports_unrecognized_lines_and_bad_letters():
    text = "The lotteries' names are S and Q.\n\nSession 1:\nThis line is not a trial.\n" + SPEKTOR2024_TEXT.split("\n")[5]
    parsed = psych201.parse_transcript(text)
    assert parsed.n_lines_checked == 4 and parsed.n_roundtrip_ok == 3
    assert parsed.unrecognized == ["This line is not a trial."]
    assert len(parsed.decisions) == 1
    # a choice letter that is not one of the named options fails the round-trip check
    bad = "The lotteries' names are S and Q.\n\nSession 1:\nThe outcomes of lottery S are 1 with 50% chance and 2 with 50% chance. The outcomes of lottery Q are 3 with 50% chance and 4 with 50% chance. You choose <<Z>>."
    parsed = psych201.parse_transcript(bad)
    assert parsed.decisions == [] and len(parsed.unrecognized) == 1
    with pytest.raises(ValueError):
        psych201.read_raw(RAW_DIR, "not_a_folder")


# ---------------------------------------------------------------------------------------------
# Order effects
# ---------------------------------------------------------------------------------------------


@_needs("order_effects", "question_order_tables.json")
def test_order_effects_rows_values_and_flags():
    with open(RAW_DIR / "order_effects" / order_effects.RAW_FILE, encoding="utf-8") as fh:
        data = json.load(fh)
    pairs = {k: v for k, v in data.items() if not k.startswith("_")}
    df, meta = order_effects.load_order_effects_with_meta()
    assert len(df) == 8 * len(pairs) == meta["n_rows"]
    _ok(df)
    _common_checks(df, "order_effects:", incentivized=False)
    assert (df["elicitation_type"] == "choice_rate").all()
    for pair, tab in pairs.items():
        sub = df[df["dataset"] == f"order_effects:{pair}"]
        assert len(sub) == 8
        for _, row in sub.iterrows():
            assert row["response"] == pytest.approx(tab[row["scenario_id"]])
            order = row["question_order_id"]
            assert row["context_tags"] == [f"order:{order}"] and row["session_id"] == order
            assert row["scenario_id"][0] == order[0]
            cov = json.loads(row["covariates"])
            assert cov["weight"] is None
            assert cov["x_verified"] == str(tab["verification"]).startswith("VERIFIED")
            assert cov["x_intervening_information"] == bool(tab["intervening_information"])
        for order in ("A_first", "B_first"):
            assert sub[sub["session_id"] == order]["response"].sum() == pytest.approx(1.0, abs=1e-3)
    n_verified = sum(str(t["verification"]).startswith("VERIFIED") for t in pairs.values())
    assert meta["n_pairs_unverified"] == len(pairs) - n_verified
    only_verified = order_effects.load_order_effects(include_unverified=False)
    assert len(only_verified) == 8 * n_verified
    assert only_verified["covariates"].map(lambda c: json.loads(c)["x_verified"]).all()
    # the SI dataset is not in hand: an empty, well-formed frame
    assert order_effects.load_wang2014_si().empty


# ---------------------------------------------------------------------------------------------
# Intake battery
# ---------------------------------------------------------------------------------------------


def _battery() -> dict:
    if not BATTERY_JSON.exists():
        pytest.skip("instrument/battery.json not present")
    return json.loads(BATTERY_JSON.read_text(encoding="utf-8"))


QUESTION_SEQUENCE = {
    "tolerance-first": ("tolerance", "sell_hold", "allocation_share"),
    "scenario-first": ("sell_hold", "allocation_share", "tolerance"),
}


def make_session(battery: dict, form: str, rng: np.random.Generator, consent: bool = True, drop_last: bool = False) -> list[dict]:
    """A complete responses array following battery.json -> output.row_rules."""
    universe = battery["design"]["scenario_universe"]
    n_pres = battery["forms"][form]["n_presentations"]
    subject_id = str(uuid.UUID(bytes=rng.bytes(16), version=4))
    session_id = rng.bytes(16).hex()
    covariates = {
        "age_band": "30-44", "wealth_band": "50-250k", "invest_experience_yrs": 7,
        "self_reported_risk_tolerance": 5, "financial_literacy_score": 3, "education": "bachelor",
    }
    rows: list[dict] = []
    t0 = pd.Timestamp("2026-09-17T10:00:00Z")
    for i in range(1, n_pres + 1):
        item = universe[int(rng.integers(len(universe)))]
        qtypes = battery["output"]["elicitation_types_used"]
        prior: list[str] = []
        for qid in QUESTION_SEQUENCE[item["question_order_id"]]:
            et = qtypes[qid]
            response = round(float(rng.integers(0, 101)) / 100, 2) if et == "allocation_pct" else int(rng.integers(0, 2))
            rows.append(
                {
                    "subject_id": subject_id, "dataset": "intake_battery", "session_id": session_id,
                    "timestamp": (t0 + pd.Timedelta(seconds=len(rows) * 7)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "position_in_session": len(rows), "scenario_id": item["scenario_id"], "loss_pct": item["loss_pct"],
                    "horizon_days": item["horizon_days"], "context_tags": list(item["context_tags"]),
                    "question_order_id": item["question_order_id"], "prior_question_ids": list(prior),
                    "elicitation_type": et, "response": response, "response_time_ms": int(rng.integers(800, 9000)),
                    "covariates": dict(covariates), "outcome_behavior": None, "incentivized": False,
                    "consent_training": consent, "battery_version": battery["battery_version"], "is_synthetic": False,
                    "source_row_ref": f"{form}:i{i:02d}:{qid}",
                }
            )
            prior.append(qid)
    return rows[:-1] if drop_last else rows


def test_intake_battery_empty_and_missing_directories(tmp_path):
    df, meta = intake_battery.load_intake_battery_with_meta(tmp_path)
    assert df.empty and list(df.columns) == S.COLUMNS and meta["n_files"] == 0
    (tmp_path / intake_battery.INTAKE_SUBDIR).mkdir()
    df = intake_battery.load_intake_battery(tmp_path)
    assert df.empty and list(df.columns) == S.COLUMNS
    assert S.validate_frame(df, expect_synthetic=False, strict=False).ok
    report = intake_battery.write_intake_battery(tmp_path / "out", tmp_path)
    assert report.ok and report.n_rows == 0 and (tmp_path / "out" / "intake_battery.parquet").exists()


def test_intake_battery_loads_generated_sessions_and_applies_pilot_rules(tmp_path, rng):
    battery = _battery()
    folder = tmp_path / intake_battery.INTAKE_SUBDIR
    folder.mkdir()
    n_short = battery["forms"]["short"]["n_presentations"] * 3
    n_full = battery["forms"]["full"]["n_presentations"] * 3

    complete_short = make_session(battery, "short", rng, consent=True)
    complete_full = make_session(battery, "full", rng, consent=False)
    incomplete = make_session(battery, "short", rng, drop_last=True)
    test_run = make_session(battery, "short", rng)
    wrong_dataset = [dict(r, dataset="something_else") for r in make_session(battery, "short", rng)]
    for name, rows in [
        ("intake_a.json", complete_short), ("intake_b.json", complete_full), ("intake_c.json", incomplete),
        ("intake_d.json", test_run), ("intake_e.json", wrong_dataset), ("intake_f_duplicate.json", complete_short),
    ]:
        (folder / name).write_text(json.dumps(rows), encoding="utf-8")
    (folder / "intake_d.session.json").write_text(json.dumps({"seed": test_run[0]["session_id"], "delay_override_active": True}), encoding="utf-8")
    (folder / "intake_a.session.json").write_text(json.dumps({"delay_override_active": False}), encoding="utf-8")
    (folder / "notes.txt").write_text("ignored", encoding="utf-8")

    df, meta = intake_battery.load_intake_battery_with_meta(tmp_path)
    assert len(complete_short) == n_short and len(complete_full) == n_full
    assert len(df) == n_short + n_full
    assert meta["n_files"] == 6 and meta["n_sessions_loaded"] == 2
    assert meta["sessions_by_form"] == {"short": 1, "full": 1}
    assert sum(v for k, v in meta["dropped"].items() if "incomplete" in k) == n_short - 1
    assert sum(v for k, v in meta["dropped"].items() if "test run" in k) == n_short
    assert sum(v for k, v in meta["dropped"].items() if "duplicate" in k) == n_short
    assert sum(v for k, v in meta["dropped"].items() if "contract" in k) == n_short
    assert any("intake_e.json" in s for s in meta["skipped_files"])

    report = S.validate_frame(df, expect_synthetic=False, strict=False)
    assert report.ok, report.errors
    assert any("consent_training=False" in w for w in report.warnings)
    assert (df["dataset"] == "intake_battery").all() and (df["battery_version"] == "1.0.0").all()
    assert set(df["elicitation_type"]) == {"binary_yes_no", "binary_sell", "allocation_pct"}
    assert df["source_row_ref"].str.startswith("intake_a.json:short:i01:").any()
    assert df["question_order_id"].isin(["tolerance-first", "scenario-first"]).all()
    cov = json.loads(df["covariates"].iloc[0])
    assert list(cov) == sorted(S.COVARIATE_KEYS) and cov["invest_experience_yrs"] == 7
    assert df["timestamp"].str.endswith("Z").all()
    a = df[df["session_id"] == complete_short[0]["session_id"]]
    assert a["position_in_session"].tolist() == list(range(n_short))
    assert a["consent_training"].all()

    # writing goes through write_events; reading back validates against the schema
    report = intake_battery.write_intake_battery(tmp_path / "out", tmp_path)
    assert report.ok and report.n_rows == len(df)
    back = S.read_events(tmp_path / "out" / "intake_battery.parquet")
    assert len(back) == len(df)


def test_intake_battery_n_presentations_fallback(tmp_path):
    assert intake_battery.n_presentations_by_form(tmp_path / "missing.json") == intake_battery.DEFAULT_N_PRESENTATIONS
    battery = _battery()
    assert intake_battery.n_presentations_by_form() == {k: v["n_presentations"] for k, v in battery["forms"].items()}


# ---------------------------------------------------------------------------------------------
# data CLI
# ---------------------------------------------------------------------------------------------


@_needs("order_effects", "question_order_tables.json")
@_needs("cpc15", "cpc15_thomas2024_aggregate.csv")
def test_data_cli_writes_parquet_reports_and_generated_readme(tmp_path, capsys):
    out = tmp_path / "processed"
    rc = data_cli.main(["--only", "order_effects", "cpc15", "--out-dir", str(out)])
    assert rc == 0
    for name in ("order_effects", "cpc15"):
        assert (out / f"{name}.parquet").exists() and (out / f"{name}.validation.md").exists()
    readme = (out / "README.md").read_text(encoding="utf-8")
    n_oe = len(order_effects.load_order_effects())
    n_cpc15 = len(cpc15.load_cpc15())
    assert f"| order_effects | D | order_effects.parquet | {n_oe} | 1 | choice_rate | yes | 0 |" in readme
    assert f"| cpc15 | B | cpc15.parquet | {n_cpc15} | {n_cpc15} | choice_rate | yes | 0 |" in readme
    assert "generated by `python -m bre.data_cli`" in readme and "--only order_effects cpc15" in readme
    assert "choices13k" not in readme.split("## Datasets")[1].split("## Notes")[0]
    back = S.read_events(out / "order_effects.parquet")
    assert len(back) == n_oe
    assert pq.read_schema(out / "cpc15.parquet").names == S.COLUMNS
    with pytest.raises(SystemExit):
        data_cli.main(["--only", "not_a_dataset", "--out-dir", str(out)])


def test_data_cli_reports_a_failing_loader_without_stopping(tmp_path, monkeypatch):
    def boom(raw_dir):
        raise RuntimeError("simulated failure")

    monkeypatch.setitem(data_cli.DATASETS, "broken", data_cli.DatasetSpec("broken", boom, "broken.parquet", "Z"))
    out = tmp_path / "processed"
    rc = data_cli.main(["--only", "broken", "order_effects", "--out-dir", str(out)])
    assert rc == 1
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "ERROR: RuntimeError: simulated failure" in readme
    assert not (out / "broken.parquet").exists()
    assert (out / "order_effects.parquet").exists()
