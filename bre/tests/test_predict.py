"""Tests of the prediction facade (``bre.predict``), the demo book (``bre.demo``) and the demo
seed (``db.demo_seed``) on a short-fit demo artifact built once per session in a temporary
``data/synthetic`` directory.

Covers: probabilities and intervals in range; the facade's point prediction equals the model's
``predict_proba`` and the draw pass equals it at the point parameters; the top driver is one of
the scenario's inputs; the drawdown capacity is non-increasing when the loss rotation is scaled
up; ``score_book`` for 300 clients runs under 2 s once warm (timing printed); the weak-match
flag for a nonsense cause frame; the interventions ranking; the profile; the demo book's
labelling and determinism; the registry round trip.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bre import demo as demo_mod
from bre import design as D
from bre import predict as P
from bre import schema as S
from bre.models.quantum import core as C
from db.models import Client, Intervention, ModelRegistry
from db.session import session_scope

TEST_FIT_STEPS = 300
TEST_FIT_RESTARTS = 1
TEST_DRAWS = 100
"""Short demo fit for the test session (the real seed uses db.demo_seed.FIT_STEPS / FIT_RESTARTS)."""

COV = {"age_band": "30-44", "wealth_band": "50-250k", "invest_experience_yrs": 5, "self_reported_risk_tolerance": 3, "financial_literacy_score": 2, "education": "bachelor"}
MARKET = {"drawdown_pct": -0.14, "duration_days": 40, "cause_frame": "analysts now expect a recession after weak earnings", "recovery_pct": 0.02, "vix_bucket": "high", "media_intensity": "high", "social_cue_prevalence": "medium"}


class DemoEnv:
    def __init__(self, root: Path) -> None:
        from db.demo_seed import demo_engine, seed_demo

        self.root = root
        self.db_path = root / "data" / "synthetic" / "demo.db"
        self.models_root = root / "runs" / "models"
        self.engine = demo_engine(self.db_path)
        self.summary = seed_demo(self.engine, models_root=self.models_root, fit_steps=TEST_FIT_STEPS, fit_restarts=TEST_FIT_RESTARTS, n_draws=TEST_DRAWS, write_files=True)
        self.artifact = P.load_active_artifact(self.engine)
        with session_scope(self.engine) as s:
            self.clients = pd.DataFrame([{"client_id": c.client_id, "display_label": c.display_label, "covariates": c.covariates} for c in s.query(Client).order_by(Client.client_id).all()])
            self.interventions = [{"id": i.id, "name": i.name, "script": i.script, "mapped_context_transform": i.mapped_context_transform, "active": i.active} for i in s.query(Intervention).all()]


@pytest.fixture(scope="session")
def demo_env(tmp_path_factory: pytest.TempPathFactory) -> DemoEnv:
    return DemoEnv(tmp_path_factory.mktemp("bre_demo"))


@pytest.fixture(scope="session")
def art(demo_env: DemoEnv):
    return demo_env.artifact


# ---------------------------------------------------------------------------------------------
# Demo book and seed
# ---------------------------------------------------------------------------------------------


def test_demo_book_is_labelled_synthetic_and_deterministic():
    clients, responses = demo_mod.build_demo_book(seed=0, n_generated=3, n_items=12)
    assert len(clients) == 3 + len(demo_mod.ARCHETYPE_NAMES)
    assert clients["is_synthetic"].all() and clients["consent_training"].all()
    assert responses["is_synthetic"].all() and responses["consent_training"].all()
    assert list(clients["display_label"][:3]) == ["Demo client 01", "Demo client 02", "Demo client 03"]
    assert set(clients["archetype"].dropna()) == set(demo_mod.ARCHETYPE_NAMES)
    assert all(lbl.startswith("Demo archetype: ") for lbl in clients.loc[clients["archetype"].notna(), "display_label"])
    assert (responses.groupby("subject_id").size() == 2 * 12).all()
    assert (responses["dataset"] == demo_mod.DATASET).all()
    S.validate_frame(responses, expect_synthetic=True, strict=True)
    again_c, again_r = demo_mod.build_demo_book(seed=0, n_generated=3, n_items=12)
    pd.testing.assert_frame_equal(responses, again_r)
    pd.testing.assert_frame_equal(clients, again_c)
    other_c, _ = demo_mod.build_demo_book(seed=1, n_generated=3, n_items=12)
    assert not other_c["truth_bloch_theta"].equals(clients["truth_bloch_theta"])
    # every covariate record has the six product keys only
    for text in clients["covariates"]:
        assert set(json.loads(text)) <= set(S.COVARIATE_KEYS)


def test_archetypes_reproduce_their_documented_latents():
    clients, _ = demo_mod.build_demo_book(seed=0, n_generated=0, n_items=12)
    assert len(clients) == len(demo_mod.ARCHETYPE_NAMES)
    for name in demo_mod.ARCHETYPE_NAMES:
        spec = demo_mod.ARCHETYPES[name]
        row = clients.set_index("archetype").loc[name]
        assert row["truth_gamma"] == pytest.approx(spec["gamma"])
        psi = np.asarray(C.prepare_state(np.asarray(spec["v"], dtype=float)))
        assert row["truth_p_sell_base"] == pytest.approx(abs(psi[1]) ** 2, abs=1e-9)
        assert json.loads(row["covariates"]) == spec["covariates"]
        assert row["truth_source"] == demo_mod.ARCHETYPE_SOURCE
    by = clients.set_index("archetype")["truth_p_sell_mean"]
    assert by["panic-prone"] > by["steady"]


def test_seed_registers_active_model_and_artifact(demo_env: DemoEnv):
    with session_scope(demo_env.engine) as s:
        rows = s.query(ModelRegistry).all()
        assert len(rows) == 1 and rows[0].is_active and rows[0].model_type in ("Q4", "Q2")
        refs = json.loads(rows[0].training_data_refs)
        assert any(r.startswith("artifact:") for r in refs)
        n_clients = s.query(Client).count()
        assert n_clients == demo_mod.N_GENERATED + len(demo_mod.ARCHETYPE_NAMES)
        assert all(json.loads(rows[0].calibrated_contexts)[i]["status"] in ("calibrated", "uncalibrated: prior only") for i in range(4))
    art = demo_env.artifact
    assert art.is_synthetic_training and art.version == rows[0].version
    assert set(art.calibrated_contexts) == set(D.NONNULL_CONTEXTS)
    for entry in art.calibrated_contexts.values():
        expected = "calibrated" if entry["n_responses"] >= 30 else "uncalibrated: prior only"
        assert entry["status"] == expected
    assert art.metrics["train_nll_per_response"] > 0 and art.metrics["steps"] == TEST_FIT_STEPS
    assert art.n_samples == TEST_DRAWS
    assert set(art.subject_ids) == set(demo_env.clients["client_id"])
    assert (demo_env.root / "data" / "synthetic" / "demo_book.parquet").exists()
    # the seed is idempotent
    from db.demo_seed import seed_demo

    again = seed_demo(demo_env.engine, models_root=demo_env.models_root, fit_steps=TEST_FIT_STEPS, fit_restarts=1, n_draws=TEST_DRAWS)
    assert again["model"]["kept"] is True and again["book"] == "kept (already seeded)"


# ---------------------------------------------------------------------------------------------
# predict_sell
# ---------------------------------------------------------------------------------------------


def test_predict_sell_contract(art):
    out = P.predict_sell(art, COV, -0.2, ["news:recession", "social:friend_sells"])
    assert 0.0 < out["p"] < 1.0
    lo80, hi80 = out["ci80"]
    lo95, hi95 = out["ci95"]
    assert 0.0 <= lo95 <= lo80 <= hi80 <= hi95 <= 1.0
    assert out["n_draws"] == TEST_DRAWS and "draws" in out["interval_note"]
    assert out["top_driver"]["driver"] in {"news:recession", "social:friend_sells", "loss"}
    assert set(out["n_calibration"]) == {"news:recession", "social:friend_sells"}
    assert all(v["n_responses"] > 0 for v in out["n_calibration"].values())
    assert out["model_version"] == art.version and out["synthetic"] is True
    assert out["wording"] == "predicted probability of selling"
    assert math.isfinite(out["interference"]["ltp"]) and out["interference"]["order_effect"] is not None
    assert out["scenario"]["on_design_grid"] and out["scenario"]["in_design_range"]
    single = P.predict_sell(art, COV, -0.1, ["news:technical"])
    assert single["interference"]["order_effect"] is None and single["interference"]["mix"] is None
    mixed = P.predict_sell(art, COV, -0.1, ["news:recession"], mix_weights={"news:recession": 0.5, "news:technical": 0.5})
    assert mixed["interference"]["mix"] is not None and math.isfinite(mixed["interference"]["mix"])
    off = P.predict_sell(art, COV, -0.5, [])
    assert not off["scenario"]["in_design_range"] and off["top_driver"]["driver"] == "loss"
    none = P.predict_sell(art, COV, None, [])
    assert none["top_driver"]["driver"] is None and "none" in none["n_calibration"]


def test_predict_sell_equals_model_predict_proba_and_draw_pass(art):
    cov = P.covariates_text(COV)
    sc = P.scorer_for(art)
    scen = [P.Scenario("c", cov, -0.2, ("news:recession", "social:friend_sells")), P.Scenario("c", cov, -0.1, ("market:recovered_5pct",), D.TOLERANCE_FIRST, 1.0), P.Scenario("c", cov, -0.05, (), D.TOLERANCE_FIRST, None)]
    df, sell_pos = P.scenario_frame(scen)
    data = art.build_data(df)
    model = art.model_instance(data)
    direct = np.asarray(model.predict_proba(art.params_for(data), data))[sell_pos]
    facade = sc.predict(scen)
    assert np.allclose(facade, direct, atol=1e-10)
    got = [P.predict_sell(art, COV, s.loss_pct, s.context_tags, s.question_order_id, s.tol_answer)["p"] for s in scen]
    assert np.allclose(got, direct, atol=1e-10)
    # the vectorized draw pass at the point parameters reproduces the point prediction
    point_only = P.Scorer(art)
    point_only.samples = [art.params]
    draws = point_only.predict_samples(scen)
    assert draws.shape == (1, 3) and np.allclose(draws[0], direct, atol=1e-10)
    # tolerance-first without an answer is the Lüders mixture of the two answered branches
    p_none = P.predict_sell(art, COV, -0.1, ["news:recession"], D.TOLERANCE_FIRST, None)["p"]
    p_yes = P.predict_sell(art, COV, -0.1, ["news:recession"], D.TOLERANCE_FIRST, 1)["p"]
    p_no = P.predict_sell(art, COV, -0.1, ["news:recession"], D.TOLERANCE_FIRST, 0)["p"]
    assert min(p_yes, p_no) - 1e-12 <= p_none <= max(p_yes, p_no) + 1e-12


def test_input_validation(art):
    with pytest.raises(ValueError):
        P.predict_sell(art, COV, -0.1, ["not:a_context"])
    with pytest.raises(ValueError):
        P.predict_sell(art, COV, -0.1, ["news:recession", "news:technical", "social:friend_sells"])
    with pytest.raises(ValueError):
        P.predict_sell(art, COV, 0.1, [])
    with pytest.raises(ValueError):
        P.predict_sell(art, COV, -0.1, [], "sideways")
    with pytest.raises(ValueError):
        P.covariates_text({"age_band": "99+"})
    assert P.check_context_tags(["none"], art.ctx_vocab) == ()
    assert json.loads(P.covariates_text({**COV, "x_extra": 1, "weight": 3})) == COV


def test_known_subject_uses_fitted_effects(art, demo_env: DemoEnv):
    cid = demo_env.clients["client_id"].iloc[0]
    cov = demo_env.clients["covariates"].iloc[0]
    known = P.predict_sell(art, cov, -0.2, ["news:recession"], subject_id=cid)
    fresh = P.predict_sell(art, cov, -0.2, ["news:recession"], subject_id="someone-else")
    assert known["scenario"]["known_subject"] and not fresh["scenario"]["known_subject"]
    assert known["p"] != pytest.approx(fresh["p"], abs=1e-9)


# ---------------------------------------------------------------------------------------------
# Drawdown capacity
# ---------------------------------------------------------------------------------------------


def test_drawdown_capacity_prefix_rule():
    assert P.drawdown_capacity(np.array([0.1, 0.2, 0.3, 0.2, 0.1])) == (0.10, "within grid")
    assert P.drawdown_capacity(np.array([0.1, 0.1, 0.2, 0.2, 0.24])) == (0.30, "at grid maximum")
    assert P.drawdown_capacity(np.array([0.3, 0.1, 0.1, 0.1, 0.1])) == (0.0, "below smallest grid loss")
    assert P.drawdown_capacity(np.array([0.1, 0.2, 0.3, 0.2, 0.1]), target=0.5) == (0.30, "at grid maximum")


def test_drawdown_capacity_monotone_in_loss_sensitivity(art, demo_env: DemoEnv):
    """Scaling ``theta_L`` up (more loss sensitivity) must not increase any client's capacity."""
    sc = P.scorer_for(art)
    grid = sorted(abs(x) for x in D.LOSS_PCTS)
    ids = demo_env.clients["client_id"].tolist()
    covs = [P.covariates_text(c) for c in demo_env.clients["covariates"]]
    scen = [P.Scenario(cid, c, -L, P.TYPICAL_CRISIS_CONTEXTS) for cid, c in zip(ids, covs) for L in grid]
    prev = None
    for k in (1.0, 1.25, 1.5, 2.0):
        params = {**art.params, "theta_L": np.asarray(art.params["theta_L"]) * k}
        p = sc.predict(scen, params=params).reshape(len(ids), len(grid))
        caps = np.asarray([P.drawdown_capacity(p[i])[0] for i in range(len(ids))])
        if prev is not None:
            assert np.all(caps <= prev + 1e-12), f"capacity increased for {int((caps > prev + 1e-12).sum())} clients at k={k}"
        prev = caps


# ---------------------------------------------------------------------------------------------
# score_book
# ---------------------------------------------------------------------------------------------


def _book_300(demo_env: DemoEnv) -> pd.DataFrame:
    parts = [demo_env.clients.assign(client_id=demo_env.clients["client_id"] + f"_{k}") for k in range(5)]
    return pd.concat(parts, ignore_index=True).head(300)


def test_score_book_contract_and_timing(art, demo_env: DemoEnv):
    book = _book_300(demo_env)
    assert len(book) == 300
    P.score_book(art, book, MARKET, demo_env.interventions)  # warm-up: compiles the forward passes for this book size
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        tab = P.score_book(art, book, MARKET, demo_env.interventions)
        times.append(time.perf_counter() - t0)
    print(f"\nscore_book(300 clients): {min(times):.3f} s (best of 3, warm); runs {[round(t, 3) for t in times]}")
    assert min(times) < 2.0
    assert len(tab) == 300 and list(tab["client_id"]) == list(book["client_id"])
    assert tab["p_sell"].between(0, 1).all() and tab["p_baseline"].between(0, 1).all()
    assert np.allclose(tab["change_vs_baseline"], tab["p_sell"] - tab["p_baseline"])
    assert (tab["ci80_low"] <= tab["ci80_high"]).all() and tab["ci80_low"].between(0, 1).all()
    scen = tab.attrs["scenario"]
    assert scen["context_tags"] == ["news:recession", "social:friend_sells"] and scen["loss_pct"] == -0.14 and not scen["weak_match"]
    assert set(tab["top_driver"].dropna()) <= set(scen["context_tags"]) | {"loss"}
    names = {it["name"] for it in demo_env.interventions}
    assert set(tab["suggested_intervention"]) <= names and (tab["suggested_delta"] <= 1e-12).all()
    assert set(tab["drawdown_capacity"]) <= {0.0, *[abs(x) for x in D.LOSS_PCTS]}
    assert set(tab["capacity_status"]) <= {"below smallest grid loss", "within grid", "at grid maximum"}
    assert tab["status"].str.startswith(("stable", "elevated", "high")).all()
    assert tab["synthetic"].all() and not tab["known_client"].any()  # replicated ids are unseen subjects
    assert tab.attrs["intervention_label"] == P.INTERVENTION_LABEL and tab.attrs["wording"] == P.PROBABILITY_WORDING
    known = P.score_book(art, demo_env.clients.head(5), MARKET, demo_env.interventions)
    assert known["known_client"].all()
    with pytest.raises(ValueError):
        P.score_book(art, pd.concat([book.head(2), book.head(2)]), MARKET)


# ---------------------------------------------------------------------------------------------
# market state mapping
# ---------------------------------------------------------------------------------------------


def test_market_state_mapping(art):
    cal = art.calibrated_contexts
    m = P.market_state_to_contexts(MARKET, cal)
    assert m["cause_context"] == "news:recession" and not m["weak_match"] and m["cause_similarity"] >= P.WEAK_MATCH_THRESHOLD
    assert m["context_tags"] == ["news:recession", "social:friend_sells"] and m["loss_pct"] == -0.14 and not m["loss_clipped"]
    nonsense = P.market_state_to_contexts({"drawdown_pct": -0.1, "cause_frame": "zxqv blorp wibble"}, cal)
    assert nonsense["weak_match"] is True and nonsense["cause_similarity"] < P.WEAK_MATCH_THRESHOLD
    tech = P.market_state_to_contexts({"drawdown_pct": -0.1, "cause_frame": "an outage at the exchange triggered automated selling"}, cal)
    assert tech["cause_context"] == "news:technical" and not tech["weak_match"]
    deep = P.market_state_to_contexts({"drawdown_pct": -0.6}, cal)
    assert deep["loss_pct"] == -0.30 and deep["loss_clipped"]
    shallow = P.market_state_to_contexts({"drawdown_pct": -0.01}, cal)
    assert shallow["loss_pct"] == -0.05 and shallow["loss_clipped"]
    percent = P.market_state_to_contexts({"drawdown_pct": -14}, cal)
    assert percent["loss_pct"] == pytest.approx(-0.14)
    none = P.market_state_to_contexts({}, cal)
    assert none["loss_pct"] == -0.05 and none["context_tags"] == [] and none["cause_context"] is None and not none["weak_match"]
    three = P.market_state_to_contexts({"drawdown_pct": -0.2, "cause_frame": "recession", "social_cue_prevalence": "high", "recovery_pct": 0.06}, cal)
    assert three["context_tags"] == ["news:recession", "social:friend_sells"] and three["dropped_contexts"] == ["market:recovered_5pct"]
    quiet = P.market_state_to_contexts({"drawdown_pct": -0.2, "cause_frame": "recession", "media_intensity": "none", "recovery_pct": 0.06}, cal)
    assert quiet["context_tags"] == ["market:recovered_5pct"]
    numeric = P.market_state_to_contexts({"drawdown_pct": -0.2, "social_cue_prevalence": 0.9}, cal)
    assert numeric["context_tags"] == ["social:friend_sells"]
    with pytest.raises(ValueError):
        P.market_state_to_contexts({"drawdown_pct": -0.2, "social_cue_prevalence": "loud"}, cal)


# ---------------------------------------------------------------------------------------------
# interventions and profile
# ---------------------------------------------------------------------------------------------


def test_rank_interventions(art, demo_env: DemoEnv):
    out = P.rank_interventions(art, COV, MARKET, demo_env.interventions)
    ranked = out["ranked"]
    assert [r["name"] for r in ranked] and all(r["label"] == P.INTERVENTION_LABEL for r in ranked)
    deltas = [r["delta_p"] for r in ranked]
    assert deltas == sorted(deltas) and [r["rank"] for r in ranked] == list(range(1, len(ranked) + 1))
    for r in ranked:
        assert r["delta_p"] == pytest.approx(r["p_after"] - r["p_before"]) and r["p_before"] == pytest.approx(out["p_before"])
        assert r["delta_ci80"] is None or r["delta_ci80"][0] <= r["delta_ci80"][1]
    social = next(r for r in ranked if r["name"] == "social_proof_counter")
    assert social["contexts_after"] == ["news:recession"]
    precommit = next(r for r in ranked if r["name"] == "precommitment_reminder")
    assert precommit["question_order_after"] == D.TOLERANCE_FIRST
    assert out["label"] == P.INTERVENTION_LABEL and out["synthetic"] is True
    # a transform that does not touch the scenario is flagged
    quiet = P.rank_interventions(art, COV, {"drawdown_pct": -0.1}, demo_env.interventions)
    assert next(r for r in quiet["ranked"] if r["name"] == "social_proof_counter")["no_change"] is True
    assert P.apply_transform(("news:recession", "social:friend_sells"), D.SCENARIO_FIRST, {"add": ["market:recovered_5pct"]}) == (("social:friend_sells", "market:recovered_5pct"), D.SCENARIO_FIRST, ["transform produced 3 contexts; kept the last 2"])


def test_profile(art, demo_env: DemoEnv):
    out = P.profile(art, COV, None, subject_id="new", book=demo_env.clients)
    assert 0 <= out["baseline"]["bloch_theta"] <= math.pi and 0 <= out["baseline"]["p_sell_base"] <= 1
    assert out["baseline"]["p_sell_base"] == pytest.approx(out["baseline"]["p_sell_base_model"], abs=1e-9)
    assert set(out["sensitivities"]) == set(art.ctx_vocab)
    for s in out["sensitivities"].values():
        assert s["theta_norm"] >= 0 and s["theta_norm_ci95"][0] <= s["theta_norm"] <= s["theta_norm_ci95"][1] + 1e-9 or s["theta_norm_ci95"][0] <= s["theta_norm_ci95"][1]
        assert s["n_calibration"]["n_responses"] > 0
    assert out["consistency"]["gamma"] > 0 and out["consistency"]["ci95"][0] <= out["consistency"]["ci95"][1]
    rs = out["risk_score"]
    assert rs["value"] == pytest.approx(math.log(rs["p_sell"] / (1 - rs["p_sell"])))
    assert 0.0 <= rs["r2_vs_angle"] <= 1.0 and rs["r2_ci95"][0] <= rs["r2_ci95"][1]
    assert out["synthetic"] is True and out["refined"] is None
    prior = [
        {"loss_pct": -0.1, "context_tags": ["news:recession"], "question_order_id": "scenario-first", "response": 1},
        {"loss_pct": -0.2, "context_tags": [], "question_order_id": "tolerance-first", "response": 1, "tol_answer": 1},
        {"loss_pct": -0.05, "context_tags": ["market:recovered_5pct"], "question_order_id": "scenario-first", "response": 0},
    ]
    refined = P.profile(art, COV, prior, subject_id="new")
    assert refined["refined"]["n_responses"] == 3 and len(refined["refined"]["u"]) == 4
    if art.model_name == "Q4":
        assert "log_gamma" in refined["refined"] and refined["consistency"]["source"].startswith("subject MAP")
    assert refined["baseline"]["p_sell_base"] != pytest.approx(out["baseline"]["p_sell_base"], abs=1e-9)
    known = P.profile(art, demo_env.clients["covariates"].iloc[0], None, subject_id=demo_env.clients["client_id"].iloc[0])
    assert known["known_subject"] is True


def test_registry_lookup_and_artifact_path(demo_env: DemoEnv):
    row = P.active_registry_row(demo_env.engine)
    assert row["is_active"] and row["version"] == demo_env.artifact.version
    path = P.artifact_path_of(row)
    assert path is not None and Path(P.resolve_artifact_dir(path), "artifact.json").exists()
    assert P.artifact_path_of({"training_data_refs": [], "metrics": {}}) is None
    assert P.artifact_path_of({"training_data_refs": [{"artifact_path": "x"}]}) == "x"
