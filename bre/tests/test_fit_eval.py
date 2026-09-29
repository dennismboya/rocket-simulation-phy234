"""Tests of the fit driver (``bre.fit``), the artifact contract (``bre.artifact``) and the
evaluation module (``bre.eval``) — fast (well under two minutes together with test_recover.py).

* ``fit_model`` on a tiny G_C table (N = 20) improves the NLL over the initialization, holds out
  a within-subject validation fold, returns a valid artifact that round-trips through
  ``save`` / ``load`` and scores rows (known subjects exactly, unseen subjects with ``u = 0``).
* The B2 (SVI) and Q2 paths of ``fit_model`` run and return draws / interference terms.
* The second-pass models (B3, B5, B6, Q3, Q5) go through ``fit_model`` with tiny settings, their
  artifacts round-trip through ``save`` / ``load`` and score rows identically; B5 takes the
  external-fit branch (``fit_external`` per restart, no draws, ``n_params`` from the model,
  byte blobs restored from float64), Q5 logs its mean ``|q|`` and an undefined ``delta_LTP``,
  Q3 fits at the default time and reports a zero order effect by construction.
* Metrics on synthetic probabilities with known answers: perfect calibration gives ECE near 0,
  a random predictor gives AUC near 0.5, a perfect predictor AUC 1; the subject bootstrap
  interval contains the point estimate.
* Splits (a)-(d): split (b) leaves only pair conditions in the test set; the others partition
  the rows as documented.
* ``decision_rule`` returns exactly the two verdict sentences of PLAN.md section 6, and
  ``compare_models`` reports parameter counts side by side.
"""

from __future__ import annotations

import numpy as np
import pytest

from bre import eval as E
from bre import fit as F
from bre.artifact import CALIBRATED, ModelArtifact, predict_rows, predict_rows_samples
from bre.models import build_model_data
from bre.registry import GENERATORS, MODEL_REGISTRY, make_model, param_counts
from bre.sim import gc


@pytest.fixture(scope="module")
def gc20():
    df, truth = gc.generate(20, 7)
    return df, truth, build_model_data(df)


@pytest.fixture(scope="module")
def gc6():
    """A smaller table (6 subjects, 1020 rows) for the second-pass fits: B6's scan and the
    Laplace Hessians of B3 / Q3 are the costly parts, and they scale with the row count."""
    df, truth = gc.generate(6, 7)
    return df, truth, build_model_data(df)


# ---------------------------------------------------------------------------------------------
# fit_model and the artifact contract
# ---------------------------------------------------------------------------------------------


def test_registry_lists_the_ten_models_and_generators(gc20):
    _, _, data = gc20
    assert set(MODEL_REGISTRY) == {"B1", "B2", "B3", "B4", "B5", "B6", "Q2", "Q3", "Q4", "Q5"}
    assert GENERATORS == {"gq": "quantum", "gc": "classical", "gf": "classical"}
    assert F.requires_external_fit(make_model("B5", data)) and not F.requires_external_fit(make_model("B1", data))
    for name in MODEL_REGISTRY:
        m = make_model(name, data)
        assert m.name == name
        p0 = m.init_params(F.jax.random.PRNGKey(0), data, 0)  # B5: a (fast) sklearn fit on the tiny table
        counts = param_counts(name, p0, m)
        assert counts["population"] <= counts["all"] == m.n_params(p0) > 0
        if name != "B5":  # the pytree count agrees with the model's count on a vocabulary without `none` entries
            assert param_counts(name, p0) == counts
        else:
            assert param_counts(name, p0)["all"] > counts["all"]  # bytes of the pickled estimators, not parameters


def test_fit_model_b1_improves_nll_and_artifact_round_trips(gc20, tmp_path):
    df, _, data = gc20
    r = F.fit_model("B1", data, restarts=2, steps=200, lr=0.05, seed=0, early_stopping_patience=3, eval_every=25, lbfgs_steps=20, n_samples=8, log_path=tmp_path / "fit.log")
    tab = r.restarts_table
    assert len(tab) == 2 and r.train_nll < tab["init_train_nll"].min() - 0.05
    assert np.isfinite(r.val_nll) and r.n_params == 63 and r.n_params_population == 63
    assert r.param_samples is not None and len(r.param_samples) == 8
    assert "laplace" in r.extra and r.extra["laplace"]["n_pop_params"] == 63
    # within-subject validation fold: disjoint, covers the sell rows, every subject on both sides
    assert len(np.intersect1d(r.train_rows, r.val_rows)) == 0
    assert len(r.train_rows) + len(r.val_rows) == int(data.sell_mask().sum())
    assert set(data.subject_idx[r.train_rows]) == set(data.subject_idx[r.val_rows]) == set(range(data.n_subjects))
    assert abs(len(r.val_rows) / (len(r.train_rows) + len(r.val_rows)) - 0.2) < 0.02
    assert (tmp_path / "fit.log").read_text().count("[fit] B1 restart") == 2
    # artifact round trip and scoring
    art = F.artifact_from_fit(r, data, training_data_refs=["tests/gc20"], metrics={"note": "test"})
    assert art.is_synthetic_training and art.model_name == "B1" and art.n_params == 63
    art.save(tmp_path / "b1")
    back = ModelArtifact.load(tmp_path / "b1")
    assert back.ctx_vocab == data.ctx_vocab and back.covariate_stats == data.covariate_stats and back.n_samples == 8
    assert back.calibrated_contexts[data.ctx_vocab[0]]["status"] == CALIBRATED
    assert back.calibrated_contexts[data.ctx_vocab[0]]["n_responses"] > 0
    assert back.calibrated_contexts[data.ctx_vocab[0]]["theta_ci95"] is not None
    p_direct = np.asarray(r.model.predict_proba(r.params, data))
    assert np.allclose(predict_rows(back, df), p_direct, atol=1e-12)
    S_ = predict_rows_samples(back, df)
    assert S_.shape == (8, data.n) and np.all((S_ > 0) & (S_ < 1))
    df_new, _ = gc.generate(3, 8)  # unseen subjects: population parameters, u = 0
    p_new = predict_rows(back, df_new)
    assert p_new.shape == (len(df_new),) and np.all((p_new > 0) & (p_new < 1))
    inst = back.model_instance()
    assert inst.name == "B1" and tuple(inst.columns) == tuple(r.model.columns)


def test_fit_model_q2_and_b2_paths(gc20, tmp_path):
    df, _, data = gc20
    q = F.fit_model("Q2", data, restarts=1, steps=60, lr=0.03, seed=1, early_stopping_patience=2, eval_every=20, lbfgs_steps=5, n_samples=3)
    assert q.train_nll < q.restarts_table["init_train_nll"].iloc[0]
    assert q.param_samples is not None and len(q.param_samples) == 3
    assert "interference_train" in q.extra and np.isfinite(q.extra["interference_train"]["ltp_mean"])
    art = F.artifact_from_fit(q, data, training_data_refs=["tests/gc20"])
    art.save(tmp_path / "q2")
    back = ModelArtifact.load(tmp_path / "q2")
    assert "Laplace" in back.notes
    assert np.allclose(predict_rows(back, df), np.asarray(q.model.predict_proba(q.params, data)), atol=1e-12)
    b = F.fit_model("B2", data, restarts=1, svi_steps=150, seed=0, n_samples=5)
    assert b.posterior is not None and b.param_samples is not None and len(b.param_samples) == 5
    assert np.isfinite(b.train_nll) and np.isfinite(b.val_nll) and "val_nll_posterior_predictive" in b.extra
    assert "waic" in b.extra and np.isfinite(b.extra["waic"]["elpd_waic_per_response"])
    art_b = F.artifact_from_fit(b, data, training_data_refs=["tests/gc20"])
    art_b.save(tmp_path / "b2")
    back_b = ModelArtifact.load(tmp_path / "b2")
    df_new, _ = gc.generate(2, 9)
    p_new = predict_rows(back_b, df_new)  # unseen subjects: beta_ctx = mu_ctx, u = 0
    assert p_new.shape == (len(df_new),) and np.all((p_new > 0) & (p_new < 1))
    assert predict_rows_samples(back_b, df_new).shape == (5, len(df_new))


TINY = dict(restarts=1, steps=20, lr=0.03, seed=2, early_stopping_patience=2, eval_every=10, lbfgs_steps=2, n_samples=2)


@pytest.mark.parametrize("name", ["B3", "B5", "B6", "Q3", "Q5"])
def test_second_pass_models_fit_and_their_artifacts_round_trip(gc6, tmp_path, name):
    df, _, data = gc6
    r = F.fit_model(name, data, log_path=tmp_path / "fit.log", **TINY)
    model = r.model
    assert r.model_name == name and np.isfinite(r.train_nll) and np.isfinite(r.val_nll)
    assert r.n_params == model.n_params(r.params) and r.n_params_population <= r.n_params
    tab = r.restarts_table
    if name == "B5":
        assert list(tab["method"]) == ["external"] and int(tab["seed"].iloc[0]) == F.external_fit_seed(F.jax.random.PRNGKey(2), 0)
        assert r.param_samples is None and "laplace" not in r.extra and "no parameter draws" in r.extra["param_samples_note"]
        assert r.params["blob"]["gbt"].dtype == np.uint8 and r.n_params == int(r.params["meta"]["n_leaves"]) + int(r.params["meta"]["n_weights"])
        assert "interference_train" not in r.extra
    else:
        assert list(tab["method"])[0].startswith("adam") and r.train_nll < tab["init_train_nll"].iloc[0]
        assert r.param_samples is not None and len(r.param_samples) == 2 and "laplace" in r.extra
    if name in ("Q3", "Q5"):
        it = r.extra["interference_train"]
        assert it["order_zero_by_construction"] and it["n_pair_rows"] > 0 and it["order_mean"] == 0.0
        if name == "Q5":
            assert not it["ltp_defined"] and np.isnan(it["ltp_mean"]) and it["mean_abs_q"] == pytest.approx(model.mean_abs_q(r.params, data.subset(r.train_rows)))
        else:
            assert it["ltp_defined"] and np.isfinite(it["ltp_mean"]) and "mean_abs_q" not in it
    # artifact round trip: same predictions, same count, blobs restored from float64 for B5
    art = F.artifact_from_fit(r, data, training_data_refs=["tests/gc20"])
    assert art.n_params == r.n_params and art.is_synthetic_training
    art.save(tmp_path / name)
    back = ModelArtifact.load(tmp_path / name)
    assert back.model_name == name and back.n_params == r.n_params and back.n_samples == (0 if name == "B5" else 2)
    p_direct = np.asarray(model.predict_proba(r.params, data))
    assert np.allclose(predict_rows(back, df), p_direct, atol=1e-12)
    assert predict_rows_samples(back, df).shape == (back.n_samples, data.n)
    if name == "B5":
        assert "no parameter draws" in back.notes and back.params["blob"]["gbt"].dtype == np.float64
    cc = back.calibrated_contexts[data.ctx_vocab[0]]
    assert cc["status"] == CALIBRATED and (cc["theta"] is None) == (name in ("B5", "B6"))
    df_new, _ = gc.generate(2, 8)  # unseen subjects
    p_new = predict_rows(back, df_new)
    assert p_new.shape == (len(df_new),) and np.all((p_new > 0) & (p_new < 1))


def test_split_within_subject_is_stratified_and_deterministic(gc20):
    _, _, data = gc20
    rows = data.sell_rows()
    a1, b1 = F.split_within_subject(data, rows, 0.2, 3)
    a2, b2 = F.split_within_subject(data, rows, 0.2, 3)
    a3, b3 = F.split_within_subject(data, rows, 0.2, 4)
    assert np.array_equal(b1, b2) and not np.array_equal(b1, b3)
    assert len(np.intersect1d(a1, b1)) == 0 and len(a1) + len(b1) == len(rows)
    strata = F.condition_strata(data)
    for s in np.unique(strata[rows]):
        for subj in range(data.n_subjects):
            grp = rows[(strata[rows] == s) & (data.subject_idx[rows] == subj)]
            held = np.intersect1d(grp, b1)
            assert len(held) == round(0.2 * len(grp))


# ---------------------------------------------------------------------------------------------
# Metrics with known answers
# ---------------------------------------------------------------------------------------------


def test_metrics_known_answers(rng):
    n = 40000
    p = rng.uniform(0.02, 0.98, size=n)
    y = (rng.uniform(size=n) < p).astype(float)
    subj = rng.integers(0, 50, size=n)
    ll = y * np.log(p) + (1 - y) * np.log1p(-p)
    assert E.ece(p, y) < 0.015  # perfectly calibrated probabilities
    assert abs(E.brier_score(p, y) - np.mean(p * (1 - p))) < 0.01
    assert abs(E.nll(ll) - np.mean(-(p * np.log(p) + (1 - p) * np.log1p(-p)))) < 0.01
    p_random = rng.uniform(size=n)
    assert abs(E.auc(p_random, y) - 0.5) < 0.02  # a random predictor
    assert E.auc(y, y) == pytest.approx(1.0)  # a perfect predictor
    assert E.auc(1 - y, y) == pytest.approx(0.0)
    assert np.isnan(E.auc(p, np.ones(n)))  # one class only
    assert E.auc(np.full(n, 0.5), y) == pytest.approx(0.5)  # all ties
    tab = E.reliability_table(p, y)
    assert len(tab) == 10 and int(tab["n"].sum()) == n and np.all(np.abs(tab["gap"]) < 0.05)
    res = E.evaluate_predictions(p, y, ll, subj, n_boot=200, seed=0)
    for key in ("nll", "brier", "ece", "auc"):
        lo, hi = res[f"{key}_ci95"]
        assert lo <= res[key] <= hi
    d = E.paired_nll_difference(ll, ll - 0.1, subj, n_boot=200, seed=0)
    assert d["diff"] == pytest.approx(-0.1) and d["excludes_zero"]
    g = E.predictive_gain_per_subject(ll, ll - 0.1, subj)
    assert len(g) == 50 and np.allclose(g["gain"], 0.1)


# ---------------------------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------------------------


def test_split_b_leaves_only_pair_conditions_in_the_test_set(gc20):
    _, _, data = gc20
    train, test = E.split_context_composition(data)
    assert len(test) > 0 and np.all(data.ctx_mask[test].all(axis=1)) and np.all(data.sell_mask()[test])
    assert np.all(data.ctx_mask[train].sum(axis=1) <= 1)
    assert len(np.intersect1d(train, test)) == 0
    assert len(test) == data.n_subjects * 5 * 12 * 2  # 5 losses x 12 ordered pairs x 2 orders
    same = E.split_rows(data, {"type": "b"})
    assert np.array_equal(same[1], test)


def test_other_splits_partition_as_documented(gc20):
    _, _, data = gc20
    train, test = E.split_subjects(data, 0.2, 0)
    assert len(set(data.subject_idx[train]) & set(data.subject_idx[test])) == 0
    assert len(set(data.subject_idx[test])) == 4 and len(train) + len(test) == data.n
    train, test = E.split_question_order(data, "tolerance-first")
    assert np.all(data.order_flag[train] == 1) and np.all(data.order_flag[test] == 0) and np.all(data.sell_mask()[test])
    train, test = E.split_temporal(data, 0.2)
    assert np.all(data.sell_mask()[test]) and len(np.intersect1d(train, test)) == 0
    for s in range(data.n_subjects):
        tr = train[data.subject_idx[train] == s]
        te = test[data.subject_idx[test] == s]
        assert data.position[tr].max() < data.position[te].min()
        assert len(te) == round(0.2 * 170)
    train, test = E.split_rows(data, "none")
    assert len(train) == data.n and len(test) == 0
    with pytest.raises(ValueError):
        E.split_rows(data, {"type": "sideways"})


def test_structural_tests_from_data_have_the_design_cells(gc20):
    _, _, data = gc20
    ltp = E.ltp_violation_from_data(data, n_boot=30, seed=0)
    assert ltp["n_cells"] == 5 * 17 and np.isfinite(ltp["overall"]) and ltp["ci95"][0] <= ltp["ci95"][1]
    oe = E.order_effects_from_data(data, n_boot=30, seed=0)
    assert oe["n_cells"] == 5 * 6 and np.isfinite(oe["mean_abs"])


def test_model_structural_checks_q5_quarter_law_and_q3_zero_order_effect(gc6):
    _, _, data = gc6
    key = F.jax.random.PRNGKey(0)
    q5 = make_model("Q5", data)
    p5 = q5.init_params(key, data, 0)
    c5 = E.model_structural_checks(q5, p5, data, n_boot=20, seed=0)
    assert not c5["ltp"]["defined"] and "not defined" in c5["ltp"]["note"]
    assert c5["order_effect"]["zero_by_construction"] and "by construction" in c5["order_effect"]["note"]
    q = c5["quarter_law"]
    assert q["reference"] == 0.25 and q["ci95"][0] <= q["mean_abs_q"] <= q["ci95"][1] and q["n_rows"] == int(data.sell_mask().sum())
    assert "decoherence" not in c5
    q3 = make_model("Q3", data)
    c3 = E.model_structural_checks(q3, q3.init_params(key, data, 0), data, rows=data.sell_rows()[:400], n_boot=20, seed=0)
    assert c3["ltp"]["defined"] and np.isfinite(c3["ltp"]["mean"]) and "quarter_law" not in c3
    assert c3["order_effect"]["zero_by_construction"] and c3["order_effect"]["mean"] == 0.0
    # a NaN interference interval (Q5 without a defined delta_LTP) never excludes zero
    ici = E.interference_ci(q5, p5, data, param_samples=[p5, p5])
    assert ici["ltp_mean_ci95"] == [float("nan")] * 2 or all(np.isnan(ici["ltp_mean_ci95"]))
    assert ici["ltp_ci_excludes_zero"] is False and ici["order_zero_by_construction"] and "mean_abs_q" in ici
    assert E.interference_ci(make_model("B1", data), {}, data) is None


# ---------------------------------------------------------------------------------------------
# Decision rule and comparison table
# ---------------------------------------------------------------------------------------------


def _results(rng, q_better: float, n_q: int, n_c: int, synthetic: bool = False):
    n = 4000
    subj = rng.integers(0, 40, size=n)
    ll_c = -0.6 + 0.05 * rng.standard_normal(n)
    ll_q = ll_c + q_better + 0.01 * rng.standard_normal(n)
    return {
        "B4": {"family": "classical", "nll": float(-ll_c.mean()), "n_params": n_c, "ll": ll_c, "subject_idx": subj, "is_synthetic": synthetic},
        "Q2": {"family": "quantum", "nll": float(-ll_q.mean()), "n_params": n_q, "ll": ll_q, "subject_idx": subj, "is_synthetic": synthetic},
    }


def test_decision_rule_returns_the_exact_verdict_sentences(rng):
    assert E.VERDICT_SUPPORTED == "Q-model supported"
    assert E.VERDICT_NONE == "no evidence of a quantum-probability advantage in the available data"
    ici = {"ltp_mean": 0.05, "ltp_mean_ci95": [0.02, 0.08]}
    out = E.decision_rule(_results(rng, 0.05, 30, 40), ici, is_real_data=True, n_boot=200)
    assert out["verdict"] == E.VERDICT_SUPPORTED
    assert all(out["details"]["conditions"].values()) and out["details"]["served_model"] == "Q2"
    # each failed condition flips the verdict
    assert E.decision_rule(_results(rng, -0.05, 30, 40), ici, is_real_data=True, n_boot=200)["verdict"] == E.VERDICT_NONE
    assert E.decision_rule(_results(rng, 0.05, 50, 40), ici, is_real_data=True, n_boot=200)["verdict"] == E.VERDICT_NONE
    assert E.decision_rule(_results(rng, 0.05, 30, 40), {"ltp_mean_ci95": [-0.01, 0.08]}, is_real_data=True, n_boot=200)["verdict"] == E.VERDICT_NONE
    # a NaN or missing interference interval does not exclude zero (Q5's undefined delta_LTP; no draws)
    for ici_nan in ({"ltp_mean_ci95": [float("nan"), float("nan")]}, {"ltp_mean_ci95": None}, None):
        out_nan = E.decision_rule(_results(rng, 0.05, 30, 40), ici_nan, is_real_data=True, n_boot=200)
        assert out_nan["verdict"] == E.VERDICT_NONE and out_nan["details"]["conditions"]["interference_ci_excludes_zero"] is False
    syn = E.decision_rule(_results(rng, 0.05, 30, 40, synthetic=True), ici, n_boot=200)
    assert syn["verdict"] == E.VERDICT_NONE and "synthetic" in syn["details"]["note"] and "data_that_would_resolve_it" in syn["details"]
    # precomputed CI path (no per-row log-likelihoods)
    res = {"B1": {"family": "classical", "nll": 0.6, "n_params": 60}, "Q4": {"family": "quantum", "nll": 0.55, "n_params": 50, "nll_diff_vs_best_classical": {"diff": -0.05, "ci95": [-0.07, -0.03]}}}
    assert E.decision_rule(res, ici, is_real_data=True)["verdict"] == E.VERDICT_SUPPORTED
    assert E.decision_rule({"B1": res["B1"]}, ici, is_real_data=True)["verdict"] == E.VERDICT_NONE
    assert "quantum-probability advantage" in E.DECISION_RULE_TEXT


def test_compare_models_side_by_side(rng):
    res = _results(rng, 0.05, 30, 40)
    df = E.compare_models({k: {kk: vv for kk, vv in v.items() if kk not in ("ll", "subject_idx")} for k, v in res.items()})
    assert list(df["model"]) == ["Q2", "B4"] and list(df["n_params"]) == [30, 40]
    assert {"family", "nll", "nll_lo", "nll_hi", "brier", "ece", "auc"} <= set(df.columns)
    md = E.markdown_table(df[["model", "family", "n_params", "nll"]])
    assert md.startswith("| model | family | n_params | nll |")
