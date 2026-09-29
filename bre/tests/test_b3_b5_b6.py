"""Tests for the classical baselines B3 (``b3_cpt``), B5 (``b5_ml``) and B6 (``b6_hmm``) on a
tiny G_C table (N = 10 subjects, generated in-test). Fast (well under 90 s together with
``test_q3_q5.py``).

* Contract: probabilities in (0, 1), ``log_lik`` zero off the sell rows, documented ``n_params``,
  restart determinism, finite gradients and a few Adam steps that improve the objective (B3,
  B6); B5's external fit (``requires_external_fit``, ``fit_external``) with AUC > 0.5 and the
  artifact byte round trip.
* B3: the value and weighting functions at known values, the closed form of ``V_sell - V_hold``
  at ``r = 0``, context independence with zero context vectors, ``separate_beta``.
* B6: the sequence layout, the state-0 / state-1 limits, causality of the filtered predictive,
  and the row-wise log-likelihood summing to a numpy forward algorithm.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from sklearn.metrics import roc_auc_score

from bre.artifact import flatten_params, unflatten_params
from bre.models import Model, ModelData, build_model_data, count_params, standard_features
from bre.models.classical.b3_cpt import B3, cpt_two_outcome, cpt_value, prelec_weight
from bre.models.classical.b5_ml import B5, gbt_n_leaves, mlp_n_weights, unpickle_blob
from bre.models.classical.b6_hmm import B6, sequence_layout, transition_matrix
from bre.registry import MODEL_REGISTRY, PER_SUBJECT_BLOCKS, make_model
from bre.sim import gc

KEY = jax.random.PRNGKey(0)
N_SUBJECTS = 10


def _finite_tree(tree) -> bool:
    return all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(tree))


def _adam_steps(model, data: ModelData, params, steps: int, lr: float):
    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        val, g = jax.value_and_grad(lambda q: model.objective(q, data))(p)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, val

    for _ in range(steps):
        params, state, _ = step(params, state)
    return params, float(model.objective(params, data))


@pytest.fixture(scope="module")
def gc_data() -> tuple[ModelData, dict]:
    df, truth = gc.generate(N_SUBJECTS, 11)
    return build_model_data(df), truth


@pytest.fixture(scope="module")
def models(gc_data):
    data, _ = gc_data
    return {"B3": B3(data), "B6": B6(data)}


# ---------------------------------------------------------------------------------------------
# Contract (gradient-fitted models)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["B3", "B6"])
def test_contract_and_adam_improves_objective(models, gc_data, name):
    data, _ = gc_data
    model = models[name]
    assert isinstance(model, Model) and model.name == name and model.family == "classical"
    assert MODEL_REGISTRY[name] is type(model) and name in PER_SUBJECT_BLOCKS
    p0 = model.init_params(KEY, data, 0)
    p1 = model.init_params(KEY, data, 1)
    again = model.init_params(KEY, data, 0)
    assert jax.tree_util.tree_structure(p0) == jax.tree_util.tree_structure(p1)
    assert any(not np.allclose(a, b) for a, b in zip(jax.tree_util.tree_leaves(p0), jax.tree_util.tree_leaves(p1)))
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(p0), jax.tree_util.tree_leaves(again)))
    p = np.asarray(model.predict_proba(p0, data))
    assert p.shape == (data.n,) and np.isfinite(p).all() and (p > 0).all() and (p < 1).all()
    ll = np.asarray(model.log_lik(p0, data))
    assert ll.shape == (data.n,) and np.all(ll[~data.sell_mask()] == 0.0) and np.all(ll[data.sell_mask()] < 0.0)
    assert isinstance(model.n_params(p0), int) and model.n_params(p0) == count_params(p0)
    assert np.isfinite(float(model.log_prior(p0)))
    obj0 = float(model.objective(p0, data))
    assert np.isfinite(obj0) and _finite_tree(jax.grad(lambda q: model.objective(q, data))(p0))
    _, obj1 = _adam_steps(model, data, p0, steps=40, lr=0.05)
    assert obj1 < obj0
    # instantiation through the registry with kwargs
    assert make_model(name, data).name == name


# ---------------------------------------------------------------------------------------------
# B3
# ---------------------------------------------------------------------------------------------


def test_b3_value_and_weighting_functions():
    lam, alpha, beta, gamma = 2.25, 0.88, 0.88, 0.65
    assert float(cpt_value(0.5, alpha, beta, lam)) == pytest.approx(0.5**alpha)
    assert float(cpt_value(-0.5, alpha, beta, lam)) == pytest.approx(-lam * 0.5**beta)
    assert float(cpt_value(0.0, alpha, beta, lam)) == 0.0
    assert np.isfinite(float(jax.grad(lambda z: cpt_value(z, alpha, beta, lam))(0.0)))  # the Z_EPS guard
    assert float(prelec_weight(1.0, gamma)) == pytest.approx(1.0, abs=1e-5)  # p is clipped to 1 - P_EPS
    assert float(prelec_weight(np.exp(-1.0), gamma)) == pytest.approx(np.exp(-1.0))  # fixed point at 1/e
    assert float(prelec_weight(0.0, gamma)) < 1e-3  # p is clipped to P_EPS = 1e-9: w(1e-9) = exp(-(20.7)^0.65) ~ 7e-4
    # degenerate lotteries (the p = 1 case carries the w(P_EPS) ~ 7e-4 residual of the clip)
    assert float(cpt_two_outcome(-0.1, 1.0, -0.4, alpha, beta, lam, gamma)) == pytest.approx(float(cpt_value(-0.1, alpha, beta, lam)), abs=1e-3)
    assert float(cpt_two_outcome(0.2, 0.3, 0.2, alpha, beta, lam, gamma)) == pytest.approx(float(cpt_value(0.2, alpha, beta, lam)))
    # mixed prospect: each outcome weighted by its own probability
    mixed = float(cpt_two_outcome(0.2, 0.3, -0.1, alpha, beta, lam, gamma))
    expected = float(prelec_weight(0.3, gamma) * cpt_value(0.2, alpha, beta, lam) + prelec_weight(0.7, gamma) * cpt_value(-0.1, alpha, beta, lam))
    assert mixed == pytest.approx(expected)


def test_b3_closed_form_at_zero_reference_and_parameter_count(models, gc_data):
    data, _ = gc_data
    b3 = models["B3"]
    p0 = b3.init_params(KEY, data, 0)
    V, S, d_x = data.n_contexts, data.n_subjects, data.d_x
    assert set(p0) == {"enc", "r0", "rho_ctx", "rho_order", "rho_tol", "pr0", "pr_ctx", "log_kappa", "log_tau"}
    assert b3.n_params(p0) == (3 * d_x + 3 + 3 * S + 3) + 2 * V + 6
    assert b3.n_params(p0, include_random_effects=False) == 3 * d_x + 6 + 2 * V + 6
    sep = B3(data, separate_beta=True)
    ps = sep.init_params(KEY, data, 0)
    assert "log_beta_ratio" in ps and sep.n_params(ps) == b3.n_params(p0) + 1
    # r = 0: V_sell - V_hold = lambda L^beta [w(1 - p_rec) kappa^beta - 1]
    params = {**p0, "r0": jnp.asarray(0.0), "rho_ctx": jnp.zeros(V), "rho_order": jnp.asarray(0.0), "rho_tol": jnp.asarray(0.0)}
    lam, alpha, beta, gamma = (np.asarray(v) for v in b3.subject_params(params, data))
    s = data.subject_idx
    p_rec = np.asarray(b3.recovery_probability(params, data))
    kappa, tau = float(np.exp(params["log_kappa"])), float(np.exp(params["log_tau"]))
    L = data.loss
    w = np.asarray(prelec_weight(1.0 - p_rec, gamma[s]))
    diff = lam[s] * L**beta[s] * (w * kappa ** beta[s] - 1.0)
    expected = 1.0 / (1.0 + np.exp(-tau * diff))
    got = np.asarray(b3.predict_proba(params, data))
    assert np.allclose(got[L > 0], expected[L > 0], atol=1e-10)
    # with zero context vectors every context condition gives the same P(sell) per (subject, L, order, tol)
    ctx_free = {**params, "pr_ctx": jnp.zeros(V)}
    p = np.asarray(b3.predict_proba(ctx_free, data))
    sell = data.sell_mask()
    tol = np.where(np.isfinite(data.tol_answer), data.tol_answer, -1.0)
    key = np.stack([data.subject_idx, data.loss, data.order_flag, tol], axis=1)[sell]
    for k in np.unique(key, axis=0):
        rows = np.flatnonzero(sell)[np.all(key == k, axis=1)]
        assert np.ptp(p[rows]) < 1e-12
    # and a reference-point shift per context breaks that
    shifted = {**ctx_free, "rho_ctx": jnp.asarray([0.05, -0.05, 0.1, -0.1])}
    rows = np.flatnonzero(sell)[np.all(key == np.unique(key, axis=0)[0], axis=1)]
    assert np.ptp(np.asarray(b3.predict_proba(shifted, data))[rows]) > 1e-4
    nat = b3.natural_params(p0)
    assert set(nat["rho_ctx"]) == set(data.ctx_vocab) and nat["kappa"] > 0 and nat["tau"] > 0 and nat["beta_ratio"] == 1.0
    summary = b3.subject_summary(p0, data)
    assert list(summary["subject_id"]) == list(data.subject_ids)
    assert (summary["lambda"] > 0).all() and (summary["alpha"] > 0).all() and (summary["gamma"] > 0).all()
    assert np.allclose(summary["beta"], summary["alpha"])


def test_b3_log_prior_is_the_encoder_shrinkage(models, gc_data):
    data, _ = gc_data
    b3 = models["B3"]
    p0 = b3.init_params(KEY, data, 0)
    assert float(b3.log_prior(p0)) == pytest.approx(float(b3.enc.log_prior(p0["enc"])))
    far = {**p0, "enc": {**p0["enc"], "u": p0["enc"]["u"] + 2.0}}
    assert float(b3.log_prior(far)) < float(b3.log_prior(p0))
    ridged = B3(data, l2=1.0)
    assert float(ridged.log_prior(p0)) == pytest.approx(float(b3.log_prior(p0)) - 0.5 * float(jnp.sum(p0["rho_ctx"] ** 2) + jnp.sum(p0["pr_ctx"] ** 2)))


# ---------------------------------------------------------------------------------------------
# B5
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def b5_fit(gc_data):
    data, _ = gc_data
    model = B5(data)
    params = model.fit_external(data, seed=3)
    return model, params


def test_b5_external_fit_contract_and_auc(gc_data, b5_fit):
    data, _ = gc_data
    model, params = b5_fit
    assert isinstance(model, Model) and model.name == "B5" and model.family == "classical"
    assert model.requires_external_fit is True and B5.requires_external_fit is True
    assert set(params) == {"blob", "meta"} and set(params["blob"]) == {"gbt", "mlp"}
    assert params["blob"]["gbt"].dtype == np.uint8 and params["blob"]["mlp"].dtype == np.uint8
    p = np.asarray(model.predict_proba(params, data))
    assert p.shape == (data.n,) and np.isfinite(p).all() and (p > 0).all() and (p < 1).all()
    ll = np.asarray(model.log_lik(params, data))
    assert np.all(ll[~data.sell_mask()] == 0.0) and np.all(ll[data.sell_mask()] < 0.0)
    assert float(model.log_prior(params)) == 0.0 and np.isfinite(float(model.objective(params, data)))
    sell = data.sell_mask()
    assert roc_auc_score(data.y[sell], p[sell]) > 0.5
    # n_params = leaves + weights, an int, and equal to the recorded meta
    ests = model.estimators(params)
    n = model.n_params(params)
    assert isinstance(n, int) and n == gbt_n_leaves(ests["gbt"]) + mlp_n_weights(ests["mlp"]) == int(params["meta"]["n_leaves"]) + int(params["meta"]["n_weights"])
    assert n > 0
    # the feature matrix is B1's (main effects + interactions)
    assert model.d == len(standard_features(data).columns) + 29 and model.features(data).shape == (data.n, model.d)
    # deterministic in the seed; init_params is a fit keyed by (rng_key, restart)
    again = model.fit_external(data, seed=3)
    assert np.array_equal(again["blob"]["gbt"], params["blob"]["gbt"]) and int(again["meta"]["seed"]) == 3
    pr = model.init_params(KEY, data, 0)
    assert np.allclose(np.asarray(model.predict_proba(pr, data)), np.asarray(model.predict_proba(model.init_params(KEY, data, 0), data)))
    assert int(pr["meta"]["seed"]) != int(model.init_params(KEY, data, 1)["meta"]["seed"])


def test_b5_variants_and_artifact_byte_round_trip(gc_data, b5_fit):
    data, _ = gc_data
    model, params = b5_fit
    parts = model.predict_proba_parts(params, data)
    p_ens = np.asarray(model.predict_proba(params, data))
    assert np.allclose(p_ens, np.clip((parts["gbt"] + parts["mlp"]) / 2, 1e-12, 1 - 1e-12))
    gbt_only, mlp_only = B5(data, estimator="gbt"), B5(data, estimator="mlp")
    assert np.allclose(np.asarray(gbt_only.predict_proba(params, data)), np.clip(parts["gbt"], 1e-12, 1 - 1e-12))
    assert np.allclose(np.asarray(mlp_only.predict_proba(params, data)), np.clip(parts["mlp"], 1e-12, 1 - 1e-12))
    assert not np.allclose(parts["gbt"], parts["mlp"])
    assert gbt_only.n_params(params) + mlp_only.n_params(params) == model.n_params(params)
    with pytest.raises(ValueError):
        B5(data, estimator="svm")
    # the artifact flattens every leaf to float64: bytes survive and the estimators reload
    flat = flatten_params(params)
    assert all(v.dtype == np.float64 for v in flat.values())
    back = unflatten_params(flat)
    assert np.allclose(np.asarray(model.predict_proba(back, data)), p_ens, atol=0)
    assert unpickle_blob(back["blob"]["gbt"]).n_iter_ == unpickle_blob(params["blob"]["gbt"]).n_iter_
    # a subset (split) is scored with the same estimators
    sub = data.subset(np.arange(0, data.n, 3))
    assert np.allclose(np.asarray(model.predict_proba(params, sub)), p_ens[::3])
    imp = model.feature_importance(params, data, n_repeats=1)
    assert list(imp.index) == list(model.columns)


# ---------------------------------------------------------------------------------------------
# B6
# ---------------------------------------------------------------------------------------------


def test_b6_layout_and_parameter_count(models, gc_data):
    data, _ = gc_data
    b6 = models["B6"]
    lay = sequence_layout(data)
    assert lay["idx"].shape == (data.n_subjects, 340) and lay["valid"].all() and lay["sell"].sum() == data.sell_mask().sum()
    # positions are increasing along every sequence and every row is placed exactly once
    for s in range(lay["idx"].shape[0]):
        rows = lay["idx"][s][lay["valid"][s]]
        assert np.all(np.diff(data.position[rows]) > 0) and len(set(data.subject_idx[rows])) == 1
    assert np.array_equal(lay["idx"][lay["seq_of_row"], lay["pos_of_row"]], np.arange(data.n))
    p0 = b6.init_params(KEY, data, 0)
    assert set(p0) == {"w_x", "w_row", "a0", "log_da", "trans_logit", "init_logit"}
    assert b6.d_row == 1 + 2 * data.n_contexts + 3 and b6.d_x == data.d_x
    assert b6.n_params(p0) == data.d_x + 2 * b6.d_row + 5 == 50
    a = np.asarray(b6.intercepts(p0))
    assert a[1] > a[0]
    T = np.asarray(transition_matrix(p0["trans_logit"]))
    assert np.allclose(T.sum(axis=1), 1.0) and (T > 0).all()
    nat = b6.natural_params(p0)
    assert 0 < nat["pi_1"] < 1 and set(nat["w_row"]) == {"state0", "state1"}
    # a subset rebuilds its own sequences (documented behaviour)
    sub = data.subset(np.arange(0, data.n, 2))
    assert sequence_layout(sub)["idx"].shape == (data.n_subjects, 170)
    assert np.asarray(b6.predict_proba(p0, sub)).shape == (sub.n,)


def test_b6_state_limits_are_plain_logistic_regressions(models, gc_data):
    data, _ = gc_data
    b6 = models["B6"]
    p0 = b6.init_params(KEY, data, 0)
    eta = np.asarray(b6.state_logits(p0, data))
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    stuck0 = {**p0, "init_logit": jnp.asarray(-40.0), "trans_logit": jnp.asarray([40.0, 40.0])}
    assert np.allclose(np.asarray(b6.predict_proba(stuck0, data)), sig(eta[:, 0]), atol=1e-12)
    stuck1 = {**stuck0, "init_logit": jnp.asarray(40.0)}
    assert np.allclose(np.asarray(b6.predict_proba(stuck1, data)), sig(eta[:, 1]), atol=1e-12)
    alpha = np.asarray(b6.filtered_states(stuck1, data))
    assert np.allclose(alpha[:, 1], 1.0)


def test_b6_filtered_predictive_is_causal_and_sums_to_the_forward_likelihood(models, gc_data):
    data, _ = gc_data
    b6 = models["B6"]
    p0 = {**b6.init_params(KEY, data, 0), "trans_logit": jnp.asarray([1.0, 0.5]), "init_logit": jnp.asarray(-0.3)}
    p0["w_row"] = p0["w_row"] + jnp.asarray([[3.0] + [0.0] * (b6.d_row - 1), [-3.0] + [0.0] * (b6.d_row - 1)])
    pred = np.asarray(b6.predict_proba(p0, data))
    ll = np.asarray(b6.log_lik(p0, data))
    # causality: flipping the last sell answer of subject 0 changes nothing before it
    lay = sequence_layout(data)
    seq0 = lay["idx"][0][lay["sell"][0]]
    last = seq0[-1]
    y2 = data.y.copy()
    y2[last] = 1.0 - y2[last]
    flipped = replace(data, y=y2)
    pred2 = np.asarray(b6.predict_proba(p0, flipped))
    before = (data.subject_idx != 0) | (data.position <= data.position[last])
    assert np.allclose(pred2[before], pred[before], atol=1e-14)  # the predictive never uses the row's own or later answers
    after = ~before  # the tolerance row placed after the flipped answer sees the updated state
    assert after.sum() <= 1 and (after.sum() == 0 or not np.allclose(pred2[after], pred[after]))
    # numpy forward algorithm on subject 0's sell rows
    eta = np.asarray(b6.state_logits(p0, data))
    T = np.asarray(transition_matrix(p0["trans_logit"]))
    pi = np.array([1 - float(jax.nn.sigmoid(p0["init_logit"])), float(jax.nn.sigmoid(p0["init_logit"]))])
    alpha = pi.copy()
    log_lik_ref = 0.0
    preds_ref = []
    for r in seq0:
        p_s = 1.0 / (1.0 + np.exp(-eta[r]))
        preds_ref.append(alpha @ p_s)
        lik = p_s if data.y[r] == 1.0 else 1.0 - p_s
        joint = alpha * lik
        log_lik_ref += np.log(joint.sum())
        alpha = (joint / joint.sum()) @ T
    assert np.allclose(pred[seq0], preds_ref, atol=1e-10)
    assert ll[seq0].sum() == pytest.approx(log_lik_ref, abs=1e-8)
    # generic parameters: the predictive genuinely moves with the history
    assert np.ptp(pred[seq0]) > 1e-3
    summary = b6.subject_summary(p0, data)
    assert list(summary["subject_id"]) == list(data.subject_ids) and summary["state1_occupancy"].between(0, 1).all()
