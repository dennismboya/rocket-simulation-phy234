"""Tests of the Q2 (context-unitary) and Q4 (open-system) models.

Fast tests: the model contract (shapes, probabilities in (0, 1), zero log-likelihood off the sell
rows, finite gradients, parameter counts, deterministic restarts); the fast forward pass equals
the row-wise core forward pass; Q2 with ``theta_L = theta_ctx = 0`` is a static classifier; Q4 at
``gamma = 0`` equals Q2 and at ``log_gamma = ln 1e3`` equals the classical Markov reference of
``tests/test_model_core.py``; the interference terms vanish for commuting settings and not
otherwise; the gauge elements leave the predictions unchanged; ``predict_new_subject`` equals
``predict_proba`` with ``u = 0``; ``none`` vocabulary entries are held at zero.

Slow tests (``-m slow``, a few minutes each): parameter recovery on G_Q data, N = 200, full design
(``fit_map`` = optax Adam on ``model.objective``), see PLAN.md section 5.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest
from scipy.stats import spearmanr

from bre import design as D
from bre.models import Model, ModelData, QuantumModelBase, build_model_data, count_params
from bre.models.base import INTERFERENCE_KEYS
from bre.models.quantum import core as C
from bre.models.quantum import q2_context_unitary as Q2M
from bre.models.quantum.q2_context_unitary import Q2, align_gauge, apply_gauge, fold_theta, none_mask, predict_new_subject
from bre.models.quantum.q4_open_system import LOG_GAMMA_CLASSICAL, Q4, lognormal_population_log_prior
from bre.sim import gq
from test_model_core import np_markov_forward

KEY = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small():
    """Five mixed-population G_Q subjects on the full design."""
    df, truth = gq.generate(5, 11)
    return df, truth, build_model_data(df)


@pytest.fixture(scope="module")
def models(small):
    _, _, data = small
    q2, q4 = Q2(data), Q4(data)
    return q2, q2.init_params(KEY, data, 0), q4, q4.init_params(KEY, data, 0)


def fit_map(model, data: ModelData, key, steps: int = 1500, lr: float = 0.02, restarts: int = 1, lr_end: float = 0.05):
    """Minimal MAP fit for the tests: Adam with a cosine-decayed learning rate on
    ``model.objective`` restricted to the sell rows (tolerance rows contribute exactly 0), best of
    ``restarts`` restarts by final objective. Returns ``(params, objective, seconds)``."""
    sub = data.subset(data.sell_rows())
    schedule = optax.cosine_decay_schedule(lr, steps, alpha=lr_end)
    opt = optax.adam(schedule)

    def objective(p):
        return model.objective(p, sub)

    @jax.jit
    def step(p, s):
        v, g = jax.value_and_grad(objective)(p)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, v

    best = None
    t0 = time.time()
    for r in range(restarts):
        p = model.init_params(key, data, restart=r)
        s = opt.init(p)
        for _ in range(steps):
            p, s, _v = step(p, s)
        v = float(objective(p))
        assert np.isfinite(v)
        if best is None or v < best[1]:
            best = (p, v)
    return best[0], best[1], time.time() - t0


def circular_correlation(a, b) -> float:
    """Jammalamadaka-Sarma circular correlation of two angle arrays."""
    a, b = np.asarray(a), np.asarray(b)
    da = np.angle(np.exp(1j * (a - np.angle(np.exp(1j * a).mean()))))
    db = np.angle(np.exp(1j * (b - np.angle(np.exp(1j * b).mean()))))
    return float(np.sum(np.sin(da) * np.sin(db)) / np.sqrt(np.sum(np.sin(da) ** 2) * np.sum(np.sin(db) ** 2)))


# ---------------------------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["q2", "q4"])
def test_contract_shapes_probabilities_loglik_and_gradients(small, models, which):
    _, _, data = small
    model, params = (models[0], models[1]) if which == "q2" else (models[2], models[3])
    assert isinstance(model, Model) and isinstance(model, QuantumModelBase)
    assert model.family == "quantum" and model.name == which.upper()
    p = np.asarray(model.predict_proba(params, data))
    assert p.shape == (data.n,) and np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))
    ll = np.asarray(model.log_lik(params, data))
    assert ll.shape == (data.n,) and np.all(ll[~data.sell_mask()] == 0.0) and np.all(ll[data.sell_mask()] < 0)
    obj = float(model.objective(params, data))
    assert np.isfinite(obj) and obj > 0
    g = jax.grad(lambda q: model.objective(q, data))(params)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(g))
    assert all(np.any(np.asarray(g[k]) != 0) for k in ("theta_L", "theta_ctx", "phi"))
    terms = model.interference_terms(params, data)
    assert set(terms) == {"ltp", "order"} and set(terms) <= set(INTERFERENCE_KEYS)
    assert terms["ltp"].shape == (data.n,) and np.all(np.isfinite(np.asarray(terms["ltp"])))
    pair = data.ctx_mask.all(axis=1)
    order = np.asarray(terms["order"])
    assert np.all(np.isnan(order[~pair])) and np.all(np.isfinite(order[pair]))
    # objective is jittable with the data baked in (what the fit driver does)
    assert np.isclose(float(jax.jit(lambda q: model.objective(q, data))(params)), obj)


def test_n_params_and_restarts(small, models):
    _, _, data = small
    q2, p2, q4, p4 = models
    d_x, S = data.d_x, data.n_subjects
    assert set(p2) == {"enc", "theta_L", "theta_ctx", "phi"}
    assert p2["theta_ctx"].shape == (data.n_contexts, 3) and p2["theta_L"].shape == (3,) and p2["phi"].shape == ()
    assert q2.n_params(p2) == count_params(p2) == (d_x * 4 + 4 + S * 4 + 4) + 3 + 3 * data.n_contexts + 1
    assert q2.n_params(p2, include_random_effects=False) == q2.n_params(p2) - S * 4
    assert set(p4) == set(p2) | {"log_gamma", "gamma_mu", "gamma_log_sigma"}
    assert p4["log_gamma"].shape == (S,)
    assert q4.n_params(p4) == count_params(p4) == q2.n_params(p2) + S + 2
    assert q4.n_params(p4, include_random_effects=False) == q2.n_params(p2, include_random_effects=False) + 2
    # deterministic in (key, restart), different across restarts, |theta| < pi, phi inside (0, pi/2)
    a = q2.init_params(KEY, data, 3)
    b = q2.init_params(KEY, data, 3)
    c = q2.init_params(KEY, data, 4)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
    assert not np.array_equal(np.asarray(a["theta_ctx"]), np.asarray(c["theta_ctx"]))
    assert np.all(np.linalg.norm(np.asarray(a["theta_ctx"]), axis=1) < np.pi) and 0 < float(a["phi"]) < np.pi / 2
    assert np.all(np.asarray(a["enc"]["u"]) == 0)
    assert np.allclose(np.asarray(q4.init_params(KEY, data, 0)["log_gamma"]), np.log(0.5), atol=0.5)


def test_log_prior_terms(small, models):
    _, _, data = small
    q2, p2, q4, p4 = models
    lp = float(q2.log_prior(p2))
    assert np.isfinite(lp)
    # the ridge pulls theta towards zero: larger theta -> smaller prior
    bigger = dict(p2, theta_ctx=p2["theta_ctx"] * 3.0)
    assert float(q2.log_prior(bigger)) < lp
    assert float(q4.log_prior(p4)) == pytest.approx(float(q2.log_prior(p4)) + float(lognormal_population_log_prior(p4["log_gamma"], p4["gamma_mu"], p4["gamma_log_sigma"])))
    # the LogNormal density is a proper log-density: matches scipy for a handful of values
    from scipy.stats import invgamma, norm

    lg = np.array([-1.0, 0.2, 0.7])
    mu, ls = 0.1, -0.3
    expected = norm.logpdf(lg, mu, np.exp(ls)).sum() + invgamma.logpdf(np.exp(2 * ls), 2.0, scale=1.0)
    assert float(lognormal_population_log_prior(lg, mu, ls)) == pytest.approx(expected)


# ---------------------------------------------------------------------------------------------
# Structural identities
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["q2", "q4"])
def test_fast_forward_equals_core_forward(small, models, which):
    _, _, data = small
    if which == "q2":
        fast, params, core = models[0], models[1], Q2(data, forward="core")
    else:
        fast, params, core = models[2], models[3], Q4(data, forward="core")
    p_fast = np.asarray(fast.predict_proba(params, data))
    p_core = np.asarray(core.predict_proba(params, data))
    assert np.allclose(p_fast, p_core, atol=1e-12)
    t_fast, t_core = fast.interference_terms(params, data), core.interference_terms(params, data)
    assert np.allclose(np.asarray(t_fast["ltp"]), np.asarray(t_core["ltp"]), atol=1e-12)
    assert np.allclose(np.asarray(t_fast["order"]), np.asarray(t_core["order"]), atol=1e-12, equal_nan=True)


def test_q2_with_zero_thetas_is_a_static_classifier(small, models):
    _, _, data = small
    q2, p2 = models[0], models[1]
    params = dict(p2, theta_L=jnp.zeros(3), theta_ctx=jnp.zeros_like(p2["theta_ctx"]))
    p = np.asarray(q2.predict_proba(params, data))
    psi = np.asarray(q2.states(params, data))
    p_sell = np.abs(psi[:, 1]) ** 2
    sf = data.order_flag == 0
    assert np.allclose(p[sf], p_sell[data.subject_idx[sf]])
    # tolerance-first rows: the Born probability of the collapsed state, independent of loss/context
    p_yes = C.tolerance_projector(params["phi"])
    for a, proj in ((1.0, p_yes), (0.0, C.IDENTITY - p_yes)):
        rows = (data.order_flag == 1) & (data.tol_answer == a)
        expected = np.asarray([float(C.born(C.lueders(jnp.asarray(psi[s]), proj), C.P_SELL)) for s in data.subject_idx[rows]])
        assert np.allclose(p[rows], expected)
    # and therefore no dependence on loss or context within a subject and order/answer
    frame = pd.DataFrame({"s": data.subject_idx, "o": data.order_flag, "a": np.nan_to_num(data.tol_answer, nan=-1), "p": p})
    assert (frame.groupby(["s", "o", "a"])["p"].nunique() == 1).all()


def test_q4_gamma_zero_equals_q2_and_large_gamma_equals_markov_reference(small, models):
    _, _, data = small
    q2, p2, q4, p4 = models
    shared = {k: p4[k] for k in ("enc", "theta_L", "theta_ctx", "phi")}
    p_q2 = np.asarray(q2.predict_proba(shared, data))
    zero = dict(p4, log_gamma=jnp.full(data.n_subjects, -40.0))
    assert np.allclose(np.asarray(q4.predict_proba(zero, data)), p_q2, atol=1e-12)
    classical = dict(p4, log_gamma=jnp.full(data.n_subjects, LOG_GAMMA_CLASSICAL))
    p_cl = np.asarray(q4.predict_proba(classical, data))
    psi = np.asarray(q4.states(classical, data))
    ctx = np.asarray(C.gather_context_thetas(q4.context_table(classical), data.ctx_idx, data.ctx_mask))
    rows = np.random.default_rng(0).choice(data.n, 80, replace=False)
    for i in rows:
        ref = np_markov_forward(
            psi[data.subject_idx[i]], np.asarray(classical["theta_L"]), data.loss[i], ctx[i], data.ctx_mask[i],
            int(data.order_flag[i]), float(classical["phi"]), data.tol_answer[i],
        )
        assert abs(ref - p_cl[i]) < 1e-4  # PLAN.md section 4: within 1e-4 (here ~1e-15)
    # generically the classical limit differs from the coherent model
    assert np.max(np.abs(p_cl - p_q2)) > 1e-3


@pytest.mark.parametrize("which", ["q2", "q4"])
def test_gauge_elements_leave_predictions_unchanged(small, models, which):
    _, _, data = small
    model, params = (models[0], models[1]) if which == "q2" else (models[2], models[3])
    p = np.asarray(model.predict_proba(params, data))
    for name in Q2M.GAUGE_ELEMENTS:
        pg = np.asarray(model.predict_proba(apply_gauge(params, name), data))
        assert np.allclose(pg, p, atol=1e-12), name
    # theta -> theta + pi theta/|theta| flips the sign of every unitary: same probabilities
    th = np.asarray(params["theta_ctx"])
    shifted = th + np.pi * th / np.linalg.norm(th, axis=1, keepdims=True)
    assert np.allclose(np.asarray(model.predict_proba(dict(params, theta_ctx=jnp.asarray(shifted)), data)), p, atol=1e-10)
    assert np.allclose(fold_theta(shifted), fold_theta(th))
    assert np.all(np.linalg.norm(fold_theta(shifted), axis=1) <= np.pi / 2 + 1e-12)
    aligned, element, corr = align_gauge(apply_gauge(params, "z_flip"), th)
    assert element == "z_flip" and corr > 0.999
    assert np.allclose(np.asarray(aligned["theta_ctx"]), fold_theta(th))


def test_interference_terms_zero_for_commuting_settings_and_nonzero_otherwise(small, models):
    _, _, data = small
    q2, p2 = models[0], models[1]
    # commuting: no rotations at all and a tolerance question aligned with the sell axis
    commuting = dict(p2, theta_L=jnp.zeros(3), theta_ctx=jnp.zeros_like(p2["theta_ctx"]), phi=jnp.asarray(0.0))
    t = q2.interference_terms(commuting, data, mix_weights={"news:recession": 0.5, "news:technical": 0.5})
    assert np.allclose(np.asarray(t["ltp"]), 0.0, atol=1e-12)
    assert np.allclose(np.nan_to_num(np.asarray(t["order"])), 0.0, atol=1e-12)
    assert np.allclose(np.asarray(t["mix"]), 0.0, atol=1e-12)
    # all context unitaries about the same axis commute: order effects vanish, LTP does not
    axis = np.array([0.6, -0.3, 0.2])
    same_axis = dict(p2, theta_ctx=jnp.asarray(np.outer([0.4, -0.8, 1.1, 0.3], axis)))
    t = q2.interference_terms(same_axis, data)
    assert np.allclose(np.nan_to_num(np.asarray(t["order"])), 0.0, atol=1e-10)
    assert np.nanmax(np.abs(np.asarray(t["ltp"]))) > 1e-3
    # generic parameters: everything non-zero, and the order term matches the core definition
    t = q2.interference_terms(p2, data, mix_weights=np.array([0.25, 0.25, 0.25, 0.25]))
    assert np.nanmax(np.abs(np.asarray(t["order"]))) > 1e-3 and np.max(np.abs(np.asarray(t["ltp"]))) > 1e-3
    assert np.max(np.abs(np.asarray(t["mix"]))) > 1e-4
    psi = np.asarray(q2.states(p2, data))
    th = np.asarray(q2.context_table(p2))
    pair_rows = np.flatnonzero(data.ctx_mask.all(axis=1))[:20]
    for i in pair_rows:
        ref = C.order_effect(jnp.asarray(psi[data.subject_idx[i]]), th[data.ctx_idx[i, 0]], th[data.ctx_idx[i, 1]], p2["theta_L"], data.loss[i])
        assert abs(float(ref) - float(t["order"][i])) < 1e-12
        ref_ltp = C.ltp_interference(jnp.asarray(psi[data.subject_idx[i]]), p2["theta_L"], data.loss[i], th[data.ctx_idx[i]], data.ctx_mask[i], p2["phi"])
        assert abs(float(ref_ltp) - float(t["ltp"][i])) < 1e-12
    # Q4: the open-system order term matches core.order_effect_rho row by row
    q4, p4 = models[2], models[3]
    t4 = q4.interference_terms(p4, data)
    psi4, th4, g4 = np.asarray(q4.states(p4, data)), np.asarray(q4.context_table(p4)), np.asarray(q4.gammas(p4))
    for i in pair_rows:
        s = data.subject_idx[i]
        ref = C.order_effect_rho(jnp.asarray(psi4[s]), th4[data.ctx_idx[i, 0]], th4[data.ctx_idx[i, 1]], g4[s], p4["theta_L"], data.loss[i])
        assert abs(float(ref) - float(t4["order"][i])) < 1e-12
        ref_ltp = C.ltp_interference_rho(C.to_rho(jnp.asarray(psi4[s])), p4["theta_L"], data.loss[i], th4[data.ctx_idx[i]], data.ctx_mask[i], p4["phi"], g4[s])
        assert abs(float(ref_ltp) - float(t4["ltp"][i])) < 1e-12
    with pytest.raises(KeyError):
        q2.interference_terms(p2, data, mix_weights={"not:a_context": 1.0})


# ---------------------------------------------------------------------------------------------
# Reporting and the API helper
# ---------------------------------------------------------------------------------------------


def test_subject_summary(small, models):
    _, _, data = small
    q2, p2, q4, p4 = models
    s2 = q2.subject_summary(p2, data)
    cols = ["subject_id", "bloch_theta", "bloch_phi", "p_sell_base"] + [f"p_sell_{D.loss_code(l)}" for l in D.LOSS_PCTS]
    assert list(s2.columns) == cols and len(s2) == data.n_subjects
    assert list(s2["subject_id"]) == list(data.subject_ids)
    psi = np.asarray(q2.states(p2, data))
    assert np.allclose(s2["bloch_theta"], 2 * np.arctan2(np.abs(psi[:, 1]), np.abs(psi[:, 0])))
    assert np.allclose(s2["p_sell_base"], np.sin(s2["bloch_theta"] / 2) ** 2)
    assert np.allclose(s2["p_sell_base"], np.abs(psi[:, 1]) ** 2)
    for loss in D.LOSS_PCTS:
        expected = [float(C.q_forward_pure(jnp.asarray(psi[s]), p2["theta_L"], abs(loss), jnp.zeros((2, 3)), jnp.zeros(2, dtype=bool), 0, p2["phi"], np.nan)) for s in range(data.n_subjects)]
        assert np.allclose(s2[f"p_sell_{D.loss_code(loss)}"], expected)
    s4 = q4.subject_summary(p4, data)
    assert list(s4.columns) == cols + ["log_gamma", "gamma"]
    assert np.allclose(s4["gamma"], np.exp(np.asarray(p4["log_gamma"])))


def test_predict_new_subject_equals_predict_proba_with_u_zero(small, models):
    df, _, data = small
    q2, p2, q4, p4 = models
    params2 = dict(p2, enc=dict(p2["enc"], u=jnp.zeros_like(p2["enc"]["u"])))
    params4 = dict(p4, enc=dict(p4["enc"], u=jnp.zeros_like(p4["enc"]["u"])))
    p_tab2 = np.asarray(q2.predict_proba(params2, data))
    p_tab4 = np.asarray(q4.predict_proba(params4, data))
    covs = df.drop_duplicates("subject_id").set_index("subject_id")["covariates"]
    rows = np.random.default_rng(1).choice(np.flatnonzero(data.sell_mask()), 40, replace=False)
    for i in rows:
        sid = data.subject_ids[data.subject_idx[i]]
        tags = [data.ctx_vocab[j] for j in data.ctx_idx[i] if j >= 0]
        order = D.TOLERANCE_FIRST if data.order_flag[i] == 1 else D.SCENARIO_FIRST
        tol = None if np.isnan(data.tol_answer[i]) else float(data.tol_answer[i])
        got = predict_new_subject(params2, covs.loc[sid], data.covariate_stats, data.ctx_vocab, -data.loss[i], tags, order, tol)
        assert abs(got - p_tab2[i]) < 1e-12
        assert abs(q2.predict_new_subject(params2, covs.loc[sid], data.covariate_stats, -data.loss[i], tags, order, tol) - p_tab2[i]) < 1e-12
        g = float(np.exp(params4["log_gamma"][data.subject_idx[i]]))
        got4 = q4.predict_new_subject(params4, covs.loc[sid], data.covariate_stats, -data.loss[i], tags, order, tol, gamma=g)
        assert abs(got4 - p_tab4[i]) < 1e-12
    # defaults: no loss and no context give the base probability; Q4 default gamma is exp(gamma_mu)
    sid = data.subject_ids[0]
    base = q2.predict_new_subject(params2, covs.loc[sid], data.covariate_stats, None)
    assert abs(base - float(q2.subject_summary(params2, data)["p_sell_base"][0])) < 1e-12
    assert q4.predict_new_subject(params4, covs.loc[sid], data.covariate_stats, -0.1, ["news:recession"]) == pytest.approx(
        q4.predict_new_subject(params4, covs.loc[sid], data.covariate_stats, -0.1, ["news:recession"], gamma=float(np.exp(params4["gamma_mu"])))
    )
    with pytest.raises(ValueError):
        predict_new_subject(params2, covs.loc[sid], data.covariate_stats, data.ctx_vocab, -0.1, ["nope:tag"])
    with pytest.raises(ValueError):
        predict_new_subject(params2, covs.loc[sid], data.covariate_stats, data.ctx_vocab, -0.1, [], "sideways")


def test_none_vocabulary_entries_are_held_at_zero(small):
    df, _, _ = small
    df = df.copy()
    # give one scenario-first single-context row an extra "ctx:none" tag (a second position)
    i = int(np.flatnonzero((df["elicitation_type"] == "binary_sell") & (df["context_tags"].map(len) == 1))[0])
    tags = list(df.at[i, "context_tags"]) + ["ctx:none"]
    df.at[i, "context_tags"] = tags
    data = build_model_data(df)
    assert data.ctx_vocab == tuple(D.NONNULL_CONTEXTS) + ("ctx:none",)
    assert none_mask(data.ctx_vocab).tolist() == [False, False, False, False, True]
    assert none_mask(("none", "news:recession")).tolist() == [True, False]
    q2 = Q2(data)
    params = q2.init_params(KEY, data, 0)
    assert np.all(np.asarray(params["theta_ctx"])[-1] == 0)
    assert q2.n_params(params) == count_params(params) - 3
    # a non-zero value in the fixed row changes nothing
    forced = dict(params, theta_ctx=params["theta_ctx"].at[-1].set(jnp.array([1.0, -2.0, 0.5])))
    assert np.allclose(np.asarray(q2.predict_proba(forced, data)), np.asarray(q2.predict_proba(params, data)))
    assert np.all(np.asarray(q2.context_table(forced))[-1] == 0)
    # the row with the extra tag equals the same row without it
    plain = build_model_data(df.assign(context_tags=df["context_tags"].map(lambda t: [x for x in t if x != "ctx:none"])), ctx_vocab=data.ctx_vocab)
    p_with = float(q2.predict_proba(forced, data)[i])
    p_without = float(Q2(plain).predict_proba(forced, plain)[i])
    assert abs(p_with - p_without) < 1e-12
    with pytest.raises(ValueError):
        q2.predict_proba(params, build_model_data(df.iloc[:10]))


# ---------------------------------------------------------------------------------------------
# Parameter recovery on G_Q data (slow)
# ---------------------------------------------------------------------------------------------

RECOVERY_N = 200
RECOVERY_STEPS_Q2 = 1500
"""Q2: 1500 Adam steps at ~40 ms each on the N = 200 sell rows."""
RECOVERY_STEPS_Q4 = 1000
"""Q4: 1000 steps at ~140 ms each (density-matrix chain), keeping the test under three minutes."""
RECOVERY_LR = 0.02


@pytest.mark.slow
def test_q2_recovers_population_context_parameters_and_states():
    """N = 200, full design, mixed_population=False (every gamma_i = 0). Q2 must recover the
    subject-mean context parameters (12 components, after gauge alignment and folding) with
    correlation > 0.9 and the subjects' Bloch polar angles with correlation > 0.8."""
    df, truth = gq.generate(RECOVERY_N, 0, mixed_population=False)
    data = build_model_data(df)
    model = Q2(data)
    params, obj, seconds = fit_map(model, data, KEY, steps=RECOVERY_STEPS_Q2, lr=RECOVERY_LR)
    mu_true = np.asarray(truth["subjects"]["theta_ctx"]).mean(axis=0)
    aligned, element, corr = align_gauge(params, mu_true)
    summary = model.subject_summary(aligned, data)
    true_theta = np.asarray(truth["subjects"]["bloch_theta"])
    corr_theta = float(np.corrcoef(summary["bloch_theta"], true_theta)[0, 1])
    corr_phi = circular_correlation(summary["bloch_phi"], np.asarray(truth["subjects"]["bloch_phi"]))
    print(
        f"\nQ2 recovery (N={RECOVERY_N}, {RECOVERY_STEPS_Q2} steps, {seconds:.0f}s, objective {obj:.1f}, gauge {element}): "
        f"theta_ctx corr {corr:.3f}; bloch_theta corr {corr_theta:.3f}; bloch_phi circular corr {corr_phi:.3f}; "
        f"theta_L true {np.asarray(truth['population']['theta_L']).round(2)} fit {np.asarray(aligned['theta_L']).round(2)}; "
        f"phi true {truth['population']['phi']:.2f} fit {float(aligned['phi']):.2f}"
    )
    assert corr > 0.9
    assert corr_theta > 0.8


@pytest.mark.slow
def test_q4_recovers_decoherence_ranking_on_mixed_population():
    """N = 200, full design, mixed population. Q4 must recover the per-subject ``log gamma_i``
    with Spearman rank correlation > 0.5."""
    df, truth = gq.generate(RECOVERY_N, 0, mixed_population=True)
    data = build_model_data(df)
    model = Q4(data)
    params, obj, seconds = fit_map(model, data, KEY, steps=RECOVERY_STEPS_Q4, lr=RECOVERY_LR)
    lg_true = np.log(np.asarray(truth["subjects"]["gamma"]))
    lg_fit = np.asarray(params["log_gamma"])
    rho = float(spearmanr(lg_true, lg_fit).correlation)
    mu_true = np.asarray(truth["subjects"]["theta_ctx"]).mean(axis=0)
    _, element, corr = align_gauge(params, mu_true)
    print(
        f"\nQ4 recovery (N={RECOVERY_N}, {RECOVERY_STEPS_Q4} steps, {seconds:.0f}s, objective {obj:.1f}, gauge {element}): "
        f"log_gamma Spearman {rho:.3f}; theta_ctx corr {corr:.3f}; "
        f"gamma_mu true {truth['population']['mu_gamma']:.2f} fit {float(params['gamma_mu']):.2f}; "
        f"sigma_gamma true {truth['population']['sigma_gamma']:.2f} fit {float(np.exp(params['gamma_log_sigma'])):.2f}"
    )
    assert rho > 0.5
