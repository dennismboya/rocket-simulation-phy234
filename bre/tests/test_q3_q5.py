"""Tests for the Q3 (``q3_dynamics``) and Q5 (``q5_qdt``) quantum-probability models on a tiny
G_C table (N = 10 subjects, generated in-test). Fast.

* Contract for both: protocol conformance, probabilities in (0, 1), ``log_lik`` zero off the
  sell rows, documented ``n_params``, finite gradients, a few Adam steps that improve the
  objective, interference keys and shapes.
* Q3: unitarity of ``expm(-iHt)``; ``t = 0`` is the static Born classifier; ``P(sell) + P(hold)
  = 1``; the recency signature (non-monotone in ``t`` with an opposing dissonance term, matching
  the closed form ``(h_x^2/|h|^2) sin^2(|h| t)``); the ``days_since_news`` hook and
  ``predict_new_subject``; ``none`` vocabulary entries held at zero.
* Q5: ``P(sell) + P(hold) = 1``, ``|q| < 0.5`` and within the admissibility bound, ``q = 0``
  reduces to the utility factor, the ``0.5 tanh`` form at ``f = 1/2``, ``quarter_law_check``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from bre import design as D
from bre.models import Model, ModelData, QuantumModelBase, build_model_data, count_params
from bre.models.base import INTERFERENCE_KEYS
from bre.models.quantum import core as C
from bre.models.quantum.q3_dynamics import (
    DAYS_KEY,
    Q3,
    T_DEFAULT,
    days_since_news_from_frame,
    dissonance_direction,
    evolution_unitaries,
    predict_new_subject,
)
from bre.models.quantum.q5_qdt import Q5, QUARTER_LAW_REFERENCE, attraction_factor
from bre.registry import MODEL_REGISTRY, PER_SUBJECT_BLOCKS
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
def small():
    df, truth = gc.generate(N_SUBJECTS, 11)
    return df, truth, build_model_data(df)


@pytest.fixture(scope="module")
def models(small):
    _, _, data = small
    q3, q5 = Q3(data), Q5(data)
    return {"Q3": (q3, q3.init_params(KEY, data, 0)), "Q5": (q5, q5.init_params(KEY, data, 0))}


# ---------------------------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Q3", "Q5"])
def test_contract_interference_and_adam(small, models, name):
    _, _, data = small
    model, params = models[name]
    assert isinstance(model, Model) and isinstance(model, QuantumModelBase)
    assert model.family == "quantum" and model.name == name and MODEL_REGISTRY[name] is type(model) and name in PER_SUBJECT_BLOCKS
    p = np.asarray(model.predict_proba(params, data))
    assert p.shape == (data.n,) and np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))
    ll = np.asarray(model.log_lik(params, data))
    assert ll.shape == (data.n,) and np.all(ll[~data.sell_mask()] == 0.0) and np.all(ll[data.sell_mask()] < 0)
    assert isinstance(model.n_params(params), int)
    obj0 = float(model.objective(params, data))
    assert np.isfinite(obj0) and obj0 > 0
    assert _finite_tree(jax.grad(lambda q: model.objective(q, data))(params))
    # restarts: deterministic in (key, restart), different across restarts
    a, b, c = model.init_params(KEY, data, 3), model.init_params(KEY, data, 3), model.init_params(KEY, data, 4)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
    assert any(not np.allclose(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(c)))
    assert np.all(np.asarray(a["enc"]["u"]) == 0)
    # interference terms: the fixed keys, (n,) shapes, order NaN off pair rows
    terms = model.interference_terms(params, data)
    assert set(terms) == {"ltp", "order"} and set(terms) <= set(INTERFERENCE_KEYS)
    pair = data.ctx_mask.all(axis=1)
    order = np.asarray(terms["order"])
    assert terms["ltp"].shape == (data.n,) and order.shape == (data.n,)
    assert np.all(np.isnan(order[~pair])) and np.all(order[pair] == 0.0)  # additive contexts: no order effect
    _, obj1 = _adam_steps(model, data, params, steps=40, lr=0.05)
    assert obj1 < obj0


def test_parameter_counts_documented(small, models):
    _, _, data = small
    q3, p3 = models["Q3"]
    q5, p5 = models["Q5"]
    d_x, S, V = data.d_x, data.n_subjects, data.n_contexts
    assert set(p3) == {"enc", "theta_p", "theta_d_angles", "w_ctx", "phi"}
    assert p3["theta_p"].shape == (3,) and p3["theta_d_angles"].shape == (2,) and p3["w_ctx"].shape == (V,) and p3["phi"].shape == ()
    assert q3.n_params(p3) == count_params(p3) == (4 * d_x + 4 + 4 * S + 4) + 3 + 2 + V + 1
    assert q3.n_params(p3, include_random_effects=False) == 4 * d_x + 8 + 6 + V
    assert set(p5) == {"enc", "b_ctx", "b_order", "b_tol", "pr_logit", "log_rho", "log_kappa", "log_tau"}
    assert q5.n_params(p5) == count_params(p5) == (d_x + 1 + S + 1) + V + 6
    assert q5.n_params(p5, include_random_effects=False) == d_x + 2 + V + 6
    assert 0 < float(p3["phi"]) < np.pi / 2


# ---------------------------------------------------------------------------------------------
# Q3
# ---------------------------------------------------------------------------------------------


def test_q3_unitarity_and_direction():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(50, 3))
    t = rng.uniform(0, 6, size=50)
    U = np.asarray(evolution_unitaries(jnp.asarray(h), jnp.asarray(t)))
    eye = np.eye(2)
    assert max(np.linalg.norm(np.conj(u).T @ u - eye) for u in U) < 1e-10
    assert max(np.linalg.norm(u @ np.conj(u).T - eye) for u in U) < 1e-10
    # expm(-i H t) with H = h.sigma, checked against scipy on a few rows
    from scipy.linalg import expm

    sig = np.asarray(C.PAULI)
    for i in range(5):
        H = np.einsum("k,kij->ij", h[i], sig)
        assert np.allclose(U[i], expm(-1j * H * t[i]), atol=1e-12)
    d = np.asarray(dissonance_direction(jnp.asarray([0.7, -1.1])))
    assert np.linalg.norm(d) == pytest.approx(1.0)
    assert np.allclose(np.asarray(dissonance_direction(jnp.asarray([0.0, 0.0]))), [0.0, 0.0, 1.0])
    assert np.allclose(np.asarray(dissonance_direction(jnp.asarray([np.pi / 2, 0.0]))), [1.0, 0.0, 0.0])


def test_q3_t_zero_is_the_static_classifier_and_probabilities_sum_to_one(small, models):
    _, _, data = small
    q3, p3 = models["Q3"]
    p_sell = np.asarray(q3.predict_proba(p3, data))
    p_hold = np.asarray(q3.predict_hold(p3, data))
    assert np.allclose(p_sell + p_hold, 1.0, atol=1e-12)
    p0 = np.asarray(q3.predict_proba_at(p3, data, t=0.0))
    psi = np.asarray(q3.states(p3, data))
    sf = data.order_flag == 0
    assert np.allclose(p0[sf], (np.abs(psi[:, 1]) ** 2)[data.subject_idx[sf]])
    p_yes = C.tolerance_projector(p3["phi"])
    for a, proj in ((1.0, p_yes), (0.0, C.IDENTITY - p_yes)):
        rows = (data.order_flag == 1) & (data.tol_answer == a)
        expected = np.asarray([float(C.born(C.lueders(jnp.asarray(psi[s]), proj), C.P_SELL)) for s in data.subject_idx[rows]])
        assert np.allclose(p0[rows], expected)
    # the default time is T_DEFAULT = 1 and predict_proba_at reproduces predict_proba there
    assert np.allclose(np.asarray(q3.predict_proba_at(p3, data, t=T_DEFAULT)), p_sell)
    assert np.allclose(np.asarray(q3.predict_proba_at(p3, data, days_since_news=np.e - 1.0)), p_sell, atol=1e-12)
    assert np.all(q3.time_of(data) == T_DEFAULT)
    with pytest.raises(ValueError):
        q3.predict_proba_at(p3, data)


def test_q3_recency_is_non_monotone_with_opposing_dissonance(small, models):
    """Chosen parameter set: psi(0) = |hold>, payoff rotation about sigma_x (theta_p = (0.8, 0, 0)
    at L = 0.2 gives h_x = 0.16), dissonance of the ``news:recession`` context along sigma_z
    with weight 0.3 (it opposes the payoff rotation by tilting the axis off the equator). Then
    P(sell)(t) = (h_x^2 / |h|^2) sin^2(|h| t): it rises to 0.22 at t = pi / (2 |h|) ~ 4.6 and
    falls back inside the news-age range [0, log1p(365)] — non-monotone. Without the dissonance
    term (w = 0) the same range is monotone (0.16 t < pi/2)."""
    _, _, data = small
    q3, p3 = models["Q3"]
    enc = {**p3["enc"], "W": jnp.zeros_like(p3["enc"]["W"]), "b": jnp.asarray([1.0, 0.0, 0.0, 0.0]), "u": jnp.zeros_like(p3["enc"]["u"])}
    params = {**p3, "enc": enc, "theta_p": jnp.asarray([0.8, 0.0, 0.0]), "theta_d_angles": jnp.asarray([0.0, 0.0]), "w_ctx": jnp.asarray([0.3, 0.0, 0.0, 0.0])}
    assert np.allclose(np.asarray(q3.states(params, data)), [[1.0, 0.0]] * data.n_subjects)
    row = int(np.flatnonzero((data.loss == 0.2) & (data.ctx_idx[:, 0] == 0) & ~data.ctx_mask[:, 1] & (data.order_flag == 0) & data.sell_mask())[0])
    one = data.subset(np.asarray([row]))  # one-row table: the curve is 80 cheap forward passes
    t_grid = np.linspace(0.0, np.log1p(365.0), 80)
    curve = np.asarray([float(q3.predict_proba_at(params, one, t=t)[0]) for t in t_grid])
    h = np.array([0.16, 0.0, 0.3])
    closed = (h[0] ** 2 / (h @ h)) * np.sin(np.linalg.norm(h) * t_grid) ** 2
    assert np.allclose(curve, closed, atol=1e-10)
    k = int(np.argmax(curve))
    assert 0 < k < len(t_grid) - 1
    assert np.all(np.diff(curve[: k + 1]) > 0) and np.all(np.diff(curve[k:]) < 0)  # rises then falls
    assert curve[k] == pytest.approx(h[0] ** 2 / (h @ h), abs=1e-3) and curve[k] < 0.3
    assert curve[0] == 0.0
    no_diss = {**params, "w_ctx": jnp.zeros(4)}
    mono = np.asarray([float(q3.predict_proba_at(no_diss, one, t=t)[0]) for t in t_grid])
    assert np.all(np.diff(mono) > 0) and np.allclose(mono, np.sin(0.16 * t_grid) ** 2, atol=1e-10)
    # the same row without any context is monotone too (payoff only), and the dissonance lowers the peak
    assert mono.max() > curve.max()


def test_q3_days_since_news_hook_and_new_subject(small, models):
    df, _, data = small
    q3, p3 = models["Q3"]
    rng = np.random.default_rng(3)
    days = rng.uniform(0, 365, size=data.n)
    q3d = Q3(data, days_since_news=days)
    assert np.allclose(q3d.time_of(data), np.log1p(days))
    p_days = np.asarray(q3d.predict_proba(p3, data))
    assert not np.allclose(p_days, np.asarray(q3.predict_proba(p3, data)))
    assert np.allclose(p_days, np.asarray(q3.predict_proba_at(p3, data, days_since_news=days)), atol=1e-12)
    # dict form by index label, partial coverage -> default elsewhere; subsets keep their times
    labels = data.index[::2].tolist()
    q3p = Q3(data, days_since_news={lab: 10.0 for lab in labels})
    t = q3p.time_of(data)
    assert np.allclose(t[::2], np.log1p(10.0)) and np.all(t[1::2] == T_DEFAULT)
    sub = data.subset(np.arange(0, data.n, 4))
    assert np.allclose(q3p.time_of(sub), np.log1p(10.0))
    with pytest.raises(ValueError):
        Q3(data, days_since_news=days[:5])
    # the frame helper reads the covariate key
    df2 = df.copy()
    import json

    first = df2.index[0]
    rec = json.loads(df2.at[first, "covariates"])
    rec[DAYS_KEY] = 30
    df2.at[first, "covariates"] = json.dumps(rec, sort_keys=True)
    got = days_since_news_from_frame(df2)
    assert got == {first: 30.0}
    # predict_new_subject equals predict_proba with u = 0 (default time and an explicit news age)
    params = {**p3, "enc": {**p3["enc"], "u": jnp.zeros_like(p3["enc"]["u"])}}
    p_tab = np.asarray(q3.predict_proba(params, data))
    p_tab_days = np.asarray(q3d.predict_proba(params, data))
    covs = df.drop_duplicates("subject_id").set_index("subject_id")["covariates"]
    rows = np.random.default_rng(1).choice(np.flatnonzero(data.sell_mask()), 25, replace=False)
    for i in rows:
        sid = data.subject_ids[data.subject_idx[i]]
        tags = [data.ctx_vocab[j] for j in data.ctx_idx[i] if j >= 0]
        order = D.TOLERANCE_FIRST if data.order_flag[i] == 1 else D.SCENARIO_FIRST
        tol = None if np.isnan(data.tol_answer[i]) else float(data.tol_answer[i])
        got = predict_new_subject(params, covs.loc[sid], data.covariate_stats, data.ctx_vocab, -data.loss[i], tags, order, tol)
        assert abs(got - p_tab[i]) < 1e-12
        got_days = q3.predict_new_subject(params, covs.loc[sid], data.covariate_stats, -data.loss[i], tags, order, tol, days_since_news=days[i])
        assert abs(got_days - p_tab_days[i]) < 1e-12
    with pytest.raises(ValueError):
        predict_new_subject(params, covs.loc[sid], data.covariate_stats, data.ctx_vocab, -0.1, [], "sideways")
    nat = q3.natural_params(p3)
    assert np.linalg.norm(nat["theta_d"]) == pytest.approx(1.0) and set(nat["w_ctx"]) == set(data.ctx_vocab)
    assert list(q3.subject_summary(p3, data)["subject_id"]) == list(data.subject_ids)


def test_q3_none_entries_held_at_zero_and_ltp_definition(small, models):
    df, _, data = small
    q3, p3 = models["Q3"]
    # ltp = P(sell | scenario-first) - Lüders mixture, and it vanishes with phi = 0 and no dynamics
    terms = q3.interference_terms(p3, data)
    assert np.all(np.isfinite(np.asarray(terms["ltp"]))) and np.max(np.abs(np.asarray(terms["ltp"]))) > 1e-3
    still = {**p3, "theta_p": jnp.zeros(3), "w_ctx": jnp.zeros(4), "phi": jnp.asarray(0.0)}
    assert np.allclose(np.asarray(q3.interference_terms(still, data)["ltp"]), 0.0, atol=1e-12)
    # a "<ns>:none" vocabulary entry is held at w = 0 and not counted
    df2 = df.copy()
    i = int(np.flatnonzero((df2["elicitation_type"] == "binary_sell") & (df2["context_tags"].map(len) == 1))[0])
    df2.at[i, "context_tags"] = list(df2.at[i, "context_tags"]) + ["ctx:none"]
    data2 = build_model_data(df2)
    m2 = Q3(data2)
    params2 = m2.init_params(KEY, data2, 0)
    assert np.asarray(params2["w_ctx"])[-1] == 0.0 and m2.n_params(params2) == count_params(params2) - 1
    forced = {**params2, "w_ctx": params2["w_ctx"].at[-1].set(2.0)}
    assert np.allclose(np.asarray(m2.predict_proba(forced, data2)), np.asarray(m2.predict_proba(params2, data2)))


# ---------------------------------------------------------------------------------------------
# Q5
# ---------------------------------------------------------------------------------------------


def test_q5_probabilities_sum_to_one_and_attraction_is_bounded(small, models):
    _, _, data = small
    q5, p5 = models["Q5"]
    p_sell = np.asarray(q5.predict_proba(p5, data))
    p_hold = np.asarray(q5.predict_hold(p5, data))
    assert np.allclose(p_sell + p_hold, 1.0, atol=1e-12)
    f = np.asarray(q5.utility_factor(p5, data))
    q = np.asarray(q5.attraction(p5, data))
    assert np.all(np.abs(q) < 0.5) and np.all(np.abs(q) <= np.minimum(f, 1 - f) + 1e-15)
    assert np.allclose(p_sell, f + q)
    # q = 0 (zero encoder and context terms) reduces to the utility factor
    enc0 = {**p5["enc"], "W": jnp.zeros_like(p5["enc"]["W"]), "b": jnp.zeros_like(p5["enc"]["b"]), "u": jnp.zeros_like(p5["enc"]["u"])}
    rational = {**p5, "enc": enc0, "b_ctx": jnp.zeros(4), "b_order": jnp.asarray(0.0), "b_tol": jnp.asarray(0.0)}
    assert np.allclose(np.asarray(q5.attraction(rational, data)), 0.0)
    assert np.allclose(np.asarray(q5.predict_proba(rational, data)), np.asarray(q5.utility_factor(rational, data)))
    # the utility factor is context-free: the same for every condition at a given L
    for L in np.unique(data.loss[data.sell_mask()]):
        assert np.ptp(f[data.sell_mask() & (data.loss == L)]) < 1e-12
    # the specified form 0.5 tanh(z) holds exactly at f = 1/2 and is scaled by 2 min(f, 1 - f) otherwise
    z = jnp.asarray([-2.0, -0.3, 0.0, 0.7, 3.0])
    assert np.allclose(np.asarray(attraction_factor(z, 0.5)), 0.5 * np.tanh(np.asarray(z)))
    assert np.allclose(np.asarray(attraction_factor(z, 0.9)), 0.5 * np.tanh(np.asarray(z)) * 0.2)
    # contexts move q: a positive b_ctx raises P(sell) on rows showing that context
    up = {**rational, "b_ctx": jnp.asarray([1.0, 0.0, 0.0, 0.0])}
    rows = data.sell_mask() & (data.ctx_idx == 0).any(axis=1)
    assert np.all(np.asarray(q5.predict_proba(up, data))[rows] > np.asarray(q5.predict_proba(rational, data))[rows])
    nat = q5.natural_params(p5)
    assert nat["rho"] > 0 and nat["kappa"] > 0 and nat["tau"] > 0 and 0 < nat["p_rec"] < 1 and set(nat["b_ctx"]) == set(data.ctx_vocab)
    assert list(q5.subject_summary(p5, data)["subject_id"]) == list(data.subject_ids)


def test_q5_quarter_law_check_and_interference_convention(small, models):
    _, _, data = small
    q5, p5 = models["Q5"]
    check = q5.quarter_law_check(p5, data, n_boot=200, seed=0)
    assert set(check) >= {"mean_abs_q", "ci95", "reference", "n_rows", "n_boot", "share_q_positive", "note"}
    assert check["reference"] == QUARTER_LAW_REFERENCE == 0.25 and check["n_rows"] == int(data.sell_mask().sum()) and check["n_boot"] == 200
    assert check["mean_abs_q"] == pytest.approx(q5.mean_abs_q(p5, data))
    lo, hi = check["ci95"]
    assert lo <= check["mean_abs_q"] <= hi and 0 <= lo and hi < 0.5
    assert "Yukalov" in check["note"]
    assert q5.quarter_law_check(p5, data, n_boot=50, seed=1) != check  # a different seed, a different interval
    # ltp is not defined for Q5 (NaN on every row); order is 0 on pairs
    terms = q5.interference_terms(p5, data)
    assert np.all(np.isnan(np.asarray(terms["ltp"])))
    pair = data.ctx_mask.all(axis=1)
    assert np.all(np.asarray(terms["order"])[pair] == 0.0)
    # the encoder prior is the shrinkage and the optional ridge acts on b_ctx
    assert float(q5.log_prior(p5)) == pytest.approx(float(q5.enc.log_prior(p5["enc"])))
    assert float(Q5(data, l2=2.0).log_prior(p5)) == pytest.approx(float(q5.log_prior(p5)) - float(jnp.sum(p5["b_ctx"] ** 2)))
