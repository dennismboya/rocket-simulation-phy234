"""Tests for the classical baselines B1 (``b1_logistic``), B2 (``b2_hier``) and B4
(``b4_bayes_updater``) on G_C-generated synthetic data.

Fast: the model contract on every model (probabilities in (0, 1) and finite, ``log_lik`` zero off
the sell rows, documented ``n_params`` counts, finite gradients, restart determinism), B1's
interaction block, B4's structural identities (``lambda = 0`` gives context-independent
predictions; the sign of the loss monotonicity, as documented in the module; positive evidence
lowers P(sell); recency discount composition), and a tiny B2 SVI/NUTS run with the
posterior-predictive machinery (``pointwise_log_lik`` shape ``(S, n)``, finite WAIC and LOO).

Slow (``@pytest.mark.slow``, each well under 3 minutes on 4 cores): parameter recovery on G_C —
B1's context coefficients by an Adam loop at N = 200 (correlation > 0.9), B4's evidence ordering
(reported, no threshold), and B2's subject intercepts by SVI at N = 100 (Spearman > 0.6).
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy.stats import pearsonr, spearmanr

from bre import design as D
from bre.models import ModelData, build_model_data, count_params, standard_features
from bre.models.classical.b1_logistic import B1, interaction_column_names
from bre.models.classical.b2_hier import B2, B2Posterior, SITES
from bre.models.classical.b4_bayes_updater import B4, crra_utility, recency_weights
from bre.sim import gc

KEY = jax.random.PRNGKey(0)


def _finite_tree(tree) -> bool:
    return all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(tree))


def _adam_fit(model, data: ModelData, params, steps: int, lr: float):
    """Minimal full-batch Adam loop on ``model.objective`` (the test's own optimizer; the fit
    driver of Phase 3 adds restarts, early stopping and the L-BFGS polish)."""
    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        val, g = jax.value_and_grad(lambda q: model.objective(q, data))(p)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, val

    val = None
    for _ in range(steps):
        params, state, val = step(params, state)
    return params, float(val)


@pytest.fixture(scope="module")
def gc_data() -> tuple[ModelData, dict]:
    df, truth = gc.generate(6, 101)
    return build_model_data(df), truth


@pytest.fixture(scope="module")
def models(gc_data):
    data, _ = gc_data
    return {"B1": B1(data), "B2": B2(data), "B4": B4(data)}


# ---------------------------------------------------------------------------------------------
# Contract shared by the three models
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["B1", "B2", "B4"])
def test_contract_probabilities_loglik_and_gradients(models, gc_data, name):
    data, _ = gc_data
    model = models[name]
    assert model.name == name and model.family == "classical"
    p0 = model.init_params(KEY, data, 0)
    p1 = model.init_params(KEY, data, 1)
    assert jax.tree_util.tree_structure(p0) == jax.tree_util.tree_structure(p1)
    assert any(not np.allclose(a, b) for a, b in zip(jax.tree_util.tree_leaves(p0), jax.tree_util.tree_leaves(p1)))
    again = model.init_params(KEY, data, 0)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(p0), jax.tree_util.tree_leaves(again)))
    p = np.asarray(model.predict_proba(p0, data))
    assert p.shape == (data.n,) and np.isfinite(p).all() and (p > 0).all() and (p < 1).all()
    ll = np.asarray(model.log_lik(p0, data))
    assert ll.shape == (data.n,)
    assert np.all(ll[~data.sell_mask()] == 0.0)
    assert np.all(ll[data.sell_mask()] < 0.0)
    assert np.isfinite(float(model.log_prior(p0)))
    assert np.isfinite(float(model.objective(p0, data)))
    assert _finite_tree(jax.grad(lambda q: model.objective(q, data))(p0))
    assert model.n_params(p0) == count_params(p0)


# ---------------------------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------------------------


def test_b1_interaction_names_and_count(models, gc_data):
    data, _ = gc_data
    b1 = models["B1"]
    K, V = data.n_ctx_positions, data.n_contexts
    assert (K, V) == (2, 4)
    assert b1.d_interactions == 1 + K * V + K * V + K * (K - 1) // 2 * V * (V - 1) == 29
    names = b1.interaction_columns
    assert names[0] == "loss*order_flag"
    assert "loss*ctx0=news:recession" in names and "loss*ctx1=market:recovered_5pct" in names
    assert "order_flag*ctx0=news:technical" in names and "order_flag*ctx1=social:friend_sells" in names
    assert "ctx0=news:recession*ctx1=news:technical" in names
    assert "ctx0=news:technical*ctx1=news:recession" in names  # ordered pairs are distinct columns
    assert not any("ctx0=" in n.split("*")[0] and "ctx0=" in n.split("*")[1] for n in names)  # no same-position products
    assert not any(n == f"ctx0={t}*ctx1={t}" for t in data.ctx_vocab for n in names)  # no same-tag products
    assert len(set(names)) == len(names)
    assert b1.columns == tuple(standard_features(data).columns) + names
    # n_params = every weight + intercept; the docstring's count for the shared design
    p0 = b1.init_params(KEY, data, 0)
    assert b1.n_params(p0) == b1.d_main + 29 + 1 == data.d_x + 1 + K * V + 1 + 2 + 29 + 1
    assert interaction_column_names(data.ctx_vocab, K) == b1._pairs


def test_b1_interaction_columns_are_products(models, gc_data):
    data, _ = gc_data
    b1 = models["B1"]
    fm = b1.features(data)
    base = standard_features(data)
    for left, right in b1._pairs:
        col = fm.column(f"{left}*{right}")
        assert np.array_equal(col, base.column(left) * base.column(right))
    # a rec>tech pair row lights exactly its ordered-pair column among the ctx*ctx products
    rows = np.flatnonzero((data.ctx_idx[:, 0] == 0) & (data.ctx_idx[:, 1] == 1))
    ctx_pairs = [n for n in b1.interaction_columns if n.startswith("ctx0=")]
    lit = [n for n in ctx_pairs if fm.column(n)[rows[0]] == 1.0]
    assert lit == ["ctx0=news:recession*ctx1=news:technical"]


def test_b1_coefficients_named_and_prior_is_ridge(models, gc_data):
    data, _ = gc_data
    b1 = models["B1"]
    p0 = b1.init_params(KEY, data, 0)
    coef = b1.coefficients(p0)
    assert list(coef.index) == list(b1.columns) + ["intercept"]
    assert float(b1.log_prior(p0)) == pytest.approx(-0.5 * b1.l2 * float(jnp.sum(p0["w"] ** 2)))
    assert len(b1.context_coefficients(p0)) == 8
    assert B1(data, l2=0.0).log_prior(p0) == 0.0


# ---------------------------------------------------------------------------------------------
# B4
# ---------------------------------------------------------------------------------------------


def test_b4_parameter_count_documented(models, gc_data):
    data, _ = gc_data
    b4 = models["B4"]
    p0 = b4.init_params(KEY, data, 0)
    V = data.n_contexts
    assert b4.n_params(p0) == (data.d_x + 1 + data.n_subjects + 1) + V + 5
    assert b4.n_params(p0, include_random_effects=False) == data.d_x + 2 + V + 5
    assert set(p0) == {"enc", "lambda", "delta_logit", "log_rho", "log_kappa", "log_tau", "beta_tol"}
    nat = b4.natural_params(p0)
    assert 0 < nat["delta"] < 1 and nat["rho"] > 0 and nat["kappa"] > 0 and nat["tau"] > 0
    assert set(nat["lambda"]) == set(data.ctx_vocab)


def test_b4_lambda_zero_is_context_independent(models, gc_data):
    data, _ = gc_data
    b4 = models["B4"]
    p0 = b4.init_params(KEY, data, 0)
    p0 = {**p0, "lambda": jnp.zeros_like(p0["lambda"]), "delta_logit": jnp.asarray(1.3)}
    p = np.asarray(b4.predict_proba(p0, data))
    sell = data.sell_mask()
    tol = np.where(np.isfinite(data.tol_answer), data.tol_answer, -1.0)
    key = np.stack([data.subject_idx, data.loss, data.order_flag, tol], axis=1)[sell]
    for k in np.unique(key, axis=0):
        rows = np.flatnonzero(sell)[np.all(key == k, axis=1)]
        assert np.ptp(p[rows]) < 1e-12  # all 17 context conditions give the same P(sell)
    # and with lambda != 0 they do not
    p1 = {**p0, "lambda": jnp.asarray([0.8, -0.4, 0.6, -0.7])}
    assert np.ptp(np.asarray(b4.predict_proba(p1, data))[np.flatnonzero(sell)[np.all(key == np.unique(key, axis=0)[0], axis=1)]]) > 1e-3


def test_b4_positive_evidence_lowers_sell_probability(models, gc_data):
    data, _ = gc_data
    b4 = models["B4"]
    p0 = b4.init_params(KEY, data, 0)
    base = {**p0, "lambda": jnp.zeros(4)}
    up = {**p0, "lambda": jnp.asarray([1.0, 0.0, 0.0, 0.0])}
    rows = np.flatnonzero(data.sell_mask() & (data.ctx_idx[:, 0] == 0) & (data.loss > 0))
    assert np.all(np.asarray(b4.predict_proba(up, data))[rows] < np.asarray(b4.predict_proba(base, data))[rows])


def test_b4_recency_discount_composition(models, gc_data):
    data, _ = gc_data
    b4 = models["B4"]
    p0 = b4.init_params(KEY, data, 0)
    lam = jnp.asarray([0.5, -0.3, 0.2, -0.1])
    delta_logit = jnp.asarray(0.4)
    params = {**p0, "lambda": lam, "delta_logit": delta_logit}
    delta = float(jax.nn.sigmoid(delta_logit))
    ev = np.asarray(b4.evidence(params, data))
    lam_np = np.asarray(lam)
    for i in range(data.n):
        c = data.ctx_idx[i][data.ctx_mask[i]]
        if len(c) == 0:
            assert ev[i] == 0.0
        elif len(c) == 1:
            assert ev[i] == pytest.approx(lam_np[c[0]])
        else:
            assert ev[i] == pytest.approx(delta * lam_np[c[0]] + lam_np[c[1]])  # the last context undiscounted
    assert np.array_equal(np.asarray(recency_weights(np.array([[True, True], [True, False], [False, False]]))), [[1, 0], [0, 0], [0, 0]])


def test_b4_monotone_in_loss_when_expected_further_loss_dominates(models, gc_data):
    """Documented sign (module docstring): with ``kappa (1 - p_rec) >= 1`` P(sell) increases in L.
    The direction is a property of (kappa, rho, p_rec), reported not assumed."""
    data, _ = gc_data
    b4 = models["B4"]
    p0 = b4.init_params(KEY, data, 0)
    enc = {**p0["enc"], "W": jnp.zeros_like(p0["enc"]["W"]), "b": jnp.zeros_like(p0["enc"]["b"])}  # p_rec = 0.5
    params = {**p0, "enc": enc, "lambda": jnp.zeros(4), "log_kappa": jnp.log(3.0), "log_rho": jnp.log(1.5), "beta_tol": jnp.asarray(0.0)}
    p = np.asarray(b4.predict_proba(params, data))
    sell = data.sell_mask()
    losses = np.sort(np.unique(data.loss[sell]))
    assert len(losses) == 5 and losses[0] > 0
    for cond in np.unique(data.ctx_idx[sell], axis=0):
        rows = np.flatnonzero(sell)[np.all(data.ctx_idx[sell] == cond, axis=1)]
        means = [p[rows][data.loss[rows] == L].mean() for L in losses]
        assert np.all(np.diff(means) > 0), (cond, means)
    # the opposite regime (kappa (1 - p_rec) < 1, here kappa = 1) makes P(sell) fall with L
    params_low = {**params, "log_kappa": jnp.log(1.0)}
    p_low = np.asarray(b4.predict_proba(params_low, data))
    rows = np.flatnonzero(sell & ~data.ctx_mask[:, 0])
    means = [p_low[rows][data.loss[rows] == L].mean() for L in losses]
    assert np.all(np.diff(means) < 0)


def test_b4_crra_utility_limits():
    w = jnp.asarray([0.5, 0.9, 1.0, 1.5])
    assert np.allclose(np.asarray(crra_utility(w, 1.0)), np.log(np.asarray(w)))
    assert np.allclose(np.asarray(crra_utility(w, 1.0 + 1e-9)), np.log(np.asarray(w)), atol=1e-6)
    assert np.allclose(np.asarray(crra_utility(w, 2.0)), 1.0 - 1.0 / np.asarray(w))
    assert float(crra_utility(1.0, 3.0)) == 0.0
    assert np.isfinite(float(crra_utility(-0.5, 2.0)))  # wealth floor
    assert np.isfinite(float(jax.grad(lambda r: crra_utility(0.7, r))(1.0)))


def test_b4_subject_summary(models, gc_data):
    data, _ = gc_data
    b4 = models["B4"]
    p0 = b4.init_params(KEY, data, 0)
    s = b4.subject_summary(p0, data)
    assert list(s["subject_id"]) == list(data.subject_ids)
    assert {"prior_recovery_logit", "prior_recovery_prob", "u"} <= set(s.columns)


# ---------------------------------------------------------------------------------------------
# B2
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def b2_tiny():
    df, truth = gc.generate(4, 202)
    data = build_model_data(df)
    model = B2(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = model.fit_svi(data, steps=300, key=jax.random.PRNGKey(1), num_samples=40)
    return data, truth, model, post


def test_b2_parameter_count_documented(models, gc_data):
    data, _ = gc_data
    b2 = models["B2"]
    p0 = b2.init_params(KEY, data, 0)
    V = data.n_contexts
    assert b2.n_params(p0) == data.d_x + data.n_subjects * (V + 1) + 2 * V + 6
    assert b2.n_params(p0, include_random_effects=False) == data.d_x + 2 * V + 6
    assert p0["beta_ctx"].shape == (data.n_subjects, V) and p0["enc"]["u"].shape == (data.n_subjects, 1)


def test_b2_svi_posterior_and_pointwise_log_lik(b2_tiny):
    data, _, model, post = b2_tiny
    assert isinstance(post, B2Posterior) and post.method == "svi"
    assert set(post.samples) == set(SITES) and post.n_draws == 40
    assert post.samples["beta_ctx"].shape == (40, data.n_subjects, data.n_contexts)
    assert np.isfinite(post.diagnostics["losses"]).all()
    p = np.asarray(model.predict_proba(post.params, data))
    assert p.shape == (data.n,) and (p > 0).all() and (p < 1).all()
    ps = np.asarray(model.predict_proba(post.params, data, sample=True, samples=post, key=jax.random.PRNGKey(5)))
    assert ps.shape == (data.n,) and (ps > 0).all() and (ps < 1).all() and not np.allclose(ps, p)
    with pytest.raises(ValueError):
        model.predict_proba(post.params, data, sample=True)
    ll = model.pointwise_log_lik(post, data, chunk_size=16)
    assert ll.shape == (40, data.n) and np.isfinite(ll).all()
    assert np.all(ll[:, ~data.sell_mask()] == 0.0) and np.all(ll[:, data.sell_mask()] < 0.0)
    # draw 0 through params_from_draw equals row 0 of the matrix
    ll0 = np.asarray(model.log_lik(model.params_from_draw(post, 0), data))
    assert np.allclose(ll0, ll[0], atol=1e-10)
    # the posterior-mean pytree is the mean of the draws
    assert np.allclose(np.asarray(post.params["beta_ctx"]), post.samples["beta_ctx"].mean(axis=0))
    # subsets (train/test splits) keep the per-subject alignment
    sub = data.subset(np.arange(0, data.n, 3))
    assert model.pointwise_log_lik(post, sub).shape == (40, sub.n)


def test_b2_waic_and_loo_finite(b2_tiny):
    data, _, model, post = b2_tiny
    ll = model.pointwise_log_lik(post, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = model.waic(post, data, log_lik=ll)
        lo = model.loo(post, data, log_lik=ll)
    n_sell = int(data.sell_mask().sum())
    assert np.isfinite(float(w.elpd_waic)) and np.isfinite(float(w.p_waic)) and float(w.se) > 0
    assert int(w.n_data_points) == n_sell and int(w.n_samples) == 40 and w.waic_i.shape == (n_sell,)
    assert np.isfinite(float(lo.elpd_loo)) and np.isfinite(float(lo.p_loo)) and lo.loo_i.shape == (n_sell,)
    # elpd_waic is the sum over sell rows of log-mean-lik minus the variance penalty
    lppd = np.sum(np.log(np.mean(np.exp(ll[:, data.sell_mask()]), axis=0)))
    assert float(w.elpd_waic) == pytest.approx(lppd - float(w.p_waic), rel=1e-6)
    summary = model.subject_summary(post.params, data)
    assert list(summary["subject_id"]) == list(data.subject_ids) and f"beta_ctx[{data.ctx_vocab[0]}]" in summary


def test_b2_nuts_smoke(rng):
    subset, _ = D.battery_subset(rng, 16)
    df, _ = gc.generate(3, 303, design=subset)
    data = build_model_data(df)
    model = B2(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = model.fit_nuts(data, jax.random.PRNGKey(2), num_warmup=40, num_samples=40)
    assert post.method == "nuts" and post.n_draws == 40
    assert post.diagnostics["divergences"] >= 0 and np.isfinite(post.diagnostics["max_rhat"])
    assert model.pointwise_log_lik(post, data).shape == (40, data.n)


def test_b2_log_prior_matches_numpyro_priors_on_the_shared_terms(models, gc_data):
    data, _ = gc_data
    b2 = models["B2"]
    p0 = b2.init_params(KEY, data, 0)
    lp = float(b2.log_prior(p0))
    assert np.isfinite(lp)
    # moving the context slopes away from their population mean lowers the prior
    far = {**p0, "beta_ctx": p0["beta_ctx"] + 3.0}
    assert float(b2.log_prior(far)) < lp


# ---------------------------------------------------------------------------------------------
# Slow: parameter recovery on G_C (run once, numbers reported in STATUS.md by the main session)
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_slow_b1_recovers_gc_context_coefficients():
    t0 = time.time()
    df, truth = gc.generate(200, 1)
    data = build_model_data(df)
    b1 = B1(data)
    params, val = _adam_fit(b1, data, b1.init_params(KEY, data, 0), steps=1500, lr=0.05)
    coef = b1.coefficients(params)
    beta = np.asarray([truth["population"]["beta_ctx"][c] for c in data.ctx_vocab])
    fitted = np.asarray([coef[f"ctx{k}={c}"] for k in range(2) for c in data.ctx_vocab])
    r = pearsonr(fitted, np.concatenate([beta, beta]))[0]
    r0 = pearsonr(fitted[:4], beta)[0]
    print(
        f"\n[slow B1 on G_C N=200] context-coefficient correlation r={r:.3f} (position 0 only r0={r0:.3f}); "
        f"fitted ctx0={np.round(fitted[:4], 3).tolist()} ctx1={np.round(fitted[4:], 3).tolist()} truth={beta.tolist()}; "
        f"beta_L fitted={coef['loss']:.2f} truth={truth['population']['beta_L']}; objective={val:.1f}; {time.time() - t0:.0f}s"
    )
    assert r > 0.9


@pytest.mark.slow
def test_slow_b4_lambda_ordering_on_gc():
    """Reported, no threshold: B4 has no additive context slope, so its evidence ``lambda_c``
    (positive = recovery more likely = fewer sales) should be anti-ordered with G_C's
    ``beta_ctx`` (positive = more sales)."""
    t0 = time.time()
    df, truth = gc.generate(200, 2)
    data = build_model_data(df)
    b4 = B4(data)
    params, val = _adam_fit(b4, data, b4.init_params(KEY, data, 0), steps=1500, lr=0.02)
    nat = b4.natural_params(params)
    beta = np.asarray([truth["population"]["beta_ctx"][c] for c in data.ctx_vocab])
    lam = np.asarray([nat["lambda"][c] for c in data.ctx_vocab])
    rho_s = spearmanr(lam, -beta)[0]
    r = pearsonr(lam, -beta)[0]
    print(
        f"\n[slow B4 on G_C N=200] lambda vs -beta_ctx: Spearman={rho_s:.3f} Pearson={r:.3f}; "
        f"lambda={np.round(lam, 3).tolist()} truth beta={beta.tolist()}; delta={nat['delta']:.3f} rho={nat['rho']:.3f} "
        f"kappa={nat['kappa']:.3f} tau={nat['tau']:.2f} beta_tol={nat['beta_tol']:.3f}; objective={val:.1f}; {time.time() - t0:.0f}s"
    )
    assert np.isfinite(rho_s) and np.isfinite(val)


@pytest.mark.slow
def test_slow_b2_svi_recovers_gc_subject_intercepts():
    t0 = time.time()
    df, truth = gc.generate(100, 3)
    data = build_model_data(df)
    b2 = B2(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = b2.fit_svi(data, steps=3000, key=jax.random.PRNGKey(7), lr=0.02, num_samples=100)
    summary = b2.subject_summary(post.params, data)
    true_int = np.asarray([truth["subjects"][s]["sell_intercept"] for s in data.subject_ids])
    rho_s = spearmanr(summary["intercept"].to_numpy(), true_int)[0]
    pop = b2.population_params(post.params)
    beta = np.asarray([truth["population"]["beta_ctx"][c] for c in data.ctx_vocab])
    mu = np.asarray([pop["mu_ctx"][c] for c in data.ctx_vocab])
    r_ctx = pearsonr(mu, beta)[0]
    print(
        f"\n[slow B2 SVI on G_C N=100] Spearman(subject intercepts)={rho_s:.3f}; mu_ctx vs beta_ctx r={r_ctx:.3f} "
        f"mu_ctx={np.round(mu, 3).tolist()} truth={beta.tolist()}; beta_L={pop['beta_L']:.2f} (truth {truth['population']['beta_L']}); "
        f"sigma_u={pop['sigma_u']:.3f}; final ELBO loss={post.diagnostics['final_loss']:.1f}; {time.time() - t0:.0f}s"
    )
    assert rho_s > 0.6
