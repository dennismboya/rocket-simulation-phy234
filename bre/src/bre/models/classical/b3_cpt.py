"""B3 — cumulative prospect theory with a context-shifted reference point (PLAN.md section 4,
classical baseline).

The investor evaluates the two prospects "sell" and "hold" with the Tversky–Kahneman (1992)
cumulative prospect theory value and weighting functions, relative to a reference point that the
contexts and the question order shift, and chooses through a logit link. Everything is a
Kolmogorov probability model: contexts enter additively (through the reference point and the
recovery belief), so ordered pairs compose without pair parameters and B3 is a fair rival on
split (b) of PLAN.md section 6.

Value and weighting (per subject ``i``)
---------------------------------------
Outcomes ``x`` are wealth changes as a fraction of the current portfolio (``x = -L`` is the loss
``L = |loss_pct|``); ``z = x - r`` is the outcome relative to the reference point ``r``::

    v_i(z) = z^alpha_i                  for z >= 0
    v_i(z) = -lambda_i (-z)^beta_i      for z < 0        (beta_i = alpha_i unless separate_beta)
    w_i(p) = exp(-(-ln p)^gamma_i)                       (Prelec one-parameter weighting)

``lambda_i`` (loss aversion), ``alpha_i`` (curvature) and ``gamma_i`` (probability weighting)
are per-subject and positive. They are ``exp`` of the three outputs of the shared
:class:`bre.models.base.Encoder` with ``d_out = 3`` (the equal-footing rule of PLAN.md section 4)::

    (log lambda_i, log alpha_i, log gamma_i) = W^T x_i + b + u_i,     u_i ~ N(0, diag(sigma^2))

so each of the three is **log-normal around its covariate-predicted value** with a fitted scale:
that is the hierarchical shrinkage of the module's parameters, and :func:`encoder_log_prior`
is its log-density (the inverse-gamma hyperprior on ``sigma^2`` keeps the joint MAP bounded).
With ``separate_beta=True`` a population scalar ``log_beta_ratio`` gives
``beta_i = alpha_i exp(log_beta_ratio)``.

Reference point and recovery belief (per row)
---------------------------------------------
::

    r      = r0 + sum_k rho[c_k] + rho_order * order_flag + rho_tol * tol_yes
    p_rec  = sigmoid(pr0 + sum_k pr_ctx[c_k])

``rho`` and ``pr_ctx`` are ``(V,)`` vectors indexed like ``ModelData.ctx_vocab``; the ``none``
condition contributes nothing. ``tol_yes`` is 1 when the prior tolerance answer of a
tolerance-first row was "yes" (0 otherwise, including NaN) — the same information the shared
classical features carry. ``p_rec`` is the subjective probability that the position recovers to
the purchase price within the horizon (a learned sigmoid of a per-context term; kept simple on
purpose).

Prospects and choice
--------------------
* ``sell``: the sure outcome ``x = -L``, value ``V_sell = v(-L - r)``.
* ``hold``: the two-outcome lottery ``{x_hi = 0 with p_rec; x_lo = -kappa L with 1 - p_rec}``
  with ``kappa = exp(log_kappa) > 0`` the further-loss multiplier, valued with the cumulative
  (rank-dependent) weights of CPT (:func:`cpt_two_outcome`; ``z_hi = -r``, ``z_lo = -kappa L - r``):

  - both outcomes gains (``z_lo >= 0``):  ``V = w(p_hi) v(z_hi) + (1 - w(p_hi)) v(z_lo)``
  - both losses (``z_hi <= 0``):          ``V = w(p_lo) v(z_lo) + (1 - w(p_lo)) v(z_hi)``
  - mixed (``z_lo < 0 < z_hi``):          ``V = w(p_hi) v(z_hi) + w(p_lo) v(z_lo)``

  (one weighting function for gains and losses, the one-parameter Prelec form above).
* ``P(sell) = sigmoid(tau (V_sell - V_hold))`` with the choice temperature ``tau = exp(log_tau)``.

Signs (reported, not assumed; ``tests/test_b3_b5_b6.py`` checks the identities): with ``r = 0``
both hold outcomes are losses, so ``V_sell - V_hold = lambda L^beta [w(1 - p_rec) kappa^beta - 1]``:
selling is preferred iff the weighted further loss outweighs the sure loss. A larger ``p_rec``
lowers ``P(sell)``. Shifting the reference point up (``rho > 0``) makes every outcome look worse;
its effect on the sell/hold difference depends on the curvature and is a fitted quantity.

Parameters
----------
``{"enc": {W (d_x, 3), b (3,), u (n_subjects, 3), log_sigma_u (3,)}, "r0": (), "rho_ctx": (V,),
"rho_order": (), "rho_tol": (), "pr0": (), "pr_ctx": (V,), "log_kappa": (), "log_tau": ()
[, "log_beta_ratio": ()]}``. ``n_params = (3 d_x + 3 + 3 n_subjects + 3) + 2 V + 6``
(+1 with ``separate_beta``); without the random effects ``u``: ``3 d_x + 6 + 2 V + 6`` (+1).
Prior: :func:`encoder_log_prior` (the log-normal shrinkage) plus an optional ridge
``-(l2 / 2) (||rho_ctx||^2 + ||pr_ctx||^2)`` (default 0).

Initialization (restart folded in with ``fold_in``; these are optimizer starting points, not
fitted values): the encoder bias starts at ``(log 2.25, log 0.88, log 0.65)``, magnitudes in the
range commonly reported for the TK form, plus the encoder's small noise; ``r0 = 0``,
``rho_*, pr_* ~ N(0, 0.1^2)``, ``log_kappa = log 3``, ``log_tau = log 10`` (plus ``N(0, 0.1^2)``).
``kappa = 3`` starts in the regime where ``P(sell)`` rises with ``L`` under the starting values.

Numerics: ``|z|^alpha`` has an infinite derivative at ``z = 0`` for ``alpha < 1``; ``cpt_value``
guards it (``|z| < Z_EPS`` gives value 0 and gradient 0), which matters because the hold
outcome ``x_hi = 0`` sits exactly at the reference point when ``r = 0``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bre.models.base import Encoder, ModelBase, Params, count_params
from bre.models.data import ModelData

Z_EPS = 1e-12
"""``|z|`` below which the value function returns 0 with a zero gradient (see module docstring)."""

P_EPS = 1e-9
"""Probabilities are clipped to ``[P_EPS, 1 - P_EPS]`` inside the Prelec weighting."""

INIT_NOISE = 0.1
"""Standard deviation of the initialization noise on the scalar and vector parameters."""

INIT_LOG_LAMBDA = float(np.log(2.25))
INIT_LOG_ALPHA = float(np.log(0.88))
INIT_LOG_GAMMA = float(np.log(0.65))
INIT_LOG_KAPPA = float(np.log(3.0))
INIT_LOG_TAU = float(np.log(10.0))


def cpt_value(z, alpha, beta, lam) -> jnp.ndarray:
    """``v(z) = z^alpha`` for ``z >= 0``, ``-lam (-z)^beta`` for ``z < 0``, with the ``Z_EPS``
    guard at zero; all arguments broadcast (per-row ``alpha``, ``beta``, ``lam``)."""
    z = jnp.asarray(z, dtype=jnp.float64)
    az = jnp.abs(z)
    ok = az > Z_EPS
    safe = jnp.where(ok, az, 1.0)
    gain = jnp.where(ok, safe ** jnp.asarray(alpha, dtype=jnp.float64), 0.0)
    loss = jnp.where(ok, safe ** jnp.asarray(beta, dtype=jnp.float64), 0.0)
    return jnp.where(z >= 0, gain, -jnp.asarray(lam, dtype=jnp.float64) * loss)


def prelec_weight(p, gamma) -> jnp.ndarray:
    """Prelec one-parameter weighting ``w(p) = exp(-(-ln p)^gamma)`` (``w(0) = 0``, ``w(1) = 1``,
    fixed point at ``1/e``); ``p`` clipped to ``[P_EPS, 1 - P_EPS]``."""
    p = jnp.clip(jnp.asarray(p, dtype=jnp.float64), P_EPS, 1.0 - P_EPS)
    return jnp.exp(-((-jnp.log(p)) ** jnp.asarray(gamma, dtype=jnp.float64)))


def cpt_two_outcome(z_hi, p_hi, z_lo, alpha, beta, lam, gamma) -> jnp.ndarray:
    """CPT value of the lottery ``{z_hi with p_hi; z_lo with 1 - p_hi}`` (``z_hi >= z_lo``,
    outcomes already relative to the reference point) with the rank-dependent weights of the
    module docstring (gains: the best outcome is weighted by ``w(p_hi)``; losses: the worst by
    ``w(p_lo)``; mixed: each by the weight of its own probability)."""
    p_hi = jnp.asarray(p_hi, dtype=jnp.float64)
    p_lo = 1.0 - p_hi
    v_hi = cpt_value(z_hi, alpha, beta, lam)
    v_lo = cpt_value(z_lo, alpha, beta, lam)
    w_hi = prelec_weight(p_hi, gamma)
    w_lo = prelec_weight(p_lo, gamma)
    both_gains = w_hi * v_hi + (1.0 - w_hi) * v_lo
    both_losses = w_lo * v_lo + (1.0 - w_lo) * v_hi
    mixed = w_hi * v_hi + w_lo * v_lo
    z_lo = jnp.asarray(z_lo, dtype=jnp.float64)
    z_hi = jnp.asarray(z_hi, dtype=jnp.float64)
    return jnp.where(z_lo >= 0, both_gains, jnp.where(z_hi <= 0, both_losses, mixed))


class B3(ModelBase):
    """Cumulative prospect theory with a context-shifted reference point (module docstring)."""

    name = "B3"
    family = "classical"

    def __init__(self, data: ModelData, *, separate_beta: bool = False, l2: float = 0.0, w_l2: float = 0.0) -> None:
        if l2 < 0 or w_l2 < 0:
            raise ValueError("l2 and w_l2 must be >= 0")
        self.enc = Encoder.for_data(data, d_out=3)
        self.ctx_vocab = tuple(data.ctx_vocab)
        self.n_contexts = int(data.n_contexts)
        self.separate_beta = bool(separate_beta)
        self.l2 = float(l2)
        self.w_l2 = float(w_l2)

    # -- model contract ---------------------------------------------------------------------------

    def init_params(self, rng_key, data: ModelData, restart: int = 0) -> Params:
        key = self.restart_key(rng_key, restart)
        keys = jax.random.split(key, 9)
        f64 = jnp.float64
        noise = lambda k, shape=(): INIT_NOISE * jax.random.normal(k, shape, dtype=f64)  # noqa: E731
        enc = self.enc.init(rng_key, restart)
        enc["b"] = enc["b"] + jnp.asarray([INIT_LOG_LAMBDA, INIT_LOG_ALPHA, INIT_LOG_GAMMA], dtype=f64)
        params: dict[str, Any] = {
            "enc": enc,
            "r0": noise(keys[0]),
            "rho_ctx": noise(keys[1], (self.n_contexts,)),
            "rho_order": noise(keys[2]),
            "rho_tol": noise(keys[3]),
            "pr0": noise(keys[4]),
            "pr_ctx": noise(keys[5], (self.n_contexts,)),
            "log_kappa": INIT_LOG_KAPPA + noise(keys[6]),
            "log_tau": INIT_LOG_TAU + noise(keys[7]),
        }
        if self.separate_beta:
            params["log_beta_ratio"] = noise(keys[8])
        return params

    def subject_params(self, params: Params, data: ModelData) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """``(lambda, alpha, beta, gamma)``, each ``(n_subjects,)``: ``exp`` of the encoder outputs
        (``beta = alpha exp(log_beta_ratio)`` with ``separate_beta``, else ``beta = alpha``)."""
        z = self.enc.apply_subjects(params["enc"], data.X)
        lam, alpha, gamma = jnp.exp(z[:, 0]), jnp.exp(z[:, 1]), jnp.exp(z[:, 2])
        beta = alpha * jnp.exp(params["log_beta_ratio"]) if self.separate_beta else alpha
        return lam, alpha, beta, gamma

    def _ctx_sum(self, vec, data: ModelData) -> jnp.ndarray:
        ctx_idx = jnp.asarray(data.ctx_idx)
        ctx_mask = jnp.asarray(data.ctx_mask)
        v = jnp.asarray(vec, dtype=jnp.float64)
        return jnp.sum(jnp.where(ctx_mask, v[jnp.where(ctx_mask, ctx_idx, 0)], 0.0), axis=1)

    def reference_point(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` ``r = r0 + sum_k rho[c_k] + rho_order order_flag + rho_tol tol_yes``."""
        tol = jnp.asarray(data.tol_answer, dtype=jnp.float64)
        tol_yes = jnp.where(jnp.isfinite(tol) & (tol == 1.0), 1.0, 0.0)
        order = jnp.asarray(data.order_flag, dtype=jnp.float64)
        return params["r0"] + self._ctx_sum(params["rho_ctx"], data) + params["rho_order"] * order + params["rho_tol"] * tol_yes

    def recovery_probability(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` ``p_rec = sigmoid(pr0 + sum_k pr_ctx[c_k])``."""
        return jax.nn.sigmoid(params["pr0"] + self._ctx_sum(params["pr_ctx"], data))

    def prospect_values(self, params: Params, data: ModelData) -> tuple[jnp.ndarray, jnp.ndarray]:
        """``(V_sell, V_hold)``, each ``(n,)`` (module docstring)."""
        lam, alpha, beta, gamma = self.subject_params(params, data)
        s = jnp.asarray(data.subject_idx)
        lam, alpha, beta, gamma = lam[s], alpha[s], beta[s], gamma[s]
        L = jnp.asarray(data.loss, dtype=jnp.float64)
        r = self.reference_point(params, data)
        p_rec = self.recovery_probability(params, data)
        kappa = jnp.exp(params["log_kappa"])
        v_sell = cpt_value(-L - r, alpha, beta, lam)
        v_hold = cpt_two_outcome(-r, p_rec, -kappa * L - r, alpha, beta, lam, gamma)
        return v_sell, v_hold

    def predict_proba(self, params: Params, data: ModelData) -> jnp.ndarray:
        v_sell, v_hold = self.prospect_values(params, data)
        return jax.nn.sigmoid(jnp.exp(params["log_tau"]) * (v_sell - v_hold))

    def log_prior(self, params: Params) -> jnp.ndarray:
        """Log-normal shrinkage of ``(lambda_i, alpha_i, gamma_i)`` through the encoder prior,
        plus the optional ridge on the context vectors."""
        lp = self.enc.log_prior(params["enc"], w_l2=self.w_l2)
        if self.l2:
            lp = lp - 0.5 * self.l2 * (jnp.sum(jnp.asarray(params["rho_ctx"]) ** 2) + jnp.sum(jnp.asarray(params["pr_ctx"]) ** 2))
        return lp

    def n_params(self, params: Params, include_random_effects: bool = True) -> int:
        """``(3 d_x + 3 + 3 n_subjects + 3) + 2 V + 6`` (+1 with ``separate_beta``); without
        ``u``: ``3 d_x + 6 + 2 V + 6`` (+1)."""
        enc = self.enc.n_params(params["enc"], include_random_effects=include_random_effects)
        rest = count_params({k: v for k, v in params.items() if k != "enc"})
        return enc + rest

    # -- reporting --------------------------------------------------------------------------------

    def natural_params(self, params: Params) -> dict[str, Any]:
        """Population-level scalars by name: ``r0``, ``rho_order``, ``rho_tol``, ``pr0``,
        ``kappa``, ``tau``, ``beta_ratio``; ``rho_ctx`` and ``pr_ctx`` by context tag; and the
        value-function parameters at zero (mean) covariates, ``lambda0, alpha0, gamma0 = exp(b)``."""
        b = np.asarray(params["enc"]["b"], dtype=np.float64)
        return {
            "r0": float(params["r0"]),
            "rho_order": float(params["rho_order"]),
            "rho_tol": float(params["rho_tol"]),
            "pr0": float(params["pr0"]),
            "kappa": float(jnp.exp(params["log_kappa"])),
            "tau": float(jnp.exp(params["log_tau"])),
            "beta_ratio": float(jnp.exp(params["log_beta_ratio"])) if self.separate_beta else 1.0,
            "lambda0": float(np.exp(b[0])),
            "alpha0": float(np.exp(b[1])),
            "gamma0": float(np.exp(b[2])),
            "rho_ctx": {tag: float(v) for tag, v in zip(self.ctx_vocab, np.asarray(params["rho_ctx"]))},
            "pr_ctx": {tag: float(v) for tag, v in zip(self.ctx_vocab, np.asarray(params["pr_ctx"]))},
        }

    def subject_summary(self, params: Params, data: ModelData) -> pd.DataFrame | None:
        """One row per subject: ``lambda``, ``alpha``, ``beta``, ``gamma`` and the random effects
        ``u_log_lambda``, ``u_log_alpha``, ``u_log_gamma``."""
        lam, alpha, beta, gamma = (np.asarray(v) for v in self.subject_params(params, data))
        u = np.asarray(params["enc"]["u"], dtype=np.float64)
        return pd.DataFrame(
            {
                "subject_id": data.subject_ids,
                "lambda": lam,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "u_log_lambda": u[:, 0],
                "u_log_alpha": u[:, 1],
                "u_log_gamma": u[:, 2],
            }
        )


__all__ = [
    "B3",
    "INIT_LOG_ALPHA",
    "INIT_LOG_GAMMA",
    "INIT_LOG_KAPPA",
    "INIT_LOG_LAMBDA",
    "INIT_LOG_TAU",
    "P_EPS",
    "Z_EPS",
    "cpt_two_outcome",
    "cpt_value",
    "prelec_weight",
]
