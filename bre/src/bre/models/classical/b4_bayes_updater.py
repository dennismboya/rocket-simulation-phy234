"""B4 — Bayesian belief updater with CRRA expected utility (PLAN.md section 4, classical baseline).

The investor holds a belief about the event R = "the market recovers within the horizon", updates
it with each context as a piece of evidence, and sells when the expected utility of locking in the
loss exceeds the expected utility of holding. No pair parameters exist: an ordered pair of
contexts composes through the sum of their log-likelihood ratios with a recency discount, which is
what makes B4 the fair classical rival of the Q-models on split (b) of PLAN.md section 6.

Belief
------
Prior log-odds of recovery for subject ``s(i)`` come from the shared encoder with ``d_out = 1``
(:class:`bre.models.base.Encoder`, the equal-footing rule)::

    prior_i = W^T x_{s(i)} + b + u_{s(i)}                       (= Encoder(x_i) + u_i)

Each context ``c`` in the vocabulary carries a learned log-likelihood ratio ``lambda_c``
(``lambda`` of shape ``(V,)``, indexed like ``ModelData.ctx_vocab``), the evidence
``log [P(c | R) / P(c | not R)]``. Evidence shown at position ``k`` of ``m`` contexts is discounted
by ``delta^(m - k)`` (``k = 1..m``: the last context enters undiscounted, the one before it by
``delta``), with ``delta = sigmoid(delta_logit)`` in ``(0, 1)``. The prior tolerance answer of a
tolerance-first row (``tol_answer`` in ``{0, 1}``, NaN otherwise -> treated as 0) enters as
further evidence with weight ``beta_tol``::

    logit p_rec,i = prior_i + sum_{k=1}^{m} delta^(m-k) lambda_{c_k} + beta_tol * tol_answer_i
    p_rec,i       = sigmoid(logit p_rec,i)

Decision
--------
CRRA utility over wealth relative to the current portfolio (wealth 1), with ``rho = exp(log_rho)
> 0`` and ``rho = 1`` the log limit::

    u(w) = (w^(1 - rho) - 1) / (1 - rho)        u(1) = 0,   u -> log w as rho -> 1

Selling now locks in the loss ``L = |loss_pct|``: ``EU_sell = u(1 - L)``. Holding pays ``u(1) = 0``
if the market recovers and a further loss otherwise, with ``kappa = exp(log_kappa) > 0`` the
further-loss multiplier::

    EU_hold = p_rec * u(1) + (1 - p_rec) * u(1 - kappa L) = (1 - p_rec) * u(1 - kappa L)
    P(sell) = sigmoid(tau * (EU_sell - EU_hold)),   tau = exp(log_tau) > 0 the choice temperature

Wealth is floored at ``W_FLOOR`` inside ``u`` so that ``kappa L >= 1`` (ruin) stays finite.

Signs and monotonicity (tests/test_b1_b2_b4.py checks them):

* ``lambda_c > 0`` raises the recovery belief, which lowers ``P(sell)`` for every ``L > 0``:
  ``EU_hold`` increases with ``p_rec`` because ``u(1 - kappa L) < 0``, so ``EU_sell - EU_hold``
  decreases. With ``lambda = 0`` every context condition gives the same probability whatever
  ``delta`` is.
* In ``L``: ``d/dL (EU_sell - EU_hold) = -u'(1 - L) + (1 - p_rec) kappa u'(1 - kappa L)``. With
  ``kappa > 1`` and ``rho > 0`` the second marginal utility exceeds the first, so ``P(sell)`` is
  increasing in ``L`` whenever ``kappa (1 - p_rec) >= 1`` (the expected further loss from holding
  outweighs the locked-in loss at the margin). When ``kappa (1 - p_rec) < 1`` a small loss makes
  holding *more* attractive as ``L`` grows (locking in the loss costs more than the expected
  further loss) and ``P(sell)`` falls before it rises; the direction is a property of the fitted
  ``(kappa, rho, p_rec)`` and is reported, not assumed.

Parameters
----------
``{"enc": {W, b, u, log_sigma_u}, "lambda": (V,), "delta_logit": (), "log_rho": (),
"log_kappa": (), "log_tau": (), "beta_tol": ()}``. ``beta_tol`` is the evidence weight of the
prior tolerance answer written in the belief equation above. Count:
``(d_x + 1 + n_subjects + 1) + V + 5``; without the random effects ``u``:
``d_x + 2 + V + 5``. Prior: the encoder's random-effect prior (:func:`encoder_log_prior`) plus
an optional ridge ``-(l2_lambda / 2) ||lambda||^2`` (default 0).

Initialization (restart folded in with ``fold_in``): encoder as :meth:`Encoder.init`;
``lambda ~ N(0, 0.1^2)``; ``delta_logit = 0 + N(0, 0.1^2)`` (``delta = 0.5``);
``log_rho = 0 + N(0, 0.1^2)`` (``rho = 1``); ``log_kappa = log 2 + N(0, 0.1^2)``;
``log_tau = log 10 + N(0, 0.1^2)``; ``beta_tol ~ N(0, 0.1^2)``. These are optimizer starting
points, not fitted values.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bre.models.base import Encoder, ModelBase, Params, count_params
from bre.models.data import ModelData

W_FLOOR = 1e-3
"""Wealth floor inside the CRRA utility: ``u`` is evaluated at ``max(w, W_FLOOR)``."""

INIT_NOISE = 0.1
"""Standard deviation of the initialization noise on the scalar parameters."""

INIT_LOG_KAPPA = float(np.log(2.0))
INIT_LOG_TAU = float(np.log(10.0))
RHO_LOG_LIMIT_EPS = 1e-6
"""``|1 - rho|`` below which ``u`` uses the log form."""


def crra_utility(w, rho) -> jnp.ndarray:
    """CRRA utility ``(w^(1-rho) - 1) / (1 - rho)`` with the ``rho = 1`` log limit and the wealth
    floor :data:`W_FLOOR`. Computed as ``expm1((1 - rho) log w) / (1 - rho)`` for numerical
    stability near ``rho = 1``; differentiable in both arguments."""
    w = jnp.maximum(jnp.asarray(w, dtype=jnp.float64), W_FLOOR)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    one_minus = 1.0 - rho
    close = jnp.abs(one_minus) < RHO_LOG_LIMIT_EPS
    safe = jnp.where(close, 1.0, one_minus)
    logw = jnp.log(w)
    return jnp.where(close, logw, jnp.expm1(safe * logw) / safe)


def recency_weights(ctx_mask) -> jnp.ndarray:
    """``(n, K)`` exponents ``m - k`` (``k`` 1-based position, ``m`` the number of contexts of the
    row): 0 for the last context, 1 for the one before it, ...; 0 where masked."""
    mask = jnp.asarray(ctx_mask, dtype=jnp.float64)
    m = jnp.sum(mask, axis=1, keepdims=True)
    k = jnp.arange(1, mask.shape[1] + 1, dtype=jnp.float64)[None, :]
    return jnp.where(mask > 0, m - k, 0.0)


class B4(ModelBase):
    """Bayesian belief updater with recency-discounted evidence and CRRA expected utility."""

    name = "B4"
    family = "classical"

    def __init__(self, data: ModelData, l2_lambda: float = 0.0, w_l2: float = 0.0) -> None:
        if l2_lambda < 0 or w_l2 < 0:
            raise ValueError("l2_lambda and w_l2 must be >= 0")
        self.enc = Encoder.for_data(data, d_out=1)
        self.ctx_vocab = tuple(data.ctx_vocab)
        self.n_contexts = int(data.n_contexts)
        self.l2_lambda = float(l2_lambda)
        self.w_l2 = float(w_l2)

    # -- model contract ---------------------------------------------------------------------------

    def init_params(self, rng_key, data: ModelData, restart: int = 0) -> Params:
        key = self.restart_key(rng_key, restart)
        k_enc, k_lam, k_d, k_r, k_k, k_t, k_b = jax.random.split(key, 7)
        f64 = jnp.float64
        noise = lambda k, shape=(): INIT_NOISE * jax.random.normal(k, shape, dtype=f64)  # noqa: E731
        return {
            "enc": self.enc.init(k_enc, 0),
            "lambda": noise(k_lam, (self.n_contexts,)),
            "delta_logit": noise(k_d),
            "log_rho": noise(k_r),
            "log_kappa": INIT_LOG_KAPPA + noise(k_k),
            "log_tau": INIT_LOG_TAU + noise(k_t),
            "beta_tol": noise(k_b),
        }

    def prior_logit(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` prior log-odds of recovery: ``Encoder(x_i) + u_i``."""
        return self.enc.apply(params["enc"], data.X, data.subject_idx)[:, 0]

    def evidence(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` discounted context evidence ``sum_k delta^(m-k) lambda_{c_k}``."""
        ctx_idx = jnp.asarray(data.ctx_idx)
        ctx_mask = jnp.asarray(data.ctx_mask)
        lam = jnp.asarray(params["lambda"], dtype=jnp.float64)
        delta = jax.nn.sigmoid(jnp.asarray(params["delta_logit"], dtype=jnp.float64))
        lam_k = jnp.where(ctx_mask, lam[jnp.where(ctx_mask, ctx_idx, 0)], 0.0)
        weights = jnp.where(ctx_mask, delta ** recency_weights(ctx_mask), 0.0)
        return jnp.sum(weights * lam_k, axis=1)

    def recovery_logit(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` posterior log-odds of recovery (belief equation of the module docstring)."""
        tol = jnp.asarray(data.tol_answer, dtype=jnp.float64)
        tol = jnp.where(jnp.isfinite(tol), tol, 0.0)
        return self.prior_logit(params, data) + self.evidence(params, data) + params["beta_tol"] * tol

    def expected_utilities(self, params: Params, data: ModelData) -> tuple[jnp.ndarray, jnp.ndarray]:
        """``(EU_sell, EU_hold)``, each ``(n,)``."""
        p_rec = jax.nn.sigmoid(self.recovery_logit(params, data))
        rho = jnp.exp(params["log_rho"])
        kappa = jnp.exp(params["log_kappa"])
        L = jnp.asarray(data.loss, dtype=jnp.float64)
        eu_sell = crra_utility(1.0 - L, rho)
        eu_hold = (1.0 - p_rec) * crra_utility(1.0 - kappa * L, rho)  # + p_rec * u(1) = 0
        return eu_sell, eu_hold

    def predict_proba(self, params: Params, data: ModelData) -> jnp.ndarray:
        eu_sell, eu_hold = self.expected_utilities(params, data)
        tau = jnp.exp(params["log_tau"])
        return jax.nn.sigmoid(tau * (eu_sell - eu_hold))

    def log_prior(self, params: Params) -> jnp.ndarray:
        lp = self.enc.log_prior(params["enc"], w_l2=self.w_l2)
        if self.l2_lambda:
            lp = lp - 0.5 * self.l2_lambda * jnp.sum(jnp.asarray(params["lambda"]) ** 2)
        return lp

    def n_params(self, params: Params, include_random_effects: bool = True) -> int:
        """``(d_x + 1 + n_subjects + 1) + V + 5``; without ``u``: ``d_x + 2 + V + 5``."""
        enc = self.enc.n_params(params["enc"], include_random_effects=include_random_effects)
        rest = count_params({k: v for k, v in params.items() if k != "enc"})
        return enc + rest

    # -- reporting --------------------------------------------------------------------------------

    def natural_params(self, params: Params) -> dict[str, Any]:
        """The interpretable scalars: ``delta``, ``rho``, ``kappa``, ``tau``, ``beta_tol`` and
        ``lambda`` by context tag."""
        return {
            "delta": float(jax.nn.sigmoid(params["delta_logit"])),
            "rho": float(jnp.exp(params["log_rho"])),
            "kappa": float(jnp.exp(params["log_kappa"])),
            "tau": float(jnp.exp(params["log_tau"])),
            "beta_tol": float(params["beta_tol"]),
            "lambda": {tag: float(v) for tag, v in zip(self.ctx_vocab, np.asarray(params["lambda"]))},
        }

    def subject_summary(self, params: Params, data: ModelData) -> pd.DataFrame | None:
        """One row per subject: the prior recovery log-odds and probability, and ``u``."""
        z = np.asarray(self.enc.apply_subjects(params["enc"], data.X))[:, 0]
        return pd.DataFrame(
            {
                "subject_id": data.subject_ids,
                "prior_recovery_logit": z,
                "prior_recovery_prob": 1.0 / (1.0 + np.exp(-z)),
                "u": np.asarray(params["enc"]["u"])[:, 0],
            }
        )


__all__ = ["B4", "INIT_LOG_KAPPA", "INIT_LOG_TAU", "W_FLOOR", "crra_utility", "recency_weights"]
