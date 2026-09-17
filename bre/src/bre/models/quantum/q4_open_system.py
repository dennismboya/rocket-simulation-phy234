"""Q4 — the open-system variant of Q2 (PLAN.md section 4): density matrices with per-investor
dephasing.

Model
-----
Everything of :mod:`bre.models.quantum.q2_context_unitary` (state from the shared encoder, loss
unitary, context unitaries in presentation order, tolerance projector with Lüders collapse)
applied to ``rho_s = |psi_s><psi_s|`` with Lindblad pure dephasing in the sell basis
(``L = sigma_z``) at the subject's rate ``gamma_s`` after ``U_L`` and after every context
unitary: the off-diagonal elements of ``rho`` are multiplied by ``exp(-2 gamma_s)`` at each step
(:func:`bre.models.quantum.core.dephase`). ``P(sell) = Tr(P_sell rho')``; this is exactly
:func:`bre.models.quantum.core.q_forward_rho` per row.

Limits (proved by ``tests/test_model_core.py`` on the core and re-checked on the model in
``tests/test_q2_q4.py``): ``gamma_s = 0`` reproduces Q2 exactly; ``gamma_s -> inf`` (the model
uses ``log_gamma = ln 1e3`` as "infinite") makes every step a classical Markov transition with
``T_ij = |(U)_ij|^2`` — the initial coherence of ``rho_s`` still enters the first step (the loss
unitary) and is then destroyed. ``gamma_s`` is therefore the per-investor *consistency score*
of PLAN.md section 4: near 0 the investor's answers show the full interference pattern
(order effects, LTP violations), large values mean context effects compose like a classical
chain of transition probabilities.

Parameters: Q2's plus ``"log_gamma": (n_subjects,)`` with the LogNormal population prior
``log gamma_s ~ N(gamma_mu, exp(gamma_log_sigma)^2)`` whose two hyperparameters
``"gamma_mu"`` and ``"gamma_log_sigma"`` are fitted; an inverse-gamma hyperprior
:data:`GAMMA_SIGMA_INV_GAMMA` on the variance keeps the joint MAP bounded (the same device as
the encoder's random-effect prior). ``n_params`` adds ``n_subjects + 2`` (``log_gamma`` is a
per-subject random effect and is left out when ``include_random_effects`` is False).

Initialization: ``log_gamma_s = log(GAMMA_INIT) + 0.1 eps_s``, ``gamma_mu = log(GAMMA_INIT)``,
``gamma_log_sigma = 0``. ``GAMMA_INIT = 0.5`` starts every subject mid-way between the coherent
and the classical regime (per-step coherence factor ``exp(-1) ~ 0.37``), where the gradient with
respect to ``log_gamma`` is largest; starting near either limit would leave ``log_gamma`` stuck
(``d exp(-2 gamma) / d log gamma = -2 gamma exp(-2 gamma)`` vanishes at both ends).

Identifiability: ``gamma_s`` is identified by how far the subject's row probabilities sit between
the coherent and the Markov predictions; with chains that are diagonal in the sell basis
(``sigma_z`` rotations only) the two coincide and ``gamma_s`` is not identified (INTERFACE.md
section 7). The gauge group of Q2 acts on Q4 unchanged (dephasing in the sell basis commutes
with both ``K`` and ``Z``).

Interference terms: ``ltp`` and ``order`` are evaluated on the open-system chain
(:func:`core.ltp_interference_rho`, :func:`core.order_effect_rho` semantics through the fast
forward pass, so that ``order`` can be non-zero for commuting unitaries when ``gamma > 0``, as
the core documents); ``mix`` is inherited from Q2 and evaluated on the pure pre-dephasing state
(a coherent superposition of context branches has no density-matrix analogue).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bre import design as _design
from bre.models.base import Params
from bre.models.data import ModelData
from bre.models.quantum import core as C
from bre.models.quantum.q2_context_unitary import Q2, forward_rho_fast, needs_mixture, predict_new_subject

GAMMA_INIT = 0.5
"""Initial dephasing rate of every subject (module docstring)."""

GAMMA_INIT_NOISE = 0.1
"""Standard deviation of the per-subject noise on the initial ``log_gamma``."""

GAMMA_SIGMA_INV_GAMMA: tuple[float, float] = (2.0, 1.0)
"""``(alpha, beta)`` of the inverse-gamma hyperprior on ``sigma_gamma^2`` (prior mean
``beta / (alpha - 1)`` = 1, i.e. one natural-log unit of spread in ``gamma``)."""

LOG_GAMMA_CLASSICAL = float(np.log(1e3))
"""``log_gamma`` value at which the model is numerically the classical Markov chain
(``exp(-2 gamma) ~ 1e-869``, i.e. 0 in float64)."""


def lognormal_population_log_prior(log_gamma, gamma_mu, gamma_log_sigma) -> jnp.ndarray:
    """``sum_s log N(log_gamma_s | gamma_mu, sigma^2) + log InvGamma(sigma^2 | alpha, beta)`` with
    ``sigma = exp(gamma_log_sigma)`` and ``(alpha, beta)`` = :data:`GAMMA_SIGMA_INV_GAMMA`; all
    constants included."""
    lg = jnp.asarray(log_gamma, dtype=jnp.float64)
    mu = jnp.asarray(gamma_mu, dtype=jnp.float64)
    log_sigma = jnp.asarray(gamma_log_sigma, dtype=jnp.float64)
    sigma2 = jnp.exp(2.0 * log_sigma)
    n = lg.shape[0]
    gauss = -0.5 * jnp.sum((lg - mu) ** 2) / sigma2 - n * log_sigma - 0.5 * n * jnp.log(2.0 * jnp.pi)
    alpha, beta = GAMMA_SIGMA_INV_GAMMA
    inv_gamma = alpha * jnp.log(beta) - jax.scipy.special.gammaln(alpha) - (alpha + 1.0) * jnp.log(sigma2) - beta / sigma2
    return gauss + inv_gamma


class Q4(Q2):
    """Open-system context-unitary model (module docstring)."""

    name = "Q4"
    family = "quantum"

    def init_params(self, rng_key, data: ModelData, restart: int = 0) -> Params:
        params = super().init_params(rng_key, data, restart)
        key = jax.random.fold_in(self.restart_key(rng_key, restart), 4)
        noise = GAMMA_INIT_NOISE * jax.random.normal(key, (data.n_subjects,), dtype=jnp.float64)
        params["log_gamma"] = float(np.log(GAMMA_INIT)) + noise
        params["gamma_mu"] = jnp.asarray(float(np.log(GAMMA_INIT)), dtype=jnp.float64)
        params["gamma_log_sigma"] = jnp.asarray(0.0, dtype=jnp.float64)
        return params

    def gammas(self, params: Params) -> jnp.ndarray:
        """``(n_subjects,)`` dephasing rates ``exp(log_gamma)``."""
        return jnp.exp(jnp.asarray(params["log_gamma"], dtype=jnp.float64))

    def _predict_rho(self, params: Params, data: ModelData, *, order_flag=None, tol_answer=None, ctx_idx=None, ctx_mask=None) -> jnp.ndarray:
        r = self._rows(data)
        if tuple(jnp.shape(params["log_gamma"])) != (data.n_subjects,):
            raise ValueError("params['log_gamma'] must have one entry per subject of the data")
        rho = jax.vmap(C.to_rho)(self.states(params, data))
        gamma = self.gammas(params)
        theta_ctx = self.context_table(params)
        order_flag = r["order_flag"] if order_flag is None else order_flag
        tol_answer = r["tol_answer"] if tol_answer is None else tol_answer
        ctx_idx = r["ctx_idx"] if ctx_idx is None else ctx_idx
        ctx_mask = r["ctx_mask"] if ctx_mask is None else ctx_mask
        if self.forward == "core":
            ctx = C.gather_context_thetas(theta_ctx, ctx_idx, ctx_mask)
            return C.q_forward_rho_batch(
                rho[r["subject_idx"]], params["theta_L"], r["loss"], ctx, ctx_mask, order_flag, params["phi"],
                tol_answer, gamma[r["subject_idx"]],
            )
        return forward_rho_fast(
            rho, gamma, params["theta_L"], theta_ctx, params["phi"], r["loss_levels"], r["loss_idx"],
            r["subject_idx"], ctx_idx, ctx_mask, order_flag, tol_answer, mixture=needs_mixture(order_flag, tol_answer),
        )

    def predict_proba(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` ``P(sell)`` with dephasing at each subject's rate."""
        return self._predict_rho(params, data)

    def log_prior(self, params: Params) -> jnp.ndarray:
        """Q2's prior plus the LogNormal population prior on ``gamma`` and its hyperprior."""
        return super().log_prior(params) + lognormal_population_log_prior(
            params["log_gamma"], params["gamma_mu"], params["gamma_log_sigma"]
        )

    def n_params(self, params: Params, include_random_effects: bool = True) -> int:
        """Q2's count + ``n_subjects`` (``log_gamma``, a random effect) + 2 hyperparameters."""
        base = super().n_params(params, include_random_effects=include_random_effects)
        n_gamma = int(jnp.shape(params["log_gamma"])[0]) if include_random_effects else 0
        return base + n_gamma + 2

    def interference_terms(self, params: Params, data: ModelData, mix_weights=None) -> dict[str, jnp.ndarray]:
        r = self._rows(data)
        n = data.n
        zeros, ones, nans = np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64), np.full(n, np.nan)
        p_sf = self._predict_rho(params, data, order_flag=zeros, tol_answer=nans)
        p_tf = self._predict_rho(params, data, order_flag=ones, tol_answer=nans)
        pair = r["ctx_mask"].all(axis=1)
        p21 = self._predict_rho(
            params, data, order_flag=zeros, tol_answer=nans,
            ctx_idx=r["ctx_idx"][:, ::-1].copy(), ctx_mask=r["ctx_mask"][:, ::-1].copy(),
        )
        out = {"ltp": p_sf - p_tf, "order": jnp.where(jnp.asarray(pair), p_sf - p21, jnp.full(n, jnp.nan))}
        if mix_weights is not None:
            out["mix"] = self._mix_term(params, data, mix_weights)
        return out

    def subject_summary(self, params: Params, data: ModelData) -> pd.DataFrame | None:
        """Q2's summary plus ``log_gamma`` and ``gamma`` (the consistency score). The loss-level
        probabilities are unchanged by dephasing (it leaves the diagonal of ``rho`` alone)."""
        out = super().subject_summary(params, data)
        lg = np.asarray(params["log_gamma"], dtype=np.float64)
        out["log_gamma"] = lg
        out["gamma"] = np.exp(lg)
        return out

    def predict_new_subject(
        self,
        params: Params,
        covariates: dict[str, Any] | str,
        covariate_stats: dict[str, tuple[float, float]],
        loss_pct: float | None,
        context_tags: tuple[str, ...] | list[str] = (),
        question_order_id: str = _design.SCENARIO_FIRST,
        tol_answer: float | None = None,
        gamma: float | None = None,
    ) -> float:
        """As Q2's, through :func:`core.q_forward_rho` at ``gamma`` (default: the fitted
        population median ``exp(gamma_mu)``, the ``u = 0`` analogue for the rate)."""
        if gamma is None:
            gamma = float(np.exp(np.asarray(params["gamma_mu"])))
        return predict_new_subject(
            params, covariates, covariate_stats, self.ctx_vocab, loss_pct, context_tags, question_order_id,
            tol_answer, gamma=gamma,
        )


__all__ = [
    "GAMMA_INIT",
    "GAMMA_INIT_NOISE",
    "GAMMA_SIGMA_INV_GAMMA",
    "LOG_GAMMA_CLASSICAL",
    "Q4",
    "lognormal_population_log_prior",
]
