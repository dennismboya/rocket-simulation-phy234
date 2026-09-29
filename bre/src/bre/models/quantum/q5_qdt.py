"""Q5 — quantum decision theory in the Yukalov–Sornette form (PLAN.md section 4): a utility
factor plus an attraction factor that sums to zero over the two alternatives.

Model
-----
For response row ``i`` of subject ``s(i)``::

    P(sell) = f_sell + q,        P(hold) = f_hold - q,        f_sell + f_hold = 1

* **Utility factor** ``f``: the softmax of the expected utilities of the two prospects under
  the CRRA valuation of B4 (:func:`bre.models.classical.b4_bayes_updater.crra_utility`)::

      EU_sell = u(1 - L)
      EU_hold = p_rec u(1) + (1 - p_rec) u(1 - kappa L) = (1 - p_rec) u(1 - kappa L)
      f_sell  = exp(tau EU_sell) / (exp(tau EU_sell) + exp(tau EU_hold)) = sigmoid(tau (EU_sell - EU_hold))

  with ``rho = exp(log_rho)`` the CRRA coefficient, ``kappa = exp(log_kappa)`` the further-loss
  multiplier, ``tau = exp(log_tau)`` the temperature and ``p_rec = sigmoid(pr_logit)`` a
  population-level recovery belief. The utility factor carries no context or subject term: it
  is the "rational" part of the decision.
* **Attraction factor** ``q``: the subject- and context-dependent deviation from ``f``::

      z_i   = a_{s(i)} + sum_k b_ctx[c_k] + b_order order_flag + b_tol tol_yes
      a_s   = Encoder(x_s) + u_s                       (shared encoder, d_out = 1)
      q_i   = 0.5 tanh(z_i) * 2 min(f_sell, f_hold)   in (-0.5, 0.5)

  ``0.5 tanh(z)`` is the specified form; the factor ``2 min(f_sell, f_hold) = 1 - |2 f_sell - 1|``
  (equal to 1 when ``f_sell = 1/2``, so the two coincide there) is the admissibility bound of
  QDT: ``P = f + q`` must stay in ``[0, 1]``, i.e. ``-f_sell <= q <= f_hold``, and without it
  ``f + 0.5 tanh(z)`` leaves ``[0, 1]`` whenever ``|f_sell - 1/2| > 1/2 - |q|``. The bound keeps
  ``q_sell = -q_hold``, ``|q| < 0.5`` and ``P(sell), P(hold)`` in ``(0, 1)`` for every row.

Contexts enter ``q`` additively and position-independently: ordered pairs compose without pair
parameters and ``P(c1, c2) = P(c2, c1)``.

What the model logs
-------------------
:meth:`Q5.attraction` returns ``q`` per row, :meth:`Q5.mean_abs_q` the mean ``|q|`` over the sell
rows, and :meth:`Q5.quarter_law_check` that mean with a subject-level (cluster) bootstrap 95%
interval next to the reference value 0.25, the empirical regularity reported by Yukalov &
Sornette (mean ``|q|`` close to 1/4 across their reviewed choice experiments). The check reports;
it does not test a hypothesis and the code claims nothing about the law beyond that reference.

Interference terms: Q5 has no non-commuting tolerance measurement (the tolerance answer enters
``q`` as a plain covariate), so ``delta_LTP`` of PLAN.md section 4 — which needs the model's own
``P(a)`` for the tolerance answer — is **not defined** for it: ``ltp`` is returned as NaN on every
row (the evaluation code treats a NaN interval as "does not exclude zero"). ``order`` is 0 on
ordered-pair rows (the additive ``z``) and NaN elsewhere. The QDT deviation from the classical
utility model is ``q`` itself, available through :meth:`attraction`.

Parameters: ``{"enc": {W (d_x, 1), b (1,), u (n_subjects, 1), log_sigma_u (1,)}, "b_ctx": (V,),
"b_order": (), "b_tol": (), "pr_logit": (), "log_rho": (), "log_kappa": (), "log_tau": ()}``;
``n_params = (d_x + 1 + n_subjects + 1) + V + 6``; without ``u``: ``d_x + 2 + V + 6``. Prior: the
encoder's random-effect prior plus an optional ridge ``-(l2 / 2) ||b_ctx||^2`` (default 0).

Initialization (restart folded in; starting points, not fitted values): encoder as
:meth:`Encoder.init`; ``b_ctx, b_order, b_tol, pr_logit ~ N(0, 0.1^2)``; ``log_rho = 0``,
``log_kappa = log 2``, ``log_tau = log 10`` (plus ``N(0, 0.1^2)``), as B4.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bre.models.base import Encoder, Params, QuantumModelBase, count_params
from bre.models.classical.b4_bayes_updater import INIT_LOG_KAPPA, INIT_LOG_TAU, crra_utility
from bre.models.data import ModelData

INIT_NOISE = 0.1
QUARTER_LAW_REFERENCE = 0.25
"""Reference value of the mean ``|q|`` reported by Yukalov & Sornette."""


def attraction_factor(z, f_sell) -> jnp.ndarray:
    """``q = 0.5 tanh(z) * 2 min(f_sell, 1 - f_sell)`` (module docstring)."""
    f = jnp.asarray(f_sell, dtype=jnp.float64)
    return 0.5 * jnp.tanh(jnp.asarray(z, dtype=jnp.float64)) * 2.0 * jnp.minimum(f, 1.0 - f)


class Q5(QuantumModelBase):
    """Quantum decision theory: ``P = f + q`` (module docstring)."""

    name = "Q5"
    family = "quantum"

    def __init__(self, data: ModelData, *, l2: float = 0.0, w_l2: float = 0.0) -> None:
        if l2 < 0 or w_l2 < 0:
            raise ValueError("l2 and w_l2 must be >= 0")
        self.enc = Encoder.for_data(data, d_out=1)
        self.ctx_vocab = tuple(data.ctx_vocab)
        self.n_contexts = int(data.n_contexts)
        self.l2 = float(l2)
        self.w_l2 = float(w_l2)

    # -- model contract ---------------------------------------------------------------------------

    def init_params(self, rng_key, data: ModelData, restart: int = 0) -> Params:
        key = self.restart_key(rng_key, restart)
        keys = jax.random.split(key, 7)
        f64 = jnp.float64
        noise = lambda k, shape=(): INIT_NOISE * jax.random.normal(k, shape, dtype=f64)  # noqa: E731
        return {
            "enc": self.enc.init(rng_key, restart),
            "b_ctx": noise(keys[0], (self.n_contexts,)),
            "b_order": noise(keys[1]),
            "b_tol": noise(keys[2]),
            "pr_logit": noise(keys[3]),
            "log_rho": noise(keys[4]),
            "log_kappa": INIT_LOG_KAPPA + noise(keys[5]),
            "log_tau": INIT_LOG_TAU + noise(keys[6]),
        }

    def utility_factor(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` ``f_sell = sigmoid(tau (EU_sell - EU_hold))``."""
        L = jnp.asarray(data.loss, dtype=jnp.float64)
        rho = jnp.exp(params["log_rho"])
        kappa = jnp.exp(params["log_kappa"])
        p_rec = jax.nn.sigmoid(params["pr_logit"])
        eu_sell = crra_utility(1.0 - L, rho)
        eu_hold = (1.0 - p_rec) * crra_utility(1.0 - kappa * L, rho)
        return jax.nn.sigmoid(jnp.exp(params["log_tau"]) * (eu_sell - eu_hold))

    def attraction_logit(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` ``z_i = a_s + sum_k b_ctx[c_k] + b_order order_flag + b_tol tol_yes``."""
        a = self.enc.apply(params["enc"], data.X, data.subject_idx)[:, 0]
        ctx_idx = jnp.asarray(data.ctx_idx)
        ctx_mask = jnp.asarray(data.ctx_mask)
        b = jnp.asarray(params["b_ctx"], dtype=jnp.float64)
        ctx = jnp.sum(jnp.where(ctx_mask, b[jnp.where(ctx_mask, ctx_idx, 0)], 0.0), axis=1)
        tol = jnp.asarray(data.tol_answer, dtype=jnp.float64)
        tol_yes = jnp.where(jnp.isfinite(tol) & (tol == 1.0), 1.0, 0.0)
        return a + ctx + params["b_order"] * jnp.asarray(data.order_flag, dtype=jnp.float64) + params["b_tol"] * tol_yes

    def attraction(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` the attraction factor ``q_i`` (``q_sell = q``, ``q_hold = -q``)."""
        return attraction_factor(self.attraction_logit(params, data), self.utility_factor(params, data))

    def predict_proba(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` ``P(sell) = f_sell + q``."""
        f = self.utility_factor(params, data)
        return f + attraction_factor(self.attraction_logit(params, data), f)

    def predict_hold(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` ``P(hold) = f_hold - q``."""
        f = self.utility_factor(params, data)
        return (1.0 - f) - attraction_factor(self.attraction_logit(params, data), f)

    def log_prior(self, params: Params) -> jnp.ndarray:
        lp = self.enc.log_prior(params["enc"], w_l2=self.w_l2)
        if self.l2:
            lp = lp - 0.5 * self.l2 * jnp.sum(jnp.asarray(params["b_ctx"]) ** 2)
        return lp

    def n_params(self, params: Params, include_random_effects: bool = True) -> int:
        """``(d_x + 1 + n_subjects + 1) + V + 6``; without ``u``: ``d_x + 2 + V + 6``."""
        enc = self.enc.n_params(params["enc"], include_random_effects=include_random_effects)
        return enc + count_params({k: v for k, v in params.items() if k != "enc"})

    # -- interference terms and the quarter-law check ---------------------------------------------

    def interference_terms(self, params: Params, data: ModelData) -> dict[str, jnp.ndarray]:
        """``ltp``: NaN (not defined for Q5, module docstring); ``order``: 0 on pair rows, NaN
        elsewhere."""
        n = data.n
        pair = jnp.asarray(np.asarray(data.ctx_mask).all(axis=1))
        return {"ltp": jnp.full(n, jnp.nan), "order": jnp.where(pair, 0.0, jnp.nan)}

    def mean_abs_q(self, params: Params, data: ModelData) -> float:
        """Mean ``|q|`` over ``data.sell_rows()``."""
        q = np.asarray(self.attraction(params, data))[data.sell_mask()]
        return float(np.mean(np.abs(q))) if q.size else float("nan")

    def quarter_law_check(self, params: Params, data: ModelData, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
        """Mean ``|q|`` over the sell rows with a subject-level bootstrap 95% interval
        (``n_boot`` resamples of subjects with replacement, ``numpy.random.default_rng(seed)``),
        the reference value :data:`QUARTER_LAW_REFERENCE` and the share of ``q`` that is
        positive. Reports; asserts nothing."""
        sell = data.sell_mask()
        q = np.asarray(self.attraction(params, data))[sell]
        sidx = np.asarray(data.subject_idx)[sell]
        out: dict[str, Any] = {"mean_abs_q": float(np.mean(np.abs(q))) if q.size else float("nan"), "reference": QUARTER_LAW_REFERENCE, "n_rows": int(q.size), "n_boot": int(n_boot), "share_q_positive": float(np.mean(q > 0)) if q.size else float("nan"), "note": "mean |q| ~ 0.25 is the empirical regularity reported by Yukalov & Sornette"}
        subjects = np.unique(sidx)
        if q.size == 0 or subjects.size == 0:
            out["ci95"] = [float("nan"), float("nan")]
            return out
        rng = np.random.default_rng(int(seed))
        groups = [np.abs(q[sidx == s]) for s in subjects]
        sums = np.asarray([g.sum() for g in groups])
        sizes = np.asarray([g.size for g in groups], dtype=np.float64)
        draws = np.empty(int(n_boot))
        for b in range(int(n_boot)):
            pick = rng.integers(0, subjects.size, size=subjects.size)
            draws[b] = sums[pick].sum() / sizes[pick].sum()
        out["ci95"] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        return out

    # -- reporting --------------------------------------------------------------------------------

    def natural_params(self, params: Params) -> dict[str, Any]:
        """``rho``, ``kappa``, ``tau``, ``p_rec``, ``b_order``, ``b_tol`` and ``b_ctx`` by tag."""
        return {
            "rho": float(jnp.exp(params["log_rho"])),
            "kappa": float(jnp.exp(params["log_kappa"])),
            "tau": float(jnp.exp(params["log_tau"])),
            "p_rec": float(jax.nn.sigmoid(params["pr_logit"])),
            "b_order": float(params["b_order"]),
            "b_tol": float(params["b_tol"]),
            "b_ctx": {tag: float(v) for tag, v in zip(self.ctx_vocab, np.asarray(params["b_ctx"]))},
        }

    def subject_summary(self, params: Params, data: ModelData) -> pd.DataFrame | None:
        """One row per subject: the attraction baseline ``a_s`` and its random effect ``u``."""
        a = np.asarray(self.enc.apply_subjects(params["enc"], data.X))[:, 0]
        return pd.DataFrame({"subject_id": data.subject_ids, "attraction_baseline": a, "u": np.asarray(params["enc"]["u"])[:, 0]})


__all__ = ["INIT_NOISE", "Q5", "QUARTER_LAW_REFERENCE", "attraction_factor"]
