"""Q3 — the quantum dynamical model (PLAN.md section 4): Schrödinger evolution of the decision
state under a payoff Hamiltonian and a dissonance Hamiltonian for a time set by the news' age.

Model
-----
For response row ``i`` of subject ``s(i)``:

* initial state ``psi_s(0) = prepare_state(Encoder(x_s) + u_s)`` in C^2 (shared encoder,
  ``d_out = 4``, gauge as in Q2);
* Hamiltonian (Hermitian, real coefficients times the Pauli matrices ``sigma = (sigma_x,
  sigma_y, sigma_z)``)::

      H_i = H_payoff(L_i) + H_dissonance(c_i)
      H_payoff(L)     = L (theta_p . sigma)                      theta_p in R^3
      H_dissonance(c) = (sum_k w[c_k]) (theta_d . sigma)         w in R^V, theta_d a unit vector

  ``theta_d = (sin a cos b, sin a sin b, cos a)`` from two angles (``theta_d_angles``), so that
  the product ``w[c] theta_d`` is identified (a free length of ``theta_d`` would trade off
  against the scale of ``w``); ``w`` of the ``none``-named vocabulary entries is held at 0;
* evolution for the row's time ``t_i``::

      psi_i(t) = expm(-i H_i t) psi_s(0) = U(-t h_i) psi_s(0),   h_i = L_i theta_p + (sum_k w[c_k]) theta_d

  with :func:`bre.models.quantum.core.unitary` (``U(theta) = expm(i theta . sigma)``), unitary
  for every real ``t`` and ``h``;
* ``P(sell) = ||P_sell psi_i(t)||^2`` (scenario-first). Tolerance-first rows: Lüders collapse of
  ``psi_s(0)`` on the observed tolerance answer (projector ``P_yes(phi)``) **before** the
  evolution; with the answer unobserved, the mixture ``sum_a ||P_a psi||^2 P(sell | a)`` (the
  same branch logic as :func:`core.q_forward_pure`).

Time variable
-------------
``t_i = log1p(days_since_news_i)``. ``ModelData`` carries no per-row covariates, so the days are
supplied to the constructor (``Q3(data, days_since_news=...)``, an ``(n,)`` array aligned with
the construction table or a ``{index label: days}`` dict; :func:`days_since_news_from_frame`
reads the covariate key :data:`DAYS_KEY` = ``x_days_since_news`` from a schema frame) and looked
up by ``ModelData.index`` label, so subsets keep their times. Rows without a value use the
design default ``t = T_DEFAULT = 1`` (the shared design does not state a news age; ``log1p(days)
= 1`` is ``days = e - 1 ~ 1.7``). The dashboard's what-if panel passes the days explicitly
through :meth:`Q3.predict_proba_at` / :func:`predict_new_subject`.

Properties (``tests/test_q3_q5.py``): ``U(-t h)`` is unitary; ``t = 0`` gives the static Born
classifier ``||P_sell psi(0)||^2`` (and the collapsed-state probabilities for tolerance-first
rows); ``P(sell) + P(hold) = 1``; recency: for ``psi(0) = |hold>`` and ``h = (h_x, 0, h_z)`` the
sell probability is ``(h_x^2 / |h|^2) sin^2(|h| t)`` — with a dissonance term along ``sigma_z``
opposing the payoff rotation the amplitude drops below 1 and the probability rises and then
falls with ``t`` (non-monotone), which is the recency signature the model is meant to carry.

Interference terms: ``ltp`` as defined in PLAN.md section 4 (scenario-first minus the
tolerance-first Lüders mixture). ``order``: the contexts enter ``H`` through the scalar
``sum_k w[c_k]``, which is symmetric in the pair, so ``Delta_order`` is **identically zero** for
every ordered pair (reported as 0 on pair rows, NaN elsewhere): Q3 predicts no order effect
between contexts, by construction.

Parameters: ``{"enc": {W (d_x, 4), b (4,), u (n_subjects, 4), log_sigma_u (4,)}, "theta_p": (3,),
"theta_d_angles": (2,), "w_ctx": (V,), "phi": ()}``; ``n_params = (4 d_x + 4 + 4 n_subjects + 4)
+ 3 + 2 + V_free + 1`` (``V_free`` the vocabulary entries not named ``none``); without ``u``:
``4 d_x + 8 + 6 + V_free``. Prior: the encoder's random-effect prior plus a weak ``N(0, s^2)``
ridge on ``theta_p`` and ``w_ctx`` (``s = theta_prior_scale`` = 2, as Q2).

Initialization (restart folded in): encoder as Q2; ``theta_p, w_ctx ~ N(0, 0.3^2)``;
``theta_d_angles = (pi/2, 0) + N(0, 0.3^2)`` (``theta_d`` near ``sigma_x``); ``phi ~ U(0.2, 1.2)``.
"""

from __future__ import annotations

import weakref
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bre import design as _design
from bre import schema as S
from bre.models.base import Encoder, Params, QuantumModelBase
from bre.models.data import ModelData
from bre.models.quantum import core as C
from bre.models.quantum.q2_context_unitary import INIT_PHI_RANGE, INIT_THETA_SCALE, THETA_PRIOR_SCALE, context_row, new_subject_state, none_mask

DAYS_KEY = "x_days_since_news"
"""Covariate key (``covariates`` JSON of a schema row) holding the news age in days."""

T_DEFAULT = 1.0
"""Design default of the time variable ``t`` for rows without a news age."""

INIT_D_ANGLES: tuple[float, float] = (float(np.pi / 2), 0.0)


def time_from_days(days) -> jnp.ndarray:
    """``t = log1p(days)`` (``days >= 0``)."""
    return jnp.log1p(jnp.asarray(days, dtype=jnp.float64))


def days_since_news_from_frame(df: pd.DataFrame, key: str = DAYS_KEY) -> dict[Any, float]:
    """``{index label: days}`` for the rows of a schema frame whose ``covariates`` JSON holds
    a finite non-negative ``key``; rows without it are left out (they get ``T_DEFAULT``)."""
    out: dict[Any, float] = {}
    for label, text in zip(df.index, df["covariates"].tolist()):
        rec = text if isinstance(text, dict) else ({} if S.is_null(text) else S.json_loads_strict(text))
        v = rec.get(key)
        if v is not None and not isinstance(v, bool) and isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)) and float(v) >= 0:
            out[label] = float(v)
    return out


def dissonance_direction(angles) -> jnp.ndarray:
    """Unit vector ``(sin a cos b, sin a sin b, cos a)`` from ``angles = (a, b)``."""
    a, b = jnp.asarray(angles, dtype=jnp.float64)
    return jnp.stack([jnp.sin(a) * jnp.cos(b), jnp.sin(a) * jnp.sin(b), jnp.cos(a)])


def hamiltonian_coefficients(theta_p, theta_d, w_ctx, loss, ctx_idx, ctx_mask) -> jnp.ndarray:
    """``(n, 3)`` ``h_i = L_i theta_p + (sum_k w[c_k]) theta_d`` (``H_i = h_i . sigma``)."""
    w = jnp.asarray(w_ctx, dtype=jnp.float64)
    idx = jnp.clip(jnp.asarray(ctx_idx), 0, w.shape[0] - 1)
    wsum = jnp.sum(jnp.where(jnp.asarray(ctx_mask, dtype=bool), w[idx], 0.0), axis=1)
    L = jnp.asarray(loss, dtype=jnp.float64)
    return L[:, None] * jnp.asarray(theta_p, dtype=jnp.float64)[None, :] + wsum[:, None] * jnp.asarray(theta_d, dtype=jnp.float64)[None, :]


def evolution_unitaries(h, t) -> jnp.ndarray:
    """``(n, 2, 2)`` ``expm(-i (h_i . sigma) t_i)`` = ``core.unitary(-t_i h_i)``."""
    t = jnp.broadcast_to(jnp.asarray(t, dtype=jnp.float64), (h.shape[0],))
    return C.unitaries(-t[:, None] * jnp.asarray(h, dtype=jnp.float64))


def forward_q3(psi_rows, U, phi, order_flag, tol_answer, projector=C.P_SELL) -> jnp.ndarray:
    """Row-wise Born probability of ``projector`` after ``U_i psi_i`` with the question-order
    branches of :func:`core.q_forward_pure` (collapse on the tolerance answer before ``U``;
    mixture when the answer is unobserved)."""
    psi = jnp.asarray(psi_rows, dtype=C.CDTYPE)
    p1 = C.tolerance_projector(phi)
    p0 = C.IDENTITY - p1
    born = jax.vmap(C.born, in_axes=(0, None))
    collapse = jax.vmap(C.lueders, in_axes=(0, None))

    def measure(v):
        return born(jnp.einsum("nij,nj->ni", U, v), projector)

    tol = jnp.asarray(tol_answer, dtype=jnp.float64)
    tol_first = jnp.asarray(order_flag) == 1
    p_sf = measure(psi)
    p_g1 = measure(collapse(psi, p1))
    p_g0 = measure(collapse(psi, p0))
    p_obs = jnp.where(tol >= 0.5, p_g1, p_g0)
    p_mix = born(psi, p1) * p_g1 + born(psi, p0) * p_g0
    p_tf = jnp.where(jnp.isnan(tol), p_mix, p_obs)
    return jnp.where(tol_first, p_tf, p_sf)


class Q3(QuantumModelBase):
    """Quantum dynamical model (module docstring)."""

    name = "Q3"
    family = "quantum"

    def __init__(
        self,
        data: ModelData,
        *,
        days_since_news: np.ndarray | dict[Any, float] | None = None,
        t_default: float = T_DEFAULT,
        theta_prior_scale: float = THETA_PRIOR_SCALE,
        w_l2: float = 0.0,
    ) -> None:
        if theta_prior_scale <= 0 or w_l2 < 0 or t_default < 0:
            raise ValueError("theta_prior_scale must be > 0, w_l2 >= 0 and t_default >= 0")
        self.enc = Encoder.for_data(data, d_out=4)
        self.ctx_vocab = tuple(data.ctx_vocab)
        self.n_contexts = len(self.ctx_vocab)
        self.fixed_ctx = none_mask(self.ctx_vocab)
        self._free = jnp.asarray(~self.fixed_ctx, dtype=jnp.float64)
        self.theta_prior_scale = float(theta_prior_scale)
        self.w_l2 = float(w_l2)
        self.t_default = float(t_default)
        if days_since_news is None:
            self.days: dict[Any, float] = {}
        elif isinstance(days_since_news, dict):
            self.days = {k: float(v) for k, v in days_since_news.items()}
        else:
            arr = np.asarray(days_since_news, dtype=np.float64).reshape(-1)
            if arr.shape[0] != data.n:
                raise ValueError("days_since_news must have one entry per row of the construction data (or be a dict by index label)")
            self.days = {lab: float(v) for lab, v in zip(data.index.tolist(), arr) if np.isfinite(v)}
        self._cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    # -- static row arrays ------------------------------------------------------------------------

    def time_of(self, data: ModelData) -> np.ndarray:
        """``(n,)`` ``t_i = log1p(days)`` for rows with a known news age, else ``t_default``."""
        days = np.asarray([self.days.get(lab, np.nan) for lab in data.index.tolist()], dtype=np.float64)
        return np.where(np.isfinite(days), np.log1p(np.where(np.isfinite(days), days, 0.0)), self.t_default)

    def _rows(self, data: ModelData) -> dict[str, np.ndarray]:
        if tuple(data.ctx_vocab) != self.ctx_vocab:
            raise ValueError("ModelData.ctx_vocab differs from the vocabulary this model was built for")
        r = self._cache.get(data)
        if r is None:
            r = {
                "subject_idx": np.asarray(data.subject_idx, dtype=np.int64),
                "loss": np.asarray(data.loss, dtype=np.float64),
                "ctx_idx": np.asarray(data.ctx_idx, dtype=np.int64),
                "ctx_mask": np.asarray(data.ctx_mask, dtype=bool),
                "order_flag": np.asarray(data.order_flag, dtype=np.int64),
                "tol_answer": np.asarray(data.tol_answer, dtype=np.float64),
                "t": self.time_of(data),
                "X": np.asarray(data.X, dtype=np.float64),
            }
            self._cache[data] = r
        return r

    # -- model contract ---------------------------------------------------------------------------

    def init_params(self, rng_key, data: ModelData, restart: int = 0) -> Params:
        key = self.restart_key(rng_key, restart)
        k_p, k_d, k_w, k_phi = jax.random.split(key, 4)
        f64 = jnp.float64
        lo, hi = INIT_PHI_RANGE
        return {
            "enc": self.enc.init(rng_key, restart),
            "theta_p": INIT_THETA_SCALE * jax.random.normal(k_p, (3,), dtype=f64),
            "theta_d_angles": jnp.asarray(INIT_D_ANGLES, dtype=f64) + INIT_THETA_SCALE * jax.random.normal(k_d, (2,), dtype=f64),
            "w_ctx": INIT_THETA_SCALE * jax.random.normal(k_w, (self.n_contexts,), dtype=f64) * self._free,
            "phi": jax.random.uniform(k_phi, (), dtype=f64, minval=lo, maxval=hi),
        }

    def states(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n_subjects, 2)`` initial states ``psi_s(0)``."""
        return jax.vmap(C.prepare_state)(self.enc.apply_subjects(params["enc"], self._rows(data)["X"]))

    def context_weights(self, params: Params) -> jnp.ndarray:
        """``(V,)`` dissonance weights with the ``none`` entries forced to zero."""
        return jnp.asarray(params["w_ctx"], dtype=jnp.float64) * self._free

    def hamiltonians(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n, 3)`` coefficient vectors ``h_i`` (``H_i = h_i . sigma``)."""
        r = self._rows(data)
        return hamiltonian_coefficients(params["theta_p"], dissonance_direction(params["theta_d_angles"]), self.context_weights(params), r["loss"], r["ctx_idx"], r["ctx_mask"])

    def _predict(self, params: Params, data: ModelData, *, t=None, order_flag=None, tol_answer=None, projector=C.P_SELL) -> jnp.ndarray:
        r = self._rows(data)
        psi = self.states(params, data)[r["subject_idx"]]
        U = evolution_unitaries(self.hamiltonians(params, data), r["t"] if t is None else t)
        return forward_q3(psi, U, params["phi"], r["order_flag"] if order_flag is None else order_flag, r["tol_answer"] if tol_answer is None else tol_answer, projector)

    def predict_proba(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` ``P(sell)`` at each row's time ``t_i`` (module docstring)."""
        return self._predict(params, data)

    def predict_proba_at(self, params: Params, data: ModelData, days_since_news=None, *, t=None) -> jnp.ndarray:
        """``(n,)`` ``P(sell)`` at an explicit news age (``days_since_news``, scalar or ``(n,)``,
        ``t = log1p(days)``) or an explicit ``t`` (scalar or ``(n,)``); the what-if entry point."""
        if (days_since_news is None) == (t is None):
            raise ValueError("pass exactly one of days_since_news and t")
        tt = time_from_days(days_since_news) if t is None else jnp.asarray(t, dtype=jnp.float64)
        return self._predict(params, data, t=tt)

    def predict_hold(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` ``P(hold) = ||P_hold psi(t)||^2`` (equals ``1 - P(sell)``)."""
        return self._predict(params, data, projector=C.P_HOLD)

    def log_prior(self, params: Params) -> jnp.ndarray:
        lp = self.enc.log_prior(params["enc"], w_l2=self.w_l2)
        ridge = jnp.sum(jnp.asarray(params["theta_p"]) ** 2) + jnp.sum(self.context_weights(params) ** 2)
        return lp - 0.5 * ridge / self.theta_prior_scale**2

    def n_params(self, params: Params, include_random_effects: bool = True) -> int:
        """Encoder scalars + 3 (``theta_p``) + 2 (``theta_d_angles``) + ``V_free`` + 1 (``phi``)."""
        enc = self.enc.n_params(params["enc"], include_random_effects=include_random_effects)
        return enc + 3 + 2 + int((~self.fixed_ctx).sum()) + 1

    # -- interference terms -------------------------------------------------------------------------

    def interference_terms(self, params: Params, data: ModelData) -> dict[str, jnp.ndarray]:
        """``ltp`` per row; ``order`` is 0 on ordered-pair rows (contexts enter ``H`` through a
        symmetric sum) and NaN elsewhere."""
        r = self._rows(data)
        n = data.n
        nans = np.full(n, np.nan)
        p_sf = self._predict(params, data, order_flag=np.zeros(n, dtype=np.int64), tol_answer=nans)
        p_tf = self._predict(params, data, order_flag=np.ones(n, dtype=np.int64), tol_answer=nans)
        pair = jnp.asarray(r["ctx_mask"].all(axis=1))
        return {"ltp": p_sf - p_tf, "order": jnp.where(pair, 0.0, jnp.nan)}

    # -- reporting --------------------------------------------------------------------------------

    def natural_params(self, params: Params) -> dict[str, Any]:
        """``theta_p``, the unit ``theta_d``, ``phi`` and the dissonance weights by context tag."""
        return {
            "theta_p": [float(v) for v in np.asarray(params["theta_p"])],
            "theta_d": [float(v) for v in np.asarray(dissonance_direction(params["theta_d_angles"]))],
            "phi": float(params["phi"]),
            "w_ctx": {tag: float(v) for tag, v in zip(self.ctx_vocab, np.asarray(self.context_weights(params)))},
            "t_default": self.t_default,
        }

    def subject_summary(self, params: Params, data: ModelData) -> pd.DataFrame | None:
        """One row per subject: Bloch angles of ``psi_s(0)`` and ``p_sell_base = ||P_sell psi(0)||^2``."""
        psi = np.asarray(self.states(params, data))
        return pd.DataFrame(
            {
                "subject_id": np.asarray(data.subject_ids),
                "bloch_theta": 2.0 * np.arctan2(np.abs(psi[:, 1]), np.abs(psi[:, 0])),
                "bloch_phi": np.angle(psi[:, 1]),
                "p_sell_base": np.abs(psi[:, 1]) ** 2,
            }
        )

    def predict_new_subject(
        self,
        params: Params,
        covariates: dict[str, Any] | str,
        covariate_stats: dict[str, tuple[float, float]],
        loss_pct: float | None,
        context_tags: tuple[str, ...] | list[str] = (),
        question_order_id: str = _design.SCENARIO_FIRST,
        tol_answer: float | None = None,
        days_since_news: float | None = None,
    ) -> float:
        """:func:`predict_new_subject` with this model's vocabulary (``u = 0``; ``days_since_news
        = None`` uses ``t_default``)."""
        return predict_new_subject(params, covariates, covariate_stats, self.ctx_vocab, loss_pct, context_tags, question_order_id, tol_answer, days_since_news, t_default=self.t_default)


def predict_new_subject(
    params: Params,
    covariates: dict[str, Any] | str,
    covariate_stats: dict[str, tuple[float, float]],
    ctx_vocab: tuple[str, ...],
    loss_pct: float | None,
    context_tags: tuple[str, ...] | list[str] = (),
    question_order_id: str = _design.SCENARIO_FIRST,
    tol_answer: float | None = None,
    days_since_news: float | None = None,
    *,
    t_default: float = T_DEFAULT,
) -> float:
    """``P(sell)`` for a new client (API / what-if path): ``psi(0)`` from the covariates with
    ``u = 0`` (:func:`bre.models.quantum.q2_context_unitary.new_subject_state`), the row's
    Hamiltonian from ``L = |loss_pct|`` and the context tags, ``t = log1p(days_since_news)``
    (``t_default`` when None), the question order and prior tolerance answer as in Q2."""
    if question_order_id not in _design.QUESTION_ORDERS:
        raise ValueError(f"unknown question order {question_order_id!r}; allowed {_design.QUESTION_ORDER_IDS}")
    psi = new_subject_state(params, covariates, covariate_stats)
    ctx_idx, ctx_mask = context_row(ctx_vocab, context_tags)
    free = jnp.asarray(~none_mask(tuple(ctx_vocab)), dtype=jnp.float64)
    w = jnp.asarray(params["w_ctx"], dtype=jnp.float64) * free
    L = 0.0 if loss_pct is None else abs(float(loss_pct))
    h = hamiltonian_coefficients(params["theta_p"], dissonance_direction(params["theta_d_angles"]), w, jnp.asarray([L]), ctx_idx[None, :], ctx_mask[None, :])
    t = float(t_default) if days_since_news is None else float(np.log1p(float(days_since_news)))
    U = evolution_unitaries(h, jnp.asarray([t]))
    order_flag = 1 if question_order_id == _design.TOLERANCE_FIRST else 0
    tol = np.nan if (tol_answer is None or order_flag == 0) else float(tol_answer)
    p = forward_q3(psi[None, :], U, params["phi"], jnp.asarray([order_flag]), jnp.asarray([tol]))
    return float(p[0])


__all__ = [
    "DAYS_KEY",
    "INIT_D_ANGLES",
    "Q3",
    "T_DEFAULT",
    "days_since_news_from_frame",
    "dissonance_direction",
    "evolution_unitaries",
    "forward_q3",
    "hamiltonian_coefficients",
    "predict_new_subject",
    "time_from_days",
]
