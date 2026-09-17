"""Q2 — the context-unitary quantum-probability model (PLAN.md section 4).

Model
-----
For response row ``i`` of subject ``s(i)`` with standardized covariates ``x`` (``ModelData.X``):

* state ``psi_s = prepare_state(Encoder(x_s) + u_s)`` in C^2 (shared encoder, ``d_out = 4``; the
  four real numbers become a unit vector with a real, non-negative hold amplitude, so two of them
  are identifiable — INTERFACE.md section 7);
* loss scenario ``U_L(L) = exp(L theta_L . G)`` with ``G_k = i sigma_k`` and ``L = |loss_pct|``;
* each context ``c`` in presentation order applies ``U_c = exp(theta_c . G)``; ``theta_none = 0``
  is the reference gauge (the ``none`` condition has no unitary, and any vocabulary entry named
  ``none`` or ``<namespace>:none`` is held at zero, :func:`none_mask`);
* the tolerance question is the projector ``P_yes = |yes><yes|``, ``|yes> = (cos phi, sin phi)``.

Scenario-first rows: ``P(sell) = ||P_sell U_{c_m} ... U_{c_1} U_L(L) psi||^2``. Tolerance-first
rows with the observed prior answer ``a``: the same chain applied to the Lüders-collapsed
``psi_a = P_a psi / ||P_a psi||``; with the answer unobserved, the mixture
``sum_a ||P_a psi||^2 P(sell | a)``. This is exactly :func:`bre.models.quantum.core.q_forward_pure`.

Parameters (``init_params``): ``{"enc": {W, b, u, log_sigma_u}, "theta_L": (3,),
"theta_ctx": (V, 3), "phi": ()}`` with ``V = len(ctx_vocab)``. ``n_params`` counts the encoder
(``d_x * 4 + 4 + n_subjects * 4 + 4``; without random effects the ``u`` block is left out), the
loss vector, ``3`` per free context row and ``phi``. Prior: the encoder's random-effect prior
(:func:`bre.models.base.encoder_log_prior`) plus a weak Gaussian ridge on every context and loss
parameter, ``-||theta||^2 / (2 s^2)`` with ``s = theta_prior_scale`` (default
:data:`THETA_PRIOR_SCALE` = 2, i.e. a ``N(0, 4)`` prior per component: it only keeps the
optimizer away from the ``|theta| ~ pi`` aliases, INTERFACE.md section 7).

Two forward implementations
---------------------------
``forward="fast"`` (default) computes the ``V`` context unitaries and one loss unitary per
distinct loss level once per evaluation (:func:`core.unitaries`) and then applies 2x2 matrix
products per row; the measurement steps use :func:`core.born`, :func:`core.lueders` and
:func:`core.tolerance_projector`. ``forward="core"`` calls :func:`core.q_forward_pure_batch`
row by row (one matrix exponential per unitary per row). The two agree to machine precision
(``tests/test_q2_q4.py``); the fast path exists because the per-row exponentials cost about
4-12 s per gradient step on the N = 200 design (34,000 sell rows), which would put the Phase 2
recovery fits out of reach, while the fast path costs tens of milliseconds.

Interference terms (PLAN.md section 4, :meth:`Q2.interference_terms`)
--------------------------------------------------------------------
* ``ltp``: ``delta_LTP = P(sell | scenario-first) - sum_a P(a, sell | tolerance-first)`` per row
  with the row's loss and contexts;
* ``order``: ``Delta_order(c1, c2) = P(sell | c1, c2) - P(sell | c2, c1)`` (scenario-first, the
  row's loss) for ordered-pair rows, NaN otherwise;
* ``mix`` (only when ``mix_weights`` is given): the context-uncertainty term
  ``delta_mix = ||P_sell normalize(sum_c sqrt(p_c) U_c psi_L)||^2 - sum_c p_c ||P_sell U_c psi_L||^2``
  evaluated on the post-loss state ``psi_L = U_L(L) psi`` of each row (the state entering the
  context step; with ``L = 0`` this is the PLAN.md formula on ``psi``).

Gauge symmetries (identifiability)
----------------------------------
Besides the state gauge fixed by ``prepare_state`` and ``theta_none = 0``, the sell likelihood is
invariant under two discrete transformations that a fit can land on (write
``psi = (cos a, e^{ib} sin a)``):

* complex conjugation ``K``: ``b -> -b``, ``theta -> (-theta_x, theta_y, -theta_z)`` for every
  unitary (``theta_L`` and every context), ``phi`` unchanged;
* conjugation by ``sigma_z`` (``Z``): ``b -> b + pi``, ``theta -> (-theta_x, -theta_y, theta_z)``,
  ``phi -> -phi``;
* their product ``KZ``: ``b -> pi - b``, ``theta -> (theta_x, -theta_y, -theta_z)``, ``phi -> -phi``.

On the encoder output ``v = (re_hold, re_sell, im_hold, im_sell)`` ``K`` negates the imaginary
components and ``Z`` negates the sell components. In addition ``U(theta + pi theta / |theta|) =
-U(theta)`` leaves every probability unchanged, so ``theta`` is identified modulo ``pi`` along
its direction (:func:`fold_theta` maps to ``|theta| <= pi / 2``). :func:`apply_gauge` and
:func:`align_gauge` implement these maps so that a fit can be compared with a known truth (the
Phase 2 recovery study); ``tests/test_q2_q4.py`` verifies that each gauge element leaves
``predict_proba`` unchanged. A ``sigma_z`` component of the *last* unitary before the sell
measurement is invisible in ``P(sell)`` (it only multiplies the sell amplitude by a phase); it
is identified through the ordered pairs in which the context comes first.

API helper
----------
:func:`predict_new_subject` scores a new client from a covariate record (standardized with the
fitting sample's ``covariate_stats``, random effect ``u = 0``) for one scenario through the exact
core forward pass; ``Q2.predict_new_subject`` binds the model's vocabulary.
"""

from __future__ import annotations

import weakref
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bre import design as _design
from bre.models.base import Encoder, Params, QuantumModelBase
from bre.models.data import ModelData, covariate_matrix
from bre.models.quantum import core as C

THETA_PRIOR_SCALE = 2.0
"""Scale ``s`` of the weak ``N(0, s^2)`` prior on every component of ``theta_L`` and
``theta_ctx`` (documented in the module docstring)."""

INIT_THETA_SCALE = 0.3
"""Standard deviation of the initial ``theta`` draws (keeps ``|theta| < pi``, INTERFACE.md section 7)."""

INIT_PHI_RANGE: tuple[float, float] = (0.2, 1.2)
"""Initial ``phi`` is drawn uniformly from this interval (strictly inside ``(0, pi / 2)``, so the
tolerance projector never starts commuting with the sell projector)."""

NONE_TAG = "none"
"""A vocabulary entry equal to this, or ending in ``:none``, is held at ``theta = 0``."""

FORWARD_MODES: tuple[str, ...] = ("fast", "core")

GAUGE_ELEMENTS: dict[str, dict[str, Any]] = {
    "identity": {"theta": (1.0, 1.0, 1.0), "v": (1.0, 1.0, 1.0, 1.0), "phi": 1.0},
    "conjugate": {"theta": (-1.0, 1.0, -1.0), "v": (1.0, 1.0, -1.0, -1.0), "phi": 1.0},
    "z_flip": {"theta": (-1.0, -1.0, 1.0), "v": (1.0, -1.0, 1.0, -1.0), "phi": -1.0},
    "conjugate_z_flip": {"theta": (1.0, -1.0, -1.0), "v": (1.0, -1.0, -1.0, 1.0), "phi": -1.0},
}
"""The four sign patterns of the discrete gauge group (module docstring): signs applied to every
``theta`` vector, to the encoder output dimensions ``(re_hold, re_sell, im_hold, im_sell)``
(hence to the columns of ``W``, ``b`` and ``u``) and to ``phi``. On ``v``: conjugation ``K``
negates the imaginary parts, ``Z`` (``psi -> sigma_z psi``) negates the sell components, and
``K Z`` (``psi -> conj(sigma_z psi)``) negates the sell real part and the hold imaginary part.
For a gauge-fixed state ``(cos a, e^{ib} sin a)`` these give ``b -> -b``, ``b -> b + pi`` and
``b -> pi - b``; ``prepare_state`` re-fixes the global phase afterwards."""


def none_mask(ctx_vocab: tuple[str, ...]) -> np.ndarray:
    """``(V,)`` bool: True for vocabulary entries held at ``theta = 0`` (the reference gauge):
    the tag ``none`` itself or any ``<namespace>:none``."""
    return np.asarray([t == NONE_TAG or t.endswith(":" + NONE_TAG) for t in ctx_vocab], dtype=bool)


# ---------------------------------------------------------------------------------------------
# Forward passes on whole tables
# ---------------------------------------------------------------------------------------------


def _loss_levels(loss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct loss magnitudes and the index of each row's level."""
    levels, idx = np.unique(np.asarray(loss, dtype=np.float64), return_inverse=True)
    return levels, idx.astype(np.int64).reshape(-1)


def _chain_pure(v, U_L, U_ctx, loss_idx, ctx_idx, ctx_mask):
    """``U_{c_K} ... U_{c_1} U_L v`` per row with precomputed unitaries; ``v (n, 2)``."""
    v = jnp.einsum("nij,nj->ni", U_L[loss_idx], v)
    V = U_ctx.shape[0]
    for k in range(ctx_idx.shape[1]):
        u = U_ctx[jnp.clip(ctx_idx[:, k], 0, V - 1)]
        v = jnp.where(ctx_mask[:, k][:, None], jnp.einsum("nij,nj->ni", u, v), v)
    return v


def _conj_rho(u, r):
    """``u r u^dagger`` per row."""
    return jnp.einsum("nij,njk,nlk->nil", u, r, jnp.conj(u))


def _chain_rho(r, gamma, U_L, U_ctx, loss_idx, ctx_idx, ctx_mask):
    """Open-system chain per row: ``dephase(U_L r U_L^+)`` then, per unmasked context,
    ``dephase(U_c r U_c^+)`` at the row's rate (:func:`core.chain_rho` with precomputed unitaries)."""
    r = jax.vmap(C.dephase)(_conj_rho(U_L[loss_idx], r), gamma)
    V = U_ctx.shape[0]
    for k in range(ctx_idx.shape[1]):
        u = U_ctx[jnp.clip(ctx_idx[:, k], 0, V - 1)]
        new = jax.vmap(C.dephase)(_conj_rho(u, r), gamma)
        r = jnp.where(ctx_mask[:, k][:, None, None], new, r)
    return r


def needs_mixture(order_flag, tol_answer) -> bool:
    """True when some tolerance-first row has no observed tolerance answer, i.e. the Lüders
    mixture branch of the forward pass is needed (static decision, taken on numpy arrays)."""
    return bool(np.any((np.asarray(order_flag) == 1) & np.isnan(np.asarray(tol_answer, dtype=np.float64))))


def _select_branches(measure, collapse, born, state, p1, p0, order_flag, tol_answer, mixture):
    """Shared branch logic of the two fast forward passes (state = ``psi`` rows or ``rho`` rows).

    ``mixture=True`` evaluates the three chains (scenario-first, collapsed on "yes", collapsed on
    "no") and selects per row exactly as :func:`core.q_forward_pure`. ``mixture=False`` (no row
    needs the Lüders mixture) runs one chain per row on the selected initial state: the
    scenario-first state, or the state collapsed on the observed answer; identical values at a
    third of the cost, which is what the optimizer uses.
    """
    tol_answer = jnp.asarray(tol_answer, dtype=C.RDTYPE)
    tol_first = jnp.asarray(order_flag) == 1
    yes = tol_answer >= 0.5
    if not mixture:
        init = jnp.where(tol_first[(...,) + (None,) * (state.ndim - 1)], jnp.where(yes[(...,) + (None,) * (state.ndim - 1)], collapse(state, p1), collapse(state, p0)), state)
        return measure(init)
    p_sf = measure(state)
    p_g1 = measure(collapse(state, p1))
    p_g0 = measure(collapse(state, p0))
    p_obs = jnp.where(yes, p_g1, p_g0)
    p_mix = born(state, p1) * p_g1 + born(state, p0) * p_g0
    p_tf = jnp.where(jnp.isnan(tol_answer), p_mix, p_obs)
    return jnp.where(tol_first, p_tf, p_sf)


@partial(jax.jit, static_argnames=("mixture",))
def forward_pure_fast(psi_subj, theta_L, theta_ctx, phi, loss_levels, loss_idx, subject_idx, ctx_idx, ctx_mask, order_flag, tol_answer, *, mixture=True):
    """Row-wise ``P(sell)`` of Q2 with the unitaries computed once per parameter table.

    ``psi_subj (S, 2)`` per subject; ``theta_ctx (V, 3)`` (rows held at zero already masked);
    ``loss_levels (U,)`` distinct loss magnitudes and ``loss_idx (n,)`` each row's level; the
    other arguments are the ``ModelData`` row arrays. Same semantics as
    :func:`core.q_forward_pure_batch` (asserted by the tests). ``mixture`` (static): whether the
    Lüders-mixture branch is needed (:func:`needs_mixture`); False is cheaper and exact when
    every tolerance-first row has an observed answer.
    """
    theta_L = jnp.asarray(theta_L, dtype=C.RDTYPE)
    U_ctx = C.unitaries(jnp.asarray(theta_ctx, dtype=C.RDTYPE))
    U_L = C.unitaries(jnp.asarray(loss_levels, dtype=C.RDTYPE)[:, None] * theta_L[None, :])
    psi = jnp.asarray(psi_subj, dtype=C.CDTYPE)[subject_idx]
    p1 = C.tolerance_projector(phi)
    p0 = C.IDENTITY - p1

    def measure(v):
        out = _chain_pure(v, U_L, U_ctx, loss_idx, ctx_idx, ctx_mask)
        return jax.vmap(C.born, in_axes=(0, None))(out, C.P_SELL)

    collapse = jax.vmap(C.lueders, in_axes=(0, None))
    born = jax.vmap(C.born, in_axes=(0, None))
    return _select_branches(measure, collapse, born, psi, p1, p0, order_flag, tol_answer, mixture)


@partial(jax.jit, static_argnames=("mixture",))
def forward_rho_fast(rho_subj, gamma_subj, theta_L, theta_ctx, phi, loss_levels, loss_idx, subject_idx, ctx_idx, ctx_mask, order_flag, tol_answer, *, mixture=True):
    """Row-wise ``P(sell)`` of Q4 (density matrices, dephasing at the subject's rate) with the
    unitaries computed once; same semantics as :func:`core.q_forward_rho_batch` (``mixture`` as
    in :func:`forward_pure_fast`)."""
    theta_L = jnp.asarray(theta_L, dtype=C.RDTYPE)
    U_ctx = C.unitaries(jnp.asarray(theta_ctx, dtype=C.RDTYPE))
    U_L = C.unitaries(jnp.asarray(loss_levels, dtype=C.RDTYPE)[:, None] * theta_L[None, :])
    rho = jnp.asarray(rho_subj, dtype=C.CDTYPE)[subject_idx]
    gamma = jnp.asarray(gamma_subj, dtype=C.RDTYPE)[subject_idx]
    p1 = C.tolerance_projector(phi)
    p0 = C.IDENTITY - p1

    def measure(r):
        out = _chain_rho(r, gamma, U_L, U_ctx, loss_idx, ctx_idx, ctx_mask)
        return jax.vmap(C.born_rho, in_axes=(0, None))(out, C.P_SELL)

    collapse = jax.vmap(C.lueders_rho, in_axes=(0, None))
    born = jax.vmap(C.born_rho, in_axes=(0, None))
    return _select_branches(measure, collapse, born, rho, p1, p0, order_flag, tol_answer, mixture)


# ---------------------------------------------------------------------------------------------
# Gauge helpers
# ---------------------------------------------------------------------------------------------


def fold_theta(theta) -> np.ndarray:
    """Map every ``theta`` vector (last axis 3) to its representative with ``|theta| <= pi / 2``:
    ``theta - pi round(|theta| / pi) theta / |theta|`` (``U(theta + pi theta/|theta|) = -U(theta)``,
    so all representatives give the same probabilities)."""
    theta = np.asarray(theta, dtype=np.float64)
    r = np.linalg.norm(theta, axis=-1, keepdims=True)
    safe = np.where(r > 0, r, 1.0)
    return theta - np.pi * np.round(r / np.pi) * theta / safe


def apply_gauge(params: Params, element: str) -> Params:
    """Return the parameters transformed by one :data:`GAUGE_ELEMENTS` entry (``predict_proba``
    is unchanged): signs on ``theta_L``, ``theta_ctx``, ``phi`` and on the encoder output
    dimensions (columns of ``W``, entries of ``b``, columns of ``u``)."""
    g = GAUGE_ELEMENTS[element]
    st = jnp.asarray(g["theta"], dtype=jnp.float64)
    sv = jnp.asarray(g["v"], dtype=jnp.float64)
    enc = dict(params["enc"])
    enc["W"] = jnp.asarray(enc["W"]) * sv[None, :]
    enc["b"] = jnp.asarray(enc["b"]) * sv
    enc["u"] = jnp.asarray(enc["u"]) * sv[None, :]
    out = dict(params)
    out["enc"] = enc
    out["theta_L"] = jnp.asarray(params["theta_L"]) * st
    out["theta_ctx"] = jnp.asarray(params["theta_ctx"]) * st[None, :]
    out["phi"] = jnp.asarray(params["phi"]) * g["phi"]
    return out


def align_gauge(params: Params, reference_theta_ctx, rows: np.ndarray | None = None) -> tuple[Params, str, float]:
    """Pick the gauge element whose folded ``theta_ctx`` (optionally restricted to ``rows``)
    correlates best with ``reference_theta_ctx`` (same shape, e.g. a generator's population
    means). Returns ``(params, element, correlation)``; the ``theta`` tables of the returned
    params are folded with :func:`fold_theta`."""
    ref = np.asarray(reference_theta_ctx, dtype=np.float64)
    best: tuple[Params, str, float] | None = None
    for name in GAUGE_ELEMENTS:
        cand = apply_gauge(params, name)
        th = fold_theta(np.asarray(cand["theta_ctx"]))
        sel = th if rows is None else th[rows]
        r = ref if rows is None else ref[rows]
        corr = float(np.corrcoef(sel.reshape(-1), r.reshape(-1))[0, 1])
        if best is None or corr > best[2]:
            cand = dict(cand)
            cand["theta_ctx"] = jnp.asarray(th)
            cand["theta_L"] = jnp.asarray(fold_theta(np.asarray(cand["theta_L"])))
            best = (cand, name, corr)
    assert best is not None
    return best


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------


class Q2(QuantumModelBase):
    """Context-unitary model (module docstring)."""

    name = "Q2"
    family = "quantum"

    def __init__(
        self,
        data: ModelData,
        *,
        theta_prior_scale: float = THETA_PRIOR_SCALE,
        w_l2: float = 0.0,
        forward: str = "fast",
    ) -> None:
        if forward not in FORWARD_MODES:
            raise ValueError(f"forward must be one of {FORWARD_MODES}; got {forward!r}")
        if theta_prior_scale <= 0 or w_l2 < 0:
            raise ValueError("theta_prior_scale must be > 0 and w_l2 >= 0")
        self.enc = Encoder.for_data(data, d_out=4)
        self.ctx_vocab = tuple(data.ctx_vocab)
        self.n_contexts = len(self.ctx_vocab)
        self.fixed_ctx = none_mask(self.ctx_vocab)
        self._free_rows = jnp.asarray(~self.fixed_ctx, dtype=jnp.float64)[:, None]
        self.theta_prior_scale = float(theta_prior_scale)
        self.w_l2 = float(w_l2)
        self.forward = forward
        self.loss_levels_design: tuple[float, ...] = tuple(abs(float(x)) for x in _design.LOSS_PCTS)
        self._cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    # -- static row arrays --------------------------------------------------------------------------

    def _rows(self, data: ModelData) -> dict[str, np.ndarray]:
        """Per-table static arrays (cached per ``ModelData`` object): distinct loss levels and
        each row's level, the context indices clipped for gathering, and the row flags."""
        if data.n_contexts != self.n_contexts or tuple(data.ctx_vocab) != self.ctx_vocab:
            raise ValueError("ModelData.ctx_vocab differs from the vocabulary this model was built for")
        cached = self._cache.get(data)
        if cached is None:
            levels, idx = _loss_levels(data.loss)
            cached = {
                "loss_levels": levels,
                "loss_idx": idx,
                "subject_idx": np.asarray(data.subject_idx, dtype=np.int64),
                "ctx_idx": np.asarray(data.ctx_idx, dtype=np.int64),
                "ctx_mask": np.asarray(data.ctx_mask, dtype=bool),
                "order_flag": np.asarray(data.order_flag, dtype=np.int64),
                "tol_answer": np.asarray(data.tol_answer, dtype=np.float64),
                "loss": np.asarray(data.loss, dtype=np.float64),
                "X": np.asarray(data.X, dtype=np.float64),
            }
            self._cache[data] = cached
        return cached

    # -- model contract ---------------------------------------------------------------------------

    def init_params(self, rng_key, data: ModelData, restart: int = 0) -> Params:
        """Encoder init of ``restart`` (``u = 0``), ``theta_L, theta_ctx ~ N(0, 0.3^2)`` (rows of
        fixed contexts at 0), ``phi ~ U(0.2, 1.2)``; deterministic in ``(rng_key, restart)``."""
        key = self.restart_key(rng_key, restart)
        k_l, k_c, k_p = jax.random.split(key, 3)
        theta_ctx = INIT_THETA_SCALE * jax.random.normal(k_c, (self.n_contexts, 3), dtype=jnp.float64)
        lo, hi = INIT_PHI_RANGE
        return {
            "enc": self.enc.init(rng_key, restart),
            "theta_L": INIT_THETA_SCALE * jax.random.normal(k_l, (3,), dtype=jnp.float64),
            "theta_ctx": theta_ctx * self._free_rows,
            "phi": jax.random.uniform(k_p, (), dtype=jnp.float64, minval=lo, maxval=hi),
        }

    def context_table(self, params: Params) -> jnp.ndarray:
        """``(V, 3)`` context parameters with the fixed (``none``) rows forced to zero."""
        return jnp.asarray(params["theta_ctx"], dtype=jnp.float64) * self._free_rows

    def states(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n_subjects, 2)`` gauge-fixed states ``psi_s = prepare_state(Encoder(x_s) + u_s)``."""
        z = self.enc.apply_subjects(params["enc"], self._rows(data)["X"])
        return jax.vmap(C.prepare_state)(z)

    def _predict_pure(self, params: Params, data: ModelData, *, order_flag=None, tol_answer=None, ctx_idx=None, ctx_mask=None) -> jnp.ndarray:
        r = self._rows(data)
        psi = self.states(params, data)
        theta_ctx = self.context_table(params)
        order_flag = r["order_flag"] if order_flag is None else order_flag
        tol_answer = r["tol_answer"] if tol_answer is None else tol_answer
        ctx_idx = r["ctx_idx"] if ctx_idx is None else ctx_idx
        ctx_mask = r["ctx_mask"] if ctx_mask is None else ctx_mask
        if self.forward == "core":
            ctx = C.gather_context_thetas(theta_ctx, ctx_idx, ctx_mask)
            return C.q_forward_pure_batch(
                psi[r["subject_idx"]], params["theta_L"], r["loss"], ctx, ctx_mask, order_flag, params["phi"], tol_answer
            )
        return forward_pure_fast(
            psi, params["theta_L"], theta_ctx, params["phi"], r["loss_levels"], r["loss_idx"], r["subject_idx"],
            ctx_idx, ctx_mask, order_flag, tol_answer, mixture=needs_mixture(order_flag, tol_answer),
        )

    def predict_proba(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` ``P(sell)`` for every row (module docstring)."""
        return self._predict_pure(params, data)

    def log_prior(self, params: Params) -> jnp.ndarray:
        """Encoder prior plus the weak ridge ``-(||theta_L||^2 + ||theta_ctx||^2) / (2 s^2)``."""
        lp = self.enc.log_prior(params["enc"], w_l2=self.w_l2)
        s2 = self.theta_prior_scale**2
        ridge = jnp.sum(jnp.asarray(params["theta_L"]) ** 2) + jnp.sum(self.context_table(params) ** 2)
        return lp - 0.5 * ridge / s2

    def n_params(self, params: Params, include_random_effects: bool = True) -> int:
        """Encoder scalars (``u`` left out when ``include_random_effects`` is False) + 3
        (``theta_L``) + 3 per free context row + 1 (``phi``)."""
        enc = self.enc.n_params(params["enc"], include_random_effects=include_random_effects)
        free_rows = int((~self.fixed_ctx).sum())
        return enc + 3 + 3 * free_rows + 1

    # -- interference terms -------------------------------------------------------------------------

    def _mix_weights(self, mix_weights) -> jnp.ndarray:
        if isinstance(mix_weights, dict):
            unknown = sorted(set(mix_weights) - set(self.ctx_vocab))
            if unknown:
                raise KeyError(f"mix_weights name unknown context(s) {unknown}")
            w = np.asarray([float(mix_weights.get(t, 0.0)) for t in self.ctx_vocab])
        else:
            w = np.asarray(mix_weights, dtype=np.float64).reshape(-1)
            if w.shape != (self.n_contexts,):
                raise ValueError(f"mix_weights must have one entry per context ({self.n_contexts})")
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("mix_weights must be non-negative with a positive sum")
        return jnp.asarray(w / w.sum())

    def _mix_term(self, params: Params, data: ModelData, mix_weights) -> jnp.ndarray:
        r = self._rows(data)
        w = self._mix_weights(mix_weights)
        psi = self.states(params, data)[r["subject_idx"]]
        theta_L = jnp.asarray(params["theta_L"])
        psi_L = jax.vmap(lambda p, L: C.loss_unitary(theta_L, L) @ p)(psi, jnp.asarray(r["loss"]))
        return jax.vmap(C.mixture_interference, in_axes=(0, None, None))(psi_L, self.context_table(params), w)

    def interference_terms(self, params: Params, data: ModelData, mix_weights=None) -> dict[str, jnp.ndarray]:
        """``ltp`` and ``order`` per row (module docstring); ``mix`` when ``mix_weights`` (a
        ``tag -> weight`` dict or a ``(V,)`` array, normalized here) is given."""
        r = self._rows(data)
        n = data.n
        nan = jnp.full(n, jnp.nan)
        p_sf = self._predict_pure(params, data, order_flag=np.zeros(n, dtype=np.int64), tol_answer=np.full(n, np.nan))
        p_tf = self._predict_pure(params, data, order_flag=np.ones(n, dtype=np.int64), tol_answer=np.full(n, np.nan))
        pair = r["ctx_mask"].all(axis=1)
        p21 = self._predict_pure(
            params, data, order_flag=np.zeros(n, dtype=np.int64), tol_answer=np.full(n, np.nan),
            ctx_idx=r["ctx_idx"][:, ::-1].copy(), ctx_mask=r["ctx_mask"][:, ::-1].copy(),
        )
        out = {"ltp": p_sf - p_tf, "order": jnp.where(jnp.asarray(pair), p_sf - p21, nan)}
        if mix_weights is not None:
            out["mix"] = self._mix_term(params, data, mix_weights)
        return out

    # -- reporting --------------------------------------------------------------------------------

    def subject_summary(self, params: Params, data: ModelData) -> pd.DataFrame | None:
        """One row per subject: Bloch angles of ``psi_s`` (``bloch_theta = 2 atan2(|psi_1|,
        |psi_0|)``, ``bloch_phi = arg psi_1``), ``p_sell_base = P(sell | L = 0, no context)``
        and ``p_sell_L05 ... p_sell_L30``, the scenario-first ``P(sell)`` at each design loss
        level with no context."""
        psi = self.states(params, data)
        psi_np = np.asarray(psi)
        out = {
            "subject_id": np.asarray(data.subject_ids),
            "bloch_theta": 2.0 * np.arctan2(np.abs(psi_np[:, 1]), np.abs(psi_np[:, 0])),
            "bloch_phi": np.angle(psi_np[:, 1]),
            "p_sell_base": np.abs(psi_np[:, 1]) ** 2,
        }
        theta_L = jnp.asarray(params["theta_L"])
        for loss in self.loss_levels_design:
            u = C.loss_unitary(theta_L, loss)
            rotated = psi @ u.T
            out[f"p_sell_{_design.loss_code(-loss)}"] = np.asarray(jax.vmap(C.born, in_axes=(0, None))(rotated, C.P_SELL))
        return pd.DataFrame(out)

    def predict_new_subject(
        self,
        params: Params,
        covariates: dict[str, Any] | str,
        covariate_stats: dict[str, tuple[float, float]],
        loss_pct: float | None,
        context_tags: tuple[str, ...] | list[str] = (),
        question_order_id: str = _design.SCENARIO_FIRST,
        tol_answer: float | None = None,
    ) -> float:
        """:func:`predict_new_subject` with this model's vocabulary (``u = 0``)."""
        return predict_new_subject(
            params, covariates, covariate_stats, self.ctx_vocab, loss_pct, context_tags, question_order_id, tol_answer
        )


# ---------------------------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------------------------


def new_subject_state(params: Params, covariates: dict[str, Any] | str, covariate_stats: dict[str, tuple[float, float]]) -> jnp.ndarray:
    """``psi`` of a new subject: ``prepare_state(W^T x + b)`` with ``x`` the covariate record
    standardized by ``covariate_stats`` (``covariate_matrix``) and the random effect ``u = 0``."""
    from bre import schema as S

    rec = S.json_loads_strict(covariates) if isinstance(covariates, str) else dict(covariates)
    X, _, _ = covariate_matrix([rec], stats=covariate_stats)
    enc = params["enc"]
    z = jnp.asarray(X[0]) @ jnp.asarray(enc["W"]) + jnp.asarray(enc["b"])
    return C.prepare_state(z)


def context_row(ctx_vocab: tuple[str, ...], context_tags, n_ctx_positions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """``ctx_idx (K,)`` and ``ctx_mask (K,)`` of one scenario's ordered context tags."""
    tags = list(context_tags)
    if len(tags) > n_ctx_positions:
        raise ValueError(f"at most {n_ctx_positions} context tags; got {tags}")
    vocab = tuple(ctx_vocab)
    idx = np.full(n_ctx_positions, -1, dtype=np.int64)
    for k, t in enumerate(tags):
        if t not in vocab:
            raise ValueError(f"unknown context tag {t!r}; vocabulary {vocab}")
        idx[k] = vocab.index(t)
    return idx, idx >= 0


def predict_new_subject(
    params: Params,
    covariates: dict[str, Any] | str,
    covariate_stats: dict[str, tuple[float, float]],
    ctx_vocab: tuple[str, ...],
    loss_pct: float | None,
    context_tags: tuple[str, ...] | list[str] = (),
    question_order_id: str = _design.SCENARIO_FIRST,
    tol_answer: float | None = None,
    gamma: float | None = None,
) -> float:
    """``P(sell)`` for a new client (API path): state from :func:`new_subject_state` (``u = 0``),
    ``L = |loss_pct|`` (0 when None), the ordered ``context_tags`` looked up in ``ctx_vocab``
    (entries named ``none`` contribute no rotation), ``question_order_id`` one of the design's
    two orders, ``tol_answer`` the prior tolerance answer (0/1) for a tolerance-first scenario
    (None = unobserved, the Lüders mixture). ``gamma=None`` uses the pure forward pass
    (:func:`core.q_forward_pure`, Q2); a rate uses :func:`core.q_forward_rho` (Q4). Equals the
    model's ``predict_proba`` on a table holding that row with ``u = 0``.
    """
    if question_order_id not in _design.QUESTION_ORDERS:
        raise ValueError(f"unknown question order {question_order_id!r}; allowed {_design.QUESTION_ORDER_IDS}")
    psi = new_subject_state(params, covariates, covariate_stats)
    ctx_idx, ctx_mask = context_row(ctx_vocab, context_tags)
    free = ~none_mask(tuple(ctx_vocab))
    theta_ctx = jnp.asarray(params["theta_ctx"], dtype=jnp.float64) * jnp.asarray(free, dtype=jnp.float64)[:, None]
    ctx = C.gather_context_thetas(theta_ctx, ctx_idx, ctx_mask)
    L = 0.0 if loss_pct is None else abs(float(loss_pct))
    order_flag = 1 if question_order_id == _design.TOLERANCE_FIRST else 0
    tol = np.nan if (tol_answer is None or order_flag == 0) else float(tol_answer)
    if gamma is None:
        p = C.q_forward_pure(psi, params["theta_L"], L, ctx, ctx_mask, order_flag, params["phi"], tol)
    else:
        p = C.q_forward_rho(C.to_rho(psi), params["theta_L"], L, ctx, ctx_mask, order_flag, params["phi"], tol, float(gamma))
    return float(p)


__all__ = [
    "FORWARD_MODES",
    "GAUGE_ELEMENTS",
    "INIT_PHI_RANGE",
    "INIT_THETA_SCALE",
    "NONE_TAG",
    "Q2",
    "THETA_PRIOR_SCALE",
    "align_gauge",
    "apply_gauge",
    "context_row",
    "fold_theta",
    "forward_pure_fast",
    "forward_rho_fast",
    "needs_mixture",
    "new_subject_state",
    "none_mask",
    "predict_new_subject",
]
