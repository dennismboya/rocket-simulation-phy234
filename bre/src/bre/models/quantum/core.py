"""Hilbert-space primitives of the Q-models in C^2 (PLAN.md section 4).

Basis and conventions (fixed for every Q-model):

* ``e0 = |hold>``, ``e1 = |sell>``; a state is a unit vector ``psi`` in C^2 or a density matrix
  ``rho`` (2 x 2, Hermitian, trace one, positive).
* Questions are rank-1 projectors: ``P_SELL = diag(0, 1)``, ``P_HOLD = diag(1, 0)`` and the
  tolerance question ``P_yes(phi) = |yes><yes|`` with ``|yes> = (cos phi, sin phi)``.
* Contexts are unitaries ``U(theta) = expm(sum_k theta_k G_k)`` with the su(2) generators
  ``G_k = i sigma_k`` (skew-Hermitian, so ``U`` is unitary for every real ``theta``).
* Probabilities follow the Born rule ``P(sell) = ||P_SELL psi||^2 = Tr(P_SELL rho)`` and an
  observed answer collapses the state by the Lüders rule ``psi -> P psi / ||P psi||``.
* Decoherence is pure dephasing in the sell basis (Lindblad operator ``sigma_z``), which damps the
  off-diagonal elements of ``rho`` by ``exp(-2 gamma t)``.

Everything is pure JAX (``jax.numpy``), differentiable, ``jit``- and ``vmap``-able, in
``complex128`` / ``float64`` (``bre.models`` switches JAX to 64-bit on import). Every public
function is wrapped in ``jax.jit`` (so eager calls pay one compilation per argument shape and are
then cheap); nesting them in a model's own ``jit`` is fine. Scalars are 0-d arrays; batched
versions of the forward passes are provided at the end of the module.

Order of operations (PLAN.md section 4, Q2/Q4): the loss unitary acts first, then the contexts in
presentation order, i.e. ``psi' = U_{c_m} ... U_{c_1} U_L(L) psi``. The tolerance question, when
asked first, collapses ``psi`` *before* the chain.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import expm

CDTYPE = jnp.complex128
"""Complex dtype of every state, projector and unitary."""
RDTYPE = jnp.float64
"""Real dtype of every parameter and probability."""

NORM_EPS = 1e-30
"""Squared-norm threshold below which a vector is treated as the zero vector (norm below 1e-15).

Used by the zero-norm guards of :func:`lueders`, :func:`lueders_rho`, :func:`prepare_state` and
:func:`normalize` so that a probability-zero branch yields a finite (zero) state and finite
gradients instead of NaN."""

# ---------------------------------------------------------------------------------------------
# Basis vectors, Pauli matrices, generators, projectors
# ---------------------------------------------------------------------------------------------

E_HOLD = jnp.array([1.0, 0.0], dtype=CDTYPE)
"""``e0 = |hold>``."""
E_SELL = jnp.array([0.0, 1.0], dtype=CDTYPE)
"""``e1 = |sell>``."""

IDENTITY = jnp.eye(2, dtype=CDTYPE)

SIGMA_X = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=CDTYPE)
SIGMA_Y = jnp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=CDTYPE)
SIGMA_Z = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=CDTYPE)

PAULI = jnp.stack([SIGMA_X, SIGMA_Y, SIGMA_Z])
"""``(3, 2, 2)``: the Pauli matrices ``(sigma_x, sigma_y, sigma_z)``, Hermitian and unitary."""

GENERATORS = 1.0j * PAULI
"""``(3, 2, 2)``: ``G_k = i sigma_k``, the skew-Hermitian su(2) basis used by :func:`unitary`.

``G_k^dagger = -G_k`` for each k, so ``expm(sum_k theta_k G_k)`` is unitary for real ``theta``."""

P_SELL = jnp.diag(jnp.array([0.0, 1.0], dtype=CDTYPE))
"""Projector on ``|sell>``: ``diag(0, 1)``."""
P_HOLD = jnp.diag(jnp.array([1.0, 0.0], dtype=CDTYPE))
"""Projector on ``|hold>``: ``diag(1, 0)``; ``P_HOLD + P_SELL = I``."""


@jax.jit
def tolerance_projector(phi) -> jnp.ndarray:
    """Projector of the "yes" answer to the tolerance question at angle ``phi`` to the hold axis.

    ``|yes> = (cos phi, sin phi)`` (real) and ``P_yes = |yes><yes|``; the "no" projector is
    ``I - P_yes = |no><no|`` with ``|no> = (-sin phi, cos phi)``. ``phi = 0`` makes the tolerance
    question commute with the sell/hold question (``P_yes = P_HOLD``); ``phi = pi/4`` is the
    maximally non-commuting case. Returns a ``(2, 2)`` complex matrix.
    """
    phi = jnp.asarray(phi, dtype=RDTYPE)
    yes = jnp.array([jnp.cos(phi), jnp.sin(phi)]).astype(CDTYPE)
    return jnp.outer(yes, jnp.conj(yes))


# ---------------------------------------------------------------------------------------------
# Unitaries
# ---------------------------------------------------------------------------------------------


@jax.jit
def unitary(theta) -> jnp.ndarray:
    """Context unitary ``U(theta) = expm(sum_k theta_k G_k)``, ``theta`` real of shape ``(3,)``.

    Because ``sum_k theta_k G_k = i theta . sigma`` this equals
    ``cos|theta| I + i sin|theta| (theta/|theta|) . sigma``; the matrix exponential
    (``jax.scipy.linalg.expm``) is used instead of the closed form so that the gradient at
    ``theta = 0`` is well defined. ``theta = 0`` gives the identity. Batches: :func:`unitaries`.
    """
    theta = jnp.asarray(theta, dtype=RDTYPE)
    a = jnp.einsum("k,kij->ij", theta.astype(CDTYPE), GENERATORS)
    return expm(a)


unitaries = jax.jit(jax.vmap(unitary))
"""``(m, 3) -> (m, 2, 2)``: :func:`unitary` applied to every row of a parameter table."""


@jax.jit
def loss_unitary(theta_L, L) -> jnp.ndarray:
    """Loss-scenario unitary ``U_L(L) = U(L * theta_L)``, ``L = |loss_pct| >= 0`` a scalar.

    The rotation angle scales linearly with the loss magnitude so that ``L = 0`` (no scenario, or
    a null ``loss_pct``) is the identity.
    """
    L = jnp.asarray(L, dtype=RDTYPE)
    return unitary(L * jnp.asarray(theta_L, dtype=RDTYPE))


# ---------------------------------------------------------------------------------------------
# Born rule, Lüders collapse, state preparation
# ---------------------------------------------------------------------------------------------


@jax.jit
def normalize(psi) -> jnp.ndarray:
    """``psi / ||psi||`` with the zero-norm guard: a vector with ``||psi||^2 < NORM_EPS`` is
    returned as the zero vector (not NaN)."""
    psi = jnp.asarray(psi, dtype=CDTYPE)
    n2 = jnp.real(jnp.vdot(psi, psi))
    ok = n2 > NORM_EPS
    n2_safe = jnp.where(ok, n2, 1.0)
    return jnp.where(ok, psi / jnp.sqrt(n2_safe), jnp.zeros_like(psi))


@jax.jit
def born(psi, P) -> jnp.ndarray:
    """Born probability ``||P psi||^2 = <psi| P |psi>`` of the outcome with projector ``P``.

    For a unit ``psi`` and a complete set of orthogonal projectors the values sum to one.
    Returns a real 0-d array.
    """
    v = jnp.asarray(P, dtype=CDTYPE) @ jnp.asarray(psi, dtype=CDTYPE)
    return jnp.real(jnp.vdot(v, v))


@jax.jit
def lueders(psi, P) -> jnp.ndarray:
    """Lüders (projective) collapse ``psi -> P psi / ||P psi||`` after observing outcome ``P``.

    Guard: when ``||P psi||^2 < NORM_EPS`` the outcome has probability zero, the post-measurement
    state is undefined, and the zero vector is returned so that any downstream Born probability
    is zero and gradients stay finite. The collapse is idempotent: ``lueders(lueders(psi, P), P)
    == lueders(psi, P)``.
    """
    return normalize(jnp.asarray(P, dtype=CDTYPE) @ jnp.asarray(psi, dtype=CDTYPE))


@jax.jit
def to_rho(psi) -> jnp.ndarray:
    """Density matrix of a pure state: ``rho = |psi><psi| = psi psi^dagger`` (``(2, 2)``)."""
    psi = jnp.asarray(psi, dtype=CDTYPE)
    return jnp.outer(psi, jnp.conj(psi))


@jax.jit
def born_rho(rho, P) -> jnp.ndarray:
    """Born probability for a density matrix: ``Tr(P rho)`` (real part; the trace of a product of
    Hermitian matrices is real up to rounding)."""
    return jnp.real(jnp.trace(jnp.asarray(P, dtype=CDTYPE) @ jnp.asarray(rho, dtype=CDTYPE)))


@jax.jit
def lueders_rho(rho, P) -> jnp.ndarray:
    """Lüders collapse of a density matrix: ``rho -> P rho P / Tr(P rho)``.

    Guard: when ``Tr(P rho) < sqrt(NORM_EPS)`` (probability-zero outcome) the zero matrix is
    returned, mirroring :func:`lueders`.
    """
    P = jnp.asarray(P, dtype=CDTYPE)
    rho = jnp.asarray(rho, dtype=CDTYPE)
    num = P @ rho @ P
    p = born_rho(rho, P)
    ok = p > jnp.sqrt(NORM_EPS)
    p_safe = jnp.where(ok, p, 1.0)
    return jnp.where(ok, num / p_safe, jnp.zeros_like(num))


@jax.jit
def prepare_state(v) -> jnp.ndarray:
    """Map four real numbers to a unit state in C^2 with the global phase fixed (the gauge).

    ``psi = normalize(v[:2] + i v[2:])`` followed by ``psi -> exp(-i arg psi_0) psi``, i.e. the
    hold amplitude ``psi_0`` is made real and non-negative. Reason: the Born rule is invariant
    under a global phase and under the norm, so of the four numbers only two are identifiable;
    fixing ``||psi|| = 1`` and ``arg psi_0 = 0`` removes exactly those two degrees of freedom
    (``psi = (cos a, e^{i b} sin a)`` with ``a in [0, pi/2]``, ``b in (-pi, pi]``). Encoders
    (``bre.models.base.Encoder`` with ``d_out = 4``) feed this function.

    Guards: ``v = 0`` returns ``|hold>``; when ``psi_0 = 0`` the phase is left as is (the gauge
    is then fixed by ``psi_1`` being arbitrary, a measure-zero set for a continuous encoder).
    """
    v = jnp.asarray(v, dtype=RDTYPE)
    psi = v[:2].astype(CDTYPE) + 1.0j * v[2:].astype(CDTYPE)
    n2 = jnp.real(jnp.vdot(psi, psi))
    ok = n2 > NORM_EPS
    psi = jnp.where(ok, psi / jnp.sqrt(jnp.where(ok, n2, 1.0)), E_HOLD)
    a0 = psi[0]
    r = jnp.abs(a0)
    ok0 = r > jnp.sqrt(NORM_EPS)
    phase = jnp.where(ok0, jnp.conj(a0) / jnp.where(ok0, r, 1.0), 1.0 + 0.0j)
    return psi * phase


# ---------------------------------------------------------------------------------------------
# Context chains and dephasing
# ---------------------------------------------------------------------------------------------


@jax.jit
def apply_contexts(psi, thetas, mask) -> jnp.ndarray:
    """Apply ``U(thetas[0])``, then ``U(thetas[1])``, ... to ``psi``, skipping masked entries.

    ``thetas`` has shape ``(m, 3)`` and ``mask`` shape ``(m,)`` (bool; False = padding, the
    unitary is replaced by the identity). Result: ``psi' = U_m ... U_1 psi`` over the unmasked
    positions in order. ``m = 0`` returns ``psi`` unchanged.
    """
    psi = jnp.asarray(psi, dtype=CDTYPE)
    thetas = jnp.asarray(thetas, dtype=RDTYPE).reshape(-1, 3)
    mask = jnp.asarray(mask, dtype=bool).reshape(-1)
    us = unitaries(thetas)

    def step(state, xs):
        u, m = xs
        return jnp.where(m, u @ state, state), None

    out, _ = lax.scan(step, psi, (us, mask))
    return out


@jax.jit
def dephase(rho, gamma, t=1.0) -> jnp.ndarray:
    """Lindblad pure dephasing in the sell basis for time ``t`` at rate ``gamma``.

    Master equation ``d rho / dt = -i [H, rho] + gamma (L rho L^dagger - rho)`` with ``H = 0``
    during the dephasing step and ``L = sigma_z`` (``L^dagger L = I``, so this is the standard
    form ``gamma (L rho L^dagger - {L^dagger L, rho} / 2)``). Since ``sigma_z rho sigma_z`` flips
    the sign of the off-diagonal elements, the diagonal is constant and
    ``rho_01(t) = exp(-2 gamma t) rho_01(0)``: the solution multiplies the off-diagonal elements
    by ``exp(-2 gamma t)``. ``gamma = 0`` is the identity; ``gamma -> inf`` projects onto the
    diagonal (a classical mixture of hold and sell).
    """
    rho = jnp.asarray(rho, dtype=CDTYPE)
    f = jnp.exp(-2.0 * jnp.asarray(gamma, dtype=RDTYPE) * jnp.asarray(t, dtype=RDTYPE))
    damp = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=RDTYPE) + f * jnp.array(
        [[0.0, 1.0], [1.0, 0.0]], dtype=RDTYPE
    )
    return rho * damp.astype(CDTYPE)


@jax.jit
def apply_contexts_rho(rho, thetas, mask, gamma=0.0) -> jnp.ndarray:
    """Density-matrix version of :func:`apply_contexts` with dephasing after every context.

    For each unmasked position ``k`` in order: ``rho -> dephase(U_k rho U_k^dagger, gamma)``.
    Masked positions leave ``rho`` untouched (no unitary, no dephasing). ``gamma = 0`` is the
    unitary chain.
    """
    rho = jnp.asarray(rho, dtype=CDTYPE)
    thetas = jnp.asarray(thetas, dtype=RDTYPE).reshape(-1, 3)
    mask = jnp.asarray(mask, dtype=bool).reshape(-1)
    us = unitaries(thetas)

    def step(state, xs):
        u, m = xs
        new = dephase(u @ state @ jnp.conj(u).T, gamma)
        return jnp.where(m, new, state), None

    out, _ = lax.scan(step, rho, (us, mask))
    return out


@jax.jit
def chain_pure(psi, theta_L, L, ctx_thetas, ctx_mask) -> jnp.ndarray:
    """The Q2 state chain: ``psi' = U_{c_m} ... U_{c_1} U_L(L) psi`` (contexts after the loss)."""
    psi1 = loss_unitary(theta_L, L) @ jnp.asarray(psi, dtype=CDTYPE)
    return apply_contexts(psi1, ctx_thetas, ctx_mask)


@jax.jit
def chain_rho(rho, theta_L, L, ctx_thetas, ctx_mask, gamma) -> jnp.ndarray:
    """The Q4 state chain: ``rho -> dephase(U_L rho U_L^dagger)`` then, per unmasked context,
    ``rho -> dephase(U_c rho U_c^dagger)``, all at rate ``gamma`` (:func:`dephase`)."""
    u = loss_unitary(theta_L, L)
    rho1 = dephase(u @ jnp.asarray(rho, dtype=CDTYPE) @ jnp.conj(u).T, gamma)
    return apply_contexts_rho(rho1, ctx_thetas, ctx_mask, gamma)


# ---------------------------------------------------------------------------------------------
# Forward passes: P(sell | loss, context sequence, question order, prior tolerance answer)
# ---------------------------------------------------------------------------------------------


def _tolerance_pair(phi_tol):
    p_yes = tolerance_projector(phi_tol)
    return p_yes, IDENTITY - p_yes


@jax.jit
def q_forward_pure(
    psi, theta_L, L, ctx_thetas, ctx_mask, order_flag, phi_tol, tol_answer, projector=P_SELL
) -> jnp.ndarray:
    """``P(sell)`` of the pure-state context-unitary model (Q2) for one response row.

    Scenario-first (``order_flag = 0``)::

        p = ||P_SELL U_{c_m} ... U_{c_1} U_L(L) psi||^2

    Tolerance-first (``order_flag = 1``) with observed answer ``a in {0, 1}`` (``P_1 =
    tolerance_projector(phi_tol)``, ``P_0 = I - P_1``)::

        psi_a = P_a psi / ||P_a psi||            (Lüders collapse)
        p     = ||P_SELL U_{c_m} ... U_{c_1} U_L(L) psi_a||^2

    Tolerance-first with ``tol_answer = NaN`` (answer not observed): the Lüders mixture
    ``p = sum_a ||P_a psi||^2 p(sell | a)``.

    ``projector`` (default ``P_SELL``) is the measured question; pass ``P_HOLD`` for ``P(hold)``.
    All branches are evaluated and selected with ``jnp.where`` so the function is ``vmap``-able
    over rows (see :func:`q_forward_pure_batch`) and differentiable in ``psi``, ``theta_L``,
    ``ctx_thetas`` and ``phi_tol``.
    """
    psi = jnp.asarray(psi, dtype=CDTYPE)
    p1, p0 = _tolerance_pair(phi_tol)
    tol_answer = jnp.asarray(tol_answer, dtype=RDTYPE)
    order_flag = jnp.asarray(order_flag)

    def measure(state):
        return born(chain_pure(state, theta_L, L, ctx_thetas, ctx_mask), projector)

    p_sf = measure(psi)
    w1, w0 = born(psi, p1), born(psi, p0)
    p_given_1 = measure(lueders(psi, p1))
    p_given_0 = measure(lueders(psi, p0))
    p_obs = jnp.where(tol_answer >= 0.5, p_given_1, p_given_0)
    p_mix = w1 * p_given_1 + w0 * p_given_0
    p_tf = jnp.where(jnp.isnan(tol_answer), p_mix, p_obs)
    return jnp.where(order_flag == 1, p_tf, p_sf)


@jax.jit
def q_forward_rho(
    rho, theta_L, L, ctx_thetas, ctx_mask, order_flag, phi_tol, tol_answer, gamma, projector=P_SELL
) -> jnp.ndarray:
    """``P(sell)`` of the open-system model (Q4): :func:`q_forward_pure` on a density matrix with
    dephasing at rate ``gamma`` after ``U_L`` and after every context unitary (:func:`chain_rho`).

    Scenario-first: ``p = Tr(P_SELL rho')`` with ``rho' = chain_rho(rho, ...)``. Tolerance-first
    with observed ``a``: ``rho_a = P_a rho P_a / Tr(P_a rho)`` then the chain; unobserved: the
    mixture ``sum_a Tr(P_a rho) p(sell | a)``. ``gamma = 0`` reproduces :func:`q_forward_pure`
    for ``rho = to_rho(psi)``; ``gamma -> inf`` turns the chain into a classical Markov chain
    whose initial distribution is ``diag(U_L rho U_L^dagger)`` and whose transition matrices
    are ``T_ij = |(U_c)_ij|^2`` (the coherence of the initial state still enters the first step).
    ``rho`` may be any density matrix; a pure state goes through :func:`to_rho`.
    """
    rho = jnp.asarray(rho, dtype=CDTYPE)
    p1, p0 = _tolerance_pair(phi_tol)
    tol_answer = jnp.asarray(tol_answer, dtype=RDTYPE)
    order_flag = jnp.asarray(order_flag)

    def measure(state):
        return born_rho(chain_rho(state, theta_L, L, ctx_thetas, ctx_mask, gamma), projector)

    p_sf = measure(rho)
    w1, w0 = born_rho(rho, p1), born_rho(rho, p0)
    p_given_1 = measure(lueders_rho(rho, p1))
    p_given_0 = measure(lueders_rho(rho, p0))
    p_obs = jnp.where(tol_answer >= 0.5, p_given_1, p_given_0)
    p_mix = w1 * p_given_1 + w0 * p_given_0
    p_tf = jnp.where(jnp.isnan(tol_answer), p_mix, p_obs)
    return jnp.where(order_flag == 1, p_tf, p_sf)


# ---------------------------------------------------------------------------------------------
# Interference terms (definitions fixed in PLAN.md section 4)
# ---------------------------------------------------------------------------------------------

_NAN = jnp.nan


@jax.jit
def ltp_interference(psi, theta_L, L, ctx_thetas, ctx_mask, phi_tol) -> jnp.ndarray:
    """Law-of-total-probability interference (primary term, PLAN.md section 4)::

        delta_LTP = P(sell | scenario-first) - sum_a P(a, sell | tolerance-first)

    with ``P(a, sell | tolerance-first) = ||P_a psi||^2 P(sell | a)``. A classical model predicts
    zero. It is zero whenever the tolerance projectors commute with the Heisenberg-picture sell
    projector ``V^dagger P_SELL V`` (``V`` the chain), e.g. ``phi_tol = 0`` with no chain
    (``theta_L = 0`` and no contexts) or with a chain diagonal in the sell basis (``sigma_z``
    rotations only). With a generic chain the term is non-zero even at ``phi_tol = 0``, because
    the first measurement destroys the hold/sell coherence that the chain then rotates.
    """
    p_sf = q_forward_pure(psi, theta_L, L, ctx_thetas, ctx_mask, 0, phi_tol, _NAN)
    p_tf = q_forward_pure(psi, theta_L, L, ctx_thetas, ctx_mask, 1, phi_tol, _NAN)
    return p_sf - p_tf


@jax.jit
def ltp_interference_rho(rho, theta_L, L, ctx_thetas, ctx_mask, phi_tol, gamma) -> jnp.ndarray:
    """:func:`ltp_interference` for the open-system chain (:func:`q_forward_rho`) at rate ``gamma``."""
    p_sf = q_forward_rho(rho, theta_L, L, ctx_thetas, ctx_mask, 0, phi_tol, _NAN, gamma)
    p_tf = q_forward_rho(rho, theta_L, L, ctx_thetas, ctx_mask, 1, phi_tol, _NAN, gamma)
    return p_sf - p_tf


@jax.jit
def mixture_interference(psi, thetas, weights, projector=P_SELL) -> jnp.ndarray:
    """Context-uncertainty interference (PLAN.md section 4) for a cause frame that is a mixture of
    contexts ``c`` with weights ``p_c`` (``thetas`` of shape ``(C, 3)``, ``weights`` ``(C,)``,
    summing to one)::

        delta_mix = ||P_SELL normalize(sum_c sqrt(p_c) U_c psi)||^2 - sum_c p_c ||P_SELL U_c psi||^2

    The first term is the Born probability of the coherent superposition of the context-rotated
    states (renormalized, because the superposition is not a unit vector in general); the second
    is the classical mixture. Zero for a single context and whenever all ``U_c`` coincide.
    """
    psi = jnp.asarray(psi, dtype=CDTYPE)
    weights = jnp.asarray(weights, dtype=RDTYPE).reshape(-1)
    thetas = jnp.asarray(thetas, dtype=RDTYPE).reshape(-1, 3)
    rotated = jnp.einsum("cij,j->ci", unitaries(thetas), psi)  # (C, 2)
    superposition = normalize(jnp.sum(jnp.sqrt(weights)[:, None].astype(CDTYPE) * rotated, axis=0))
    p_coherent = born(superposition, projector)
    p_classical = jnp.sum(weights * jax.vmap(born, in_axes=(0, None))(rotated, projector))
    return p_coherent - p_classical


@jax.jit
def order_effect(psi, theta_1, theta_2, theta_L=None, L=0.0, projector=P_SELL) -> jnp.ndarray:
    """Order effect of a context pair (PLAN.md section 4)::

        Delta_order(c1, c2) = P(sell | c1, c2) - P(sell | c2, c1)

    with ``P(sell | c1, c2) = ||P_SELL U_{c2} U_{c1} U_L(L) psi||^2`` (scenario-first, no prior
    tolerance answer). ``theta_L`` defaults to zero (no loss rotation). Zero when ``U_{c1}`` and
    ``U_{c2}`` commute, e.g. both generated along the same Pauli axis.
    """
    theta_L = jnp.zeros(3, dtype=RDTYPE) if theta_L is None else jnp.asarray(theta_L, dtype=RDTYPE)
    t1 = jnp.asarray(theta_1, dtype=RDTYPE)
    t2 = jnp.asarray(theta_2, dtype=RDTYPE)
    mask = jnp.array([True, True])
    p12 = born(chain_pure(psi, theta_L, L, jnp.stack([t1, t2]), mask), projector)
    p21 = born(chain_pure(psi, theta_L, L, jnp.stack([t2, t1]), mask), projector)
    return p12 - p21


@partial(jax.jit, static_argnames=("is_rho",))
def order_effect_rho(psi_or_rho, theta_1, theta_2, gamma, theta_L=None, L=0.0, projector=P_SELL, *, is_rho=False):
    """:func:`order_effect` for the open-system chain at rate ``gamma``.

    The first argument is a density matrix when ``is_rho`` is True, otherwise a pure state that is
    converted with :func:`to_rho`. Note that the dephasing step between the two contexts is not
    unitary and does not commute with ``sigma_x``/``sigma_y`` rotations, so two commuting
    unitaries can still show an order effect at ``gamma > 0``; it vanishes for every ``gamma``
    only when both also commute with the dephasing (``sigma_z`` rotations).
    """
    rho = jnp.asarray(psi_or_rho, dtype=CDTYPE) if is_rho else to_rho(psi_or_rho)
    theta_L = jnp.zeros(3, dtype=RDTYPE) if theta_L is None else jnp.asarray(theta_L, dtype=RDTYPE)
    t1 = jnp.asarray(theta_1, dtype=RDTYPE)
    t2 = jnp.asarray(theta_2, dtype=RDTYPE)
    mask = jnp.array([True, True])
    p12 = born_rho(chain_rho(rho, theta_L, L, jnp.stack([t1, t2]), mask, gamma), projector)
    p21 = born_rho(chain_rho(rho, theta_L, L, jnp.stack([t2, t1]), mask, gamma), projector)
    return p12 - p21


# ---------------------------------------------------------------------------------------------
# Batched helpers for whole ModelData tables
# ---------------------------------------------------------------------------------------------


@jax.jit
def gather_context_thetas(theta_ctx, ctx_idx, ctx_mask) -> jnp.ndarray:
    """Look up per-row context parameters: ``(V, 3)`` table, ``(n, K)`` indices with ``-1``
    padding, ``(n, K)`` mask -> ``(n, K, 3)`` with zeros (identity unitaries) at masked slots.

    ``V`` is the size of ``ModelData.ctx_vocab``; the ``none`` condition has no row in the table
    (``theta_none = 0`` is the reference gauge of PLAN.md section 4) and appears as an all-False
    mask row.
    """
    theta_ctx = jnp.asarray(theta_ctx, dtype=RDTYPE)
    idx = jnp.clip(jnp.asarray(ctx_idx), 0, theta_ctx.shape[0] - 1)
    mask = jnp.asarray(ctx_mask, dtype=bool)
    return jnp.where(mask[..., None], theta_ctx[idx], 0.0)


q_forward_pure_batch = jax.jit(jax.vmap(q_forward_pure, in_axes=(0, None, 0, 0, 0, 0, None, 0)))
"""Row-wise :func:`q_forward_pure`: ``psi (n, 2)``, ``theta_L (3,)`` shared, ``L (n,)``,
``ctx_thetas (n, K, 3)``, ``ctx_mask (n, K)``, ``order_flag (n,)``, ``phi_tol`` shared,
``tol_answer (n,)`` -> ``P(sell) (n,)``."""

q_forward_rho_batch = jax.jit(jax.vmap(q_forward_rho, in_axes=(0, None, 0, 0, 0, 0, None, 0, 0)))
"""Row-wise :func:`q_forward_rho`: as :data:`q_forward_pure_batch` with ``rho (n, 2, 2)`` and a
per-row ``gamma (n,)`` (Q4 has one rate per investor: pass ``gamma_subject[subject_idx]``)."""

ltp_interference_batch = jax.jit(jax.vmap(ltp_interference, in_axes=(0, None, 0, 0, 0, None)))
"""Row-wise :func:`ltp_interference` -> ``(n,)``."""

ltp_interference_rho_batch = jax.jit(jax.vmap(ltp_interference_rho, in_axes=(0, None, 0, 0, 0, None, 0)))
"""Row-wise :func:`ltp_interference_rho` -> ``(n,)``."""

__all__ = [
    "CDTYPE",
    "RDTYPE",
    "NORM_EPS",
    "E_HOLD",
    "E_SELL",
    "IDENTITY",
    "SIGMA_X",
    "SIGMA_Y",
    "SIGMA_Z",
    "PAULI",
    "GENERATORS",
    "P_SELL",
    "P_HOLD",
    "tolerance_projector",
    "unitary",
    "unitaries",
    "loss_unitary",
    "normalize",
    "born",
    "lueders",
    "to_rho",
    "born_rho",
    "lueders_rho",
    "prepare_state",
    "apply_contexts",
    "dephase",
    "apply_contexts_rho",
    "chain_pure",
    "chain_rho",
    "q_forward_pure",
    "q_forward_rho",
    "ltp_interference",
    "ltp_interference_rho",
    "mixture_interference",
    "order_effect",
    "order_effect_rho",
    "gather_context_thetas",
    "q_forward_pure_batch",
    "q_forward_rho_batch",
    "ltp_interference_batch",
    "ltp_interference_rho_batch",
]
