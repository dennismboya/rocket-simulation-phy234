"""Tests for the model core: ``bre.models.quantum.core`` (Hilbert-space math), ``bre.models.data``
(ModelData) and ``bre.models.base`` (model contract, Encoder, standard features).

The math tests follow PLAN.md section 4: unitarity, probabilities summing to one, static limits,
the gamma = 0 and gamma -> inf limits of the open-system chain (the latter against an
independent numpy/scipy Markov-chain reference written here), Lüders idempotence and the three
interference terms. Randomness comes from the ``rng`` fixture (seed 0).
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm as scipy_expm

from bre import design as D
from bre import schema as S
from bre.models import (
    Encoder,
    Model,
    ModelBase,
    ModelData,
    QuantumModelBase,
    bernoulli_log_lik,
    build_model_data,
    count_params,
    encoder_log_prior,
    standard_features,
)
from bre.models import data as MD
from bre.models.quantum import core as C

TOL = 1e-10


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def rand_theta(rng: np.random.Generator, scale: float = 1.5) -> jnp.ndarray:
    return jnp.asarray(rng.normal(size=3) * scale)


def rand_state(rng: np.random.Generator) -> jnp.ndarray:
    return C.prepare_state(jnp.asarray(rng.normal(size=4)))


def np_unitary(theta: np.ndarray) -> np.ndarray:
    """Independent reference: ``expm(i theta . sigma)`` with scipy."""
    sig = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    return scipy_expm(1j * sum(t * s for t, s in zip(np.asarray(theta, float), sig)))


def np_markov_forward(psi, theta_L, L, ctx_thetas, ctx_mask, order_flag, phi, tol_answer) -> float:
    """Reference for the gamma -> inf limit of ``q_forward_rho``: after the loss unitary the state
    is fully dephased, so the remaining chain is a classical Markov chain over {hold, sell} with
    transition matrices ``T_ij = |(U_c)_ij|^2`` and initial distribution ``diag(U_L rho U_L^+)``.
    The tolerance-first branches collapse ``rho`` with the Lüders rule before the chain."""
    psi = np.asarray(psi, complex)
    rho = np.outer(psi, psi.conj())

    def chain(rho0: np.ndarray) -> float:
        u = np_unitary(L * np.asarray(theta_L, float))
        p = np.real(np.diag(u @ rho0 @ u.conj().T))
        for th, m in zip(np.asarray(ctx_thetas, float), np.asarray(ctx_mask, bool)):
            if m:
                t = np.abs(np_unitary(th)) ** 2
                p = t @ p
        return float(p[1])

    if order_flag == 0:
        return chain(rho)
    yes = np.array([np.cos(phi), np.sin(phi)], dtype=complex)
    p1 = np.outer(yes, yes.conj())
    p0 = np.eye(2) - p1

    def collapse(proj):
        pr = float(np.real(np.trace(proj @ rho)))
        return pr, (proj @ rho @ proj) / pr

    w1, r1 = collapse(p1)
    w0, r0 = collapse(p0)
    if np.isnan(tol_answer):
        return w1 * chain(r1) + w0 * chain(r0)
    return chain(r1) if tol_answer == 1 else chain(r0)


def _event(**kw) -> S.DecisionEvent:
    base = dict(
        subject_id="s000",
        dataset="synthetic_test",
        session_id="full",
        timestamp=None,
        position_in_session=0,
        scenario_id="L10|none",
        loss_pct=-0.10,
        horizon_days=365,
        context_tags=[],
        question_order_id=D.SCENARIO_FIRST,
        prior_question_ids=[],
        elicitation_type="binary_sell",
        response=0.0,
        response_time_ms=None,
        covariates={},
        outcome_behavior=None,
        incentivized=False,
        consent_training=True,
        battery_version=D.DESIGN_VERSION,
        is_synthetic=True,
        source_row_ref="test",
    )
    base.update(kw)
    return S.DecisionEvent(**base)


def design_frame(rng: np.random.Generator, n_subjects: int = 3, with_tolerance_rows: bool = True) -> pd.DataFrame:
    """``n_subjects`` subjects answering the full 170-item design; with ``with_tolerance_rows``
    every item is stored as its two rows (tolerance + sell in the item's order), so tolerance-first
    sell rows have a tolerance row at position - 1."""
    design = D.full_design()
    covs = D.sample_covariates(rng, n_subjects)
    events = []
    for s in range(n_subjects):
        cov = covs.iloc[s].to_dict()
        if s == 1:  # exercise missing covariates
            cov = {"age_band": cov["age_band"]}
        pos = 0
        for row in design.itertuples(index=False):
            tol_first = row.question_order_id == D.TOLERANCE_FIRST
            sell_kw = dict(
                subject_id=f"s{s:03d}",
                position_in_session=pos,
                scenario_id=row.scenario_id,
                loss_pct=row.loss_pct,
                horizon_days=row.horizon_days,
                context_tags=list(row.context_tags),
                question_order_id=row.question_order_id,
                prior_question_ids=list(row.prior_question_ids),
                elicitation_type="binary_sell",
                response=float(rng.integers(0, 2)),
                covariates=cov,
                source_row_ref=row.item_id,
            )
            tol_kw = dict(
                subject_id=f"s{s:03d}",
                scenario_id=D.TOLERANCE_QUESTION_ID,
                loss_pct=None,
                horizon_days=None,
                context_tags=[],
                question_order_id=row.question_order_id,
                prior_question_ids=[] if tol_first else [row.scenario_id],
                elicitation_type="binary_yes_no",
                response=float(rng.integers(0, 2)),
                covariates=cov,
                source_row_ref=row.item_id + "|tol",
            )
            if not with_tolerance_rows:
                events.append(_event(**sell_kw))
                pos += 1
                continue
            if tol_first:
                events.append(_event(position_in_session=pos, **tol_kw))
                sell_kw["position_in_session"] = pos + 1
                events.append(_event(**sell_kw))
            else:
                events.append(_event(**sell_kw))
                events.append(_event(position_in_session=pos + 1, **tol_kw))
            pos += 2
    return S.events_to_frame(events)


# ---------------------------------------------------------------------------------------------
# core: Pauli matrices, generators, unitaries
# ---------------------------------------------------------------------------------------------


def test_x64_enabled_and_dtypes():
    assert jax.config.jax_enable_x64 is True
    assert C.PAULI.dtype == jnp.complex128
    assert C.unitary(jnp.zeros(3)).dtype == jnp.complex128


def test_pauli_hermitian_and_generators_skew_hermitian():
    for k in range(3):
        sigma = np.asarray(C.PAULI[k])
        g = np.asarray(C.GENERATORS[k])
        assert np.allclose(sigma, sigma.conj().T, atol=TOL)
        assert np.allclose(sigma @ sigma, np.eye(2), atol=TOL)
        assert np.allclose(g.conj().T, -g, atol=TOL)
        assert np.allclose(g, 1j * sigma, atol=TOL)


def test_unitary_at_zero_is_identity():
    assert np.allclose(np.asarray(C.unitary(jnp.zeros(3))), np.eye(2), atol=TOL)
    assert np.allclose(np.asarray(C.loss_unitary(jnp.array([1.0, 2.0, 3.0]), 0.0)), np.eye(2), atol=TOL)


def test_unitary_is_unitary_for_random_theta(rng):
    thetas = jnp.asarray(rng.normal(size=(50, 3)) * 3.0)
    us = np.asarray(C.unitaries(thetas))
    assert us.shape == (50, 2, 2)
    for u in us:
        assert np.abs(u.conj().T @ u - np.eye(2)).max() < TOL
        assert np.abs(u @ u.conj().T - np.eye(2)).max() < TOL
        assert abs(abs(np.linalg.det(u)) - 1.0) < TOL


def test_unitary_matches_closed_form_and_scipy(rng):
    theta = rng.normal(size=3)
    u = np.asarray(C.unitary(jnp.asarray(theta)))
    a = np.linalg.norm(theta)
    n = theta / a
    closed = np.cos(a) * np.eye(2) + 1j * np.sin(a) * sum(n[k] * np.asarray(C.PAULI[k]) for k in range(3))
    assert np.allclose(u, closed, atol=TOL)
    assert np.allclose(u, np_unitary(theta), atol=TOL)


def test_loss_unitary_scales_angle_with_loss(rng):
    theta = rand_theta(rng)
    assert np.allclose(np.asarray(C.loss_unitary(theta, 0.3)), np.asarray(C.unitary(0.3 * theta)), atol=TOL)


# ---------------------------------------------------------------------------------------------
# core: projectors, Born rule, Lüders collapse
# ---------------------------------------------------------------------------------------------


def test_sell_hold_projectors_complete_and_orthogonal():
    ps, ph = np.asarray(C.P_SELL), np.asarray(C.P_HOLD)
    assert np.allclose(ps + ph, np.eye(2))
    assert np.allclose(ps @ ph, 0)
    assert np.allclose(ps @ ps, ps) and np.allclose(ph @ ph, ph)
    assert np.allclose(ps @ np.asarray(C.E_SELL), np.asarray(C.E_SELL))
    assert np.allclose(ps @ np.asarray(C.E_HOLD), 0)


@pytest.mark.parametrize("phi", [0.0, 0.3, np.pi / 4, 1.2, np.pi / 2])
def test_tolerance_projector_idempotent_hermitian_complete(phi):
    p1 = np.asarray(C.tolerance_projector(phi))
    p0 = np.eye(2) - p1
    assert np.allclose(p1 @ p1, p1, atol=TOL)
    assert np.allclose(p1, p1.conj().T, atol=TOL)
    assert abs(np.trace(p1) - 1.0) < TOL
    assert np.allclose(p0 @ p0, p0, atol=TOL)
    assert np.allclose(p1 + p0, np.eye(2), atol=TOL)
    assert np.allclose(p1 @ p0, 0, atol=TOL)


def test_tolerance_projector_phi_zero_is_hold_and_half_pi_is_sell():
    assert np.allclose(np.asarray(C.tolerance_projector(0.0)), np.asarray(C.P_HOLD), atol=TOL)
    assert np.allclose(np.asarray(C.tolerance_projector(np.pi / 2)), np.asarray(C.P_SELL), atol=TOL)


def test_born_probabilities_sum_to_one(rng):
    for _ in range(10):
        psi = rand_state(rng)
        phi = rng.uniform(0, np.pi)
        p1 = C.tolerance_projector(phi)
        assert abs(float(C.born(psi, C.P_SELL) + C.born(psi, C.P_HOLD)) - 1.0) < TOL
        assert abs(float(C.born(psi, p1) + C.born(psi, C.IDENTITY - p1)) - 1.0) < TOL
        assert 0.0 <= float(C.born(psi, C.P_SELL)) <= 1.0


def test_lueders_normalized_and_idempotent(rng):
    psi = rand_state(rng)
    p1 = C.tolerance_projector(0.7)
    post = C.lueders(psi, p1)
    assert abs(float(jnp.vdot(post, post).real) - 1.0) < TOL
    assert np.allclose(np.asarray(C.lueders(post, p1)), np.asarray(post), atol=TOL)
    assert abs(float(C.born(post, p1)) - 1.0) < TOL  # the answer is now certain


def test_lueders_zero_norm_guard_is_finite():
    post = C.lueders(C.E_HOLD, C.P_SELL)  # P(sell) = 0 exactly
    assert np.all(np.isfinite(np.asarray(post)))
    assert np.allclose(np.asarray(post), 0)
    post_rho = C.lueders_rho(C.to_rho(C.E_HOLD), C.P_SELL)
    assert np.all(np.isfinite(np.asarray(post_rho)))
    assert np.allclose(np.asarray(post_rho), 0)
    grad = jax.grad(lambda v: C.born(C.lueders(C.prepare_state(v), C.P_SELL), C.P_SELL))(jnp.array([1.0, 0.0, 0.0, 0.0]))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_density_matrix_born_and_lueders_match_pure(rng):
    psi = rand_state(rng)
    rho = C.to_rho(psi)
    p1 = C.tolerance_projector(1.1)
    assert abs(float(C.born_rho(rho, p1)) - float(C.born(psi, p1))) < TOL
    assert np.allclose(np.asarray(C.lueders_rho(rho, p1)), np.asarray(C.to_rho(C.lueders(psi, p1))), atol=TOL)


def test_to_rho_is_pure_hermitian_trace_one(rng):
    rho = np.asarray(C.to_rho(rand_state(rng)))
    assert np.allclose(rho, rho.conj().T, atol=TOL)
    assert abs(np.trace(rho) - 1.0) < TOL
    assert np.allclose(rho @ rho, rho, atol=TOL)


# ---------------------------------------------------------------------------------------------
# core: state preparation and gauge
# ---------------------------------------------------------------------------------------------


def test_prepare_state_normalized_with_real_nonnegative_hold_amplitude(rng):
    for _ in range(20):
        v = rng.normal(size=4) * 3
        psi = np.asarray(C.prepare_state(jnp.asarray(v)))
        assert abs(np.vdot(psi, psi).real - 1.0) < TOL
        assert abs(psi[0].imag) < TOL
        assert psi[0].real >= 0.0
        raw = v[:2] + 1j * v[2:]
        raw = raw / np.linalg.norm(raw)
        # same Born probabilities as the un-gauged normalized vector
        assert abs(abs(psi[1]) ** 2 - abs(raw[1]) ** 2) < TOL
        # equal up to a global phase
        assert abs(abs(np.vdot(raw, psi)) - 1.0) < TOL


def test_prepare_state_guards_zero_vector_and_zero_hold_amplitude():
    assert np.allclose(np.asarray(C.prepare_state(jnp.zeros(4))), np.asarray(C.E_HOLD))
    psi = np.asarray(C.prepare_state(jnp.array([0.0, 2.0, 0.0, 0.0])))
    assert np.allclose(psi, np.asarray(C.E_SELL))
    assert np.all(np.isfinite(psi))


# ---------------------------------------------------------------------------------------------
# core: context chains and dephasing
# ---------------------------------------------------------------------------------------------


def test_apply_contexts_order_and_mask(rng):
    psi = rand_state(rng)
    t1, t2 = rand_theta(rng), rand_theta(rng)
    u1, u2 = np.asarray(C.unitary(t1)), np.asarray(C.unitary(t2))
    out = np.asarray(C.apply_contexts(psi, jnp.stack([t1, t2]), jnp.array([True, True])))
    assert np.allclose(out, u2 @ u1 @ np.asarray(psi), atol=TOL)  # U_1 first, then U_2
    only_second = np.asarray(C.apply_contexts(psi, jnp.stack([t1, t2]), jnp.array([False, True])))
    assert np.allclose(only_second, u2 @ np.asarray(psi), atol=TOL)
    none = np.asarray(C.apply_contexts(psi, jnp.stack([t1, t2]), jnp.array([False, False])))
    assert np.allclose(none, np.asarray(psi), atol=TOL)
    empty = np.asarray(C.apply_contexts(psi, jnp.zeros((0, 3)), jnp.zeros((0,), bool)))
    assert np.allclose(empty, np.asarray(psi), atol=TOL)


def test_apply_contexts_rho_matches_pure_without_dephasing(rng):
    psi = rand_state(rng)
    thetas = jnp.stack([rand_theta(rng), rand_theta(rng)])
    mask = jnp.array([True, True])
    out_psi = C.apply_contexts(psi, thetas, mask)
    out_rho = C.apply_contexts_rho(C.to_rho(psi), thetas, mask, gamma=0.0)
    assert np.allclose(np.asarray(out_rho), np.asarray(C.to_rho(out_psi)), atol=TOL)


def test_dephase_gamma_zero_is_identity(rng):
    rho = C.to_rho(rand_state(rng))
    assert np.allclose(np.asarray(C.dephase(rho, 0.0)), np.asarray(rho), atol=TOL)


def test_dephase_large_gamma_makes_rho_diagonal(rng):
    rho = np.asarray(C.to_rho(rand_state(rng)))
    out = np.asarray(C.dephase(jnp.asarray(rho), 1e3))
    assert abs(out[0, 1]) < 1e-12 and abs(out[1, 0]) < 1e-12
    assert np.allclose(np.diag(out), np.diag(rho), atol=TOL)


def test_dephase_solution_and_trace_preservation(rng):
    rho = np.asarray(C.to_rho(rand_state(rng)))
    gamma, t = 0.37, 2.5
    out = np.asarray(C.dephase(jnp.asarray(rho), gamma, t))
    assert abs(out[0, 1] - np.exp(-2 * gamma * t) * rho[0, 1]) < TOL
    assert abs(np.trace(out) - 1.0) < TOL
    assert np.allclose(out, out.conj().T, atol=TOL)
    # matches the Lindblad generator integrated numerically: d rho/dt = gamma (Z rho Z - rho)
    z = np.asarray(C.SIGMA_Z)
    r, dt = rho.copy(), 1e-4
    for _ in range(int(round(t / dt))):
        r = r + dt * gamma * (z @ r @ z - r)
    assert np.abs(r - out).max() < 1e-3


# ---------------------------------------------------------------------------------------------
# core: forward passes
# ---------------------------------------------------------------------------------------------


def _random_case(rng, n_ctx: int = 2):
    psi = rand_state(rng)
    theta_L = rand_theta(rng)
    L = float(rng.choice(np.abs(D.LOSS_PCTS)))
    ctx = jnp.asarray(rng.normal(size=(n_ctx, 3)) * 1.5)
    mask = jnp.asarray(rng.integers(0, 2, size=n_ctx).astype(bool))
    phi = float(rng.uniform(0, np.pi / 2))
    return psi, theta_L, L, ctx, mask, phi


@pytest.mark.parametrize("order_flag,tol_answer", [(0, np.nan), (1, np.nan), (1, 0.0), (1, 1.0)])
def test_q_forward_pure_probabilities_in_unit_interval_and_complementary(rng, order_flag, tol_answer):
    for _ in range(10):
        psi, theta_L, L, ctx, mask, phi = _random_case(rng)
        p_sell = float(C.q_forward_pure(psi, theta_L, L, ctx, mask, order_flag, phi, tol_answer))
        p_hold = float(C.q_forward_pure(psi, theta_L, L, ctx, mask, order_flag, phi, tol_answer, projector=C.P_HOLD))
        assert 0.0 <= p_sell <= 1.0
        assert abs(p_sell + p_hold - 1.0) < TOL


def test_q_forward_static_born_probability_when_all_thetas_zero(rng):
    psi = rand_state(rng)
    zeros = jnp.zeros(3)
    ctx = jnp.zeros((2, 3))
    mask = jnp.array([True, True])
    p = float(C.q_forward_pure(psi, zeros, 0.3, ctx, mask, 0, 0.9, np.nan))
    assert abs(p - float(C.born(psi, C.P_SELL))) < TOL
    assert abs(p - abs(np.asarray(psi)[1]) ** 2) < TOL


def test_q_forward_tolerance_first_mixture_is_lueders_average(rng):
    psi, theta_L, L, ctx, mask, phi = _random_case(rng)
    p1 = C.tolerance_projector(phi)
    w1 = float(C.born(psi, p1))
    p_yes = float(C.q_forward_pure(psi, theta_L, L, ctx, mask, 1, phi, 1.0))
    p_no = float(C.q_forward_pure(psi, theta_L, L, ctx, mask, 1, phi, 0.0))
    p_mix = float(C.q_forward_pure(psi, theta_L, L, ctx, mask, 1, phi, np.nan))
    assert abs(p_mix - (w1 * p_yes + (1 - w1) * p_no)) < TOL
    # observed-answer branches equal the explicit collapse + chain
    chain_yes = C.chain_pure(C.lueders(psi, p1), theta_L, L, ctx, mask)
    assert abs(p_yes - float(C.born(chain_yes, C.P_SELL))) < TOL


def test_q_forward_scenario_first_ignores_tolerance_inputs(rng):
    psi, theta_L, L, ctx, mask, _ = _random_case(rng)
    ps = [float(C.q_forward_pure(psi, theta_L, L, ctx, mask, 0, phi, a)) for phi in (0.0, 0.7) for a in (np.nan, 0.0, 1.0)]
    assert max(ps) - min(ps) < TOL


@pytest.mark.parametrize("order_flag,tol_answer", [(0, np.nan), (1, np.nan), (1, 0.0), (1, 1.0)])
def test_q_forward_rho_gamma_zero_equals_pure(rng, order_flag, tol_answer):
    for _ in range(5):
        psi, theta_L, L, ctx, mask, phi = _random_case(rng)
        p_pure = float(C.q_forward_pure(psi, theta_L, L, ctx, mask, order_flag, phi, tol_answer))
        p_rho = float(C.q_forward_rho(C.to_rho(psi), theta_L, L, ctx, mask, order_flag, phi, tol_answer, 0.0))
        assert abs(p_pure - p_rho) < TOL


@pytest.mark.parametrize("order_flag,tol_answer", [(0, np.nan), (1, np.nan), (1, 0.0), (1, 1.0)])
def test_q_forward_rho_large_gamma_equals_classical_markov_chain(rng, order_flag, tol_answer):
    for _ in range(5):
        psi, theta_L, L, ctx, mask, phi = _random_case(rng)
        p_rho = float(C.q_forward_rho(C.to_rho(psi), theta_L, L, ctx, mask, order_flag, phi, tol_answer, 1e3))
        p_ref = np_markov_forward(np.asarray(psi), np.asarray(theta_L), L, np.asarray(ctx), np.asarray(mask), order_flag, phi, tol_answer)
        assert abs(p_rho - p_ref) < TOL


def test_q_forward_rho_large_gamma_differs_from_pure_generically(rng):
    psi, theta_L, L, ctx, mask, phi = _random_case(rng)
    mask = jnp.array([True, True])
    p_pure = float(C.q_forward_pure(psi, theta_L, L, ctx, mask, 0, phi, np.nan))
    p_cl = float(C.q_forward_rho(C.to_rho(psi), theta_L, L, ctx, mask, 0, phi, np.nan, 1e3))
    assert abs(p_pure - p_cl) > 1e-3  # coherence matters for a generic chain


def test_batched_forward_matches_loop_and_jits(rng):
    n = 12
    psis = jnp.stack([rand_state(rng) for _ in range(n)])
    theta_L = rand_theta(rng)
    L = jnp.asarray(rng.choice(np.abs(D.LOSS_PCTS), size=n))
    ctx = jnp.asarray(rng.normal(size=(n, 2, 3)))
    mask = jnp.asarray(rng.integers(0, 2, size=(n, 2)).astype(bool))
    order = jnp.asarray(rng.integers(0, 2, size=n))
    tol = jnp.asarray(np.where(rng.integers(0, 2, size=n) == 1, rng.integers(0, 2, size=n).astype(float), np.nan))
    gamma = jnp.asarray(rng.uniform(0, 2, size=n))
    phi = 0.6
    batch = np.asarray(jax.jit(C.q_forward_pure_batch)(psis, theta_L, L, ctx, mask, order, phi, tol))
    batch_rho = np.asarray(jax.jit(C.q_forward_rho_batch)(jax.vmap(C.to_rho)(psis), theta_L, L, ctx, mask, order, phi, tol, gamma))
    for i in range(n):
        p = float(C.q_forward_pure(psis[i], theta_L, L[i], ctx[i], mask[i], order[i], phi, tol[i]))
        assert abs(batch[i] - p) < TOL
        r = float(C.q_forward_rho(C.to_rho(psis[i]), theta_L, L[i], ctx[i], mask[i], order[i], phi, tol[i], gamma[i]))
        assert abs(batch_rho[i] - r) < TOL
    assert np.all((batch >= 0) & (batch <= 1)) and np.all((batch_rho >= 0) & (batch_rho <= 1))


def test_gradients_are_finite(rng):
    psi_v = jnp.asarray(rng.normal(size=4))
    theta_L = rand_theta(rng)
    ctx = jnp.asarray(rng.normal(size=(2, 3)))
    mask = jnp.array([True, True])

    def f(v, tL, tc, phi):
        return C.q_forward_pure(C.prepare_state(v), tL, 0.2, tc, mask, 1, phi, np.nan)

    grads = jax.grad(f, argnums=(0, 1, 2, 3))(psi_v, theta_L, ctx, 0.5)
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g)))
    # gradient at theta = 0 is finite (expm, not the closed form with |theta| in it)
    g0 = jax.grad(lambda t: C.q_forward_pure(C.prepare_state(psi_v), t, 0.2, jnp.zeros((2, 3)), mask, 0, 0.5, np.nan))(jnp.zeros(3))
    assert np.all(np.isfinite(np.asarray(g0)))
    g_rho = jax.grad(lambda gm: C.q_forward_rho(C.to_rho(C.prepare_state(psi_v)), theta_L, 0.2, ctx, mask, 0, 0.5, np.nan, gm))(0.3)
    assert np.isfinite(float(g_rho))


def test_gather_context_thetas(rng):
    table = jnp.asarray(rng.normal(size=(5, 3)))
    idx = jnp.array([[0, 3], [4, -1], [-1, -1]])
    mask = idx >= 0
    out = np.asarray(C.gather_context_thetas(table, idx, mask))
    assert out.shape == (3, 2, 3)
    assert np.allclose(out[0, 0], np.asarray(table[0])) and np.allclose(out[0, 1], np.asarray(table[3]))
    assert np.allclose(out[1, 0], np.asarray(table[4])) and np.allclose(out[1, 1], 0)
    assert np.allclose(out[2], 0)


# ---------------------------------------------------------------------------------------------
# core: interference terms
# ---------------------------------------------------------------------------------------------


def test_ltp_interference_zero_for_commuting_projectors(rng):
    psi = rand_state(rng)
    # phi = 0: the tolerance question is the hold/sell question; with no chain the LTP holds
    d0 = float(C.ltp_interference(psi, jnp.zeros(3), 0.0, jnp.zeros((2, 3)), jnp.array([False, False]), 0.0))
    assert abs(d0) < TOL
    # ... and with a chain that is diagonal in the sell basis (sigma_z rotations only)
    z_only = jnp.array([0.0, 0.0, 1.3])
    ctx_z = jnp.array([[0.0, 0.0, -0.4], [0.0, 0.0, 2.2]])
    dz = float(C.ltp_interference(psi, z_only, 0.2, ctx_z, jnp.array([True, True]), 0.0))
    assert abs(dz) < TOL


def test_ltp_interference_nonzero_for_phi_quarter_pi(rng):
    psi = rand_state(rng)
    d = float(C.ltp_interference(psi, jnp.zeros(3), 0.0, jnp.zeros((2, 3)), jnp.array([False, False]), np.pi / 4))
    assert abs(d) > 1e-3
    # equals the definition written out
    chain = lambda s: float(C.born(s, C.P_SELL))  # noqa: E731
    p1 = C.tolerance_projector(np.pi / 4)
    p0 = C.IDENTITY - p1
    expected = chain(psi) - (float(C.born(psi, p1)) * chain(C.lueders(psi, p1)) + float(C.born(psi, p0)) * chain(C.lueders(psi, p0)))
    assert abs(d - expected) < TOL
    # closed form for the identity chain: 2 Re <psi|P_0 P_sell P_1|psi>
    v = np.asarray(psi)
    closed = 2 * np.real(v.conj() @ np.asarray(p0) @ np.asarray(C.P_SELL) @ np.asarray(p1) @ v)
    assert abs(d - closed) < TOL


def test_ltp_interference_rho_matches_pure_at_gamma_zero(rng):
    psi, theta_L, L, ctx, mask, phi = _random_case(rng)
    d_pure = float(C.ltp_interference(psi, theta_L, L, ctx, mask, phi))
    d_rho = float(C.ltp_interference_rho(C.to_rho(psi), theta_L, L, ctx, mask, phi, 0.0))
    assert abs(d_pure - d_rho) < TOL
    batch = np.asarray(C.ltp_interference_batch(jnp.stack([psi, psi]), theta_L, jnp.array([L, L]), jnp.stack([ctx, ctx]), jnp.stack([mask, mask]), phi))
    assert np.allclose(batch, d_pure, atol=TOL)


def test_mixture_interference_zero_for_single_or_identical_contexts(rng):
    psi = rand_state(rng)
    t = rand_theta(rng)
    assert abs(float(C.mixture_interference(psi, t[None, :], jnp.array([1.0])))) < TOL
    assert abs(float(C.mixture_interference(psi, jnp.stack([t, t, t]), jnp.array([0.2, 0.3, 0.5])))) < TOL


def test_mixture_interference_nonzero_generic_and_matches_definition(rng):
    psi = rand_state(rng)
    thetas = jnp.stack([rand_theta(rng), rand_theta(rng)])
    w = jnp.array([0.4, 0.6])
    d = float(C.mixture_interference(psi, thetas, w))
    assert abs(d) > 1e-4
    us = [np.asarray(C.unitary(t)) for t in thetas]
    v = np.asarray(psi)
    sup = sum(np.sqrt(float(wc)) * (u @ v) for wc, u in zip(w, us))
    sup = sup / np.linalg.norm(sup)
    expected = abs(sup[1]) ** 2 - sum(float(wc) * abs((u @ v)[1]) ** 2 for wc, u in zip(w, us))
    assert abs(d - expected) < TOL


def test_order_effect_zero_for_commuting_generators(rng):
    psi = rand_state(rng)
    t1 = jnp.array([0.0, 0.9, 0.0])
    t2 = jnp.array([0.0, -1.7, 0.0])
    assert abs(float(C.order_effect(psi, t1, t2))) < TOL
    assert abs(float(C.order_effect(psi, t1, t2, theta_L=rand_theta(rng), L=0.2))) < TOL
    assert abs(float(C.order_effect_rho(psi, t1, t2, gamma=0.0))) < TOL
    # dephasing between the two steps does not commute with sigma_y rotations, so the open-system
    # order effect of commuting unitaries is zero only when they also commute with the dephasing
    # (sigma_z rotations), for any gamma
    assert abs(float(C.order_effect_rho(psi, t1, t2, gamma=0.4))) > 1e-4
    z1, z2 = jnp.array([0.0, 0.0, 0.8]), jnp.array([0.0, 0.0, -2.1])
    for gamma in (0.0, 0.4, 1e3):
        assert abs(float(C.order_effect_rho(psi, z1, z2, gamma=gamma, theta_L=rand_theta(rng), L=0.1))) < TOL
    # parallel generators along any common axis commute too
    axis = rng.normal(size=3)
    assert abs(float(C.order_effect(psi, jnp.asarray(0.7 * axis), jnp.asarray(-1.1 * axis)))) < TOL


def test_order_effect_nonzero_generic_and_antisymmetric(rng):
    psi = rand_state(rng)
    t1, t2 = rand_theta(rng), rand_theta(rng)
    d = float(C.order_effect(psi, t1, t2))
    assert abs(d) > 1e-3
    assert abs(d + float(C.order_effect(psi, t2, t1))) < TOL
    u1, u2 = np.asarray(C.unitary(t1)), np.asarray(C.unitary(t2))
    v = np.asarray(psi)
    expected = abs((u2 @ u1 @ v)[1]) ** 2 - abs((u1 @ u2 @ v)[1]) ** 2
    assert abs(d - expected) < TOL


# ---------------------------------------------------------------------------------------------
# data: ModelData from the full design
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def design_data():
    rng = np.random.default_rng(0)
    df = design_frame(rng, n_subjects=3, with_tolerance_rows=True)
    return df, build_model_data(df)


def test_design_frame_is_valid(design_data):
    df, _ = design_data
    assert S.validate_frame(df, strict=False).ok
    assert len(df) == 3 * 2 * D.N_ITEMS


def test_model_data_shapes_and_subjects(design_data):
    df, data = design_data
    n = len(df)
    assert data.n == n and data.n_subjects == 3
    assert list(data.subject_ids) == ["s000", "s001", "s002"]
    assert data.subject_idx.shape == (n,) and data.subject_idx.dtype == np.int64
    assert data.X.shape == (3, len(data.X_columns))
    for name, shape in [("loss", (n,)), ("loss_signed", (n,)), ("ctx_idx", (n, 2)), ("ctx_mask", (n, 2)), ("order_flag", (n,)), ("tol_answer", (n,)), ("y", (n,)), ("w", (n,)), ("row_kind", (n,)), ("position", (n,)), ("is_synthetic", (n,)), ("index", (n,))]:
        assert getattr(data, name).shape == shape, name
    assert data.ctx_vocab == tuple(D.NONNULL_CONTEXTS)
    assert set(data.row_kind) == {"sell", "tol"}
    assert data.sell_rows().shape == (3 * D.N_ITEMS,)
    assert np.array_equal(np.asarray(df["subject_id"].astype(object).to_numpy()), data.subject_ids[data.subject_idx])
    assert data.is_synthetic.all()
    assert np.array_equal(data.index, np.asarray(df.index))
    assert np.all(data.w == 1.0)


def test_model_data_loss_and_contexts_match_frame(design_data):
    df, data = design_data
    lp = df["loss_pct"].to_numpy(dtype=float)
    assert np.allclose(data.loss, np.where(np.isnan(lp), 0.0, np.abs(lp)))
    assert np.allclose(data.loss_signed, np.where(np.isnan(lp), 0.0, lp))
    for i, tags in enumerate(df["context_tags"].tolist()):
        got = [data.ctx_vocab[j] for j in data.ctx_idx[i] if j >= 0]
        assert got == list(tags)
        assert data.ctx_mask[i].tolist() == [k < len(tags) for k in range(2)]
    # every one of the 17 conditions appears with the right number of tags
    n_tags = data.ctx_mask.sum(axis=1)
    sell = data.sell_mask()
    assert np.bincount(n_tags[sell], minlength=3).tolist() == [3 * 10, 3 * 40, 3 * 120]


def test_model_data_order_flag_and_tol_answer_join(design_data):
    df, data = design_data
    qo = df["question_order_id"].astype(object).to_numpy()
    assert np.array_equal(data.order_flag, (qo == D.TOLERANCE_FIRST).astype(np.int64))
    sell = data.sell_mask()
    tf = sell & (data.order_flag == 1)
    sf = sell & (data.order_flag == 0)
    assert tf.sum() == 3 * D.N_ITEMS // 2 and sf.sum() == 3 * D.N_ITEMS // 2
    # every tolerance-first sell row has its answer; scenario-first rows and tolerance rows do not
    assert np.isfinite(data.tol_answer[tf]).all()
    assert np.isnan(data.tol_answer[sf]).all()
    assert np.isnan(data.tol_answer[data.row_kind == "tol"]).all()
    # the joined value is the response of the row just before it in the same session
    by_key = {(r.subject_id, r.session_id, int(r.position_in_session)): r for r in df.itertuples(index=False)}
    for i in np.flatnonzero(tf):
        r = df.iloc[i]
        prev = by_key[(r.subject_id, r.session_id, int(r.position_in_session) - 1)]
        assert prev.scenario_id == D.TOLERANCE_QUESTION_ID
        assert data.tol_answer[i] == prev.response
    assert set(np.unique(data.tol_answer[tf]).tolist()) <= {0.0, 1.0}


def test_tol_answer_nan_when_tolerance_rows_absent(rng):
    df = design_frame(rng, n_subjects=1, with_tolerance_rows=False)
    data = build_model_data(df)
    assert data.n == D.N_ITEMS
    assert np.isnan(data.tol_answer).all()
    assert data.order_flag.sum() == D.N_ITEMS // 2
    assert set(data.row_kind) == {"sell"}


def test_model_data_covariate_matrix_columns_and_missing(design_data):
    df, data = design_data
    cols = data.X_columns
    assert cols == MD.covariate_columns()
    assert "age_band=18-29" in cols and "invest_experience_yrs" in cols and "education__missing" in cols
    # subject s001 has only age_band -> missing indicators for the other five, one-hot for age
    i1 = list(data.subject_ids).index("s001")
    row = dict(zip(cols, data.X[i1]))
    for f in D.COVARIATE_KEYS:
        assert row[f + MD.MISSING_SUFFIX] == (0.0 if f == "age_band" else 1.0)
    assert sum(row[f"age_band={lvl}"] for lvl in D.COVARIATE_SPEC["age_band"].levels) == 1.0
    assert row["invest_experience_yrs"] == 0.0
    # subjects with full covariates: no missing flags; numerics z-scored over the present values
    covs = [json.loads(df[df["subject_id"] == s]["covariates"].iloc[0]) for s in data.subject_ids]
    for name in ("invest_experience_yrs", "self_reported_risk_tolerance", "financial_literacy_score"):
        vals = np.array([c[name] for c in covs if name in c], dtype=float)
        mean, std = data.covariate_stats[name]
        assert abs(mean - vals.mean()) < 1e-12
        j = cols.index(name)
        present = [i for i, c in enumerate(covs) if name in c]
        assert np.allclose(data.X[present, j], (vals - mean) / std)
    assert data.X.dtype == np.float64


def test_covariate_matrix_reuses_stats_and_handles_unknown_level():
    recs = [{"invest_experience_yrs": 10, "age_band": "30-44"}, {"invest_experience_yrs": 20, "age_band": "nonsense"}]
    X, cols, stats = MD.covariate_matrix(recs)
    assert stats["invest_experience_yrs"] == (15.0, 5.0)
    X2, _, stats2 = MD.covariate_matrix([{"invest_experience_yrs": 25}], stats=stats)
    assert stats2["invest_experience_yrs"] == stats["invest_experience_yrs"]
    assert X2[0, cols.index("invest_experience_yrs")] == 2.0
    assert X[1, cols.index("age_band__missing")] == 1.0 and X[0, cols.index("age_band=30-44")] == 1.0
    # constant numeric column: std guard -> zeros, not NaN
    X3, _, stats3 = MD.covariate_matrix([{"financial_literacy_score": 3}, {"financial_literacy_score": 3}])
    assert stats3["financial_literacy_score"][1] == 1.0
    assert np.all(np.isfinite(X3))


# ---------------------------------------------------------------------------------------------
# data: hand-built frames (weights, rates, unknown tags, edge cases)
# ---------------------------------------------------------------------------------------------


def hand_frame() -> pd.DataFrame:
    """Two subjects in one real dataset: tolerance-first and scenario-first items, an aggregate
    ``choice_rate`` row with a weight, a ``likert`` row, a tag outside the design, a null loss."""
    ev = [
        # subject a, session 1: tolerance (pos 0) then sell (pos 1) -> tolerance-first
        _event(subject_id="a", dataset="real_x", session_id="1", position_in_session=0, scenario_id=D.TOLERANCE_QUESTION_ID, loss_pct=None, horizon_days=None, elicitation_type="binary_yes_no", response=1.0, question_order_id=D.TOLERANCE_FIRST, is_synthetic=False, covariates={"age_band": "60+", "invest_experience_yrs": 12}),
        _event(subject_id="a", dataset="real_x", session_id="1", position_in_session=1, scenario_id="L20|rec>friend", loss_pct=-0.20, context_tags=["news:recession", "social:friend_sells"], question_order_id=D.TOLERANCE_FIRST, prior_question_ids=[D.TOLERANCE_QUESTION_ID], response=1.0, is_synthetic=False, covariates={"age_band": "60+", "invest_experience_yrs": 12}),
        # subject a, session 1: scenario-first item (sell at 2, tolerance at 3), then a sell row at 4
        # immediately after that tolerance row but scenario-first -> tol_answer stays NaN
        _event(subject_id="a", dataset="real_x", session_id="1", position_in_session=2, scenario_id="L05|tech", loss_pct=-0.05, context_tags=["news:technical"], question_order_id=D.SCENARIO_FIRST, response=0.0, is_synthetic=False, covariates={"age_band": "60+", "invest_experience_yrs": 12}),
        _event(subject_id="a", dataset="real_x", session_id="1", position_in_session=3, scenario_id=D.TOLERANCE_QUESTION_ID, loss_pct=None, horizon_days=None, elicitation_type="binary_yes_no", response=0.0, question_order_id=D.SCENARIO_FIRST, prior_question_ids=["L05|tech"], is_synthetic=False, covariates={"age_band": "60+", "invest_experience_yrs": 12}),
        _event(subject_id="a", dataset="real_x", session_id="1", position_in_session=4, scenario_id="L10|none", loss_pct=-0.10, question_order_id=D.SCENARIO_FIRST, response=1.0, is_synthetic=False, covariates={"age_band": "60+", "invest_experience_yrs": 12}),
        # subject a, session 2: tolerance-first sell row whose tolerance row is missing -> NaN
        _event(subject_id="a", dataset="real_x", session_id="2", position_in_session=5, scenario_id="L30|recov", loss_pct=-0.30, context_tags=["market:recovered_5pct"], question_order_id=D.TOLERANCE_FIRST, prior_question_ids=[D.TOLERANCE_QUESTION_ID], response=0.0, is_synthetic=False, covariates={"age_band": "60+", "invest_experience_yrs": 12}),
        # subject b: aggregate rate row with weight and an external tag; a likert row; null order id
        _event(subject_id="b", dataset="real_x", session_id="1", position_in_session=0, scenario_id="p17", loss_pct=0.25, context_tags=["exp:loss"], question_order_id=None, elicitation_type="choice_rate", response=0.37, battery_version="ext-1", is_synthetic=False, covariates={"weight": 40, "x_problem": 17}),
        _event(subject_id="b", dataset="real_x", session_id="1", position_in_session=1, scenario_id="q1", loss_pct=None, horizon_days=None, question_order_id=None, elicitation_type="likert", response=5.0, battery_version="ext-1", is_synthetic=False, covariates={}),
        # subject b: null order id but the tolerance question listed as prior -> order_flag = 1
        _event(subject_id="b", dataset="real_x", session_id="1", position_in_session=2, scenario_id=D.TOLERANCE_QUESTION_ID, loss_pct=None, horizon_days=None, elicitation_type="binary_yes_no", response=1.0, question_order_id=None, battery_version="ext-1", is_synthetic=False, covariates={}),
        _event(subject_id="b", dataset="real_x", session_id="1", position_in_session=3, scenario_id="L15|none", loss_pct=-0.15, question_order_id=None, prior_question_ids=[D.TOLERANCE_QUESTION_ID], response=0.0, battery_version="ext-1", is_synthetic=False, covariates={}),
    ]
    return S.events_to_frame(ev)


def test_hand_frame_row_kinds_weights_and_sell_rows():
    df = hand_frame()
    data = build_model_data(df)
    assert data.row_kind.tolist() == ["tol", "sell", "sell", "tol", "sell", "sell", "rate", "other", "tol", "sell"]
    assert data.sell_rows().tolist() == [1, 2, 4, 5, 6, 9]
    assert data.w.tolist() == [1, 1, 1, 1, 1, 1, 40, 1, 1, 1]
    assert data.y[6] == 0.37 and data.y[7] == 5.0
    assert not data.is_synthetic.any()
    assert data.summary()["row_kinds"] == {"other": 1, "rate": 1, "sell": 5, "tol": 3}


def test_hand_frame_order_flag_and_tol_answer_rules():
    data = build_model_data(hand_frame())
    assert data.order_flag.tolist() == [1, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    ta = data.tol_answer
    assert ta[1] == 1.0  # tolerance-first with the tolerance row at position 0
    assert np.isnan(ta[4])  # scenario-first: the preceding tolerance row belongs to the previous item
    assert np.isnan(ta[5])  # tolerance-first but the tolerance row is missing (other session)
    assert ta[9] == 1.0  # order recovered from prior_question_ids, answer at position 2
    assert np.isnan(ta[[0, 2, 3, 6, 7, 8]]).all()


def test_hand_frame_contexts_extend_vocab_and_signed_loss():
    data = build_model_data(hand_frame())
    assert data.ctx_vocab == tuple(D.NONNULL_CONTEXTS) + ("exp:loss",)
    assert data.ctx_idx[6].tolist() == [4, -1] and data.ctx_mask[6].tolist() == [True, False]
    assert data.ctx_idx[1].tolist() == [0, 2]
    assert data.loss[6] == 0.25 and data.loss_signed[6] == 0.25
    assert data.loss[1] == 0.20 and data.loss_signed[1] == -0.20
    assert data.loss[0] == 0.0 and data.loss_signed[7] == 0.0
    fixed = build_model_data(hand_frame(), ctx_vocab=data.ctx_vocab)
    assert fixed.ctx_vocab == data.ctx_vocab
    with pytest.raises(ValueError, match="not in the given ctx_vocab"):
        build_model_data(hand_frame(), ctx_vocab=tuple(D.NONNULL_CONTEXTS))
    with pytest.raises(ValueError, match="must start with"):
        build_model_data(hand_frame(), ctx_vocab=("exp:loss",))


def test_build_model_data_sell_types_option_and_subset():
    df = hand_frame()
    data = build_model_data(df, sell_types=("binary_sell", "likert"))
    assert data.row_kind[7] == "sell"
    sub = data.subset(data.sell_rows())
    assert sub.n == 7 and sub.n_subjects == data.n_subjects
    assert sub.X.shape == data.X.shape and sub.ctx_vocab == data.ctx_vocab
    assert np.array_equal(sub.index, data.index[data.sell_rows()])
    mask_sub = data.subset(data.subject_idx == 0)
    assert mask_sub.n == 6 and set(mask_sub.subject_idx.tolist()) == {0}
    j = data.jax()
    assert j["y"].shape == (data.n,) and bool(j["sell_mask"][7])


def test_build_model_data_edge_cases(rng):
    df = hand_frame()
    # consent filter
    df2 = df.copy()
    df2.loc[df2.index[7], "consent_training"] = False
    assert build_model_data(df2).n == len(df) - 1
    assert build_model_data(df2, require_consent=False).n == len(df)
    # subject id collision across datasets
    df3 = df.copy()
    df3.loc[df3.index[2:], "dataset"] = "real_y"  # subject a now appears in real_x and real_y
    with pytest.raises(ValueError, match="more than one dataset"):
        build_model_data(df3)
    keyed = build_model_data(df3, subject_key="dataset/subject_id")
    assert keyed.subject_ids.tolist() == ["real_x/a", "real_y/a", "real_y/b"]
    assert keyed.subject_idx.tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    # too many contexts
    df4 = df.copy()
    df4.at[df4.index[1], "context_tags"] = ["news:recession", "social:friend_sells", "news:technical"]
    with pytest.raises(ValueError, match="context tags"):
        build_model_data(df4)
    assert build_model_data(df4, n_ctx_positions=3).ctx_idx.shape == (len(df), 3)
    # missing columns
    with pytest.raises(S.SchemaError):
        build_model_data(df.drop(columns=["response"]))
    # validate=True on a valid frame passes
    assert build_model_data(df, validate=True).n == len(df)


# ---------------------------------------------------------------------------------------------
# base: helpers, Encoder, features, model contract
# ---------------------------------------------------------------------------------------------


def test_count_params_counts_real_scalars_and_complex_twice():
    params = {"W": jnp.zeros((3, 4)), "b": jnp.zeros(4), "nested": {"c": jnp.zeros(2, dtype=jnp.complex128), "s": jnp.zeros(())}}
    assert count_params(params) == 12 + 4 + 4 + 1
    assert count_params(params, exclude=("W",)) == 9
    assert count_params([jnp.ones(3), (jnp.ones(2),)]) == 5


def test_bernoulli_log_lik_convention():
    p = jnp.array([0.8, 0.8, 0.5, 0.9])
    y = jnp.array([1.0, 0.0, 0.37, 1.0])
    w = jnp.array([1.0, 1.0, 40.0, 1.0])
    mask = jnp.array([True, True, True, False])
    ll = np.asarray(bernoulli_log_lik(p, y, w, mask))
    assert abs(ll[0] - np.log(0.8)) < TOL
    assert abs(ll[1] - np.log(0.2)) < TOL
    assert abs(ll[2] - 40 * (0.37 * np.log(0.5) + 0.63 * np.log(0.5))) < TOL
    assert ll[3] == 0.0
    # clipping keeps p = 0 / 1 finite
    assert np.all(np.isfinite(np.asarray(bernoulli_log_lik(jnp.array([0.0, 1.0]), jnp.array([1.0, 0.0]), jnp.ones(2), jnp.ones(2, bool)))))


def test_encoder_shapes_restarts_and_apply(design_data):
    _, data = design_data
    enc = Encoder.for_data(data, d_out=4)
    key = jax.random.PRNGKey(0)
    p0, p1, p0b = enc.init(key, 0), enc.init(key, 1), enc.init(key, 0)
    assert p0["W"].shape == (data.d_x, 4) and p0["b"].shape == (4,)
    assert p0["u"].shape == (data.n_subjects, 4) and p0["log_sigma_u"].shape == (4,)
    assert all(a.dtype == jnp.float64 for a in jax.tree_util.tree_leaves(p0))
    assert not np.allclose(np.asarray(p0["W"]), np.asarray(p1["W"]))  # restarts differ
    assert np.array_equal(np.asarray(p0["W"]), np.asarray(p0b["W"]))  # and are reproducible
    z = enc.apply(p0, data.X, data.subject_idx)
    assert z.shape == (data.n, 4)
    zs = enc.apply_subjects(p0, data.X)
    assert np.allclose(np.asarray(z), np.asarray(zs)[data.subject_idx])
    assert np.allclose(np.asarray(zs), np.asarray(data.X @ p0["W"] + p0["b"] + p0["u"]))
    assert enc.n_params(p0) == count_params(p0)
    assert enc.n_params(p0, include_random_effects=False) == count_params(p0) - data.n_subjects * 4
    psi = jax.vmap(C.prepare_state)(z)
    assert psi.shape == (data.n, 2)
    assert np.allclose(np.asarray(jax.vmap(lambda s: C.born(s, C.P_SELL) + C.born(s, C.P_HOLD))(psi)), 1.0, atol=TOL)


def test_encoder_log_prior_penalizes_random_effects_and_is_bounded():
    enc = Encoder(d_x=3, d_out=2, n_subjects=5)
    p = enc.init(jax.random.PRNGKey(1), 0)
    lp0 = float(encoder_log_prior(p))
    assert np.isfinite(lp0)
    p_big = dict(p, u=jnp.ones((5, 2)) * 3.0)
    assert float(encoder_log_prior(p_big)) < lp0
    # fitted scale: shrinking sigma to zero with u = 0 must not diverge to +inf (hyperprior)
    vals = [float(encoder_log_prior(dict(p, log_sigma_u=jnp.full((2,), s)))) for s in (0.0, -2.0, -5.0, -10.0)]
    assert all(np.isfinite(v) for v in vals) and vals[-1] < vals[0]
    # ridge on W lowers the value
    assert float(enc.log_prior(p, w_l2=1.0)) < lp0
    # the log-prior is differentiable in all parameters
    g = jax.grad(encoder_log_prior)(p)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(g))


def test_standard_features_columns_and_values(design_data):
    _, data = design_data
    fm = standard_features(data)
    n, K, V = data.n, data.n_ctx_positions, data.n_contexts
    assert fm.F.shape == (n, data.d_x + 1 + K * V + 1 + 2)
    assert fm.columns[: data.d_x] == data.X_columns
    assert fm.columns[data.d_x] == "loss"
    ctx_cols = [f"ctx{k}={t}" for k in range(K) for t in data.ctx_vocab]
    assert list(fm.columns[data.d_x + 1 : data.d_x + 1 + K * V]) == ctx_cols
    assert fm.columns[-3:] == ("order_flag", "tol_answered", "tol_yes")
    assert np.allclose(fm.column("loss"), data.loss)
    assert np.allclose(fm.column("order_flag"), data.order_flag)
    assert np.allclose(fm.column("tol_answered"), np.isfinite(data.tol_answer))
    assert np.allclose(fm.column("tol_yes"), np.nan_to_num(data.tol_answer) == 1.0)
    assert np.allclose(fm.F[:, : data.d_x], data.X[data.subject_idx])
    ctx_block = fm.F[:, data.d_x + 1 : data.d_x + 1 + K * V]
    assert np.array_equal(ctx_block.sum(axis=1), data.ctx_mask.sum(axis=1))
    for i in range(0, n, 37):
        for k in range(K):
            if data.ctx_mask[i, k]:
                assert fm.column(f"ctx{k}={data.ctx_vocab[data.ctx_idx[i, k]]}")[i] == 1.0
    signed = standard_features(data, loss_column="loss_signed")
    assert signed.columns[data.d_x] == "loss_signed" and np.allclose(signed.column("loss_signed"), data.loss_signed)
    assert fm.F.dtype == np.float64


class _StaticBornModel(QuantumModelBase):
    """Toy Q-model for the contract test: P(sell) = ||P_SELL psi_i||^2, psi_i from the encoder."""

    name = "toy-static-born"

    def __init__(self, data: ModelData):
        self.enc = Encoder.for_data(data, d_out=4)

    def init_params(self, rng_key, data, restart=0):
        return {"enc": self.enc.init(rng_key, restart)}

    def predict_proba(self, params, data):
        psi = jax.vmap(C.prepare_state)(self.enc.apply(params["enc"], data.X, data.subject_idx))
        return jax.vmap(C.born, in_axes=(0, None))(psi, C.P_SELL)

    def log_prior(self, params):
        return self.enc.log_prior(params["enc"])

    def interference_terms(self, params, data):
        n = data.n
        return {"ltp": jnp.zeros(n), "order": jnp.full(n, jnp.nan)}


def test_model_contract_with_toy_model(design_data):
    _, data = design_data
    model = _StaticBornModel(data)
    assert isinstance(model, Model) and isinstance(model, ModelBase)
    assert model.family == "quantum" and model.name == "toy-static-born"
    params = model.init_params(jax.random.PRNGKey(0), data, restart=2)
    p = np.asarray(model.predict_proba(params, data))
    assert p.shape == (data.n,) and np.all((p >= 0) & (p <= 1))
    ll = np.asarray(model.log_lik(params, data))
    assert ll.shape == (data.n,)
    assert np.all(ll[~data.sell_mask()] == 0.0)
    assert np.all(ll[data.sell_mask()] < 0.0)
    obj = float(model.objective(params, data))
    assert np.isfinite(obj) and obj > 0
    assert model.n_params(params) == count_params(params)
    assert model.subject_summary(params, data) is None
    terms = model.interference_terms(params, data)
    assert set(terms) >= {"ltp", "order"} and terms["ltp"].shape == (data.n,)
    g = jax.grad(lambda q: model.objective(q, data))(params)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(g))
    assert not isinstance(object(), Model)


def test_full_q2_style_forward_on_model_data_jits(design_data):
    """A Q2-shaped forward pass on the design's ModelData through the batched core helpers."""
    _, data = design_data
    enc = Encoder.for_data(data, d_out=4)
    key = jax.random.PRNGKey(3)
    params = {
        "enc": enc.init(key, 0),
        "theta_L": jnp.array([0.5, -0.3, 0.1]),
        "theta_ctx": jax.random.normal(key, (data.n_contexts, 3), dtype=jnp.float64),
        "phi": jnp.asarray(0.4),
    }
    arrays = data.jax()

    @jax.jit
    def forward(params):
        psi = jax.vmap(C.prepare_state)(enc.apply(params["enc"], arrays["X"], arrays["subject_idx"]))
        ctx_thetas = C.gather_context_thetas(params["theta_ctx"], arrays["ctx_idx"], arrays["ctx_mask"])
        return C.q_forward_pure_batch(psi, params["theta_L"], arrays["loss"], ctx_thetas, arrays["ctx_mask"], arrays["order_flag"], params["phi"], arrays["tol_answer"])

    p = np.asarray(forward(params))
    assert p.shape == (data.n,) and np.all((p >= 0) & (p <= 1)) and np.all(np.isfinite(p))
    # the none condition with L > 0 still rotates by U_L; same subject, same L, same order -> same p
    ll = np.asarray(bernoulli_log_lik(p, arrays["y"], arrays["w"], arrays["sell_mask"]))
    assert np.all(ll[~data.sell_mask()] == 0.0) and np.isfinite(ll.sum())
