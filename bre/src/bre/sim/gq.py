"""G_Q — the quantum-probability generator of the Phase 2 recovery study (PLAN.md section 5).

Every table produced here is synthetic (``is_synthetic = True`` on every row) and every number

# Population-default rationale (2026-09-29): the loss unitary rotates about the y axis of the Bloch
# sphere (theta_L = (0, -1.5, 0)) and the state prior b keeps subjects hold-leaning, so that the
# no-context sell probability rises with the loss for most subjects (probe on 150 subjects: mean
# P(sell) 0.34 at -5% to 0.58 at -30%, 92% monotone). The earlier defaults rotated about an axis
# nearly parallel to the typical state, leaving the loss response flat and theta_L unidentified.
# Synthetic design choice, not fitted to data.
in :data:`POPULATION_DEFAULTS` is a labelled synthetic design choice (``SOURCE``), chosen so
that the generator exercises all the structure the Q-models can represent; none of it is fitted
to, or quoted from, any dataset. :func:`population_from_fit` maps fitted Q2/Q4 parameters to a
population dict so that later public-data fits can replace the defaults with provenance.

Generative process (the primitives are those of ``bre.models.quantum.core``, so the generator
and the Q4 model share one code path: :func:`core.q_forward_rho_batch`, :func:`core.chain_rho`,
:func:`core.lueders_rho`, :func:`core.born_rho`).

Per subject ``i`` with standardized covariate row ``x_i`` (:func:`bre.models.data.covariate_matrix`,
the same design matrix ``ModelData.X`` the models read):

* state pre-image ``v_i = W^T x_i + b + eps_i``, ``eps_i ~ N(0, sigma_v^2 I_4)``, and
  ``psi_i = prepare_state(v_i)`` (unit vector in C^2, hold amplitude real, PLAN.md section 4);
* context parameters ``theta_{c,i} ~ N(mu_c, diag(sigma_c^2))`` for each of the four non-null
  design contexts (3-vectors in the su(2) basis; ``theta_none = 0`` is the reference gauge);
* decoherence rate ``gamma_i ~ LogNormal(mu_gamma, sigma_gamma)`` spanning near-0 (fully
  coherent, "quantum") to large (classical Markov chain); with ``mixed_population=False`` every
  ``gamma_i = 0`` and the data are exactly Q2-distributed (up to the per-subject spread of
  ``theta_{c,i}``, which Q2 summarizes by its population table);
* the population loss rotation ``theta_L`` (``U_L(L) = U(L theta_L)``) and the tolerance angle
  ``phi`` (``P_yes = |yes><yes|``, ``|yes> = (cos phi, sin phi)``) are shared by all subjects.

Per design item (``bre.design.full_design``: loss ``L``, ordered context tuple, question order),
with ``rho_i = |psi_i><psi_i|`` prepared afresh for every item:

* tolerance-first: the tolerance answer ``a ~ Bernoulli(Tr(P_yes rho_i))`` is drawn first; the
  state collapses by the Lüders rule ``rho_a = P_a rho_i P_a / Tr(P_a rho_i)``; then the chain
  ``U_L(L)`` and the context unitaries in presentation order act, with dephasing at rate
  ``gamma_i`` after ``U_L`` and after every context (:func:`core.chain_rho`), and
  ``sell ~ Bernoulli(Tr(P_sell rho'))``. In one call this is
  ``core.q_forward_rho(rho_i, theta_L, L, theta_{c,i}, mask, order_flag=1, phi, a, gamma_i)``.
* scenario-first: ``sell ~ Bernoulli(Tr(P_sell chain_rho(rho_i, ...)))`` without any tolerance
  measurement (``order_flag=0``); the state then collapses on the observed sell/hold outcome and
  the tolerance answer is drawn from it: ``a ~ Bernoulli(Tr(P_yes lueders_rho(rho', P_sell or
  P_hold)))`` (which equals ``sin^2 phi`` after "sell" and ``cos^2 phi`` after "hold").

Rows are written through :func:`bre.sim.common.assemble_events` (two rows per item in the
item's question order, the layout ``bre.models.data.build_model_data`` reads back) and the truth
dict records every latent quantity: the population, ``covariate_stats``, per-subject ``v_i``,
``psi_i`` (real/imaginary parts and the Bloch angles), ``theta_{c,i}``, ``gamma_i``, and per item
the probabilities that were actually used to draw the two answers together with the answers.

Identifiability notes for whoever fits this data (see also ``q2_context_unitary``): the sell
likelihood is invariant under complex conjugation of everything
(``b -> -b, theta -> (-theta_x, theta_y, -theta_z)``) and under conjugation by ``sigma_z``
(``b -> b + pi, theta -> (-theta_x, -theta_y, theta_z), phi -> -phi``), so fitted context
parameters are compared with the truth up to these four sign patterns and modulo ``pi`` along
their direction.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bre import design as _design
from bre.models.data import covariate_columns, covariate_matrix
from bre.models.quantum import core as C
from bre.sim import common

NAME = "gq"
"""Generator id (``bre.sim.GENERATOR_NAMES``)."""

DATASET = "synthetic_gq"
"""``dataset`` column of every generated row."""

SESSION_ID = "full"
"""``session_id`` of every generated row: one session holding the whole design."""

SOURCE = "synthetic default, not fitted to data"
"""Provenance label carried by :data:`POPULATION_DEFAULTS` (CLAUDE.md rule 3)."""

STATE_DIMS: tuple[str, ...] = ("re_hold", "re_sell", "im_hold", "im_sell")
"""Meaning of the four components of the state pre-image ``v`` fed to ``prepare_state``."""

D_STATE = len(STATE_DIMS)

# ---------------------------------------------------------------------------------------------
# Population defaults (synthetic design choices, labelled)
# ---------------------------------------------------------------------------------------------

_W_NAMED: dict[str, tuple[float, float, float, float]] = {
    # column name -> effect on (re_hold, re_sell, im_hold, im_sell). Signs and sizes are design
    # choices that make the covariate map non-trivial so encoder recovery can be tested; they
    # are not empirical claims about investors.
    "self_reported_risk_tolerance": (0.0, -0.25, 0.0, -0.10),
    "invest_experience_yrs": (0.0, -0.15, 0.0, 0.05),
    "financial_literacy_score": (0.0, -0.10, 0.0, 0.0),
    "age_band=60+": (0.0, 0.20, 0.0, 0.0),
    "age_band=18-29": (0.0, -0.10, 0.0, 0.10),
    "wealth_band=<50k": (0.0, 0.15, 0.0, 0.0),
    "wealth_band=>1M": (0.0, -0.15, 0.0, 0.0),
    "education=graduate": (0.0, -0.05, 0.0, 0.05),
}


def _default_W(columns: tuple[str, ...]) -> np.ndarray:
    W = np.zeros((len(columns), D_STATE), dtype=np.float64)
    for name, row in _W_NAMED.items():
        if name not in columns:
            raise KeyError(f"POPULATION_DEFAULTS names covariate column {name!r} that covariate_columns() lacks")
        W[columns.index(name)] = row
    return W


_COLUMNS = covariate_columns()

POPULATION_DEFAULTS: dict[str, Any] = {
    "SOURCE": SOURCE,
    "covariate_columns": tuple(_COLUMNS),
    "W": _default_W(_COLUMNS),
    "b": np.array([1.0, 0.45, 0.0, 0.15]),
    "sigma_v": np.array([0.3, 0.3, 0.3, 0.3]),
    "theta_L": np.array([0.0, -1.5, 0.0]),
    "contexts": tuple(_design.NONNULL_CONTEXTS),
    "mu_c": np.array(
        [
            [0.6, 0.2, 0.3],  # news:recession
            [-0.3, 0.4, 0.2],  # news:technical
            [0.4, -0.5, 0.1],  # social:friend_sells
            [-0.5, -0.2, 0.4],  # market:recovered_5pct
        ]
    ),
    "sigma_c": np.full((4, 3), 0.15),
    "phi": 0.55,
    "mu_gamma": float(np.log(0.3)),
    "sigma_gamma": 1.5,
}
"""Population parameters of G_Q. Every entry is a synthetic design choice (``SOURCE``):

* ``W (d_x, 4)``, ``b (4,)``, ``sigma_v (4,)``: the covariate-to-state map and its noise; the
  baseline state ``prepare_state(b)`` has ``P(sell | L = 0) = |psi_1|^2 ~ 0.27``;
* ``theta_L (3,)``: loss rotation, ``|L theta_L| ~ 0.5`` at the largest design loss (30%);
* ``mu_c (4, 3)``, ``sigma_c (4, 3)``: context means (generic, pairwise non-commuting
  directions so that order effects exist) and their per-subject spread, rows in ``contexts``
  order (``bre.design.NONNULL_CONTEXTS``);
* ``phi``: tolerance angle (non-commuting with the sell question, so LTP interference exists);
* ``mu_gamma``, ``sigma_gamma``: the LogNormal of the per-subject dephasing rate
  (median 0.3; 5th-95th percentiles about 0.025-3.6, i.e. per-step coherence factors
  ``exp(-2 gamma)`` from ~0.95 down to ~0.001).
"""

_ARRAY_KEYS: dict[str, tuple[int, ...]] = {
    "W": (len(_COLUMNS), D_STATE),
    "b": (D_STATE,),
    "sigma_v": (D_STATE,),
    "theta_L": (3,),
    "mu_c": (len(_design.NONNULL_CONTEXTS), 3),
    "sigma_c": (len(_design.NONNULL_CONTEXTS), 3),
}
_SCALAR_KEYS: tuple[str, ...] = ("phi", "mu_gamma", "sigma_gamma")


def resolve_population(population: dict[str, Any] | None = None) -> dict[str, Any]:
    """:data:`POPULATION_DEFAULTS` updated with ``population`` (partial overrides allowed),
    arrays coerced to float64 with their shapes checked; unknown keys raise. ``sigma_v`` and
    ``sigma_c`` may be given as scalars (broadcast). The ``SOURCE`` of a partially overridden
    population is the override's ``SOURCE`` when given, else a string naming the overridden keys."""
    pop = {k: (np.array(v, dtype=np.float64) if isinstance(v, np.ndarray) else v) for k, v in POPULATION_DEFAULTS.items()}
    if population:
        unknown = sorted(set(population) - set(pop))
        if unknown:
            raise KeyError(f"unknown population key(s) {unknown}; allowed {sorted(pop)}")
        overridden = [k for k in population if k != "SOURCE"]
        pop.update(population)
        if "SOURCE" not in population and overridden:
            pop["SOURCE"] = f"{SOURCE}; overridden keys {overridden}"
    for key, shape in _ARRAY_KEYS.items():
        arr = np.asarray(pop[key], dtype=np.float64)
        if key in ("sigma_v", "sigma_c") and arr.ndim == 0:
            arr = np.full(shape, float(arr))
        if arr.shape != shape:
            raise ValueError(f"population[{key!r}] must have shape {shape}; got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"population[{key!r}] has non-finite entries")
        if key in ("sigma_v", "sigma_c") and np.any(arr < 0):
            raise ValueError(f"population[{key!r}] must be >= 0")
        pop[key] = arr
    for key in _SCALAR_KEYS:
        val = float(pop[key])
        if not np.isfinite(val):
            raise ValueError(f"population[{key!r}] must be finite")
        pop[key] = val
    if pop["sigma_gamma"] < 0:
        raise ValueError("population['sigma_gamma'] must be >= 0")
    if tuple(pop["contexts"]) != tuple(_design.NONNULL_CONTEXTS):
        raise ValueError("population['contexts'] must equal design.NONNULL_CONTEXTS (row order of mu_c)")
    if tuple(pop["covariate_columns"]) != tuple(_COLUMNS):
        raise ValueError("population['covariate_columns'] must equal bre.models.data.covariate_columns()")
    return pop


# ---------------------------------------------------------------------------------------------
# Forward passes (batched over subjects x items through the core primitives)
# ---------------------------------------------------------------------------------------------


def bloch_angles(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bloch-sphere angles of gauge-fixed states ``(m, 2)``: polar ``theta = 2 atan2(|psi_1|,
    |psi_0|)`` in ``[0, pi]`` (``P(sell) = sin^2(theta / 2)``) and azimuth ``phi = arg psi_1``
    in ``(-pi, pi]`` (``arg psi_0 = 0`` by :func:`core.prepare_state`)."""
    psi = np.asarray(psi)
    polar = 2.0 * np.arctan2(np.abs(psi[:, 1]), np.abs(psi[:, 0]))
    azimuth = np.angle(psi[:, 1])
    return polar, azimuth


@jax.jit
def _expand(rho, theta_ctx, gamma, loss, ctx_idx, ctx_mask, order_flag):
    """Broadcast per-subject and per-item arrays to ``n * m`` rows (subject-major)."""
    n, m = rho.shape[0], loss.shape[0]
    rho_rows = jnp.repeat(rho, m, axis=0)
    ctx_rows = jax.vmap(C.gather_context_thetas, in_axes=(0, None, None))(theta_ctx, ctx_idx, ctx_mask)
    ctx_rows = ctx_rows.reshape(n * m, ctx_idx.shape[1], 3)
    return (
        rho_rows,
        ctx_rows,
        jnp.repeat(gamma, m),
        jnp.tile(loss, n),
        jnp.tile(ctx_mask, (n, 1)),
        jnp.tile(order_flag, n),
    )


@jax.jit
def sell_probabilities(rho, theta_ctx, gamma, theta_L, phi, loss, ctx_idx, ctx_mask, order_flag, tol_answer):
    """``P(sell)`` of every (subject, item) pair, ``(n, m)``, through :func:`core.q_forward_rho_batch`.

    ``rho (n, 2, 2)``, ``theta_ctx (n, V, 3)``, ``gamma (n,)`` per subject; ``loss (m,)``,
    ``ctx_idx (m, K)``, ``ctx_mask (m, K)``, ``order_flag (m,)`` per item (the
    :func:`bre.sim.common.item_arrays` encoding); ``tol_answer (n, m)`` the prior tolerance
    answer (0/1) for tolerance-first items and NaN for scenario-first ones.
    """
    n, m = rho.shape[0], loss.shape[0]
    rho_rows, ctx_rows, gamma_rows, loss_rows, mask_rows, order_rows = _expand(
        rho, theta_ctx, gamma, loss, ctx_idx, ctx_mask, order_flag
    )
    p = C.q_forward_rho_batch(
        rho_rows, theta_L, loss_rows, ctx_rows, mask_rows, order_rows, phi, tol_answer.reshape(-1), gamma_rows
    )
    return p.reshape(n, m)


@jax.jit
def post_sell_tolerance_probabilities(rho, theta_ctx, gamma, theta_L, phi, loss, ctx_idx, ctx_mask, sell):
    """``P(yes)`` of the tolerance question asked *after* the sell question, ``(n, m)``: the
    scenario-first chain :func:`core.chain_rho` (no tolerance measurement), Lüders collapse on the
    observed sell/hold outcome ``sell (n, m)`` (:func:`core.lueders_rho`), then
    ``Tr(P_yes rho)`` (:func:`core.born_rho`)."""
    n, m = rho.shape[0], loss.shape[0]
    rho_rows, ctx_rows, gamma_rows, loss_rows, mask_rows, _ = _expand(
        rho, theta_ctx, gamma, loss, ctx_idx, ctx_mask, jnp.zeros(m, dtype=jnp.int64)
    )
    rho_post = jax.vmap(C.chain_rho, in_axes=(0, None, 0, 0, 0, 0))(
        rho_rows, theta_L, loss_rows, ctx_rows, mask_rows, gamma_rows
    )
    s = jnp.asarray(sell, dtype=C.RDTYPE).reshape(-1)
    proj = jnp.where(s[:, None, None] >= 0.5, C.P_SELL[None], C.P_HOLD[None])
    collapsed = jax.vmap(C.lueders_rho)(rho_post, proj)
    p_yes = jax.vmap(C.born_rho, in_axes=(0, None))(collapsed, C.tolerance_projector(phi))
    return p_yes.reshape(n, m)


# ---------------------------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------------------------


def draw_subjects(
    rng: np.random.Generator, X: np.ndarray, pop: dict[str, Any], mixed_population: bool
) -> dict[str, np.ndarray]:
    """Per-subject latents from a resolved population: ``v``, ``psi``, ``theta_ctx (n, 4, 3)``,
    ``gamma (n,)`` (all zero when ``mixed_population`` is False) and the Bloch angles.
    Draw order (fixed for reproducibility): state noise, context deviations, log-rates."""
    n = X.shape[0]
    eps = rng.normal(size=(n, D_STATE)) * pop["sigma_v"]
    v = X @ pop["W"] + pop["b"] + eps
    z_ctx = rng.normal(size=(n,) + pop["mu_c"].shape)
    theta_ctx = pop["mu_c"] + pop["sigma_c"] * z_ctx
    z_gamma = rng.normal(size=n)
    if mixed_population:
        log_gamma = pop["mu_gamma"] + pop["sigma_gamma"] * z_gamma
        gamma = np.exp(log_gamma)
    else:
        log_gamma = np.full(n, -np.inf)
        gamma = np.zeros(n)
    psi = np.asarray(jax.vmap(C.prepare_state)(jnp.asarray(v)))
    polar, azimuth = bloch_angles(psi)
    return {
        "v": v,
        "state_noise": eps,
        "psi": psi,
        "bloch_theta": polar,
        "bloch_phi": azimuth,
        "theta_ctx": theta_ctx,
        "gamma": gamma,
        "log_gamma": log_gamma,
    }


def generate(
    n_subjects: int,
    seed: int,
    population: dict[str, Any] | None = None,
    design: ModuleType | pd.DataFrame | None = None,
    mixed_population: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate ``n_subjects`` synthetic subjects answering the design (default: all 170 items).

    Returns ``(frame, truth)``: the DecisionEvent frame (``2 * n_items`` rows per subject, all
    ``is_synthetic = True``, ``dataset = "synthetic_gq"``) and the truth dict (module docstring).
    ``population`` overrides entries of :data:`POPULATION_DEFAULTS`; ``mixed_population=False``
    sets every ``gamma_i = 0`` (pure Q2 data). All randomness comes from
    ``numpy.random.default_rng(seed)`` in a fixed draw order, so the output is a deterministic
    function of ``(n_subjects, seed, population, design, mixed_population)``.
    """
    if n_subjects < 0:
        raise ValueError("n_subjects must be >= 0")
    pop = resolve_population(population)
    rng = np.random.default_rng(int(seed))
    subjects = common.make_subjects(rng, n_subjects)
    records = common.covariate_records(subjects)
    X, columns, stats = covariate_matrix(records)
    assert tuple(columns) == tuple(pop["covariate_columns"])
    latents = draw_subjects(rng, X, pop, mixed_population)

    items = common.design_items(design)
    arrays = common.item_arrays(items)
    m = len(items)
    n = n_subjects
    tol_first = arrays["order_flag"] == 1

    # Answers. Draw order: tolerance-first tolerance answers, sell answers, scenario-first
    # tolerance answers (each a uniform per (subject, item) compared with its probability).
    theta_L = jnp.asarray(pop["theta_L"])
    phi = jnp.asarray(pop["phi"])
    loss = jnp.asarray(arrays["loss"])
    ctx_idx = jnp.asarray(arrays["ctx_idx"])
    ctx_mask = jnp.asarray(arrays["ctx_mask"])
    order_flag = jnp.asarray(arrays["order_flag"])
    if n > 0 and m > 0:
        rho = jax.vmap(C.to_rho)(jnp.asarray(latents["psi"]))
        theta_ctx = jnp.asarray(latents["theta_ctx"])
        gamma = jnp.asarray(latents["gamma"])
        p_yes_first = np.asarray(
            jax.vmap(C.born_rho, in_axes=(0, None))(rho, C.tolerance_projector(phi))
        )  # (n,) the initial state's P(yes), the same for every item
        u_tol = rng.random((n, m))
        tol = (u_tol < p_yes_first[:, None]).astype(np.float64)
        tol_answer = np.where(tol_first[None, :], tol, np.nan)
        p_sell = np.asarray(
            sell_probabilities(rho, theta_ctx, gamma, theta_L, phi, loss, ctx_idx, ctx_mask, order_flag, jnp.asarray(tol_answer))
        )
        u_sell = rng.random((n, m))
        sell = (u_sell < p_sell).astype(np.float64)
        p_yes_post = np.asarray(
            post_sell_tolerance_probabilities(rho, theta_ctx, gamma, theta_L, phi, loss, ctx_idx, ctx_mask, jnp.asarray(sell))
        )
        u_tol2 = rng.random((n, m))
        tol_post = (u_tol2 < p_yes_post).astype(np.float64)
        tol = np.where(tol_first[None, :], tol, tol_post)
        p_tol = np.where(tol_first[None, :], p_yes_first[:, None], p_yes_post)
    else:
        p_yes_first = np.zeros(n)
        p_sell = np.zeros((n, m))
        p_tol = np.zeros((n, m))
        sell = np.zeros((n, m))
        tol = np.zeros((n, m))
    if not (np.all(np.isfinite(p_sell)) and np.all((p_sell >= 0) & (p_sell <= 1))):
        raise RuntimeError("G_Q produced a sell probability outside [0, 1]")

    frames = []
    for i in range(n):
        frames.append(
            common.assemble_events(
                subject_id=str(subjects["subject_id"].iloc[i]),
                covariates=records[i],
                items_df=items,
                sell_responses=sell[i],
                tol_responses=tol[i],
                dataset=DATASET,
                battery_version=_design.DESIGN_VERSION,
                session_id=SESSION_ID,
                rng=rng,
                generator=NAME,
                seed=int(seed),
            )
        )
    df = common.concat_events(frames)

    truth: dict[str, Any] = {
        "generator": NAME,
        "seed": int(seed),
        "n_subjects": int(n),
        "n_items": int(m),
        "mixed_population": bool(mixed_population),
        "design_version": _design.DESIGN_VERSION,
        "dataset": DATASET,
        "session_id": SESSION_ID,
        "population": {k: (list(v) if isinstance(v, tuple) else v) for k, v in pop.items()},
        "covariate_columns": list(columns),
        "covariate_stats": {k: [float(a), float(b)] for k, (a, b) in stats.items()},
        "state_dims": list(STATE_DIMS),
        "subjects": {
            "subject_id": subjects["subject_id"].tolist(),
            "covariates": records,
            "X": X,
            "v": latents["v"],
            "state_noise": latents["state_noise"],
            "psi_re": np.real(latents["psi"]),
            "psi_im": np.imag(latents["psi"]),
            "bloch_theta": latents["bloch_theta"],
            "bloch_phi": latents["bloch_phi"],
            "theta_ctx": latents["theta_ctx"],
            "theta_ctx_contexts": list(pop["contexts"]),
            "gamma": latents["gamma"],
            "log_gamma": [None if not np.isfinite(g) else float(g) for g in latents["log_gamma"]],
            "p_yes_initial": p_yes_first,
        },
        "items": {
            "item_id": items["item_id"].tolist(),
            "order_flag": arrays["order_flag"],
            "p_sell": p_sell,
            "p_tol": p_tol,
            "sell": sell,
            "tol": tol,
        },
    }
    return df, truth


# ---------------------------------------------------------------------------------------------
# From fitted parameters back to a population
# ---------------------------------------------------------------------------------------------


def population_from_fit(
    params: dict[str, Any],
    ctx_vocab: tuple[str, ...],
    x_columns: tuple[str, ...] | None = None,
    source: str = "fitted Q2/Q4 parameters",
    sigma_c: float | np.ndarray = 0.0,
) -> dict[str, Any]:
    """Map fitted Q2/Q4 parameters to a G_Q population dict (for :func:`generate`).

    ``W``, ``b`` come from ``params["enc"]``, ``sigma_v = exp(log_sigma_u)`` (the fitted
    random-effect scale), ``theta_L``, ``phi`` as fitted, ``mu_c`` the rows of
    ``params["theta_ctx"]`` matching ``design.NONNULL_CONTEXTS`` in ``ctx_vocab``; ``sigma_c``
    is 0 unless given (Q2/Q4 fit one population table, not a per-subject spread). Q4 params
    (``gamma_mu``, ``gamma_log_sigma``) give ``mu_gamma``/``sigma_gamma``; otherwise the
    dephasing entries keep their synthetic defaults and ``SOURCE`` says so. ``x_columns`` (the
    fit's ``ModelData.X_columns``) must equal :func:`covariate_columns` when given.
    """
    enc = params["enc"]
    W = np.asarray(enc["W"], dtype=np.float64)
    if x_columns is not None and tuple(x_columns) != tuple(_COLUMNS):
        raise ValueError("the fit's X_columns differ from covariate_columns(); cannot map W")
    if W.shape != (len(_COLUMNS), D_STATE):
        raise ValueError(f"params['enc']['W'] must have shape {(len(_COLUMNS), D_STATE)}; got {W.shape}")
    vocab = tuple(ctx_vocab)
    theta_ctx = np.asarray(params["theta_ctx"], dtype=np.float64)
    if theta_ctx.shape != (len(vocab), 3):
        raise ValueError(f"params['theta_ctx'] must have shape {(len(vocab), 3)}; got {theta_ctx.shape}")
    missing = [c for c in _design.NONNULL_CONTEXTS if c not in vocab]
    if missing:
        raise ValueError(f"ctx_vocab lacks design context(s) {missing}")
    mu_c = np.stack([theta_ctx[vocab.index(c)] for c in _design.NONNULL_CONTEXTS])
    pop: dict[str, Any] = {
        "SOURCE": source,
        "W": W,
        "b": np.asarray(enc["b"], dtype=np.float64),
        "sigma_v": np.exp(np.asarray(enc["log_sigma_u"], dtype=np.float64)),
        "theta_L": np.asarray(params["theta_L"], dtype=np.float64),
        "mu_c": mu_c,
        "sigma_c": np.broadcast_to(np.asarray(sigma_c, dtype=np.float64), mu_c.shape).copy(),
        "phi": float(np.asarray(params["phi"])),
    }
    if "gamma_mu" in params and "gamma_log_sigma" in params:
        pop["mu_gamma"] = float(np.asarray(params["gamma_mu"]))
        pop["sigma_gamma"] = float(np.exp(np.asarray(params["gamma_log_sigma"])))
    else:
        pop["SOURCE"] = f"{source}; mu_gamma/sigma_gamma are the {SOURCE}"
    return pop


__all__ = [
    "DATASET",
    "D_STATE",
    "NAME",
    "POPULATION_DEFAULTS",
    "SESSION_ID",
    "SOURCE",
    "STATE_DIMS",
    "bloch_angles",
    "draw_subjects",
    "generate",
    "population_from_fit",
    "post_sell_tolerance_probabilities",
    "resolve_population",
    "sell_probabilities",
]
