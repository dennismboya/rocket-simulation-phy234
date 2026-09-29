"""G_F — the flexible-logistic generator of the Phase 2 recovery study (PLAN.md section 5).

Everything produced here is synthetic (CLAUDE.md rule 3): every row carries ``is_synthetic = True``
and every population default below is a labelled synthetic design choice
(``POPULATION_DEFAULTS["SOURCE"]``), not a number fitted to or quoted from data.

Role
----
G_F is the flexibility control: a classical logistic with *position-specific* context effects and a
free interaction term for *every ordered pair* of contexts, so its context effects are neither
additive (as in G_C) nor composable from single-context parameters (as in Q2's product of
unitaries or B4's summed evidence). B1 can represent it exactly; Q2, Q4 and B4 cannot. It also
exercises the order and prior-answer features (``beta_order``, ``beta_tol`` non-zero by default),
which makes ``delta_LTP != 0`` through a classical mechanism (the answer given to the tolerance
question shifts the sell log-odds); ``tests/test_sim_gc_gf.py`` records that this is expected.

Generative model
----------------
Subject ``i`` has an intercept ``u_i ~ N(0, sigma_u^2)`` and a pair-interaction multiplier
``s_i = exp(N(0, sigma_pair_scale^2))`` (``mixed_population=True``; with ``False`` every subject
has ``u_i = 0`` and ``s_i = 1``). For an item with loss ``L = |loss_pct|``, contexts
``c_1..c_m`` (m <= 2) at positions ``k = 0..m-1``, question order and, for a tolerance-first
item, the tolerance answer ``a`` given just before::

    eta = intercept + u_i + beta_L L + sum_k beta_ctx_by_position[k][c_k]
          + s_i * beta_pair["<c_1>><c_2>"]           (only when m = 2; keys are condition ids)
          + beta_order * order_flag + beta_tol * a

    P(sell) = sigmoid(eta)

The tolerance answer comes from its own logistic on the latent, ``P(a = 1) = sigmoid(tol_intercept
+ tol_u_slope * u_i)``. Items are independent given the latent ``(u_i, s_i)``.

Non-additivity (checked analytically from the truth in the tests): for an ordered pair the
log-odds displacement from ``none`` is ``beta_ctx[0][c_1] + beta_ctx[1][c_2] + s_i beta_pair[c_1>c_2]``,
which differs from the sum of the two single-context displacements ``beta_ctx[0][c_1] +
beta_ctx[0][c_2]`` and from the reversed pair, so ``P(sell | c_1, c_2) != P(sell | c_2, c_1)`` in
general (an order effect between contexts with a purely classical origin).

Output
------
``generate(n_subjects, seed, population=None, design=None, mixed_population=True)`` returns the
DecisionEvent frame (two rows per item, see :func:`bre.sim.common.assemble_events`; ``dataset =
"sim_gf"``) and the truth dict (population used, per-subject ``u``, ``pair_scale`` and tolerance
log-odds). :func:`sell_probability` and :func:`tolerance_probability` recompute any item's
probabilities from the truth alone.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from bre import design as _design
from bre.sim import common

NAME = "gf"
DATASET = "sim_gf"

SOURCE = "synthetic default, not fitted to data"

POPULATION_DEFAULTS: dict[str, Any] = {
    "SOURCE": SOURCE,
    "intercept": -1.0,
    "sigma_u": 0.8,
    "sigma_pair_scale": 0.5,  # sd of log s_i, the per-subject multiplier of the pair terms
    "beta_L": 6.0,
    # position-specific main effects: index 0 = first (or only) context shown, 1 = second
    "beta_ctx_by_position": [
        {"news:recession": 0.7, "news:technical": -0.3, "social:friend_sells": 0.5, "market:recovered_5pct": -0.6},
        {"news:recession": 0.9, "news:technical": -0.5, "social:friend_sells": 0.7, "market:recovered_5pct": -0.9},
    ],
    # one free term per ordered pair, keyed by the design's condition id (bre.design.condition_id);
    # chosen so that pairs are non-additive and order-asymmetric. Synthetic design choices.
    "beta_pair": {
        "rec>tech": -0.8,
        "tech>rec": 0.4,
        "rec>friend": 0.6,
        "friend>rec": 0.3,
        "rec>recov": -0.5,
        "recov>rec": 0.2,
        "tech>friend": 0.2,
        "friend>tech": -0.6,
        "tech>recov": -0.4,
        "recov>tech": -0.2,
        "friend>recov": -0.7,
        "recov>friend": 0.5,
    },
    "beta_order": 0.3,
    "beta_tol": 0.5,
    # tolerance answer
    "tol_intercept": 0.0,
    "tol_u_slope": 1.0,
}
"""Population parameters of G_F. ``SOURCE`` says what they are: synthetic design choices."""


# ---------------------------------------------------------------------------------------------
# Population handling
# ---------------------------------------------------------------------------------------------


def resolve_population(population: dict[str, Any] | None = None, design: ModuleType | None = None) -> dict[str, Any]:
    """:data:`POPULATION_DEFAULTS` with ``population`` overrides merged in (``beta_pair`` and each
    position's dict in ``beta_ctx_by_position`` merge one level deep). Unknown keys, missing
    contexts or missing ordered pairs raise."""
    design = _design if design is None else design
    pop: dict[str, Any] = {}
    for k, v in POPULATION_DEFAULTS.items():
        if k == "beta_ctx_by_position":
            pop[k] = [dict(d) for d in v]
        elif isinstance(v, dict):
            pop[k] = dict(v)
        else:
            pop[k] = v
    if population:
        unknown = sorted(k for k in population if k not in POPULATION_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown G_F population key(s) {unknown}; allowed {sorted(POPULATION_DEFAULTS)}")
        for k, v in population.items():
            if k == "beta_pair" and isinstance(v, dict):
                pop[k] = {**pop[k], **v}
            elif k == "beta_ctx_by_position":
                if len(v) != 2:
                    raise ValueError("beta_ctx_by_position must have one dict per position (2)")
                pop[k] = [{**pop[k][j], **dict(v[j])} for j in range(2)]
            else:
                pop[k] = v
        if "SOURCE" not in population:
            pop["SOURCE"] = SOURCE + " (with caller overrides: " + ", ".join(sorted(population)) + ")"
    for j in range(2):
        missing = [c for c in design.NONNULL_CONTEXTS if c not in pop["beta_ctx_by_position"][j]]
        unknown = [c for c in pop["beta_ctx_by_position"][j] if c not in design.NONNULL_CONTEXTS]
        if missing or unknown:
            raise ValueError(f"beta_ctx_by_position[{j}] lacks context(s) {missing} / has unknown context(s) {unknown}")
    pair_ids = [design.condition_id(c) for c in design.context_conditions() if len(c) == 2]
    missing = [p for p in pair_ids if p not in pop["beta_pair"]]
    unknown = [p for p in pop["beta_pair"] if p not in pair_ids]
    if missing or unknown:
        raise ValueError(f"beta_pair lacks ordered pair(s) {missing} / has unknown pair id(s) {unknown}; ids are {pair_ids}")
    if float(pop["sigma_u"]) < 0 or float(pop["sigma_pair_scale"]) < 0:
        raise ValueError("sigma_u and sigma_pair_scale must be >= 0")
    return pop


def context_tables(pop: dict[str, Any], design: ModuleType | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(B_ctx (2, V), B_pair (V, V))`` indexed like ``design.NONNULL_CONTEXTS``: the
    position-specific main effects and the ordered-pair terms (``B_pair[a, b]`` is the term of the
    pair shown as ``a`` then ``b``; the diagonal is 0 and unused)."""
    design = _design if design is None else design
    V = len(design.NONNULL_CONTEXTS)
    B_ctx = np.asarray(
        [[float(pop["beta_ctx_by_position"][k][c]) for c in design.NONNULL_CONTEXTS] for k in range(2)],
        dtype=np.float64,
    )
    B_pair = np.zeros((V, V), dtype=np.float64)
    for a_i, a in enumerate(design.NONNULL_CONTEXTS):
        for b_i, b in enumerate(design.NONNULL_CONTEXTS):
            if a != b:
                B_pair[a_i, b_i] = float(pop["beta_pair"][design.condition_id((a, b))])
    return B_ctx, B_pair


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


# ---------------------------------------------------------------------------------------------
# Analytic probabilities from the truth
# ---------------------------------------------------------------------------------------------


def sell_logit(
    truth: dict[str, Any],
    u: np.ndarray,
    pair_scale: np.ndarray,
    loss: np.ndarray,
    ctx_idx: np.ndarray,
    ctx_mask: np.ndarray,
    order_flag: np.ndarray,
    tol_answer: np.ndarray | None = None,
    design: ModuleType | None = None,
) -> np.ndarray:
    """Sell log-odds ``eta`` of the module docstring for rows with per-row subject latents ``u``
    and ``pair_scale``, loss, context indices/mask (:func:`bre.sim.common.item_arrays` layout),
    order flag and tolerance answer (NaN or None = not answered before, contributes 0)."""
    pop = truth["population"]
    B_ctx, B_pair = context_tables(pop, design)
    ctx_idx = np.asarray(ctx_idx)
    ctx_mask = np.asarray(ctx_mask, dtype=bool)
    if ctx_idx.shape[1] != 2:
        raise ValueError("G_F expects two context positions")
    safe = np.where(ctx_mask, ctx_idx, 0)
    main = np.where(ctx_mask[:, 0], B_ctx[0][safe[:, 0]], 0.0) + np.where(ctx_mask[:, 1], B_ctx[1][safe[:, 1]], 0.0)
    both = ctx_mask[:, 0] & ctx_mask[:, 1]
    pair = np.where(both, B_pair[safe[:, 0], safe[:, 1]], 0.0) * np.asarray(pair_scale, dtype=np.float64)
    eta = (
        float(pop["intercept"])
        + np.asarray(u, dtype=np.float64)
        + float(pop["beta_L"]) * np.asarray(loss, dtype=np.float64)
        + main
        + pair
        + float(pop["beta_order"]) * np.asarray(order_flag, dtype=np.float64)
    )
    if tol_answer is not None:
        a = np.asarray(tol_answer, dtype=np.float64)
        eta = eta + float(pop["beta_tol"]) * np.where(np.isfinite(a), a, 0.0)
    return eta


def subject_arrays(truth: dict[str, Any], subject_ids: list[str] | np.ndarray) -> dict[str, np.ndarray]:
    """Per-subject latent quantities from the truth for the given ids, as arrays."""
    subs = truth["subjects"]
    ids = list(subject_ids)
    return {
        "u": np.asarray([float(subs[s]["u"]) for s in ids], dtype=np.float64),
        "pair_scale": np.asarray([float(subs[s]["pair_scale"]) for s in ids], dtype=np.float64),
        "tol_logit": np.asarray([float(subs[s]["tol_logit"]) for s in ids], dtype=np.float64),
    }


def sell_probability(
    truth: dict[str, Any],
    subject_ids: list[str] | np.ndarray,
    loss: np.ndarray,
    ctx_idx: np.ndarray,
    ctx_mask: np.ndarray,
    order_flag: np.ndarray,
    tol_answer: np.ndarray | None = None,
    design: ModuleType | None = None,
) -> np.ndarray:
    """``P(sell)`` per row from the truth alone (one subject id per row)."""
    sa = subject_arrays(truth, subject_ids)
    return _sigmoid(sell_logit(truth, sa["u"], sa["pair_scale"], loss, ctx_idx, ctx_mask, order_flag, tol_answer, design))


def tolerance_probability(truth: dict[str, Any], subject_ids: list[str] | np.ndarray) -> np.ndarray:
    """``P(a = 1)`` per row from the truth (one subject id per row)."""
    return _sigmoid(subject_arrays(truth, subject_ids)["tol_logit"])


# ---------------------------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------------------------


def draw_latents(rng: np.random.Generator, n: int, pop: dict[str, Any], mixed_population: bool) -> dict[str, np.ndarray]:
    """Per-subject latents ``u`` ``(n,)``, ``pair_scale`` ``(n,)`` and the tolerance log-odds."""
    if mixed_population:
        u = rng.normal(0.0, float(pop["sigma_u"]), size=n)
        pair_scale = np.exp(rng.normal(0.0, float(pop["sigma_pair_scale"]), size=n))
    else:
        u = np.zeros(n, dtype=np.float64)
        pair_scale = np.ones(n, dtype=np.float64)
    return {
        "u": u,
        "pair_scale": pair_scale,
        "tol_logit": float(pop["tol_intercept"]) + float(pop["tol_u_slope"]) * u,
    }


def generate(
    n_subjects: int,
    seed: int,
    population: dict[str, Any] | None = None,
    design: ModuleType | pd.DataFrame | None = None,
    mixed_population: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate ``n_subjects`` synthetic subjects answering the design (default: the full 170-item
    design) under G_F. Returns ``(frame, truth)``; see the module docstring. Deterministic in
    ``seed``."""
    if n_subjects < 1:
        raise ValueError("n_subjects must be >= 1")
    design_mod = _design if not isinstance(design, ModuleType) else design
    pop = resolve_population(population, design_mod)
    items = common.design_items(design)
    arrays = common.item_arrays(items, design_mod)
    m = len(items)
    master = np.random.default_rng(int(seed))

    subjects = common.make_subjects(master, n_subjects, design_mod)
    latents = draw_latents(master, n_subjects, pop, mixed_population)
    child_seeds = master.integers(0, 2**31 - 1, size=n_subjects)
    truth: dict[str, Any] = {
        "generator": NAME,
        "seed": int(seed),
        "n_subjects": int(n_subjects),
        "design_version": design_mod.DESIGN_VERSION,
        "n_items": int(m),
        "mixed_population": bool(mixed_population),
        "population": pop,
        "context_vocab": list(design_mod.NONNULL_CONTEXTS),
        "subjects": {},
    }

    loss, ctx_idx, ctx_mask, order_flag = arrays["loss"], arrays["ctx_idx"], arrays["ctx_mask"], arrays["order_flag"]
    tol_first = order_flag == 1
    frames: list[pd.DataFrame] = []
    for i in range(n_subjects):
        sid = str(subjects["subject_id"].iloc[i])
        s = int(child_seeds[i])
        rng_i = np.random.default_rng([s, 1])
        tol = (rng_i.uniform(size=m) < _sigmoid(latents["tol_logit"][i])).astype(np.float64)
        eta = sell_logit(
            truth,
            np.full(m, latents["u"][i]),
            np.full(m, latents["pair_scale"][i]),
            loss,
            ctx_idx,
            ctx_mask,
            order_flag,
            np.where(tol_first, tol, np.nan),
            design_mod,
        )
        sell = (rng_i.uniform(size=m) < _sigmoid(eta)).astype(np.float64)
        frames.append(
            common.assemble_events(
                sid,
                str(subjects["covariates"].iloc[i]),
                items,
                sell,
                tol,
                DATASET,
                design_mod.DESIGN_VERSION,
                "full",
                s,
                generator=NAME,
                seed=int(seed),
                design=design_mod,
            )
        )
        truth["subjects"][sid] = {
            "u": float(latents["u"][i]),
            "pair_scale": float(latents["pair_scale"][i]),
            "tol_logit": float(latents["tol_logit"][i]),
            "child_seed": s,
        }
    return common.concat_events(frames), truth


__all__ = [
    "DATASET",
    "NAME",
    "POPULATION_DEFAULTS",
    "SOURCE",
    "context_tables",
    "draw_latents",
    "generate",
    "resolve_population",
    "sell_logit",
    "sell_probability",
    "subject_arrays",
    "tolerance_probability",
]
