"""G_C — the classical latent-type generator of the Phase 2 recovery study (PLAN.md section 5).

Everything produced here is synthetic (CLAUDE.md rule 3): every row carries ``is_synthetic = True``
and every population default below is a labelled synthetic design choice
(``POPULATION_DEFAULTS["SOURCE"]``), not a number fitted to or quoted from data.

Generative model
----------------
Subject ``i`` has a discrete latent type ``t_i`` in ``{0, .., K-1}`` (``K = 3``: hold-prone,
neutral, sell-prone) and a continuous intercept ``u_i ~ N(0, sigma_u^2)``. Type probabilities are a
softmax of a linear map of the standardized covariates ``x_i`` (:func:`bre.models.data.covariate_matrix`,
the same standardization the models see)::

    P(t_i = k | x_i) = softmax(x_i^T M + m)_k          M (d_x, K) from ``type_map``, m = ``type_bias``

The sell log-odds of an item with loss ``L = |loss_pct|``, contexts ``c_1..c_m`` (m <= 2), question
order and, for a tolerance-first item, the tolerance answer ``a`` given just before, are additive
and position-independent::

    eta = intercept + type_intercept[t_i] + u_i + beta_L L + sum_k beta_ctx[c_k]
          + beta_order * order_flag + beta_tol * a            (+ beta_prev * y_prev, Markov variant)

    P(sell) = sigmoid(eta)

The tolerance answer comes from its own logistic on the same latent::

    P(a = 1) = sigmoid(tol_intercept + tol_type_effects[t_i] + tol_u_slope * u_i)

Answers are conditionally independent given the latent ``(t_i, u_i)``: the sell answer and the
tolerance answer of an item are drawn independently once the latent is fixed (with the defaults
``beta_order = beta_tol = 0``), and items are independent of each other. The Markov variant
(``markov=True``) adds a first-order dependence ``beta_prev * y_prev`` on the sell answer of the
previously presented item (``y_prev = 0`` for the first item of the session); this is the
"sequential answers" case B6 is meant for.

No interference by construction
-------------------------------
G_C is a joint Kolmogorov distribution over all answers, so the law of total probability holds
exactly for every subject and item: ``P(sell | tolerance-first) = sum_a P(a) P(sell | a)``. The
interference term of PLAN.md section 4, ``delta_LTP = P(sell | scenario-first) - sum_a P(a, sell |
tolerance-first)``, is therefore ``sigmoid(eta) - sum_a P(a) sigmoid(eta + beta_order + beta_tol a)``,
which vanishes identically under the defaults ``beta_order = beta_tol = 0`` (:func:`ltp_interference`
computes it from the truth; ``tests/test_sim_gc_gf.py`` checks it). Non-zero ``beta_order`` or
``beta_tol`` produce a *classical* order effect — the act of answering the tolerance question, or
the answer itself, shifts the sell log-odds — that the classical features ``order_flag`` /
``tol_yes`` capture exactly; it is offered as a labelled variant, not as the default, because it
makes ``delta_LTP != 0`` and the structural tests of PLAN.md section 6 would then flag a classical
generator. Contexts compose additively and position-independently, so ``P(sell | c1, c2) =
P(sell | c2, c1)`` (no order effect between contexts) and ``logit P(c1, c2) - logit P(none) =
[logit P(c1) - logit P(none)] + [logit P(c2) - logit P(none)]`` exactly.

Output
------
``generate(n_subjects, seed, population=None, design=None, mixed_population=True, markov=False)``
returns the DecisionEvent frame (two rows per item, see :func:`bre.sim.common.assemble_events`;
``dataset = "sim_gc"``) and the truth dict: the population actually used (defaults merged with
``population`` overrides), the standardized-covariate column names, and per subject its type, type
probabilities, ``u``, the resulting sell intercept and tolerance log-odds. :func:`sell_probability`
and :func:`tolerance_probability` recompute any item's probabilities from the truth alone, which is
what the recovery study compares fitted models against. ``mixed_population=False`` switches the
type mixture off (every subject is the neutral type; only ``u_i`` varies).

Presentation order: items are permuted per subject by :func:`bre.sim.common.assemble_events` from
an integer seed; this module draws the same permutation from the same seed beforehand so the
Markov variant can generate answers in presentation order, and asserts afterwards that the
assembled positions agree.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from bre import design as _design
from bre.models.data import covariate_matrix
from bre.sim import common

NAME = "gc"
DATASET = "sim_gc"

SOURCE = "synthetic default, not fitted to data"

POPULATION_DEFAULTS: dict[str, Any] = {
    "SOURCE": SOURCE,
    # latent types
    "n_types": 3,
    "type_labels": ["hold-prone", "neutral", "sell-prone"],
    "type_intercepts": [-1.5, 0.0, 1.5],
    "type_bias": [0.0, 0.0, 0.0],
    # linear map from standardized covariate columns (bre.models.data.covariate_columns names) to
    # the K type logits; columns not listed have weight 0. Direction of the design choice: stated
    # risk tolerance, experience, literacy, age and wealth push towards the hold-prone type; youth
    # and low wealth towards the sell-prone type. Magnitudes are arbitrary and synthetic.
    "type_map": {
        "self_reported_risk_tolerance": [0.6, 0.0, -0.6],
        "invest_experience_yrs": [0.4, 0.0, -0.4],
        "financial_literacy_score": [0.3, 0.0, -0.3],
        "age_band=18-29": [-0.3, 0.0, 0.3],
        "age_band=60+": [0.3, 0.0, -0.3],
        "wealth_band=<50k": [-0.2, 0.0, 0.2],
        "wealth_band=>1M": [0.2, 0.0, -0.2],
    },
    # continuous subject intercept
    "sigma_u": 0.8,
    # sell log-odds
    "intercept": -1.0,
    "beta_L": 6.0,
    "beta_ctx": {
        "news:recession": 0.8,
        "news:technical": -0.4,
        "social:friend_sells": 0.6,
        "market:recovered_5pct": -0.7,
    },
    "beta_order": 0.0,
    "beta_tol": 0.0,
    "beta_prev": 0.8,  # used only with markov=True
    # tolerance answer
    "tol_intercept": 0.0,
    "tol_type_effects": [-0.8, 0.0, 0.8],
    "tol_u_slope": 1.0,
}
"""Population parameters of G_C. ``SOURCE`` says what they are: synthetic design choices."""

_DICT_KEYS = ("type_map", "beta_ctx")


# ---------------------------------------------------------------------------------------------
# Population handling
# ---------------------------------------------------------------------------------------------


def resolve_population(population: dict[str, Any] | None = None) -> dict[str, Any]:
    """:data:`POPULATION_DEFAULTS` with ``population`` overrides merged in (one level deep for the
    dict-valued entries ``type_map`` and ``beta_ctx``). Unknown keys raise; ``SOURCE`` is replaced
    by the caller's value when given so an override is never labelled as the default."""
    pop: dict[str, Any] = {}
    for k, v in POPULATION_DEFAULTS.items():
        if isinstance(v, dict):
            pop[k] = dict(v)
        elif isinstance(v, list):
            pop[k] = list(v)
        else:
            pop[k] = v
    if population:
        unknown = sorted(k for k in population if k not in POPULATION_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown G_C population key(s) {unknown}; allowed {sorted(POPULATION_DEFAULTS)}")
        for k, v in population.items():
            if k in _DICT_KEYS and isinstance(v, dict):
                pop[k] = {**pop[k], **v}
            else:
                pop[k] = v
        if "SOURCE" not in population:
            pop["SOURCE"] = SOURCE + " (with caller overrides: " + ", ".join(sorted(population)) + ")"
    K = int(pop["n_types"])
    for key in ("type_intercepts", "type_bias", "tol_type_effects", "type_labels"):
        if len(pop[key]) != K:
            raise ValueError(f"population[{key!r}] must have n_types = {K} entries")
    for col, row in pop["type_map"].items():
        if len(row) != K:
            raise ValueError(f"population['type_map'][{col!r}] must have n_types = {K} entries")
    unknown = [c for c in pop["beta_ctx"] if c not in _design.NONNULL_CONTEXTS]
    missing = [c for c in _design.NONNULL_CONTEXTS if c not in pop["beta_ctx"]]
    if unknown or missing:
        raise ValueError(f"population['beta_ctx'] has unknown context(s) {unknown} / lacks {missing}")
    if float(pop["sigma_u"]) < 0:
        raise ValueError("sigma_u must be >= 0")
    return pop


def type_map_matrix(pop: dict[str, Any], columns: tuple[str, ...]) -> np.ndarray:
    """``(d_x, K)`` matrix ``M`` of the type logits from ``pop["type_map"]`` (rows named by the
    standardized covariate columns; unlisted columns are zero; a listed name that is not a column
    raises)."""
    K = int(pop["n_types"])
    M = np.zeros((len(columns), K), dtype=np.float64)
    pos = {c: j for j, c in enumerate(columns)}
    for col, row in pop["type_map"].items():
        if col not in pos:
            raise ValueError(f"type_map column {col!r} is not a covariate column; known: {list(columns)}")
        M[pos[col]] = np.asarray(row, dtype=np.float64)
    return M


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def context_effects(pop: dict[str, Any], design: ModuleType | None = None) -> np.ndarray:
    """``beta_ctx`` as a ``(V,)`` array indexed like ``design.NONNULL_CONTEXTS``."""
    design = _design if design is None else design
    missing = [c for c in design.NONNULL_CONTEXTS if c not in pop["beta_ctx"]]
    if missing:
        raise ValueError(f"population['beta_ctx'] lacks context(s) {missing}")
    return np.asarray([float(pop["beta_ctx"][c]) for c in design.NONNULL_CONTEXTS], dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# Analytic probabilities from the truth
# ---------------------------------------------------------------------------------------------


def sell_logit(
    truth: dict[str, Any],
    sell_intercept: np.ndarray,
    loss: np.ndarray,
    ctx_idx: np.ndarray,
    ctx_mask: np.ndarray,
    order_flag: np.ndarray,
    tol_answer: np.ndarray | None = None,
    prev_sell: np.ndarray | None = None,
    design: ModuleType | None = None,
) -> np.ndarray:
    """Sell log-odds ``eta`` of the module docstring for rows with the given per-row subject sell
    intercept (``intercept + type_intercept[t] + u``; see the truth's ``subjects``), loss, context
    indices/mask (``bre.sim.common.item_arrays`` layout), order flag, tolerance answer (NaN or
    None = not answered before, contributes 0) and previous sell answer (None = 0)."""
    pop = truth["population"]
    beta = context_effects(pop, design)
    ctx_idx = np.asarray(ctx_idx)
    ctx_mask = np.asarray(ctx_mask, dtype=bool)
    ctx = np.where(ctx_mask, beta[np.where(ctx_mask, ctx_idx, 0)], 0.0).sum(axis=1)
    eta = (
        np.asarray(sell_intercept, dtype=np.float64)
        + float(pop["beta_L"]) * np.asarray(loss, dtype=np.float64)
        + ctx
        + float(pop["beta_order"]) * np.asarray(order_flag, dtype=np.float64)
    )
    if tol_answer is not None:
        a = np.asarray(tol_answer, dtype=np.float64)
        eta = eta + float(pop["beta_tol"]) * np.where(np.isfinite(a), a, 0.0)
    if prev_sell is not None:
        eta = eta + float(truth["beta_prev_effective"]) * np.asarray(prev_sell, dtype=np.float64)
    return eta


def subject_arrays(truth: dict[str, Any], subject_ids: list[str] | np.ndarray) -> dict[str, np.ndarray]:
    """Per-subject latent quantities from the truth for the given ids, as arrays."""
    subs = truth["subjects"]
    ids = list(subject_ids)
    return {
        "type": np.asarray([int(subs[s]["type"]) for s in ids], dtype=np.int64),
        "u": np.asarray([float(subs[s]["u"]) for s in ids], dtype=np.float64),
        "sell_intercept": np.asarray([float(subs[s]["sell_intercept"]) for s in ids], dtype=np.float64),
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
    prev_sell: np.ndarray | None = None,
    design: ModuleType | None = None,
) -> np.ndarray:
    """``P(sell)`` per row from the truth alone (one subject id per row)."""
    sa = subject_arrays(truth, subject_ids)
    return _sigmoid(sell_logit(truth, sa["sell_intercept"], loss, ctx_idx, ctx_mask, order_flag, tol_answer, prev_sell, design))


def tolerance_probability(truth: dict[str, Any], subject_ids: list[str] | np.ndarray) -> np.ndarray:
    """``P(a = 1)`` per row from the truth (one subject id per row)."""
    return _sigmoid(subject_arrays(truth, subject_ids)["tol_logit"])


def ltp_interference(
    truth: dict[str, Any],
    subject_ids: list[str] | np.ndarray,
    loss: np.ndarray,
    ctx_idx: np.ndarray,
    ctx_mask: np.ndarray,
    design: ModuleType | None = None,
) -> np.ndarray:
    """``delta_LTP = P(sell | scenario-first) - sum_a P(a) P(sell | a, tolerance-first)`` per row,
    computed from the truth (PLAN.md section 4 definition). Identically zero when ``beta_order =
    beta_tol = 0``; otherwise the classical order displacement of the module docstring."""
    n = len(subject_ids)
    zeros = np.zeros(n, dtype=np.int64)
    ones = np.ones(n, dtype=np.int64)
    p_sf = sell_probability(truth, subject_ids, loss, ctx_idx, ctx_mask, zeros, None, None, design)
    p_a1 = tolerance_probability(truth, subject_ids)
    p_yes = sell_probability(truth, subject_ids, loss, ctx_idx, ctx_mask, ones, np.ones(n), None, design)
    p_no = sell_probability(truth, subject_ids, loss, ctx_idx, ctx_mask, ones, np.zeros(n), None, design)
    return p_sf - (p_a1 * p_yes + (1.0 - p_a1) * p_no)


# ---------------------------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------------------------


def draw_latents(
    rng: np.random.Generator,
    X: np.ndarray,
    columns: tuple[str, ...],
    pop: dict[str, Any],
    mixed_population: bool,
) -> dict[str, np.ndarray]:
    """Per-subject latents: type probabilities ``(n, K)``, type ``(n,)``, ``u`` ``(n,)``, the sell
    intercept ``intercept + type_intercept[type] + u`` and the tolerance log-odds. With
    ``mixed_population=False`` every subject is the neutral type (index ``n_types // 2``) with
    probability one; ``u`` is drawn either way."""
    n = X.shape[0]
    K = int(pop["n_types"])
    if mixed_population:
        M = type_map_matrix(pop, columns)
        probs = _softmax(X @ M + np.asarray(pop["type_bias"], dtype=np.float64)[None, :])
        cum = probs.cumsum(axis=1)
        r = rng.uniform(size=(n, 1))
        types = np.minimum((r > cum).sum(axis=1), K - 1).astype(np.int64)
    else:
        neutral = K // 2
        probs = np.zeros((n, K), dtype=np.float64)
        probs[:, neutral] = 1.0
        types = np.full(n, neutral, dtype=np.int64)
    u = rng.normal(0.0, float(pop["sigma_u"]), size=n)
    type_int = np.asarray(pop["type_intercepts"], dtype=np.float64)[types]
    tol_eff = np.asarray(pop["tol_type_effects"], dtype=np.float64)[types]
    return {
        "type_probs": probs,
        "type": types,
        "u": u,
        "sell_intercept": float(pop["intercept"]) + type_int + u,
        "tol_logit": float(pop["tol_intercept"]) + tol_eff + float(pop["tol_u_slope"]) * u,
    }


def generate(
    n_subjects: int,
    seed: int,
    population: dict[str, Any] | None = None,
    design: ModuleType | pd.DataFrame | None = None,
    mixed_population: bool = True,
    markov: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate ``n_subjects`` synthetic subjects answering the design (default: the full 170-item
    design) under G_C. Returns ``(frame, truth)``; see the module docstring. Deterministic in
    ``seed`` (a ``numpy.random.default_rng(seed)`` master stream drives covariates, latents and
    per-subject child seeds)."""
    if n_subjects < 1:
        raise ValueError("n_subjects must be >= 1")
    pop = resolve_population(population)
    design_mod = _design if not isinstance(design, ModuleType) else design
    items = common.design_items(design)
    arrays = common.item_arrays(items, design_mod)
    m = len(items)
    master = np.random.default_rng(int(seed))

    subjects = common.make_subjects(master, n_subjects, design_mod)
    records = common.covariate_records(subjects)
    X, columns, stats = covariate_matrix(records, spec=design_mod.COVARIATE_SPEC)
    latents = draw_latents(master, X, columns, pop, mixed_population)
    child_seeds = master.integers(0, 2**31 - 1, size=n_subjects)
    beta_prev = float(pop["beta_prev"]) if markov else 0.0
    truth: dict[str, Any] = {
        "generator": NAME,
        "seed": int(seed),
        "n_subjects": int(n_subjects),
        "design_version": design_mod.DESIGN_VERSION,
        "n_items": int(m),
        "mixed_population": bool(mixed_population),
        "markov": bool(markov),
        "beta_prev_effective": beta_prev,
        "population": pop,
        "covariate_columns": list(columns),
        "covariate_stats": {k: [float(v[0]), float(v[1])] for k, v in stats.items()},
        "context_vocab": list(design_mod.NONNULL_CONTEXTS),
        "subjects": {},
    }

    loss, ctx_idx, ctx_mask, order_flag = arrays["loss"], arrays["ctx_idx"], arrays["ctx_mask"], arrays["order_flag"]
    tol_first = order_flag == 1
    frames: list[pd.DataFrame] = []
    for i in range(n_subjects):
        sid = str(subjects["subject_id"].iloc[i])
        s = int(child_seeds[i])
        perm = np.random.default_rng(s).permutation(m)  # what assemble_events(rng=s) will draw
        rng_i = np.random.default_rng([s, 1])
        p_tol = _sigmoid(latents["tol_logit"][i])
        tol = (rng_i.uniform(size=m) < p_tol).astype(np.float64)
        base = sell_logit(
            truth,
            np.full(m, latents["sell_intercept"][i]),
            loss,
            ctx_idx,
            ctx_mask,
            order_flag,
            np.where(tol_first, tol, np.nan),
            None,
            design_mod,
        )
        u_sell = rng_i.uniform(size=m)
        sell = np.zeros(m, dtype=np.float64)
        if markov and beta_prev != 0.0:
            prev = 0.0
            for k in perm.tolist():
                p = _sigmoid(base[k] + beta_prev * prev)
                sell[k] = float(u_sell[k] < p)
                prev = sell[k]
        else:
            sell = (u_sell < _sigmoid(base)).astype(np.float64)
        frame = common.assemble_events(
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
        _check_presentation_order(frame, items, perm, tol_first)
        frames.append(frame)
        truth["subjects"][sid] = {
            "type": int(latents["type"][i]),
            "type_label": str(pop["type_labels"][int(latents["type"][i])]),
            "type_probs": [float(v) for v in latents["type_probs"][i]],
            "u": float(latents["u"][i]),
            "sell_intercept": float(latents["sell_intercept"][i]),
            "tol_logit": float(latents["tol_logit"][i]),
            "child_seed": s,
        }
    df = common.concat_events(frames)
    truth["type_counts"] = {str(k): int((latents["type"] == k).sum()) for k in range(int(pop["n_types"]))}
    return df, truth


def _check_presentation_order(frame: pd.DataFrame, items: pd.DataFrame, perm: np.ndarray, tol_first: np.ndarray) -> None:
    """Assert that :func:`bre.sim.common.assemble_events` placed item ``perm[j]`` at positions
    ``2j``/``2j+1`` (the order the Markov variant generated the answers in)."""
    sell_rows = frame[frame["elicitation_type"] == "binary_sell"]
    expected_pos = np.where(tol_first[perm], 2 * np.arange(len(perm)) + 1, 2 * np.arange(len(perm)))
    got = sell_rows.sort_values("position_in_session")
    ok = (got["position_in_session"].to_numpy() == np.sort(expected_pos)).all() and (
        got["scenario_id"].to_numpy() == items["scenario_id"].to_numpy()[perm]
    ).all()
    if not ok:
        raise RuntimeError("G_C: assemble_events did not use the expected presentation order; check bre.sim.common")


__all__ = [
    "DATASET",
    "NAME",
    "POPULATION_DEFAULTS",
    "SOURCE",
    "context_effects",
    "draw_latents",
    "generate",
    "ltp_interference",
    "resolve_population",
    "sell_logit",
    "sell_probability",
    "subject_arrays",
    "tolerance_probability",
    "type_map_matrix",
]
