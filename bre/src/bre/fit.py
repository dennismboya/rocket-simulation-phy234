"""Fit driver of the BRE models (PLAN.md section 4, equal-footing rules).

:func:`fit_model` fits one registered model (``bre.registry.MODEL_REGISTRY``) on a ``ModelData``:

* MAP models (B1, B3, B4, B6, Q2, Q3, Q4, Q5): ``restarts`` random restarts of full-batch Adam
  on ``model.objective`` (negative log-posterior) with a cosine-decayed learning rate, **early
  stopping** on a validation fold, then an **L-BFGS polish** (``optax.lbfgs`` with its zoom line
  search) of the early-stopped parameters; the restart with the lowest validation NLL wins.
* B2 (hierarchical Bayesian, NumPyro): ``restarts`` SVI runs (``B2.fit_svi``, ``svi_steps`` steps,
  different PRNG keys), the run with the lowest validation NLL of its posterior-mean parameters
  wins and its posterior draws become ``param_samples``; WAIC and PSIS-LOO of the winning run on
  the training rows are recorded through ``B2.waic`` / ``B2.loo``.
* Externally fitted models (``model.requires_external_fit`` is True: B5, whose parameters are
  pickled scikit-learn estimators): ``restarts`` calls of ``model.fit_external(train, seed)``
  with the seed of restart ``r`` derived from ``(seed, r)`` (:func:`external_fit_seed`); the
  fit with the lowest validation NLL wins. No Adam, no L-BFGS polish and **no Laplace draws**
  (there is no differentiable objective): ``param_samples`` is None and the result carries a
  note saying so. Parameter counts always come from ``model.n_params`` (``registry.param_counts``
  with the model instance): for B5 the pytree leaves are bytes, not parameters.

Every Q-model (any model with ``interference_terms``) has its mean interference terms on the
training rows logged (``extra["interference_train"]``: ``delta_LTP`` and ``Delta_order``; a NaN
mean means the model does not define the term, as Q5's ``ltp``); a model with ``mean_abs_q``
(Q5) has its mean ``|q|`` logged next to them. Q3 is fitted at each row's default time ``t``
(no news age in the shared design; see ``bre.models.quantum.q3_dynamics``).

Validation fold (documented rule): ``val_frac`` (default 20%) of the training sell rows is held
out **within subject**, stratified by condition type (number of contexts x question order), so
that every subject and every condition type appears on both sides and the fold measures
generalization to new items of known subjects, not to new subjects (that is split (a) of the
evaluation protocol, ``bre.eval``). Early stopping evaluates the validation NLL per response
every ``eval_every`` steps and stops after ``early_stopping_patience`` evaluations without
improvement, keeping the best-validation parameters. The L-BFGS polish is accepted only when it
lowers the training objective and does not raise the validation NLL by more than
``POLISH_VAL_TOLERANCE`` nats per response; otherwise the pre-polish parameters are kept and
the restart table says so.

Uncertainty: :func:`laplace_samples` draws ``n_samples`` parameter vectors from a diagonal
Laplace approximation at the optimum for the MAP models (the approximation and its limits are
documented in :mod:`bre.artifact`); B2 uses its posterior draws; externally fitted models get
none (the draws need ``jax.hessian`` of the objective, which the external models lack).

Config-driven use: ``python -m bre.fit experiments/<name>.yaml`` (:func:`run_experiment`) reads
the dataset, model, split and optimizer settings from the YAML, fits on the split's training
rows, evaluates the held-out rows with :mod:`bre.eval`, and saves a :class:`bre.artifact.ModelArtifact`
under ``runs/fits/<name>/`` plus a metrics JSON under ``reports/fits/<name>.json``. Without an
argument it runs every experiment listed in ``experiments/fit_all.yaml``. Every random choice
(validation fold, restarts, draws) is a function of the YAML's seeds (CLAUDE.md rule 5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import yaml
from jax.flatten_util import ravel_pytree

from bre import design as _design
from bre import schema as S
from bre.artifact import (
    ARTIFACT_VERSION,
    ModelArtifact,
    calibrated_contexts_table,
    jsonable,
    to_numpy_tree,
)
from bre.models.base import Model
from bre.models.data import ModelData, build_model_data
from bre.registry import MODEL_FAMILY, PER_SUBJECT_BLOCKS, make_model, param_counts

PROJECT_ROOT = S.PROJECT_ROOT
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FITS_DIR = PROJECT_ROOT / "runs" / "fits"
REPORTS_FITS_DIR = PROJECT_ROOT / "reports" / "fits"
FIT_ALL = EXPERIMENTS_DIR / "fit_all.yaml"

POLISH_VAL_TOLERANCE = 1e-4
"""Nats per response by which the L-BFGS polish may raise the validation NLL and still be kept."""

EARLY_STOP_MIN_DELTA = 1e-5
"""Improvement of the validation NLL per response below which an evaluation counts as no progress."""

LR_END_FRACTION = 0.05
"""``alpha`` of ``optax.cosine_decay_schedule``: the learning rate decays to this fraction of ``lr``."""

LAPLACE_MIN_CURVATURE = 1e-6
"""Floor on the diagonal Hessian entries of the Laplace approximation (flat directions)."""

LAPLACE_MAX_SD = 3.0
"""Cap on the Laplace standard deviation of any parameter (flat directions would otherwise give
draws that leave the region where the quadratic approximation means anything)."""

DEFAULT_FIT_SETTINGS: dict[str, Any] = {
    "restarts": 5,
    "steps": 1500,
    "lr": 0.02,
    "val_frac": 0.2,
    "seed": 0,
    "early_stopping_patience": 5,
    "eval_every": 25,
    "lbfgs_polish": True,
    "lbfgs_steps": 100,
    "n_samples": 50,
    "svi_steps": 2000,
    "svi_lr": 0.01,
    "svi_particles": 1,
}
"""Optimizer settings a YAML may override under ``fit:``."""


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def nll_per_response(ll, mask) -> float:
    """Held-out NLL per response: ``-mean(ll[mask])`` (INTERFACE.md, weights left in)."""
    ll = np.asarray(ll, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return float("nan")
    return float(-ll[mask].mean())


def condition_strata(data: ModelData) -> np.ndarray:
    """Per-row stratum label ``"<n_contexts>|<order_flag>"`` (condition type x question order)."""
    n_ctx = data.ctx_mask.sum(axis=1)
    return np.asarray([f"{int(k)}|{int(o)}" for k, o in zip(n_ctx, data.order_flag)], dtype=object)


def split_within_subject(
    data: ModelData,
    rows: np.ndarray,
    frac: float,
    seed: int,
    strata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out ``frac`` of ``rows`` inside every (subject, stratum) group.

    ``rows`` are row indices; ``strata`` (default :func:`condition_strata`) labels every row of
    ``data``. Within each group the rows are shuffled with ``numpy.random.default_rng(seed)``
    (one generator, groups visited in sorted order, so the split is a deterministic function of
    ``seed``) and ``round(frac * n_group)`` of them go to the held-out part. Returns
    ``(kept_rows, held_out_rows)`` as sorted index arrays.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if not (0.0 <= frac < 1.0):
        raise ValueError("frac must be in [0, 1)")
    strata = condition_strata(data) if strata is None else np.asarray(strata, dtype=object)
    rng = np.random.default_rng(int(seed))
    key = pd.DataFrame({"row": rows, "subject": data.subject_idx[rows], "stratum": strata[rows]})
    held: list[np.ndarray] = []
    for _, grp in key.sort_values(["subject", "stratum", "row"]).groupby(["subject", "stratum"], sort=True):
        r = grp["row"].to_numpy()
        n_hold = int(round(frac * len(r)))
        if n_hold > 0:
            held.append(rng.permutation(r)[:n_hold])
    held_out = np.sort(np.concatenate(held)) if held else np.zeros(0, dtype=np.int64)
    kept = np.setdiff1d(rows, held_out)
    return kept, held_out


def _finite_tree(tree) -> bool:
    return all(bool(np.isfinite(np.asarray(leaf)).all()) for leaf in jax.tree_util.tree_leaves(tree))


def requires_external_fit(model: Model) -> bool:
    """True for a model whose parameters are not fitted by gradient steps (``B5``: the class
    attribute ``requires_external_fit``); such a model has no differentiable objective, so no
    Adam, no L-BFGS polish and no Laplace draws apply to it."""
    return bool(getattr(model, "requires_external_fit", False))


def external_fit_seed(key, restart: int) -> int:
    """The integer seed of restart ``restart`` of an external fit: ``randint(fold_in(key,
    restart))`` in ``[0, 2^31)``, the same derivation ``B5.init_params`` uses, so that the fit
    driver and a direct ``init_params`` call agree for the same ``(seed, restart)``."""
    return int(jax.random.randint(jax.random.fold_in(key, int(restart)), (), 0, 2**31 - 1))


NO_DRAWS_NOTE = ("no parameter draws: the model is fitted externally (pickled estimators, no differentiable "
                 "objective), so neither a Laplace approximation nor posterior draws exist; intervals are not available")
"""``extra["param_samples_note"]`` (and the artifact note) of an externally fitted model."""


def _log(log_path: str | Path | None, line: str, echo: bool = True) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = f"{stamp} {line}"
    if echo:
        print(text, flush=True)
    if log_path is not None:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")


# ---------------------------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------------------------


@dataclass
class FitResult:
    """What :func:`fit_model` returns (numpy pytrees; the B2 posterior object rides along)."""

    model_name: str
    params: dict[str, Any]
    param_samples: list[dict[str, Any]] | None
    train_nll: float
    val_nll: float
    restarts_table: pd.DataFrame
    n_params: int
    wall_time: float
    n_params_population: int = 0
    best_restart: int = -1
    train_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    val_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    settings: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    posterior: Any = None
    model: Any = None

    def metrics(self) -> dict[str, Any]:
        """Scalars and the restart table, JSON-ready."""
        out = {
            "model_name": self.model_name,
            "train_nll": self.train_nll,
            "val_nll": self.val_nll,
            "n_params": self.n_params,
            "n_params_population": self.n_params_population,
            "wall_time_s": self.wall_time,
            "best_restart": self.best_restart,
            "n_train_rows": int(len(self.train_rows)),
            "n_val_rows": int(len(self.val_rows)),
            "n_samples": 0 if self.param_samples is None else len(self.param_samples),
            "settings": dict(self.settings),
            "restarts": jsonable(self.restarts_table.to_dict(orient="records")),
        }
        out.update(jsonable(self.extra))
        return out


# ---------------------------------------------------------------------------------------------
# Adam with early stopping, L-BFGS polish
# ---------------------------------------------------------------------------------------------


def adam_early_stopping(
    model: Model,
    params: Any,
    train: ModelData,
    val: ModelData,
    *,
    steps: int,
    lr: float,
    patience: int,
    eval_every: int,
) -> tuple[Any, dict[str, Any]]:
    """Full-batch Adam on ``model.objective(params, train)`` with early stopping on the
    validation NLL per response (module docstring). Returns the best-validation parameters and
    a history dict (``val_history`` of ``(step, val_nll)``, ``steps_run``, ``stopped_early``,
    ``init_objective``, ``final_objective``, ``diverged``)."""
    steps = int(steps)
    schedule = optax.cosine_decay_schedule(float(lr), max(steps, 1), alpha=LR_END_FRACTION)
    opt = optax.adam(schedule)
    state = opt.init(params)

    def objective(p):
        return model.objective(p, train)

    @jax.jit
    def step(p, s):
        v, g = jax.value_and_grad(objective)(p)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, v

    val_sell = np.flatnonzero(val.sell_mask())  # concrete indices: usable inside jit

    def val_nll(p):
        return -jnp.mean(model.log_lik(p, val)[val_sell])

    init_obj = float(objective(params))
    # The first evaluation on each table is eager: models that cache per-table arrays (B6, Q3)
    # would otherwise cache tracers created inside the trace and leak them into later traces.
    best_val = float(val_nll(params)) if val.n else float("inf")
    val_fn = jax.jit(val_nll)
    best_params = params
    history: list[tuple[int, float]] = [(0, best_val)]
    bad = 0
    stopped_early = False
    diverged = False
    last_obj = init_obj
    t = 0
    for t in range(1, steps + 1):
        params, state, v = step(params, state)
        last_obj = float(v)
        if not np.isfinite(last_obj):
            diverged = True
            break
        if t % int(eval_every) == 0 or t == steps:
            if val.n:
                vn = float(val_fn(params))
                history.append((t, vn))
                if vn < best_val - EARLY_STOP_MIN_DELTA:
                    best_val, best_params, bad = vn, params, 0
                else:
                    bad += 1
                    if bad >= int(patience):
                        stopped_early = True
                        break
            else:
                best_params = params
    if not val.n:
        best_params = params
    return best_params, {
        "val_history": history,
        "steps_run": int(t),
        "stopped_early": stopped_early,
        "diverged": diverged,
        "init_objective": init_obj,
        "final_objective": last_obj,
        "best_val_nll": best_val,
    }


def polish_lbfgs(model: Model, params: Any, train: ModelData, *, max_iter: int, rel_tol: float = 1e-7) -> tuple[Any, dict[str, Any]]:
    """L-BFGS (``optax.lbfgs``, zoom line search) on ``model.objective(params, train)`` from
    ``params`` for at most ``max_iter`` iterations; stops when the objective changes by less
    than ``rel_tol`` relatively for three iterations in a row or becomes non-finite (then the
    last finite parameters are returned). Returns ``(params, info)``."""

    def objective(p):
        return model.objective(p, train)

    opt = optax.lbfgs()
    state = opt.init(params)
    value_and_grad = optax.value_and_grad_from_state(objective)

    @jax.jit
    def step(p, s):
        v, g = value_and_grad(p, state=s)
        upd, s = opt.update(g, s, p, value=v, grad=g, value_fn=objective)
        return optax.apply_updates(p, upd), s, v

    start = float(objective(params))
    prev = start
    best_p, best_v = params, start
    still = 0
    n_iter = 0
    for n_iter in range(1, int(max_iter) + 1):
        new_p, state, v = step(params, state)
        v = float(v)
        if not np.isfinite(v) or not _finite_tree(new_p):
            break
        params = new_p
        v_now = float(objective(params))
        if not np.isfinite(v_now):
            break
        if v_now < best_v:
            best_p, best_v = params, v_now
        if abs(prev - v_now) <= rel_tol * max(1.0, abs(v_now)):
            still += 1
            if still >= 3:
                break
        else:
            still = 0
        prev = v_now
    return best_p, {"objective_before": start, "objective_after": best_v, "iterations": int(n_iter)}


# ---------------------------------------------------------------------------------------------
# Diagonal Laplace draws
# ---------------------------------------------------------------------------------------------


def _split_population(model_name: str, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(population_block, per_subject_block)`` of a params dict (registry.PER_SUBJECT_BLOCKS)."""
    pop = {k: (dict(v) if isinstance(v, dict) else v) for k, v in params.items()}
    subj: dict[str, Any] = {}
    for path in PER_SUBJECT_BLOCKS[model_name]:
        if len(path) == 1:
            subj[path[0]] = pop.pop(path[0])
        else:
            subj.setdefault(path[0], {})[path[1]] = pop[path[0]].pop(path[1])
    return pop, subj


def _merge(pop: dict[str, Any], subj: dict[str, Any]) -> dict[str, Any]:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in pop.items()}
    for k, v in subj.items():
        if isinstance(v, dict):
            out.setdefault(k, {}).update(v)
        else:
            out[k] = v
    return out


def hessian_diagonal(f: Callable[[Any], Any], x) -> np.ndarray:
    """``diag(H)`` of the scalar function ``f`` at the vector ``x`` through one Hessian-vector
    product per coordinate (``jax.jvp`` of ``jax.grad(f)`` along each basis vector, jitted
    once): the same numbers as ``np.diag(jax.hessian(f)(x))``, at the memory of a single
    gradient pass. ``jax.hessian`` (forward-over-reverse over every coordinate at once)
    materializes ``n`` reverse passes together and was killed by the memory limit for Q3 on a
    3,400-row table; the per-coordinate products stay flat in memory and cost ``n`` gradient
    evaluations in time."""
    x = jnp.asarray(x, dtype=jnp.float64)
    n = int(x.shape[0])
    grad_f = jax.grad(f)

    @jax.jit
    def hvp_entry(j):
        e = jnp.zeros(n, dtype=jnp.float64).at[j].set(1.0)
        return jax.jvp(grad_f, (x,), (e,))[1][j]

    return np.asarray([float(hvp_entry(j)) for j in range(n)], dtype=np.float64)


def laplace_samples(
    model: Model,
    model_name: str,
    params: dict[str, Any],
    train: ModelData,
    n_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``n_samples`` draws from the diagonal Laplace approximation at ``params`` (see
    :mod:`bre.artifact` for the approximation and its limits): the diagonal of the Hessian of
    ``model.objective`` with respect to the population block (per-subject blocks held fixed) is
    computed by :func:`hessian_diagonal`; each population parameter ``j`` gets
    ``sd_j = min(LAPLACE_MAX_SD, 1 / sqrt(max(H_jj, LAPLACE_MIN_CURVATURE)))`` and the draws are
    ``theta* + sd * z`` with ``z ~ N(0, I)`` from ``numpy.random.default_rng(seed)``. Returns
    the draws (numpy pytrees) and a diagnostics dict (``n_pop_params``, ``n_flat_directions``,
    ``n_capped``, ``min_curvature``, ``median_sd``)."""
    pop, subj = _split_population(model_name, params)
    pop_j = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=jnp.float64), pop)
    vec, unravel = ravel_pytree(pop_j)
    subj_j = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=jnp.float64), subj)

    def objective_vec(v):
        return model.objective(_merge(unravel(v), subj_j), train)

    diag = hessian_diagonal(objective_vec, vec)
    flat = ~np.isfinite(diag) | (diag < LAPLACE_MIN_CURVATURE)
    curv = np.where(flat, LAPLACE_MIN_CURVATURE, diag)
    sd = 1.0 / np.sqrt(curv)
    capped = sd > LAPLACE_MAX_SD
    sd = np.minimum(sd, LAPLACE_MAX_SD)
    rng = np.random.default_rng(int(seed))
    draws: list[dict[str, Any]] = []
    base = np.asarray(vec)
    for _ in range(int(n_samples)):
        z = rng.standard_normal(base.shape[0])
        draws.append(to_numpy_tree(_merge(unravel(jnp.asarray(base + sd * z)), subj)))
    diag_info = {
        "method": "diagonal Laplace at the MAP, population block only (per-subject blocks fixed)",
        "n_pop_params": int(base.shape[0]),
        "n_flat_directions": int(flat.sum()),
        "n_capped": int(capped.sum()),
        "min_curvature": float(np.nanmin(diag)) if diag.size else float("nan"),
        "median_sd": float(np.median(sd)) if sd.size else float("nan"),
    }
    return draws, diag_info


# ---------------------------------------------------------------------------------------------
# Interference terms on a fold
# ---------------------------------------------------------------------------------------------


def interference_summary(model: Model, params: dict[str, Any], data: ModelData) -> dict[str, Any]:
    """Mean interference terms of a Q-model over the sell rows of ``data``: ``ltp_mean`` /
    ``ltp_abs_mean`` (NaN, with ``ltp_defined = False``, when the model returns NaN on every
    row, as Q5 does: ``delta_LTP`` is not defined for it), ``order_mean`` / ``order_abs_mean``
    over the ordered-pair rows (NaN without pair rows) and, for a model with ``mean_abs_q``
    (Q5), ``mean_abs_q``. ``order_zero_by_construction`` is True when every pair row has an
    order term of exactly 0 (Q3 and Q5 compose contexts symmetrically)."""
    sell = data.sell_mask()
    terms = model.interference_terms(params, data)
    ltp = np.asarray(terms["ltp"], dtype=np.float64)[sell]
    order = np.asarray(terms["order"], dtype=np.float64)[sell]
    ltp_ok, order_ok = bool(np.isfinite(ltp).any()), bool(np.isfinite(order).any())
    out: dict[str, Any] = {
        "n_rows": int(sell.sum()),
        "ltp_defined": ltp_ok,
        "ltp_mean": float(np.nanmean(ltp)) if ltp_ok else float("nan"),
        "ltp_abs_mean": float(np.nanmean(np.abs(ltp))) if ltp_ok else float("nan"),
        "n_pair_rows": int(np.isfinite(order).sum()),
        "order_mean": float(np.nanmean(order)) if order_ok else float("nan"),
        "order_abs_mean": float(np.nanmean(np.abs(order))) if order_ok else float("nan"),
        "order_zero_by_construction": bool(order_ok and np.all(order[np.isfinite(order)] == 0.0)),
    }
    if hasattr(model, "mean_abs_q"):
        out["mean_abs_q"] = float(model.mean_abs_q(params, data))
    return out


# ---------------------------------------------------------------------------------------------
# fit_model
# ---------------------------------------------------------------------------------------------


def fit_model(
    model_name: str,
    data: ModelData,
    *,
    restarts: int = 5,
    steps: int = 1500,
    lr: float = 0.02,
    val_frac: float = 0.2,
    seed: int = 0,
    early_stopping_patience: int = 5,
    eval_every: int = 25,
    lbfgs_polish: bool = True,
    lbfgs_steps: int = 100,
    log_path: str | Path | None = None,
    n_samples: int = 50,
    svi_steps: int = 2000,
    svi_lr: float = 0.01,
    svi_particles: int = 1,
    rows: np.ndarray | None = None,
    model_kwargs: dict[str, Any] | None = None,
    echo: bool = False,
    model: Model | None = None,
) -> FitResult:
    """Fit ``model_name`` on the sell rows of ``data`` (or on ``rows`` of it), module docstring.

    ``seed`` drives the validation fold, the restart initializations (``jax.random.PRNGKey(seed)``
    with the restart folded in by the model), the SVI keys and the Laplace draws. ``rows``
    restricts the training rows (e.g. the training half of an evaluation split); tolerance rows
    among them are carried and contribute zero to the likelihood. A progress line per restart
    goes to ``log_path`` (and to stdout when ``echo``).
    """
    t_start = time.time()
    if model_name not in MODEL_FAMILY:
        raise KeyError(f"unknown model {model_name!r}")
    all_rows = data.sell_rows() if rows is None else np.intersect1d(np.asarray(rows, dtype=np.int64), data.sell_rows())
    if len(all_rows) == 0:
        raise ValueError("no sell rows to fit on")
    train_rows, val_rows = split_within_subject(data, all_rows, float(val_frac), int(seed))
    train = data.subset(train_rows)
    val = data.subset(val_rows)
    model = make_model(model_name, data, **(model_kwargs or {})) if model is None else model
    key = jax.random.PRNGKey(int(seed))
    settings = {
        "restarts": int(restarts),
        "steps": int(steps),
        "lr": float(lr),
        "val_frac": float(val_frac),
        "seed": int(seed),
        "early_stopping_patience": int(early_stopping_patience),
        "eval_every": int(eval_every),
        "lbfgs_polish": bool(lbfgs_polish),
        "lbfgs_steps": int(lbfgs_steps),
        "n_samples": int(n_samples),
        "svi_steps": int(svi_steps),
        "svi_lr": float(svi_lr),
        "svi_particles": int(svi_particles),
        "model_kwargs": dict(model_kwargs or {}),
        "validation_fold": "within-subject, stratified by (n_contexts, question order)",
    }
    rows_tab: list[dict[str, Any]] = []
    extra: dict[str, Any] = {}

    def val_nll_of(p) -> float:
        return nll_per_response(np.asarray(model.log_lik(p, val)), val.sell_mask()) if val.n else float("nan")

    def train_nll_of(p) -> float:
        return nll_per_response(np.asarray(model.log_lik(p, train)), train.sell_mask())

    best: tuple[int, float, Any, Any] | None = None  # (restart, val_nll, params, posterior)
    external = requires_external_fit(model)

    if external:
        for r in range(int(restarts)):
            t0 = time.time()
            seed_r = external_fit_seed(key, r)
            p_np = to_numpy_tree(model.fit_external(train, seed_r))
            vn, tn = val_nll_of(p_np), train_nll_of(p_np)
            rows_tab.append({"restart": r, "method": "external", "seed": seed_r, "train_nll": tn, "val_nll": vn, "seconds": time.time() - t0})
            _log(log_path, f"[fit] {model_name} restart {r + 1}/{restarts}: external fit (seed {seed_r}), train_nll={tn:.4f} val_nll={vn:.4f} ({time.time() - t0:.0f}s)", echo)
            if best is None or (np.isfinite(vn) and vn < best[1]) or (not np.isfinite(best[1])):
                best = (r, vn, p_np, None)
        assert best is not None
        r_best, val_nll, params, _ = best
        posterior = None
        samples = None
        extra["param_samples_note"] = NO_DRAWS_NOTE
        extra["external_fit"] = {"method": "model.fit_external(train, seed) per restart; best validation NLL kept", "seeds": [int(row["seed"]) for row in rows_tab], "n_samples_requested": int(n_samples)}
        if int(n_samples) > 0:
            _log(log_path, f"[fit] {model_name}: n_samples={n_samples} requested but the model is fitted externally; no draws", echo)
    elif model_name == "B2":
        for r in range(int(restarts)):
            t0 = time.time()
            k = jax.random.fold_in(key, r)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                post = model.fit_svi(train, steps=int(svi_steps), key=k, lr=float(svi_lr), num_samples=max(int(n_samples), 20), num_particles=int(svi_particles))
            p = to_numpy_tree(post.params)
            vn, tn = val_nll_of(p), train_nll_of(p)
            rows_tab.append({"restart": r, "method": "svi", "steps_run": int(svi_steps), "final_loss": float(post.diagnostics["final_loss"]), "train_nll": tn, "val_nll": vn, "seconds": time.time() - t0})
            _log(log_path, f"[fit] {model_name} restart {r + 1}/{restarts}: svi {svi_steps} steps, elbo_loss={post.diagnostics['final_loss']:.1f} train_nll={tn:.4f} val_nll={vn:.4f} ({time.time() - t0:.0f}s)", echo)
            if best is None or (np.isfinite(vn) and vn < best[1]):
                best = (r, vn, p, post)
        assert best is not None
        r_best, val_nll, params, post = best
        samples = [to_numpy_tree(model.params_from_draw(post, s)) for s in range(int(n_samples))] if int(n_samples) > 0 else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ll = model.pointwise_log_lik(post, train)
            try:
                w = model.waic(post, train, log_lik=ll)
                lo = model.loo(post, train, log_lik=ll)
                n_sell = int(train.sell_mask().sum())
                extra["waic"] = {"elpd_waic": float(w.elpd_waic), "p_waic": float(w.p_waic), "se": float(w.se), "elpd_waic_per_response": float(w.elpd_waic) / n_sell, "warning": bool(w.warning)}
                extra["loo"] = {"elpd_loo": float(lo.elpd_loo), "p_loo": float(lo.p_loo), "se": float(lo.se), "elpd_loo_per_response": float(lo.elpd_loo) / n_sell, "warning": bool(lo.warning), "pareto_k_max": float(np.max(np.asarray(lo.pareto_k)))}
            except Exception as exc:  # noqa: BLE001 - ArviZ failures are reported, not fatal
                extra["waic_error"] = f"{type(exc).__name__}: {exc}"
        if val.n:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                llv = model.pointwise_log_lik(post, val)[:, val.sell_mask()]
            extra["val_nll_posterior_predictive"] = float(-np.mean(np.log(np.mean(np.exp(llv), axis=0))))
        extra["svi_diagnostics"] = {k: v for k, v in post.diagnostics.items() if k != "losses"}
        posterior = post
    else:
        for r in range(int(restarts)):
            t0 = time.time()
            p0 = model.init_params(key, data, r)
            init_nll = train_nll_of(p0)
            p, hist = adam_early_stopping(model, p0, train, val, steps=int(steps), lr=float(lr), patience=int(early_stopping_patience), eval_every=int(eval_every))
            vn_pre = val_nll_of(p)
            tn_pre = train_nll_of(p)
            polish_info: dict[str, Any] = {"applied": False}
            if lbfgs_polish and not hist["diverged"] and int(lbfgs_steps) > 0:
                p_pol, info = polish_lbfgs(model, p, train, max_iter=int(lbfgs_steps))
                vn_pol = val_nll_of(p_pol)
                ok = info["objective_after"] < info["objective_before"] and (not np.isfinite(vn_pre) or not np.isfinite(vn_pol) or vn_pol <= vn_pre + POLISH_VAL_TOLERANCE)
                polish_info = {"applied": bool(ok), **info, "val_nll_after": vn_pol}
                if ok:
                    p, vn_pre = p_pol, vn_pol
            vn, tn = vn_pre, train_nll_of(p)
            p_np = to_numpy_tree(p)
            rows_tab.append({
                "restart": r, "method": "adam+lbfgs" if polish_info.get("applied") else "adam", "init_train_nll": init_nll,
                "steps_run": hist["steps_run"], "stopped_early": hist["stopped_early"], "diverged": hist["diverged"],
                "init_objective": hist["init_objective"], "final_objective": hist["final_objective"],
                "polish_applied": polish_info.get("applied", False), "polish_iterations": polish_info.get("iterations", 0),
                "polish_objective_after": polish_info.get("objective_after", float("nan")),
                "train_nll_pre_polish": tn_pre, "train_nll": tn, "val_nll": vn, "seconds": time.time() - t0,
            })
            _log(log_path, f"[fit] {model_name} restart {r + 1}/{restarts}: adam {hist['steps_run']} steps (early_stop={hist['stopped_early']}, diverged={hist['diverged']}), polish={polish_info.get('applied', False)}/{polish_info.get('iterations', 0)} it, init_nll={init_nll:.4f} train_nll={tn:.4f} val_nll={vn:.4f} ({time.time() - t0:.0f}s)", echo)
            if not hist["diverged"] and (best is None or (np.isfinite(vn) and vn < best[1]) or (not np.isfinite(best[1]))):
                best = (r, vn, p_np, None)
        if best is None:
            raise RuntimeError(f"{model_name}: every restart diverged")
        r_best, val_nll, params, _ = best
        posterior = None
        samples = None
        if int(n_samples) > 0 and not external:  # Laplace draws need jax.hessian of the objective
            t0 = time.time()
            samples, lap = laplace_samples(model, model_name, params, train, int(n_samples), int(seed))
            extra["laplace"] = {**lap, "seconds": time.time() - t0}
            _log(log_path, f"[fit] {model_name} laplace: {lap['n_pop_params']} population params, {lap['n_flat_directions']} flat, {lap['n_capped']} capped ({time.time() - t0:.0f}s)", echo)

    counts = param_counts(model_name, params, model)
    if hasattr(model, "interference_terms"):
        extra["interference_train"] = interference_summary(model, params, train)
        it = extra["interference_train"]
        _log(log_path, f"[fit] {model_name} interference on the training rows: ltp_mean={it['ltp_mean']:.4f} order_mean={it['order_mean']:.4f}" + (f" mean_abs_q={it['mean_abs_q']:.4f}" if "mean_abs_q" in it else ""), echo)
    result = FitResult(
        model_name=model_name,
        params=params,
        param_samples=samples,
        train_nll=train_nll_of(params),
        val_nll=float(val_nll),
        restarts_table=pd.DataFrame(rows_tab),
        n_params=counts["all"],
        n_params_population=counts["population"],
        wall_time=time.time() - t_start,
        best_restart=int(r_best),
        train_rows=train_rows,
        val_rows=val_rows,
        settings=settings,
        extra=extra,
        posterior=posterior,
        model=model,
    )
    _log(log_path, f"[fit] {model_name} done: best restart {r_best}, train_nll={result.train_nll:.4f} val_nll={result.val_nll:.4f}, n_params={counts['all']} (population {counts['population']}), {result.wall_time:.0f}s", echo)
    return result


# ---------------------------------------------------------------------------------------------
# Artifacts from fits
# ---------------------------------------------------------------------------------------------


def artifact_from_fit(
    result: FitResult,
    data: ModelData,
    *,
    training_data_refs: list[str],
    metrics: dict[str, Any] | None = None,
    notes: str = "",
) -> ModelArtifact:
    """Package a :class:`FitResult` as a :class:`bre.artifact.ModelArtifact` (training subjects
    stored, calibrated-context table from the training rows, provenance and metrics)."""
    model = result.model
    b1_cols = getattr(model, "columns", None) if result.model_name == "B1" else None
    cc = calibrated_contexts_table(result.model_name, result.params, data, rows=result.train_rows, param_samples=result.param_samples, b1_columns=b1_cols)
    m = dict(result.metrics())
    if metrics:
        m.update(jsonable(metrics))
    is_syn = bool(np.asarray(data.is_synthetic).any())
    laplace_note = ""
    if result.model_name != "B2" and result.param_samples:
        laplace_note = (" param_samples: diagonal Laplace approximation at the MAP over the population block "
                        "(per-subject blocks fixed; curvature floored at %g, sd capped at %g); local, correlation-free, "
                        "blind to gauge copies — see bre.artifact." % (LAPLACE_MIN_CURVATURE, LAPLACE_MAX_SD))
    elif result.model_name == "B2" and result.param_samples:
        laplace_note = " param_samples: posterior draws from B2's SVI guide (AutoNormal, mean-field)."
    elif result.extra.get("param_samples_note"):
        laplace_note = " param_samples: " + str(result.extra["param_samples_note"]) + "."
    return ModelArtifact(
        version=ARTIFACT_VERSION,
        model_name=result.model_name,
        family=MODEL_FAMILY[result.model_name],
        params=result.params,
        param_samples=result.param_samples,
        ctx_vocab=tuple(data.ctx_vocab),
        covariate_stats=dict(data.covariate_stats),
        X_columns=tuple(data.X_columns),
        design_version=_design.DESIGN_VERSION,
        training_data_refs=list(training_data_refs),
        metrics=m,
        calibrated_contexts=cc,
        n_params=result.n_params,
        created_at=ModelArtifact.now(),
        is_synthetic_training=is_syn,
        notes=(notes + laplace_note).strip(),
        subject_ids=tuple(data.subject_ids),
        n_ctx_positions=int(data.n_ctx_positions),
        model_kwargs=dict(result.settings.get("model_kwargs", {})),
    )


# ---------------------------------------------------------------------------------------------
# Config-driven experiments
# ---------------------------------------------------------------------------------------------


def resolve_path(p: str | Path) -> Path:
    """Relative paths are relative to the project root ``bre/`` (not the working directory)."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg.setdefault("name", path.stem)
    cfg["_path"] = str(path)
    return cfg


def load_dataset(cfg: dict[str, Any]) -> tuple[pd.DataFrame, ModelData, str]:
    """Read the config's dataset (a schema parquet) and build its ``ModelData`` with the
    ``data:`` options. Returns ``(frame, data, reference)`` with ``reference`` the path relative
    to the project root (the artifact's ``training_data_refs`` entry)."""
    if "dataset" not in cfg:
        raise KeyError(f"{cfg.get('_path', 'config')}: 'dataset' is required")
    path = resolve_path(cfg["dataset"])
    df = S.read_events(path)
    opts = dict(cfg.get("data") or {})
    for k in ("sell_types", "rate_types"):
        if k in opts:
            opts[k] = tuple(opts[k])
    data = build_model_data(df, **opts)
    try:
        ref = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        ref = str(path)
    return df, data, ref


def fit_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """``DEFAULT_FIT_SETTINGS`` updated with the config's ``fit:`` block (unknown keys raise)."""
    fit = dict(cfg.get("fit") or {})
    unknown = sorted(set(fit) - set(DEFAULT_FIT_SETTINGS))
    if unknown:
        raise KeyError(f"unknown fit setting(s) {unknown}; allowed {sorted(DEFAULT_FIT_SETTINGS)}")
    return {**DEFAULT_FIT_SETTINGS, **fit}


def run_experiment(cfg: dict[str, Any] | str | Path, *, echo: bool = True) -> tuple[ModelArtifact, Path, Path]:
    """Run one fit experiment (module docstring). Returns ``(artifact, artifact_dir, metrics_path)``."""
    from bre import eval as E  # noqa: PLC0415 - eval imports fit; keep the cycle lazy

    if not isinstance(cfg, dict):
        cfg = load_config(cfg)
    name = cfg["name"]
    model_name = cfg["model"]
    if model_name not in MODEL_FAMILY:
        raise KeyError(f"{name}: unknown model {model_name!r}")
    out = dict(cfg.get("output") or {})
    art_dir = resolve_path(out.get("artifact_dir", FITS_DIR / name))
    metrics_path = resolve_path(out.get("metrics_path", REPORTS_FITS_DIR / f"{name}.json"))
    log_path = art_dir / "fit.log"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=False), encoding="utf-8")
    _log(log_path, f"[fit] experiment {name}: model={model_name} dataset={cfg['dataset']}", echo)
    df, data, ref = load_dataset(cfg)
    split_cfg = dict(cfg.get("split") or {"type": "none"})
    train_rows, test_rows = E.split_rows(data, split_cfg)
    _log(log_path, f"[fit] data: {data.summary()}; split={split_cfg} -> train rows {len(train_rows)}, test rows {len(test_rows)}", echo)
    settings = fit_settings(cfg)
    result = fit_model(model_name, data, rows=train_rows, log_path=log_path, echo=echo, model_kwargs=dict(cfg.get("model_kwargs") or {}), **settings)
    metrics: dict[str, Any] = {"experiment": name, "dataset": ref, "split": split_cfg, "n_train_rows": int(len(train_rows)), "n_test_rows": int(len(test_rows))}
    if len(test_rows):
        boot = dict(cfg.get("bootstrap") or {})
        held = E.evaluate_model(result.model, result.params, data, test_rows, n_boot=int(boot.get("n_draws", 1000)), seed=int(boot.get("seed", 0)), param_samples=result.param_samples, posterior=result.posterior)
        metrics["holdout"] = {k: v for k, v in held.items() if k not in ("ll", "p")}
        _log(log_path, f"[fit] held-out ({split_cfg.get('type')}): nll={held['nll']:.4f} [{held['nll_ci95'][0]:.4f}, {held['nll_ci95'][1]:.4f}] brier={held['brier']:.4f} ece={held['ece']:.4f} auc={held['auc']:.4f}", echo)
    artifact = artifact_from_fit(result, data, training_data_refs=[ref], metrics=metrics, notes=str(cfg.get("notes", "")))
    artifact.save(art_dir)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": name, "config": {k: v for k, v in cfg.items() if not k.startswith("_")}, "artifact_dir": str(art_dir.relative_to(PROJECT_ROOT)) if art_dir.is_relative_to(PROJECT_ROOT) else str(art_dir), "is_synthetic": artifact.is_synthetic_training, "metrics": artifact.metrics, "calibrated_contexts": artifact.calibrated_contexts, "n_params": artifact.n_params}
    metrics_path.write_text(json.dumps(jsonable(payload), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    _log(log_path, f"[fit] saved artifact {art_dir} and metrics {metrics_path}", echo)
    return artifact, art_dir, metrics_path


def experiments_listed(path: str | Path = FIT_ALL) -> list[Path]:
    """The experiment YAMLs named by a ``*_all.yaml`` (``experiments: [paths]``, relative to the
    project root or to the list file's directory)."""
    path = resolve_path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: list[Path] = []
    for p in cfg.get("experiments") or []:
        cand = resolve_path(p)
        if not cand.exists():
            cand = path.parent / p
        out.append(cand)
    return out


def expand_configs(paths: list[Path]) -> list[Path]:
    """Experiment YAMLs, with any list file (``experiments: [...]``) expanded in place."""
    out: list[Path] = []
    for p in paths:
        p = resolve_path(p)
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if "experiments" in cfg and "model" not in cfg and "models" not in cfg:
            out.extend(experiments_listed(p))
        else:
            out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bre.fit", description="Fit one BRE model from an experiment YAML (default: every experiment in experiments/fit_all.yaml).")
    parser.add_argument("config", nargs="*", help="experiment YAML(s)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    configs = expand_configs([Path(c) for c in args.config] or [FIT_ALL])
    if not configs:
        print("no experiments listed", file=sys.stderr)
        return 1
    failures = 0
    for c in configs:
        try:
            run_experiment(c, echo=not args.quiet)
        except Exception as exc:  # noqa: BLE001 - report and continue with the list
            failures += 1
            print(f"[fit] {c}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FIT_SETTINGS",
    "FitResult",
    "adam_early_stopping",
    "artifact_from_fit",
    "condition_strata",
    "experiments_listed",
    "external_fit_seed",
    "fit_model",
    "fit_settings",
    "hessian_diagonal",
    "interference_summary",
    "laplace_samples",
    "polish_lbfgs",
    "load_config",
    "load_dataset",
    "nll_per_response",
    "requires_external_fit",
    "resolve_path",
    "run_experiment",
    "split_within_subject",
]
