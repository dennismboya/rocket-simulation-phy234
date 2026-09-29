"""Phase 4: the pre-registered evaluation of PLAN.md section 6 on the real data in hand.

Primary dataset: CPC18 individual choices in the pairs variant of :mod:`bre.realdata`
(feedback blocks 2-5, context = the experienced outcomes of the one or two preceding trials,
response = safer option chosen; the analog of retreat to safety, not panic selling — the report
says so). Secondary: ``psych201_spektor2024lossaversion`` (described 50/50 lotteries, context =
outcome domain). ``python -m bre.phase4 --dataset cpc18 [--subjects N] [--steps S] [--models ...]``;
``--smoke`` runs the reduced settings of :data:`SMOKE_SETTINGS` (60 subjects, 300 steps, one
restart) for tests and timing; ``--config experiments/p4-*.yaml`` reads a run description.

Splits (letters of PLAN.md section 6, :func:`make_splits`):

(a) held-out subjects, 20% (``bre.eval.split_subjects``, stratified by dataset);
(b) held-out context compositions: train on the rows with at most one ``exp:*`` tag, test on
    the ordered pairs (``bre.realdata.split_pairs_by_exp_tags``; pairs variant only — a table
    without two-context rows reports the split as not applicable);
(c) held-out question order: **not applicable** to any real table in hand (no dataset carries
    a question-order manipulation; CPC18 and the Psych-201 tasks have no tolerance question).
    The split is listed with that reason and nothing is fitted for it;
(d) temporal: per subject, the first 80% of the trials in ``(session, position)`` order train,
    the last 20% of the sell rows test (``bre.eval.split_temporal``).

Models: every registered model that can fit the table (default all ten: B5 through its external
fit, B2 through SVI, Q3 at its default time ``t`` since no news age exists here). Every fit goes
through :func:`bre.fit.fit_model` with the same settings (restarts, Adam + early stopping,
L-BFGS polish, diagonal-Laplace draws for the MAP models).

Metrics (PLAN.md section 6): held-out NLL per response, Brier, ECE with the reliability table,
AUC, WAIC / PSIS-LOO for B2, parameter counts (all scalars and population only), the per-subject
predictive gain against the best classical model of the split, and 1000-draw subject-level
bootstrap intervals on every metric. The metrics are those of :mod:`bre.eval`; the AUC
bootstrap uses :func:`auc_fast`, a vectorised weighted Mann-Whitney AUC that equals
``bre.eval.auc`` (tested) and keeps the 1000 draws on 370,000 rows within seconds.

Structural tests that apply: order effects by scenario on the pairs variant (per problem,
``P(safe | exp:loss > exp:gain) - P(safe | exp:gain > exp:loss)``, subject bootstrap); the
interference-term interval of every Q-model on the split (b) test rows (``bre.eval.interference_ci``:
``delta_LTP`` and ``Delta_order`` over the Laplace / posterior draws); the decoherence-rate
distribution of Q4; the quarter-law check of Q5. The LTP violation from the data and the QQ
equality need a question-order manipulation and are reported as not applicable. Note on
``delta_LTP``: the tolerance projector angle ``phi`` is not identified on data without a
tolerance question (no row is tolerance-first), so the ``delta_LTP`` interval on these tables
is a model-internal counterfactual driven by the prior and the initialization; the report says
so next to the number. ``Delta_order`` (the order effect of the two experienced outcomes) is
identified on the pairs variant and is reported next to it.

Decision rule: :func:`bre.eval.decision_rule` applied verbatim to the split (b) results (the
best Q-model against the best classical model by held-out NLL, paired subject bootstrap of the
difference, matched-or-lower parameter count, interference interval); ``verdict.json`` carries
the exact verdict sentence. Without an applicable split (b) the verdict is the negative one with
the reason.

Outputs (``reports/phase4/<name>/``): ``metrics.json`` (every number), ``table.md`` (all models
side by side per split with parameter counts), ``structural.md``, ``verdict.json``,
``verdict.md``; fits, logs, per-job caches and artifacts under ``runs/phase4/<name>/``. A job
(split x model) whose cache exists with the same settings is not refitted, so a long run can be
split over several invocations (``--models`` subsets) and assembled by a final call; ``--refit``
ignores the cache. ``--estimate-only`` prints the wall-time estimate from the measured smoke
costs (:data:`SECONDS_PER_ROW_STEP`) scaled by rows; a run over two hours is a GATE (CLAUDE.md
rule 4) and must be split by the main session.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bre import eval as E
from bre import realdata as R
from bre import schema as S
from bre.artifact import jsonable, to_numpy_tree
from bre.fit import DEFAULT_FIT_SETTINGS, _log, fit_settings, load_config, resolve_path
from bre.models.data import ModelData
from bre.registry import MODEL_FAMILY, MODEL_REGISTRY

PROJECT_ROOT = S.PROJECT_ROOT
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase4"
RUNS_DIR = PROJECT_ROOT / "runs" / "phase4"

DEFAULT_MODELS: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5", "B6", "Q2", "Q3", "Q4", "Q5")
DEFAULT_SPLITS: tuple[str, ...] = ("a", "b", "c", "d")
SPLIT_NAMES: dict[str, str] = {"a": "subjects", "b": "pairs", "c": "order", "d": "temporal"}
TEST_FRAC = 0.2

DATASET_KEYS: dict[str, str] = {
    "cpc18": "cpc18",
    "psych201_spektor2024lossaversion": "psych201_spektor2024lossaversion",
    "spektor2024": "psych201_spektor2024lossaversion",
    "psych201_spektor2019contexteffects": "psych201_spektor2019contexteffects",
    "psych201_olschewski2024skewness": "psych201_olschewski2024skewness",
}
"""``--dataset`` names -> :data:`bre.realdata.DATASETS` keys (``cpc18`` takes ``--variant``)."""

SPLIT_C_NOT_APPLICABLE = (
    "split (c) held-out question order is not applicable: no real dataset in hand has a question-order "
    "manipulation (CPC18 and the Psych-201 tasks ask no tolerance question); it applies to the intake "
    "battery data only"
)
SPLIT_B_NOT_APPLICABLE = "split (b) needs ordered two-context rows; this table has none (only the CPC18 pairs variant has them)"
LTP_NOT_APPLICABLE = "the LTP violation from the data needs both question orders of the same scenario; no real table in hand has a question-order manipulation"
QQ_NOT_APPLICABLE = "the QQ equality needs within-subject answers to two questions in both orders; dataset D (aggregate tables) is covered by reports/q1_qq_equality.md, none of the tables here qualify"
PHI_NOTE = (
    "delta_LTP is a model-internal counterfactual on this table: no row is tolerance-first, so the tolerance "
    "angle phi is not identified by the data and the delta_LTP interval reflects the prior, the initialization "
    "and the Laplace approximation, not an observed order effect; Delta_order (exp:loss>exp:gain vs the reverse) "
    "is the identified interference quantity on the pairs variant"
)
ANALOG_NOTE = (
    "CPC18 is an analog: the response is retreat to the lower-variance option after experienced outcomes in a "
    "lab lottery task, not panic selling; the loss magnitude is the previous obtained payoff relative to the "
    "problem's largest absolute outcome (PLAN.md section 6)"
)

SMOKE_SETTINGS: dict[str, Any] = {"restarts": 1, "steps": 300, "lr": 0.02, "val_frac": 0.2, "early_stopping_patience": 3, "eval_every": 25, "lbfgs_polish": True, "lbfgs_steps": 20, "n_samples": 10, "svi_steps": 300, "svi_lr": 0.01, "svi_particles": 1}
SMOKE_SUBJECTS = 60
"""``--smoke``: 60 subjects, 300 steps, one restart (the compute rule for tests and timing)."""

FULL_SETTINGS: dict[str, Any] = dict(DEFAULT_FIT_SETTINGS)
"""The PLAN.md section 4 settings (5 restarts, 1500 steps, early stopping, L-BFGS polish, 50 draws)."""

MODEL_SETTINGS: dict[str, dict[str, Any]] = {"B5": {"n_samples": 0}}
"""Per-model overrides of the fit settings (B5 has no draws: external fit)."""

MODEL_KWARGS_DEFAULT: dict[str, dict[str, Any]] = {"B5": {"n_threads": 1}}
"""Constructor keywords per model (B5 single-threaded, as the recovery study)."""

# Measured on the smoke run of 2026-09-29 (CPC18 pairs, 30 subjects, 300 steps, one restart,
# ``--smoke``, one worker while three cores and 13 GB were held by the recovery grid): seconds of
# fit wall time per fit row per optimizer step (Adam steps + 3 x L-BFGS iterations; JIT
# compilation, the Laplace draws and the polish are inside the wall time, so at 11,000 fit rows
# the numbers are compile-inflated upper bounds), used by :func:`estimate_seconds` scaled by rows
# and steps. Only B1 and Q2 are measured (the 10-model smoke could not run on the memory-bound
# build machine); a model outside the table is estimated at Q2's cost and flagged. For reference,
# ``bre.recover.SECONDS_PER_STEP`` (N = 200 synthetic tables, 34,000 rows, 4 workers) gives per
# row and step: B1 6e-8, B4 1.2e-7, Q2 2.6e-6, Q4 7.6e-6, B2 (SVI) 8.8e-7. B2: per row per SVI
# step; B5: per row per external fit. Replace with new measurements from
# ``metrics.json["timing"]`` of a later run.
SECONDS_PER_ROW_STEP: dict[str, float] = {"B1": 2.2e-06, "Q2": 1.4e-05}
SECONDS_PER_ROW_EVAL: float = 7.0e-04
"""Seconds per test row of the held-out evaluation (bootstrap, interference draws), measured on
the same smoke run (median over its six jobs; compile-inflated on small tables)."""
GATE_MINUTES = 120.0


# ---------------------------------------------------------------------------------------------
# Fast weighted AUC (same value as bre.eval.auc)
# ---------------------------------------------------------------------------------------------


def _auc_groups(p: np.ndarray) -> np.ndarray:
    """Group index of every row among the sorted distinct values of ``p``."""
    return np.unique(np.asarray(p, dtype=np.float64), return_inverse=True)[1].reshape(-1)


def auc_from_groups(groups: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Mann-Whitney AUC with ties counted half, from precomputed value groups
    (``groups`` ascending in ``p``): ``sum_g pos_w[g] (neg_w below g + 0.5 neg_w[g]) / (W+ W-)``."""
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    pos = (y >= 0.5) & (w > 0)
    neg = (y < 0.5) & (w > 0)
    wp, wn = float(w[pos].sum()), float(w[neg].sum())
    if wp <= 0 or wn <= 0:
        return float("nan")
    n_g = int(groups.max()) + 1 if groups.size else 0
    neg_g = np.bincount(groups, weights=np.where(neg, w, 0.0), minlength=n_g)
    pos_g = np.bincount(groups, weights=np.where(pos, w, 0.0), minlength=n_g)
    below = np.concatenate([[0.0], np.cumsum(neg_g)[:-1]])
    return float(np.sum(pos_g * (below + 0.5 * neg_g)) / (wp * wn))


def auc_fast(p, y, w=None) -> float:
    """Vectorised weighted AUC, equal to :func:`bre.eval.auc` (Mann-Whitney, ties half)."""
    p = np.asarray(p, dtype=np.float64)
    w = np.ones(p.shape[0]) if w is None else np.asarray(w, dtype=np.float64)
    return auc_from_groups(_auc_groups(p), y, w)


# ---------------------------------------------------------------------------------------------
# Held-out evaluation (bre.eval metrics, fast bootstrap)
# ---------------------------------------------------------------------------------------------


def evaluate_rows(model, params, data: ModelData, rows: np.ndarray, *, n_boot: int = 1000, seed: int = 0, param_samples=None, posterior=None) -> dict[str, Any]:
    """The dict of :func:`bre.eval.evaluate_model` (metrics, reliability, bootstrap intervals,
    ``ll`` / ``p`` / ``subject_idx`` / ``rows`` of the held-out sell rows, posterior-predictive
    NLL for B2), with the subject bootstrap of the AUC through :func:`auc_from_groups`."""
    rows = np.asarray(rows, dtype=np.int64)
    sub = data.subset(rows)
    mask = sub.sell_mask()
    p = np.asarray(model.predict_proba(params, sub), dtype=np.float64)[mask]
    ll = np.asarray(model.log_lik(params, sub), dtype=np.float64)[mask]
    y, w, sidx = sub.y[mask], sub.w[mask], sub.subject_idx[mask]
    out = E.evaluate_predictions(p, y, ll, sidx, w, n_boot=0, seed=seed)
    out["auc"] = auc_fast(p, y, w)
    if p.shape[0] and n_boot > 0:
        groups = _auc_groups(p)
        for key, fn in (
            ("nll", lambda m: E.nll(ll, m)),
            ("brier", lambda m: E.brier_score(p, y, w * m)),
            ("ece", lambda m: E.ece(p, y, w * m)),
            ("auc", lambda m: auc_from_groups(groups, y, w * m)),
        ):
            lo, hi, _ = E.subject_bootstrap(fn, sidx, n_boot=n_boot, seed=seed)
            out[f"{key}_ci95"] = [lo, hi]
        out["bootstrap"] = {"n_draws": int(n_boot), "seed": int(seed), "unit": "subject"}
    out["rows"], out["ll"], out["p"], out["subject_idx"] = rows[mask], ll, p, sidx
    if posterior is not None and hasattr(model, "pointwise_log_lik"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            llm = model.pointwise_log_lik(posterior, sub)[:, mask]
        out["nll_posterior_predictive"] = float(-np.mean(np.log(np.mean(np.exp(llm), axis=0))))
    elif param_samples:
        lls = np.stack([np.asarray(model.log_lik(s, sub), dtype=np.float64)[mask] for s in param_samples])
        out["nll_samples_predictive"] = float(-np.mean(np.log(np.mean(np.exp(lls), axis=0))))
    return out


# ---------------------------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------------------------


def make_splits(data: ModelData, letters=DEFAULT_SPLITS, seed: int = 0) -> dict[str, dict[str, Any]]:
    """Letter -> ``{"name", "applicable", "reason", "train", "test"}`` (module docstring)."""
    out: dict[str, dict[str, Any]] = {}
    for s in letters:
        if s not in SPLIT_NAMES:
            raise ValueError(f"unknown split {s!r}; allowed {sorted(SPLIT_NAMES)}")
        entry: dict[str, Any] = {"name": SPLIT_NAMES[s], "applicable": True, "reason": None}
        if s == "a":
            entry["train"], entry["test"] = E.split_subjects(data, TEST_FRAC, seed)
        elif s == "b":
            tr, te = R.split_pairs_by_exp_tags(data)
            if te.size == 0:
                entry.update(applicable=False, reason=SPLIT_B_NOT_APPLICABLE, train=np.zeros(0, dtype=np.int64), test=np.zeros(0, dtype=np.int64))
            else:
                entry["train"], entry["test"] = tr, te
        elif s == "c":
            entry.update(applicable=False, reason=SPLIT_C_NOT_APPLICABLE, train=np.zeros(0, dtype=np.int64), test=np.zeros(0, dtype=np.int64))
        else:
            entry["train"], entry["test"] = E.split_temporal(data, TEST_FRAC)
        out[s] = entry
    return out


# ---------------------------------------------------------------------------------------------
# Structural test from the data: order effects by problem (pairs variant)
# ---------------------------------------------------------------------------------------------


def order_effects_by_problem(frame: pd.DataFrame, data: ModelData, rows: np.ndarray | None = None, *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Order effects of the two experienced outcomes by scenario (problem): among the sell rows
    with the ordered pair ``exp:loss > exp:gain`` (forward) or ``exp:gain > exp:loss``
    (reverse), per problem ``P(safe | forward) - P(safe | reverse)``; ``mean_over_problems``
    (problems with both orders), ``pooled`` (all rows), a subject bootstrap 95% interval on
    both, and the per-problem table. ``frame`` supplies ``scenario_id`` (same row order as
    ``data``)."""
    rows = data.sell_rows() if rows is None else np.intersect1d(np.asarray(rows), data.sell_rows())
    vocab = list(data.ctx_vocab)
    if "exp:loss" not in vocab or "exp:gain" not in vocab or rows.size == 0:
        return {"applicable": False, "reason": "no exp:loss / exp:gain ordered pairs in this table", "n_rows": 0}
    i_loss, i_gain = vocab.index("exp:loss"), vocab.index("exp:gain")
    ci0, ci1 = data.ctx_idx[rows, 0], data.ctx_idx[rows, 1]
    both = data.ctx_mask[rows].all(axis=1)
    fwd = both & (ci0 == i_loss) & (ci1 == i_gain)
    rev = both & (ci0 == i_gain) & (ci1 == i_loss)
    keep = fwd | rev
    if keep.sum() == 0:
        return {"applicable": False, "reason": "no mixed ordered pairs (exp:loss>exp:gain or the reverse) among the rows", "n_rows": 0}
    r = rows[keep]
    is_fwd = fwd[keep]
    y, sidx = data.y[r], data.subject_idx[r]
    scen = frame["scenario_id"].astype(str).to_numpy()[r]
    cells, cidx = np.unique(scen, return_inverse=True)
    n_c = len(cells)

    def per_problem(m: np.ndarray) -> np.ndarray:
        wf, wr = np.where(is_fwd, m, 0.0), np.where(~is_fwd, m, 0.0)
        sf, sr = np.bincount(cidx, weights=wf, minlength=n_c), np.bincount(cidx, weights=wr, minlength=n_c)
        yf, yr = np.bincount(cidx, weights=wf * y, minlength=n_c), np.bincount(cidx, weights=wr * y, minlength=n_c)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = yf / sf - yr / sr
        return np.where((sf > 0) & (sr > 0), d, np.nan)

    def pooled(m: np.ndarray) -> float:
        wf, wr = np.where(is_fwd, m, 0.0), np.where(~is_fwd, m, 0.0)
        return float(np.sum(wf * y) / wf.sum() - np.sum(wr * y) / wr.sum()) if wf.sum() > 0 and wr.sum() > 0 else float("nan")

    ones = np.ones(r.size)
    d = per_problem(ones)
    lo, hi, _ = E.subject_bootstrap(lambda m: float(np.nanmean(per_problem(m))) if np.isfinite(per_problem(m)).any() else float("nan"), sidx, n_boot=n_boot, seed=seed)
    plo, phi_, _ = E.subject_bootstrap(pooled, sidx, n_boot=n_boot, seed=seed)
    n_fwd = np.bincount(cidx, weights=is_fwd.astype(float), minlength=n_c)
    n_rev = np.bincount(cidx, weights=(~is_fwd).astype(float), minlength=n_c)
    table = [{"scenario": str(c), "delta_order": float(v), "n_forward": int(a), "n_reverse": int(b)} for c, v, a, b in zip(cells, d, n_fwd, n_rev)]
    return {
        "applicable": True,
        "definition": "per problem: P(safe | exp:loss > exp:gain) - P(safe | exp:gain > exp:loss); forward = the loss was experienced two trials back and the gain on the last trial",
        "n_rows": int(r.size),
        "n_forward_rows": int(is_fwd.sum()),
        "n_reverse_rows": int((~is_fwd).sum()),
        "n_problems": int(np.isfinite(d).sum()),
        "mean_over_problems": float(np.nanmean(d)) if np.isfinite(d).any() else float("nan"),
        "mean_over_problems_ci95": [lo, hi],
        "mean_abs_over_problems": float(np.nanmean(np.abs(d))) if np.isfinite(d).any() else float("nan"),
        "share_problems_positive": float(np.nanmean(d > 0)) if np.isfinite(d).any() else float("nan"),
        "pooled": pooled(ones),
        "pooled_ci95": [plo, phi_],
        "bootstrap": {"n_draws": int(n_boot), "seed": int(seed), "unit": "subject"},
        "problems": table,
    }


# ---------------------------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------------------------


def dataset_key(dataset: str, variant: str = "pairs") -> str:
    """``bre.realdata`` key of a ``--dataset`` / ``--variant`` pair."""
    if dataset not in DATASET_KEYS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(DATASET_KEYS)}")
    key = DATASET_KEYS[dataset]
    if key == "cpc18":
        if variant not in ("pairs", "singles"):
            raise ValueError("variant must be 'pairs' or 'singles'")
        return "cpc18_pairs" if variant == "pairs" else "cpc18"
    return key


_DATA_CACHE: dict[tuple[str, int | None, int], tuple[pd.DataFrame, ModelData]] = {}


def load_data(key: str, subjects: int | None, seed: int, *, validate: bool = True) -> tuple[pd.DataFrame, ModelData]:
    """:func:`bre.realdata.load_real`, cached per process (workers reuse the table across jobs)."""
    k = (key, subjects, int(seed))
    if k not in _DATA_CACHE:
        _DATA_CACHE[k] = R.load_real(key, subjects=subjects, seed=seed, validate=validate)
    return _DATA_CACHE[k]


def job_settings(model: str, base: dict[str, Any]) -> dict[str, Any]:
    """Fit settings of ``model``: ``base`` with :data:`MODEL_SETTINGS` merged on top."""
    return {**base, **MODEL_SETTINGS.get(model, {})}


def cache_matches(cached: dict[str, Any], job: dict[str, Any]) -> bool:
    """A cached job result is reused only when its settings, subsample and bootstrap agree."""
    want = _cache_key(job)
    return cached.get("cache_key") == want


def _cache_key(job: dict[str, Any]) -> dict[str, Any]:
    return jsonable({"dataset_key": job["dataset_key"], "subjects": job["subjects"], "seed": job["seed"], "split": job["split"], "model": job["model"], "settings": job["settings"], "model_kwargs": job["model_kwargs"], "n_boot": job["n_boot"]})


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Fit one model on the training rows of one split, score the test rows, run the model's
    structural checks, save the artifact and the cache file; returns the plain result dict.
    Runs in the driver process or in a worker."""
    warnings.filterwarnings("ignore")
    from bre.fit import artifact_from_fit, fit_model

    t0 = time.time()
    frame, data = load_data(job["dataset_key"], job["subjects"], job["seed"], validate=False)
    split = make_splits(data, (job["split"],), job["seed"])[job["split"]]
    train, test = split["train"], split["test"]
    model_name, settings, kw = job["model"], dict(job["settings"]), dict(job["model_kwargs"])
    log_path = job.get("log_path")
    fit = fit_model(model_name, data, rows=train, log_path=log_path, echo=False, model_kwargs=kw, **settings)
    t_fit = time.time()
    held = evaluate_rows(fit.model, fit.params, data, test, n_boot=int(job["n_boot"]), seed=int(job["seed"]), param_samples=fit.param_samples, posterior=fit.posterior)
    entry: dict[str, Any] = {k: v for k, v in held.items() if k not in ("rows", "ll", "p", "subject_idx")}
    tab = fit.restarts_table
    if "steps_run" in tab.columns:
        steps_total = int(tab["steps_run"].sum() + 3 * tab.get("polish_iterations", pd.Series(dtype=float)).fillna(0).sum())
    elif model_name == "B2":
        steps_total = int(settings["restarts"] * settings["svi_steps"])
    else:
        steps_total = int(settings["restarts"])
    n_fit_rows = int(len(fit.train_rows))
    entry.update({
        "family": MODEL_FAMILY[model_name],
        "n_params": int(fit.n_params),
        "n_params_population": int(fit.n_params_population),
        "train_nll": float(fit.train_nll),
        "val_nll": float(fit.val_nll),
        "best_restart": int(fit.best_restart),
        "fit_wall_time_s": float(fit.wall_time),
        "is_synthetic": bool(np.asarray(data.is_synthetic).any()),
        "restarts": jsonable(tab.to_dict(orient="records")),
        "n_train_rows": int(len(train)),
        "n_fit_rows": n_fit_rows,
        "n_test_rows": int(len(held["rows"])),
        "timing": {
            "fit_wall_time_s": float(fit.wall_time),
            "eval_seconds": 0.0,
            "steps_total": steps_total,
            "sec_per_row_step": float(fit.wall_time / max(n_fit_rows * steps_total, 1)),
        },
    })
    for k in ("waic", "loo", "laplace", "interference_train", "external_fit", "param_samples_note", "svi_diagnostics", "val_nll_posterior_predictive"):
        if k in fit.extra:
            entry[k] = fit.extra[k]
    if hasattr(fit.model, "interference_terms"):
        entry["interference_test"] = E.interference_ci(fit.model, fit.params, data, test, fit.param_samples)
        checks = E.model_structural_checks(fit.model, fit.params, data, test, n_boot=int(job["n_boot"]), seed=int(job["seed"]))
        entry["structural_checks_test"] = checks
        if "decoherence" in checks:
            entry["decoherence"] = checks["decoherence"]
        if "quarter_law" in checks:
            entry["quarter_law_test"] = checks["quarter_law"]
    if job.get("artifact_dir"):
        art = artifact_from_fit(fit, data, training_data_refs=[job["data_ref"]], metrics={"phase4": {"split": job["split"], "holdout": {k: v for k, v in entry.items() if k not in ("restarts", "reliability")}}}, notes=f"Phase 4 fit on the training rows of split ({job['split']}) of {job['dataset_key']}.")
        art.save(Path(job["artifact_dir"]) / job["split"] / model_name)
    entry["timing"]["eval_seconds"] = float(time.time() - t_fit)
    entry["timing"]["eval_sec_per_test_row"] = float((time.time() - t_fit) / max(len(held["rows"]), 1))
    out = {
        "split": job["split"], "model": model_name, "entry": entry,
        "ll": np.asarray(held["ll"]), "p": np.asarray(held["p"]), "subject_idx": np.asarray(held["subject_idx"]), "rows": np.asarray(held["rows"]),
        "params": to_numpy_tree(fit.params), "job_seconds": time.time() - t0, "cache_key": _cache_key(job),
    }
    if job.get("cache_path"):
        Path(job["cache_path"]).parent.mkdir(parents=True, exist_ok=True)
        with open(job["cache_path"], "wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def estimate_seconds(model: str, n_train_rows: int, n_test_rows: int, settings: dict[str, Any]) -> tuple[float, bool]:
    """``(seconds, measured)`` worst-case estimate of one job from :data:`SECONDS_PER_ROW_STEP`
    (no early stopping) and :data:`SECONDS_PER_ROW_EVAL`; ``measured`` is False when the
    model's cost is not in the table (the largest known cost is used)."""
    rows = int(n_train_rows) * (1.0 - float(settings.get("val_frac", 0.2)))
    measured = model in SECONDS_PER_ROW_STEP
    cost = SECONDS_PER_ROW_STEP.get(model, max(SECONDS_PER_ROW_STEP.values()) if SECONDS_PER_ROW_STEP else 0.0)
    if model == "B2":
        steps = int(settings["restarts"]) * int(settings["svi_steps"])
    elif getattr(MODEL_REGISTRY[model], "requires_external_fit", False):
        steps = int(settings["restarts"])
    else:
        steps = int(settings["restarts"]) * (int(settings["steps"]) + 3 * int(settings.get("lbfgs_steps", 0)))
    return cost * rows * steps + SECONDS_PER_ROW_EVAL * int(n_test_rows), measured


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------


def run_phase4(
    dataset: str = "cpc18",
    *,
    variant: str = "pairs",
    subjects: int | None = None,
    seed: int = 0,
    models=DEFAULT_MODELS,
    splits=DEFAULT_SPLITS,
    settings: dict[str, Any] | None = None,
    model_kwargs: dict[str, dict[str, Any]] | None = None,
    n_boot: int = 1000,
    name: str | None = None,
    out_dir: str | Path | None = None,
    runs_dir: str | Path | None = None,
    workers: int = 1,
    refit: bool = False,
    save_artifacts: bool = True,
    smoke: bool = False,
    estimate_only: bool = False,
    echo: bool = True,
    command: str = "",
) -> dict[str, Any]:
    """Run the evaluation (module docstring) and write the outputs; returns the results dict."""
    t_start = time.time()
    key = dataset_key(dataset, variant)
    if smoke:
        subjects = SMOKE_SUBJECTS if subjects is None else subjects
        base = {**SMOKE_SETTINGS, **(settings or {})}
    else:
        base = {**FULL_SETTINGS, **(settings or {})}
    base["seed"] = int(seed)
    models = [m for m in models]
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(f"unknown model(s) {unknown}")
    name = name or (key + (f"-n{subjects}" if subjects else "") + ("-smoke" if smoke else ""))
    out_dir = Path(out_dir) if out_dir else REPORTS_DIR / name
    runs_dir = Path(runs_dir) if runs_dir else RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / "phase4.log"
    mk = {**{m: dict(v) for m, v in MODEL_KWARGS_DEFAULT.items()}, **{m: dict(v) for m, v in (model_kwargs or {}).items()}}
    _log(log_path, f"[phase4] {name}: dataset={key} subjects={subjects} seed={seed} models={models} splits={list(splits)} workers={workers} smoke={smoke} settings={json.dumps(jsonable(base))}", echo)
    frame, data = load_data(key, subjects, seed, validate=True)
    if bool(np.asarray(data.is_synthetic).any()):
        raise S.SchemaError("phase4 runs on real data only")
    split_info = make_splits(data, splits, seed)
    for s, info in split_info.items():
        _log(log_path, f"[phase4] split ({s}) {info['name']}: " + (f"{len(info['train'])} train rows, {len(info['test'])} test rows" if info["applicable"] else f"not applicable — {info['reason']}"), echo)
    data_ref = "loader:cpc18(pairs=True) [data/raw/cpc18/cpc18_calibration_raw.csv.gz]" if key == "cpc18_pairs" else f"data/processed/{R.DATASETS[key]['file']}"
    jobs: list[dict[str, Any]] = []
    for s, info in split_info.items():
        if not info["applicable"]:
            continue
        for m in models:
            jobs.append({
                "dataset_key": key, "subjects": subjects, "seed": int(seed), "split": s, "model": m,
                "settings": job_settings(m, base), "model_kwargs": mk.get(m, {}), "n_boot": int(n_boot),
                "log_path": str(log_path), "cache_path": str(runs_dir / "jobs" / f"{s}__{m}.pkl"),
                "artifact_dir": str(runs_dir / "artifacts") if save_artifacts else None, "data_ref": data_ref,
                "n_train_rows": int(len(info["train"])), "n_test_rows": int(len(info["test"])),
            })
    order = {"Q4": 0, "Q3": 1, "Q2": 2, "B2": 3, "B3": 4, "B6": 5, "Q5": 6, "B4": 7, "B5": 8, "B1": 9}
    jobs.sort(key=lambda j: (order.get(j["model"], 5), -j["n_train_rows"]))
    est_rows = []
    total_est = 0.0
    for j in jobs:
        sec, measured = estimate_seconds(j["model"], j["n_train_rows"], j["n_test_rows"], j["settings"])
        est_rows.append({"split": j["split"], "model": j["model"], "n_train_rows": j["n_train_rows"], "n_test_rows": j["n_test_rows"], "estimated_seconds": sec, "measured_cost": measured})
        total_est += sec
    est_minutes = total_est / max(int(workers), 1) / 60.0
    estimate = {"jobs": est_rows, "total_minutes_sequential": total_est / 60.0, "workers": int(workers), "total_minutes": est_minutes, "gate_minutes": GATE_MINUTES, "over_gate": bool(est_minutes > GATE_MINUTES), "basis": "SECONDS_PER_ROW_STEP x fit rows x optimizer steps (no early stopping) + SECONDS_PER_ROW_EVAL x test rows; measured on the smoke run" if SECONDS_PER_ROW_STEP else "no measured costs in SECONDS_PER_ROW_STEP yet: run --smoke first"}
    _log(log_path, f"[phase4] {len(jobs)} jobs; wall-time estimate {est_minutes:.0f} min with {workers} worker(s)" + (" — OVER THE 2-HOUR GATE: split the run (--models subsets, fewer restarts/steps, --subjects)" if estimate["over_gate"] else ""), echo)
    if estimate_only:
        for r in est_rows:
            _log(log_path, f"[phase4]   ({r['split']}) {r['model']}: {r['estimated_seconds'] / 60:.1f} min for {r['n_train_rows']} train / {r['n_test_rows']} test rows" + ("" if r["measured_cost"] else " (cost not measured; largest known)"), echo)
        return {"name": name, "estimate": estimate, "jobs": len(jobs)}

    results_by_job: dict[tuple[str, str], dict[str, Any]] = {}
    todo: list[dict[str, Any]] = []
    for j in jobs:
        cp = Path(j["cache_path"])
        if cp.exists() and not refit:
            with open(cp, "rb") as fh:
                cached = pickle.load(fh)
            if cache_matches(cached, j):
                results_by_job[(j["split"], j["model"])] = cached
                _log(log_path, f"[phase4] cached ({j['split']}) {j['model']}: nll={cached['entry']['nll']:.4f} (reused; --refit to redo)", echo)
                continue
            _log(log_path, f"[phase4] cache of ({j['split']}) {j['model']} has other settings; refitting", echo)
        todo.append(j)
    failed: list[dict[str, Any]] = []

    def done(j: dict[str, Any], r: dict[str, Any]) -> None:
        results_by_job[(j["split"], j["model"])] = r
        e = r["entry"]
        _log(log_path, f"[phase4] done ({j['split']}) {j['model']}: nll={e['nll']:.4f} CI={np.round(e['nll_ci95'], 4).tolist()} brier={e['brier']:.4f} auc={e['auc']:.4f} n_params={e['n_params']} ({r['job_seconds']:.0f}s; fit {e['fit_wall_time_s']:.0f}s) [{len(results_by_job)}/{len(jobs)}]", echo)

    if int(workers) > 1 and todo:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from bre.recover import _worker_init

        _worker_init()
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=mp.get_context("spawn"), initializer=_worker_init) as pool:
            futures = {pool.submit(run_job, j): j for j in todo}
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    done(j, fut.result())
                except Exception as exc:  # noqa: BLE001
                    failed.append({"split": j["split"], "model": j["model"], "error": f"{type(exc).__name__}: {exc}"})
                    _log(log_path, f"[phase4] FAILED ({j['split']}) {j['model']}: {type(exc).__name__}: {exc}", echo)
    else:
        for j in todo:
            try:
                done(j, run_job(j))
            except Exception as exc:  # noqa: BLE001
                failed.append({"split": j["split"], "model": j["model"], "error": f"{type(exc).__name__}: {exc}"})
                _log(log_path, f"[phase4] FAILED ({j['split']}) {j['model']}: {type(exc).__name__}: {exc}", echo)

    results = assemble_results(frame, data, key, split_info, results_by_job, models, n_boot=int(n_boot), seed=int(seed))
    results.update({
        "name": name, "dataset": dataset, "dataset_key": key, "variant": variant if key.startswith("cpc18") else None, "subjects": subjects, "seed": int(seed),
        "smoke": bool(smoke), "fit_settings": jsonable(base), "model_settings": MODEL_SETTINGS, "model_kwargs": mk, "models_requested": models,
        "failed": failed, "estimate": estimate, "command": command, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": time.time() - t_start, "data_ref": data_ref, "notes": {"analog": ANALOG_NOTE, "delta_ltp": PHI_NOTE},
    })
    write_outputs(results, out_dir)
    _log(log_path, f"[phase4] verdict: {results['verdict']['verdict']}; wrote {out_dir} ({results['wall_time_s']:.0f}s)", echo)
    return results


def assemble_results(frame: pd.DataFrame, data: ModelData, key: str, split_info: dict[str, dict[str, Any]], results_by_job: dict[tuple[str, str], dict[str, Any]], models: list[str], *, n_boot: int, seed: int) -> dict[str, Any]:
    """Per-split tables, paired differences against the best classical model, structural tests
    and the decision rule from the per-job results."""
    out: dict[str, Any] = {"is_synthetic": bool(np.asarray(data.is_synthetic).any()), "data": data.summary(), "splits": {}, "structural": {}, "timing": {}}
    best_q_interference: dict[str, Any] | None = None
    split_b: dict[str, dict[str, Any]] | None = None
    timing_rows = []
    for s, info in split_info.items():
        sres: dict[str, Any] = {"name": info["name"], "letter": s, "applicable": info["applicable"], "reason": info["reason"], "n_train_rows": int(len(info["train"])), "n_test_rows": int(len(info["test"])), "models": {}, "table": []}
        if not info["applicable"]:
            out["splits"][s] = sres
            continue
        per_model = {m: results_by_job[(s, m)] for m in models if (s, m) in results_by_job}
        entries = {m: dict(r["entry"]) for m, r in per_model.items()}
        classical = [m for m in entries if entries[m]["family"] == "classical"]
        quantum = [m for m in entries if entries[m]["family"] == "quantum"]
        best_c = min(classical, key=lambda k: entries[k]["nll"]) if classical else None
        best_q = min(quantum, key=lambda k: entries[k]["nll"]) if quantum else None
        if best_c is not None:
            ll_c, sidx_c = per_model[best_c]["ll"], per_model[best_c]["subject_idx"]
            for m, r in per_model.items():
                if m != best_c and len(r["ll"]) == len(ll_c):
                    entries[m]["nll_diff_vs_best_classical"] = {**E.paired_nll_difference(r["ll"], ll_c, r["subject_idx"], n_boot=n_boot, seed=seed), "reference": best_c}
                    gain = E.predictive_gain_per_subject(r["ll"], ll_c, r["subject_idx"], data.subject_ids)
                    entries[m]["predictive_gain_vs_best_classical"] = {"reference": best_c, "mean": float(gain["gain"].mean()), "median": float(gain["gain"].median()), "share_positive": float((gain["gain"] > 0).mean()), "n_subjects": int(len(gain)), "quantiles": {str(q): float(gain["gain"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}
        for m, e in entries.items():
            timing_rows.append({"split": s, "model": m, **e["timing"], "n_train_rows": e["n_train_rows"], "n_fit_rows": e["n_fit_rows"], "n_test_rows": e["n_test_rows"], "job_seconds": per_model[m]["job_seconds"], "n_params_population": e["n_params_population"]})
        table = E.compare_models(entries)
        if len(table):
            table["fit_seconds"] = [entries[m]["fit_wall_time_s"] for m in table["model"]]
            table["gain_share_positive"] = [(entries[m].get("predictive_gain_vs_best_classical") or {}).get("share_positive", float("nan")) for m in table["model"]]
        sres.update({"models": entries, "table": table.to_dict(orient="records"), "best_classical": best_c, "best_quantum": best_q})
        for m, e in entries.items():
            if e.get("structural_checks_test"):
                out["structural"].setdefault("model_checks", []).append({"split": s, "model": m, **e["structural_checks_test"]})
        if s == "b":
            split_b = {m: {**entries[m], "ll": per_model[m]["ll"], "subject_idx": per_model[m]["subject_idx"]} for m in entries}
            if best_q is not None:
                best_q_interference = entries[best_q].get("interference_test")
        out["splits"][s] = sres
    out["timing"] = {"jobs": timing_rows}
    # structural tests from the data
    b_info = split_info.get("b")
    rows_b = b_info["test"] if (b_info is not None and b_info["applicable"]) else None
    out["structural"]["order_effects_by_problem"] = order_effects_by_problem(frame, data, rows_b, n_boot=n_boot, seed=seed) if rows_b is not None else {"applicable": False, "reason": SPLIT_B_NOT_APPLICABLE}
    out["structural"]["ltp_violation_from_data"] = {"applicable": False, "reason": LTP_NOT_APPLICABLE}
    out["structural"]["qq_equality"] = {"applicable": False, "reason": QQ_NOT_APPLICABLE}
    out["structural"]["delta_ltp_note"] = PHI_NOTE
    # decision rule
    if split_b is not None and split_b:
        out["verdict"] = E.decision_rule(split_b, best_q_interference, is_real_data=not out["is_synthetic"], n_boot=n_boot, seed=seed)
    else:
        reason = (b_info["reason"] if b_info is not None and not b_info["applicable"] else "split (b) was not evaluated (no model finished on it)")
        out["verdict"] = {"verdict": E.VERDICT_NONE, "details": {"reason": reason, "rule": E.DECISION_RULE_TEXT, "conditions": {}, "data_that_would_resolve_it": "within-subject responses to ordered context pairs with both question orders on the shared design (intake battery, instrument/; or the gated UAS / LISS panels, data/REGISTRATION.md)"}}
    out["verdict_data"] = f"{key.replace('_pairs', ' pairs').replace('_', ' ')} split (b)" if (b_info is not None and b_info["applicable"]) else f"{key}: split (b) not applicable"
    return out


# ---------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------


def verdict_payload(results: dict[str, Any]) -> dict[str, Any]:
    """The ``verdict.json`` dict: the exact verdict sentence and the numbers behind it."""
    v = results["verdict"]
    d = v.get("details", {})
    ici = d.get("interference_ltp_ci95")
    return {
        "verdict": v["verdict"],
        "best_quantum": d.get("best_quantum"),
        "best_classical": d.get("best_classical"),
        "nll_diff": d.get("nll_diff_quantum_minus_classical"),
        "ci95": d.get("nll_diff_ci95"),
        "interference_ci": ici,
        "n_params_q": d.get("n_params_quantum"),
        "n_params_c": d.get("n_params_classical"),
        "data": results["verdict_data"],
        "nll_quantum": d.get("nll_quantum"),
        "nll_classical": d.get("nll_classical"),
        "conditions": d.get("conditions", {}),
        "failed_conditions": d.get("failed_conditions"),
        "served_model": d.get("served_model"),
        "served_model_family": d.get("served_model_family"),
        "reason": d.get("reason"),
        "is_real_data": d.get("is_real_data", not results.get("is_synthetic", False)),
        "n_test_rows_split_b": (results["splits"].get("b") or {}).get("n_test_rows"),
        "n_train_rows_split_b": (results["splits"].get("b") or {}).get("n_train_rows"),
        "subjects": results.get("subjects"),
        "smoke": results.get("smoke", False),
        "dataset_key": results.get("dataset_key"),
        "generated_at": results.get("generated_at"),
        "rule": E.DECISION_RULE_TEXT,
    }


def _f(v: Any, fmt: str = ".4f") -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "nan" if not np.isfinite(x) else format(x, fmt)


def _ci_str(ci: Any, fmt: str = ".4f") -> str:
    if not ci or len(ci) != 2:
        return "n/a"
    return f"[{_f(ci[0], fmt)}, {_f(ci[1], fmt)}]"


def table_markdown(results: dict[str, Any]) -> str:
    lines = [f"# Phase 4 tables: {results['name']}", ""]
    lines.append(f"Dataset `{results['dataset_key']}` ({'SYNTHETIC' if results.get('is_synthetic') else 'real'} data), {results['data']['n_subjects']} subjects, {results['data']['n_sell_rows']} sell rows" + (f" (subject subsample {results['subjects']}, seed {results['seed']})" if results.get("subjects") else "") + f"; {'SMOKE settings' if results.get('smoke') else 'full settings'}: {json.dumps(results['fit_settings'])}. Generated by `bre.phase4`; no hand-typed numbers.")
    lines.append("")
    lines.append(results["notes"]["analog"])
    lines.append("")
    for s, sres in results["splits"].items():
        head = f"## Split ({s}) {sres['name']}"
        if not sres["applicable"]:
            lines += [head, "", f"Not applicable: {sres['reason']}", ""]
            continue
        lines += [head + f": {sres['n_train_rows']} train rows, {sres['n_test_rows']} test rows", ""]
        if not sres["table"]:
            lines += ["(no model finished on this split)", ""]
            continue
        lines.append("| model | family | n_params | n_params_population | NLL | NLL CI95 | Brier | ECE | AUC | NLL (predictive draws) | elpd_waic/resp | NLL diff vs best classical | diff CI95 | subjects with gain > 0 | fit s |")
        lines.append("|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---:|")
        for r in sres["table"]:
            lines.append(f"| {r['model']} | {r['family']} | {int(r['n_params'])} | {int(r['n_params_population'])} | {_f(r['nll'])} | {_ci_str([r['nll_lo'], r['nll_hi']])} | {_f(r['brier'])} | {_f(r['ece'])} | {_f(r['auc'])} | {_f(r['nll_ppd'])} | {_f(r['elpd_waic_per_response'])} | {_f(r['nll_diff_vs_best_classical'])} | {_ci_str([r['nll_diff_lo'], r['nll_diff_hi']])} | {_f(r['gain_share_positive'], '.2f')} | {_f(r['fit_seconds'], '.0f')} |")
        lines.append("")
        lines.append(f"Best classical: {sres.get('best_classical')}; best Q-model: {sres.get('best_quantum')}. NLL = held-out negative log-likelihood per response (lower is better); the difference and its subject-bootstrap CI are against the best classical model of the split (negative = better than it).")
        lines.append("")
    if results.get("failed"):
        lines += ["## Failed jobs", ""] + [f"- ({f['split']}) {f['model']}: {f['error']}" for f in results["failed"]] + [""]
    return "\n".join(lines)


def structural_markdown(results: dict[str, Any]) -> str:
    st = results["structural"]
    lines = [f"# Phase 4 structural tests: {results['name']}", ""]
    lines.append(f"Dataset `{results['dataset_key']}` ({'SYNTHETIC' if results.get('is_synthetic') else 'real'} data). Generated by `bre.phase4`; no hand-typed numbers.")
    lines.append("")
    oe = st.get("order_effects_by_problem", {})
    lines.append("## Order effects by scenario (from the data, split (b) test rows)")
    lines.append("")
    if oe.get("applicable"):
        lines.append(f"- definition: {oe['definition']}")
        lines.append(f"- rows: {oe['n_rows']} ({oe['n_forward_rows']} forward, {oe['n_reverse_rows']} reverse) over {oe['n_problems']} problems with both orders")
        lines.append(f"- mean over problems: {_f(oe['mean_over_problems'])}, CI95 {_ci_str(oe['mean_over_problems_ci95'])}; mean |delta|: {_f(oe['mean_abs_over_problems'])}; share of problems with delta > 0: {_f(oe['share_problems_positive'], '.2f')}")
        lines.append(f"- pooled over rows: {_f(oe['pooled'])}, CI95 {_ci_str(oe['pooled_ci95'])} (subject bootstrap, {oe['bootstrap']['n_draws']} draws)")
        top = sorted([p for p in oe["problems"] if np.isfinite(p["delta_order"])], key=lambda p: -abs(p["delta_order"]))[:10]
        if top:
            lines += ["", "| problem | delta_order | n forward | n reverse |", "|---|---|---:|---:|"] + [f"| {p['scenario']} | {_f(p['delta_order'])} | {p['n_forward']} | {p['n_reverse']} |" for p in top]
    else:
        lines.append(f"- not applicable: {oe.get('reason')}")
    lines.append("")
    lines.append("## LTP violation and QQ equality")
    lines.append("")
    lines.append(f"- LTP violation from the data: not applicable — {st['ltp_violation_from_data']['reason']}")
    lines.append(f"- QQ equality: not applicable — {st['qq_equality']['reason']}")
    lines.append("")
    lines.append("## Model-level checks (Q-models, held-out rows of each split)")
    lines.append("")
    lines.append(f"Note: {st['delta_ltp_note']}")
    lines.append("")
    for s, sres in results["splits"].items():
        for m, e in (sres.get("models") or {}).items():
            it = e.get("interference_test")
            if it:
                lines.append(f"- {m} on split ({s}) ({it['n_rows']} rows): mean delta_LTP = {_f(it['ltp_mean'])} (CI95 over {it.get('n_samples', 0)} draws {_ci_str(it.get('ltp_mean_ci95'))}, excludes zero: {it.get('ltp_ci_excludes_zero')}), mean |delta_LTP| = {_f(it['ltp_abs_mean'])}; mean Delta_order = {_f(it['order_mean'])} (CI95 {_ci_str(it.get('order_mean_ci95'))}) over {it['n_pair_rows']} pair rows" + ("; order effect zero by construction" if it.get("order_zero_by_construction") else "") + ("; delta_LTP undefined for this model" if not it.get("ltp_defined", True) else ""))
            if e.get("decoherence"):
                d = e["decoherence"]
                lines.append(f"- {m} on split ({s}): decoherence rates gamma_i over {d['n_subjects']} subjects: median {_f(d['gamma_quantiles']['0.5'], '.3f')}, 5-95% {_f(d['gamma_quantiles']['0.05'], '.3f')}-{_f(d['gamma_quantiles']['0.95'], '.3f')}; share near-classical (gamma > 3) {_f(d['share_near_classical'], '.2f')}, near-coherent (gamma < 0.05) {_f(d['share_near_coherent'], '.2f')}; population gamma_mu {_f(d['gamma_mu'], '.3f')}, sigma {_f(d['sigma_gamma'], '.3f')}")
            if e.get("quarter_law_test"):
                q = e["quarter_law_test"]
                lines.append(f"- {m} on split ({s}): quarter-law check: mean |q| = {_f(q.get('mean_abs_q'))}, CI95 {_ci_str(q.get('ci95'))} vs reference {q.get('reference')}; share of q > 0: {_f(q.get('share_q_positive'), '.2f')} ({q.get('note')})")
    lines.append("")
    return "\n".join(lines)


def verdict_markdown(results: dict[str, Any]) -> str:
    v = results["verdict"]
    d = v.get("details", {})
    pay = verdict_payload(results)
    lines = [f"# Phase 4 verdict: {results['name']}", ""]
    lines.append(f"**{v['verdict']}**")
    lines.append("")
    lines.append(f"Data: {pay['data']} ({'SYNTHETIC' if results.get('is_synthetic') else 'real'} data; {results['data']['n_subjects']} subjects" + (f", subsample of {results['subjects']}" if results.get("subjects") else "") + (", SMOKE settings — not a result" if results.get("smoke") else "") + "). Generated by `bre.phase4`.")
    lines.append("")
    if d.get("conditions"):
        for k, ok in d["conditions"].items():
            lines.append(f"- {k}: {'yes' if ok else 'no'}")
        lines.append(f"- best Q-model {d.get('best_quantum')} (NLL {_f(d.get('nll_quantum'))}, {d.get('n_params_quantum')} params) vs best classical {d.get('best_classical')} (NLL {_f(d.get('nll_classical'))}, {d.get('n_params_classical')} params); NLL difference (Q - classical) {_f(d.get('nll_diff_quantum_minus_classical'))}, CI95 {_ci_str(d.get('nll_diff_ci95'))}")
        lines.append(f"- interference (delta_LTP) CI95 of the best Q-model on the split (b) test rows: {_ci_str(d.get('interference_ltp_ci95'))}")
        lines.append(f"- served model (lowest held-out NLL on split (b)): {d.get('served_model')} ({d.get('served_model_family')})")
    if d.get("reason"):
        lines.append(f"- reason: {d['reason']}")
    if d.get("failed_conditions"):
        lines.append(f"- failed conditions: {', '.join(d['failed_conditions'])}")
    if d.get("data_that_would_resolve_it"):
        lines.append(f"- data that would resolve it: {d['data_that_would_resolve_it']}")
    lines.append("")
    lines.append(f"Note on the interference condition: {results['notes']['delta_ltp']}")
    lines.append("")
    lines.append(results["notes"]["analog"])
    lines.append("")
    lines.append(f"Decision rule (PLAN.md section 6, verbatim): {E.DECISION_RULE_TEXT}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(results: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(jsonable(results), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "table.md").write_text(table_markdown(results), encoding="utf-8")
    (out_dir / "structural.md").write_text(structural_markdown(results), encoding="utf-8")
    (out_dir / "verdict.json").write_text(json.dumps(jsonable(verdict_payload(results)), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "verdict.md").write_text(verdict_markdown(results), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bre.phase4", description="Phase 4 pre-registered evaluation on real data (PLAN.md section 6).")
    parser.add_argument("--config", default=None, help="experiments/p4-*.yaml run description (command-line flags override it)")
    parser.add_argument("--dataset", default=None, choices=sorted(DATASET_KEYS))
    parser.add_argument("--variant", default=None, choices=["pairs", "singles"])
    parser.add_argument("--subjects", type=int, default=None, help="deterministic subject subsample (default: all)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None, help="registered model ids or 'all'")
    parser.add_argument("--splits", nargs="+", default=None, help="letters a b c d (default all four; c is reported as not applicable)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--svi-steps", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default=None, help="output directory (default reports/phase4/<name>)")
    parser.add_argument("--runs-dir", default=None, help="fits, caches, logs (default runs/phase4/<name>)")
    parser.add_argument("--smoke", action="store_true", help="60 subjects, 300 steps, one restart (tests and timing)")
    parser.add_argument("--refit", action="store_true", help="ignore cached job results")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--estimate-only", action="store_true", help="print the wall-time estimate and exit")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    cfg: dict[str, Any] = load_config(args.config) if args.config else {}
    fit_cfg = dict(cfg.get("fit") or {})
    for k, v in (("steps", args.steps), ("restarts", args.restarts), ("svi_steps", args.svi_steps), ("n_samples", args.n_samples)):
        if v is not None:
            fit_cfg[k] = v
    models = args.models or cfg.get("models") or list(DEFAULT_MODELS)
    if isinstance(models, str):
        models = [models]
    if len(models) == 1 and str(models[0]).lower() == "all":
        models = list(DEFAULT_MODELS)
    boot = dict(cfg.get("bootstrap") or {})
    out = dict(cfg.get("output") or {})
    smoke = bool(args.smoke or cfg.get("smoke", False))
    run_phase4(
        args.dataset or cfg.get("dataset", "cpc18"),
        variant=args.variant or cfg.get("variant", "pairs"),
        subjects=args.subjects if args.subjects is not None else cfg.get("subjects"),
        seed=int(args.seed if args.seed is not None else cfg.get("seed", 0)),
        models=list(models),
        splits=tuple(args.splits or cfg.get("splits") or DEFAULT_SPLITS),
        settings=fit_cfg,
        model_kwargs=dict(cfg.get("model_kwargs") or {}),
        n_boot=int(args.n_boot if args.n_boot is not None else boot.get("n_draws", 1000)),
        name=args.name or cfg.get("name"),
        out_dir=resolve_path(args.out) if args.out else (resolve_path(out["dir"]) if out.get("dir") else None),
        runs_dir=resolve_path(args.runs_dir) if args.runs_dir else (resolve_path(out["runs_dir"]) if out.get("runs_dir") else None),
        workers=int(args.workers if args.workers is not None else cfg.get("workers", 1)),
        refit=bool(args.refit),
        save_artifacts=not args.no_artifacts,
        smoke=smoke,
        estimate_only=bool(args.estimate_only),
        echo=not args.quiet,
        command="python -m bre.phase4 " + " ".join(argv if argv is not None else sys.argv[1:]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MODELS",
    "DEFAULT_SPLITS",
    "SECONDS_PER_ROW_STEP",
    "SMOKE_SETTINGS",
    "SMOKE_SUBJECTS",
    "SPLIT_C_NOT_APPLICABLE",
    "assemble_results",
    "auc_fast",
    "dataset_key",
    "estimate_seconds",
    "evaluate_rows",
    "load_data",
    "make_splits",
    "order_effects_by_problem",
    "run_job",
    "run_phase4",
    "verdict_payload",
    "write_outputs",
]
