"""Pre-registered evaluation of PLAN.md section 6: splits, metrics, structural tests, verdict.

Metrics (per model, on the held-out rows of a split): held-out NLL per response (primary,
``-mean(ll)`` over the sell rows with the likelihood weights left in), Brier score, expected
calibration error with 10 equal-width probability bins and its reliability table, AUC
(Mann-Whitney, ties counted half), the per-subject predictive gain against a reference model
(mean log-likelihood per response of the model minus the reference, per subject) and
subject-level bootstrap 95% intervals (``n_boot`` resamples of subjects with replacement, every
metric recomputed with the resampled subject multiplicities as row weights; seedable). For B2
the posterior-predictive NLL (log of the mean likelihood over posterior draws) is reported next
to the plug-in NLL, and WAIC / PSIS-LOO come from the fit (:mod:`bre.fit`, ``B2.waic`` /
``B2.loo``).

Splits (PLAN.md section 6), each returning ``(train_rows, test_rows)`` as row indices:

(a) ``subjects``: 20% of the subjects held out, stratified by ``dataset`` (test rows = every row
    of a held-out subject; train rows = every row of the others).
(b) ``pairs``: train on the single-context and ``none`` conditions, test on the ordered pairs
    (test rows = sell rows with two context tags; train rows = every row with at most one).
    Classical models use their additive context effects here; B4 composes evidence without
    pair parameters and is the natural rival.
(c) ``order``: train on one question order (``train_order``, default tolerance-first), test on
    the other (sell rows).
(d) ``temporal``: per subject, rows ordered by ``(session_id, position_in_session)``; the last
    ``test_frac`` of the sell rows are the test set, everything earlier trains.

Structural tests: the LTP violation estimated from the data (per scenario, ``P(sell |
scenario-first) - P(sell | tolerance-first)``, the latter being ``sum_a P(a, sell)``, with a
subject bootstrap CI), order effects by scenario (``P(sell | c1, c2) - P(sell | c2, c1)``), the
interference-term interval of a Q-model (mean ``delta_LTP`` and mean ``Delta_order`` over the
rows, 2.5/97.5 percentiles over ``param_samples``) and, per Q-model, the model-level checks of
:func:`model_structural_checks`: the decoherence-rate distribution of a model with per-subject
``log_gamma`` (Q4), the quarter-law check of a model with ``quarter_law_check`` (Q5: mean ``|q|``
with a subject bootstrap interval next to the reference 0.25), whether ``delta_LTP`` is defined
for the model at all (Q5 returns NaN: no non-commuting tolerance measurement) and whether
``Delta_order`` is identically zero on every ordered-pair row (Q3 and Q5 compose contexts
symmetrically, so their zero order effect is a property of the model, not a fitted result).

Decision rule: :func:`decision_rule` implements the PLAN.md section 6 text verbatim; its
``verdict`` is one of :data:`VERDICT_SUPPORTED` and :data:`VERDICT_NONE`. An interference
interval that is missing or NaN (no draws, or a model whose ``delta_LTP`` is undefined) does
**not** exclude zero: the condition fails and the verdict is :data:`VERDICT_NONE`.

Any registered model (``bre.registry.MODEL_REGISTRY``) can be listed under ``models:``; the
per-model extras are picked by capability (``interference_terms``, ``quarter_law_check``,
``log_gamma`` in the parameters), not by name.

Config-driven use: ``python -m bre.eval experiments/<name>.yaml`` fits every listed model on the
training rows of every listed split (through :func:`bre.fit.fit_model`), evaluates the test
rows, runs the structural tests, applies the decision rule to split (b) and writes a metrics
JSON and a markdown table under ``reports/eval/``. Without an argument it runs every experiment
listed in ``experiments/eval_all.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from bre import design as _design
from bre import schema as S
from bre.artifact import jsonable
from bre.fit import (
    FIT_ALL,
    PROJECT_ROOT,
    _log,
    artifact_from_fit,
    experiments_listed,
    fit_model,
    fit_settings,
    interference_summary,
    load_config,
    load_dataset,
    nll_per_response,
    resolve_path,
)
from bre.models.data import ModelData
from bre.registry import MODEL_FAMILY

EVAL_ALL = PROJECT_ROOT / "experiments" / "eval_all.yaml"
EVAL_RUNS_DIR = PROJECT_ROOT / "runs" / "eval"
REPORTS_EVAL_DIR = PROJECT_ROOT / "reports" / "eval"

VERDICT_SUPPORTED = "Q-model supported"
VERDICT_NONE = "no evidence of a quantum-probability advantage in the available data"
"""The two verdict sentences of PLAN.md section 6 (verbatim; the second keeps the owner's phrase)."""

DECISION_RULE_TEXT = (
    '"Q-model supported" only if, on real data, the best Q-model beats the best classical baseline on '
    "split (b) by held-out NLL with a bootstrap 95% CI excluding zero, at matched or lower parameter "
    'count, and the interference-term CI excludes zero. Otherwise the conclusion is "no evidence of a '
    'quantum-probability advantage in the available data", and the report names the data that would '
    "resolve it. The dashboard serves whichever model wins on held-out NLL; if that is a classical "
    "model, the screen says so."
)
"""PLAN.md section 6 decision rule, verbatim (printed on the transparency page)."""

SPLIT_TYPES: dict[str, str] = {"a": "subjects", "b": "pairs", "c": "order", "d": "temporal", "none": "none"}
"""Letter aliases of PLAN.md section 6 -> split names accepted by :func:`split_rows`."""

N_ECE_BINS = 10


# ---------------------------------------------------------------------------------------------
# Metrics on probability vectors
# ---------------------------------------------------------------------------------------------


def _weights(w, n: int) -> np.ndarray:
    return np.ones(n, dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64)


def brier_score(p, y, w=None) -> float:
    """Weighted mean squared error ``sum w (p - y)^2 / sum w``."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = _weights(w, p.shape[0])
    if w.sum() <= 0:
        return float("nan")
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def reliability_table(p, y, w=None, n_bins: int = N_ECE_BINS) -> pd.DataFrame:
    """Equal-width bins on ``[0, 1]``: ``bin, lo, hi, n, weight, mean_p, frac_pos, gap``."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = _weights(w, p.shape[0])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        wb = w[m]
        tot = float(wb.sum())
        if tot > 0:
            mp = float(np.sum(wb * p[m]) / tot)
            fp = float(np.sum(wb * y[m]) / tot)
        else:
            mp, fp = float("nan"), float("nan")
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()), "weight": tot, "mean_p": mp, "frac_pos": fp, "gap": (fp - mp) if tot > 0 else float("nan")})
    return pd.DataFrame(rows)


def ece(p, y, w=None, n_bins: int = N_ECE_BINS) -> float:
    """Expected calibration error: weight-averaged ``|frac_pos - mean_p|`` over the bins."""
    tab = reliability_table(p, y, w, n_bins)
    tot = float(tab["weight"].sum())
    if tot <= 0:
        return float("nan")
    ok = tab["weight"] > 0
    return float(np.sum(tab.loc[ok, "weight"] * np.abs(tab.loc[ok, "gap"])) / tot)


def auc(p, y, w=None) -> float:
    """Weighted AUC = P(p_pos > p_neg) + 0.5 P(tie) (Mann-Whitney); NaN with one class only."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = _weights(w, p.shape[0])
    pos, neg = (y >= 0.5) & (w > 0), (y < 0.5) & (w > 0)
    wp, wn = float(w[pos].sum()), float(w[neg].sum())
    if wp <= 0 or wn <= 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps, ws, ys = p[order], w[order], y[order]
    # cumulative negative weight strictly below each distinct value, plus half of the ties
    is_neg = ys < 0.5
    neg_w = np.where(is_neg, ws, 0.0)
    cum_below = np.concatenate([[0.0], np.cumsum(neg_w)[:-1]])
    # handle ties: group equal p
    _, first, counts = np.unique(ps, return_index=True, return_counts=True)
    total = 0.0
    for f, c in zip(first, counts):
        seg = slice(f, f + c)
        below = cum_below[f]
        tie_neg = float(neg_w[seg].sum())
        pos_w_seg = float(np.where(~is_neg[seg], ws[seg], 0.0).sum())
        total += pos_w_seg * (below + 0.5 * tie_neg)
    return float(total / (wp * wn))


def nll(ll, w=None) -> float:
    """Weighted mean negative log-likelihood per response (``w`` = row multiplicities)."""
    ll = np.asarray(ll, dtype=np.float64)
    w = _weights(w, ll.shape[0])
    if w.sum() <= 0:
        return float("nan")
    return float(-np.sum(w * ll) / np.sum(w))


# ---------------------------------------------------------------------------------------------
# Subject-level bootstrap
# ---------------------------------------------------------------------------------------------


def subject_bootstrap(
    stat: Callable[[np.ndarray], float],
    subject_idx: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """``stat(row_weights) -> float`` evaluated under ``n_boot`` resamples of the subjects with
    replacement (row weight = how often the row's subject was drawn). Returns
    ``(lo, hi, draws)`` with the 2.5/97.5 percentiles of the finite draws."""
    subject_idx = np.asarray(subject_idx, dtype=np.int64)
    if subject_idx.size == 0:
        return float("nan"), float("nan"), np.zeros(0)
    uniq, inv = np.unique(subject_idx, return_inverse=True)
    S_ = len(uniq)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        counts = np.bincount(rng.integers(0, S_, size=S_), minlength=S_).astype(np.float64)
        draws[b] = stat(counts[inv])
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan"), draws
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)), draws


def evaluate_predictions(p, y, ll, subject_idx, w=None, *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """All metrics of the module docstring for one probability vector on the held-out sell rows
    (``p``, ``y``, ``ll`` (per-row log-likelihood), ``subject_idx`` aligned; ``w`` the
    likelihood weights, default 1). Bootstrap intervals resample subjects."""
    p, y, ll = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(ll, dtype=np.float64)
    w = _weights(w, p.shape[0])
    out: dict[str, Any] = {
        "n_rows": int(p.shape[0]),
        "n_subjects": int(len(np.unique(subject_idx))) if p.shape[0] else 0,
        "nll": nll(ll),
        "brier": brier_score(p, y, w),
        "ece": ece(p, y, w),
        "auc": auc(p, y, w),
        "reliability": reliability_table(p, y, w).to_dict(orient="records"),
    }
    if p.shape[0] and n_boot > 0:
        for key, fn in (
            ("nll", lambda m: nll(ll, m)),
            ("brier", lambda m: brier_score(p, y, w * m)),
            ("ece", lambda m: ece(p, y, w * m)),
            ("auc", lambda m: auc(p, y, w * m)),
        ):
            lo, hi, _ = subject_bootstrap(fn, subject_idx, n_boot=n_boot, seed=seed)
            out[f"{key}_ci95"] = [lo, hi]
    else:
        for key in ("nll", "brier", "ece", "auc"):
            out[f"{key}_ci95"] = [float("nan"), float("nan")]
    out["bootstrap"] = {"n_draws": int(n_boot), "seed": int(seed), "unit": "subject"}
    return out


def evaluate_model(
    model,
    params: dict[str, Any],
    data: ModelData,
    rows: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    param_samples: list[dict[str, Any]] | None = None,
    posterior: Any = None,
) -> dict[str, Any]:
    """Score ``rows`` of ``data`` with ``model``/``params``: :func:`evaluate_predictions` on the
    sell rows among them, plus ``ll`` and ``p`` (per selected sell row, for comparisons), the
    posterior-predictive NLL for B2 when ``posterior`` is given, and the mean log-likelihood
    under ``param_samples`` (``nll_samples_mean``) when draws are given."""
    rows = np.asarray(rows, dtype=np.int64)
    sub = data.subset(rows)
    mask = sub.sell_mask()
    p = np.asarray(model.predict_proba(params, sub), dtype=np.float64)[mask]
    ll = np.asarray(model.log_lik(params, sub), dtype=np.float64)[mask]
    y = sub.y[mask]
    w = sub.w[mask]
    sidx = sub.subject_idx[mask]
    out = evaluate_predictions(p, y, ll, sidx, w, n_boot=n_boot, seed=seed)
    out["rows"] = rows[mask]
    out["ll"] = ll
    out["p"] = p
    out["subject_idx"] = sidx
    if posterior is not None and hasattr(model, "pointwise_log_lik"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            llm = model.pointwise_log_lik(posterior, sub)[:, mask]
        out["nll_posterior_predictive"] = float(-np.mean(np.log(np.mean(np.exp(llm), axis=0))))
    elif param_samples:
        lls = np.stack([np.asarray(model.log_lik(s, sub), dtype=np.float64)[mask] for s in param_samples])
        out["nll_samples_predictive"] = float(-np.mean(np.log(np.mean(np.exp(lls), axis=0))))
    return out


def predictive_gain_per_subject(ll_model, ll_ref, subject_idx, subject_ids=None) -> pd.DataFrame:
    """Per subject: mean log-likelihood per response of the model minus the reference
    (``gain > 0`` = the model predicts that subject's held-out responses better), with
    ``n_rows``; sorted by subject index."""
    ll_model, ll_ref = np.asarray(ll_model, dtype=np.float64), np.asarray(ll_ref, dtype=np.float64)
    subject_idx = np.asarray(subject_idx, dtype=np.int64)
    df = pd.DataFrame({"subject_idx": subject_idx, "d": ll_model - ll_ref})
    g = df.groupby("subject_idx")["d"].agg(["mean", "size"]).reset_index()
    g.columns = ["subject_idx", "gain", "n_rows"]
    if subject_ids is not None:
        g.insert(1, "subject_id", np.asarray(subject_ids)[g["subject_idx"].to_numpy()])
    return g


def paired_nll_difference(ll_a, ll_b, subject_idx, *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """``NLL_a - NLL_b`` on the same rows with a subject-level paired bootstrap 95% CI
    (negative = ``a`` better)."""
    ll_a, ll_b = np.asarray(ll_a, dtype=np.float64), np.asarray(ll_b, dtype=np.float64)
    d = ll_b - ll_a  # NLL_a - NLL_b = mean(ll_b - ll_a)
    est = float(np.mean(d)) if d.size else float("nan")
    lo, hi, _ = subject_bootstrap(lambda m: float(np.sum(m * d) / np.sum(m)) if np.sum(m) > 0 else float("nan"), subject_idx, n_boot=n_boot, seed=seed)
    return {"diff": est, "ci95": [lo, hi], "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (hi < 0 or lo > 0))}


# ---------------------------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------------------------


def split_subjects(data: ModelData, test_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(a) held-out subjects, ``test_frac`` per ``dataset`` stratum (at least one subject per
    stratum with two or more subjects). Every row of a test subject is a test row."""
    rng = np.random.default_rng(int(seed))
    subj_dataset = pd.DataFrame({"s": data.subject_idx, "d": data.dataset}).drop_duplicates("s").sort_values("s")
    test_subjects: list[int] = []
    for _, grp in subj_dataset.groupby("d", sort=True):
        s = grp["s"].to_numpy()
        n_test = int(round(test_frac * len(s)))
        if len(s) >= 2:
            n_test = max(n_test, 1)
        n_test = min(n_test, len(s) - 1) if len(s) >= 2 else 0
        test_subjects.extend(rng.permutation(s)[:n_test].tolist())
    is_test = np.isin(data.subject_idx, np.asarray(test_subjects, dtype=np.int64))
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def split_context_composition(data: ModelData) -> tuple[np.ndarray, np.ndarray]:
    """(b) train on ``none`` and single-context rows (every row kind), test on the ordered
    pairs (sell rows with two context tags)."""
    n_ctx = data.ctx_mask.sum(axis=1)
    test = np.flatnonzero((n_ctx >= 2) & data.sell_mask())
    train = np.flatnonzero(n_ctx <= 1)
    return train, test


def split_question_order(data: ModelData, train_order: str = _design.TOLERANCE_FIRST) -> tuple[np.ndarray, np.ndarray]:
    """(c) train on one question order, test on the sell rows of the other."""
    if train_order not in _design.QUESTION_ORDER_IDS:
        raise ValueError(f"train_order must be one of {_design.QUESTION_ORDER_IDS}")
    flag = 1 if train_order == _design.TOLERANCE_FIRST else 0
    train = np.flatnonzero(data.order_flag == flag)
    test = np.flatnonzero((data.order_flag != flag) & data.sell_mask())
    return train, test


def split_temporal(data: ModelData, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """(d) per subject, order rows by ``(session_id, position)``; the last ``test_frac`` of the
    subject's sell rows are test rows, every earlier row (any kind) trains."""
    order = np.lexsort((data.position, data.session_id, data.subject_idx))
    sell = data.sell_mask()
    test_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    for s in np.unique(data.subject_idx):
        rows = order[data.subject_idx[order] == s]
        sell_rows = rows[sell[rows]]
        n_test = int(round(test_frac * len(sell_rows)))
        if n_test == 0:
            train_parts.append(rows)
            continue
        cut_row = sell_rows[-n_test]
        pos_cut = np.flatnonzero(rows == cut_row)[0]
        test_parts.append(rows[pos_cut:][sell[rows[pos_cut:]]])
        train_parts.append(rows[:pos_cut])
    train = np.sort(np.concatenate(train_parts)) if train_parts else np.zeros(0, dtype=np.int64)
    test = np.sort(np.concatenate(test_parts)) if test_parts else np.zeros(0, dtype=np.int64)
    return train, test


def split_rows(data: ModelData, cfg: dict[str, Any] | str) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch on ``cfg["type"]`` (``subjects``/``a``, ``pairs``/``b``, ``order``/``c``,
    ``temporal``/``d``, ``none`` = everything trains, empty test set)."""
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    t = str(cfg.get("type", "none")).lower()
    t = SPLIT_TYPES.get(t, t)
    if t == "subjects":
        return split_subjects(data, float(cfg.get("test_frac", 0.2)), int(cfg.get("seed", 0)))
    if t == "pairs":
        return split_context_composition(data)
    if t == "order":
        return split_question_order(data, str(cfg.get("train_order", _design.TOLERANCE_FIRST)))
    if t == "temporal":
        return split_temporal(data, float(cfg.get("test_frac", 0.2)))
    if t == "none":
        return np.arange(data.n, dtype=np.int64), np.zeros(0, dtype=np.int64)
    raise ValueError(f"unknown split type {t!r}; allowed {sorted(set(SPLIT_TYPES.values()))}")


# ---------------------------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------------------------


def _scenario_key(data: ModelData, rows: np.ndarray) -> np.ndarray:
    tags = [">".join(data.ctx_vocab[j] for j in data.ctx_idx[i] if j >= 0) or "none" for i in rows]
    return np.asarray([f"L{abs(data.loss_signed[i]):.2f}|{t}" for i, t in zip(rows, tags)], dtype=object)


def ltp_violation_from_data(data: ModelData, rows: np.ndarray | None = None, *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """LTP violation estimated from the responses: per scenario cell (loss x context sequence),
    ``P(sell | scenario-first) - P(sell | tolerance-first)`` where the second term is the
    marginal over the observed tolerance answer (``sum_a P(a, sell | tolerance-first)``);
    ``overall`` is the mean over cells with both orders observed, with a subject bootstrap CI
    (each cell recomputed under the resampled subject weights). Classical (Kolmogorov)
    responses have expectation 0 up to any classical order effect encoded in ``order_flag``;
    the G_C generator with ``beta_order = beta_tol = 0`` has exactly 0."""
    rows = data.sell_rows() if rows is None else np.intersect1d(np.asarray(rows), data.sell_rows())
    if rows.size == 0:
        return {"overall": float("nan"), "ci95": [float("nan"), float("nan")], "n_cells": 0, "cells": []}
    key = _scenario_key(data, rows)
    y, flag, sidx = data.y[rows], data.order_flag[rows], data.subject_idx[rows]
    cells, cidx = np.unique(key, return_inverse=True)

    def per_cell(m: np.ndarray) -> np.ndarray:
        out = np.full(len(cells), np.nan)
        for c in range(len(cells)):
            sel = cidx == c
            sf, tf = sel & (flag == 0), sel & (flag == 1)
            wsf, wtf = m[sf].sum(), m[tf].sum()
            if wsf > 0 and wtf > 0:
                out[c] = np.sum(m[sf] * y[sf]) / wsf - np.sum(m[tf] * y[tf]) / wtf
        return out

    ones = np.ones(rows.size)
    d = per_cell(ones)
    lo, hi, _ = subject_bootstrap(lambda m: float(np.nanmean(per_cell(m))) if np.isfinite(per_cell(m)).any() else float("nan"), sidx, n_boot=n_boot, seed=seed)
    table = [{"scenario": str(c), "delta_ltp": float(v), "n_scenario_first": int((cidx == i).sum() - ((cidx == i) & (flag == 1)).sum()), "n_tolerance_first": int(((cidx == i) & (flag == 1)).sum())} for i, (c, v) in enumerate(zip(cells, d))]
    return {"overall": float(np.nanmean(d)) if np.isfinite(d).any() else float("nan"), "abs_mean": float(np.nanmean(np.abs(d))) if np.isfinite(d).any() else float("nan"), "ci95": [lo, hi], "n_cells": int(np.isfinite(d).sum()), "cells": table}


def order_effects_from_data(data: ModelData, rows: np.ndarray | None = None, *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Order effects by scenario: for each loss and unordered context pair ``{c1, c2}``,
    ``P(sell | c1, c2) - P(sell | c2, c1)`` from scenario-first and tolerance-first rows pooled,
    with a subject bootstrap CI on the mean absolute effect."""
    rows = data.sell_rows() if rows is None else np.intersect1d(np.asarray(rows), data.sell_rows())
    pair = rows[data.ctx_mask[rows].all(axis=1)]
    if pair.size == 0:
        return {"mean_abs": float("nan"), "ci95": [float("nan"), float("nan")], "pairs": []}
    a, b = data.ctx_idx[pair, 0], data.ctx_idx[pair, 1]
    lo_i, hi_i = np.minimum(a, b), np.maximum(a, b)
    forward = a < b
    loss = np.round(data.loss[pair], 4)
    key = np.asarray([f"{L:.2f}|{data.ctx_vocab[i]}|{data.ctx_vocab[j]}" for L, i, j in zip(loss, lo_i, hi_i)], dtype=object)
    cells, cidx = np.unique(key, return_inverse=True)
    y, sidx = data.y[pair], data.subject_idx[pair]

    def per_cell(m: np.ndarray) -> np.ndarray:
        out = np.full(len(cells), np.nan)
        for c in range(len(cells)):
            sel = cidx == c
            f, r = sel & forward, sel & ~forward
            wf, wr = m[f].sum(), m[r].sum()
            if wf > 0 and wr > 0:
                out[c] = np.sum(m[f] * y[f]) / wf - np.sum(m[r] * y[r]) / wr
        return out

    d = per_cell(np.ones(pair.size))
    lo, hi, _ = subject_bootstrap(lambda m: float(np.nanmean(np.abs(per_cell(m)))), sidx, n_boot=n_boot, seed=seed)
    table = [{"scenario": str(c), "delta_order": float(v)} for c, v in zip(cells, d)]
    return {"mean_abs": float(np.nanmean(np.abs(d))) if np.isfinite(d).any() else float("nan"), "mean": float(np.nanmean(d)) if np.isfinite(d).any() else float("nan"), "ci95": [lo, hi], "n_cells": int(np.isfinite(d).sum()), "pairs": table}


def interference_ci(model, params: dict[str, Any], data: ModelData, rows: np.ndarray | None = None, param_samples: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """For a Q-model: the mean ``delta_LTP`` over the sell rows and the mean ``Delta_order`` over
    the pair rows at ``params``, and their 2.5/97.5 percentiles over ``param_samples`` (an
    interval from the diagonal Laplace draws for MAP fits, see :mod:`bre.artifact`; ``None``
    bounds without draws). A model whose ``delta_LTP`` is undefined (NaN on every row, Q5)
    gets NaN means, NaN bounds and ``ltp_ci_excludes_zero = False`` (``ltp_defined`` says why):
    a NaN interval never counts as excluding zero. Returns None for a classical model."""
    if not hasattr(model, "interference_terms"):
        return None
    rows = data.sell_rows() if rows is None else np.intersect1d(np.asarray(rows), data.sell_rows())
    sub = data.subset(rows)

    def stats(p: dict[str, Any]) -> tuple[float, float, float]:
        t = interference_summary(model, p, sub)
        return t["ltp_mean"], t["ltp_abs_mean"], t["order_mean"]

    def pct(v: np.ndarray) -> list[float]:
        fin = v[np.isfinite(v)]
        return [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if fin.size else [float("nan"), float("nan")]

    base = interference_summary(model, params, sub)
    out: dict[str, Any] = {"ltp_mean": base["ltp_mean"], "ltp_abs_mean": base["ltp_abs_mean"], "order_mean": base["order_mean"], "n_rows": base["n_rows"], "n_pair_rows": base["n_pair_rows"], "ltp_defined": base["ltp_defined"], "order_zero_by_construction": base["order_zero_by_construction"]}
    if "mean_abs_q" in base:
        out["mean_abs_q"] = base["mean_abs_q"]
    if param_samples:
        draws = np.asarray([stats(s) for s in param_samples], dtype=np.float64).reshape(-1, 3)
        out["ltp_mean_ci95"] = pct(draws[:, 0])
        out["ltp_abs_mean_ci95"] = pct(draws[:, 1])
        out["order_mean_ci95"] = pct(draws[:, 2])
        out["n_samples"] = int(len(param_samples))
        lo, hi = out["ltp_mean_ci95"]
        out["ltp_ci_excludes_zero"] = bool(np.isfinite(lo) and np.isfinite(hi) and (hi < 0 or lo > 0))
    else:
        out["ltp_mean_ci95"] = None
        out["ltp_ci_excludes_zero"] = None
    return out


def decoherence_distribution(params: dict[str, Any]) -> dict[str, Any] | None:
    """Quantiles of ``gamma_i = exp(log_gamma_i)`` (Q4's per-investor consistency score) and the
    fitted population ``(gamma_mu, sigma_gamma)``; None when ``params`` has no ``log_gamma``."""
    if "log_gamma" not in params:
        return None
    lg = np.asarray(params["log_gamma"], dtype=np.float64)
    g = np.exp(lg)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    return {
        "n_subjects": int(lg.shape[0]),
        "gamma_quantiles": {str(q): float(np.quantile(g, q)) for q in qs},
        "log_gamma_mean": float(np.mean(lg)),
        "log_gamma_sd": float(np.std(lg)),
        "gamma_mu": float(np.asarray(params.get("gamma_mu", np.nan))),
        "sigma_gamma": float(np.exp(np.asarray(params.get("gamma_log_sigma", np.nan)))),
        "share_near_classical": float(np.mean(g > 3.0)),
        "share_near_coherent": float(np.mean(g < 0.05)),
    }


ORDER_ZERO_NOTE = ("Delta_order = 0 on every ordered-pair row: the model composes contexts symmetrically, so it "
                   "predicts no order effect by construction (a property of the model, not a fitted result)")
LTP_UNDEFINED_NOTE = ("delta_LTP is not defined for this model (no non-commuting tolerance measurement; the "
                      "interference terms return NaN), so the interference-interval condition of the decision "
                      "rule cannot hold for it")


def model_structural_checks(model, params: dict[str, Any], data: ModelData, rows: np.ndarray | None = None, *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Model-level structural checks of a Q-model at ``params`` on the sell rows among ``rows``
    (default all): ``ltp`` (whether ``delta_LTP`` is defined, with :data:`LTP_UNDEFINED_NOTE`
    when it is not), ``order_effect`` (``zero_by_construction`` when every pair row has
    ``Delta_order = 0`` exactly, with :data:`ORDER_ZERO_NOTE`; ``n_pair_rows``; the mean
    absolute order term otherwise), ``quarter_law`` (``model.quarter_law_check`` when the model
    has one: Q5's mean ``|q|`` with a subject bootstrap interval and the reference 0.25) and
    ``decoherence`` (:func:`decoherence_distribution` when the parameters carry ``log_gamma``).
    Every entry reports; none asserts a hypothesis."""
    rows = data.sell_rows() if rows is None else np.intersect1d(np.asarray(rows), data.sell_rows())
    sub = data.subset(rows)
    base = interference_summary(model, params, sub)
    out: dict[str, Any] = {
        "n_rows": base["n_rows"],
        "ltp": {"defined": base["ltp_defined"], "mean": base["ltp_mean"], "abs_mean": base["ltp_abs_mean"], "note": None if base["ltp_defined"] else LTP_UNDEFINED_NOTE},
        "order_effect": {"n_pair_rows": base["n_pair_rows"], "mean": base["order_mean"], "abs_mean": base["order_abs_mean"], "zero_by_construction": base["order_zero_by_construction"], "note": ORDER_ZERO_NOTE if base["order_zero_by_construction"] else None},
    }
    if hasattr(model, "quarter_law_check"):
        out["quarter_law"] = model.quarter_law_check(params, sub, n_boot=int(n_boot), seed=int(seed))
    dec = decoherence_distribution(params)
    if dec is not None:
        out["decoherence"] = dec
    return out


# ---------------------------------------------------------------------------------------------
# Comparison and verdict
# ---------------------------------------------------------------------------------------------


def compare_models(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Side-by-side table of per-model result dicts (keys ``family``, ``n_params``,
    ``n_params_population``, ``nll``, ``nll_ci95``, ``brier``, ``ece``, ``auc``, optional
    ``nll_posterior_predictive``, ``waic``/``loo`` per response, ``nll_diff_vs_best_classical``);
    sorted by held-out NLL. Missing keys become NaN."""
    rows = []
    for name, r in results.items():
        ci = r.get("nll_ci95") or [float("nan"), float("nan")]
        diff = r.get("nll_diff_vs_best_classical") or {}
        rows.append({
            "model": name,
            "family": r.get("family", MODEL_FAMILY.get(name, "")),
            "n_params": r.get("n_params", float("nan")),
            "n_params_population": r.get("n_params_population", float("nan")),
            "nll": r.get("nll", float("nan")),
            "nll_lo": ci[0],
            "nll_hi": ci[1],
            "brier": r.get("brier", float("nan")),
            "ece": r.get("ece", float("nan")),
            "auc": r.get("auc", float("nan")),
            "nll_ppd": r.get("nll_posterior_predictive", r.get("nll_samples_predictive", float("nan"))),
            "elpd_waic_per_response": (r.get("waic") or {}).get("elpd_waic_per_response", float("nan")),
            "elpd_loo_per_response": (r.get("loo") or {}).get("elpd_loo_per_response", float("nan")),
            "nll_diff_vs_best_classical": diff.get("diff", float("nan")),
            "nll_diff_lo": (diff.get("ci95") or [float("nan"), float("nan")])[0],
            "nll_diff_hi": (diff.get("ci95") or [float("nan"), float("nan")])[1],
            "n_rows": r.get("n_rows", float("nan")),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("nll", kind="mergesort").reset_index(drop=True)
    return df


def decision_rule(
    results_split_b: dict[str, dict[str, Any]],
    interference_ci: dict[str, Any] | None,
    *,
    is_real_data: bool | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    param_count_key: str = "n_params",
) -> dict[str, Any]:
    """PLAN.md section 6 decision rule on the split (b) results (module docstring).

    ``results_split_b``: model -> dict with ``family``, ``nll``, ``n_params`` (and
    ``n_params_population``), and either per-row ``ll`` with a shared ``subject_idx`` (the
    paired bootstrap of the NLL difference is computed here) or a precomputed
    ``nll_diff_vs_best_classical`` (``{"diff", "ci95"}``); optionally ``is_synthetic``.
    ``interference_ci``: the :func:`interference_ci` dict of the best Q-model (its
    ``ltp_mean_ci95`` must exclude zero; a missing or NaN interval — no draws, or a model whose
    ``delta_LTP`` is undefined — does not exclude zero). ``is_real_data`` defaults to ``not
    any(is_synthetic)``; on synthetic data the positive verdict is never issued and the details
    say which conditions held. ``param_count_key`` picks the count compared ("matched or lower"
    means ``count_q <= count_c``). Returns ``{"verdict", "details"}`` with ``verdict`` one of
    :data:`VERDICT_SUPPORTED` / :data:`VERDICT_NONE`.
    """
    classical = {k: v for k, v in results_split_b.items() if v.get("family", MODEL_FAMILY.get(k)) == "classical"}
    quantum = {k: v for k, v in results_split_b.items() if v.get("family", MODEL_FAMILY.get(k)) == "quantum"}
    if is_real_data is None:
        is_real_data = not any(bool(v.get("is_synthetic", False)) for v in results_split_b.values())
    details: dict[str, Any] = {"rule": DECISION_RULE_TEXT, "split": "b (held-out context compositions: train singles, test ordered pairs)", "is_real_data": bool(is_real_data), "param_count_key": param_count_key}
    if not classical or not quantum:
        details["reason"] = "both a classical and a quantum-probability model are needed on split (b)"
        details["conditions"] = {}
        details["served_model"] = min(results_split_b, key=lambda k: results_split_b[k].get("nll", np.inf)) if results_split_b else None
        return {"verdict": VERDICT_NONE, "details": details}
    best_c = min(classical, key=lambda k: classical[k].get("nll", np.inf))
    best_q = min(quantum, key=lambda k: quantum[k].get("nll", np.inf))
    rc, rq = classical[best_c], quantum[best_q]
    if "ll" in rq and "ll" in rc and rq.get("subject_idx") is not None:
        diff = paired_nll_difference(rq["ll"], rc["ll"], rq["subject_idx"], n_boot=n_boot, seed=seed)
    else:
        diff = dict(rq.get("nll_diff_vs_best_classical") or {"diff": float(rq.get("nll", np.nan)) - float(rc.get("nll", np.nan)), "ci95": [float("nan"), float("nan")]})
    lo, hi = diff.get("ci95", [float("nan"), float("nan")])
    beats = bool(np.isfinite(hi) and hi < 0)
    nq, nc = rq.get(param_count_key, np.nan), rc.get(param_count_key, np.nan)
    count_ok = bool(np.isfinite(nq) and np.isfinite(nc) and nq <= nc)
    ici = (interference_ci or {}).get("ltp_mean_ci95")
    # a missing or NaN interval (no draws; delta_LTP undefined for the model) does not exclude zero
    interf_ok = bool(ici is not None and len(ici) == 2 and np.all(np.isfinite(ici)) and (ici[1] < 0 or ici[0] > 0))
    conditions = {"beats_best_classical_nll_ci_excludes_zero": beats, "param_count_matched_or_lower": count_ok, "interference_ci_excludes_zero": interf_ok, "real_data": bool(is_real_data)}
    served = min(results_split_b, key=lambda k: results_split_b[k].get("nll", np.inf))
    details.update({
        "best_quantum": best_q,
        "best_classical": best_c,
        "nll_quantum": float(rq.get("nll", np.nan)),
        "nll_classical": float(rc.get("nll", np.nan)),
        "nll_diff_quantum_minus_classical": diff.get("diff"),
        "nll_diff_ci95": [lo, hi],
        "n_params_quantum": nq,
        "n_params_classical": nc,
        "interference_ltp_ci95": ici,
        "conditions": conditions,
        "served_model": served,
        "served_model_family": results_split_b[served].get("family", MODEL_FAMILY.get(served)),
    })
    if all(conditions.values()):
        return {"verdict": VERDICT_SUPPORTED, "details": details}
    failed = [k for k, v in conditions.items() if not v]
    details["failed_conditions"] = failed
    if not is_real_data:
        details["note"] = "synthetic data: the rule applies to real data only; the other conditions " + ("held" if all(v for k, v in conditions.items() if k != "real_data") else "did not all hold")
    details["data_that_would_resolve_it"] = (
        "within-subject responses to ordered context pairs with both question orders on the shared design "
        "(intake battery, instrument/; or the gated UAS / LISS panels, data/REGISTRATION.md)"
    )
    return {"verdict": VERDICT_NONE, "details": details}


# ---------------------------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------------------------


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """A GitHub-flavoured markdown table of a DataFrame (numbers formatted with ``floatfmt``)."""
    if df.empty:
        return "(empty)\n"
    cols = list(df.columns)

    def fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "nan" if not np.isfinite(v) else format(float(v), floatfmt)
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.astype(object).iterrows():  # object dtype keeps ints as ints per cell
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Config-driven evaluation
# ---------------------------------------------------------------------------------------------


def run_experiment(cfg: dict[str, Any] | str | Path, *, echo: bool = True) -> tuple[dict[str, Any], Path, Path]:
    """Run one evaluation experiment (module docstring). Returns ``(results, json_path, md_path)``."""
    if not isinstance(cfg, dict):
        cfg = load_config(cfg)
    name = cfg["name"]
    models = list(cfg.get("models") or list(MODEL_FAMILY))
    splits = list(cfg.get("splits") or ["subjects", "pairs", "order", "temporal"])
    boot = dict(cfg.get("bootstrap") or {})
    n_boot, bseed = int(boot.get("n_draws", 1000)), int(boot.get("seed", 0))
    out = dict(cfg.get("output") or {})
    run_dir = resolve_path(out.get("dir", EVAL_RUNS_DIR / name))
    json_path = resolve_path(out.get("metrics_path", REPORTS_EVAL_DIR / f"{name}.json"))
    md_path = resolve_path(out.get("table_path", REPORTS_EVAL_DIR / f"{name}.md"))
    log_path = run_dir / "eval.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=False), encoding="utf-8")
    t_start = time.time()
    _log(log_path, f"[eval] experiment {name}: models={models} splits={splits} dataset={cfg['dataset']}", echo)
    df, data, ref = load_dataset(cfg)
    is_syn = bool(np.asarray(data.is_synthetic).any())
    settings = fit_settings(cfg)
    results: dict[str, Any] = {"experiment": name, "dataset": ref, "is_synthetic": is_syn, "data": data.summary(), "fit_settings": settings, "splits": {}, "structural": {}}
    results["structural"]["ltp_violation_from_data"] = ltp_violation_from_data(data, n_boot=n_boot, seed=bseed)
    results["structural"]["order_effects_from_data"] = order_effects_from_data(data, n_boot=n_boot, seed=bseed)
    _log(log_path, f"[eval] data LTP violation overall={results['structural']['ltp_violation_from_data']['overall']:.4f} CI={results['structural']['ltp_violation_from_data']['ci95']}; order effects mean|d|={results['structural']['order_effects_from_data']['mean_abs']:.4f}", echo)
    best_q_interference: dict[str, Any] | None = None
    split_b_results: dict[str, dict[str, Any]] | None = None
    for split in splits:
        split_cfg = split if isinstance(split, dict) else {"type": split}
        stype = SPLIT_TYPES.get(str(split_cfg.get("type")).lower(), str(split_cfg.get("type")).lower())
        train_rows, test_rows = split_rows(data, split_cfg)
        _log(log_path, f"[eval] split {stype}: {len(train_rows)} train rows, {len(test_rows)} test rows", echo)
        per_model: dict[str, dict[str, Any]] = {}
        for m in models:
            t0 = time.time()
            fit = fit_model(m, data, rows=train_rows, log_path=log_path, echo=echo, model_kwargs=dict((cfg.get("model_kwargs") or {}).get(m, {})), **settings)
            held = evaluate_model(fit.model, fit.params, data, test_rows, n_boot=n_boot, seed=bseed, param_samples=fit.param_samples, posterior=fit.posterior)
            entry = {k: v for k, v in held.items() if k not in ("rows", "ll", "p", "subject_idx")}
            entry.update({"family": MODEL_FAMILY[m], "n_params": fit.n_params, "n_params_population": fit.n_params_population, "train_nll": fit.train_nll, "val_nll": fit.val_nll, "wall_time_s": fit.wall_time, "is_synthetic": is_syn})
            for k in ("waic", "loo", "laplace", "interference_train", "external_fit", "param_samples_note"):
                if k in fit.extra:
                    entry[k] = fit.extra[k]
            if hasattr(fit.model, "interference_terms"):
                entry["interference_test"] = interference_ci(fit.model, fit.params, data, test_rows, fit.param_samples)
                checks = model_structural_checks(fit.model, fit.params, data, test_rows, n_boot=n_boot, seed=bseed)
                entry["structural_checks_test"] = checks
                if "decoherence" in checks:
                    entry["decoherence"] = checks["decoherence"]
                if "quarter_law" in checks:
                    entry["quarter_law_test"] = checks["quarter_law"]
                results["structural"].setdefault("model_checks", []).append({"split": stype, "model": m, **checks})
            entry["_ll"], entry["_subject_idx"] = held["ll"], held["subject_idx"]
            per_model[m] = entry
            art = artifact_from_fit(fit, data, training_data_refs=[ref], metrics={"experiment": name, "split": split_cfg, "holdout": {k: v for k, v in entry.items() if not k.startswith("_")}})
            art.save(run_dir / stype / m)
            _log(log_path, f"[eval] {stype}/{m}: nll={entry['nll']:.4f} CI={np.round(entry['nll_ci95'], 4).tolist()} brier={entry['brier']:.4f} ece={entry['ece']:.4f} auc={entry['auc']:.4f} n_params={fit.n_params} ({time.time() - t0:.0f}s)", echo)
        classical = [m for m in per_model if per_model[m]["family"] == "classical"]
        if classical:
            best_c = min(classical, key=lambda k: per_model[k]["nll"])
            for m, e in per_model.items():
                if m != best_c and len(e["_ll"]) == len(per_model[best_c]["_ll"]):
                    e["nll_diff_vs_best_classical"] = {**paired_nll_difference(e["_ll"], per_model[best_c]["_ll"], e["_subject_idx"], n_boot=n_boot, seed=bseed), "reference": best_c}
                    gain = predictive_gain_per_subject(e["_ll"], per_model[best_c]["_ll"], e["_subject_idx"], data.subject_ids)
                    e["predictive_gain_vs_best_classical"] = {"reference": best_c, "mean": float(gain["gain"].mean()), "median": float(gain["gain"].median()), "share_positive": float((gain["gain"] > 0).mean()), "quantiles": {str(q): float(gain["gain"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}
        table = compare_models({m: {k: v for k, v in e.items() if not k.startswith("_")} for m, e in per_model.items()})
        results["splits"][stype] = {"config": split_cfg, "n_train_rows": int(len(train_rows)), "n_test_rows": int(len(test_rows)), "models": {m: {k: v for k, v in e.items() if not k.startswith("_")} for m, e in per_model.items()}, "table": table.to_dict(orient="records")}
        if stype == "pairs":
            split_b_results = {m: {**{k: v for k, v in e.items() if not k.startswith("_")}, "ll": e["_ll"], "subject_idx": e["_subject_idx"]} for m, e in per_model.items()}
            qs = [m for m in per_model if per_model[m]["family"] == "quantum"]
            if qs:
                best_q = min(qs, key=lambda k: per_model[k]["nll"])
                best_q_interference = per_model[best_q].get("interference_test")
    if split_b_results is not None:
        results["verdict"] = decision_rule(split_b_results, best_q_interference, is_real_data=not is_syn, n_boot=n_boot, seed=bseed)
    else:
        results["verdict"] = {"verdict": VERDICT_NONE, "details": {"reason": "split (b) was not evaluated", "rule": DECISION_RULE_TEXT}}
    results["wall_time_s"] = time.time() - t_start
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(jsonable(results), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(results_markdown(results), encoding="utf-8")
    _log(log_path, f"[eval] verdict: {results['verdict']['verdict']}; wrote {json_path} and {md_path} ({results['wall_time_s']:.0f}s)", echo)
    return results, json_path, md_path


def results_markdown(results: dict[str, Any]) -> str:
    """The markdown report of :func:`run_experiment` (tables per split, structural tests, verdict)."""
    lines = [f"# Evaluation: {results['experiment']}", ""]
    lines.append(f"Dataset: `{results['dataset']}` — **{'SYNTHETIC' if results.get('is_synthetic') else 'real'}** data; "
                 f"{results['data']['n_subjects']} subjects, {results['data']['n_sell_rows']} sell rows. Generated by `bre.eval`; no hand-typed numbers.")
    lines.append("")
    for stype, sres in results["splits"].items():
        lines.append(f"## Split {stype}: {sres['n_train_rows']} train rows, {sres['n_test_rows']} test rows")
        lines.append("")
        cols = ["model", "family", "n_params", "n_params_population", "nll", "nll_lo", "nll_hi", "brier", "ece", "auc", "nll_diff_vs_best_classical", "nll_diff_lo", "nll_diff_hi"]
        df = pd.DataFrame(sres["table"])
        lines.append(markdown_table(df[[c for c in cols if c in df.columns]]))
        for m, e in sres["models"].items():
            if e.get("interference_test"):
                it = e["interference_test"]
                lines.append(f"- {m} interference on the test rows: mean delta_LTP = {it['ltp_mean']:.4f} (CI95 {it.get('ltp_mean_ci95')}), mean |delta_LTP| = {it['ltp_abs_mean']:.4f}, mean Delta_order = {it['order_mean']:.4f}")
            if e.get("decoherence"):
                d = e["decoherence"]
                lines.append(f"- {m} decoherence rates gamma_i: median {d['gamma_quantiles']['0.5']:.3f}, 5-95% {d['gamma_quantiles']['0.05']:.3f}-{d['gamma_quantiles']['0.95']:.3f}; share near-classical (gamma>3) {d['share_near_classical']:.2f}, near-coherent (gamma<0.05) {d['share_near_coherent']:.2f}")
            if e.get("predictive_gain_vs_best_classical"):
                g = e["predictive_gain_vs_best_classical"]
                lines.append(f"- {m} per-subject predictive gain vs {g['reference']}: mean {g['mean']:.4f}, median {g['median']:.4f}, share of subjects with gain > 0: {g['share_positive']:.2f}")
        lines.append("")
    st = results.get("structural", {})
    if st:
        lines.append("## Structural tests from the data")
        lines.append("")
        ltp = st.get("ltp_violation_from_data", {})
        lines.append(f"- LTP violation (mean over {ltp.get('n_cells')} scenario cells of P(sell | scenario-first) - P(sell | tolerance-first)): {ltp.get('overall', float('nan')):.4f}, CI95 {np.round(ltp.get('ci95', [np.nan, np.nan]), 4).tolist()}; mean |delta| {ltp.get('abs_mean', float('nan')):.4f}")
        oe = st.get("order_effects_from_data", {})
        lines.append(f"- Order effects by scenario (mean |P(sell | c1,c2) - P(sell | c2,c1)| over {oe.get('n_cells')} cells): {oe.get('mean_abs', float('nan')):.4f}, CI95 {np.round(oe.get('ci95', [np.nan, np.nan]), 4).tolist()}")
        lines.append("")
        if st.get("model_checks"):
            lines.append("### Model-level structural checks (Q-models, held-out rows of each split)")
            lines.append("")
            for c in st["model_checks"]:
                tag = f"{c['model']} on split {c['split']} ({c['n_rows']} rows)"
                ltp, oe_m = c.get("ltp", {}), c.get("order_effect", {})
                if ltp.get("defined"):
                    lines.append(f"- {tag}: mean delta_LTP = {ltp.get('mean', float('nan')):.4f}, mean |delta_LTP| = {ltp.get('abs_mean', float('nan')):.4f}")
                else:
                    lines.append(f"- {tag}: {ltp.get('note')}")
                if oe_m.get("zero_by_construction"):
                    lines.append(f"- {tag}: order effect identically 0 on its {oe_m.get('n_pair_rows')} pair rows — {oe_m.get('note')}")
                elif oe_m.get("n_pair_rows", 0) > 0:
                    lines.append(f"- {tag}: mean Delta_order = {oe_m.get('mean', float('nan')):.4f}, mean |Delta_order| = {oe_m.get('abs_mean', float('nan')):.4f} over {oe_m.get('n_pair_rows')} pair rows")
                else:
                    lines.append(f"- {tag}: no ordered-pair rows in this fold; the order effect is not evaluated here")
                if c.get("quarter_law"):
                    q = c["quarter_law"]
                    lines.append(f"- {tag}: quarter-law check: mean |q| = {q.get('mean_abs_q', float('nan')):.4f}, CI95 {np.round(q.get('ci95', [np.nan, np.nan]), 4).tolist()} vs reference {q.get('reference')} ({q.get('note')}); share of q > 0: {q.get('share_q_positive', float('nan')):.2f}")
                if c.get("decoherence"):
                    d = c["decoherence"]
                    lines.append(f"- {tag}: decoherence rates gamma_i: median {d['gamma_quantiles']['0.5']:.3f}, 5-95% {d['gamma_quantiles']['0.05']:.3f}-{d['gamma_quantiles']['0.95']:.3f}")
            lines.append("")
    v = results.get("verdict", {})
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{v.get('verdict')}**")
    lines.append("")
    d = v.get("details", {})
    if d.get("conditions"):
        for k, ok in d["conditions"].items():
            lines.append(f"- {k}: {'yes' if ok else 'no'}")
        lines.append(f"- best Q-model {d.get('best_quantum')} (NLL {d.get('nll_quantum', float('nan')):.4f}, {d.get('n_params_quantum')} params) vs best classical {d.get('best_classical')} (NLL {d.get('nll_classical', float('nan')):.4f}, {d.get('n_params_classical')} params); NLL difference {d.get('nll_diff_quantum_minus_classical', float('nan')):.4f} CI95 {d.get('nll_diff_ci95')}")
        lines.append(f"- served model (lowest held-out NLL on split b): {d.get('served_model')} ({d.get('served_model_family')})")
    if d.get("note"):
        lines.append(f"- note: {d['note']}")
    if d.get("data_that_would_resolve_it"):
        lines.append(f"- data that would resolve it: {d['data_that_would_resolve_it']}")
    lines.append("")
    lines.append(f"Decision rule (PLAN.md section 6, verbatim): {DECISION_RULE_TEXT}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bre.eval", description="Pre-registered evaluation from an experiment YAML (default: experiments/eval_all.yaml).")
    parser.add_argument("config", nargs="*")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    from bre.fit import expand_configs

    configs = expand_configs([Path(c) for c in args.config] or [EVAL_ALL])
    if not configs:
        print("no experiments listed", file=sys.stderr)
        return 1
    failures = 0
    for c in configs:
        try:
            run_experiment(c, echo=not args.quiet)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[eval] {c}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DECISION_RULE_TEXT",
    "SPLIT_TYPES",
    "VERDICT_NONE",
    "VERDICT_SUPPORTED",
    "auc",
    "brier_score",
    "compare_models",
    "decision_rule",
    "decoherence_distribution",
    "ece",
    "evaluate_model",
    "evaluate_predictions",
    "interference_ci",
    "ltp_violation_from_data",
    "markdown_table",
    "model_structural_checks",
    "nll",
    "order_effects_from_data",
    "paired_nll_difference",
    "predictive_gain_per_subject",
    "reliability_table",
    "results_markdown",
    "run_experiment",
    "split_context_composition",
    "split_question_order",
    "split_rows",
    "split_subjects",
    "split_temporal",
    "subject_bootstrap",
]
