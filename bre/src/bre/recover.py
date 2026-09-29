"""Phase 2 model-recovery study (PLAN.md section 5; ``make recover``).

For every generator in ``{gq, gc, gf}`` (``bre.sim``), every ``N`` and every seed of the grid:

1. the synthetic table ``data/synthetic/<gen>_n<N>_seed<seed>.parquet`` (the stem
   ``bre.sim.cli.file_stem`` writes for ``make sim``) is read when present, together with its
   ``.truth.json``, and generated otherwise (same seed, same table);
2. the sell rows are split **within subject**, 80/20, stratified by condition type (number of
   contexts x question order, :func:`bre.fit.condition_strata`): the 20% is the *selection
   fold*, the 80% is what every model is fitted on (:func:`bre.fit.fit_model`, which holds out
   its own validation fold for early stopping inside that 80%);
3. every model in the list is fitted and scored on the selection fold by held-out NLL per
   response; the model with the lowest NLL is *selected*, and the selection is *correct* when
   its family (classical / quantum) is the generator's family (``bre.registry.GENERATORS``).
   Any model of ``bre.registry.MODEL_REGISTRY`` may be listed (``--models``); the default grid
   is the first-pass five (``DEFAULT_MODELS``). WAIC and PSIS-LOO are recorded for B2 (the
   only Bayesian fit; they do not compare across models at this stage); the interference
   terms of every Q-model on the fit fold ride along;
4. parameter recovery: for G_Q the population context means ``theta_c`` (the subject-mean of
   the generator's per-subject ``theta_{c,i}``) against the Q2 / Q4 tables after gauge
   alignment and folding (``align_gauge`` of ``bre.models.quantum.q2_context_unitary``), the
   per-subject ``log gamma_i`` against Q4's fitted ``log_gamma`` (Spearman), and the subjects'
   Bloch polar angles; for G_C the context betas against B1's per-position main effects and B2's
   ``mu_ctx`` (Pearson) and the subject intercepts against B2's (Spearman), plus B4's evidence
   ordering (Spearman with ``-beta``, reported).

Outputs (``reports/recovery/``): ``confusion_N<N>.md`` (true generator x selected model, and x
selected family), ``recovery_scatter_*.png`` (matplotlib Agg), ``N_target.json`` (the smallest
``N`` of the grid at which the family-level selection is correct in at least 90% of the
generator x seed cells, and "responses needed for calibration" = that ``N`` x items per
subject; null with the reason when no ``N`` reaches 90%), ``results.json`` (every number) and
``summary.md`` (generated from ``results.json``; no hand-typed numbers). The log of the run is
``runs/recover/<timestamp>.log`` with one progress line per fit; fitted parameters are saved as
artifacts under ``runs/recover/<timestamp>/``.

Runtime: fits are distributed over ``--workers`` processes (one process per (dataset, model)
job, the long Q4 jobs first). Each worker is pinned to one XLA thread so that the processes do
not compete for the same cores (measured on the 4-core build machine: one fit alone does not
use more than one core, four single-threaded fits run with ~2.3x the throughput of one).
Before starting, the driver estimates the wall time from measured per-step costs and refuses a
run that would exceed ``--max-minutes`` (default 110, under the two-hour GATE of CLAUDE.md
rule 4) unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bre import schema as S
from bre.registry import GENERATORS, MODEL_FAMILY, MODEL_REGISTRY

PROJECT_ROOT = S.PROJECT_ROOT
REPORTS_DIR = PROJECT_ROOT / "reports" / "recovery"
RUNS_DIR = PROJECT_ROOT / "runs" / "recover"
DEFAULT_DATA_DIR = S.SYNTHETIC_DIR

DEFAULT_MODELS: tuple[str, ...] = ("B1", "B2", "B4", "Q2", "Q4")
DEFAULT_GENERATORS: tuple[str, ...] = ("gq", "gc", "gf")

SELECTION_FRAC = 0.2
"""Share of every subject's sell rows (per condition-type stratum) held out as the selection fold."""

N_TARGET_THRESHOLD = 0.9
"""Share of correct family-level selections an ``N`` must reach to become the calibration target."""

N_TARGET_RULE = (
    "N_target is the smallest N of the grid at which the model selected by held-out NLL on the "
    "within-subject 20% selection fold belongs to the true generator's family in at least 90% of "
    "the (generator, seed) cells at that N; responses needed for calibration = N_target x items per "
    "subject (170 on the shared design). Null when no N of the grid reaches 90%."
)

SECONDS_PER_STEP: dict[str, float] = {"B1": 0.002, "B4": 0.004, "Q2": 0.09, "Q4": 0.26, "B2": 0.03}
"""Measured full-batch step costs on the N = 200 design (34,000 sell rows) with four
single-threaded workers running at once (2026-09-17 build machine); scaled linearly with the
number of sell rows for the runtime estimate."""

SECONDS_PER_STEP_ASSUMED: dict[str, float] = {"B3": 0.006, "B6": 0.02, "Q3": 0.09, "Q5": 0.005}
"""**Assumed, not measured** step costs of the second-pass models, used only by
:func:`estimate_minutes` (the pre-run wall-time guard): B3 and Q5 are encoder models of B4's
size with a few more element-wise operations; B6 runs a ``lax.scan`` over session positions;
Q3 evaluates one 2x2 matrix exponential per row like Q2. A model missing from both tables is
estimated at the largest measured cost (Q4's). Replace an entry with a measured value once a
fit of that model has been timed on the build machine (the log carries the seconds per fit)."""

EXTERNAL_FIT_SECONDS: float = 60.0
"""Assumed wall time of one external fit (B5: boosting + MLP on ~22,000 rows, one thread) for
the runtime estimate; scaled with the number of rows like the step costs."""

XLA_SINGLE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

MODEL_SETTINGS_DEFAULT: dict[str, dict[str, Any]] = {
    "classical": {"restarts": 3, "steps": 1000, "lr": 0.02, "lbfgs_steps": 50},
    "quantum": {"restarts": 2, "steps": 1000, "lr": 0.02, "lbfgs_steps": 50},
    "B1": {"lr": 0.05},
    "B2": {"restarts": 3, "svi_steps": 2000, "svi_lr": 0.01},
}
"""Fit settings by family, with per-model overrides merged on top (``B1`` takes the learning rate
its own recovery test uses; ``B2`` is SVI). The first pass uses 2 restarts for the Q-models and
3 for the classical ones, 1000 Adam steps with early stopping and a 50-iteration L-BFGS polish:
sized so that the N = 200, 3-seed grid finishes within about 40 minutes on 4 cores (PLAN.md
section 4 asks for 5 restarts in the final fits; ``experiments/*.yaml`` carry those)."""


# ---------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------


def dataset_stem(gen: str, n: int, seed: int) -> str:
    return f"{gen}_n{int(n)}_seed{int(seed)}"


def ensure_dataset(gen: str, n: int, seed: int, data_dir: Path, log=print) -> tuple[Path, Path]:
    """Read or generate ``<gen>_n<N>_seed<seed>`` under ``data_dir`` (a ``data/synthetic``
    directory); returns the parquet and truth paths. A parquet without its truth file is
    regenerated (same seed, same table) so that parameter recovery has its reference."""
    from bre.sim import cli as sim_cli
    from bre.sim import common

    stem = dataset_stem(gen, n, seed)
    pq = common.parquet_path(stem, data_dir)
    tr = common.truth_path(stem, data_dir)
    if pq.exists() and tr.exists():
        return pq, tr
    mod = sim_cli.load_generator(gen)
    if mod is None:
        raise FileNotFoundError(f"generator bre.sim.{gen} does not exist")
    t0 = time.time()
    df, truth = mod.generate(int(n), int(seed))
    common.write_synthetic(df, truth, stem, data_dir)
    log(f"[recover] generated {stem}: {len(df)} rows in {time.time() - t0:.0f}s")
    return pq, tr


# ---------------------------------------------------------------------------------------------
# One (dataset, model) job
# ---------------------------------------------------------------------------------------------


def _worker_init() -> None:
    flags = os.environ.get("XLA_FLAGS", "")
    if "intra_op_parallelism_threads" not in flags:
        os.environ["XLA_FLAGS"] = (flags + " " + XLA_SINGLE_THREAD).strip()


def selection_split(data, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """80/20 within-subject split of the sell rows stratified by condition type (module docstring)."""
    from bre.fit import split_within_subject

    return split_within_subject(data, data.sell_rows(), SELECTION_FRAC, int(seed))


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Fit one model on one dataset's fit fold and score the selection fold. Runs in a worker
    process; returns a plain dict (numbers, small numpy arrays) for the parent to aggregate."""
    import warnings

    warnings.filterwarnings("ignore")
    from bre import eval as E
    from bre.fit import artifact_from_fit, fit_model, nll_per_response
    from bre.models import build_model_data

    t0 = time.time()
    gen, n, seed, model_name = job["gen"], int(job["n"]), int(job["seed"]), job["model"]
    stem = dataset_stem(gen, n, seed)
    df = S.read_events(job["parquet"])
    data = build_model_data(df)
    fit_rows, sel_rows = selection_split(data, seed)
    settings = dict(job["settings"])
    log_path = job.get("log_path")
    res = fit_model(model_name, data, rows=fit_rows, log_path=log_path, echo=False, n_samples=0, **settings)
    held = E.evaluate_model(res.model, res.params, data, sel_rows, n_boot=int(job.get("n_boot", 200)), seed=seed, posterior=res.posterior)
    sub = data.subset(sel_rows)
    out: dict[str, Any] = {
        "gen": gen, "n": n, "seed": seed, "model": model_name, "family": MODEL_FAMILY[model_name], "stem": stem,
        "n_fit_rows": int(len(fit_rows)), "n_selection_rows": int(len(sel_rows)),
        "selection_nll": float(held["nll"]), "selection_nll_ci95": held["nll_ci95"], "selection_brier": float(held["brier"]),
        "selection_ece": float(held["ece"]), "selection_auc": float(held["auc"]),
        "train_nll": float(res.train_nll), "val_nll": float(res.val_nll), "n_params": int(res.n_params), "n_params_population": int(res.n_params_population),
        "best_restart": int(res.best_restart), "restarts": res.restarts_table.to_dict(orient="records"),
        "wall_time_s": float(res.wall_time), "job_seconds": 0.0,
        "selection_ll": np.asarray(held["ll"]), "selection_subject_idx": np.asarray(held["subject_idx"]),
        "selection_nll_check": nll_per_response(np.asarray(res.model.log_lik(res.params, sub)), sub.sell_mask()),
    }
    if "waic" in res.extra:
        out["waic"] = res.extra["waic"]
        out["loo"] = res.extra.get("loo")
    if "nll_posterior_predictive" in held:
        out["selection_nll_posterior_predictive"] = float(held["nll_posterior_predictive"])
    # parameters needed for the recovery scatter plots
    p = res.params
    summary: dict[str, Any] = {"subject_ids": [str(s) for s in data.subject_ids]}
    if hasattr(res.model, "interference_terms"):
        out["interference_train"] = res.extra.get("interference_train")
    if model_name in ("Q2", "Q4"):
        summary["theta_ctx"] = np.asarray(res.model.context_table(p))
        summary["theta_L"] = np.asarray(p["theta_L"])
        summary["phi"] = float(np.asarray(p["phi"]))
        ss = res.model.subject_summary(p, data)
        summary["bloch_theta"] = ss["bloch_theta"].to_numpy()
        summary["bloch_phi"] = ss["bloch_phi"].to_numpy()
        summary["enc_W"] = np.asarray(p["enc"]["W"])
        summary["enc_b"] = np.asarray(p["enc"]["b"])
        if model_name == "Q4":
            summary["log_gamma"] = np.asarray(p["log_gamma"])
            summary["gamma_mu"] = float(np.asarray(p["gamma_mu"]))
            summary["gamma_sigma"] = float(np.exp(np.asarray(p["gamma_log_sigma"])))
    elif model_name == "B1":
        coef = res.model.coefficients(p)
        summary["coefficients"] = {str(k): float(v) for k, v in coef.items()}
        summary["ctx_coefficients"] = {str(k): float(v) for k, v in res.model.context_coefficients(p).items()}
    elif model_name == "B2":
        pop = res.model.population_params(p)
        summary["population"] = pop
        ss = res.model.subject_summary(p, data)
        summary["subject_intercept"] = ss["intercept"].to_numpy()
        summary["beta_ctx_subject"] = np.asarray(p["beta_ctx"])
    elif model_name == "B4":
        summary["natural"] = res.model.natural_params(p)
        ss = res.model.subject_summary(p, data)
        summary["prior_recovery_logit"] = ss["prior_recovery_logit"].to_numpy()
    else:  # any other registered model: its named population parameters, when it reports them
        if hasattr(res.model, "natural_params"):
            summary["natural"] = res.model.natural_params(p)
        if hasattr(res.model, "quarter_law_check"):
            out["quarter_law_train"] = res.model.quarter_law_check(p, data.subset(fit_rows), n_boot=int(job.get("n_boot", 200)), seed=seed)
    summary["ctx_vocab"] = list(data.ctx_vocab)
    out["summary"] = summary
    if job.get("artifact_dir"):
        art = artifact_from_fit(res, data, training_data_refs=[str(Path(job["parquet"]).relative_to(PROJECT_ROOT)) if Path(job["parquet"]).is_relative_to(PROJECT_ROOT) else str(job["parquet"])], metrics={"recovery": {k: v for k, v in out.items() if k in ("selection_nll", "selection_nll_ci95", "selection_brier", "selection_ece", "selection_auc", "n_fit_rows", "n_selection_rows")}}, notes="Phase 2 recovery-study fit on the 80% fit fold of a synthetic table.")
        art.save(Path(job["artifact_dir"]) / stem / model_name)
    out["job_seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------------------------


def _pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 3:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def recovery_gq(truth: dict[str, Any], fits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """G_Q recovery numbers for one dataset: ``theta_ctx`` correlation after gauge alignment
    (Q2, Q4), ``log gamma_i`` Spearman and ``bloch_theta`` Pearson (per Q-model)."""
    from bre.models.quantum.q2_context_unitary import align_gauge, fold_theta

    subs = truth["subjects"]
    mu_true = np.asarray(subs["theta_ctx"], dtype=np.float64).mean(axis=0)  # (4, 3)
    out: dict[str, Any] = {"theta_ctx_true_mean": mu_true.tolist(), "theta_L_true": truth["population"]["theta_L"], "phi_true": truth["population"]["phi"]}
    for m in ("Q2", "Q4"):
        f = fits.get(m)
        if f is None:
            continue
        s = f["summary"]
        vocab = list(s["ctx_vocab"])
        rows = [vocab.index(c) for c in subs["theta_ctx_contexts"]]
        params = {"enc": {"W": s["enc_W"], "b": s["enc_b"], "u": np.zeros((1, 4))}, "theta_L": s["theta_L"], "theta_ctx": np.asarray(s["theta_ctx"]), "phi": s["phi"]}
        aligned, element, corr = align_gauge(params, np.asarray(mu_true)[np.argsort(rows)] if rows != sorted(rows) else mu_true, rows=None if len(vocab) == 4 else np.asarray(rows))
        th_fit = fold_theta(np.asarray(aligned["theta_ctx"]))[rows]
        entry: dict[str, Any] = {
            "gauge": element,
            "theta_ctx_pearson": _pearson(th_fit, mu_true),
            "theta_ctx_fit_aligned": th_fit.tolist(),
            "theta_ctx_rmse": float(np.sqrt(np.mean((th_fit - mu_true) ** 2))),
            "theta_L_fit_aligned": np.asarray(aligned["theta_L"]).tolist(),
            "phi_fit_aligned": float(np.asarray(aligned["phi"])),
            "bloch_theta_pearson": _pearson(s["bloch_theta"], np.asarray(subs["bloch_theta"])),
        }
        if m == "Q4":
            lg_true = np.log(np.asarray(subs["gamma"], dtype=np.float64))
            finite = np.isfinite(lg_true)
            entry["log_gamma_spearman"] = _spearman(lg_true[finite], np.asarray(s["log_gamma"])[finite]) if finite.sum() >= 3 else float("nan")
            entry["log_gamma_pearson"] = _pearson(lg_true[finite], np.asarray(s["log_gamma"])[finite]) if finite.sum() >= 3 else float("nan")
            entry["gamma_mu_fit"] = s["gamma_mu"]
            entry["gamma_mu_true"] = truth["population"]["mu_gamma"]
            entry["sigma_gamma_fit"] = s["gamma_sigma"]
            entry["sigma_gamma_true"] = truth["population"]["sigma_gamma"]
            entry["log_gamma_true"] = lg_true.tolist()
            entry["log_gamma_fit"] = np.asarray(s["log_gamma"]).tolist()
        out[m] = entry
    return out


def recovery_gc(truth: dict[str, Any], fits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """G_C recovery numbers: context betas vs B1 (per-position main effects) and B2 (``mu_ctx``),
    subject intercepts vs B2, B4's evidence ordering (reported)."""
    pop = truth["population"]
    vocab = truth["context_vocab"]
    beta = np.asarray([pop["beta_ctx"][c] for c in vocab], dtype=np.float64)
    out: dict[str, Any] = {"beta_ctx_true": beta.tolist(), "beta_L_true": pop["beta_L"], "ctx_vocab": list(vocab)}
    if "B1" in fits:
        cc = fits["B1"]["summary"]["ctx_coefficients"]
        fitted = np.asarray([[cc[f"ctx{k}={c}"] for c in vocab] for k in range(2)])
        out["B1"] = {
            "beta_ctx_pearson_both_positions": _pearson(fitted.reshape(-1), np.concatenate([beta, beta])),
            "beta_ctx_pearson_position0": _pearson(fitted[0], beta),
            "beta_ctx_fit_position0": fitted[0].tolist(),
            "beta_ctx_fit_position1": fitted[1].tolist(),
            "beta_L_fit": fits["B1"]["summary"]["coefficients"].get("loss"),
        }
    if "B2" in fits:
        s = fits["B2"]["summary"]
        mu = np.asarray([s["population"]["mu_ctx"][c] for c in vocab])
        ids = s["subject_ids"]
        true_int = np.asarray([truth["subjects"][i]["sell_intercept"] for i in ids])
        out["B2"] = {
            "mu_ctx_pearson": _pearson(mu, beta),
            "mu_ctx_fit": mu.tolist(),
            "beta_L_fit": s["population"]["beta_L"],
            "subject_intercept_spearman": _spearman(np.asarray(s["subject_intercept"]), true_int),
            "subject_intercept_pearson": _pearson(np.asarray(s["subject_intercept"]), true_int),
            "subject_intercept_true": true_int.tolist(),
            "subject_intercept_fit": np.asarray(s["subject_intercept"]).tolist(),
        }
    if "B4" in fits:
        nat = fits["B4"]["summary"]["natural"]
        lam = np.asarray([nat["lambda"][c] for c in vocab])
        out["B4"] = {"lambda_vs_minus_beta_spearman": _spearman(lam, -beta), "lambda_fit": lam.tolist(), "delta": nat["delta"], "rho": nat["rho"], "kappa": nat["kappa"], "tau": nat["tau"]}
    return out


# ---------------------------------------------------------------------------------------------
# Aggregation, tables, figures
# ---------------------------------------------------------------------------------------------


def aggregate(results: list[dict[str, Any]], models: list[str]) -> pd.DataFrame:
    """One row per (gen, n, seed) with the selection-fold NLL of every model, the selected
    model/family and whether the family is the generator's."""
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for r in results:
        key = (r["gen"], r["n"], r["seed"])
        rows.setdefault(key, {"gen": r["gen"], "n": r["n"], "seed": r["seed"], "true_family": GENERATORS[r["gen"]]})
        rows[key][f"nll_{r['model']}"] = r["selection_nll"]
    out = []
    for key, row in sorted(rows.items()):
        nlls = {m: row.get(f"nll_{m}", np.nan) for m in models}
        finite = {m: v for m, v in nlls.items() if np.isfinite(v)}
        sel = min(finite, key=finite.get) if finite else None
        row["selected_model"] = sel
        row["selected_family"] = MODEL_FAMILY[sel] if sel else None
        row["correct_family"] = bool(sel and MODEL_FAMILY[sel] == row["true_family"])
        # margin: best of the other family minus the selected
        if sel:
            other = [v for m, v in finite.items() if MODEL_FAMILY[m] != MODEL_FAMILY[sel]]
            row["nll_margin_vs_other_family"] = float(min(other) - finite[sel]) if other else float("nan")
        out.append(row)
    return pd.DataFrame(out)


def confusion_tables(table: pd.DataFrame, n: int, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(generator x selected model counts, generator x selected family counts) at ``N = n``."""
    sub = table[table["n"] == n]
    gens = [g for g in DEFAULT_GENERATORS if g in set(sub["gen"])]
    by_model = pd.DataFrame(0, index=gens, columns=models, dtype=int)
    by_family = pd.DataFrame(0, index=gens, columns=["classical", "quantum"], dtype=int)
    for _, r in sub.iterrows():
        if r["selected_model"]:
            by_model.loc[r["gen"], r["selected_model"]] += 1
            by_family.loc[r["gen"], r["selected_family"]] += 1
    by_model.index.name = "true generator"
    by_family.index.name = "true generator"
    return by_model, by_family


def confusion_markdown(table: pd.DataFrame, n: int, models: list[str], results: list[dict[str, Any]]) -> str:
    from bre.eval import markdown_table

    by_model, by_family = confusion_tables(table, n, models)
    sub = table[table["n"] == n]
    lines = [f"# Recovery study: confusion matrices at N = {n}", ""]
    lines.append(f"Selection by held-out NLL per response on the within-subject 20% selection fold; {len(sub)} (generator, seed) cells; "
                 f"correct family-level selections: {int(sub['correct_family'].sum())}/{len(sub)} = {sub['correct_family'].mean():.2f}. Synthetic data only (is_synthetic = True). Generated by `bre.recover`.")
    lines.append("")
    lines.append("## True generator x selected model")
    lines.append("")
    lines.append(markdown_table(by_model.reset_index(), floatfmt=".0f"))
    lines.append("## True generator x selected family")
    lines.append("")
    lines.append(markdown_table(by_family.reset_index(), floatfmt=".0f"))
    lines.append("## Selection-fold NLL per response by cell")
    lines.append("")
    cols = ["gen", "seed"] + [f"nll_{m}" for m in models if f"nll_{m}" in sub.columns] + ["selected_model", "correct_family", "nll_margin_vs_other_family"]
    lines.append(markdown_table(sub[cols].reset_index(drop=True)))
    lines.append("## Fit diagnostics per cell and model")
    lines.append("")
    diag = pd.DataFrame([{
        "gen": r["gen"], "seed": r["seed"], "model": r["model"], "n_params": r["n_params"], "train_nll": r["train_nll"], "val_nll": r["val_nll"],
        "selection_nll": r["selection_nll"], "sel_lo": r["selection_nll_ci95"][0], "sel_hi": r["selection_nll_ci95"][1], "auc": r["selection_auc"],
        "steps": sum(int(x.get("steps_run", 0)) for x in r["restarts"]), "seconds": r["wall_time_s"],
        "elpd_waic_per_response": (r.get("waic") or {}).get("elpd_waic_per_response", float("nan")),
    } for r in results if r["n"] == n]).sort_values(["gen", "seed", "model"])
    lines.append(markdown_table(diag.reset_index(drop=True)))
    return "\n".join(lines)


def n_target(table: pd.DataFrame, items_per_subject: int) -> dict[str, Any]:
    per_n = table.groupby("n")["correct_family"].agg(["mean", "size", "sum"]).reset_index().sort_values("n")
    reaching = per_n[per_n["mean"] >= N_TARGET_THRESHOLD]
    out: dict[str, Any] = {
        "rule": N_TARGET_RULE,
        "threshold": N_TARGET_THRESHOLD,
        "items_per_subject": int(items_per_subject),
        "grid": [{"n": int(r["n"]), "correct_share": float(r["mean"]), "n_cells": int(r["size"]), "n_correct": int(r["sum"])} for _, r in per_n.iterrows()],
        "is_synthetic": True,
    }
    if len(reaching):
        n_star = int(reaching.iloc[0]["n"])
        out.update({"N_target": n_star, "responses_needed_for_calibration": n_star * int(items_per_subject), "reached": True})
    else:
        out.update({"N_target": None, "responses_needed_for_calibration": None, "reached": False, "reason": f"no N of the grid {[int(v) for v in per_n['n']]} reaches {N_TARGET_THRESHOLD:.0%} correct family-level selection; extend the grid (PLAN.md section 5: 100, 200, 400, 800) or fix identifiability before any real-data fit"})
    return out


def _legend(ax, size: int) -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=size)


def scatter_figures(recovery: dict[str, Any], n: int, out_dir: Path) -> list[Path]:
    """Recovery scatter plots (matplotlib Agg): G_Q theta_ctx (Q2, Q4), G_Q log gamma (Q4),
    G_C context betas (B1, B2) and subject intercepts (B2); every seed of ``N = n`` pooled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    gq_cells = [v for (g, nn, s), v in recovery.items() if g == "gq" and nn == n]
    if gq_cells:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, m in zip(axes[:2], ("Q2", "Q4")):
            for cell in gq_cells:
                e = cell.get(m)
                if e:
                    ax.scatter(np.asarray(cell["theta_ctx_true_mean"]).reshape(-1), np.asarray(e["theta_ctx_fit_aligned"]).reshape(-1), s=22, alpha=0.8, label=f"seed {cell['seed']} (r={e['theta_ctx_pearson']:.2f}, {e['gauge']})")
            lim = [-1.2, 1.2]
            ax.plot(lim, lim, color="gray", lw=0.8)
            ax.set_xlabel("true population theta_c (subject mean)")
            ax.set_ylabel(f"{m} fitted theta_c (gauge-aligned, folded)")
            ax.set_title(f"G_Q theta_c recovery, {m}, N={n}")
            _legend(ax, 7)
        ax = axes[2]
        for cell in gq_cells:
            e = cell.get("Q4")
            if e and "log_gamma_true" in e:
                ax.scatter(e["log_gamma_true"], e["log_gamma_fit"], s=8, alpha=0.5, label=f"seed {cell['seed']} (Spearman={e['log_gamma_spearman']:.2f})")
        ax.set_xlabel("true log gamma_i")
        ax.set_ylabel("Q4 fitted log gamma_i")
        ax.set_title(f"G_Q decoherence recovery, Q4, N={n}")
        _legend(ax, 7)
        fig.suptitle("SYNTHETIC data (G_Q); parameter recovery", fontsize=9)
        fig.tight_layout()
        p = out_dir / f"recovery_scatter_gq_N{n}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)
    gc_cells = [v for (g, nn, s), v in recovery.items() if g == "gc" and nn == n]
    if gc_cells:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        ax = axes[0]
        for cell in gc_cells:
            e = cell.get("B1")
            if e:
                ax.scatter(cell["beta_ctx_true"], e["beta_ctx_fit_position0"], s=30, marker="o", alpha=0.8, label=f"seed {cell['seed']} pos 0 (r={e['beta_ctx_pearson_position0']:.2f})")
                ax.scatter(cell["beta_ctx_true"], e["beta_ctx_fit_position1"], s=30, marker="x", alpha=0.8, label=f"seed {cell['seed']} pos 1")
        ax.plot([-1, 1], [-1, 1], color="gray", lw=0.8)
        ax.set_xlabel("true beta_ctx")
        ax.set_ylabel("B1 fitted ctx main effects")
        ax.set_title(f"G_C context betas, B1, N={n}")
        _legend(ax, 6)
        ax = axes[1]
        for cell in gc_cells:
            e = cell.get("B2")
            if e:
                ax.scatter(cell["beta_ctx_true"], e["mu_ctx_fit"], s=30, alpha=0.8, label=f"seed {cell['seed']} (r={e['mu_ctx_pearson']:.2f})")
        ax.plot([-1, 1], [-1, 1], color="gray", lw=0.8)
        ax.set_xlabel("true beta_ctx")
        ax.set_ylabel("B2 fitted mu_ctx")
        ax.set_title(f"G_C context betas, B2, N={n}")
        _legend(ax, 7)
        ax = axes[2]
        for cell in gc_cells:
            e = cell.get("B2")
            if e:
                ax.scatter(e["subject_intercept_true"], e["subject_intercept_fit"], s=8, alpha=0.5, label=f"seed {cell['seed']} (Spearman={e['subject_intercept_spearman']:.2f})")
        ax.set_xlabel("true subject sell intercept")
        ax.set_ylabel("B2 fitted intercept alpha_s")
        ax.set_title(f"G_C subject intercepts, B2, N={n}")
        _legend(ax, 7)
        fig.suptitle("SYNTHETIC data (G_C); parameter recovery", fontsize=9)
        fig.tight_layout()
        p = out_dir / f"recovery_scatter_gc_N{n}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)
    return written


def summary_markdown(payload: dict[str, Any]) -> str:
    from bre.eval import markdown_table

    lines = ["# Phase 2 recovery study: summary", ""]
    lines.append(f"Generated by `bre.recover` on {payload['finished_at']} from `results.json` (no hand-typed numbers). "
                 f"All data are SYNTHETIC (`is_synthetic = True`; generators G_Q, G_C, G_F of `bre.sim`). Wall time {payload['wall_time_s'] / 60:.1f} min "
                 f"with {payload['workers']} worker process(es); {payload['n_jobs']} fits ({payload['n_failed']} failed).")
    lines.append("")
    lines.append(f"Grid: generators {payload['generators']}, N {payload['ns']}, seeds {payload['seeds']}, models {payload['models']}. "
                 f"Fit settings per family: {json.dumps(payload['model_settings'])}. Selection fold: within-subject {SELECTION_FRAC:.0%} of the sell rows, stratified by condition type.")
    lines.append("")
    table = pd.DataFrame(payload["table"])
    lines.append("## Correct family-level selection by N")
    lines.append("")
    per_n = pd.DataFrame(payload["n_target"]["grid"])
    lines.append(markdown_table(per_n, floatfmt=".2f"))
    nt = payload["n_target"]
    if nt["reached"]:
        lines.append(f"**N_target = {nt['N_target']}** subjects -> responses needed for calibration = {nt['responses_needed_for_calibration']} ({nt['N_target']} x {nt['items_per_subject']} items). Rule: {nt['rule']}")
    else:
        lines.append(f"**N_target not reached.** {nt['reason']} Rule: {nt['rule']}")
    lines.append("")
    for n in payload["ns"]:
        lines.append(f"## N = {n}")
        lines.append("")
        by_model, by_family = confusion_tables(table, n, payload["models"])
        lines.append("True generator x selected model:")
        lines.append("")
        lines.append(markdown_table(by_model.reset_index(), floatfmt=".0f"))
        lines.append("True generator x selected family:")
        lines.append("")
        lines.append(markdown_table(by_family.reset_index(), floatfmt=".0f"))
        sub = table[table["n"] == n]
        cols = ["gen", "seed"] + [f"nll_{m}" for m in payload["models"] if f"nll_{m}" in sub.columns] + ["selected_model", "correct_family", "nll_margin_vs_other_family"]
        lines.append("Selection-fold NLL per response (lower is better):")
        lines.append("")
        lines.append(markdown_table(sub[cols].reset_index(drop=True)))
        rec = payload["recovery"]
        gq_rows = [{"seed": r["seed"], "model": m, "theta_ctx_pearson": r[m]["theta_ctx_pearson"], "theta_ctx_rmse": r[m]["theta_ctx_rmse"], "bloch_theta_pearson": r[m]["bloch_theta_pearson"], "gauge": r[m]["gauge"], "log_gamma_spearman": r[m].get("log_gamma_spearman", float("nan")), "phi_fit": r[m]["phi_fit_aligned"], "phi_true": r["phi_true"]} for r in rec if r["gen"] == "gq" and r["n"] == n for m in ("Q2", "Q4") if m in r]
        if gq_rows:
            lines.append("G_Q parameter recovery (population theta_c after gauge alignment; per-subject log gamma_i; Bloch polar angle):")
            lines.append("")
            lines.append(markdown_table(pd.DataFrame(gq_rows), floatfmt=".3f"))
        gc_rows = []
        for r in rec:
            if r["gen"] == "gc" and r["n"] == n:
                row = {"seed": r["seed"]}
                if "B1" in r:
                    row["B1_beta_ctx_pearson"] = r["B1"]["beta_ctx_pearson_both_positions"]
                    row["B1_beta_L"] = r["B1"]["beta_L_fit"]
                if "B2" in r:
                    row["B2_mu_ctx_pearson"] = r["B2"]["mu_ctx_pearson"]
                    row["B2_intercept_spearman"] = r["B2"]["subject_intercept_spearman"]
                    row["B2_beta_L"] = r["B2"]["beta_L_fit"]
                if "B4" in r:
                    row["B4_lambda_vs_minus_beta_spearman"] = r["B4"]["lambda_vs_minus_beta_spearman"]
                row["beta_L_true"] = r["beta_L_true"]
                gc_rows.append(row)
        if gc_rows:
            lines.append("G_C parameter recovery (context betas vs B1 / B2, subject intercepts vs B2, B4 evidence ordering):")
            lines.append("")
            lines.append(markdown_table(pd.DataFrame(gc_rows), floatfmt=".3f"))
        lines.append("")
    if payload.get("failed"):
        lines.append("## Failed fits")
        lines.append("")
        for f in payload["failed"]:
            lines.append(f"- {f['gen']} N={f['n']} seed={f['seed']} {f['model']}: {f['error']}")
        lines.append("")
    lines.append("## Reading the tables")
    lines.append("")
    lines.append("- A correct cell means the family (classical / quantum) of the model with the lowest selection-fold NLL is the generator's. G_Q data should select Q2 or Q4 (Q4 is the generating process); G_C data B1 or B2 (B2 is the generating process); G_F data B1 (the only model with ordered-pair terms).")
    lines.append("- Negative results are results: a wrong selection or a poor recovery correlation is reported as is; identifiability fixes (PLAN.md section 5) come before any real-data fit.")
    lines.append("- The Q-models' `theta_ctx` are identified only up to the gauge group (four sign patterns) and modulo pi along their direction; the correlations are computed after `align_gauge` + `fold_theta`, and a `sigma_z` component of a last unitary is expected to recover poorly (INTERFACE.md section 7).")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------


def step_cost(model_name: str) -> tuple[float, bool]:
    """``(seconds per full-batch step, measured)`` of a model for :func:`estimate_minutes`:
    :data:`SECONDS_PER_STEP` when measured, :data:`SECONDS_PER_STEP_ASSUMED` otherwise, and
    the largest measured cost for a model in neither table."""
    if model_name in SECONDS_PER_STEP:
        return SECONDS_PER_STEP[model_name], True
    return SECONDS_PER_STEP_ASSUMED.get(model_name, max(SECONDS_PER_STEP.values())), False


def estimate_minutes(jobs: list[dict[str, Any]], workers: int) -> float:
    """Rough wall-time estimate from :data:`SECONDS_PER_STEP` (worst case: no early stopping;
    assumed costs for models without a measurement, :func:`step_cost`; externally fitted
    models at :data:`EXTERNAL_FIT_SECONDS` per restart)."""
    total = 0.0
    for j in jobs:
        m, s = j["model"], j["settings"]
        scale = (j["n"] * 170 * 0.8 * 0.8) / 34000.0 * 1.25  # rows in the fit fold relative to the benchmark table, plus polish/eval overhead
        if m == "B2":
            total += s["restarts"] * s["svi_steps"] * SECONDS_PER_STEP[m] * max(scale, 0.4) + 20
        elif getattr(MODEL_REGISTRY[m], "requires_external_fit", False):
            total += s.get("restarts", 1) * EXTERNAL_FIT_SECONDS * scale + 10
        else:
            cost, _ = step_cost(m)
            total += s.get("restarts", 1) * (s.get("steps", 0) + 3 * s.get("lbfgs_steps", 0)) * cost * scale + 10
    return total / max(workers, 1) / 60.0


def build_jobs(generators, ns, seeds, models, data_dir: Path, model_settings: dict[str, dict[str, Any]], log_path: Path, artifact_dir: Path | None, n_boot: int, log=print) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for gen in generators:
        for n in ns:
            for seed in seeds:
                pq, tr = ensure_dataset(gen, n, seed, data_dir, log)
                for m in models:
                    settings = {**(model_settings.get(MODEL_FAMILY[m]) or {}), **(model_settings.get(m) or {})}
                    if m == "B2":
                        settings = {k: v for k, v in settings.items() if k in ("restarts", "svi_steps", "svi_lr", "svi_particles")}
                    jobs.append({"gen": gen, "n": int(n), "seed": int(seed), "model": m, "parquet": str(pq), "truth": str(tr), "settings": settings, "log_path": str(log_path), "artifact_dir": None if artifact_dir is None else str(artifact_dir), "n_boot": int(n_boot)})
    order = {"Q4": 0, "Q2": 1, "B2": 2, "B4": 3, "B1": 4}
    jobs.sort(key=lambda j: (order.get(j["model"], 9), -j["n"]))
    return jobs


def run(
    *,
    generators=DEFAULT_GENERATORS,
    ns=(200,),
    seeds=(0, 1, 2),
    models=DEFAULT_MODELS,
    workers: int = 4,
    data_dir: Path = DEFAULT_DATA_DIR,
    out_dir: Path = REPORTS_DIR,
    runs_dir: Path = RUNS_DIR,
    model_settings: dict[str, dict[str, Any]] | None = None,
    n_boot: int = 200,
    max_minutes: float = 110.0,
    force: bool = False,
    save_artifacts: bool = True,
    tag: str | None = None,
) -> dict[str, Any]:
    """Run the grid (module docstring) and write every output. Returns the results payload."""
    from bre.artifact import jsonable

    generators, ns, seeds, models = list(generators), [int(v) for v in ns], [int(v) for v in seeds], list(models)
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(f"unknown model(s) {unknown}")
    unknown_g = [g for g in generators if g not in GENERATORS]
    if unknown_g:
        raise KeyError(f"unknown generator(s) {unknown_g}")
    settings = {k: dict(v) for k, v in MODEL_SETTINGS_DEFAULT.items()}
    for k, v in (model_settings or {}).items():
        settings[k] = {**settings.get(k, {}), **v}
    stamp = tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / f"{stamp}.log"
    artifact_dir = runs_dir / stamp if save_artifacts else None

    def log(line: str) -> None:
        from bre.fit import _log

        _log(log_path, line, echo=True)

    t_start = time.time()
    log(f"[recover] start: generators={generators} ns={ns} seeds={seeds} models={models} workers={workers} settings={json.dumps(settings)}")
    jobs = build_jobs(generators, ns, seeds, models, Path(data_dir), settings, log_path, artifact_dir, n_boot, log)
    est = estimate_minutes(jobs, workers)
    assumed = sorted({m for m in models if m != "B2" and not step_cost(m)[1] and not getattr(MODEL_REGISTRY[m], "requires_external_fit", False)})
    log(f"[recover] {len(jobs)} fits queued; worst-case wall-time estimate {est:.0f} min with {workers} worker(s) (early stopping usually cuts this)" + (f"; step costs of {assumed} are assumed, not measured (SECONDS_PER_STEP_ASSUMED)" if assumed else ""))
    if est > max_minutes and not force:
        raise RuntimeError(f"estimated {est:.0f} min exceeds --max-minutes {max_minutes:.0f}; reduce the grid (seeds, steps, restarts) or pass --force (a run over two hours is a GATE, CLAUDE.md rule 4)")
    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    if workers > 1:
        _worker_init()  # the parent's environment is inherited by the spawned workers
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=int(workers), mp_context=mp.get_context("spawn"), initializer=_worker_init) as pool:
            futures = {pool.submit(run_job, j): j for j in jobs}
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failed.append({**{k: j[k] for k in ("gen", "n", "seed", "model")}, "error": f"{type(exc).__name__}: {exc}"})
                    log(f"[recover] FAILED {j['gen']} N={j['n']} seed={j['seed']} {j['model']}: {type(exc).__name__}: {exc}")
                    continue
                results.append(r)
                log(f"[recover] done {r['gen']} N={r['n']} seed={r['seed']} {r['model']}: selection_nll={r['selection_nll']:.4f} train_nll={r['train_nll']:.4f} val_nll={r['val_nll']:.4f} n_params={r['n_params']} ({r['job_seconds']:.0f}s) [{len(results) + len(failed)}/{len(jobs)}, {(time.time() - t_start) / 60:.1f} min elapsed]")
    else:
        for j in jobs:
            try:
                r = run_job(j)
            except Exception as exc:  # noqa: BLE001
                failed.append({**{k: j[k] for k in ("gen", "n", "seed", "model")}, "error": f"{type(exc).__name__}: {exc}"})
                log(f"[recover] FAILED {j['gen']} N={j['n']} seed={j['seed']} {j['model']}: {type(exc).__name__}: {exc}")
                continue
            results.append(r)
            log(f"[recover] done {r['gen']} N={r['n']} seed={r['seed']} {r['model']}: selection_nll={r['selection_nll']:.4f} train_nll={r['train_nll']:.4f} val_nll={r['val_nll']:.4f} n_params={r['n_params']} ({r['job_seconds']:.0f}s) [{len(results) + len(failed)}/{len(jobs)}, {(time.time() - t_start) / 60:.1f} min elapsed]")
    if not results:
        raise RuntimeError("every fit failed; see the log")

    table = aggregate(results, models)
    recovery: dict[tuple[str, int, int], dict[str, Any]] = {}
    fits_by_cell: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for r in results:
        fits_by_cell.setdefault((r["gen"], r["n"], r["seed"]), {})[r["model"]] = r
    for (gen, n, seed), fits in sorted(fits_by_cell.items()):
        truth = S.json_loads_strict((Path(data_dir) / f"{dataset_stem(gen, n, seed)}.truth.json").read_text(encoding="utf-8"))
        try:
            if gen == "gq":
                recovery[(gen, n, seed)] = {"gen": gen, "n": n, "seed": seed, **recovery_gq(truth, fits)}
            elif gen == "gc":
                recovery[(gen, n, seed)] = {"gen": gen, "n": n, "seed": seed, **recovery_gc(truth, fits)}
        except Exception as exc:  # noqa: BLE001
            log(f"[recover] parameter recovery failed for {gen} N={n} seed={seed}: {type(exc).__name__}: {exc}")
            recovery[(gen, n, seed)] = {"gen": gen, "n": n, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
    items = int(json.loads(Path(jobs[0]["truth"]).read_text(encoding="utf-8")).get("n_items", 170))
    nt = n_target(table, items)
    figures: list[Path] = []
    for n in ns:
        (out_dir / f"confusion_N{n}.md").write_text(confusion_markdown(table, n, models, results), encoding="utf-8")
        figures.extend(scatter_figures(recovery, n, out_dir))
    (out_dir / "N_target.json").write_text(json.dumps(jsonable(nt), indent=1) + "\n", encoding="utf-8")
    slim = [{k: v for k, v in r.items() if k not in ("selection_ll", "selection_subject_idx", "summary")} | {"summary": {k: v for k, v in r["summary"].items() if k not in ("subject_ids", "beta_ctx_subject")}} for r in results]
    payload = {
        "is_synthetic": True,
        "started_at": datetime.fromtimestamp(t_start, timezone.utc).isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": time.time() - t_start,
        "workers": int(workers),
        "generators": generators, "ns": ns, "seeds": seeds, "models": models,
        "model_settings": settings, "selection_frac": SELECTION_FRAC, "n_boot": int(n_boot),
        "n_jobs": len(jobs), "n_failed": len(failed), "failed": failed,
        "log": str(log_path), "artifact_dir": None if artifact_dir is None else str(artifact_dir),
        "table": table.to_dict(orient="records"),
        "n_target": nt,
        "recovery": [v for _, v in sorted(recovery.items())],
        "fits": slim,
        "figures": [str(p) for p in figures],
    }
    (out_dir / "results.json").write_text(json.dumps(jsonable(payload), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_markdown(jsonable(payload)), encoding="utf-8")
    log(f"[recover] finished in {payload['wall_time_s'] / 60:.1f} min; N_target={nt['N_target']}; outputs in {out_dir}")
    return payload


def _parse_list(text: str) -> list[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bre.recover", description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", default="200", help="comma list of N (subjects per table)")
    parser.add_argument("--seeds", default="0,1,2", help="comma list of seeds")
    parser.add_argument("--generators", default=",".join(DEFAULT_GENERATORS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=None, help="restarts for the classical MAP models (default 3)")
    parser.add_argument("--q-restarts", type=int, default=None, help="restarts for Q2/Q4 (default 2)")
    parser.add_argument("--steps", type=int, default=None, help="Adam step cap for the MAP models (default 1000, early stopping)")
    parser.add_argument("--svi-steps", type=int, default=None, help="B2 SVI steps (default 2000)")
    parser.add_argument("--lbfgs-steps", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=200, help="subject bootstrap draws for the selection-fold CIs")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out", default=str(REPORTS_DIR))
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--max-minutes", type=float, default=110.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--tag", default=None, help="name of the run (default: UTC timestamp)")
    args = parser.parse_args(argv)
    ms: dict[str, dict[str, Any]] = {"classical": {}, "quantum": {}, "B2": {}}
    if args.restarts is not None:
        ms["classical"]["restarts"] = args.restarts
        ms["B2"]["restarts"] = args.restarts
    if args.q_restarts is not None:
        ms["quantum"]["restarts"] = args.q_restarts
    if args.steps is not None:
        ms["classical"]["steps"] = args.steps
        ms["quantum"]["steps"] = args.steps
    if args.lbfgs_steps is not None:
        ms["classical"]["lbfgs_steps"] = args.lbfgs_steps
        ms["quantum"]["lbfgs_steps"] = args.lbfgs_steps
    if args.svi_steps is not None:
        ms["B2"]["svi_steps"] = args.svi_steps
    data_dir = Path(args.data_dir)
    if not S.is_synthetic_path(data_dir / "x.parquet"):
        parser.error(f"--data-dir must be a data/synthetic directory (CLAUDE.md rule 3); got {data_dir}")
    try:
        run(
            generators=_parse_list(args.generators), ns=[int(v) for v in _parse_list(args.n)], seeds=[int(v) for v in _parse_list(args.seeds)],
            models=_parse_list(args.models), workers=args.workers, data_dir=data_dir, out_dir=Path(args.out), runs_dir=Path(args.runs_dir),
            model_settings=ms, n_boot=args.n_boot, max_minutes=args.max_minutes, force=args.force, save_artifacts=not args.no_artifacts, tag=args.tag,
        )
    except RuntimeError as exc:
        print(f"[recover] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_GENERATORS",
    "DEFAULT_MODELS",
    "EXTERNAL_FIT_SECONDS",
    "MODEL_SETTINGS_DEFAULT",
    "N_TARGET_RULE",
    "N_TARGET_THRESHOLD",
    "SECONDS_PER_STEP",
    "SECONDS_PER_STEP_ASSUMED",
    "SELECTION_FRAC",
    "aggregate",
    "confusion_tables",
    "dataset_stem",
    "ensure_dataset",
    "estimate_minutes",
    "n_target",
    "recovery_gc",
    "recovery_gq",
    "run",
    "run_job",
    "selection_split",
    "step_cost",
]
