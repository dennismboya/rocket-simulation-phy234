"""Retrain the served model on the consented responses in the database (Phase 6, ``make retrain``).

:func:`retrain` does, in order:

1. pull every response row with ``consent_training = True`` from the database
   (``db.session.responses_to_frame``; rows without consent never enter a fit);
2. hold out 20% of the subjects (``bre.eval.split_subjects``, seeded) and fit the active
   model type (or ``model_type``) on the rest with :func:`bre.fit.fit_model` (its own
   within-subject validation fold drives early stopping; the held-out subjects are never seen);
3. evaluate the held-out subjects with :func:`bre.eval.evaluate_model`, scoring every held-out
   subject as *unseen* (``u = 0`` and the population values of the per-subject blocks,
   :func:`bre.artifact.params_for_data` without training ids), which is how the API scores a
   client the model was not fitted on;
4. score the **current active artifact** on the same held-out rows in the same way (the
   reference) and read its recorded held-out NLL (``metrics.held_out.nll``) when it has one;
5. promote the refit (register it as the active model) **only if** its held-out NLL is not worse
   than the reference (``nll_new <= nll_reference + PROMOTION_TOLERANCE``); otherwise the artifact
   is saved and registered inactive, and the result says why. Without an active artifact (or
   when the active artifact cannot score the rows, e.g. new context tags) the refit is promoted
   when its held-out NLL is finite;
6. save the artifact under ``<models_root>/<version>/`` (``retrain-<model>-<UTC stamp>``), the
   log and ``result.json`` under ``runs/retrain/<stamp>/``, and an ``audit_log`` row
   (``action = "retrain"``) with the numbers behind the decision.

Every random choice is a function of ``seed`` (subject split, restarts, Laplace draws;
CLAUDE.md rule 5). ``prior_strength`` (a dashboard setting) multiplies the strength of the
context-rotation prior of Q2/Q4 (``theta_prior_scale = THETA_PRIOR_SCALE / prior_strength``);
other models ignore it and the result says so. ``is_synthetic_training`` of the refit follows the
data (a demo database yields a synthetic-training artifact, and the API keeps demo mode on).

CLI: ``python -m bre.retrain [--model Q4] [--steps N] [--restarts R] [--n-samples K] [--seed S]
[--db sqlite:///...] [--models-root DIR]`` prints the result as JSON (``make retrain``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from bre import predict as P
from bre import schema as S
from bre.artifact import CALIBRATED, UNCALIBRATED, ModelArtifact, jsonable, params_for_data
from bre.eval import evaluate_model, split_subjects
from bre.fit import _log, artifact_from_fit, fit_model
from bre.models.data import build_model_data
from bre.models.quantum.q2_context_unitary import THETA_PRIOR_SCALE
from bre.registry import MODEL_FAMILY

RETRAIN_RUNS_DIR = S.PROJECT_ROOT / "runs" / "retrain"
MODELS_ROOT = S.PROJECT_ROOT / "runs" / "models"

PROMOTION_TOLERANCE = 1e-9
"""``nll_new <= nll_reference + PROMOTION_TOLERANCE`` promotes (equal counts as not worse)."""

MIN_SUBJECTS = 5
"""Fewer consented subjects with sell rows than this: no fit (the result says ``skipped``)."""

TEST_FRAC = 0.2
DEFAULT_STEPS = 1500
DEFAULT_RESTARTS = 3
DEFAULT_N_SAMPLES = 50
DEFAULT_VAL_FRAC = 0.2
DEFAULT_N_BOOT = 200

ACTOR = "retrain"

PROMOTION_RULE = (
    "promote only if the refit's held-out NLL per response (20% of the consented subjects held out, "
    "every held-out subject scored as unseen: u = 0 and the population values of the per-subject blocks) "
    "is not worse than the current active artifact scored on the same held-out rows in the same way; "
    "the active artifact's own recorded held-out NLL (metrics.held_out.nll) is reported next to it"
)


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------


def consented_responses(engine: Engine) -> pd.DataFrame:
    """Every stored response row with training consent (schema columns, frame order by
    dataset / subject / session / position). Scoring rows (``bre.predict.SCORING_DATASET``) are
    never stored, so nothing has to be excluded besides the consent flag."""
    from db.session import responses_to_frame, session_scope

    with session_scope(engine) as s:
        df = responses_to_frame(s)
    if len(df) == 0:
        return df
    keep = df["consent_training"].map(bool).to_numpy() & (df["dataset"] != P.SCORING_DATASET).to_numpy()
    return df.loc[keep].reset_index(drop=True)


def _known_tag_mask(df: pd.DataFrame, vocab: tuple[str, ...]) -> np.ndarray:
    return np.asarray([all(t in vocab for t in (tags if isinstance(tags, (list, tuple)) else [])) for tags in df["context_tags"].tolist()], dtype=bool)


# ---------------------------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------------------------


def _summary(held: dict[str, Any]) -> dict[str, Any]:
    return {k: held.get(k) for k in ("nll", "nll_ci95", "brier", "brier_ci95", "ece", "ece_ci95", "auc", "auc_ci95", "n_rows", "n_subjects", "reliability")}


def held_out_nll_of_fit(fit: Any, data: Any, test_rows: np.ndarray, *, n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Held-out metrics of a :class:`bre.fit.FitResult` on ``test_rows`` of ``data`` with every
    subject scored as unseen (module docstring, step 3)."""
    params = params_for_data(fit.model_name, fit.params, data, None)
    held = evaluate_model(fit.model, params, data, test_rows, n_boot=n_boot, seed=seed)
    return {**_summary(held), "scoring": "unseen-subject (u = 0, population per-subject values)"}


def reference_held_out(artifact: ModelArtifact, df: pd.DataFrame, test_rows: np.ndarray, *, n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> dict[str, Any] | None:
    """The active artifact scored on the same held-out rows as the refit, every subject unseen
    (module docstring, step 4). Rows whose context tags the artifact does not know are left out
    (``n_rows_dropped`` says how many); None when no held-out row can be scored."""
    mask = _known_tag_mask(df, artifact.ctx_vocab)
    test_rows = np.asarray(test_rows, dtype=np.int64)
    if not mask.all():
        keep_idx = np.flatnonzero(mask)
        pos = {int(i): k for k, i in enumerate(keep_idx)}
        sub_df = df.loc[keep_idx].reset_index(drop=True)
        rows = np.asarray([pos[int(i)] for i in test_rows if int(i) in pos], dtype=np.int64)
        dropped = int(len(test_rows) - len(rows))
    else:
        sub_df, rows, dropped = df, test_rows, 0
    if len(rows) == 0:
        return None
    data = artifact.build_data(sub_df)
    model = artifact.model_instance(data)
    params = params_for_data(artifact.model_name, artifact.params, data, None)
    held = evaluate_model(model, params, data, rows, n_boot=n_boot, seed=seed)
    return {**_summary(held), "n_rows_dropped": dropped, "scoring": "unseen-subject (u = 0, population per-subject values)", "version": artifact.version, "model_type": artifact.model_name}


def recorded_held_out_nll(artifact: ModelArtifact | None) -> float | None:
    """``metrics.held_out.nll`` of an artifact when it recorded a held-out evaluation."""
    if artifact is None:
        return None
    ho = (artifact.metrics or {}).get("held_out")
    if isinstance(ho, dict) and ho.get("nll") is not None and np.isfinite(float(ho["nll"])):
        return float(ho["nll"])
    return None


def decide_promotion(nll_new: float | None, nll_reference: float | None) -> tuple[bool, str]:
    """The promotion rule (:data:`PROMOTION_RULE`) as a pure function."""
    if nll_new is None or not np.isfinite(nll_new):
        return False, "refit held-out NLL is not finite"
    if nll_reference is None or not np.isfinite(nll_reference):
        return True, "no active artifact could be scored on the held-out rows: first model on this data"
    if nll_new <= nll_reference + PROMOTION_TOLERANCE:
        return True, f"held-out NLL {nll_new:.4f} <= reference {nll_reference:.4f}"
    return False, f"held-out NLL {nll_new:.4f} is worse than the active artifact's {nll_reference:.4f}; not promoted"


# ---------------------------------------------------------------------------------------------
# Retrain
# ---------------------------------------------------------------------------------------------


def _model_kwargs(model_type: str, prior_strength: float) -> tuple[dict[str, Any], str | None]:
    if prior_strength <= 0:
        raise ValueError("prior_strength must be > 0")
    if model_type in ("Q2", "Q4") and abs(prior_strength - 1.0) > 1e-12:
        return {"theta_prior_scale": float(THETA_PRIOR_SCALE) / float(prior_strength)}, None
    if abs(prior_strength - 1.0) > 1e-12:
        return {}, f"prior_strength={prior_strength} applies to Q2/Q4 only; {model_type} keeps its own priors"
    return {}, None


def retrain(
    engine: Engine,
    *,
    model_type: str | None = None,
    steps: int = DEFAULT_STEPS,
    restarts: int = DEFAULT_RESTARTS,
    n_samples: int = DEFAULT_N_SAMPLES,
    val_frac: float = DEFAULT_VAL_FRAC,
    seed: int = 0,
    test_frac: float = TEST_FRAC,
    prior_strength: float = 1.0,
    n_boot: int = DEFAULT_N_BOOT,
    models_root: str | Path | None = None,
    runs_root: str | Path | None = None,
    actor: str = ACTOR,
    echo: bool = False,
) -> dict[str, Any]:
    """Pull, refit, evaluate, promote-if-not-worse (module docstring). Returns the result dict
    that is also written to ``runs/retrain/<stamp>/result.json``."""
    from db.demo_seed import CALIBRATION_MIN_RESPONSES, register_artifact
    from db.session import session_scope, write_audit

    t_start = time.time()
    run_id = stamp()
    run_dir = Path(runs_root) if runs_root is not None else RETRAIN_RUNS_DIR
    run_dir = run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "retrain.log"
    root = Path(models_root) if models_root is not None else MODELS_ROOT
    db_label = str(engine.url.database or engine.url)

    active_row = P.active_registry_row(engine)
    active: ModelArtifact | None = None
    if active_row is not None:
        try:
            active = P.load_active_artifact(engine)
        except (LookupError, OSError, ValueError) as exc:
            _log(log_path, f"[retrain] active artifact could not be loaded: {exc}", echo)
    mt = model_type or (active.model_name if active is not None else "Q4")
    if mt not in MODEL_FAMILY:
        raise KeyError(f"unknown model type {mt!r}; registered {sorted(MODEL_FAMILY)}")
    result: dict[str, Any] = {
        "run_id": run_id, "run_dir": str(run_dir), "db": db_label, "model_type": mt, "actor": actor,
        "settings": {"steps": int(steps), "restarts": int(restarts), "n_samples": int(n_samples), "val_frac": float(val_frac), "seed": int(seed),
                     "test_frac": float(test_frac), "prior_strength": float(prior_strength), "n_boot": int(n_boot)},
        "active_before": None if active is None else {"version": active.version, "model_type": active.model_name, "is_synthetic_training": bool(active.is_synthetic_training),
                                                       "recorded_held_out_nll": recorded_held_out_nll(active)},
        "promotion_rule": PROMOTION_RULE, "promoted": False, "status": "started",
    }
    _log(log_path, f"[retrain] run {run_id}: db={db_label} model={mt} active={None if active is None else active.version} steps={steps} restarts={restarts}", echo)

    df = consented_responses(engine)
    n_subjects = int(df.loc[df["elicitation_type"] == "binary_sell", "subject_id"].nunique()) if len(df) else 0
    result["data"] = {"n_rows_consented": int(len(df)), "n_subjects_with_sell_rows": n_subjects,
                      "datasets": sorted(df["dataset"].unique().tolist()) if len(df) else [],
                      "is_synthetic": bool(df["is_synthetic"].any()) if len(df) else None,
                      "consent_rule": "rows with consent_training = False are never used"}
    if n_subjects < MIN_SUBJECTS:
        result["status"] = f"skipped: {n_subjects} consented subjects with sell rows, at least {MIN_SUBJECTS} needed"
        result["reason"] = result["status"]
        _finish(result, run_dir, log_path, echo, t_start)
        with session_scope(engine) as s:
            write_audit(s, actor=actor, action="retrain", entity="model_registry", entity_id="-", payload={"run_id": run_id, "promoted": False, "status": result["status"]})
        return result

    data = build_model_data(df)
    train_rows, test_rows = split_subjects(data, float(test_frac), int(seed))
    result["split"] = {"type": "subjects (bre.eval.split_subjects, stratified by dataset)", "test_frac": float(test_frac), "seed": int(seed),
                       "n_train_rows": int(len(train_rows)), "n_test_rows": int(len(test_rows)),
                       "n_test_subjects": int(len(np.unique(data.subject_idx[test_rows]))), "ctx_vocab": list(data.ctx_vocab)}
    kwargs, prior_note = _model_kwargs(mt, float(prior_strength))
    if prior_note:
        result["notes"] = [prior_note]
    _log(log_path, f"[retrain] {len(df)} consented rows, {data.n_subjects} subjects; split: {len(train_rows)} train / {len(test_rows)} test rows; model_kwargs={kwargs}", echo)

    fit = fit_model(mt, data, rows=train_rows, restarts=int(restarts), steps=int(steps), val_frac=float(val_frac), seed=int(seed),
                    n_samples=int(n_samples), svi_steps=int(steps), log_path=log_path, echo=echo, model_kwargs=kwargs)
    new_eval = held_out_nll_of_fit(fit, data, test_rows, n_boot=int(n_boot), seed=int(seed))
    ref_eval = None
    if active is not None:
        try:
            ref_eval = reference_held_out(active, df, test_rows, n_boot=int(n_boot), seed=int(seed))
        except Exception as exc:  # noqa: BLE001 - the reference is reported, never fatal
            _log(log_path, f"[retrain] reference scoring failed: {type(exc).__name__}: {exc}", echo)
            ref_eval = {"error": f"{type(exc).__name__}: {exc}"}
    nll_new = new_eval.get("nll")
    nll_ref = None if not ref_eval or ref_eval.get("nll") is None else float(ref_eval["nll"])
    promoted, reason = decide_promotion(None if nll_new is None else float(nll_new), nll_ref)
    result["held_out"] = {"new": new_eval, "reference": ref_eval, "recorded_reference_nll": recorded_held_out_nll(active),
                          "nll_new": nll_new, "nll_reference": nll_ref}
    result["fit"] = {"train_nll": fit.train_nll, "val_nll": fit.val_nll, "n_params": fit.n_params, "n_params_population": fit.n_params_population,
                     "best_restart": fit.best_restart, "wall_time_s": fit.wall_time, "model_kwargs": kwargs}
    _log(log_path, f"[retrain] held-out NLL new={nll_new} reference={nll_ref} -> promoted={promoted} ({reason})", echo)

    version = f"retrain-{mt.lower()}-{run_id}"
    sell_train = np.isin(np.arange(data.n), train_rows) & data.sell_mask()
    n_no_ctx = int((sell_train & ~data.ctx_mask.any(axis=1)).sum())
    refs = [f"db:{db_label}"] + [f"dataset:{d} (consented rows from the database)" for d in result["data"]["datasets"]]
    if active is not None:
        refs.append(f"reference:{active.version}")
    metrics = {
        "held_out": {**new_eval, "split": result["split"], "reference": ref_eval, "promotion": {"promoted": promoted, "reason": reason, "rule": PROMOTION_RULE}},
        "n_responses": int(sell_train.sum()), "n_subjects": int(data.n_subjects), "n_responses_no_context": n_no_ctx,
        "retrain": {"run_id": run_id, "settings": result["settings"], "consent_rule": result["data"]["consent_rule"]},
        "held_out_note": "held-out subjects (20%) scored as unseen; see held_out",
    }
    notes = (f"Retrained on {int(sell_train.sum())} consented sell responses of {data.n_subjects} subjects from {db_label} "
             f"({'SYNTHETIC' if result['data']['is_synthetic'] else 'real'} rows). {reason}.")
    artifact = artifact_from_fit(fit, data, training_data_refs=refs, metrics=metrics, notes=notes)
    artifact.version = version
    for tag, entry in artifact.calibrated_contexts.items():
        entry["status"] = CALIBRATED if int(entry.get("n_responses", 0)) >= CALIBRATION_MIN_RESPONSES else UNCALIBRATED
        entry["min_responses"] = CALIBRATION_MIN_RESPONSES
    artifact_dir = root / version
    artifact.save(artifact_dir)
    register_artifact(engine, artifact, artifact_dir, activate=promoted, actor=actor)
    result.update({"version": version, "artifact_dir": str(artifact_dir), "promoted": promoted, "reason": reason,
                   "is_synthetic_training": bool(artifact.is_synthetic_training), "calibrated_contexts": jsonable(artifact.calibrated_contexts),
                   "status": "promoted" if promoted else "not promoted"})
    with session_scope(engine) as s:
        write_audit(s, actor=actor, action="retrain", entity="model_registry", entity_id=version,
                    payload={"run_id": run_id, "promoted": promoted, "reason": reason, "nll_new": nll_new, "nll_reference": nll_ref,
                             "recorded_reference_nll": result["held_out"]["recorded_reference_nll"], "model_type": mt,
                             "n_rows_consented": int(len(df)), "n_subjects": int(data.n_subjects), "active_before": None if active is None else active.version,
                             "is_synthetic_training": bool(artifact.is_synthetic_training), "settings": result["settings"]})
    _finish(result, run_dir, log_path, echo, t_start)
    return result


def _finish(result: dict[str, Any], run_dir: Path, log_path: Path, echo: bool, t_start: float) -> None:
    result["wall_time_s"] = time.time() - t_start
    (run_dir / "result.json").write_text(json.dumps(jsonable(result), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    _log(log_path, f"[retrain] {result['status']} ({result['wall_time_s']:.0f}s); result at {run_dir / 'result.json'}", echo)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from db.session import get_engine, init_db

    parser = argparse.ArgumentParser(description="Retrain the served model on the consented responses in the database (promote only if not worse).")
    parser.add_argument("--db", default=None, help="SQLAlchemy URL (default $BRE_DB_URL, else the demo database under data/synthetic)")
    parser.add_argument("--model", default=None, help="model type (default: the active model's type)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--runs-root", default=None)
    args = parser.parse_args(argv)
    url = args.db or os.environ.get("BRE_DB_URL")
    if not url:
        from db.demo_seed import DEMO_DB_PATH

        url = f"sqlite:///{DEMO_DB_PATH}"
    engine = get_engine(url)
    init_db(engine)
    result = retrain(engine, model_type=args.model, steps=args.steps, restarts=args.restarts, n_samples=args.n_samples, seed=args.seed,
                     prior_strength=args.prior_strength, models_root=args.models_root, runs_root=args.runs_root, echo=True)
    print(json.dumps(jsonable({k: v for k, v in result.items() if k not in ("calibrated_contexts",)}), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACTOR", "DEFAULT_N_SAMPLES", "DEFAULT_RESTARTS", "DEFAULT_STEPS", "MIN_SUBJECTS", "MODELS_ROOT", "PROMOTION_RULE", "PROMOTION_TOLERANCE",
    "RETRAIN_RUNS_DIR", "TEST_FRAC", "consented_responses", "decide_promotion", "held_out_nll_of_fit", "main", "recorded_held_out_nll",
    "reference_held_out", "retrain", "stamp",
]
