"""Seed the demo database and fit the demo model (Phase 5 demo mode).

``seed_demo(engine)`` loads the synthetic demo book (:func:`bre.demo.build_demo_book`) into a
SQLite database that must live under ``data/synthetic/`` (the DB layer refuses synthetic rows
anywhere else), seeds the five advisor interventions (:func:`db.seed.seed_interventions`), fits
the demo model on the book's own synthetic responses and registers it in ``model_registry`` as
the active model. Everything the API serves in demo mode traces back to this function.

The demo model
--------------
Q4 (the open-system context-unitary model; Q2 when the Q4 fit fails) fitted by MAP with the
minimal optax loop of ``tests/test_q2_q4.py`` (Adam, cosine-decayed learning rate,
:data:`FIT_STEPS` <= 1500 steps, :data:`FIT_RESTARTS` restarts, best final objective kept). The
artifact (``bre.artifact.ModelArtifact``, saved under ``runs/models/demo-<model>-v<revision>/``,
:data:`DEMO_ARTIFACT_REVISION`) carries
``is_synthetic_training = True``, the training subjects' ids (so book clients keep their fitted
random effects), the train NLL per response, and ``calibrated_contexts`` computed from the
training rows: a context is ``"calibrated"`` when at least :data:`CALIBRATION_MIN_RESPONSES`
training sell rows show it, else ``"uncalibrated: prior only"``. Its metrics also hold the
``loss_response`` diagnostic of the training book (:func:`bre.predict.loss_response_diagnostic`:
the share of subjects whose fitted ``P(sell | L, no context)`` is non-decreasing over the design
grid, and the mean ``P(sell)`` per loss level).

Intervals come from :func:`laplace_samples`: a **diagonal Laplace approximation** at the
optimum over the population-level parameters (``bre.artifact`` module docstring lists its
limits; :data:`LAPLACE_MIN_CURVATURE` floors the curvature and :data:`LAPLACE_MAX_SD` caps the
standard deviation of unidentified directions), :data:`N_DRAWS` draws with a fixed seed. If
``bre.fit.fit_model`` exists it is preferred for the fit itself (same artifact contract).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from bre import demo as demo_mod
from bre import design as D
from bre import schema as S
from bre.artifact import CALIBRATED, UNCALIBRATED, ModelArtifact, calibrated_contexts_table, jsonable, to_numpy_tree
from bre.models import build_model_data
from bre.models.data import ModelData
from bre.registry import MODEL_FAMILY, PER_SUBJECT_BLOCKS, make_model, param_counts
from db.models import Client, ModelRegistry, utcnow
from db.seed import seed_interventions
from db.session import frame_to_responses, get_engine, init_db, session_scope, write_audit

DEMO_DB_PATH = S.SYNTHETIC_DIR / "demo.db"
"""Default location of the demo database (must be under ``data/synthetic``)."""

MODELS_ROOT = S.PROJECT_ROOT / "runs" / "models"
"""Artifacts are saved under ``runs/models/<version>/``."""

DEMO_MODEL = "Q4"
FALLBACK_MODEL = "Q2"
DEMO_ARTIFACT_REVISION = 2
"""Revision of the demo artifact: v1 was fitted on the book generated with the G_Q defaults before
2026-09-29 (loss rotation about the x axis); v2 on the book from the current defaults (``theta_L``
about the y axis, ``b = (1, 0.45, 0, 0.15)``, so the synthetic loss response rises with the loss).
A rebuild registers the new version as the active model and leaves older artifact directories on
disk (their registry rows are deactivated)."""

ARTIFACT_VERSION_TEMPLATE = "demo-{model}-v" + str(DEMO_ARTIFACT_REVISION)

FIT_STEPS = 1500
FIT_RESTARTS = 3
FIT_LR = 0.02
FIT_LR_END = 0.05
"""Optimizer settings of the demo fit (``tests/test_q2_q4.py`` pattern)."""

CALIBRATION_MIN_RESPONSES = 30
"""A context is calibrated when at least this many training sell rows show it (design choice:
about two batteries' worth of a context at the book's item mix)."""

N_DRAWS = 200
LAPLACE_MIN_CURVATURE = 1.0
LAPLACE_MAX_SD = 1.0
LAPLACE_SEED = 0
"""Laplace draws: curvature floor / standard-deviation cap for flat (unidentified) directions."""

ACTOR = "demo_seed"


# ---------------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------------


def fit_map(model, data: ModelData, key, steps: int = FIT_STEPS, lr: float = FIT_LR, restarts: int = FIT_RESTARTS, lr_end: float = FIT_LR_END) -> tuple[dict[str, Any], float, float, list[float]]:
    """MAP fit: Adam with a cosine-decayed learning rate on ``model.objective`` restricted to the
    sell rows, best of ``restarts`` restarts by final objective. Returns ``(params, objective,
    seconds, objectives_per_restart)``."""
    if steps < 1 or steps > 1500:
        raise ValueError("steps must be in [1, 1500] for the demo fit")
    sub = data.subset(data.sell_rows())
    schedule = optax.cosine_decay_schedule(lr, steps, alpha=lr_end)
    opt = optax.adam(schedule)

    def objective(p):
        return model.objective(p, sub)

    @jax.jit
    def step(p, s):
        v, g = jax.value_and_grad(objective)(p)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, v

    best = None
    finals: list[float] = []
    t0 = time.time()
    for r in range(restarts):
        p = model.init_params(key, data, restart=r)
        s = opt.init(p)
        for _ in range(steps):
            p, s, _v = step(p, s)
        v = float(objective(p))
        finals.append(v)
        if not np.isfinite(v):
            continue
        if best is None or v < best[1]:
            best = (p, v)
    if best is None:
        raise RuntimeError("every restart ended with a non-finite objective")
    return best[0], best[1], time.time() - t0, finals


def _split_population(model_name: str, params: dict[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, ...], np.ndarray]]:
    """Population subtree of ``params`` and the per-subject blocks (by path)."""
    pop = to_numpy_tree(params)
    fixed: dict[tuple[str, ...], np.ndarray] = {}
    for path in PER_SUBJECT_BLOCKS[model_name]:
        node = pop
        for p in path[:-1]:
            node = node[p]
        fixed[path] = np.asarray(node.pop(path[-1]))
    return pop, fixed


def _merge(pop: dict[str, Any], fixed: dict[tuple[str, ...], Any]) -> dict[str, Any]:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in pop.items()}
    for path, block in fixed.items():
        node = out
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = block
    return out


def laplace_samples(model, model_name: str, params: dict[str, Any], data: ModelData, k: int = N_DRAWS, seed: int = LAPLACE_SEED) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Diagonal Laplace draws at the optimum (module docstring): ``theta_pop ~ N(theta_pop*,
    diag(1 / H_jj))`` with ``H`` the Hessian of ``model.objective`` over the population block
    (Hessian-vector products along every basis vector), per-subject blocks fixed. Returns the
    draws as full parameter pytrees and a diagnostics dict."""
    sub = data.subset(data.sell_rows())
    pop, fixed = _split_population(model_name, params)
    flat, unravel = jax.flatten_util.ravel_pytree({k: (jnp.asarray(v) if not isinstance(v, dict) else {kk: jnp.asarray(vv) for kk, vv in v.items()}) for k, v in pop.items()})
    fixed_j = {path: jnp.asarray(v) for path, v in fixed.items()}

    def objective(x):
        return model.objective(_merge(unravel(x), fixed_j), sub)

    grad = jax.grad(objective)
    hvp = jax.jit(lambda v: jax.jvp(grad, (flat,), (v,))[1])
    n = int(flat.shape[0])
    eye = np.eye(n)
    diag = np.asarray([float(hvp(jnp.asarray(eye[j]))[j]) for j in range(n)])
    curvature = np.where(np.isfinite(diag), diag, 0.0)
    floored = curvature < LAPLACE_MIN_CURVATURE
    sd = 1.0 / np.sqrt(np.maximum(curvature, LAPLACE_MIN_CURVATURE))
    sd = np.minimum(sd, LAPLACE_MAX_SD)
    rng = np.random.default_rng(seed)
    mean = np.asarray(flat)
    draws = []
    for _ in range(int(k)):
        x = mean + sd * rng.standard_normal(n)
        draws.append(to_numpy_tree(_merge(unravel(jnp.asarray(x)), fixed)))
    diag_info = {
        "method": "diagonal Laplace at the MAP over the population-level parameters; per-subject blocks fixed",
        "n_population_params": n, "n_floored": int(floored.sum()), "min_curvature": LAPLACE_MIN_CURVATURE, "max_sd": LAPLACE_MAX_SD,
        "sd_min": float(sd.min()), "sd_max": float(sd.max()), "n_draws": int(k), "seed": int(seed),
    }
    return draws, diag_info


def calibration_table(model_name: str, params: dict[str, Any], data: ModelData, samples: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """``bre.artifact.calibrated_contexts_table`` re-statused with :data:`CALIBRATION_MIN_RESPONSES`."""
    table = calibrated_contexts_table(model_name, params, data, param_samples=samples)
    for tag, entry in table.items():
        entry["status"] = CALIBRATED if entry["n_responses"] >= CALIBRATION_MIN_RESPONSES else UNCALIBRATED
        entry["min_responses"] = CALIBRATION_MIN_RESPONSES
    return table


def fit_demo_model(
    responses: pd.DataFrame,
    *,
    model_name: str = DEMO_MODEL,
    steps: int = FIT_STEPS,
    restarts: int = FIT_RESTARTS,
    seed: int = 0,
    n_draws: int = N_DRAWS,
    training_refs: list[str] | None = None,
    allow_fallback: bool = True,
) -> tuple[ModelArtifact, dict[str, Any]]:
    """Fit the demo model on the book's synthetic responses and build its artifact (not saved).

    Tries ``bre.fit.fit_model`` when it exists (same artifact contract), else the local
    :func:`fit_map`. On a failure of ``model_name`` (exception or non-finite objective) the
    :data:`FALLBACK_MODEL` is fitted instead when ``allow_fallback``. Returns ``(artifact,
    fit_summary)``.
    """
    if not bool(responses["is_synthetic"].all()):
        raise ValueError("the demo model is fitted on synthetic responses only")
    data = build_model_data(responses)
    key = jax.random.PRNGKey(int(seed))
    order = [model_name] + ([FALLBACK_MODEL] if allow_fallback and model_name != FALLBACK_MODEL else [])
    errors: list[str] = []
    for name in order:
        try:
            model = make_model(name, data)
            params, obj, seconds, finals = fit_map(model, data, key, steps=steps, restarts=restarts)
            break
        except Exception as exc:  # noqa: BLE001 - fall back to the next model, report why
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
    else:
        raise RuntimeError("demo fit failed: " + "; ".join(errors))
    ll = np.asarray(model.log_lik(params, data))
    sell = data.sell_mask()
    train_nll = float(-ll[sell].mean())
    p = np.asarray(model.predict_proba(params, data))[sell]
    y = data.y[sell]
    brier = float(np.mean((p - y) ** 2))
    samples, laplace = laplace_samples(model, name, params, data, k=n_draws, seed=seed) if n_draws > 0 else (None, {"method": "none"})
    table = calibration_table(name, params, data, samples)
    counts = param_counts(name, params)
    no_ctx = int((sell & ~data.ctx_mask.any(axis=1)).sum())
    metrics = {
        "train_nll_per_response": train_nll, "train_brier": brier, "objective": float(obj), "objectives_per_restart": finals,
        "steps": int(steps), "restarts": int(restarts), "lr": FIT_LR, "seconds": seconds, "n_responses": int(sell.sum()),
        "n_subjects": int(data.n_subjects), "n_responses_no_context": no_ctx, "n_params_all": counts["all"], "n_params_population": counts["population"],
        "laplace": laplace, "held_out": None, "held_out_note": "demo fit on the whole synthetic book; no held-out evaluation (Phase 4 evaluates real data)",
        "fallback_errors": errors,
    }
    notes = (
        f"Demo model fitted on the synthetic demo book only (bre.demo.build_demo_book; {data.n_subjects} synthetic investors, "
        f"{int(sell.sum())} sell responses). Every number it produces is a synthetic-training result. "
        f"MAP via Adam ({steps} steps, {restarts} restarts, best objective kept). Intervals: {laplace['method']} "
        f"({laplace.get('n_draws', 0)} draws). Calibrated context = at least {CALIBRATION_MIN_RESPONSES} training sell rows."
        + (f" Fallback used after: {errors}" if errors else "")
    )
    artifact = ModelArtifact(
        version=ARTIFACT_VERSION_TEMPLATE.format(model=name.lower()),
        model_name=name,
        family=MODEL_FAMILY[name],
        params=to_numpy_tree(params),
        param_samples=samples,
        ctx_vocab=tuple(data.ctx_vocab),
        covariate_stats=dict(data.covariate_stats),
        X_columns=tuple(data.X_columns),
        design_version=D.DESIGN_VERSION,
        training_data_refs=list(training_refs or [f"dataset:{demo_mod.DATASET}"]),
        metrics=jsonable(metrics),
        calibrated_contexts=jsonable(table),
        n_params=int(counts["all"]),
        created_at=ModelArtifact.now(),
        is_synthetic_training=True,
        notes=notes,
        subject_ids=tuple(str(s) for s in data.subject_ids),
    )
    from bre.predict import loss_response_diagnostic

    book = responses.drop_duplicates("subject_id")[["subject_id", "covariates"]].rename(columns={"subject_id": "client_id"}).reset_index(drop=True)
    loss_response = loss_response_diagnostic(artifact, book)
    artifact.metrics["loss_response"] = {k: loss_response[k] for k in ("monotone_share", "n_clients", "n_monotone", "loss_levels", "mean_p_sell_by_loss", "p_sell_by_loss_ci80", "context", "question_order_id", "definition")}
    artifact.metrics["loss_response_monotone_share"] = loss_response["monotone_share"]
    summary = {"model": name, "steps": steps, "restarts": restarts, "train_nll_per_response": train_nll, "objective": float(obj), "seconds": seconds, "fallback_errors": errors,
               "theta_L": np.asarray(artifact.params["theta_L"], dtype=np.float64).tolist(), "loss_response_monotone_share": loss_response["monotone_share"],
               "mean_p_sell_by_loss": loss_response["mean_p_sell_by_loss"]}
    return artifact, summary


# ---------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------


def registry_calibrated_list(table: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The artifact's ``calibrated_contexts`` dict as the JSON array the registry column holds."""
    return [{"context": tag, **{k: v for k, v in entry.items() if k in ("n_responses", "status", "theta", "min_responses")}} for tag, entry in table.items()]


def register_artifact(engine: Engine, artifact: ModelArtifact, artifact_dir: Path, *, activate: bool = True, actor: str = ACTOR) -> None:
    """Upsert a ``model_registry`` row for ``artifact`` (``artifact:<dir>`` in
    ``training_data_refs`` names the directory), activate it and deactivate every other row."""
    try:
        rel = str(Path(artifact_dir).resolve().relative_to(S.PROJECT_ROOT.resolve()))
    except ValueError:
        rel = str(Path(artifact_dir).resolve())
    refs = list(artifact.training_data_refs) + [f"artifact:{rel}"]
    with session_scope(engine) as s:
        if activate:
            s.execute(update(ModelRegistry).values(is_active=False))
        row = s.get(ModelRegistry, artifact.version)
        values = dict(
            model_type=artifact.model_name,
            training_data_refs=json.dumps(refs),
            metrics=json.dumps(jsonable({**artifact.metrics, "is_synthetic_training": artifact.is_synthetic_training, "family": artifact.family, "n_params": artifact.n_params, "created_at": artifact.created_at})),
            calibrated_contexts=json.dumps(registry_calibrated_list(artifact.calibrated_contexts)),
            promoted_at=utcnow() if activate else None,
            is_active=bool(activate),
        )
        if row is None:
            s.add(ModelRegistry(version=artifact.version, **values))
        else:
            for k, v in values.items():
                setattr(row, k, v)
        s.flush()
        write_audit(s, actor=actor, action="register_model", entity="model_registry", entity_id=artifact.version,
                    payload={"artifact_dir": rel, "is_active": activate, "is_synthetic_training": artifact.is_synthetic_training})


# ---------------------------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------------------------


def demo_engine(db_path: str | Path | None = None) -> Engine:
    """Engine of the demo database (default :data:`DEMO_DB_PATH`)."""
    path = Path(db_path) if db_path is not None else DEMO_DB_PATH
    return get_engine(f"sqlite:///{path}")


def load_book(engine: Engine, clients: pd.DataFrame, responses: pd.DataFrame, *, actor: str = ACTOR) -> int:
    """Insert the book's clients and responses (one ``frame_to_responses`` call per client so
    the rows link to their client); returns the number of response rows written."""
    n = 0
    with session_scope(engine) as s:
        for rec in clients.to_dict(orient="records"):
            s.add(Client(client_id=rec["client_id"], display_label=rec["display_label"], covariates=rec["covariates"]))
        s.flush()
        for cid in clients["client_id"].tolist():
            part = responses[responses["subject_id"] == cid].reset_index(drop=True)
            n += frame_to_responses(s, part, client_id=cid, actor=actor)
        write_audit(s, actor=actor, action="seed_demo_book", entity="clients", entity_id="-",
                    payload={"n_clients": int(len(clients)), "n_responses": n, "is_synthetic": True, "dataset": demo_mod.DATASET})
    return n


def seed_demo(
    engine: Engine | None = None,
    *,
    seed: int = 0,
    n_generated: int = demo_mod.N_GENERATED,
    n_items: int = demo_mod.BATTERY_ITEMS,
    models_root: str | Path | None = None,
    model_name: str = DEMO_MODEL,
    fit_steps: int = FIT_STEPS,
    fit_restarts: int = FIT_RESTARTS,
    n_draws: int = N_DRAWS,
    rebuild: bool = False,
    write_files: bool = True,
) -> dict[str, Any]:
    """Load the demo book, seed the interventions, fit and register the demo model.

    Idempotent: when the database already holds clients and ``rebuild`` is False, the book is
    left alone and only a missing model is fitted. With ``rebuild`` the clients (and, by
    cascade, their responses and logs) are deleted and everything is rebuilt. Returns a summary
    dict (paths, counts, the fit summary).
    """
    engine = engine if engine is not None else demo_engine()
    init_db(engine)
    root = Path(models_root) if models_root is not None else MODELS_ROOT
    summary: dict[str, Any] = {"db_url": str(engine.url), "models_root": str(root)}
    with session_scope(engine) as s:
        existing = s.scalars(select(Client.client_id)).all()
        seeded = seed_interventions(s)
    summary["interventions_seeded"] = seeded
    clients: pd.DataFrame | None = None
    responses: pd.DataFrame | None = None
    if existing and rebuild:
        from db.session import delete_client

        with session_scope(engine) as s:
            for cid in existing:
                delete_client(s, cid, actor=ACTOR)
        existing = []
    if not existing:
        clients, responses = demo_mod.build_demo_book(seed, n_generated, n_items)
        summary["n_responses_loaded"] = load_book(engine, clients, responses)
        summary["n_clients"] = int(len(clients))
        if write_files:
            db_file = engine.url.database
            out_dir = Path(db_file).parent if db_file else S.SYNTHETIC_DIR
            try:
                pq_path, cl_path = demo_mod.write_demo_book(clients, responses, out_dir)
                summary["book_files"] = [str(pq_path), str(cl_path)]
            except Exception as exc:  # noqa: BLE001 - the DB is the source of truth; files are a convenience
                summary["book_files_error"] = f"{type(exc).__name__}: {exc}"
    else:
        summary["n_clients"] = len(existing)
        summary["book"] = "kept (already seeded)"
    from bre.predict import active_registry_row, artifact_path_of, resolve_artifact_dir

    row = active_registry_row(engine)
    have_model = row is not None and artifact_path_of(row) is not None and Path(resolve_artifact_dir(artifact_path_of(row)), "artifact.json").exists()
    if have_model and not rebuild:
        summary["model"] = {"version": row["version"], "kept": True}
        return summary
    if responses is None:
        from db.session import responses_to_frame

        with session_scope(engine) as s:
            responses = responses_to_frame(s, dataset=demo_mod.DATASET, is_synthetic=True)
    refs = [
        f"dataset:{demo_mod.DATASET} (bre.demo.build_demo_book seed={seed}, {n_generated} G_Q investors + {len(demo_mod.ARCHETYPE_NAMES)} archetypes, {n_items} items each)",
        f"db:{engine.url.database}",
        f"generator:bre.sim.gq POPULATION_DEFAULTS ({demo_mod.gq.SOURCE}); archetypes: bre.demo.ARCHETYPES ({demo_mod.ARCHETYPE_SOURCE})",
    ]
    artifact, fit_summary = fit_demo_model(responses, model_name=model_name, steps=fit_steps, restarts=fit_restarts, seed=seed, n_draws=n_draws, training_refs=refs)
    artifact_dir = root / artifact.version
    artifact.save(artifact_dir)
    register_artifact(engine, artifact, artifact_dir)
    summary["model"] = {**fit_summary, "version": artifact.version, "artifact_dir": str(artifact_dir), "kept": False}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the demo database (synthetic book + demo model).")
    parser.add_argument("--db", default=str(DEMO_DB_PATH), help="SQLite file under data/synthetic/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=FIT_STEPS)
    parser.add_argument("--restarts", type=int, default=FIT_RESTARTS)
    parser.add_argument("--rebuild", action="store_true", help="delete the seeded book and refit")
    args = parser.parse_args(argv)
    summary = seed_demo(demo_engine(args.db), seed=args.seed, fit_steps=args.steps, fit_restarts=args.restarts, rebuild=args.rebuild)
    print(json.dumps(jsonable(summary), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACTOR",
    "CALIBRATION_MIN_RESPONSES",
    "DEMO_ARTIFACT_REVISION",
    "DEMO_DB_PATH",
    "DEMO_MODEL",
    "FALLBACK_MODEL",
    "FIT_RESTARTS",
    "FIT_STEPS",
    "LAPLACE_MAX_SD",
    "LAPLACE_MIN_CURVATURE",
    "MODELS_ROOT",
    "N_DRAWS",
    "calibration_table",
    "demo_engine",
    "fit_demo_model",
    "fit_map",
    "laplace_samples",
    "load_book",
    "register_artifact",
    "registry_calibrated_list",
    "seed_demo",
]
