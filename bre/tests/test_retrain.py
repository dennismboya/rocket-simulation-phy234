"""Tests of ``bre.retrain`` (``make retrain``, ``POST /retrain``) on a small demo database
(10 generated investors + the 5 archetypes, short demo fit) built once per module in a
temporary ``data/synthetic`` directory: the consent filter, the held-out split, the artifact and
registry rows, the audit row, and the promotion rule (a worse refit is never promoted; a
not-worse one is, and the API then serves it).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select, update

from bre import predict as P
from bre import retrain as R
from db.models import AuditLog, ModelRegistry, Response
from db.session import session_scope

SMALL_N = 10
FIT_STEPS = 60
RETRAIN_STEPS = 40


class SmallEnv:
    def __init__(self, root: Path) -> None:
        from db.demo_seed import demo_engine, seed_demo

        self.root = root
        self.db_path = root / "data" / "synthetic" / "demo.db"
        self.models_root = root / "runs" / "models"
        self.runs_root = root / "runs" / "retrain"
        self.engine = demo_engine(self.db_path)
        self.summary = seed_demo(self.engine, n_generated=SMALL_N, models_root=self.models_root, fit_steps=FIT_STEPS, fit_restarts=1, n_draws=5, write_files=False)
        self.artifact = P.load_active_artifact(self.engine)


@pytest.fixture(scope="module")
def small_env(tmp_path_factory: pytest.TempPathFactory) -> SmallEnv:
    return SmallEnv(tmp_path_factory.mktemp("bre_retrain"))


def _active_version(env: SmallEnv) -> str:
    row = P.active_registry_row(env.engine)
    assert row is not None
    return str(row["version"])


def _run(env: SmallEnv, **kw) -> dict:
    return R.retrain(env.engine, steps=RETRAIN_STEPS, restarts=1, n_samples=2, n_boot=5, models_root=env.models_root, runs_root=env.runs_root, **kw)


# ---------------------------------------------------------------------------------------------
# Unit rules
# ---------------------------------------------------------------------------------------------


def test_decide_promotion_rule():
    assert R.decide_promotion(0.60, 0.61) == (True, "held-out NLL 0.6000 <= reference 0.6100")
    assert R.decide_promotion(0.61, 0.61)[0] is True  # equal counts as not worse
    ok, why = R.decide_promotion(0.62, 0.61)
    assert ok is False and "worse" in why
    assert R.decide_promotion(0.62, None)[0] is True  # no reference: first model on this data
    assert R.decide_promotion(float("nan"), 0.61)[0] is False and R.decide_promotion(None, None)[0] is False


def test_consented_responses_excludes_rows_without_consent(small_env: SmallEnv):
    df = R.consented_responses(small_env.engine)
    n_all = len(df)
    assert n_all > 0 and df["consent_training"].all() and df["is_synthetic"].all()
    subject = df["subject_id"].iloc[0]
    with session_scope(small_env.engine) as s:
        s.execute(update(Response).where(Response.subject_id == subject).values(consent_training=False))
    try:
        df2 = R.consented_responses(small_env.engine)
        assert subject not in set(df2["subject_id"]) and len(df2) == n_all - int((df["subject_id"] == subject).sum())
    finally:
        with session_scope(small_env.engine) as s:
            s.execute(update(Response).where(Response.subject_id == subject).values(consent_training=True))


# ---------------------------------------------------------------------------------------------
# The full path
# ---------------------------------------------------------------------------------------------


def test_retrain_writes_artifact_registry_row_log_and_audit(small_env: SmallEnv):
    before = _active_version(small_env)
    res = _run(small_env, seed=1)
    assert res["status"] in ("promoted", "not promoted") and res["model_type"] == small_env.artifact.model_name
    assert res["data"]["n_subjects_with_sell_rows"] == SMALL_N + 5 and res["data"]["is_synthetic"] is True and res["data"]["n_rows_consented"] > 0
    sp = res["split"]
    assert sp["n_test_subjects"] >= 1 and sp["n_train_rows"] + sp["n_test_rows"] == res["data"]["n_rows_consented"] and sp["seed"] == 1
    ho = res["held_out"]
    assert np.isfinite(ho["nll_new"]) and np.isfinite(ho["nll_reference"]) and ho["new"]["n_rows"] > 0 and ho["reference"]["version"] == small_env.artifact.version
    assert ho["new"]["scoring"].startswith("unseen-subject") and ho["reference"]["scoring"].startswith("unseen-subject")
    assert res["promoted"] == (ho["nll_new"] <= ho["nll_reference"] + R.PROMOTION_TOLERANCE)
    assert res["version"].startswith("retrain-q") and (Path(res["artifact_dir"]) / "artifact.json").exists()
    assert (Path(res["run_dir"]) / "result.json").exists() and (Path(res["run_dir"]) / "retrain.log").read_text().count("[retrain]") >= 3
    saved = json.loads((Path(res["run_dir"]) / "result.json").read_text())
    assert saved["version"] == res["version"] and saved["promotion_rule"] == R.PROMOTION_RULE
    with session_scope(small_env.engine) as s:
        row = s.get(ModelRegistry, res["version"])
        assert row is not None and bool(row.is_active) == res["promoted"]
        metrics = json.loads(row.metrics)
        assert metrics["held_out"]["nll"] == ho["nll_new"] and metrics["held_out"]["promotion"]["promoted"] == res["promoted"] and metrics["is_synthetic_training"] is True
        audit = s.scalars(select(AuditLog).where(AuditLog.action == "retrain", AuditLog.entity_id == res["version"])).all()
        assert len(audit) == 1 and json.loads(audit[0].payload)["promoted"] == res["promoted"] and json.loads(audit[0].payload)["nll_reference"] == ho["nll_reference"]
    assert _active_version(small_env) == (res["version"] if res["promoted"] else before)
    art = P.load_artifact(res["artifact_dir"])
    assert art.is_synthetic_training and art.metrics["n_responses_no_context"] >= 0 and set(art.calibrated_contexts) >= set(small_env.artifact.ctx_vocab)
    assert all(e["status"] in ("calibrated", "uncalibrated: prior only") and e["min_responses"] == 30 for e in art.calibrated_contexts.values())
    # restore the original active model for the next tests
    from db.demo_seed import register_artifact

    orig_dir = small_env.models_root / small_env.artifact.version
    register_artifact(small_env.engine, small_env.artifact, orig_dir, activate=True)
    assert _active_version(small_env) == small_env.artifact.version


def test_retrain_never_promotes_a_worse_model(small_env: SmallEnv, monkeypatch: pytest.MonkeyPatch):
    before = _active_version(small_env)
    real = R.held_out_nll_of_fit

    def worse(fit, data, test_rows, **kw):
        out = real(fit, data, test_rows, **kw)
        out["nll"] = float(out["nll"]) + 1.0  # a refit that is one nat per response worse than it really is
        return out

    monkeypatch.setattr(R, "held_out_nll_of_fit", worse)
    res = _run(small_env, seed=2)
    assert res["promoted"] is False and res["status"] == "not promoted" and "worse" in res["reason"]
    assert res["held_out"]["nll_new"] > res["held_out"]["nll_reference"]
    assert _active_version(small_env) == before
    with session_scope(small_env.engine) as s:
        row = s.get(ModelRegistry, res["version"])
        assert row is not None and not row.is_active and row.promoted_at is None
        assert s.execute(select(func.count()).select_from(ModelRegistry).where(ModelRegistry.is_active == True)).scalar_one() == 1  # noqa: E712


def test_retrain_promotes_a_not_worse_model_and_the_api_serves_it(small_env: SmallEnv, monkeypatch: pytest.MonkeyPatch):
    before = _active_version(small_env)
    real = R.held_out_nll_of_fit

    def better(fit, data, test_rows, **kw):
        out = real(fit, data, test_rows, **kw)
        out["nll"] = 0.01  # not worse than any reference
        return out

    monkeypatch.setattr(R, "held_out_nll_of_fit", better)
    res = _run(small_env, seed=3)
    assert res["promoted"] is True and res["status"] == "promoted"
    assert _active_version(small_env) == res["version"] != before
    art = P.load_active_artifact(small_env.engine)
    assert art.version == res["version"] and art.metrics["held_out"]["nll"] == 0.01
    with session_scope(small_env.engine) as s:
        rows = {r.version: r for r in s.scalars(select(ModelRegistry)).all()}
        assert rows[res["version"]].is_active and not rows[before].is_active and rows[res["version"]].promoted_at is not None
    # a later run compares against the promoted artifact's recorded held-out NLL as well
    monkeypatch.setattr(R, "held_out_nll_of_fit", real)
    res2 = _run(small_env, seed=3)
    assert res2["held_out"]["recorded_reference_nll"] == 0.01 and res2["held_out"]["reference"]["version"] == res["version"]
    # restore the original model
    from db.demo_seed import register_artifact

    register_artifact(small_env.engine, small_env.artifact, small_env.models_root / small_env.artifact.version, activate=True)


def test_retrain_skips_when_too_few_consented_subjects(tmp_path: Path):
    from db.demo_seed import demo_engine, seed_demo

    db = tmp_path / "data" / "synthetic" / "tiny.db"
    engine = demo_engine(db)
    seed_demo(engine, n_generated=0, models_root=tmp_path / "models", fit_steps=20, fit_restarts=1, n_draws=0, write_files=False)  # 5 archetypes
    with session_scope(engine) as s:
        keep = s.scalars(select(Response.subject_id).distinct()).all()[:2]
        s.execute(update(Response).where(Response.subject_id.not_in(keep)).values(consent_training=False))
    res = R.retrain(engine, steps=5, restarts=1, n_samples=0, models_root=tmp_path / "models", runs_root=tmp_path / "runs")
    assert res["status"].startswith("skipped") and res["promoted"] is False and "version" not in res
    with session_scope(engine) as s:
        assert s.execute(select(func.count()).select_from(AuditLog).where(AuditLog.action == "retrain")).scalar_one() == 1


def test_prior_strength_maps_to_the_context_prior_scale():
    from bre.models.quantum.q2_context_unitary import THETA_PRIOR_SCALE

    assert R._model_kwargs("Q4", 1.0) == ({}, None)
    kw, note = R._model_kwargs("Q4", 2.0)
    assert kw == {"theta_prior_scale": THETA_PRIOR_SCALE / 2.0} and note is None
    kw, note = R._model_kwargs("B2", 2.0)
    assert kw == {} and "Q2/Q4 only" in note
    with pytest.raises(ValueError):
        R._model_kwargs("Q4", 0.0)


def test_post_retrain_endpoint(small_env: SmallEnv):
    from fastapi.testclient import TestClient

    env = {"BRE_DB_URL": f"sqlite:///{small_env.db_path}", "BRE_MODELS_ROOT": str(small_env.models_root), "BRE_API_WARMUP": "0", "BRE_API_AUTOSEED": "0", "BRE_MARKET_NETWORK": "0"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    from api.main import app

    try:
        with TestClient(app) as c:
            before = c.get("/model").json()["version"]
            r = c.post("/retrain", json={"steps": RETRAIN_STEPS, "restarts": 1, "n_samples": 2, "n_boot": 5})
            assert r.status_code == 200, r.text
            b = r.json()
            assert b["status"] in ("promoted", "not promoted") and b["model_type"] == small_env.artifact.model_name and b["run_id"] == b["result"]["run_id"]
            assert np.isfinite(b["nll_new"]) and np.isfinite(b["nll_reference"]) and b["promoted"] == (b["nll_new"] <= b["nll_reference"] + R.PROMOTION_TOLERANCE)
            assert b["active_version"] == (b["version"] if b["promoted"] else before) and c.get("/model").json()["version"] == b["active_version"]
            assert b["demo_mode"] is True
            versions = {x["version"]: x for x in c.get("/model/registry").json()["rows"]}
            assert b["version"] in versions and versions[b["version"]]["held_out_nll"] == b["nll_new"]
            assert "model_type" in {str(x) for e in c.post("/retrain", json={"model_type": "B5"}).json()["detail"] for x in e["loc"]}
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        from db.demo_seed import register_artifact

        register_artifact(small_env.engine, small_env.artifact, small_env.models_root / small_env.artifact.version, activate=True)
