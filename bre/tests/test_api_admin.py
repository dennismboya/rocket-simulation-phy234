"""Contract tests of the Phase 6 endpoints (``api/phase6.py``): settings, contexts, scenarios,
interventions, intake (in-session rows and uploads), export / delete (cascade to every table),
registry / activation (admin flag), transparency (the exact "not yet run" sentence, no Phase 4
numbers before the verdict exists) and the response-history route.

Runs against a **copy** of the short-fit demo database of ``tests/test_predict.py`` so nothing
here changes what ``tests/test_api.py`` sees.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from db.models import AuditLog, InterventionLog, PredictionLog, Response
from db.session import get_engine, session_scope
from test_predict import COV, MARKET, DemoEnv, demo_env  # noqa: F401 - session fixture reused

APP_DIR = Path(__file__).resolve().parents[1] / "app"
API_DIR = Path(__file__).resolve().parents[1] / "api"
NEW_PATHS = {
    "/settings", "/contexts", "/scenarios", "/interventions/versions", "/clients/{client_id}/responses", "/intake/upload", "/intake/config",
    "/clients/{client_id}/export", "/clients/{client_id}", "/model/registry", "/model/activate", "/transparency", "/retrain",
}
NOT_RUN = "Decision rule not yet run: no real-data numbers are shown"


class AdminEnv:
    def __init__(self, root: Path, demo: DemoEnv) -> None:
        self.root = root
        self.db_path = root / "data" / "synthetic" / "demo.db"
        self.db_path.parent.mkdir(parents=True)
        shutil.copy(demo.db_path, self.db_path)
        self.models_root = demo.models_root
        self.engine = get_engine(f"sqlite:///{self.db_path}")
        self.artifact = demo.artifact


@pytest.fixture(scope="module")
def admin_env(tmp_path_factory: pytest.TempPathFactory, demo_env: DemoEnv) -> AdminEnv:
    return AdminEnv(tmp_path_factory.mktemp("bre_admin"), demo_env)


@pytest.fixture(scope="module")
def client(admin_env: AdminEnv):
    env = {"BRE_DB_URL": f"sqlite:///{admin_env.db_path}", "BRE_MODELS_ROOT": str(admin_env.models_root), "BRE_API_WARMUP": "0", "BRE_API_AUTOSEED": "0",
           "BRE_MARKET_NETWORK": "0", "BRE_REPORTS_DIR": str(admin_env.root / "reports_absent")}
    old = {k: os.environ.get(k) for k in list(env) + ["BRE_ADMIN"]}
    os.environ.pop("BRE_ADMIN", None)
    os.environ.update(env)
    from api.main import app

    try:
        with TestClient(app) as c:
            yield c
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _locs(resp) -> set[str]:
    assert resp.status_code == 422, resp.text
    return {str(x) for e in resp.json()["detail"] for x in e["loc"]}


def _table_counts(db_path: Path, client_id: str) -> dict[str, int]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = {}
        for t in ("responses", "predictions_log", "intervention_log"):
            out[t] = con.execute(f"SELECT COUNT(*) FROM {t} WHERE client_id = ?", (client_id,)).fetchone()[0]
        out["clients"] = con.execute("SELECT COUNT(*) FROM clients WHERE client_id = ?", (client_id,)).fetchone()[0]
        out["bre_documents"] = sum(1 for (p,) in con.execute("SELECT payload FROM bre_documents WHERE kind = 'intake_session'") if json.loads(p).get("client_id") == client_id)
        return out
    finally:
        con.close()


def _intake_rows(client_id: str, *, form: str = "short", is_synthetic: bool = True, consent: bool = True, seed: str | None = None, subject_id: str | None = None):
    import sys

    sys.path.insert(0, str(APP_DIR))
    from components import intake as I

    seed = seed or I.new_seed()
    pres, assignment = I.build_presentations(seed, form)
    rows, pos = [], 0
    cov = I.covariates_json(COV)
    for k, p in enumerate(pres):
        r = I.rows_for_presentation(p, {"tolerance": k % 2, "sell_hold": 1 - k % 2, "allocation_share": 40}, form=form, subject_id=subject_id or client_id, session_id=seed,
                                    covariates_json=cov, consent_training=consent, is_synthetic=is_synthetic, position_start=pos, timestamp=I.now_iso(), response_time_ms=1500)
        rows.extend(r)
        pos += len(r)
    return rows, assignment, seed


# ---------------------------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------------------------


def test_openapi_has_the_new_routes_and_committed_file_is_current(client: TestClient):
    from api.main import app

    live = app.openapi()
    assert NEW_PATHS <= set(live["paths"])
    committed = json.loads((API_DIR / "openapi.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(live, sort_keys=True)), "api/openapi.json is stale: run `python api/export_openapi.py`"
    # the Phase 5 routes and their response fields are untouched
    for path in ("/health", "/model", "/clients", "/interventions", "/market", "/predict", "/profile", "/score_book", "/score_book/csv", "/interventions/rank"):
        assert path in live["paths"]
    assert {"p", "ci80", "ci95", "interference", "top_driver", "n_calibration", "scenario", "client_id", "prediction_log_id"} <= set(live["components"]["schemas"]["PredictResponse"]["properties"])


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------


def test_settings_defaults_validation_and_demo_mode(client: TestClient):
    r = client.get("/settings")
    assert r.status_code == 200, r.text
    body = r.json()
    s = body["settings"]
    assert s["alert_threshold"] == 0.5 and s["status_thresholds"] == {"elevated": 0.25, "high": 0.5} and s["capacity_target"] == 0.25
    assert s["prior_strength"] == 1.0 and s["typical_crisis_contexts"] == ["news:recession", "social:friend_sells"] and s["demo_mode"] is False
    assert set(body["defaults"]) == set(s) and set(body["rules"]) == set(s) and body["version"] == 0
    assert body["demo_mode"] is True and body["demo_mode_forced_by_data"] is True  # synthetic model + database under data/synthetic
    r = client.put("/settings", json={"alert_threshold": 0.4, "status_thresholds": {"elevated": 0.2, "high": 0.6}, "capacity_target": 0.3, "typical_crisis_contexts": ["social:friend_sells"], "retrain": {"steps": 20, "restarts": 1}})
    assert r.status_code == 200, r.text
    s = r.json()["settings"]
    assert s["alert_threshold"] == 0.4 and s["status_thresholds"] == {"elevated": 0.2, "high": 0.6} and s["capacity_target"] == 0.3 and s["retrain"]["steps"] == 20 and s["retrain"]["n_samples"] == 50
    assert r.json()["version"] == 1
    # the scoring uses the settings: status thresholds and the crisis contexts of the capacity
    r = client.post("/score_book", json={"clients": [{"client_id": "DEMO-C01", "covariates": COV}], "market_state": MARKET})
    assert r.status_code == 200 and r.json()["meta"]["status_thresholds"] == {"elevated": 0.2, "high": 0.6} and r.json()["meta"]["crisis_contexts"] == ["social:friend_sells"]
    assert "['social:friend_sells']" in r.json()["meta"]["capacity_definition"]
    # validation names the field
    assert "alert_threshold" in _locs(client.put("/settings", json={"alert_threshold": 1.5}))
    assert "body" in _locs(client.put("/settings", json={"status_thresholds": {"elevated": 0.9, "high": 0.1}}))
    assert "typical_crisis_contexts" in _locs(client.put("/settings", json={"typical_crisis_contexts": ["news:unknown_tag"]}))
    assert "extra" in _locs(client.put("/settings", json={"extra": 1}))
    # demo mode toggle: forced on stays on; GET /model reflects it
    r = client.put("/settings", json={"demo_mode": True})
    assert r.status_code == 200 and r.json()["settings"]["demo_mode"] is True and r.json()["demo_mode"] is True
    assert client.get("/model").json()["demo_mode"] is True and client.get("/health").json()["demo_mode"] is True
    r = client.put("/settings", json={"demo_mode": False})
    assert r.json()["settings"]["demo_mode"] is False and r.json()["demo_mode"] is True  # synthetic data keeps demo mode on
    client.put("/settings", json={"alert_threshold": 0.5, "status_thresholds": {"elevated": 0.25, "high": 0.5}, "capacity_target": 0.25, "typical_crisis_contexts": ["news:recession", "social:friend_sells"]})
    with session_scope(get_engine(f"sqlite:///{os.environ['BRE_DB_URL'].removeprefix('sqlite:///')}")) as s:
        assert s.execute(select(func.count()).select_from(AuditLog).where(AuditLog.action == "settings")).scalar_one() >= 3


# ---------------------------------------------------------------------------------------------
# Contexts, scenarios, interventions
# ---------------------------------------------------------------------------------------------


def test_contexts_uncalibrated_flag_and_versions(client: TestClient, admin_env: AdminEnv):
    r = client.get("/contexts")
    assert r.status_code == 200, r.text
    rows = r.json()
    assert [c["tag"] for c in rows][:4] == ["news:recession", "news:technical", "social:friend_sells", "market:recovered_5pct"]
    assert all(c["in_design"] and c["in_served_vocabulary"] and c["version"] == 1 and c["display"] for c in rows[:4])
    assert all(c["calibration"]["status"] == admin_env.artifact.calibrated_contexts[c["tag"]]["status"] for c in rows[:4])
    assert all(c["calibration"]["theta"] is not None and len(c["calibration"]["theta"]) == 3 for c in rows[:4])
    r = client.post("/contexts", json={"tag": "news:tariff_shock", "kind": "news", "display": "The news reports a tariff shock."})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["calibration"]["status"] == "uncalibrated: prior only" and c["calibration"]["n_responses"] == 0 and c["calibration"]["theta"] is None
    assert c["in_design"] is False and c["in_served_vocabulary"] is False and c["version"] == 1 and c["triggers_delay"] is True
    # retire = a new version; the old one stays in the history
    r = client.post("/contexts", json={"tag": "news:tariff_shock", "kind": "news", "display": "The news reports a tariff shock.", "active": False})
    assert r.status_code == 200 and r.json()["version"] == 2 and r.json()["active"] is False
    tags = {c["tag"]: c for c in client.get("/contexts").json()}
    assert tags["news:tariff_shock"]["version"] == 2 and tags["news:tariff_shock"]["active"] is False
    assert "tag" in _locs(client.post("/contexts", json={"tag": "no namespace", "display": "x"}))


def test_scenarios_and_interventions_are_versioned(client: TestClient):
    r = client.get("/scenarios")
    assert r.status_code == 200
    texts = r.json()["texts"]
    assert set(r.json()["keys"]) == set(texts) and texts["intro"]["version"] == 1 and "invested" in texts["intro"]["text"]
    assert "avoids investment losses" in texts["tolerance_question"]["text"]
    before = texts["intro"]["text"]
    r = client.post("/scenarios", json={"key": "intro", "text": "Edited intro."})
    assert r.status_code == 200 and r.json()["version"] == 2
    hist = client.get("/scenarios", params={"history": "true"}).json()["history"]
    intro_versions = [h for h in hist if h["key"] == "intro"]
    assert [h["version"] for h in intro_versions] == [1, 2] and intro_versions[0]["payload"]["text"] == before
    assert "key" in _locs(client.post("/scenarios", json={"key": "nope", "text": "x"}))
    assert "text" in _locs(client.post("/scenarios", json={"key": "intro", "text": "   "}))
    # interventions: the live row changes and a version is appended; bad transforms are refused
    r = client.put("/interventions", json={"name": "cash_sleeve_check", "script": "New script.", "mapped_context_transform": {"remove": ["news:technical"]}})
    assert r.status_code == 200, r.text
    assert r.json()["version"] == 1 and r.json()["script"] == "New script." and r.json()["mapped_context_transform"] == {"remove": ["news:technical"]}
    live = {i["name"]: i for i in client.get("/interventions").json()}
    assert live["cash_sleeve_check"]["script"] == "New script." and live["cash_sleeve_check"]["label"] == "predicted effect, not causally validated"
    assert "mapped_context_transform" in _locs(client.put("/interventions", json={"name": "cash_sleeve_check", "mapped_context_transform": {"add": ["bogus:tag"]}}))
    assert client.put("/interventions", json={"name": "does_not_exist", "script": "x"}).status_code == 404
    r = client.put("/interventions", json={"name": "cash_sleeve_check", "active": False})
    assert r.status_code == 200 and r.json()["version"] == 2 and r.json()["active"] is False
    versions = {v["name"]: v for v in client.get("/interventions/versions", params={"history": "true"}).json()}
    assert len(versions["cash_sleeve_check"]["history"]) == 2 and versions["reframe_time_horizon"]["version"] == 0
    r = client.post("/interventions/rank", json={"covariates": COV, "market_state": MARKET})
    assert "cash_sleeve_check" in r.json()["inactive"]
    client.put("/interventions", json={"name": "cash_sleeve_check", "active": True})


# ---------------------------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------------------------


def test_intake_config(client: TestClient):
    r = client.get("/intake/config")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["battery_version"] == "1.0.0" and b["forms"]["full"]["n_presentations"] == 14 and b["forms"]["short"]["n_presentations"] == 5
    assert b["instrument_url"].startswith("file://") and b["instrument_url"].endswith("index.html") and b["demo_mode"] is True
    assert b["intake"]["consent"]["checkboxes"][1]["maps_to"] == "consent_training" and len(b["intake"]["financial_literacy_quiz"]["items"]) == 5


def test_post_responses_stores_rows_records_assignment_and_refuses_real_rows(client: TestClient, admin_env: AdminEnv):
    rows, assignment, seed = _intake_rows("INTAKE-A")
    r = client.post("/clients/INTAKE-A/responses", json={"rows": rows, "assignment": assignment, "session": {"form": "short", "seed": seed}, "display_label": "Intake A"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["n_rows"] == 15 and b["is_synthetic"] is True and b["consent_training"] is True and b["dataset"] == "intake_battery" and b["document_version"] == 1 and "SYNTHETIC" in b["note"]
    clients = {c["client_id"]: c for c in client.get("/clients").json()}
    assert clients["INTAKE-A"]["n_responses"] == 15 and clients["INTAKE-A"]["display_label"] == "Intake A" and clients["INTAKE-A"]["covariates"]["age_band"] == COV["age_band"]
    # the response history route serves them (tolerance rows keyed by the design's tolerance id)
    hist = client.get("/clients/INTAKE-A/responses").json()
    assert len(hist) == 15 and {h["elicitation_type"] for h in hist} == {"binary_yes_no", "binary_sell", "allocation_pct"}
    assert all(h["scenario_id"] == "tolerance" for h in hist if h["elicitation_type"] == "binary_yes_no") and all(h["is_synthetic"] for h in hist)
    # the assignment and session are recorded in a document and an audit row
    exp = client.get("/clients/INTAKE-A/export").json()
    assert len(exp["intake_sessions"]) == 1
    doc = exp["intake_sessions"][0]["payload"]
    assert doc["assignment"]["form"] == "short" and len(doc["assignment"]["item_ids"]) == 5 and doc["assignment"]["seed"] == seed and doc["session"]["seed"] == seed
    with session_scope(admin_env.engine) as s:
        assert s.execute(select(func.count()).select_from(AuditLog).where(AuditLog.action == "intake_session", AuditLog.entity_id == "INTAKE-A")).scalar_one() == 1
    # duplicate session -> 409; real rows into the synthetic database -> 409; bad rows -> 422 naming rows
    assert client.post("/clients/INTAKE-A/responses", json={"rows": rows}).status_code == 409
    real_rows, _, _ = _intake_rows("INTAKE-A", is_synthetic=False)
    r = client.post("/clients/INTAKE-A/responses", json={"rows": real_rows})
    assert r.status_code == 409 and "data/synthetic" in r.json()["detail"][0]["msg"]
    bad = [dict(x) for x in _intake_rows("INTAKE-B")[0]]
    bad[0]["response"] = 7
    assert "rows" in _locs(client.post("/clients/INTAKE-B/responses", json={"rows": bad}))
    other, _, _ = _intake_rows("INTAKE-B", subject_id="someone-else")
    assert "rows" in _locs(client.post("/clients/INTAKE-B/responses", json={"rows": other}))
    assert "client_id" in _locs(client.post("/clients/api:batch/responses", json={"rows": rows}))


def test_intake_upload_uses_the_loader_contract(client: TestClient):
    rows, _, seed = _intake_rows("UPLOAD-1", is_synthetic=False, subject_id="11111111-2222-4333-8444-555555555555")
    content = json.dumps(rows).encode()
    fname = f"intake_{seed}.json"
    # real rows into the demo database: refused unless relabelled synthetic (demo mode)
    r = client.post("/intake/upload", files={"file": (fname, io.BytesIO(content), "application/json")}, data={"client_id": "UPLOAD-1"})
    assert r.status_code == 409
    r = client.post("/intake/upload", files={"file": (fname, io.BytesIO(content), "application/json")}, data={"client_id": "UPLOAD-1", "store_as_synthetic": "true", "display_label": "Uploaded"})
    assert r.status_code == 200, r.text
    assert r.json()["n_rows"] == 15 and r.json()["is_synthetic"] is True and r.json()["session_id"] == seed
    hist = client.get("/clients/UPLOAD-1/responses").json()
    assert len(hist) == 15 and all(h["source_row_ref"].startswith(fname + ":short:i") for h in hist)
    exp = client.get("/clients/UPLOAD-1/export").json()
    assert exp["intake_sessions"][0]["payload"]["session"]["original_subject_id"] == "11111111-2222-4333-8444-555555555555"
    # contract violations
    r = client.post("/intake/upload", files={"file": ("x.json", io.BytesIO(json.dumps(rows[:6]).encode()), "application/json")}, data={"client_id": "UPLOAD-2", "store_as_synthetic": "true"})
    assert r.status_code == 422 and "incomplete session" in r.json()["detail"][0]["msg"]
    broken = [dict(x) for x in rows]
    broken[0]["dataset"] = "other"
    r = client.post("/intake/upload", files={"file": ("x.json", io.BytesIO(json.dumps(broken).encode()), "application/json")}, data={"client_id": "UPLOAD-2", "store_as_synthetic": "true"})
    assert r.status_code == 422 and "contract" in r.json()["detail"][0]["msg"]
    r = client.post("/intake/upload", files={"file": ("x.json", io.BytesIO(b"{not json"), "application/json")}, data={"client_id": "UPLOAD-2"})
    assert r.status_code == 422 and "file" in _locs(r)


# ---------------------------------------------------------------------------------------------
# Export and delete (cascade to every table)
# ---------------------------------------------------------------------------------------------


def test_delete_client_removes_every_row_in_every_table(client: TestClient, admin_env: AdminEnv):
    rows, assignment, seed = _intake_rows("DEL-1")
    assert client.post("/clients/DEL-1/responses", json={"rows": rows, "assignment": assignment, "session": {"form": "short", "seed": seed}}).status_code == 200
    assert client.post("/predict", json={"client_id": "DEL-1", "covariates": COV, "loss_pct": -0.2, "context_tags": ["news:recession"]}).status_code == 200
    with session_scope(admin_env.engine) as s:
        iv_id = s.execute(select(__import__("db.models", fromlist=["Intervention"]).Intervention.id)).scalars().first()
        s.add(InterventionLog(client_id="DEL-1", intervention_id=iv_id, advisor_note="test"))
    before = _table_counts(admin_env.db_path, "DEL-1")
    assert before == {"responses": 15, "predictions_log": 1, "intervention_log": 1, "clients": 1, "bre_documents": 1}
    exp = client.get("/clients/DEL-1/export")
    assert exp.status_code == 200 and len(exp.json()["responses"]) == 15 and len(exp.json()["predictions_log"]) == 1 and len(exp.json()["intervention_log"]) == 1 and len(exp.json()["intake_sessions"]) == 1
    assert exp.json()["synthetic"] is True and exp.json()["demo_mode"] is True
    r = client.delete("/clients/DEL-1")
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == {"responses": 15, "predictions_log": 1, "intervention_log": 1, "bre_documents": 1}
    after = _table_counts(admin_env.db_path, "DEL-1")
    assert after == {"responses": 0, "predictions_log": 0, "intervention_log": 0, "clients": 0, "bre_documents": 0}
    with session_scope(admin_env.engine) as s:
        row = s.get(AuditLog, r.json()["audit_log_id"])
        assert row is not None and row.action == "delete_client" and row.entity_id == "DEL-1"
        assert s.execute(select(func.count()).select_from(Response).where(Response.client_id == "DEL-1")).scalar_one() == 0
        assert s.execute(select(func.count()).select_from(PredictionLog).where(PredictionLog.client_id == "DEL-1")).scalar_one() == 0
    assert client.get("/clients/DEL-1/export").status_code == 404 and client.delete("/clients/DEL-1").status_code == 404
    assert client.delete("/clients/api:anonymous").status_code == 409
    assert client.get("/clients/nobody/responses").status_code == 404


# ---------------------------------------------------------------------------------------------
# Registry, activation, transparency
# ---------------------------------------------------------------------------------------------


def test_registry_and_activation_need_the_admin_flag(client: TestClient, admin_env: AdminEnv):
    r = client.get("/model/registry")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["active_version"] == admin_env.artifact.version and b["admin"] is False and b["selectable_model_types"] == ["Q2", "Q4", "B2"]
    active = [x for x in b["rows"] if x["is_active"]]
    assert len(active) == 1 and active[0]["version"] == admin_env.artifact.version and active[0]["artifact_exists"] and active[0]["is_synthetic_training"] is True and active[0]["train_nll"] > 0
    assert client.post("/model/activate", json={"version": admin_env.artifact.version}).status_code == 403
    assert client.get("/model/registry", headers={"X-BRE-Admin": "1"}).json()["admin"] is True
    # a second registered version (a copy of the artifact under another version) can be activated and switched back
    from bre.artifact import ModelArtifact
    from db.demo_seed import register_artifact

    copy = ModelArtifact.load(next(p for p in admin_env.models_root.iterdir() if (p / "artifact.json").exists()))
    copy.version = "test-copy-v1"
    copy_dir = admin_env.root / "runs" / "models" / "test-copy-v1"
    copy.save(copy_dir)
    register_artifact(admin_env.engine, copy, copy_dir, activate=False)
    r = client.post("/model/activate", json={"version": "test-copy-v1"}, headers={"X-BRE-Admin": "1"})
    assert r.status_code == 200, r.text
    assert r.json()["activated"] == "test-copy-v1" and r.json()["previous"] == admin_env.artifact.version and r.json()["synthetic"] is True
    assert client.get("/model").json()["version"] == "test-copy-v1" and client.get("/health").json()["model_version"] == "test-copy-v1"
    rows = {x["version"]: x for x in client.get("/model/registry").json()["rows"]}
    assert rows["test-copy-v1"]["is_active"] and not rows[admin_env.artifact.version]["is_active"]
    assert client.post("/model/activate", json={"version": "missing-version"}, headers={"X-BRE-Admin": "1"}).status_code == 404
    r = client.post("/model/activate", json={"version": admin_env.artifact.version}, headers={"X-BRE-Admin": "1"})
    assert r.status_code == 200 and client.get("/model").json()["version"] == admin_env.artifact.version
    with session_scope(admin_env.engine) as s:
        assert s.execute(select(func.count()).select_from(AuditLog).where(AuditLog.action == "activate_model")).scalar_one() == 2


def test_transparency_withholds_real_numbers_until_the_verdict_exists(client: TestClient, admin_env: AdminEnv, monkeypatch: pytest.MonkeyPatch):
    r = client.get("/transparency")
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["verdict"] is None and t["verdict_status"] == NOT_RUN and t["phase4_metrics"] is None and t["real_data_numbers_shown"] is False
    assert "Q-model supported" in t["decision_rule"] and t["model_version"] == admin_env.artifact.version and t["n_params"] == admin_env.artifact.n_params
    assert t["is_synthetic_training"] is True and t["demo_mode"] is True
    # synthetic numbers are returned (shown under the banner): training metrics and the in-sample reliability table
    assert t["training_metrics"]["is_synthetic"] is True and t["training_metrics"]["train_nll_per_response"] > 0
    cal = t["calibration_plot"]
    assert cal["is_synthetic"] is True and cal["n_rows"] > 0 and len(cal["reliability"]) == 10 and 0 <= cal["ece"] <= 1 and "in-sample" in cal["scope"]
    assert t["n_target"] is None or t["n_target"]["items_per_subject"] == 170
    prov = t["provenance"]
    assert any(p["dataset"] == "choices13k" and p["real_or_synthetic"] == "real" and p["catalog_entry"] == "A" and p["n_rows"] == 12225 for p in prov)
    assert any(p["dataset"] == "synthetic_demo_book" and p["real_or_synthetic"] == "synthetic" and p["used_by_served_model"] for p in prov)
    assert all({"dataset", "n_rows", "n_subjects", "real_or_synthetic", "license"} <= set(p) for p in prov)
    assert t["model_card"] is None and "promote only if" in t["retrain_rule"]
    # with a verdict file the sentence goes away and the verdict is returned verbatim
    rdir = admin_env.root / "reports_with_verdict"
    (rdir / "phase4").mkdir(parents=True)
    verdict = {"verdict": "no evidence of a quantum-probability advantage in the available data", "details": {"split": "b"}, "metrics": {"B1": {"nll": 0.61}, "Q4": {"nll": 0.62}}}
    (rdir / "phase4" / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    (rdir / "MODEL_CARD.md").write_text("# Model card\n\ntest card\n", encoding="utf-8")
    monkeypatch.setenv("BRE_REPORTS_DIR", str(rdir))
    t = client.get("/transparency").json()
    assert t["verdict"] == verdict and t["verdict_status"] == verdict["verdict"] and t["phase4_metrics"] == verdict["metrics"] and t["real_data_numbers_shown"] is True
    assert t["model_card"].startswith("# Model card") and t["n_target"] is None
