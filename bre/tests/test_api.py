"""Contract tests of the FastAPI service (``api/main.py``) with ``httpx``'s TestClient.

The service runs against the short-fit demo database of ``tests/test_predict.py`` (session
fixture ``demo_env``), reached through ``BRE_DB_URL`` / ``BRE_MODELS_ROOT``; the JIT warm-up is
off (``BRE_API_WARMUP=0``). Covers every endpoint, the 422 cases with field names, the
``predictions_log`` / ``audit_log`` increments, the ``GET /model`` fields and the committed
``api/openapi.json``.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from db.models import AuditLog, Client, PredictionLog
from db.session import session_scope
from test_predict import COV, MARKET, DemoEnv, demo_env  # noqa: F401 - session fixture reused

API_DIR = Path(__file__).resolve().parents[1] / "api"


@pytest.fixture(scope="session")
def client(demo_env: DemoEnv):
    env = {"BRE_DB_URL": f"sqlite:///{demo_env.db_path}", "BRE_MODELS_ROOT": str(demo_env.models_root), "BRE_API_WARMUP": "0", "BRE_API_AUTOSEED": "0", "BRE_MARKET_NETWORK": "0"}
    old = {k: os.environ.get(k) for k in env}
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


def _counts(demo_env: DemoEnv) -> tuple[int, int]:
    with session_scope(demo_env.engine) as s:
        return int(s.execute(select(func.count()).select_from(PredictionLog)).scalar_one()), int(s.execute(select(func.count()).select_from(AuditLog)).scalar_one())


def _locs(resp) -> set[str]:
    assert resp.status_code == 422, resp.text
    return {str(x) for e in resp.json()["detail"] for x in e["loc"]}


# ---------------------------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------------------------


def test_health(client: TestClient, demo_env: DemoEnv):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_version"] == demo_env.artifact.version
    assert body["demo_mode"] is True and body["synthetic"] is True and "demo.db" in body["db"]


def test_model_info_fields(client: TestClient, demo_env: DemoEnv):
    r = client.get("/model")
    assert r.status_code == 200
    m = r.json()
    art = demo_env.artifact
    assert m["version"] == art.version and m["model_type"] == art.model_name and m["family"] == "quantum"
    assert m["is_synthetic_training"] is True and m["demo_mode"] is True and m["synthetic"] is True
    assert m["metrics"]["train_nll_per_response"] > 0 and m["metrics"]["steps"] == art.metrics["steps"]
    assert set(m["calibrated_contexts"]) == set(art.ctx_vocab)
    assert all(v["status"] in ("calibrated", "uncalibrated: prior only") and "n_responses" in v for v in m["calibrated_contexts"].values())
    assert m["n_params"] == art.n_params and m["n_draws"] == art.n_samples
    prov = m["provenance"]
    assert prov["training_data_refs"] == list(art.training_data_refs)
    names = [e["name"] for e in prov["catalog_entries"]]
    assert any("choices13k" in n for n in names) and len(names) >= 5
    assert prov["catalog_entries_used_in_training"] == [] and "synthetic demo book" in prov["note"]
    assert "Q-model supported" in m["decision_rule"] and m["evaluation_status"].startswith("not evaluated")
    assert m["wording"] == "predicted probability of selling" and "draws" in m["interval_source"]
    assert m["promoted_at"] is not None and m["design_version"] == "1.0.0"


def test_clients_interventions_market(client: TestClient, demo_env: DemoEnv):
    r = client.get("/clients")
    assert r.status_code == 200
    rows = r.json()
    ids = {c["client_id"] for c in rows}
    assert ids == set(demo_env.clients["client_id"]) and not any(i.startswith("api:") for i in ids)
    first = next(c for c in rows if c["client_id"] == "DEMO-C01")
    assert first["display_label"] == "Demo client 01" and first["known_to_model"] and first["synthetic"] and first["n_responses"] == 32
    assert set(first["covariates"]) <= {"age_band", "wealth_band", "invest_experience_yrs", "self_reported_risk_tolerance", "financial_literacy_score", "education"}
    r = client.get("/interventions")
    assert r.status_code == 200 and len(r.json()) == 5
    assert all(i["label"] == "predicted effect, not causally validated" and isinstance(i["mapped_context_transform"], dict) for i in r.json())
    r = client.get("/market")
    assert r.status_code == 200
    body = r.json()
    assert body["market_state"]["source"] == "cached-file" and body["demo_mode"] is True
    assert body["scenario"]["loss_pct"] <= -0.05 and "context_tags" in body["scenario"]


# ---------------------------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------------------------


def test_predict_logs_and_fields(client: TestClient, demo_env: DemoEnv):
    before = _counts(demo_env)
    payload = {"client_id": "api-test-1", "display_label": "API test", "covariates": COV, "loss_pct": -0.2, "context_tags": ["news:recession", "social:friend_sells"]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 < body["p"] < 1 and body["ci80"][0] <= body["ci80"][1] and body["ci95"][0] <= body["ci80"][0]
    assert body["synthetic"] is True and body["demo_mode"] is True and body["model_version"] == demo_env.artifact.version
    assert body["top_driver"]["driver"] in ("news:recession", "social:friend_sells", "loss")
    assert set(body["n_calibration"]) == {"news:recession", "social:friend_sells"}
    assert body["interference"]["order_effect"] is not None and body["client_id"] == "api-test-1"
    after = _counts(demo_env)
    assert after[0] == before[0] + 1 and after[1] >= before[1] + 1
    with session_scope(demo_env.engine) as s:
        c = s.get(Client, "api-test-1")
        assert c is not None and c.display_label == "API test" and json.loads(c.covariates) == COV
        row = s.get(PredictionLog, body["prediction_log_id"])
        assert row.client_id == "api-test-1" and json.loads(row.outputs)["kind"] == "predict"
        assert s.execute(select(func.count()).select_from(AuditLog).where(AuditLog.action == "predict:predict")).scalar_one() >= 1
    # anonymous request: logged under the reserved client
    before = _counts(demo_env)
    r = client.post("/predict", json={"covariates": {}, "loss_pct": -0.1, "context_tags": ["none"]})
    assert r.status_code == 200 and r.json()["client_id"] == "api:anonymous"
    assert _counts(demo_env)[0] == before[0] + 1
    # tolerance-first with an answer and a mixture
    r = client.post("/predict", json={"covariates": COV, "loss_pct": -0.1, "context_tags": ["news:recession"], "question_order_id": "tolerance-first", "tol_answer": 1, "mix_weights": {"news:recession": 0.5, "news:technical": 0.5}})
    assert r.status_code == 200 and r.json()["interference"]["mix"] is not None
    # prior responses refine the subject first
    prior = [{"loss_pct": -0.1, "context_tags": ["news:recession"], "response": 1}, {"loss_pct": -0.2, "context_tags": [], "question_order_id": "tolerance-first", "response": 1, "tol_answer": 0}]
    r = client.post("/predict", json={"client_id": "api-test-2", "covariates": COV, "loss_pct": -0.2, "context_tags": [], "prior_responses": prior})
    assert r.status_code == 200 and r.json()["refined"]["n_responses"] == 2


@pytest.mark.parametrize(
    "patch,field",
    [
        ({"loss_pct": 0.2}, "loss_pct"),
        ({"loss_pct": -1.5}, "loss_pct"),
        ({"covariates": {**COV, "age_band": "99+"}}, "age_band"),
        ({"covariates": {**COV, "self_reported_risk_tolerance": 9}}, "self_reported_risk_tolerance"),
        ({"context_tags": ["news:recession", "news:technical", "social:friend_sells"]}, "context_tags"),
        ({"context_tags": ["not:a_context"]}, "context_tags"),
        ({"question_order_id": "sideways"}, "question_order_id"),
        ({"tol_answer": 2}, "tol_answer"),
        ({"mix_weights": {"news:recession": -1.0}}, "mix_weights"),
        ({"client_id": "bad id with spaces"}, "client_id"),
        ({"extra": 1}, "extra"),
    ],
)
def test_predict_422_names_the_field(client: TestClient, patch, field):
    payload = {"covariates": COV, "loss_pct": -0.2, "context_tags": ["news:recession"], **patch}
    assert field in _locs(client.post("/predict", json=payload))


def test_predict_missing_loss_is_422(client: TestClient):
    assert "loss_pct" in _locs(client.post("/predict", json={"covariates": COV}))


# ---------------------------------------------------------------------------------------------
# /profile
# ---------------------------------------------------------------------------------------------


def test_profile(client: TestClient, demo_env: DemoEnv):
    before = _counts(demo_env)
    r = client.post("/profile", json={"client_id": "api-profile", "covariates": COV})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["sensitivities"]) == set(demo_env.artifact.ctx_vocab)
    assert 0 <= body["baseline"]["p_sell_base"] <= 1 and body["consistency"]["gamma"] > 0
    assert body["risk_score"]["r2_vs_angle"] is not None and 0 <= body["risk_score"]["r2_vs_angle"] <= 1
    assert body["synthetic"] is True and body["refined"] is None
    assert _counts(demo_env)[0] == before[0] + 1
    prior = [{"loss_pct": -0.1, "context_tags": ["news:recession"], "response": 1}]
    r = client.post("/profile", json={"covariates": COV, "prior_responses": prior, "include_book_r2": False})
    assert r.status_code == 200 and r.json()["refined"]["n_responses"] == 1 and r.json()["risk_score"]["r2_vs_angle"] is None
    assert "response" in _locs(client.post("/profile", json={"covariates": COV, "prior_responses": [{"loss_pct": -0.1, "response": 2}]}))


# ---------------------------------------------------------------------------------------------
# /score_book
# ---------------------------------------------------------------------------------------------


def _clients_payload(demo_env: DemoEnv, n: int = 4) -> list[dict]:
    out = []
    for rec in demo_env.clients.head(n).to_dict(orient="records"):
        out.append({"client_id": rec["client_id"], "display_label": rec["display_label"], "covariates": json.loads(rec["covariates"])})
    return out


def test_score_book_json(client: TestClient, demo_env: DemoEnv):
    clients = _clients_payload(demo_env, 4) + [{"client_id": "new-upload-1", "covariates": COV}]
    before = _counts(demo_env)
    r = client.post("/score_book", json={"clients": clients, "market_state": MARKET, "target": 0.25})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["rows"]) == 5 and body["synthetic"] is True and body["demo_mode"] is True
    assert body["scenario"]["context_tags"] == ["news:recession", "social:friend_sells"] and body["scenario"]["weak_match"] is False
    assert set(body["n_calibration"]) == {"news:recession", "social:friend_sells"}
    assert body["meta"]["intervention_label"] == "predicted effect, not causally validated" and body["meta"]["n_clients"] == 5
    known = [row for row in body["rows"] if row["client_id"] != "new-upload-1"]
    assert all(row["known_client"] for row in known) and not next(row for row in body["rows"] if row["client_id"] == "new-upload-1")["known_client"]
    for row in body["rows"]:
        assert 0 <= row["p_sell"] <= 1 and row["ci80_low"] <= row["ci80_high"] and row["prediction_log_id"] is not None
        assert row["status"].split(";")[0] in ("stable", "elevated", "high")
    after = _counts(demo_env)
    assert after[0] == before[0] + 5
    with session_scope(demo_env.engine) as s:
        assert s.get(Client, "new-upload-1") is None  # not registered by default
        new_row = s.get(PredictionLog, next(row for row in body["rows"] if row["client_id"] == "new-upload-1")["prediction_log_id"])
        assert new_row.client_id == "api:batch" and json.loads(new_row.outputs)["inputs"]["client_id"] == "new-upload-1"
    r = client.post("/score_book", json={"clients": [{"client_id": "new-upload-2", "display_label": "Uploaded", "covariates": COV}], "market_state": MARKET, "register_unknown": True})
    assert r.status_code == 200
    with session_scope(demo_env.engine) as s:
        assert s.get(Client, "new-upload-2").display_label == "Uploaded"
    # weak match is reported for a nonsense cause frame
    r = client.post("/score_book", json={"clients": clients[:1], "market_state": {"drawdown_pct": -0.1, "cause_frame": "zxqv blorp"}})
    assert r.status_code == 200 and r.json()["scenario"]["weak_match"] is True


@pytest.mark.parametrize(
    "payload,field",
    [
        ({"clients": [], "market_state": MARKET}, "clients"),
        ({"clients": [{"client_id": "a", "covariates": COV}, {"client_id": "a", "covariates": COV}], "market_state": MARKET}, "clients"),
        ({"clients": [{"client_id": "a", "covariates": COV}], "market_state": {"vix_bucket": "wild"}}, "vix_bucket"),
        ({"clients": [{"client_id": "a", "covariates": COV}], "market_state": MARKET, "target": 1.5}, "target"),
        ({"clients": [{"client_id": "a", "covariates": {"education": "phd"}}], "market_state": MARKET}, "education"),
        ({"clients": [{"client_id": "a", "covariates": COV}]}, "market_state"),
        ({"clients": [{"client_id": "a", "covariates": COV}], "market_state": {"social_cue_prevalence": 3}}, "social_cue_prevalence"),
    ],
)
def test_score_book_422(client: TestClient, payload, field):
    assert field in _locs(client.post("/score_book", json=payload))


def test_score_book_csv(client: TestClient, demo_env: DemoEnv):
    csv = "client_id,display_label,age_band,wealth_band,invest_experience_yrs,self_reported_risk_tolerance,financial_literacy_score,education\n"
    csv += "csv-1,CSV one,30-44,50-250k,5,3,2,bachelor\ncsv-2,,60+,>1M,30,5,,graduate\n"
    before = _counts(demo_env)
    r = client.post("/score_book/csv", files={"file": ("book.csv", io.BytesIO(csv.encode()), "text/csv")}, data={"market_state": json.dumps(MARKET), "target": "0.3"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [row["client_id"] for row in body["rows"]] == ["csv-1", "csv-2"] and body["meta"]["target"] == 0.3
    assert body["rows"][1]["display_label"] is None
    assert _counts(demo_env)[0] == before[0] + 2
    bad = "name,age_band\nx,30-44\n"
    r = client.post("/score_book/csv", files={"file": ("book.csv", io.BytesIO(bad.encode()), "text/csv")}, data={"market_state": "{}"})
    assert r.status_code == 422 and "client_id" in _locs(r)
    bad_row = "client_id,age_band\ncsv-3,99+\n"
    r = client.post("/score_book/csv", files={"file": ("book.csv", io.BytesIO(bad_row.encode()), "text/csv")}, data={"market_state": "{}"})
    locs = _locs(r)
    assert "age_band" in locs and "row 2" in locs
    r = client.post("/score_book/csv", files={"file": ("book.csv", io.BytesIO(csv.encode()), "text/csv")}, data={"market_state": '{"vix_bucket": "wild"}'})
    assert "vix_bucket" in _locs(r)


# ---------------------------------------------------------------------------------------------
# /interventions/rank
# ---------------------------------------------------------------------------------------------


def test_rank_interventions(client: TestClient, demo_env: DemoEnv):
    before = _counts(demo_env)
    r = client.post("/interventions/rank", json={"client_id": "api-rank", "covariates": COV, "market_state": MARKET})
    assert r.status_code == 200, r.text
    body = r.json()
    ranked = body["ranked"]
    assert len(ranked) == 5 and [x["rank"] for x in ranked] == [1, 2, 3, 4, 5]
    assert [x["delta_p"] for x in ranked] == sorted(x["delta_p"] for x in ranked)
    assert all(x["label"] == "predicted effect, not causally validated" for x in ranked) and body["label"] == ranked[0]["label"]
    assert body["synthetic"] is True and body["scenario"]["context_tags"] == ["news:recession", "social:friend_sells"]
    assert _counts(demo_env)[0] == before[0] + 1
    assert "market_state" in _locs(client.post("/interventions/rank", json={"covariates": COV}))
    assert "drawdown_pct" in _locs(client.post("/interventions/rank", json={"covariates": COV, "market_state": {"drawdown_pct": "deep"}}))


# ---------------------------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------------------------


def test_openapi_json_matches_app(client: TestClient):
    from api.main import app

    committed = json.loads((API_DIR / "openapi.json").read_text(encoding="utf-8"))
    live = json.loads(json.dumps(app.openapi(), sort_keys=True))
    assert committed == live, "api/openapi.json is stale: run `python api/export_openapi.py`"
    paths = set(live["paths"])
    assert {"/health", "/model", "/clients", "/interventions", "/market", "/predict", "/profile", "/score_book", "/score_book/csv", "/interventions/rank"} <= paths
    served = client.get("/openapi.json")
    assert served.status_code == 200 and set(served.json()["paths"]) == paths
