"""Phase 6 acceptance tests of dashboard pages 4-7 (intake battery, scenario and context editor,
transparency, settings) with Streamlit's ``AppTest`` against a live API on a copy of the demo
database (same fixture pattern as ``tests/test_dashboard.py``; the API runs with ``BRE_ADMIN=1``
so the activation control is exercised).

Checks: the SYNTHETIC DATA banner on pages 4-7; an in-session short-form battery stores 15 rows
with its randomized assignment recorded; the send-a-link upload path; an uncalibrated context is
flagged in red on the editor; the exact "Decision rule not yet run" sentence and no Phase 4
metric table on the transparency page before the verdict exists; the settings page changes the
triage alert threshold; exporting and deleting a client (every table empty afterwards); changing
a what-if input changes every dependent number on the client profile; and the wording rule.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

BRE = Path(__file__).resolve().parents[1]
APP = BRE / "app"
HOME = APP / "Home.py"
TRIAGE = "pages/1_Book_triage.py"
PROFILE = "pages/2_Client_profile.py"
INTAKE = "pages/4_Intake_battery.py"
EDITOR = "pages/5_Scenario_editor.py"
TRANSPARENCY = "pages/6_Transparency.py"
SETTINGS = "pages/7_Settings.py"
RUN_TIMEOUT_S = 600.0
API_START_TIMEOUT_S = 420.0
NOT_RUN = "Decision rule not yet run: no real-data numbers are shown"

sys.path.insert(0, str(APP))
from test_dashboard import SHORT_FIT_DRAWS, SHORT_FIT_STEPS, _seeded_demo_db  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def pages_api(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("bre_pages")
    db_dir = root / "data" / "synthetic"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "demo.db"
    models_root = BRE / "runs" / "models"
    seeded = _seeded_demo_db()
    if seeded is not None:
        shutil.copy(seeded, db_path)
    else:
        from db.demo_seed import demo_engine, seed_demo

        models_root = root / "runs" / "models"
        seed_demo(demo_engine(db_path), models_root=models_root, fit_steps=SHORT_FIT_STEPS, fit_restarts=1, n_draws=SHORT_FIT_DRAWS, write_files=False)
    port = _free_port()
    env = {**os.environ, "BRE_DB_URL": f"sqlite:///{db_path}", "BRE_MODELS_ROOT": str(models_root), "BRE_API_AUTOSEED": "0", "BRE_API_WARMUP": "1", "BRE_MARKET_NETWORK": "0",
           "BRE_ADMIN": "1", "BRE_REPORTS_DIR": str(root / "reports_absent"), "PYTHONPATH": os.pathsep.join([str(BRE / "src"), str(BRE)])}
    log_path = root / "api.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"], cwd=BRE, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + API_START_TIMEOUT_S
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"API exited early:\n{log_path.read_text(encoding='utf-8')[-4000:]}")
        try:
            r = httpx.get(base + "/health", timeout=5.0)
            if r.status_code == 200 and r.json().get("status") == "ok":
                ready = True
                break
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    if not ready:
        proc.kill()
        raise RuntimeError(f"API did not become ready:\n{log_path.read_text(encoding='utf-8')[-4000:]}")
    old = {k: os.environ.get(k) for k in ("BRE_API_URL", "BRE_ADMIN")}
    os.environ["BRE_API_URL"] = base
    os.environ["BRE_ADMIN"] = "1"
    try:
        yield {"base": base, "db_path": db_path}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture()
def home(pages_api) -> AppTest:
    at = AppTest.from_file(str(HOME), default_timeout=RUN_TIMEOUT_S)
    at.run()
    _ok(at)
    return at


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value) for e in at.exception]


def _texts(at: AppTest) -> list[str]:
    out: list[str] = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "text", "warning", "error", "success", "info"):
        out.extend(str(e.value) for e in at.get(kind))
    return out


def _page(at: AppTest, page: str) -> AppTest:
    at.switch_page(page)
    at.run()
    _ok(at)
    return at


def _metric(at: AppTest, label: str) -> str:
    for m in at.metric:
        if m.label == label:
            return str(m.value)
    raise AssertionError(f"no metric {label!r}; have {[m.label for m in at.metric]}")


def _counts(db_path: Path, client_id: str) -> dict[str, int]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = {t: con.execute(f"SELECT COUNT(*) FROM {t} WHERE client_id = ?", (client_id,)).fetchone()[0] for t in ("responses", "predictions_log", "intervention_log")}
        out["clients"] = con.execute("SELECT COUNT(*) FROM clients WHERE client_id = ?", (client_id,)).fetchone()[0]
        out["bre_documents"] = sum(1 for (p,) in con.execute("SELECT payload FROM bre_documents WHERE kind = 'intake_session'") if json.loads(p).get("client_id") == client_id)
        return out
    finally:
        con.close()


# ---------------------------------------------------------------------------------------------
# Banner and wording
# ---------------------------------------------------------------------------------------------


def test_banner_on_pages_4_to_7_in_demo_mode(home: AppTest):
    assert home.session_state["demo_mode"] is True
    for page in (INTAKE, EDITOR, TRANSPARENCY, SETTINGS):
        _page(home, page)
        assert any("SYNTHETIC DATA" in t for t in _texts(home)), page
        assert sum("SYNTHETIC DATA" in str(w.value) for w in home.warning) >= 2, page


def test_no_will_sell_wording_on_the_new_pages():
    offenders = [str(p) for p in (APP / "pages").glob("*.py") if "will sell" in p.read_text(encoding="utf-8").lower()]
    offenders += [str(p) for p in (APP / "components").glob("*.py") if "will sell" in p.read_text(encoding="utf-8").lower()]
    assert not offenders, offenders


# ---------------------------------------------------------------------------------------------
# Page 4: intake battery
# ---------------------------------------------------------------------------------------------


def test_in_session_short_form_stores_rows_and_records_the_assignment(home: AppTest, pages_api):
    _page(home, INTAKE)
    home.selectbox(key="ik_client").set_value("New client…").run()
    home.text_input(key="ik_client_id").set_value("PAGE-INTAKE-1").run()
    home.radio(key="ik_form").set_value("short").run()
    home.button(key="ik_start").click().run()
    _ok(home)
    S = home.session_state["intake"]
    assert S["stage"] == "consent" and len(S["presentations"]) == 5 and S["form"] == "short" and re.fullmatch(r"[0-9a-f]{32}", S["seed"])
    assignment = S["assignment"]
    assert assignment["form"] == "short" and len(assignment["item_ids"]) == 5 and assignment["n_presentations"] == 5 and assignment["seed"] == S["seed"]
    assert assignment["condition_class_counts"] == {"none": 1, "single": 4, "pair": 0} and sorted(assignment["loss_counts"].values()) == [1, 1, 1, 1, 1]
    assert sorted(assignment["order_counts"].values()) == [2, 3]
    # consent: the training checkbox sets consent_training on every row
    assert home.button(key="ik_consent_next").disabled
    home.checkbox(key="ik_consent_participate").check().run()
    home.checkbox(key="ik_consent_training").check().run()
    home.button(key="ik_consent_next").click().run()
    _ok(home)
    assert home.session_state["intake"]["stage"] == "about"
    for r in home.radio:
        if r.key and (r.key.startswith("ik_cov_") or r.key.startswith("ik_quiz_")):
            r.set_value(r.options[0])
    home.run()
    home.button(key="ik_about_next").click().run()
    _ok(home)
    S = home.session_state["intake"]
    assert S["stage"] == "items" and set(S["covariates"]) >= {"age_band", "education", "wealth_band", "invest_experience_yrs", "self_reported_risk_tolerance", "financial_literacy_score"}
    for k in range(5):
        home.radio(key=f"it{k}_tol").set_value("Yes" if k % 2 else "No")
        home.radio(key=f"it{k}_sell").set_value("Sell" if k % 2 else "Hold")
        home.slider(key=f"it{k}_alloc").set_value(25)
        home.run()
        home.button(key=f"it{k}_submit").click().run()
        _ok(home)
    S = home.session_state["intake"]
    assert S["stage"] == "done" and len(S["rows"]) == 15
    res = S["result"]
    assert "error" not in res and res["n_rows"] == 15 and res["is_synthetic"] is True and res["consent_training"] is True
    assert any("Session complete" in t for t in _texts(home))
    assert any("Download responses JSON" in b.label for b in home.download_button)
    # rows: three per presentation in the item's order, tolerance rows keyed by the design's tolerance id, positions consecutive
    rows = S["rows"]
    assert [r["position_in_session"] for r in rows] == list(range(15)) and all(r["is_synthetic"] and r["consent_training"] and r["dataset"] == "intake_battery" for r in rows)
    assert {r["elicitation_type"] for r in rows} == {"binary_yes_no", "binary_sell", "allocation_pct"}
    for p, k in zip(S["presentations"], range(0, 15, 3)):
        trio = rows[k:k + 3]
        seq = ["tolerance", "sell_hold", "allocation_share"] if p["question_order_id"] == "tolerance-first" else ["sell_hold", "allocation_share", "tolerance"]
        assert [r["source_row_ref"].split(":")[-1] for r in trio] == seq and trio[1]["prior_question_ids"] == [seq[0]]
    # stored through the API: the client exists with 15 responses; the assignment is in the intake-session document
    base = pages_api["base"]
    clients = {c["client_id"]: c for c in httpx.get(base + "/clients", timeout=60).json()}
    assert clients["PAGE-INTAKE-1"]["n_responses"] == 15
    exp = httpx.get(base + "/clients/PAGE-INTAKE-1/export", timeout=60).json()
    assert len(exp["intake_sessions"]) == 1 and exp["intake_sessions"][0]["payload"]["assignment"]["item_ids"] == assignment["item_ids"]
    assert exp["intake_sessions"][0]["payload"]["session"]["text_versions"]["scenario"]["intro"] >= 1
    assert _counts(pages_api["db_path"], "PAGE-INTAKE-1")["responses"] == 15


def test_send_a_link_shows_the_instrument_url_and_uploads_a_file(home: AppTest, pages_api):
    from components import intake as I

    _page(home, INTAKE)
    home.radio(key="ik_mode").set_value("Send a link (static instrument)").run()
    _ok(home)
    code = [str(c.value) for c in home.code]
    assert code and code[0].startswith("file://") and code[0].endswith("index.html")
    seed = I.new_seed()
    pres, _ = I.build_presentations(seed, "short")
    rows, pos = [], 0
    for p in pres:
        r = I.rows_for_presentation(p, {"tolerance": 1, "sell_hold": 0, "allocation_share": 10}, form="short", subject_id="c0ffee00-0000-4000-8000-000000000000", session_id=seed,
                                    covariates_json=I.covariates_json({"age_band": "45-59"}), consent_training=True, is_synthetic=False, position_start=pos, timestamp=I.now_iso(), response_time_ms=700)
        rows.extend(r)
        pos += 3
    home.text_input(key="ik_link_new_id").set_value("PAGE-UPLOAD-1").run()
    assert any("PAGE-UPLOAD-1" in t for t in _texts(home))
    home.file_uploader(key="ik_upload").upload(f"intake_{seed}.json", json.dumps(rows).encode(), "application/json").run()
    home.button(key="ik_upload_button").click().run()
    _ok(home)
    res = home.session_state["intake_upload_result"]
    assert "error" not in res and res["n_rows"] == 15 and res["is_synthetic"] is True and res["client_id"] == "PAGE-UPLOAD-1"
    assert any("Stored 15 rows" in t for t in _texts(home))
    assert _counts(pages_api["db_path"], "PAGE-UPLOAD-1")["responses"] == 15


# ---------------------------------------------------------------------------------------------
# Page 5: editor
# ---------------------------------------------------------------------------------------------


def test_editor_flags_an_uncalibrated_context(home: AppTest, pages_api):
    _page(home, EDITOR)
    assert home.session_state["editor_uncalibrated"] == [] or all(t in httpx.get(pages_api["base"] + "/model", timeout=60).json()["calibrated_contexts"] for t in home.session_state["editor_uncalibrated"])
    home.text_input(key="ce_tag").set_value("news:tariff_shock").run()
    home.text_area(key="ce_display").set_value("The news reports that a tariff shock is pulling the market down.").run()
    home.button(key="ce_add").click().run()
    _ok(home)
    assert "news:tariff_shock" in home.session_state["editor_uncalibrated"]
    lines = [m for m in (str(e.value) for e in home.markdown) if "`news:tariff_shock`" in m]
    assert lines and ":red[**uncalibrated: prior only**]" in lines[0] and "N = 0" in lines[0]
    assert any("Uncalibrated (prior only): " in str(w.value) and "news:tariff_shock" in str(w.value) for w in home.warning)
    # the design contexts keep their calibrated status from GET /model
    table = httpx.get(pages_api["base"] + "/model", timeout=60).json()["calibrated_contexts"]
    for tag, e in table.items():
        line = next(m for m in (str(x.value) for x in home.markdown) if f"`{tag}`" in m)
        assert e["status"] in line and f"N = {e['n_responses']}" in line
    # scenario texts and intervention scripts save as new versions
    home.text_area(key="sc_intro").set_value("Edited intro from the page test.").run()
    home.button(key="sc_save_intro").click().run()
    _ok(home)
    scen = httpx.get(pages_api["base"] + "/scenarios", params={"history": "true"}, timeout=60).json()
    assert scen["texts"]["intro"]["version"] == 2 and scen["texts"]["intro"]["text"] == "Edited intro from the page test." and any(h["key"] == "intro" and h["version"] == 1 for h in scen["history"])
    home.text_area(key="iv_script_cash_sleeve_check").set_value("Edited script from the page test.").run()
    home.button(key="iv_save_cash_sleeve_check").click().run()
    _ok(home)
    live = {i["name"]: i for i in httpx.get(pages_api["base"] + "/interventions", timeout=60).json()}
    assert live["cash_sleeve_check"]["script"] == "Edited script from the page test."


# ---------------------------------------------------------------------------------------------
# Page 6: transparency
# ---------------------------------------------------------------------------------------------


def test_transparency_shows_the_not_run_sentence_and_no_phase4_table(home: AppTest, pages_api):
    _page(home, TRANSPARENCY)
    texts = _texts(home)
    assert any(NOT_RUN in t for t in texts)
    assert home.session_state["transparency_phase4_rendered"] is False and home.session_state["transparency_real_numbers_shown"] is False
    assert not any("Held-out metrics (Phase 4)" in t for t in texts)
    for d in home.dataframe:
        assert not ({"nll_ci95", "brier_ci95"} & set(map(str, d.value.columns))), "a Phase 4 metric table was rendered before the verdict exists"
    assert any("Q-model supported" in t for t in texts)  # the decision rule verbatim
    model = httpx.get(pages_api["base"] + "/model", timeout=60).json()
    assert _metric(home, "Version") == model["version"] and _metric(home, "Parameters").replace(",", "") == str(model["n_params"]) and _metric(home, "Trained on synthetic data") == "yes"
    assert any("SYNTHETIC: every number in this section" in t for t in texts)
    frames = [d.value for d in home.dataframe]
    assert any({"version", "type", "active"} <= set(map(str, f.columns)) for f in frames)  # registry
    assert any({"dataset", "real_or_synthetic", "license"} <= set(map(str, f.columns)) for f in frames)  # provenance
    assert any({"bin", "mean_p", "frac_pos"} <= set(map(str, f.columns)) for f in frames)  # in-sample reliability (synthetic)
    assert home.selectbox(key="tr_version") is not None and home.button(key="tr_activate") is not None  # admin control (BRE_ADMIN=1)
    assert home.button(key="tr_retrain") is not None


# ---------------------------------------------------------------------------------------------
# Page 7: settings, export, delete
# ---------------------------------------------------------------------------------------------


def test_settings_change_the_triage_threshold_and_delete_removes_every_row(home: AppTest, pages_api):
    _page(home, SETTINGS)
    home.number_input(key="st_alert").set_value(0.15).run()
    home.toggle(key="st_demo").set_value(True).run()
    home.button(key="st_save").click().run()
    _ok(home)
    settings = httpx.get(pages_api["base"] + "/settings", timeout=60).json()
    assert settings["settings"]["alert_threshold"] == 0.15 and settings["settings"]["demo_mode"] is True and settings["demo_mode"] is True
    assert home.session_state["alert_threshold"] == 0.15
    _page(home, TRIAGE)
    alerts = [str(e.value) for e in list(home.error) + list(home.success)]
    assert any("above threshold 0.15" in a for a in alerts), alerts
    assert any("SYNTHETIC DATA" in t for t in _texts(home))
    # export a client
    _page(home, SETTINGS)
    home.selectbox(key="st_export_client").set_value("PAGE-INTAKE-1").run()
    home.button(key="st_export").click().run()
    _ok(home)
    dump = json.loads(home.session_state["export_json"])
    assert dump["client"]["client_id"] == "PAGE-INTAKE-1" and len(dump["responses"]) == 15 and len(dump["intake_sessions"]) == 1 and dump["synthetic"] is True
    assert any(b.key == "st_export_download" for b in home.download_button)
    # delete it: every table empty afterwards
    before = _counts(pages_api["db_path"], "PAGE-INTAKE-1")
    assert before["responses"] == 15 and before["clients"] == 1 and before["bre_documents"] == 1
    home.selectbox(key="st_delete_client").set_value("PAGE-INTAKE-1").run()
    assert home.button(key="st_delete").disabled
    home.checkbox(key="st_delete_confirm").check().run()
    home.button(key="st_delete").click().run()
    _ok(home)
    res = home.session_state["delete_result"]
    assert res["client_id"] == "PAGE-INTAKE-1" and res["deleted"]["responses"] == 15 and res["deleted"]["bre_documents"] == 1 and res["audit_log_id"] > 0
    assert _counts(pages_api["db_path"], "PAGE-INTAKE-1") == {"responses": 0, "predictions_log": 0, "intervention_log": 0, "clients": 0, "bre_documents": 0}
    assert "PAGE-INTAKE-1" not in {c["client_id"] for c in httpx.get(pages_api["base"] + "/clients", timeout=60).json()}
    httpx.put(pages_api["base"] + "/settings", json={"alert_threshold": 0.5, "demo_mode": False}, timeout=60)


# ---------------------------------------------------------------------------------------------
# Page 3: the what-if panel recomputes with every input
# ---------------------------------------------------------------------------------------------


def test_what_if_inputs_change_every_dependent_number(home: AppTest):
    _page(home, PROFILE)
    wording = "predicted probability of selling"
    p0, d0, top0 = _metric(home, wording), _metric(home, "Change vs baseline"), _metric(home, "Top driver")
    home.slider(key="wi_loss").set_value(35.0).run()
    _ok(home)
    p1, d1 = _metric(home, wording), _metric(home, "Change vs baseline")
    assert p1 != p0 and d1 != d0
    home.selectbox(key="wi_social").set_value("high").run()
    _ok(home)
    p2, d2, top2 = _metric(home, wording), _metric(home, "Change vs baseline"), _metric(home, "Top driver")
    assert p2 != p1 and d2 != d1
    assert any("friend sells" in t for t in _texts(home) if t.startswith("**Scenario the API built"))  or any("friend sells" in t for t in _texts(home))
    assert top2 in ("friend sells", "loss", "recession news", "technical-fault news", "5% recovery", "none")
