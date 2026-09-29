"""Phase 6 acceptance tests of the advisor dashboard (``app/``) with Streamlit's ``AppTest``.

A live API (``uvicorn api.main:app``) is started once per session on a free port against a
copy of the seeded demo database (``data/synthetic/demo.db`` with its active model under
``runs/models``; when that is missing, a short-fit demo is seeded into the temporary directory
as ``tests/test_predict.py`` does). The pages run in this process and reach the service through
``BRE_API_URL``.

Checks (PLAN.md, Phase 6): the SYNTHETIC DATA banner on every page in demo mode; the drawdown
slider changes the triage probabilities; the threshold alert count updates; the CSV export's
first line is the banner; the client-profile heatmap has 5 loss rows x 17 context columns; the
PDF export carries the banner; a weak cause-frame match is flagged; the market button fills the
bar; interventions logged in the session show up on the triage page; and no file under
``app/`` contains the string "will sell".
"""

from __future__ import annotations

import io
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
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

BRE = Path(__file__).resolve().parents[1]
APP = BRE / "app"
HOME = APP / "Home.py"
TRIAGE = "pages/1_Book_triage.py"
PROFILE = "pages/2_Client_profile.py"
RUN_TIMEOUT_S = 600.0
API_START_TIMEOUT_S = 420.0

SHORT_FIT_STEPS = 300
SHORT_FIT_DRAWS = 100


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _seeded_demo_db() -> Path | None:
    """The repository's demo database when it holds an active model whose artifact exists."""
    db = BRE / "data" / "synthetic" / "demo.db"
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT training_data_refs FROM model_registry WHERE is_active = 1 LIMIT 1").fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    for ref in str(row[0]).strip("[]").replace('"', "").split(","):
        ref = ref.strip()
        if ref.startswith("artifact:"):
            path = Path(ref[len("artifact:"):])
            if not path.is_absolute():
                path = BRE / path
            return db if (path / "artifact.json").exists() else None
    return None


@pytest.fixture(scope="session")
def api_server(tmp_path_factory: pytest.TempPathFactory) -> str:
    root = tmp_path_factory.mktemp("bre_dashboard")
    db_dir = root / "data" / "synthetic"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "demo.db"
    models_root = BRE / "runs" / "models"
    seeded = _seeded_demo_db()
    if seeded is not None:
        shutil.copy(seeded, db_path)
    else:  # fresh clone: short demo fit into the temporary directory
        from db.demo_seed import demo_engine, seed_demo

        models_root = root / "runs" / "models"
        seed_demo(demo_engine(db_path), models_root=models_root, fit_steps=SHORT_FIT_STEPS, fit_restarts=1, n_draws=SHORT_FIT_DRAWS, write_files=False)
    port = _free_port()
    env = {
        **os.environ,
        "BRE_DB_URL": f"sqlite:///{db_path}",
        "BRE_MODELS_ROOT": str(models_root),
        "BRE_API_AUTOSEED": "0",
        "BRE_API_WARMUP": "1",
        "BRE_MARKET_NETWORK": "0",
        "PYTHONPATH": os.pathsep.join([str(BRE / "src"), str(BRE)]),
    }
    log_path = root / "api.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
            cwd=BRE, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
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
        raise RuntimeError(f"API did not become ready within {API_START_TIMEOUT_S:.0f} s:\n{log_path.read_text(encoding='utf-8')[-4000:]}")
    os.environ["BRE_API_URL"] = base  # the AppTest scripts run in this process
    try:
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture()
def home(api_server: str) -> AppTest:
    os.environ["BRE_API_URL"] = api_server
    at = AppTest.from_file(str(HOME), default_timeout=RUN_TIMEOUT_S)
    at.run()
    _no_exceptions(at)
    return at


def _no_exceptions(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def _texts(at: AppTest) -> list[str]:
    out: list[str] = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "text", "warning", "error", "success", "info"):
        out.extend(str(e.value) for e in at.get(kind))
    return out


def _alert(at: AppTest) -> tuple[int, float]:
    for e in list(at.error) + list(at.success) + list(at.warning):
        m = re.search(r"(\d+) clients above threshold (\d\.\d\d)", str(e.value))
        if m:
            return int(m.group(1)), float(m.group(2))
    raise AssertionError("no threshold alert found")


def _triage_table(at: AppTest):
    frames = [d.value for d in at.dataframe if "p_sell" in d.value.columns]
    assert frames, "no triage table rendered"
    return frames[0]


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


def test_banner_on_every_page_in_demo_mode(home: AppTest):
    assert home.session_state["demo_mode"] is True
    for page in (None, TRIAGE, PROFILE):
        if page:
            home.switch_page(page)
            home.run()
        _no_exceptions(home)
        texts = _texts(home)
        assert any("SYNTHETIC DATA" in t for t in texts), page
        # the banner is rendered both in the main area and in the sidebar
        assert sum("SYNTHETIC DATA" in str(w.value) for w in home.warning) >= 2, page


def test_drawdown_slider_changes_triage_probabilities(home: AppTest):
    home.switch_page(TRIAGE)
    home.run()
    _no_exceptions(home)
    before = _triage_table(home).set_index("client_id")["p_sell"]
    assert home.session_state["market"]["drawdown_pct"] == -10.0
    home.slider(key="ms_drawdown_pct").set_value(-30.0).run()
    _no_exceptions(home)
    assert home.session_state["market"]["drawdown_pct"] == -30.0
    after = _triage_table(home).set_index("client_id")["p_sell"]
    assert set(before.index) == set(after.index) and len(before) > 0
    joined = before.to_frame("before").join(after.to_frame("after"))
    assert not np.allclose(joined["before"], joined["after"])
    # most clients move (the Q-model's loss rotation is not monotone for every client, so no direction is asserted)
    assert (np.abs(joined["after"] - joined["before"]) > 0.01).mean() > 0.5
    assert ((joined >= 0) & (joined <= 1)).all().all()
    # every row carries its interval and the calibration N
    table = _triage_table(home)
    assert table["interval80"].str.startswith("[").all()
    assert (table["calibration_n"] > 0).all()


def test_threshold_alert_count_updates(home: AppTest):
    home.switch_page(TRIAGE)
    home.run()
    _no_exceptions(home)
    n0, t0 = _alert(home)
    table = _triage_table(home)
    assert n0 == int((table["p_sell"] >= t0).sum())
    home.number_input(key="alert_threshold").set_value(0.0).run()
    _no_exceptions(home)
    n_all, t_all = _alert(home)
    assert t_all == 0.0 and n_all == len(table)
    home.number_input(key="alert_threshold").set_value(1.0).run()
    n_none, t_none = _alert(home)
    assert t_none == 1.0 and n_none == 0


def test_csv_export_header_carries_the_banner(home: AppTest):
    home.switch_page(TRIAGE)
    home.run()
    _no_exceptions(home)
    assert any("CSV" in b.label for b in home.download_button)
    lines = home.session_state["triage_csv"].splitlines()
    assert lines[0].startswith("# SYNTHETIC DATA")
    assert lines[1].startswith("# model=") and "predicted probability of selling" in lines[1] and "market_state=" in lines[1]
    assert "client_id" in lines[2] and "p_sell" in lines[2] and "interval80" in lines[2]
    assert len(lines) == 3 + len(_triage_table(home))
    assert "will sell" not in home.session_state["triage_csv"].lower()


def test_client_profile_heatmap_and_pdf(home: AppTest):
    home.switch_page(PROFILE)
    home.run()
    _no_exceptions(home)
    heat = home.session_state["profile_heatmap"]
    assert heat.shape == (5, 17)
    assert list(heat.index) == ["5% loss", "10% loss", "15% loss", "20% loss", "30% loss"]
    assert "no context" in heat.columns
    assert sum("›" in c for c in heat.columns) == 12 and sum("›" not in c for c in heat.columns) == 5
    assert ((heat >= 0) & (heat <= 1)).all().all()
    shapes = [d.value.shape for d in home.dataframe]
    assert (5, 17) in shapes
    pdf = home.session_state["profile_pdf"]
    assert pdf[:5] == b"%PDF-"
    from pypdf import PdfReader

    text = " ".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    assert "SYNTHETIC DATA" in text and "Behavioral Risk Profile" in text
    assert "not causally validated" in text
    assert "will sell" not in text.lower()
    labels = [b.label for b in home.download_button]
    assert any("PDF" in lbl for lbl in labels)


def test_intervention_log_reaches_triage_status(home: AppTest):
    home.switch_page(PROFILE)
    home.run()
    _no_exceptions(home)
    cid = home.session_state["selected_client"]
    assert any("not causally validated" in str(w.value) for w in home.warning)
    home.button(key="log_button").click().run()
    _no_exceptions(home)
    entry = home.session_state["client_store"][cid]
    assert entry["status"] == "intervention logged" and len(entry["log"]) == 1 and entry["log"][0]["persisted"] is False
    home.switch_page(TRIAGE)
    home.run()
    _no_exceptions(home)
    table = _triage_table(home).set_index("client_id")
    assert table.loc[cid, "contact_status"] == "intervention logged"
    assert table.loc[cid, "last_contact"] != "—"


def test_weak_cause_frame_match_is_flagged(home: AppTest):
    home.selectbox(key="ms_cause_choice").set_value("free text…").run()
    home.text_input(key="ms_cause_frame").set_value("xyzzy plugh").run()
    _no_exceptions(home)
    assert any("Weak match" in str(w.value) for w in home.warning)
    table = httpx.get(os.environ["BRE_API_URL"] + "/model", timeout=30.0).json()["calibrated_contexts"]
    calibrated = [t for t, e in table.items() if t.startswith("news:") and e.get("status") == "calibrated"]
    assert calibrated, "GET /model reported no calibrated news context"
    assert all(any(t in o for o in home.selectbox(key="ms_cause_choice").options) for t in calibrated)  # the dropdown lists them
    home.selectbox(key="ms_cause_choice").set_value(calibrated[0]).run()
    _no_exceptions(home)
    assert not any("Weak match" in str(w.value) for w in home.warning)


def test_load_current_market_button(home: AppTest):
    home.button(key="ms_load").click().run()
    _no_exceptions(home)
    info = home.session_state["market_loaded"]
    assert "error" not in info and info.get("source")
    assert any("Loaded from" in str(c.value) for c in home.caption)
    m = home.session_state["market"]
    assert -50.0 <= m["drawdown_pct"] <= 20.0 and m["duration_days"] >= 0


def test_no_will_sell_wording_in_app():
    offenders = []
    for path in APP.rglob("*"):
        if path.suffix in (".py", ".md", ".toml", ".txt") and path.is_file():
            if "will sell" in path.read_text(encoding="utf-8").lower():
                offenders.append(str(path))
    assert not offenders, offenders
