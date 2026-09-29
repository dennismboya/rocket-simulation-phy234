"""Intake battery (Phase 6, page 4): the Phase 7 instrument embedded in the dashboard.

Two ways to collect a client's responses:

* **In-session**: the advisor runs the client through the scenario battery here. The items are
  drawn with the instrument's randomization scheme (``components.intake``:
  ``bre.design.battery_subset`` seeded by the session id, the full form's two verbatim repeats,
  randomized question order) and the assignment is recorded with the session. Consent is asked
  first; the optional training-consent checkbox sets ``consent_training`` on every row (rows
  without it never enter a fit). Progress and completion are shown; the rows are written through
  ``POST /clients/{client_id}/responses`` (validated by ``bre.schema`` on the API side).
* **Send a link**: the static instrument's URL (``GET /intake/config``) and the client id the
  advisor gives the client; the downloaded ``intake_<session_id>.json`` is uploaded here
  (``POST /intake/upload``, checked with the loader's contract before it is stored).

Forms: ``full`` (12 items + 2 repeats, 8-12 minutes) and ``short`` (5 items, about 3 minutes,
for annual renewals). In demo mode the database only takes synthetic rows, so everything
collected here is stored as SYNTHETIC and labelled so.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components import api_client as api  # noqa: E402
from components import chrome, intake, market_bar  # noqa: E402

health = chrome.page_setup("Intake battery")
base = api.base_url()
try:
    model = api.model_info(base)
    book = api.clients(base)
    cfg = api.intake_config(base)
except api.ApiError as exc:
    chrome.stop_on_error(exc)
market_bar.render(model)
chrome.load_settings(base)
demo = chrome.demo_mode()
forms_cfg = cfg.get("forms") or {}
intake_texts = cfg.get("intake") or {}
NEW_CLIENT = "New client…"

st.caption(
    f"Battery {cfg.get('battery_version')} on design {cfg.get('design_version')}; the same randomization scheme as the static instrument. "
    f"Storage: {cfg.get('storage_note')}"
)
if demo:
    st.info("Demo mode: responses recorded here are stored as SYNTHETIC rows in the demo database; they never enter a real-data fit.", icon="ℹ️")

S = st.session_state.setdefault("intake", {"stage": "setup"})


def _reset() -> None:
    st.session_state["intake"] = {"stage": "setup"}


def _client_options() -> list[str]:
    return [NEW_CLIENT] + [c["client_id"] for c in book]


def _label(cid: str) -> str:
    if cid == NEW_CLIENT:
        return cid
    c = next((x for x in book if x["client_id"] == cid), None)
    return f"{(c or {}).get('display_label') or cid} ({cid})"


# ---------------------------------------------------------------------------------------------
# Stage: setup (in-session start, or send a link + upload)
# ---------------------------------------------------------------------------------------------

if S["stage"] == "setup":
    mode = st.radio("How is the client answering?", ["In-session (advisor-led)", "Send a link (static instrument)"], key="ik_mode", horizontal=True)
    if mode.startswith("In-session"):
        c1, c2, c3 = st.columns([2, 2, 1])
        choice = c1.selectbox("Client", _client_options(), format_func=_label, key="ik_client")
        new_id = c2.text_input("New client id", key="ik_client_id", max_chars=64, help="letters, digits, _ . : - ; no names") if choice == NEW_CLIENT else ""
        new_label = c2.text_input("Display label (optional, the only identifying text stored)", key="ik_client_label", max_chars=80) if choice == NEW_CLIENT else ""
        form = c3.radio("Form", ["full", "short"], key="ik_form", format_func=lambda f: f"{f} ({(forms_cfg.get(f) or {}).get('n_presentations', intake.FORMS[f]['n_items'] + intake.FORMS[f]['n_repeats'])} presentations; "
                        + ("8-12 min" if f == "full" else "about 3 min, annual renewal") + ")")
        if st.button("Start session", key="ik_start", type="primary"):
            cid = new_id.strip() if choice == NEW_CLIENT else choice
            if not cid:
                st.error("Enter a client id.")
            else:
                seed = intake.new_seed()
                fc = forms_cfg.get(form) or {}
                pres, assignment = intake.build_presentations(seed, form, n_items=fc.get("n_items"), n_repeats=fc.get("n_repeats"))
                st.session_state["intake"] = {
                    "stage": "consent", "client_id": cid, "display_label": (new_label.strip() or None) if choice == NEW_CLIENT else None, "new_client": choice == NEW_CLIENT,
                    "form": form, "seed": seed, "presentations": pres, "assignment": assignment, "current": 0, "rows": [], "position": 0,
                    "started_at": intake.now_iso(), "consent_training": False, "covariates": {}, "quiz_score": None, "shown_at": None, "result": None,
                }
                st.rerun()
    else:
        st.subheader("Send a link")
        st.code(cfg.get("instrument_url") or "", language=None)
        st.caption(f"Instrument URL source: {cfg.get('instrument_url_source')}. URL parameters: {json.dumps(cfg.get('url_params') or {})}.")
        c1, c2 = st.columns(2)
        link_client = c1.selectbox("Client the session belongs to", [c["client_id"] for c in book] or [""], format_func=_label, key="ik_link_client")
        link_new = c2.text_input("…or a new client id", key="ik_link_new_id", max_chars=64)
        target_id = link_new.strip() or link_client
        st.markdown(f"Tell the client to keep this id with the file they send back: **`{target_id}`**. The downloaded file is named "
                    f"`{(cfg.get('file_names') or {}).get('responses', 'intake_<session_id>.json')}`; upload it below.")
        up = st.file_uploader("Responses file (intake_<session_id>.json)", type=["json"], key="ik_upload")
        as_syn = st.checkbox("Store as synthetic (demo database)", value=demo, disabled=demo, key="ik_upload_synthetic",
                             help="the demo database under data/synthetic refuses real rows, so in demo mode uploads are stored as synthetic and labelled so")
        if st.button("Upload responses", key="ik_upload_button", disabled=up is None or not target_id):
            try:
                res = api.upload_intake(base, target_id, up.name, up.getvalue(), store_as_synthetic=bool(as_syn))
                st.session_state["intake_upload_result"] = res
                st.success(f"Stored {res['n_rows']} rows for client {res['client_id']} (session {res['session_id']}); {res['note']}.")
            except api.ApiError as exc:
                st.session_state["intake_upload_result"] = {"error": str(exc)}
                st.error(f"Upload refused: {exc}")
        prev = st.session_state.get("intake_upload_result")
        if prev and "error" not in prev:
            st.json(prev)

# ---------------------------------------------------------------------------------------------
# Stage: consent
# ---------------------------------------------------------------------------------------------

elif S["stage"] == "consent":
    consent = intake_texts.get("consent") or {}
    st.subheader(consent.get("title") or "Before you start")
    for para in consent.get("text") or []:
        st.write(para)
    if demo:
        st.caption("Demo mode: this session is recorded as synthetic demo data.")
    participate = st.checkbox("I have read the above and agree to take part.", key="ik_consent_participate")
    training = st.checkbox("I consent to my anonymised responses being used to train the model", key="ik_consent_training")
    st.caption("Only rows with training consent enter a fit; the answers are stored either way for the advisor's use.")
    c1, c2 = st.columns([1, 5])
    if c1.button("Continue", key="ik_consent_next", disabled=not participate, type="primary"):
        S["consent_training"] = bool(training)
        S["stage"] = "about"
        st.rerun()
    if c2.button("Cancel session", key="ik_cancel_consent"):
        _reset()
        st.rerun()

# ---------------------------------------------------------------------------------------------
# Stage: about you (covariates + financial-literacy quiz)
# ---------------------------------------------------------------------------------------------

elif S["stage"] == "about":
    st.subheader("About you")
    instr = intake_texts.get("instructions") or {}
    for para in instr.get("text") or []:
        st.caption(para.replace("{n_scenarios}", str(len(S["presentations"]))))
    values: dict[str, object] = {}
    for field in intake_texts.get("covariates") or []:
        key = field["key"]
        if field.get("input") == "radio":
            opts = field.get("options") or []
            labels = [o["label"] for o in opts]
            pick = st.radio(field["text"], labels, key=f"ik_cov_{key}", index=None)
            values[key] = next((o["value"] for o in opts if o["label"] == pick), None) if pick is not None else None
        elif field.get("input") == "number":
            values[key] = int(st.number_input(field["text"], min_value=int(field.get("min", 0)), max_value=int(field.get("max", 40)), value=int(field.get("min", 0)), step=1, key=f"ik_cov_{key}"))
        elif field.get("input") == "likert":
            anchors = field.get("anchors") or {}
            values[key] = int(st.slider(f"{field['text']} ({anchors.get('1', '')} … {anchors.get('7', '')})", int(field.get("min", 1)), int(field.get("max", 7)), 4, key=f"ik_cov_{key}"))
    quiz = intake_texts.get("financial_literacy_quiz") or {}
    if quiz.get("items"):
        st.markdown(f"**Financial literacy.** {quiz.get('intro', '')}")
        answers: dict[str, int | None] = {}
        for it in quiz["items"]:
            pick = st.radio(it["text"], it["options"], key=f"ik_quiz_{it['item_id']}", index=None)
            answers[it["item_id"]] = None if pick is None else it["options"].index(pick)
        score = intake.quiz_score(answers, quiz["items"])
        values["financial_literacy_score"] = score
    required_radios = [f["key"] for f in (intake_texts.get("covariates") or []) if f.get("input") == "radio"]
    missing = [k for k in required_radios if st.session_state.get(f"ik_cov_{k}") is None]
    c1, c2 = st.columns([1, 5])
    if c1.button("Begin the scenarios", key="ik_about_next", disabled=bool(missing), type="primary"):
        S["covariates"] = values
        S["quiz_score"] = values.get("financial_literacy_score")
        S["stage"] = "items"
        S["current"] = 0
        S["shown_at"] = time.time()
        st.rerun()
    if missing:
        c2.caption(f"Answer every question (a 'Prefer not to say' is allowed): missing {', '.join(missing)}.")
    if c2.button("Cancel session", key="ik_cancel_about"):
        _reset()
        st.rerun()

# ---------------------------------------------------------------------------------------------
# Stage: items (one presentation per screen)
# ---------------------------------------------------------------------------------------------

elif S["stage"] == "items":
    pres = S["presentations"]
    k = int(S["current"])
    p = pres[k]
    n = len(pres)
    st.progress((k) / n, text=f"Scenario {k + 1} of {n}")
    try:
        texts = {kk: v["text"] for kk, v in (api.scenarios(base).get("texts") or {}).items()}
        ctx_rows = api.contexts(base)
    except api.ApiError as exc:
        chrome.stop_on_error(exc)
    ctx_display = {c["tag"]: c["display"] for c in ctx_rows if c.get("active", True)}
    text_versions = {"scenario": {kk: v["version"] for kk, v in (api.scenarios(base).get("texts") or {}).items()}, "contexts": {c["tag"]: c["version"] for c in ctx_rows}}
    tol_q = texts.get("tolerance_question") or intake_texts.get("questions", {}).get("tolerance", {}).get("text", "")
    sell_q = texts.get("sell_hold_question") or "What would you do with this position today?"
    alloc_q = texts.get("allocation_question") or "What share of this position would you sell?"
    prefix = f"it{k}_"
    tol_first = p["question_order_id"] == "tolerance-first"
    if S.get("shown_at") is None:
        S["shown_at"] = time.time()

    def tolerance_widget() -> int | None:
        pick = st.radio(tol_q, ["Yes", "No"], key=prefix + "tol", index=None, horizontal=True)
        return None if pick is None else (1 if pick == "Yes" else 0)

    tol_answer = tolerance_widget() if tol_first else None
    with st.container(border=True):
        for para in intake.scenario_paragraphs(texts, p, ctx_display):
            st.write(para)
    if p["has_news"]:
        dp = intake_texts.get("delay_page") or {}
        st.info(f"**{dp.get('title', 'A few days later')}** — {texts.get('delay_page_text') or dp.get('text', '')}", icon="⏳")
        st.caption(f"The static instrument shows this page for {dp.get('display_seconds', 10)} s before the decision; in-session the timer is not enforced (recorded in the session log).")
    sell_pick = st.radio(sell_q, ["Sell", "Hold"], key=prefix + "sell", index=None, horizontal=True)
    alloc = st.slider(alloc_q, 0, 100, 50, step=1, key=prefix + "alloc", format="%d%%", help="0% = sell nothing, 100% = sell everything")
    if not tol_first:
        tol_answer = tolerance_widget()
    ready = sell_pick is not None and tol_answer is not None
    c1, c2 = st.columns([1, 5])
    if c1.button("Record answers", key=prefix + "submit", disabled=not ready, type="primary"):
        rt_ms = (time.time() - float(S["shown_at"])) * 1000.0
        rows = intake.rows_for_presentation(
            p, {"tolerance": tol_answer, "sell_hold": 1 if sell_pick == "Sell" else 0, "allocation_share": alloc}, form=S["form"], subject_id=S["client_id"],
            session_id=S["seed"], covariates_json=intake.covariates_json(S["covariates"]), consent_training=bool(S["consent_training"]), is_synthetic=demo,
            position_start=int(S["position"]), timestamp=intake.now_iso(), response_time_ms=rt_ms,
        )
        S["rows"].extend(rows)
        S["position"] += len(rows)
        S["current"] = k + 1
        S["shown_at"] = time.time()
        if S["current"] >= n:
            session_info = {
                "form": S["form"], "seed": S["seed"], "mode": "in-session (advisor-led)", "consent_training": bool(S["consent_training"]), "started_at": S["started_at"],
                "finished_at": intake.now_iso(), "rt_granularity": intake.RT_GRANULARITY, "delay_page": "shown; timer not enforced in-session", "quiz_score": S["quiz_score"],
                "text_versions": text_versions, "is_synthetic": demo, "is_synthetic_reason": "demo mode: the database under data/synthetic takes synthetic rows only" if demo else None,
                "battery_version": intake.BATTERY_VERSION, "n_presentations": n,
            }
            try:
                S["result"] = api.post_responses(base, S["client_id"], S["rows"], S["assignment"], session_info, S.get("display_label"))
            except api.ApiError as exc:
                S["result"] = {"error": str(exc)}
            S["session_info"] = session_info
            S["stage"] = "done"
        st.rerun()
    if c2.button("Cancel session", key=prefix + "cancel"):
        _reset()
        st.rerun()
    st.caption(f"Presentation {p['index']}: item {p['item_id']} · question order {p['question_order_id']} · rows so far {len(S['rows'])} · session {S['seed']}")

# ---------------------------------------------------------------------------------------------
# Stage: done
# ---------------------------------------------------------------------------------------------

elif S["stage"] == "done":
    res = S.get("result") or {}
    if "error" in res:
        st.error(f"The responses were not stored: {res['error']}")
        st.download_button("Download the responses JSON (to retry or archive)", json.dumps(S["rows"], indent=1), file_name=intake.responses_file_name(S["seed"]), mime="application/json", key="ik_download_failed")
    else:
        st.success(f"Session complete: {res.get('n_rows')} rows stored for client {res.get('client_id')} (session {res.get('session_id')}). {res.get('note')}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Presentations", len(S["presentations"]))
        m2.metric("Rows stored", res.get("n_rows"))
        m3.metric("Training consent", "yes" if res.get("consent_training") else "no")
        m4.metric("Stored as", "synthetic (demo)" if res.get("is_synthetic") else "real")
        st.download_button("Download responses JSON", json.dumps(S["rows"], indent=1), file_name=intake.responses_file_name(S["seed"]), mime="application/json", key="ik_download")
    with st.expander("Recorded assignment (randomized item draw, question orders, repeats)"):
        st.json(S["assignment"])
    with st.expander("Session record"):
        st.json(S.get("session_info") or {})
    c1, c2 = st.columns([1, 1])
    if c1.button("Open this client's profile", key="ik_open_profile"):
        st.session_state["selected_client"] = S["client_id"]
        st.switch_page("pages/2_Client_profile.py")
    if c2.button("Start another session", key="ik_reset"):
        _reset()
        st.rerun()
