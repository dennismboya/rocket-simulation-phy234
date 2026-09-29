"""Scenario and context editor (Phase 6, page 5).

Edit the scenario texts the intake battery shows, add or retire contexts (for example
``news:tariff_shock``) and edit the intervention scripts. Everything is versioned by the API
(``api.store``: every save appends a new version with a timestamp; past versions and past
responses are never rewritten). Every context shows its calibration status from the active
artifact's ``calibrated_contexts``: the number of consented training responses behind it, the
fitted rotation ``theta`` with its 95% interval, or ``uncalibrated: prior only`` in red. A
context that is not in the served model's vocabulary (a newly added one) has no parameter and
is uncalibrated until the model is refitted on responses that carry it; the shared design of
the battery (design 1.0.0) does not draw it, so such responses come from a future design
version.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import api_client as api  # noqa: E402
from components import chrome, market_bar  # noqa: E402

health = chrome.page_setup("Scenario and context editor")
base = api.base_url()
try:
    model = api.model_info(base)
    contexts = api.contexts(base)
    scen = api.scenarios(base, history=True)
    ivs = api.interventions_versions(base, history=True)
except api.ApiError as exc:
    chrome.stop_on_error(exc)
market_bar.render(model)
chrome.load_settings(base)
st.caption(scen.get("note") or "")

# ---------------------------------------------------------------------------------------------
# 1. Contexts with calibration status
# ---------------------------------------------------------------------------------------------

st.subheader("Contexts")
st.caption(f"Calibration rule: {model.get('calibration_rule')}. Status from the served model {model.get('version')} ({model.get('model_type')}); theta is the context's su(2) rotation vector (Q2/Q4), 95% interval over the artifact's draws.")
uncal: list[str] = []
for c in contexts:
    cal = c.get("calibration") or {}
    status = str(cal.get("status") or "")
    if status.startswith("uncalibrated"):
        uncal.append(c["tag"])
    theta = cal.get("theta")
    ci = cal.get("theta_ci95")
    theta_txt = "no parameter" if theta is None else (f"θ = [{', '.join(f'{float(x):+.2f}' for x in theta)}]" + (f", 95% [{', '.join(f'{float(x):+.2f}' for x in ci[0])}] … [{', '.join(f'{float(x):+.2f}' for x in ci[1])}]" if ci else ""))
    flags = ("design" if c.get("in_design") else "added") + ("" if c.get("active", True) else ", RETIRED") + ("" if c.get("in_served_vocabulary") else ", not in the served vocabulary")
    st.markdown(
        f"**{chrome.context_label(c['tag'])}** `{c['tag']}` ({c.get('kind')}; {flags}; v{c.get('version')}) — "
        f"{chrome.uncalibrated_markdown(status)} — consented training responses N = {cal.get('n_responses')} (minimum {cal.get('min_responses')}) — {theta_txt}"
    )
    st.caption(f"Sentence: {c.get('display')}" + (f" — note: {c.get('note')}" if c.get("note") else ""))
st.session_state["editor_uncalibrated"] = uncal
if uncal:
    st.warning(f"Uncalibrated (prior only): {', '.join(uncal)}. Predictions that involve them rest on the prior; the triage status carries the flag 'uncalibrated context'.", icon="⚠️")
st.dataframe(pd.DataFrame([{"tag": c["tag"], "kind": c.get("kind"), "active": c.get("active", True), "version": c.get("version"), "updated": c.get("updated_at"), "in design": c.get("in_design"),
                            "in served vocabulary": c.get("in_served_vocabulary"), "N": (c.get("calibration") or {}).get("n_responses"), "status": (c.get("calibration") or {}).get("status"),
                            "sentence": c.get("display")} for c in contexts]), hide_index=True, width="stretch")

with st.expander("Add a context or write a new version of one", expanded=True):
    st.caption("A new tag is namespace:value (e.g. news:tariff_shock). It appears here as uncalibrated: prior only until a model is refitted on responses that carry it.")
    e1, e2 = st.columns([1, 1])
    tag = e1.text_input("Tag", key="ce_tag", max_chars=80, placeholder="news:tariff_shock")
    kind = e2.selectbox("Kind", ["news", "social", "market", "other"], key="ce_kind")
    display = st.text_area("Sentence shown in the battery", key="ce_display", max_chars=2000, height=90)
    note = st.text_input("Note (optional)", key="ce_note", max_chars=500)
    if st.button("Add / update context", key="ce_add", type="primary"):
        try:
            out = api.post_context(base, {"tag": tag.strip(), "kind": kind, "display": display.strip(), "active": True, "note": note or None})
            st.session_state["editor_last_context"] = out
            st.success(f"Stored {out['tag']} v{out['version']}: {out['calibration']['status']} (N = {out['calibration']['n_responses']}).")
            st.rerun()
        except api.ApiError as exc:
            st.error(f"Not stored: {exc}")
    r1, r2 = st.columns([2, 1])
    retire_tag = r1.selectbox("Retire or reactivate", [c["tag"] for c in contexts], key="ce_retire_tag", format_func=lambda t: f"{t} ({'active' if next(c for c in contexts if c['tag'] == t).get('active', True) else 'retired'})")
    if r2.button("Toggle active", key="ce_retire"):
        cur = next(c for c in contexts if c["tag"] == retire_tag)
        try:
            out = api.post_context(base, {"tag": cur["tag"], "kind": cur.get("kind") or "other", "display": cur.get("display") or "", "active": not cur.get("active", True),
                                          "triggers_delay": cur.get("triggers_delay"), "note": "retired from the editor" if cur.get("active", True) else "reactivated from the editor"})
            st.success(f"{out['tag']} is now {'active' if out['active'] else 'retired'} (v{out['version']}).")
            st.rerun()
        except api.ApiError as exc:
            st.error(f"Not stored: {exc}")

# ---------------------------------------------------------------------------------------------
# 2. Scenario texts
# ---------------------------------------------------------------------------------------------

st.subheader("Scenario texts")
st.caption("Placeholders such as {loss_pct_display} are filled per item. Saving creates a new version; the intake battery records the version it showed with every session.")
texts = scen.get("texts") or {}
for key in scen.get("keys") or list(texts):
    entry = texts.get(key) or {}
    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        new_text = c1.text_area(f"{key} (v{entry.get('version')}, {entry.get('updated_at', '')[:19]})", value=entry.get("text", ""), key=f"sc_{key}", height=90, max_chars=4000)
        if c2.button("Save", key=f"sc_save_{key}", disabled=new_text.strip() == (entry.get("text") or "").strip()):
            try:
                out = api.post_scenario(base, key, new_text.strip(), "edited on the dashboard")
                st.success(f"{key}: version {out['version']} saved.")
                st.rerun()
            except api.ApiError as exc:
                st.error(f"Not saved: {exc}")
hist = scen.get("history") or []
if hist:
    with st.expander(f"Version history ({len(hist)} versions)"):
        st.dataframe(pd.DataFrame([{"key": h["key"], "version": h["version"], "created": h["created_at"], "actor": h["actor"], "text": (h.get("payload") or {}).get("text", "")} for h in hist]), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------------------
# 3. Intervention scripts
# ---------------------------------------------------------------------------------------------

st.subheader("Intervention scripts")
st.caption(f"Every effect an intervention shows on the other pages is a {chrome.INTERVENTION_LABEL_FALLBACK}; the transform (add / remove context tags, question_order) is the modelling assumption behind it. Retiring (active = off) removes it from the ranking.")
for iv in ivs:
    with st.container(border=True):
        st.markdown(f"**{iv['name']}** (v{iv.get('version', 0)}{', ' + str(iv.get('updated_at'))[:19] if iv.get('updated_at') else ''}) — {iv.get('label')}")
        script = st.text_area("Script", value=iv.get("script", ""), key=f"iv_script_{iv['name']}", height=110, max_chars=4000)
        c1, c2, c3 = st.columns([3, 1, 1])
        tf_text = c1.text_input("Context transform (JSON: add / remove / question_order)", value=json.dumps(iv.get("mapped_context_transform") or {}, sort_keys=True), key=f"iv_tf_{iv['name']}")
        active = c2.checkbox("Active", value=bool(iv.get("active", True)), key=f"iv_active_{iv['name']}")
        if c3.button("Save", key=f"iv_save_{iv['name']}"):
            try:
                tf = json.loads(tf_text)
            except ValueError as exc:
                st.error(f"Transform is not valid JSON: {exc}")
                tf = None
            if tf is not None:
                body = {"name": iv["name"], "script": script, "mapped_context_transform": tf, "active": bool(active), "note": "edited on the dashboard"}
                try:
                    out = api.put_intervention(base, body)
                    st.success(f"{out['name']}: version {out['version']} saved.")
                    st.rerun()
                except api.ApiError as exc:
                    st.error(f"Not saved: {exc}")
        if iv.get("history"):
            with st.expander(f"Versions of {iv['name']} ({len(iv['history'])})"):
                st.dataframe(pd.DataFrame([{"version": h["version"], "created": h["created_at"], "active": (h.get("payload") or {}).get("active"), "script": (h.get("payload") or {}).get("script", "")[:120]} for h in iv["history"]]), hide_index=True, width="stretch")
