"""Settings (Phase 6, page 7): alert thresholds, target probabilities, prior strength, which
contexts count as the "typical crisis", the demo-mode toggle, and per-client export / delete.

Everything is stored by the API (``GET/PUT /settings``, a versioned document; the defaults and
ranges come back with every read). Export is ``GET /clients/{id}/export`` (every stored row of
the client, JSON); delete is ``DELETE /clients/{id}`` and cascades to every table
(responses, predictions_log, intervention_log, intake-session documents) with an audit row.
Demo mode can be forced on here; it cannot be switched off while the served model was trained
on synthetic data or the database lives under data/synthetic.
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

health = chrome.page_setup("Settings")
base = api.base_url()
try:
    model = api.model_info(base)
    book = api.clients(base)
    cfg = api.get_settings(base)
except api.ApiError as exc:
    chrome.stop_on_error(exc)
market_bar.render(model)
chrome.load_settings(base)
s = cfg.get("settings") or {}
defaults = cfg.get("defaults") or {}
rules = cfg.get("rules") or {}
vocab = chrome.ordered_contexts(list(model.get("calibrated_contexts") or {}))

# ---------------------------------------------------------------------------------------------
# 1. Editable settings
# ---------------------------------------------------------------------------------------------

st.subheader("Thresholds, targets and priors")
st.caption(f"Settings version {cfg.get('version')} (updated {cfg.get('updated_at') or 'never'}); defaults and ranges below. Every value here feeds the API's scoring, so the other pages recompute with it.")
c1, c2, c3 = st.columns(3)
alert = c1.number_input(f"Alert threshold ({chrome.WORDING})", 0.0, 1.0, float(s.get("alert_threshold", 0.5)), step=0.05, format="%.2f", key="st_alert", help=rules.get("alert_threshold"))
elevated = c2.number_input("Status 'elevated' from", 0.0, 1.0, float((s.get("status_thresholds") or {}).get("elevated", 0.25)), step=0.05, format="%.2f", key="st_elevated", help=rules.get("status_thresholds"))
high = c3.number_input("Status 'high' from", 0.0, 1.0, float((s.get("status_thresholds") or {}).get("high", 0.5)), step=0.05, format="%.2f", key="st_high", help=rules.get("status_thresholds"))
c4, c5, c6 = st.columns(3)
target = c4.number_input(f"Drawdown-capacity target ({chrome.WORDING})", 0.05, 0.95, float(s.get("capacity_target", 0.25)), step=0.05, format="%.2f", key="st_target", help=rules.get("capacity_target"))
prior = c5.number_input("Prior strength (refits)", 0.05, 20.0, float(s.get("prior_strength", 1.0)), step=0.05, format="%.2f", key="st_prior", help=rules.get("prior_strength") + "; multiplies the strength of the context-rotation prior of a Q2/Q4 refit (1 = the fit default)")
crisis = c6.multiselect("Typical crisis contexts (ordered, at most two)", vocab, default=[t for t in (s.get("typical_crisis_contexts") or []) if t in vocab], max_selections=2, key="st_crisis",
                        format_func=chrome.context_label, help="the ordered context sequence behind the behavioral drawdown capacity and its guardrail")
d1, d2, d3, d4 = st.columns(4)
demo_toggle = d1.toggle("Force demo mode (banner on every screen)", value=bool(s.get("demo_mode", False)), key="st_demo", help="cannot switch demo mode off while the data is synthetic")
rt = s.get("retrain") or {}
steps = d2.number_input("Retrain: steps", 1, 5000, int(rt.get("steps", 1500)), step=50, key="st_steps")
restarts = d3.number_input("Retrain: restarts", 1, 10, int(rt.get("restarts", 3)), step=1, key="st_restarts")
n_samples = d4.number_input("Retrain: parameter draws", 0, 500, int(rt.get("n_samples", 50)), step=10, key="st_nsamples")
if cfg.get("demo_mode_forced_by_data"):
    st.caption("Demo mode is forced by the data (synthetic-training model or a database under data/synthetic); the toggle can only add the banner, never remove it.")
if st.button("Save settings", key="st_save", type="primary"):
    patch = {"alert_threshold": float(alert), "status_thresholds": {"elevated": float(elevated), "high": float(high)}, "capacity_target": float(target), "prior_strength": float(prior),
             "typical_crisis_contexts": list(crisis), "demo_mode": bool(demo_toggle), "retrain": {"steps": int(steps), "restarts": int(restarts), "n_samples": int(n_samples)}}
    try:
        out = api.put_settings(base, patch)
        st.session_state["settings_saved"] = out
        st.session_state["alert_threshold"] = float(out["settings"]["alert_threshold"])
        st.session_state["capacity_target"] = float(out["settings"]["capacity_target"])
        st.success(f"Saved settings version {out.get('version')}.")
        st.rerun()
    except api.ApiError as exc:
        st.error(f"Not saved: {exc}")
with st.expander("Defaults and rules"):
    st.dataframe(pd.DataFrame([{"setting": k, "default": json.dumps(v), "current": json.dumps(s.get(k)), "rule": rules.get(k, "")} for k, v in defaults.items()]), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------------------
# 2. Export and delete client data
# ---------------------------------------------------------------------------------------------

st.subheader("Client data: export and delete")
ids = [c["client_id"] for c in book]
labels = {c["client_id"]: c.get("display_label") or c["client_id"] for c in book}
if not ids:
    st.info("No clients in the served book.")
else:
    e1, e2 = st.columns([3, 1])
    exp_id = e1.selectbox("Export every stored row of a client (JSON)", ids, format_func=lambda i: f"{labels[i]} ({i})", key="st_export_client")
    if e2.button("Prepare export", key="st_export"):
        try:
            dump = api.export_client(base, exp_id)
            st.session_state["export_json"] = json.dumps(dump, indent=1)
            st.session_state["export_client_id"] = exp_id
        except api.ApiError as exc:
            st.error(f"Export failed: {exc}")
    if st.session_state.get("export_json") and st.session_state.get("export_client_id") == exp_id:
        dump = json.loads(st.session_state["export_json"])
        st.caption(f"{len(dump.get('responses', []))} responses, {len(dump.get('predictions_log', []))} logged predictions, {len(dump.get('intervention_log', []))} intervention-log rows, "
                   f"{len(dump.get('intake_sessions', []))} intake sessions; synthetic: {dump.get('synthetic')}; exported {dump.get('exported_at')}.")
        st.download_button("Download export (JSON)", st.session_state["export_json"], file_name=f"bre_client_{exp_id}.json", mime="application/json", key="st_export_download")
    st.divider()
    x1, x2, x3 = st.columns([3, 2, 1])
    del_id = x1.selectbox("Delete a client and every row that belongs to it", ids, format_func=lambda i: f"{labels[i]} ({i})", key="st_delete_client")
    confirm = x2.checkbox(f"I understand this removes the client, its responses, logged predictions, intervention log and intake sessions", key="st_delete_confirm")
    if x3.button("Delete", key="st_delete", disabled=not confirm, type="secondary"):
        try:
            out = api.delete_client(base, del_id)
            st.session_state["delete_result"] = out
            st.session_state.pop("export_json", None)
            st.success(f"Deleted {out['client_id']}: {out['deleted']} (audit row {out['audit_log_id']}).")
            st.rerun()
        except api.ApiError as exc:
            st.error(f"Delete failed: {exc}")
    last = st.session_state.get("delete_result")
    if last:
        st.caption(f"Last delete: {last.get('client_id')} → {last.get('deleted')} (audit row {last.get('audit_log_id')}).")
