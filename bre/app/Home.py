"""BRE advisor dashboard — Home (Phase 6).

Market-state bar (sidebar, every page), the book summary under the current market state and
the served model's transparency card. Every number comes from the BRE API; nothing here
imports model code. Run with ``make dashboard`` (starts the API when it is not listening).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import api_client as api  # noqa: E402
from components import chrome, market_bar  # noqa: E402

health = chrome.page_setup("Behavioral Risk Engine — advisor dashboard")
base = api.base_url()
try:
    model = api.model_info(base)
    book = api.clients(base)
except api.ApiError as exc:
    chrome.stop_on_error(exc)

payload = market_bar.render(model)
target = float(st.session_state.setdefault("capacity_target", 0.25))
try:
    scored = api.score_book(base, api.dumps(api.client_payload(book)), api.dumps(payload), target)
except api.ApiError as exc:
    chrome.stop_on_error(exc)

market_bar.render_scenario(scored["scenario"], scored["n_calibration"])
rows = scored["rows"]
meta = scored["meta"]
if "alert_threshold" not in st.session_state:
    st.session_state["alert_threshold"] = float(meta["status_thresholds"]["high"])
threshold = float(st.session_state["alert_threshold"])

st.subheader("Book under the current market state")
frame = pd.DataFrame(rows)
n_above = int((frame["p_sell"] >= threshold).sum()) if len(frame) else 0
n_min = chrome.calibration_min_n(scored["n_calibration"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clients scored", len(frame))
c2.metric(f"Clients above threshold {threshold:.2f}", n_above, help="Edit the threshold on the Book triage page.")
if len(frame):
    med = frame["p_sell"].median()
    lo = frame["ci80_low"].median() if frame["ci80_low"].notna().any() else None
    hi = frame["ci80_high"].median() if frame["ci80_high"].notna().any() else None
    c3.metric(f"Median {chrome.WORDING}", f"{med:.2f}", help="Median over clients of the point prediction; the median 80% interval bounds are shown below.")
    c3.caption(f"median 80% interval [{chrome.fmt_p(lo)}, {chrome.fmt_p(hi)}]; calibration N (min over scenario contexts) = {n_min}")
c4.metric("Served model", f"{health.get('model_version')}", help=f"{health.get('model_type')} ({model.get('family')}); synthetic training: {model.get('is_synthetic_training')}")
st.caption(f"Intervals: {meta.get('interval')}. Status thresholds: {meta.get('status_thresholds')}.")

st.page_link("pages/1_Book_triage.py", label="Book triage — one row per client, alerts, CSV export", icon="📋")
st.page_link("pages/2_Client_profile.py", label="Client profile — risk state, sensitivities, what-if, interventions, PDF", icon="👤")

with st.expander("Served model and provenance (transparency)", expanded=False):
    st.write(
        f"**{model.get('version')}** — type {model.get('model_type')}, family {model.get('family')}, {model.get('n_params')} parameters "
        f"({model.get('n_params_population')} population-level), design {model.get('design_version')}, created {model.get('created_at')}, promoted {model.get('promoted_at')}."
    )
    st.write(f"**Evaluation status:** {model.get('evaluation_status')}")
    st.write(f"**Pre-registered decision rule:** {model.get('decision_rule')}")
    st.write(f"**Intervals:** {model.get('interval_source')} ({model.get('n_draws')} draws)")
    st.write(f"**Calibration rule:** {model.get('calibration_rule')}")
    cal = model.get("calibrated_contexts") or {}
    cal_df = pd.DataFrame([{"context": chrome.context_label(t), "tag": t, "training responses (N)": e.get("n_responses"), "status": e.get("status"), "minimum N": e.get("min_responses")} for t, e in cal.items()])
    st.dataframe(cal_df, hide_index=True, width="stretch")
    prov = model.get("provenance") or {}
    st.write(f"**Training data:** {'; '.join(model.get('training_data_refs') or [])}")
    st.write(f"**Provenance note:** {prov.get('note')}")
    used = prov.get("catalog_entries_used_in_training") or []
    st.write("**Catalogued datasets used in training:** " + (", ".join(f"{e['id']}. {e['name']}" for e in used) if used else "none"))
    metrics = model.get("metrics") or {}
    st.write(f"**Training fit:** NLL per response {chrome.fmt_p(metrics.get('train_nll_per_response'), 3)}, Brier {chrome.fmt_p(metrics.get('train_brier'), 3)} on {metrics.get('n_responses')} responses of {metrics.get('n_subjects')} subjects; held out: {metrics.get('held_out_note')}")
    st.write(f"**Notes:** {model.get('notes')}")
